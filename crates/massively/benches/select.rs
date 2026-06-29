use cubecl::wgpu::WgpuRuntime;
mod common;

use common::{Runtime, SIZES, iter_gpu, select_flags, sync};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use cubecl::prelude::*;
use massively::op::BinaryPredicateOp;
use massively::{Executor, SoA1, copy_where, partition, remove_where, unique};

fn alternating_signed(len: usize) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let value = (index % 251) as f32 + 1.0;
            if index % 2 == 0 { value } else { -value }
        })
        .collect()
}

fn alternating_flags(len: usize) -> Vec<u32> {
    (0..len)
        .map(|index| if index % 2 == 0 { 1 } else { 0 })
        .collect()
}

fn repeated_pairs(len: usize) -> Vec<f32> {
    (0..len).map(|index| (index / 2) as f32).collect()
}

struct Equal;

#[cubecl::cube]
impl BinaryPredicateOp<WgpuRuntime, (f32,)> for Equal {
    fn apply(lhs: (f32,), rhs: (f32,)) -> bool {
        lhs.0 == rhs.0
    }
}

fn check_copy_where(exec: &Executor<WgpuRuntime>) {
    let values = exec.to_device(&[-1.0_f32, 2.0, -3.0, 4.0]).unwrap();
    let stencil = exec.to_device(&[0_u32, 1, 0, 1]).unwrap();
    let SoA1(output) =
        copy_where(&exec, massively::SoA1(values.slice(..)), stencil.slice(..)).unwrap();
    assert_eq!(exec.to_host(&output).unwrap(), vec![2.0, 4.0]);
}

fn check_selection_family(exec: &Executor<WgpuRuntime>) {
    let values = exec.to_device(&[-1.0_f32, 2.0, -3.0, 4.0]).unwrap();
    let stencil = exec.to_device(&[0_u32, 1, 0, 1]).unwrap();

    let SoA1(removed) =
        remove_where(&exec, massively::SoA1(values.slice(..)), stencil.slice(..)).unwrap();
    assert_eq!(exec.to_host(&removed).unwrap(), vec![-1.0, -3.0]);

    let (SoA1(positives), SoA1(non_positives)) =
        partition(&exec, massively::SoA1(values.slice(..)), Positive).unwrap();
    assert_eq!(exec.to_host(&positives).unwrap(), vec![2.0, 4.0]);
    assert_eq!(exec.to_host(&non_positives).unwrap(), vec![-1.0, -3.0]);

    let repeated = exec.to_device(&[1.0_f32, 1.0, 2.0, 2.0, 3.0]).unwrap();
    let SoA1(unique_values) = unique(&exec, massively::SoA1(repeated.slice(..)), Equal).unwrap();
    assert_eq!(exec.to_host(&unique_values).unwrap(), vec![1.0, 2.0, 3.0]);
}

struct Positive;

#[cubecl::cube]
impl massively::op::PredicateOp<WgpuRuntime, (f32,)> for Positive {
    fn apply(input: (f32,)) -> bool {
        input.0 > 0.0
    }
}

fn bench_select(c: &mut Criterion) {
    let mut copy_group = c.benchmark_group("copy_where_selectivity");
    for backend in Runtime::available() {
        let exec = backend.exec();
        check_copy_where(&exec);

        for &selected_per_100 in &[0_usize, 1, 50, 99, 100] {
            for &len in SIZES {
                let values = exec.to_device(&alternating_signed(len)).unwrap();
                let stencil = exec
                    .to_device(&select_flags(len, selected_per_100))
                    .unwrap();
                sync(&exec);
                copy_group.bench_function(
                    BenchmarkId::new(format!("{}-{}pct", backend.name(), selected_per_100), len),
                    |b| {
                        iter_gpu(b, || {
                            let output = copy_where(
                                &exec,
                                massively::SoA1(black_box(values.slice(..))),
                                black_box(stencil.slice(..)),
                            )
                            .unwrap();
                            let output_len = output.0.len();
                            drop(output);
                            sync(&exec);
                            black_box(output_len)
                        })
                    },
                );
            }
        }
    }
    copy_group.finish();

    let mut remove_group = c.benchmark_group("remove_where");
    for backend in Runtime::available() {
        let exec = backend.exec();
        check_selection_family(&exec);

        for &len in SIZES {
            let values = exec.to_device(&alternating_signed(len)).unwrap();
            let stencil = exec.to_device(&alternating_flags(len)).unwrap();
            sync(&exec);
            remove_group.bench_function(BenchmarkId::new(backend.name(), len), |b| {
                iter_gpu(b, || {
                    let output = remove_where(
                        &exec,
                        massively::SoA1(black_box(values.slice(..))),
                        black_box(stencil.slice(..)),
                    )
                    .unwrap();
                    let output_len = output.0.len();
                    drop(output);
                    sync(&exec);
                    black_box(output_len)
                })
            });
        }
    }
    remove_group.finish();

    let mut partition_group = c.benchmark_group("partition");
    for backend in Runtime::available() {
        let exec = backend.exec();
        check_selection_family(&exec);

        for &len in SIZES {
            let values = exec.to_device(&alternating_signed(len)).unwrap();
            sync(&exec);
            partition_group.bench_function(BenchmarkId::new(backend.name(), len), |b| {
                iter_gpu(b, || {
                    let output = partition(
                        &exec,
                        massively::SoA1(black_box(values.slice(..))),
                        Positive,
                    )
                    .unwrap();
                    let output_len = output.0.0.len() + output.1.0.len();
                    drop(output);
                    sync(&exec);
                    black_box(output_len)
                })
            });
        }
    }
    partition_group.finish();

    let mut unique_group = c.benchmark_group("unique");
    for backend in Runtime::available() {
        let exec = backend.exec();
        check_selection_family(&exec);

        for &len in SIZES {
            let values = exec.to_device(&repeated_pairs(len)).unwrap();
            sync(&exec);
            unique_group.bench_function(BenchmarkId::new(backend.name(), len), |b| {
                iter_gpu(b, || {
                    let output =
                        unique(&exec, massively::SoA1(black_box(values.slice(..))), Equal).unwrap();
                    let output_len = output.0.len();
                    drop(output);
                    sync(&exec);
                    black_box(output_len)
                })
            });
        }
    }
    unique_group.finish();
}

criterion_group! {
    name = benches;
    config = common::criterion();
    targets = bench_select
}
criterion_main!(benches);
