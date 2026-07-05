use cubecl::wgpu::WgpuRuntime;
mod common;

use common::{Runtime, SIZES, iter_gpu, select_flags, sync};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use cubecl::prelude::*;
use massively::op::BinaryPredicateOp;
use massively::{Executor, copy_where, partition, remove_where, unique};

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

fn offset_u32(len: usize, offset: u32) -> Vec<u32> {
    (0..len).map(|index| index as u32 + offset).collect()
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
    let output = exec.to_device(&[0.0_f32; 4]).unwrap();
    let len = copy_where(
        &exec,
        massively::Zip1(values.slice(..)),
        stencil.slice(..),
        massively::Zip1(output.slice_mut(..)),
    )
    .unwrap();
    assert_eq!(exec.to_host(&output.slice(..len)).unwrap(), vec![2.0, 4.0]);
}

fn check_selection_family(exec: &Executor<WgpuRuntime>) {
    let values = exec.to_device(&[-1.0_f32, 2.0, -3.0, 4.0]).unwrap();
    let stencil = exec.to_device(&[0_u32, 1, 0, 1]).unwrap();

    let removed = exec.to_device(&[0.0_f32; 4]).unwrap();
    let len = remove_where(
        &exec,
        massively::Zip1(values.slice(..)),
        stencil.slice(..),
        massively::Zip1(removed.slice_mut(..)),
    )
    .unwrap();
    assert_eq!(
        exec.to_host(&removed.slice(..len)).unwrap(),
        vec![-1.0, -3.0]
    );

    let partitioned = exec.to_device(&[0.0_f32; 4]).unwrap();
    let split = partition(
        &exec,
        massively::Zip1(values.slice(..)),
        Positive,
        (),
        massively::Zip1(partitioned.slice_mut(..)),
    )
    .unwrap();
    assert_eq!(
        exec.to_host(&partitioned.slice(..split)).unwrap(),
        vec![2.0, 4.0]
    );
    assert_eq!(
        exec.to_host(&partitioned.slice(split..)).unwrap(),
        vec![-1.0, -3.0]
    );

    let repeated = exec.to_device(&[1.0_f32, 1.0, 2.0, 2.0, 3.0]).unwrap();
    let unique_values = exec.to_device(&[0.0_f32; 5]).unwrap();
    let len = unique(
        &exec,
        massively::Zip1(repeated.slice(..)),
        Equal,
        massively::Zip1(unique_values.slice_mut(..)),
    )
    .unwrap();
    assert_eq!(
        exec.to_host(&unique_values.slice(..len)).unwrap(),
        vec![1.0, 2.0, 3.0]
    );
}

fn check_wide_copy_remove_where(exec: &Executor<WgpuRuntime>) {
    let a = exec.to_device(&[1_u32, 2, 3, 4, 5]).unwrap();
    let b = exec.to_device(&[11_u32, 12, 13, 14, 15]).unwrap();
    let c = exec.to_device(&[21_u32, 22, 23, 24, 25]).unwrap();
    let d = exec.to_device(&[31_u32, 32, 33, 34, 35]).unwrap();
    let e = exec.to_device(&[41_u32, 42, 43, 44, 45]).unwrap();
    let f = exec.to_device(&[51_u32, 52, 53, 54, 55]).unwrap();
    let g = exec.to_device(&[61_u32, 62, 63, 64, 65]).unwrap();
    let stencil = exec.to_device(&[1_u32, 0, 1, 0, 1]).unwrap();
    let out_a = exec.to_device(&[0_u32; 5]).unwrap();
    let out_b = exec.to_device(&[0_u32; 5]).unwrap();
    let out_c = exec.to_device(&[0_u32; 5]).unwrap();
    let out_d = exec.to_device(&[0_u32; 5]).unwrap();
    let out_e = exec.to_device(&[0_u32; 5]).unwrap();
    let out_f = exec.to_device(&[0_u32; 5]).unwrap();
    let out_g = exec.to_device(&[0_u32; 5]).unwrap();

    let selected = copy_where(
        exec,
        massively::Zip7(
            a.slice(..),
            b.slice(..),
            c.slice(..),
            d.slice(..),
            e.slice(..),
            f.slice(..),
            g.slice(..),
        ),
        stencil.slice(..),
        massively::Zip7(
            out_a.slice_mut(..),
            out_b.slice_mut(..),
            out_c.slice_mut(..),
            out_d.slice_mut(..),
            out_e.slice_mut(..),
            out_f.slice_mut(..),
            out_g.slice_mut(..),
        ),
    )
    .unwrap();
    assert_eq!(selected, 3);

    let remaining = remove_where(
        exec,
        massively::Zip7(
            a.slice(..),
            b.slice(..),
            c.slice(..),
            d.slice(..),
            e.slice(..),
            f.slice(..),
            g.slice(..),
        ),
        stencil.slice(..),
        massively::Zip7(
            out_a.slice_mut(..),
            out_b.slice_mut(..),
            out_c.slice_mut(..),
            out_d.slice_mut(..),
            out_e.slice_mut(..),
            out_f.slice_mut(..),
            out_g.slice_mut(..),
        ),
    )
    .unwrap();
    assert_eq!(remaining, 2);
}

fn check_wide_partition(exec: &Executor<WgpuRuntime>) {
    let a = exec.to_device(&[0_u32, 1, 2, 3, 4]).unwrap();
    let b = exec.to_device(&[10_u32, 11, 12, 13, 14]).unwrap();
    let c = exec.to_device(&[20_u32, 21, 22, 23, 24]).unwrap();
    let d = exec.to_device(&[30_u32, 31, 32, 33, 34]).unwrap();
    let e = exec.to_device(&[40_u32, 41, 42, 43, 44]).unwrap();
    let f = exec.to_device(&[50_u32, 51, 52, 53, 54]).unwrap();
    let g = exec.to_device(&[60_u32, 61, 62, 63, 64]).unwrap();
    let out_a = exec.to_device(&[0_u32; 5]).unwrap();
    let out_b = exec.to_device(&[0_u32; 5]).unwrap();
    let out_c = exec.to_device(&[0_u32; 5]).unwrap();
    let out_d = exec.to_device(&[0_u32; 5]).unwrap();
    let out_e = exec.to_device(&[0_u32; 5]).unwrap();
    let out_f = exec.to_device(&[0_u32; 5]).unwrap();
    let out_g = exec.to_device(&[0_u32; 5]).unwrap();

    let split = partition(
        exec,
        massively::Zip7(
            a.slice(..),
            b.slice(..),
            c.slice(..),
            d.slice(..),
            e.slice(..),
            f.slice(..),
            g.slice(..),
        ),
        FirstColumnEven,
        (),
        massively::Zip7(
            out_a.slice_mut(..),
            out_b.slice_mut(..),
            out_c.slice_mut(..),
            out_d.slice_mut(..),
            out_e.slice_mut(..),
            out_f.slice_mut(..),
            out_g.slice_mut(..),
        ),
    )
    .unwrap();
    assert_eq!(split, 3);
    assert_eq!(exec.to_host(&out_a.slice(split..)).unwrap().len(), 2);
}

struct Positive;

#[cubecl::cube]
impl massively::op::PredicateOp<WgpuRuntime, (f32,)> for Positive {
    type Env = ();

    fn apply(_env: (), input: (f32,)) -> bool {
        input.0 > 0.0
    }
}

struct FirstColumnEven;

#[cubecl::cube]
impl massively::op::PredicateOp<WgpuRuntime, (u32, u32, u32, u32, u32, u32, u32)>
    for FirstColumnEven
{
    type Env = ();

    fn apply(_env: (), input: (u32, u32, u32, u32, u32, u32, u32)) -> bool {
        input.0 % 2u32 == 0u32
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
                let output = exec.to_device(&vec![0.0_f32; len]).unwrap();
                sync(&exec);
                copy_group.bench_function(
                    BenchmarkId::new(format!("{}-{}pct", backend.name(), selected_per_100), len),
                    |b| {
                        iter_gpu(b, || {
                            let output_len = copy_where(
                                &exec,
                                massively::Zip1(black_box(values.slice(..))),
                                black_box(stencil.slice(..)),
                                massively::Zip1(black_box(output.slice_mut(..))),
                            )
                            .unwrap();
                            sync(&exec);
                            black_box((output_len, &output))
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
            let output = exec.to_device(&vec![0.0_f32; len]).unwrap();
            sync(&exec);
            remove_group.bench_function(BenchmarkId::new(backend.name(), len), |b| {
                iter_gpu(b, || {
                    let output_len = remove_where(
                        &exec,
                        massively::Zip1(black_box(values.slice(..))),
                        black_box(stencil.slice(..)),
                        massively::Zip1(black_box(output.slice_mut(..))),
                    )
                    .unwrap();
                    sync(&exec);
                    black_box((output_len, &output))
                })
            });
        }
    }
    remove_group.finish();

    let mut wide_group = c.benchmark_group("wide_tuple_copy_remove_where");
    for backend in Runtime::available() {
        let exec = backend.exec();
        check_wide_copy_remove_where(&exec);

        for &len in SIZES {
            let col_a = exec.to_device(&offset_u32(len, 0)).unwrap();
            let col_b = exec.to_device(&offset_u32(len, 10)).unwrap();
            let col_c = exec.to_device(&offset_u32(len, 20)).unwrap();
            let col_d = exec.to_device(&offset_u32(len, 30)).unwrap();
            let col_e = exec.to_device(&offset_u32(len, 40)).unwrap();
            let col_f = exec.to_device(&offset_u32(len, 50)).unwrap();
            let col_g = exec.to_device(&offset_u32(len, 60)).unwrap();
            let stencil = exec.to_device(&alternating_flags(len)).unwrap();
            let out_a = exec.to_device(&vec![0_u32; len]).unwrap();
            let out_b = exec.to_device(&vec![0_u32; len]).unwrap();
            let out_c = exec.to_device(&vec![0_u32; len]).unwrap();
            let out_d = exec.to_device(&vec![0_u32; len]).unwrap();
            let out_e = exec.to_device(&vec![0_u32; len]).unwrap();
            let out_f = exec.to_device(&vec![0_u32; len]).unwrap();
            let out_g = exec.to_device(&vec![0_u32; len]).unwrap();
            sync(&exec);
            wide_group.bench_function(
                BenchmarkId::new(format!("{}-copy", backend.name()), len),
                |b| {
                    iter_gpu(b, || {
                        let output_len = copy_where(
                            &exec,
                            massively::Zip7(
                                black_box(col_a.slice(..)),
                                black_box(col_b.slice(..)),
                                black_box(col_c.slice(..)),
                                black_box(col_d.slice(..)),
                                black_box(col_e.slice(..)),
                                black_box(col_f.slice(..)),
                                black_box(col_g.slice(..)),
                            ),
                            black_box(stencil.slice(..)),
                            massively::Zip7(
                                black_box(out_a.slice_mut(..)),
                                black_box(out_b.slice_mut(..)),
                                black_box(out_c.slice_mut(..)),
                                black_box(out_d.slice_mut(..)),
                                black_box(out_e.slice_mut(..)),
                                black_box(out_f.slice_mut(..)),
                                black_box(out_g.slice_mut(..)),
                            ),
                        )
                        .unwrap();
                        sync(&exec);
                        black_box((
                            output_len, &out_a, &out_b, &out_c, &out_d, &out_e, &out_f, &out_g,
                        ))
                    })
                },
            );
            wide_group.bench_function(
                BenchmarkId::new(format!("{}-remove", backend.name()), len),
                |b| {
                    iter_gpu(b, || {
                        let output_len = remove_where(
                            &exec,
                            massively::Zip7(
                                black_box(col_a.slice(..)),
                                black_box(col_b.slice(..)),
                                black_box(col_c.slice(..)),
                                black_box(col_d.slice(..)),
                                black_box(col_e.slice(..)),
                                black_box(col_f.slice(..)),
                                black_box(col_g.slice(..)),
                            ),
                            black_box(stencil.slice(..)),
                            massively::Zip7(
                                black_box(out_a.slice_mut(..)),
                                black_box(out_b.slice_mut(..)),
                                black_box(out_c.slice_mut(..)),
                                black_box(out_d.slice_mut(..)),
                                black_box(out_e.slice_mut(..)),
                                black_box(out_f.slice_mut(..)),
                                black_box(out_g.slice_mut(..)),
                            ),
                        )
                        .unwrap();
                        sync(&exec);
                        black_box((
                            output_len, &out_a, &out_b, &out_c, &out_d, &out_e, &out_f, &out_g,
                        ))
                    })
                },
            );
        }
    }
    wide_group.finish();

    let mut partition_group = c.benchmark_group("partition");
    for backend in Runtime::available() {
        let exec = backend.exec();
        check_selection_family(&exec);

        for &len in SIZES {
            let values = exec.to_device(&alternating_signed(len)).unwrap();
            let output = exec.to_device(&vec![0.0_f32; len]).unwrap();
            sync(&exec);
            partition_group.bench_function(BenchmarkId::new(backend.name(), len), |b| {
                iter_gpu(b, || {
                    let split = partition(
                        &exec,
                        massively::Zip1(black_box(values.slice(..))),
                        Positive,
                        (),
                        massively::Zip1(black_box(output.slice_mut(..))),
                    )
                    .unwrap();
                    sync(&exec);
                    black_box((split, &output))
                })
            });
        }
    }
    partition_group.finish();

    let mut wide_partition_group = c.benchmark_group("wide_tuple_partition");
    for backend in Runtime::available() {
        let exec = backend.exec();
        check_wide_partition(&exec);

        for &len in SIZES {
            let col_a = exec.to_device(&offset_u32(len, 0)).unwrap();
            let col_b = exec.to_device(&offset_u32(len, 10)).unwrap();
            let col_c = exec.to_device(&offset_u32(len, 20)).unwrap();
            let col_d = exec.to_device(&offset_u32(len, 30)).unwrap();
            let col_e = exec.to_device(&offset_u32(len, 40)).unwrap();
            let col_f = exec.to_device(&offset_u32(len, 50)).unwrap();
            let col_g = exec.to_device(&offset_u32(len, 60)).unwrap();
            let out_a = exec.to_device(&vec![0_u32; len]).unwrap();
            let out_b = exec.to_device(&vec![0_u32; len]).unwrap();
            let out_c = exec.to_device(&vec![0_u32; len]).unwrap();
            let out_d = exec.to_device(&vec![0_u32; len]).unwrap();
            let out_e = exec.to_device(&vec![0_u32; len]).unwrap();
            let out_f = exec.to_device(&vec![0_u32; len]).unwrap();
            let out_g = exec.to_device(&vec![0_u32; len]).unwrap();
            sync(&exec);
            wide_partition_group.bench_function(BenchmarkId::new(backend.name(), len), |b| {
                iter_gpu(b, || {
                    let split = partition(
                        &exec,
                        massively::Zip7(
                            black_box(col_a.slice(..)),
                            black_box(col_b.slice(..)),
                            black_box(col_c.slice(..)),
                            black_box(col_d.slice(..)),
                            black_box(col_e.slice(..)),
                            black_box(col_f.slice(..)),
                            black_box(col_g.slice(..)),
                        ),
                        FirstColumnEven,
                        (),
                        massively::Zip7(
                            black_box(out_a.slice_mut(..)),
                            black_box(out_b.slice_mut(..)),
                            black_box(out_c.slice_mut(..)),
                            black_box(out_d.slice_mut(..)),
                            black_box(out_e.slice_mut(..)),
                            black_box(out_f.slice_mut(..)),
                            black_box(out_g.slice_mut(..)),
                        ),
                    )
                    .unwrap();
                    sync(&exec);
                    black_box((
                        split, &out_a, &out_b, &out_c, &out_d, &out_e, &out_f, &out_g,
                    ))
                })
            });
        }
    }
    wide_partition_group.finish();

    let mut unique_group = c.benchmark_group("unique");
    for backend in Runtime::available() {
        let exec = backend.exec();
        check_selection_family(&exec);

        for &len in SIZES {
            let values = exec.to_device(&repeated_pairs(len)).unwrap();
            let output = exec.to_device(&vec![0.0_f32; len]).unwrap();
            sync(&exec);
            unique_group.bench_function(BenchmarkId::new(backend.name(), len), |b| {
                iter_gpu(b, || {
                    let output_len = unique(
                        &exec,
                        massively::Zip1(black_box(values.slice(..))),
                        Equal,
                        massively::Zip1(black_box(output.slice_mut(..))),
                    )
                    .unwrap();
                    sync(&exec);
                    black_box((output_len, &output))
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
