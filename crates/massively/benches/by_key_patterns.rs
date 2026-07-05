use cubecl::wgpu::WgpuRuntime;
mod common;

use common::{Runtime, SIZES, dense_f32, iter_gpu, run_keys, sync};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use cubecl::prelude::*;
use massively::op::{BinaryPredicateOp, ReductionOp};
use massively::{Executor, exclusive_scan_by_key, inclusive_scan_by_key, reduce_by_key};

struct Sum;

#[cubecl::cube]
impl ReductionOp<WgpuRuntime, (f32,)> for Sum {
    fn apply(lhs: (f32,), rhs: (f32,)) -> (f32,) {
        (lhs.0 + rhs.0,)
    }
}

struct KeyEq;

#[cubecl::cube]
impl BinaryPredicateOp<WgpuRuntime, (u32,)> for KeyEq {
    fn apply(lhs: (u32,), rhs: (u32,)) -> bool {
        lhs.0 == rhs.0
    }
}

fn key_patterns(len: usize) -> Vec<(&'static str, Vec<u32>)> {
    vec![
        ("run1", run_keys(len, 1)),
        ("run8", run_keys(len, 8)),
        ("run128", run_keys(len, 128)),
        ("all", run_keys(len, len)),
    ]
}

fn check_by_key(exec: &Executor<WgpuRuntime>) {
    let keys = exec.to_device(&[0_u32, 0, 1, 1]).unwrap();
    let values = exec.to_device(&[1.0_f32, 2.0, 10.0, 20.0]).unwrap();

    let inclusive = exec.to_device(&[0.0_f32; 4]).unwrap();
    inclusive_scan_by_key(
        exec,
        massively::SoA1(keys.slice(..)),
        massively::SoA1(values.slice(..)),
        KeyEq,
        Sum,
        massively::SoA1(inclusive.slice_mut(..)),
    )
    .unwrap();
    assert_eq!(
        exec.to_host(&inclusive).unwrap(),
        vec![1.0, 3.0, 10.0, 30.0]
    );

    let exclusive = exec.to_device(&[0.0_f32; 4]).unwrap();
    exclusive_scan_by_key(
        exec,
        massively::SoA1(keys.slice(..)),
        massively::SoA1(values.slice(..)),
        KeyEq,
        (0.0,),
        Sum,
        massively::SoA1(exclusive.slice_mut(..)),
    )
    .unwrap();
    assert_eq!(exec.to_host(&exclusive).unwrap(), vec![0.0, 1.0, 0.0, 10.0]);
}

fn bench_by_key_patterns(c: &mut Criterion) {
    let mut inclusive_group = c.benchmark_group("inclusive_scan_by_key_patterns");
    for backend in Runtime::available() {
        let exec = backend.exec();
        check_by_key(&exec);

        for &len in SIZES {
            let values = exec.to_device(&dense_f32(len)).unwrap();
            let output = exec.to_device(&vec![0.0_f32; len]).unwrap();
            for (pattern, host_keys) in key_patterns(len) {
                let keys = exec.to_device(&host_keys).unwrap();
                sync(&exec);
                inclusive_group.bench_function(
                    BenchmarkId::new(format!("{}-{pattern}", backend.name()), len),
                    |b| {
                        iter_gpu(b, || {
                            inclusive_scan_by_key(
                                &exec,
                                massively::SoA1(black_box(keys.slice(..))),
                                massively::SoA1(black_box(values.slice(..))),
                                KeyEq,
                                Sum,
                                massively::SoA1(black_box(output.slice_mut(..))),
                            )
                            .unwrap();
                            sync(&exec);
                            black_box(&output)
                        })
                    },
                );
            }
        }
    }
    inclusive_group.finish();

    let mut exclusive_group = c.benchmark_group("exclusive_scan_by_key_patterns");
    for backend in Runtime::available() {
        let exec = backend.exec();
        check_by_key(&exec);

        for &len in SIZES {
            let values = exec.to_device(&dense_f32(len)).unwrap();
            let output = exec.to_device(&vec![0.0_f32; len]).unwrap();
            for (pattern, host_keys) in key_patterns(len) {
                let keys = exec.to_device(&host_keys).unwrap();
                sync(&exec);
                exclusive_group.bench_function(
                    BenchmarkId::new(format!("{}-{pattern}", backend.name()), len),
                    |b| {
                        iter_gpu(b, || {
                            exclusive_scan_by_key(
                                &exec,
                                massively::SoA1(black_box(keys.slice(..))),
                                massively::SoA1(black_box(values.slice(..))),
                                KeyEq,
                                (0.0,),
                                Sum,
                                massively::SoA1(black_box(output.slice_mut(..))),
                            )
                            .unwrap();
                            sync(&exec);
                            black_box(&output)
                        })
                    },
                );
            }
        }
    }
    exclusive_group.finish();

    let mut reduce_group = c.benchmark_group("reduce_by_key_patterns");
    for backend in Runtime::available() {
        let exec = backend.exec();
        check_by_key(&exec);

        for &len in SIZES {
            let values = exec.to_device(&dense_f32(len)).unwrap();
            let out_keys = exec.to_device(&vec![0_u32; len]).unwrap();
            let out_values = exec.to_device(&vec![0.0_f32; len]).unwrap();
            for (pattern, host_keys) in key_patterns(len) {
                let keys = exec.to_device(&host_keys).unwrap();
                sync(&exec);
                reduce_group.bench_function(
                    BenchmarkId::new(format!("{}-{pattern}", backend.name()), len),
                    |b| {
                        iter_gpu(b, || {
                            let len = reduce_by_key(
                                &exec,
                                massively::SoA1(black_box(keys.slice(..))),
                                massively::SoA1(black_box(values.slice(..))),
                                KeyEq,
                                (0.0,),
                                Sum,
                                massively::SoA1(black_box(out_keys.slice_mut(..))),
                                massively::SoA1(black_box(out_values.slice_mut(..))),
                            )
                            .unwrap();
                            sync(&exec);
                            black_box((len, &out_keys, &out_values))
                        })
                    },
                );
            }
        }
    }
    reduce_group.finish();
}

criterion_group! {
    name = benches;
    config = common::criterion();
    targets = bench_by_key_patterns
}
criterion_main!(benches);
