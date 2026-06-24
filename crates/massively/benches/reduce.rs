use cubecl::wgpu::WgpuRuntime;
mod common;

use common::{Runtime, SIZES, dense_f32, sync};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use cubecl::prelude::*;
use massively::op::{BinaryPredicateOp, ReductionOp};
use massively::{DeviceVec, Executor, reduce, reduce_by_key};

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

fn keys(len: usize) -> Vec<u32> {
    (0..len).map(|index| (index / 8) as u32).collect()
}

fn check_reduce(exec: &Executor<WgpuRuntime>) {
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let output = reduce(&exec, massively::SoA1(values.slice(..)), (0.0,), Sum).unwrap();
    assert_eq!(output, (10.0,));
}

fn check_reduce_by_key(exec: &Executor<WgpuRuntime>) {
    let keys = exec.to_device(&[0_u32, 0, 1, 1]).unwrap();
    let values = exec.to_device(&[1.0_f32, 2.0, 10.0, 20.0]).unwrap();
    let ((out_keys,), (out_values,)) = reduce_by_key(
        &exec,
        massively::SoA1(keys.slice(..)),
        massively::SoA1(values.slice(..)),
        KeyEq,
        (0.0,),
        Sum,
    )
    .unwrap();
    assert_eq!(exec.to_host(&out_keys).unwrap(), vec![0, 1]);
    assert_eq!(exec.to_host(&out_values).unwrap(), vec![3.0, 30.0]);
}

fn bench_reduce(c: &mut Criterion) {
    let mut reduce_group = c.benchmark_group("reduce");
    for backend in Runtime::available() {
        let exec = backend.exec();
        check_reduce(&exec);

        for &len in SIZES {
            let values = exec.to_device(&dense_f32(len)).unwrap();
            sync(&exec);
            reduce_group.bench_function(BenchmarkId::new(backend.name(), len), |b| {
                b.iter(|| {
                    let output = reduce(
                        &exec,
                        massively::SoA1(black_box(values.slice(..))),
                        (0.0,),
                        Sum,
                    )
                    .unwrap();
                    sync(&exec);
                    black_box(output)
                })
            });
        }
    }
    reduce_group.finish();

    let mut reduce_by_key_group = c.benchmark_group("reduce_by_key");
    for backend in Runtime::available() {
        let exec = backend.exec();
        check_reduce_by_key(&exec);

        for &len in SIZES {
            let keys = exec.to_device(&keys(len)).unwrap();
            let values = exec.to_device(&dense_f32(len)).unwrap();
            sync(&exec);
            reduce_by_key_group.bench_function(BenchmarkId::new(backend.name(), len), |b| {
                b.iter(|| {
                    let output: (
                        (DeviceVec<WgpuRuntime, u32>,),
                        (DeviceVec<WgpuRuntime, f32>,),
                    ) = reduce_by_key(
                        &exec,
                        massively::SoA1(black_box(keys.slice(..))),
                        massively::SoA1(black_box(values.slice(..))),
                        KeyEq,
                        (0.0,),
                        Sum,
                    )
                    .unwrap();
                    sync(&exec);
                    black_box(output)
                })
            });
        }
    }
    reduce_by_key_group.finish();
}

criterion_group!(benches, bench_reduce);
criterion_main!(benches);
