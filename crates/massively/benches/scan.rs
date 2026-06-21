mod common;

use common::{Backend, SIZES, dense_f32, sync};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use cubecl::prelude::*;
use massively::op::{BinaryOp, BinaryPredicateOp};
use massively::{
    DeviceVec, Executor, Wgpu, exclusive_scan_by_key, inclusive_scan, inclusive_scan_by_key,
};

struct Sum;

#[cubecl::cube]
impl BinaryOp<(f32,)> for Sum {
    fn apply(lhs: (f32,), rhs: (f32,)) -> (f32,) {
        (lhs.0 + rhs.0,)
    }
}

struct KeyEq;

#[cubecl::cube]
impl BinaryPredicateOp<(u32,)> for KeyEq {
    fn apply(lhs: (u32,), rhs: (u32,)) -> bool {
        lhs.0 == rhs.0
    }
}

fn keys(len: usize) -> Vec<u32> {
    (0..len).map(|index| (index / 8) as u32).collect()
}

fn check_scan(exec: &Executor<Wgpu>) {
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let (output,) = inclusive_scan(&exec, (values.slice(..),), Sum).unwrap();
    assert_eq!(exec.to_host(&output).unwrap(), vec![1.0, 3.0, 6.0, 10.0]);
}

fn check_scan_by_key(exec: &Executor<Wgpu>) {
    let keys = exec.to_device(&[0_u32, 0, 1, 1]).unwrap();
    let values = exec.to_device(&[1.0_f32, 2.0, 10.0, 20.0]).unwrap();
    let (output,) =
        inclusive_scan_by_key(&exec, (keys.slice(..),), (values.slice(..),), KeyEq, Sum).unwrap();
    assert_eq!(exec.to_host(&output).unwrap(), vec![1.0, 3.0, 10.0, 30.0]);

    let (output,) = exclusive_scan_by_key(
        &exec,
        (keys.slice(..),),
        (values.slice(..),),
        KeyEq,
        (0.0,),
        Sum,
    )
    .unwrap();
    assert_eq!(exec.to_host(&output).unwrap(), vec![0.0, 1.0, 0.0, 10.0]);
}

fn bench_scan(c: &mut Criterion) {
    let mut scan_group = c.benchmark_group("inclusive_scan");
    for backend in Backend::available() {
        let exec = backend.exec();
        check_scan(&exec);

        for &len in SIZES {
            let values = exec.to_device(&dense_f32(len)).unwrap();
            sync(&exec);
            scan_group.bench_function(BenchmarkId::new(backend.name(), len), |b| {
                b.iter(|| {
                    let output: (DeviceVec<Wgpu, f32>,) =
                        inclusive_scan(&exec, (black_box(values.slice(..)),), Sum).unwrap();
                    sync(&exec);
                    black_box(output)
                })
            });
        }
    }
    scan_group.finish();

    let mut by_key_group = c.benchmark_group("inclusive_scan_by_key");
    for backend in Backend::available() {
        let exec = backend.exec();
        check_scan_by_key(&exec);

        for &len in SIZES {
            let keys = exec.to_device(&keys(len)).unwrap();
            let values = exec.to_device(&dense_f32(len)).unwrap();
            sync(&exec);
            by_key_group.bench_function(BenchmarkId::new(backend.name(), len), |b| {
                b.iter(|| {
                    let output: (DeviceVec<Wgpu, f32>,) = inclusive_scan_by_key(
                        &exec,
                        (black_box(keys.slice(..)),),
                        (black_box(values.slice(..)),),
                        KeyEq,
                        Sum,
                    )
                    .unwrap();
                    sync(&exec);
                    black_box(output)
                })
            });
        }
    }
    by_key_group.finish();

    let mut exclusive_by_key_group = c.benchmark_group("exclusive_scan_by_key");
    for backend in Backend::available() {
        let exec = backend.exec();
        check_scan_by_key(&exec);

        for &len in SIZES {
            let keys = exec.to_device(&keys(len)).unwrap();
            let values = exec.to_device(&dense_f32(len)).unwrap();
            sync(&exec);
            exclusive_by_key_group.bench_function(BenchmarkId::new(backend.name(), len), |b| {
                b.iter(|| {
                    let output: (DeviceVec<Wgpu, f32>,) = exclusive_scan_by_key(
                        &exec,
                        (black_box(keys.slice(..)),),
                        (black_box(values.slice(..)),),
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
    exclusive_by_key_group.finish();
}

criterion_group!(benches, bench_scan);
criterion_main!(benches);
