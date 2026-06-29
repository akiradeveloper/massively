use cubecl::wgpu::WgpuRuntime;
mod common;

use common::{Runtime, SIZES, ascending_u32, iter_gpu, sync};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use cubecl::prelude::*;
use massively::op::BinaryPredicateOp;
use massively::{Executor, SoA1, lower_bound, upper_bound};

struct Less;

#[cubecl::cube]
impl BinaryPredicateOp<WgpuRuntime, (u32,)> for Less {
    fn apply(lhs: (u32,), rhs: (u32,)) -> bool {
        lhs.0 < rhs.0
    }
}

fn query_u32(len: usize) -> Vec<u32> {
    (0..len)
        .map(|index| ((index * 1_103_515_245 + 12_345) % (len.max(1) * 2)) as u32)
        .collect()
}

fn check_search(exec: &Executor<WgpuRuntime>) {
    let input = exec.to_device(&[0_u32, 0, 2, 2, 2]).unwrap();
    let values = exec.to_device(&[0_u32, 1, 2]).unwrap();
    let lower = lower_bound(exec, SoA1(input.slice(..)), SoA1(values.slice(..)), Less).unwrap();
    assert_eq!(exec.to_host(&lower).unwrap(), vec![0, 2, 2]);
    let upper = upper_bound(exec, SoA1(input.slice(..)), SoA1(values.slice(..)), Less).unwrap();
    assert_eq!(exec.to_host(&upper).unwrap(), vec![2, 2, 5]);
}

fn bench_search(c: &mut Criterion) {
    let mut lower_group = c.benchmark_group("lower_bound");
    for backend in Runtime::available() {
        let exec = backend.exec();
        check_search(&exec);

        for &len in SIZES {
            let input = exec.to_device(&ascending_u32(len)).unwrap();
            let values = exec.to_device(&query_u32(len)).unwrap();
            sync(&exec);
            lower_group.bench_function(BenchmarkId::new(backend.name(), len), |b| {
                iter_gpu(b, || {
                    let output = lower_bound(
                        &exec,
                        SoA1(black_box(input.slice(..))),
                        SoA1(black_box(values.slice(..))),
                        Less,
                    )
                    .unwrap();
                    sync(&exec);
                    black_box(output)
                })
            });
        }
    }
    lower_group.finish();

    let mut upper_group = c.benchmark_group("upper_bound");
    for backend in Runtime::available() {
        let exec = backend.exec();
        check_search(&exec);

        for &len in SIZES {
            let input = exec.to_device(&ascending_u32(len)).unwrap();
            let values = exec.to_device(&query_u32(len)).unwrap();
            sync(&exec);
            upper_group.bench_function(BenchmarkId::new(backend.name(), len), |b| {
                iter_gpu(b, || {
                    let output = upper_bound(
                        &exec,
                        SoA1(black_box(input.slice(..))),
                        SoA1(black_box(values.slice(..))),
                        Less,
                    )
                    .unwrap();
                    sync(&exec);
                    black_box(output)
                })
            });
        }
    }
    upper_group.finish();
}

criterion_group! {
    name = benches;
    config = common::criterion();
    targets = bench_search
}
criterion_main!(benches);
