use cubecl::wgpu::WgpuRuntime;
mod common;

use common::{Runtime, SORT_SIZES, descending_f32, iter_gpu, shuffled_u32, sync};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use cubecl::prelude::*;
use massively::op::BinaryPredicateOp;
use massively::{Executor, SoA1, sort_by_key};

struct Less;

#[cubecl::cube]
impl BinaryPredicateOp<WgpuRuntime, (u32,)> for Less {
    fn apply(lhs: (u32,), rhs: (u32,)) -> bool {
        lhs.0 < rhs.0
    }
}

fn check_sort_by_key(exec: &Executor<WgpuRuntime>) {
    let keys = exec.to_device(&[2_u32, 0, 1]).unwrap();
    let values = exec.to_device(&[20.0_f32, 0.0, 10.0]).unwrap();
    let (SoA1(keys), SoA1(values)) = sort_by_key(
        &exec,
        massively::SoA1(keys.slice(..)),
        massively::SoA1(values.slice(..)),
        Less,
    )
    .unwrap();
    assert_eq!(exec.to_host(&keys).unwrap(), vec![0, 1, 2]);
    assert_eq!(exec.to_host(&values).unwrap(), vec![0.0, 10.0, 20.0]);
}

fn bench_sort(c: &mut Criterion) {
    let mut group = c.benchmark_group("sort_by_key");
    for backend in Runtime::available() {
        let exec = backend.exec();
        check_sort_by_key(&exec);

        for &len in SORT_SIZES {
            let keys = exec.to_device(&shuffled_u32(len)).unwrap();
            let values = exec.to_device(&descending_f32(len)).unwrap();
            sync(&exec);
            group.bench_function(BenchmarkId::new(backend.name(), len), |b| {
                iter_gpu(b, || {
                    let output = sort_by_key(
                        &exec,
                        massively::SoA1(black_box(keys.slice(..))),
                        massively::SoA1(black_box(values.slice(..))),
                        Less,
                    )
                    .unwrap();
                    sync(&exec);
                    black_box(output)
                })
            });
        }
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = common::criterion();
    targets = bench_sort
}
criterion_main!(benches);
