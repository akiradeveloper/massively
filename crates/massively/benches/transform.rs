use cubecl::wgpu::WgpuRuntime;
mod common;

use common::{Runtime, SIZES, dense_f32, iter_gpu, sync};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use cubecl::prelude::*;
use massively::op::UnaryOp;
use massively::{Executor, transform};

struct MulTwo;

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, (f32,)> for MulTwo {
    type Output = (f32,);

    fn apply(input: (f32,)) -> (f32,) {
        (input.0 * 2.0,)
    }
}

fn check_transform(exec: &Executor<WgpuRuntime>) {
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let output = exec.to_device(&[0.0_f32; 3]).unwrap();
    transform(
        exec,
        massively::SoA1(values.slice(..)),
        MulTwo,
        massively::SoA1(output.slice_mut(..)),
    )
    .unwrap();
    assert_eq!(exec.to_host(&output).unwrap(), vec![2.0, 4.0, 6.0]);
}

fn bench_transform(c: &mut Criterion) {
    let mut transform_group = c.benchmark_group("transform");
    for backend in Runtime::available() {
        let exec = backend.exec();
        check_transform(&exec);

        for &len in SIZES {
            let values = exec.to_device(&dense_f32(len)).unwrap();
            let output = exec.to_device(&vec![0.0_f32; len]).unwrap();
            sync(&exec);
            transform_group.bench_function(BenchmarkId::new(backend.name(), len), |b| {
                iter_gpu(b, || {
                    transform(
                        &exec,
                        massively::SoA1(black_box(values.slice(..))),
                        MulTwo,
                        massively::SoA1(output.slice_mut(..)),
                    )
                    .unwrap();
                    sync(&exec);
                    black_box(output.len())
                })
            });
        }
    }
    transform_group.finish();
}

criterion_group! {
    name = benches;
    config = common::criterion();
    targets = bench_transform
}
criterion_main!(benches);
