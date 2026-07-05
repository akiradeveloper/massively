use cubecl::wgpu::WgpuRuntime;
mod common;

use common::{Runtime, SIZES, dense_f32, iter_gpu, reverse_indices, sync};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use cubecl::prelude::*;
use massively::op::UnaryOp;
use massively::{Executor, scatter, transform};

struct MulTwo;

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, (f32,)> for MulTwo {
    type Env = ();
    type Output = (f32,);

    fn apply(_env: (), input: (f32,)) -> (f32,) {
        (input.0 * 2.0,)
    }
}

fn check_scatter(exec: &Executor<WgpuRuntime>) {
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let indices = exec.to_device(&[3_u32, 2, 1, 0]).unwrap();
    let output = exec.to_device(&[0.0_f32; 4]).unwrap();
    scatter(
        &exec,
        massively::Zip1(values.slice(..)),
        indices.slice(..),
        massively::Zip1(output.slice_mut(..)),
    )
    .unwrap();
    assert_eq!(exec.to_host(&output).unwrap(), vec![4.0, 3.0, 2.0, 1.0]);
}

fn bench_scatter(c: &mut Criterion) {
    let mut group = c.benchmark_group("scatter");
    for backend in Runtime::available() {
        let exec = backend.exec();
        check_scatter(&exec);

        for &len in SIZES {
            let input = exec.to_device(&dense_f32(len)).unwrap();
            let indices = exec.to_device(&reverse_indices(len)).unwrap();
            let values = exec.to_device(&vec![0.0_f32; len]).unwrap();
            let output = exec.to_device(&vec![0.0_f32; len]).unwrap();
            transform(
                &exec,
                massively::Zip1(input.slice(..)),
                MulTwo,
                (),
                massively::Zip1(values.slice_mut(..)),
            )
            .unwrap();
            sync(&exec);
            group.bench_function(BenchmarkId::new(backend.name(), len), |b| {
                iter_gpu(b, || {
                    scatter(
                        &exec,
                        massively::Zip1(black_box(values.slice(..))),
                        black_box(indices.slice(..)),
                        massively::Zip1(output.slice_mut(..)),
                    )
                    .unwrap();
                    sync(&exec);
                    black_box(output.len())
                })
            });
        }
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = common::criterion();
    targets = bench_scatter
}
criterion_main!(benches);
