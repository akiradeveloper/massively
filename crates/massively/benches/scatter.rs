mod common;

use common::{Backend, SIZES, dense_f32, reverse_indices, sync};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use cubecl::prelude::*;
use massively::op::UnaryOp;
use massively::{DeviceVec, Executor, Wgpu, scatter, transform};

struct MulTwo;

#[cubecl::cube]
impl UnaryOp<Wgpu, (f32,)> for MulTwo {
    type Output = (f32,);

    fn apply(input: (f32,)) -> (f32,) {
        (input.0 * 2.0,)
    }
}

fn check_scatter(exec: &Executor<Wgpu>) {
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let indices = exec.to_device(&[3_u32, 2, 1, 0]).unwrap();
    let (output,) = scatter(
        &exec,
        massively::SoA1(values.slice(..)),
        indices.slice(..),
        4,
        (0.0_f32,),
    )
    .unwrap();
    assert_eq!(exec.to_host(&output).unwrap(), vec![4.0, 3.0, 2.0, 1.0]);
}

fn bench_scatter(c: &mut Criterion) {
    let mut group = c.benchmark_group("scatter");
    for backend in Backend::available() {
        let exec = backend.exec();
        check_scatter(&exec);

        for &len in SIZES {
            let input = exec.to_device(&dense_f32(len)).unwrap();
            let indices = exec.to_device(&reverse_indices(len)).unwrap();
            let (values,) = transform(&exec, massively::SoA1(input.slice(..)), MulTwo).unwrap();
            sync(&exec);
            group.bench_function(BenchmarkId::new(backend.name(), len), |b| {
                b.iter(|| {
                    let output: (DeviceVec<Wgpu, f32>,) = scatter(
                        &exec,
                        massively::SoA1(black_box(values.slice(..))),
                        black_box(indices.slice(..)),
                        len,
                        (0.0_f32,),
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

criterion_group!(benches, bench_scatter);
criterion_main!(benches);
