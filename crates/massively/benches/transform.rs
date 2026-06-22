mod common;

use common::{Backend, SIZES, dense_f32, sync};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use cubecl::prelude::*;
use massively::op::UnaryOp;
use massively::{DeviceVec, Executor, Wgpu, transform};

struct MulTwo;

#[cubecl::cube]
impl UnaryOp<Wgpu, (f32,)> for MulTwo {
    type Output = (f32,);

    fn apply(input: (f32,)) -> (f32,) {
        (input.0 * 2.0,)
    }
}

fn check_transform(exec: &Executor<Wgpu>) {
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let (output,) = transform(&exec, massively::SoA1(values.slice(..)), MulTwo).unwrap();
    assert_eq!(exec.to_host(&output).unwrap(), vec![2.0, 4.0, 6.0]);
}

fn bench_transform(c: &mut Criterion) {
    let mut transform_group = c.benchmark_group("transform");
    for backend in Backend::available() {
        let exec = backend.exec();
        check_transform(&exec);

        for &len in SIZES {
            let values = exec.to_device(&dense_f32(len)).unwrap();
            sync(&exec);
            transform_group.bench_function(BenchmarkId::new(backend.name(), len), |b| {
                b.iter(|| {
                    let output: (DeviceVec<Wgpu, f32>,) =
                        transform(&exec, massively::SoA1(black_box(values.slice(..))), MulTwo)
                            .unwrap();
                    sync(&exec);
                    black_box(output)
                })
            });
        }
    }
    transform_group.finish();
}

criterion_group!(benches, bench_transform);
criterion_main!(benches);
