mod common;

use common::{Backend, SIZES, dense_f32, reverse_indices, sync};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use cubecl::prelude::*;
use massively::op::UnaryOp;
use massively::{CubeWgpu, DeviceVec, Wgpu, scatter, transform};

struct MulTwo;

#[cubecl::cube]
impl UnaryOp<(f32,)> for MulTwo {
    type Output = (f32,);

    fn apply(input: (f32,)) -> (f32,) {
        (input.0 * 2.0,)
    }
}

fn check_scatter(policy: &CubeWgpu) {
    let values = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let indices = policy.to_device(&[3_u32, 2, 1, 0]).unwrap();
    let (output,) = scatter((values.slice(..),), (indices.slice(..),), 4, (0.0_f32,)).unwrap();
    assert_eq!(output.to_vec().unwrap(), vec![4.0, 3.0, 2.0, 1.0]);
}

fn bench_scatter(c: &mut Criterion) {
    let mut group = c.benchmark_group("scatter");
    for backend in Backend::available() {
        let policy = backend.policy();
        check_scatter(&policy);

        for &len in SIZES {
            let input = policy.to_device(&dense_f32(len)).unwrap();
            let indices = policy.to_device(&reverse_indices(len)).unwrap();
            let (values,) = transform((input.slice(..),), MulTwo).unwrap();
            sync(&policy);
            group.bench_function(BenchmarkId::new(backend.name(), len), |b| {
                b.iter(|| {
                    let output: (DeviceVec<Wgpu, f32>,) = scatter(
                        (black_box(values.slice(..)),),
                        (black_box(indices.slice(..)),),
                        len,
                        (0.0_f32,),
                    )
                    .unwrap();
                    sync(&policy);
                    black_box(output)
                })
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_scatter);
criterion_main!(benches);
