mod common;

use common::{Backend, SIZES, dense_f32, sync};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use massively::op::UnaryOp;
use massively::{CubeWgpu, transform, unzip};

struct MulTwo;

#[cubecl::cube]
impl UnaryOp<f32> for MulTwo {
    type Output = f32;

    fn apply(input: f32) -> f32 {
        input * 2.0
    }
}

fn check_transform(policy: &CubeWgpu) {
    let values = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let output = unzip(transform(&values, MulTwo).unwrap()).unwrap();
    assert_eq!(output.to_vec().unwrap(), vec![2.0, 4.0, 6.0]);
}

fn bench_transform(c: &mut Criterion) {
    let mut transform_group = c.benchmark_group("transform");
    for backend in Backend::available() {
        let policy = backend.policy();
        check_transform(&policy);

        for &len in SIZES {
            let values = policy.to_device(&dense_f32(len)).unwrap();
            sync(&policy);
            transform_group.bench_function(BenchmarkId::new(backend.name(), len), |b| {
                b.iter(|| {
                    let output = transform(black_box(&values), MulTwo).unwrap();
                    sync(&policy);
                    black_box(output)
                })
            });
        }
    }
    transform_group.finish();
}

criterion_group!(benches, bench_transform);
criterion_main!(benches);
