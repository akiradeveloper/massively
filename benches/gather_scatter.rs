mod common;

use common::{Backend, SIZES, dense_f32, reverse_indices, sync};
use criterion::{BatchSize, BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use massively::op::UnaryOp;
use massively::{CubeWgpu, gather, scatter, transform};

struct MulTwo;

#[cubecl::cube]
impl UnaryOp<f32> for MulTwo {
    type Output = f32;

    fn apply(input: f32) -> f32 {
        input * 2.0
    }
}

fn check_gather(policy: &CubeWgpu) {
    let values = policy.to_device(&[10.0_f32, 20.0, 30.0, 40.0]).unwrap();
    let indices = policy.to_device(&[3_u32, 2, 1, 0]).unwrap();
    let output = gather(&values, &indices).unwrap();
    assert_eq!(output.to_vec().unwrap(), vec![40.0, 30.0, 20.0, 10.0]);
}

fn check_scatter(policy: &CubeWgpu) {
    let values = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let indices = policy.to_device(&[3_u32, 2, 1, 0]).unwrap();
    let initial = policy.device_filled(4, 0.0_f32).unwrap();
    let output = scatter(&values, &indices, initial).unwrap();
    assert_eq!(output.to_vec().unwrap(), vec![4.0, 3.0, 2.0, 1.0]);
}

fn bench_gather_scatter(c: &mut Criterion) {
    let mut gather_group = c.benchmark_group("gather");
    for backend in Backend::available() {
        let policy = backend.policy();
        check_gather(&policy);

        for &len in SIZES {
            let values = policy.to_device(&dense_f32(len)).unwrap();
            let indices = policy.to_device(&reverse_indices(len)).unwrap();
            sync(&policy);
            gather_group.bench_function(BenchmarkId::new(backend.name(), len), |b| {
                b.iter(|| {
                    let output = gather(black_box(&values), black_box(&indices)).unwrap();
                    sync(&policy);
                    black_box(output)
                })
            });
        }
    }
    gather_group.finish();

    let mut scatter_group = c.benchmark_group("scatter");
    for backend in Backend::available() {
        let policy = backend.policy();
        check_scatter(&policy);

        for &len in SIZES {
            let input = policy.to_device(&dense_f32(len)).unwrap();
            let indices = policy.to_device(&reverse_indices(len)).unwrap();
            let values = transform(&input, MulTwo).unwrap();
            sync(&policy);
            scatter_group.bench_function(BenchmarkId::new(backend.name(), len), |b| {
                b.iter_batched(
                    || {
                        let initial = policy.device_filled(len, 0.0_f32).unwrap();
                        sync(&policy);
                        initial
                    },
                    |initial| {
                        let output =
                            scatter(black_box(&values), black_box(&indices), initial).unwrap();
                        sync(&policy);
                        black_box(output)
                    },
                    BatchSize::SmallInput,
                )
            });
        }
    }
    scatter_group.finish();
}

criterion_group!(benches, bench_gather_scatter);
criterion_main!(benches);
