mod common;

use common::{Backend, SIZES, dense_f32, reverse_indices, sync};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use massively::{CubeWgpu, DeviceVec, Wgpu, gather};

fn check_gather(policy: &CubeWgpu) {
    let values = policy.to_device(&[10.0_f32, 20.0, 30.0, 40.0]).unwrap();
    let indices = policy.to_device(&[3_u32, 2, 1, 0]).unwrap();
    let (output,) = gather((values.slice(..),), (indices.slice(..),)).unwrap();
    assert_eq!(output.to_vec().unwrap(), vec![40.0, 30.0, 20.0, 10.0]);
}

fn bench_gather(c: &mut Criterion) {
    let mut group = c.benchmark_group("gather");
    for backend in Backend::available() {
        let policy = backend.policy();
        check_gather(&policy);

        for &len in SIZES {
            let values = policy.to_device(&dense_f32(len)).unwrap();
            let indices = policy.to_device(&reverse_indices(len)).unwrap();
            sync(&policy);
            group.bench_function(BenchmarkId::new(backend.name(), len), |b| {
                b.iter(|| {
                    let output: (DeviceVec<Wgpu, f32>,) = gather(
                        (black_box(values.slice(..)),),
                        (black_box(indices.slice(..)),),
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

criterion_group!(benches, bench_gather);
criterion_main!(benches);
