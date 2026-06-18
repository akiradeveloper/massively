mod common;

use common::{Backend, SORT_SIZES, descending_f32, shuffled_u32, sync};
use criterion::{BatchSize, BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use massively::op::BinaryPredicateOp;
use massively::{CubeWgpu, sort_by_key};

struct Less;

#[cubecl::cube]
impl BinaryPredicateOp<f32> for Less {
    fn apply(lhs: f32, rhs: f32) -> bool {
        lhs < rhs
    }
}

#[cubecl::cube]
impl BinaryPredicateOp<u32> for Less {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs < rhs
    }
}

fn check_sort_by_key(policy: &CubeWgpu) {
    let keys = policy.to_device(&[2_u32, 0, 1]).unwrap();
    let values = policy.to_device(&[20.0_f32, 0.0, 10.0]).unwrap();
    let (keys, values) = sort_by_key(&keys, &values, Less).unwrap();
    assert_eq!(keys.to_vec().unwrap(), vec![0, 1, 2]);
    assert_eq!(values.to_vec().unwrap(), vec![0.0, 10.0, 20.0]);
}

fn bench_sort(c: &mut Criterion) {
    let mut group = c.benchmark_group("sort_by_key");
    for backend in Backend::available() {
        let policy = backend.policy();
        check_sort_by_key(&policy);

        for &len in SORT_SIZES {
            let host_keys = shuffled_u32(len);
            let host_values = descending_f32(len);
            group.bench_function(BenchmarkId::new(backend.name(), len), |b| {
                b.iter_batched(
                    || {
                        let input = (
                            policy.to_device(&host_keys).unwrap(),
                            policy.to_device(&host_values).unwrap(),
                        );
                        sync(&policy);
                        input
                    },
                    |(keys, values)| {
                        let output = sort_by_key(&keys, &values, Less).unwrap();
                        sync(&policy);
                        black_box(output)
                    },
                    BatchSize::SmallInput,
                )
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_sort);
criterion_main!(benches);
