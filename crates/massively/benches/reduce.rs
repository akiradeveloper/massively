mod common;

use common::{Backend, SIZES, dense_f32, sync};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use massively::op::{BinaryOp, BinaryPredicateOp};
use massively::{CubeWgpu, reduce, reduce_by_key};

struct Sum;

#[cubecl::cube]
impl BinaryOp<f32> for Sum {
    fn apply(lhs: f32, rhs: f32) -> f32 {
        lhs + rhs
    }
}

struct KeyEq;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for KeyEq {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs == rhs
    }
}

fn keys(len: usize) -> Vec<u32> {
    (0..len).map(|index| (index / 8) as u32).collect()
}

fn check_reduce(policy: &CubeWgpu) {
    let values = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let output = reduce(&values, 0.0, Sum).unwrap();
    assert_eq!(output, 10.0);
}

fn check_reduce_by_key(policy: &CubeWgpu) {
    let keys = policy.to_device(&[0_u32, 0, 1, 1]).unwrap();
    let values = policy.to_device(&[1.0_f32, 2.0, 10.0, 20.0]).unwrap();
    let (out_keys, out_values) = reduce_by_key(&keys, &values, KeyEq, 0.0, Sum).unwrap();
    assert_eq!(out_keys.to_vec().unwrap(), vec![0, 1]);
    assert_eq!(out_values.to_vec().unwrap(), vec![3.0, 30.0]);
}

fn bench_reduce(c: &mut Criterion) {
    let mut reduce_group = c.benchmark_group("reduce");
    for backend in Backend::available() {
        let policy = backend.policy();
        check_reduce(&policy);

        for &len in SIZES {
            let values = policy.to_device(&dense_f32(len)).unwrap();
            sync(&policy);
            reduce_group.bench_function(BenchmarkId::new(backend.name(), len), |b| {
                b.iter(|| {
                    let output = reduce(black_box(&values), 0.0, Sum).unwrap();
                    sync(&policy);
                    black_box(output)
                })
            });
        }
    }
    reduce_group.finish();

    let mut reduce_by_key_group = c.benchmark_group("reduce_by_key");
    for backend in Backend::available() {
        let policy = backend.policy();
        check_reduce_by_key(&policy);

        for &len in SIZES {
            let keys = policy.to_device(&keys(len)).unwrap();
            let values = policy.to_device(&dense_f32(len)).unwrap();
            sync(&policy);
            reduce_by_key_group.bench_function(BenchmarkId::new(backend.name(), len), |b| {
                b.iter(|| {
                    let output =
                        reduce_by_key(black_box(&keys), black_box(&values), KeyEq, 0.0, Sum)
                            .unwrap();
                    sync(&policy);
                    black_box(output)
                })
            });
        }
    }
    reduce_by_key_group.finish();
}

criterion_group!(benches, bench_reduce);
criterion_main!(benches);
