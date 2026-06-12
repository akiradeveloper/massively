mod common;

use common::{Backend, SIZES, sync};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use massively::op::PredicateOp;
use massively::{CubeWgpu, copy_if, unzip};

struct Positive;

#[cubecl::cube]
impl PredicateOp<f32> for Positive {
    fn apply(input: f32) -> bool {
        input > 0.0
    }
}

fn alternating_signed(len: usize) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let value = (index % 251) as f32 + 1.0;
            if index % 2 == 0 { value } else { -value }
        })
        .collect()
}

fn check_copy_if(policy: &CubeWgpu) {
    let values = policy.to_device(&[-1.0_f32, 2.0, -3.0, 4.0]).unwrap();
    let output = unzip(copy_if(&values, Positive).unwrap()).unwrap();
    assert_eq!(output.to_vec().unwrap(), vec![2.0, 4.0]);
}

fn bench_select(c: &mut Criterion) {
    let mut copy_group = c.benchmark_group("copy_if");
    for backend in Backend::available() {
        let policy = backend.policy();
        check_copy_if(&policy);

        for &len in SIZES {
            let values = policy.to_device(&alternating_signed(len)).unwrap();
            sync(&policy);
            copy_group.bench_function(BenchmarkId::new(backend.name(), len), |b| {
                b.iter(|| {
                    let output = copy_if(black_box(&values), Positive).unwrap();
                    sync(&policy);
                    black_box(output)
                })
            });
        }
    }
    copy_group.finish();
}

criterion_group!(benches, bench_select);
criterion_main!(benches);
