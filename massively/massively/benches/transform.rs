mod common;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use cubecl::prelude::*;
use massively::{op::UnaryOp, vector::transform};

struct MulTwo;

#[cubecl::cube]
impl UnaryOp<f32> for MulTwo {
    type Output = f32;
    fn apply(input: f32) -> f32 {
        input * 2.0
    }
}

fn bench_transform(c: &mut Criterion) {
    let exec = common::exec();
    let mut group = c.benchmark_group("transform");
    for &len in common::SIZES {
        let input = exec.to_device(&common::dense_f32(len));
        exec.sync().unwrap();
        group.bench_function(BenchmarkId::new("gpu", len), |b| {
            b.iter(|| {
                black_box(transform(&exec, black_box(input.slice(..)), MulTwo).unwrap());
                exec.sync().unwrap();
            })
        });
    }
    group.finish();
}

criterion_group! { name = benches; config = common::criterion(); targets = bench_transform }
criterion_main!(benches);
