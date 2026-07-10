mod common;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use cubecl::prelude::*;
use massively::{BinaryPredicateOp, ReductionOp, reduce, reduce_by_key};

struct Sum;
struct Equal;
#[cubecl::cube]
impl ReductionOp<f32> for Sum {
    fn apply(lhs: f32, rhs: f32) -> f32 {
        lhs + rhs
    }
}
#[cubecl::cube]
impl BinaryPredicateOp<u32> for Equal {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs == rhs
    }
}

fn bench_reduce(c: &mut Criterion) {
    let exec = common::exec();
    let mut group = c.benchmark_group("reduce");
    for &len in common::SIZES {
        let values = exec.to_device(&common::dense_f32(len));
        let keys = exec.to_device(&common::run_keys(len, 8));
        let out_keys = exec.alloc::<u32>(len);
        let out_values = exec.alloc::<f32>(len);
        exec.sync().unwrap();
        group.bench_function(BenchmarkId::new("reduce", len), |b| {
            b.iter(|| black_box(reduce(&exec, values.slice(..), 0.0, Sum).unwrap()))
        });
        group.bench_function(BenchmarkId::new("reduce_by_key", len), |b| {
            b.iter(|| {
                black_box(
                    reduce_by_key(
                        &exec,
                        keys.slice(..),
                        values.slice(..),
                        Equal,
                        0.0,
                        Sum,
                        out_keys.slice_mut(..),
                        out_values.slice_mut(..),
                    )
                    .unwrap(),
                )
            })
        });
    }
    group.finish();
}
criterion_group! { name = benches; config = common::criterion(); targets = bench_reduce }
criterion_main!(benches);
