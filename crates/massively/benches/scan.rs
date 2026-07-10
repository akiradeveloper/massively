mod common;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use cubecl::prelude::*;
use massively::{
    BinaryPredicateOp, ReductionOp, exclusive_scan, exclusive_scan_by_key, inclusive_scan,
    inclusive_scan_by_key,
};

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

fn bench_scan(c: &mut Criterion) {
    let exec = common::exec();
    let mut group = c.benchmark_group("scan");
    for &len in common::SIZES {
        let values = exec.to_device(&common::dense_f32(len));
        let keys = exec.to_device(&common::run_keys(len, 8));
        let output = exec.alloc::<f32>(len);
        exec.sync().unwrap();
        group.bench_function(BenchmarkId::new("inclusive", len), |b| {
            b.iter(|| {
                inclusive_scan(&exec, values.slice(..), Sum, output.slice_mut(..)).unwrap();
                exec.sync().unwrap();
            })
        });
        group.bench_function(BenchmarkId::new("exclusive", len), |b| {
            b.iter(|| {
                exclusive_scan(&exec, values.slice(..), 0.0, Sum, output.slice_mut(..)).unwrap();
                exec.sync().unwrap();
            })
        });
        group.bench_function(BenchmarkId::new("inclusive_by_key", len), |b| {
            b.iter(|| {
                inclusive_scan_by_key(
                    &exec,
                    keys.slice(..),
                    values.slice(..),
                    Equal,
                    Sum,
                    output.slice_mut(..),
                )
                .unwrap();
                exec.sync().unwrap();
            })
        });
        group.bench_function(BenchmarkId::new("exclusive_by_key", len), |b| {
            b.iter(|| {
                exclusive_scan_by_key(
                    &exec,
                    keys.slice(..),
                    values.slice(..),
                    Equal,
                    0.0,
                    Sum,
                    output.slice_mut(..),
                )
                .unwrap();
                exec.sync().unwrap();
            })
        });
    }
    group.finish();
}
criterion_group! { name = benches; config = common::criterion(); targets = bench_scan }
criterion_main!(benches);
