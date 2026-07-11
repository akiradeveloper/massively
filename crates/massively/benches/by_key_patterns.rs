mod common;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use cubecl::prelude::*;
use massively::{
    op::BinaryPredicateOp, op::ReductionOp, vector::exclusive_scan_by_key,
    vector::inclusive_scan_by_key, vector::reduce_by_key, vector::unique_by_key,
};

struct Equal;
struct Sum;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for Equal {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs == rhs
    }
}

#[cubecl::cube]
impl ReductionOp<f32> for Sum {
    fn apply(lhs: f32, rhs: f32) -> f32 {
        lhs + rhs
    }
}

fn key_patterns(len: usize) -> [(String, Vec<u32>); 4] {
    [
        ("run1".into(), common::run_keys(len, 1)),
        ("run8".into(), common::run_keys(len, 8)),
        ("run128".into(), common::run_keys(len, 128)),
        ("all".into(), common::run_keys(len, len)),
    ]
}

fn bench_by_key_patterns(c: &mut Criterion) {
    let exec = common::exec();
    let mut inclusive = c.benchmark_group("inclusive_scan_by_key_patterns");
    for &len in common::SIZES {
        let values = exec.to_device(&common::dense_f32(len));
        let output = exec.alloc::<f32>(len);
        for (pattern, host_keys) in key_patterns(len) {
            let keys = exec.to_device(&host_keys);
            inclusive.bench_function(BenchmarkId::new(pattern, len), |b| {
                b.iter(|| {
                    inclusive_scan_by_key(
                        &exec,
                        black_box(keys.slice(..)),
                        black_box(values.slice(..)),
                        Equal,
                        Sum,
                        black_box(output.slice_mut(..)),
                    )
                    .unwrap();
                    exec.sync().unwrap();
                    black_box(&output);
                })
            });
        }
    }
    inclusive.finish();

    let mut exclusive = c.benchmark_group("exclusive_scan_by_key_patterns");
    for &len in common::SIZES {
        let values = exec.to_device(&common::dense_f32(len));
        let output = exec.alloc::<f32>(len);
        for (pattern, host_keys) in key_patterns(len) {
            let keys = exec.to_device(&host_keys);
            exclusive.bench_function(BenchmarkId::new(pattern, len), |b| {
                b.iter(|| {
                    exclusive_scan_by_key(
                        &exec,
                        black_box(keys.slice(..)),
                        black_box(values.slice(..)),
                        Equal,
                        0.0,
                        Sum,
                        black_box(output.slice_mut(..)),
                    )
                    .unwrap();
                    exec.sync().unwrap();
                    black_box(&output);
                })
            });
        }
    }
    exclusive.finish();

    let mut reduce = c.benchmark_group("reduce_by_key_patterns");
    for &len in common::SIZES {
        let values = exec.to_device(&common::dense_f32(len));
        let out_keys = exec.alloc::<u32>(len);
        let out_values = exec.alloc::<f32>(len);
        for (pattern, host_keys) in key_patterns(len) {
            let keys = exec.to_device(&host_keys);
            reduce.bench_function(BenchmarkId::new(pattern, len), |b| {
                b.iter(|| {
                    let output_len = reduce_by_key(
                        &exec,
                        black_box(keys.slice(..)),
                        black_box(values.slice(..)),
                        Equal,
                        0.0,
                        Sum,
                        black_box(out_keys.slice_mut(..)),
                        black_box(out_values.slice_mut(..)),
                    )
                    .unwrap();
                    exec.sync().unwrap();
                    black_box((output_len, &out_keys, &out_values));
                })
            });
        }
    }
    reduce.finish();

    let mut unique = c.benchmark_group("unique_by_key_patterns");
    for &len in common::SIZES {
        let values = exec.to_device(&(0..len as u32).collect::<Vec<_>>());
        let out_keys = exec.alloc::<u32>(len);
        let out_values = exec.alloc::<u32>(len);
        for (pattern, host_keys) in key_patterns(len) {
            let keys = exec.to_device(&host_keys);
            unique.bench_function(BenchmarkId::new(pattern, len), |b| {
                b.iter(|| {
                    let output_len = unique_by_key(
                        &exec,
                        black_box(keys.slice(..)),
                        black_box(values.slice(..)),
                        Equal,
                        black_box(out_keys.slice_mut(..)),
                        black_box(out_values.slice_mut(..)),
                    )
                    .unwrap();
                    exec.sync().unwrap();
                    black_box((output_len, &out_keys, &out_values));
                })
            });
        }
    }
    unique.finish();
}

criterion_group! {
    name = benches;
    config = common::criterion();
    targets = bench_by_key_patterns
}
criterion_main!(benches);
