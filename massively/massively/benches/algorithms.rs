use std::time::Duration;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{
    Executor, op::BinaryPredicateOp, op::ReductionOp, op::UnaryOp, vector::copy_where,
    vector::gather, vector::inclusive_scan, vector::inclusive_scan_by_key, vector::reduce,
    vector::reduce_by_key, vector::scatter, vector::sort_by_key, vector::transform,
};

const SIZES: &[usize] = &[1_024, 16 * 1_024, 256 * 1_024, 1_024 * 1_024];
const SORT_SIZES: &[usize] = &[1_024, 16 * 1_024, 256 * 1_024];

struct MulTwo;
struct SumF32;
struct EqualU32;
struct LessU32;

#[cubecl::cube]
impl UnaryOp<f32> for MulTwo {
    type Output = f32;

    fn apply(input: f32) -> f32 {
        input * 2.0
    }
}

#[cubecl::cube]
impl ReductionOp<f32> for SumF32 {
    fn apply(lhs: f32, rhs: f32) -> f32 {
        lhs + rhs
    }
}

#[cubecl::cube]
impl BinaryPredicateOp<u32> for EqualU32 {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs == rhs
    }
}

#[cubecl::cube]
impl BinaryPredicateOp<u32> for LessU32 {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs < rhs
    }
}

fn dense_f32(len: usize) -> Vec<f32> {
    (0..len).map(|index| (index % 251) as f32).collect()
}

fn shuffled_u32(len: usize) -> Vec<u32> {
    (0..len)
        .map(|index| ((index * 1_103_515_245 + 12_345) % len.max(1)) as u32)
        .collect()
}

fn bench_linear_algorithms(c: &mut Criterion) {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let mut group = c.benchmark_group("linear_algorithms");

    for &len in SIZES {
        let values = exec.to_device(&dense_f32(len));
        let output = exec.alloc::<f32>(len);
        let flags = exec.to_device(
            &(0..len)
                .map(|index| (index % 2 == 0) as u32)
                .collect::<Vec<_>>(),
        );
        let indices = exec.to_device(&(0..len).rev().map(|index| index as u32).collect::<Vec<_>>());
        exec.sync().unwrap();

        group.bench_function(BenchmarkId::new("transform", len), |b| {
            b.iter(|| {
                black_box(transform(&exec, black_box(values.slice(..)), MulTwo).unwrap());
                exec.sync().unwrap();
            })
        });
        group.bench_function(BenchmarkId::new("reduce", len), |b| {
            b.iter(|| black_box(reduce(&exec, values.slice(..), 0.0, SumF32).unwrap()))
        });
        group.bench_function(BenchmarkId::new("inclusive_scan", len), |b| {
            b.iter(|| {
                black_box(inclusive_scan(&exec, values.slice(..), SumF32).unwrap());
                exec.sync().unwrap();
            })
        });
        group.bench_function(BenchmarkId::new("copy_where_half", len), |b| {
            b.iter(|| {
                black_box(copy_where(&exec, values.slice(..), flags.slice(..)).unwrap());
            })
        });
        group.bench_function(BenchmarkId::new("gather_reverse", len), |b| {
            b.iter(|| {
                black_box(gather(&exec, values.slice(..), indices.slice(..)).unwrap());
                exec.sync().unwrap();
            })
        });
        group.bench_function(BenchmarkId::new("scatter_reverse", len), |b| {
            b.iter(|| {
                scatter(
                    &exec,
                    values.slice(..),
                    indices.slice(..),
                    output.slice_mut(..),
                )
                .unwrap();
                exec.sync().unwrap();
            })
        });
    }
    group.finish();
}

fn bench_ordering_and_by_key(c: &mut Criterion) {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let mut group = c.benchmark_group("ordering_and_by_key");

    for &len in SORT_SIZES {
        let keys = exec.to_device(&shuffled_u32(len));
        let values = exec.to_device(&dense_f32(len));
        exec.sync().unwrap();
        group.bench_function(BenchmarkId::new("sort_by_key", len), |b| {
            b.iter(|| {
                black_box(sort_by_key(&exec, keys.slice(..), values.slice(..), LessU32).unwrap());
                exec.sync().unwrap();
            })
        });
    }

    for &len in SIZES {
        let keys = exec.to_device(&(0..len).map(|index| (index / 8) as u32).collect::<Vec<_>>());
        let values = exec.to_device(&dense_f32(len));
        exec.sync().unwrap();
        group.bench_function(BenchmarkId::new("inclusive_scan_by_key", len), |b| {
            b.iter(|| {
                black_box(
                    inclusive_scan_by_key(
                        &exec,
                        keys.slice(..),
                        values.slice(..),
                        EqualU32,
                        SumF32,
                    )
                    .unwrap(),
                );
                exec.sync().unwrap();
            })
        });
        group.bench_function(BenchmarkId::new("reduce_by_key", len), |b| {
            b.iter(|| {
                black_box(
                    reduce_by_key(
                        &exec,
                        keys.slice(..),
                        values.slice(..),
                        EqualU32,
                        0.0,
                        SumF32,
                    )
                    .unwrap(),
                );
            })
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(100))
        .measurement_time(Duration::from_millis(250));
    targets = bench_linear_algorithms, bench_ordering_and_by_key
}
criterion_main!(benches);
