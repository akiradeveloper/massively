use cubecl::wgpu::WgpuRuntime;
mod common;

use common::{
    Runtime, SIZES, SORT_SIZES, ascending_u32, descending_u32, even_u32, iter_gpu, odd_u32, sync,
};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use cubecl::prelude::*;
use massively::op::BinaryPredicateOp;
use massively::{Executor, merge, set_difference, set_intersection, set_union, sort};

struct Less;

#[cubecl::cube]
impl BinaryPredicateOp<WgpuRuntime, (u32,)> for Less {
    fn apply(lhs: (u32,), rhs: (u32,)) -> bool {
        lhs.0 < rhs.0
    }
}

fn shifted_u32(len: usize, offset: usize) -> Vec<u32> {
    (0..len).map(|index| (index + offset) as u32).collect()
}

fn check_ordering(exec: &Executor<WgpuRuntime>) {
    let values = exec.to_device(&[3_u32, 1, 2]).unwrap();
    let sorted = exec.to_device(&[0_u32; 3]).unwrap();
    sort(
        exec,
        massively::Zip1(values.slice(..)),
        Less,
        massively::Zip1(sorted.slice_mut(..)),
    )
    .unwrap();
    assert_eq!(exec.to_host(&sorted).unwrap(), vec![1, 2, 3]);

    let left = exec.to_device(&[1_u32, 3]).unwrap();
    let right = exec.to_device(&[2_u32, 4]).unwrap();
    let merged = exec.to_device(&[0_u32; 4]).unwrap();
    merge(
        exec,
        massively::Zip1(left.slice(..)),
        massively::Zip1(right.slice(..)),
        Less,
        massively::Zip1(merged.slice_mut(..)),
    )
    .unwrap();
    assert_eq!(exec.to_host(&merged).unwrap(), vec![1, 2, 3, 4]);
}

fn bench_ordering(c: &mut Criterion) {
    let mut sort_group = c.benchmark_group("sort");
    for backend in Runtime::available() {
        let exec = backend.exec();
        check_ordering(&exec);

        for &len in SORT_SIZES {
            let values = exec.to_device(&descending_u32(len)).unwrap();
            let output = exec.to_device(&vec![0_u32; len]).unwrap();
            sync(&exec);
            sort_group.bench_function(BenchmarkId::new(backend.name(), len), |b| {
                iter_gpu(b, || {
                    sort(
                        &exec,
                        massively::Zip1(black_box(values.slice(..))),
                        Less,
                        massively::Zip1(black_box(output.slice_mut(..))),
                    )
                    .unwrap();
                    sync(&exec);
                    black_box(&output)
                })
            });
        }
    }
    sort_group.finish();

    let mut merge_group = c.benchmark_group("merge");
    for backend in Runtime::available() {
        let exec = backend.exec();
        check_ordering(&exec);

        for &len in SIZES {
            let left = exec.to_device(&even_u32(len)).unwrap();
            let right = exec.to_device(&odd_u32(len)).unwrap();
            let output = exec.to_device(&vec![0_u32; len * 2]).unwrap();
            sync(&exec);
            merge_group.bench_function(BenchmarkId::new(backend.name(), len), |b| {
                iter_gpu(b, || {
                    merge(
                        &exec,
                        massively::Zip1(black_box(left.slice(..))),
                        massively::Zip1(black_box(right.slice(..))),
                        Less,
                        massively::Zip1(black_box(output.slice_mut(..))),
                    )
                    .unwrap();
                    sync(&exec);
                    black_box(&output)
                })
            });
        }
    }
    merge_group.finish();

    let mut union_group = c.benchmark_group("set_union");
    for backend in Runtime::available() {
        let exec = backend.exec();
        check_ordering(&exec);

        for &len in SIZES {
            let left = exec.to_device(&ascending_u32(len)).unwrap();
            let right = exec.to_device(&shifted_u32(len, len / 2)).unwrap();
            let output = exec.to_device(&vec![0_u32; len * 2]).unwrap();
            sync(&exec);
            union_group.bench_function(BenchmarkId::new(backend.name(), len), |b| {
                iter_gpu(b, || {
                    let output_len = set_union(
                        &exec,
                        massively::Zip1(black_box(left.slice(..))),
                        massively::Zip1(black_box(right.slice(..))),
                        Less,
                        massively::Zip1(black_box(output.slice_mut(..))),
                    )
                    .unwrap();
                    sync(&exec);
                    black_box((output_len, &output))
                })
            });
        }
    }
    union_group.finish();

    let mut intersection_group = c.benchmark_group("set_intersection");
    for backend in Runtime::available() {
        let exec = backend.exec();
        check_ordering(&exec);

        for &len in SIZES {
            let left = exec.to_device(&ascending_u32(len)).unwrap();
            let right = exec.to_device(&shifted_u32(len, len / 2)).unwrap();
            let output = exec.to_device(&vec![0_u32; len]).unwrap();
            sync(&exec);
            intersection_group.bench_function(BenchmarkId::new(backend.name(), len), |b| {
                iter_gpu(b, || {
                    let output_len = set_intersection(
                        &exec,
                        massively::Zip1(black_box(left.slice(..))),
                        massively::Zip1(black_box(right.slice(..))),
                        Less,
                        massively::Zip1(black_box(output.slice_mut(..))),
                    )
                    .unwrap();
                    sync(&exec);
                    black_box((output_len, &output))
                })
            });
        }
    }
    intersection_group.finish();

    let mut difference_group = c.benchmark_group("set_difference");
    for backend in Runtime::available() {
        let exec = backend.exec();
        check_ordering(&exec);

        for &len in SIZES {
            let left = exec.to_device(&ascending_u32(len)).unwrap();
            let right = exec.to_device(&shifted_u32(len, len / 2)).unwrap();
            let output = exec.to_device(&vec![0_u32; len]).unwrap();
            sync(&exec);
            difference_group.bench_function(BenchmarkId::new(backend.name(), len), |b| {
                iter_gpu(b, || {
                    let output_len = set_difference(
                        &exec,
                        massively::Zip1(black_box(left.slice(..))),
                        massively::Zip1(black_box(right.slice(..))),
                        Less,
                        massively::Zip1(black_box(output.slice_mut(..))),
                    )
                    .unwrap();
                    sync(&exec);
                    black_box((output_len, &output))
                })
            });
        }
    }
    difference_group.finish();
}

criterion_group! {
    name = benches;
    config = common::criterion();
    targets = bench_ordering
}
criterion_main!(benches);
