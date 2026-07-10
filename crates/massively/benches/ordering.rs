mod common;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use cubecl::prelude::*;
use massively::{BinaryPredicateOp, merge, set_difference, set_intersection, set_union, sort};

struct Less;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for Less {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs < rhs
    }
}

fn ascending(len: usize) -> Vec<u32> {
    (0..len).map(|index| index as u32).collect()
}

fn shifted(len: usize) -> Vec<u32> {
    (0..len).map(|index| (index + len / 2) as u32).collect()
}

fn bench_ordering(c: &mut Criterion) {
    let exec = common::exec();
    let mut sort_group = c.benchmark_group("sort");
    for &len in common::SORT_SIZES {
        let values = exec.to_device(&common::shuffled_u32(len));
        let output = exec.alloc::<u32>(len);
        sort_group.bench_function(BenchmarkId::new("gpu", len), |b| {
            b.iter(|| {
                sort(
                    &exec,
                    black_box(values.slice(..)),
                    Less,
                    black_box(output.slice_mut(..)),
                )
                .unwrap();
                exec.sync().unwrap();
                black_box(&output);
            })
        });
    }
    sort_group.finish();

    let mut merge_group = c.benchmark_group("merge");
    for &len in common::SIZES {
        let left = exec.to_device(&(0..len).map(|index| (index * 2) as u32).collect::<Vec<_>>());
        let right = exec.to_device(
            &(0..len)
                .map(|index| (index * 2 + 1) as u32)
                .collect::<Vec<_>>(),
        );
        let output = exec.alloc::<u32>(len * 2);
        merge_group.bench_function(BenchmarkId::new("gpu", len), |b| {
            b.iter(|| {
                merge(
                    &exec,
                    black_box(left.slice(..)),
                    black_box(right.slice(..)),
                    Less,
                    black_box(output.slice_mut(..)),
                )
                .unwrap();
                exec.sync().unwrap();
                black_box(&output);
            })
        });
    }
    merge_group.finish();

    macro_rules! set_benchmark {
        ($group_name:literal, $algorithm:ident, $capacity:expr) => {{
            let mut group = c.benchmark_group($group_name);
            for &len in common::SIZES {
                let left = exec.to_device(&ascending(len));
                let right = exec.to_device(&shifted(len));
                let output = exec.alloc::<u32>($capacity(len));
                group.bench_function(BenchmarkId::new("gpu", len), |b| {
                    b.iter(|| {
                        let output_len = $algorithm(
                            &exec,
                            black_box(left.slice(..)),
                            black_box(right.slice(..)),
                            Less,
                            black_box(output.slice_mut(..)),
                        )
                        .unwrap();
                        exec.sync().unwrap();
                        black_box((output_len, &output));
                    })
                });
            }
            group.finish();
        }};
    }

    set_benchmark!("set_union", set_union, |len| len * 2);
    set_benchmark!("set_intersection", set_intersection, |len| len);
    set_benchmark!("set_difference", set_difference, |len| len);
}

criterion_group! {
    name = benches;
    config = common::criterion();
    targets = bench_ordering
}
criterion_main!(benches);
