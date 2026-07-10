mod common;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use cubecl::prelude::*;
use massively::{BinaryPredicateOp, sort_by_key};

struct Less;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for Less {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs < rhs
    }
}

fn bench_sort_by_key(c: &mut Criterion) {
    let exec = common::exec();
    let mut group = c.benchmark_group("sort_by_key");

    for &len in common::SORT_SIZES {
        let keys = exec.to_device(&common::shuffled_u32(len));
        let values = exec.to_device(&common::dense_f32(len));
        let out_keys = exec.alloc::<u32>(len);
        let out_values = exec.alloc::<f32>(len);
        exec.sync().unwrap();

        group.bench_function(BenchmarkId::new("gpu", len), |b| {
            b.iter(|| {
                sort_by_key(
                    &exec,
                    black_box(keys.slice(..)),
                    black_box(values.slice(..)),
                    Less,
                    black_box(out_keys.slice_mut(..)),
                    black_box(out_values.slice_mut(..)),
                )
                .unwrap();
                exec.sync().unwrap();
                black_box((&out_keys, &out_values));
            })
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = common::criterion();
    targets = bench_sort_by_key
}
criterion_main!(benches);
