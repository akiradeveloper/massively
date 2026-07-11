mod common;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use cubecl::prelude::*;
use massively::{
    op::PredicateOp, vector::copy_where, vector::partition, vector::remove_where, zip7,
};

struct Positive;
#[cubecl::cube]
impl PredicateOp<f32> for Positive {
    fn apply(input: f32) -> bool {
        input > 0.0
    }
}

fn bench_select(c: &mut Criterion) {
    let exec = common::exec();
    let mut group = c.benchmark_group("select");
    for &len in common::SIZES {
        let host = common::dense_f32(len);
        let input = exec.to_device(&host);
        let output = exec.alloc::<f32>(len);
        for &rate in &[0usize, 50, 100] {
            let flags = exec.to_device(&common::flags(len, rate));
            group.bench_function(BenchmarkId::new(format!("copy_where_{rate}"), len), |b| {
                b.iter(|| {
                    criterion::black_box(
                        copy_where(
                            &exec,
                            input.slice(..),
                            flags.slice(..),
                            output.slice_mut(..),
                        )
                        .unwrap(),
                    );
                })
            });
            group.bench_function(BenchmarkId::new(format!("remove_where_{rate}"), len), |b| {
                b.iter(|| {
                    criterion::black_box(
                        remove_where(
                            &exec,
                            input.slice(..),
                            flags.slice(..),
                            output.slice_mut(..),
                        )
                        .unwrap(),
                    );
                })
            });
        }
        group.bench_function(BenchmarkId::new("partition", len), |b| {
            b.iter(|| {
                criterion::black_box(
                    partition(&exec, input.slice(..), Positive, output.slice_mut(..)).unwrap(),
                );
            })
        });
    }
    for &len in common::SORT_SIZES {
        let columns: Vec<_> = (0..7)
            .map(|column| {
                exec.to_device(&(0..len).map(|i| (i + column) as u32).collect::<Vec<_>>())
            })
            .collect();
        let outputs: Vec<_> = (0..7).map(|_| exec.alloc::<u32>(len)).collect();
        let flags = exec.to_device(&common::flags(len, 50));
        group.bench_function(BenchmarkId::new("copy_where_zip7", len), |b| {
            b.iter(|| {
                criterion::black_box(
                    copy_where(
                        &exec,
                        zip7(
                            columns[0].slice(..),
                            columns[1].slice(..),
                            columns[2].slice(..),
                            columns[3].slice(..),
                            columns[4].slice(..),
                            columns[5].slice(..),
                            columns[6].slice(..),
                        ),
                        flags.slice(..),
                        zip7(
                            outputs[0].slice_mut(..),
                            outputs[1].slice_mut(..),
                            outputs[2].slice_mut(..),
                            outputs[3].slice_mut(..),
                            outputs[4].slice_mut(..),
                            outputs[5].slice_mut(..),
                            outputs[6].slice_mut(..),
                        ),
                    )
                    .unwrap(),
                );
            })
        });
    }
    group.finish();
}
criterion_group! { name = benches; config = common::criterion(); targets = bench_select }
criterion_main!(benches);
