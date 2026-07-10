mod common;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use massively::scatter;

fn bench_scatter(c: &mut Criterion) {
    let exec = common::exec();
    let mut group = c.benchmark_group("scatter");
    for &len in common::SIZES {
        let input = exec.to_device(&common::dense_f32(len));
        let indices = exec.to_device(&common::reverse_indices(len));
        let output = exec.alloc::<f32>(len);
        exec.sync().unwrap();
        group.bench_function(BenchmarkId::new("reverse", len), |b| {
            b.iter(|| {
                scatter(
                    &exec,
                    input.slice(..),
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
criterion_group! { name = benches; config = common::criterion(); targets = bench_scatter }
criterion_main!(benches);
