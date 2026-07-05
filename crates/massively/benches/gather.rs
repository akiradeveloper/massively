use cubecl::wgpu::WgpuRuntime;
mod common;

use common::{Runtime, SIZES, dense_f32, iter_gpu, reverse_indices, sync};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use massively::{Executor, gather};

fn check_gather(exec: &Executor<WgpuRuntime>) {
    let values = exec.to_device(&[10.0_f32, 20.0, 30.0, 40.0]).unwrap();
    let indices = exec.to_device(&[3_u32, 2, 1, 0]).unwrap();
    let output = exec.to_device(&[0.0_f32; 4]).unwrap();
    gather(
        exec,
        massively::Zip1(values.slice(..)),
        indices.slice(..),
        massively::Zip1(output.slice_mut(..)),
    )
    .unwrap();
    assert_eq!(exec.to_host(&output).unwrap(), vec![40.0, 30.0, 20.0, 10.0]);
}

fn bench_gather(c: &mut Criterion) {
    let mut group = c.benchmark_group("gather");
    for backend in Runtime::available() {
        let exec = backend.exec();
        check_gather(&exec);

        for &len in SIZES {
            let values = exec.to_device(&dense_f32(len)).unwrap();
            let indices = exec.to_device(&reverse_indices(len)).unwrap();
            let output = exec.to_device(&vec![0.0_f32; len]).unwrap();
            sync(&exec);
            group.bench_function(BenchmarkId::new(backend.name(), len), |b| {
                iter_gpu(b, || {
                    gather(
                        &exec,
                        massively::Zip1(black_box(values.slice(..))),
                        black_box(indices.slice(..)),
                        massively::Zip1(output.slice_mut(..)),
                    )
                    .unwrap();
                    sync(&exec);
                    black_box(output.len())
                })
            });
        }
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = common::criterion();
    targets = bench_gather
}
criterion_main!(benches);
