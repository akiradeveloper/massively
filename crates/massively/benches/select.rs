use cubecl::wgpu::WgpuRuntime;
mod common;

use common::{Runtime, SIZES, sync};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use massively::{Executor, copy_where};

fn alternating_signed(len: usize) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let value = (index % 251) as f32 + 1.0;
            if index % 2 == 0 { value } else { -value }
        })
        .collect()
}

fn alternating_flags(len: usize) -> Vec<u32> {
    (0..len)
        .map(|index| if index % 2 == 0 { 1 } else { 0 })
        .collect()
}

fn check_copy_where(exec: &Executor<WgpuRuntime>) {
    let values = exec.to_device(&[-1.0_f32, 2.0, -3.0, 4.0]).unwrap();
    let stencil = exec.to_device(&[0_u32, 1, 0, 1]).unwrap();
    let (output,) =
        copy_where(&exec, massively::SoA1(values.slice(..)), stencil.slice(..)).unwrap();
    assert_eq!(exec.to_host(&output).unwrap(), vec![2.0, 4.0]);
}

fn bench_select(c: &mut Criterion) {
    let mut copy_group = c.benchmark_group("copy_where");
    for backend in Runtime::available() {
        let exec = backend.exec();
        check_copy_where(&exec);

        for &len in SIZES {
            let values = exec.to_device(&alternating_signed(len)).unwrap();
            let stencil = exec.to_device(&alternating_flags(len)).unwrap();
            sync(&exec);
            copy_group.bench_function(BenchmarkId::new(backend.name(), len), |b| {
                b.iter(|| {
                    let output = copy_where(
                        &exec,
                        massively::SoA1(black_box(values.slice(..))),
                        black_box(stencil.slice(..)),
                    )
                    .unwrap();
                    sync(&exec);
                    black_box(output)
                })
            });
        }
    }
    copy_group.finish();
}

criterion_group!(benches, bench_select);
criterion_main!(benches);
