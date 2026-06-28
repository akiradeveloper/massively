use cubecl::wgpu::WgpuRuntime;
mod common;

use common::{
    Runtime, SIZES, ascending_u32, dense_f32, half_select_flags, iter_gpu, reverse_indices,
    select_flags, sync,
};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use massively::{Executor, gather_where, replace_where, scatter_where};

fn check_where_indexed(exec: &Executor<WgpuRuntime>) {
    let values = exec.to_device(&[10.0_f32, -20.0, 30.0]).unwrap();
    let indices = exec.to_device(&[2_u32, 0, 1]).unwrap();
    let stencil = exec.to_device(&[1_u32, 1, 0]).unwrap();
    let output = exec.to_device(&[0.0_f32; 3]).unwrap();

    gather_where(
        exec,
        massively::SoA1(values.slice(..)),
        indices.slice(..),
        stencil.slice(..),
        massively::SoA1(output.slice_mut(..)),
    )
    .unwrap();
    assert_eq!(exec.to_host(&output).unwrap(), vec![30.0, 10.0, 0.0]);

    let scatter_stencil = exec.to_device(&[1_u32, 0, 1]).unwrap();
    let scattered = exec.to_device(&[0.0_f32; 3]).unwrap();
    scatter_where(
        exec,
        massively::SoA1(values.slice(..)),
        indices.slice(..),
        scatter_stencil.slice(..),
        massively::SoA1(scattered.slice_mut(..)),
    )
    .unwrap();
    assert_eq!(exec.to_host(&scattered).unwrap(), vec![0.0, 30.0, 10.0]);

    replace_where(
        exec,
        (0.0,),
        scatter_stencil.slice(..),
        massively::SoA1(output.slice_mut(..)),
    )
    .unwrap();
}

fn bench_where_indexed(c: &mut Criterion) {
    let mut gather_group = c.benchmark_group("gather_where");
    for backend in Runtime::available() {
        let exec = backend.exec();
        check_where_indexed(&exec);

        for &len in SIZES {
            let values = exec.to_device(&dense_f32(len)).unwrap();
            let indices = exec.to_device(&reverse_indices(len)).unwrap();
            let stencil = exec.to_device(&half_select_flags(len)).unwrap();
            let output = exec.to_device(&vec![0.0_f32; len]).unwrap();
            sync(&exec);
            gather_group.bench_function(BenchmarkId::new(backend.name(), len), |b| {
                iter_gpu(b, || {
                    gather_where(
                        &exec,
                        massively::SoA1(black_box(values.slice(..))),
                        black_box(indices.slice(..)),
                        black_box(stencil.slice(..)),
                        massively::SoA1(output.slice_mut(..)),
                    )
                    .unwrap();
                    sync(&exec);
                    black_box(output.len())
                })
            });
        }
    }
    gather_group.finish();

    let mut scatter_group = c.benchmark_group("scatter_where");
    for backend in Runtime::available() {
        let exec = backend.exec();
        check_where_indexed(&exec);

        for &len in SIZES {
            let values = exec.to_device(&dense_f32(len)).unwrap();
            let indices = exec.to_device(&reverse_indices(len)).unwrap();
            let stencil = exec.to_device(&half_select_flags(len)).unwrap();
            let output = exec.to_device(&vec![0.0_f32; len]).unwrap();
            sync(&exec);
            scatter_group.bench_function(BenchmarkId::new(backend.name(), len), |b| {
                iter_gpu(b, || {
                    scatter_where(
                        &exec,
                        massively::SoA1(black_box(values.slice(..))),
                        black_box(indices.slice(..)),
                        black_box(stencil.slice(..)),
                        massively::SoA1(output.slice_mut(..)),
                    )
                    .unwrap();
                    sync(&exec);
                    black_box(output.len())
                })
            });
        }
    }
    scatter_group.finish();

    let mut replace_group = c.benchmark_group("replace_where");
    for backend in Runtime::available() {
        let exec = backend.exec();
        check_where_indexed(&exec);

        for &len in SIZES {
            let values = exec.to_device(&dense_f32(len)).unwrap();
            let stencil = exec.to_device(&half_select_flags(len)).unwrap();
            sync(&exec);
            replace_group.bench_function(BenchmarkId::new(backend.name(), len), |b| {
                iter_gpu(b, || {
                    replace_where(
                        &exec,
                        (0.0,),
                        black_box(stencil.slice(..)),
                        massively::SoA1(values.slice_mut(..)),
                    )
                    .unwrap();
                    sync(&exec);
                    black_box(values.len())
                })
            });
        }
    }
    replace_group.finish();

    let selectivity_patterns = [
        ("0pct", 0_usize),
        ("50pct", 50_usize),
        ("100pct", 100_usize),
    ];
    let index_patterns: [(&str, fn(usize) -> Vec<u32>); 2] =
        [("identity", ascending_u32), ("reverse", reverse_indices)];

    let mut gather_selectivity_group = c.benchmark_group("gather_where_selectivity");
    for backend in Runtime::available() {
        let exec = backend.exec();
        check_where_indexed(&exec);

        for (index_name, make_indices) in index_patterns {
            for (selectivity_name, selected_per_100) in selectivity_patterns {
                for &len in SIZES {
                    let values = exec.to_device(&dense_f32(len)).unwrap();
                    let indices = exec.to_device(&make_indices(len)).unwrap();
                    let stencil = exec
                        .to_device(&select_flags(len, selected_per_100))
                        .unwrap();
                    let output = exec.to_device(&vec![0.0_f32; len]).unwrap();
                    sync(&exec);
                    gather_selectivity_group.bench_function(
                        BenchmarkId::new(
                            format!("{}-{}-{}", backend.name(), index_name, selectivity_name),
                            len,
                        ),
                        |b| {
                            iter_gpu(b, || {
                                gather_where(
                                    &exec,
                                    massively::SoA1(black_box(values.slice(..))),
                                    black_box(indices.slice(..)),
                                    black_box(stencil.slice(..)),
                                    massively::SoA1(output.slice_mut(..)),
                                )
                                .unwrap();
                                sync(&exec);
                                black_box(output.len())
                            })
                        },
                    );
                }
            }
        }
    }
    gather_selectivity_group.finish();

    let mut scatter_selectivity_group = c.benchmark_group("scatter_where_selectivity");
    for backend in Runtime::available() {
        let exec = backend.exec();
        check_where_indexed(&exec);

        for (index_name, make_indices) in index_patterns {
            for (selectivity_name, selected_per_100) in selectivity_patterns {
                for &len in SIZES {
                    let values = exec.to_device(&dense_f32(len)).unwrap();
                    let indices = exec.to_device(&make_indices(len)).unwrap();
                    let stencil = exec
                        .to_device(&select_flags(len, selected_per_100))
                        .unwrap();
                    let output = exec.to_device(&vec![0.0_f32; len]).unwrap();
                    sync(&exec);
                    scatter_selectivity_group.bench_function(
                        BenchmarkId::new(
                            format!("{}-{}-{}", backend.name(), index_name, selectivity_name),
                            len,
                        ),
                        |b| {
                            iter_gpu(b, || {
                                scatter_where(
                                    &exec,
                                    massively::SoA1(black_box(values.slice(..))),
                                    black_box(indices.slice(..)),
                                    black_box(stencil.slice(..)),
                                    massively::SoA1(output.slice_mut(..)),
                                )
                                .unwrap();
                                sync(&exec);
                                black_box(output.len())
                            })
                        },
                    );
                }
            }
        }
    }
    scatter_selectivity_group.finish();

    let mut replace_selectivity_group = c.benchmark_group("replace_where_selectivity");
    for backend in Runtime::available() {
        let exec = backend.exec();
        check_where_indexed(&exec);

        for (selectivity_name, selected_per_100) in selectivity_patterns {
            for &len in SIZES {
                let values = exec.to_device(&dense_f32(len)).unwrap();
                let stencil = exec
                    .to_device(&select_flags(len, selected_per_100))
                    .unwrap();
                sync(&exec);
                replace_selectivity_group.bench_function(
                    BenchmarkId::new(format!("{}-{}", backend.name(), selectivity_name), len),
                    |b| {
                        iter_gpu(b, || {
                            replace_where(
                                &exec,
                                (0.0,),
                                black_box(stencil.slice(..)),
                                massively::SoA1(values.slice_mut(..)),
                            )
                            .unwrap();
                            sync(&exec);
                            black_box(values.len())
                        })
                    },
                );
            }
        }
    }
    replace_selectivity_group.finish();
}

criterion_group! {
    name = benches;
    config = common::criterion();
    targets = bench_where_indexed
}
criterion_main!(benches);
