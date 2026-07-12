mod common;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use graph_algorithms::{
    bc, bfs, color, forman_ricci, geo, hits, kcore, mst, ppr, pr, spgemm, spmv, sssp, tc,
};
use massively::Executor;

fn bench_single_pass(c: &mut Criterion, exec: &Executor<WgpuRuntime>) {
    let mut group = c.benchmark_group("graph_end_to_end_single_pass");
    for &vertices in common::SINGLE_PASS_SIZES {
        let fixture = common::Fixture::new(vertices);
        group.throughput(Throughput::Elements(fixture.graph.neighbors.len() as u64));

        group.bench_function(BenchmarkId::new("spmv", vertices), |b| {
            b.iter(|| {
                black_box(spmv::solve(exec, &fixture.matrix, &fixture.vector).unwrap());
            })
        });
        group.bench_function(BenchmarkId::new("forman_ricci", vertices), |b| {
            b.iter(|| {
                black_box(forman_ricci::solve(exec, &fixture.graph).unwrap());
            })
        });
        group.bench_function(BenchmarkId::new("minimum_spanning_forest", vertices), |b| {
            b.iter(|| {
                black_box(mst::solve(exec, &fixture.matrix).unwrap());
            })
        });
        group.bench_function(BenchmarkId::new("triangle_count", vertices), |b| {
            b.iter(|| {
                black_box(tc::solve(exec, &fixture.graph).unwrap());
            })
        });
    }
    group.finish();
}

fn bench_iterative(c: &mut Criterion, exec: &Executor<WgpuRuntime>) {
    let mut group = c.benchmark_group("graph_end_to_end_iterative");
    for &vertices in common::ITERATIVE_SIZES {
        let fixture = common::Fixture::new(vertices);
        group.throughput(Throughput::Elements(fixture.graph.neighbors.len() as u64));

        group.bench_function(BenchmarkId::new("bfs", vertices), |b| {
            b.iter(|| {
                black_box(bfs::solve(exec, &fixture.graph, 0).unwrap());
            })
        });
        group.bench_function(BenchmarkId::new("sssp", vertices), |b| {
            b.iter(|| {
                black_box(sssp::solve(exec, &fixture.graph, &fixture.weights_u32, 0).unwrap());
            })
        });
        group.bench_function(BenchmarkId::new("page_rank_3_iterations", vertices), |b| {
            b.iter(|| {
                black_box(pr::solve(exec, &fixture.graph, 0.85, common::ITERATIONS).unwrap());
            })
        });
        group.bench_function(
            BenchmarkId::new("personalized_page_rank_3_iterations", vertices),
            |b| {
                b.iter(|| {
                    black_box(
                        ppr::solve(exec, &fixture.graph, 0, 0.85, common::ITERATIONS).unwrap(),
                    );
                })
            },
        );
        group.bench_function(BenchmarkId::new("hits_3_iterations", vertices), |b| {
            b.iter(|| {
                black_box(hits::solve(exec, &fixture.graph, common::ITERATIONS).unwrap());
            })
        });
        group.bench_function(
            BenchmarkId::new("geolocation_3_iterations", vertices),
            |b| {
                b.iter(|| {
                    black_box(
                        geo::solve(
                            exec,
                            &fixture.graph,
                            &fixture.coordinates,
                            &fixture.known,
                            common::ITERATIONS,
                        )
                        .unwrap(),
                    );
                })
            },
        );
    }
    group.finish();
}

fn bench_host_orchestrated(c: &mut Criterion, exec: &Executor<WgpuRuntime>) {
    let mut group = c.benchmark_group("graph_end_to_end_host_orchestrated");
    for &vertices in common::HOST_ORCHESTRATED_SIZES {
        let fixture = common::Fixture::new(vertices);
        group.throughput(Throughput::Elements(fixture.graph.neighbors.len() as u64));

        group.bench_function(BenchmarkId::new("betweenness_centrality", vertices), |b| {
            b.iter(|| {
                black_box(bc::solve(exec, &fixture.graph).unwrap());
            })
        });
        group.bench_function(BenchmarkId::new("coloring", vertices), |b| {
            b.iter(|| {
                black_box(color::solve(exec, &fixture.graph).unwrap());
            })
        });
        group.bench_function(BenchmarkId::new("kcore", vertices), |b| {
            b.iter(|| {
                black_box(kcore::solve(exec, &fixture.graph).unwrap());
            })
        });
        group.bench_function(BenchmarkId::new("boolean_spgemm", vertices), |b| {
            b.iter(|| {
                black_box(spgemm::solve(exec, &fixture.graph, &fixture.graph).unwrap());
            })
        });
    }
    group.finish();
}

fn bench_algorithms(c: &mut Criterion) {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    bench_single_pass(c, &exec);
    bench_iterative(c, &exec);
    bench_host_orchestrated(c, &exec);
}

criterion_group! { name = benches; config = common::criterion(); targets = bench_algorithms }
criterion_main!(benches);
