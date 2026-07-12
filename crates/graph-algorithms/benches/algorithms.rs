mod common;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use graph_algorithms::{
    DeviceCsr, DeviceWeightedCsr, bc, bfs, color, forman_ricci, geo, hits, kcore, mst, ppr, pr,
    spgemm, spmv, sssp, tc,
};
use massively::{Executor, zip2};

fn bench_single_pass(c: &mut Criterion, exec: &Executor<WgpuRuntime>) {
    let mut group = c.benchmark_group("graph_device_resident_single_pass");
    for &vertices in common::SINGLE_PASS_SIZES {
        let fixture = common::Fixture::new(vertices);
        let matrix = DeviceWeightedCsr::<_, f32>::from_host(exec, &fixture.matrix).unwrap();
        let vector = exec.to_device(&fixture.vector);
        exec.sync().unwrap();

        group.throughput(Throughput::Elements(matrix.graph().edge_count() as u64));
        group.bench_function(BenchmarkId::new("spmv", vertices), |b| {
            b.iter(|| {
                let output = spmv::solve(exec, &matrix, &vector).unwrap();
                exec.sync().unwrap();
                black_box(output)
            })
        });
        group.bench_function(BenchmarkId::new("forman_ricci", vertices), |b| {
            b.iter(|| {
                let output = forman_ricci::solve(exec, matrix.graph()).unwrap();
                exec.sync().unwrap();
                black_box(output)
            })
        });
        group.bench_function(BenchmarkId::new("triangle_count", vertices), |b| {
            b.iter(|| {
                let output = tc::solve(exec, matrix.graph()).unwrap();
                exec.sync().unwrap();
                black_box(output)
            })
        });
    }
    group.finish();
}

fn bench_iterative(c: &mut Criterion, exec: &Executor<WgpuRuntime>) {
    let mut group = c.benchmark_group("graph_device_resident_iterative");
    for &vertices in common::ITERATIVE_SIZES {
        let fixture = common::Fixture::new(vertices);
        let graph = DeviceCsr::from_host(exec, &fixture.graph).unwrap();
        let weighted_graph =
            DeviceWeightedCsr::from_parts(graph.clone(), exec.to_device(&fixture.weights_u32))
                .unwrap();
        let coordinates = zip2(
            exec.to_device(
                &fixture
                    .coordinates
                    .iter()
                    .map(|coordinate| coordinate.0)
                    .collect::<Vec<_>>(),
            ),
            exec.to_device(
                &fixture
                    .coordinates
                    .iter()
                    .map(|coordinate| coordinate.1)
                    .collect::<Vec<_>>(),
            ),
        );
        let known = exec.to_device(
            &fixture
                .known
                .iter()
                .map(|&known| u32::from(known))
                .collect::<Vec<_>>(),
        );
        exec.sync().unwrap();

        group.throughput(Throughput::Elements(graph.edge_count() as u64));
        group.bench_function(BenchmarkId::new("bfs", vertices), |b| {
            b.iter(|| {
                let output = bfs::solve(exec, &graph, 0).unwrap();
                exec.sync().unwrap();
                black_box(output)
            })
        });
        group.bench_function(BenchmarkId::new("sssp", vertices), |b| {
            b.iter(|| {
                let output = sssp::solve(exec, &weighted_graph, 0).unwrap();
                exec.sync().unwrap();
                black_box(output)
            })
        });
        group.bench_function(BenchmarkId::new("page_rank", vertices), |b| {
            b.iter(|| {
                let output = pr::solve(exec, &graph, 0.85, common::ITERATIONS).unwrap();
                exec.sync().unwrap();
                black_box(output)
            })
        });
        group.bench_function(BenchmarkId::new("personalized_page_rank", vertices), |b| {
            b.iter(|| {
                let output = ppr::solve(exec, &graph, 0, 0.85, common::ITERATIONS).unwrap();
                exec.sync().unwrap();
                black_box(output)
            })
        });
        group.bench_function(BenchmarkId::new("hits", vertices), |b| {
            b.iter(|| {
                let output = hits::solve(exec, &graph, common::ITERATIONS).unwrap();
                exec.sync().unwrap();
                black_box(output)
            })
        });
        group.bench_function(BenchmarkId::new("geolocation", vertices), |b| {
            b.iter(|| {
                let output =
                    geo::solve(exec, &graph, &coordinates, &known, common::ITERATIONS).unwrap();
                exec.sync().unwrap();
                black_box(output)
            })
        });
    }
    group.finish();
}

fn bench_control(c: &mut Criterion, exec: &Executor<WgpuRuntime>) {
    let mut group = c.benchmark_group("graph_device_resident_control");
    for &vertices in common::CONTROL_SIZES {
        let fixture = common::Fixture::new(vertices);
        let graph = DeviceCsr::from_host(exec, &fixture.graph).unwrap();
        let matrix = DeviceWeightedCsr::<_, f32>::from_host(exec, &fixture.matrix).unwrap();
        exec.sync().unwrap();

        group.throughput(Throughput::Elements(graph.edge_count() as u64));
        group.bench_function(BenchmarkId::new("coloring", vertices), |b| {
            b.iter(|| {
                let output = color::solve(exec, &graph).unwrap();
                exec.sync().unwrap();
                black_box(output)
            })
        });
        group.bench_function(BenchmarkId::new("k_core", vertices), |b| {
            b.iter(|| {
                let output = kcore::solve(exec, &graph).unwrap();
                exec.sync().unwrap();
                black_box(output)
            })
        });
        group.bench_function(BenchmarkId::new("minimum_spanning_forest", vertices), |b| {
            b.iter(|| {
                let output = mst::solve(exec, &matrix).unwrap();
                exec.sync().unwrap();
                black_box(output)
            })
        });
        group.bench_function(BenchmarkId::new("betweenness_centrality", vertices), |b| {
            b.iter(|| {
                let output = bc::solve(exec, &graph).unwrap();
                exec.sync().unwrap();
                black_box(output)
            })
        });
        group.bench_function(BenchmarkId::new("boolean_spgemm", vertices), |b| {
            b.iter(|| {
                let output = spgemm::solve(exec, &graph, &graph).unwrap();
                exec.sync().unwrap();
                black_box(output)
            })
        });
    }
    group.finish();
}

fn bench_algorithms(c: &mut Criterion) {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    bench_single_pass(c, &exec);
    bench_iterative(c, &exec);
    bench_control(c, &exec);
}

criterion_group! { name = benches; config = common::criterion(); targets = bench_algorithms }
criterion_main!(benches);
