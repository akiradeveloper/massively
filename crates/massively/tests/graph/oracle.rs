use std::{cmp::Reverse, collections::BTreeSet};

use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::Executor;
use proptest::{
    prelude::*,
    test_runner::{Config, TestCaseResult, TestRunner},
};
use recipes::graph::{self, CsrGraph, WeightedCsr};

const CASES: u32 = 24;
const ITERATIONS: usize = 3;
const INF: u32 = 1_000_000_000;

#[derive(Clone, Debug)]
struct GraphCase {
    graph: CsrGraph,
    weights_u32: Vec<u32>,
    weights_f32: Vec<f32>,
    vector: Vec<f32>,
    coordinates: Vec<(f32, f32)>,
    known: Vec<bool>,
    source: u32,
}

fn graph_case() -> impl Strategy<Value = GraphCase> {
    (1usize..8)
        .prop_flat_map(|vertices| {
            let possible_edges = vertices * vertices.saturating_sub(1) / 2;
            (
                Just(vertices),
                prop::collection::vec(any::<bool>(), possible_edges),
                prop::collection::vec(1u32..10, possible_edges),
                prop::collection::vec(-8i16..9, vertices),
                prop::collection::vec((-8i16..9, -8i16..9), vertices),
                prop::collection::vec(any::<bool>(), vertices),
                0usize..vertices,
            )
        })
        .prop_map(
            |(vertices, present, edge_weights, vector, coordinates, known, source)| {
                let mut rows = vec![Vec::<(u32, u32)>::new(); vertices];
                let mut pair = 0;
                for lhs in 0..vertices {
                    for rhs in lhs + 1..vertices {
                        if present[pair] {
                            let weight = edge_weights[pair];
                            rows[lhs].push((rhs as u32, weight));
                            rows[rhs].push((lhs as u32, weight));
                        }
                        pair += 1;
                    }
                }

                let mut offsets = Vec::with_capacity(vertices + 1);
                let mut neighbors = Vec::new();
                let mut weights_u32 = Vec::new();
                offsets.push(0);
                for row in &mut rows {
                    row.sort_unstable_by_key(|&(destination, _)| destination);
                    for &(destination, weight) in row.iter() {
                        neighbors.push(destination);
                        weights_u32.push(weight);
                    }
                    offsets.push(neighbors.len() as u32);
                }

                GraphCase {
                    graph: CsrGraph::new(offsets, neighbors),
                    weights_f32: weights_u32.iter().map(|&weight| weight as f32).collect(),
                    weights_u32,
                    vector: vector.into_iter().map(f32::from).collect(),
                    coordinates: coordinates
                        .into_iter()
                        .map(|(x, y)| (f32::from(x), f32::from(y)))
                        .collect(),
                    known,
                    source: source as u32,
                }
            },
        )
}

fn assert_near(actual: &[f32], expected: &[f32], tolerance: f32) -> TestCaseResult {
    prop_assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        prop_assert!(
            (actual - expected).abs() <= tolerance,
            "index={index}, actual={actual}, expected={expected}"
        );
    }
    Ok(())
}

fn assert_coordinates_near(
    actual: &[(f32, f32)],
    expected: &[(f32, f32)],
    tolerance: f32,
) -> TestCaseResult {
    prop_assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        prop_assert!(
            (actual.0 - expected.0).abs() <= tolerance
                && (actual.1 - expected.1).abs() <= tolerance,
            "index={index}, actual={actual:?}, expected={expected:?}"
        );
    }
    Ok(())
}

fn run_graph_cases(test: impl Fn(&Executor<WgpuRuntime>, GraphCase) -> TestCaseResult) {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let mut runner = TestRunner::new(Config {
        cases: CASES,
        ..Config::default()
    });

    runner.run(&graph_case(), |case| test(&exec, case)).unwrap();
}

#[test]
fn bfs_matches_cpu_reference() {
    run_graph_cases(|exec, case| {
        let expected = reference::bfs(&case.graph, case.source);
        let actual = graph::bfs::solve(exec, &case.graph, case.source).unwrap();
        prop_assert_eq!(actual, expected);
        Ok(())
    });
}

#[test]
fn sssp_matches_cpu_reference() {
    run_graph_cases(|exec, case| {
        let expected = reference::sssp(&case.graph, &case.weights_u32, case.source);
        let actual = graph::sssp::solve(exec, &case.graph, &case.weights_u32, case.source).unwrap();
        prop_assert_eq!(actual, expected);
        Ok(())
    });
}

#[test]
fn spmv_matches_cpu_reference() {
    run_graph_cases(|exec, case| {
        let expected = reference::spmv(&case.graph, &case.weights_f32, &case.vector);
        let matrix = WeightedCsr::new(case.graph, case.weights_f32);
        let actual = graph::spmv::solve(exec, &matrix, &case.vector).unwrap();
        assert_near(&actual, &expected, 1e-5)
    });
}

#[test]
fn page_rank_matches_cpu_reference() {
    run_graph_cases(|exec, case| {
        let expected = reference::page_rank(&case.graph, 0.85, ITERATIONS);
        let actual = graph::pr::solve(exec, &case.graph, 0.85, ITERATIONS).unwrap();
        assert_near(&actual, &expected, 2e-4)
    });
}

#[test]
fn personalized_page_rank_matches_cpu_reference() {
    run_graph_cases(|exec, case| {
        let expected =
            reference::personalized_page_rank(&case.graph, case.source, 0.85, ITERATIONS);
        let actual = graph::ppr::solve(exec, &case.graph, case.source, 0.85, ITERATIONS).unwrap();
        assert_near(&actual, &expected, 2e-4)
    });
}

#[test]
fn hits_matches_cpu_reference() {
    run_graph_cases(|exec, case| {
        let expected = reference::hits(&case.graph, ITERATIONS);
        let actual = graph::hits::solve(exec, &case.graph, ITERATIONS).unwrap();
        assert_near(&actual.0, &expected.0, 2e-4)?;
        assert_near(&actual.1, &expected.1, 2e-4)
    });
}

#[test]
fn geolocation_matches_cpu_reference() {
    run_graph_cases(|exec, case| {
        let expected = reference::geo(&case.graph, &case.coordinates, &case.known, ITERATIONS);
        let actual = graph::geo::solve(
            exec,
            &case.graph,
            &case.coordinates,
            &case.known,
            ITERATIONS,
        )
        .unwrap();
        assert_coordinates_near(&actual, &expected, 2e-4)
    });
}

#[test]
fn spgemm_matches_cpu_reference() {
    run_graph_cases(|exec, case| {
        let expected = reference::spgemm(&case.graph, &case.graph);
        let actual = graph::spgemm::solve(exec, &case.graph, &case.graph).unwrap();
        prop_assert_eq!(actual, expected);
        Ok(())
    });
}

#[test]
fn triangle_count_matches_cpu_reference() {
    run_graph_cases(|exec, case| {
        let expected = reference::triangle_count(&case.graph);
        let actual = graph::tc::solve(exec, &case.graph).unwrap();
        prop_assert_eq!(actual, expected);
        Ok(())
    });
}

#[test]
fn graph_coloring_matches_cpu_reference() {
    run_graph_cases(|exec, case| {
        let expected = reference::color(&case.graph);
        let actual = graph::color::solve(exec, &case.graph).unwrap();
        prop_assert_eq!(actual, expected);
        Ok(())
    });
}

#[test]
fn kcore_matches_cpu_reference() {
    run_graph_cases(|exec, case| {
        let expected = reference::kcore(&case.graph);
        let actual = graph::kcore::solve(exec, &case.graph).unwrap();
        prop_assert_eq!(actual, expected);
        Ok(())
    });
}

#[test]
fn forman_ricci_matches_cpu_reference() {
    run_graph_cases(|exec, case| {
        let expected = reference::forman_ricci(&case.graph);
        let actual = graph::forman_ricci::solve(exec, &case.graph).unwrap();
        prop_assert_eq!(actual, expected);
        Ok(())
    });
}

#[test]
fn betweenness_centrality_matches_cpu_reference() {
    run_graph_cases(|exec, case| {
        let expected = reference::betweenness_centrality(&case.graph);
        let actual = graph::bc::solve(exec, &case.graph).unwrap();
        assert_near(&actual, &expected, 2e-4)
    });
}

#[test]
fn minimum_spanning_forest_matches_cpu_reference() {
    run_graph_cases(|exec, case| {
        let expected = reference::minimum_spanning_forest(&case.graph, &case.weights_f32);
        let matrix = WeightedCsr::new(case.graph, case.weights_f32);
        let actual = graph::mst::solve(exec, &matrix).unwrap();
        prop_assert_eq!(actual.len(), expected.0);
        let actual_weight = actual.iter().map(|edge| edge.2).sum::<f32>();
        prop_assert!((actual_weight - expected.1).abs() <= 1e-5);
        Ok(())
    });
}

mod reference {
    use super::*;

    pub fn bfs(graph: &CsrGraph, source: u32) -> Vec<u32> {
        let mut distance = vec![u32::MAX; graph.vertex_count()];
        let mut queue = std::collections::VecDeque::from([source as usize]);
        distance[source as usize] = 0;
        while let Some(vertex) = queue.pop_front() {
            for &destination in graph.row(vertex) {
                let destination = destination as usize;
                if distance[destination] == u32::MAX {
                    distance[destination] = distance[vertex] + 1;
                    queue.push_back(destination);
                }
            }
        }
        distance
    }

    pub fn sssp(graph: &CsrGraph, weights: &[u32], source: u32) -> Vec<u32> {
        let mut distance = vec![INF; graph.vertex_count()];
        distance[source as usize] = 0;
        for _ in 0..graph.vertex_count().saturating_sub(1) {
            let mut changed = false;
            for vertex in 0..graph.vertex_count() {
                for edge in graph.offsets[vertex] as usize..graph.offsets[vertex + 1] as usize {
                    let destination = graph.neighbors[edge] as usize;
                    let proposal = distance[vertex].saturating_add(weights[edge]).min(INF);
                    if proposal < distance[destination] {
                        distance[destination] = proposal;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        distance
    }

    pub fn spmv(graph: &CsrGraph, weights: &[f32], vector: &[f32]) -> Vec<f32> {
        (0..graph.vertex_count())
            .map(|vertex| {
                (graph.offsets[vertex] as usize..graph.offsets[vertex + 1] as usize)
                    .map(|edge| weights[edge] * vector[graph.neighbors[edge] as usize])
                    .sum()
            })
            .collect()
    }

    pub fn page_rank(graph: &CsrGraph, damping: f32, iterations: usize) -> Vec<f32> {
        let n = graph.vertex_count();
        let degree = degrees(graph);
        let mut rank = vec![1.0 / n as f32; n];
        for _ in 0..iterations {
            let dangling = (0..n)
                .filter(|&vertex| degree[vertex] == 0)
                .map(|vertex| rank[vertex])
                .sum::<f32>();
            let mut next = vec![(1.0 - damping + damping * dangling) / n as f32; n];
            for source in 0..n {
                if degree[source] != 0 {
                    let contribution = damping * rank[source] / degree[source] as f32;
                    for &destination in graph.row(source) {
                        next[destination as usize] += contribution;
                    }
                }
            }
            rank = next;
        }
        rank
    }

    pub fn personalized_page_rank(
        graph: &CsrGraph,
        root: u32,
        damping: f32,
        iterations: usize,
    ) -> Vec<f32> {
        let n = graph.vertex_count();
        let degree = degrees(graph);
        let mut rank = vec![1.0 / n as f32; n];
        for _ in 0..iterations {
            let dangling = (0..n)
                .filter(|&vertex| degree[vertex] == 0)
                .map(|vertex| rank[vertex])
                .sum::<f32>();
            let mut next = vec![damping * dangling / n as f32; n];
            next[root as usize] += 1.0 - damping;
            for source in 0..n {
                if degree[source] != 0 {
                    let contribution = damping * rank[source] / degree[source] as f32;
                    for &destination in graph.row(source) {
                        next[destination as usize] += contribution;
                    }
                }
            }
            rank = next;
        }
        rank
    }

    pub fn hits(graph: &CsrGraph, iterations: usize) -> (Vec<f32>, Vec<f32>) {
        let n = graph.vertex_count();
        let mut hubs = vec![1.0f32; n];
        let mut authorities = vec![1.0f32; n];
        for _ in 0..iterations {
            authorities.fill(0.0);
            for source in 0..n {
                for &destination in graph.row(source) {
                    authorities[destination as usize] += hubs[source];
                }
            }
            normalize(&mut authorities);

            hubs.fill(0.0);
            for source in 0..n {
                for &destination in graph.row(source) {
                    hubs[source] += authorities[destination as usize];
                }
            }
            normalize(&mut hubs);
        }
        (hubs, authorities)
    }

    pub fn geo(
        graph: &CsrGraph,
        initial: &[(f32, f32)],
        known: &[bool],
        iterations: usize,
    ) -> Vec<(f32, f32)> {
        let mut coordinates = initial.to_vec();
        for _ in 0..iterations {
            let previous = coordinates.clone();
            for vertex in 0..graph.vertex_count() {
                if known[vertex] || graph.row(vertex).is_empty() {
                    continue;
                }
                let (sum_x, sum_y) =
                    graph
                        .row(vertex)
                        .iter()
                        .fold((0.0, 0.0), |(sum_x, sum_y), &destination| {
                            let value = previous[destination as usize];
                            (sum_x + value.0, sum_y + value.1)
                        });
                coordinates[vertex] = (
                    sum_x / graph.row(vertex).len() as f32,
                    sum_y / graph.row(vertex).len() as f32,
                );
            }
        }
        coordinates
    }

    pub fn spgemm(lhs: &CsrGraph, rhs: &CsrGraph) -> CsrGraph {
        let mut offsets = Vec::with_capacity(lhs.vertex_count() + 1);
        let mut neighbors = Vec::new();
        offsets.push(0);
        for source in 0..lhs.vertex_count() {
            let mut row = BTreeSet::new();
            for &middle in lhs.row(source) {
                row.extend(rhs.row(middle as usize).iter().copied());
            }
            neighbors.extend(row);
            offsets.push(neighbors.len() as u32);
        }
        CsrGraph::new(offsets, neighbors)
    }

    pub fn triangle_count(graph: &CsrGraph) -> u32 {
        let mut triangles = 0;
        for lhs in 0..graph.vertex_count() {
            for rhs in lhs + 1..graph.vertex_count() {
                if graph.row(lhs).binary_search(&(rhs as u32)).is_err() {
                    continue;
                }
                for third in rhs + 1..graph.vertex_count() {
                    if graph.row(lhs).binary_search(&(third as u32)).is_ok()
                        && graph.row(rhs).binary_search(&(third as u32)).is_ok()
                    {
                        triangles += 1;
                    }
                }
            }
        }
        triangles
    }

    pub fn color(graph: &CsrGraph) -> Vec<u32> {
        let degree = degrees(graph);
        let mut order: Vec<_> = (0..graph.vertex_count()).collect();
        order.sort_by_key(|&vertex| Reverse(degree[vertex]));
        let mut colors = vec![u32::MAX; graph.vertex_count()];
        for vertex in order {
            let mut used = vec![false; graph.vertex_count() + 1];
            for &destination in graph.row(vertex) {
                let color = colors[destination as usize];
                if color != u32::MAX {
                    used[color as usize] = true;
                }
            }
            colors[vertex] = used.iter().position(|&used| !used).unwrap() as u32;
        }
        colors
    }

    pub fn kcore(graph: &CsrGraph) -> Vec<u32> {
        let mut current_degree = degrees(graph);
        let mut removed = vec![false; graph.vertex_count()];
        let mut core = vec![0; graph.vertex_count()];
        let mut running_core = 0;
        for _ in 0..graph.vertex_count() {
            let vertex = (0..graph.vertex_count())
                .filter(|&vertex| !removed[vertex])
                .min_by_key(|&vertex| current_degree[vertex])
                .unwrap();
            running_core = running_core.max(current_degree[vertex]);
            core[vertex] = running_core;
            removed[vertex] = true;
            for &destination in graph.row(vertex) {
                let destination = destination as usize;
                if !removed[destination] {
                    current_degree[destination] = current_degree[destination].saturating_sub(1);
                }
            }
        }
        core
    }

    pub fn forman_ricci(graph: &CsrGraph) -> Vec<i32> {
        let degree = degrees(graph);
        let mut output = Vec::with_capacity(graph.neighbors.len());
        for source in 0..graph.vertex_count() {
            for &destination in graph.row(source) {
                output.push(4 - degree[source] as i32 - degree[destination as usize] as i32);
            }
        }
        output
    }

    pub fn betweenness_centrality(graph: &CsrGraph) -> Vec<f32> {
        let n = graph.vertex_count();
        let mut centrality = vec![0.0; n];
        for source in 0..n {
            let distance = bfs(graph, source as u32);
            let mut order: Vec<_> = (0..n)
                .filter(|&vertex| distance[vertex] != u32::MAX)
                .collect();
            order.sort_by_key(|&vertex| distance[vertex]);
            let mut paths = vec![0.0; n];
            paths[source] = 1.0;
            for &vertex in &order {
                for &destination in graph.row(vertex) {
                    let destination = destination as usize;
                    if distance[destination] == distance[vertex] + 1 {
                        paths[destination] += paths[vertex];
                    }
                }
            }
            let mut dependency = vec![0.0; n];
            for &vertex in order.iter().rev() {
                for &destination in graph.row(vertex) {
                    let destination = destination as usize;
                    if distance[destination] == distance[vertex] + 1 && paths[destination] != 0.0 {
                        dependency[vertex] +=
                            paths[vertex] / paths[destination] * (1.0 + dependency[destination]);
                    }
                }
                if vertex != source {
                    centrality[vertex] += dependency[vertex];
                }
            }
        }
        centrality
    }

    pub fn minimum_spanning_forest(graph: &CsrGraph, weights: &[f32]) -> (usize, f32) {
        let mut edges = Vec::new();
        for source in 0..graph.vertex_count() {
            for edge in graph.offsets[source] as usize..graph.offsets[source + 1] as usize {
                let destination = graph.neighbors[edge] as usize;
                if source <= destination {
                    edges.push((source, destination, weights[edge]));
                }
            }
        }
        edges.sort_by(|lhs, rhs| lhs.2.partial_cmp(&rhs.2).unwrap());
        let mut sets = DisjointSet::new(graph.vertex_count());
        let mut count = 0;
        let mut total = 0.0;
        for (lhs, rhs, weight) in edges {
            if sets.union(lhs, rhs) {
                count += 1;
                total += weight;
            }
        }
        (count, total)
    }

    fn degrees(graph: &CsrGraph) -> Vec<u32> {
        (0..graph.vertex_count())
            .map(|vertex| graph.row(vertex).len() as u32)
            .collect()
    }

    fn normalize(values: &mut [f32]) {
        let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
        if norm != 0.0 {
            for value in values {
                *value /= norm;
            }
        }
    }

    struct DisjointSet {
        parent: Vec<usize>,
        rank: Vec<u8>,
    }

    impl DisjointSet {
        fn new(len: usize) -> Self {
            Self {
                parent: (0..len).collect(),
                rank: vec![0; len],
            }
        }

        fn find(&mut self, value: usize) -> usize {
            if self.parent[value] != value {
                self.parent[value] = self.find(self.parent[value]);
            }
            self.parent[value]
        }

        fn union(&mut self, lhs: usize, rhs: usize) -> bool {
            let mut lhs = self.find(lhs);
            let mut rhs = self.find(rhs);
            if lhs == rhs {
                return false;
            }
            if self.rank[lhs] < self.rank[rhs] {
                std::mem::swap(&mut lhs, &mut rhs);
            }
            self.parent[rhs] = lhs;
            if self.rank[lhs] == self.rank[rhs] {
                self.rank[lhs] += 1;
            }
            true
        }
    }
}
