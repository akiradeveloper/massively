//! Unweighted Brandes betweenness centrality using the BFS recipe for every source.

use cubecl::prelude::Runtime;
use massively::Executor;

use super::{
    bfs,
    common::{self, CsrGraph},
};

pub fn solve<R: Runtime>(exec: &Executor<R>, graph: &CsrGraph) -> common::Result<Vec<f32>> {
    let n = graph.vertex_count();
    let mut centrality = vec![0.0f32; n];

    for source in 0..n {
        let distance = bfs::solve(exec, graph, source as u32)?;
        let mut order: Vec<usize> = (0..n)
            .filter(|&vertex| distance[vertex] != u32::MAX)
            .collect();
        order.sort_by_key(|&vertex| distance[vertex]);

        let mut paths = vec![0.0f32; n];
        paths[source] = 1.0;
        for &vertex in &order {
            for &neighbor in graph.row(vertex) {
                let neighbor = neighbor as usize;
                if distance[neighbor] == distance[vertex] + 1 {
                    paths[neighbor] += paths[vertex];
                }
            }
        }

        let mut dependency = vec![0.0f32; n];
        for &vertex in order.iter().rev() {
            for &neighbor in graph.row(vertex) {
                let neighbor = neighbor as usize;
                if distance[neighbor] == distance[vertex] + 1 && paths[neighbor] != 0.0 {
                    dependency[vertex] +=
                        paths[vertex] / paths[neighbor] * (1.0 + dependency[neighbor]);
                }
            }
            if vertex != source {
                centrality[vertex] += dependency[vertex];
            }
        }
    }

    Ok(centrality)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn middle_vertices_dominate_a_path() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        common::assert_near(
            &solve(&exec, &common::path_graph()).unwrap(),
            &[0.0, 4.0, 4.0, 0.0],
            1e-5,
        );
    }
}
