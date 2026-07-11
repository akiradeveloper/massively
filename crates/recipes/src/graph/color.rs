//! Degree-ordered greedy graph coloring with GPU neighbor-degree computation.

use cubecl::prelude::Runtime;
use massively::Executor;

use super::common::{self, CsrGraph};

pub fn solve<R: Runtime>(exec: &Executor<R>, graph: &CsrGraph) -> common::Result<Vec<u32>> {
    let degree = common::degrees(exec, graph)?;
    let mut order: Vec<usize> = (0..graph.vertex_count()).collect();
    order.sort_by_key(|&vertex| std::cmp::Reverse(degree[vertex]));

    let mut colors = vec![u32::MAX; graph.vertex_count()];
    for vertex in order {
        let mut used = vec![false; graph.vertex_count() + 1];
        for &neighbor in graph.row(vertex) {
            let color = colors[neighbor as usize];
            if color != u32::MAX {
                used[color as usize] = true;
            }
        }
        colors[vertex] = used.iter().position(|used| !*used).unwrap() as u32;
    }

    Ok(colors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn adjacent_vertices_have_distinct_colors() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        let graph = common::sample_graph();
        let colors = solve(&exec, &graph).unwrap();
        for source in 0..graph.vertex_count() {
            for &destination in graph.row(source) {
                assert_ne!(colors[source], colors[destination as usize]);
            }
        }
        assert_eq!(colors.iter().copied().max().unwrap() + 1, 3);
    }
}
