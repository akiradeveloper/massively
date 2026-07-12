//! K-core decomposition initialized by GPU neighbor reduction.

use cubecl::prelude::Runtime;
use massively::Executor;

use super::common::{self, CsrGraph};

pub fn solve<R: Runtime>(exec: &Executor<R>, graph: &CsrGraph) -> common::Result<Vec<u32>> {
    let mut current_degree = common::degrees(exec, graph)?;
    let mut removed = vec![false; graph.vertex_count()];
    let mut core = vec![0u32; graph.vertex_count()];
    let mut running_core = 0u32;

    for _ in 0..graph.vertex_count() {
        let vertex = (0..graph.vertex_count())
            .filter(|&vertex| !removed[vertex])
            .min_by_key(|&vertex| current_degree[vertex])
            .unwrap();
        running_core = running_core.max(current_degree[vertex]);
        core[vertex] = running_core;
        removed[vertex] = true;
        for &neighbor in graph.row(vertex) {
            let neighbor = neighbor as usize;
            if !removed[neighbor] {
                current_degree[neighbor] = current_degree[neighbor].saturating_sub(1);
            }
        }
    }

    Ok(core)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn two_triangles_share_a_two_core() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        assert_eq!(
            solve(&exec, &common::sample_graph()).unwrap(),
            vec![2, 2, 2, 2]
        );
    }
}
