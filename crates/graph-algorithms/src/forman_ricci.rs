//! Unweighted Forman–Ricci edge curvature from GPU-computed degrees.

use cubecl::prelude::Runtime;
use massively::Executor;

use super::common::{self, CsrGraph};

pub fn solve<R: Runtime>(exec: &Executor<R>, graph: &CsrGraph) -> common::Result<Vec<i32>> {
    let degree = exec.to_host(&common::degrees(exec, graph)?)?;
    Ok(graph
        .edge_sources()
        .into_iter()
        .zip(graph.neighbors.iter().copied())
        .map(|(source, destination)| {
            4 - degree[source as usize] as i32 - degree[destination as usize] as i32
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn curvature_matches_endpoint_degrees() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        assert_eq!(
            solve(&exec, &common::sample_graph()).unwrap(),
            vec![-1, -1, -1, -2, -1, -1, -2, -1, -1, -1]
        );
    }
}
