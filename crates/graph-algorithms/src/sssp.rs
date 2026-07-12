//! Frontier SSSP expressed as weighted destination-state relaxation.

use cubecl::prelude::*;
use massively::{Executor, graph, op::UnaryOp, zip2};

use super::common::{self, CsrGraph, DeviceGraph};

const INF: u32 = 1_000_000_000;

struct AddDistance;

#[cubecl::cube]
impl UnaryOp<(u32, u32)> for AddDistance {
    type Output = u32;

    fn apply(input: (u32, u32)) -> u32 {
        if input.0 >= INF {
            INF
        } else {
            u32::min(input.0 + input.1, INF)
        }
    }
}

pub fn solve<R: Runtime>(
    exec: &Executor<R>,
    graph: &CsrGraph,
    weights: &[u32],
    source: u32,
) -> common::Result<Vec<u32>> {
    assert_eq!(weights.len(), graph.neighbors.len());
    let n = graph.vertex_count();
    let device_graph = DeviceGraph::new(exec, graph);
    let weights = exec.to_device(weights);
    let mut distance = vec![INF; n];
    distance[source as usize] = 0;
    let distance = exec.to_device(&distance);
    let mut frontier = exec.to_device(&[source]);
    while !frontier.is_empty() {
        frontier = graph::traverse(exec, device_graph.csr(), frontier.slice(..))?
            .map(
                zip2(
                    graph::source(distance.slice(..)),
                    graph::edge(weights.slice(..)),
                ),
                AddDistance,
            )
            .relax_min_by_destination(exec, INF, distance.slice(..), distance.slice_mut(..))?;
    }

    exec.to_host(&distance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn weighted_path_distances_accumulate() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        assert_eq!(
            solve(&exec, &common::path_graph(), &[1, 1, 2, 2, 3, 3], 0).unwrap(),
            vec![0, 1, 3, 6]
        );
    }
}
