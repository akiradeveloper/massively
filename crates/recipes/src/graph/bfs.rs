//! Breadth-first search as destination-state relaxation over a traversal.

use cubecl::prelude::*;
use massively::{Executor, graph, op::UnaryOp};

use super::common::{self, CsrGraph, DeviceGraph};

struct AddOne;

#[cubecl::cube]
impl UnaryOp<u32> for AddOne {
    type Output = u32;

    fn apply(distance: u32) -> u32 {
        distance + 1u32
    }
}

pub fn solve<R: Runtime>(
    exec: &Executor<R>,
    graph: &CsrGraph,
    source: u32,
) -> common::Result<Vec<u32>> {
    let mut distance = vec![u32::MAX; graph.vertex_count()];
    distance[source as usize] = 0;
    let distance = exec.to_device(&distance);
    let device_graph = DeviceGraph::new(exec, graph);
    let mut frontier = exec.to_device(&[source]);
    let mut frontier_len = 1u32;

    while frontier_len != 0 {
        let next = exec.alloc::<u32>(graph.vertex_count());
        frontier_len = graph::traverse(
            exec,
            device_graph.csr(),
            frontier.slice(..frontier_len as usize),
        )?
        .map(graph::source(distance.slice(..)), AddOne)
        .relax_min_by_destination(
            exec,
            u32::MAX,
            distance.slice(..),
            distance.slice_mut(..),
            next.slice_mut(..),
        )?;
        frontier = next;
    }

    exec.to_host(&distance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn path_levels() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        assert_eq!(
            solve(&exec, &common::path_graph(), 0).unwrap(),
            vec![0, 1, 2, 3]
        );
    }
}
