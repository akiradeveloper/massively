//! Triangle counting with one batched intersection over a forward-oriented graph.

use cubecl::prelude::*;
use massively::{Executor, graph, op::ReductionOp};

use super::common::{self, CsrGraph, DeviceGraph};

struct SumU32;

#[cubecl::cube]
impl ReductionOp<u32> for SumU32 {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        lhs + rhs
    }
}

pub fn solve<R: Runtime>(exec: &Executor<R>, graph: &CsrGraph) -> common::Result<u32> {
    let mut offsets = Vec::with_capacity(graph.vertex_count() + 1);
    let mut destinations = Vec::new();
    offsets.push(0);
    for source in 0..graph.vertex_count() {
        destinations.extend(
            graph
                .row(source)
                .iter()
                .copied()
                .filter(|&destination| (source as u32) < destination),
        );
        offsets.push(destinations.len() as u32);
    }
    if destinations.is_empty() {
        return Ok(0);
    }

    let oriented = CsrGraph::new(offsets, destinations);
    let sources = oriented.edge_sources();
    let targets = oriented.neighbors.clone();
    let device_graph = DeviceGraph::new(exec, &oriented);
    let sources = exec.to_device(&sources);
    let targets = exec.to_device(&targets);
    let counts = exec.alloc::<u32>(targets.len());
    graph::intersect_count(
        exec,
        device_graph.csr(),
        sources.slice(..),
        targets.slice(..),
        counts.slice_mut(..),
    )?;

    massively::vector::reduce(exec, counts.slice(..), 0, SumU32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn shared_edge_forms_two_triangles() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        assert_eq!(solve(&exec, &common::sample_graph()).unwrap(), 2);
    }
}
