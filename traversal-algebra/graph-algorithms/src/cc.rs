//! Connected components by monotone minimum-label propagation.
//!
//! The input is interpreted as an undirected graph (each undirected edge is
//! represented by both CSR directions).  The returned label of every vertex
//! is the smallest vertex identifier in its connected component.

use cubecl::prelude::*;
use massively::{DeviceVec, Executor, graph, lazy, op::Identity, vector};

use super::common::{self, DeviceCsr};

pub fn solve<R: Runtime>(
    exec: &Executor<R>,
    graph: &DeviceCsr<R>,
) -> common::Result<DeviceVec<R, u32>> {
    let n = graph.vertex_count();
    let labels = vector::transform(exec, lazy::counting(0).take(n), Identity)?;
    let mut frontier = vector::transform(exec, lazy::counting(0).take(n), Identity)?;

    while !frontier.is_empty() {
        frontier = graph::traverse(exec, graph.csr(), frontier.slice(..))?
            .map(graph::source(labels.slice(..)), Identity)
            .relax_min_by_destination(exec, u32::MAX, labels.slice(..), labels.slice_mut(..))?;
    }

    Ok(labels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CsrGraph;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn labels_disconnected_components_by_their_minimum_vertex() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        let host = CsrGraph::new(vec![0, 1, 2, 3, 4, 4], vec![1, 0, 3, 2]);
        let graph = DeviceCsr::from_host(&exec, &host).unwrap();
        let labels = solve(&exec, &graph).unwrap();
        assert_eq!(exec.to_host(&labels).unwrap(), vec![0, 0, 2, 2, 4]);
    }
}
