//! Breadth-first search as destination-state relaxation over a traversal.

use cubecl::prelude::*;
use massively::{DeviceVec, Executor, graph, op::UnaryOp, vector};

use super::common::{self, DeviceCsr};

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
    graph: &DeviceCsr<R>,
    source: u32,
) -> common::Result<DeviceVec<R, u32>> {
    assert!(source < graph.vertex_count());
    let distance = vector::fill(exec, graph.vertex_count() as usize, u32::MAX)?;
    let mut frontier = vector::fill(exec, 1, source)?;
    let zero = vector::fill(exec, 1, 0u32)?;
    vector::scatter(
        exec,
        zero.slice(..),
        frontier.slice(..),
        distance.slice_mut(..),
    )?;

    while !frontier.is_empty() {
        frontier = graph::traverse(exec, graph.csr(), frontier.slice(..))?
            .map(graph::source(distance.slice(..)), AddOne)
            .relax_min_by_destination(exec, u32::MAX, distance.slice(..), distance.slice_mut(..))?;
    }

    Ok(distance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn path_levels() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        let graph = DeviceCsr::from_host(&exec, &common::path_graph()).unwrap();
        assert_eq!(
            exec.to_host(&solve(&exec, &graph, 0).unwrap()).unwrap(),
            vec![0, 1, 2, 3]
        );
    }
}
