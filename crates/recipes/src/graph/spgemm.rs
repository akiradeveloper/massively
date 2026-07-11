//! Boolean sparse matrix multiplication using traversal emission and list canonicalization.

use cubecl::prelude::*;
use massively::{Executor, graph, op::BinaryPredicateOp, op::Identity};

use super::common::{self, CsrGraph, DeviceGraph};

struct LessU32;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for LessU32 {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs < rhs
    }
}

struct EqualU32;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for EqualU32 {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs == rhs
    }
}

pub fn solve<R: Runtime>(
    exec: &Executor<R>,
    lhs: &CsrGraph,
    rhs: &CsrGraph,
) -> common::Result<CsrGraph> {
    assert_eq!(lhs.vertex_count(), rhs.vertex_count());
    let rhs = DeviceGraph::new(exec, rhs);
    let mut offsets = Vec::with_capacity(lhs.vertex_count() + 1);
    let mut neighbors = Vec::new();
    offsets.push(0);

    for vertex in 0..lhs.vertex_count() {
        if lhs.row(vertex).is_empty() {
            offsets.push(neighbors.len() as u32);
            continue;
        }
        let frontier = exec.to_device(lhs.row(vertex));
        let traversal = graph::traverse(exec, rhs.csr(), frontier.slice(..))?;
        let edge_count = traversal.edge_count() as usize;
        if edge_count != 0 {
            let emitted = exec.alloc::<u32>(edge_count);
            traversal
                .map(graph::destination_id(), Identity)
                .emit(exec, emitted.slice_mut(..))?;
            let sorted = exec.alloc::<u32>(edge_count);
            massively::vector::sort(exec, emitted.slice(..), LessU32, sorted.slice_mut(..))?;
            let unique = exec.alloc::<u32>(edge_count);
            let len =
                massively::vector::unique(exec, sorted.slice(..), EqualU32, unique.slice_mut(..))?;
            neighbors.extend(exec.to_host(&unique.slice(..len as usize))?);
        }
        offsets.push(neighbors.len() as u32);
    }

    Ok(CsrGraph::new(offsets, neighbors))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn path_squared_contains_two_hop_pairs() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        assert_eq!(
            solve(&exec, &common::path_graph(), &common::path_graph()).unwrap(),
            CsrGraph::new(vec![0, 2, 4, 6, 8], vec![0, 2, 1, 3, 0, 2, 1, 3])
        );
    }
}
