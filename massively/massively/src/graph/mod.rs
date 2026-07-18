//! Algebraic graph traversal over lazy edge-context expressions.
//!
//! A traversal selects edges from a vertex frontier. Edge expressions describe values read at
//! source vertices, destination vertices, or CSR edge positions. A terminal then emits edge
//! results or reduces them by source or destination. Execution control and expansion
//! materialization remain private and can be replaced by fused kernels without changing this API.

mod control;
#[doc(hidden)]
pub mod expr;
mod intersection;
mod traversal;

pub use expr::{
    Destination, DestinationId, Edge, EdgeExpr, EdgeId, Source, SourceId, destination,
    destination_id, edge, edge_id, source, source_id,
};
pub use intersection::intersect_count;
pub use traversal::{MappedTraversal, Traversal, traverse};

/// A compressed sparse row topology view.
#[derive(Clone, Copy, Debug)]
pub struct Csr<Destinations, Offsets> {
    destinations: Destinations,
    offsets: Offsets,
}

impl<Destinations, Offsets> Csr<Destinations, Offsets> {
    /// Constructs a CSR topology from destination vertex IDs and vertex offsets.
    pub const fn new(destinations: Destinations, offsets: Offsets) -> Self {
        Self {
            destinations,
            offsets,
        }
    }

    /// Returns the flat destination vertex stream.
    pub const fn destinations(&self) -> &Destinations {
        &self.destinations
    }

    /// Returns the vertex offsets.
    pub const fn offsets(&self) -> &Offsets {
        &self.offsets
    }

    /// Decomposes this topology view.
    pub fn into_parts(self) -> (Destinations, Offsets) {
        (self.destinations, self.offsets)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Executor, op::Identity, op::ReductionOp, op::UnaryOp, zip2};
    use cubecl::prelude::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    struct Add;

    #[cubecl::cube]
    impl ReductionOp<u32> for Add {
        fn apply(lhs: u32, rhs: u32) -> u32 {
            lhs + rhs
        }
    }

    struct PairAdd;

    #[cubecl::cube]
    impl ReductionOp<(u32, u32)> for PairAdd {
        fn apply(lhs: (u32, u32), rhs: (u32, u32)) -> (u32, u32) {
            (lhs.0 + rhs.0, lhs.1 + rhs.1)
        }
    }

    struct SourcePlusEdge;

    #[cubecl::cube]
    impl UnaryOp<(u32, u32)> for SourcePlusEdge {
        type Output = u32;

        fn apply(input: (u32, u32)) -> u32 {
            input.0 + input.1
        }
    }

    struct AddOne;

    #[cubecl::cube]
    impl UnaryOp<u32> for AddOne {
        type Output = u32;

        fn apply(input: u32) -> u32 {
            input + 1u32
        }
    }

    #[test]
    fn traversal_terminals_use_edge_context_expressions() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let offsets = exec.to_device(&[0_u32, 2, 3, 5]);
        let destinations = exec.to_device(&[1_u32, 2, 2, 0, 1]);
        let frontier = exec.to_device(&[2_u32, 0]);

        let emitted = traverse(
            &exec,
            Csr::new(destinations.slice(..), offsets.slice(..)),
            frontier.slice(..),
        )
        .unwrap()
        .map(destination_id(), Identity)
        .emit(&exec)
        .unwrap();
        assert_eq!(exec.to_host(&emitted).unwrap(), vec![0, 1, 1, 2]);

        let edge_values = exec.to_device(&[10_u32, 20, 30, 40, 50]);
        let source_reduced = traverse(
            &exec,
            Csr::new(destinations.slice(..), offsets.slice(..)),
            frontier.slice(..),
        )
        .unwrap()
        .map(edge(edge_values.slice(..)), Identity)
        .reduce_by_source(&exec, 0, Add)
        .unwrap();
        assert_eq!(exec.to_host(&source_reduced).unwrap(), vec![90, 30]);

        let vertex_values = exec.to_device(&[1_u32, 10, 100]);
        let destination_reduced = traverse(
            &exec,
            Csr::new(destinations.slice(..), offsets.slice(..)),
            frontier.slice(..),
        )
        .unwrap()
        .map(
            zip2(source(vertex_values.slice(..)), edge_id()),
            SourcePlusEdge,
        )
        .reduce_by_destination(&exec, 0, Add)
        .unwrap();
        assert_eq!(
            exec.to_host(&destination_reduced).unwrap(),
            vec![103, 105, 2]
        );

        let state_left = exec.to_device(&[10_u32, 20, 30]);
        let state_right = exec.to_device(&[1_u32, 2, 3]);
        traverse(
            &exec,
            Csr::new(destinations.slice(..), offsets.slice(..)),
            frontier.slice(..),
        )
        .unwrap()
        .map(
            zip2(source(vertex_values.slice(..)), edge(edge_values.slice(..))),
            Identity,
        )
        .update_by_destination(
            &exec,
            (0, 0),
            PairAdd,
            zip2(state_left.slice_mut(..), state_right.slice_mut(..)),
        )
        .unwrap();
        assert_eq!(exec.to_host(&state_left).unwrap(), vec![110, 121, 31]);
        assert_eq!(exec.to_host(&state_right).unwrap(), vec![41, 62, 23]);

        let distance = exec.to_device(&[0_u32, u32::MAX, u32::MAX]);
        let single_source = exec.to_device(&[0_u32]);
        let next = traverse(
            &exec,
            Csr::new(destinations.slice(..), offsets.slice(..)),
            single_source.slice(..),
        )
        .unwrap()
        .map(source(distance.slice(..)), AddOne)
        .relax_min_by_destination(&exec, u32::MAX, distance.slice(..), distance.slice_mut(..))
        .unwrap();
        assert_eq!(exec.to_host(&distance).unwrap(), vec![0, 1, 1]);
        assert_eq!(exec.to_host(&next).unwrap(), vec![1, 2]);
    }

    #[test]
    fn batched_intersection_counts_each_adjacency_pair() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let offsets = exec.to_device(&[0_u32, 2, 3, 5]);
        let destinations = exec.to_device(&[1_u32, 2, 2, 0, 1]);
        let sources = exec.to_device(&[0_u32, 1]);
        let targets = exec.to_device(&[1_u32, 2]);
        let counts = intersect_count(
            &exec,
            Csr::new(destinations.slice(..), offsets.slice(..)),
            sources.slice(..),
            targets.slice(..),
        )
        .unwrap();

        assert_eq!(exec.to_host(&counts).unwrap(), vec![1, 0]);
    }
}
