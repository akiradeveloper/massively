//! Semantic traversal construction and terminal operations.

// The private bound seals edge expressions while keeping traversal control out of the API.
#![allow(private_bounds, private_interfaces)]

use cubecl::prelude::*;

use crate::{
    Error, Executor, MAlloc, MIndex, MIter, MIterMut, MStorage, MVal, MVec, op::BinaryPredicateOp,
    op::ReductionOp, op::UnaryOp, seg::SegmentIterator,
};

use super::{
    Csr,
    control::TraversalControl,
    expr::{EdgeExpr, EdgeExprImpl},
};

struct IndexLess;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for IndexLess {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs < rhs
    }
}

struct IndexEqual;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for IndexEqual {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs == rhs
    }
}

struct MinU32;

#[cubecl::cube]
impl ReductionOp<u32> for MinU32 {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        u32::min(lhs, rhs)
    }
}

struct ApplyMinU32;

#[cubecl::cube]
impl UnaryOp<(u32, u32)> for ApplyMinU32 {
    type Output = u32;

    fn apply(input: (u32, u32)) -> u32 {
        u32::min(input.0, input.1)
    }
}

struct LoweredU32;

#[cubecl::cube]
impl UnaryOp<(u32, u32)> for LoweredU32 {
    type Output = u32;

    fn apply(input: (u32, u32)) -> u32 {
        if input.1 < input.0 { 1u32 } else { 0u32 }
    }
}

/// A selected, ordered stream of graph edges.
///
/// The stream is semantic: private expansion data may be materialized by the current lowering,
/// but is not part of this API and can be replaced by fused edge kernels.
pub struct Traversal<R: Runtime> {
    control: TraversalControl<R>,
}

/// An edge traversal with a lazy input expression and map operation.
pub struct MappedTraversal<R: Runtime, Expr, Map> {
    traversal: Traversal<R>,
    expr: Expr,
    map: Map,
}

/// Selects all outgoing edges of the vertices in `frontier`.
///
/// `max_edges` is a host-known physical allocation bound. The actual number of
/// selected edges remains internal and device-resident until
/// [`Traversal::edge_count`] or a variable-length terminal requests it. If
/// the bound is too small, terminal operations safely process its prefix.
/// Construction and composition do not read device-produced lengths back.
pub fn traverse<R, Destinations, Offsets, Frontier>(
    exec: &Executor<R>,
    graph: Csr<Destinations, Offsets>,
    frontier: Frontier,
    max_edges: MIndex,
) -> Result<Traversal<R>, Error>
where
    R: Runtime,
    Destinations: MIter<R, Item = MIndex>,
    Offsets: MIter<R, Item = MIndex>,
    Frontier: MIter<R, Item = MIndex>,
{
    let (destinations, offsets) = graph.into_parts();
    Ok(Traversal {
        control: TraversalControl::new(exec, destinations, offsets, frontier, max_edges)?,
    })
}

impl<R: Runtime> Traversal<R> {
    /// Number of edges selected by this traversal.
    ///
    /// This can exceed [`Self::capacity`] when the supplied physical bound was
    /// too small.  Terminal results contain the first `capacity` edges.
    /// This is an explicit synchronization boundary.
    pub fn edge_count(&self, exec: &Executor<R>) -> Result<MIndex, Error> {
        self.control.required_len.read(exec)
    }

    /// Host-known physical edge capacity supplied to [`traverse`].
    pub const fn capacity(&self) -> MIndex {
        self.control.capacity
    }

    /// Whether the selected edge stream fits in the physical capacity.
    ///
    /// This is an explicit synchronization boundary.
    pub fn fits(&self, exec: &Executor<R>) -> Result<bool, Error> {
        Ok(self.edge_count(exec)? <= self.capacity())
    }

    /// Attaches a lazy edge expression and map operation.
    pub fn map<Expr, Map>(self, expr: Expr, map: Map) -> MappedTraversal<R, Expr, Map> {
        MappedTraversal {
            traversal: self,
            expr,
            map,
        }
    }
}

impl<R, Expr, Map> MappedTraversal<R, Expr, Map>
where
    R: Runtime,
    Expr: EdgeExpr<R> + EdgeExprImpl<R>,
    Map: UnaryOp<Expr::Item>,
    Map::Output: MAlloc<R> + Copy,
{
    /// Returns mapped storage with an exact host-visible logical length.
    pub fn emit(self, exec: &Executor<R>) -> Result<MVec<R, Map::Output>, Error> {
        let capacity = self.traversal.control.capacity as usize;
        let output = exec.alloc::<Map::Output>(capacity);
        let len = self.emit_into(exec, output.slice_mut(..))?;
        crate::api::iter::into_exact_prefix::<R, Map::Output>(exec, output, len.read(exec)?)
    }

    /// Writes one mapped item per traversed edge into caller-provided storage.
    fn emit_into<Output>(self, exec: &Executor<R>, output: Output) -> Result<MVal<R, MIndex>, Error>
    where
        Output: MIterMut<R, Item = Map::Output>,
    {
        let len = self.traversal.control.output_len.clone();
        let input = self.expr.materialize(exec, &self.traversal.control)?;
        crate::api::algorithm::transform::transform_prefix_into(
            exec,
            input.slice(..),
            self.map,
            len.scratch_storage(),
            output,
        )?;
        Ok(len)
    }
}

impl<R, Expr, Map> MappedTraversal<R, Expr, Map>
where
    R: Runtime,
    Expr: EdgeExpr<R> + EdgeExprImpl<R>,
    Map: UnaryOp<Expr::Item>,
    Map::Output: MAlloc<R> + Copy,
{
    /// Maps every selected edge and reduces results independently for each input source.
    pub fn reduce_by_source<ReduceOp>(
        self,
        exec: &Executor<R>,
        init: Map::Output,
        reduce: ReduceOp,
    ) -> Result<MVec<R, Map::Output>, Error>
    where
        ReduceOp: ReductionOp<Map::Output>,
    {
        let init = exec.value(init)?;
        let output = exec.alloc::<Map::Output>(self.traversal.control.source_count as usize);
        self.reduce_by_source_into(exec, init, reduce, output.slice_mut(..))?;
        Ok(output)
    }

    /// Reduces independently for each input source into caller-provided storage.
    fn reduce_by_source_into<Output, ReduceOp>(
        self,
        exec: &Executor<R>,
        init: MVal<R, Map::Output>,
        reduce: ReduceOp,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R, Item = Map::Output>,
        ReduceOp: ReductionOp<Map::Output>,
    {
        let input = self.expr.materialize(exec, &self.traversal.control)?;
        let mapped =
            exec.full_value::<Map::Output>(self.traversal.control.capacity as usize, &init)?;
        crate::api::algorithm::transform::transform_prefix_into(
            exec,
            input.slice(..),
            self.map,
            self.traversal.control.output_len.scratch_storage(),
            mapped.slice_mut(..),
        )?;
        crate::seg::reduce_segments(
            exec,
            SegmentIterator::new(
                mapped.slice(..),
                self.traversal.control.output_offsets.slice(..),
            ),
            init,
            reduce,
            output,
        )
    }

    /// Maps every selected edge and reduces colliding results by destination vertex.
    pub fn reduce_by_destination<ReduceOp>(
        self,
        exec: &Executor<R>,
        init: Map::Output,
        reduce: ReduceOp,
    ) -> Result<MVec<R, Map::Output>, Error>
    where
        ReduceOp: ReductionOp<Map::Output>,
    {
        let init = exec.value(init)?;
        let output = exec.alloc::<Map::Output>(self.traversal.control.vertex_count as usize);
        crate::api::algorithm::fill_value(exec, &init, output.slice_mut(..))?;
        self.reduce_by_destination_into(exec, init, reduce, output.slice_mut(..))?;
        Ok(output)
    }

    /// Reduces by destination into caller-provided initialized storage.
    fn reduce_by_destination_into<Output, ReduceOp>(
        self,
        exec: &Executor<R>,
        init: MVal<R, Map::Output>,
        reduce: ReduceOp,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R, Item = Map::Output>,
        ReduceOp: ReductionOp<Map::Output>,
    {
        let input = self.expr.materialize(exec, &self.traversal.control)?;
        let mapped =
            exec.full_value::<Map::Output>(self.traversal.control.capacity as usize, &init)?;
        crate::api::algorithm::transform::transform_prefix_into(
            exec,
            input.slice(..),
            self.map,
            self.traversal.control.output_len.scratch_storage(),
            mapped.slice_mut(..),
        )?;
        crate::api::algorithm::scatter_reduce_value(
            exec,
            mapped.slice(..),
            self.traversal.control.destinations.slice(..),
            init,
            reduce,
            output,
        )
    }

    /// Reduces proposals by destination and combines them into existing vertex state.
    ///
    /// This is the stateful interpretation of [`reduce_by_destination`](Self::reduce_by_destination):
    /// the destination output is read, combined with its reduced proposal, and written back. The
    /// state item may be multi-column.
    pub fn update_by_destination<StateOutput, ReduceOp>(
        self,
        exec: &Executor<R>,
        proposal_init: Map::Output,
        reduce: ReduceOp,
        state_output: StateOutput,
    ) -> Result<(), Error>
    where
        StateOutput: MIterMut<R, Item = Map::Output>,
        ReduceOp: ReductionOp<Map::Output>,
    {
        self.reduce_by_destination_into(exec, exec.value(proposal_init)?, reduce, state_output)
    }

    /// Applies minimum proposals by destination and emits vertices whose state decreased.
    ///
    /// The generic lowering sorts proposals to guarantee one writer per destination. A future
    /// atomic-min lowering can preserve this contract without changing traversal programs.
    pub fn relax_min_by_destination<State, StateOutput>(
        self,
        exec: &Executor<R>,
        infinity: u32,
        state: State,
        state_output: StateOutput,
    ) -> Result<MVec<R, u32>, Error>
    where
        Map: UnaryOp<Expr::Item, Output = u32>,
        State: MIter<R, Item = u32>,
        StateOutput: MIterMut<R, Item = u32>,
        u32: crate::RowAlloc<R, RowStorage = crate::DeviceVec<R, u32>>
            + MAlloc<R, Owned = crate::DeviceVec<R, u32>>,
    {
        let capacity = self.traversal.control.capacity as usize;
        let infinity = exec.value(infinity)?;
        let next = exec.alloc::<u32>(capacity);
        let len = self.relax_min_by_destination_into(
            exec,
            infinity,
            state,
            state_output,
            next.slice_mut(..),
        )?;
        crate::api::iter::into_exact_prefix::<R, u32>(exec, next, len.read(exec)?)
    }

    /// Stateful caller-provided output variant of [`Self::relax_min_by_destination`].
    fn relax_min_by_destination_into<State, StateOutput, Next>(
        self,
        exec: &Executor<R>,
        infinity: MVal<R, u32>,
        state: State,
        state_output: StateOutput,
        next: Next,
    ) -> Result<MVal<R, MIndex>, Error>
    where
        Map: UnaryOp<Expr::Item, Output = u32>,
        State: MIter<R, Item = u32>,
        StateOutput: MIterMut<R, Item = u32>,
        Next: MIterMut<R, Item = u32>,
        u32: crate::RowAlloc<R, RowStorage = crate::DeviceVec<R, u32>>
            + MAlloc<R, Owned = crate::DeviceVec<R, u32>>,
    {
        let edge_capacity = self.traversal.control.capacity;
        if edge_capacity == 0 {
            return exec.value(0);
        }

        let input = self.expr.materialize(exec, &self.traversal.control)?;
        let proposals = exec.full_value(edge_capacity as usize, &infinity)?;
        crate::api::algorithm::transform::transform_prefix_into(
            exec,
            input.slice(..),
            self.map,
            self.traversal.control.output_len.scratch_storage(),
            proposals.slice_mut(..),
        )?;

        let sorted_destinations = exec.alloc_column::<u32>(edge_capacity as usize);
        let sorted_proposals = exec.alloc_column::<u32>(edge_capacity as usize);
        crate::vector::sort_by_key_into(
            exec,
            self.traversal.control.destinations.slice(..),
            proposals.slice(..),
            IndexLess,
            sorted_destinations.slice_mut(..),
            sorted_proposals.slice_mut(..),
        )?;

        let unique_destinations = exec.alloc_column::<u32>(edge_capacity as usize);
        let reduced_proposals = exec.alloc_column::<u32>(edge_capacity as usize);
        let zero = exec.value(0u32)?;
        crate::api::algorithm::fill_value(exec, &zero, unique_destinations.slice_mut(..))?;
        crate::api::algorithm::fill_value(exec, &infinity, reduced_proposals.slice_mut(..))?;
        let unique_len = crate::vector::reduce_by_key_into(
            exec,
            sorted_destinations.slice(..),
            sorted_proposals.slice(..),
            IndexEqual,
            infinity,
            MinU32,
            unique_destinations.slice_mut(..),
            reduced_proposals.slice_mut(..),
        )?;

        let old_state = exec.alloc_column::<u32>(edge_capacity as usize);
        crate::vector::gather_into(
            exec,
            state,
            unique_destinations.slice(..),
            old_state.slice_mut(..),
        )?;

        let state_and_proposal = || crate::zip2(old_state.slice(..), reduced_proposals.slice(..));
        let flags = exec.alloc_column::<u32>(edge_capacity as usize);
        crate::vector::transform_into(exec, state_and_proposal(), LoweredU32, flags.slice_mut(..))?;

        let new_state = exec.alloc_column::<u32>(edge_capacity as usize);
        crate::vector::transform_into(
            exec,
            state_and_proposal(),
            ApplyMinU32,
            new_state.slice_mut(..),
        )?;
        crate::vector::scatter_prefix(
            exec,
            new_state.slice(..),
            unique_destinations.slice(..),
            unique_len.scratch_storage(),
            state_output,
        )?;

        crate::vector::copy_where_into(
            exec,
            unique_destinations.slice(..),
            crate::lazy::map(flags.slice(..), crate::op::NonZero),
            next,
        )
    }
}
