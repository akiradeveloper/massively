//! Semantic traversal construction and terminal operations.

// The private bound seals edge expressions while keeping traversal control out of the API.
#![allow(private_bounds, private_interfaces)]

use cubecl::prelude::*;

use crate::{
    Allocable, Canonicalizable, Error, Executor, MIndex, MIter, MIterMut, MStorage, MVec,
    Materializable, WritableFrom,
    op::BinaryPredicateOp,
    op::ReductionOp,
    op::UnaryOp,
    seg::{ForEachSegment, Reduce, SegmentIterator, SummarizingExecutableInto},
};

use super::{Csr, EdgeExpr, control::TraversalControl, expr::EdgeExprImpl};

struct IndexLess;

#[cubecl::cube]
impl BinaryPredicateOp<MIndex> for IndexLess {
    fn apply(lhs: MIndex, rhs: MIndex) -> bool {
        lhs < rhs
    }
}

struct IndexEqual;

#[cubecl::cube]
impl BinaryPredicateOp<MIndex> for IndexEqual {
    fn apply(lhs: MIndex, rhs: MIndex) -> bool {
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
pub fn traverse<R, Destinations, Offsets, Frontier>(
    exec: &Executor<R>,
    graph: Csr<Destinations, Offsets>,
    frontier: Frontier,
) -> Result<Traversal<R>, Error>
where
    R: Runtime,
    Destinations: MIter<R, Item = MIndex>,
    Offsets: MIter<R, Item = MIndex>,
    Frontier: MIter<R, Item = MIndex>,
{
    let (destinations, offsets) = graph.into_parts();
    Ok(Traversal {
        control: TraversalControl::new(exec, destinations, offsets, frontier)?,
    })
}

impl<R: Runtime> Traversal<R> {
    /// Number of edges selected by this traversal.
    pub const fn edge_count(&self) -> MIndex {
        self.control.output_len
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
    Map::Output: Allocable<R>
        + Canonicalizable<Canonical = Map::Output>
        + Materializable<R, Materialized = Map::Output>
        + Copy,
{
    /// Returns one mapped item per traversed edge.
    pub fn emit(self, exec: &Executor<R>) -> Result<MVec<R, Map::Output>, Error> {
        let output = exec.alloc_mvec::<Map::Output>(self.traversal.control.output_len as usize);
        self.emit_into(exec, output.slice_mut(..))?;
        Ok(output)
    }

    /// Writes one mapped item per traversed edge into caller-provided storage.
    fn emit_into<Output>(self, exec: &Executor<R>, output: Output) -> Result<(), Error>
    where
        Output: MIterMut<R>,
        Output::Item: WritableFrom<Map::Output>,
    {
        let input = self.expr.materialize(exec, &self.traversal.control)?;
        crate::vector::transform_into(exec, input.slice(..), self.map, output)
    }

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
        let output = exec.alloc_mvec::<Map::Output>(self.traversal.control.source_count as usize);
        self.reduce_by_source_into(exec, init, reduce, output.slice_mut(..))?;
        Ok(output)
    }

    /// Reduces independently for each input source into caller-provided storage.
    fn reduce_by_source_into<Output, ReduceOp>(
        self,
        exec: &Executor<R>,
        init: Map::Output,
        reduce: ReduceOp,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R>,
        Output::Item: WritableFrom<Map::Output>,
        ReduceOp: ReductionOp<Map::Output>,
    {
        let input = self.expr.materialize(exec, &self.traversal.control)?;
        let mapped =
            <Map::Output as Allocable<R>>::alloc(exec, self.traversal.control.output_len as usize);
        crate::vector::transform_into(exec, input.slice(..), self.map, mapped.slice_mut(..))?;
        ForEachSegment(Reduce(reduce, init)).run_into(
            exec,
            SegmentIterator::new(
                mapped.slice(..),
                self.traversal.control.output_offsets.slice(..),
            ),
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
        let output = crate::vector::fill(exec, self.traversal.control.vertex_count as usize, init)?;
        self.reduce_by_destination_into(exec, init, reduce, output.slice_mut(..))?;
        Ok(output)
    }

    /// Reduces by destination into caller-provided initialized storage.
    fn reduce_by_destination_into<Output, ReduceOp>(
        self,
        exec: &Executor<R>,
        init: Map::Output,
        reduce: ReduceOp,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R, Item = Map::Output>,
        ReduceOp: ReductionOp<Map::Output>,
    {
        let input = self.expr.materialize(exec, &self.traversal.control)?;
        let mapped =
            <Map::Output as Allocable<R>>::alloc(exec, self.traversal.control.output_len as usize);
        crate::vector::transform_into(exec, input.slice(..), self.map, mapped.slice_mut(..))?;
        crate::vector::scatter_reduce(
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
        self.reduce_by_destination_into(exec, proposal_init, reduce, state_output)
    }

    /// Applies minimum proposals by destination and emits vertices whose state decreased.
    ///
    /// The generic lowering sorts proposals to guarantee one writer per destination. A future
    /// atomic-min lowering can preserve this contract without changing traversal programs.
    pub fn relax_min_by_destination<State, StateOutput>(
        self,
        exec: &Executor<R>,
        infinity: MIndex,
        state: State,
        state_output: StateOutput,
    ) -> Result<MVec<R, MIndex>, Error>
    where
        Map: UnaryOp<Expr::Item, Output = MIndex>,
        State: MIter<R, Item = MIndex>,
        StateOutput: MIterMut<R, Item = MIndex>,
        MIndex: crate::CanonicalAlloc<R, CanonicalStorage = crate::DeviceVec<R, MIndex>>,
    {
        let capacity = self.traversal.control.output_len as usize;
        let mut next = exec.alloc_mvec::<MIndex>(capacity);
        let len = self.relax_min_by_destination_into(
            exec,
            infinity,
            state,
            state_output,
            next.slice_mut(..),
        )?;
        next.truncate(len);
        Ok(next)
    }

    /// Stateful caller-provided output variant of [`Self::relax_min_by_destination`].
    fn relax_min_by_destination_into<State, StateOutput, Next>(
        self,
        exec: &Executor<R>,
        infinity: MIndex,
        state: State,
        state_output: StateOutput,
        next: Next,
    ) -> Result<MIndex, Error>
    where
        Map: UnaryOp<Expr::Item, Output = MIndex>,
        State: MIter<R, Item = MIndex>,
        StateOutput: MIterMut<R, Item = MIndex>,
        Next: MIterMut<R, Item = MIndex>,
        MIndex: crate::CanonicalAlloc<R, CanonicalStorage = crate::DeviceVec<R, MIndex>>,
    {
        let edge_count = self.traversal.control.output_len;
        if edge_count == 0 {
            return Ok(0);
        }

        let input = self.expr.materialize(exec, &self.traversal.control)?;
        let proposals = exec.alloc_column::<MIndex>(edge_count as usize);
        crate::vector::transform_into(exec, input.slice(..), self.map, proposals.slice_mut(..))?;

        let sorted_destinations = exec.alloc_column::<MIndex>(edge_count as usize);
        let sorted_proposals = exec.alloc_column::<MIndex>(edge_count as usize);
        crate::vector::sort_by_key_into(
            exec,
            self.traversal.control.destinations.slice(..),
            proposals.slice(..),
            IndexLess,
            sorted_destinations.slice_mut(..),
            sorted_proposals.slice_mut(..),
        )?;

        let unique_destinations = exec.alloc_column::<MIndex>(edge_count as usize);
        let reduced_proposals = exec.alloc_column::<MIndex>(edge_count as usize);
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

        let old_state = exec.alloc_column::<MIndex>(unique_len as usize);
        crate::vector::gather_into(
            exec,
            state,
            unique_destinations.slice(..unique_len as usize),
            old_state.slice_mut(..),
        )?;

        let state_and_proposal = || {
            crate::zip2(
                old_state.slice(..),
                reduced_proposals.slice(..unique_len as usize),
            )
        };
        let flags = exec.alloc_column::<MIndex>(unique_len as usize);
        crate::vector::transform_into(exec, state_and_proposal(), LoweredU32, flags.slice_mut(..))?;

        let new_state = exec.alloc_column::<MIndex>(unique_len as usize);
        crate::vector::transform_into(
            exec,
            state_and_proposal(),
            ApplyMinU32,
            new_state.slice_mut(..),
        )?;
        crate::vector::scatter(
            exec,
            new_state.slice(..),
            unique_destinations.slice(..unique_len as usize),
            state_output,
        )?;

        crate::vector::copy_where_into(
            exec,
            unique_destinations.slice(..unique_len as usize),
            flags.slice(..),
            next,
        )
    }
}
