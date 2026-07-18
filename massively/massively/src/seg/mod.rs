//! Algorithms applied independently to offset-delimited segments.
//!
//! A [`SegmentIterator`] value combines a flat [`MIter`] with offsets. Segment `i` is
//! the half-open range `offsets[i]..offsets[i + 1]` in the flat values.
//!
//! Segment algorithms return owned device storage. Length-preserving algorithms
//! return new flat values with the original offsets, length-changing algorithms
//! return values with rebuilt offsets, and reductions return one value per
//! segment.

pub(crate) mod control;
mod segment;

pub use segment::Segment;
pub(crate) use segment::{SegmentExpand, SegmentReader};

use cubecl::prelude::*;
use std::{marker::PhantomData, ops::RangeBounds};

use crate::{
    Error, Executor, MAllocItem, MIter, MIterMut, MStorage, MVec, op::BinaryPredicateOp,
    op::PredicateOp, op::ReductionOp, op::UnaryOp,
};

/// A flat value stream and the offsets delimiting its segments.
///
/// When `Values` and `Offsets` are logical iterators, this is itself an
/// [`MIter`] whose item is [`Segment<Values::Item>`](Segment). Each item is a
/// read-only adapter over the shared value stream and supports `len()` and
/// unchecked `at()` access inside CubeCL operations.
#[derive(Clone, Copy, Debug)]
pub struct SegmentIterator<Values, Offsets> {
    values: Values,
    offsets: Offsets,
}

impl<Values, Offsets> SegmentIterator<Values, Offsets> {
    /// Combines flat values and offsets into a segment iterator.
    pub const fn new(values: Values, offsets: Offsets) -> Self {
        Self { values, offsets }
    }

    /// Returns the flat values.
    pub const fn values(&self) -> &Values {
        &self.values
    }

    /// Returns the segment offsets.
    pub const fn offsets(&self) -> &Offsets {
        &self.offsets
    }

    /// Decomposes this input into its flat values and offsets.
    pub fn into_parts(self) -> (Values, Offsets) {
        (self.values, self.offsets)
    }
}

/// Physical segmented read created when a logical [`SegmentIterator`] is consumed.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct SegmentRead<Values, Offsets> {
    values: Values,
    offsets: Offsets,
}

impl<Values, Offsets> SegmentRead<Values, Offsets> {
    pub(crate) const fn new(values: Values, offsets: Offsets) -> Self {
        Self { values, offsets }
    }

    pub(crate) const fn values(&self) -> &Values {
        &self.values
    }

    pub(crate) const fn offsets(&self) -> &Offsets {
        &self.offsets
    }
}

#[doc(hidden)]
impl<R, Values, Offsets> MIter<R> for SegmentIterator<Values, Offsets>
where
    R: Runtime,
    Values: MIter<R>,
    Offsets: MIter<R, Item = u32>,
    SegmentRead<Values::Read, Offsets::Read>: crate::core::facade::KernelInput<R, Item = Segment<Values::Item>>
        + crate::core::facade::IterLength
        + crate::read::SliceExpression,
{
    type Item = Segment<Values::Item>;
    type Read = SegmentRead<Values::Read, Offsets::Read>;
    type Slice = crate::read::Slice<R, Self::Read>;

    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice
    where
        Bounds: RangeBounds<usize>,
    {
        let input = self.clone().lower_read();
        let len = crate::core::facade::IterLength::logical_len(&input)
            .expect("cannot slice segmented input with an invalid length");
        let (start, count) = crate::api::iter::resolve_iter_range(len, range);
        crate::read::Slice::new(crate::read::SliceExpression::slice_expression(
            &input, start, count,
        ))
    }

    fn len(&self) -> Result<usize, Error> {
        Ok(MIter::len(&self.offsets)?.saturating_sub(1))
    }

    fn lower_read(self) -> Self::Read {
        SegmentRead::new(self.values.lower_read(), self.offsets.lower_read())
    }
}

/// Mutable flat values and offsets for a length-changing segmented result.
#[derive(Debug)]
pub(crate) struct SegmentedMut<Values, Offsets> {
    values: Values,
    offsets: Offsets,
}

impl<Values, Offsets> SegmentedMut<Values, Offsets> {
    /// Combines preallocated flat value and offset outputs.
    pub(crate) const fn new(values: Values, offsets: Offsets) -> Self {
        Self { values, offsets }
    }

    /// Decomposes this output into its flat values and offsets.
    pub(crate) fn into_parts(self) -> (Values, Offsets) {
        (self.values, self.offsets)
    }
}

/// Lifts one ordinary algorithm so it runs independently for every segment.
#[derive(Clone, Copy, Debug)]
pub struct ForEachSegment<Algorithm>(pub Algorithm);

/// Transforms every item and returns the new values with the original offsets.
#[derive(Clone, Copy, Debug)]
pub struct Transform<Op>(pub Op);

/// Stably sorts the items within every segment and preserves the offsets.
#[derive(Clone, Copy, Debug)]
pub struct Sort<Less>(pub Less);

/// Reverses the items within every segment and preserves the offsets.
#[derive(Clone, Copy, Debug, Default)]
pub struct Reverse;

/// Computes an inclusive scan within every segment and preserves the offsets.
#[derive(Clone, Copy, Debug)]
pub struct InclusiveScan<Op>(pub Op);

/// Computes an exclusive scan within every segment and preserves the offsets.
///
/// Each segment starts from `init`.
#[derive(Clone, Copy, Debug)]
pub struct ExclusiveScan<Op, Init>(pub Op, pub Init);

/// Computes adjacent differences within every segment and preserves the offsets.
#[derive(Clone, Copy, Debug)]
pub struct AdjacentDifference<Op>(pub Op);

/// Keeps the first item of each adjacent-equal run within every segment.
#[derive(Clone, Copy, Debug)]
pub struct Unique<Equal>(pub Equal);

/// Keeps the items satisfying a predicate within every segment.
#[derive(Clone, Copy, Debug)]
pub struct Filter<Pred>(pub Pred);

/// Reduces every segment to one item, starting from `init`.
#[derive(Clone, Copy, Debug)]
pub struct Reduce<Op, Init>(pub Op, pub Init);

/// Counts the items satisfying a predicate within every segment.
#[derive(Clone, Copy, Debug)]
pub struct CountIf<Pred>(pub Pred);

/// Tests whether every item satisfies a predicate within every segment.
#[derive(Clone, Copy, Debug)]
pub struct AllOf<Pred>(pub Pred);

/// Tests whether any item satisfies a predicate within every segment.
#[derive(Clone, Copy, Debug)]
pub struct AnyOf<Pred>(pub Pred);

/// Tests whether no item satisfies a predicate within every segment.
#[derive(Clone, Copy, Debug)]
pub struct NoneOf<Pred>(pub Pred);

/// Tests whether every segment is sorted.
#[derive(Clone, Copy, Debug)]
pub struct IsSorted<Less>(pub Less);

/// Finds the sorted prefix length of every segment.
#[derive(Clone, Copy, Debug)]
pub struct IsSortedUntil<Less>(pub Less);

/// Execution contract for segment algorithms that preserve segment lengths.
///
/// The output contains only flat values. Its segment offsets are the input
/// offsets and therefore do not need to be copied.
#[doc(hidden)]
pub(crate) trait LengthPreservingExecutableInto<R, Input, InputOffsets, Output>:
    Sized
where
    R: Runtime,
    Input: MIter<R>,
    InputOffsets: MIter<R, Item = u32>,
    Output: MIterMut<R>,
{
    /// Executes the algorithm for every segment.
    fn run_into(
        self,
        exec: &Executor<R>,
        input: SegmentIterator<Input, InputOffsets>,
        output: Output,
    ) -> Result<(), Error>;
}

/// Execution contract for algorithms that compact each segment independently.
///
/// Output segment lengths never exceed their corresponding input lengths. The
/// caller provides capacity for both flat values and rebuilt offsets, and the
/// returned index is the total number of flat values written.
#[doc(hidden)]
pub(crate) trait CompactingExecutableInto<R, Input, InputOffsets, Output, OutputOffsets>:
    Sized
where
    R: Runtime,
    Input: MIter<R>,
    InputOffsets: MIter<R, Item = u32>,
    Output: MIterMut<R>,
    OutputOffsets: MIterMut<R, Item = u32>,
{
    /// Executes the algorithm for every segment.
    fn run_into(
        self,
        exec: &Executor<R>,
        input: SegmentIterator<Input, InputOffsets>,
        output: SegmentedMut<Output, OutputOffsets>,
    ) -> Result<u32, Error>;
}

/// Execution contract for segment algorithms that write one item per segment.
#[doc(hidden)]
pub(crate) trait SummarizingExecutableInto<R, Input, InputOffsets, Output>: Sized
where
    R: Runtime,
    Input: MIter<R>,
    InputOffsets: MIter<R, Item = u32>,
    Output: MIterMut<R>,
{
    /// Executes the algorithm for every segment.
    fn run_into(
        self,
        exec: &Executor<R>,
        input: SegmentIterator<Input, InputOffsets>,
        output: Output,
    ) -> Result<(), Error>;
}

/// Owned-result execution contract for segment algorithms.
///
/// Each algorithm selects its complete output shape. Length-preserving
/// algorithms return new values with the input offsets, compacting algorithms
/// return rebuilt segments, and summarizing algorithms return one value per
/// segment.
pub trait Executable<R, Input, InputOffsets>: Sized
where
    R: Runtime,
    Input: MIter<R>,
    InputOffsets: MIter<R, Item = u32>,
{
    /// Complete owned result selected by the segment algorithm.
    type Output;

    /// Executes this algorithm independently for every input segment.
    fn run(
        self,
        exec: &Executor<R>,
        input: SegmentIterator<Input, InputOffsets>,
    ) -> Result<Self::Output, Error>;
}

impl<R, Input, InputOffsets, Op> Executable<R, Input, InputOffsets>
    for ForEachSegment<Transform<Op>>
where
    R: Runtime,
    Input: MIter<R>,
    InputOffsets: MIter<R, Item = u32>,
    Op: UnaryOp<Input::Item>,
    Op::Output: MAllocItem<R>,
{
    type Output = SegmentIterator<MVec<R, Op::Output>, InputOffsets>;

    fn run(
        self,
        exec: &Executor<R>,
        input: SegmentIterator<Input, InputOffsets>,
    ) -> Result<Self::Output, Error> {
        let Transform(op) = self.0;
        let (values, offsets) = input.into_parts();
        let output = crate::vector::transform(exec, values, op)?;
        Ok(SegmentIterator::new(output, offsets))
    }
}

impl<R, Input, InputOffsets, Output, Op>
    LengthPreservingExecutableInto<R, Input, InputOffsets, Output> for ForEachSegment<Transform<Op>>
where
    R: Runtime,
    Input: MIter<R>,
    InputOffsets: MIter<R, Item = u32>,
    Output: MIterMut<R>,
    Op: UnaryOp<Input::Item, Output = Output::Item>,
{
    fn run_into(
        self,
        exec: &Executor<R>,
        input: SegmentIterator<Input, InputOffsets>,
        output: Output,
    ) -> Result<(), Error> {
        let Transform(op) = self.0;
        let (values, _) = input.into_parts();
        crate::api::algorithm::transform::transform_into(exec, values, op, output)
    }
}

impl<R, Input, InputOffsets, Output, Less>
    LengthPreservingExecutableInto<R, Input, InputOffsets, Output> for ForEachSegment<Sort<Less>>
where
    R: Runtime,
    Input: MIter<R, Item = Output::Item>,
    InputOffsets: MIter<R, Item = u32>,
    Output: MIterMut<R>,
    Less: BinaryPredicateOp<Input::Item>,
{
    fn run_into(
        self,
        exec: &Executor<R>,
        input: SegmentIterator<Input, InputOffsets>,
        output: Output,
    ) -> Result<(), Error> {
        let Sort(less) = self.0;
        let (values, offsets) = input.into_parts();
        let value_len = values.len()? as usize;
        let control = control::SegmentControl::new(exec, offsets, value_len)?;

        let values = crate::api::iter::lower_fixed::<R, _>(values);
        let ordering = crate::ordering::sort_control_with(exec, values.clone(), less)?;
        let sorted_ids = exec.alloc::<u32>(value_len);
        crate::api::algorithm::apply_permutation_into(
            exec,
            crate::api::iter::lower_fixed::<R, _>(control.ids.slice(..)),
            ordering.column(),
            sorted_ids.slice_mut(..),
        )?;

        let id_ordering = crate::ordering::sort_control_with(
            exec,
            crate::api::iter::lower_fixed::<R, _>(sorted_ids.slice(..)),
            control::LessU32,
        )?;
        let permutation = exec.alloc::<u32>(value_len);
        crate::api::algorithm::apply_permutation_into(
            exec,
            crate::api::iter::lower_fixed::<R, _>(ordering.slice(..)),
            id_ordering.column(),
            permutation.slice_mut(..),
        )?;
        crate::api::algorithm::apply_permutation_into(exec, values, permutation.column(), output)
    }
}

impl<R, Input, InputOffsets, Output, Op>
    LengthPreservingExecutableInto<R, Input, InputOffsets, Output>
    for ForEachSegment<AdjacentDifference<Op>>
where
    R: Runtime,
    Input: MIter<R, Item = Output::Item> + Clone,
    InputOffsets: MIter<R, Item = u32>,
    Output: MIterMut<R>,
    Output::SliceMut: MIterMut<R, Item = Output::Item>,
    Op: ReductionOp<Input::Item>,
{
    fn run_into(
        self,
        exec: &Executor<R>,
        input: SegmentIterator<Input, InputOffsets>,
        output: Output,
    ) -> Result<(), Error> {
        let AdjacentDifference(op) = self.0;
        let (values, offsets) = input.into_parts();
        let control = control::SegmentControl::new(exec, offsets, values.len()? as usize)?;
        crate::vector::adjacent_difference_into(exec, values.clone(), op, output.slice_mut(..))?;
        crate::vector::transform_where_raw(
            exec,
            values,
            crate::op::Identity,
            control.heads.slice(..),
            output,
        )
    }
}

impl<R, Input, InputOffsets, Output, Op>
    LengthPreservingExecutableInto<R, Input, InputOffsets, Output>
    for ForEachSegment<InclusiveScan<Op>>
where
    R: Runtime,
    Input: MIter<R, Item = Output::Item>,
    InputOffsets: MIter<R, Item = u32>,
    Output: MIterMut<R>,
    Op: ReductionOp<Input::Item>,
{
    fn run_into(
        self,
        exec: &Executor<R>,
        input: SegmentIterator<Input, InputOffsets>,
        output: Output,
    ) -> Result<(), Error> {
        let InclusiveScan(op) = self.0;
        let (values, offsets) = input.into_parts();
        let control = control::SegmentControl::new(exec, offsets, values.len()? as usize)?;
        crate::vector::inclusive_scan_by_key_into(
            exec,
            control.ids.slice(..),
            values,
            control::EqualU32,
            op,
            output,
        )
    }
}

impl<R, Input, InputOffsets, Output, Op, Init>
    LengthPreservingExecutableInto<R, Input, InputOffsets, Output>
    for ForEachSegment<ExclusiveScan<Op, Init>>
where
    R: Runtime,
    Input: MIter<R, Item = Init>,
    InputOffsets: MIter<R, Item = u32>,
    Output: MIterMut<R, Item = Init>,
    Init: CubeType,
    Op: ReductionOp<Init>,
{
    fn run_into(
        self,
        exec: &Executor<R>,
        input: SegmentIterator<Input, InputOffsets>,
        output: Output,
    ) -> Result<(), Error> {
        let ExclusiveScan(op, init) = self.0;
        let (values, offsets) = input.into_parts();
        let control = control::SegmentControl::new(exec, offsets, values.len()? as usize)?;
        crate::vector::exclusive_scan_by_key_into(
            exec,
            control.ids.slice(..),
            values,
            control::EqualU32,
            init,
            op,
            output,
        )
    }
}

impl<R, Input, InputOffsets, Output> LengthPreservingExecutableInto<R, Input, InputOffsets, Output>
    for ForEachSegment<Reverse>
where
    R: Runtime,
    Input: MIter<R, Item = Output::Item>,
    InputOffsets: MIter<R, Item = u32>,
    Output: MIterMut<R>,
{
    fn run_into(
        self,
        exec: &Executor<R>,
        input: SegmentIterator<Input, InputOffsets>,
        output: Output,
    ) -> Result<(), Error> {
        let (values, offsets) = input.into_parts();
        let control = control::SegmentControl::new(exec, offsets, values.len()? as usize)?;
        let indices = control.reverse_indices(exec)?;
        crate::vector::gather_raw_into(exec, values, indices.slice(..), output)
    }
}

impl<R, Input, InputOffsets, Output, OutputOffsets, Equal>
    CompactingExecutableInto<R, Input, InputOffsets, Output, OutputOffsets>
    for ForEachSegment<Unique<Equal>>
where
    R: Runtime,
    Input: MIter<R, Item = Output::Item>,
    InputOffsets: MIter<R, Item = u32>,
    Output: MIterMut<R>,
    OutputOffsets: MIterMut<R, Item = u32>,
    Equal: BinaryPredicateOp<Input::Item>,
{
    fn run_into(
        self,
        exec: &Executor<R>,
        input: SegmentIterator<Input, InputOffsets>,
        output: SegmentedMut<Output, OutputOffsets>,
    ) -> Result<u32, Error> {
        let Unique(_equal) = self.0;
        let (values, offsets) = input.into_parts();
        let value_len = values.len()? as usize;
        let control = control::SegmentControl::new(exec, offsets, value_len)?;
        let values = crate::api::iter::lower_fixed::<R, _>(values);
        let flags = crate::ordering::unique_head_flags::<R, _, Equal>(exec, values.clone())?;
        control.merge_heads(exec, &flags)?;
        let (output_values, output_offsets) = output.into_parts();
        control.compact(exec, values, flags, output_values, output_offsets)
    }
}

struct PredicateMap<Pred>(PhantomData<fn() -> Pred>);

#[cubecl::cube]
impl<Item, Pred> UnaryOp<Item> for PredicateMap<Pred>
where
    Item: CubeType + 'static,
    Pred: PredicateOp<Item>,
{
    type Output = u32;

    fn apply(input: Item) -> u32 {
        if Pred::apply(input) { 1u32 } else { 0u32 }
    }
}

struct NegatedPredicateMap<Pred>(PhantomData<fn() -> Pred>);

#[cubecl::cube]
impl<Item, Pred> UnaryOp<Item> for NegatedPredicateMap<Pred>
where
    Item: CubeType + 'static,
    Pred: PredicateOp<Item>,
{
    type Output = u32;

    fn apply(input: Item) -> u32 {
        if Pred::apply(input) { 0u32 } else { 1u32 }
    }
}

struct Decrement;

#[cubecl::cube]
impl UnaryOp<u32> for Decrement {
    type Output = u32;

    fn apply(input: u32) -> u32 {
        input - 1u32
    }
}

struct InvertFlag;

#[cubecl::cube]
impl UnaryOp<u32> for InvertFlag {
    type Output = u32;

    fn apply(input: u32) -> u32 {
        if input == 0u32 { 1u32 } else { 0u32 }
    }
}

pub(crate) fn reduce_segments<R, Values, Offsets, Output, Op>(
    exec: &Executor<R>,
    input: SegmentIterator<Values, Offsets>,
    init: Values::Item,
    op: Op,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Values: MIter<R>,
    Values::Item: MAllocItem<R> + Copy,
    Offsets: MIter<R, Item = u32>,
    Output: MIterMut<R, Item = Values::Item>,
    Op: ReductionOp<Values::Item>,
{
    let (values, offsets) = input.into_parts();
    let value_len = values.len()? as usize;
    let control = control::SegmentControl::new(exec, offsets, value_len)?;
    let keys = exec.alloc::<u32>(control.segment_count);
    let reduced = exec.alloc::<Values::Item>(control.segment_count);
    let count = crate::vector::reduce_by_key_into(
        exec,
        control.ids.slice(..),
        values,
        control::EqualU32,
        init,
        op,
        keys.slice_mut(..),
        reduced.slice_mut(..),
    )?;

    let result = exec.alloc::<Values::Item>(control.segment_count);
    result.slice_mut(..).fill_with(exec, init)?;
    if count != 0 {
        let indices = exec.alloc::<u32>(count as usize);
        crate::api::algorithm::transform::transform_into(
            exec,
            keys.slice(..count as usize),
            Decrement,
            indices.slice_mut(..),
        )?;
        crate::vector::scatter_raw(
            exec,
            reduced.slice(..count as usize),
            indices.slice(..),
            result.slice_mut(..),
        )?;
    }
    crate::api::algorithm::transform::transform_into(
        exec,
        result.slice(..),
        crate::op::Identity,
        output,
    )
}

impl<R, Input, InputOffsets, Output, OutputOffsets, Pred>
    CompactingExecutableInto<R, Input, InputOffsets, Output, OutputOffsets>
    for ForEachSegment<Filter<Pred>>
where
    R: Runtime,
    Input: MIter<R, Item = Output::Item> + Clone,
    InputOffsets: MIter<R, Item = u32>,
    Output: MIterMut<R>,
    OutputOffsets: MIterMut<R, Item = u32>,
    Pred: PredicateOp<Input::Item>,
{
    fn run_into(
        self,
        exec: &Executor<R>,
        input: SegmentIterator<Input, InputOffsets>,
        output: SegmentedMut<Output, OutputOffsets>,
    ) -> Result<u32, Error> {
        let (values, offsets) = input.into_parts();
        let value_len = values.len()? as usize;
        let control = control::SegmentControl::new(exec, offsets, value_len)?;
        let flags = exec.alloc::<u32>(value_len);
        crate::api::algorithm::transform::transform_into(
            exec,
            values.clone(),
            PredicateMap::<Pred>(PhantomData),
            flags.slice_mut(..),
        )?;
        let (output_values, output_offsets) = output.into_parts();
        control.compact(
            exec,
            crate::api::iter::lower_fixed::<R, _>(values),
            flags,
            output_values,
            output_offsets,
        )
    }
}

impl<R, Input, InputOffsets, Output, Op, Init>
    SummarizingExecutableInto<R, Input, InputOffsets, Output> for ForEachSegment<Reduce<Op, Init>>
where
    R: Runtime,
    Input: MIter<R, Item = Init>,
    InputOffsets: MIter<R, Item = u32>,
    Output: MIterMut<R, Item = Init>,
    Init: MAllocItem<R> + Copy,
    Op: ReductionOp<Input::Item>,
{
    fn run_into(
        self,
        exec: &Executor<R>,
        input: SegmentIterator<Input, InputOffsets>,
        output: Output,
    ) -> Result<(), Error> {
        let Reduce(op, init) = self.0;
        reduce_segments(exec, input, init, op, output)
    }
}

macro_rules! impl_predicate_summary {
    ($algorithm:ident, $map:ident, $reducer:expr, $init:expr) => {
        impl<R, Input, InputOffsets, Output, Pred>
            SummarizingExecutableInto<R, Input, InputOffsets, Output>
            for ForEachSegment<$algorithm<Pred>>
        where
            R: Runtime,
            Input: MIter<R>,
            InputOffsets: MIter<R, Item = u32>,
            Output: MIterMut<R, Item = u32>,
            Pred: PredicateOp<Input::Item>,
        {
            fn run_into(
                self,
                exec: &Executor<R>,
                input: SegmentIterator<Input, InputOffsets>,
                output: Output,
            ) -> Result<(), Error> {
                let (values, offsets) = input.into_parts();
                let value_len = values.len()? as usize;
                let mapped = exec.alloc::<u32>(value_len);
                crate::api::algorithm::transform::transform_into(
                    exec,
                    values,
                    $map::<Pred>(PhantomData),
                    mapped.slice_mut(..),
                )?;
                reduce_segments(
                    exec,
                    SegmentIterator::new(mapped.slice(..), offsets),
                    $init,
                    $reducer,
                    output,
                )
            }
        }
    };
}

impl_predicate_summary!(CountIf, PredicateMap, control::SumU32, 0u32);
impl_predicate_summary!(AllOf, PredicateMap, control::MinU32, 1u32);
impl_predicate_summary!(AnyOf, PredicateMap, control::MaxU32, 0u32);
impl_predicate_summary!(NoneOf, NegatedPredicateMap, control::MinU32, 1u32);

impl<R, Input, InputOffsets, Output, Less> SummarizingExecutableInto<R, Input, InputOffsets, Output>
    for ForEachSegment<IsSorted<Less>>
where
    R: Runtime,
    Input: MIter<R>,
    InputOffsets: MIter<R, Item = u32>,
    Output: MIterMut<R, Item = u32>,
    Less: BinaryPredicateOp<Input::Item>,
{
    fn run_into(
        self,
        exec: &Executor<R>,
        input: SegmentIterator<Input, InputOffsets>,
        output: Output,
    ) -> Result<(), Error> {
        let IsSorted(_less) = self.0;
        let (values, offsets) = input.into_parts();
        let value_len = values.len()? as usize;
        let control = control::SegmentControl::new(exec, offsets, value_len)?;
        let breaks = crate::ordering::sorted_break_flags::<R, _, Less>(
            exec,
            crate::api::iter::lower_fixed::<R, _>(values),
        )?;
        control.clear_heads(exec, &breaks)?;
        let ordered = exec.alloc::<u32>(value_len);
        crate::api::algorithm::transform::transform_into(
            exec,
            breaks.slice(..),
            InvertFlag,
            ordered.slice_mut(..),
        )?;
        reduce_segments(
            exec,
            SegmentIterator::new(ordered.slice(..), control.offsets.slice(..)),
            1u32,
            control::MinU32,
            output,
        )
    }
}

impl<R, Input, InputOffsets, Output, Less> SummarizingExecutableInto<R, Input, InputOffsets, Output>
    for ForEachSegment<IsSortedUntil<Less>>
where
    R: Runtime,
    Input: MIter<R>,
    InputOffsets: MIter<R, Item = u32>,
    Output: MIterMut<R, Item = u32>,
    Less: BinaryPredicateOp<Input::Item>,
{
    fn run_into(
        self,
        exec: &Executor<R>,
        input: SegmentIterator<Input, InputOffsets>,
        output: Output,
    ) -> Result<(), Error> {
        let IsSortedUntil(_less) = self.0;
        let (values, offsets) = input.into_parts();
        let value_len = values.len()? as usize;
        let control = control::SegmentControl::new(exec, offsets, value_len)?;
        let breaks = crate::ordering::sorted_break_flags::<R, _, Less>(
            exec,
            crate::api::iter::lower_fixed::<R, _>(values),
        )?;
        let candidates = control.sorted_until_candidates(exec, &breaks)?;
        let reduced = exec.alloc::<u32>(control.segment_count);
        reduce_segments(
            exec,
            SegmentIterator::new(candidates.slice(..), control.offsets.slice(..)),
            u32::MAX,
            control::MinU32,
            reduced.slice_mut(..),
        )?;
        let result = control.finish_sorted_until(exec, &reduced)?;
        crate::api::algorithm::transform::transform_into(
            exec,
            result.slice(..),
            crate::op::Identity,
            output,
        )
    }
}

macro_rules! impl_owned_length_preserving_input {
    ($algorithm:ty, $( $bound:tt )+) => {
        impl<R, Input, InputOffsets, Op> Executable<R, Input, InputOffsets>
            for ForEachSegment<$algorithm>
        where
            R: Runtime,
            Input: MIter<R>,
            Input::Item: MAllocItem<R>,
            InputOffsets: MIter<R, Item = u32>,
            $( $bound )+
        {
            type Output = SegmentIterator<MVec<R, Input::Item>, InputOffsets>;

            fn run(
                self,
                exec: &Executor<R>,
                input: SegmentIterator<Input, InputOffsets>,
            ) -> Result<Self::Output, Error> {
                let (values, offsets) = input.into_parts();
                let output = exec.alloc::<Input::Item>(values.len()? as usize);
                LengthPreservingExecutableInto::run_into(
                    self,
                    exec,
                    SegmentIterator::new(values, offsets.clone()),
                    output.slice_mut(..),
                )?;
                Ok(SegmentIterator::new(output, offsets))
            }
        }
    };
}

impl_owned_length_preserving_input!(Sort<Op>, Op: BinaryPredicateOp<Input::Item>);
impl_owned_length_preserving_input!(InclusiveScan<Op>,
    Op: ReductionOp<Input::Item>
);

impl<R, Input, InputOffsets, Op, Init> Executable<R, Input, InputOffsets>
    for ForEachSegment<ExclusiveScan<Op, Init>>
where
    R: Runtime,
    Input: MIter<R, Item = Init>,
    InputOffsets: MIter<R, Item = u32>,
    Init: MAllocItem<R>,
    Op: ReductionOp<Init>,
{
    type Output = SegmentIterator<MVec<R, Input::Item>, InputOffsets>;

    fn run(
        self,
        exec: &Executor<R>,
        input: SegmentIterator<Input, InputOffsets>,
    ) -> Result<Self::Output, Error> {
        let (values, offsets) = input.into_parts();
        let output = exec.alloc::<Input::Item>(values.len()? as usize);
        LengthPreservingExecutableInto::run_into(
            self,
            exec,
            SegmentIterator::new(values, offsets.clone()),
            output.slice_mut(..),
        )?;
        Ok(SegmentIterator::new(output, offsets))
    }
}

impl<R, Input, InputOffsets> Executable<R, Input, InputOffsets> for ForEachSegment<Reverse>
where
    R: Runtime,
    Input: MIter<R>,
    Input::Item: MAllocItem<R>,
    InputOffsets: MIter<R, Item = u32>,
{
    type Output = SegmentIterator<MVec<R, Input::Item>, InputOffsets>;

    fn run(
        self,
        exec: &Executor<R>,
        input: SegmentIterator<Input, InputOffsets>,
    ) -> Result<Self::Output, Error> {
        let (values, offsets) = input.into_parts();
        let output = exec.alloc::<Input::Item>(values.len()? as usize);
        LengthPreservingExecutableInto::run_into(
            self,
            exec,
            SegmentIterator::new(values, offsets.clone()),
            output.slice_mut(..),
        )?;
        Ok(SegmentIterator::new(output, offsets))
    }
}

impl<R, Input, InputOffsets, Op> Executable<R, Input, InputOffsets>
    for ForEachSegment<AdjacentDifference<Op>>
where
    R: Runtime,
    Input: MIter<R> + Clone,
    Input::Item: MAllocItem<R>,
    InputOffsets: MIter<R, Item = u32>,
    Op: ReductionOp<Input::Item>,
{
    type Output = SegmentIterator<MVec<R, Input::Item>, InputOffsets>;

    fn run(
        self,
        exec: &Executor<R>,
        input: SegmentIterator<Input, InputOffsets>,
    ) -> Result<Self::Output, Error> {
        let AdjacentDifference(op) = self.0;
        let (values, offsets) = input.into_parts();
        let control = control::SegmentControl::new(exec, offsets.clone(), values.len()? as usize)?;
        let output = crate::vector::adjacent_difference(exec, values.clone(), op)?;
        crate::vector::transform_where_raw(
            exec,
            values,
            crate::op::Identity,
            control.heads.column(),
            output.slice_mut(..),
        )?;
        Ok(SegmentIterator::new(output, offsets))
    }
}

macro_rules! impl_owned_compacting {
    ($algorithm:ty, $( $bound:tt )+) => {
        impl<R, Input, InputOffsets, Op> Executable<R, Input, InputOffsets>
            for ForEachSegment<$algorithm>
        where
            R: Runtime,
            Input: MIter<R> + Clone,
            Input::Item: MAllocItem<R>,
            InputOffsets: MIter<R, Item = u32>,
            $( $bound )+
        {
            type Output = SegmentIterator<MVec<R, Input::Item>, MVec<R, u32>>;

            fn run(
                self,
                exec: &Executor<R>,
                input: SegmentIterator<Input, InputOffsets>,
            ) -> Result<Self::Output, Error> {
                let mut values = exec.alloc::<Input::Item>(input.values().len()? as usize);
                let offsets = exec.alloc::<u32>(input.offsets().len()? as usize);
                let written = CompactingExecutableInto::run_into(
                    self,
                    exec,
                    input,
                    SegmentedMut::new(values.slice_mut(..), offsets.slice_mut(..)),
                )?;
                values.truncate(written as usize);
                Ok(SegmentIterator::new(values, offsets))
            }
        }
    };
}

impl_owned_compacting!(Unique<Op>, Op: BinaryPredicateOp<Input::Item>);
impl_owned_compacting!(Filter<Op>, Op: PredicateOp<Input::Item>);

impl<R, Input, InputOffsets, Op, Init> Executable<R, Input, InputOffsets>
    for ForEachSegment<Reduce<Op, Init>>
where
    R: Runtime,
    Input: MIter<R, Item = Init>,
    InputOffsets: MIter<R, Item = u32>,
    Init: MAllocItem<R> + Copy,
    Op: ReductionOp<Init>,
{
    type Output = MVec<R, Input::Item>;

    fn run(
        self,
        exec: &Executor<R>,
        input: SegmentIterator<Input, InputOffsets>,
    ) -> Result<Self::Output, Error> {
        let offset_len = input.offsets().len()? as usize;
        let segment_count = offset_len
            .checked_sub(1)
            .ok_or(Error::LengthMismatch { left: 1, right: 0 })?;
        let output = exec.alloc::<Input::Item>(segment_count);
        SummarizingExecutableInto::run_into(self, exec, input, output.slice_mut(..))?;
        Ok(output)
    }
}

macro_rules! impl_owned_summary_index {
    ($algorithm:ty, $( $bound:tt )+) => {
        impl<R, Input, InputOffsets, Op> Executable<R, Input, InputOffsets>
            for ForEachSegment<$algorithm>
        where
            R: Runtime,
            Input: MIter<R>,
            InputOffsets: MIter<R, Item = u32>,
            $( $bound )+
        {
            type Output = MVec<R, u32>;

            fn run(
                self,
                exec: &Executor<R>,
                input: SegmentIterator<Input, InputOffsets>,
            ) -> Result<Self::Output, Error> {
                let offset_len = input.offsets().len()? as usize;
                let segment_count = offset_len
                    .checked_sub(1)
                    .ok_or(Error::LengthMismatch { left: 1, right: 0 })?;
                let output = exec.alloc::<u32>(segment_count);
                SummarizingExecutableInto::run_into(self, exec, input, output.slice_mut(..))?;
                Ok(output)
            }
        }
    };
}

impl_owned_summary_index!(CountIf<Op>, Op: PredicateOp<Input::Item>);
impl_owned_summary_index!(AllOf<Op>, Op: PredicateOp<Input::Item>);
impl_owned_summary_index!(AnyOf<Op>, Op: PredicateOp<Input::Item>);
impl_owned_summary_index!(NoneOf<Op>, Op: PredicateOp<Input::Item>);
impl_owned_summary_index!(IsSorted<Op>, Op: BinaryPredicateOp<Input::Item>);
impl_owned_summary_index!(IsSortedUntil<Op>, Op: BinaryPredicateOp<Input::Item>);
