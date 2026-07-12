//! Algorithms applied independently to offset-delimited segments.
//!
//! A [`Segmented`] value combines a flat [`MIter`] with offsets. Segment `i` is
//! the half-open range `offsets[i]..offsets[i + 1]` in the flat values.
//!
//! Segment algorithms return owned device storage. Length-preserving algorithms
//! return flat values, length-changing algorithms return values with rebuilt
//! offsets, and reductions return one value per segment.

pub(crate) mod control;

use cubecl::prelude::*;
use std::marker::PhantomData;

use crate::{
    Allocable, Error, Executor, MIndex, MIter, MIterMut, MStorage, MVec, Materializable,
    WritableFrom, op::BinaryPredicateOp, op::PredicateOp, op::ReductionOp, op::UnaryOp,
};

/// A flat value stream and the offsets delimiting its segments.
#[derive(Clone, Copy, Debug)]
pub struct Segmented<Values, Offsets> {
    values: Values,
    offsets: Offsets,
}

impl<Values, Offsets> Segmented<Values, Offsets> {
    /// Combines flat values and offsets into a segmented input.
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

/// Maps every item while preserving segment offsets.
#[derive(Clone, Copy, Debug)]
pub struct Map<Op>(pub Op);

/// Stably sorts the items within every segment.
#[derive(Clone, Copy, Debug)]
pub struct Sort<Less>(pub Less);

/// Reverses the items within every segment.
#[derive(Clone, Copy, Debug, Default)]
pub struct Reverse;

/// Computes an inclusive scan within every segment.
#[derive(Clone, Copy, Debug)]
pub struct InclusiveScan<Op>(pub Op);

/// Computes an exclusive scan within every segment, starting from `init`.
#[derive(Clone, Copy, Debug)]
pub struct ExclusiveScan<Op, Init>(pub Op, pub Init);

/// Computes adjacent differences without crossing segment boundaries.
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
pub(crate) trait MapLikeExecutableInto<R, Input, InputOffsets, Output>: Sized
where
    R: Runtime,
    Input: MIter<R>,
    InputOffsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
{
    /// Executes the algorithm for every segment.
    fn run_into(
        self,
        exec: &Executor<R>,
        input: Segmented<Input, InputOffsets>,
        output: Output,
    ) -> Result<(), Error>;
}

/// Execution contract for segment algorithms that can change segment lengths.
///
/// The caller provides capacity for both the flat values and the rebuilt
/// offsets. The returned index is the total number of flat values written.
#[doc(hidden)]
pub(crate) trait UniqueLikeExecutableInto<R, Input, InputOffsets, Output, OutputOffsets>:
    Sized
where
    R: Runtime,
    Input: MIter<R>,
    InputOffsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
    OutputOffsets: MIterMut<R, Item = MIndex>,
{
    /// Executes the algorithm for every segment.
    fn run_into(
        self,
        exec: &Executor<R>,
        input: Segmented<Input, InputOffsets>,
        output: SegmentedMut<Output, OutputOffsets>,
    ) -> Result<MIndex, Error>;
}

/// Execution contract for segment algorithms that write one item per segment.
#[doc(hidden)]
pub(crate) trait ReduceLikeExecutableInto<R, Input, InputOffsets, Output>: Sized
where
    R: Runtime,
    Input: MIter<R>,
    InputOffsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
{
    /// Executes the algorithm for every segment.
    fn run_into(
        self,
        exec: &Executor<R>,
        input: Segmented<Input, InputOffsets>,
        output: Output,
    ) -> Result<(), Error>;
}

/// Owned-result execution contract for segment algorithms that preserve flat length.
pub trait MapLikeExecutable<R, Input, InputOffsets>: Sized
where
    R: Runtime,
    Input: MIter<R>,
    InputOffsets: MIter<R, Item = MIndex>,
{
    type OutputItem: Materializable<R>;

    fn run(
        self,
        exec: &Executor<R>,
        input: Segmented<Input, InputOffsets>,
    ) -> Result<MVec<R, Self::OutputItem>, Error>;
}

/// Owned-result execution contract for length-changing segment algorithms.
pub trait UniqueLikeExecutable<R, Input, InputOffsets>: Sized
where
    R: Runtime,
    Input: MIter<R>,
    InputOffsets: MIter<R, Item = MIndex>,
{
    type OutputItem: Materializable<R>;

    fn run(
        self,
        exec: &Executor<R>,
        input: Segmented<Input, InputOffsets>,
    ) -> Result<Segmented<MVec<R, Self::OutputItem>, MVec<R, MIndex>>, Error>;
}

/// Owned-result execution contract for algorithms that return one item per segment.
pub trait ReduceLikeExecutable<R, Input, InputOffsets>: Sized
where
    R: Runtime,
    Input: MIter<R>,
    InputOffsets: MIter<R, Item = MIndex>,
{
    type OutputItem: Materializable<R>;

    fn run(
        self,
        exec: &Executor<R>,
        input: Segmented<Input, InputOffsets>,
    ) -> Result<MVec<R, Self::OutputItem>, Error>;
}

impl<R, Input, InputOffsets, Op> MapLikeExecutable<R, Input, InputOffsets>
    for ForEachSegment<Map<Op>>
where
    R: Runtime,
    Input: MIter<R>,
    InputOffsets: MIter<R, Item = MIndex>,
    Op: UnaryOp<Input::Item>,
    Op::Output: Materializable<R>,
{
    type OutputItem = Op::Output;

    fn run(
        self,
        exec: &Executor<R>,
        input: Segmented<Input, InputOffsets>,
    ) -> Result<MVec<R, Self::OutputItem>, Error> {
        let output = exec.alloc_mvec::<Self::OutputItem>(input.values().len()? as usize);
        MapLikeExecutableInto::run_into(self, exec, input, output.slice_mut(..))?;
        Ok(output)
    }
}

impl<R, Input, InputOffsets, Output, Op> MapLikeExecutableInto<R, Input, InputOffsets, Output>
    for ForEachSegment<Map<Op>>
where
    R: Runtime,
    Input: MIter<R>,
    InputOffsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
    Op: UnaryOp<Input::Item>,
    Output::Item: WritableFrom<<Op as UnaryOp<Input::Item>>::Output>,
{
    fn run_into(
        self,
        exec: &Executor<R>,
        input: Segmented<Input, InputOffsets>,
        output: Output,
    ) -> Result<(), Error> {
        let Map(op) = self.0;
        let (values, _) = input.into_parts();
        crate::api::algorithm::transform::transform_into(exec, values, op, output)
    }
}

impl<R, Input, InputOffsets, Output, Less> MapLikeExecutableInto<R, Input, InputOffsets, Output>
    for ForEachSegment<Sort<Less>>
where
    R: Runtime,
    Input: MIter<R>,
    InputOffsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
    Less: BinaryPredicateOp<Input::Item>,
    Output::Item: WritableFrom<Input::Item>,
{
    fn run_into(
        self,
        exec: &Executor<R>,
        input: Segmented<Input, InputOffsets>,
        output: Output,
    ) -> Result<(), Error> {
        let Sort(less) = self.0;
        let (values, offsets) = input.into_parts();
        let value_len = values.len()? as usize;
        let control = control::SegmentControl::new(exec, offsets, value_len)?;

        let values = crate::api::iter::lower::<R, _>(values);
        let ordering = crate::ordering::sort_control_with(exec, values.clone(), less)?;
        let sorted_ids = exec.alloc::<u32>(value_len);
        crate::indexed::apply_permutation(
            exec,
            crate::api::iter::lower::<R, _>(control.ids.slice(..)),
            ordering.column(),
            <crate::DeviceSliceMut<u32> as MIterMut<R>>::lower_output(sorted_ids.slice_mut(..)),
        )?;

        let id_ordering = crate::ordering::sort_control_with(
            exec,
            crate::api::iter::lower::<R, _>(sorted_ids.slice(..)),
            control::LessU32,
        )?;
        let permutation = exec.alloc::<u32>(value_len);
        crate::indexed::apply_permutation(
            exec,
            crate::api::iter::lower::<R, _>(ordering.slice(..)),
            id_ordering.column(),
            <crate::DeviceSliceMut<u32> as MIterMut<R>>::lower_output(permutation.slice_mut(..)),
        )?;
        crate::indexed::apply_permutation(
            exec,
            values,
            permutation.column(),
            output.lower_output_from::<Input::Item>(),
        )
    }
}

impl<R, Input, InputOffsets, Output, Op> MapLikeExecutableInto<R, Input, InputOffsets, Output>
    for ForEachSegment<AdjacentDifference<Op>>
where
    R: Runtime,
    Input: MIter<R> + Clone,
    InputOffsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
    Output::Item: WritableFrom<Input::Item>,
    Output::SliceMut: MIterMut<R>,
    <Output::SliceMut as MIterMut<R>>::Item: WritableFrom<Input::Item>,
    Op: ReductionOp<Input::Item>,
{
    fn run_into(
        self,
        exec: &Executor<R>,
        input: Segmented<Input, InputOffsets>,
        output: Output,
    ) -> Result<(), Error> {
        let AdjacentDifference(op) = self.0;
        let (values, offsets) = input.into_parts();
        let control = control::SegmentControl::new(exec, offsets, values.len()? as usize)?;
        crate::vector::adjacent_difference_into(exec, values.clone(), op, output.slice_mut(..))?;
        crate::vector::transform_where(
            exec,
            values,
            crate::op::Identity,
            control.heads.slice(..),
            output,
        )
    }
}

impl<R, Input, InputOffsets, Output, Op> MapLikeExecutableInto<R, Input, InputOffsets, Output>
    for ForEachSegment<InclusiveScan<Op>>
where
    R: Runtime,
    Input: MIter<R>,
    InputOffsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
    Op: ReductionOp<Input::Item>,
    Output::Item: WritableFrom<Input::Item>,
{
    fn run_into(
        self,
        exec: &Executor<R>,
        input: Segmented<Input, InputOffsets>,
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

impl<R, Input, InputOffsets, Output, Op, Init> MapLikeExecutableInto<R, Input, InputOffsets, Output>
    for ForEachSegment<ExclusiveScan<Op, Init>>
where
    R: Runtime,
    Input: MIter<R, Item = Init>,
    InputOffsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
    Init: Materializable<R>,
    Op: ReductionOp<Init>,
    Output::Item: WritableFrom<Input::Item>,
{
    fn run_into(
        self,
        exec: &Executor<R>,
        input: Segmented<Input, InputOffsets>,
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

impl<R, Input, InputOffsets, Output> MapLikeExecutableInto<R, Input, InputOffsets, Output>
    for ForEachSegment<Reverse>
where
    R: Runtime,
    Input: MIter<R>,
    InputOffsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
    Output::Item: WritableFrom<Input::Item>,
{
    fn run_into(
        self,
        exec: &Executor<R>,
        input: Segmented<Input, InputOffsets>,
        output: Output,
    ) -> Result<(), Error> {
        let (values, offsets) = input.into_parts();
        let control = control::SegmentControl::new(exec, offsets, values.len()? as usize)?;
        let indices = control.reverse_indices(exec)?;
        crate::vector::gather_into(exec, values, indices.slice(..), output)
    }
}

impl<R, Input, InputOffsets, Output, OutputOffsets, Equal>
    UniqueLikeExecutableInto<R, Input, InputOffsets, Output, OutputOffsets>
    for ForEachSegment<Unique<Equal>>
where
    R: Runtime,
    Input: MIter<R>,
    InputOffsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
    OutputOffsets: MIterMut<R, Item = MIndex>,
    Equal: BinaryPredicateOp<Input::Item>,
    Output::Item: WritableFrom<Input::Item>,
{
    fn run_into(
        self,
        exec: &Executor<R>,
        input: Segmented<Input, InputOffsets>,
        output: SegmentedMut<Output, OutputOffsets>,
    ) -> Result<MIndex, Error> {
        let Unique(_equal) = self.0;
        let (values, offsets) = input.into_parts();
        let value_len = values.len()? as usize;
        let control = control::SegmentControl::new(exec, offsets, value_len)?;
        let values = crate::api::iter::lower::<R, _>(values);
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
    input: Segmented<Values, Offsets>,
    init: Values::Item,
    op: Op,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Values: MIter<R>,
    Values::Item: Allocable<R> + Copy,
    Offsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
    Output::Item: WritableFrom<Values::Item>,
    Op: ReductionOp<Values::Item>,
{
    let (values, offsets) = input.into_parts();
    let value_len = values.len()? as usize;
    let control = control::SegmentControl::new(exec, offsets, value_len)?;
    let keys = exec.alloc::<u32>(control.segment_count);
    let reduced = <Values::Item as Allocable<R>>::alloc(exec, control.segment_count);
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

    let result = <Values::Item as Allocable<R>>::alloc(exec, control.segment_count);
    result.slice_mut(..).fill_with(exec, init)?;
    if count != 0 {
        let indices = exec.alloc::<u32>(count as usize);
        crate::api::algorithm::transform::transform_into(
            exec,
            keys.slice(..count as usize),
            Decrement,
            indices.slice_mut(..),
        )?;
        crate::vector::scatter(
            exec,
            reduced.slice(..count),
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
    UniqueLikeExecutableInto<R, Input, InputOffsets, Output, OutputOffsets>
    for ForEachSegment<Filter<Pred>>
where
    R: Runtime,
    Input: MIter<R> + Clone,
    InputOffsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
    OutputOffsets: MIterMut<R, Item = MIndex>,
    Pred: PredicateOp<Input::Item>,
    Output::Item: WritableFrom<Input::Item>,
{
    fn run_into(
        self,
        exec: &Executor<R>,
        input: Segmented<Input, InputOffsets>,
        output: SegmentedMut<Output, OutputOffsets>,
    ) -> Result<MIndex, Error> {
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
            crate::api::iter::lower::<R, _>(values),
            flags,
            output_values,
            output_offsets,
        )
    }
}

impl<R, Input, InputOffsets, Output, Op, Init>
    ReduceLikeExecutableInto<R, Input, InputOffsets, Output> for ForEachSegment<Reduce<Op, Init>>
where
    R: Runtime,
    Input: MIter<R, Item = Init>,
    InputOffsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
    Init: CubeType + Allocable<R> + Copy,
    Op: ReductionOp<Input::Item>,
    Output::Item: WritableFrom<Input::Item>,
{
    fn run_into(
        self,
        exec: &Executor<R>,
        input: Segmented<Input, InputOffsets>,
        output: Output,
    ) -> Result<(), Error> {
        let Reduce(op, init) = self.0;
        reduce_segments(exec, input, init, op, output)
    }
}

macro_rules! impl_predicate_reduce_like {
    ($algorithm:ident, $map:ident, $reducer:expr, $init:expr) => {
        impl<R, Input, InputOffsets, Output, Pred>
            ReduceLikeExecutableInto<R, Input, InputOffsets, Output>
            for ForEachSegment<$algorithm<Pred>>
        where
            R: Runtime,
            Input: MIter<R>,
            InputOffsets: MIter<R, Item = MIndex>,
            Output: MIterMut<R>,
            Pred: PredicateOp<Input::Item>,
            Output::Item: WritableFrom<MIndex>,
        {
            fn run_into(
                self,
                exec: &Executor<R>,
                input: Segmented<Input, InputOffsets>,
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
                    Segmented::new(mapped.slice(..), offsets),
                    $init,
                    $reducer,
                    output,
                )
            }
        }
    };
}

impl_predicate_reduce_like!(CountIf, PredicateMap, control::SumU32, 0u32);
impl_predicate_reduce_like!(AllOf, PredicateMap, control::MinU32, 1u32);
impl_predicate_reduce_like!(AnyOf, PredicateMap, control::MaxU32, 0u32);
impl_predicate_reduce_like!(NoneOf, NegatedPredicateMap, control::MinU32, 1u32);

impl<R, Input, InputOffsets, Output, Less> ReduceLikeExecutableInto<R, Input, InputOffsets, Output>
    for ForEachSegment<IsSorted<Less>>
where
    R: Runtime,
    Input: MIter<R>,
    InputOffsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
    Less: BinaryPredicateOp<Input::Item>,
    Output::Item: WritableFrom<MIndex>,
{
    fn run_into(
        self,
        exec: &Executor<R>,
        input: Segmented<Input, InputOffsets>,
        output: Output,
    ) -> Result<(), Error> {
        let IsSorted(_less) = self.0;
        let (values, offsets) = input.into_parts();
        let value_len = values.len()? as usize;
        let control = control::SegmentControl::new(exec, offsets, value_len)?;
        let breaks = crate::ordering::sorted_break_flags::<R, _, Less>(
            exec,
            crate::api::iter::lower::<R, _>(values),
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
            Segmented::new(ordered.slice(..), control.offsets.slice(..)),
            1u32,
            control::MinU32,
            output,
        )
    }
}

impl<R, Input, InputOffsets, Output, Less> ReduceLikeExecutableInto<R, Input, InputOffsets, Output>
    for ForEachSegment<IsSortedUntil<Less>>
where
    R: Runtime,
    Input: MIter<R>,
    InputOffsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
    Less: BinaryPredicateOp<Input::Item>,
    Output::Item: WritableFrom<MIndex>,
{
    fn run_into(
        self,
        exec: &Executor<R>,
        input: Segmented<Input, InputOffsets>,
        output: Output,
    ) -> Result<(), Error> {
        let IsSortedUntil(_less) = self.0;
        let (values, offsets) = input.into_parts();
        let value_len = values.len()? as usize;
        let control = control::SegmentControl::new(exec, offsets, value_len)?;
        let breaks = crate::ordering::sorted_break_flags::<R, _, Less>(
            exec,
            crate::api::iter::lower::<R, _>(values),
        )?;
        let candidates = control.sorted_until_candidates(exec, &breaks)?;
        let reduced = exec.alloc::<u32>(control.segment_count);
        reduce_segments(
            exec,
            Segmented::new(candidates.slice(..), control.offsets.slice(..)),
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

macro_rules! impl_owned_map_input {
    ($algorithm:ty, $( $bound:tt )+) => {
        impl<R, Input, InputOffsets, Op> MapLikeExecutable<R, Input, InputOffsets>
            for ForEachSegment<$algorithm>
        where
            R: Runtime,
            Input: MIter<R>,
            Input::Item: Materializable<R>,
            InputOffsets: MIter<R, Item = MIndex>,
            $( $bound )+
        {
            type OutputItem = Input::Item;

            fn run(
                self,
                exec: &Executor<R>,
                input: Segmented<Input, InputOffsets>,
            ) -> Result<MVec<R, Self::OutputItem>, Error> {
                let output = exec.alloc_mvec::<Self::OutputItem>(input.values().len()? as usize);
                MapLikeExecutableInto::run_into(self, exec, input, output.slice_mut(..))?;
                Ok(output)
            }
        }
    };
}

impl_owned_map_input!(Sort<Op>, Op: BinaryPredicateOp<Input::Item>);
impl_owned_map_input!(InclusiveScan<Op>, Op: ReductionOp<Input::Item>);

impl<R, Input, InputOffsets, Op, Init> MapLikeExecutable<R, Input, InputOffsets>
    for ForEachSegment<ExclusiveScan<Op, Init>>
where
    R: Runtime,
    Input: MIter<R, Item = Init>,
    InputOffsets: MIter<R, Item = MIndex>,
    Init: Materializable<R>,
    Op: ReductionOp<Init>,
{
    type OutputItem = Input::Item;

    fn run(
        self,
        exec: &Executor<R>,
        input: Segmented<Input, InputOffsets>,
    ) -> Result<MVec<R, Self::OutputItem>, Error> {
        let output = exec.alloc_mvec::<Self::OutputItem>(input.values().len()? as usize);
        MapLikeExecutableInto::run_into(self, exec, input, output.slice_mut(..))?;
        Ok(output)
    }
}

impl<R, Input, InputOffsets> MapLikeExecutable<R, Input, InputOffsets> for ForEachSegment<Reverse>
where
    R: Runtime,
    Input: MIter<R>,
    Input::Item: Materializable<R>,
    InputOffsets: MIter<R, Item = MIndex>,
{
    type OutputItem = Input::Item;

    fn run(
        self,
        exec: &Executor<R>,
        input: Segmented<Input, InputOffsets>,
    ) -> Result<MVec<R, Self::OutputItem>, Error> {
        let output = exec.alloc_mvec::<Self::OutputItem>(input.values().len()? as usize);
        MapLikeExecutableInto::run_into(self, exec, input, output.slice_mut(..))?;
        Ok(output)
    }
}

impl<R, Input, InputOffsets, Op> MapLikeExecutable<R, Input, InputOffsets>
    for ForEachSegment<AdjacentDifference<Op>>
where
    R: Runtime,
    Input: MIter<R> + Clone,
    Input::Item: Materializable<R>,
    InputOffsets: MIter<R, Item = MIndex>,
    Op: ReductionOp<Input::Item>,
{
    type OutputItem = Input::Item;

    fn run(
        self,
        exec: &Executor<R>,
        input: Segmented<Input, InputOffsets>,
    ) -> Result<MVec<R, Self::OutputItem>, Error> {
        let AdjacentDifference(op) = self.0;
        let (values, offsets) = input.into_parts();
        let control = control::SegmentControl::new(exec, offsets, values.len()? as usize)?;
        let output = crate::vector::adjacent_difference(exec, values.clone(), op)?;
        crate::vector::transform_where(
            exec,
            values,
            crate::op::Identity,
            control.heads.column(),
            output.slice_mut(..),
        )?;
        Ok(output)
    }
}

macro_rules! impl_owned_unique_like {
    ($algorithm:ty, $( $bound:tt )+) => {
        impl<R, Input, InputOffsets, Op> UniqueLikeExecutable<R, Input, InputOffsets>
            for ForEachSegment<$algorithm>
        where
            R: Runtime,
            Input: MIter<R> + Clone,
            Input::Item: Materializable<R>,
            InputOffsets: MIter<R, Item = MIndex>,
            $( $bound )+
        {
            type OutputItem = Input::Item;

            fn run(
                self,
                exec: &Executor<R>,
                input: Segmented<Input, InputOffsets>,
            ) -> Result<Segmented<MVec<R, Self::OutputItem>, MVec<R, MIndex>>, Error> {
                let mut values =
                    exec.alloc_mvec::<Self::OutputItem>(input.values().len()? as usize);
                let offsets = exec.alloc_mvec::<MIndex>(input.offsets().len()? as usize);
                let written = UniqueLikeExecutableInto::run_into(
                    self,
                    exec,
                    input,
                    SegmentedMut::new(values.slice_mut(..), offsets.slice_mut(..)),
                )?;
                values.truncate(written);
                Ok(Segmented::new(values, offsets))
            }
        }
    };
}

impl_owned_unique_like!(Unique<Op>, Op: BinaryPredicateOp<Input::Item>);
impl_owned_unique_like!(Filter<Op>, Op: PredicateOp<Input::Item>);

impl<R, Input, InputOffsets, Op, Init> ReduceLikeExecutable<R, Input, InputOffsets>
    for ForEachSegment<Reduce<Op, Init>>
where
    R: Runtime,
    Input: MIter<R, Item = Init>,
    InputOffsets: MIter<R, Item = MIndex>,
    Init: Allocable<R> + Materializable<R> + Copy,
    Op: ReductionOp<Init>,
{
    type OutputItem = Input::Item;

    fn run(
        self,
        exec: &Executor<R>,
        input: Segmented<Input, InputOffsets>,
    ) -> Result<MVec<R, Self::OutputItem>, Error> {
        let offset_len = input.offsets().len()? as usize;
        let segment_count = offset_len
            .checked_sub(1)
            .ok_or(Error::LengthMismatch { left: 1, right: 0 })?;
        let output = exec.alloc_mvec::<Self::OutputItem>(segment_count);
        ReduceLikeExecutableInto::run_into(self, exec, input, output.slice_mut(..))?;
        Ok(output)
    }
}

macro_rules! impl_owned_reduce_index {
    ($algorithm:ty, $( $bound:tt )+) => {
        impl<R, Input, InputOffsets, Op> ReduceLikeExecutable<R, Input, InputOffsets>
            for ForEachSegment<$algorithm>
        where
            R: Runtime,
            Input: MIter<R>,
            InputOffsets: MIter<R, Item = MIndex>,
            $( $bound )+
        {
            type OutputItem = MIndex;

            fn run(
                self,
                exec: &Executor<R>,
                input: Segmented<Input, InputOffsets>,
            ) -> Result<MVec<R, Self::OutputItem>, Error> {
                let offset_len = input.offsets().len()? as usize;
                let segment_count = offset_len
                    .checked_sub(1)
                    .ok_or(Error::LengthMismatch { left: 1, right: 0 })?;
                let output = exec.alloc_mvec::<MIndex>(segment_count);
                ReduceLikeExecutableInto::run_into(self, exec, input, output.slice_mut(..))?;
                Ok(output)
            }
        }
    };
}

impl_owned_reduce_index!(CountIf<Op>, Op: PredicateOp<Input::Item>);
impl_owned_reduce_index!(AllOf<Op>, Op: PredicateOp<Input::Item>);
impl_owned_reduce_index!(AnyOf<Op>, Op: PredicateOp<Input::Item>);
impl_owned_reduce_index!(NoneOf<Op>, Op: PredicateOp<Input::Item>);
impl_owned_reduce_index!(IsSorted<Op>, Op: BinaryPredicateOp<Input::Item>);
impl_owned_reduce_index!(IsSortedUntil<Op>, Op: BinaryPredicateOp<Input::Item>);
