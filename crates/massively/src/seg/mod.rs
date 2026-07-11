//! Algorithms applied independently to offset-delimited segments.
//!
//! A [`Segmented`] value combines a flat [`MIter`] with offsets. Segment `i` is
//! the half-open range `offsets[i]..offsets[i + 1]` in the flat values.
//!
//! Segment algorithms have three output contracts:
//!
//! - [`MapLikeExecutable`] preserves the flat length and reuses the input offsets.
//! - [`UniqueLikeExecutable`] may change each segment's length and writes new offsets.
//! - [`ReduceLikeExecutable`] writes one item per segment.

pub(crate) mod control;

use cubecl::prelude::*;
use std::marker::PhantomData;

use crate::{
    Error, Executor, MAlloc, MIndex, MItem, MIter, MIterMut, MStorage, WriteFrom,
    op::BinaryPredicateOp, op::PredicateOp, op::ReductionOp, op::UnaryOp,
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
pub struct SegmentedMut<Values, Offsets> {
    values: Values,
    offsets: Offsets,
}

impl<Values, Offsets> SegmentedMut<Values, Offsets> {
    /// Combines preallocated flat value and offset outputs.
    pub const fn new(values: Values, offsets: Offsets) -> Self {
        Self { values, offsets }
    }

    /// Returns the flat value output.
    pub const fn values(&self) -> &Values {
        &self.values
    }

    /// Returns the offset output.
    pub const fn offsets(&self) -> &Offsets {
        &self.offsets
    }

    /// Decomposes this output into its flat values and offsets.
    pub fn into_parts(self) -> (Values, Offsets) {
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
pub trait MapLikeExecutable<R, Input, InputOffsets, Output>: Sized
where
    R: Runtime,
    Input: MIter<R>,
    InputOffsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
{
    /// Executes the algorithm for every segment.
    fn run(
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
pub trait UniqueLikeExecutable<R, Input, InputOffsets, Output, OutputOffsets>: Sized
where
    R: Runtime,
    Input: MIter<R>,
    InputOffsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
    OutputOffsets: MIterMut<R, Item = MIndex>,
{
    /// Executes the algorithm for every segment.
    fn run(
        self,
        exec: &Executor<R>,
        input: Segmented<Input, InputOffsets>,
        output: SegmentedMut<Output, OutputOffsets>,
    ) -> Result<MIndex, Error>;
}

/// Execution contract for segment algorithms that write one item per segment.
pub trait ReduceLikeExecutable<R, Input, InputOffsets, Output>: Sized
where
    R: Runtime,
    Input: MIter<R>,
    InputOffsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
{
    /// Executes the algorithm for every segment.
    fn run(
        self,
        exec: &Executor<R>,
        input: Segmented<Input, InputOffsets>,
        output: Output,
    ) -> Result<(), Error>;
}

impl<R, Input, InputOffsets, Output, Op> MapLikeExecutable<R, Input, InputOffsets, Output>
    for ForEachSegment<Map<Op>>
where
    R: Runtime,
    Input: MIter<R>,
    InputOffsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
    Op: UnaryOp<Input::Item>,
    Output::Item: WriteFrom<<Op as UnaryOp<Input::Item>>::Output>,
{
    fn run(
        self,
        exec: &Executor<R>,
        input: Segmented<Input, InputOffsets>,
        output: Output,
    ) -> Result<(), Error> {
        let Map(op) = self.0;
        let (values, _) = input.into_parts();
        values.transform_into(exec, op, output)
    }
}

impl<R, Input, InputOffsets, Output, Less> MapLikeExecutable<R, Input, InputOffsets, Output>
    for ForEachSegment<Sort<Less>>
where
    R: Runtime,
    Input: MIter<R>,
    InputOffsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
    Less: BinaryPredicateOp<Input::Item>,
    Output::Item: WriteFrom<Input::Item>,
{
    fn run(
        self,
        exec: &Executor<R>,
        input: Segmented<Input, InputOffsets>,
        output: Output,
    ) -> Result<(), Error> {
        let Sort(less) = self.0;
        let (values, offsets) = input.into_parts();
        let value_len = values.len()? as usize;
        let control = control::SegmentControl::new(exec, offsets, value_len)?;

        let values =
            <Input::Read as crate::core::facade::AnyRead<R>>::normalize(values.lower_read(), exec)?;
        let ordering = <<Input::Item as MItem<R>>::Kernel as crate::core::facade::KernelItem<
            R,
            Input::Item,
        >>::sort_ordering(exec, values, less)?;
        let sorted_ids = exec.alloc::<u32>(value_len);
        control.ids.slice(..).indexed_with(
            exec,
            ordering.control.permutation().column(),
            None,
            false,
            sorted_ids.slice_mut(..),
        )?;

        let sorted_ids = <_ as crate::core::facade::AnyRead<R>>::normalize(
            <crate::DeviceSlice<u32> as MIter<R>>::lower_read(sorted_ids.slice(..)),
            exec,
        )?;
        let id_ordering =
            <<u32 as MItem<R>>::Kernel as crate::core::facade::KernelItem<R, u32>>::sort_ordering(
                exec,
                sorted_ids,
                control::LessU32,
            )?;

        crate::core::facade::KernelWrite::gather_storage(
            output.lower_write_from::<Input::Item>(),
            exec,
            &ordering.sorted_keys,
            id_ordering.control.permutation().column(),
        )
    }
}

impl<R, Input, InputOffsets, Output, Op> MapLikeExecutable<R, Input, InputOffsets, Output>
    for ForEachSegment<AdjacentDifference<Op>>
where
    R: Runtime,
    Input: MIter<R> + Clone,
    InputOffsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
    Output::Item: WriteFrom<Input::Item>,
    Output::SliceMut: MIterMut<R>,
    <Output::SliceMut as MIterMut<R>>::Item: WriteFrom<Input::Item>,
    Op: ReductionOp<Input::Item>,
{
    fn run(
        self,
        exec: &Executor<R>,
        input: Segmented<Input, InputOffsets>,
        output: Output,
    ) -> Result<(), Error> {
        let AdjacentDifference(op) = self.0;
        let (values, offsets) = input.into_parts();
        let control = control::SegmentControl::new(exec, offsets, values.len()? as usize)?;
        values
            .clone()
            .adjacent_difference_with(exec, op, output.slice_mut(..))?;
        values.transform_where_with(
            exec,
            crate::op::Identity,
            control.heads.column(),
            output.slice_mut(..),
        )
    }
}

impl<R, Input, InputOffsets, Output, Op> MapLikeExecutable<R, Input, InputOffsets, Output>
    for ForEachSegment<InclusiveScan<Op>>
where
    R: Runtime,
    Input: MIter<R>,
    InputOffsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
    Op: ReductionOp<Input::Item>,
    Output::Item: WriteFrom<Input::Item>,
{
    fn run(
        self,
        exec: &Executor<R>,
        input: Segmented<Input, InputOffsets>,
        output: Output,
    ) -> Result<(), Error> {
        let InclusiveScan(op) = self.0;
        let (values, offsets) = input.into_parts();
        let control = control::SegmentControl::new(exec, offsets, values.len()? as usize)?;
        control
            .ids
            .slice(..)
            .scan_by_key_with(exec, values, control::EqualU32, None, op, output)
    }
}

impl<R, Input, InputOffsets, Output, Op, Init> MapLikeExecutable<R, Input, InputOffsets, Output>
    for ForEachSegment<ExclusiveScan<Op, Init>>
where
    R: Runtime,
    Input: MIter<R, Item = Init>,
    InputOffsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
    Init: CubeType,
    Op: ReductionOp<Input::Item>,
    Output::Item: WriteFrom<Input::Item>,
{
    fn run(
        self,
        exec: &Executor<R>,
        input: Segmented<Input, InputOffsets>,
        output: Output,
    ) -> Result<(), Error> {
        let ExclusiveScan(op, init) = self.0;
        let (values, offsets) = input.into_parts();
        let control = control::SegmentControl::new(exec, offsets, values.len()? as usize)?;
        control.ids.slice(..).scan_by_key_with(
            exec,
            values,
            control::EqualU32,
            Some(init),
            op,
            output,
        )
    }
}

impl<R, Input, InputOffsets, Output> MapLikeExecutable<R, Input, InputOffsets, Output>
    for ForEachSegment<Reverse>
where
    R: Runtime,
    Input: MIter<R>,
    InputOffsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
    Output::Item: WriteFrom<Input::Item>,
{
    fn run(
        self,
        exec: &Executor<R>,
        input: Segmented<Input, InputOffsets>,
        output: Output,
    ) -> Result<(), Error> {
        let (values, offsets) = input.into_parts();
        let control = control::SegmentControl::new(exec, offsets, values.len()? as usize)?;
        let indices = control.reverse_indices(exec)?;
        values.indexed_with(exec, indices.column(), None, false, output)
    }
}

impl<R, Input, InputOffsets, Output, OutputOffsets, Equal>
    UniqueLikeExecutable<R, Input, InputOffsets, Output, OutputOffsets>
    for ForEachSegment<Unique<Equal>>
where
    R: Runtime,
    Input: MIter<R>,
    InputOffsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
    OutputOffsets: MIterMut<R, Item = MIndex>,
    Equal: BinaryPredicateOp<Input::Item>,
    Output::Item: WriteFrom<Input::Item>,
{
    fn run(
        self,
        exec: &Executor<R>,
        input: Segmented<Input, InputOffsets>,
        output: SegmentedMut<Output, OutputOffsets>,
    ) -> Result<MIndex, Error> {
        let Unique(equal) = self.0;
        let (values, offsets) = input.into_parts();
        let value_len = values.len()? as usize;
        let control = control::SegmentControl::new(exec, offsets, value_len)?;
        let storage =
            <Input::Read as crate::core::facade::AnyRead<R>>::normalize(values.lower_read(), exec)?;
        let flags = <<Input::Item as MItem<R>>::Kernel as crate::core::facade::KernelItem<
            R,
            Input::Item,
        >>::segment_heads(exec, &storage, equal)?;
        control.merge_heads(exec, &flags)?;
        let (output_values, output_offsets) = output.into_parts();
        control.compact::<Input::Item, _, _>(exec, &storage, flags, output_values, output_offsets)
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
    Values::Item: MAlloc<R> + Copy,
    Offsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
    Output::Item: WriteFrom<Values::Item>,
    Op: ReductionOp<Values::Item>,
{
    let (values, offsets) = input.into_parts();
    let value_len = values.len()? as usize;
    let control = control::SegmentControl::new(exec, offsets, value_len)?;
    let keys = exec.alloc::<u32>(control.segment_count);
    let reduced = <Values::Item as MAlloc<R>>::alloc(exec, control.segment_count);
    let count = control.ids.slice(..).reduce_by_key_with(
        exec,
        values,
        control::EqualU32,
        init,
        op,
        keys.slice_mut(..),
        reduced.slice_mut(..),
    )?;

    let result = <Values::Item as MAlloc<R>>::alloc(exec, control.segment_count);
    result.slice_mut(..).fill_with(exec, init)?;
    if count != 0 {
        let indices = exec.alloc::<u32>(count as usize);
        keys.slice(..count as usize)
            .transform_into(exec, Decrement, indices.slice_mut(..))?;
        reduced.slice(..count).indexed_with(
            exec,
            indices.column(),
            None,
            true,
            result.slice_mut(..),
        )?;
    }
    result
        .slice(..)
        .transform_into(exec, crate::op::Identity, output)
}

impl<R, Input, InputOffsets, Output, OutputOffsets, Pred>
    UniqueLikeExecutable<R, Input, InputOffsets, Output, OutputOffsets>
    for ForEachSegment<Filter<Pred>>
where
    R: Runtime,
    Input: MIter<R> + Clone,
    InputOffsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
    OutputOffsets: MIterMut<R, Item = MIndex>,
    Pred: PredicateOp<Input::Item>,
    Output::Item: WriteFrom<Input::Item>,
{
    fn run(
        self,
        exec: &Executor<R>,
        input: Segmented<Input, InputOffsets>,
        output: SegmentedMut<Output, OutputOffsets>,
    ) -> Result<MIndex, Error> {
        let (values, offsets) = input.into_parts();
        let value_len = values.len()? as usize;
        let control = control::SegmentControl::new(exec, offsets, value_len)?;
        let flags = exec.alloc::<u32>(value_len);
        values.clone().transform_into(
            exec,
            PredicateMap::<Pred>(PhantomData),
            flags.slice_mut(..),
        )?;
        let storage =
            <Input::Read as crate::core::facade::AnyRead<R>>::normalize(values.lower_read(), exec)?;
        let (output_values, output_offsets) = output.into_parts();
        control.compact::<Input::Item, _, _>(exec, &storage, flags, output_values, output_offsets)
    }
}

impl<R, Input, InputOffsets, Output, Op, Init> ReduceLikeExecutable<R, Input, InputOffsets, Output>
    for ForEachSegment<Reduce<Op, Init>>
where
    R: Runtime,
    Input: MIter<R, Item = Init>,
    InputOffsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
    Init: CubeType + MAlloc<R> + Copy,
    Op: ReductionOp<Input::Item>,
    Output::Item: WriteFrom<Input::Item>,
{
    fn run(
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
            ReduceLikeExecutable<R, Input, InputOffsets, Output>
            for ForEachSegment<$algorithm<Pred>>
        where
            R: Runtime,
            Input: MIter<R>,
            InputOffsets: MIter<R, Item = MIndex>,
            Output: MIterMut<R>,
            Pred: PredicateOp<Input::Item>,
            Output::Item: WriteFrom<MIndex>,
        {
            fn run(
                self,
                exec: &Executor<R>,
                input: Segmented<Input, InputOffsets>,
                output: Output,
            ) -> Result<(), Error> {
                let (values, offsets) = input.into_parts();
                let value_len = values.len()? as usize;
                let mapped = exec.alloc::<u32>(value_len);
                values.transform_into(exec, $map::<Pred>(PhantomData), mapped.slice_mut(..))?;
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

impl<R, Input, InputOffsets, Output, Less> ReduceLikeExecutable<R, Input, InputOffsets, Output>
    for ForEachSegment<IsSorted<Less>>
where
    R: Runtime,
    Input: MIter<R>,
    InputOffsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
    Less: BinaryPredicateOp<Input::Item>,
    Output::Item: WriteFrom<MIndex>,
{
    fn run(
        self,
        exec: &Executor<R>,
        input: Segmented<Input, InputOffsets>,
        output: Output,
    ) -> Result<(), Error> {
        let IsSorted(less) = self.0;
        let (values, offsets) = input.into_parts();
        let value_len = values.len()? as usize;
        let control = control::SegmentControl::new(exec, offsets, value_len)?;
        let storage =
            <Input::Read as crate::core::facade::AnyRead<R>>::normalize(values.lower_read(), exec)?;
        let breaks = <<Input::Item as MItem<R>>::Kernel as crate::core::facade::KernelItem<
            R,
            Input::Item,
        >>::sorted_breaks(exec, &storage, less)?;
        control.clear_heads(exec, &breaks)?;
        let ordered = exec.alloc::<u32>(value_len);
        breaks
            .slice(..)
            .transform_into(exec, InvertFlag, ordered.slice_mut(..))?;
        reduce_segments(
            exec,
            Segmented::new(ordered.slice(..), control.offsets.slice(..)),
            1u32,
            control::MinU32,
            output,
        )
    }
}

impl<R, Input, InputOffsets, Output, Less> ReduceLikeExecutable<R, Input, InputOffsets, Output>
    for ForEachSegment<IsSortedUntil<Less>>
where
    R: Runtime,
    Input: MIter<R>,
    InputOffsets: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
    Less: BinaryPredicateOp<Input::Item>,
    Output::Item: WriteFrom<MIndex>,
{
    fn run(
        self,
        exec: &Executor<R>,
        input: Segmented<Input, InputOffsets>,
        output: Output,
    ) -> Result<(), Error> {
        let IsSortedUntil(less) = self.0;
        let (values, offsets) = input.into_parts();
        let value_len = values.len()? as usize;
        let control = control::SegmentControl::new(exec, offsets, value_len)?;
        let storage =
            <Input::Read as crate::core::facade::AnyRead<R>>::normalize(values.lower_read(), exec)?;
        let breaks = <<Input::Item as MItem<R>>::Kernel as crate::core::facade::KernelItem<
            R,
            Input::Item,
        >>::sorted_breaks(exec, &storage, less)?;
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
        result
            .slice(..)
            .transform_into(exec, crate::op::Identity, output)
    }
}
