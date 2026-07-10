use cubecl::prelude::Runtime;
use std::ops::{Bound, RangeBounds};

use crate::{Error, Executor, MIndex};

pub use crate::core::iter::Zip;
pub use crate::core::storage::WriteFrom;

fn resolve_iter_range<Bounds>(len: MIndex, range: Bounds) -> (usize, usize)
where
    Bounds: RangeBounds<MIndex>,
{
    let start = match range.start_bound() {
        Bound::Included(&start) => start,
        Bound::Excluded(&start) => start.checked_add(1).expect("slice start overflow"),
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        Bound::Included(&end) => end.checked_add(1).expect("slice end overflow"),
        Bound::Excluded(&end) => end,
        Bound::Unbounded => len,
    };
    assert!(
        start <= end,
        "slice start ({start}) is greater than slice end ({end})"
    );
    assert!(
        end <= len,
        "slice end ({end}) is out of bounds for iterator of length {len}"
    );
    (start as usize, (end - start) as usize)
}

/// Logical item supported by the public iterator and storage facade.
#[doc(hidden)]
pub trait MItem<R: Runtime>: crate::StorageLayout + crate::MAlloc<R> {
    #[doc(hidden)]
    type Kernel: private::KernelItem<R, Self>;
}

#[doc(hidden)]
impl<R, Item> MItem<R> for Item
where
    R: Runtime,
    Item: crate::StorageLayout + crate::MAlloc<R>,
    Item::StorageLeaves: private::KernelItem<R, Item>,
{
    type Kernel = Item::StorageLeaves;
}

/// Public read-only logical row stream.
///
/// Device slices, lazy expressions, and [`Zip`] trees implement this trait.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, MIter, transform};
/// use massively::{lazy, op::Identity};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let values = lazy::counting(10).take(5);
/// let middle = values.slice(1..4);
/// let output = exec.alloc::<u32>(middle.len().unwrap() as usize);
/// transform(&exec, middle, Identity, output.slice_mut(..)).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![11, 12, 13]);
/// ```
pub trait MIter<R: Runtime>: Sized {
    type Item: MItem<R>;

    #[doc(hidden)]
    type Slice;

    /// Returns a zero-copy logical subrange of this iterator.
    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice
    where
        Bounds: RangeBounds<MIndex>;

    #[doc(hidden)]
    type Read: private::AnyRead<R, Item = Self::Item>;

    #[doc(hidden)]
    fn lower_read(self) -> Self::Read;

    fn len(&self) -> Result<MIndex, Error>;

    fn is_empty(&self) -> Result<bool, Error> {
        Ok(self.len()? == 0)
    }

    #[doc(hidden)]
    fn transform_into<Output, Op>(
        self,
        exec: &Executor<R>,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R>,
        Op: crate::UnaryOp<Self::Item>,
        Output::Item: crate::WriteFrom<<Op as crate::UnaryOp<Self::Item>>::Output>;

    #[doc(hidden)]
    fn count_if_with<Pred>(self, exec: &Executor<R>, pred: Pred) -> Result<MIndex, Error>
    where
        Pred: crate::PredicateOp<Self::Item>;

    #[doc(hidden)]
    fn all_of_with<Pred>(self, exec: &Executor<R>, pred: Pred) -> Result<bool, Error>
    where
        Pred: crate::PredicateOp<Self::Item>;

    #[doc(hidden)]
    fn any_of_with<Pred>(self, exec: &Executor<R>, pred: Pred) -> Result<bool, Error>
    where
        Pred: crate::PredicateOp<Self::Item>;

    #[doc(hidden)]
    fn none_of_with<Pred>(self, exec: &Executor<R>, pred: Pred) -> Result<bool, Error>
    where
        Pred: crate::PredicateOp<Self::Item>;

    #[doc(hidden)]
    fn find_if_with<Pred>(self, exec: &Executor<R>, pred: Pred) -> Result<Option<MIndex>, Error>
    where
        Pred: crate::PredicateOp<Self::Item>;

    #[doc(hidden)]
    fn is_partitioned_with<Pred>(self, exec: &Executor<R>, pred: Pred) -> Result<bool, Error>
    where
        Pred: crate::PredicateOp<Self::Item>;

    #[doc(hidden)]
    fn reduce_with<Op>(
        self,
        exec: &Executor<R>,
        init: Self::Item,
        op: Op,
    ) -> Result<Self::Item, Error>
    where
        Op: crate::ReductionOp<Self::Item>;

    #[doc(hidden)]
    fn adjacent_find_with<Equal>(
        self,
        exec: &Executor<R>,
        equal: Equal,
    ) -> Result<Option<MIndex>, Error>
    where
        Equal: crate::BinaryPredicateOp<Self::Item>;

    #[doc(hidden)]
    fn is_sorted_until_with<Less>(self, exec: &Executor<R>, less: Less) -> Result<MIndex, Error>
    where
        Less: crate::BinaryPredicateOp<Self::Item>;

    #[doc(hidden)]
    fn is_sorted_with<Less>(self, exec: &Executor<R>, less: Less) -> Result<bool, Error>
    where
        Less: crate::BinaryPredicateOp<Self::Item>;

    #[doc(hidden)]
    fn min_element_with<Less>(
        self,
        exec: &Executor<R>,
        less: Less,
    ) -> Result<Option<MIndex>, Error>
    where
        Less: crate::BinaryPredicateOp<Self::Item>;

    #[doc(hidden)]
    fn max_element_with<Less>(
        self,
        exec: &Executor<R>,
        less: Less,
    ) -> Result<Option<MIndex>, Error>
    where
        Less: crate::BinaryPredicateOp<Self::Item>;

    #[doc(hidden)]
    fn minmax_element_with<Less>(
        self,
        exec: &Executor<R>,
        less: Less,
    ) -> Result<Option<(MIndex, MIndex)>, Error>
    where
        Less: crate::BinaryPredicateOp<Self::Item>;

    #[doc(hidden)]
    fn inclusive_scan_with<Output, Op>(
        self,
        exec: &Executor<R>,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R>,
        Output::Item: crate::WriteFrom<Self::Item>,
        Op: crate::ReductionOp<Self::Item>;

    #[doc(hidden)]
    fn adjacent_difference_with<Output, Op>(
        self,
        exec: &Executor<R>,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R>,
        Output::Item: crate::WriteFrom<Self::Item>,
        Op: crate::ReductionOp<Self::Item>;

    #[doc(hidden)]
    fn sort_with<Output, Less>(
        self,
        exec: &Executor<R>,
        less: Less,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R>,
        Output::Item: crate::WriteFrom<Self::Item>,
        Less: crate::BinaryPredicateOp<Self::Item>;

    #[doc(hidden)]
    fn unique_with<Output, Equal>(
        self,
        exec: &Executor<R>,
        equal: Equal,
        output: Output,
    ) -> Result<MIndex, Error>
    where
        Output: MIterMut<R>,
        Output::Item: crate::WriteFrom<Self::Item>,
        Equal: crate::BinaryPredicateOp<Self::Item>;

    #[doc(hidden)]
    fn materialize_u32(self, exec: &Executor<R>) -> Result<crate::DeviceVec<R, MIndex>, Error>
    where
        Self: MIter<R, Item = MIndex>,
    {
        let len = self.len()? as usize;
        let output = exec.alloc::<MIndex>(len);
        self.transform_into(exec, crate::op::Identity, output.slice_mut(..))?;
        Ok(output)
    }

    #[doc(hidden)]
    fn select_with_flags<Output>(
        self,
        exec: &Executor<R>,
        flags: crate::Column<MIndex>,
        invert: bool,
        output: Output,
    ) -> Result<MIndex, Error>
    where
        Output: MIterMut<R>,
        Output::Item: crate::WriteFrom<Self::Item>;

    #[doc(hidden)]
    fn partition_with<Output, Pred>(
        self,
        exec: &Executor<R>,
        pred: Pred,
        output: Output,
    ) -> Result<MIndex, Error>
    where
        Output: MIterMut<R>,
        Output::Item: crate::WriteFrom<Self::Item>,
        Pred: crate::PredicateOp<Self::Item>;

    #[doc(hidden)]
    fn indexed_with<Output>(
        self,
        exec: &Executor<R>,
        indices: crate::Column<MIndex>,
        flags: Option<crate::Column<MIndex>>,
        scatter: bool,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R>,
        Output::Item: crate::WriteFrom<Self::Item>;

    #[doc(hidden)]
    fn reverse_with<Output>(self, exec: &Executor<R>, output: Output) -> Result<(), Error>
    where
        Output: MIterMut<R>,
        Output::Item: crate::WriteFrom<Self::Item>,
    {
        let len = self.len()? as usize;
        let indices = <crate::ReverseCounting as MIter<R>>::materialize_u32(
            crate::ReverseCounting::new(len),
            exec,
        )?;
        self.indexed_with(exec, indices.column(), None, false, output)
    }

    #[doc(hidden)]
    fn transform_where_with<Output, Op>(
        self,
        exec: &Executor<R>,
        op: Op,
        flags: crate::Column<MIndex>,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R>,
        Op: crate::UnaryOp<Self::Item>,
        Output::Item: crate::WriteFrom<<Op as crate::UnaryOp<Self::Item>>::Output>;

    #[doc(hidden)]
    fn exclusive_scan_with<Output, Op>(
        self,
        exec: &Executor<R>,
        init: Self::Item,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R>,
        Output::Item: crate::WriteFrom<Self::Item>,
        Op: crate::ReductionOp<Self::Item>;

    #[doc(hidden)]
    fn equal_with<Right, Equal>(
        self,
        exec: &Executor<R>,
        right: Right,
        equal: Equal,
    ) -> Result<bool, Error>
    where
        Right: MIter<R, Item = Self::Item>,
        Equal: crate::BinaryPredicateOp<Self::Item>;

    #[doc(hidden)]
    fn mismatch_with<Right, Equal>(
        self,
        exec: &Executor<R>,
        right: Right,
        equal: Equal,
    ) -> Result<Option<MIndex>, Error>
    where
        Right: MIter<R, Item = Self::Item>,
        Equal: crate::BinaryPredicateOp<Self::Item>;

    #[doc(hidden)]
    fn lexicographical_with<Right, Less>(
        self,
        exec: &Executor<R>,
        right: Right,
        less: Less,
    ) -> Result<bool, Error>
    where
        Right: MIter<R, Item = Self::Item>,
        Less: crate::BinaryPredicateOp<Self::Item>;

    #[doc(hidden)]
    fn find_first_of_with<Needles, Equal>(
        self,
        exec: &Executor<R>,
        needles: Needles,
        equal: Equal,
    ) -> Result<Option<MIndex>, Error>
    where
        Needles: MIter<R, Item = Self::Item>,
        Equal: crate::BinaryPredicateOp<Self::Item>;

    #[doc(hidden)]
    fn bounds_with<Values, Less, Output>(
        self,
        exec: &Executor<R>,
        values: Values,
        less: Less,
        upper: bool,
        output: Output,
    ) -> Result<(), Error>
    where
        Values: MIter<R, Item = Self::Item>,
        Less: crate::BinaryPredicateOp<Self::Item>,
        Output: MIterMut<R>,
        Output::Item: crate::WriteFrom<MIndex>;

    #[doc(hidden)]
    fn merge_with<Right, Less, Output>(
        self,
        exec: &Executor<R>,
        right: Right,
        less: Less,
        output: Output,
    ) -> Result<(), Error>
    where
        Right: MIter<R, Item = Self::Item>,
        Less: crate::BinaryPredicateOp<Self::Item>,
        Output: MIterMut<R>,
        Output::Item: crate::WriteFrom<Self::Item>;

    #[doc(hidden)]
    fn set_with<Right, Less, Output>(
        self,
        exec: &Executor<R>,
        right: Right,
        less: Less,
        output: Output,
        mode: u8,
    ) -> Result<MIndex, Error>
    where
        Right: MIter<R, Item = Self::Item>,
        Less: crate::BinaryPredicateOp<Self::Item>,
        Output: MIterMut<R>,
        Output::Item: crate::WriteFrom<Self::Item>;

    #[doc(hidden)]
    fn sort_by_key_with<Values, Less, KeyOutput, ValueOutput>(
        self,
        exec: &Executor<R>,
        values: Values,
        less: Less,
        key_output: KeyOutput,
        value_output: ValueOutput,
    ) -> Result<(), Error>
    where
        Values: MIter<R>,
        Less: crate::BinaryPredicateOp<Self::Item>,
        KeyOutput: MIterMut<R>,
        KeyOutput::Item: crate::WriteFrom<Self::Item>,
        ValueOutput: MIterMut<R>,
        ValueOutput::Item: crate::WriteFrom<Values::Item>;

    #[doc(hidden)]
    fn scan_by_key_with<Values, Equal, Op, Output>(
        self,
        exec: &Executor<R>,
        values: Values,
        equal: Equal,
        init: Option<Values::Item>,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Values: MIter<R>,
        Equal: crate::BinaryPredicateOp<Self::Item>,
        Op: crate::ReductionOp<Values::Item>,
        Output: MIterMut<R>,
        Output::Item: crate::WriteFrom<Values::Item>;

    #[doc(hidden)]
    fn reduce_by_key_with<Values, Equal, Op, KeyOutput, ValueOutput>(
        self,
        exec: &Executor<R>,
        values: Values,
        equal: Equal,
        init: Values::Item,
        op: Op,
        key_output: KeyOutput,
        value_output: ValueOutput,
    ) -> Result<MIndex, Error>
    where
        Values: MIter<R>,
        Equal: crate::BinaryPredicateOp<Self::Item>,
        Op: crate::ReductionOp<Values::Item>,
        KeyOutput: MIterMut<R>,
        KeyOutput::Item: crate::WriteFrom<Self::Item>,
        ValueOutput: MIterMut<R>,
        ValueOutput::Item: crate::WriteFrom<Values::Item>;

    #[doc(hidden)]
    fn unique_by_key_with<Values, Equal, KeyOutput, ValueOutput>(
        self,
        exec: &Executor<R>,
        values: Values,
        equal: Equal,
        key_output: KeyOutput,
        value_output: ValueOutput,
    ) -> Result<MIndex, Error>
    where
        Values: MIter<R>,
        Equal: crate::BinaryPredicateOp<Self::Item>,
        KeyOutput: MIterMut<R>,
        KeyOutput::Item: crate::WriteFrom<Self::Item>,
        ValueOutput: MIterMut<R>,
        ValueOutput::Item: crate::WriteFrom<Values::Item>;

    #[doc(hidden)]
    fn merge_by_key_with<LeftValues, RightKeys, RightValues, Less, KeyOutput, ValueOutput>(
        self,
        exec: &Executor<R>,
        left_values: LeftValues,
        right_keys: RightKeys,
        right_values: RightValues,
        less: Less,
        key_output: KeyOutput,
        value_output: ValueOutput,
    ) -> Result<(), Error>
    where
        LeftValues: MIter<R>,
        RightKeys: MIter<R, Item = Self::Item>,
        RightValues: MIter<R, Item = LeftValues::Item>,
        Less: crate::BinaryPredicateOp<Self::Item>,
        KeyOutput: MIterMut<R>,
        KeyOutput::Item: crate::WriteFrom<Self::Item>,
        ValueOutput: MIterMut<R>,
        ValueOutput::Item: crate::WriteFrom<LeftValues::Item>;
}

/// Public preallocated output stream.
///
/// Device mutable slices and [`Zip`] trees of mutable slices implement this trait.
pub trait MIterMut<R: Runtime>: Sized {
    type Item: MItem<R>;

    #[doc(hidden)]
    type Slice;

    #[doc(hidden)]
    type SliceMut;

    /// Returns a read-only zero-copy subrange of this output.
    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice
    where
        Bounds: RangeBounds<MIndex>;

    /// Returns a mutable zero-copy subrange of this output.
    fn slice_mut<Bounds>(&self, range: Bounds) -> Self::SliceMut
    where
        Bounds: RangeBounds<MIndex>;

    fn len(&self) -> Result<MIndex, Error>;

    fn is_empty(&self) -> Result<bool, Error> {
        Ok(self.len()? == 0)
    }

    #[doc(hidden)]
    type Write: private::KernelWrite<R, Item = Self::Item>;

    #[doc(hidden)]
    fn lower_write(self) -> Self::Write;

    #[doc(hidden)]
    type ReboundWrite<Source>: private::KernelWrite<R, Item = Source>
    where
        Source: MItem<R>,
        Self::Item: crate::WriteFrom<Source>;

    #[doc(hidden)]
    fn lower_write_from<Source>(self) -> Self::ReboundWrite<Source>
    where
        Source: MItem<R>,
        Self::Item: crate::WriteFrom<Source>;

    #[doc(hidden)]
    fn fill_with(self, exec: &Executor<R>, value: Self::Item) -> Result<(), Error> {
        <Self::Write as private::KernelWrite<R>>::fill(self.lower_write(), exec, value)
    }

    #[doc(hidden)]
    fn replace_with_flags(
        self,
        exec: &Executor<R>,
        value: Self::Item,
        flags: crate::Column<MIndex>,
    ) -> Result<(), Error> {
        <Self::Write as private::KernelWrite<R>>::replace(self.lower_write(), exec, value, flags)
    }
}

#[doc(hidden)]
impl<R, Input, Item> MIter<R> for Input
where
    R: Runtime,
    Input: private::IterLength
        + Clone
        + crate::read::ReadExpression<Item = Item>
        + crate::read::SliceExpression
        + crate::read::LowerReadExpression
        + crate::reduce::StageRead<R, crate::read::Env0>,
    Input::Slots: private::KernelRead<R, Input>,
    Item: MItem<R>,
    <Item as crate::StorageLayout>::StorageLeaves: private::KernelItem<R, Item>,
{
    type Item = Item;
    type Slice = crate::read::Slice<R, Input>;
    type Read = private::Read<Input, Input::Slots>;

    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice
    where
        Bounds: RangeBounds<MIndex>,
    {
        let len = private::IterLength::logical_len(self)
            .expect("cannot slice an iterator with an invalid length");
        let len = MIndex::try_from(len).expect("iterator length exceeds MIndex");
        let (start, len) = resolve_iter_range(len, range);
        crate::read::Slice::new(self.slice_expression(start, len))
    }

    fn lower_read(self) -> Self::Read {
        private::Read::new(self)
    }

    fn len(&self) -> Result<MIndex, Error> {
        let len = private::IterLength::logical_len(self)?;
        MIndex::try_from(len).map_err(|_| Error::LengthTooLarge { len })
    }

    fn transform_into<Output, Op>(
        self,
        exec: &Executor<R>,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R>,
        Op: crate::UnaryOp<Self::Item>,
        Output::Item: crate::WriteFrom<<Op as crate::UnaryOp<Self::Item>>::Output>,
    {
        <Input::Slots as private::KernelRead<R, Input>>::transform(
            self,
            exec,
            op,
            output.lower_write(),
        )
    }

    fn count_if_with<Pred>(self, exec: &Executor<R>, pred: Pred) -> Result<MIndex, Error>
    where
        Pred: crate::PredicateOp<Self::Item>,
    {
        <Input::Slots as private::KernelRead<R, Input>>::count_if(self, exec, pred)
    }

    fn all_of_with<Pred>(self, exec: &Executor<R>, pred: Pred) -> Result<bool, Error>
    where
        Pred: crate::PredicateOp<Self::Item>,
    {
        <Input::Slots as private::KernelRead<R, Input>>::all_of(self, exec, pred)
    }

    fn any_of_with<Pred>(self, exec: &Executor<R>, pred: Pred) -> Result<bool, Error>
    where
        Pred: crate::PredicateOp<Self::Item>,
    {
        <Input::Slots as private::KernelRead<R, Input>>::any_of(self, exec, pred)
    }

    fn none_of_with<Pred>(self, exec: &Executor<R>, pred: Pred) -> Result<bool, Error>
    where
        Pred: crate::PredicateOp<Self::Item>,
    {
        <Input::Slots as private::KernelRead<R, Input>>::none_of(self, exec, pred)
    }

    fn find_if_with<Pred>(self, exec: &Executor<R>, pred: Pred) -> Result<Option<MIndex>, Error>
    where
        Pred: crate::PredicateOp<Self::Item>,
    {
        <Input::Slots as private::KernelRead<R, Input>>::find_if(self, exec, pred)
    }

    fn is_partitioned_with<Pred>(self, exec: &Executor<R>, pred: Pred) -> Result<bool, Error>
    where
        Pred: crate::PredicateOp<Self::Item>,
    {
        <Input::Slots as private::KernelRead<R, Input>>::is_partitioned(self, exec, pred)
    }

    fn reduce_with<Op>(
        self,
        exec: &Executor<R>,
        init: Self::Item,
        op: Op,
    ) -> Result<Self::Item, Error>
    where
        Op: crate::ReductionOp<Self::Item>,
    {
        <Input::Slots as private::KernelRead<R, Input>>::reduce(self, exec, init, op)
    }

    fn adjacent_find_with<Equal>(
        self,
        exec: &Executor<R>,
        equal: Equal,
    ) -> Result<Option<MIndex>, Error>
    where
        Equal: crate::BinaryPredicateOp<Self::Item>,
    {
        <Input::Slots as private::KernelRead<R, Input>>::adjacent_find(self, exec, equal)
    }

    fn is_sorted_until_with<Less>(self, exec: &Executor<R>, less: Less) -> Result<MIndex, Error>
    where
        Less: crate::BinaryPredicateOp<Self::Item>,
    {
        <Input::Slots as private::KernelRead<R, Input>>::is_sorted_until(self, exec, less)
    }

    fn is_sorted_with<Less>(self, exec: &Executor<R>, less: Less) -> Result<bool, Error>
    where
        Less: crate::BinaryPredicateOp<Self::Item>,
    {
        <Input::Slots as private::KernelRead<R, Input>>::is_sorted(self, exec, less)
    }

    fn min_element_with<Less>(self, exec: &Executor<R>, less: Less) -> Result<Option<MIndex>, Error>
    where
        Less: crate::BinaryPredicateOp<Self::Item>,
    {
        <Input::Slots as private::KernelRead<R, Input>>::min_element(self, exec, less)
    }

    fn max_element_with<Less>(self, exec: &Executor<R>, less: Less) -> Result<Option<MIndex>, Error>
    where
        Less: crate::BinaryPredicateOp<Self::Item>,
    {
        <Input::Slots as private::KernelRead<R, Input>>::max_element(self, exec, less)
    }

    fn minmax_element_with<Less>(
        self,
        exec: &Executor<R>,
        less: Less,
    ) -> Result<Option<(MIndex, MIndex)>, Error>
    where
        Less: crate::BinaryPredicateOp<Self::Item>,
    {
        <Input::Slots as private::KernelRead<R, Input>>::minmax_element(self, exec, less)
    }

    fn inclusive_scan_with<Output, Op>(
        self,
        exec: &Executor<R>,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R>,
        Output::Item: crate::WriteFrom<Self::Item>,
        Op: crate::ReductionOp<Self::Item>,
    {
        <Input::Slots as private::KernelRead<R, Input>>::inclusive_scan(
            self,
            exec,
            op,
            output.lower_write_from::<Self::Item>(),
        )
    }

    fn adjacent_difference_with<Output, Op>(
        self,
        exec: &Executor<R>,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R>,
        Output::Item: crate::WriteFrom<Self::Item>,
        Op: crate::ReductionOp<Self::Item>,
    {
        <Input::Slots as private::KernelRead<R, Input>>::adjacent_difference(
            self,
            exec,
            op,
            output.lower_write_from::<Self::Item>(),
        )
    }

    fn sort_with<Output, Less>(
        self,
        exec: &Executor<R>,
        less: Less,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R>,
        Output::Item: crate::WriteFrom<Self::Item>,
        Less: crate::BinaryPredicateOp<Self::Item>,
    {
        <Input::Slots as private::KernelRead<R, Input>>::sort(
            self,
            exec,
            less,
            output.lower_write_from::<Self::Item>(),
        )
    }

    fn unique_with<Output, Equal>(
        self,
        exec: &Executor<R>,
        equal: Equal,
        output: Output,
    ) -> Result<MIndex, Error>
    where
        Output: MIterMut<R>,
        Output::Item: crate::WriteFrom<Self::Item>,
        Equal: crate::BinaryPredicateOp<Self::Item>,
    {
        <Input::Slots as private::KernelRead<R, Input>>::unique(
            self,
            exec,
            equal,
            output.lower_write_from::<Self::Item>(),
        )
    }

    fn select_with_flags<Output>(
        self,
        exec: &Executor<R>,
        flags: crate::Column<MIndex>,
        invert: bool,
        output: Output,
    ) -> Result<MIndex, Error>
    where
        Output: MIterMut<R>,
        Output::Item: crate::WriteFrom<Self::Item>,
    {
        <Input::Slots as private::KernelRead<R, Input>>::select(
            self,
            exec,
            flags,
            invert,
            output.lower_write_from::<Self::Item>(),
        )
    }

    fn partition_with<Output, Pred>(
        self,
        exec: &Executor<R>,
        pred: Pred,
        output: Output,
    ) -> Result<MIndex, Error>
    where
        Output: MIterMut<R>,
        Output::Item: crate::WriteFrom<Self::Item>,
        Pred: crate::PredicateOp<Self::Item>,
    {
        <Input::Slots as private::KernelRead<R, Input>>::partition(
            self,
            exec,
            pred,
            output.lower_write_from::<Self::Item>(),
        )
    }

    fn indexed_with<Output>(
        self,
        exec: &Executor<R>,
        indices: crate::Column<MIndex>,
        flags: Option<crate::Column<MIndex>>,
        scatter: bool,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R>,
        Output::Item: crate::WriteFrom<Self::Item>,
    {
        <Input::Slots as private::KernelRead<R, Input>>::indexed(
            self,
            exec,
            indices,
            flags,
            scatter,
            output.lower_write_from::<Self::Item>(),
        )
    }

    fn transform_where_with<Output, Op>(
        self,
        exec: &Executor<R>,
        op: Op,
        flags: crate::Column<MIndex>,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R>,
        Op: crate::UnaryOp<Self::Item>,
        Output::Item: crate::WriteFrom<<Op as crate::UnaryOp<Self::Item>>::Output>,
    {
        <Input::Slots as private::KernelRead<R, Input>>::transform_where(
            self,
            exec,
            op,
            flags,
            output.lower_write(),
        )
    }

    fn exclusive_scan_with<Output, Op>(
        self,
        exec: &Executor<R>,
        init: Self::Item,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R>,
        Output::Item: crate::WriteFrom<Self::Item>,
        Op: crate::ReductionOp<Self::Item>,
    {
        <Input::Slots as private::KernelRead<R, Input>>::exclusive_scan(
            self,
            exec,
            init,
            op,
            output.lower_write_from::<Self::Item>(),
        )
    }

    fn equal_with<Right, Equal>(
        self,
        exec: &Executor<R>,
        right: Right,
        equal: Equal,
    ) -> Result<bool, Error>
    where
        Right: MIter<R, Item = Self::Item>,
        Equal: crate::BinaryPredicateOp<Self::Item>,
    {
        match private::run_pair(exec, self.lower_read(), right.lower_read(), equal, 0)? {
            private::PairResult::Bool(value) => Ok(value),
            _ => unreachable!(),
        }
    }

    fn mismatch_with<Right, Equal>(
        self,
        exec: &Executor<R>,
        right: Right,
        equal: Equal,
    ) -> Result<Option<MIndex>, Error>
    where
        Right: MIter<R, Item = Self::Item>,
        Equal: crate::BinaryPredicateOp<Self::Item>,
    {
        match private::run_pair(exec, self.lower_read(), right.lower_read(), equal, 1)? {
            private::PairResult::Index(value) => Ok(value),
            _ => unreachable!(),
        }
    }

    fn lexicographical_with<Right, Less>(
        self,
        exec: &Executor<R>,
        right: Right,
        less: Less,
    ) -> Result<bool, Error>
    where
        Right: MIter<R, Item = Self::Item>,
        Less: crate::BinaryPredicateOp<Self::Item>,
    {
        match private::run_pair(exec, self.lower_read(), right.lower_read(), less, 2)? {
            private::PairResult::Bool(value) => Ok(value),
            _ => unreachable!(),
        }
    }

    fn find_first_of_with<Needles, Equal>(
        self,
        exec: &Executor<R>,
        needles: Needles,
        equal: Equal,
    ) -> Result<Option<MIndex>, Error>
    where
        Needles: MIter<R, Item = Self::Item>,
        Equal: crate::BinaryPredicateOp<Self::Item>,
    {
        match private::run_pair(exec, self.lower_read(), needles.lower_read(), equal, 3)? {
            private::PairResult::Index(value) => Ok(value),
            _ => unreachable!(),
        }
    }

    fn bounds_with<Values, Less, Output>(
        self,
        exec: &Executor<R>,
        values: Values,
        less: Less,
        upper: bool,
        output: Output,
    ) -> Result<(), Error>
    where
        Values: MIter<R, Item = Self::Item>,
        Less: crate::BinaryPredicateOp<Self::Item>,
        Output: MIterMut<R>,
        Output::Item: crate::WriteFrom<MIndex>,
    {
        let source = <Self::Read as private::AnyRead<R>>::normalize(self.lower_read(), exec)?;
        let values = <Values::Read as private::AnyRead<R>>::normalize(values.lower_read(), exec)?;
        let bounds =
            <<Self::Item as crate::StorageLayout>::StorageLeaves as private::KernelItem<
                R,
                Self::Item,
            >>::bounds(exec, &source, &values, less, upper)?;
        bounds
            .column()
            .transform_into(exec, crate::op::Identity, output)
    }

    fn merge_with<Right, Less, Output>(
        self,
        exec: &Executor<R>,
        right: Right,
        less: Less,
        output: Output,
    ) -> Result<(), Error>
    where
        Right: MIter<R, Item = Self::Item>,
        Less: crate::BinaryPredicateOp<Self::Item>,
        Output: MIterMut<R>,
        Output::Item: crate::WriteFrom<Self::Item>,
    {
        let left = <Self::Read as private::AnyRead<R>>::normalize(self.lower_read(), exec)?;
        let right = <Right::Read as private::AnyRead<R>>::normalize(right.lower_read(), exec)?;
        let control = <<Self::Item as MItem<R>>::Kernel as private::KernelItem<
            R,
            Self::Item,
        >>::merge_control(exec, &left, &right, less)?;
        private::KernelWrite::concat_storage(
            output.lower_write_from::<Self::Item>(),
            exec,
            &left,
            &right,
            &control,
        )
    }

    fn set_with<Right, Less, Output>(
        self,
        exec: &Executor<R>,
        right: Right,
        less: Less,
        output: Output,
        mode: u8,
    ) -> Result<MIndex, Error>
    where
        Right: MIter<R, Item = Self::Item>,
        Less: crate::BinaryPredicateOp<Self::Item>,
        Output: MIterMut<R>,
        Output::Item: crate::WriteFrom<Self::Item>,
    {
        let left = <Self::Read as private::AnyRead<R>>::normalize(self.lower_read(), exec)?;
        let right = <Right::Read as private::AnyRead<R>>::normalize(right.lower_read(), exec)?;
        private::KernelWrite::set_storage(
            output.lower_write_from::<Self::Item>(),
            exec,
            &left,
            &right,
            less,
            mode,
        )
    }

    fn sort_by_key_with<Values, Less, KeyOutput, ValueOutput>(
        self,
        exec: &Executor<R>,
        values: Values,
        less: Less,
        key_output: KeyOutput,
        value_output: ValueOutput,
    ) -> Result<(), Error>
    where
        Values: MIter<R>,
        Less: crate::BinaryPredicateOp<Self::Item>,
        KeyOutput: MIterMut<R>,
        KeyOutput::Item: crate::WriteFrom<Self::Item>,
        ValueOutput: MIterMut<R>,
        ValueOutput::Item: crate::WriteFrom<Values::Item>,
    {
        let keys = <Self::Read as private::AnyRead<R>>::normalize(self.lower_read(), exec)?;
        let values = <Values::Read as private::AnyRead<R>>::normalize(values.lower_read(), exec)?;
        let key_len = crate::MStorage::len(&keys)?;
        let value_len = crate::MStorage::len(&values)?;
        if key_len != value_len {
            return Err(Error::LengthMismatch {
                left: key_len,
                right: value_len,
            });
        }
        let permutation = <<Self::Item as MItem<R>>::Kernel as private::KernelItem<
            R,
            Self::Item,
        >>::sort_control(exec, &keys, less)?;
        private::KernelWrite::gather_storage(
            key_output.lower_write_from::<Self::Item>(),
            exec,
            &keys,
            permutation.column(),
        )?;
        private::KernelWrite::gather_storage(
            value_output.lower_write_from::<Values::Item>(),
            exec,
            &values,
            permutation.column(),
        )
    }

    fn scan_by_key_with<Values, Equal, Op, Output>(
        self,
        exec: &Executor<R>,
        values: Values,
        equal: Equal,
        init: Option<Values::Item>,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Values: MIter<R>,
        Equal: crate::BinaryPredicateOp<Self::Item>,
        Op: crate::ReductionOp<Values::Item>,
        Output: MIterMut<R>,
        Output::Item: crate::WriteFrom<Values::Item>,
    {
        let keys = <Self::Read as private::AnyRead<R>>::normalize(self.lower_read(), exec)?;
        let values = <Values::Read as private::AnyRead<R>>::normalize(values.lower_read(), exec)?;
        let heads = <<Self::Item as MItem<R>>::Kernel as private::KernelItem<
            R,
            Self::Item,
        >>::segment_heads(exec, &keys, equal)?;
        let mode = u8::from(init.is_some());
        let scanned = <<Values::Item as MItem<R>>::Kernel as private::KernelItem<
            R,
            Values::Item,
        >>::segmented(exec, &values, &heads, init, op, mode)?;
        private::KernelWrite::materialize_storage(
            output.lower_write_from::<Values::Item>(),
            exec,
            &scanned,
        )
    }

    fn reduce_by_key_with<Values, Equal, Op, KeyOutput, ValueOutput>(
        self,
        exec: &Executor<R>,
        values: Values,
        equal: Equal,
        init: Values::Item,
        op: Op,
        key_output: KeyOutput,
        value_output: ValueOutput,
    ) -> Result<MIndex, Error>
    where
        Values: MIter<R>,
        Equal: crate::BinaryPredicateOp<Self::Item>,
        Op: crate::ReductionOp<Values::Item>,
        KeyOutput: MIterMut<R>,
        KeyOutput::Item: crate::WriteFrom<Self::Item>,
        ValueOutput: MIterMut<R>,
        ValueOutput::Item: crate::WriteFrom<Values::Item>,
    {
        let keys = <Self::Read as private::AnyRead<R>>::normalize(self.lower_read(), exec)?;
        let values = <Values::Read as private::AnyRead<R>>::normalize(values.lower_read(), exec)?;
        let key_len = crate::MStorage::len(&keys)?;
        let value_len = crate::MStorage::len(&values)?;
        if key_len != value_len {
            return Err(Error::LengthMismatch {
                left: key_len,
                right: value_len,
            });
        }
        let heads = <<Self::Item as MItem<R>>::Kernel as private::KernelItem<
            R,
            Self::Item,
        >>::segment_heads(exec, &keys, equal)?;
        let tails = crate::core::by_key::segment_tails(exec, &heads)?;
        let reduced = <<Values::Item as MItem<R>>::Kernel as private::KernelItem<
            R,
            Values::Item,
        >>::segmented(exec, &values, &heads, Some(init), op, 2)?;
        let key_count = private::KernelWrite::select_storage(
            key_output.lower_write_from::<Self::Item>(),
            exec,
            &keys,
            heads,
        )?;
        let value_count = private::KernelWrite::select_storage(
            value_output.lower_write_from::<Values::Item>(),
            exec,
            &reduced,
            tails,
        )?;
        debug_assert_eq!(key_count, value_count);
        Ok(key_count)
    }

    fn unique_by_key_with<Values, Equal, KeyOutput, ValueOutput>(
        self,
        exec: &Executor<R>,
        values: Values,
        equal: Equal,
        key_output: KeyOutput,
        value_output: ValueOutput,
    ) -> Result<MIndex, Error>
    where
        Values: MIter<R>,
        Equal: crate::BinaryPredicateOp<Self::Item>,
        KeyOutput: MIterMut<R>,
        KeyOutput::Item: crate::WriteFrom<Self::Item>,
        ValueOutput: MIterMut<R>,
        ValueOutput::Item: crate::WriteFrom<Values::Item>,
    {
        let keys = <Self::Read as private::AnyRead<R>>::normalize(self.lower_read(), exec)?;
        let values = <Values::Read as private::AnyRead<R>>::normalize(values.lower_read(), exec)?;
        let key_len = crate::MStorage::len(&keys)?;
        let value_len = crate::MStorage::len(&values)?;
        if key_len != value_len {
            return Err(Error::LengthMismatch {
                left: key_len,
                right: value_len,
            });
        }
        let heads = <<Self::Item as MItem<R>>::Kernel as private::KernelItem<
            R,
            Self::Item,
        >>::segment_heads(exec, &keys, equal)?;
        let key_count = private::KernelWrite::select_storage(
            key_output.lower_write_from::<Self::Item>(),
            exec,
            &keys,
            heads.clone(),
        )?;
        let value_count = private::KernelWrite::select_storage(
            value_output.lower_write_from::<Values::Item>(),
            exec,
            &values,
            heads,
        )?;
        debug_assert_eq!(key_count, value_count);
        Ok(key_count)
    }

    fn merge_by_key_with<LeftValues, RightKeys, RightValues, Less, KeyOutput, ValueOutput>(
        self,
        exec: &Executor<R>,
        left_values: LeftValues,
        right_keys: RightKeys,
        right_values: RightValues,
        less: Less,
        key_output: KeyOutput,
        value_output: ValueOutput,
    ) -> Result<(), Error>
    where
        LeftValues: MIter<R>,
        RightKeys: MIter<R, Item = Self::Item>,
        RightValues: MIter<R, Item = LeftValues::Item>,
        Less: crate::BinaryPredicateOp<Self::Item>,
        KeyOutput: MIterMut<R>,
        KeyOutput::Item: crate::WriteFrom<Self::Item>,
        ValueOutput: MIterMut<R>,
        ValueOutput::Item: crate::WriteFrom<LeftValues::Item>,
    {
        let left_keys = <Self::Read as private::AnyRead<R>>::normalize(self.lower_read(), exec)?;
        let left_values =
            <LeftValues::Read as private::AnyRead<R>>::normalize(left_values.lower_read(), exec)?;
        let right_keys =
            <RightKeys::Read as private::AnyRead<R>>::normalize(right_keys.lower_read(), exec)?;
        let right_values =
            <RightValues::Read as private::AnyRead<R>>::normalize(right_values.lower_read(), exec)?;
        let left_key_len = crate::MStorage::len(&left_keys)?;
        let left_value_len = crate::MStorage::len(&left_values)?;
        let right_key_len = crate::MStorage::len(&right_keys)?;
        let right_value_len = crate::MStorage::len(&right_values)?;
        if left_key_len != left_value_len {
            return Err(Error::LengthMismatch {
                left: left_key_len,
                right: left_value_len,
            });
        }
        if right_key_len != right_value_len {
            return Err(Error::LengthMismatch {
                left: right_key_len,
                right: right_value_len,
            });
        }
        let control = <<Self::Item as MItem<R>>::Kernel as private::KernelItem<
            R,
            Self::Item,
        >>::merge_control(exec, &left_keys, &right_keys, less)?;
        private::KernelWrite::concat_storage(
            key_output.lower_write_from::<Self::Item>(),
            exec,
            &left_keys,
            &right_keys,
            &control,
        )?;
        private::KernelWrite::concat_storage(
            value_output.lower_write_from::<LeftValues::Item>(),
            exec,
            &left_values,
            &right_values,
            &control,
        )
    }
}

#[doc(hidden)]
impl<R, Output> MIterMut<R> for Output
where
    R: Runtime,
    Output: crate::output::OutputExpression
        + crate::output::LowerOutputExpression
        + crate::output::ReadOutput
        + crate::output::StageOutput<R, crate::read::Env0>
        + crate::selection::FillOutput<R>
        + crate::selection::ReplaceOutput<R>
        + crate::output::SliceOutput,
    private::Write<Output, Output::Slots>:
        private::KernelWrite<R, Item = Output::Item, Output = Output>,
    Output::Item: MItem<R>,
{
    type Item = <Output as crate::output::OutputExpression>::Item;
    type Slice = crate::read::Slice<R, Output::Read>;
    type SliceMut = crate::output::Slice<R, Output>;
    type Write = private::Write<Output, Output::Slots>;
    type ReboundWrite<Source>
        = <<Source as MItem<R>>::Kernel as private::KernelItem<R, Source>>::ReboundWrite<Output>
    where
        Source: MItem<R>,
        Self::Item: crate::WriteFrom<Source>;

    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice
    where
        Bounds: RangeBounds<MIndex>,
    {
        let len = crate::output::OutputExpression::logical_len(self)
            .expect("cannot slice an output with an invalid length");
        let len = MIndex::try_from(len).expect("output length exceeds MIndex");
        let (start, len) = resolve_iter_range(len, range);
        crate::read::Slice::new(self.slice_read(start..start + len))
    }

    fn slice_mut<Bounds>(&self, range: Bounds) -> Self::SliceMut
    where
        Bounds: RangeBounds<MIndex>,
    {
        let len = crate::output::OutputExpression::logical_len(self)
            .expect("cannot slice an output with an invalid length");
        let len = MIndex::try_from(len).expect("output length exceeds MIndex");
        let (start, len) = resolve_iter_range(len, range);
        crate::output::Slice::new(self.slice_output(start..start + len))
    }

    fn len(&self) -> Result<MIndex, Error> {
        let len = crate::output::OutputExpression::logical_len(self)?;
        MIndex::try_from(len).map_err(|_| Error::LengthTooLarge { len })
    }

    fn lower_write(self) -> Self::Write {
        private::Write::new(self)
    }

    fn lower_write_from<Source>(self) -> Self::ReboundWrite<Source>
    where
        Source: MItem<R>,
        Self::Item: crate::WriteFrom<Source>,
    {
        <<Source as MItem<R>>::Kernel as private::KernelItem<R, Source>>::rebind_write(self)
    }
}

use crate::core::facade as private;

/// Combines two iterators into one iterator of paired items.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, UnaryOp, transform, zip2};
///
/// struct AddPair;
///
/// #[cubecl::cube]
/// impl UnaryOp<(u32, u32)> for AddPair {
///     type Output = u32;
///
///     fn apply(value: (u32, u32)) -> u32 {
///         value.0 + value.1
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let left = exec.to_device(&[1_u32, 2, 3]);
/// let right = exec.to_device(&[10_u32, 20, 30]);
/// let output = exec.alloc::<u32>(3);
/// let input = zip2(left.slice(..), right.slice(..));
/// transform(&exec, input, AddPair, output.slice_mut(..)).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![11, 22, 33]);
/// ```
pub fn zip2<A, B>(a: A, b: B) -> Zip<A, B> {
    Zip::new(a, b)
}

/// Combines three iterators into an iterator whose item is [`crate::Tuple3`].
///
/// Like all `zipN` helpers, this constructs the internal left-associated binary tree.
/// See [`zip2`] for a complete example.
pub fn zip3<A, B, C>(a: A, b: B, c: C) -> Zip<Zip<A, B>, C> {
    Zip::new(Zip::new(a, b), c)
}

/// Combines four iterators into an iterator whose item is [`crate::Tuple4`].
///
/// See [`zip2`] for a complete example.
pub fn zip4<A, B, C, D>(a: A, b: B, c: C, d: D) -> Zip<Zip<Zip<A, B>, C>, D> {
    Zip::new(zip3(a, b, c), d)
}

/// Combines five iterators into an iterator whose item is [`crate::Tuple5`].
///
/// See [`zip2`] for a complete example.
pub fn zip5<A, B, C, D, E>(a: A, b: B, c: C, d: D, e: E) -> Zip<Zip<Zip<Zip<A, B>, C>, D>, E> {
    Zip::new(zip4(a, b, c, d), e)
}

/// Combines six iterators into an iterator whose item is [`crate::Tuple6`].
///
/// See [`zip2`] for a complete example.
pub fn zip6<A, B, C, D, E, F>(
    a: A,
    b: B,
    c: C,
    d: D,
    e: E,
    f: F,
) -> Zip<Zip<Zip<Zip<Zip<A, B>, C>, D>, E>, F> {
    Zip::new(zip5(a, b, c, d, e), f)
}

/// Combines seven iterators into an iterator whose item is [`crate::Tuple7`].
///
/// See [`zip2`] for a complete example.
pub fn zip7<A, B, C, D, E, F, G>(
    a: A,
    b: B,
    c: C,
    d: D,
    e: E,
    f: F,
    g: G,
) -> Zip<Zip<Zip<Zip<Zip<Zip<A, B>, C>, D>, E>, F>, G> {
    Zip::new(zip6(a, b, c, d, e, f), g)
}
