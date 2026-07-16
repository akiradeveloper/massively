use cubecl::prelude::{CubeType, Runtime};
use std::ops::{Bound, RangeBounds};

use crate::{Error, Executor};

pub use crate::core::iter::Zip;
pub use crate::core::storage::WritableFrom;

/// Owned canonical storage for one left-associated logical item type.
pub trait MStorage<R: Runtime>: Sized {
    type Item: CanonicalForm<R, Storage = Self>;

    /// Allocates uninitialized storage for `len` logical items.
    ///
    /// The storage must be completely written before it is read.
    #[doc(hidden)]
    fn allocate(exec: &Executor<R>, len: usize) -> Self;

    fn len(&self) -> Result<usize, Error>;

    fn is_empty(&self) -> Result<bool, Error> {
        Ok(self.len()? == 0)
    }

    /// Shrinks the logical length without copying or reallocating device memory.
    ///
    /// Multi-column storage truncates every physical column to the same length.
    fn truncate(&mut self, len: usize);

    fn slice<Bounds>(&self, range: Bounds) -> impl MIter<R, Item = Self::Item> + '_
    where
        Bounds: RangeBounds<usize>;

    fn slice_mut<Bounds>(&self, range: Bounds) -> impl MIterMut<R, Item = Self::Item> + '_
    where
        Bounds: RangeBounds<usize>;
}

/// A canonical logical item with owned device storage.
///
/// Only scalar items and canonical left-associated tuple shapes implement
/// this trait. Semantic tuple association is handled by [`ToCanonical`].
pub trait CanonicalForm<R: Runtime>: CubeType + WritableFrom<Self> + Sized {
    type Storage: MStorage<R, Item = Self>;
}

/// Internal runtime-independent mapping to a canonical left-associated shape.
#[doc(hidden)]
pub trait CanonicalShape: CubeType + Sized {
    type Canonical: CubeType + WritableFrom<Self>;
}

/// Sealed physical contract for algorithms that normalize a semantic item
/// into canonical storage.
#[doc(hidden)]
pub(crate) trait CanonicalAbi<R: Runtime>: MItem<R> + crate::CanonicalAlloc<R> {}

impl<R, Item> CanonicalAbi<R> for Item
where
    R: Runtime,
    Item: MItem<R> + crate::CanonicalAlloc<R>,
{
}

/// Sealed physical contract for algorithms that require canonical scratch
/// storage for a semantic item.
#[doc(hidden)]
pub(crate) trait ScratchAbi<R: Runtime>:
    CanonicalAbi<R> + crate::core::allocation::ScratchStorage<R>
{
}

impl<R, Item> ScratchAbi<R> for Item
where
    R: Runtime,
    Item: CanonicalAbi<R> + crate::core::allocation::ScratchStorage<R>,
{
}

/// Sealed storage-shape dispatch for algorithms that materialize before
/// sorting. Public iterator APIs remain independent of physical arity.
pub(crate) trait SortAbi<R: Runtime>: MItem<R> + crate::CanonicalAlloc<R> {
    fn sort_storage<Less>(
        exec: &Executor<R>,
        input: <Self as crate::CanonicalAlloc<R>>::CanonicalStorage,
        carry_indices: bool,
    ) -> Result<
        crate::ordering::sort::OrderingResult<
            R,
            <Self as crate::CanonicalAlloc<R>>::CanonicalStorage,
        >,
        Error,
    >
    where
        Less: crate::op::BinaryPredicateOp<Self>;
}

impl<R, Item> SortAbi<R> for Item
where
    R: Runtime,
    Item: MItem<R> + crate::CanonicalAlloc<R>,
    <Item as crate::StorageLayout>::StorageLeaves: crate::ordering::sort::SortLeaves<R, Item>,
{
    fn sort_storage<Less>(
        exec: &Executor<R>,
        input: <Self as crate::CanonicalAlloc<R>>::CanonicalStorage,
        carry_indices: bool,
    ) -> Result<
        crate::ordering::sort::OrderingResult<
            R,
            <Self as crate::CanonicalAlloc<R>>::CanonicalStorage,
        >,
        Error,
    >
    where
        Less: crate::op::BinaryPredicateOp<Self>,
    {
        <<Self as crate::StorageLayout>::StorageLeaves as crate::ordering::sort::SortLeaves<
            R,
            Self,
        >>::sort_storage::<Less>(exec, input, carry_indices)
    }
}

/// Seals canonicalization while carrying backend-only implementation
/// capabilities. These are not part of the public materialization model.
#[doc(hidden)]
pub(crate) trait Sealed<R: Runtime>: ScratchAbi<R> + SortAbi<R> {}

impl<R, Item> Sealed<R> for Item
where
    R: Runtime,
    Item: ScratchAbi<R> + SortAbi<R>,
{
}

/// An item that can be written into owned canonical storage on `R`.
///
/// This single capability guarantees both that the canonical storage can be
/// allocated and that the semantic item can be written into it.
#[allow(private_bounds)]
pub trait ToCanonical<R: Runtime>: CubeType + Sized + Sealed<R> {
    /// The unique left-associated item written at an owned-storage boundary.
    type Canonical: CanonicalForm<R> + WritableFrom<Self>;
}

#[doc(hidden)]
#[allow(private_bounds)]
impl<R, Item> ToCanonical<R> for Item
where
    R: Runtime,
    Item: CanonicalShape + ScratchAbi<R> + SortAbi<R>,
    <Item as CanonicalShape>::Canonical: CanonicalForm<R>,
{
    type Canonical = <Item as CanonicalShape>::Canonical;
}

/// Owned canonical device storage for a semantic item type.
///
/// A scalar item maps to one [`crate::DeviceVec`]. A multi-column item maps to
/// a left-associated [`Zip`] tree of `DeviceVec`s. Non-canonical tuple
/// association is normalized through [`ToCanonical::Canonical`].
pub type MVec<R, Item> = <<Item as ToCanonical<R>>::Canonical as CanonicalForm<R>>::Storage;

pub(crate) fn resolve_iter_range<Bounds>(len: usize, range: Bounds) -> (usize, usize)
where
    Bounds: RangeBounds<usize>,
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
    (start, end - start)
}

/// Lowers a logical iterator while preserving its actual physical read arity.
pub(crate) fn lower<R, Input>(input: Input) -> Input::Read
where
    R: Runtime,
    Input: MIter<R>,
{
    input.lower_read()
}

/// Lowers a logical iterator and selects the current fixed thirteen-slot ABI.
///
/// Keeping this conversion explicit at consumer call sites leaves room for an
/// exact-arity launch policy without changing [`MIter`] or read expressions.
pub(crate) fn lower_fixed<R, Input>(input: Input) -> crate::read::FixedRead<Input::Read>
where
    R: Runtime,
    Input: MIter<R>,
{
    private::KernelInput::into_fixed(lower::<R, _>(input))
}

/// Materializes an already-physical `u32` iterator through the fixed read ABI.
pub(crate) fn materialize_u32<R, Input>(
    exec: &Executor<R>,
    input: Input,
) -> Result<crate::DeviceVec<R, u32>, Error>
where
    R: Runtime,
    Input: MIter<R, Item = u32>,
{
    let len = input.len()?;
    let output = exec.alloc::<u32>(len);
    let input = lower_fixed::<R, _>(input);
    let output_view = output.slice_mut(..);
    crate::transform::materialize_fixed(exec, &input, &output_view)?;
    Ok(output)
}

/// Internal marker for values supported by the physical storage ABI.
///
/// This is deliberately not required by [`MIter`]; read-only semantic values
/// may have no storage layout. Implementing this marker does not imply that
/// new owned storage can be allocated.
#[doc(hidden)]
pub trait MItem<R: Runtime>:
    crate::StorageLayout<
        StorageLeaves: private::KernelValue<
            StorageArity = <Self as crate::StorageLayout>::StorageArity,
        > + private::KernelOutputLeaves,
    >
{
}

#[doc(hidden)]
impl<R, Item> MItem<R> for Item
where
    R: Runtime,
    Item: crate::StorageLayout,
    Item::StorageLeaves: private::KernelValue<StorageArity = <Item as crate::StorageLayout>::StorageArity>
        + private::KernelOutputLeaves,
{
}

/// Public read-only logical row stream.
///
/// Device slices, lazy expressions, and [`Zip`] trees implement this trait.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, MIter, lazy, vector::gather};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let values = exec.to_device(&[10_u32, 20, 30, 40, 50]);
/// let indices = lazy::counting(0).take(5);
/// let middle = indices.slice(1..4);
/// let output = gather(&exec, values.slice(..), middle).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![20, 30, 40]);
/// ```
pub trait MIter<R: Runtime>: Clone + Sized {
    /// Semantic value produced by one indexed read.
    ///
    /// Reading does not imply that the value has a storage layout, can be
    /// allocated, or can cross a write boundary.
    type Item: CubeType + Send + Sync + 'static;

    /// Exact-arity device read plan for this iterator.
    #[doc(hidden)]
    type Read: private::KernelInput<R, Item = Self::Item>;

    #[doc(hidden)]
    type Slice: MIter<R, Item = Self::Item>;

    /// Returns a zero-copy logical subrange of this iterator.
    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice
    where
        Bounds: RangeBounds<usize>;

    fn len(&self) -> Result<usize, Error>;

    fn is_empty(&self) -> Result<bool, Error> {
        Ok(self.len()? == 0)
    }

    #[doc(hidden)]
    fn lower_read(self) -> Self::Read;
}

/// Internal bridge from a logical row stream to GPU algorithm dispatch.
///
/// This is deliberately separate from [`MIter`]: being a readable view does
/// not require proving that every kernel algorithm can consume the view.
#[doc(hidden)]
#[cfg(any())]
pub trait MIterKernel<R: Runtime>: MIter<R> {
    type Read: private::AnyRead<R, Item = Self::Item>;

    fn lower_read(self) -> Self::Read;

    #[doc(hidden)]
    fn count_if_with<Pred>(self, exec: &Executor<R>, pred: Pred) -> Result<u32, Error>
    where
        Pred: crate::op::PredicateOp<Self::Item>;

    #[doc(hidden)]
    fn all_of_with<Pred>(self, exec: &Executor<R>, pred: Pred) -> Result<bool, Error>
    where
        Pred: crate::op::PredicateOp<Self::Item>;

    #[doc(hidden)]
    fn any_of_with<Pred>(self, exec: &Executor<R>, pred: Pred) -> Result<bool, Error>
    where
        Pred: crate::op::PredicateOp<Self::Item>;

    #[doc(hidden)]
    fn none_of_with<Pred>(self, exec: &Executor<R>, pred: Pred) -> Result<bool, Error>
    where
        Pred: crate::op::PredicateOp<Self::Item>;

    #[doc(hidden)]
    fn find_if_with<Pred>(self, exec: &Executor<R>, pred: Pred) -> Result<Option<u32>, Error>
    where
        Pred: crate::op::PredicateOp<Self::Item>;

    #[doc(hidden)]
    fn is_partitioned_with<Pred>(self, exec: &Executor<R>, pred: Pred) -> Result<bool, Error>
    where
        Pred: crate::op::PredicateOp<Self::Item>;

    #[doc(hidden)]
    fn reduce_with<Op>(
        self,
        exec: &Executor<R>,
        init: Self::Item,
        op: Op,
    ) -> Result<Self::Item, Error>
    where
        Self: MIterStorageKernel<R>,
        Op: crate::op::ReductionOp<Self::Item>;

    #[doc(hidden)]
    fn adjacent_find_with<Equal>(
        self,
        exec: &Executor<R>,
        equal: Equal,
    ) -> Result<Option<u32>, Error>
    where
        Self: MIterStorageKernel<R>,
        Equal: crate::op::BinaryPredicateOp<Self::Item>;

    #[doc(hidden)]
    fn is_sorted_until_with<Less>(self, exec: &Executor<R>, less: Less) -> Result<u32, Error>
    where
        Self: MIterStorageKernel<R>,
        Less: crate::op::BinaryPredicateOp<Self::Item>;

    #[doc(hidden)]
    fn is_sorted_with<Less>(self, exec: &Executor<R>, less: Less) -> Result<bool, Error>
    where
        Self: MIterStorageKernel<R>,
        Less: crate::op::BinaryPredicateOp<Self::Item>;

    #[doc(hidden)]
    fn min_element_with<Less>(self, exec: &Executor<R>, less: Less) -> Result<Option<u32>, Error>
    where
        Self: MIterStorageKernel<R>,
        Less: crate::op::BinaryPredicateOp<Self::Item>;

    #[doc(hidden)]
    fn max_element_with<Less>(self, exec: &Executor<R>, less: Less) -> Result<Option<u32>, Error>
    where
        Self: MIterStorageKernel<R>,
        Less: crate::op::BinaryPredicateOp<Self::Item>;

    #[doc(hidden)]
    fn minmax_element_with<Less>(
        self,
        exec: &Executor<R>,
        less: Less,
    ) -> Result<Option<(u32, u32)>, Error>
    where
        Self: MIterStorageKernel<R>,
        Less: crate::op::BinaryPredicateOp<Self::Item>;

    #[doc(hidden)]
    fn inclusive_scan_with<Output, Op>(
        self,
        exec: &Executor<R>,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Self: MIterStorageKernel<R>,
        Output: MIterMut<R>,
        Output::Item: crate::WritableFrom<Self::Item>,
        Op: crate::op::ReductionOp<Self::Item>;

    #[doc(hidden)]
    fn adjacent_difference_with<Output, Op>(
        self,
        exec: &Executor<R>,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Self: MIterStorageKernel<R>,
        Output: MIterMut<R>,
        Output::Item: crate::WritableFrom<Self::Item>,
        Op: crate::op::ReductionOp<Self::Item>;

    #[doc(hidden)]
    fn sort_with<Output, Less>(
        self,
        exec: &Executor<R>,
        less: Less,
        output: Output,
    ) -> Result<(), Error>
    where
        Self: MIterStorageKernel<R>,
        Output: MIterMut<R>,
        Output::Item: crate::WritableFrom<Self::Item>,
        Less: crate::op::BinaryPredicateOp<Self::Item>;

    #[doc(hidden)]
    fn unique_with<Output, Equal>(
        self,
        exec: &Executor<R>,
        equal: Equal,
        output: Output,
    ) -> Result<u32, Error>
    where
        Self: MIterStorageKernel<R>,
        Output: MIterMut<R>,
        Output::Item: crate::WritableFrom<Self::Item>,
        Equal: crate::op::BinaryPredicateOp<Self::Item>;

    #[doc(hidden)]
    fn materialize_u32(self, exec: &Executor<R>) -> Result<crate::DeviceVec<R, u32>, Error>
    where
        Self: MIterKernel<R, Item = u32>,
    {
        let len = self.len()? as usize;
        let output = exec.alloc::<u32>(len);
        crate::api::algorithm::transform::transform_into(
            exec,
            self,
            crate::op::Identity,
            output.slice_mut(..),
        )?;
        Ok(output)
    }

    #[doc(hidden)]
    fn select_with_flags<Output>(
        self,
        exec: &Executor<R>,
        flags: crate::Column<u32>,
        invert: bool,
        output: Output,
    ) -> Result<u32, Error>
    where
        Self: MIterStorageKernel<R>,
        Output: MIterMut<R>,
        Output::Item: crate::WritableFrom<Self::Item>;

    #[doc(hidden)]
    fn partition_with<Output, Pred>(
        self,
        exec: &Executor<R>,
        pred: Pred,
        output: Output,
    ) -> Result<u32, Error>
    where
        Self: MIterStorageKernel<R>,
        Output: MIterMut<R>,
        Output::Item: crate::WritableFrom<Self::Item>,
        Pred: crate::op::PredicateOp<Self::Item>;

    #[doc(hidden)]
    fn indexed_with<Output>(
        self,
        exec: &Executor<R>,
        indices: crate::Column<u32>,
        flags: Option<crate::Column<u32>>,
        scatter: bool,
        output: Output,
    ) -> Result<(), Error>
    where
        Self: MIterStorageKernel<R>,
        Output: MIterMut<R>,
        Output::Item: crate::WritableFrom<Self::Item>;

    #[doc(hidden)]
    fn reverse_with<Output>(self, exec: &Executor<R>, output: Output) -> Result<(), Error>
    where
        Self: MIterStorageKernel<R>,
        Output: MIterMut<R>,
        Output::Item: crate::WritableFrom<Self::Item>,
    {
        let len = self.len()? as usize;
        let indices = <crate::ReverseCounting as MIterKernel<R>>::materialize_u32(
            crate::ReverseCounting::new(len),
            exec,
        )?;
        self.indexed_with(exec, indices.column(), None, false, output)
    }

    #[doc(hidden)]
    fn transform_where_with<Output, Op, OutItem>(
        self,
        exec: &Executor<R>,
        op: Op,
        flags: crate::Column<u32>,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R>,
        Op: crate::op::UnaryOp<Self::Item, Output = OutItem>,
        OutItem: crate::StorageLayout,
        Output::Item: crate::WritableFrom<OutItem>;

    #[doc(hidden)]
    fn exclusive_scan_with<Output, Op>(
        self,
        exec: &Executor<R>,
        init: Self::Item,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Self: MIterStorageKernel<R>,
        Output: MIterMut<R>,
        Output::Item: crate::WritableFrom<Self::Item>,
        Op: crate::op::ReductionOp<Self::Item>;

    #[doc(hidden)]
    fn equal_with<Right, Equal>(
        self,
        exec: &Executor<R>,
        right: Right,
        equal: Equal,
    ) -> Result<bool, Error>
    where
        Self: MIterStorageKernel<R>,
        Right: MIterStorageKernel<R, Item = Self::Item>,
        Equal: crate::op::BinaryPredicateOp<Self::Item>;

    #[doc(hidden)]
    fn mismatch_with<Right, Equal>(
        self,
        exec: &Executor<R>,
        right: Right,
        equal: Equal,
    ) -> Result<Option<u32>, Error>
    where
        Self: MIterStorageKernel<R>,
        Right: MIterStorageKernel<R, Item = Self::Item>,
        Equal: crate::op::BinaryPredicateOp<Self::Item>;

    #[doc(hidden)]
    fn lexicographical_with<Right, Less>(
        self,
        exec: &Executor<R>,
        right: Right,
        less: Less,
    ) -> Result<bool, Error>
    where
        Self: MIterStorageKernel<R>,
        Right: MIterStorageKernel<R, Item = Self::Item>,
        Less: crate::op::BinaryPredicateOp<Self::Item>;

    #[doc(hidden)]
    fn find_first_of_with<Needles, Equal>(
        self,
        exec: &Executor<R>,
        needles: Needles,
        equal: Equal,
    ) -> Result<Option<u32>, Error>
    where
        Self: MIterStorageKernel<R>,
        Needles: MIterStorageKernel<R, Item = Self::Item>,
        Equal: crate::op::BinaryPredicateOp<Self::Item>;

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
        Self: MIterStorageKernel<R>,
        Values: MIterStorageKernel<R, Item = Self::Item>,
        Less: crate::op::BinaryPredicateOp<Self::Item>,
        Output: MIterMut<R>,
        Output::Item: crate::WritableFrom<u32>;

    #[doc(hidden)]
    fn merge_with<Right, Less, Output>(
        self,
        exec: &Executor<R>,
        right: Right,
        less: Less,
        output: Output,
    ) -> Result<(), Error>
    where
        Self: MIterStorageKernel<R>,
        Right: MIterStorageKernel<R, Item = Self::Item>,
        Less: crate::op::BinaryPredicateOp<Self::Item>,
        Output: MIterMut<R>,
        Output::Item: crate::WritableFrom<Self::Item>;

    #[doc(hidden)]
    fn set_with<Right, Less, Output>(
        self,
        exec: &Executor<R>,
        right: Right,
        less: Less,
        output: Output,
        mode: u8,
    ) -> Result<u32, Error>
    where
        Self: MIterStorageKernel<R>,
        Right: MIterStorageKernel<R, Item = Self::Item>,
        Less: crate::op::BinaryPredicateOp<Self::Item>,
        Output: MIterMut<R>,
        Output::Item: crate::WritableFrom<Self::Item>;

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
        Self: MIterStorageKernel<R>,
        Values: MIterStorageKernel<R>,
        Less: crate::op::BinaryPredicateOp<Self::Item>,
        KeyOutput: MIterMut<R>,
        KeyOutput::Item: crate::WritableFrom<Self::Item>,
        ValueOutput: MIterMut<R>,
        ValueOutput::Item: crate::WritableFrom<Values::Item>;

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
        Self: MIterStorageKernel<R>,
        Values: MIterStorageKernel<R>,
        Equal: crate::op::BinaryPredicateOp<Self::Item>,
        Op: crate::op::ReductionOp<Values::Item>,
        Output: MIterMut<R>,
        Output::Item: crate::WritableFrom<Values::Item>;

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
    ) -> Result<u32, Error>
    where
        Self: MIterStorageKernel<R>,
        Values: MIterStorageKernel<R>,
        Equal: crate::op::BinaryPredicateOp<Self::Item>,
        Op: crate::op::ReductionOp<Values::Item>,
        KeyOutput: MIterMut<R>,
        KeyOutput::Item: crate::WritableFrom<Self::Item>,
        ValueOutput: MIterMut<R>,
        ValueOutput::Item: crate::WritableFrom<Values::Item>;

    #[doc(hidden)]
    fn unique_by_key_with<Values, Equal, KeyOutput, ValueOutput>(
        self,
        exec: &Executor<R>,
        values: Values,
        equal: Equal,
        key_output: KeyOutput,
        value_output: ValueOutput,
    ) -> Result<u32, Error>
    where
        Self: MIterStorageKernel<R>,
        Values: MIterStorageKernel<R>,
        Equal: crate::op::BinaryPredicateOp<Self::Item>,
        KeyOutput: MIterMut<R>,
        KeyOutput::Item: crate::WritableFrom<Self::Item>,
        ValueOutput: MIterMut<R>,
        ValueOutput::Item: crate::WritableFrom<Values::Item>;

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
        Self: MIterStorageKernel<R>,
        LeftValues: MIterStorageKernel<R>,
        RightKeys: MIterStorageKernel<R, Item = Self::Item>,
        RightValues: MIterStorageKernel<R, Item = LeftValues::Item>,
        Less: crate::op::BinaryPredicateOp<Self::Item>,
        KeyOutput: MIterMut<R>,
        KeyOutput::Item: crate::WritableFrom<Self::Item>,
        ValueOutput: MIterMut<R>,
        ValueOutput::Item: crate::WritableFrom<LeftValues::Item>;
}

/// Internal capability for algorithms that operate on canonical item storage.
#[doc(hidden)]
#[cfg(any())]
pub trait MIterStorageKernel<R: Runtime>:
    MIterKernel<R, Leaves: private::KernelItem<R, Self::Item>>
{
}

#[doc(hidden)]
#[cfg(any())]
impl<R, Input> MIterStorageKernel<R> for Input
where
    R: Runtime,
    Input: MIterKernel<R>,
    Input::Leaves: private::KernelItem<R, Input::Item>,
{
}

/// Public preallocated output stream.
///
/// Device mutable slices and [`Zip`] trees of mutable slices implement this trait.
#[allow(private_bounds)]
pub trait MIterMut<R: Runtime>: Sized {
    /// Semantic value stored by one output row.
    ///
    /// A preallocated destination needs a physical storage layout, but it does
    /// not need to be allocatable as new owned storage.
    type Item: MItem<R> + crate::core::allocation::ScratchStorage<R>;

    #[doc(hidden)]
    type Slice;

    #[doc(hidden)]
    type SliceMut;

    /// Returns a read-only zero-copy subrange of this output.
    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice
    where
        Bounds: RangeBounds<usize>;

    /// Returns a mutable zero-copy subrange of this output.
    fn slice_mut<Bounds>(&self, range: Bounds) -> Self::SliceMut
    where
        Bounds: RangeBounds<usize>;

    fn len(&self) -> Result<usize, Error>;

    fn is_empty(&self) -> Result<bool, Error> {
        Ok(self.len()? == 0)
    }

    /// Internal fixed-ABI output tree. This is a structural lowering contract
    /// and contains no algorithm operations.
    #[doc(hidden)]
    type OutputSlots: crate::output::PaddedOutputSlots<
            Leaves = <Self::Item as crate::StorageLayout>::StorageLeaves,
        > + crate::output::OutputSlotEnvironment<
            StorageArity = <Self::Item as crate::StorageLayout>::StorageArity,
        >;

    #[doc(hidden)]
    type LoweredOutput: crate::output::OutputExpression<Item = Self::Item>
        + crate::output::LowerOutputExpression<Slots = Self::OutputSlots>
        + crate::output::StageOutput<R, crate::read::Env0>
        + private::KernelOutput<R>
        + crate::selection::FillOutput<R>
        + crate::output::SliceOutput;

    #[doc(hidden)]
    fn lower_output(self) -> Self::LoweredOutput;

    #[doc(hidden)]
    fn fill_with(self, exec: &Executor<R>, value: Self::Item) -> Result<(), Error> {
        crate::selection::fill(exec, value, self.lower_output())
    }
}

#[doc(hidden)]
impl<R, Input, Item> MIter<R> for Input
where
    R: Runtime,
    Input: Clone
        + private::IterLength
        + private::KernelInput<R, Item = Item>
        + crate::read::SliceExpression
        + crate::read::LowerReadExpression,
    Item: CubeType + Send + Sync + 'static,
{
    type Item = Item;
    type Read = Input;
    type Slice = crate::read::Slice<R, Input>;

    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice
    where
        Bounds: RangeBounds<usize>,
    {
        let len = private::IterLength::logical_len(self)
            .expect("cannot slice an iterator with an invalid length");
        let (start, len) = resolve_iter_range(len, range);
        crate::read::Slice::new(self.slice_expression(start, len))
    }

    fn len(&self) -> Result<usize, Error> {
        private::IterLength::logical_len(self)
    }

    fn lower_read(self) -> Self::Read {
        self
    }
}

#[doc(hidden)]
#[cfg(any())]
impl<R, Input> MIterKernel<R> for Input
where
    R: Runtime,
    Input: MIter<R>
        + Clone
        + crate::read::ReadExpression<Item = <Input as MIter<R>>::Item>
        + crate::read::LowerReadExpression
        + crate::reduce::StageRead<R, crate::read::Env0>,
    Input::Slots: crate::read::PaddedReadSlots,
    <Input as MIter<R>>::Leaves: private::KernelItem<R, <Input as MIter<R>>::Item>,
    <<Input as MIter<R>>::Item as crate::CanonicalAlloc<R>>::CanonicalStorage:
        crate::CanonicalStorage<
                R,
                WriteSlots = <<Input as MIter<R>>::Leaves as private::KernelItem<
                    R,
                    <Input as MIter<R>>::Item,
                >>::WriteSlots,
            >,
    crate::read::KernelReadSlots<Input::Slots>: private::KernelRead<
            R,
            crate::read::FixedRead<Input>,
            <Input as MIter<R>>::Item,
            <<Input as MIter<R>>::Item as crate::StorageLayout>::StorageLeaves,
        >,
{
    type Read = private::Read<
        crate::read::FixedRead<Input>,
        crate::read::KernelReadSlots<Input::Slots>,
        <Input as MIter<R>>::Item,
        <<Input as MIter<R>>::Item as crate::StorageLayout>::StorageLeaves,
    >;

    fn lower_read(self) -> Self::Read {
        private::Read::new(crate::read::FixedRead::new(self))
    }

    fn count_if_with<Pred>(self, exec: &Executor<R>, pred: Pred) -> Result<u32, Error>
    where
        Pred: crate::op::PredicateOp<Self::Item>,
    {
        <crate::read::KernelReadSlots<Input::Slots> as private::KernelRead<
            R,
            crate::read::FixedRead<Input>,
        >>::count_if(crate::read::FixedRead::new(self), exec, pred)
    }

    fn all_of_with<Pred>(self, exec: &Executor<R>, pred: Pred) -> Result<bool, Error>
    where
        Pred: crate::op::PredicateOp<Self::Item>,
    {
        <crate::read::KernelReadSlots<Input::Slots> as private::KernelRead<
            R,
            crate::read::FixedRead<Input>,
        >>::all_of(crate::read::FixedRead::new(self), exec, pred)
    }

    fn any_of_with<Pred>(self, exec: &Executor<R>, pred: Pred) -> Result<bool, Error>
    where
        Pred: crate::op::PredicateOp<Self::Item>,
    {
        <crate::read::KernelReadSlots<Input::Slots> as private::KernelRead<
            R,
            crate::read::FixedRead<Input>,
        >>::any_of(crate::read::FixedRead::new(self), exec, pred)
    }

    fn none_of_with<Pred>(self, exec: &Executor<R>, pred: Pred) -> Result<bool, Error>
    where
        Pred: crate::op::PredicateOp<Self::Item>,
    {
        <crate::read::KernelReadSlots<Input::Slots> as private::KernelRead<
            R,
            crate::read::FixedRead<Input>,
        >>::none_of(crate::read::FixedRead::new(self), exec, pred)
    }

    fn find_if_with<Pred>(self, exec: &Executor<R>, pred: Pred) -> Result<Option<u32>, Error>
    where
        Pred: crate::op::PredicateOp<Self::Item>,
    {
        <crate::read::KernelReadSlots<Input::Slots> as private::KernelRead<
            R,
            crate::read::FixedRead<Input>,
        >>::find_if(crate::read::FixedRead::new(self), exec, pred)
    }

    fn is_partitioned_with<Pred>(self, exec: &Executor<R>, pred: Pred) -> Result<bool, Error>
    where
        Pred: crate::op::PredicateOp<Self::Item>,
    {
        <crate::read::KernelReadSlots<Input::Slots> as private::KernelRead<
            R,
            crate::read::FixedRead<Input>,
        >>::is_partitioned(crate::read::FixedRead::new(self), exec, pred)
    }

    fn reduce_with<Op>(
        self,
        exec: &Executor<R>,
        init: Self::Item,
        op: Op,
    ) -> Result<Self::Item, Error>
    where
        Self: MIterStorageKernel<R>,
        Op: crate::op::ReductionOp<Self::Item>,
    {
        <crate::read::KernelReadSlots<Input::Slots> as private::KernelRead<
            R,
            crate::read::FixedRead<Input>,
        >>::reduce(crate::read::FixedRead::new(self), exec, init, op)
    }

    fn adjacent_find_with<Equal>(
        self,
        exec: &Executor<R>,
        equal: Equal,
    ) -> Result<Option<u32>, Error>
    where
        Self: MIterStorageKernel<R>,
        Equal: crate::op::BinaryPredicateOp<Self::Item>,
    {
        <crate::read::KernelReadSlots<Input::Slots> as private::KernelRead<
            R,
            crate::read::FixedRead<Input>,
        >>::adjacent_find(crate::read::FixedRead::new(self), exec, equal)
    }

    fn is_sorted_until_with<Less>(self, exec: &Executor<R>, less: Less) -> Result<u32, Error>
    where
        Self: MIterStorageKernel<R>,
        Less: crate::op::BinaryPredicateOp<Self::Item>,
    {
        <crate::read::KernelReadSlots<Input::Slots> as private::KernelRead<
            R,
            crate::read::FixedRead<Input>,
        >>::is_sorted_until(crate::read::FixedRead::new(self), exec, less)
    }

    fn is_sorted_with<Less>(self, exec: &Executor<R>, less: Less) -> Result<bool, Error>
    where
        Self: MIterStorageKernel<R>,
        Less: crate::op::BinaryPredicateOp<Self::Item>,
    {
        <crate::read::KernelReadSlots<Input::Slots> as private::KernelRead<
            R,
            crate::read::FixedRead<Input>,
        >>::is_sorted(crate::read::FixedRead::new(self), exec, less)
    }

    fn min_element_with<Less>(self, exec: &Executor<R>, less: Less) -> Result<Option<u32>, Error>
    where
        Self: MIterStorageKernel<R>,
        Less: crate::op::BinaryPredicateOp<Self::Item>,
    {
        <crate::read::KernelReadSlots<Input::Slots> as private::KernelRead<
            R,
            crate::read::FixedRead<Input>,
        >>::min_element(crate::read::FixedRead::new(self), exec, less)
    }

    fn max_element_with<Less>(self, exec: &Executor<R>, less: Less) -> Result<Option<u32>, Error>
    where
        Self: MIterStorageKernel<R>,
        Less: crate::op::BinaryPredicateOp<Self::Item>,
    {
        <crate::read::KernelReadSlots<Input::Slots> as private::KernelRead<
            R,
            crate::read::FixedRead<Input>,
        >>::max_element(crate::read::FixedRead::new(self), exec, less)
    }

    fn minmax_element_with<Less>(
        self,
        exec: &Executor<R>,
        less: Less,
    ) -> Result<Option<(u32, u32)>, Error>
    where
        Self: MIterStorageKernel<R>,
        Less: crate::op::BinaryPredicateOp<Self::Item>,
    {
        <crate::read::KernelReadSlots<Input::Slots> as private::KernelRead<
            R,
            crate::read::FixedRead<Input>,
        >>::minmax_element(crate::read::FixedRead::new(self), exec, less)
    }

    fn inclusive_scan_with<Output, Op>(
        self,
        exec: &Executor<R>,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Self: MIterStorageKernel<R>,
        Output: MIterMut<R>,
        Output::Item: crate::WritableFrom<Self::Item>,
        Op: crate::op::ReductionOp<Self::Item>,
    {
        <crate::read::KernelReadSlots<Input::Slots> as private::KernelRead<
            R,
            crate::read::FixedRead<Input>,
        >>::inclusive_scan(
            crate::read::FixedRead::new(self),
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
        Self: MIterStorageKernel<R>,
        Output: MIterMut<R>,
        Output::Item: crate::WritableFrom<Self::Item>,
        Op: crate::op::ReductionOp<Self::Item>,
    {
        <crate::read::KernelReadSlots<Input::Slots> as private::KernelRead<
            R,
            crate::read::FixedRead<Input>,
        >>::adjacent_difference(
            crate::read::FixedRead::new(self),
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
        Self: MIterStorageKernel<R>,
        Output: MIterMut<R>,
        Output::Item: crate::WritableFrom<Self::Item>,
        Less: crate::op::BinaryPredicateOp<Self::Item>,
    {
        <crate::read::KernelReadSlots<Input::Slots> as private::KernelRead<
            R,
            crate::read::FixedRead<Input>,
        >>::sort(
            crate::read::FixedRead::new(self),
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
    ) -> Result<u32, Error>
    where
        Self: MIterStorageKernel<R>,
        Output: MIterMut<R>,
        Output::Item: crate::WritableFrom<Self::Item>,
        Equal: crate::op::BinaryPredicateOp<Self::Item>,
    {
        <crate::read::KernelReadSlots<Input::Slots> as private::KernelRead<
            R,
            crate::read::FixedRead<Input>,
        >>::unique(
            crate::read::FixedRead::new(self),
            exec,
            equal,
            output.lower_write_from::<Self::Item>(),
        )
    }

    fn select_with_flags<Output>(
        self,
        exec: &Executor<R>,
        flags: crate::Column<u32>,
        invert: bool,
        output: Output,
    ) -> Result<u32, Error>
    where
        Self: MIterStorageKernel<R>,
        Output: MIterMut<R>,
        Output::Item: crate::WritableFrom<Self::Item>,
    {
        <crate::read::KernelReadSlots<Input::Slots> as private::KernelRead<
            R,
            crate::read::FixedRead<Input>,
        >>::select(
            crate::read::FixedRead::new(self),
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
    ) -> Result<u32, Error>
    where
        Self: MIterStorageKernel<R>,
        Output: MIterMut<R>,
        Output::Item: crate::WritableFrom<Self::Item>,
        Pred: crate::op::PredicateOp<Self::Item>,
    {
        <crate::read::KernelReadSlots<Input::Slots> as private::KernelRead<
            R,
            crate::read::FixedRead<Input>,
        >>::partition(
            crate::read::FixedRead::new(self),
            exec,
            pred,
            output.lower_write_from::<Self::Item>(),
        )
    }

    fn indexed_with<Output>(
        self,
        exec: &Executor<R>,
        indices: crate::Column<u32>,
        flags: Option<crate::Column<u32>>,
        scatter: bool,
        output: Output,
    ) -> Result<(), Error>
    where
        Self: MIterStorageKernel<R>,
        Output: MIterMut<R>,
        Output::Item: crate::WritableFrom<Self::Item>,
    {
        <crate::read::KernelReadSlots<Input::Slots> as private::KernelRead<
            R,
            crate::read::FixedRead<Input>,
        >>::indexed(
            crate::read::FixedRead::new(self),
            exec,
            indices,
            flags,
            scatter,
            output.lower_write_from::<Self::Item>(),
        )
    }

    fn transform_where_with<Output, Op, OutItem>(
        self,
        exec: &Executor<R>,
        op: Op,
        flags: crate::Column<u32>,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R>,
        Op: crate::op::UnaryOp<<Input as MIter<R>>::Item, Output = OutItem>,
        OutItem: crate::StorageLayout,
        Output::Item: crate::WritableFrom<OutItem>,
    {
        <crate::read::KernelReadSlots<Input::Slots> as private::KernelRead<
            R,
            crate::read::FixedRead<Input>,
        >>::transform_where(
            crate::read::FixedRead::new(self),
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
        Self: MIterStorageKernel<R>,
        Output: MIterMut<R>,
        Output::Item: crate::WritableFrom<Self::Item>,
        Op: crate::op::ReductionOp<Self::Item>,
    {
        <crate::read::KernelReadSlots<Input::Slots> as private::KernelRead<
            R,
            crate::read::FixedRead<Input>,
        >>::exclusive_scan(
            crate::read::FixedRead::new(self),
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
        Self: MIterStorageKernel<R>,
        Right: MIterStorageKernel<R, Item = Self::Item>,
        Equal: crate::op::BinaryPredicateOp<Self::Item>,
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
    ) -> Result<Option<u32>, Error>
    where
        Self: MIterStorageKernel<R>,
        Right: MIterStorageKernel<R, Item = Self::Item>,
        Equal: crate::op::BinaryPredicateOp<Self::Item>,
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
        Self: MIterStorageKernel<R>,
        Right: MIterStorageKernel<R, Item = Self::Item>,
        Less: crate::op::BinaryPredicateOp<Self::Item>,
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
    ) -> Result<Option<u32>, Error>
    where
        Self: MIterStorageKernel<R>,
        Needles: MIterStorageKernel<R, Item = Self::Item>,
        Equal: crate::op::BinaryPredicateOp<Self::Item>,
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
        Self: MIterStorageKernel<R>,
        Values: MIterStorageKernel<R, Item = Self::Item>,
        Less: crate::op::BinaryPredicateOp<Self::Item>,
        Output: MIterMut<R>,
        Output::Item: crate::WritableFrom<u32>,
    {
        let source = <Self::Read as private::AnyRead<R>>::normalize(self.lower_read(), exec)?;
        let values = <Values::Read as private::AnyRead<R>>::normalize(values.lower_read(), exec)?;
        let bounds = <Self::Leaves as private::KernelItem<R, Self::Item>>::bounds(
            exec, &source, &values, less, upper,
        )?;
        crate::api::algorithm::transform::transform_into(
            exec,
            bounds.column(),
            crate::op::Identity,
            output,
        )
    }

    fn merge_with<Right, Less, Output>(
        self,
        exec: &Executor<R>,
        right: Right,
        less: Less,
        output: Output,
    ) -> Result<(), Error>
    where
        Self: MIterStorageKernel<R>,
        Right: MIterStorageKernel<R, Item = Self::Item>,
        Less: crate::op::BinaryPredicateOp<Self::Item>,
        Output: MIterMut<R>,
        Output::Item: crate::WritableFrom<Self::Item>,
    {
        let left = <Self::Read as private::AnyRead<R>>::normalize(self.lower_read(), exec)?;
        let right = <Right::Read as private::AnyRead<R>>::normalize(right.lower_read(), exec)?;
        let control = <Self::Leaves as private::KernelItem<R, Self::Item>>::merge_control(
            exec, &left, &right, less,
        )?;
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
    ) -> Result<u32, Error>
    where
        Self: MIterStorageKernel<R>,
        Right: MIterStorageKernel<R, Item = Self::Item>,
        Less: crate::op::BinaryPredicateOp<Self::Item>,
        Output: MIterMut<R>,
        Output::Item: crate::WritableFrom<Self::Item>,
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
        Self: MIterStorageKernel<R>,
        Values: MIterStorageKernel<R>,
        Less: crate::op::BinaryPredicateOp<Self::Item>,
        KeyOutput: MIterMut<R>,
        KeyOutput::Item: crate::WritableFrom<Self::Item>,
        ValueOutput: MIterMut<R>,
        ValueOutput::Item: crate::WritableFrom<Values::Item>,
    {
        let keys = <Self::Read as private::AnyRead<R>>::normalize(self.lower_read(), exec)?;
        let key_len = crate::CanonicalStorage::len(&keys)?;
        let value_len = values.len()? as usize;
        if key_len != value_len {
            return Err(Error::LengthMismatch {
                left: key_len,
                right: value_len,
            });
        }
        let ordering =
            <Self::Leaves as private::KernelItem<R, Self::Item>>::sort_ordering(exec, keys, less)?;
        private::KernelWrite::materialize_storage(
            key_output.lower_write_from::<Self::Item>(),
            exec,
            &ordering.sorted_keys,
        )?;
        values.indexed_with(
            exec,
            ordering.control.permutation().column(),
            None,
            false,
            value_output,
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
        Self: MIterStorageKernel<R>,
        Values: MIterStorageKernel<R>,
        Equal: crate::op::BinaryPredicateOp<Self::Item>,
        Op: crate::op::ReductionOp<Values::Item>,
        Output: MIterMut<R>,
        Output::Item: crate::WritableFrom<Values::Item>,
    {
        let keys = <Self::Read as private::AnyRead<R>>::normalize(self.lower_read(), exec)?;
        let values = <Values::Read as private::AnyRead<R>>::normalize(values.lower_read(), exec)?;
        let heads = <Self::Leaves as private::KernelItem<R, Self::Item>>::segment_heads(
            exec, &keys, equal,
        )?;
        let mode = u8::from(init.is_some());
        let scanned = <Values::Leaves as private::KernelItem<R, Values::Item>>::segmented(
            exec, &values, &heads, init, op, mode,
        )?;
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
    ) -> Result<u32, Error>
    where
        Self: MIterStorageKernel<R>,
        Values: MIterStorageKernel<R>,
        Equal: crate::op::BinaryPredicateOp<Self::Item>,
        Op: crate::op::ReductionOp<Values::Item>,
        KeyOutput: MIterMut<R>,
        KeyOutput::Item: crate::WritableFrom<Self::Item>,
        ValueOutput: MIterMut<R>,
        ValueOutput::Item: crate::WritableFrom<Values::Item>,
    {
        let keys = <Self::Read as private::AnyRead<R>>::normalize(self.lower_read(), exec)?;
        let values = <Values::Read as private::AnyRead<R>>::normalize(values.lower_read(), exec)?;
        let key_len = crate::CanonicalStorage::len(&keys)?;
        let value_len = crate::CanonicalStorage::len(&values)?;
        if key_len != value_len {
            return Err(Error::LengthMismatch {
                left: key_len,
                right: value_len,
            });
        }
        let heads = <Self::Leaves as private::KernelItem<R, Self::Item>>::segment_heads(
            exec, &keys, equal,
        )?;
        let reduced = <Values::Leaves as private::KernelItem<R, Values::Item>>::segmented(
            exec,
            &values,
            &heads,
            Some(init),
            op,
            2,
        )?;
        let head_control = crate::core::selection::SelectionControl::from_flags(exec, heads)?;
        let tail_control = crate::core::by_key::tail_control_from_heads(exec, &head_control)?;
        let key_count = private::KernelWrite::select_storage_control(
            key_output.lower_write_from::<Self::Item>(),
            exec,
            &keys,
            &head_control,
        )?;
        let value_count = private::KernelWrite::select_storage_control(
            value_output.lower_write_from::<Values::Item>(),
            exec,
            &reduced,
            &tail_control,
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
    ) -> Result<u32, Error>
    where
        Self: MIterStorageKernel<R>,
        Values: MIterStorageKernel<R>,
        Equal: crate::op::BinaryPredicateOp<Self::Item>,
        KeyOutput: MIterMut<R>,
        KeyOutput::Item: crate::WritableFrom<Self::Item>,
        ValueOutput: MIterMut<R>,
        ValueOutput::Item: crate::WritableFrom<Values::Item>,
    {
        let keys = <Self::Read as private::AnyRead<R>>::normalize(self.lower_read(), exec)?;
        let values = <Values::Read as private::AnyRead<R>>::normalize(values.lower_read(), exec)?;
        let key_len = crate::CanonicalStorage::len(&keys)?;
        let value_len = crate::CanonicalStorage::len(&values)?;
        if key_len != value_len {
            return Err(Error::LengthMismatch {
                left: key_len,
                right: value_len,
            });
        }
        let heads = <Self::Leaves as private::KernelItem<R, Self::Item>>::segment_heads(
            exec, &keys, equal,
        )?;
        let control = crate::core::selection::SelectionControl::from_flags(exec, heads)?;
        let key_count = private::KernelWrite::select_storage_control(
            key_output.lower_write_from::<Self::Item>(),
            exec,
            &keys,
            &control,
        )?;
        let value_count = private::KernelWrite::select_storage_control(
            value_output.lower_write_from::<Values::Item>(),
            exec,
            &values,
            &control,
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
        Self: MIterStorageKernel<R>,
        LeftValues: MIterStorageKernel<R>,
        RightKeys: MIterStorageKernel<R, Item = Self::Item>,
        RightValues: MIterStorageKernel<R, Item = LeftValues::Item>,
        Less: crate::op::BinaryPredicateOp<Self::Item>,
        KeyOutput: MIterMut<R>,
        KeyOutput::Item: crate::WritableFrom<Self::Item>,
        ValueOutput: MIterMut<R>,
        ValueOutput::Item: crate::WritableFrom<LeftValues::Item>,
    {
        let left_keys = <Self::Read as private::AnyRead<R>>::normalize(self.lower_read(), exec)?;
        let left_values =
            <LeftValues::Read as private::AnyRead<R>>::normalize(left_values.lower_read(), exec)?;
        let right_keys =
            <RightKeys::Read as private::AnyRead<R>>::normalize(right_keys.lower_read(), exec)?;
        let right_values =
            <RightValues::Read as private::AnyRead<R>>::normalize(right_values.lower_read(), exec)?;
        let left_key_len = crate::CanonicalStorage::len(&left_keys)?;
        let left_value_len = crate::CanonicalStorage::len(&left_values)?;
        let right_key_len = crate::CanonicalStorage::len(&right_keys)?;
        let right_value_len = crate::CanonicalStorage::len(&right_values)?;
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
        let control = <Self::Leaves as private::KernelItem<R, Self::Item>>::merge_control(
            exec,
            &left_keys,
            &right_keys,
            less,
        )?;
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
        + crate::output::SliceOutput,
    Output::Item: MItem<R> + crate::core::allocation::ScratchStorage<R>,
    Output::Slots: crate::output::PaddedOutputSlots<
            Leaves = <Output::Item as crate::StorageLayout>::StorageLeaves,
        > + crate::output::OutputSlotEnvironment<
            StorageArity = <Output::Item as crate::StorageLayout>::StorageArity,
        >,
{
    type Item = <Output as crate::output::OutputExpression>::Item;
    type Slice = crate::read::Slice<R, Output::Read>;
    type SliceMut = crate::output::Slice<R, Output>;
    type OutputSlots = Output::Slots;
    type LoweredOutput = Output;
    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice
    where
        Bounds: RangeBounds<usize>,
    {
        let len = crate::output::OutputExpression::logical_len(self)
            .expect("cannot slice an output with an invalid length");
        let (start, len) = resolve_iter_range(len, range);
        crate::read::Slice::new(self.slice_read(start..start + len))
    }

    fn slice_mut<Bounds>(&self, range: Bounds) -> Self::SliceMut
    where
        Bounds: RangeBounds<usize>,
    {
        let len = crate::output::OutputExpression::logical_len(self)
            .expect("cannot slice an output with an invalid length");
        let (start, len) = resolve_iter_range(len, range);
        crate::output::Slice::new(self.slice_output(start..start + len))
    }

    fn len(&self) -> Result<usize, Error> {
        crate::output::OutputExpression::logical_len(self)
    }

    fn lower_output(self) -> Self::LoweredOutput {
        self
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
/// use massively::{Executor, op, vector::transform, zip2};
///
/// struct AddPair;
///
/// #[cubecl::cube]
/// impl op::UnaryOp<(u32, u32)> for AddPair {
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
/// let input = zip2(left.slice(..), right.slice(..));
/// let output = transform(&exec, input, AddPair).unwrap();
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

/// Combines eight iterators into an iterator whose item is [`crate::Tuple8`].
#[allow(clippy::too_many_arguments)]
pub fn zip8<A, B, C, D, E, F, G, H>(
    a: A,
    b: B,
    c: C,
    d: D,
    e: E,
    f: F,
    g: G,
    h: H,
) -> Zip<Zip<Zip<Zip<Zip<Zip<Zip<A, B>, C>, D>, E>, F>, G>, H> {
    Zip::new(zip7(a, b, c, d, e, f, g), h)
}

/// Combines nine iterators into an iterator whose item is [`crate::Tuple9`].
#[allow(clippy::too_many_arguments)]
pub fn zip9<A, B, C, D, E, F, G, H, I>(
    a: A,
    b: B,
    c: C,
    d: D,
    e: E,
    f: F,
    g: G,
    h: H,
    i: I,
) -> Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<A, B>, C>, D>, E>, F>, G>, H>, I> {
    Zip::new(zip8(a, b, c, d, e, f, g, h), i)
}

/// Combines ten iterators into an iterator whose item is [`crate::Tuple10`].
#[allow(clippy::too_many_arguments)]
pub fn zip10<A, B, C, D, E, F, G, H, I, J>(
    a: A,
    b: B,
    c: C,
    d: D,
    e: E,
    f: F,
    g: G,
    h: H,
    i: I,
    j: J,
) -> Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<A, B>, C>, D>, E>, F>, G>, H>, I>, J> {
    Zip::new(zip9(a, b, c, d, e, f, g, h, i), j)
}

/// Combines eleven iterators into an iterator whose item is [`crate::Tuple11`].
#[allow(clippy::too_many_arguments)]
pub fn zip11<A, B, C, D, E, F, G, H, I, J, K>(
    a: A,
    b: B,
    c: C,
    d: D,
    e: E,
    f: F,
    g: G,
    h: H,
    i: I,
    j: J,
    k: K,
) -> Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<A, B>, C>, D>, E>, F>, G>, H>, I>, J>, K> {
    Zip::new(zip10(a, b, c, d, e, f, g, h, i, j), k)
}

/// Combines twelve iterators into an iterator whose item is [`crate::Tuple12`].
#[allow(clippy::too_many_arguments)]
pub fn zip12<A, B, C, D, E, F, G, H, I, J, K, L>(
    a: A,
    b: B,
    c: C,
    d: D,
    e: E,
    f: F,
    g: G,
    h: H,
    i: I,
    j: J,
    k: K,
    l: L,
) -> Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<A, B>, C>, D>, E>, F>, G>, H>, I>, J>, K>, L> {
    Zip::new(zip11(a, b, c, d, e, f, g, h, i, j, k), l)
}

/// Decomposes the [`Zip`] produced by [`zip2`] into its two children.
///
/// This consumes the `Zip` and moves out its children without copying their data. In particular,
/// this can split a two-column [`MVec`] into its two owning [`crate::DeviceVec`] columns.
///
/// # Examples
///
/// ```
/// use massively::{unzip2, zip2};
///
/// let zipped = zip2(String::from("left"), vec![1, 2, 3]);
/// let (left, right) = unzip2(zipped);
///
/// assert_eq!(left, "left");
/// assert_eq!(right, vec![1, 2, 3]);
/// ```
pub fn unzip2<A, B>(zipped: Zip<A, B>) -> (A, B) {
    zipped.into_parts()
}

/// Decomposes the [`Zip`] produced by [`zip3`] into its three children.
pub fn unzip3<A, B, C>(zipped: Zip<Zip<A, B>, C>) -> (A, B, C) {
    let (ab, c) = zipped.into_parts();
    let (a, b) = unzip2(ab);
    (a, b, c)
}

/// Decomposes the [`Zip`] produced by [`zip4`] into its four children.
pub fn unzip4<A, B, C, D>(zipped: Zip<Zip<Zip<A, B>, C>, D>) -> (A, B, C, D) {
    let (abc, d) = zipped.into_parts();
    let (a, b, c) = unzip3(abc);
    (a, b, c, d)
}

/// Decomposes the [`Zip`] produced by [`zip5`] into its five children.
pub fn unzip5<A, B, C, D, E>(zipped: Zip<Zip<Zip<Zip<A, B>, C>, D>, E>) -> (A, B, C, D, E) {
    let (abcd, e) = zipped.into_parts();
    let (a, b, c, d) = unzip4(abcd);
    (a, b, c, d, e)
}

/// Decomposes the [`Zip`] produced by [`zip6`] into its six children.
pub fn unzip6<A, B, C, D, E, F>(
    zipped: Zip<Zip<Zip<Zip<Zip<A, B>, C>, D>, E>, F>,
) -> (A, B, C, D, E, F) {
    let (abcde, f) = zipped.into_parts();
    let (a, b, c, d, e) = unzip5(abcde);
    (a, b, c, d, e, f)
}

/// Decomposes the [`Zip`] produced by [`zip7`] into its seven children.
pub fn unzip7<A, B, C, D, E, F, G>(
    zipped: Zip<Zip<Zip<Zip<Zip<Zip<A, B>, C>, D>, E>, F>, G>,
) -> (A, B, C, D, E, F, G) {
    let (abcdef, g) = zipped.into_parts();
    let (a, b, c, d, e, f) = unzip6(abcdef);
    (a, b, c, d, e, f, g)
}

/// Decomposes the [`Zip`] produced by [`zip8`] into its eight children.
pub fn unzip8<A, B, C, D, E, F, G, H>(
    zipped: Zip<Zip<Zip<Zip<Zip<Zip<Zip<A, B>, C>, D>, E>, F>, G>, H>,
) -> (A, B, C, D, E, F, G, H) {
    let (abcdefg, h) = zipped.into_parts();
    let (a, b, c, d, e, f, g) = unzip7(abcdefg);
    (a, b, c, d, e, f, g, h)
}

/// Decomposes the [`Zip`] produced by [`zip9`] into its nine children.
pub fn unzip9<A, B, C, D, E, F, G, H, I>(
    zipped: Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<A, B>, C>, D>, E>, F>, G>, H>, I>,
) -> (A, B, C, D, E, F, G, H, I) {
    let (abcdefgh, i) = zipped.into_parts();
    let (a, b, c, d, e, f, g, h) = unzip8(abcdefgh);
    (a, b, c, d, e, f, g, h, i)
}

/// Decomposes the [`Zip`] produced by [`zip10`] into its ten children.
pub fn unzip10<A, B, C, D, E, F, G, H, I, J>(
    zipped: Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<A, B>, C>, D>, E>, F>, G>, H>, I>, J>,
) -> (A, B, C, D, E, F, G, H, I, J) {
    let (abcdefghi, j) = zipped.into_parts();
    let (a, b, c, d, e, f, g, h, i) = unzip9(abcdefghi);
    (a, b, c, d, e, f, g, h, i, j)
}

/// Decomposes the [`Zip`] produced by [`zip11`] into its eleven children.
pub fn unzip11<A, B, C, D, E, F, G, H, I, J, K>(
    zipped: Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<A, B>, C>, D>, E>, F>, G>, H>, I>, J>, K>,
) -> (A, B, C, D, E, F, G, H, I, J, K) {
    let (abcdefghij, k) = zipped.into_parts();
    let (a, b, c, d, e, f, g, h, i, j) = unzip10(abcdefghij);
    (a, b, c, d, e, f, g, h, i, j, k)
}

/// Decomposes the [`Zip`] produced by [`zip12`] into its twelve children.
pub fn unzip12<A, B, C, D, E, F, G, H, I, J, K, L>(
    zipped: Zip<
        Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<Zip<A, B>, C>, D>, E>, F>, G>, H>, I>, J>, K>,
        L,
    >,
) -> (A, B, C, D, E, F, G, H, I, J, K, L) {
    let (abcdefghijk, l) = zipped.into_parts();
    let (a, b, c, d, e, f, g, h, i, j, k) = unzip11(abcdefghijk);
    (a, b, c, d, e, f, g, h, i, j, k, l)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{A1, A2, A13, ReadExpression, StorageLayout, read::FixedRead};
    use cubecl::prelude::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    use static_assertions::{assert_impl_all, assert_not_impl_any, assert_type_eq_all};

    #[derive(CubeType, Clone, Copy)]
    struct ReadOnlyValue {
        value: u32,
    }

    #[derive(CubeType, Clone, Copy)]
    struct StoredOnlyValue {
        value: u32,
    }

    struct StoredOnlyLayout;

    #[cubecl::cube]
    impl crate::storage::Decompose<StoredOnlyValue> for StoredOnlyLayout {
        type Leaves = crate::storage::Last<u32>;

        fn decompose(item: StoredOnlyValue) -> Self::Leaves {
            crate::storage::Last::new(item.value)
        }
    }

    #[cubecl::cube]
    impl crate::storage::Recompose<StoredOnlyValue> for StoredOnlyLayout {
        type Leaves = crate::storage::Last<u32>;

        fn recompose(leaves: Self::Leaves) -> StoredOnlyValue {
            StoredOnlyValue {
                value: leaves.value,
            }
        }
    }

    impl StorageLayout for StoredOnlyValue {
        type StorageArity = crate::S1;
        type StorageLeaves = crate::storage::Last<u32>;
        type DeviceLayout = StoredOnlyLayout;

        fn into_storage_leaves(self) -> Self::StorageLeaves {
            crate::storage::Last::<u32> { value: self.value }
        }

        fn from_storage_leaves(leaves: Self::StorageLeaves) -> Self {
            Self {
                value: leaves.value,
            }
        }
    }

    struct MakeReadOnly;

    struct MakeReadOnlyFromU64;

    struct ReadOnlyEqual;

    struct ReadOnlyLess;

    #[cubecl::cube]
    impl crate::op::UnaryOp<u32> for MakeReadOnly {
        type Output = ReadOnlyValue;

        fn apply(value: u32) -> ReadOnlyValue {
            ReadOnlyValue { value }
        }
    }

    #[cubecl::cube]
    impl crate::op::UnaryOp<u64> for MakeReadOnlyFromU64 {
        type Output = ReadOnlyValue;

        fn apply(value: u64) -> ReadOnlyValue {
            ReadOnlyValue {
                value: value as u32,
            }
        }
    }

    #[cubecl::cube]
    impl crate::op::BinaryPredicateOp<ReadOnlyValue> for ReadOnlyEqual {
        fn apply(lhs: ReadOnlyValue, rhs: ReadOnlyValue) -> bool {
            lhs.value == rhs.value
        }
    }

    #[cubecl::cube]
    impl crate::op::BinaryPredicateOp<ReadOnlyValue> for ReadOnlyLess {
        fn apply(lhs: ReadOnlyValue, rhs: ReadOnlyValue) -> bool {
            lhs.value < rhs.value
        }
    }

    #[cubecl::cube]
    impl WritableFrom<ReadOnlyValue> for u32 {
        fn write_from(source: ReadOnlyValue) -> u32 {
            source.value
        }

        fn read_source(output: u32) -> ReadOnlyValue {
            ReadOnlyValue { value: output }
        }
    }

    #[cubecl::cube]
    impl WritableFrom<ReadOnlyValue> for StoredOnlyValue {
        fn write_from(source: ReadOnlyValue) -> StoredOnlyValue {
            StoredOnlyValue {
                value: source.value,
            }
        }

        fn read_source(output: StoredOnlyValue) -> ReadOnlyValue {
            ReadOnlyValue {
                value: output.value,
            }
        }
    }

    type ReadOnlyIter = crate::read::Transform<crate::Counting, MakeReadOnly>;
    type ExactRead = <ReadOnlyIter as MIter<WgpuRuntime>>::Read;
    type TwoColumnRead = <Zip<crate::Counting, crate::Counting> as MIter<WgpuRuntime>>::Read;
    type Fixed = FixedRead<ExactRead>;

    #[test]
    fn readable_item_does_not_require_a_storage_layout() {
        assert_not_impl_any!(ReadOnlyValue: StorageLayout);
        assert_impl_all!(ReadOnlyIter: MIter<WgpuRuntime>);
    }

    #[test]
    fn logical_lowering_retains_exact_arity_until_fixed_adapter() {
        assert_type_eq_all!(<ExactRead as ReadExpression>::ReadArity, A1);
        assert_type_eq_all!(<TwoColumnRead as ReadExpression>::ReadArity, A2);
        assert_type_eq_all!(<Fixed as ReadExpression>::ReadArity, A13);
    }

    #[test]
    fn non_storage_value_can_cross_an_explicit_write_boundary() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let input = crate::read::Transform::new(crate::Counting::new(7, 3), MakeReadOnly);
        let output = exec.alloc::<u32>(3);

        crate::api::algorithm::transform::transform_into(
            &exec,
            input,
            crate::op::Identity,
            output.slice_mut(..),
        )
        .unwrap();

        assert_eq!(exec.to_host(&output).unwrap(), vec![7, 8, 9]);
    }

    #[test]
    fn preallocated_output_does_not_require_allocation_capability() {
        type Output = crate::output::ReassociatedOutput<
            crate::DeviceSliceMut<u32>,
            StoredOnlyValue,
            crate::read::Env1<u32>,
        >;

        assert_impl_all!(StoredOnlyValue: MItem<WgpuRuntime>);
        assert_not_impl_any!(StoredOnlyValue: ToCanonical<WgpuRuntime>);
        assert_impl_all!(Output: MIterMut<WgpuRuntime>);

        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let input = crate::read::Transform::new(crate::Counting::new(4, 3), MakeReadOnly);
        let backing = exec.alloc::<u32>(3);
        let output =
            crate::output::ReassociatedOutput::<_, StoredOnlyValue, crate::read::Env1<u32>>::new(
                backing.slice_mut(..),
            );

        crate::api::algorithm::transform::transform_into(&exec, input, crate::op::Identity, output)
            .unwrap();

        assert_eq!(exec.to_host(&backing).unwrap(), vec![4, 5, 6]);
    }

    #[test]
    fn non_storage_keys_support_comparison_without_materialization() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let keys = crate::read::Transform::new(crate::Counting::new(7, 3), MakeReadOnly);

        assert!(crate::vector::is_sorted(&exec, keys, ReadOnlyLess).unwrap());
    }

    #[test]
    fn non_storage_keys_can_build_a_sort_permutation() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let backing = exec.to_device(&[9_u32, 7, 8]);
        let keys = crate::read::Transform::new(backing.column(), MakeReadOnly);

        let permutation = crate::ordering::sort_control_with(
            &exec,
            lower_fixed::<WgpuRuntime, _>(keys),
            ReadOnlyLess,
        )
        .unwrap();

        assert_eq!(exec.to_host(&permutation).unwrap(), vec![1, 2, 0]);
    }

    #[test]
    fn value_only_by_key_algorithms_accept_read_only_keys() {
        assert_not_impl_any!(ReadOnlyValue: ToCanonical<WgpuRuntime>);

        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);

        let sort_key_storage = exec.to_device(&[3_u32, 1, 2]);
        let sort_keys = crate::read::Transform::new(sort_key_storage.column(), MakeReadOnly);
        let sort_values = exec.to_device(&[30_u32, 10, 20]);
        let sorted =
            crate::vector::sort_by_key(&exec, sort_keys, sort_values.slice(..), ReadOnlyLess)
                .unwrap();
        assert_eq!(exec.to_host(&sorted).unwrap(), vec![10, 20, 30]);

        let unique_key_storage = exec.to_device(&[1_u32, 1, 2, 2]);
        let unique_keys = crate::read::Transform::new(unique_key_storage.column(), MakeReadOnly);
        let unique_values = exec.to_device(&[10_u32, 11, 20, 21]);
        let unique = crate::vector::unique_by_key(
            &exec,
            unique_keys,
            unique_values.slice(..),
            ReadOnlyEqual,
        )
        .unwrap();
        assert_eq!(exec.to_host(&unique).unwrap(), vec![10, 20]);

        let left_key_storage = exec.to_device(&[1_u32, 3]);
        let right_key_storage = exec.to_device(&[2_u64, 4]);
        let left_keys = crate::read::Transform::new(left_key_storage.column(), MakeReadOnly);
        let right_keys =
            crate::read::Transform::new(right_key_storage.column(), MakeReadOnlyFromU64);
        let left_values = exec.to_device(&[10_u32, 30]);
        let right_values = exec.to_device(&[20_u32, 40]);
        let merged = crate::vector::merge_by_key(
            &exec,
            left_keys,
            left_values.slice(..),
            right_keys,
            right_values.slice(..),
            ReadOnlyLess,
        )
        .unwrap();
        assert_eq!(exec.to_host(&merged).unwrap(), vec![10, 20, 30, 40]);
    }

    #[test]
    fn two_input_comparison_accepts_independent_physical_slot_types() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let left = crate::read::Transform::new(crate::Counting::new(7, 3), MakeReadOnly);
        let right_values = exec.to_device(&[7_u64, 8, 9]);
        let right = crate::read::Transform::new(right_values.column(), MakeReadOnlyFromU64);

        assert!(crate::vector::equal(&exec, left, right, ReadOnlyEqual).unwrap());
    }
}
