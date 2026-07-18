use core::marker::PhantomData;
use cubecl::prelude::{CubeType, Runtime};
use std::ops::{Bound, RangeBounds};

use crate::core::iter::Zip;
use crate::{Error, Executor};
use crate::{
    output::{ReadOutput, SliceOutput},
    read::SliceExpression,
};

/// Owned device storage for one flat logical row type.
pub trait MStorage<R: Runtime>: Sized {
    type Item: CubeType + Send + Sync + 'static;

    /// Owned physical columns in the same flat order as [`Self::Item`].
    ///
    /// A scalar row returns one [`crate::DeviceVec`]. A tuple row returns a
    /// native tuple of device vectors, regardless of the internal storage tree.
    type Columns;

    #[doc(hidden)]
    type Slice<'a>: MIter<R, Item = Self::Item>
    where
        Self: 'a;

    #[doc(hidden)]
    type SliceMut<'a>: MIterMut<R, Item = Self::Item>
    where
        Self: 'a;

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

    /// Consumes this storage and returns its columns as a flat native tuple.
    fn into_columns(self) -> Self::Columns;

    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice<'_>
    where
        Bounds: RangeBounds<usize>;

    fn slice_mut<Bounds>(&self, range: Bounds) -> Self::SliceMut<'_>
    where
        Bounds: RangeBounds<usize>;
}

/// Sealed physical contract for algorithms that require scratch storage.
#[doc(hidden)]
pub(crate) trait ScratchAbi<R: Runtime>:
    KernelRow + crate::RowAlloc<R> + crate::core::allocation::ScratchStorage<R>
{
}

impl<R, Item> ScratchAbi<R> for Item
where
    R: Runtime,
    Item: KernelRow + crate::RowAlloc<R> + crate::core::allocation::ScratchStorage<R>,
{
}

/// Sealed storage-shape dispatch for algorithms that materialize before
/// sorting. Public iterator APIs remain independent of physical arity.
pub(crate) trait SortAbi<R: Runtime>: KernelRow + crate::RowAlloc<R> {
    fn sort_storage<Less>(
        exec: &Executor<R>,
        input: <Self as crate::RowAlloc<R>>::RowStorage,
        carry_indices: bool,
    ) -> Result<
        crate::ordering::sort::OrderingResult<R, <Self as crate::RowAlloc<R>>::RowStorage>,
        Error,
    >
    where
        Less: crate::op::BinaryPredicateOp<Self>;
}

impl<R, Item> SortAbi<R> for Item
where
    R: Runtime,
    Item: KernelRow + crate::RowAlloc<R>,
    <Item as crate::StorageLayout>::StorageLeaves: crate::ordering::sort::SortLeaves<R, Item>,
{
    fn sort_storage<Less>(
        exec: &Executor<R>,
        input: <Self as crate::RowAlloc<R>>::RowStorage,
        carry_indices: bool,
    ) -> Result<
        crate::ordering::sort::OrderingResult<R, <Self as crate::RowAlloc<R>>::RowStorage>,
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

/// Backend-only capabilities carried by every storable public row.
#[doc(hidden)]
pub(crate) trait Sealed<R: Runtime>: ScratchAbi<R> + SortAbi<R> {}

impl<R, Item> Sealed<R> for Item
where
    R: Runtime,
    Item: ScratchAbi<R> + SortAbi<R>,
{
}

/// A scalar or compound numeric key with a stable radix ordering.
///
/// Scalar leaves may be `u8`, `u16`, `u32`, `u64`, `i8`, `i16`, `i32`, `i64`,
/// `f32`, or `f64`, provided the runtime supports that scalar type. Integers use
/// their natural ascending numeric order. Floating-point leaves use the same
/// total order as [`f32::total_cmp`] and [`f64::total_cmp`]. Compound keys use
/// lexicographic leaf order, with the leftmost leaf as the primary key.
pub trait RadixKey<R: Runtime>: MItem<R> {
    #[doc(hidden)]
    fn radix_permutation(
        exec: &Executor<R>,
        keys: &MVec<R, Self>,
        len: usize,
    ) -> Result<crate::DeviceVec<R, u32>, Error>;
}

#[doc(hidden)]
#[allow(private_bounds)]
impl<R, Item> RadixKey<R> for Item
where
    R: Runtime,
    Item: MItem<R>,
    MVec<R, Item>: crate::radix::RadixStorage<R>,
{
    fn radix_permutation(
        exec: &Executor<R>,
        keys: &MVec<R, Self>,
        len: usize,
    ) -> Result<crate::DeviceVec<R, u32>, Error> {
        crate::radix::permutation(exec, keys, len)
    }
}

/// Owned device storage for a flat row type.
pub type MVec<R, Item> = <Item as MItem<R>>::Storage;

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
pub(crate) trait KernelRow:
    crate::StorageLayout<
        StorageLeaves: private::KernelValue<
            StorageArity = <Self as crate::StorageLayout>::StorageArity,
        > + private::KernelOutputLeaves,
    >
{
}

impl<Item> KernelRow for Item
where
    Item: crate::StorageLayout,
    Item::StorageLeaves:
        private::KernelValue<StorageArity = Item::StorageArity> + private::KernelOutputLeaves,
{
}

#[doc(hidden)]
#[allow(private_bounds)]
pub trait MItem<R: Runtime>: KernelRow + Sealed<R> {
    /// The owned SoA storage used for this flat row.
    #[doc(hidden)]
    type Storage: MStorage<R, Item = Self>;
}

#[doc(hidden)]
impl<R, Item> MItem<R> for Item
where
    R: Runtime,
    Item: KernelRow + crate::RowAlloc<R> + Sealed<R>,
    <Item as crate::RowAlloc<R>>::RowStorage: MStorage<R, Item = Item>,
{
    type Storage = <Item as crate::RowAlloc<R>>::RowStorage;
}

#[doc(hidden)]
pub(crate) trait MutableItem<R: Runtime>:
    KernelRow + crate::core::allocation::ScratchStorage<R>
{
}

impl<R, Item> MutableItem<R> for Item
where
    R: Runtime,
    Item: KernelRow + crate::core::allocation::ScratchStorage<R>,
{
}

/// Public read-only logical row stream.
///
/// Device slices, lazy expressions, and values returned by the `zipN` helpers
/// implement this trait. Every tuple item is a native flat tuple, independent
/// of how calls to [`zip2`] are grouped.
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
    type Read: private::KernelInput<R, Item = Self::Item>
        + private::IterLength
        + crate::read::SliceExpression;

    #[doc(hidden)]
    type Slice;

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

/// Public preallocated output stream.
///
/// Device mutable slices and values returned by the `zipN` helpers implement
/// this trait. Their logical item is always a native flat tuple.
#[allow(private_bounds)]
pub trait MIterMut<R: Runtime>: Sized {
    /// Semantic value stored by one output row.
    ///
    /// A preallocated destination needs a physical storage layout, but it does
    /// not need to be allocatable as new owned storage.
    type Item: MutableItem<R>;

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
impl<R, Output> MIterMut<R> for Output
where
    R: Runtime,
    Output: crate::output::OutputExpression
        + crate::output::LowerOutputExpression
        + crate::output::ReadOutput
        + crate::output::StageOutput<R, crate::read::Env0>
        + crate::selection::FillOutput<R>
        + crate::output::SliceOutput,
    Output::Item: crate::StorageLayout<
            StorageLeaves: private::KernelValue<
                StorageArity = <Output::Item as crate::StorageLayout>::StorageArity,
            > + private::KernelOutputLeaves,
        > + crate::core::allocation::ScratchStorage<R>,
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

/// Logical read view over opaque owned row storage.
#[doc(hidden)]
pub struct StorageSlice<'a, R, Storage> {
    storage: &'a Storage,
    start: usize,
    len: usize,
    _runtime: PhantomData<fn() -> R>,
}

impl<R, Storage> Copy for StorageSlice<'_, R, Storage> {}

impl<R, Storage> Clone for StorageSlice<'_, R, Storage> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, R, Storage> StorageSlice<'a, R, Storage> {
    pub(crate) const fn new(storage: &'a Storage, start: usize, len: usize) -> Self {
        Self {
            storage,
            start,
            len,
            _runtime: PhantomData,
        }
    }
}

impl<R, Storage> MIter<R> for StorageSlice<'_, R, Storage>
where
    R: Runtime,
    Storage: crate::RowStorage<R>,
    Storage::Read: private::KernelInput<R, Item = Storage::Item>
        + private::IterLength
        + crate::read::SliceExpression,
{
    type Item = Storage::Item;
    type Read = Storage::Read;
    type Slice = Self;

    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice
    where
        Bounds: RangeBounds<usize>,
    {
        let (start, len) = resolve_iter_range(self.len, range);
        Self::new(self.storage, self.start + start, len)
    }

    fn len(&self) -> Result<usize, Error> {
        Ok(self.len)
    }

    fn lower_read(self) -> Self::Read {
        crate::RowStorage::slice(self.storage, self.start..self.start + self.len)
    }
}

/// Logical mutable view over opaque owned row storage.
#[doc(hidden)]
pub struct StorageSliceMut<'a, R, Storage> {
    storage: &'a Storage,
    start: usize,
    len: usize,
    _runtime: PhantomData<fn() -> R>,
}

impl<R, Storage> Copy for StorageSliceMut<'_, R, Storage> {}

impl<R, Storage> Clone for StorageSliceMut<'_, R, Storage> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, R, Storage> StorageSliceMut<'a, R, Storage> {
    pub(crate) const fn new(storage: &'a Storage, start: usize, len: usize) -> Self {
        Self {
            storage,
            start,
            len,
            _runtime: PhantomData,
        }
    }
}

impl<'a, R, Storage> MIterMut<R> for StorageSliceMut<'a, R, Storage>
where
    R: Runtime,
    Storage: crate::RowStorage<R>,
    Storage::Item: MutableItem<R>,
    Storage::Read: private::KernelInput<R, Item = Storage::Item>
        + private::IterLength
        + crate::read::SliceExpression,
    Storage::Write:
        ReadOutput + private::KernelOutput<R> + crate::selection::FillOutput<R> + SliceOutput,
    Storage::WriteSlots: crate::output::OutputSlotEnvironment<
            StorageArity = <Storage::Item as crate::StorageLayout>::StorageArity,
        >,
{
    type Item = Storage::Item;
    type Slice = StorageSlice<'a, R, Storage>;
    type SliceMut = Self;
    type OutputSlots = Storage::WriteSlots;
    type LoweredOutput = Storage::Write;

    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice
    where
        Bounds: RangeBounds<usize>,
    {
        let (start, len) = resolve_iter_range(self.len, range);
        StorageSlice::new(self.storage, self.start + start, len)
    }

    fn slice_mut<Bounds>(&self, range: Bounds) -> Self::SliceMut
    where
        Bounds: RangeBounds<usize>,
    {
        let (start, len) = resolve_iter_range(self.len, range);
        Self::new(self.storage, self.start + start, len)
    }

    fn len(&self) -> Result<usize, Error> {
        Ok(self.len)
    }

    fn lower_output(self) -> Self::LoweredOutput {
        crate::RowStorage::slice_mut(self.storage, self.start..self.start + self.len)
    }
}

/// Logical composition of two iterator schemas.
///
/// Its operands are lowered into the private physical `Zip` tree only when an
/// algorithm consumes it. The wrapper itself carries no public tree-shape
/// semantics: its item is the flat concatenation of both operand items.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct Zipped<Left, Right>(Left, Right);

impl<Left, Right> Zipped<Left, Right> {
    pub(crate) const fn new(left: Left, right: Right) -> Self {
        Self(left, right)
    }

    pub(crate) fn into_parts(self) -> (Left, Right) {
        (self.0, self.1)
    }
}

impl<R, Left, Right> MIter<R> for Zipped<Left, Right>
where
    R: Runtime,
    Left: MIter<R>,
    Right: MIter<R>,
    Zip<Left::Read, Right::Read>:
        private::KernelInput<R> + private::IterLength + crate::read::SliceExpression,
{
    type Item = <Zip<Left::Read, Right::Read> as crate::ReadExpression>::Item;
    type Read = Zip<Left::Read, Right::Read>;
    type Slice = crate::read::Slice<R, Self::Read>;

    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice
    where
        Bounds: RangeBounds<usize>,
    {
        let input = self.clone().lower_read();
        let len =
            private::IterLength::logical_len(&input).expect("zip operands have equal lengths");
        let (start, count) = resolve_iter_range(len, range);
        crate::read::Slice::new(input.slice_expression(start, count))
    }

    fn len(&self) -> Result<usize, Error> {
        let left = self.0.len()?;
        let right = self.1.len()?;
        if left != right {
            return Err(Error::LengthMismatch { left, right });
        }
        Ok(left)
    }

    fn lower_read(self) -> Self::Read {
        Zip::new(self.0.lower_read(), self.1.lower_read())
    }
}

impl<R, Left, Right> MIterMut<R> for Zipped<Left, Right>
where
    R: Runtime,
    Left: MIterMut<R> + Clone,
    Right: MIterMut<R> + Clone,
    Zip<Left::LoweredOutput, Right::LoweredOutput>: crate::output::OutputExpression
        + crate::output::LowerOutputExpression
        + crate::output::ReadOutput
        + crate::output::StageOutput<R, crate::read::Env0>
        + crate::selection::FillOutput<R>
        + crate::output::SliceOutput
        + private::KernelOutput<R>,
    <Zip<Left::LoweredOutput, Right::LoweredOutput> as crate::output::OutputExpression>::Item:
        MutableItem<R>,
    <Zip<Left::LoweredOutput, Right::LoweredOutput> as crate::output::LowerOutputExpression>::Slots:
        crate::output::PaddedOutputSlots<
            Leaves = <<Zip<Left::LoweredOutput, Right::LoweredOutput> as crate::output::OutputExpression>::Item as crate::StorageLayout>::StorageLeaves,
        > + crate::output::OutputSlotEnvironment<
            StorageArity = <<Zip<Left::LoweredOutput, Right::LoweredOutput> as crate::output::OutputExpression>::Item as crate::StorageLayout>::StorageArity,
        >,
{
    type Item =
        <Zip<Left::LoweredOutput, Right::LoweredOutput> as crate::output::OutputExpression>::Item;
    type Slice = crate::read::Slice<
        R,
        <Zip<Left::LoweredOutput, Right::LoweredOutput> as crate::output::ReadOutput>::Read,
    >;
    type SliceMut = crate::output::Slice<R, Zip<Left::LoweredOutput, Right::LoweredOutput>>;
    type OutputSlots = <Zip<Left::LoweredOutput, Right::LoweredOutput> as crate::output::LowerOutputExpression>::Slots;
    type LoweredOutput = Zip<Left::LoweredOutput, Right::LoweredOutput>;

    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice
    where
        Bounds: RangeBounds<usize>,
    {
        let output = Zip::new(
            self.0.clone().lower_output(),
            self.1.clone().lower_output(),
        );
        let len = crate::output::OutputExpression::logical_len(&output)
            .expect("zip outputs have equal lengths");
        let (start, count) = resolve_iter_range(len, range);
        crate::read::Slice::new(output.slice_read(start..start + count))
    }

    fn slice_mut<Bounds>(&self, range: Bounds) -> Self::SliceMut
    where
        Bounds: RangeBounds<usize>,
    {
        let output = Zip::new(
            self.0.clone().lower_output(),
            self.1.clone().lower_output(),
        );
        let len = crate::output::OutputExpression::logical_len(&output)
            .expect("zip outputs have equal lengths");
        let (start, count) = resolve_iter_range(len, range);
        crate::output::Slice::new(output.slice_output(start..start + count))
    }

    fn len(&self) -> Result<usize, Error> {
        let left = MIterMut::len(&self.0)?;
        let right = MIterMut::len(&self.1)?;
        if left != right {
            return Err(Error::LengthMismatch { left, right });
        }
        Ok(left)
    }

    fn lower_output(self) -> Self::LoweredOutput {
        Zip::new(self.0.lower_output(), self.1.lower_output())
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
pub fn zip2<A, B>(a: A, b: B) -> Zipped<A, B> {
    Zipped::new(a, b)
}

/// Combines three iterators into an iterator whose item is `(A, B, C)`.
///
/// See [`zip2`] for a complete example. Grouping `zip2` calls differently does
/// not change the flat logical item type.
pub fn zip3<A, B, C>(a: A, b: B, c: C) -> Zipped<Zipped<A, B>, C> {
    Zipped::new(Zipped::new(a, b), c)
}

/// Combines four iterators into an iterator whose item is `(A, B, C, D)`.
///
/// See [`zip2`] for a complete example.
pub fn zip4<A, B, C, D>(a: A, b: B, c: C, d: D) -> Zipped<Zipped<Zipped<A, B>, C>, D> {
    Zipped::new(zip3(a, b, c), d)
}

/// Combines five iterators into a flat five-element tuple iterator.
///
/// See [`zip2`] for a complete example.
pub fn zip5<A, B, C, D, E>(
    a: A,
    b: B,
    c: C,
    d: D,
    e: E,
) -> Zipped<Zipped<Zipped<Zipped<A, B>, C>, D>, E> {
    Zipped::new(zip4(a, b, c, d), e)
}

/// Combines six iterators into a flat six-element tuple iterator.
///
/// See [`zip2`] for a complete example.
pub fn zip6<A, B, C, D, E, F>(
    a: A,
    b: B,
    c: C,
    d: D,
    e: E,
    f: F,
) -> Zipped<Zipped<Zipped<Zipped<Zipped<A, B>, C>, D>, E>, F> {
    Zipped::new(zip5(a, b, c, d, e), f)
}

/// Combines seven iterators into a flat seven-element tuple iterator.
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
) -> Zipped<Zipped<Zipped<Zipped<Zipped<Zipped<A, B>, C>, D>, E>, F>, G> {
    Zipped::new(zip6(a, b, c, d, e, f), g)
}

/// Combines eight iterators into a flat eight-element tuple iterator.
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
) -> Zipped<Zipped<Zipped<Zipped<Zipped<Zipped<Zipped<A, B>, C>, D>, E>, F>, G>, H> {
    Zipped::new(zip7(a, b, c, d, e, f, g), h)
}

/// Combines nine iterators into a flat nine-element tuple iterator.
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
) -> Zipped<Zipped<Zipped<Zipped<Zipped<Zipped<Zipped<Zipped<A, B>, C>, D>, E>, F>, G>, H>, I> {
    Zipped::new(zip8(a, b, c, d, e, f, g, h), i)
}

/// Combines ten iterators into a flat ten-element tuple iterator.
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
) -> Zipped<
    Zipped<Zipped<Zipped<Zipped<Zipped<Zipped<Zipped<Zipped<A, B>, C>, D>, E>, F>, G>, H>, I>,
    J,
> {
    Zipped::new(zip9(a, b, c, d, e, f, g, h, i), j)
}

/// Combines eleven iterators into a flat eleven-element tuple iterator.
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
) -> Zipped<
    Zipped<
        Zipped<Zipped<Zipped<Zipped<Zipped<Zipped<Zipped<Zipped<A, B>, C>, D>, E>, F>, G>, H>, I>,
        J,
    >,
    K,
> {
    Zipped::new(zip10(a, b, c, d, e, f, g, h, i, j), k)
}

/// Combines twelve iterators into a flat twelve-element tuple iterator.
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
) -> Zipped<
    Zipped<
        Zipped<
            Zipped<
                Zipped<Zipped<Zipped<Zipped<Zipped<Zipped<Zipped<A, B>, C>, D>, E>, F>, G>, H>,
                I,
            >,
            J,
        >,
        K,
    >,
    L,
> {
    Zipped::new(zip11(a, b, c, d, e, f, g, h, i, j, k), l)
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

    type ReadOnlyIter = crate::read::Transform<crate::Counting, MakeReadOnly>;
    type ExactRead = <ReadOnlyIter as MIter<WgpuRuntime>>::Read;
    type TwoColumnRead = <Zipped<crate::Counting, crate::Counting> as MIter<WgpuRuntime>>::Read;
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
        assert_not_impl_any!(ReadOnlyValue: MItem<WgpuRuntime>);

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
