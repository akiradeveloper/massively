//! Recursive output trees and physical storage staging.

use core::marker::PhantomData;
use cubecl::prelude::Runtime;
use std::ops::RangeBounds;

use crate::{
    Column, DeviceSliceMut, Error, MStorageElement, S1, S2, S3, S4, S5, S6, S7, StorageLayout, Zip,
    read::{Env0, Env1, Env2, Env3, Env4, Env5, Env6, Env7},
    storage::StorageArity,
};

/// Presents a physically compatible output sink as another semantic item
/// shape. The buffers are unchanged; only the item reconstructed at the write
/// boundary is different.
#[doc(hidden)]
pub struct ReassociatedOutput<Output, Source, Slots> {
    output: Output,
    _marker: PhantomData<fn() -> (Source, Slots)>,
}

impl<Output, Source, Slots> ReassociatedOutput<Output, Source, Slots> {
    pub fn new(output: Output) -> Self {
        Self {
            output,
            _marker: PhantomData,
        }
    }

    pub(crate) fn into_inner(self) -> Output {
        self.output
    }
}

/// A zero-copy mutable subrange tied to the runtime selected by `MIterMut`.
#[derive(Clone, Copy, Debug)]
pub struct Slice<R, Output> {
    output: Output,
    _runtime: PhantomData<fn() -> R>,
}

impl<R, Output> Slice<R, Output> {
    pub const fn new(output: Output) -> Self {
        Self {
            output,
            _runtime: PhantomData,
        }
    }

    pub(crate) fn into_inner(self) -> Output {
        self.output
    }
}

impl<R, Output> OutputExpression for Slice<R, Output>
where
    Output: OutputExpression,
{
    type Item = Output::Item;
    type StorageArity = Output::StorageArity;

    fn logical_len(&self) -> Result<usize, Error> {
        self.output.logical_len()
    }
}

impl<R, Output> SliceOutput for Slice<R, Output>
where
    Output: SliceOutput,
{
    fn slice_output<Range: RangeBounds<usize>>(&self, range: Range) -> Self {
        Self::new(self.output.slice_output(range))
    }
}

impl<R, Output> ReadOutput for Slice<R, Output>
where
    Output: ReadOutput,
{
    type Read = crate::read::Slice<R, Output::Read>;

    fn slice_read<Range: RangeBounds<usize>>(&self, range: Range) -> Self::Read {
        crate::read::Slice::new(self.output.slice_read(range))
    }
}

impl<Output, Source, Slots> OutputExpression for ReassociatedOutput<Output, Source, Slots>
where
    Output: OutputExpression,
    Source: StorageLayout,
    Output::Item: crate::WriteFrom<Source>,
    Slots: OutputSlotEnvironment<StorageArity = Source::StorageArity>,
{
    type Item = Source;
    type StorageArity = Source::StorageArity;

    fn logical_len(&self) -> Result<usize, Error> {
        self.output.logical_len()
    }
}

impl<Output, Source, Slots> SliceOutput for ReassociatedOutput<Output, Source, Slots>
where
    Output: SliceOutput,
    Source: StorageLayout,
    Output::Item: crate::WriteFrom<Source>,
    Slots: OutputSlotEnvironment<StorageArity = Source::StorageArity>,
{
    fn slice_output<Range: RangeBounds<usize>>(&self, range: Range) -> Self {
        Self::new(self.output.slice_output(range))
    }
}

/// A mutable output whose public shape is a binary tree.
pub trait OutputExpression {
    type Item: StorageLayout;
    type StorageArity: StorageArity;

    fn logical_len(&self) -> Result<usize, Error>;
}

/// Creates same-shaped subviews of a recursive output tree.
pub trait SliceOutput: OutputExpression + Sized {
    fn slice_output<Range: RangeBounds<usize>>(&self, range: Range) -> Self;
}

/// Creates a read-only view over a mutable output tree.
#[doc(hidden)]
pub trait ReadOutput: OutputExpression {
    type Read: crate::read::ReadExpression<Item = Self::Item>;

    fn slice_read<Range: RangeBounds<usize>>(&self, range: Range) -> Self::Read;
}

impl<T> SliceOutput for DeviceSliceMut<T>
where
    T: MStorageElement + StorageLayout<StorageArity = S1>,
{
    fn slice_output<Range: RangeBounds<usize>>(&self, range: Range) -> Self {
        self.slice_mut(range)
    }
}

impl<T> OutputExpression for DeviceSliceMut<T>
where
    T: MStorageElement + StorageLayout<StorageArity = S1>,
{
    type Item = T;
    type StorageArity = S1;

    fn logical_len(&self) -> Result<usize, Error> {
        Ok(self.len)
    }
}

impl<T> ReadOutput for DeviceSliceMut<T>
where
    T: MStorageElement + StorageLayout<StorageArity = S1>,
{
    type Read = Column<T>;

    fn slice_read<Range: RangeBounds<usize>>(&self, range: Range) -> Self::Read {
        self.slice(range)
    }
}

impl<Left, Right> OutputExpression for Zip<Left, Right>
where
    Left: OutputExpression,
    Right: OutputExpression,
    (Left::Item, Right::Item): StorageLayout,
{
    type Item = (Left::Item, Right::Item);
    type StorageArity = <Self::Item as StorageLayout>::StorageArity;

    fn logical_len(&self) -> Result<usize, Error> {
        let left = self.0.logical_len()?;
        let right = self.1.logical_len()?;
        if left != right {
            return Err(Error::LengthMismatch { left, right });
        }
        Ok(left)
    }
}

impl<Left, Right> SliceOutput for Zip<Left, Right>
where
    Left: SliceOutput,
    Right: SliceOutput,
    Zip<Left, Right>: OutputExpression,
{
    fn slice_output<Range: RangeBounds<usize>>(&self, range: Range) -> Self {
        let len = self
            .logical_len()
            .expect("output columns have equal lengths");
        let (start, count) = crate::read::resolve_slice_range(len, range);
        Zip::new(
            self.0.slice_output(start..start + count),
            self.1.slice_output(start..start + count),
        )
    }
}

impl<Left, Right> ReadOutput for Zip<Left, Right>
where
    Left: ReadOutput,
    Right: ReadOutput,
    Zip<Left, Right>: OutputExpression,
    Zip<Left::Read, Right::Read>:
        crate::read::ReadExpression<Item = <Zip<Left, Right> as OutputExpression>::Item>,
{
    type Read = Zip<Left::Read, Right::Read>;

    fn slice_read<Range: RangeBounds<usize>>(&self, range: Range) -> Self::Read {
        let len = self
            .logical_len()
            .expect("output columns have equal lengths");
        let (start, count) = crate::read::resolve_slice_range(len, range);
        Zip::new(
            self.0.slice_read(start..start + count),
            self.1.slice_read(start..start + count),
        )
    }
}

/// Binds output leaves left-to-right to an environment.
#[doc(hidden)]
pub trait BindOutputSlots<Env> {
    type NextEnv;
}

impl<Output, Source, Slots> BindOutputSlots<Env0> for ReassociatedOutput<Output, Source, Slots>
where
    Source: StorageLayout,
    Slots: OutputSlotEnvironment<StorageArity = Source::StorageArity>,
{
    type NextEnv = Slots;
}

macro_rules! impl_output_leaf_binding {
    (impl <$( $env_ty:ident ),*> $env:ty => $next:ty) => {
        impl<T, $( $env_ty ),*> BindOutputSlots<$env> for DeviceSliceMut<T>
        where
            T: MStorageElement,
        {
            type NextEnv = $next;
        }
    };
}

impl_output_leaf_binding!(impl <> Env0 => Env1<T>);
impl_output_leaf_binding!(impl <L0> Env1<L0> => Env2<L0, T>);
impl_output_leaf_binding!(impl <L0, L1> Env2<L0, L1> => Env3<L0, L1, T>);
impl_output_leaf_binding!(impl <L0, L1, L2> Env3<L0, L1, L2> => Env4<L0, L1, L2, T>);
impl_output_leaf_binding!(impl <L0, L1, L2, L3> Env4<L0, L1, L2, L3> => Env5<L0, L1, L2, L3, T>);
impl_output_leaf_binding!(impl <L0, L1, L2, L3, L4> Env5<L0, L1, L2, L3, L4> => Env6<L0, L1, L2, L3, L4, T>);
impl_output_leaf_binding!(impl <L0, L1, L2, L3, L4, L5> Env6<L0, L1, L2, L3, L4, L5> => Env7<L0, L1, L2, L3, L4, L5, T>);

impl<Left, Right, Env> BindOutputSlots<Env> for Zip<Left, Right>
where
    Left: BindOutputSlots<Env>,
    Right: BindOutputSlots<Left::NextEnv>,
{
    type NextEnv = Right::NextEnv;
}

impl<R, Output, Env> BindOutputSlots<Env> for Slice<R, Output>
where
    Output: BindOutputSlots<Env>,
{
    type NextEnv = Output::NextEnv;
}

#[doc(hidden)]
pub trait OutputSlotEnvironment {
    type StorageArity: StorageArity;
}

impl<L0> OutputSlotEnvironment for Env1<L0> {
    type StorageArity = S1;
}
impl<L0, L1> OutputSlotEnvironment for Env2<L0, L1> {
    type StorageArity = S2;
}
impl<L0, L1, L2> OutputSlotEnvironment for Env3<L0, L1, L2> {
    type StorageArity = S3;
}
impl<L0, L1, L2, L3> OutputSlotEnvironment for Env4<L0, L1, L2, L3> {
    type StorageArity = S4;
}
impl<L0, L1, L2, L3, L4> OutputSlotEnvironment for Env5<L0, L1, L2, L3, L4> {
    type StorageArity = S5;
}
impl<L0, L1, L2, L3, L4, L5> OutputSlotEnvironment for Env6<L0, L1, L2, L3, L4, L5> {
    type StorageArity = S6;
}
impl<L0, L1, L2, L3, L4, L5, L6> OutputSlotEnvironment for Env7<L0, L1, L2, L3, L4, L5, L6> {
    type StorageArity = S7;
}

/// Fully bound output tree.
#[doc(hidden)]
pub trait LowerOutputExpression: OutputExpression {
    type Slots: OutputSlotEnvironment<StorageArity = Self::StorageArity>;
}

impl<Output> LowerOutputExpression for Output
where
    Output: OutputExpression + BindOutputSlots<Env0>,
    Output::NextEnv: OutputSlotEnvironment<StorageArity = Output::StorageArity>,
{
    type Slots = Output::NextEnv;
}

#[doc(hidden)]
pub struct OutputBindings {
    pub(crate) slots: Vec<(cubecl::server::Handle, usize)>,
    pub(crate) offsets: Vec<u32>,
}

impl OutputBindings {
    pub(crate) fn new() -> Self {
        Self {
            slots: Vec::new(),
            offsets: Vec::new(),
        }
    }

    fn push(&mut self, handle: cubecl::server::Handle, len: usize, offset: u32) {
        self.slots.push((handle, len));
        self.offsets.push(offset);
    }
}

/// Stages output buffers using the same left-first traversal as storage layout.
#[doc(hidden)]
pub trait StageOutput<R: Runtime, Env>: BindOutputSlots<Env> {
    fn stage_output(&self, owner: u64, bindings: &mut OutputBindings) -> Result<(), Error>;
}

impl<R, Output, Source, Slots> StageOutput<R, Env0> for ReassociatedOutput<Output, Source, Slots>
where
    R: Runtime,
    Output: OutputExpression + StageOutput<R, Env0>,
    Source: StorageLayout,
    Output::Item: crate::WriteFrom<Source>,
    Slots: OutputSlotEnvironment<StorageArity = Source::StorageArity>,
{
    fn stage_output(&self, owner: u64, bindings: &mut OutputBindings) -> Result<(), Error> {
        self.output.stage_output(owner, bindings)
    }
}

macro_rules! impl_output_leaf_staging {
    (impl <$( $env_ty:ident ),*> $env:ty) => {
        impl<R, T, $( $env_ty ),*> StageOutput<R, $env> for DeviceSliceMut<T>
        where
            R: Runtime,
            T: MStorageElement,
            DeviceSliceMut<T>: BindOutputSlots<$env>,
        {
            fn stage_output(
                &self,
                owner: u64,
                bindings: &mut OutputBindings,
            ) -> Result<(), Error> {
                if self.owner != owner {
                    return Err(Error::ForeignExecutor);
                }
                bindings.push(self.handle.clone(), self.buffer_len, self.offset);
                Ok(())
            }
        }
    };
}

impl_output_leaf_staging!(impl <> Env0);
impl_output_leaf_staging!(impl <L0> Env1<L0>);
impl_output_leaf_staging!(impl <L0, L1> Env2<L0, L1>);
impl_output_leaf_staging!(impl <L0, L1, L2> Env3<L0, L1, L2>);
impl_output_leaf_staging!(impl <L0, L1, L2, L3> Env4<L0, L1, L2, L3>);
impl_output_leaf_staging!(impl <L0, L1, L2, L3, L4> Env5<L0, L1, L2, L3, L4>);
impl_output_leaf_staging!(impl <L0, L1, L2, L3, L4, L5> Env6<L0, L1, L2, L3, L4, L5>);

impl<R, Left, Right, Env> StageOutput<R, Env> for Zip<Left, Right>
where
    R: Runtime,
    Left: StageOutput<R, Env>,
    Right: StageOutput<R, Left::NextEnv>,
    Zip<Left, Right>: BindOutputSlots<Env>,
{
    fn stage_output(&self, owner: u64, bindings: &mut OutputBindings) -> Result<(), Error> {
        self.0.stage_output(owner, bindings)?;
        self.1.stage_output(owner, bindings)
    }
}

impl<R, Output, Env> StageOutput<R, Env> for Slice<R, Output>
where
    R: Runtime,
    Output: StageOutput<R, Env>,
    Slice<R, Output>: BindOutputSlots<Env>,
{
    fn stage_output(&self, owner: u64, bindings: &mut OutputBindings) -> Result<(), Error> {
        self.output.stage_output(owner, bindings)
    }
}
