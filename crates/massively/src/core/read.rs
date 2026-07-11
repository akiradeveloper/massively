//! Read-expression trees and deterministic slot binding.

use core::marker::PhantomData;
use cubecl::prelude::*;
use std::ops::{Bound, RangeBounds};

use crate::{
    Zip,
    arity::{A1, A2, A3, A4, A5, A6, A7, A8, AddArity, ReadArity},
    eval::{
        AdjacentExpr, AdjacentIndexedTransformExpr, Broadcast, Count, DeviceExpr, Direct, Eval1,
        Eval2, Eval3, Eval4, Eval5, Eval6, Eval7, Eval8, IndexedTransformExpr, PermuteExpr,
        ReassociateExpr, ReverseCount, Slot0, Slot1, Slot2, Slot3, Slot4, Slot5, Slot6, Slot7,
        TransformExpr, ZipExpr,
    },
    op::{IndexedBinaryOp, IndexedUnaryOp, UnaryOp},
    reduce::ReductionOp,
    storage::{StorageLayout, WriteFrom},
    value::MStorageElement,
};

/// Read-only contiguous device view. Cloning a view does not copy device data.
#[derive(Clone, Debug, Default)]
pub struct DeviceSlice<T> {
    pub(crate) handle: Option<cubecl::server::Handle>,
    pub(crate) len: usize,
    pub(crate) buffer_len: usize,
    pub(crate) offset: u32,
    pub(crate) owner: Option<u64>,
    _item: PhantomData<fn() -> T>,
}

impl<T> DeviceSlice<T> {
    pub const fn new() -> Self {
        Self {
            handle: None,
            len: 0,
            buffer_len: 0,
            offset: 0,
            owner: None,
            _item: PhantomData,
        }
    }

    pub(crate) fn from_handle(
        handle: cubecl::server::Handle,
        len: usize,
        offset: u32,
        owner: u64,
        buffer_len: usize,
    ) -> Self {
        Self {
            handle: Some(handle),
            len,
            buffer_len,
            offset,
            owner: Some(owner),
            _item: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns a read-only subview without copying device data.
    ///
    /// # Examples
    ///
    /// ```
    /// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    /// use massively::Executor;
    ///
    /// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    /// let values = exec.to_device(&[1_u32, 2, 3, 4, 5]);
    /// let nested = values.slice(1..5).slice(1..3);
    ///
    /// assert_eq!(exec.to_host(&nested).unwrap(), vec![3, 4]);
    /// ```
    pub fn slice<Range>(&self, range: Range) -> Self
    where
        Range: RangeBounds<usize>,
    {
        let (relative, len) = resolve_slice_range(self.len, range);
        Self {
            handle: self.handle.clone(),
            len,
            buffer_len: self.buffer_len,
            offset: self.offset + relative as u32,
            owner: self.owner,
            _item: PhantomData,
        }
    }
}

/// Internal name for a staged physical read leaf.
pub(crate) type Column<T> = DeviceSlice<T>;

pub(crate) fn resolve_slice_range<Range>(len: usize, range: Range) -> (usize, usize)
where
    Range: RangeBounds<usize>,
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
        "slice end ({end}) is out of bounds for length {len}"
    );
    (start, end - start)
}

/// A scalar leaf broadcast to every logical index.
#[derive(Clone, Copy, Debug)]
pub struct Constant<T> {
    pub value: T,
    pub len: usize,
}

impl<T> Constant<T> {
    pub const fn new(value: T, len: usize) -> Self {
        Self { value, len }
    }
}

/// A `u32` counting leaf.
#[derive(Clone, Copy, Debug)]
pub struct Counting {
    pub start: u32,
    pub len: usize,
}

/// A descending `u32` index leaf yielding `len - 1 - index`.
#[derive(Clone, Copy, Debug)]
pub struct ReverseCounting {
    pub start: usize,
    pub len: usize,
}

impl ReverseCounting {
    pub const fn new(len: usize) -> Self {
        Self {
            start: len.saturating_sub(1),
            len,
        }
    }
}

/// A lazy expression that reads its input in reverse order.
#[derive(Clone, Copy, Debug)]
pub struct Reverse<Values> {
    pub(crate) values: Values,
    pub(crate) offset: usize,
    pub(crate) len: Option<usize>,
}

impl<Values> Reverse<Values> {
    pub const fn new(values: Values) -> Self {
        Self {
            values,
            offset: 0,
            len: None,
        }
    }

    pub(crate) fn indices(&self, input_len: usize) -> ReverseCounting {
        ReverseCounting {
            start: input_len.saturating_sub(1).saturating_sub(self.offset),
            len: self.len.unwrap_or(input_len),
        }
    }
}

impl Counting {
    pub const fn new(start: u32, len: usize) -> Self {
        Self { start, len }
    }
}

/// A finite view of an otherwise unbounded lazy source.
#[derive(Clone, Copy, Debug)]
pub struct Taken<Source> {
    pub(crate) source: Source,
    pub(crate) offset: u32,
    pub(crate) len: u32,
}

impl<Source> Taken<Source> {
    pub(crate) const fn new(source: Source, len: u32) -> Self {
        Self {
            source,
            offset: 0,
            len,
        }
    }

    pub(crate) fn lower(&self) -> Source::Read
    where
        Source: TakenSource,
    {
        self.source.lower(self.offset, self.len)
    }
}

/// Internal lowering contract for unbounded lazy sources.
#[doc(hidden)]
pub trait TakenSource {
    type Read: ReadExpression;

    fn lower(&self, offset: u32, len: u32) -> Self::Read;
}

/// A lazy unary transform expression.
#[derive(Debug)]
pub struct Transform<Input, Op> {
    pub input: Input,
    _op: PhantomData<fn() -> Op>,
}

/// A lazy transform that also receives the logical input index.
#[doc(hidden)]
#[derive(Debug)]
pub struct IndexedTransform<Input, Op> {
    pub input: Input,
    _op: PhantomData<fn() -> Op>,
}

/// A lazy adjacent transform that also receives the current logical index.
#[doc(hidden)]
#[derive(Debug)]
pub struct AdjacentIndexedTransform<Input, Op> {
    pub input: Input,
    _op: PhantomData<fn() -> Op>,
}

/// A lazy adjacent reduction expression.
#[doc(hidden)]
#[derive(Debug)]
pub struct Adjacent<Input, Op> {
    pub input: Input,
    _op: PhantomData<fn() -> Op>,
}

impl<Input: Clone, Op> Clone for Transform<Input, Op> {
    fn clone(&self) -> Self {
        Self {
            input: self.input.clone(),
            _op: PhantomData,
        }
    }
}

impl<Input: Copy, Op> Copy for Transform<Input, Op> {}

impl<Input: Clone, Op> Clone for IndexedTransform<Input, Op> {
    fn clone(&self) -> Self {
        Self {
            input: self.input.clone(),
            _op: PhantomData,
        }
    }
}

impl<Input: Copy, Op> Copy for IndexedTransform<Input, Op> {}

impl<Input: Clone, Op> Clone for AdjacentIndexedTransform<Input, Op> {
    fn clone(&self) -> Self {
        Self {
            input: self.input.clone(),
            _op: PhantomData,
        }
    }
}

impl<Input: Copy, Op> Copy for AdjacentIndexedTransform<Input, Op> {}

impl<Input: Clone, Op> Clone for Adjacent<Input, Op> {
    fn clone(&self) -> Self {
        Self {
            input: self.input.clone(),
            _op: PhantomData,
        }
    }
}

impl<Input: Copy, Op> Copy for Adjacent<Input, Op> {}

impl<Input, Op> Adjacent<Input, Op> {
    pub fn new(input: Input, _op: Op) -> Self {
        Self {
            input,
            _op: PhantomData,
        }
    }
}

impl<Input, Op> Transform<Input, Op> {
    pub fn new(input: Input, _op: Op) -> Self {
        Self {
            input,
            _op: PhantomData,
        }
    }
}

impl<Input, Op> IndexedTransform<Input, Op> {
    pub fn new(input: Input, _op: Op) -> Self {
        Self {
            input,
            _op: PhantomData,
        }
    }
}

impl<Input, Op> AdjacentIndexedTransform<Input, Op> {
    pub fn new(input: Input, _op: Op) -> Self {
        Self {
            input,
            _op: PhantomData,
        }
    }
}

/// A lazy permutation expression evaluating `values[indices[index]]`.
#[derive(Clone, Copy, Debug)]
pub struct Permute<Values, Indices> {
    pub values: Values,
    pub indices: Indices,
}

impl<Values, Indices> Permute<Values, Indices> {
    pub const fn new(values: Values, indices: Indices) -> Self {
        Self { values, indices }
    }
}

#[doc(hidden)]
#[derive(Debug)]
pub struct Reassociate<Input, Output> {
    pub input: Input,
    _output: PhantomData<fn() -> Output>,
}

/// A zero-copy logical subrange of a read expression.
///
/// The contained expression has already been sliced according to its own
/// semantics. Keeping a distinct adapter type makes slicing part of iterator
/// composition without adding a staged read leaf or changing read arity.
#[derive(Clone, Copy, Debug)]
pub struct Slice<Runtime, Input> {
    pub input: Input,
    _runtime: PhantomData<fn() -> Runtime>,
}

impl<Runtime, Input> Slice<Runtime, Input> {
    pub const fn new(input: Input) -> Self {
        Self {
            input,
            _runtime: PhantomData,
        }
    }
}

impl<Input: Clone, Output> Clone for Reassociate<Input, Output> {
    fn clone(&self) -> Self {
        Self::new(self.input.clone())
    }
}

impl<Input: Copy, Output> Copy for Reassociate<Input, Output> {}

impl<Input, Output> Reassociate<Input, Output> {
    pub fn new(input: Input) -> Self {
        Self {
            input,
            _output: PhantomData,
        }
    }
}

/// Semantic item and physical read arity of a read-expression tree.
pub trait ReadExpression {
    type Item: StorageLayout;
    type ReadArity: ReadArity;
}

impl<T> ReadExpression for Column<T>
where
    T: MStorageElement,
{
    type Item = T;
    type ReadArity = A1;
}

impl<T> ReadExpression for Constant<T>
where
    T: MStorageElement,
{
    type Item = T;
    type ReadArity = A1;
}

impl ReadExpression for Counting {
    type Item = u32;
    type ReadArity = A1;
}

impl ReadExpression for ReverseCounting {
    type Item = u32;
    type ReadArity = A1;
}

impl<Source> ReadExpression for Taken<Source>
where
    Source: TakenSource,
{
    type Item = <Source::Read as ReadExpression>::Item;
    type ReadArity = <Source::Read as ReadExpression>::ReadArity;
}

impl<Left, Right> ReadExpression for Zip<Left, Right>
where
    Left: ReadExpression,
    Right: ReadExpression,
    Left::ReadArity: AddArity<Right::ReadArity>,
    (Left::Item, Right::Item): StorageLayout,
{
    type Item = (Left::Item, Right::Item);
    type ReadArity = <Left::ReadArity as AddArity<Right::ReadArity>>::Output;
}

impl<Input, Op> ReadExpression for Transform<Input, Op>
where
    Input: ReadExpression,
    Op: UnaryOp<Input::Item>,
{
    type Item = Op::Output;
    type ReadArity = Input::ReadArity;
}

impl<Input, Op> ReadExpression for IndexedTransform<Input, Op>
where
    Input: ReadExpression,
    Op: IndexedUnaryOp<Input::Item>,
{
    type Item = Op::Output;
    type ReadArity = Input::ReadArity;
}

impl<Input, Op> ReadExpression for AdjacentIndexedTransform<Input, Op>
where
    Input: ReadExpression,
    Op: IndexedBinaryOp<Input::Item>,
{
    type Item = Op::Output;
    type ReadArity = Input::ReadArity;
}

impl<Input, Op> ReadExpression for Adjacent<Input, Op>
where
    Input: ReadExpression,
    Input::Item: StorageLayout,
    Op: ReductionOp<Input::Item>,
{
    type Item = Input::Item;
    type ReadArity = Input::ReadArity;
}

impl<Values, Indices> ReadExpression for Permute<Values, Indices>
where
    Values: ReadExpression,
    Indices: ReadExpression<Item = u32>,
    Values::ReadArity: AddArity<Indices::ReadArity>,
{
    type Item = Values::Item;
    type ReadArity = <Values::ReadArity as AddArity<Indices::ReadArity>>::Output;
}

impl<Values> ReadExpression for Reverse<Values>
where
    Values: ReadExpression,
    Values::ReadArity: AddArity<A1>,
{
    type Item = Values::Item;
    type ReadArity = <Values::ReadArity as AddArity<A1>>::Output;
}

impl<Input, Output> ReadExpression for Reassociate<Input, Output>
where
    Input: ReadExpression,
    Input::Item: StorageLayout,
    Output: StorageLayout + WriteFrom<Input::Item> + 'static,
{
    type Item = Output;
    type ReadArity = Input::ReadArity;
}

impl<Runtime, Input> ReadExpression for Slice<Runtime, Input>
where
    Input: ReadExpression,
{
    type Item = Input::Item;
    type ReadArity = Input::ReadArity;
}

/// Produces a same-arity expression for a logical subrange.
///
/// This operation is defined per expression node so that index-sensitive lazy
/// expressions such as [`Permute`] slice their logical input, rather than
/// blindly shifting every physical leaf.
#[doc(hidden)]
pub trait SliceExpression: ReadExpression + Sized {
    fn slice_expression(&self, start: usize, len: usize) -> Self;
}

impl<T> SliceExpression for Column<T>
where
    T: MStorageElement,
{
    fn slice_expression(&self, start: usize, len: usize) -> Self {
        self.slice(start..start + len)
    }
}

impl<T> SliceExpression for Constant<T>
where
    T: MStorageElement,
{
    fn slice_expression(&self, _start: usize, len: usize) -> Self {
        Self::new(self.value, len)
    }
}

impl SliceExpression for Counting {
    fn slice_expression(&self, start: usize, len: usize) -> Self {
        let start = u32::try_from(start).expect("slice start exceeds MIndex");
        Self::new(
            self.start
                .checked_add(start)
                .expect("counting slice start overflow"),
            len,
        )
    }
}

impl SliceExpression for ReverseCounting {
    fn slice_expression(&self, start: usize, len: usize) -> Self {
        Self {
            start: self.start.saturating_sub(start),
            len,
        }
    }
}

impl<Source> SliceExpression for Taken<Source>
where
    Source: TakenSource + Clone,
{
    fn slice_expression(&self, start: usize, len: usize) -> Self {
        let start = u32::try_from(start).expect("slice start exceeds MIndex");
        let len = u32::try_from(len).expect("slice length exceeds MIndex");
        Self {
            source: self.source.clone(),
            offset: self
                .offset
                .checked_add(start)
                .expect("taken slice offset overflow"),
            len,
        }
    }
}

impl<Left, Right> SliceExpression for Zip<Left, Right>
where
    Left: SliceExpression,
    Right: SliceExpression,
    Zip<Left, Right>: ReadExpression,
{
    fn slice_expression(&self, start: usize, len: usize) -> Self {
        Zip::new(
            self.0.slice_expression(start, len),
            self.1.slice_expression(start, len),
        )
    }
}

impl<Input, Op> SliceExpression for Transform<Input, Op>
where
    Input: SliceExpression,
    Op: UnaryOp<Input::Item>,
{
    fn slice_expression(&self, start: usize, len: usize) -> Self {
        Transform {
            input: self.input.slice_expression(start, len),
            _op: PhantomData,
        }
    }
}

impl<Input, Op> SliceExpression for IndexedTransform<Input, Op>
where
    Input: ReadExpression + SliceExpression,
    Op: IndexedUnaryOp<Input::Item>,
{
    fn slice_expression(&self, start: usize, len: usize) -> Self {
        Self {
            input: self.input.slice_expression(start, len),
            _op: PhantomData,
        }
    }
}

impl<Input, Op> SliceExpression for AdjacentIndexedTransform<Input, Op>
where
    Input: ReadExpression + SliceExpression,
    Op: IndexedBinaryOp<Input::Item>,
{
    fn slice_expression(&self, start: usize, len: usize) -> Self {
        Self {
            input: self.input.slice_expression(start, len),
            _op: PhantomData,
        }
    }
}

impl<Values, Indices> SliceExpression for Permute<Values, Indices>
where
    Values: ReadExpression + Clone,
    Indices: SliceExpression<Item = u32>,
    Permute<Values, Indices>: ReadExpression,
{
    fn slice_expression(&self, start: usize, len: usize) -> Self {
        Permute::new(
            self.values.clone(),
            self.indices.slice_expression(start, len),
        )
    }
}

impl<Values> SliceExpression for Reverse<Values>
where
    Values: ReadExpression + Clone,
    Reverse<Values>: ReadExpression,
{
    fn slice_expression(&self, start: usize, len: usize) -> Self {
        Self {
            values: self.values.clone(),
            offset: self
                .offset
                .checked_add(start)
                .expect("reverse slice offset overflow"),
            len: Some(len),
        }
    }
}

impl<Input, Output> SliceExpression for Reassociate<Input, Output>
where
    Input: SliceExpression,
    Input::Item: StorageLayout,
    Output: StorageLayout + WriteFrom<Input::Item> + 'static,
{
    fn slice_expression(&self, start: usize, len: usize) -> Self {
        Reassociate::new(self.input.slice_expression(start, len))
    }
}

impl<Runtime, Input> SliceExpression for Slice<Runtime, Input>
where
    Input: SliceExpression,
{
    fn slice_expression(&self, start: usize, len: usize) -> Self {
        Self::new(self.input.slice_expression(start, len))
    }
}

/// Empty slot environment used as the root binding state.
#[doc(hidden)]
pub struct Env0;
#[doc(hidden)]
pub struct Env1<L0>(PhantomData<fn() -> L0>);
#[doc(hidden)]
pub struct Env2<L0, L1>(PhantomData<fn() -> (L0, L1)>);
#[doc(hidden)]
pub struct Env3<L0, L1, L2>(PhantomData<fn() -> (L0, L1, L2)>);
#[doc(hidden)]
pub struct Env4<L0, L1, L2, L3>(PhantomData<fn() -> (L0, L1, L2, L3)>);
#[doc(hidden)]
pub struct Env5<L0, L1, L2, L3, L4>(PhantomData<fn() -> (L0, L1, L2, L3, L4)>);
#[doc(hidden)]
pub struct Env6<L0, L1, L2, L3, L4, L5>(PhantomData<fn() -> (L0, L1, L2, L3, L4, L5)>);
#[doc(hidden)]
pub struct Env7<L0, L1, L2, L3, L4, L5, L6>(PhantomData<fn() -> (L0, L1, L2, L3, L4, L5, L6)>);
#[doc(hidden)]
pub struct Env8<L0, L1, L2, L3, L4, L5, L6, L7>(
    PhantomData<fn() -> (L0, L1, L2, L3, L4, L5, L6, L7)>,
);

/// Binds an expression's leaves to consecutive slots, starting at `Env`.
///
/// Binary nodes always bind the left subtree first, then pass its resulting
/// environment to the right subtree.
#[doc(hidden)]
pub trait BindSlots<Env> {
    type Expr;
    type NextEnv;
}

macro_rules! impl_leaf_binding {
    (impl <$( $env_ty:ident ),*> $env:ty => $column_next:ty, $counting_next:ty, $slot:ident) => {
        impl<T, $( $env_ty ),*> BindSlots<$env> for Column<T>
        where
            T: MStorageElement,
        {
            type Expr = $slot<T, Direct>;
            type NextEnv = $column_next;
        }

        impl<T, $( $env_ty ),*> BindSlots<$env> for Constant<T>
        where
            T: MStorageElement,
        {
            type Expr = $slot<T, Broadcast>;
            type NextEnv = $column_next;
        }

        impl<$( $env_ty ),*> BindSlots<$env> for Counting {
            type Expr = $slot<u32, Count>;
            type NextEnv = $counting_next;
        }

        impl<$( $env_ty ),*> BindSlots<$env> for ReverseCounting {
            type Expr = $slot<u32, ReverseCount>;
            type NextEnv = $counting_next;
        }
    };
}

impl_leaf_binding!(impl <> Env0 => Env1<T>, Env1<u32>, Slot0);
impl_leaf_binding!(impl <L0> Env1<L0> => Env2<L0, T>, Env2<L0, u32>, Slot1);
impl_leaf_binding!(impl <L0, L1> Env2<L0, L1> => Env3<L0, L1, T>, Env3<L0, L1, u32>, Slot2);
impl_leaf_binding!(impl <L0, L1, L2> Env3<L0, L1, L2> => Env4<L0, L1, L2, T>, Env4<L0, L1, L2, u32>, Slot3);
impl_leaf_binding!(impl <L0, L1, L2, L3> Env4<L0, L1, L2, L3> => Env5<L0, L1, L2, L3, T>, Env5<L0, L1, L2, L3, u32>, Slot4);
impl_leaf_binding!(impl <L0, L1, L2, L3, L4> Env5<L0, L1, L2, L3, L4> => Env6<L0, L1, L2, L3, L4, T>, Env6<L0, L1, L2, L3, L4, u32>, Slot5);
impl_leaf_binding!(impl <L0, L1, L2, L3, L4, L5> Env6<L0, L1, L2, L3, L4, L5> => Env7<L0, L1, L2, L3, L4, L5, T>, Env7<L0, L1, L2, L3, L4, L5, u32>, Slot6);
impl_leaf_binding!(impl <L0, L1, L2, L3, L4, L5, L6> Env7<L0, L1, L2, L3, L4, L5, L6> => Env8<L0, L1, L2, L3, L4, L5, L6, T>, Env8<L0, L1, L2, L3, L4, L5, L6, u32>, Slot7);

impl<Source, Env> BindSlots<Env> for Taken<Source>
where
    Source: TakenSource,
    Source::Read: BindSlots<Env>,
{
    type Expr = <Source::Read as BindSlots<Env>>::Expr;
    type NextEnv = <Source::Read as BindSlots<Env>>::NextEnv;
}

impl<Left, Right, Env> BindSlots<Env> for Zip<Left, Right>
where
    Left: BindSlots<Env>,
    Right: BindSlots<Left::NextEnv>,
{
    type Expr = ZipExpr<Left::Expr, Right::Expr>;
    type NextEnv = Right::NextEnv;
}

impl<Input, Op, Env> BindSlots<Env> for Transform<Input, Op>
where
    Input: ReadExpression + BindSlots<Env>,
    Op: UnaryOp<Input::Item>,
{
    type Expr = TransformExpr<Input::Expr, Input::Item, Op>;
    type NextEnv = Input::NextEnv;
}

impl<Input, Op, Env> BindSlots<Env> for IndexedTransform<Input, Op>
where
    Input: ReadExpression + BindSlots<Env>,
    Op: IndexedUnaryOp<Input::Item>,
{
    type Expr = IndexedTransformExpr<Input::Expr, Input::Item, Op>;
    type NextEnv = Input::NextEnv;
}

impl<Input, Op, Env> BindSlots<Env> for AdjacentIndexedTransform<Input, Op>
where
    Input: ReadExpression + BindSlots<Env>,
    Op: IndexedBinaryOp<Input::Item>,
{
    type Expr = AdjacentIndexedTransformExpr<Input::Expr, Input::Item, Op>;
    type NextEnv = Input::NextEnv;
}

impl<Input, Op, Env> BindSlots<Env> for Adjacent<Input, Op>
where
    Input: ReadExpression + BindSlots<Env>,
    Input::Item: StorageLayout,
    <Input::Item as StorageLayout>::StorageLeaves: crate::storage::SelectLeaves,
    Op: ReductionOp<Input::Item>,
{
    type Expr = AdjacentExpr<
        Input::Expr,
        Input::Item,
        Op,
        <Input::Item as StorageLayout>::DeviceLayout,
        <Input::Item as StorageLayout>::StorageLeaves,
    >;
    type NextEnv = Input::NextEnv;
}

impl<Values, Indices, Env> BindSlots<Env> for Permute<Values, Indices>
where
    Values: BindSlots<Env>,
    Indices: BindSlots<Values::NextEnv>,
{
    type Expr = PermuteExpr<Values::Expr, Indices::Expr>;
    type NextEnv = Indices::NextEnv;
}

impl<Values, Env> BindSlots<Env> for Reverse<Values>
where
    Values: BindSlots<Env>,
    ReverseCounting: BindSlots<Values::NextEnv>,
{
    type Expr = PermuteExpr<Values::Expr, <ReverseCounting as BindSlots<Values::NextEnv>>::Expr>;
    type NextEnv = <ReverseCounting as BindSlots<Values::NextEnv>>::NextEnv;
}

impl<Input, Output, Env> BindSlots<Env> for Reassociate<Input, Output>
where
    Input: ReadExpression + BindSlots<Env>,
    Input::Item: StorageLayout,
    Output: StorageLayout + WriteFrom<Input::Item> + 'static,
{
    type Expr = ReassociateExpr<
        Input::Expr,
        Input::Item,
        Output,
        <Input::Item as StorageLayout>::DeviceLayout,
        <Output as StorageLayout>::DeviceLayout,
    >;
    type NextEnv = Input::NextEnv;
}

impl<Runtime, Input, Env> BindSlots<Env> for Slice<Runtime, Input>
where
    Input: BindSlots<Env>,
{
    type Expr = Input::Expr;
    type NextEnv = Input::NextEnv;
}

/// A non-empty final slot environment and its read arity.
#[doc(hidden)]
pub trait SlotEnvironment {
    type Arity: ReadArity;
}

impl<L0> SlotEnvironment for Env1<L0> {
    type Arity = A1;
}
impl<L0, L1> SlotEnvironment for Env2<L0, L1> {
    type Arity = A2;
}
impl<L0, L1, L2> SlotEnvironment for Env3<L0, L1, L2> {
    type Arity = A3;
}
impl<L0, L1, L2, L3> SlotEnvironment for Env4<L0, L1, L2, L3> {
    type Arity = A4;
}
impl<L0, L1, L2, L3, L4> SlotEnvironment for Env5<L0, L1, L2, L3, L4> {
    type Arity = A5;
}
impl<L0, L1, L2, L3, L4, L5> SlotEnvironment for Env6<L0, L1, L2, L3, L4, L5> {
    type Arity = A6;
}
impl<L0, L1, L2, L3, L4, L5, L6> SlotEnvironment for Env7<L0, L1, L2, L3, L4, L5, L6> {
    type Arity = A7;
}
impl<L0, L1, L2, L3, L4, L5, L6, L7> SlotEnvironment for Env8<L0, L1, L2, L3, L4, L5, L6, L7> {
    type Arity = A8;
}

/// Connects a final slot environment to its arity-specific evaluator.
#[doc(hidden)]
pub trait EvalEnvironment<Expr, Item: CubeType>: SlotEnvironment {}

impl<Expr, Item, L0> EvalEnvironment<Expr, Item> for Env1<L0>
where
    Item: CubeType,
    L0: MStorageElement,
    Expr: Eval1<Item, L0>,
{
}
impl<Expr, Item, L0, L1> EvalEnvironment<Expr, Item> for Env2<L0, L1>
where
    Item: CubeType,
    L0: MStorageElement,
    L1: MStorageElement,
    Expr: Eval2<Item, L0, L1>,
{
}
impl<Expr, Item, L0, L1, L2> EvalEnvironment<Expr, Item> for Env3<L0, L1, L2>
where
    Item: CubeType,
    L0: MStorageElement,
    L1: MStorageElement,
    L2: MStorageElement,
    Expr: Eval3<Item, L0, L1, L2>,
{
}
impl<Expr, Item, L0, L1, L2, L3> EvalEnvironment<Expr, Item> for Env4<L0, L1, L2, L3>
where
    Item: CubeType,
    L0: MStorageElement,
    L1: MStorageElement,
    L2: MStorageElement,
    L3: MStorageElement,
    Expr: Eval4<Item, L0, L1, L2, L3>,
{
}
impl<Expr, Item, L0, L1, L2, L3, L4> EvalEnvironment<Expr, Item> for Env5<L0, L1, L2, L3, L4>
where
    Item: CubeType,
    L0: MStorageElement,
    L1: MStorageElement,
    L2: MStorageElement,
    L3: MStorageElement,
    L4: MStorageElement,
    Expr: Eval5<Item, L0, L1, L2, L3, L4>,
{
}
impl<Expr, Item, L0, L1, L2, L3, L4, L5> EvalEnvironment<Expr, Item>
    for Env6<L0, L1, L2, L3, L4, L5>
where
    Item: CubeType,
    L0: MStorageElement,
    L1: MStorageElement,
    L2: MStorageElement,
    L3: MStorageElement,
    L4: MStorageElement,
    L5: MStorageElement,
    Expr: Eval6<Item, L0, L1, L2, L3, L4, L5>,
{
}
impl<Expr, Item, L0, L1, L2, L3, L4, L5, L6> EvalEnvironment<Expr, Item>
    for Env7<L0, L1, L2, L3, L4, L5, L6>
where
    Item: CubeType,
    L0: MStorageElement,
    L1: MStorageElement,
    L2: MStorageElement,
    L3: MStorageElement,
    L4: MStorageElement,
    L5: MStorageElement,
    L6: MStorageElement,
    Expr: Eval7<Item, L0, L1, L2, L3, L4, L5, L6>,
{
}
impl<Expr, Item, L0, L1, L2, L3, L4, L5, L6, L7> EvalEnvironment<Expr, Item>
    for Env8<L0, L1, L2, L3, L4, L5, L6, L7>
where
    Item: CubeType,
    L0: MStorageElement,
    L1: MStorageElement,
    L2: MStorageElement,
    L3: MStorageElement,
    L4: MStorageElement,
    L5: MStorageElement,
    L6: MStorageElement,
    L7: MStorageElement,
    Expr: Eval8<Item, L0, L1, L2, L3, L4, L5, L6, L7>,
{
}

/// Fully bound form of a read expression.
///
/// This is the bridge from recursively computed [`ReadArity`] to the matching
/// `EvalN` trait and ordered leaf types.
#[doc(hidden)]
pub trait LowerReadExpression: ReadExpression {
    type DeviceExpr: DeviceExpr<Self::Item>;
    type Slots: SlotEnvironment<Arity = Self::ReadArity>
        + EvalEnvironment<Self::DeviceExpr, Self::Item>;
}

impl<Input> LowerReadExpression for Input
where
    Input: ReadExpression + BindSlots<Env0>,
    Input::Expr: DeviceExpr<<Input as ReadExpression>::Item>,
    Input::NextEnv: SlotEnvironment<Arity = <Input as ReadExpression>::ReadArity>
        + EvalEnvironment<Input::Expr, <Input as ReadExpression>::Item>,
{
    type DeviceExpr = Input::Expr;
    type Slots = Input::NextEnv;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        eval::*,
        op::{Identity, UnaryOp},
    };
    use static_assertions::{assert_impl_all, assert_not_impl_any};

    type Seven = Zip<
        Column<u8>,
        Zip<
            Column<u16>,
            Zip<Column<u32>, Zip<Column<u64>, Zip<Column<i8>, Zip<Column<i16>, Column<i32>>>>>,
        >,
    >;
    type SevenItem = (u8, (u16, (u32, (u64, (i8, (i16, i32))))));
    type Lazified = Transform<Permute<Seven, Counting>, Identity>;
    type Unsupported9 = Transform<Permute<Lazified, Counting>, Identity>;
    type One = Column<u8>;
    type Two = Zip<Column<u8>, Column<u16>>;
    type Three = Zip<Two, Column<u32>>;
    type Four = Zip<Three, Column<u64>>;
    type Five = Zip<Four, Column<i8>>;
    type Six = Zip<Five, Column<i16>>;

    type SevenDevice = ZipExpr<
        Slot0<u8, Direct>,
        ZipExpr<
            Slot1<u16, Direct>,
            ZipExpr<
                Slot2<u32, Direct>,
                ZipExpr<
                    Slot3<u64, Direct>,
                    ZipExpr<Slot4<i8, Direct>, ZipExpr<Slot5<i16, Direct>, Slot6<i32, Direct>>>,
                >,
            >,
        >,
    >;
    type LazifiedDevice =
        TransformExpr<PermuteExpr<SevenDevice, Slot7<u32, Count>>, SevenItem, Identity>;
    type LazifiedSlots = Env8<u8, u16, u32, u64, i8, i16, i32, u32>;

    type RuntimeSevenItem = (u32, (u32, (u32, (u32, (u32, (u32, u32))))));
    type RuntimeSevenDevice = ZipExpr<
        Slot0<u32, Direct>,
        ZipExpr<
            Slot1<u32, Direct>,
            ZipExpr<
                Slot2<u32, Direct>,
                ZipExpr<
                    Slot3<u32, Direct>,
                    ZipExpr<Slot4<u32, Direct>, ZipExpr<Slot5<u32, Direct>, Slot6<u32, Direct>>>,
                >,
            >,
        >,
    >;
    type RuntimeLazifiedDevice = TransformExpr<
        PermuteExpr<RuntimeSevenDevice, Slot7<u32, Count>>,
        RuntimeSevenItem,
        Identity,
    >;
    type RuntimeMixedDevice =
        TransformExpr<ZipExpr<Slot0<u32, Broadcast>, Slot1<u32, Count>>, (u32, u32), AddPair>;

    struct AddPair;

    #[cubecl::cube]
    impl UnaryOp<(u32, u32)> for AddPair {
        type Output = u32;

        fn apply(input: (u32, u32)) -> u32 {
            input.0 + input.1
        }
    }

    #[test]
    fn read_arity_and_semantic_item_are_independent() {
        fn assert_shape<E, Item, Arity>()
        where
            E: ReadExpression<Item = Item, ReadArity = Arity>,
            Item: CubeType + 'static,
            Arity: ReadArity,
        {
        }

        fn assert_arity<E, Arity>()
        where
            E: ReadExpression<ReadArity = Arity>,
            Arity: ReadArity,
        {
        }

        type Pair = Zip<Column<u32>, Column<u32>>;
        assert_arity::<One, A1>();
        assert_arity::<Two, A2>();
        assert_arity::<Three, A3>();
        assert_arity::<Four, A4>();
        assert_arity::<Five, A5>();
        assert_arity::<Six, A6>();
        assert_shape::<Seven, SevenItem, A7>();
        assert_shape::<Lazified, SevenItem, A8>();
        assert_shape::<Transform<Pair, AddPair>, u32, A2>();
    }

    #[test]
    fn binding_is_left_to_right_and_selects_eval8() {
        fn assert_binding<E, Expr, Slots>()
        where
            E: BindSlots<Env0, Expr = Expr, NextEnv = Slots>
                + LowerReadExpression<DeviceExpr = Expr, Slots = Slots>,
            E: ReadExpression,
            Expr: DeviceExpr<E::Item>,
            Slots: SlotEnvironment<Arity = E::ReadArity> + EvalEnvironment<Expr, E::Item>,
        {
        }

        assert_binding::<Lazified, LazifiedDevice, LazifiedSlots>();

        type Mixed = Zip<Constant<u16>, Counting>;
        type MixedDevice = ZipExpr<Slot0<u16, Broadcast>, Slot1<u32, Count>>;
        assert_binding::<Mixed, MixedDevice, Env2<u16, u32>>();
    }

    #[cubecl::cube]
    #[allow(dead_code)]
    fn cubecl_compiles_eval8(
        slot0: &[u8],
        slot1: &[u16],
        slot2: &[u32],
        slot3: &[u64],
        slot4: &[i8],
        slot5: &[i16],
        slot6: &[i32],
        slot7: &[u32],
        offsets: &[u32],
        index: usize,
    ) -> SevenItem {
        LazifiedDevice::eval8(
            slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, offsets, index,
        )
    }

    #[cubecl::cube(launch_unchecked)]
    fn eval8_runtime_kernel(
        slot0: &[u32],
        slot1: &[u32],
        slot2: &[u32],
        slot3: &[u32],
        slot4: &[u32],
        slot5: &[u32],
        slot6: &[u32],
        slot7: &[u32],
        offsets: &[u32],
        output: &mut [u32],
    ) {
        let index = ABSOLUTE_POS as usize;
        if index < output.len() {
            let value = RuntimeLazifiedDevice::eval8(
                slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, offsets, index,
            );
            output[index] = value.0
                + value.1.0
                + value.1.1.0
                + value.1.1.1.0
                + value.1.1.1.1.0
                + value.1.1.1.1.1.0
                + value.1.1.1.1.1.1;
        }
    }

    #[cubecl::cube(launch_unchecked)]
    fn eval2_modes_runtime_kernel(
        constant: &[u32],
        counting: &[u32],
        offsets: &[u32],
        output: &mut [u32],
    ) {
        let index = ABSOLUTE_POS as usize;
        if index < output.len() {
            output[index] = RuntimeMixedDevice::eval2(constant, counting, offsets, index);
        }
    }

    #[test]
    fn eval8_executes_nested_zip_permute_and_transform_on_cubecl() {
        use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

        let client = WgpuRuntime::client(&WgpuDevice::DefaultDevice);
        let columns = [
            [10_u32, 11, 12, 13, 14],
            [20_u32, 21, 22, 23, 24],
            [30_u32, 31, 32, 33, 34],
            [40_u32, 41, 42, 43, 44],
            [50_u32, 51, 52, 53, 54],
            [60_u32, 61, 62, 63, 64],
            [70_u32, 71, 72, 73, 74],
        ];
        let handles = columns.map(|column| client.create_from_slice(u32::as_bytes(&column)));
        let count = client.create_from_slice(u32::as_bytes(&[1]));
        let offsets = client.create_from_slice(u32::as_bytes(&[0, 1, 2, 0, 1, 2, 0, 0]));
        let output = client.empty(2 * size_of::<u32>());

        unsafe {
            eval8_runtime_kernel::launch_unchecked::<WgpuRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(2),
                BufferArg::from_raw_parts(handles[0].clone(), columns[0].len()),
                BufferArg::from_raw_parts(handles[1].clone(), columns[1].len()),
                BufferArg::from_raw_parts(handles[2].clone(), columns[2].len()),
                BufferArg::from_raw_parts(handles[3].clone(), columns[3].len()),
                BufferArg::from_raw_parts(handles[4].clone(), columns[4].len()),
                BufferArg::from_raw_parts(handles[5].clone(), columns[5].len()),
                BufferArg::from_raw_parts(handles[6].clone(), columns[6].len()),
                BufferArg::from_raw_parts(count, 1),
                BufferArg::from_raw_parts(offsets, 8),
                BufferArg::from_raw_parts(output.clone(), 2),
            );
        }

        let bytes = client.read_one_unchecked(output);
        let actual = u32::from_bytes(&bytes);
        assert_eq!(actual, &[293, 300]);

        let constant = client.create_from_slice(u32::as_bytes(&[100]));
        let counting = client.create_from_slice(u32::as_bytes(&[5]));
        let offsets = client.create_from_slice(u32::as_bytes(&[99, 3]));
        let output = client.empty(2 * size_of::<u32>());
        unsafe {
            eval2_modes_runtime_kernel::launch_unchecked::<WgpuRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(2),
                BufferArg::from_raw_parts(constant, 1),
                BufferArg::from_raw_parts(counting, 1),
                BufferArg::from_raw_parts(offsets, 2),
                BufferArg::from_raw_parts(output.clone(), 2),
            );
        }
        let bytes = client.read_one_unchecked(output);
        let actual = u32::from_bytes(&bytes);
        assert_eq!(actual, &[108, 109]);
    }

    assert_impl_all!(One: ReadExpression, LowerReadExpression);
    assert_impl_all!(Two: ReadExpression, LowerReadExpression);
    assert_impl_all!(Three: ReadExpression, LowerReadExpression);
    assert_impl_all!(Four: ReadExpression, LowerReadExpression);
    assert_impl_all!(Five: ReadExpression, LowerReadExpression);
    assert_impl_all!(Six: ReadExpression, LowerReadExpression);
    assert_impl_all!(Seven: ReadExpression, LowerReadExpression);
    assert_impl_all!(Lazified: ReadExpression, LowerReadExpression);
    assert_impl_all!(Transform<Zip<Column<u32>, Column<u32>>, AddPair>: ReadExpression, LowerReadExpression);
    assert_not_impl_any!(Unsupported9: ReadExpression, LowerReadExpression);
}
