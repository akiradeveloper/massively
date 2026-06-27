use std::marker::PhantomData;

use cubecl::frontend::PartialOrdExpand;
use cubecl::prelude::Runtime;

use crate::op;
use crate::runtime::Scalar;
use crate::value::MItem;

#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default)]
pub struct KernelOp<B, Op>(PhantomData<fn() -> (B, Op)>);

impl<B, Op> KernelOp<B, Op> {
    pub(super) fn new() -> Self {
        Self(PhantomData)
    }
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default)]
pub struct KernelTuple1Op<B, Op>(PhantomData<fn() -> (B, Op)>);

impl<B, Op> KernelTuple1Op<B, Op> {
    pub(super) fn new() -> Self {
        Self(PhantomData)
    }
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default)]
pub struct KernelTuple1InnerProductOp<B, Op, Output>(PhantomData<fn() -> (B, Op, Output)>);

impl<B, Op, Output> KernelTuple1InnerProductOp<B, Op, Output> {
    pub(super) fn new() -> Self {
        Self(PhantomData)
    }
}

#[cubecl::cube]
impl<B, T, Op> crate::detail::op::kernel::UnaryOp<T> for KernelTuple1Op<B, Op>
where
    B: Runtime,
    T: Scalar,
    Op: op::UnaryOp<B, (T,), Output = (T,)>,
{
    type Output = T;

    fn apply(input: T) -> T {
        Op::apply((input,)).0
    }
}

#[cubecl::cube]
impl<B, T, Op> crate::detail::op::kernel::BinaryOp<T> for KernelTuple1Op<B, Op>
where
    B: Runtime,
    T: Scalar,
    Op: op::ReductionOp<B, (T,)>,
{
    fn apply(lhs: T, rhs: T) -> T {
        Op::apply((lhs,), (rhs,)).0
    }
}

#[cubecl::cube]
impl<B, T, Op> crate::detail::op::kernel::PredicateOp<T> for KernelTuple1Op<B, Op>
where
    B: Runtime,
    T: Scalar,
    Op: op::PredicateOp<B, (T,)>,
{
    fn apply(input: T) -> bool {
        Op::apply((input,))
    }
}

#[cubecl::cube]
impl<B, T, Op> crate::detail::op::kernel::BinaryPredicateOp<T> for KernelTuple1Op<B, Op>
where
    B: Runtime,
    T: Scalar,
    Op: op::BinaryPredicateOp<B, (T,)>,
{
    fn apply(lhs: T, rhs: T) -> bool {
        Op::apply((lhs,), (rhs,))
    }
}

#[cubecl::cube]
impl<B, Left, Right, Op, Output> op::UnaryOp<B, (Left, Right)>
    for KernelTuple1InnerProductOp<B, Op, Output>
where
    B: Runtime,
    Left: Scalar,
    Right: Scalar,
    Output: MItem<B>,
    Output: 'static,
    Op: op::BinaryOp<B, (Left,), (Right,), Output = Output>,
{
    type Output = Output;

    fn apply(input: (Left, Right)) -> Self::Output {
        Op::apply((input.0,), (input.1,))
    }
}

#[cubecl::cube]
impl<B, Input, Op> crate::detail::op::kernel::UnaryOp<Input> for KernelOp<B, Op>
where
    B: Runtime,
    Input: MItem<B>,
    Op: op::UnaryOp<B, Input>,
{
    type Output = Op::Output;

    fn apply(input: Input) -> Self::Output {
        Op::apply(input)
    }
}

#[cubecl::cube]
impl<B, Item, Op> crate::detail::op::kernel::BinaryOp<Item> for KernelOp<B, Op>
where
    B: Runtime,
    Item: MItem<B>,
    Op: op::ReductionOp<B, Item>,
{
    fn apply(lhs: Item, rhs: Item) -> Item {
        Op::apply(lhs, rhs)
    }
}

#[cubecl::cube]
impl<B, Item, Op> crate::detail::op::kernel::PredicateOp<Item> for KernelOp<B, Op>
where
    B: Runtime,
    Item: MItem<B>,
    Op: op::PredicateOp<B, Item>,
{
    fn apply(input: Item) -> bool {
        Op::apply(input)
    }
}

#[cubecl::cube]
impl<B, Item, Op> crate::detail::op::kernel::BinaryPredicateOp<Item> for KernelOp<B, Op>
where
    B: Runtime,
    Item: MItem<B>,
    Op: op::BinaryPredicateOp<B, Item>,
{
    fn apply(lhs: Item, rhs: Item) -> bool {
        Op::apply(lhs, rhs)
    }
}

#[doc(hidden)]
pub struct StencilFlag;

#[cubecl::cube]
impl<B> op::PredicateOp<B, (u32,)> for StencilFlag
where
    B: Runtime,
{
    fn apply(input: (u32,)) -> bool {
        input.0 > 0
    }
}
