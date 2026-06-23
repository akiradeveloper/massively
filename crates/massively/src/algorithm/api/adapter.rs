use std::marker::PhantomData;

use cubecl::frontend::PartialOrdExpand;

use super::{Backend, MItem, Scalar, op};

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
    B: Backend,
    T: Scalar,
    Op: op::UnaryOp<B, (T,), Output = (T,)>,
{
    type Output = T;

    fn apply(input: T) -> T {
        Op::apply((input,)).0
    }
}

#[cubecl::cube]
impl<B, T, Op> crate::detail::op::kernel::BinaryOp2<T> for KernelTuple1Op<B, Op>
where
    B: Backend,
    T: Scalar,
    Op: op::BinaryOp1<B, (T,)>,
{
    fn apply(lhs: T, rhs: T) -> T {
        Op::apply((lhs,), (rhs,)).0
    }
}

#[cubecl::cube]
impl<B, T, Op> crate::detail::op::kernel::PredicateOp1<T> for KernelTuple1Op<B, Op>
where
    B: Backend,
    T: Scalar,
    Op: op::PredicateOp1<B, (T,)>,
{
    fn apply(input: T) -> bool {
        Op::apply((input,))
    }
}

#[cubecl::cube]
impl<B, T, Op> crate::detail::op::kernel::PredicateOp2<T> for KernelTuple1Op<B, Op>
where
    B: Backend,
    T: Scalar,
    Op: op::PredicateOp2<B, (T,)>,
{
    fn apply(lhs: T, rhs: T) -> bool {
        Op::apply((lhs,), (rhs,))
    }
}

#[cubecl::cube]
impl<B, Left, Right, Op, Output> op::UnaryOp<B, (Left, Right)>
    for KernelTuple1InnerProductOp<B, Op, Output>
where
    B: Backend,
    Left: Scalar,
    Right: Scalar,
    Output: MItem<B>,
    Output: 'static,
    Op: op::BinaryOp2<B, (Left,), (Right,), Output = Output>,
{
    type Output = Output;

    fn apply(input: (Left, Right)) -> Self::Output {
        Op::apply((input.0,), (input.1,))
    }
}

#[cubecl::cube]
impl<B, Input, Op> crate::detail::op::kernel::UnaryOp<Input> for KernelOp<B, Op>
where
    B: Backend,
    Input: MItem<B>,
    Op: op::UnaryOp<B, Input>,
{
    type Output = Op::Output;

    fn apply(input: Input) -> Self::Output {
        Op::apply(input)
    }
}

#[cubecl::cube]
impl<B, Item, Op> crate::detail::op::kernel::BinaryOp2<Item> for KernelOp<B, Op>
where
    B: Backend,
    Item: MItem<B>,
    Op: op::BinaryOp1<B, Item>,
{
    fn apply(lhs: Item, rhs: Item) -> Item {
        Op::apply(lhs, rhs)
    }
}

#[cubecl::cube]
impl<B, Item, Op> crate::detail::op::kernel::PredicateOp1<Item> for KernelOp<B, Op>
where
    B: Backend,
    Item: MItem<B>,
    Op: op::PredicateOp1<B, Item>,
{
    fn apply(input: Item) -> bool {
        Op::apply(input)
    }
}

#[cubecl::cube]
impl<B, Item, Op> crate::detail::op::kernel::PredicateOp2<Item> for KernelOp<B, Op>
where
    B: Backend,
    Item: MItem<B>,
    Op: op::PredicateOp2<B, Item>,
{
    fn apply(lhs: Item, rhs: Item) -> bool {
        Op::apply(lhs, rhs)
    }
}

#[doc(hidden)]
pub struct StencilFlag;

#[cubecl::cube]
impl<B> op::PredicateOp1<B, (u32,)> for StencilFlag
where
    B: Backend,
{
    fn apply(input: (u32,)) -> bool {
        input.0 > 0
    }
}
