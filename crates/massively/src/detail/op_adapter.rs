use std::marker::PhantomData;

use cubecl::frontend::PartialOrdExpand;
use cubecl::prelude::Runtime;

use crate::op;
use crate::runtime::Scalar;
use crate::value::MItem;

#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default)]
pub struct KernelOp<R, Op>(PhantomData<fn() -> (R, Op)>);

impl<R, Op> KernelOp<R, Op> {
    pub(crate) fn new() -> Self {
        Self(PhantomData)
    }
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default)]
pub struct KernelTuple1Op<R, Op>(PhantomData<fn() -> (R, Op)>);

impl<R, Op> KernelTuple1Op<R, Op> {
    pub(super) fn new() -> Self {
        Self(PhantomData)
    }
}

#[cubecl::cube]
impl<R, T, Op> crate::detail::op::kernel::UnaryOp<T> for KernelTuple1Op<R, Op>
where
    R: Runtime,
    T: Scalar,
    Op: op::UnaryOp<R, (T,), Output = (T,)>,
{
    type Output = T;

    fn apply(input: T) -> T {
        Op::apply((input,)).0
    }
}

#[cubecl::cube]
impl<R, T, Op> crate::detail::op::kernel::BinaryOp<T> for KernelTuple1Op<R, Op>
where
    R: Runtime,
    T: Scalar,
    Op: op::ReductionOp<R, (T,)>,
{
    fn apply(lhs: T, rhs: T) -> T {
        Op::apply((lhs,), (rhs,)).0
    }
}

#[cubecl::cube]
impl<R, T, Op> crate::detail::op::kernel::PredicateOp<T> for KernelTuple1Op<R, Op>
where
    R: Runtime,
    T: Scalar,
    Op: op::PredicateOp<R, (T,)>,
{
    fn apply(input: T) -> bool {
        Op::apply((input,))
    }
}

#[cubecl::cube]
impl<R, T, Op> crate::detail::op::kernel::BinaryPredicateOp<T> for KernelTuple1Op<R, Op>
where
    R: Runtime,
    T: Scalar,
    Op: op::BinaryPredicateOp<R, (T,)>,
{
    fn apply(lhs: T, rhs: T) -> bool {
        Op::apply((lhs,), (rhs,))
    }
}

#[cubecl::cube]
impl<R, Input, Op> crate::detail::op::kernel::UnaryOp<Input> for KernelOp<R, Op>
where
    R: Runtime,
    Input: MItem<R>,
    Op: op::UnaryOp<R, Input>,
{
    type Output = Op::Output;

    fn apply(input: Input) -> Self::Output {
        Op::apply(input)
    }
}

#[cubecl::cube]
impl<R, Item, Op> crate::detail::op::kernel::BinaryOp<Item> for KernelOp<R, Op>
where
    R: Runtime,
    Item: MItem<R>,
    Op: op::ReductionOp<R, Item>,
{
    fn apply(lhs: Item, rhs: Item) -> Item {
        Op::apply(lhs, rhs)
    }
}

#[cubecl::cube]
impl<R, Item, Op> crate::detail::op::kernel::PredicateOp<Item> for KernelOp<R, Op>
where
    R: Runtime,
    Item: MItem<R>,
    Op: op::PredicateOp<R, Item>,
{
    fn apply(input: Item) -> bool {
        Op::apply(input)
    }
}

#[cubecl::cube]
impl<R, Item, Op> crate::detail::op::kernel::BinaryPredicateOp<Item> for KernelOp<R, Op>
where
    R: Runtime,
    Item: MItem<R>,
    Op: op::BinaryPredicateOp<R, Item>,
{
    fn apply(lhs: Item, rhs: Item) -> bool {
        Op::apply(lhs, rhs)
    }
}

#[doc(hidden)]
pub struct StencilFlag;

#[cubecl::cube]
impl<R> op::PredicateOp<R, (u32,)> for StencilFlag
where
    R: Runtime,
{
    fn apply(input: (u32,)) -> bool {
        input.0 > 0
    }
}
