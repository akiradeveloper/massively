use std::marker::PhantomData;

use cubecl::frontend::PartialOrdExpand;
use cubecl::prelude::Runtime;

use crate::op;
use crate::value::{MItem, MStorageElement};

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
    T: MStorageElement + MItem<R>,
    Op: op::UnaryOp<R, (T,), Output = (T,)>,
{
    type Env = Op::Env;
    type Output = T;

    fn apply(env: Self::Env, input: T) -> T {
        Op::apply(env, (input,)).0
    }
}

#[cubecl::cube]
impl<R, T, Op> crate::detail::op::kernel::BinaryOp<T> for KernelTuple1Op<R, Op>
where
    R: Runtime,
    T: MStorageElement + MItem<R>,
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
    T: MStorageElement + MItem<R>,
    Op: op::PredicateOp<R, (T,)>,
{
    type Env = Op::Env;

    fn apply(env: Self::Env, input: T) -> bool {
        Op::apply(env, (input,))
    }
}

#[cubecl::cube]
impl<R, T, Op> crate::detail::op::kernel::BinaryPredicateOp<T> for KernelTuple1Op<R, Op>
where
    R: Runtime,
    T: MStorageElement,
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
    type Env = Op::Env;
    type Output = Op::Output;

    fn apply(env: Self::Env, input: Input) -> Self::Output {
        Op::apply(env, input)
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
    type Env = Op::Env;

    fn apply(env: Self::Env, input: Item) -> bool {
        Op::apply(env, input)
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
#[derive(Clone, Copy, Debug, Default)]
pub struct KernelScalarInputOp<R, Op>(PhantomData<fn() -> (R, Op)>);

#[cubecl::cube]
impl<R, T, Op> crate::detail::op::kernel::UnaryOp<(T,)> for KernelScalarInputOp<R, Op>
where
    R: Runtime,
    T: MStorageElement + MItem<R>,
    Op: op::UnaryOp<R, T>,
{
    type Env = Op::Env;
    type Output = Op::Output;

    fn apply(env: Self::Env, input: (T,)) -> Self::Output {
        Op::apply(env, input.0)
    }
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default)]
pub struct KernelScalarTuple1Op<R, Op>(PhantomData<fn() -> (R, Op)>);

impl<R, Op> KernelScalarTuple1Op<R, Op> {
    pub(super) fn new() -> Self {
        Self(PhantomData)
    }
}

#[cubecl::cube]
impl<R, T, Op> crate::detail::op::kernel::BinaryOp<(T,)> for KernelScalarTuple1Op<R, Op>
where
    R: Runtime,
    T: MStorageElement + MItem<R>,
    Op: op::ReductionOp<R, T>,
{
    fn apply(lhs: (T,), rhs: (T,)) -> (T,) {
        (Op::apply(lhs.0, rhs.0),)
    }
}

#[cubecl::cube]
impl<R, T, Op> crate::detail::op::kernel::PredicateOp<(T,)> for KernelScalarTuple1Op<R, Op>
where
    R: Runtime,
    T: MStorageElement + MItem<R>,
    Op: op::PredicateOp<R, T>,
{
    type Env = Op::Env;

    fn apply(env: Self::Env, input: (T,)) -> bool {
        Op::apply(env, input.0)
    }
}

#[cubecl::cube]
impl<R, T, Op> crate::detail::op::kernel::BinaryPredicateOp<(T,)> for KernelScalarTuple1Op<R, Op>
where
    R: Runtime,
    T: MStorageElement + MItem<R>,
    Op: op::BinaryPredicateOp<R, T>,
{
    fn apply(lhs: (T,), rhs: (T,)) -> bool {
        Op::apply(lhs.0, rhs.0)
    }
}

#[doc(hidden)]
pub struct StencilFlag;

#[cubecl::cube]
impl<R> op::PredicateOp<R, (u32,)> for StencilFlag
where
    R: Runtime,
{
    type Env = ();

    fn apply(_env: (), input: (u32,)) -> bool {
        input.0 > 0
    }
}
