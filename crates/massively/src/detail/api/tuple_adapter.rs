use std::marker::PhantomData;

use crate::detail::op::kernel::{BinaryOp, BinaryPredicateOp, PredicateOp};
use cubecl::prelude::*;

#[doc(hidden)]
pub struct Tuple1Less<Less> {
    _less: PhantomData<fn() -> Less>,
}

impl<Less> Default for Tuple1Less<Less> {
    fn default() -> Self {
        Self { _less: PhantomData }
    }
}

#[cubecl::cube]
impl<T, Less> BinaryPredicateOp<T> for Tuple1Less<Less>
where
    T: CubePrimitive + CubeElement,
    Less: BinaryPredicateOp<(T,)>,
{
    fn apply(lhs: T, rhs: T) -> bool {
        Less::apply((lhs,), (rhs,))
    }
}

#[doc(hidden)]
pub struct Tuple1BinaryOp<Op> {
    _op: PhantomData<fn() -> Op>,
}

impl<Op> Default for Tuple1BinaryOp<Op> {
    fn default() -> Self {
        Self { _op: PhantomData }
    }
}

#[cubecl::cube]
impl<T, Op> BinaryOp<T> for Tuple1BinaryOp<Op>
where
    T: CubePrimitive + CubeElement,
    Op: BinaryOp<(T,)>,
{
    fn apply(lhs: T, rhs: T) -> T {
        Op::apply((lhs,), (rhs,)).0
    }
}

#[doc(hidden)]
pub struct Tuple1PredicateOp<Op> {
    _op: PhantomData<fn() -> Op>,
}

impl<Op> Default for Tuple1PredicateOp<Op> {
    fn default() -> Self {
        Self { _op: PhantomData }
    }
}

#[cubecl::cube]
impl<T, Op> PredicateOp<T> for Tuple1PredicateOp<Op>
where
    T: CubePrimitive + CubeElement,
    Op: PredicateOp<(T,)>,
{
    fn apply(input: T) -> bool {
        Op::apply((input,))
    }
}
