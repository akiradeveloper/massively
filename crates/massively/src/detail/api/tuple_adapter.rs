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
pub struct Tuple2AsTuple3Less<Less> {
    _less: PhantomData<fn() -> Less>,
}

impl<Less> Default for Tuple2AsTuple3Less<Less> {
    fn default() -> Self {
        Self { _less: PhantomData }
    }
}

#[cubecl::cube]
impl<A, B, C, Less> BinaryPredicateOp<(A, B, C)> for Tuple2AsTuple3Less<Less>
where
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    C: CubePrimitive + CubeElement,
    Less: BinaryPredicateOp<(A, B)>,
{
    fn apply(lhs: (A, B, C), rhs: (A, B, C)) -> bool {
        Less::apply((lhs.0, lhs.1), (rhs.0, rhs.1))
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

#[doc(hidden)]
pub struct Tuple4AsTuple7BinaryOp<Op> {
    _op: PhantomData<fn() -> Op>,
}

impl<Op> Default for Tuple4AsTuple7BinaryOp<Op> {
    fn default() -> Self {
        Self { _op: PhantomData }
    }
}

#[cubecl::cube]
impl<A, B, C, D, Op> BinaryOp<(A, B, C, D, u32, u32, u32)> for Tuple4AsTuple7BinaryOp<Op>
where
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    C: CubePrimitive + CubeElement,
    D: CubePrimitive + CubeElement,
    Op: BinaryOp<(A, B, C, D)>,
{
    fn apply(
        lhs: (A, B, C, D, u32, u32, u32),
        rhs: (A, B, C, D, u32, u32, u32),
    ) -> (A, B, C, D, u32, u32, u32) {
        let out = Op::apply((lhs.0, lhs.1, lhs.2, lhs.3), (rhs.0, rhs.1, rhs.2, rhs.3));
        (out.0, out.1, out.2, out.3, lhs.4, lhs.5, lhs.6)
    }
}

#[doc(hidden)]
pub struct Tuple4AsTuple7PredicateOp<Op> {
    _op: PhantomData<fn() -> Op>,
}

impl<Op> Default for Tuple4AsTuple7PredicateOp<Op> {
    fn default() -> Self {
        Self { _op: PhantomData }
    }
}

#[cubecl::cube]
impl<A, B, C, D, Op> PredicateOp<(A, B, C, D, u32, u32, u32)> for Tuple4AsTuple7PredicateOp<Op>
where
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    C: CubePrimitive + CubeElement,
    D: CubePrimitive + CubeElement,
    Op: PredicateOp<(A, B, C, D)>,
{
    fn apply(input: (A, B, C, D, u32, u32, u32)) -> bool {
        Op::apply((input.0, input.1, input.2, input.3))
    }
}

#[doc(hidden)]
pub struct Tuple4AsTuple7BinaryPredicateOp<Op> {
    _op: PhantomData<fn() -> Op>,
}

impl<Op> Default for Tuple4AsTuple7BinaryPredicateOp<Op> {
    fn default() -> Self {
        Self { _op: PhantomData }
    }
}

#[cubecl::cube]
impl<A, B, C, D, Op> BinaryPredicateOp<(A, B, C, D, u32, u32, u32)>
    for Tuple4AsTuple7BinaryPredicateOp<Op>
where
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    C: CubePrimitive + CubeElement,
    D: CubePrimitive + CubeElement,
    Op: BinaryPredicateOp<(A, B, C, D)>,
{
    fn apply(lhs: (A, B, C, D, u32, u32, u32), rhs: (A, B, C, D, u32, u32, u32)) -> bool {
        Op::apply((lhs.0, lhs.1, lhs.2, lhs.3), (rhs.0, rhs.1, rhs.2, rhs.3))
    }
}

#[doc(hidden)]
pub struct Tuple5AsTuple7BinaryOp<Op> {
    _op: PhantomData<fn() -> Op>,
}

impl<Op> Default for Tuple5AsTuple7BinaryOp<Op> {
    fn default() -> Self {
        Self { _op: PhantomData }
    }
}

#[cubecl::cube]
impl<A, B, C, D, E, Op> BinaryOp<(A, B, C, D, E, u32, u32)> for Tuple5AsTuple7BinaryOp<Op>
where
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    C: CubePrimitive + CubeElement,
    D: CubePrimitive + CubeElement,
    E: CubePrimitive + CubeElement,
    Op: BinaryOp<(A, B, C, D, E)>,
{
    fn apply(
        lhs: (A, B, C, D, E, u32, u32),
        rhs: (A, B, C, D, E, u32, u32),
    ) -> (A, B, C, D, E, u32, u32) {
        let out = Op::apply(
            (lhs.0, lhs.1, lhs.2, lhs.3, lhs.4),
            (rhs.0, rhs.1, rhs.2, rhs.3, rhs.4),
        );
        (out.0, out.1, out.2, out.3, out.4, lhs.5, lhs.6)
    }
}

#[doc(hidden)]
pub struct Tuple5AsTuple7PredicateOp<Op> {
    _op: PhantomData<fn() -> Op>,
}

impl<Op> Default for Tuple5AsTuple7PredicateOp<Op> {
    fn default() -> Self {
        Self { _op: PhantomData }
    }
}

#[cubecl::cube]
impl<A, B, C, D, E, Op> PredicateOp<(A, B, C, D, E, u32, u32)> for Tuple5AsTuple7PredicateOp<Op>
where
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    C: CubePrimitive + CubeElement,
    D: CubePrimitive + CubeElement,
    E: CubePrimitive + CubeElement,
    Op: PredicateOp<(A, B, C, D, E)>,
{
    fn apply(input: (A, B, C, D, E, u32, u32)) -> bool {
        Op::apply((input.0, input.1, input.2, input.3, input.4))
    }
}

#[doc(hidden)]
pub struct Tuple5AsTuple7BinaryPredicateOp<Op> {
    _op: PhantomData<fn() -> Op>,
}

impl<Op> Default for Tuple5AsTuple7BinaryPredicateOp<Op> {
    fn default() -> Self {
        Self { _op: PhantomData }
    }
}

#[cubecl::cube]
impl<A, B, C, D, E, Op> BinaryPredicateOp<(A, B, C, D, E, u32, u32)>
    for Tuple5AsTuple7BinaryPredicateOp<Op>
where
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    C: CubePrimitive + CubeElement,
    D: CubePrimitive + CubeElement,
    E: CubePrimitive + CubeElement,
    Op: BinaryPredicateOp<(A, B, C, D, E)>,
{
    fn apply(lhs: (A, B, C, D, E, u32, u32), rhs: (A, B, C, D, E, u32, u32)) -> bool {
        Op::apply(
            (lhs.0, lhs.1, lhs.2, lhs.3, lhs.4),
            (rhs.0, rhs.1, rhs.2, rhs.3, rhs.4),
        )
    }
}

#[doc(hidden)]
pub struct Tuple6AsTuple7BinaryOp<Op> {
    _op: PhantomData<fn() -> Op>,
}

impl<Op> Default for Tuple6AsTuple7BinaryOp<Op> {
    fn default() -> Self {
        Self { _op: PhantomData }
    }
}

#[doc(hidden)]
pub struct Tuple6AsTuple7PredicateOp<Op> {
    _op: PhantomData<fn() -> Op>,
}

impl<Op> Default for Tuple6AsTuple7PredicateOp<Op> {
    fn default() -> Self {
        Self { _op: PhantomData }
    }
}

#[cubecl::cube]
impl<A, B, C, D, E, F, Op> PredicateOp<(A, B, C, D, E, F, u32)> for Tuple6AsTuple7PredicateOp<Op>
where
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    C: CubePrimitive + CubeElement,
    D: CubePrimitive + CubeElement,
    E: CubePrimitive + CubeElement,
    F: CubePrimitive + CubeElement,
    Op: PredicateOp<(A, B, C, D, E, F)>,
{
    fn apply(input: (A, B, C, D, E, F, u32)) -> bool {
        Op::apply((input.0, input.1, input.2, input.3, input.4, input.5))
    }
}

#[doc(hidden)]
pub struct Tuple6AsTuple7BinaryPredicateOp<Op> {
    _op: PhantomData<fn() -> Op>,
}

impl<Op> Default for Tuple6AsTuple7BinaryPredicateOp<Op> {
    fn default() -> Self {
        Self { _op: PhantomData }
    }
}

#[cubecl::cube]
impl<A, B, C, D, E, F, Op> BinaryPredicateOp<(A, B, C, D, E, F, u32)>
    for Tuple6AsTuple7BinaryPredicateOp<Op>
where
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    C: CubePrimitive + CubeElement,
    D: CubePrimitive + CubeElement,
    E: CubePrimitive + CubeElement,
    F: CubePrimitive + CubeElement,
    Op: BinaryPredicateOp<(A, B, C, D, E, F)>,
{
    fn apply(lhs: (A, B, C, D, E, F, u32), rhs: (A, B, C, D, E, F, u32)) -> bool {
        Op::apply(
            (lhs.0, lhs.1, lhs.2, lhs.3, lhs.4, lhs.5),
            (rhs.0, rhs.1, rhs.2, rhs.3, rhs.4, rhs.5),
        )
    }
}

#[cubecl::cube]
impl<A, B, C, D, E, F, Op> BinaryOp<(A, B, C, D, E, F, u32)> for Tuple6AsTuple7BinaryOp<Op>
where
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    C: CubePrimitive + CubeElement,
    D: CubePrimitive + CubeElement,
    E: CubePrimitive + CubeElement,
    F: CubePrimitive + CubeElement,
    Op: BinaryOp<(A, B, C, D, E, F)>,
{
    fn apply(
        lhs: (A, B, C, D, E, F, u32),
        rhs: (A, B, C, D, E, F, u32),
    ) -> (A, B, C, D, E, F, u32) {
        let out = Op::apply(
            (lhs.0, lhs.1, lhs.2, lhs.3, lhs.4, lhs.5),
            (rhs.0, rhs.1, rhs.2, rhs.3, rhs.4, rhs.5),
        );
        (out.0, out.1, out.2, out.3, out.4, out.5, lhs.6)
    }
}
