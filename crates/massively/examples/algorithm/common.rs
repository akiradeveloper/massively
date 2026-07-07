#![allow(dead_code)]

use cubecl::prelude::*;
use massively::op::{BinaryOp, BinaryPredicateOp, PredicateOp, ReductionOp, UnaryOp};

pub type Result = std::result::Result<(), massively::Error>;

pub fn bool_stencil<R, Op>(len: massively::MIndex, op: Op) -> impl massively::MIter<R, Item = bool>
where
    R: cubecl::prelude::Runtime,
    Op: UnaryOp<R, u32, Output = bool>,
{
    massively::lazy::transform(massively::lazy::counting(0).take(len), op)
}

pub struct IndexOdd;

#[cubecl::cube]
impl<R> UnaryOp<R, u32> for IndexOdd
where
    R: cubecl::prelude::Runtime,
{
    type Output = bool;

    fn apply(input: u32) -> bool {
        input % 2 == 1
    }
}

pub struct IndexNot1;

#[cubecl::cube]
impl<R> UnaryOp<R, u32> for IndexNot1
where
    R: cubecl::prelude::Runtime,
{
    type Output = bool;

    fn apply(input: u32) -> bool {
        input != 1
    }
}

pub struct IndexLt2;

#[cubecl::cube]
impl<R> UnaryOp<R, u32> for IndexLt2
where
    R: cubecl::prelude::Runtime,
{
    type Output = bool;

    fn apply(input: u32) -> bool {
        input < 2
    }
}

pub struct AddOne;

#[cubecl::cube]
impl<R> UnaryOp<R, (f32,)> for AddOne
where
    R: cubecl::prelude::Runtime,
{
    type Output = (f32,);

    fn apply(input: (f32,)) -> (f32,) {
        (input.0 + 1.0,)
    }
}

pub struct Square;

#[cubecl::cube]
impl<R> UnaryOp<R, (f32,)> for Square
where
    R: cubecl::prelude::Runtime,
{
    type Output = (f32,);

    fn apply(input: (f32,)) -> (f32,) {
        (input.0 * input.0,)
    }
}

pub struct PairProduct;

#[cubecl::cube]
impl<R> BinaryOp<R, (f32,), (f32,)> for PairProduct
where
    R: cubecl::prelude::Runtime,
{
    type Output = (f32,);

    fn apply(lhs: (f32,), rhs: (f32,)) -> (f32,) {
        (lhs.0 * rhs.0,)
    }
}

pub struct SumF32;

#[cubecl::cube]
impl<R> ReductionOp<R, (f32,)> for SumF32
where
    R: cubecl::prelude::Runtime,
{
    fn apply(lhs: (f32,), rhs: (f32,)) -> (f32,) {
        (lhs.0 + rhs.0,)
    }
}

pub struct TupleSumF32;

#[cubecl::cube]
impl<R> ReductionOp<R, (f32,)> for TupleSumF32
where
    R: cubecl::prelude::Runtime,
{
    fn apply(lhs: (f32,), rhs: (f32,)) -> (f32,) {
        (lhs.0 + rhs.0,)
    }
}

pub struct SumU32;

#[cubecl::cube]
impl<R> ReductionOp<R, (u32,)> for SumU32
where
    R: cubecl::prelude::Runtime,
{
    fn apply(lhs: (u32,), rhs: (u32,)) -> (u32,) {
        (lhs.0 + rhs.0,)
    }
}

pub struct MulF32;

#[cubecl::cube]
impl<R> ReductionOp<R, (f32,)> for MulF32
where
    R: cubecl::prelude::Runtime,
{
    fn apply(lhs: (f32,), rhs: (f32,)) -> (f32,) {
        (lhs.0 * rhs.0,)
    }
}

pub struct LessF32;

#[cubecl::cube]
impl<R> BinaryPredicateOp<R, (f32,)> for LessF32
where
    R: cubecl::prelude::Runtime,
{
    fn apply(lhs: (f32,), rhs: (f32,)) -> bool {
        lhs.0 < rhs.0
    }
}

pub struct LessU32;

#[cubecl::cube]
impl<R> BinaryPredicateOp<R, (u32,)> for LessU32
where
    R: cubecl::prelude::Runtime,
{
    fn apply(lhs: (u32,), rhs: (u32,)) -> bool {
        lhs.0 < rhs.0
    }
}

pub struct EqualF32;

#[cubecl::cube]
impl<R> BinaryPredicateOp<R, (f32,)> for EqualF32
where
    R: cubecl::prelude::Runtime,
{
    fn apply(lhs: (f32,), rhs: (f32,)) -> bool {
        lhs.0 == rhs.0
    }
}

pub struct EqualU32;

#[cubecl::cube]
impl<R> BinaryPredicateOp<R, (u32,)> for EqualU32
where
    R: cubecl::prelude::Runtime,
{
    fn apply(lhs: (u32,), rhs: (u32,)) -> bool {
        lhs.0 == rhs.0
    }
}

pub struct Positive;

#[cubecl::cube]
impl<R> PredicateOp<R, (f32,)> for Positive
where
    R: cubecl::prelude::Runtime,
{
    fn apply(input: (f32,)) -> bool {
        input.0 > 0.0
    }
}

pub struct GreaterThanTwo;

#[cubecl::cube]
impl<R> PredicateOp<R, (f32,)> for GreaterThanTwo
where
    R: cubecl::prelude::Runtime,
{
    fn apply(input: (f32,)) -> bool {
        input.0 > 2.0
    }
}

pub struct EvenU32;

#[cubecl::cube]
impl<R> PredicateOp<R, (u32,)> for EvenU32
where
    R: cubecl::prelude::Runtime,
{
    fn apply(input: (u32,)) -> bool {
        input.0 % 2 == 0
    }
}
