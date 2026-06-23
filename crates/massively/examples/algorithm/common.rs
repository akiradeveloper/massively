#![allow(dead_code)]

use cubecl::prelude::*;
use massively::op::{BinaryOp, BinaryPredicateOp, PredicateOp, ReductionOp, UnaryOp};

pub type Result = std::result::Result<(), massively::Error>;

pub struct AddOne;

#[cubecl::cube]
impl<B> UnaryOp<B, (f32,)> for AddOne
where
    B: massively::Backend,
{
    type Output = (f32,);

    fn apply(input: (f32,)) -> (f32,) {
        (input.0 + 1.0,)
    }
}

pub struct Square;

#[cubecl::cube]
impl<B> UnaryOp<B, (f32,)> for Square
where
    B: massively::Backend,
{
    type Output = (f32,);

    fn apply(input: (f32,)) -> (f32,) {
        (input.0 * input.0,)
    }
}

pub struct PairProduct;

#[cubecl::cube]
impl<B> BinaryOp<B, (f32,), (f32,)> for PairProduct
where
    B: massively::Backend,
{
    type Output = (f32,);

    fn apply(lhs: (f32,), rhs: (f32,)) -> (f32,) {
        (lhs.0 * rhs.0,)
    }
}

pub struct SumF32;

#[cubecl::cube]
impl<B> ReductionOp<B, (f32,)> for SumF32
where
    B: massively::Backend,
{
    fn apply(lhs: (f32,), rhs: (f32,)) -> (f32,) {
        (lhs.0 + rhs.0,)
    }
}

pub struct TupleSumF32;

#[cubecl::cube]
impl<B> ReductionOp<B, (f32,)> for TupleSumF32
where
    B: massively::Backend,
{
    fn apply(lhs: (f32,), rhs: (f32,)) -> (f32,) {
        (lhs.0 + rhs.0,)
    }
}

pub struct SumU32;

#[cubecl::cube]
impl<B> ReductionOp<B, (u32,)> for SumU32
where
    B: massively::Backend,
{
    fn apply(lhs: (u32,), rhs: (u32,)) -> (u32,) {
        (lhs.0 + rhs.0,)
    }
}

pub struct MulF32;

#[cubecl::cube]
impl<B> ReductionOp<B, (f32,)> for MulF32
where
    B: massively::Backend,
{
    fn apply(lhs: (f32,), rhs: (f32,)) -> (f32,) {
        (lhs.0 * rhs.0,)
    }
}

pub struct LessF32;

#[cubecl::cube]
impl<B> BinaryPredicateOp<B, (f32,)> for LessF32
where
    B: massively::Backend,
{
    fn apply(lhs: (f32,), rhs: (f32,)) -> bool {
        lhs.0 < rhs.0
    }
}

pub struct LessU32;

#[cubecl::cube]
impl<B> BinaryPredicateOp<B, (u32,)> for LessU32
where
    B: massively::Backend,
{
    fn apply(lhs: (u32,), rhs: (u32,)) -> bool {
        lhs.0 < rhs.0
    }
}

pub struct EqualF32;

#[cubecl::cube]
impl<B> BinaryPredicateOp<B, (f32,)> for EqualF32
where
    B: massively::Backend,
{
    fn apply(lhs: (f32,), rhs: (f32,)) -> bool {
        lhs.0 == rhs.0
    }
}

pub struct EqualU32;

#[cubecl::cube]
impl<B> BinaryPredicateOp<B, (u32,)> for EqualU32
where
    B: massively::Backend,
{
    fn apply(lhs: (u32,), rhs: (u32,)) -> bool {
        lhs.0 == rhs.0
    }
}

pub struct Positive;

#[cubecl::cube]
impl<B> PredicateOp<B, (f32,)> for Positive
where
    B: massively::Backend,
{
    fn apply(input: (f32,)) -> bool {
        input.0 > 0.0
    }
}

pub struct GreaterThanTwo;

#[cubecl::cube]
impl<B> PredicateOp<B, (f32,)> for GreaterThanTwo
where
    B: massively::Backend,
{
    fn apply(input: (f32,)) -> bool {
        input.0 > 2.0
    }
}

pub struct EvenU32;

#[cubecl::cube]
impl<B> PredicateOp<B, (u32,)> for EvenU32
where
    B: massively::Backend,
{
    fn apply(input: (u32,)) -> bool {
        input.0 % 2 == 0
    }
}
