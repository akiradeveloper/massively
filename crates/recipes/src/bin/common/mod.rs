#![allow(dead_code)]

use cubecl::frontend::PartialEqExpand;
use cubecl::prelude::*;
use massively::op::{BinaryPredicateOp, ReductionOp};

pub type Result<T = ()> = std::result::Result<T, massively::Error>;

pub struct U32Flag;

#[cubecl::cube]
impl<R> massively::op::UnaryOp<R, u32> for U32Flag
where
    R: cubecl::prelude::Runtime,
{
    type Output = bool;

    fn apply(input: u32) -> bool {
        input != 0
    }
}

pub fn assert_f32_near(actual: f32, expected: f32, tolerance: f32) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual}, expected={expected}, tolerance={tolerance}"
    );
}

pub struct SumU32;

#[cubecl::cube]
impl<B> ReductionOp<B, (u32,)> for SumU32
where
    B: cubecl::prelude::Runtime,
{
    fn apply(lhs: (u32,), rhs: (u32,)) -> (u32,) {
        (lhs.0 + rhs.0,)
    }
}

pub struct SumF32;

#[cubecl::cube]
impl<B> ReductionOp<B, (f32,)> for SumF32
where
    B: cubecl::prelude::Runtime,
{
    fn apply(lhs: (f32,), rhs: (f32,)) -> (f32,) {
        (lhs.0 + rhs.0,)
    }
}

pub struct LessU32;

#[cubecl::cube]
impl<B> BinaryPredicateOp<B, (u32,)> for LessU32
where
    B: cubecl::prelude::Runtime,
{
    fn apply(lhs: (u32,), rhs: (u32,)) -> bool {
        lhs.0 < rhs.0
    }
}

pub struct LessF32;

#[cubecl::cube]
impl<B> BinaryPredicateOp<B, (f32,)> for LessF32
where
    B: cubecl::prelude::Runtime,
{
    fn apply(lhs: (f32,), rhs: (f32,)) -> bool {
        lhs.0 < rhs.0
    }
}

pub struct EqualU32;

#[cubecl::cube]
impl<B> BinaryPredicateOp<B, (u32,)> for EqualU32
where
    B: cubecl::prelude::Runtime,
{
    fn apply(lhs: (u32,), rhs: (u32,)) -> bool {
        lhs.0 == rhs.0
    }
}

pub struct EqualF32;

#[cubecl::cube]
impl<B> BinaryPredicateOp<B, (f32,)> for EqualF32
where
    B: cubecl::prelude::Runtime,
{
    fn apply(lhs: (f32,), rhs: (f32,)) -> bool {
        lhs.0 == rhs.0
    }
}
