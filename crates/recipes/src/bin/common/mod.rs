#![allow(dead_code)]

use cubecl::prelude::*;
use massively::op::{BinaryOp1, PredicateOp2};

pub type Result<T = ()> = std::result::Result<T, massively::Error>;

pub fn assert_f32_near(actual: f32, expected: f32, tolerance: f32) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual}, expected={expected}, tolerance={tolerance}"
    );
}

pub struct SumU32;

#[cubecl::cube]
impl<B> BinaryOp1<B, (u32,)> for SumU32
where
    B: massively::Backend,
{
    fn apply(lhs: (u32,), rhs: (u32,)) -> (u32,) {
        (lhs.0 + rhs.0,)
    }
}

pub struct SumF32;

#[cubecl::cube]
impl<B> BinaryOp1<B, (f32,)> for SumF32
where
    B: massively::Backend,
{
    fn apply(lhs: (f32,), rhs: (f32,)) -> (f32,) {
        (lhs.0 + rhs.0,)
    }
}

pub struct LessU32;

#[cubecl::cube]
impl<B> PredicateOp2<B, (u32,)> for LessU32
where
    B: massively::Backend,
{
    fn apply(lhs: (u32,), rhs: (u32,)) -> bool {
        lhs.0 < rhs.0
    }
}

pub struct LessF32;

#[cubecl::cube]
impl<B> PredicateOp2<B, (f32,)> for LessF32
where
    B: massively::Backend,
{
    fn apply(lhs: (f32,), rhs: (f32,)) -> bool {
        lhs.0 < rhs.0
    }
}

pub struct EqualU32;

#[cubecl::cube]
impl<B> PredicateOp2<B, (u32,)> for EqualU32
where
    B: massively::Backend,
{
    fn apply(lhs: (u32,), rhs: (u32,)) -> bool {
        lhs.0 == rhs.0
    }
}

pub struct EqualF32;

#[cubecl::cube]
impl<B> PredicateOp2<B, (f32,)> for EqualF32
where
    B: massively::Backend,
{
    fn apply(lhs: (f32,), rhs: (f32,)) -> bool {
        lhs.0 == rhs.0
    }
}
