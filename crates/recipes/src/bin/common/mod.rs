#![allow(dead_code)]

use cubecl::frontend::PartialEqExpand;
use cubecl::prelude::*;
use massively::{BinaryPredicateOp, ReductionOp, UnaryOp};

pub type Result<T = ()> = std::result::Result<T, massively::Error>;

pub struct U32Flag;

#[cubecl::cube]
impl UnaryOp<u32> for U32Flag {
    type Output = u32;

    fn apply(input: u32) -> u32 {
        if input != 0u32 { 1u32 } else { 0u32 }
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
impl ReductionOp<u32> for SumU32 {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        lhs + rhs
    }
}

pub struct SumF32;

#[cubecl::cube]
impl ReductionOp<f32> for SumF32 {
    fn apply(lhs: f32, rhs: f32) -> f32 {
        lhs + rhs
    }
}

pub struct LessU32;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for LessU32 {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs < rhs
    }
}

pub struct LessF32;

#[cubecl::cube]
impl BinaryPredicateOp<f32> for LessF32 {
    fn apply(lhs: f32, rhs: f32) -> bool {
        lhs < rhs
    }
}

pub struct EqualU32;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for EqualU32 {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs == rhs
    }
}

pub struct EqualF32;

#[cubecl::cube]
impl BinaryPredicateOp<f32> for EqualF32 {
    fn apply(lhs: f32, rhs: f32) -> bool {
        lhs == rhs
    }
}
