#![allow(dead_code)]

use massively::op::{BinaryOp, BinaryPredicateOp, PredicateOp, UnaryOp};

pub type Result = std::result::Result<(), massively::Error>;

pub struct AddOne;

#[cubecl::cube]
impl UnaryOp<f32> for AddOne {
    type Output = f32;

    fn apply(input: f32) -> f32 {
        input + 1.0
    }
}

pub struct Square;

#[cubecl::cube]
impl UnaryOp<f32> for Square {
    type Output = f32;

    fn apply(input: f32) -> f32 {
        input * input
    }
}

pub struct PairProduct;

#[cubecl::cube]
impl UnaryOp<(f32, f32)> for PairProduct {
    type Output = f32;

    fn apply(input: (f32, f32)) -> f32 {
        input.0 * input.1
    }
}

pub struct SumF32;

#[cubecl::cube]
impl BinaryOp<f32> for SumF32 {
    fn apply(lhs: f32, rhs: f32) -> f32 {
        lhs + rhs
    }
}

pub struct SumU32;

#[cubecl::cube]
impl BinaryOp<u32> for SumU32 {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        lhs + rhs
    }
}

pub struct MulF32;

#[cubecl::cube]
impl BinaryOp<f32> for MulF32 {
    fn apply(lhs: f32, rhs: f32) -> f32 {
        lhs * rhs
    }
}

pub struct LessF32;

#[cubecl::cube]
impl BinaryPredicateOp<f32> for LessF32 {
    fn apply(lhs: f32, rhs: f32) -> bool {
        lhs < rhs
    }
}

pub struct LessU32;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for LessU32 {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs < rhs
    }
}

pub struct EqualF32;

#[cubecl::cube]
impl BinaryPredicateOp<f32> for EqualF32 {
    fn apply(lhs: f32, rhs: f32) -> bool {
        lhs == rhs
    }
}

pub struct EqualU32;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for EqualU32 {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs == rhs
    }
}

pub struct Positive;

#[cubecl::cube]
impl PredicateOp<f32> for Positive {
    fn apply(input: f32) -> bool {
        input > 0.0
    }
}

pub struct GreaterThanTwo;

#[cubecl::cube]
impl PredicateOp<f32> for GreaterThanTwo {
    fn apply(input: f32) -> bool {
        input > 2.0
    }
}

pub struct EvenU32;

#[cubecl::cube]
impl PredicateOp<u32> for EvenU32 {
    fn apply(input: u32) -> bool {
        input % 2 == 0
    }
}
