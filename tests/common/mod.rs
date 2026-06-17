#![allow(dead_code, unused_imports)]

pub(crate) use massively::op::{BinaryOp, BinaryPredicateOp, PredicateOp, UnaryOp};
pub(crate) use massively::{
    CubeWgpu, copy_if, count_if, exclusive_scan, exclusive_scan_by_key, find_if, gather, gather_if,
    inclusive_scan, inclusive_scan_by_key, reduce, reduce_by_key, remove_if, scatter, scatter_if,
    sort, sort_by_key, transform, unique_by_key, unzip, vzip, vzip3, vzip4, vzip12, zip, zip3,
    zip4, zip12,
};

pub(crate) fn policy() -> CubeWgpu {
    CubeWgpu::cpu()
}

pub(crate) struct Sum;

#[cubecl::cube]
impl BinaryOp<f32> for Sum {
    fn apply(lhs: f32, rhs: f32) -> f32 {
        lhs + rhs
    }
}

#[cubecl::cube]
impl BinaryOp<u32> for Sum {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        lhs + rhs
    }
}

pub(crate) struct Less;

#[cubecl::cube]
impl BinaryPredicateOp<f32> for Less {
    fn apply(lhs: f32, rhs: f32) -> bool {
        lhs < rhs
    }
}

pub(crate) struct LessU32;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for LessU32 {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs < rhs
    }
}

pub(crate) struct MixedTupleLess;

#[cubecl::cube]
impl BinaryPredicateOp<(f32, u32)> for MixedTupleLess {
    fn apply(lhs: (f32, u32), rhs: (f32, u32)) -> bool {
        lhs.0 < rhs.0 || (lhs.0 == rhs.0 && lhs.1 < rhs.1)
    }
}

pub(crate) struct MixedTuple3Less;

#[cubecl::cube]
impl BinaryPredicateOp<(f32, u32, f32)> for MixedTuple3Less {
    fn apply(lhs: (f32, u32, f32), rhs: (f32, u32, f32)) -> bool {
        lhs.1 < rhs.1 || (lhs.1 == rhs.1 && lhs.0 < rhs.0)
    }
}

pub(crate) struct Tuple4Less;

#[cubecl::cube]
impl BinaryPredicateOp<(f32, f32, f32, f32)> for Tuple4Less {
    fn apply(lhs: (f32, f32, f32, f32), rhs: (f32, f32, f32, f32)) -> bool {
        lhs.0 < rhs.0 || (lhs.0 == rhs.0 && lhs.1 < rhs.1)
    }
}

pub(crate) struct Tuple12Less;

#[cubecl::cube]
impl BinaryPredicateOp<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>
    for Tuple12Less
{
    fn apply(
        lhs: (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32),
        rhs: (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32),
    ) -> bool {
        lhs.0 < rhs.0
    }
}

pub(crate) struct EqualU32;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for EqualU32 {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs == rhs
    }
}

pub(crate) struct Double;

#[cubecl::cube]
impl UnaryOp<f32> for Double {
    type Output = f32;

    fn apply(input: f32) -> f32 {
        input * 2.0
    }
}

pub(crate) struct GreaterThanFour;

#[cubecl::cube]
impl PredicateOp<f32> for GreaterThanFour {
    fn apply(input: f32) -> bool {
        input > 4.0
    }
}

pub(crate) struct NonZero;

#[cubecl::cube]
impl PredicateOp<u32> for NonZero {
    fn apply(input: u32) -> bool {
        input != 0
    }
}

pub(crate) struct Tuple12MixedFirstGreaterThanOne;

#[cubecl::cube]
impl PredicateOp<(f32, u32, f32, u32, f32, u32, f32, u32, f32, u32, f32, u32)>
    for Tuple12MixedFirstGreaterThanOne
{
    fn apply(input: (f32, u32, f32, u32, f32, u32, f32, u32, f32, u32, f32, u32)) -> bool {
        input.0 > 1.0
    }
}

pub(crate) struct PairMixedSplit;

#[cubecl::cube]
impl UnaryOp<(f32, u32)> for PairMixedSplit {
    type Output = (f32, u32);

    fn apply(input: (f32, u32)) -> (f32, u32) {
        (input.0 + 10.0, input.1 + 1)
    }
}

pub(crate) struct PairScaleAndTag;

#[cubecl::cube]
impl UnaryOp<(f32, u32)> for PairScaleAndTag {
    type Output = (f32, u32);

    fn apply(input: (f32, u32)) -> (f32, u32) {
        (input.0 * 2.0, input.1 + 1)
    }
}

pub(crate) struct Tuple3MixedSplit;

#[cubecl::cube]
impl UnaryOp<(f32, u32, f32)> for Tuple3MixedSplit {
    type Output = (f32, u32, f32);

    fn apply(input: (f32, u32, f32)) -> (f32, u32, f32) {
        (input.0 + input.2, input.1 + 1, input.2 + input.0)
    }
}

pub(crate) struct PairMixedFirstPositive;

#[cubecl::cube]
impl PredicateOp<(f32, u32)> for PairMixedFirstPositive {
    fn apply(input: (f32, u32)) -> bool {
        input.0 > 0.0
    }
}

pub(crate) struct PairMixedTagIsTwenty;

#[cubecl::cube]
impl PredicateOp<(f32, u32)> for PairMixedTagIsTwenty {
    fn apply(input: (f32, u32)) -> bool {
        input.1 == 20
    }
}

pub(crate) struct Tuple3MixedTagIsTwenty;

#[cubecl::cube]
impl PredicateOp<(f32, u32, f32)> for Tuple3MixedTagIsTwenty {
    fn apply(input: (f32, u32, f32)) -> bool {
        input.1 == 20 && input.2 > 0.0
    }
}

pub(crate) struct Tuple3MixedFirstPositive;

#[cubecl::cube]
impl PredicateOp<(f32, u32, f32)> for Tuple3MixedFirstPositive {
    fn apply(input: (f32, u32, f32)) -> bool {
        input.0 > 0.0
    }
}
