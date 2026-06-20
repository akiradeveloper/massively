#![allow(dead_code, unused_imports)]

pub(crate) use cubecl::prelude::*;
pub(crate) use massively::op::{BinaryOp, BinaryPredicateOp, PredicateOp, UnaryOp};
pub(crate) use massively::{
    CubeWgpu, adjacent_difference, adjacent_find, copy_if, count_if, equal, equal_range,
    exclusive_scan, exclusive_scan_by_key, find_first_of, find_if, gather, gather_if,
    inclusive_scan, inclusive_scan_by_key, inner_product, is_partitioned, is_sorted,
    is_sorted_until, lexicographical_compare, lower_bound, max_element, merge, merge_by_key,
    min_element, minmax_element, mismatch, partition, reduce, reduce_by_key, remove_if, replace_if,
    reverse, scatter, scatter_if, set_difference, set_intersection, set_union, sort, sort_by_key,
    transform, unique, unique_by_key, upper_bound,
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
impl BinaryOp<(f32,)> for Sum {
    fn apply(lhs: (f32,), rhs: (f32,)) -> (f32,) {
        (lhs.0 + rhs.0,)
    }
}

#[cubecl::cube]
impl BinaryOp<u32> for Sum {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        lhs + rhs
    }
}

#[cubecl::cube]
impl BinaryOp<(u32,)> for Sum {
    fn apply(lhs: (u32,), rhs: (u32,)) -> (u32,) {
        (lhs.0 + rhs.0,)
    }
}

pub(crate) struct TupleSum;

#[cubecl::cube]
impl BinaryOp<(f32,)> for TupleSum {
    fn apply(lhs: (f32,), rhs: (f32,)) -> (f32,) {
        (lhs.0 + rhs.0,)
    }
}

#[cubecl::cube]
impl BinaryOp<(f32, u32)> for TupleSum {
    fn apply(lhs: (f32, u32), rhs: (f32, u32)) -> (f32, u32) {
        (lhs.0 + rhs.0, lhs.1 + rhs.1)
    }
}

#[cubecl::cube]
impl BinaryOp<(f32, u32, f32)> for TupleSum {
    fn apply(lhs: (f32, u32, f32), rhs: (f32, u32, f32)) -> (f32, u32, f32) {
        (lhs.0 + rhs.0, lhs.1 + rhs.1, lhs.2 + rhs.2)
    }
}

pub(crate) struct Less;

#[cubecl::cube]
impl BinaryPredicateOp<f32> for Less {
    fn apply(lhs: f32, rhs: f32) -> bool {
        lhs < rhs
    }
}

#[cubecl::cube]
impl BinaryPredicateOp<(f32,)> for Less {
    fn apply(lhs: (f32,), rhs: (f32,)) -> bool {
        lhs.0 < rhs.0
    }
}

pub(crate) struct LessU32;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for LessU32 {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs < rhs
    }
}

#[cubecl::cube]
impl BinaryPredicateOp<(u32,)> for LessU32 {
    fn apply(lhs: (u32,), rhs: (u32,)) -> bool {
        lhs.0 < rhs.0
    }
}

pub(crate) struct MixedTupleLess;

#[cubecl::cube]
impl BinaryPredicateOp<(f32, u32)> for MixedTupleLess {
    fn apply(lhs: (f32, u32), rhs: (f32, u32)) -> bool {
        lhs.0 < rhs.0 || (lhs.0 == rhs.0 && lhs.1 < rhs.1)
    }
}

pub(crate) struct MixedTupleEqual;

#[cubecl::cube]
impl BinaryPredicateOp<(f32, u32)> for MixedTupleEqual {
    fn apply(lhs: (f32, u32), rhs: (f32, u32)) -> bool {
        lhs.0 == rhs.0 && lhs.1 == rhs.1
    }
}

pub(crate) struct MixedTupleFirstEqual;

#[cubecl::cube]
impl BinaryPredicateOp<(f32, u32)> for MixedTupleFirstEqual {
    fn apply(lhs: (f32, u32), rhs: (f32, u32)) -> bool {
        lhs.0 == rhs.0
    }
}

pub(crate) struct MixedTuple3Less;

#[cubecl::cube]
impl BinaryPredicateOp<(f32, u32, f32)> for MixedTuple3Less {
    fn apply(lhs: (f32, u32, f32), rhs: (f32, u32, f32)) -> bool {
        lhs.1 < rhs.1 || (lhs.1 == rhs.1 && lhs.0 < rhs.0)
    }
}

pub(crate) struct MixedTuple3Equal;

#[cubecl::cube]
impl BinaryPredicateOp<(f32, u32, f32)> for MixedTuple3Equal {
    fn apply(lhs: (f32, u32, f32), rhs: (f32, u32, f32)) -> bool {
        lhs.0 == rhs.0 && lhs.1 == rhs.1 && lhs.2 == rhs.2
    }
}

pub(crate) struct MixedTuple3FirstEqual;

#[cubecl::cube]
impl BinaryPredicateOp<(f32, u32, f32)> for MixedTuple3FirstEqual {
    fn apply(lhs: (f32, u32, f32), rhs: (f32, u32, f32)) -> bool {
        lhs.0 == rhs.0
    }
}

pub(crate) struct Tuple4Less;

#[cubecl::cube]
impl BinaryPredicateOp<(f32, f32, f32, f32)> for Tuple4Less {
    fn apply(lhs: (f32, f32, f32, f32), rhs: (f32, f32, f32, f32)) -> bool {
        lhs.0 < rhs.0 || (lhs.0 == rhs.0 && lhs.1 < rhs.1)
    }
}

pub(crate) struct Tuple4Equal;

#[cubecl::cube]
impl BinaryPredicateOp<(f32, f32, f32, f32)> for Tuple4Equal {
    fn apply(lhs: (f32, f32, f32, f32), rhs: (f32, f32, f32, f32)) -> bool {
        lhs.0 == rhs.0 && lhs.1 == rhs.1 && lhs.2 == rhs.2 && lhs.3 == rhs.3
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

pub(crate) struct Tuple12MixedLess;

#[cubecl::cube]
impl BinaryPredicateOp<(f32, u32, f32, u32, f32, u32, f32, u32, f32, u32, f32, u32)>
    for Tuple12MixedLess
{
    fn apply(
        lhs: (f32, u32, f32, u32, f32, u32, f32, u32, f32, u32, f32, u32),
        rhs: (f32, u32, f32, u32, f32, u32, f32, u32, f32, u32, f32, u32),
    ) -> bool {
        lhs.0 < rhs.0 || (lhs.0 == rhs.0 && lhs.1 < rhs.1)
    }
}

pub(crate) struct Tuple12MixedTailLess;

#[cubecl::cube]
impl BinaryPredicateOp<(f32, u32, f32, u32, f32, u32, f32, u32, f32, u32, f32, u32)>
    for Tuple12MixedTailLess
{
    fn apply(
        lhs: (f32, u32, f32, u32, f32, u32, f32, u32, f32, u32, f32, u32),
        rhs: (f32, u32, f32, u32, f32, u32, f32, u32, f32, u32, f32, u32),
    ) -> bool {
        lhs.11 < rhs.11 || (lhs.11 == rhs.11 && lhs.10 < rhs.10)
    }
}

pub(crate) struct EqualU32;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for EqualU32 {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs == rhs
    }
}

#[cubecl::cube]
impl BinaryPredicateOp<(u32,)> for EqualU32 {
    fn apply(lhs: (u32,), rhs: (u32,)) -> bool {
        lhs.0 == rhs.0
    }
}

pub(crate) struct SameParityU32;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for SameParityU32 {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs % 2 == rhs % 2
    }
}

#[cubecl::cube]
impl BinaryPredicateOp<(u32,)> for SameParityU32 {
    fn apply(lhs: (u32,), rhs: (u32,)) -> bool {
        lhs.0 % 2 == rhs.0 % 2
    }
}

pub(crate) struct NeverEqualU32;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for NeverEqualU32 {
    fn apply(lhs: u32, _rhs: u32) -> bool {
        lhs != lhs
    }
}

#[cubecl::cube]
impl BinaryPredicateOp<(u32,)> for NeverEqualU32 {
    fn apply(lhs: (u32,), _rhs: (u32,)) -> bool {
        lhs.0 != lhs.0
    }
}

pub(crate) struct Tuple12MixedEqual;

#[cubecl::cube]
impl BinaryPredicateOp<(f32, u32, f32, u32, f32, u32, f32, u32, f32, u32, f32, u32)>
    for Tuple12MixedEqual
{
    fn apply(
        lhs: (f32, u32, f32, u32, f32, u32, f32, u32, f32, u32, f32, u32),
        rhs: (f32, u32, f32, u32, f32, u32, f32, u32, f32, u32, f32, u32),
    ) -> bool {
        lhs.0 == rhs.0 && lhs.1 == rhs.1 && lhs.10 == rhs.10 && lhs.11 == rhs.11
    }
}

pub(crate) struct Double;

#[cubecl::cube]
impl UnaryOp<(f32,)> for Double {
    type Output = (f32,);

    fn apply(input: (f32,)) -> (f32,) {
        (input.0 * 2.0,)
    }
}

pub(crate) struct ScalarToTuple5Mixed;

#[cubecl::cube]
impl UnaryOp<(f32,)> for ScalarToTuple5Mixed {
    type Output = (f32, u32, f32, u32, f32);

    fn apply(input: (f32,)) -> (f32, u32, f32, u32, f32) {
        (
            input.0 + 1.0,
            input.0 as u32 + 2,
            input.0 + 3.0,
            input.0 as u32 + 4,
            input.0 + 5.0,
        )
    }
}

pub(crate) struct ScalarToTuple12Mixed;

#[cubecl::cube]
impl UnaryOp<(u32,)> for ScalarToTuple12Mixed {
    type Output = (f32, u32, f32, u32, f32, u32, f32, u32, f32, u32, f32, u32);

    fn apply(input: (u32,)) -> (f32, u32, f32, u32, f32, u32, f32, u32, f32, u32, f32, u32) {
        (
            input.0 as f32 + 1.0,
            input.0 + 2,
            input.0 as f32 + 3.0,
            input.0 + 4,
            input.0 as f32 + 5.0,
            input.0 + 6,
            input.0 as f32 + 7.0,
            input.0 + 8,
            input.0 as f32 + 9.0,
            input.0 + 10,
            input.0 as f32 + 11.0,
            input.0 + 12,
        )
    }
}

pub(crate) struct GreaterThanFour;

#[cubecl::cube]
impl PredicateOp<f32> for GreaterThanFour {
    fn apply(input: f32) -> bool {
        input > 4.0
    }
}

pub(crate) struct F32GreaterThanOne;

#[cubecl::cube]
impl PredicateOp<f32> for F32GreaterThanOne {
    fn apply(input: f32) -> bool {
        input > 1.0
    }
}

#[cubecl::cube]
impl PredicateOp<(f32,)> for F32GreaterThanOne {
    fn apply(input: (f32,)) -> bool {
        input.0 > 1.0
    }
}

pub(crate) struct NonZero;

#[cubecl::cube]
impl PredicateOp<u32> for NonZero {
    fn apply(input: u32) -> bool {
        input != 0
    }
}

#[cubecl::cube]
impl PredicateOp<(u32,)> for NonZero {
    fn apply(input: (u32,)) -> bool {
        input.0 != 0
    }
}

#[cubecl::cube]
impl PredicateOp<f32> for NonZero {
    fn apply(input: f32) -> bool {
        input != 0.0
    }
}

pub(crate) struct U32IsTwenty;

#[cubecl::cube]
impl PredicateOp<u32> for U32IsTwenty {
    fn apply(input: u32) -> bool {
        input == 20
    }
}

#[cubecl::cube]
impl PredicateOp<(u32,)> for U32IsTwenty {
    fn apply(input: (u32,)) -> bool {
        input.0 == 20
    }
}

pub(crate) struct MixedStencilKeep;

#[cubecl::cube]
impl PredicateOp<(f32, u32)> for MixedStencilKeep {
    fn apply(input: (f32, u32)) -> bool {
        input.0 > 1.0 && input.1 != 0
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

pub(crate) struct Tuple12MixedTailPredicate;

#[cubecl::cube]
impl PredicateOp<(f32, u32, f32, u32, f32, u32, f32, u32, f32, u32, f32, u32)>
    for Tuple12MixedTailPredicate
{
    fn apply(input: (f32, u32, f32, u32, f32, u32, f32, u32, f32, u32, f32, u32)) -> bool {
        input.1 >= 20 && input.10 > 750.0 && input.11 != 9000
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

pub(crate) struct Tuple4MixedSplit;

#[cubecl::cube]
impl UnaryOp<(f32, u32, f32, u32)> for Tuple4MixedSplit {
    type Output = (f32, u32, f32, u32);

    fn apply(input: (f32, u32, f32, u32)) -> (f32, u32, f32, u32) {
        (
            input.0 + input.2,
            input.1 + 2,
            input.2 + input.0,
            input.3 + 4,
        )
    }
}

pub(crate) struct Tuple3To5MixedSplit;

#[cubecl::cube]
impl UnaryOp<(f32, u32, f32)> for Tuple3To5MixedSplit {
    type Output = (f32, u32, f32, u32, f32);

    fn apply(input: (f32, u32, f32)) -> (f32, u32, f32, u32, f32) {
        (
            input.0 + input.2,
            input.1 + 10,
            input.2 - input.0,
            input.1 + 20,
            input.0 * input.2,
        )
    }
}

pub(crate) struct Tuple5To3MixedSplit;

#[cubecl::cube]
impl UnaryOp<(f32, u32, f32, u32, f32)> for Tuple5To3MixedSplit {
    type Output = (f32, u32, f32);

    fn apply(input: (f32, u32, f32, u32, f32)) -> (f32, u32, f32) {
        (
            input.0 + input.2 + input.4,
            input.1 + input.3,
            input.4 - input.0,
        )
    }
}

pub(crate) struct Tuple2To12MixedExpand;

#[cubecl::cube]
impl UnaryOp<(f32, u32)> for Tuple2To12MixedExpand {
    type Output = (f32, u32, f32, u32, f32, u32, f32, u32, f32, u32, f32, u32);

    fn apply(input: (f32, u32)) -> (f32, u32, f32, u32, f32, u32, f32, u32, f32, u32, f32, u32) {
        (
            input.0 + 1.0,
            input.1 + 2,
            input.0 + 3.0,
            input.1 + 4,
            input.0 + 5.0,
            input.1 + 6,
            input.0 + 7.0,
            input.1 + 8,
            input.0 + 9.0,
            input.1 + 10,
            input.0 + 11.0,
            input.1 + 12,
        )
    }
}

pub(crate) struct Tuple12To2MixedProject;

#[cubecl::cube]
impl UnaryOp<(f32, u32, f32, u32, f32, u32, f32, u32, f32, u32, f32, u32)>
    for Tuple12To2MixedProject
{
    type Output = (f32, u32);

    fn apply(input: (f32, u32, f32, u32, f32, u32, f32, u32, f32, u32, f32, u32)) -> (f32, u32) {
        (input.0 + input.2 + input.10, input.1 + input.5 + input.11)
    }
}

pub(crate) struct TupleWideMixedSplit;

macro_rules! impl_tuple_wide_mixed_split {
    (
        $input:ident,
        ($($ty:ty),+),
        ($($out:expr),+)
    ) => {
        #[cubecl::cube]
        impl UnaryOp<($($ty),+)> for TupleWideMixedSplit {
            type Output = ($($ty),+);

            fn apply($input: ($($ty),+)) -> ($($ty),+) {
                ($($out),+)
            }
        }
    };
}

impl_tuple_wide_mixed_split!(
    input,
    (f32, u32, f32, u32, f32),
    (
        input.0 + 1.0,
        input.1 + 2,
        input.2 + 3.0,
        input.3 + 4,
        input.4 + 5.0
    )
);
impl_tuple_wide_mixed_split!(
    input,
    (f32, u32, f32, u32, f32, u32),
    (
        input.0 + 1.0,
        input.1 + 2,
        input.2 + 3.0,
        input.3 + 4,
        input.4 + 5.0,
        input.5 + 6
    )
);
impl_tuple_wide_mixed_split!(
    input,
    (f32, u32, f32, u32, f32, u32, f32),
    (
        input.0 + 1.0,
        input.1 + 2,
        input.2 + 3.0,
        input.3 + 4,
        input.4 + 5.0,
        input.5 + 6,
        input.6 + 7.0
    )
);
impl_tuple_wide_mixed_split!(
    input,
    (f32, u32, f32, u32, f32, u32, f32, u32),
    (
        input.0 + 1.0,
        input.1 + 2,
        input.2 + 3.0,
        input.3 + 4,
        input.4 + 5.0,
        input.5 + 6,
        input.6 + 7.0,
        input.7 + 8
    )
);
impl_tuple_wide_mixed_split!(
    input,
    (f32, u32, f32, u32, f32, u32, f32, u32, f32),
    (
        input.0 + 1.0,
        input.1 + 2,
        input.2 + 3.0,
        input.3 + 4,
        input.4 + 5.0,
        input.5 + 6,
        input.6 + 7.0,
        input.7 + 8,
        input.8 + 9.0
    )
);
impl_tuple_wide_mixed_split!(
    input,
    (f32, u32, f32, u32, f32, u32, f32, u32, f32, u32),
    (
        input.0 + 1.0,
        input.1 + 2,
        input.2 + 3.0,
        input.3 + 4,
        input.4 + 5.0,
        input.5 + 6,
        input.6 + 7.0,
        input.7 + 8,
        input.8 + 9.0,
        input.9 + 10
    )
);
impl_tuple_wide_mixed_split!(
    input,
    (f32, u32, f32, u32, f32, u32, f32, u32, f32, u32, f32),
    (
        input.0 + 1.0,
        input.1 + 2,
        input.2 + 3.0,
        input.3 + 4,
        input.4 + 5.0,
        input.5 + 6,
        input.6 + 7.0,
        input.7 + 8,
        input.8 + 9.0,
        input.9 + 10,
        input.10 + 11.0
    )
);

pub(crate) struct Tuple12MixedSplit;

#[cubecl::cube]
impl UnaryOp<(f32, u32, f32, u32, f32, u32, f32, u32, f32, u32, f32, u32)> for Tuple12MixedSplit {
    type Output = (f32, u32, f32, u32, f32, u32, f32, u32, f32, u32, f32, u32);

    fn apply(
        input: (f32, u32, f32, u32, f32, u32, f32, u32, f32, u32, f32, u32),
    ) -> (f32, u32, f32, u32, f32, u32, f32, u32, f32, u32, f32, u32) {
        (
            input.0 + 1.0,
            input.1 + 2,
            input.2 + 3.0,
            input.3 + 4,
            input.4 + 5.0,
            input.5 + 6,
            input.6 + 7.0,
            input.7 + 8,
            input.8 + 9.0,
            input.9 + 10,
            input.10 + 11.0,
            input.11 + 12,
        )
    }
}

pub(crate) struct Tuple12MixedChecksum;

#[cubecl::cube]
impl UnaryOp<(f32, u32, f32, u32, f32, u32, f32, u32, f32, u32, f32, u32)>
    for Tuple12MixedChecksum
{
    type Output = (f32,);

    fn apply(input: (f32, u32, f32, u32, f32, u32, f32, u32, f32, u32, f32, u32)) -> (f32,) {
        (input.0
            + input.1 as f32
            + input.2
            + input.3 as f32
            + input.4
            + input.5 as f32
            + input.6
            + input.7 as f32
            + input.8
            + input.9 as f32
            + input.10
            + input.11 as f32,)
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
