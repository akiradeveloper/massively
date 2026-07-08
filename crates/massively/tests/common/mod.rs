#![allow(dead_code, unused_imports)]
pub(crate) use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

pub(crate) use cubecl::prelude::*;
pub(crate) use massively::algorithm::op::{BinaryPredicateOp, PredicateOp, ReductionOp, UnaryOp};
pub(crate) use massively::{
    Executor, MIter, MIterMut, MStorage, adjacent_difference, adjacent_find, copy_where, count_if,
    equal, exclusive_scan, exclusive_scan_by_key, fill, find_first_of, find_if, gather,
    gather_where, inclusive_scan, inclusive_scan_by_key, is_partitioned, is_sorted,
    is_sorted_until, lexicographical_compare, lower_bound, max_element, merge, merge_by_key,
    min_element, minmax_element, mismatch, partition, reduce, reduce_by_key, remove_where,
    replace_where, reverse, scatter, scatter_where, set_difference, set_intersection, set_union,
    sort, sort_by_key, transform, transform_where, unique, unique_by_key, upper_bound,
};

pub(crate) fn exec() -> Executor<WgpuRuntime> {
    Executor::<WgpuRuntime>::new(WgpuDevice::Cpu)
}

pub(crate) struct Sum;

#[cubecl::cube]
impl ReductionOp<WgpuRuntime, f32> for Sum {
    fn apply(lhs: f32, rhs: f32) -> f32 {
        lhs + rhs
    }
}

#[cubecl::cube]
impl ReductionOp<WgpuRuntime, (f32,)> for Sum {
    fn apply(lhs: (f32,), rhs: (f32,)) -> (f32,) {
        (lhs.0 + rhs.0,)
    }
}

#[cubecl::cube]
impl ReductionOp<WgpuRuntime, (u32,)> for Sum {
    fn apply(lhs: (u32,), rhs: (u32,)) -> (u32,) {
        (lhs.0 + rhs.0,)
    }
}

pub(crate) struct MaxU32;

#[cubecl::cube]
impl ReductionOp<WgpuRuntime, (u32,)> for MaxU32 {
    fn apply(lhs: (u32,), rhs: (u32,)) -> (u32,) {
        (lhs.0.max(rhs.0),)
    }
}

pub(crate) struct TupleSum;

#[cubecl::cube]
impl ReductionOp<WgpuRuntime, (f32,)> for TupleSum {
    fn apply(lhs: (f32,), rhs: (f32,)) -> (f32,) {
        (lhs.0 + rhs.0,)
    }
}

#[cubecl::cube]
impl ReductionOp<WgpuRuntime, (f32, u32)> for TupleSum {
    fn apply(lhs: (f32, u32), rhs: (f32, u32)) -> (f32, u32) {
        (lhs.0 + rhs.0, lhs.1 + rhs.1)
    }
}

#[cubecl::cube]
impl ReductionOp<WgpuRuntime, (f32, u32, f32)> for TupleSum {
    fn apply(lhs: (f32, u32, f32), rhs: (f32, u32, f32)) -> (f32, u32, f32) {
        (lhs.0 + rhs.0, lhs.1 + rhs.1, lhs.2 + rhs.2)
    }
}

#[cubecl::cube]
impl ReductionOp<WgpuRuntime, (f32, u32, f32, u32, f32, u32, f32)> for TupleSum {
    fn apply(
        lhs: (f32, u32, f32, u32, f32, u32, f32),
        rhs: (f32, u32, f32, u32, f32, u32, f32),
    ) -> (f32, u32, f32, u32, f32, u32, f32) {
        (
            lhs.0 + rhs.0,
            lhs.1 + rhs.1,
            lhs.2 + rhs.2,
            lhs.3 + rhs.3,
            lhs.4 + rhs.4,
            lhs.5 + rhs.5,
            lhs.6 + rhs.6,
        )
    }
}

pub(crate) struct Less;

#[cubecl::cube]
impl BinaryPredicateOp<WgpuRuntime, (f32,)> for Less {
    fn apply(lhs: (f32,), rhs: (f32,)) -> bool {
        lhs.0 < rhs.0
    }
}

#[cubecl::cube]
impl BinaryPredicateOp<WgpuRuntime, f32> for Less {
    fn apply(lhs: f32, rhs: f32) -> bool {
        lhs < rhs
    }
}

pub(crate) struct EqualF32;

#[cubecl::cube]
impl BinaryPredicateOp<WgpuRuntime, (f32,)> for EqualF32 {
    fn apply(lhs: (f32,), rhs: (f32,)) -> bool {
        lhs.0 == rhs.0
    }
}

#[cubecl::cube]
impl BinaryPredicateOp<WgpuRuntime, f32> for EqualF32 {
    fn apply(lhs: f32, rhs: f32) -> bool {
        lhs == rhs
    }
}

pub(crate) struct LessU32;

#[cubecl::cube]
impl BinaryPredicateOp<WgpuRuntime, (u32,)> for LessU32 {
    fn apply(lhs: (u32,), rhs: (u32,)) -> bool {
        lhs.0 < rhs.0
    }
}

#[cubecl::cube]
impl BinaryPredicateOp<WgpuRuntime, u32> for LessU32 {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs < rhs
    }
}

pub(crate) struct SameLowNibbleU32;

#[cubecl::cube]
impl BinaryPredicateOp<WgpuRuntime, (u32,)> for SameLowNibbleU32 {
    fn apply(lhs: (u32,), rhs: (u32,)) -> bool {
        (lhs.0 & 0x0f) == (rhs.0 & 0x0f)
    }
}

pub(crate) fn bool_stencil<Op>(
    len: massively::MIndex,
    op: Op,
) -> impl MIter<WgpuRuntime, Item = bool>
where
    Op: UnaryOp<WgpuRuntime, u32, Output = bool>,
{
    massively::lazy::transform(massively::lazy::counting(0).take(len), op)
}

pub(crate) fn bool_stencil_from<Op>(
    start: massively::MIndex,
    len: massively::MIndex,
    op: Op,
) -> impl MIter<WgpuRuntime, Item = bool>
where
    Op: UnaryOp<WgpuRuntime, u32, Output = bool>,
{
    massively::lazy::transform(massively::lazy::counting(start).take(len), op)
}

pub(crate) struct IndexNonZero;

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, u32> for IndexNonZero {
    type Output = bool;

    fn apply(input: u32) -> bool {
        input > 0
    }
}

pub(crate) struct IndexGe2;

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, u32> for IndexGe2 {
    type Output = bool;

    fn apply(input: u32) -> bool {
        input >= 2
    }
}

pub(crate) struct IndexEq2;

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, u32> for IndexEq2 {
    type Output = bool;

    fn apply(input: u32) -> bool {
        input == 2
    }
}

pub(crate) struct IndexBetween1And2;

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, u32> for IndexBetween1And2 {
    type Output = bool;

    fn apply(input: u32) -> bool {
        input >= 1 && input <= 2
    }
}

pub(crate) struct IndexEven;

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, u32> for IndexEven {
    type Output = bool;

    fn apply(input: u32) -> bool {
        input % 2 == 0
    }
}

pub(crate) struct IndexOdd;

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, u32> for IndexOdd {
    type Output = bool;

    fn apply(input: u32) -> bool {
        input % 2 == 1
    }
}

pub(crate) struct IndexNot1;

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, u32> for IndexNot1 {
    type Output = bool;

    fn apply(input: u32) -> bool {
        input != 1
    }
}

pub(crate) struct MixedTupleLess;

#[cubecl::cube]
impl BinaryPredicateOp<WgpuRuntime, (f32, u32)> for MixedTupleLess {
    fn apply(lhs: (f32, u32), rhs: (f32, u32)) -> bool {
        lhs.0 < rhs.0 || (lhs.0 == rhs.0 && lhs.1 < rhs.1)
    }
}

pub(crate) struct MixedTupleEqual;

#[cubecl::cube]
impl BinaryPredicateOp<WgpuRuntime, (f32, u32)> for MixedTupleEqual {
    fn apply(lhs: (f32, u32), rhs: (f32, u32)) -> bool {
        lhs.0 == rhs.0 && lhs.1 == rhs.1
    }
}

pub(crate) struct MixedTupleFirstEqual;

#[cubecl::cube]
impl BinaryPredicateOp<WgpuRuntime, (f32, u32)> for MixedTupleFirstEqual {
    fn apply(lhs: (f32, u32), rhs: (f32, u32)) -> bool {
        lhs.0 == rhs.0
    }
}

pub(crate) struct MixedTuple3Less;

#[cubecl::cube]
impl BinaryPredicateOp<WgpuRuntime, (f32, u32, f32)> for MixedTuple3Less {
    fn apply(lhs: (f32, u32, f32), rhs: (f32, u32, f32)) -> bool {
        lhs.1 < rhs.1 || (lhs.1 == rhs.1 && lhs.0 < rhs.0)
    }
}

pub(crate) struct FirstAscSecondDescU32;

#[cubecl::cube]
impl BinaryPredicateOp<WgpuRuntime, (u32, u32)> for FirstAscSecondDescU32 {
    fn apply(lhs: (u32, u32), rhs: (u32, u32)) -> bool {
        lhs.0 < rhs.0 || (lhs.0 == rhs.0 && lhs.1 > rhs.1)
    }
}

pub(crate) struct FirstOnlyLessU32;

#[cubecl::cube]
impl BinaryPredicateOp<WgpuRuntime, (u32, u32)> for FirstOnlyLessU32 {
    fn apply(lhs: (u32, u32), rhs: (u32, u32)) -> bool {
        lhs.0 < rhs.0
    }
}

pub(crate) struct MixedTuple3LexLess;

#[cubecl::cube]
impl BinaryPredicateOp<WgpuRuntime, (f32, u32, f32)> for MixedTuple3LexLess {
    fn apply(lhs: (f32, u32, f32), rhs: (f32, u32, f32)) -> bool {
        lhs.0 < rhs.0 || (lhs.0 == rhs.0 && (lhs.1 < rhs.1 || (lhs.1 == rhs.1 && lhs.2 < rhs.2)))
    }
}

pub(crate) struct MixedTuple3Equal;

#[cubecl::cube]
impl BinaryPredicateOp<WgpuRuntime, (f32, u32, f32)> for MixedTuple3Equal {
    fn apply(lhs: (f32, u32, f32), rhs: (f32, u32, f32)) -> bool {
        lhs.0 == rhs.0 && lhs.1 == rhs.1 && lhs.2 == rhs.2
    }
}

pub(crate) struct MixedTuple3FirstEqual;

#[cubecl::cube]
impl BinaryPredicateOp<WgpuRuntime, (f32, u32, f32)> for MixedTuple3FirstEqual {
    fn apply(lhs: (f32, u32, f32), rhs: (f32, u32, f32)) -> bool {
        lhs.0 == rhs.0
    }
}

pub(crate) struct Tuple4Less;

#[cubecl::cube]
impl BinaryPredicateOp<WgpuRuntime, (f32, f32, f32, f32)> for Tuple4Less {
    fn apply(lhs: (f32, f32, f32, f32), rhs: (f32, f32, f32, f32)) -> bool {
        lhs.0 < rhs.0 || (lhs.0 == rhs.0 && lhs.1 < rhs.1)
    }
}

pub(crate) struct Tuple4Equal;

#[cubecl::cube]
impl BinaryPredicateOp<WgpuRuntime, (f32, f32, f32, f32)> for Tuple4Equal {
    fn apply(lhs: (f32, f32, f32, f32), rhs: (f32, f32, f32, f32)) -> bool {
        lhs.0 == rhs.0 && lhs.1 == rhs.1 && lhs.2 == rhs.2 && lhs.3 == rhs.3
    }
}

pub(crate) struct EqualU32;

#[cubecl::cube]
impl BinaryPredicateOp<WgpuRuntime, (u32,)> for EqualU32 {
    fn apply(lhs: (u32,), rhs: (u32,)) -> bool {
        lhs.0 == rhs.0
    }
}

#[cubecl::cube]
impl BinaryPredicateOp<WgpuRuntime, u32> for EqualU32 {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs == rhs
    }
}

pub(crate) struct SameParityU32;

#[cubecl::cube]
impl BinaryPredicateOp<WgpuRuntime, (u32,)> for SameParityU32 {
    fn apply(lhs: (u32,), rhs: (u32,)) -> bool {
        lhs.0 % 2 == rhs.0 % 2
    }
}

pub(crate) struct GreaterThanF32;

#[cubecl::cube]
impl PredicateOp<WgpuRuntime, (f32, f32)> for GreaterThanF32 {
    fn apply(input: (f32, f32)) -> bool {
        input.0 > input.1
    }
}

pub(crate) struct NeverEqualU32;

#[cubecl::cube]
impl BinaryPredicateOp<WgpuRuntime, (u32,)> for NeverEqualU32 {
    fn apply(lhs: (u32,), _rhs: (u32,)) -> bool {
        lhs.0 != lhs.0
    }
}

pub(crate) struct Tuple7MixedEqual;

#[cubecl::cube]
impl BinaryPredicateOp<WgpuRuntime, (f32, u32, f32, u32, f32, u32, f32)> for Tuple7MixedEqual {
    fn apply(
        lhs: (f32, u32, f32, u32, f32, u32, f32),
        rhs: (f32, u32, f32, u32, f32, u32, f32),
    ) -> bool {
        lhs.0 == rhs.0
            && lhs.1 == rhs.1
            && lhs.2 == rhs.2
            && lhs.3 == rhs.3
            && lhs.4 == rhs.4
            && lhs.5 == rhs.5
            && lhs.6 == rhs.6
    }
}

pub(crate) struct Tuple7MixedLess;

#[cubecl::cube]
impl BinaryPredicateOp<WgpuRuntime, (f32, u32, f32, u32, f32, u32, f32)> for Tuple7MixedLess {
    fn apply(
        lhs: (f32, u32, f32, u32, f32, u32, f32),
        rhs: (f32, u32, f32, u32, f32, u32, f32),
    ) -> bool {
        lhs.1 < rhs.1 || (lhs.1 == rhs.1 && lhs.0 < rhs.0)
    }
}

pub(crate) struct Double;

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, (f32,)> for Double {
    type Output = (f32,);

    fn apply(input: (f32,)) -> (f32,) {
        (input.0 * 2.0,)
    }
}

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, f32> for Double {
    type Output = (f32,);

    fn apply(input: f32) -> (f32,) {
        (input * 2.0,)
    }
}

pub(crate) struct AddOneIndex;

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, u32> for AddOneIndex {
    type Output = u32;

    fn apply(input: u32) -> u32 {
        input + 1
    }
}

pub(crate) struct ScalarToTuple5Mixed;

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, (f32,)> for ScalarToTuple5Mixed {
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

pub(crate) struct ScalarToTuple7Mixed;

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, (u32,)> for ScalarToTuple7Mixed {
    type Output = (u32, f32, u32, f32, u32, f32, u32);

    fn apply(input: (u32,)) -> (u32, f32, u32, f32, u32, f32, u32) {
        (
            input.0 + 1,
            input.0 as f32 + 2.0,
            input.0 + 3,
            input.0 as f32 + 4.0,
            input.0 + 5,
            input.0 as f32 + 6.0,
            input.0 + 7,
        )
    }
}

pub(crate) struct GreaterThanFour;

#[cubecl::cube]
impl PredicateOp<WgpuRuntime, (f32,)> for GreaterThanFour {
    fn apply(input: (f32,)) -> bool {
        input.0 > 4.0
    }
}

pub(crate) struct F32GreaterThanOne;

#[cubecl::cube]
impl PredicateOp<WgpuRuntime, (f32,)> for F32GreaterThanOne {
    fn apply(input: (f32,)) -> bool {
        input.0 > 1.0
    }
}

pub(crate) struct NonZero;

#[cubecl::cube]
impl PredicateOp<WgpuRuntime, (u32,)> for NonZero {
    fn apply(input: (u32,)) -> bool {
        input.0 != 0
    }
}

pub(crate) struct U32IsTwenty;

#[cubecl::cube]
impl PredicateOp<WgpuRuntime, (u32,)> for U32IsTwenty {
    fn apply(input: (u32,)) -> bool {
        input.0 == 20
    }
}

pub(crate) struct MixedStencilKeep;

#[cubecl::cube]
impl PredicateOp<WgpuRuntime, (f32, u32)> for MixedStencilKeep {
    fn apply(input: (f32, u32)) -> bool {
        input.0 > 1.0 && input.1 != 0
    }
}

pub(crate) struct PairMixedSplit;

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, (f32, u32)> for PairMixedSplit {
    type Output = (f32, u32);

    fn apply(input: (f32, u32)) -> (f32, u32) {
        (input.0 + 10.0, input.1 + 1)
    }
}

pub(crate) struct PairScaleAndTag;

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, (f32, u32)> for PairScaleAndTag {
    type Output = (f32, u32);

    fn apply(input: (f32, u32)) -> (f32, u32) {
        (input.0 * 2.0, input.1 + 1)
    }
}

pub(crate) struct Tuple3MixedSplit;

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, (f32, u32, f32)> for Tuple3MixedSplit {
    type Output = (f32, u32, f32);

    fn apply(input: (f32, u32, f32)) -> (f32, u32, f32) {
        (input.0 + input.2, input.1 + 1, input.2 + input.0)
    }
}

pub(crate) struct NestedTuple3MixedSplit;

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, ((f32, u32), f32)> for NestedTuple3MixedSplit {
    type Output = (f32, u32, f32);

    fn apply(input: ((f32, u32), f32)) -> (f32, u32, f32) {
        (input.0.0 + input.1, input.0.1 + 1, input.1 + input.0.0)
    }
}

pub(crate) struct Tuple4MixedSplit;

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, (f32, u32, f32, u32)> for Tuple4MixedSplit {
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
impl UnaryOp<WgpuRuntime, (f32, u32, f32)> for Tuple3To5MixedSplit {
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
impl UnaryOp<WgpuRuntime, (f32, u32, f32, u32, f32)> for Tuple5To3MixedSplit {
    type Output = (f32, u32, f32);

    fn apply(input: (f32, u32, f32, u32, f32)) -> (f32, u32, f32) {
        (
            input.0 + input.2 + input.4,
            input.1 + input.3,
            input.4 - input.0,
        )
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
        impl UnaryOp<WgpuRuntime, ($($ty), +)> for TupleWideMixedSplit {
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

pub(crate) struct PairMixedFirstPositive;

#[cubecl::cube]
impl PredicateOp<WgpuRuntime, (f32, u32)> for PairMixedFirstPositive {
    fn apply(input: (f32, u32)) -> bool {
        input.0 > 0.0
    }
}

pub(crate) struct PairMixedTagIsTwenty;

#[cubecl::cube]
impl PredicateOp<WgpuRuntime, (f32, u32)> for PairMixedTagIsTwenty {
    fn apply(input: (f32, u32)) -> bool {
        input.1 == 20
    }
}

pub(crate) struct Tuple3MixedTagIsTwenty;

#[cubecl::cube]
impl PredicateOp<WgpuRuntime, (f32, u32, f32)> for Tuple3MixedTagIsTwenty {
    fn apply(input: (f32, u32, f32)) -> bool {
        input.1 == 20 && input.2 > 0.0
    }
}

pub(crate) struct NestedTuple3MixedTagIsTwenty;

#[cubecl::cube]
impl PredicateOp<WgpuRuntime, ((f32, u32), f32)> for NestedTuple3MixedTagIsTwenty {
    fn apply(input: ((f32, u32), f32)) -> bool {
        input.0.1 == 20 && input.1 > 0.0
    }
}

pub(crate) struct Tuple3MixedFirstPositive;

#[cubecl::cube]
impl PredicateOp<WgpuRuntime, (f32, u32, f32)> for Tuple3MixedFirstPositive {
    fn apply(input: (f32, u32, f32)) -> bool {
        input.0 > 0.0
    }
}
