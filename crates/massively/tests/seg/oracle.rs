#[path = "../vector/oracle/common.rs"]
#[allow(dead_code)]
mod common;

use cubecl::prelude::*;
use massively::seg::{
    AdjacentDifference, AllOf, AnyOf, CountIf, ExclusiveScan, Executable, Filter, ForEachSegment,
    InclusiveScan, IsSorted, IsSortedUntil, NoneOf, Reduce, Reverse, SegmentIterator, Sort,
    Transform, Unique,
};
use massively::{op::BinaryPredicateOp, op::PredicateOp, op::ReductionOp, op::UnaryOp, zip2};
use oracle_ref::seg as oracle;
use proptest::prelude::*;

use common::*;

const SEGMENT_LENGTHS: [usize; 12] = [0, 1, 2, 31, 32, 33, 127, 128, 129, 255, 256, 257];

fn oracle_segments() -> impl Strategy<Value = Vec<Vec<u32>>> {
    prop::collection::vec(
        prop::sample::select(&SEGMENT_LENGTHS)
            .prop_flat_map(|len| prop::collection::vec(0_u32..100, len)),
        0..6,
    )
}

fn flatten<T: Clone>(segments: &[Vec<T>]) -> (Vec<T>, Vec<u32>) {
    let mut values = Vec::new();
    let mut offsets = Vec::with_capacity(segments.len() + 1);
    offsets.push(0);
    for segment in segments {
        values.extend_from_slice(segment);
        offsets.push(values.len() as u32);
    }
    (values, offsets)
}

macro_rules! length_preserving_case {
    ($name:ident, $algorithm:expr, $oracle:expr) => {
        proptest! {
            #![proptest_config(ProptestConfig { cases: CASES, .. ProptestConfig::default() })]
            #[test]
            fn $name(segments in oracle_segments()) {
                let exec = exec();
                let (values, offsets) = flatten(&segments);
                let values_gpu = exec.to_device(&values);
                let offsets_gpu = exec.to_device(&offsets);
                let output = ForEachSegment($algorithm).run(
                    &exec,
                    SegmentIterator::new(
                        lazify(values_gpu.slice(..)),
                        lazify(offsets_gpu.slice(..)),
                    ),
                ).unwrap();

                let (expected, expected_offsets) = flatten(&$oracle(&segments));
                let output_offsets = massively::vector::transform(
                    &exec,
                    output.offsets().clone(),
                    massively::op::Identity,
                ).unwrap();
                prop_assert_eq!(exec.to_host(output.values()).unwrap(), expected);
                prop_assert_eq!(exec.to_host(&output_offsets).unwrap(), expected_offsets);
            }
        }
    };
}

macro_rules! compacting_case {
    ($name:ident, $algorithm:expr, $oracle:expr) => {
        proptest! {
            #![proptest_config(ProptestConfig { cases: CASES, .. ProptestConfig::default() })]
            #[test]
            fn $name(segments in oracle_segments()) {
                let exec = exec();
                let (values, offsets) = flatten(&segments);
                let values_gpu = exec.to_device(&values);
                let offsets_gpu = exec.to_device(&offsets);
                let output = ForEachSegment($algorithm).run(
                    &exec,
                    SegmentIterator::new(
                        lazify(values_gpu.slice(..)),
                        lazify(offsets_gpu.slice(..)),
                    ),
                ).unwrap();

                let (expected_values, expected_offsets) = flatten(&$oracle(&segments));
                prop_assert_eq!(exec.to_host(output.values()).unwrap(), expected_values);
                prop_assert_eq!(exec.to_host(output.offsets()).unwrap(), expected_offsets);
            }
        }
    };
}

macro_rules! summarizing_case {
    ($name:ident, $algorithm:expr, $oracle:expr) => {
        proptest! {
            #![proptest_config(ProptestConfig { cases: CASES, .. ProptestConfig::default() })]
            #[test]
            fn $name(segments in oracle_segments()) {
                let exec = exec();
                let (values, offsets) = flatten(&segments);
                let values_gpu = exec.to_device(&values);
                let offsets_gpu = exec.to_device(&offsets);
                let output = ForEachSegment($algorithm).run(
                    &exec,
                    SegmentIterator::new(
                        lazify(values_gpu.slice(..)),
                        lazify(offsets_gpu.slice(..)),
                    ),
                ).unwrap();

                prop_assert_eq!(exec.to_host(&output).unwrap(), $oracle(&segments));
            }
        }
    };
}

length_preserving_case!(seg_transform, Transform(AddOne), |segments| oracle::map(
    segments, AddOne
));
length_preserving_case!(seg_sort, Sort(Less), |segments| oracle::sort(
    segments, Less
));
length_preserving_case!(seg_reverse, Reverse, oracle::reverse);
length_preserving_case!(seg_inclusive_scan, InclusiveScan(Sum), |segments| {
    oracle::inclusive_scan(segments, Sum)
});
length_preserving_case!(seg_exclusive_scan, ExclusiveScan(Sum, 7), |segments| {
    oracle::exclusive_scan(segments, Sum, 7)
});
length_preserving_case!(
    seg_adjacent_difference,
    AdjacentDifference(Sum),
    |segments| oracle::adjacent_difference(segments, Sum)
);

compacting_case!(seg_unique, Unique(Equal), |segments| oracle::unique(
    segments, Equal
));
compacting_case!(seg_filter, Filter(Even), |segments| oracle::filter(
    segments, Even
));

summarizing_case!(seg_reduce, Reduce(Sum, 7), |segments| oracle::reduce(
    segments, Sum, 7
));
summarizing_case!(seg_count_if, CountIf(Even), |segments| {
    oracle::count_if(segments, Even)
});
summarizing_case!(seg_all_of, AllOf(Even), |segments| oracle::all_of(
    segments, Even
));
summarizing_case!(seg_any_of, AnyOf(Even), |segments| oracle::any_of(
    segments, Even
));
summarizing_case!(seg_none_of, NoneOf(Even), |segments| oracle::none_of(
    segments, Even
));
summarizing_case!(seg_is_sorted, IsSorted(Less), |segments| {
    oracle::is_sorted(segments, Less)
});
summarizing_case!(seg_is_sorted_until, IsSortedUntil(Less), |segments| {
    oracle::is_sorted_until(segments, Less)
});

type Pair = (u32, u32);

struct PairAddOne;
struct PairSum;
struct PairEven;
struct PairEqual;
struct PairLess;

#[cubecl::cube]
impl UnaryOp<Pair> for PairAddOne {
    type Output = Pair;

    fn apply(input: Pair) -> Pair {
        (input.0 + 1u32, input.1 + 1u32)
    }
}

impl oracle_ref::op::UnaryOp<Pair> for PairAddOne {
    type Output = Pair;

    fn apply(input: Pair) -> Pair {
        (input.0 + 1, input.1 + 1)
    }
}

#[cubecl::cube]
impl ReductionOp<Pair> for PairSum {
    fn apply(lhs: Pair, rhs: Pair) -> Pair {
        (lhs.0 + rhs.0, lhs.1 + rhs.1)
    }
}

impl oracle_ref::op::ReductionOp<Pair> for PairSum {
    fn apply(lhs: Pair, rhs: Pair) -> Pair {
        (lhs.0 + rhs.0, lhs.1 + rhs.1)
    }
}

#[cubecl::cube]
impl PredicateOp<Pair> for PairEven {
    fn apply(input: Pair) -> bool {
        input.0 % 2u32 == 0u32
    }
}

impl oracle_ref::op::PredicateOp<Pair> for PairEven {
    fn apply(input: Pair) -> bool {
        input.0 % 2 == 0
    }
}

#[cubecl::cube]
impl BinaryPredicateOp<Pair> for PairEqual {
    fn apply(lhs: Pair, rhs: Pair) -> bool {
        lhs.0 == rhs.0 && lhs.1 == rhs.1
    }
}

impl oracle_ref::op::BinaryPredicateOp<Pair> for PairEqual {
    fn apply(lhs: Pair, rhs: Pair) -> bool {
        lhs == rhs
    }
}

#[cubecl::cube]
impl BinaryPredicateOp<Pair> for PairLess {
    fn apply(lhs: Pair, rhs: Pair) -> bool {
        lhs.0 < rhs.0
    }
}

impl oracle_ref::op::BinaryPredicateOp<Pair> for PairLess {
    fn apply(lhs: Pair, rhs: Pair) -> bool {
        lhs.0 < rhs.0
    }
}

fn oracle_pair_segments() -> impl Strategy<Value = Vec<Vec<Pair>>> {
    oracle_segments().prop_map(|segments| {
        segments
            .into_iter()
            .enumerate()
            .map(|(segment, values)| {
                values
                    .into_iter()
                    .enumerate()
                    .map(|(index, value)| {
                        (
                            value,
                            value.wrapping_add(segment as u32 * 17 + index as u32),
                        )
                    })
                    .collect()
            })
            .collect()
    })
}

fn pair_columns(rows: &[Pair]) -> (Vec<u32>, Vec<u32>) {
    rows.iter().copied().unzip()
}

fn pair_rows(first: Vec<u32>, second: Vec<u32>) -> Vec<Pair> {
    first.into_iter().zip(second).collect()
}

macro_rules! pair_length_preserving_case {
    ($name:ident, $algorithm:expr, $oracle:expr) => {
        proptest! {
            #![proptest_config(ProptestConfig { cases: CASES, .. ProptestConfig::default() })]
            #[test]
            fn $name(segments in oracle_pair_segments()) {
                let exec = exec();
                let (rows, offsets) = flatten(&segments);
                let (first, second) = pair_columns(&rows);
                let first_gpu = exec.to_device(&first);
                let second_gpu = exec.to_device(&second);
                let offsets_gpu = exec.to_device(&offsets);
                let output = ForEachSegment($algorithm).run(
                    &exec,
                    SegmentIterator::new(
                        zip2(lazify(first_gpu.slice(..)), lazify(second_gpu.slice(..))),
                        lazify(offsets_gpu.slice(..)),
                    ),
                ).unwrap();

                let (expected, expected_offsets) = flatten(&$oracle(&segments));
                prop_assert_eq!(
                    pair_rows(
                        exec.to_host(&output.values().0).unwrap(),
                        exec.to_host(&output.values().1).unwrap(),
                    ),
                    expected,
                );
                let output_offsets = massively::vector::transform(
                    &exec,
                    output.offsets().clone(),
                    massively::op::Identity,
                ).unwrap();
                prop_assert_eq!(exec.to_host(&output_offsets).unwrap(), expected_offsets);
            }
        }
    };
}

macro_rules! pair_compacting_case {
    ($name:ident, $algorithm:expr, $oracle:expr) => {
        proptest! {
            #![proptest_config(ProptestConfig { cases: CASES, .. ProptestConfig::default() })]
            #[test]
            fn $name(segments in oracle_pair_segments()) {
                let exec = exec();
                let (rows, offsets) = flatten(&segments);
                let (first, second) = pair_columns(&rows);
                let first_gpu = exec.to_device(&first);
                let second_gpu = exec.to_device(&second);
                let offsets_gpu = exec.to_device(&offsets);
                let output = ForEachSegment($algorithm).run(
                    &exec,
                    SegmentIterator::new(
                        zip2(lazify(first_gpu.slice(..)), lazify(second_gpu.slice(..))),
                        lazify(offsets_gpu.slice(..)),
                    ),
                ).unwrap();

                let (expected, expected_offsets) = flatten(&$oracle(&segments));
                prop_assert_eq!(
                    pair_rows(
                        exec.to_host(&output.values().0).unwrap(),
                        exec.to_host(&output.values().1).unwrap(),
                    ),
                    expected,
                );
                prop_assert_eq!(exec.to_host(output.offsets()).unwrap(), expected_offsets);
            }
        }
    };
}

macro_rules! pair_item_reduce_case {
    ($name:ident, $algorithm:expr, $oracle:expr) => {
        proptest! {
            #![proptest_config(ProptestConfig { cases: CASES, .. ProptestConfig::default() })]
            #[test]
            fn $name(segments in oracle_pair_segments()) {
                let exec = exec();
                let (rows, offsets) = flatten(&segments);
                let (first, second) = pair_columns(&rows);
                let first_gpu = exec.to_device(&first);
                let second_gpu = exec.to_device(&second);
                let offsets_gpu = exec.to_device(&offsets);
                let output = ForEachSegment($algorithm).run(
                    &exec,
                    SegmentIterator::new(
                        zip2(lazify(first_gpu.slice(..)), lazify(second_gpu.slice(..))),
                        lazify(offsets_gpu.slice(..)),
                    ),
                ).unwrap();

                prop_assert_eq!(
                    pair_rows(
                        exec.to_host(&output.0).unwrap(),
                        exec.to_host(&output.1).unwrap(),
                    ),
                    $oracle(&segments),
                );
            }
        }
    };
}

macro_rules! pair_flag_reduce_case {
    ($name:ident, $algorithm:expr, $oracle:expr) => {
        proptest! {
            #![proptest_config(ProptestConfig { cases: CASES, .. ProptestConfig::default() })]
            #[test]
            fn $name(segments in oracle_pair_segments()) {
                let exec = exec();
                let (rows, offsets) = flatten(&segments);
                let (first, second) = pair_columns(&rows);
                let first_gpu = exec.to_device(&first);
                let second_gpu = exec.to_device(&second);
                let offsets_gpu = exec.to_device(&offsets);
                let output = ForEachSegment($algorithm).run(
                    &exec,
                    SegmentIterator::new(
                        zip2(lazify(first_gpu.slice(..)), lazify(second_gpu.slice(..))),
                        lazify(offsets_gpu.slice(..)),
                    ),
                ).unwrap();

                prop_assert_eq!(exec.to_host(&output).unwrap(), $oracle(&segments));
            }
        }
    };
}

pair_length_preserving_case!(seg_pair_transform, Transform(PairAddOne), |segments| {
    oracle::map(segments, PairAddOne)
});
pair_length_preserving_case!(seg_pair_sort, Sort(PairLess), |segments| oracle::sort(
    segments, PairLess
));
pair_length_preserving_case!(seg_pair_reverse, Reverse, oracle::reverse);
pair_length_preserving_case!(
    seg_pair_inclusive_scan,
    InclusiveScan(PairSum),
    |segments| oracle::inclusive_scan(segments, PairSum)
);
pair_length_preserving_case!(
    seg_pair_exclusive_scan,
    ExclusiveScan(PairSum, (7, 11)),
    |segments| oracle::exclusive_scan(segments, PairSum, (7, 11))
);
pair_length_preserving_case!(
    seg_pair_adjacent_difference,
    AdjacentDifference(PairSum),
    |segments| oracle::adjacent_difference(segments, PairSum)
);

pair_compacting_case!(seg_pair_unique, Unique(PairEqual), |segments| {
    oracle::unique(segments, PairEqual)
});
pair_compacting_case!(seg_pair_filter, Filter(PairEven), |segments| {
    oracle::filter(segments, PairEven)
});

pair_item_reduce_case!(seg_pair_reduce, Reduce(PairSum, (7, 11)), |segments| {
    oracle::reduce(segments, PairSum, (7, 11))
});
pair_flag_reduce_case!(seg_pair_count_if, CountIf(PairEven), |segments| {
    oracle::count_if(segments, PairEven)
});
pair_flag_reduce_case!(seg_pair_all_of, AllOf(PairEven), |segments| {
    oracle::all_of(segments, PairEven)
});
pair_flag_reduce_case!(seg_pair_any_of, AnyOf(PairEven), |segments| {
    oracle::any_of(segments, PairEven)
});
pair_flag_reduce_case!(seg_pair_none_of, NoneOf(PairEven), |segments| {
    oracle::none_of(segments, PairEven)
});
pair_flag_reduce_case!(seg_pair_is_sorted, IsSorted(PairLess), |segments| {
    oracle::is_sorted(segments, PairLess)
});
pair_flag_reduce_case!(
    seg_pair_is_sorted_until,
    IsSortedUntil(PairLess),
    |segments| oracle::is_sorted_until(segments, PairLess)
);
