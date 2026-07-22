use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{
    Executor, MIter,
    op::BinaryPredicateOp,
    op::ExpandOp,
    op::PredicateOp,
    op::ReductionOp,
    op::UnaryOp,
    seg::{
        AllOf, Executable, FlatMap, ForEachSegment, Map, Reduce, Segment, SegmentIterator, Unique,
    },
    vector::{copy_where, equal, is_sorted, map as vector_map},
};

struct CastU64;

#[cubecl::cube]
impl UnaryOp<u32> for CastU64 {
    type Output = u64;

    fn apply(value: u32) -> u64 {
        u64::cast_from(value)
    }
}

struct Equal;
struct Even;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for Equal {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs == rhs
    }
}

#[cubecl::cube]
impl PredicateOp<u32> for Even {
    fn apply(value: u32) -> bool {
        value % 2u32 == 0u32
    }
}

struct RepeatValue;

#[cubecl::cube]
impl ExpandOp<u32> for RepeatValue {
    type Output = u32;

    fn count(input: u32) -> u32 {
        input
    }

    fn generate(input: u32, local_index: u32) -> u32 {
        input * 10 + local_index
    }
}

struct Add;

#[cubecl::cube]
impl ReductionOp<u32> for Add {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        lhs + rhs
    }
}

struct LexicographicalBytes;

struct SliceLength;

#[cubecl::cube]
impl UnaryOp<Segment<u32>> for SliceLength {
    type Output = u32;

    fn apply(value: Segment<u32>) -> u32 {
        value.len()
    }
}

#[cubecl::cube]
impl BinaryPredicateOp<Segment<u32>> for LexicographicalBytes {
    fn apply(lhs: Segment<u32>, rhs: Segment<u32>) -> bool {
        let lhs_len = lhs.len();
        let rhs_len = rhs.len();
        let common_len = if lhs_len < rhs_len { lhs_len } else { rhs_len };
        let cursor = RuntimeCell::<u32>::new(0u32);
        let ordering = RuntimeCell::<u32>::new(0u32);

        while cursor.read() < common_len && ordering.read() == 0u32 {
            let lhs_item = lhs.at(cursor.read());
            let rhs_item = rhs.at(cursor.read());
            if lhs_item < rhs_item {
                ordering.store(1u32);
            } else if rhs_item < lhs_item {
                ordering.store(2u32);
            }
            cursor.store(cursor.read() + 1u32);
        }

        if ordering.read() == 1u32 {
            true
        } else if ordering.read() == 2u32 {
            false
        } else {
            lhs_len < rhs_len
        }
    }
}

#[derive(CubeType, Clone, Copy)]
struct Code {
    value: u32,
}

struct WrapCode;

#[cubecl::cube]
impl UnaryOp<u32> for WrapCode {
    type Output = Code;

    fn apply(value: u32) -> Code {
        Code { value }
    }
}

struct LexicographicalCodes;

#[cubecl::cube]
impl BinaryPredicateOp<Segment<Code>> for LexicographicalCodes {
    fn apply(lhs: Segment<Code>, rhs: Segment<Code>) -> bool {
        let lhs_len = lhs.len();
        let rhs_len = rhs.len();
        let common_len = if lhs_len < rhs_len { lhs_len } else { rhs_len };
        let cursor = RuntimeCell::<u32>::new(0u32);
        let ordering = RuntimeCell::<u32>::new(0u32);

        while cursor.read() < common_len && ordering.read() == 0u32 {
            let lhs_item = lhs.at(cursor.read()).value;
            let rhs_item = rhs.at(cursor.read()).value;
            if lhs_item < rhs_item {
                ordering.store(1u32);
            } else if rhs_item < lhs_item {
                ordering.store(2u32);
            }
            cursor.store(cursor.read() + 1u32);
        }

        if ordering.read() == 1u32 {
            true
        } else if ordering.read() == 2u32 {
            false
        } else {
            lhs_len < rhs_len
        }
    }
}

struct SlicesEqual;

#[cubecl::cube]
impl BinaryPredicateOp<Segment<u32>> for SlicesEqual {
    fn apply(lhs: Segment<u32>, rhs: Segment<u32>) -> bool {
        if lhs.len() != rhs.len() {
            false
        } else {
            let cursor = RuntimeCell::<u32>::new(0u32);
            let equal = RuntimeCell::<u32>::new(1u32);
            while cursor.read() < lhs.len() && equal.read() != 0u32 {
                if lhs.at(cursor.read()) != rhs.at(cursor.read()) {
                    equal.store(0u32);
                }
                cursor.store(cursor.read() + 1u32);
            }
            equal.read() != 0u32
        }
    }
}

#[test]
fn segmented_wrappers_expose_their_parts() {
    let input = SegmentIterator::new([1_u32, 2], [0_u32, 2]);
    assert_eq!(input.values(), &[1, 2]);
    assert_eq!(input.offsets(), &[0, 2]);
    assert_eq!(input.into_parts(), ([1, 2], [0, 2]));
}

#[test]
fn segmented_algorithms_return_owned_device_results() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let values = exec.to_device(&[1_u32, 1, 2, 3, 3]);
    let offsets = exec.to_device(&[0_u32, 3, 5]);

    let mapped = ForEachSegment(Map(CastU64))
        .run(
            &exec,
            SegmentIterator::new(values.slice(..), offsets.slice(..)),
        )
        .unwrap();
    assert_eq!(
        exec.to_host(mapped.values()).unwrap(),
        vec![1_u64, 1, 2, 3, 3]
    );
    assert_eq!(exec.to_host(mapped.offsets()).unwrap(), vec![0, 3, 5]);

    let unique = ForEachSegment(Unique(Equal))
        .run(
            &exec,
            SegmentIterator::new(values.slice(..), offsets.slice(..)),
        )
        .unwrap();
    assert_eq!(exec.to_host(unique.values()).unwrap(), vec![1, 2, 3]);
    assert_eq!(exec.to_host(unique.offsets()).unwrap(), vec![0, 2, 3]);

    let reduced = ForEachSegment(Reduce(Add, 0))
        .run(
            &exec,
            SegmentIterator::new(values.slice(..), offsets.slice(..)),
        )
        .unwrap();
    assert_eq!(exec.to_host(&reduced).unwrap(), vec![4, 6]);
}

#[test]
fn segmented_flat_map_rebuilds_offsets_and_preserves_empty_segments() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let values = exec.to_device(&[2_u32, 0, 1, 3]);
    let offsets = exec.to_device(&[0_u32, 2, 2, 4]);

    let output = ForEachSegment(FlatMap(RepeatValue))
        .run(
            &exec,
            SegmentIterator::new(values.slice(..), offsets.slice(..)),
        )
        .unwrap();

    assert_eq!(
        exec.to_host(output.values()).unwrap(),
        vec![20, 21, 10, 30, 31, 32]
    );
    assert_eq!(exec.to_host(output.offsets()).unwrap(), vec![0, 2, 2, 6]);
}

#[test]
fn segmented_boolean_results_are_bool_iterators() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let values = exec.to_device(&[2_u32, 4, 3, 6]);
    let offsets = exec.to_device(&[0_u32, 2, 4]);
    let flags = ForEachSegment(AllOf(Even))
        .run(
            &exec,
            SegmentIterator::new(values.slice(..), offsets.slice(..)),
        )
        .unwrap();

    fn assert_bool_iter<R: Runtime, Input: MIter<R, Item = bool>>(_input: &Input) {}
    assert_bool_iter::<WgpuRuntime, _>(&flags);
    assert_eq!(exec.to_host(&flags).unwrap(), vec![true, false]);

    let input = exec.to_device(&[10_u32, 20]);
    let selected = copy_where(&exec, input.slice(..), flags.slice(..)).unwrap();
    assert_eq!(exec.to_host(&selected).unwrap(), vec![10]);
}

#[test]
fn segment_iterator_is_an_miter_of_shared_read_only_segments() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    // [], [a], [aa], [ab], [b]
    let values = exec.to_device(&[
        b'a' as u32,
        b'a' as u32,
        b'a' as u32,
        b'a' as u32,
        b'b' as u32,
        b'b' as u32,
    ]);
    let offsets = exec.to_device(&[0_u32, 0, 1, 3, 5, 6]);
    let rows = SegmentIterator::new(values.slice(..), offsets.slice(..));

    fn assert_item<R: Runtime, Input: MIter<R, Item = Segment<u32>>>(_input: &Input) {}
    assert_item::<WgpuRuntime, _>(&rows);
    assert_eq!(<_ as MIter<WgpuRuntime>>::len(&rows).unwrap(), 5);
    let lengths = vector_map(&exec, rows.clone(), SliceLength).unwrap();
    assert_eq!(exec.to_host(&lengths).unwrap(), vec![0, 1, 2, 2, 1]);
    assert!(is_sorted(&exec, rows.clone(), LexicographicalBytes).unwrap());

    let lazy_rows = SegmentIterator::new(
        massively::lazy::map(values.slice(..), WrapCode),
        offsets.slice(..),
    );
    fn assert_lazy_item<R: Runtime, Input: MIter<R, Item = Segment<Code>>>(_input: &Input) {}
    assert_lazy_item::<WgpuRuntime, _>(&lazy_rows);
    assert!(is_sorted(&exec, lazy_rows, LexicographicalCodes).unwrap());

    let middle = rows.slice(1..4);
    assert_eq!(<_ as MIter<WgpuRuntime>>::len(&middle).unwrap(), 3);
    assert!(is_sorted(&exec, middle, LexicographicalBytes).unwrap());

    // [ab], [aa]
    let unsorted_values = exec.to_device(&[b'a' as u32, b'b' as u32, b'a' as u32, b'a' as u32]);
    let unsorted_offsets = exec.to_device(&[0_u32, 2, 4]);
    let unsorted = SegmentIterator::new(unsorted_values.slice(..), unsorted_offsets.slice(..));
    assert!(!is_sorted(&exec, unsorted, LexicographicalBytes).unwrap());

    // The two operands use independent backing expressions and absolute offsets.
    let left_values = exec.to_device(&[1_u32, 2, 3]);
    let left_offsets = exec.to_device(&[0_u32, 1, 3]);
    let right_values = exec.to_device(&[99_u32, 1, 2, 3]);
    let right_offsets = exec.to_device(&[1_u32, 2, 4]);
    let left = SegmentIterator::new(left_values.slice(..), left_offsets.slice(..));
    let right = SegmentIterator::new(
        massively::lazy::map(right_values.slice(..), massively::op::Identity),
        right_offsets.slice(..),
    );
    assert!(equal(&exec, left, right, SlicesEqual).unwrap());
}
