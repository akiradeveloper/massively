use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::seg::{Segment, SegmentIterator};
use massively::{
    Executor, MIter, MStorage, lazy, op::ReductionOp, op::UnaryOp, vector::gather, vector::map,
    vector::reduce, zip2, zip7,
};

struct Double;
struct Sum;
struct LookupTable;
struct LookupPairTable;
struct LookupTwoTables;

#[cubecl::cube]
impl UnaryOp<massively::MIndex> for Double {
    type Output = u32;

    fn apply(input: massively::MIndex) -> u32 {
        input * 2u32
    }
}

#[cubecl::cube]
impl ReductionOp<u32> for Sum {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        lhs + rhs
    }
}

#[cubecl::cube]
impl UnaryOp<(u32, Segment<u32>)> for LookupTable {
    type Output = u32;

    fn apply(input: (u32, Segment<u32>)) -> u32 {
        input.1.at(input.0)
    }
}

#[cubecl::cube]
impl UnaryOp<(u32, Segment<(u32, u32)>)> for LookupPairTable {
    type Output = u32;

    fn apply(input: (u32, Segment<(u32, u32)>)) -> u32 {
        let row = input.1.at(input.0);
        row.0 + row.1
    }
}

#[cubecl::cube]
impl UnaryOp<(u32, Segment<u32>, Segment<u32>)> for LookupTwoTables {
    type Output = u32;

    fn apply(input: (u32, Segment<u32>, Segment<u32>)) -> u32 {
        input.1.at(input.0) + input.2.at(input.0)
    }
}

#[test]
fn public_lazy_constructors_compose_as_miter() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);

    let constant: lazy::Taken<lazy::Constant<u32>> = lazy::constant(3_u32).take(4);
    assert_eq!(MIter::<WgpuRuntime>::len(&constant).unwrap(), 4);
    assert_eq!(reduce(&exec, constant, 0, Sum).unwrap(), 12);

    let counting: lazy::Taken<lazy::Counting> = lazy::counting(1).take(4);
    let output = map(
        &exec,
        lazy::identity(lazy::map(counting, Double)),
        massively::op::Identity,
    )
    .unwrap();
    assert_eq!(exec.to_host(&output).unwrap(), vec![2, 4, 6, 8]);

    let values = exec.to_device(&[10_u32, 20, 30, 40]);
    let permuted = lazy::permute(values.slice(..), lazy::counting(0).take(4));
    assert_eq!(MIter::<WgpuRuntime>::len(&permuted).unwrap(), 4);
    assert_eq!(reduce(&exec, permuted, 0, Sum).unwrap(), 100);
}

#[test]
fn with_table_shares_an_entire_lazy_iterator_with_every_context() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let indices = exec.to_device(&[3_u32, 0, 2, 1]);
    let values = exec.to_device(&[5_u32, 10, 15, 20]);
    let table = lazy::map(values.slice(..), Double);
    let input = lazy::with_table(indices.slice(..), table);

    assert_eq!(MIter::<WgpuRuntime>::len(&input).unwrap(), 4);
    let output = map(&exec, input, LookupTable).unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![40, 10, 30, 20]);
}

#[test]
fn with_table_supports_multi_column_tables() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let indices = exec.to_device(&[2_u32, 0, 1]);
    let left = exec.to_device(&[1_u32, 2, 3]);
    let right = exec.to_device(&[10_u32, 20, 30]);
    let table = zip2(left.slice(..), right.slice(..));

    let output = map(
        &exec,
        lazy::with_table(indices.slice(..), table),
        LookupPairTable,
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![33, 11, 22]);
}

#[test]
fn slicing_with_table_slices_only_the_contexts() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let indices = exec.to_device(&[4_u32, 3, 2, 1, 0]);
    let table = exec.to_device(&[999_u32, 10, 20, 30, 40, 50, 999]);
    let input = lazy::with_table(indices.slice(..), table.slice(1..6))
        .slice(1..5)
        .slice(1..3);

    assert_eq!(MIter::<WgpuRuntime>::len(&input).unwrap(), 2);
    let output = map(&exec, input, LookupTable).unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![30, 20]);
}

#[test]
fn with_table_can_be_nested_without_materializing_intermediates() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let indices = exec.to_device(&[2_u32, 0, 1]);
    let first = exec.to_device(&[1_u32, 2, 3]);
    let second = exec.to_device(&[10_u32, 20, 30]);
    let input = lazy::with_table(
        lazy::with_table(indices.slice(..), first.slice(..)),
        second.slice(..),
    );

    fn assert_flat_item<R: Runtime, Input: MIter<R, Item = (u32, Segment<u32>, Segment<u32>)>>(
        _input: &Input,
    ) {
    }
    assert_flat_item::<WgpuRuntime, _>(&input);

    let output = map(&exec, input, LookupTwoTables).unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![33, 11, 22]);
}

#[test]
fn with_table_matches_permuted_single_segment_iterators() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let indices = exec.to_device(&[2_u32, 0, 1]);
    let first = exec.to_device(&[1_u32, 2, 3]);
    let second = exec.to_device(&[10_u32, 20, 30]);
    let table_offsets = exec.to_device(&[0_u32, 3]);
    let repeated_index = exec.to_device(&[0_u32, 0, 0]);

    let first_table = lazy::permute(
        SegmentIterator::new(first.slice(..), table_offsets.slice(..)),
        repeated_index.slice(..),
    );
    let second_table = lazy::permute(
        SegmentIterator::new(second.slice(..), table_offsets.slice(..)),
        repeated_index.slice(..),
    );
    let composed = massively::zip3(indices.slice(..), first_table, second_table);
    let nested = lazy::with_table(
        lazy::with_table(indices.slice(..), first.slice(..)),
        second.slice(..),
    );

    let composed_output = map(&exec, composed, LookupTwoTables).unwrap();
    let nested_output = map(&exec, nested, LookupTwoTables).unwrap();

    assert_eq!(
        exec.to_host(&composed_output).unwrap(),
        exec.to_host(&nested_output).unwrap()
    );
}

#[test]
fn taken_tracks_nested_slice_offsets() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let taken: lazy::Taken<lazy::Counting> = lazy::counting(10).take(8);
    let sliced = taken.slice(2..6).slice(1..3);

    assert_eq!(MIter::<WgpuRuntime>::len(&sliced).unwrap(), 2);
    let values = exec.to_device(&(0_u32..20).collect::<Vec<_>>());
    let output = gather(&exec, values.slice(..), sliced).unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![13, 14]);
}

#[test]
fn slicing_a_lazy_permutation_slices_its_logical_rows() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let values = exec.to_device(&[10_u32, 20, 30, 40, 50, 60]);
    let indices = exec.to_device(&[4_u32, 1, 5, 0, 3, 2]);

    let sliced = lazy::permute(values.slice(..), indices.slice(..))
        .slice(1..5)
        .slice(1..3);
    assert_eq!(MIter::<WgpuRuntime>::len(&sliced).unwrap(), 2);

    let output = map(&exec, sliced, massively::op::Identity).unwrap();
    assert_eq!(exec.to_host(&output).unwrap(), vec![60, 10]);
}

#[test]
fn slicing_does_not_increase_read_arity_eight() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let inputs: Vec<_> = (0_u32..7)
        .map(|column| {
            let base = column * 10;
            exec.to_device(&[base, base + 1, base + 2, base + 3])
        })
        .collect();

    let sliced = lazy::permute(
        zip7(
            inputs[0].slice(..),
            inputs[1].slice(..),
            inputs[2].slice(..),
            inputs[3].slice(..),
            inputs[4].slice(..),
            inputs[5].slice(..),
            inputs[6].slice(..),
        ),
        lazy::counting(0).take(4),
    )
    .slice(1..3);

    let outputs = map(&exec, sliced, massively::op::Identity).unwrap();

    let (a, b, c, d, e, f, g) = MStorage::into_columns(outputs);
    let outputs = [&a, &b, &c, &d, &e, &f, &g];
    for (column, output) in outputs.into_iter().enumerate() {
        let base = column as u32 * 10;
        assert_eq!(exec.to_host(output).unwrap(), vec![base + 1, base + 2]);
    }
}

#[test]
fn reverse_composes_with_slicing_and_multi_column_inputs() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let empty = exec.alloc::<u32>(0);
    let reversed_empty = lazy::reverse(empty.slice(..));
    assert_eq!(MIter::<WgpuRuntime>::len(&reversed_empty).unwrap(), 0);
    let empty_output = map(&exec, reversed_empty, massively::op::Identity).unwrap();
    assert_eq!(empty_output.len(), 0);

    let values = exec.to_device(&[10_u32, 20, 30, 40, 50]);
    let middle = lazy::reverse(values.slice(..)).slice(1..4).slice(1..2);

    let output = map(&exec, middle, massively::op::Identity).unwrap();
    assert_eq!(exec.to_host(&output).unwrap(), vec![30]);

    let first = exec.to_device(&[1_u32, 2, 3]);
    let second = exec.to_device(&[10_u32, 20, 30]);
    let reversed = lazy::reverse(zip2(first.slice(..), second.slice(..)));

    assert_eq!(MIter::<WgpuRuntime>::len(&reversed).unwrap(), 3);
    let output = map(&exec, reversed, massively::op::Identity).unwrap();
    let (output_first, output_second) = MStorage::into_columns(output);
    assert_eq!(exec.to_host(&output_first).unwrap(), vec![3, 2, 1]);
    assert_eq!(exec.to_host(&output_second).unwrap(), vec![30, 20, 10]);
}
