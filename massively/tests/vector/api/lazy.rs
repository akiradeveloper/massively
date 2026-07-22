use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{
    Executor, MIter, MStorage, lazy, op::ReductionOp, op::UnaryOp, vector::gather, vector::map,
    vector::reduce, zip2, zip7,
};

struct Double;
struct Sum;

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
