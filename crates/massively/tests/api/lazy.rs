use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, MIter, ReductionOp, UnaryOp, lazy, reduce, transform, zip7};

struct Double;
struct Sum;

#[cubecl::cube]
impl UnaryOp<u32> for Double {
    type Output = u32;

    fn apply(input: u32) -> u32 {
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
    let output = exec.to_device(&[0_u32; 4]);
    transform(
        &exec,
        lazy::identity(lazy::transform(counting, Double)),
        massively::op::Identity,
        output.slice_mut(..),
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
    let output = exec.to_device(&[0_u32; 2]);
    let taken: lazy::Taken<lazy::Counting> = lazy::counting(10).take(8);
    let sliced = taken.slice(2..6).slice(1..3);

    assert_eq!(MIter::<WgpuRuntime>::len(&sliced).unwrap(), 2);
    transform(&exec, sliced, massively::op::Identity, output.slice_mut(..)).unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![13, 14]);
}

#[test]
fn slicing_a_lazy_permutation_slices_its_logical_rows() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let values = exec.to_device(&[10_u32, 20, 30, 40, 50, 60]);
    let indices = exec.to_device(&[4_u32, 1, 5, 0, 3, 2]);
    let output = exec.to_device(&[0_u32; 2]);

    let sliced = lazy::permute(values.slice(..), indices.slice(..))
        .slice(1..5)
        .slice(1..3);
    assert_eq!(MIter::<WgpuRuntime>::len(&sliced).unwrap(), 2);

    transform(&exec, sliced, massively::op::Identity, output.slice_mut(..)).unwrap();
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
    let outputs: Vec<_> = (0..7).map(|_| exec.to_device(&[0_u32; 2])).collect();

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

    transform(
        &exec,
        sliced,
        massively::op::Identity,
        zip7(
            outputs[0].slice_mut(..),
            outputs[1].slice_mut(..),
            outputs[2].slice_mut(..),
            outputs[3].slice_mut(..),
            outputs[4].slice_mut(..),
            outputs[5].slice_mut(..),
            outputs[6].slice_mut(..),
        ),
    )
    .unwrap();

    for (column, output) in outputs.iter().enumerate() {
        let base = column as u32 * 10;
        assert_eq!(exec.to_host(output).unwrap(), vec![base + 1, base + 2]);
    }
}
