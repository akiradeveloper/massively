use cubecl::prelude::*;
use massively::prelude::*;

use crate::{WgpuDevice, WgpuRuntime};

struct PairOddFlag;

#[cubecl::cube]
impl<R> massively::op::UnaryOp<R, (u32, u32)> for PairOddFlag
where
    R: Runtime,
{
    type Output = (u32,);

    fn apply(input: (u32, u32)) -> (u32,) {
        (input.0 % 2,)
    }
}

struct PairSumOddFlag;

#[cubecl::cube]
impl<R> massively::op::UnaryOp<R, (u32, u32)> for PairSumOddFlag
where
    R: Runtime,
{
    type Output = (u32,);

    fn apply(input: (u32, u32)) -> (u32,) {
        ((input.0 + input.1) % 2,)
    }
}

struct AddOne;

#[cubecl::cube]
impl massively::op::UnaryOp<WgpuRuntime, (u32,)> for AddOne {
    type Output = (u32,);

    fn apply(input: (u32,)) -> (u32,) {
        (input.0 + 1,)
    }
}

#[test]
fn copy_where_accepts_constant_slice_stencil() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let input = exec.to_device(&[10_u32, 20, 30]).unwrap();

    let (output,) = massively::copy_where(
        &exec,
        SoA1(input.slice(..)),
        massively::slice::constant_slice(3, 1),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![10, 20, 30]);
}

#[test]
fn transform_where_accepts_constant_slice_stencil() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let input = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let output = exec.to_device(&[0_u32; 3]).unwrap();

    massively::transform_where(
        &exec,
        SoA1(input.slice(..)),
        AddOne,
        massively::slice::constant_slice(3, 1),
        SoA1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![11, 21, 31]);
}

#[test]
fn replace_where_accepts_constant_slice_stencil() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let output = exec.to_device(&[10_u32, 20, 30]).unwrap();

    massively::replace_where(
        &exec,
        (7_u32,),
        massively::slice::constant_slice(3, 1),
        SoA1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![7, 7, 7]);
}

#[test]
fn remove_where_accepts_transform_slice_stencil_from_two_column_iter() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let left = exec.to_device(&[10_u32, 21, 30, 43]).unwrap();
    let right = exec.to_device(&[100_u32, 200, 300, 400]).unwrap();

    let stencil =
        massively::slice::transform_slice(SoA2(left.slice(..), right.slice(..)), PairOddFlag);
    let (out_left, out_right) =
        massively::remove_where(&exec, SoA2(left.slice(..), right.slice(..)), stencil).unwrap();

    assert_eq!(exec.to_host(&out_left).unwrap(), vec![10, 30]);
    assert_eq!(exec.to_host(&out_right).unwrap(), vec![100, 300]);
}

#[test]
fn copy_where_accepts_transform_slice_stencil_from_two_column_iter() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let values = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let left = exec.to_device(&[1_u32, 2, 3, 4]).unwrap();
    let right = exec.to_device(&[100_u32, 200, 300, 400]).unwrap();

    let stencil =
        massively::slice::transform_slice(SoA2(left.slice(..), right.slice(..)), PairOddFlag);
    let (output,) = massively::copy_where(&exec, SoA1(values.slice(..)), stencil).unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![10, 30]);
}

#[test]
fn copy_where_accepts_transform_slice_stencil_from_two_column_sum() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
    let values = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let left = exec.to_device(&[1_u32, 1, 2, 2]).unwrap();
    let right = exec.to_device(&[0_u32, 1, 1, 2]).unwrap();

    let stencil =
        massively::slice::transform_slice(SoA2(left.slice(..), right.slice(..)), PairSumOddFlag);
    let (output,) = massively::copy_where(&exec, SoA1(values.slice(..)), stencil).unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![10, 30]);
}
