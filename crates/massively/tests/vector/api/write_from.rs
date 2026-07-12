use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{op::*, vector::*, *};

type Right = (u32, (u32, u32));

fn exec() -> Executor<WgpuRuntime> {
    Executor::new(WgpuDevice::DefaultDevice)
}

fn right_input(
    a: &DeviceVec<WgpuRuntime, u32>,
    b: &DeviceVec<WgpuRuntime, u32>,
    c: &DeviceVec<WgpuRuntime, u32>,
) -> Zip<DeviceSlice<u32>, Zip<DeviceSlice<u32>, DeviceSlice<u32>>> {
    zip2(a.slice(..), zip2(b.slice(..), c.slice(..)))
}

struct SumRight;

#[cubecl::cube]
impl ReductionOp<Right> for SumRight {
    fn apply(lhs: Right, rhs: Right) -> Right {
        (lhs.0 + rhs.0, (lhs.1.0 + rhs.1.0, lhs.1.1 + rhs.1.1))
    }
}

struct LessRight;

#[cubecl::cube]
impl BinaryPredicateOp<Right> for LessRight {
    fn apply(lhs: Right, rhs: Right) -> bool {
        lhs.0 < rhs.0
    }
}

struct LessU32;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for LessU32 {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs < rhs
    }
}

struct AddOne;

#[cubecl::cube]
impl UnaryOp<Right> for AddOne {
    type Output = Right;

    fn apply(input: Right) -> Right {
        (input.0 + 1, (input.1.0 + 1, input.1.1 + 1))
    }
}

#[test]
fn inclusive_scan_writes_right_associated_items_to_left_associated_output() {
    let exec = exec();
    let a = exec.to_device(&[1_u32, 2, 3]);
    let b = exec.to_device(&[10_u32, 20, 30]);
    let c = exec.to_device(&[100_u32, 200, 300]);
    let output = inclusive_scan(&exec, right_input(&a, &b, &c), SumRight).unwrap();

    assert_eq!(exec.to_host(&output.0.0).unwrap(), vec![1, 3, 6]);
    assert_eq!(exec.to_host(&output.0.1).unwrap(), vec![10, 30, 60]);
    assert_eq!(exec.to_host(&output.1).unwrap(), vec![100, 300, 600]);
}

#[test]
fn sort_writes_right_associated_items_to_left_associated_output() {
    let exec = exec();
    let a = exec.to_device(&[3_u32, 1, 2]);
    let b = exec.to_device(&[30_u32, 10, 20]);
    let c = exec.to_device(&[300_u32, 100, 200]);
    let output = sort(&exec, right_input(&a, &b, &c), LessRight).unwrap();

    assert_eq!(exec.to_host(&output.0.0).unwrap(), vec![1, 2, 3]);
    assert_eq!(exec.to_host(&output.0.1).unwrap(), vec![10, 20, 30]);
    assert_eq!(exec.to_host(&output.1).unwrap(), vec![100, 200, 300]);
}

#[test]
fn copy_where_writes_right_associated_items_to_left_associated_output() {
    let exec = exec();
    let a = exec.to_device(&[1_u32, 2, 3, 4]);
    let b = exec.to_device(&[10_u32, 20, 30, 40]);
    let c = exec.to_device(&[100_u32, 200, 300, 400]);
    let flags = exec.to_device(&[0_u32, 1, 1, 0]);
    let output = copy_where(&exec, right_input(&a, &b, &c), flags.slice(..)).unwrap();

    assert_eq!(exec.to_host(&output.0.0).unwrap(), vec![2, 3]);
    assert_eq!(exec.to_host(&output.0.1).unwrap(), vec![20, 30]);
    assert_eq!(exec.to_host(&output.1).unwrap(), vec![200, 300]);
}

#[test]
fn gather_writes_right_associated_items_to_left_associated_output() {
    let exec = exec();
    let a = exec.to_device(&[1_u32, 2, 3]);
    let b = exec.to_device(&[10_u32, 20, 30]);
    let c = exec.to_device(&[100_u32, 200, 300]);
    let indices = exec.to_device(&[2_u32, 0]);
    let output = gather(&exec, right_input(&a, &b, &c), indices.slice(..)).unwrap();

    assert_eq!(exec.to_host(&output.0.0).unwrap(), vec![3, 1]);
    assert_eq!(exec.to_host(&output.0.1).unwrap(), vec![30, 10]);
    assert_eq!(exec.to_host(&output.1).unwrap(), vec![300, 100]);
}

#[test]
fn merge_writes_right_associated_items_to_left_associated_output() {
    let exec = exec();
    let la = exec.to_device(&[1_u32, 3]);
    let lb = exec.to_device(&[10_u32, 30]);
    let lc = exec.to_device(&[100_u32, 300]);
    let ra = exec.to_device(&[2_u32, 4]);
    let rb = exec.to_device(&[20_u32, 40]);
    let rc = exec.to_device(&[200_u32, 400]);
    let output = merge(
        &exec,
        right_input(&la, &lb, &lc),
        right_input(&ra, &rb, &rc),
        LessRight,
    )
    .unwrap();

    assert_eq!(exec.to_host(&output.0.0).unwrap(), vec![1, 2, 3, 4]);
    assert_eq!(exec.to_host(&output.0.1).unwrap(), vec![10, 20, 30, 40]);
    assert_eq!(exec.to_host(&output.1).unwrap(), vec![100, 200, 300, 400]);
}

#[test]
fn sort_by_key_reassociates_value_output() {
    let exec = exec();
    let keys = exec.to_device(&[3_u32, 1, 2]);
    let a = exec.to_device(&[30_u32, 10, 20]);
    let b = exec.to_device(&[300_u32, 100, 200]);
    let c = exec.to_device(&[3_000_u32, 1_000, 2_000]);
    let (out_keys, output) =
        sort_by_key(&exec, keys.slice(..), right_input(&a, &b, &c), LessU32).unwrap();

    assert_eq!(exec.to_host(&out_keys).unwrap(), vec![1, 2, 3]);
    assert_eq!(exec.to_host(&output.0.0).unwrap(), vec![10, 20, 30]);
    assert_eq!(exec.to_host(&output.0.1).unwrap(), vec![100, 200, 300]);
    assert_eq!(exec.to_host(&output.1).unwrap(), vec![1_000, 2_000, 3_000]);
}

#[test]
fn transform_where_reassociates_operation_output() {
    let exec = exec();
    let a = exec.to_device(&[1_u32, 2, 3]);
    let b = exec.to_device(&[10_u32, 20, 30]);
    let c = exec.to_device(&[100_u32, 200, 300]);
    let flags = exec.to_device(&[1_u32, 0, 1]);
    let oa = exec.to_device(&[90_u32, 90, 90]);
    let ob = exec.to_device(&[80_u32, 80, 80]);
    let oc = exec.to_device(&[70_u32, 70, 70]);

    transform_where(
        &exec,
        right_input(&a, &b, &c),
        AddOne,
        flags.slice(..),
        zip3(oa.slice_mut(..), ob.slice_mut(..), oc.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&oa).unwrap(), vec![2, 90, 4]);
    assert_eq!(exec.to_host(&ob).unwrap(), vec![11, 80, 31]);
    assert_eq!(exec.to_host(&oc).unwrap(), vec![101, 70, 301]);
}
