use crate::common::*;

struct AddOneU32;

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, (u32,)> for AddOneU32 {
    type Output = (u32,);

    fn apply(input: (u32,)) -> (u32,) {
        (input.0 + 1,)
    }
}

struct PairToU32;

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, (u32, u32)> for PairToU32 {
    type Output = (u32,);

    fn apply(input: (u32, u32)) -> (u32,) {
        (input.0 + input.1 * 10,)
    }
}

#[test]
fn map_returns_owned_single_column_output() {
    let exec = exec();
    let input = exec.to_device(&[1_u32, 2, 3]).unwrap();

    let massively::SoA1(output) = map(&exec, massively::SoA1(input.slice(..)), AddOneU32).unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![2, 3, 4]);
}

#[test]
fn map_returns_owned_output_from_multi_column_input() {
    let exec = exec();
    let left = exec.to_device(&[1_u32, 2, 3]).unwrap();
    let right = exec.to_device(&[4_u32, 5, 6]).unwrap();

    let massively::SoA1(output) = map(
        &exec,
        massively::SoA2(left.slice(..), right.slice(..)),
        PairToU32,
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![41, 52, 63]);
}

#[test]
fn permute_returns_owned_single_column_output() {
    let exec = exec();
    let input = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let indices = exec.to_device(&[3_u32, 1, 0]).unwrap();

    let massively::SoA1(output) =
        permute(&exec, massively::SoA1(input.slice(..)), indices.slice(..)).unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![40, 20, 10]);
}

#[test]
fn permute_returns_owned_two_column_output() {
    let exec = exec();
    let left = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let right = exec.to_device(&[1_u32, 2, 3, 4]).unwrap();
    let indices = exec.to_device(&[2_u32, 0, 3]).unwrap();

    let massively::SoA2(out_left, out_right) = permute(
        &exec,
        massively::SoA2(left.slice(..), right.slice(..)),
        indices.slice(..),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_left).unwrap(), vec![30, 10, 40]);
    assert_eq!(exec.to_host(&out_right).unwrap(), vec![3, 1, 4]);
}

#[test]
fn permute_returns_owned_three_column_output() {
    let exec = exec();
    let a = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let b = exec.to_device(&[1_u32, 2, 3, 4]).unwrap();
    let c = exec.to_device(&[100_u32, 200, 300, 400]).unwrap();
    let indices = exec.to_device(&[1_u32, 3]).unwrap();

    let massively::SoA3(out_a, out_b, out_c) = permute(
        &exec,
        massively::SoA3(a.slice(..), b.slice(..), c.slice(..)),
        indices.slice(..),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_a).unwrap(), vec![20, 40]);
    assert_eq!(exec.to_host(&out_b).unwrap(), vec![2, 4]);
    assert_eq!(exec.to_host(&out_c).unwrap(), vec![200, 400]);
}

#[test]
fn where_algorithms_accept_device_slice_stencil() {
    let exec = exec();
    let input = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let stencil = exec.to_device(&[0_u32, 1, 0, 1]).unwrap();

    let massively::SoA1(copied) =
        copy_where(&exec, massively::SoA1(input.slice(..)), stencil.slice(..)).unwrap();
    assert_eq!(exec.to_host(&copied).unwrap(), vec![20, 40]);

    let transformed = exec.constant(4, 0_u32).unwrap();
    transform_where(
        &exec,
        massively::SoA1(input.slice(..)),
        AddOneU32,
        stencil.slice(..),
        massively::SoA1(transformed.slice_mut(..)),
    )
    .unwrap();
    assert_eq!(exec.to_host(&transformed).unwrap(), vec![0, 21, 0, 41]);
}

#[test]
fn owned_soa_result_can_feed_next_algorithm() {
    let exec = exec();
    let input = exec.to_device(&[4_u32, 1, 3, 2]).unwrap();

    let sorted = sort(&exec, massively::SoA1(input.slice(..)), LessU32).unwrap();
    let massively::SoA1(output) = map(&exec, sorted.slice(..), AddOneU32).unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![2, 3, 4, 5]);
}

#[test]
fn owned_soa_result_can_be_sliced_before_next_algorithm() {
    let exec = exec();
    let input = exec.to_device(&[4_u32, 1, 3, 2]).unwrap();

    let sorted = sort(&exec, massively::SoA1(input.slice(..)), LessU32).unwrap();
    let massively::SoA1(output) = map(&exec, sorted.slice(1..3), AddOneU32).unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![3, 4]);
}

#[test]
fn permuted_owned_soa_can_feed_selection_algorithm() {
    let exec = exec();
    let left = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let right = exec.to_device(&[1_u32, 2, 3, 4]).unwrap();
    let indices = exec.to_device(&[3_u32, 1, 0]).unwrap();
    let stencil = exec.to_device(&[1_u32, 0, 1]).unwrap();

    let permuted: massively::SoA2<_, _> = permute(
        &exec,
        massively::SoA2(left.slice(..), right.slice(..)),
        indices.slice(..),
    )
    .unwrap();
    let massively::SoA2(out_left, out_right) =
        copy_where(&exec, permuted.slice(..), stencil.slice(..)).unwrap();

    assert_eq!(exec.to_host(&out_left).unwrap(), vec![40, 10]);
    assert_eq!(exec.to_host(&out_right).unwrap(), vec![4, 1]);
}

#[test]
fn owned_soa_slice_mut_can_be_used_as_output() {
    let exec = exec();
    let input = exec.to_device(&[1_u32, 2, 3]).unwrap();
    let output = massively::SoA1(exec.constant(5, 0_u32).unwrap());

    transform(
        &exec,
        massively::SoA1(input.slice(..)),
        AddOneU32,
        output.slice_mut(1..4),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output.0).unwrap(), vec![0, 2, 3, 4, 0]);
}
