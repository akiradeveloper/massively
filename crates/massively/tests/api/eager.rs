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

struct AddOffset;

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, (u32, u32)> for AddOffset {
    type Output = (u32,);

    fn apply(input: (u32, u32)) -> (u32,) {
        (input.0 + input.1,)
    }
}

struct Square;

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, (u32,)> for Square {
    type Output = (u32,);

    fn apply(input: (u32,)) -> (u32,) {
        (input.0 * input.0,)
    }
}

struct LessThan;

#[cubecl::cube]
impl PredicateOp<WgpuRuntime, (u32, u32)> for LessThan {
    fn apply(input: (u32, u32)) -> bool {
        input.0 < input.1
    }
}

struct IdentityU32Tuple;

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, (u32,)> for IdentityU32Tuple {
    type Output = (u32,);

    fn apply(input: (u32,)) -> (u32,) {
        input
    }
}

struct IdentityF32U32Tuple;

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, (f32, u32)> for IdentityF32U32Tuple {
    type Output = (f32, u32);

    fn apply(input: (f32, u32)) -> (f32, u32) {
        input
    }
}

#[test]
fn map_returns_owned_single_column_output() {
    let exec = exec();
    let input = exec.to_device(&[1_u32, 2, 3]).unwrap();

    let output = exec.to_device(&[0_u32; 3]).unwrap();
    transform(
        &exec,
        massively::Zip1(input.slice(..)),
        AddOneU32,
        massively::Zip1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![2, 3, 4]);
}

#[test]
fn stateful_unary_op_carries_value() {
    let exec = exec();
    let input = exec.to_device(&[1_u32, 2, 3]).unwrap();

    let output = exec.to_device(&[0_u32; 3]).unwrap();
    transform(
        &exec,
        massively::Zip2(
            input.slice(..),
            massively::lazy::constant(10_u32).take(input.len()),
        ),
        AddOffset,
        massively::Zip1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![11, 12, 13]);
}

#[test]
fn stateless_unary_op_runs_without_env() {
    let exec = exec();
    let input = exec.to_device(&[2_u32, 3, 4]).unwrap();

    let output = exec.to_device(&[0_u32; 3]).unwrap();
    transform(
        &exec,
        massively::Zip1(input.slice(..)),
        Square,
        massively::Zip1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![4, 9, 16]);
}

#[test]
fn composed_unary_op_reads_external_value_from_input() {
    let exec = exec();
    let input = exec.to_device(&[1_u32, 2, 3]).unwrap();
    let op = massively::op::compose(AddOffset, Square);

    let output = exec.to_device(&[0_u32; 3]).unwrap();
    transform(
        &exec,
        massively::Zip2(
            input.slice(..),
            massively::lazy::constant(2_u32).take(input.len()),
        ),
        op,
        massively::Zip1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![9, 16, 25]);
}

#[test]
fn public_identity_can_force_lazy_transform_path() {
    let exec = exec();
    let input = exec.to_device(&[1_u32, 2, 3]).unwrap();

    let output = exec.to_device(&[0_u32; 3]).unwrap();
    transform(
        &exec,
        massively::lazy::transform(massively::Zip1(input.slice(..)), massively::op::Identity),
        massively::op::Identity,
        massively::Zip1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![1, 2, 3]);
}

#[test]
fn lazy_constant_materializes_single_column() {
    let exec = exec();

    let output = exec.to_device(&[0_u32; 3]).unwrap();
    transform(
        &exec,
        massively::lazy::constant((42_u32,)).take(3),
        IdentityU32Tuple,
        massively::Zip1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![42, 42, 42]);
}

#[test]
fn lazy_constant_materializes_multi_column() {
    let exec = exec();

    let values = exec.to_device(&[0.0_f32; 3]).unwrap();
    let tags = exec.to_device(&[0_u32; 3]).unwrap();
    transform(
        &exec,
        massively::lazy::constant((1.5_f32, 9_u32)).take(3),
        IdentityF32U32Tuple,
        massively::Zip2(values.slice_mut(..), tags.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&values).unwrap(), vec![1.5, 1.5, 1.5]);
    assert_eq!(exec.to_host(&tags).unwrap(), vec![9, 9, 9]);
}

#[test]
fn predicate_reads_external_value_from_input() {
    let exec = exec();
    let input = exec.to_device(&[1_u32, 3, 5, 7]).unwrap();

    assert!(
        !massively::all_of(
            &exec,
            massively::Zip2(
                input.slice(..),
                massively::lazy::constant(5_u32).take(input.len()),
            ),
            LessThan,
        )
        .unwrap()
    );
    assert_eq!(
        count_if(
            &exec,
            massively::Zip2(
                input.slice(..),
                massively::lazy::constant(5_u32).take(input.len()),
            ),
            LessThan,
        )
        .unwrap(),
        2
    );
    assert_eq!(
        find_if(
            &exec,
            massively::Zip2(
                input.slice(..),
                massively::lazy::constant(4_u32).take(input.len()),
            ),
            LessThan,
        )
        .unwrap(),
        Some(0)
    );
}

#[test]
fn map_returns_owned_output_from_multi_column_input() {
    let exec = exec();
    let left = exec.to_device(&[1_u32, 2, 3]).unwrap();
    let right = exec.to_device(&[4_u32, 5, 6]).unwrap();

    let output = exec.to_device(&[0_u32; 3]).unwrap();
    transform(
        &exec,
        massively::Zip2(left.slice(..), right.slice(..)),
        PairToU32,
        massively::Zip1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![41, 52, 63]);
}

#[test]
fn permute_returns_owned_single_column_output() {
    let exec = exec();
    let input = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let indices = exec.to_device(&[3_u32, 1, 0]).unwrap();

    let output = exec.to_device(&[0_u32; 3]).unwrap();
    gather(
        &exec,
        massively::Zip1(input.slice(..)),
        indices.slice(..),
        massively::Zip1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![40, 20, 10]);
}

#[test]
fn permute_returns_owned_two_column_output() {
    let exec = exec();
    let left = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let right = exec.to_device(&[1_u32, 2, 3, 4]).unwrap();
    let indices = exec.to_device(&[2_u32, 0, 3]).unwrap();

    let out_left = exec.to_device(&[0_u32; 3]).unwrap();
    let out_right = exec.to_device(&[0_u32; 3]).unwrap();
    gather(
        &exec,
        massively::Zip2(left.slice(..), right.slice(..)),
        indices.slice(..),
        massively::Zip2(out_left.slice_mut(..), out_right.slice_mut(..)),
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

    let out_a = exec.to_device(&[0_u32; 2]).unwrap();
    let out_b = exec.to_device(&[0_u32; 2]).unwrap();
    let out_c = exec.to_device(&[0_u32; 2]).unwrap();
    gather(
        &exec,
        massively::Zip3(a.slice(..), b.slice(..), c.slice(..)),
        indices.slice(..),
        massively::Zip3(
            out_a.slice_mut(..),
            out_b.slice_mut(..),
            out_c.slice_mut(..),
        ),
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

    let copied = exec.to_device(&[0_u32; 4]).unwrap();
    let copied_len = copy_where(
        &exec,
        massively::Zip1(input.slice(..)),
        bool_stencil(4, IndexOdd),
        massively::Zip1(copied.slice_mut(..)),
    )
    .unwrap();
    assert_eq!(
        exec.to_host(&copied.slice(..copied_len)).unwrap(),
        vec![20, 40]
    );

    let transformed = exec.full(4, 0_u32).unwrap();
    transform_where(
        &exec,
        massively::Zip1(input.slice(..)),
        AddOneU32,
        bool_stencil(4, IndexOdd),
        massively::Zip1(transformed.slice_mut(..)),
    )
    .unwrap();
    assert_eq!(exec.to_host(&transformed).unwrap(), vec![0, 21, 0, 41]);
}

#[test]
fn owned_zip_result_can_feed_next_algorithm() {
    let exec = exec();
    let input = exec.to_device(&[4_u32, 1, 3, 2]).unwrap();

    let sorted = exec.to_device(&[0_u32; 4]).unwrap();
    sort(
        &exec,
        massively::Zip1(input.slice(..)),
        LessU32,
        massively::Zip1(sorted.slice_mut(..)),
    )
    .unwrap();
    let output = exec.to_device(&[0_u32; 4]).unwrap();
    transform(
        &exec,
        massively::Zip1(sorted.slice(..)),
        AddOneU32,
        massively::Zip1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![2, 3, 4, 5]);
}

#[test]
fn owned_zip_result_can_be_sliced_before_next_algorithm() {
    let exec = exec();
    let input = exec.to_device(&[4_u32, 1, 3, 2]).unwrap();

    let sorted = exec.to_device(&[0_u32; 4]).unwrap();
    sort(
        &exec,
        massively::Zip1(input.slice(..)),
        LessU32,
        massively::Zip1(sorted.slice_mut(..)),
    )
    .unwrap();
    let output = exec.to_device(&[0_u32; 2]).unwrap();
    transform(
        &exec,
        massively::Zip1(sorted.slice(1..3)),
        AddOneU32,
        massively::Zip1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![3, 4]);
}

#[test]
fn permuted_owned_zip_can_feed_selection_algorithm() {
    let exec = exec();
    let left = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let right = exec.to_device(&[1_u32, 2, 3, 4]).unwrap();
    let indices = exec.to_device(&[3_u32, 1, 0]).unwrap();
    let stencil = bool_stencil(3, IndexNot1);

    let permuted_left = exec.to_device(&[0_u32; 3]).unwrap();
    let permuted_right = exec.to_device(&[0_u32; 3]).unwrap();
    let permuted = massively::Zip2(permuted_left, permuted_right);
    gather(
        &exec,
        massively::Zip2(left.slice(..), right.slice(..)),
        indices.slice(..),
        permuted.slice_mut(..),
    )
    .unwrap();
    let out_left = exec.to_device(&[0_u32; 3]).unwrap();
    let out_right = exec.to_device(&[0_u32; 3]).unwrap();
    let len = copy_where(
        &exec,
        permuted.slice(..),
        stencil,
        massively::Zip2(out_left.slice_mut(..), out_right.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_left.slice(..len)).unwrap(), vec![40, 10]);
    assert_eq!(exec.to_host(&out_right.slice(..len)).unwrap(), vec![4, 1]);
}

#[test]
fn owned_zip_slice_mut_can_be_used_as_output() {
    let exec = exec();
    let input = exec.to_device(&[1_u32, 2, 3]).unwrap();
    let output = massively::Zip1(exec.full(5, 0_u32).unwrap());

    transform(
        &exec,
        massively::Zip1(input.slice(..)),
        AddOneU32,
        output.slice_mut(1..4),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output.0).unwrap(), vec![0, 2, 3, 4, 0]);
}
