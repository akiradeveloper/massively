use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{
    Executor, Tuple2, Tuple3, Tuple4, Tuple5, Tuple6, Tuple7, Tuple8, Tuple9, Tuple10, Tuple11,
    Tuple12, flatten3, flatten4, flatten5, flatten6, flatten7, flatten8, flatten9, flatten10,
    flatten11, flatten12, op::UnaryOp, tuple2, tuple3, tuple4, tuple5, tuple6, tuple7, tuple8,
    tuple9, tuple10, tuple11, tuple12, vector::transform,
};

struct CanonicalTuple;

#[cubecl::cube]
impl UnaryOp<u32> for CanonicalTuple {
    type Output = Tuple12<u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32>;

    fn apply(input: u32) -> Self::Output {
        tuple12(
            input,
            input + 1,
            input + 2,
            input + 3,
            input + 4,
            input + 5,
            input + 6,
            input + 7,
            input + 8,
            input + 9,
            input + 10,
            input + 11,
        )
    }
}

struct FlattenTuples;

#[cubecl::cube]
impl UnaryOp<u32> for FlattenTuples {
    type Output = Tuple12<u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32>;

    fn apply(input: u32) -> Self::Output {
        let (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11) = flatten12(tuple12(
            input,
            input + 1,
            input + 2,
            input + 3,
            input + 4,
            input + 5,
            input + 6,
            input + 7,
            input + 8,
            input + 9,
            input + 10,
            input + 11,
        ));
        tuple12(a11, a10, a9, a8, a7, a6, a5, a4, a3, a2, a1, a0)
    }
}

#[test]
fn tuple_aliases_and_constructors_are_ordinary_rust_values() {
    let value2: Tuple2<u8, u16> = tuple2(1, 2);
    let value3: Tuple3<u8, u16, u32> = tuple3(1, 2, 3);
    let value4: Tuple4<u8, u16, u32, u64> = tuple4(1, 2, 3, 4);
    let value5: Tuple5<u8, u16, u32, u64, i8> = tuple5(1, 2, 3, 4, -5);
    let value6: Tuple6<u8, u16, u32, u64, i8, i16> = tuple6(1, 2, 3, 4, -5, -6);
    let value7: Tuple7<u8, u16, u32, u64, i8, i16, i32> = tuple7(1, 2, 3, 4, -5, -6, -7);
    let value8: Tuple8<u8, u16, u32, u64, i8, i16, i32, i64> = tuple8(1, 2, 3, 4, -5, -6, -7, -8);
    let value9: Tuple9<u8, u16, u32, u64, i8, i16, i32, i64, f32> =
        tuple9(1, 2, 3, 4, -5, -6, -7, -8, 9.0);
    let value10: Tuple10<u8, u16, u32, u64, i8, i16, i32, i64, f32, f64> =
        tuple10(1, 2, 3, 4, -5, -6, -7, -8, 9.0, 10.0);
    let value11: Tuple11<u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, u8> =
        tuple11(1, 2, 3, 4, -5, -6, -7, -8, 9.0, 10.0, 11);
    let value12: Tuple12<u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, u8, u16> =
        tuple12(1, 2, 3, 4, -5, -6, -7, -8, 9.0, 10.0, 11, 12);

    assert_eq!(value2, (1, 2));
    assert_eq!(flatten3(value3), (1, 2, 3));
    assert_eq!(flatten4(value4), (1, 2, 3, 4));
    assert_eq!(flatten5(value5), (1, 2, 3, 4, -5));
    assert_eq!(flatten6(value6), (1, 2, 3, 4, -5, -6));
    assert_eq!(flatten7(value7), (1, 2, 3, 4, -5, -6, -7));
    assert_eq!(flatten8(value8), (1, 2, 3, 4, -5, -6, -7, -8));
    assert_eq!(flatten9(value9), (1, 2, 3, 4, -5, -6, -7, -8, 9.0));
    assert_eq!(flatten10(value10), (1, 2, 3, 4, -5, -6, -7, -8, 9.0, 10.0));
    assert_eq!(
        flatten11(value11),
        (1, 2, 3, 4, -5, -6, -7, -8, 9.0, 10.0, 11)
    );
    assert_eq!(
        flatten12(value12),
        (1, 2, 3, 4, -5, -6, -7, -8, 9.0, 10.0, 11, 12)
    );
}

#[test]
fn flatten_helpers_destructure_public_tuple_values_on_the_host() {
    assert_eq!(flatten3(tuple3(1_u8, 2_u16, 3_u32)), (1, 2, 3));
    assert_eq!(flatten4(tuple4(1_u8, 2_u16, 3_u32, 4_u64)), (1, 2, 3, 4));
    assert_eq!(
        flatten5(tuple5(1_u8, 2_u16, 3_u32, 4_u64, -5_i8)),
        (1, 2, 3, 4, -5)
    );
    assert_eq!(
        flatten6(tuple6(1_u8, 2_u16, 3_u32, 4_u64, -5_i8, -6_i16)),
        (1, 2, 3, 4, -5, -6)
    );
    assert_eq!(
        flatten7(tuple7(1_u8, 2_u16, 3_u32, 4_u64, -5_i8, -6_i16, -7_i32)),
        (1, 2, 3, 4, -5, -6, -7)
    );
}

#[test]
fn flatten_helpers_hide_tuple_representation_in_cube_ops() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let input = exec.to_device(&[10_u32, 20]);
    let outputs = transform(&exec, input.slice(..), FlattenTuples).unwrap();

    let offsets = [11_u32, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0];
    let outputs = [
        &outputs.0.0.0.0.0.0.0.0.0.0.0,
        &outputs.0.0.0.0.0.0.0.0.0.0.1,
        &outputs.0.0.0.0.0.0.0.0.0.1,
        &outputs.0.0.0.0.0.0.0.0.1,
        &outputs.0.0.0.0.0.0.0.1,
        &outputs.0.0.0.0.0.0.1,
        &outputs.0.0.0.0.0.1,
        &outputs.0.0.0.0.1,
        &outputs.0.0.0.1,
        &outputs.0.0.1,
        &outputs.0.1,
        &outputs.1,
    ];
    for (column, offset) in outputs.into_iter().zip(offsets) {
        assert_eq!(
            exec.to_host(column).unwrap(),
            vec![10 + offset, 20 + offset]
        );
    }
}

#[test]
fn tuple_aliases_and_constructors_are_canonical_in_cube_ops() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let input = exec.to_device(&[10_u32, 20]);
    let outputs = transform(&exec, input.slice(..), CanonicalTuple).unwrap();

    let outputs = [
        &outputs.0.0.0.0.0.0.0.0.0.0.0,
        &outputs.0.0.0.0.0.0.0.0.0.0.1,
        &outputs.0.0.0.0.0.0.0.0.0.1,
        &outputs.0.0.0.0.0.0.0.0.1,
        &outputs.0.0.0.0.0.0.0.1,
        &outputs.0.0.0.0.0.0.1,
        &outputs.0.0.0.0.0.1,
        &outputs.0.0.0.0.1,
        &outputs.0.0.0.1,
        &outputs.0.0.1,
        &outputs.0.1,
        &outputs.1,
    ];
    for (column, offset) in outputs.into_iter().zip(0_u32..) {
        assert_eq!(
            exec.to_host(column).unwrap(),
            vec![10 + offset, 20 + offset]
        );
    }
}
