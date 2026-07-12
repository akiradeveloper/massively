use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{
    Executor, Tuple2, Tuple3, Tuple4, Tuple5, Tuple6, Tuple7, flatten3, flatten4, flatten5,
    flatten6, flatten7, op::UnaryOp, tuple2, tuple3, tuple4, tuple5, tuple6, tuple7,
    vector::transform,
};

struct CanonicalTuple;

#[cubecl::cube]
impl UnaryOp<u32> for CanonicalTuple {
    type Output = Tuple7<u32, u32, u32, u32, u32, u32, u32>;

    fn apply(input: u32) -> Self::Output {
        let pair: Tuple2<u32, u32> = tuple2(input, input + 1);
        let triple: Tuple3<u32, u32, u32> = tuple3(pair.0, pair.1, input + 2);
        let quadruple: Tuple4<u32, u32, u32, u32> =
            tuple4(triple.0.0, triple.0.1, triple.1, input + 3);
        let quintuple: Tuple5<u32, u32, u32, u32, u32> = tuple5(
            quadruple.0.0.0,
            quadruple.0.0.1,
            quadruple.0.1,
            quadruple.1,
            input + 4,
        );
        let sextuple: Tuple6<u32, u32, u32, u32, u32, u32> = tuple6(
            quintuple.0.0.0.0,
            quintuple.0.0.0.1,
            quintuple.0.0.1,
            quintuple.0.1,
            quintuple.1,
            input + 5,
        );
        tuple7(
            sextuple.0.0.0.0.0,
            sextuple.0.0.0.0.1,
            sextuple.0.0.0.1,
            sextuple.0.0.1,
            sextuple.0.1,
            sextuple.1,
            input + 6,
        )
    }
}

struct FlattenTuples;

#[cubecl::cube]
impl UnaryOp<u32> for FlattenTuples {
    type Output = Tuple7<u32, u32, u32, u32, u32, u32, u32>;

    fn apply(input: u32) -> Self::Output {
        let (a0, _, a2) = flatten3(tuple3(input, input + 1, input + 2));
        let (b0, _, _, b3) = flatten4(tuple4(input, input + 1, input + 2, input + 3));
        let (_, _, _, _, c4) = flatten5(tuple5(input, input + 1, input + 2, input + 3, input + 4));
        let (_, _, _, _, _, d5) = flatten6(tuple6(
            input,
            input + 1,
            input + 2,
            input + 3,
            input + 4,
            input + 5,
        ));
        let (_, _, _, _, _, _, e6) = flatten7(tuple7(
            input,
            input + 1,
            input + 2,
            input + 3,
            input + 4,
            input + 5,
            input + 6,
        ));
        tuple7(a2, b3, c4, d5, e6, a0, b0)
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

    assert_eq!(value2, (1, 2));
    assert_eq!(flatten3(value3), (1, 2, 3));
    assert_eq!(flatten4(value4), (1, 2, 3, 4));
    assert_eq!(flatten5(value5), (1, 2, 3, 4, -5));
    assert_eq!(flatten6(value6), (1, 2, 3, 4, -5, -6));
    assert_eq!(flatten7(value7), (1, 2, 3, 4, -5, -6, -7));
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

    let offsets = [2_u32, 3, 4, 5, 6, 0, 0];
    let outputs = [
        &outputs.0.0.0.0.0.0,
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
        &outputs.0.0.0.0.0.0,
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
