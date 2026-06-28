use crate::common::*;

struct AddOneU32;

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, (u32,)> for AddOneU32 {
    type Output = (u32,);

    fn apply(input: (u32,)) -> (u32,) {
        (input.0 + 1,)
    }
}

struct SeededIndexF32;

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, (u64, u32)> for SeededIndexF32 {
    type Output = (f32,);

    fn apply(input: (u64, u32)) -> (f32,) {
        ((input.0 as f32) + (input.1 as f32),)
    }
}

struct PairInsideTen;

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, (f32, f32)> for PairInsideTen {
    type Output = (u32,);

    fn apply(input: (f32, f32)) -> (u32,) {
        ((input.0 + input.1 < 10.0) as u32,)
    }
}

struct PairSumU32;

#[cubecl::cube]
impl ReductionOp<WgpuRuntime, (u32, u32)> for PairSumU32 {
    fn apply(lhs: (u32, u32), rhs: (u32, u32)) -> (u32, u32) {
        (lhs.0 + rhs.0, lhs.1 + rhs.1)
    }
}

struct TripleSumU32;

#[cubecl::cube]
impl ReductionOp<WgpuRuntime, (u32, u32, u32)> for TripleSumU32 {
    fn apply(lhs: (u32, u32, u32), rhs: (u32, u32, u32)) -> (u32, u32, u32) {
        (lhs.0 + rhs.0, lhs.1 + rhs.1, lhs.2 + rhs.2)
    }
}

#[test]
fn soa1_accepts_constant_slice_as_miter() {
    let exec = exec();

    let sum = reduce(
        &exec,
        massively::SoA1(massively::slice::constant_slice(4, 7_u32)),
        (0_u32,),
        Sum,
    )
    .unwrap();

    assert_eq!(sum, (28,));
}

#[test]
fn soa1_accepts_transform_slice_as_miter() {
    let exec = exec();
    let input = exec.to_device(&[1_u32, 2, 3, 4]).unwrap();
    let lazy = massively::slice::transform_slice(massively::SoA1(input.slice(..)), AddOneU32);

    let sum = reduce(&exec, massively::SoA1(lazy), (0_u32,), Sum).unwrap();

    assert_eq!(sum, (14,));
}

#[test]
fn soa2_accepts_mixed_mslice_columns() {
    let exec = exec();
    let input = exec.to_device(&[1_u32, 2, 3]).unwrap();
    let lazy = massively::slice::transform_slice(massively::SoA1(input.slice(..)), AddOneU32);

    let sum = reduce(
        &exec,
        massively::SoA2(lazy, massively::slice::constant_slice(3, 10_u32)),
        (0_u32, 0_u32),
        PairSumU32,
    )
    .unwrap();

    assert_eq!(sum, (9, 30));
}

#[test]
fn transform_accepts_binary_transform_slice_columns() {
    let exec = exec();
    let len = 5;
    let seed1 = massively::slice::constant_slice(len, 1_u64);
    let seed2 = massively::slice::constant_slice(len, 2_u64);
    let idx1 = massively::slice::tabulate_slice(len);
    let idx2 = massively::slice::tabulate_slice(len);
    let x = massively::slice::transform_slice(massively::SoA2(seed1, idx1), SeededIndexF32);
    let y = massively::slice::transform_slice(massively::SoA2(seed2, idx2), SeededIndexF32);
    let out = exec.filled(len, 0_u32).unwrap();

    transform(
        &exec,
        massively::SoA2(x, y),
        PairInsideTen,
        massively::SoA1(out.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out).unwrap(), vec![1, 1, 1, 1, 0]);
}

#[test]
fn transform_slice_accepts_binary_transform_slice_columns() {
    let exec = exec();
    let len = 5;
    let seed1 = massively::slice::constant_slice(len, 1_u64);
    let seed2 = massively::slice::constant_slice(len, 2_u64);
    let idx1 = massively::slice::tabulate_slice(len);
    let idx2 = massively::slice::tabulate_slice(len);
    let x = massively::slice::transform_slice(massively::SoA2(seed1, idx1), SeededIndexF32);
    let y = massively::slice::transform_slice(massively::SoA2(seed2, idx2), SeededIndexF32);

    let hits = massively::slice::transform_slice(massively::SoA2(x, y), PairInsideTen);
    let sum = reduce(&exec, massively::SoA1(hits), (0_u32,), Sum).unwrap();

    assert_eq!(sum, (4,));
}

#[test]
fn soa3_accepts_mixed_mslice_columns() {
    let exec = exec();
    let input = exec.to_device(&[1_u32, 2, 3]).unwrap();
    let lazy = massively::slice::transform_slice(massively::SoA1(input.slice(..)), AddOneU32);

    let sum = reduce(
        &exec,
        massively::SoA3(
            input.slice(..),
            lazy,
            massively::slice::constant_slice(3, 100_u32),
        ),
        (0_u32, 0_u32, 0_u32),
        TripleSumU32,
    )
    .unwrap();

    assert_eq!(sum, (6, 9, 300));
}

#[test]
fn by_key_accepts_lazy_single_column_keys() {
    let exec = exec();
    let raw_keys = exec.to_device(&[0_u32, 0, 1, 1]).unwrap();
    let values = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let keys = massively::slice::transform_slice(massively::SoA1(raw_keys.slice(..)), AddOneU32);

    let out = inclusive_scan_by_key(
        &exec,
        massively::SoA1(keys),
        massively::SoA1(values.slice(..)),
        EqualU32,
        Sum,
    )
    .unwrap();

    assert_eq!(exec.to_host(&out.0).unwrap(), vec![10, 30, 30, 70]);
}

#[test]
fn pair_algorithm_accepts_lazy_right_single_column() {
    let exec = exec();
    let left = exec.to_device(&[2_u32, 3, 4]).unwrap();
    let right_base = exec.to_device(&[1_u32, 2, 3]).unwrap();
    let right = massively::slice::transform_slice(massively::SoA1(right_base.slice(..)), AddOneU32);

    let is_equal = equal(
        &exec,
        massively::SoA1(left.slice(..)),
        massively::SoA1(right),
        EqualU32,
    )
    .unwrap();

    assert!(is_equal);
}

#[test]
fn merge_accepts_lazy_right_single_column() {
    let exec = exec();
    let left = exec.to_device(&[1_u32, 3]).unwrap();
    let right_base = exec.to_device(&[1_u32, 3]).unwrap();
    let right = massively::slice::transform_slice(massively::SoA1(right_base.slice(..)), AddOneU32);

    let out = merge(
        &exec,
        massively::SoA1(left.slice(..)),
        massively::SoA1(right),
        LessU32,
    )
    .unwrap();

    assert_eq!(exec.to_host(&out.0).unwrap(), vec![1, 2, 3, 4]);
}

#[test]
fn merge_by_key_accepts_lazy_right_single_key_and_value() {
    let exec = exec();
    let left_keys = exec.to_device(&[1_u32, 3]).unwrap();
    let left_values = exec.to_device(&[10_u32, 30]).unwrap();
    let right_key_base = exec.to_device(&[1_u32, 3]).unwrap();
    let right_value_base = exec.to_device(&[20_u32, 40]).unwrap();
    let right_keys =
        massively::slice::transform_slice(massively::SoA1(right_key_base.slice(..)), AddOneU32);
    let right_values =
        massively::slice::transform_slice(massively::SoA1(right_value_base.slice(..)), AddOneU32);

    let ((out_keys,), (out_values,)) = merge_by_key(
        &exec,
        massively::SoA1(left_keys.slice(..)),
        massively::SoA1(left_values.slice(..)),
        massively::SoA1(right_keys),
        massively::SoA1(right_values),
        LessU32,
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_keys).unwrap(), vec![1, 2, 3, 4]);
    assert_eq!(exec.to_host(&out_values).unwrap(), vec![10, 21, 30, 41]);
}
