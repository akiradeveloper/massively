use crate::common::*;

struct DetectHitTuple;
struct DetectHitScalar;
struct DetectPositiveTuple1Scalar;
struct CountHitTuple;
struct CountHitScalar;
struct OneTupleU32;
struct OneScalarU32;

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, (f32, f32)> for DetectHitTuple {
    type Output = (u32,);

    fn apply(input: (f32, f32)) -> (u32,) {
        let d2 = input.0 * input.0 + input.1 * input.1;
        if d2 <= 1.0 { (1_u32,) } else { (0_u32,) }
    }
}

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, (f32, f32)> for DetectHitScalar {
    type Output = u32;

    fn apply(input: (f32, f32)) -> u32 {
        let d2 = input.0 * input.0 + input.1 * input.1;
        if d2 <= 1.0 { 1_u32 } else { 0_u32 }
    }
}

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, (f32,)> for DetectPositiveTuple1Scalar {
    type Output = u32;

    fn apply(input: (f32,)) -> u32 {
        if input.0 > 0.0 { 1_u32 } else { 0_u32 }
    }
}

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, u32> for OneTupleU32 {
    type Output = (u32,);

    fn apply(_input: u32) -> (u32,) {
        (1_u32,)
    }
}

#[cubecl::cube]
impl UnaryOp<WgpuRuntime, u32> for OneScalarU32 {
    type Output = u32;

    fn apply(_input: u32) -> u32 {
        1_u32
    }
}

#[cubecl::cube]
impl ReductionOp<WgpuRuntime, (u32,)> for CountHitTuple {
    fn apply(lhs: (u32,), rhs: (u32,)) -> (u32,) {
        (lhs.0 + rhs.0,)
    }
}

#[cubecl::cube]
impl ReductionOp<WgpuRuntime, u32> for CountHitScalar {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        lhs + rhs
    }
}

#[test]
fn reduce_accepts_tuple_columns() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let sum = reduce(
        &exec,
        massively::Zip2(a.slice(..), b.slice(..)),
        (0.0_f32, 0_u32),
        TupleSum,
    )
    .unwrap();
    assert_eq!(sum, (6.0, 60));
}

#[test]
fn reduce_accepts_single_column_as_tuple_item() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let sum = reduce(&exec, massively::Zip1(a.slice(..)), (0.0_f32,), TupleSum).unwrap();
    assert_eq!(sum, (6.0,));
}

#[test]
fn reduce_accepts_lazy_unary_transform_to_tuple1() {
    let exec = exec();
    let ones = massively::lazy::transform(massively::lazy::counting(0).take(16), OneTupleU32);
    assert_eq!(reduce(&exec, ones, (0_u32,), CountHitTuple).unwrap(), (16,));
}

#[test]
fn reduce_accepts_lazy_unary_transform_to_scalar() {
    let exec = exec();
    let ones = massively::lazy::transform(massively::lazy::counting(0).take(16), OneScalarU32);
    assert_eq!(reduce(&exec, ones, 0_u32, CountHitScalar).unwrap(), 16);
}

#[test]
fn reduce_accepts_lazy_tuple_transform_to_tuple1() {
    let exec = exec();
    let hits = massively::lazy::transform(
        massively::Zip2(
            massively::lazy::constant(0.5_f32).take(16),
            massively::lazy::constant(0.5_f32).take(16),
        ),
        DetectHitTuple,
    );
    assert_eq!(reduce(&exec, hits, (0_u32,), CountHitTuple).unwrap(), (16,));
}

#[test]
fn reduce_accepts_lazy_tuple_transform_to_scalar() {
    let exec = exec();
    let hits = massively::lazy::transform(
        massively::Zip2(
            massively::lazy::constant(0.5_f32).take(16),
            massively::lazy::constant(0.5_f32).take(16),
        ),
        DetectHitScalar,
    );
    assert_eq!(reduce(&exec, hits, 0_u32, CountHitScalar).unwrap(), 16);
}

#[test]
fn reduce_accepts_lazy_tuple1_transform_to_scalar() {
    let exec = exec();
    let hits = massively::lazy::transform(
        massively::Zip1(massively::lazy::constant(0.5_f32).take(16)),
        DetectPositiveTuple1Scalar,
    );
    assert_eq!(reduce(&exec, hits, 0_u32, CountHitScalar).unwrap(), 16);
}

#[test]
fn reduce_accepts_device_tuple_transform_to_scalar() {
    let exec = exec();
    let x = exec.to_device(&[0.5_f32; 16]).unwrap();
    let y = exec.to_device(&[0.5_f32; 16]).unwrap();
    let hits =
        massively::lazy::transform(massively::Zip2(x.slice(..), y.slice(..)), DetectHitScalar);
    assert_eq!(reduce(&exec, hits, 0_u32, CountHitScalar).unwrap(), 16);
}

#[test]
fn reduce_accepts_three_column_tuple_item_op() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let c = exec.to_device(&[100.0_f32, 200.0, 300.0]).unwrap();
    let sum = reduce(
        &exec,
        massively::Zip3(a.slice(..), b.slice(..), c.slice(..)),
        (0.0_f32, 0_u32, 0.0_f32),
        TupleSum,
    )
    .unwrap();
    assert_eq!(sum, (6.0, 60, 600.0));
}

#[test]
fn reduce_accepts_seven_tuple_columns() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let c = exec.to_device(&[100.0_f32, 200.0, 300.0]).unwrap();
    let d = exec.to_device(&[1000_u32, 2000, 3000]).unwrap();
    let e = exec.to_device(&[10000.0_f32, 20000.0, 30000.0]).unwrap();
    let f = exec.to_device(&[100000_u32, 200000, 300000]).unwrap();
    let g = exec
        .to_device(&[1000000.0_f32, 2000000.0, 3000000.0])
        .unwrap();

    let sum = reduce(
        &exec,
        massively::Zip7(
            a.slice(..),
            b.slice(..),
            c.slice(..),
            d.slice(..),
            e.slice(..),
            f.slice(..),
            g.slice(..),
        ),
        (0.0_f32, 0_u32, 0.0_f32, 0_u32, 0.0_f32, 0_u32, 0.0_f32),
        TupleSum,
    )
    .unwrap();

    assert_eq!(sum, (6.0, 60, 600.0, 6000, 60000.0, 600000, 6000000.0));
}
