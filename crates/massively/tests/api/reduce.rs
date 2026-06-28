use crate::common::*;

#[test]
fn reduce_accepts_tuple_columns() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let sum = reduce(
        &exec,
        massively::SoA2(a.slice(..), b.slice(..)),
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
    let sum = reduce(&exec, massively::SoA1(a.slice(..)), (0.0_f32,), TupleSum).unwrap();
    assert_eq!(sum, (6.0,));
}

#[test]
fn reduce_accepts_three_column_tuple_item_op() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let c = exec.to_device(&[100.0_f32, 200.0, 300.0]).unwrap();
    let sum = reduce(
        &exec,
        massively::SoA3(a.slice(..), b.slice(..), c.slice(..)),
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
        massively::SoA7(
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
