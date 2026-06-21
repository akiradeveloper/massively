use crate::common::*;

#[test]
fn exclusive_scan_accepts_tuple_columns() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let output = exclusive_scan(
        &exec,
        (a.slice(..), b.slice(..)),
        (0.0_f32, 0_u32),
        TupleSum,
    )
    .unwrap();
    let (a, b) = output;
    assert_eq!(exec.to_host(&a).unwrap(), vec![0.0, 1.0, 3.0]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![0, 10, 30]);
}

#[test]
fn exclusive_scan_accepts_single_column_as_tuple_item() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();

    let (a,) = exclusive_scan(&exec, (a.slice(..),), (0.0_f32,), TupleSum).unwrap();

    assert_eq!(exec.to_host(&a).unwrap(), vec![0.0, 1.0, 3.0]);
}

#[test]
fn exclusive_scan_accepts_three_column_tuple_item_op() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let c = exec.to_device(&[100.0_f32, 200.0, 300.0]).unwrap();

    let (a, b, c) = exclusive_scan(
        &exec,
        (a.slice(..), b.slice(..), c.slice(..)),
        (0.0_f32, 0_u32, 0.0_f32),
        TupleSum,
    )
    .unwrap();

    assert_eq!(exec.to_host(&a).unwrap(), vec![0.0, 1.0, 3.0]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![0, 10, 30]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![0.0, 100.0, 300.0]);
}
