use crate::common::*;

#[cfg(any())]
#[test]
fn exclusive_scan_by_key_accepts_tuple_keys() {
    let exec = exec();
    let key_a = exec.to_device(&[1.0_f32, 1.0, 2.0, 2.0, 2.0]).unwrap();
    let key_b = exec.to_device(&[10_u32, 10, 20, 20, 30]).unwrap();
    let values = exec.to_device(&[1_u32, 2, 3, 4, 5]).unwrap();

    let output = exclusive_scan_by_key(
        &exec,
        (key_a.slice(..), key_b.slice(..)),
        (values.slice(..),),
        MixedTupleEqual,
        (0_u32,),
        Sum,
    )
    .unwrap();
    let (output,) = output;
    assert_eq!(exec.to_host(&output).unwrap(), vec![0, 1, 0, 3, 0]);
}

#[test]
fn exclusive_scan_by_key_uses_supplied_key_equality() {
    let exec = exec();
    let keys = exec.to_device(&[0_u32, 0, 1]).unwrap();
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let ids = exec.to_device(&[10_u32, 20, 30]).unwrap();

    let output = exclusive_scan_by_key(
        &exec,
        (keys.slice(..),),
        (values.slice(..), ids.slice(..)),
        NeverEqualU32,
        (100.0_f32, 1000_u32),
        TupleSum,
    )
    .unwrap();
    let (values, ids) = output;
    assert_eq!(exec.to_host(&values).unwrap(), vec![100.0, 100.0, 100.0]);
    assert_eq!(exec.to_host(&ids).unwrap(), vec![1000, 1000, 1000]);
}

#[test]
fn exclusive_scan_by_key_accepts_tuple_values() {
    let exec = exec();
    let keys = exec.to_device(&[0_u32, 0, 1, 1, 1, 2]).unwrap();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 30, 40, 50, 60]).unwrap();
    let c = exec
        .to_device(&[100.0_f32, 200.0, 300.0, 400.0, 500.0, 600.0])
        .unwrap();

    let output = exclusive_scan_by_key(
        &exec,
        (keys.slice(..),),
        (a.slice(..), b.slice(..), c.slice(..)),
        EqualU32,
        (0.0_f32, 0_u32, 0.0_f32),
        TupleSum,
    )
    .unwrap();
    let (a, b, c) = output;
    assert_eq!(
        exec.to_host(&a).unwrap(),
        vec![0.0, 1.0, 0.0, 3.0, 7.0, 0.0]
    );
    assert_eq!(exec.to_host(&b).unwrap(), vec![0, 10, 0, 30, 70, 0]);
    assert_eq!(
        exec.to_host(&c).unwrap(),
        vec![0.0, 100.0, 0.0, 300.0, 700.0, 0.0]
    );
}
