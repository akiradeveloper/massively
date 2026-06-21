use crate::common::*;

#[cfg(any())]
#[test]
fn sort_by_key_accepts_borrowed_tuple_keys() {
    let exec = exec();
    let key_a = exec.to_device(&[3.0_f32, 1.0, 2.0]).unwrap();
    let key_b = exec.to_device(&[30_u32, 10, 20]).unwrap();
    let values = exec.to_device(&[30_u32, 10, 20]).unwrap();

    let (keys, values) = sort_by_key(
        &exec,
        (key_a.slice(..), key_b.slice(..)),
        (values.slice(..),),
        MixedTupleLess,
    )
    .unwrap();
    let (key_a, key_b) = keys;
    let (values,) = values;
    assert_eq!(exec.to_host(&key_a).unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(exec.to_host(&key_b).unwrap(), vec![10, 20, 30]);
    assert_eq!(exec.to_host(&values).unwrap(), vec![10, 20, 30]);
}

#[test]
fn sort_by_key_accepts_tuple_values() {
    let exec = exec();
    let keys = exec.to_device(&[3_u32, 1, 2]).unwrap();
    let a = exec.to_device(&[30_u32, 10, 20]).unwrap();
    let b = exec.to_device(&[30.0_f32, 10.0, 20.0]).unwrap();
    let c = exec.to_device(&[300_u32, 100, 200]).unwrap();

    let (keys, values) = sort_by_key(
        &exec,
        (keys.slice(..),),
        (a.slice(..), b.slice(..), c.slice(..)),
        LessU32,
    )
    .unwrap();
    let (keys,) = keys;
    let (a, b, c) = values;
    assert_eq!(exec.to_host(&keys).unwrap(), vec![1, 2, 3]);
    assert_eq!(exec.to_host(&a).unwrap(), vec![10, 20, 30]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![10.0, 20.0, 30.0]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![100, 200, 300]);
}

#[cfg(any())]
#[test]
fn sort_by_key_reports_value_length_mismatch() {
    let exec = exec();
    let key_a = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let key_b = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let values = exec.to_device(&[1_u32, 2]).unwrap();

    let err = sort_by_key(
        &exec,
        (key_a.slice(..), key_b.slice(..)),
        (values.slice(..),),
        MixedTupleLess,
    )
    .unwrap_err();
    assert!(matches!(err, massively::Error::LengthMismatch { .. }));
}
