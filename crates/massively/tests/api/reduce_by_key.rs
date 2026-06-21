use crate::common::*;

#[test]
fn reduce_by_key_uses_supplied_key_equality() {
    let exec = exec();
    let keys = exec.to_device(&[0_u32, 2, 4, 1, 3]).unwrap();
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0, 5.0]).unwrap();

    let (keys, values) = reduce_by_key(
        &exec,
        (keys.slice(..),),
        (values.slice(..),),
        SameParityU32,
        (0.0,),
        Sum,
    )
    .unwrap();
    let (keys,) = keys;
    let (values,) = values;
    assert_eq!(exec.to_host(&keys).unwrap(), vec![4, 3]);
    assert_eq!(exec.to_host(&values).unwrap(), vec![6.0, 9.0]);
}

#[test]
fn reduce_by_key_accepts_tuple_values() {
    let exec = exec();
    let keys = exec.to_device(&[1_u32, 1, 2, 2, 2]).unwrap();
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let ids = exec.to_device(&[10_u32, 20, 30, 40, 50]).unwrap();

    let (keys, values) = reduce_by_key(
        &exec,
        (keys.slice(..),),
        (values.slice(..), ids.slice(..)),
        EqualU32,
        (0.0_f32, 0_u32),
        TupleSum,
    )
    .unwrap();
    let (keys,) = keys;
    let (values, ids) = values;
    assert_eq!(exec.to_host(&keys).unwrap(), vec![1, 2]);
    assert_eq!(exec.to_host(&values).unwrap(), vec![3.0, 12.0]);
    assert_eq!(exec.to_host(&ids).unwrap(), vec![30, 120]);
}

#[test]
fn reduce_by_key_accepts_three_tuple_values() {
    let exec = exec();
    let keys = exec.to_device(&[1_u32, 1, 2, 2, 2]).unwrap();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 30, 40, 50]).unwrap();
    let c = exec
        .to_device(&[100.0_f32, 200.0, 300.0, 400.0, 500.0])
        .unwrap();

    let (keys, values) = reduce_by_key(
        &exec,
        (keys.slice(..),),
        (a.slice(..), b.slice(..), c.slice(..)),
        EqualU32,
        (0.0_f32, 0_u32, 0.0_f32),
        TupleSum,
    )
    .unwrap();
    let (keys,) = keys;
    let (a, b, c) = values;
    assert_eq!(exec.to_host(&keys).unwrap(), vec![1, 2]);
    assert_eq!(exec.to_host(&a).unwrap(), vec![3.0, 12.0]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![30, 120]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![300.0, 1200.0]);
}

#[cfg(any())]
#[test]
fn reduce_by_key_accepts_tuple_keys() {
    let exec = exec();
    let key_a = exec.to_device(&[1.0_f32, 1.0, 2.0, 2.0, 2.0]).unwrap();
    let key_b = exec.to_device(&[10_u32, 10, 20, 20, 30]).unwrap();
    let values = exec.to_device(&[1_u32, 2, 3, 4, 5]).unwrap();

    let (keys, values) = reduce_by_key(
        &exec,
        (key_a.slice(..), key_b.slice(..)),
        (values.slice(..),),
        MixedTupleEqual,
        0_u32,
        Sum,
    )
    .unwrap();
    let (key_a, key_b) = keys;
    let (values,) = values;
    assert_eq!(exec.to_host(&key_a).unwrap(), vec![1.0, 2.0, 2.0]);
    assert_eq!(exec.to_host(&key_b).unwrap(), vec![10, 20, 30]);
    assert_eq!(exec.to_host(&values).unwrap(), vec![3, 7, 5]);
}
