use crate::common::*;

#[test]
fn unique_by_key_accepts_tuple_values() {
    let policy = policy();
    let keys = policy.to_device(&[0_u32, 0, 1]).unwrap();
    let a = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let c = policy.to_device(&[100.0_f32, 200.0, 300.0]).unwrap();

    let (keys, values) = unique_by_key(
        (keys.slice(..),),
        (a.slice(..), b.slice(..), c.slice(..)),
        EqualU32,
    )
    .unwrap();
    let (keys,) = keys;
    let (a, b, c) = values;

    assert_eq!(keys.to_vec().unwrap(), vec![0, 1]);
    assert_eq!(a.to_vec().unwrap(), vec![1.0, 3.0]);
    assert_eq!(b.to_vec().unwrap(), vec![10, 30]);
    assert_eq!(c.to_vec().unwrap(), vec![100.0, 300.0]);
}

#[test]
fn unique_by_key_accepts_tuple_values_with_multiple_runs() {
    let policy = policy();
    let keys = policy.to_device(&[0_u32, 0, 0, 2, 3, 3]).unwrap();
    let a = policy
        .to_device(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap();
    let b = policy.to_device(&[10_u32, 20, 30, 40, 50, 60]).unwrap();
    let c = policy
        .to_device(&[100.0_f32, 200.0, 300.0, 400.0, 500.0, 600.0])
        .unwrap();

    let (keys, values) = unique_by_key(
        (keys.slice(..),),
        (a.slice(..), b.slice(..), c.slice(..)),
        EqualU32,
    )
    .unwrap();
    let (keys,) = keys;
    let (a, b, c) = values;

    assert_eq!(keys.to_vec().unwrap(), vec![0, 2, 3]);
    assert_eq!(a.to_vec().unwrap(), vec![1.0, 4.0, 5.0]);
    assert_eq!(b.to_vec().unwrap(), vec![10, 40, 50]);
    assert_eq!(c.to_vec().unwrap(), vec![100.0, 400.0, 500.0]);
}

#[cfg(any())]
#[test]
fn unique_by_key_accepts_borrowed_tuple_keys() {
    let policy = policy();
    let key_a = policy.to_device(&[1.0_f32, 1.0, 2.0, 2.0, 2.0]).unwrap();
    let key_b = policy.to_device(&[10_u32, 10, 20, 20, 30]).unwrap();
    let values = policy.to_device(&[100_u32, 101, 200, 201, 300]).unwrap();

    let (keys, values) =
        unique_by_key((key_a.slice(..), key_b.slice(..)), values, MixedTupleEqual).unwrap();
    let (key_a, key_b) = keys;
    let (values,) = values;

    assert_eq!(key_a.to_vec().unwrap(), vec![1.0, 2.0, 2.0]);
    assert_eq!(key_b.to_vec().unwrap(), vec![10, 20, 30]);
    assert_eq!(values.to_vec().unwrap(), vec![100, 200, 300]);
}

#[cfg(any())]
#[test]
fn unique_by_tuple_key_reports_value_length_mismatch_for_wide_values() {
    let policy = policy();
    let key_a = policy.to_device(&[1.0_f32, 1.0, 2.0]).unwrap();
    let key_b = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let value_a = policy.to_device(&[1_u32, 2]).unwrap();
    let value_b = policy.to_device(&[10.0_f32, 20.0]).unwrap();
    let value_c = policy.to_device(&[100_u32, 200]).unwrap();
    let value_d = policy.to_device(&[1000.0_f32, 2000.0]).unwrap();

    let err = unique_by_key(
        zip(key_a.slice(..), key_b.slice(..)),
        zip4(
            value_a.slice(..),
            value_b.slice(..),
            value_c.slice(..),
            value_d.slice(..),
        ),
        MixedTupleEqual,
    )
    .unwrap_err();

    assert_eq!(
        err,
        massively::Error::LengthMismatch {
            input: 2,
            output: 3
        }
    );
}

#[cfg(any())]
#[test]
fn unique_by_tuple_key_with_wide_values_uses_supplied_key_equality() {
    let policy = policy();
    let key_a = policy.to_device(&[1.0_f32, 1.0, 2.0, 2.0]).unwrap();
    let key_b = policy.to_device(&[10_u32, 20, 10, 20]).unwrap();
    let value_a = policy.to_device(&[1_u32, 2, 3, 4]).unwrap();
    let value_b = policy.to_device(&[10.0_f32, 20.0, 30.0, 40.0]).unwrap();
    let value_c = policy.to_device(&[100_u32, 200, 300, 400]).unwrap();
    let value_d = policy
        .to_device(&[1000.0_f32, 2000.0, 3000.0, 4000.0])
        .unwrap();

    let (keys, values) = unique_by_key(
        zip(key_a.slice(..), key_b.slice(..)),
        zip4(
            value_a.slice(..),
            value_b.slice(..),
            value_c.slice(..),
            value_d.slice(..),
        ),
        MixedTupleFirstEqual,
    )
    .unwrap();
    let (key_a, key_b) = keys;
    let (value_a, value_b, value_c, value_d) = values;

    assert_eq!(key_a.to_vec().unwrap(), vec![1.0, 2.0]);
    assert_eq!(key_b.to_vec().unwrap(), vec![10, 10]);
    assert_eq!(value_a.to_vec().unwrap(), vec![1, 3]);
    assert_eq!(value_b.to_vec().unwrap(), vec![10.0, 30.0]);
    assert_eq!(value_c.to_vec().unwrap(), vec![100, 300]);
    assert_eq!(value_d.to_vec().unwrap(), vec![1000.0, 3000.0]);
}
