use crate::common::*;

#[cfg(any())]
#[test]
fn sort_by_key_accepts_borrowed_tuple_keys() {
    let policy = policy();
    let key_a = policy.to_device(&[3.0_f32, 1.0, 2.0]).unwrap();
    let key_b = policy.to_device(&[30_u32, 10, 20]).unwrap();
    let values = policy.to_device(&[30_u32, 10, 20]).unwrap();

    let (keys, values) = sort_by_key((&key_a, &key_b), &values, MixedTupleLess).unwrap();
    let (key_a, key_b) = keys;
    let (values,) = values;
    assert_eq!(key_a.to_vec().unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(key_b.to_vec().unwrap(), vec![10, 20, 30]);
    assert_eq!(values.to_vec().unwrap(), vec![10, 20, 30]);
}

#[test]
fn sort_by_key_accepts_tuple_values() {
    let policy = policy();
    let keys = policy.to_device(&[3_u32, 1, 2]).unwrap();
    let a = policy.to_device(&[30_u32, 10, 20]).unwrap();
    let b = policy.to_device(&[30.0_f32, 10.0, 20.0]).unwrap();
    let c = policy.to_device(&[300_u32, 100, 200]).unwrap();

    let (keys, values) = sort_by_key((&keys,), (&a, &b, &c), LessU32).unwrap();
    let (keys,) = keys;
    let (a, b, c) = values;
    assert_eq!(keys.to_vec().unwrap(), vec![1, 2, 3]);
    assert_eq!(a.to_vec().unwrap(), vec![10, 20, 30]);
    assert_eq!(b.to_vec().unwrap(), vec![10.0, 20.0, 30.0]);
    assert_eq!(c.to_vec().unwrap(), vec![100, 200, 300]);
}

#[cfg(any())]
#[test]
fn sort_by_key_reports_value_length_mismatch() {
    let policy = policy();
    let key_a = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let key_b = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let values = policy.to_device(&[1_u32, 2]).unwrap();

    let err = sort_by_key((&key_a, &key_b), &values, MixedTupleLess).unwrap_err();
    assert!(matches!(err, massively::Error::LengthMismatch { .. }));
}
