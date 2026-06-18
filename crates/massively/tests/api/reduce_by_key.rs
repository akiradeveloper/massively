use crate::common::*;

#[test]
fn reduce_by_key_uses_supplied_key_equality() {
    let policy = policy();
    let keys = policy.to_device(&[0_u32, 2, 4, 1, 3]).unwrap();
    let values = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0, 5.0]).unwrap();

    let (keys, values) = reduce_by_key(&keys, &values, SameParityU32, 0.0, Sum).unwrap();
    assert_eq!(keys.to_vec().unwrap(), vec![4, 3]);
    assert_eq!(values.to_vec().unwrap(), vec![6.0, 9.0]);
}

#[test]
fn reduce_by_key_accepts_tuple_values() {
    let policy = policy();
    let keys = policy.to_device(&[1_u32, 1, 2, 2, 2]).unwrap();
    let values = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let ids = policy.to_device(&[10_u32, 20, 30, 40, 50]).unwrap();

    let (keys, values) =
        reduce_by_key(&keys, zip(&values, &ids), EqualU32, (0.0_f32, 0_u32), Sum).unwrap();
    let (values, ids) = values;
    assert_eq!(keys.to_vec().unwrap(), vec![1, 2]);
    assert_eq!(values.to_vec().unwrap(), vec![3.0, 12.0]);
    assert_eq!(ids.to_vec().unwrap(), vec![30, 120]);
}

#[test]
fn reduce_by_key_accepts_tuple_keys() {
    let policy = policy();
    let key_a = policy.to_device(&[1.0_f32, 1.0, 2.0, 2.0, 2.0]).unwrap();
    let key_b = policy.to_device(&[10_u32, 10, 20, 20, 30]).unwrap();
    let values = policy.to_device(&[1_u32, 2, 3, 4, 5]).unwrap();

    let (keys, values) =
        reduce_by_key(zip(&key_a, &key_b), &values, MixedTupleEqual, 0_u32, Sum).unwrap();
    let (key_a, key_b) = keys;
    assert_eq!(key_a.to_vec().unwrap(), vec![1.0, 2.0, 2.0]);
    assert_eq!(key_b.to_vec().unwrap(), vec![10, 20, 30]);
    assert_eq!(values.to_vec().unwrap(), vec![3, 7, 5]);
}
