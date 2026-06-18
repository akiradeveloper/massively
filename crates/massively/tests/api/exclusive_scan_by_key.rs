use crate::common::*;

#[test]
fn exclusive_scan_by_key_accepts_tuple_keys() {
    let policy = policy();
    let key_a = policy.to_device(&[1.0_f32, 1.0, 2.0, 2.0, 2.0]).unwrap();
    let key_b = policy.to_device(&[10_u32, 10, 20, 20, 30]).unwrap();
    let values = policy.to_device(&[1_u32, 2, 3, 4, 5]).unwrap();

    let output =
        exclusive_scan_by_key(zip(&key_a, &key_b), &values, MixedTupleEqual, 0_u32, Sum).unwrap();
    assert_eq!(output.to_vec().unwrap(), vec![0, 1, 0, 3, 0]);
}

#[test]
fn exclusive_scan_by_key_uses_supplied_key_equality() {
    let policy = policy();
    let keys = policy.to_device(&[0_u32, 0, 1]).unwrap();
    let values = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let ids = policy.to_device(&[10_u32, 20, 30]).unwrap();

    let output = exclusive_scan_by_key(
        &keys,
        zip(&values, &ids),
        NeverEqualU32,
        (100.0_f32, 1000_u32),
        Sum,
    )
    .unwrap();
    let (values, ids) = output;
    assert_eq!(values.to_vec().unwrap(), vec![100.0, 100.0, 100.0]);
    assert_eq!(ids.to_vec().unwrap(), vec![1000, 1000, 1000]);
}
