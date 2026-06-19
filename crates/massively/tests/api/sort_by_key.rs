use crate::common::*;

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

#[cfg(any())]
#[test]
fn sort_by_key_accepts_soa12_values() {
    let policy = policy();
    let keys = policy.to_device(&[3_u32, 1, 2]).unwrap();
    let a = policy.to_device(&[30_u32, 10, 20]).unwrap();
    let b = policy.to_device(&[30.0_f32, 10.0, 20.0]).unwrap();
    let c = policy.to_device(&[300_u32, 100, 200]).unwrap();
    let d = policy.to_device(&[300.0_f32, 100.0, 200.0]).unwrap();
    let e = policy.to_device(&[31_u32, 11, 21]).unwrap();
    let f = policy.to_device(&[31.0_f32, 11.0, 21.0]).unwrap();
    let g = policy.to_device(&[32_u32, 12, 22]).unwrap();
    let h = policy.to_device(&[32.0_f32, 12.0, 22.0]).unwrap();
    let i = policy.to_device(&[33_u32, 13, 23]).unwrap();
    let j = policy.to_device(&[33.0_f32, 13.0, 23.0]).unwrap();
    let k = policy.to_device(&[34_u32, 14, 24]).unwrap();
    let l = policy.to_device(&[34.0_f32, 14.0, 24.0]).unwrap();

    let (keys, values) = sort_by_key(
        &keys,
        zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
        LessU32,
    )
    .unwrap();
    let (a, _, _, _, _, _, _, _, _, _, _, l) = values;
    assert_eq!(keys.to_vec().unwrap(), vec![1, 2, 3]);
    assert_eq!(a.to_vec().unwrap(), vec![10, 20, 30]);
    assert_eq!(l.to_vec().unwrap(), vec![14.0, 24.0, 34.0]);
}

#[test]
fn sort_by_key_reports_value_length_mismatch() {
    let policy = policy();
    let key_a = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let key_b = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let values = policy.to_device(&[1_u32, 2]).unwrap();

    let err = sort_by_key((&key_a, &key_b), &values, MixedTupleLess).unwrap_err();
    assert!(matches!(err, massively::Error::LengthMismatch { .. }));
}
