use crate::common::*;

#[cfg(any())]
#[test]
fn unique_by_key_accepts_soa12_values() {
    let policy = policy();
    let keys = policy.to_device(&[0_u32, 0, 1]).unwrap();
    let a = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let c = policy.to_device(&[100.0_f32, 200.0, 300.0]).unwrap();
    let d = policy.to_device(&[1000_u32, 2000, 3000]).unwrap();
    let e = policy.to_device(&[4.0_f32, 5.0, 6.0]).unwrap();
    let f = policy.to_device(&[40_u32, 50, 60]).unwrap();
    let g = policy.to_device(&[400.0_f32, 500.0, 600.0]).unwrap();
    let h = policy.to_device(&[4000_u32, 5000, 6000]).unwrap();
    let i = policy.to_device(&[7.0_f32, 8.0, 9.0]).unwrap();
    let j = policy.to_device(&[70_u32, 80, 90]).unwrap();
    let k = policy.to_device(&[700.0_f32, 800.0, 900.0]).unwrap();
    let l = policy.to_device(&[100_u32, 200, 300]).unwrap();

    let (keys, values) = unique_by_key(
        keys,
        zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
        EqualU32,
    )
    .unwrap();
    let keys = keys;
    let (a, b, c, d, e, f, g, h, i, j, k, l) = values;

    assert_eq!(keys.to_vec().unwrap(), vec![0, 1]);
    assert_eq!(a.to_vec().unwrap(), vec![1.0, 3.0]);
    assert_eq!(b.to_vec().unwrap(), vec![10, 30]);
    assert_eq!(c.to_vec().unwrap(), vec![100.0, 300.0]);
    assert_eq!(d.to_vec().unwrap(), vec![1000, 3000]);
    assert_eq!(e.to_vec().unwrap(), vec![4.0, 6.0]);
    assert_eq!(f.to_vec().unwrap(), vec![40, 60]);
    assert_eq!(g.to_vec().unwrap(), vec![400.0, 600.0]);
    assert_eq!(h.to_vec().unwrap(), vec![4000, 6000]);
    assert_eq!(i.to_vec().unwrap(), vec![7.0, 9.0]);
    assert_eq!(j.to_vec().unwrap(), vec![70, 90]);
    assert_eq!(k.to_vec().unwrap(), vec![700.0, 900.0]);
    assert_eq!(l.to_vec().unwrap(), vec![100, 300]);
}

#[cfg(any())]
#[test]
fn unique_by_key_accepts_soa12_values_with_multiple_runs() {
    let policy = policy();
    let keys = policy.to_device(&[0_u32, 0, 0, 2, 3, 3]).unwrap();
    let a = policy
        .to_device(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap();
    let b = policy.to_device(&[10_u32, 20, 30, 40, 50, 60]).unwrap();
    let c = policy
        .to_device(&[100.0_f32, 200.0, 300.0, 400.0, 500.0, 600.0])
        .unwrap();
    let d = policy
        .to_device(&[1000_u32, 2000, 3000, 4000, 5000, 6000])
        .unwrap();
    let e = policy
        .to_device(&[7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0])
        .unwrap();
    let f = policy.to_device(&[70_u32, 80, 90, 100, 110, 120]).unwrap();
    let g = policy
        .to_device(&[700.0_f32, 800.0, 900.0, 1000.0, 1100.0, 1200.0])
        .unwrap();
    let h = policy
        .to_device(&[7000_u32, 8000, 9000, 10000, 11000, 12000])
        .unwrap();
    let i = policy
        .to_device(&[13.0_f32, 14.0, 15.0, 16.0, 17.0, 18.0])
        .unwrap();
    let j = policy
        .to_device(&[130_u32, 140, 150, 160, 170, 180])
        .unwrap();
    let k = policy
        .to_device(&[1300.0_f32, 1400.0, 1500.0, 1600.0, 1700.0, 1800.0])
        .unwrap();
    let l = policy
        .to_device(&[13000_u32, 14000, 15000, 16000, 17000, 18000])
        .unwrap();

    let (keys, values) = unique_by_key(
        keys,
        zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
        EqualU32,
    )
    .unwrap();
    let keys = keys;
    let (a, b, c, d, e, f, g, h, i, j, k, l) = values;

    assert_eq!(keys.to_vec().unwrap(), vec![0, 2, 3]);
    assert_eq!(a.to_vec().unwrap(), vec![1.0, 4.0, 5.0]);
    assert_eq!(b.to_vec().unwrap(), vec![10, 40, 50]);
    assert_eq!(c.to_vec().unwrap(), vec![100.0, 400.0, 500.0]);
    assert_eq!(d.to_vec().unwrap(), vec![1000, 4000, 5000]);
    assert_eq!(e.to_vec().unwrap(), vec![7.0, 10.0, 11.0]);
    assert_eq!(f.to_vec().unwrap(), vec![70, 100, 110]);
    assert_eq!(g.to_vec().unwrap(), vec![700.0, 1000.0, 1100.0]);
    assert_eq!(h.to_vec().unwrap(), vec![7000, 10000, 11000]);
    assert_eq!(i.to_vec().unwrap(), vec![13.0, 16.0, 17.0]);
    assert_eq!(j.to_vec().unwrap(), vec![130, 160, 170]);
    assert_eq!(k.to_vec().unwrap(), vec![1300.0, 1600.0, 1700.0]);
    assert_eq!(l.to_vec().unwrap(), vec![13000, 16000, 17000]);
}

#[test]
fn unique_by_key_accepts_borrowed_tuple_keys() {
    let policy = policy();
    let key_a = policy.to_device(&[1.0_f32, 1.0, 2.0, 2.0, 2.0]).unwrap();
    let key_b = policy.to_device(&[10_u32, 10, 20, 20, 30]).unwrap();
    let values = policy.to_device(&[100_u32, 101, 200, 201, 300]).unwrap();

    let (keys, values) = unique_by_key((&key_a, &key_b), values, MixedTupleEqual).unwrap();
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
        zip(&key_a, &key_b),
        zip4(&value_a, &value_b, &value_c, &value_d),
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
        zip(&key_a, &key_b),
        zip4(&value_a, &value_b, &value_c, &value_d),
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
