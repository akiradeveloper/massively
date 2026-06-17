mod common;
use common::*;

#[test]
fn by_key_reduce_and_scan_accept_heterogeneous_value_columns() {
    let policy = policy();
    let keys = policy.to_device(&[1_u32, 1, 2, 2, 2]).unwrap();
    let values = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let ids = policy.to_device(&[10_u32, 20, 30, 40, 50]).unwrap();

    let (out_keys, reduced) =
        reduce_by_key(vzip(&values, &ids), &keys, (0.0_f32, 0_u32), Sum).unwrap();
    let (reduced_values, reduced_ids) = unzip(reduced).unwrap();
    let out_keys = unzip(out_keys).unwrap();
    assert_eq!(out_keys.to_vec().unwrap(), vec![1, 2]);
    assert_eq!(reduced_values.to_vec().unwrap(), vec![3.0, 12.0]);
    assert_eq!(reduced_ids.to_vec().unwrap(), vec![30, 120]);

    let inclusive = inclusive_scan_by_key(vzip(&values, &ids), &keys, EqualU32, Sum).unwrap();
    let (inclusive_values, inclusive_ids) = unzip(inclusive).unwrap();
    assert_eq!(
        inclusive_values.to_vec().unwrap(),
        vec![1.0, 3.0, 3.0, 7.0, 12.0]
    );
    assert_eq!(inclusive_ids.to_vec().unwrap(), vec![10, 30, 30, 70, 120]);

    let exclusive =
        exclusive_scan_by_key(vzip(&values, &ids), &keys, (0.0_f32, 0_u32), EqualU32, Sum).unwrap();
    let (exclusive_values, exclusive_ids) = unzip(exclusive).unwrap();
    assert_eq!(
        exclusive_values.to_vec().unwrap(),
        vec![0.0, 1.0, 0.0, 3.0, 7.0]
    );
    assert_eq!(exclusive_ids.to_vec().unwrap(), vec![0, 10, 0, 30, 70]);
}

#[test]
fn reduce_by_key_accepts_one_component_device_soa_values() {
    let policy = policy();
    let keys = policy.to_device(&[0_u32, 0, 1, 1, 1, 2]).unwrap();
    let values = policy
        .to_device(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap();

    let (out_keys, out_values) = reduce_by_key(&values, &keys, 0.0, Sum).unwrap();
    let out_keys = unzip(out_keys).unwrap();
    let out_values = unzip(out_values).unwrap();

    assert_eq!(out_keys.to_vec().unwrap(), vec![0, 1, 2]);
    assert_eq!(out_values.to_vec().unwrap(), vec![3.0, 12.0, 6.0]);

    let mapped = transform(vzip(&values, &keys), PairScaleAndTag).unwrap();
    let (mapped, transformed_keys) = unzip(mapped).unwrap();
    assert_eq!(transformed_keys.to_vec().unwrap(), vec![1, 1, 2, 2, 2, 3]);
    let (out_keys, out_values) = reduce_by_key(&mapped, &keys, 0.0, Sum).unwrap();
    let out_keys = unzip(out_keys).unwrap();
    let out_values = unzip(out_values).unwrap();

    assert_eq!(out_keys.to_vec().unwrap(), vec![0, 1, 2]);
    assert_eq!(out_values.to_vec().unwrap(), vec![6.0, 24.0, 12.0]);
}

#[test]
fn by_key_algorithms_accept_wide_sovas() {
    let policy = policy();
    let keys = policy.to_device(&[0_u32, 0, 1]).unwrap();
    let a = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let c = policy.to_device(&[100.0_f32, 200.0, 300.0]).unwrap();
    let d = policy.to_device(&[1000_u32, 2000, 3000]).unwrap();

    let inclusive = inclusive_scan_by_key(vzip4(&a, &b, &c, &d), &keys, EqualU32, Sum).unwrap();
    let (a_out, b_out, c_out, d_out) = unzip(inclusive).unwrap();
    assert_eq!(a_out.to_vec().unwrap(), vec![1.0, 3.0, 3.0]);
    assert_eq!(b_out.to_vec().unwrap(), vec![10, 30, 30]);
    assert_eq!(c_out.to_vec().unwrap(), vec![100.0, 300.0, 300.0]);
    assert_eq!(d_out.to_vec().unwrap(), vec![1000, 3000, 3000]);

    let exclusive = exclusive_scan_by_key(
        vzip4(&a, &b, &c, &d),
        &keys,
        (0.0, 100_u32, 1000.0, 10000_u32),
        EqualU32,
        Sum,
    )
    .unwrap();
    let (a_out, b_out, c_out, d_out) = unzip(exclusive).unwrap();
    assert_eq!(a_out.to_vec().unwrap(), vec![0.0, 1.0, 0.0]);
    assert_eq!(b_out.to_vec().unwrap(), vec![100, 110, 100]);
    assert_eq!(c_out.to_vec().unwrap(), vec![1000.0, 1100.0, 1000.0]);
    assert_eq!(d_out.to_vec().unwrap(), vec![10000, 11000, 10000]);

    let (out_keys, out_values) =
        reduce_by_key(vzip4(&a, &b, &c, &d), &keys, (0.0, 0_u32, 0.0, 0_u32), Sum).unwrap();
    let out_keys = unzip(out_keys).unwrap();
    let (a_out, b_out, c_out, d_out) = unzip(out_values).unwrap();
    assert_eq!(out_keys.to_vec().unwrap(), vec![0, 1]);
    assert_eq!(a_out.to_vec().unwrap(), vec![3.0, 3.0]);
    assert_eq!(b_out.to_vec().unwrap(), vec![30, 30]);
    assert_eq!(c_out.to_vec().unwrap(), vec![300.0, 300.0]);
    assert_eq!(d_out.to_vec().unwrap(), vec![3000, 3000]);
}

#[test]
fn reduce_by_key_accepts_sova12_values() {
    let policy = policy();
    let keys = policy.to_device(&[0_u32, 0, 1]).unwrap();
    let a = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = policy.to_device(&[1_u32, 2, 3]).unwrap();
    let c = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let d = policy.to_device(&[1_u32, 2, 3]).unwrap();
    let e = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let f = policy.to_device(&[1_u32, 2, 3]).unwrap();
    let g = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let h = policy.to_device(&[1_u32, 2, 3]).unwrap();
    let i = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let j = policy.to_device(&[1_u32, 2, 3]).unwrap();
    let k = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let l = policy.to_device(&[1_u32, 2, 3]).unwrap();

    let (_out_keys, out_values) = reduce_by_key(
        vzip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
        &keys,
        (
            0.0, 0_u32, 0.0, 0_u32, 0.0, 0_u32, 0.0, 0_u32, 0.0, 0_u32, 0.0, 0_u32,
        ),
        Sum,
    )
    .unwrap();
    let (_a, _b, _c, _d, _e, _f, _g, _h, _i, _j, _k, l_out) = unzip(out_values).unwrap();
    assert_eq!(l_out.to_vec().unwrap(), vec![3, 3]);
}

#[test]
fn unique_by_key_accepts_soa12_values() {
    let policy = policy();
    let keys = policy.to_device(&[0_u32, 0, 1]).unwrap();
    let a = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let c = policy.to_device(&[10.0_f32, 20.0, 30.0]).unwrap();
    let d = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let e = policy.to_device(&[10.0_f32, 20.0, 30.0]).unwrap();
    let f = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let g = policy.to_device(&[10.0_f32, 20.0, 30.0]).unwrap();
    let h = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let i = policy.to_device(&[10.0_f32, 20.0, 30.0]).unwrap();
    let j = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let k = policy.to_device(&[10.0_f32, 20.0, 30.0]).unwrap();
    let l = policy.to_device(&[100_u32, 200, 300]).unwrap();

    let (keys, values) =
        unique_by_key(keys, zip12(a, b, c, d, e, f, g, h, i, j, k, l), EqualU32).unwrap();
    let keys = unzip(keys).unwrap();
    let (a, _b, _c, _d, _e, _f, _g, _h, _i, _j, _k, l) = unzip(values).unwrap();

    assert_eq!(keys.to_vec().unwrap(), vec![0, 1]);
    assert_eq!(a.to_vec().unwrap(), vec![1.0, 3.0]);
    assert_eq!(l.to_vec().unwrap(), vec![100, 300]);
}
