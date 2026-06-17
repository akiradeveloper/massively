mod common;
use common::*;

#[test]
fn by_key_apis_follow_concept_argument_order() {
    let policy = policy();
    let keys = policy.to_device(&[0_u32, 0, 1, 1]).unwrap();
    let values = policy.to_device(&[1.0_f32, 2.0, 10.0, 20.0]).unwrap();

    let exclusive = exclusive_scan_by_key(&keys, &values, EqualU32, 0.0, Sum).unwrap();
    assert_eq!(exclusive.to_vec().unwrap(), vec![0.0, 1.0, 0.0, 10.0]);

    let (out_keys, out_values) = reduce_by_key(&keys, &values, EqualU32, 0.0, Sum).unwrap();
    assert_eq!(out_keys.to_vec().unwrap(), vec![0, 1]);
    assert_eq!(out_values.to_vec().unwrap(), vec![3.0, 30.0]);
}

#[test]
fn reduce_by_key_uses_supplied_key_equality() {
    let policy = policy();
    let keys = policy.to_device(&[0_u32, 2, 4, 1, 3]).unwrap();
    let values = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0, 5.0]).unwrap();

    let (out_keys, out_values) = reduce_by_key(&keys, &values, SameParityU32, 0.0, Sum).unwrap();

    assert_eq!(out_keys.to_vec().unwrap(), vec![4, 3]);
    assert_eq!(out_values.to_vec().unwrap(), vec![6.0, 9.0]);
}

#[test]
fn reduce_by_key_with_tuple_values_uses_supplied_key_equality_for_every_value_column() {
    let policy = policy();
    let keys = policy.to_device(&[0_u32, 0, 1]).unwrap();
    let values = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let ids = policy.to_device(&[10_u32, 20, 30]).unwrap();

    let (out_keys, reduced) = reduce_by_key(
        &keys,
        zip(&values, &ids),
        NeverEqualU32,
        (100.0_f32, 1000_u32),
        Sum,
    )
    .unwrap();
    let (reduced_values, reduced_ids) = reduced;

    assert_eq!(out_keys.to_vec().unwrap(), vec![0, 0, 1]);
    assert_eq!(reduced_values.to_vec().unwrap(), vec![101.0, 102.0, 103.0]);
    assert_eq!(reduced_ids.to_vec().unwrap(), vec![1010, 1020, 1030]);
}

#[test]
fn exclusive_scan_by_key_with_tuple_values_uses_supplied_key_equality_for_every_value_column() {
    let policy = policy();
    let keys = policy.to_device(&[0_u32, 0, 1]).unwrap();
    let values = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let ids = policy.to_device(&[10_u32, 20, 30]).unwrap();

    let scanned = exclusive_scan_by_key(
        &keys,
        zip(&values, &ids),
        NeverEqualU32,
        (100.0_f32, 1000_u32),
        Sum,
    )
    .unwrap();
    let (scanned_values, scanned_ids) = scanned;

    assert_eq!(scanned_values.to_vec().unwrap(), vec![100.0, 100.0, 100.0]);
    assert_eq!(scanned_ids.to_vec().unwrap(), vec![1000, 1000, 1000]);
}

#[test]
fn by_key_reduce_and_scan_accept_heterogeneous_value_columns() {
    let policy = policy();
    let keys = policy.to_device(&[1_u32, 1, 2, 2, 2]).unwrap();
    let values = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let ids = policy.to_device(&[10_u32, 20, 30, 40, 50]).unwrap();

    let (out_keys, reduced) =
        reduce_by_key(&keys, zip(&values, &ids), EqualU32, (0.0_f32, 0_u32), Sum).unwrap();
    let (reduced_values, reduced_ids) = reduced;
    let out_keys = out_keys;
    assert_eq!(out_keys.to_vec().unwrap(), vec![1, 2]);
    assert_eq!(reduced_values.to_vec().unwrap(), vec![3.0, 12.0]);
    assert_eq!(reduced_ids.to_vec().unwrap(), vec![30, 120]);

    let inclusive = inclusive_scan_by_key(&keys, zip(&values, &ids), EqualU32, Sum).unwrap();
    let (inclusive_values, inclusive_ids) = inclusive;
    assert_eq!(
        inclusive_values.to_vec().unwrap(),
        vec![1.0, 3.0, 3.0, 7.0, 12.0]
    );
    assert_eq!(inclusive_ids.to_vec().unwrap(), vec![10, 30, 30, 70, 120]);

    let exclusive =
        exclusive_scan_by_key(&keys, zip(&values, &ids), EqualU32, (0.0_f32, 0_u32), Sum).unwrap();
    let (exclusive_values, exclusive_ids) = exclusive;
    assert_eq!(
        exclusive_values.to_vec().unwrap(),
        vec![0.0, 1.0, 0.0, 3.0, 7.0]
    );
    assert_eq!(exclusive_ids.to_vec().unwrap(), vec![0, 10, 0, 30, 70]);
}

#[test]
fn by_key_reduce_and_scan_accept_borrowed_heterogeneous_value_soas() {
    let policy = policy();
    let keys = policy.to_device(&[1_u32, 1, 2, 2, 2]).unwrap();

    let values = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let ids = policy.to_device(&[10_u32, 20, 30, 40, 50]).unwrap();
    let inclusive = inclusive_scan_by_key(&keys, zip(&values, &ids), EqualU32, Sum).unwrap();
    let (values, ids) = inclusive;
    assert_eq!(values.to_vec().unwrap(), vec![1.0, 3.0, 3.0, 7.0, 12.0]);
    assert_eq!(ids.to_vec().unwrap(), vec![10, 30, 30, 70, 120]);

    let values = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let ids = policy.to_device(&[10_u32, 20, 30, 40, 50]).unwrap();
    let exclusive =
        exclusive_scan_by_key(&keys, zip(&values, &ids), EqualU32, (0.0_f32, 0_u32), Sum).unwrap();
    let (values, ids) = exclusive;
    assert_eq!(values.to_vec().unwrap(), vec![0.0, 1.0, 0.0, 3.0, 7.0]);
    assert_eq!(ids.to_vec().unwrap(), vec![0, 10, 0, 30, 70]);

    let values = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let ids = policy.to_device(&[10_u32, 20, 30, 40, 50]).unwrap();
    let (out_keys, reduced) =
        reduce_by_key(&keys, zip(&values, &ids), EqualU32, (0.0_f32, 0_u32), Sum).unwrap();
    let (values, ids) = reduced;
    let out_keys = out_keys;
    assert_eq!(out_keys.to_vec().unwrap(), vec![1, 2]);
    assert_eq!(values.to_vec().unwrap(), vec![3.0, 12.0]);
    assert_eq!(ids.to_vec().unwrap(), vec![30, 120]);
}

#[test]
fn reduce_by_key_accepts_one_component_soa_values() {
    let policy = policy();
    let keys = policy.to_device(&[0_u32, 0, 1, 1, 1, 2]).unwrap();
    let values = policy
        .to_device(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap();

    let (out_keys, out_values) = reduce_by_key(&keys, &values, EqualU32, 0.0, Sum).unwrap();
    let out_keys = out_keys;
    let out_values = out_values;

    assert_eq!(out_keys.to_vec().unwrap(), vec![0, 1, 2]);
    assert_eq!(out_values.to_vec().unwrap(), vec![3.0, 12.0, 6.0]);

    let mapped = transform(zip(&values, &keys), PairScaleAndTag).unwrap();
    let (mapped, transformed_keys) = mapped;
    assert_eq!(transformed_keys.to_vec().unwrap(), vec![1, 1, 2, 2, 2, 3]);
    let (out_keys, out_values) = reduce_by_key(&keys, &mapped, EqualU32, 0.0, Sum).unwrap();
    let out_keys = out_keys;
    let out_values = out_values;

    assert_eq!(out_keys.to_vec().unwrap(), vec![0, 1, 2]);
    assert_eq!(out_values.to_vec().unwrap(), vec![6.0, 24.0, 12.0]);
}

#[test]
fn by_key_algorithms_accept_wide_soas() {
    let policy = policy();
    let keys = policy.to_device(&[0_u32, 0, 1]).unwrap();
    let a = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let c = policy.to_device(&[100.0_f32, 200.0, 300.0]).unwrap();
    let d = policy.to_device(&[1000_u32, 2000, 3000]).unwrap();

    let inclusive = inclusive_scan_by_key(&keys, zip4(&a, &b, &c, &d), EqualU32, Sum).unwrap();
    let (a_out, b_out, c_out, d_out) = inclusive;
    assert_eq!(a_out.to_vec().unwrap(), vec![1.0, 3.0, 3.0]);
    assert_eq!(b_out.to_vec().unwrap(), vec![10, 30, 30]);
    assert_eq!(c_out.to_vec().unwrap(), vec![100.0, 300.0, 300.0]);
    assert_eq!(d_out.to_vec().unwrap(), vec![1000, 3000, 3000]);

    let exclusive = exclusive_scan_by_key(
        &keys,
        zip4(&a, &b, &c, &d),
        EqualU32,
        (0.0, 100_u32, 1000.0, 10000_u32),
        Sum,
    )
    .unwrap();
    let (a_out, b_out, c_out, d_out) = exclusive;
    assert_eq!(a_out.to_vec().unwrap(), vec![0.0, 1.0, 0.0]);
    assert_eq!(b_out.to_vec().unwrap(), vec![100, 110, 100]);
    assert_eq!(c_out.to_vec().unwrap(), vec![1000.0, 1100.0, 1000.0]);
    assert_eq!(d_out.to_vec().unwrap(), vec![10000, 11000, 10000]);

    let (out_keys, out_values) = reduce_by_key(
        &keys,
        zip4(&a, &b, &c, &d),
        EqualU32,
        (0.0, 0_u32, 0.0, 0_u32),
        Sum,
    )
    .unwrap();
    let out_keys = out_keys;
    let (a_out, b_out, c_out, d_out) = out_values;
    assert_eq!(out_keys.to_vec().unwrap(), vec![0, 1]);
    assert_eq!(a_out.to_vec().unwrap(), vec![3.0, 3.0]);
    assert_eq!(b_out.to_vec().unwrap(), vec![30, 30]);
    assert_eq!(c_out.to_vec().unwrap(), vec![300.0, 300.0]);
    assert_eq!(d_out.to_vec().unwrap(), vec![3000, 3000]);
}

#[test]
fn reduce_by_key_accepts_soa12_values() {
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
    let l = policy.to_device(&[7000_u32, 8000, 9000]).unwrap();

    let (out_keys, out_values) = reduce_by_key(
        &keys,
        zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
        EqualU32,
        (
            0.0, 0_u32, 0.0, 0_u32, 0.0, 0_u32, 0.0, 0_u32, 0.0, 0_u32, 0.0, 0_u32,
        ),
        Sum,
    )
    .unwrap();
    let out_keys = out_keys;
    let (a, b, c, d, e, f, g, h, i, j, k, l) = out_values;
    assert_eq!(out_keys.to_vec().unwrap(), vec![0, 1]);
    assert_eq!(a.to_vec().unwrap(), vec![3.0, 3.0]);
    assert_eq!(b.to_vec().unwrap(), vec![30, 30]);
    assert_eq!(c.to_vec().unwrap(), vec![300.0, 300.0]);
    assert_eq!(d.to_vec().unwrap(), vec![3000, 3000]);
    assert_eq!(e.to_vec().unwrap(), vec![9.0, 6.0]);
    assert_eq!(f.to_vec().unwrap(), vec![90, 60]);
    assert_eq!(g.to_vec().unwrap(), vec![900.0, 600.0]);
    assert_eq!(h.to_vec().unwrap(), vec![9000, 6000]);
    assert_eq!(i.to_vec().unwrap(), vec![15.0, 9.0]);
    assert_eq!(j.to_vec().unwrap(), vec![150, 90]);
    assert_eq!(k.to_vec().unwrap(), vec![1500.0, 900.0]);
    assert_eq!(l.to_vec().unwrap(), vec![15000, 9000]);
}

#[test]
fn reduce_by_key_accepts_soa12_values_with_multiple_segments() {
    let policy = policy();
    let keys = policy.to_device(&[0_u32, 0, 1, 1, 1, 3]).unwrap();
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

    let (out_keys, out_values) = reduce_by_key(
        &keys,
        zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
        EqualU32,
        (
            0.0, 0_u32, 0.0, 0_u32, 0.0, 0_u32, 0.0, 0_u32, 0.0, 0_u32, 0.0, 0_u32,
        ),
        Sum,
    )
    .unwrap();
    let out_keys = out_keys;
    let (a, b, c, d, e, f, g, h, i, j, k, l) = out_values;

    assert_eq!(out_keys.to_vec().unwrap(), vec![0, 1, 3]);
    assert_eq!(a.to_vec().unwrap(), vec![3.0, 12.0, 6.0]);
    assert_eq!(b.to_vec().unwrap(), vec![30, 120, 60]);
    assert_eq!(c.to_vec().unwrap(), vec![300.0, 1200.0, 600.0]);
    assert_eq!(d.to_vec().unwrap(), vec![3000, 12000, 6000]);
    assert_eq!(e.to_vec().unwrap(), vec![15.0, 30.0, 12.0]);
    assert_eq!(f.to_vec().unwrap(), vec![150, 300, 120]);
    assert_eq!(g.to_vec().unwrap(), vec![1500.0, 3000.0, 1200.0]);
    assert_eq!(h.to_vec().unwrap(), vec![15000, 30000, 12000]);
    assert_eq!(i.to_vec().unwrap(), vec![27.0, 48.0, 18.0]);
    assert_eq!(j.to_vec().unwrap(), vec![270, 480, 180]);
    assert_eq!(k.to_vec().unwrap(), vec![2700.0, 4800.0, 1800.0]);
    assert_eq!(l.to_vec().unwrap(), vec![27000, 48000, 18000]);
}

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
