use crate::common::*;

#[test]
fn transform_zip_output_returns_storage() {
    let policy = policy();
    let values = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let tags = policy.to_device(&[10_u32, 20, 30]).unwrap();

    let output = transform(zip(&values, &tags), PairMixedSplit).unwrap();
    let (values, tags) = output;

    assert_eq!(values.to_vec().unwrap(), vec![11.0, 12.0, 13.0]);
    assert_eq!(tags.to_vec().unwrap(), vec![11, 21, 31]);
}

#[test]
fn transform_returns_device_storage() {
    let policy = policy();
    let left = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let right = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let split = transform(zip(&left, &right), PairMixedSplit).unwrap();
    let (values, tags) = split;

    assert_eq!(values.to_vec().unwrap(), vec![11.0, 12.0, 13.0]);
    assert_eq!(tags.to_vec().unwrap(), vec![11, 21, 31]);
}

#[test]
fn transform_tuple_output_maps_to_storage_output() {
    let policy = policy();
    let values = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let tags = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let bias = policy.to_device(&[100.0_f32, 200.0, 300.0]).unwrap();

    let split = transform(zip3(&values, &tags, &bias), Tuple3MixedSplit).unwrap();
    let (values, flags, bias) = split;
    assert_eq!(values.to_vec().unwrap(), vec![101.0, 202.0, 303.0]);
    assert_eq!(flags.to_vec().unwrap(), vec![11, 21, 31]);
    assert_eq!(bias.to_vec().unwrap(), vec![101.0, 202.0, 303.0]);
}

#[test]
fn scalar_transform_returns_soa1_storage() {
    let policy = policy();
    let input = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();

    let output = transform(&input, Double).unwrap();
    let output = output;

    assert_eq!(output.to_vec().unwrap(), vec![2.0, 4.0, 6.0]);
}

#[test]
fn unary_transform_accepts_wide_tuple_outputs() {
    let policy = policy();
    let input = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();

    let output = transform(&input, ScalarToTuple5Mixed).unwrap();
    let (a, b, c, d, e) = output;

    assert_eq!(a.to_vec().unwrap(), vec![2.0, 3.0, 4.0]);
    assert_eq!(b.to_vec().unwrap(), vec![3, 4, 5]);
    assert_eq!(c.to_vec().unwrap(), vec![4.0, 5.0, 6.0]);
    assert_eq!(d.to_vec().unwrap(), vec![5, 6, 7]);
    assert_eq!(e.to_vec().unwrap(), vec![6.0, 7.0, 8.0]);
}

#[test]
fn unary_transform_accepts_tuple12_output_and_checks_every_column() {
    let policy = policy();
    let input = policy.to_device(&[10_u32, 20]).unwrap();

    let output = transform(&input, ScalarToTuple12Mixed).unwrap();
    let (a, b, c, d, e, f, g, h, i, j, k, l) = output;

    assert_eq!(a.to_vec().unwrap(), vec![11.0, 21.0]);
    assert_eq!(b.to_vec().unwrap(), vec![12, 22]);
    assert_eq!(c.to_vec().unwrap(), vec![13.0, 23.0]);
    assert_eq!(d.to_vec().unwrap(), vec![14, 24]);
    assert_eq!(e.to_vec().unwrap(), vec![15.0, 25.0]);
    assert_eq!(f.to_vec().unwrap(), vec![16, 26]);
    assert_eq!(g.to_vec().unwrap(), vec![17.0, 27.0]);
    assert_eq!(h.to_vec().unwrap(), vec![18, 28]);
    assert_eq!(i.to_vec().unwrap(), vec![19.0, 29.0]);
    assert_eq!(j.to_vec().unwrap(), vec![20, 30]);
    assert_eq!(k.to_vec().unwrap(), vec![21.0, 31.0]);
    assert_eq!(l.to_vec().unwrap(), vec![22, 32]);
}

#[test]
fn tuple_transform_uses_flat_soa_input() {
    let policy = policy();
    let lhs = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let rhs = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let bias = policy.to_device(&[100.0_f32, 200.0, 300.0]).unwrap();

    let output = transform(zip3(&lhs, &rhs, &bias), Tuple3MixedSplit).unwrap();
    let (values, tags, adjusted_bias) = output;

    assert_eq!(values.to_vec().unwrap(), vec![101.0, 202.0, 303.0]);
    assert_eq!(tags.to_vec().unwrap(), vec![11, 21, 31]);
    assert_eq!(adjusted_bias.to_vec().unwrap(), vec![101.0, 202.0, 303.0]);
}

#[test]
fn transform_accepts_heterogeneous_tuple_inputs() {
    let policy = policy();
    let values = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let tags = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let bias = policy.to_device(&[100.0_f32, 200.0, 300.0]).unwrap();
    let pair_output = transform(zip(&values, &tags), PairMixedSplit).unwrap();
    let (pair_values, pair_tags) = pair_output;
    assert_eq!(pair_values.to_vec().unwrap(), vec![11.0, 12.0, 13.0]);
    assert_eq!(pair_tags.to_vec().unwrap(), vec![11, 21, 31]);

    let tuple3_output = transform(zip3(&values, &tags, &bias), Tuple3MixedSplit).unwrap();
    let (tuple_values, tuple_tags, tuple_bias) = tuple3_output;
    assert_eq!(tuple_values.to_vec().unwrap(), vec![101.0, 202.0, 303.0]);
    assert_eq!(tuple_tags.to_vec().unwrap(), vec![11, 21, 31]);
    assert_eq!(tuple_bias.to_vec().unwrap(), vec![101.0, 202.0, 303.0]);
}

#[test]
fn transform_accepts_soa4_heterogeneous_inputs_and_checks_every_column() {
    let policy = policy();
    let a = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let c = policy.to_device(&[100.0_f32, 200.0, 300.0]).unwrap();
    let d = policy.to_device(&[1000_u32, 2000, 3000]).unwrap();

    let output = transform(zip4(&a, &b, &c, &d), Tuple4MixedSplit).unwrap();
    let (a, b, c, d) = output;

    assert_eq!(a.to_vec().unwrap(), vec![101.0, 202.0, 303.0]);
    assert_eq!(b.to_vec().unwrap(), vec![12, 22, 32]);
    assert_eq!(c.to_vec().unwrap(), vec![101.0, 202.0, 303.0]);
    assert_eq!(d.to_vec().unwrap(), vec![1004, 2004, 3004]);
}

#[test]
fn transform_accepts_mismatched_input_and_output_tuple_widths() {
    let policy = policy();
    let a = policy.to_device(&[1.0_f32, 2.0]).unwrap();
    let b = policy.to_device(&[10_u32, 20]).unwrap();
    let c = policy.to_device(&[100.0_f32, 200.0]).unwrap();
    let d = policy.to_device(&[1000_u32, 2000]).unwrap();
    let e = policy.to_device(&[10000.0_f32, 20000.0]).unwrap();

    let out_5_to_3 = transform(zip5(&a, &b, &c, &d, &e), Tuple5To3MixedSplit).unwrap();
    let (x, y, z) = out_5_to_3;
    assert_eq!(x.to_vec().unwrap(), vec![10101.0, 20202.0]);
    assert_eq!(y.to_vec().unwrap(), vec![1010, 2020]);
    assert_eq!(z.to_vec().unwrap(), vec![9999.0, 19998.0]);

    let out_3_to_5 = transform(zip3(&a, &b, &c), Tuple3To5MixedSplit).unwrap();
    let (x, y, z, w, v) = out_3_to_5;
    assert_eq!(x.to_vec().unwrap(), vec![101.0, 202.0]);
    assert_eq!(y.to_vec().unwrap(), vec![20, 30]);
    assert_eq!(z.to_vec().unwrap(), vec![99.0, 198.0]);
    assert_eq!(w.to_vec().unwrap(), vec![30, 40]);
    assert_eq!(v.to_vec().unwrap(), vec![100.0, 400.0]);
}

#[test]
fn transform_accepts_extreme_mismatched_tuple_widths() {
    let policy = policy();
    let a = policy.to_device(&[1.0_f32, 2.0]).unwrap();
    let b = policy.to_device(&[10_u32, 20]).unwrap();

    let expanded = transform(zip(&a, &b), Tuple2To12MixedExpand).unwrap();
    let (a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, a6, b6) = expanded;
    assert_eq!(a1.to_vec().unwrap(), vec![2.0, 3.0]);
    assert_eq!(b1.to_vec().unwrap(), vec![12, 22]);
    assert_eq!(a2.to_vec().unwrap(), vec![4.0, 5.0]);
    assert_eq!(b2.to_vec().unwrap(), vec![14, 24]);
    assert_eq!(a3.to_vec().unwrap(), vec![6.0, 7.0]);
    assert_eq!(b3.to_vec().unwrap(), vec![16, 26]);
    assert_eq!(a4.to_vec().unwrap(), vec![8.0, 9.0]);
    assert_eq!(b4.to_vec().unwrap(), vec![18, 28]);
    assert_eq!(a5.to_vec().unwrap(), vec![10.0, 11.0]);
    assert_eq!(b5.to_vec().unwrap(), vec![20, 30]);
    assert_eq!(a6.to_vec().unwrap(), vec![12.0, 13.0]);
    assert_eq!(b6.to_vec().unwrap(), vec![22, 32]);

    let c = policy.to_device(&[100.0_f32, 200.0]).unwrap();
    let d = policy.to_device(&[1000_u32, 2000]).unwrap();
    let e = policy.to_device(&[10000.0_f32, 20000.0]).unwrap();
    let f = policy.to_device(&[100000_u32, 200000]).unwrap();
    let g = policy.to_device(&[1000000.0_f32, 2000000.0]).unwrap();
    let h = policy.to_device(&[7_u32, 8]).unwrap();
    let i = policy.to_device(&[70.0_f32, 80.0]).unwrap();
    let j = policy.to_device(&[700_u32, 800]).unwrap();
    let k = policy.to_device(&[7000.0_f32, 8000.0]).unwrap();
    let l = policy.to_device(&[70000_u32, 80000]).unwrap();

    let projected = transform(
        zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
        Tuple12To2MixedProject,
    )
    .unwrap();
    let (x, y) = projected;
    assert_eq!(x.to_vec().unwrap(), vec![7101.0, 8202.0]);
    assert_eq!(y.to_vec().unwrap(), vec![170010, 280020]);
}

#[test]
fn transform_accepts_soa5_to_soa11_heterogeneous_tuple_outputs() {
    let policy = policy();
    let a = policy.to_device(&[1.0_f32]).unwrap();
    let b = policy.to_device(&[10_u32]).unwrap();
    let c = policy.to_device(&[100.0_f32]).unwrap();
    let d = policy.to_device(&[1000_u32]).unwrap();
    let e = policy.to_device(&[10000.0_f32]).unwrap();
    let f = policy.to_device(&[100000_u32]).unwrap();
    let g = policy.to_device(&[1000000.0_f32]).unwrap();
    let h = policy.to_device(&[7_u32]).unwrap();
    let i = policy.to_device(&[70.0_f32]).unwrap();
    let j = policy.to_device(&[700_u32]).unwrap();
    let k = policy.to_device(&[7000.0_f32]).unwrap();

    let (a5, b5, c5, d5, e5) = transform(zip5(&a, &b, &c, &d, &e), TupleWideMixedSplit).unwrap();
    assert_eq!(a5.to_vec().unwrap(), vec![2.0]);
    assert_eq!(b5.to_vec().unwrap(), vec![12]);
    assert_eq!(c5.to_vec().unwrap(), vec![103.0]);
    assert_eq!(d5.to_vec().unwrap(), vec![1004]);
    assert_eq!(e5.to_vec().unwrap(), vec![10005.0]);

    let (a6, b6, c6, d6, e6, f6) =
        transform(zip6(&a, &b, &c, &d, &e, &f), TupleWideMixedSplit).unwrap();
    assert_eq!(a6.to_vec().unwrap(), vec![2.0]);
    assert_eq!(b6.to_vec().unwrap(), vec![12]);
    assert_eq!(c6.to_vec().unwrap(), vec![103.0]);
    assert_eq!(d6.to_vec().unwrap(), vec![1004]);
    assert_eq!(e6.to_vec().unwrap(), vec![10005.0]);
    assert_eq!(f6.to_vec().unwrap(), vec![100006]);

    let (a7, b7, c7, d7, e7, f7, g7) =
        transform(zip7(&a, &b, &c, &d, &e, &f, &g), TupleWideMixedSplit).unwrap();
    assert_eq!(a7.to_vec().unwrap(), vec![2.0]);
    assert_eq!(b7.to_vec().unwrap(), vec![12]);
    assert_eq!(c7.to_vec().unwrap(), vec![103.0]);
    assert_eq!(d7.to_vec().unwrap(), vec![1004]);
    assert_eq!(e7.to_vec().unwrap(), vec![10005.0]);
    assert_eq!(f7.to_vec().unwrap(), vec![100006]);
    assert_eq!(g7.to_vec().unwrap(), vec![1000007.0]);

    let (a8, b8, c8, d8, e8, f8, g8, h8) =
        transform(zip8(&a, &b, &c, &d, &e, &f, &g, &h), TupleWideMixedSplit).unwrap();
    assert_eq!(a8.to_vec().unwrap(), vec![2.0]);
    assert_eq!(b8.to_vec().unwrap(), vec![12]);
    assert_eq!(c8.to_vec().unwrap(), vec![103.0]);
    assert_eq!(d8.to_vec().unwrap(), vec![1004]);
    assert_eq!(e8.to_vec().unwrap(), vec![10005.0]);
    assert_eq!(f8.to_vec().unwrap(), vec![100006]);
    assert_eq!(g8.to_vec().unwrap(), vec![1000007.0]);
    assert_eq!(h8.to_vec().unwrap(), vec![15]);

    let (a9, b9, c9, d9, e9, f9, g9, h9, i9) = transform(
        zip9(&a, &b, &c, &d, &e, &f, &g, &h, &i),
        TupleWideMixedSplit,
    )
    .unwrap();
    assert_eq!(a9.to_vec().unwrap(), vec![2.0]);
    assert_eq!(b9.to_vec().unwrap(), vec![12]);
    assert_eq!(c9.to_vec().unwrap(), vec![103.0]);
    assert_eq!(d9.to_vec().unwrap(), vec![1004]);
    assert_eq!(e9.to_vec().unwrap(), vec![10005.0]);
    assert_eq!(f9.to_vec().unwrap(), vec![100006]);
    assert_eq!(g9.to_vec().unwrap(), vec![1000007.0]);
    assert_eq!(h9.to_vec().unwrap(), vec![15]);
    assert_eq!(i9.to_vec().unwrap(), vec![79.0]);

    let (a10, b10, c10, d10, e10, f10, g10, h10, i10, j10) = transform(
        zip10(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j),
        TupleWideMixedSplit,
    )
    .unwrap();
    assert_eq!(a10.to_vec().unwrap(), vec![2.0]);
    assert_eq!(b10.to_vec().unwrap(), vec![12]);
    assert_eq!(c10.to_vec().unwrap(), vec![103.0]);
    assert_eq!(d10.to_vec().unwrap(), vec![1004]);
    assert_eq!(e10.to_vec().unwrap(), vec![10005.0]);
    assert_eq!(f10.to_vec().unwrap(), vec![100006]);
    assert_eq!(g10.to_vec().unwrap(), vec![1000007.0]);
    assert_eq!(h10.to_vec().unwrap(), vec![15]);
    assert_eq!(i10.to_vec().unwrap(), vec![79.0]);
    assert_eq!(j10.to_vec().unwrap(), vec![710]);

    let (a11, b11, c11, d11, e11, f11, g11, h11, i11, j11, k11) = transform(
        zip11(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k),
        TupleWideMixedSplit,
    )
    .unwrap();
    assert_eq!(a11.to_vec().unwrap(), vec![2.0]);
    assert_eq!(b11.to_vec().unwrap(), vec![12]);
    assert_eq!(c11.to_vec().unwrap(), vec![103.0]);
    assert_eq!(d11.to_vec().unwrap(), vec![1004]);
    assert_eq!(e11.to_vec().unwrap(), vec![10005.0]);
    assert_eq!(f11.to_vec().unwrap(), vec![100006]);
    assert_eq!(g11.to_vec().unwrap(), vec![1000007.0]);
    assert_eq!(h11.to_vec().unwrap(), vec![15]);
    assert_eq!(i11.to_vec().unwrap(), vec![79.0]);
    assert_eq!(j11.to_vec().unwrap(), vec![710]);
    assert_eq!(k11.to_vec().unwrap(), vec![7011.0]);
}

#[test]
fn transform_accepts_soa12_heterogeneous_inputs_and_checks_every_column() {
    let policy = policy();
    let a = policy.to_device(&[1.0_f32, 2.0]).unwrap();
    let b = policy.to_device(&[10_u32, 20]).unwrap();
    let c = policy.to_device(&[100.0_f32, 200.0]).unwrap();
    let d = policy.to_device(&[1000_u32, 2000]).unwrap();
    let e = policy.to_device(&[10000.0_f32, 20000.0]).unwrap();
    let f = policy.to_device(&[100000_u32, 200000]).unwrap();
    let g = policy.to_device(&[1000000.0_f32, 2000000.0]).unwrap();
    let h = policy.to_device(&[7_u32, 8]).unwrap();
    let i = policy.to_device(&[70.0_f32, 80.0]).unwrap();
    let j = policy.to_device(&[700_u32, 800]).unwrap();
    let k = policy.to_device(&[7000.0_f32, 8000.0]).unwrap();
    let l = policy.to_device(&[70000_u32, 80000]).unwrap();

    let output = transform(
        zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
        Tuple12MixedSplit,
    )
    .unwrap();
    let (a, b, c, d, e, f, g, h, i, j, k, l) = output;

    assert_eq!(a.to_vec().unwrap(), vec![2.0, 3.0]);
    assert_eq!(b.to_vec().unwrap(), vec![12, 22]);
    assert_eq!(c.to_vec().unwrap(), vec![103.0, 203.0]);
    assert_eq!(d.to_vec().unwrap(), vec![1004, 2004]);
    assert_eq!(e.to_vec().unwrap(), vec![10005.0, 20005.0]);
    assert_eq!(f.to_vec().unwrap(), vec![100006, 200006]);
    assert_eq!(g.to_vec().unwrap(), vec![1000007.0, 2000007.0]);
    assert_eq!(h.to_vec().unwrap(), vec![15, 16]);
    assert_eq!(i.to_vec().unwrap(), vec![79.0, 89.0]);
    assert_eq!(j.to_vec().unwrap(), vec![710, 810]);
    assert_eq!(k.to_vec().unwrap(), vec![7011.0, 8011.0]);
    assert_eq!(l.to_vec().unwrap(), vec![70012, 80012]);
}

#[test]
fn transform_accepts_soa12_heterogeneous_inputs_to_scalar_output() {
    let policy = policy();
    let a = policy.to_device(&[1.0_f32, 2.0]).unwrap();
    let b = policy.to_device(&[10_u32, 20]).unwrap();
    let c = policy.to_device(&[100.0_f32, 200.0]).unwrap();
    let d = policy.to_device(&[1000_u32, 2000]).unwrap();
    let e = policy.to_device(&[3.0_f32, 4.0]).unwrap();
    let f = policy.to_device(&[30_u32, 40]).unwrap();
    let g = policy.to_device(&[300.0_f32, 400.0]).unwrap();
    let h = policy.to_device(&[3000_u32, 4000]).unwrap();
    let i = policy.to_device(&[5.0_f32, 6.0]).unwrap();
    let j = policy.to_device(&[50_u32, 60]).unwrap();
    let k = policy.to_device(&[500.0_f32, 600.0]).unwrap();
    let l = policy.to_device(&[5000_u32, 6000]).unwrap();

    let output = transform(
        zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
        Tuple12MixedChecksum,
    )
    .unwrap();
    let output = output;

    assert_eq!(output.to_vec().unwrap(), vec![9999.0, 13332.0]);
}

#[test]
fn transform_zip_flattens_soa1_columns() {
    let policy = policy();
    let left = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let right = policy.to_device(&[10_u32, 20, 30]).unwrap();

    let split = transform(zip(&left, &right), PairMixedSplit).unwrap();
    let (values, tags) = split;

    assert_eq!(values.to_vec().unwrap(), vec![11.0, 12.0, 13.0]);
    assert_eq!(tags.to_vec().unwrap(), vec![11, 21, 31]);
}
