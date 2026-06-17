mod common;
use common::*;

#[test]
fn by_key_sort_merge_unique_accept_declared_key_value_layouts() {
    let policy = policy();

    let key_a = policy.to_device(&[3.0_f32, 1.0, 2.0]).unwrap();
    let key_b = policy.to_device(&[30.0_f32, 10.0, 20.0]).unwrap();
    let key_c = policy.to_device(&[3.1_f32, 1.1, 2.1]).unwrap();
    let key_d = policy.to_device(&[31.0_f32, 11.0, 21.0]).unwrap();
    let va = policy.to_device(&[30_u32, 10, 20]).unwrap();
    let vb = policy.to_device(&[30.0_f32, 10.0, 20.0]).unwrap();
    let vc = policy.to_device(&[300_u32, 100, 200]).unwrap();
    let vd = policy.to_device(&[300.0_f32, 100.0, 200.0]).unwrap();
    let ve = policy.to_device(&[31_u32, 11, 21]).unwrap();
    let vf = policy.to_device(&[31.0_f32, 11.0, 21.0]).unwrap();
    let vg = policy.to_device(&[32_u32, 12, 22]).unwrap();
    let vh = policy.to_device(&[32.0_f32, 12.0, 22.0]).unwrap();
    let vi = policy.to_device(&[33_u32, 13, 23]).unwrap();
    let vj = policy.to_device(&[33.0_f32, 13.0, 23.0]).unwrap();
    let vk = policy.to_device(&[34_u32, 14, 24]).unwrap();
    let vl = policy.to_device(&[34.0_f32, 14.0, 24.0]).unwrap();

    let sort_key_a = policy.to_device(&[3.0_f32, 1.0, 2.0]).unwrap();
    let sort_key_b = policy.to_device(&[30.0_f32, 10.0, 20.0]).unwrap();
    let sort_key_c = policy.to_device(&[3.1_f32, 1.1, 2.1]).unwrap();
    let sort_key_d = policy.to_device(&[31.0_f32, 11.0, 21.0]).unwrap();
    let sort_va = policy.to_device(&[30_u32, 10, 20]).unwrap();
    let sort_vb = policy.to_device(&[30.0_f32, 10.0, 20.0]).unwrap();
    let sort_vc = policy.to_device(&[300_u32, 100, 200]).unwrap();
    let sort_vd = policy.to_device(&[300.0_f32, 100.0, 200.0]).unwrap();
    let sort_ve = policy.to_device(&[31_u32, 11, 21]).unwrap();
    let sort_vf = policy.to_device(&[31.0_f32, 11.0, 21.0]).unwrap();
    let sort_vg = policy.to_device(&[32_u32, 12, 22]).unwrap();
    let sort_vh = policy.to_device(&[32.0_f32, 12.0, 22.0]).unwrap();
    let sort_vi = policy.to_device(&[33_u32, 13, 23]).unwrap();
    let sort_vj = policy.to_device(&[33.0_f32, 13.0, 23.0]).unwrap();
    let sort_vk = policy.to_device(&[34_u32, 14, 24]).unwrap();
    let sort_vl = policy.to_device(&[34.0_f32, 14.0, 24.0]).unwrap();
    let (keys, values) = sort_by_key(
        zip4(&sort_key_a, &sort_key_b, &sort_key_c, &sort_key_d),
        zip12(
            &sort_va, &sort_vb, &sort_vc, &sort_vd, &sort_ve, &sort_vf, &sort_vg, &sort_vh,
            &sort_vi, &sort_vj, &sort_vk, &sort_vl,
        ),
        Tuple4Less,
    )
    .unwrap();
    let (out_a, _, _, _) = keys;
    let (out_va, _, _, _, _, _, _, _, _, _, _, out_vl) = values;
    assert_eq!(out_a.to_vec().unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(out_va.to_vec().unwrap(), vec![10, 20, 30]);
    assert_eq!(out_vl.to_vec().unwrap(), vec![14.0, 24.0, 34.0]);

    let right_key_a = policy.to_device(&[3.5_f32, 1.5, 2.5]).unwrap();
    let right_key_b = policy.to_device(&[35.0_f32, 15.0, 25.0]).unwrap();
    let right_key_c = policy.to_device(&[3.6_f32, 1.6, 2.6]).unwrap();
    let right_key_d = policy.to_device(&[36.0_f32, 16.0, 26.0]).unwrap();
    let (keys, values) = merge_by_key(
        zip4(&key_a, &key_b, &key_c, &key_d),
        zip12(&va, &vb, &vc, &vd, &ve, &vf, &vg, &vh, &vi, &vj, &vk, &vl),
        zip4(&right_key_a, &right_key_b, &right_key_c, &right_key_d),
        zip12(&va, &vb, &vc, &vd, &ve, &vf, &vg, &vh, &vi, &vj, &vk, &vl),
        Tuple4Less,
    )
    .unwrap();
    let (out_a, _, _, _) = keys;
    let (out_va, _, _, _, _, _, _, _, _, _, _, out_vl) = values;
    assert_eq!(out_a.to_vec().unwrap(), vec![1.0, 1.5, 2.0, 2.5, 3.0, 3.5]);
    assert_eq!(out_va.to_vec().unwrap(), vec![10, 10, 20, 20, 30, 30]);
    assert_eq!(
        out_vl.to_vec().unwrap(),
        vec![14.0, 14.0, 24.0, 24.0, 34.0, 34.0]
    );

    let (keys, values) = unique_by_key(
        zip4(&key_a, &key_b, &key_c, &key_d),
        zip12(&va, &vb, &vc, &vd, &ve, &vf, &vg, &vh, &vi, &vj, &vk, &vl),
        Tuple4Equal,
    )
    .unwrap();
    let (out_a, _, _, _) = keys;
    let (out_va, _, _, _, _, _, _, _, _, _, _, out_vl) = values;
    assert_eq!(out_a.to_vec().unwrap(), vec![3.0, 1.0, 2.0]);
    assert_eq!(out_va.to_vec().unwrap(), vec![30, 10, 20]);
    assert_eq!(out_vl.to_vec().unwrap(), vec![34.0, 14.0, 24.0]);
}

#[test]
fn by_key_algorithms_accept_soa12_keys_and_soa12_values() {
    let policy = policy();

    let key_a = policy.to_device(&[1.0_f32, 1.0, 2.0, 2.0, 3.0]).unwrap();
    let key_b = policy.to_device(&[10_u32, 10, 20, 20, 30]).unwrap();
    let key_c = policy.to_device(&[1.1_f32, 1.1, 2.1, 2.1, 3.1]).unwrap();
    let key_d = policy.to_device(&[11_u32, 11, 21, 21, 31]).unwrap();
    let key_e = policy.to_device(&[1.2_f32, 1.2, 2.2, 2.2, 3.2]).unwrap();
    let key_f = policy.to_device(&[12_u32, 12, 22, 22, 32]).unwrap();
    let key_g = policy.to_device(&[1.3_f32, 1.3, 2.3, 2.3, 3.3]).unwrap();
    let key_h = policy.to_device(&[13_u32, 13, 23, 23, 33]).unwrap();
    let key_i = policy.to_device(&[1.4_f32, 1.4, 2.4, 2.4, 3.4]).unwrap();
    let key_j = policy.to_device(&[14_u32, 14, 24, 24, 34]).unwrap();
    let key_k = policy.to_device(&[1.5_f32, 1.5, 2.5, 2.5, 3.5]).unwrap();
    let key_l = policy.to_device(&[15_u32, 15, 25, 25, 35]).unwrap();

    let va = policy.to_device(&[1_u32, 2, 3, 4, 5]).unwrap();
    let vb = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let vc = policy.to_device(&[10_u32, 20, 30, 40, 50]).unwrap();
    let vd = policy
        .to_device(&[10.0_f32, 20.0, 30.0, 40.0, 50.0])
        .unwrap();
    let ve = policy.to_device(&[100_u32, 200, 300, 400, 500]).unwrap();
    let vf = policy
        .to_device(&[100.0_f32, 200.0, 300.0, 400.0, 500.0])
        .unwrap();
    let vg = policy.to_device(&[7_u32, 8, 9, 10, 11]).unwrap();
    let vh = policy.to_device(&[7.0_f32, 8.0, 9.0, 10.0, 11.0]).unwrap();
    let vi = policy.to_device(&[70_u32, 80, 90, 100, 110]).unwrap();
    let vj = policy
        .to_device(&[70.0_f32, 80.0, 90.0, 100.0, 110.0])
        .unwrap();
    let vk = policy.to_device(&[700_u32, 800, 900, 1000, 1100]).unwrap();
    let vl = policy
        .to_device(&[700.0_f32, 800.0, 900.0, 1000.0, 1100.0])
        .unwrap();

    let values = zip12(&va, &vb, &vc, &vd, &ve, &vf, &vg, &vh, &vi, &vj, &vk, &vl);
    let keys = zip12(
        &key_a, &key_b, &key_c, &key_d, &key_e, &key_f, &key_g, &key_h, &key_i, &key_j, &key_k,
        &key_l,
    );
    let inclusive = inclusive_scan_by_key(keys, values, Tuple12MixedEqual, Sum).unwrap();
    let (oa, ob, oc, od, oe, of, og, oh, oi, oj, ok, ol) = inclusive;
    assert_eq!(oa.to_vec().unwrap(), vec![1, 3, 3, 7, 5]);
    assert_eq!(ob.to_vec().unwrap(), vec![1.0, 3.0, 3.0, 7.0, 5.0]);
    assert_eq!(oc.to_vec().unwrap(), vec![10, 30, 30, 70, 50]);
    assert_eq!(od.to_vec().unwrap(), vec![10.0, 30.0, 30.0, 70.0, 50.0]);
    assert_eq!(oe.to_vec().unwrap(), vec![100, 300, 300, 700, 500]);
    assert_eq!(
        of.to_vec().unwrap(),
        vec![100.0, 300.0, 300.0, 700.0, 500.0]
    );
    assert_eq!(og.to_vec().unwrap(), vec![7, 15, 9, 19, 11]);
    assert_eq!(oh.to_vec().unwrap(), vec![7.0, 15.0, 9.0, 19.0, 11.0]);
    assert_eq!(oi.to_vec().unwrap(), vec![70, 150, 90, 190, 110]);
    assert_eq!(oj.to_vec().unwrap(), vec![70.0, 150.0, 90.0, 190.0, 110.0]);
    assert_eq!(ok.to_vec().unwrap(), vec![700, 1500, 900, 1900, 1100]);
    assert_eq!(
        ol.to_vec().unwrap(),
        vec![700.0, 1500.0, 900.0, 1900.0, 1100.0]
    );

    let values = zip12(&va, &vb, &vc, &vd, &ve, &vf, &vg, &vh, &vi, &vj, &vk, &vl);
    let keys = zip12(
        &key_a, &key_b, &key_c, &key_d, &key_e, &key_f, &key_g, &key_h, &key_i, &key_j, &key_k,
        &key_l,
    );
    let exclusive = exclusive_scan_by_key(
        keys,
        values,
        Tuple12MixedEqual,
        (
            0_u32, 0.0_f32, 0_u32, 0.0_f32, 0_u32, 0.0_f32, 0_u32, 0.0_f32, 0_u32, 0.0_f32, 0_u32,
            0.0_f32,
        ),
        Sum,
    )
    .unwrap();
    let (oa, ob, oc, od, oe, of, og, oh, oi, oj, ok, ol) = exclusive;
    assert_eq!(oa.to_vec().unwrap(), vec![0, 1, 0, 3, 0]);
    assert_eq!(ob.to_vec().unwrap(), vec![0.0, 1.0, 0.0, 3.0, 0.0]);
    assert_eq!(oc.to_vec().unwrap(), vec![0, 10, 0, 30, 0]);
    assert_eq!(od.to_vec().unwrap(), vec![0.0, 10.0, 0.0, 30.0, 0.0]);
    assert_eq!(oe.to_vec().unwrap(), vec![0, 100, 0, 300, 0]);
    assert_eq!(of.to_vec().unwrap(), vec![0.0, 100.0, 0.0, 300.0, 0.0]);
    assert_eq!(og.to_vec().unwrap(), vec![0, 7, 0, 9, 0]);
    assert_eq!(oh.to_vec().unwrap(), vec![0.0, 7.0, 0.0, 9.0, 0.0]);
    assert_eq!(oi.to_vec().unwrap(), vec![0, 70, 0, 90, 0]);
    assert_eq!(oj.to_vec().unwrap(), vec![0.0, 70.0, 0.0, 90.0, 0.0]);
    assert_eq!(ok.to_vec().unwrap(), vec![0, 700, 0, 900, 0]);
    assert_eq!(ol.to_vec().unwrap(), vec![0.0, 700.0, 0.0, 900.0, 0.0]);

    let values = zip12(&va, &vb, &vc, &vd, &ve, &vf, &vg, &vh, &vi, &vj, &vk, &vl);
    let keys = zip12(
        &key_a, &key_b, &key_c, &key_d, &key_e, &key_f, &key_g, &key_h, &key_i, &key_j, &key_k,
        &key_l,
    );
    let (keys, values) = reduce_by_key(
        keys,
        values,
        Tuple12MixedEqual,
        (
            0_u32, 0.0_f32, 0_u32, 0.0_f32, 0_u32, 0.0_f32, 0_u32, 0.0_f32, 0_u32, 0.0_f32, 0_u32,
            0.0_f32,
        ),
        Sum,
    )
    .unwrap();
    let (ka, kb, _, _, _, _, _, _, _, _, kk, kl) = keys;
    let (oa, ob, oc, od, oe, of, og, oh, oi, oj, ok, ol) = values;
    assert_eq!(ka.to_vec().unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(kb.to_vec().unwrap(), vec![10, 20, 30]);
    assert_eq!(kk.to_vec().unwrap(), vec![1.5, 2.5, 3.5]);
    assert_eq!(kl.to_vec().unwrap(), vec![15, 25, 35]);
    assert_eq!(oa.to_vec().unwrap(), vec![3, 7, 5]);
    assert_eq!(ob.to_vec().unwrap(), vec![3.0, 7.0, 5.0]);
    assert_eq!(oc.to_vec().unwrap(), vec![30, 70, 50]);
    assert_eq!(od.to_vec().unwrap(), vec![30.0, 70.0, 50.0]);
    assert_eq!(oe.to_vec().unwrap(), vec![300, 700, 500]);
    assert_eq!(of.to_vec().unwrap(), vec![300.0, 700.0, 500.0]);
    assert_eq!(og.to_vec().unwrap(), vec![15, 19, 11]);
    assert_eq!(oh.to_vec().unwrap(), vec![15.0, 19.0, 11.0]);
    assert_eq!(oi.to_vec().unwrap(), vec![150, 190, 110]);
    assert_eq!(oj.to_vec().unwrap(), vec![150.0, 190.0, 110.0]);
    assert_eq!(ok.to_vec().unwrap(), vec![1500, 1900, 1100]);
    assert_eq!(ol.to_vec().unwrap(), vec![1500.0, 1900.0, 1100.0]);
}

#[test]
fn scan_and_reduce_by_tuple_key_accept_borrowed_heterogeneous_value_soas() {
    let policy = policy();
    let key_a = policy.to_device(&[1.0_f32, 1.0, 2.0, 2.0, 2.0]).unwrap();
    let key_b = policy.to_device(&[10_u32, 10, 20, 20, 30]).unwrap();

    let values = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let ids = policy.to_device(&[10_u32, 20, 30, 40, 50]).unwrap();
    let inclusive = inclusive_scan_by_key(
        zip(&key_a, &key_b),
        zip(&values, &ids),
        MixedTupleEqual,
        Sum,
    )
    .unwrap();
    let (values, ids) = inclusive;
    assert_eq!(values.to_vec().unwrap(), vec![1.0, 3.0, 3.0, 7.0, 5.0]);
    assert_eq!(ids.to_vec().unwrap(), vec![10, 30, 30, 70, 50]);

    let values = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let ids = policy.to_device(&[10_u32, 20, 30, 40, 50]).unwrap();
    let exclusive = exclusive_scan_by_key(
        zip(&key_a, &key_b),
        zip(&values, &ids),
        MixedTupleEqual,
        (0.0_f32, 0_u32),
        Sum,
    )
    .unwrap();
    let (values, ids) = exclusive;
    assert_eq!(values.to_vec().unwrap(), vec![0.0, 1.0, 0.0, 3.0, 0.0]);
    assert_eq!(ids.to_vec().unwrap(), vec![0, 10, 0, 30, 0]);

    let values = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let ids = policy.to_device(&[10_u32, 20, 30, 40, 50]).unwrap();
    let (keys, reduced) = reduce_by_key(
        zip(&key_a, &key_b),
        zip(&values, &ids),
        MixedTupleEqual,
        (0.0_f32, 0_u32),
        Sum,
    )
    .unwrap();
    let (key_a, key_b) = keys;
    let (values, ids) = reduced;
    assert_eq!(key_a.to_vec().unwrap(), vec![1.0, 2.0, 2.0]);
    assert_eq!(key_b.to_vec().unwrap(), vec![10, 20, 30]);
    assert_eq!(values.to_vec().unwrap(), vec![3.0, 7.0, 5.0]);
    assert_eq!(ids.to_vec().unwrap(), vec![30, 70, 50]);
}

#[test]
fn scan_and_reduce_by_key_accept_borrowed_heterogeneous_key_and_value_soas() {
    let policy = policy();

    let key_a = policy.to_device(&[1.0_f32, 1.0, 2.0, 2.0, 2.0]).unwrap();
    let key_b = policy.to_device(&[10_u32, 10, 20, 20, 30]).unwrap();
    let values = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let ids = policy.to_device(&[10_u32, 20, 30, 40, 50]).unwrap();
    let inclusive = inclusive_scan_by_key(
        zip(&key_a, &key_b),
        zip(&values, &ids),
        MixedTupleEqual,
        Sum,
    )
    .unwrap();
    let (values, ids) = inclusive;
    assert_eq!(values.to_vec().unwrap(), vec![1.0, 3.0, 3.0, 7.0, 5.0]);
    assert_eq!(ids.to_vec().unwrap(), vec![10, 30, 30, 70, 50]);

    let key_a = policy.to_device(&[1.0_f32, 1.0, 2.0, 2.0, 2.0]).unwrap();
    let key_b = policy.to_device(&[10_u32, 10, 20, 20, 30]).unwrap();
    let values = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let ids = policy.to_device(&[10_u32, 20, 30, 40, 50]).unwrap();
    let exclusive = exclusive_scan_by_key(
        zip(&key_a, &key_b),
        zip(&values, &ids),
        MixedTupleEqual,
        (0.0_f32, 0_u32),
        Sum,
    )
    .unwrap();
    let (values, ids) = exclusive;
    assert_eq!(values.to_vec().unwrap(), vec![0.0, 1.0, 0.0, 3.0, 0.0]);
    assert_eq!(ids.to_vec().unwrap(), vec![0, 10, 0, 30, 0]);

    let key_a = policy.to_device(&[1.0_f32, 1.0, 2.0, 2.0, 2.0]).unwrap();
    let key_b = policy.to_device(&[10_u32, 10, 20, 20, 30]).unwrap();
    let values = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let ids = policy.to_device(&[10_u32, 20, 30, 40, 50]).unwrap();
    let (keys, reduced) = reduce_by_key(
        zip(&key_a, &key_b),
        zip(&values, &ids),
        MixedTupleEqual,
        (0.0_f32, 0_u32),
        Sum,
    )
    .unwrap();
    let (key_a, key_b) = keys;
    let (values, ids) = reduced;
    assert_eq!(key_a.to_vec().unwrap(), vec![1.0, 2.0, 2.0]);
    assert_eq!(key_b.to_vec().unwrap(), vec![10, 20, 30]);
    assert_eq!(values.to_vec().unwrap(), vec![3.0, 7.0, 5.0]);
    assert_eq!(ids.to_vec().unwrap(), vec![30, 70, 50]);
}

#[test]
fn ordering_and_unique_by_key_accept_borrowed_heterogeneous_key_soas() {
    let policy = policy();

    let key_a = policy.to_device(&[2.0_f32, 1.0, 2.0, 1.0]).unwrap();
    let key_b = policy.to_device(&[20_u32, 10, 10, 20]).unwrap();
    let values = policy.to_device(&[200_u32, 100, 210, 120]).unwrap();
    let (keys, values) = sort_by_key(zip(&key_a, &key_b), &values, MixedTupleLess).unwrap();
    let (key_a, key_b) = keys;
    let values = values;
    assert_eq!(key_a.to_vec().unwrap(), vec![1.0, 1.0, 2.0, 2.0]);
    assert_eq!(key_b.to_vec().unwrap(), vec![10, 20, 10, 20]);
    assert_eq!(values.to_vec().unwrap(), vec![100, 120, 210, 200]);

    let left_key_a = policy.to_device(&[1.0_f32, 2.0]).unwrap();
    let left_key_b = policy.to_device(&[10_u32, 20]).unwrap();
    let left_values = policy.to_device(&[100_u32, 200]).unwrap();
    let right_key_a = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let right_key_b = policy.to_device(&[20_u32, 10, 30]).unwrap();
    let right_values = policy.to_device(&[120_u32, 210, 300]).unwrap();
    let (keys, values) = merge_by_key(
        zip(&left_key_a, &left_key_b),
        left_values,
        zip(&right_key_a, &right_key_b),
        right_values,
        MixedTupleLess,
    )
    .unwrap();
    let (key_a, key_b) = keys;
    let values = values;
    assert_eq!(key_a.to_vec().unwrap(), vec![1.0, 1.0, 2.0, 2.0, 3.0]);
    assert_eq!(key_b.to_vec().unwrap(), vec![10, 20, 10, 20, 30]);
    assert_eq!(values.to_vec().unwrap(), vec![100, 120, 210, 200, 300]);

    let key_a = policy.to_device(&[1.0_f32, 1.0, 2.0, 2.0, 3.0]).unwrap();
    let key_b = policy.to_device(&[10_u32, 10, 20, 30, 30]).unwrap();
    let values = policy.to_device(&[100_u32, 101, 200, 230, 300]).unwrap();
    let (keys, values) = unique_by_key(zip(&key_a, &key_b), values, MixedTupleEqual).unwrap();
    let (key_a, key_b) = keys;
    let values = values;
    assert_eq!(key_a.to_vec().unwrap(), vec![1.0, 2.0, 2.0, 3.0]);
    assert_eq!(key_b.to_vec().unwrap(), vec![10, 20, 30, 30]);
    assert_eq!(values.to_vec().unwrap(), vec![100, 200, 230, 300]);
}

#[test]
fn sort_and_unique_by_key_accept_borrowed_key_and_value_soas() {
    let policy = policy();

    let key_a = policy.to_device(&[2.0_f32, 1.0, 2.0, 1.0]).unwrap();
    let key_b = policy.to_device(&[20_u32, 10, 10, 20]).unwrap();
    let values = policy.to_device(&[200_u32, 100, 210, 120]).unwrap();
    let ids = policy.to_device(&[20.0_f32, 10.0, 21.0, 12.0]).unwrap();
    let (keys, values) =
        sort_by_key(zip(&key_a, &key_b), zip(&values, &ids), MixedTupleLess).unwrap();
    let (key_a, key_b) = keys;
    let (values, ids) = values;
    assert_eq!(key_a.to_vec().unwrap(), vec![1.0, 1.0, 2.0, 2.0]);
    assert_eq!(key_b.to_vec().unwrap(), vec![10, 20, 10, 20]);
    assert_eq!(values.to_vec().unwrap(), vec![100, 120, 210, 200]);
    assert_eq!(ids.to_vec().unwrap(), vec![10.0, 12.0, 21.0, 20.0]);

    let key_a = policy.to_device(&[1.0_f32, 1.0, 2.0, 2.0, 3.0]).unwrap();
    let key_b = policy.to_device(&[10_u32, 10, 20, 30, 30]).unwrap();
    let values = policy.to_device(&[100_u32, 101, 200, 230, 300]).unwrap();
    let ids = policy
        .to_device(&[10.0_f32, 10.1, 20.0, 23.0, 30.0])
        .unwrap();
    let (keys, values) =
        unique_by_key(zip(&key_a, &key_b), zip(&values, &ids), MixedTupleEqual).unwrap();
    let (key_a, key_b) = keys;
    let (values, ids) = values;
    assert_eq!(key_a.to_vec().unwrap(), vec![1.0, 2.0, 2.0, 3.0]);
    assert_eq!(key_b.to_vec().unwrap(), vec![10, 20, 30, 30]);
    assert_eq!(values.to_vec().unwrap(), vec![100, 200, 230, 300]);
    assert_eq!(ids.to_vec().unwrap(), vec![10.0, 20.0, 23.0, 30.0]);
}

#[test]
fn sort_by_key_accepts_borrowed_heterogeneous_soa_keys() {
    let policy = policy();
    let key_a = policy.to_device(&[2.0_f32, 1.0, 2.0, 1.0]).unwrap();
    let key_b = policy.to_device(&[20_u32, 10, 10, 20]).unwrap();
    let values = policy.to_device(&[200_u32, 100, 210, 120]).unwrap();

    let (keys, values) = sort_by_key(zip(&key_a, &key_b), &values, MixedTupleLess).unwrap();
    let (key_a, key_b) = keys;
    let values = values;

    assert_eq!(key_a.to_vec().unwrap(), vec![1.0, 1.0, 2.0, 2.0]);
    assert_eq!(key_b.to_vec().unwrap(), vec![10, 20, 10, 20]);
    assert_eq!(values.to_vec().unwrap(), vec![100, 120, 210, 200]);
}

#[test]
fn merge_by_key_accepts_borrowed_heterogeneous_soa_keys() {
    let policy = policy();
    let left_key_a = policy.to_device(&[1.0_f32, 2.0]).unwrap();
    let left_key_b = policy.to_device(&[10_u32, 20]).unwrap();
    let left_values = policy.to_device(&[100_u32, 200]).unwrap();
    let right_key_a = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let right_key_b = policy.to_device(&[20_u32, 10, 30]).unwrap();
    let right_values = policy.to_device(&[120_u32, 210, 300]).unwrap();

    let (keys, values) = merge_by_key(
        zip(&left_key_a, &left_key_b),
        &left_values,
        zip(&right_key_a, &right_key_b),
        &right_values,
        MixedTupleLess,
    )
    .unwrap();
    let (key_a, key_b) = keys;
    let values = values;

    assert_eq!(key_a.to_vec().unwrap(), vec![1.0, 1.0, 2.0, 2.0, 3.0]);
    assert_eq!(key_b.to_vec().unwrap(), vec![10, 20, 10, 20, 30]);
    assert_eq!(values.to_vec().unwrap(), vec![100, 120, 210, 200, 300]);
}

#[test]
fn reduce_by_key_accepts_borrowed_heterogeneous_soa_keys() {
    let policy = policy();
    let key_a = policy.to_device(&[1.0_f32, 1.0, 2.0, 2.0, 2.0]).unwrap();
    let key_b = policy.to_device(&[10_u32, 10, 20, 20, 30]).unwrap();
    let values = policy.to_device(&[1_u32, 2, 3, 4, 5]).unwrap();

    let (keys, values) =
        reduce_by_key(zip(&key_a, &key_b), &values, MixedTupleEqual, 0_u32, Sum).unwrap();
    let (key_a, key_b) = keys;
    let values = values;

    assert_eq!(key_a.to_vec().unwrap(), vec![1.0, 2.0, 2.0]);
    assert_eq!(key_b.to_vec().unwrap(), vec![10, 20, 30]);
    assert_eq!(values.to_vec().unwrap(), vec![3, 7, 5]);
}

#[test]
fn reduce_by_tuple_key_uses_supplied_key_equality() {
    let policy = policy();
    let key_a = policy.to_device(&[1.0_f32, 1.0, 2.0, 2.0]).unwrap();
    let key_b = policy.to_device(&[10_u32, 20, 10, 20]).unwrap();
    let values = policy.to_device(&[1_u32, 2, 3, 4]).unwrap();

    let (keys, values) = reduce_by_key(
        zip(&key_a, &key_b),
        &values,
        MixedTupleFirstEqual,
        0_u32,
        Sum,
    )
    .unwrap();
    let (key_a, key_b) = keys;

    assert_eq!(key_a.to_vec().unwrap(), vec![1.0, 2.0]);
    assert_eq!(key_b.to_vec().unwrap(), vec![20, 20]);
    assert_eq!(values.to_vec().unwrap(), vec![3, 7]);
}

#[test]
fn reduce_by_tuple_key_with_tuple_values_uses_supplied_key_equality() {
    let policy = policy();
    let key_a = policy.to_device(&[1.0_f32, 1.0, 2.0, 2.0]).unwrap();
    let key_b = policy.to_device(&[10_u32, 20, 10, 20]).unwrap();
    let value_a = policy.to_device(&[1_u32, 2, 3, 4]).unwrap();
    let value_b = policy.to_device(&[10.0_f32, 20.0, 30.0, 40.0]).unwrap();

    let (keys, values) = reduce_by_key(
        zip(&key_a, &key_b),
        zip(&value_a, &value_b),
        MixedTupleFirstEqual,
        (0_u32, 0.0_f32),
        Sum,
    )
    .unwrap();
    let (key_a, key_b) = keys;
    let (value_a, value_b) = values;

    assert_eq!(key_a.to_vec().unwrap(), vec![1.0, 2.0]);
    assert_eq!(key_b.to_vec().unwrap(), vec![20, 20]);
    assert_eq!(value_a.to_vec().unwrap(), vec![3, 7]);
    assert_eq!(value_b.to_vec().unwrap(), vec![30.0, 70.0]);
}

#[test]
fn reduce_by_three_tuple_key_with_tuple_values_uses_supplied_key_equality() {
    let policy = policy();
    let key_a = policy.to_device(&[1.0_f32, 1.0, 2.0, 2.0]).unwrap();
    let key_b = policy.to_device(&[10_u32, 20, 10, 20]).unwrap();
    let key_c = policy.to_device(&[100.0_f32, 200.0, 100.0, 200.0]).unwrap();
    let value_a = policy.to_device(&[1_u32, 2, 3, 4]).unwrap();
    let value_b = policy.to_device(&[10.0_f32, 20.0, 30.0, 40.0]).unwrap();

    let (keys, values) = reduce_by_key(
        zip3(&key_a, &key_b, &key_c),
        zip(&value_a, &value_b),
        MixedTuple3FirstEqual,
        (0_u32, 0.0_f32),
        Sum,
    )
    .unwrap();
    let (key_a, key_b, key_c) = keys;
    let (value_a, value_b) = values;

    assert_eq!(key_a.to_vec().unwrap(), vec![1.0, 2.0]);
    assert_eq!(key_b.to_vec().unwrap(), vec![20, 20]);
    assert_eq!(key_c.to_vec().unwrap(), vec![200.0, 200.0]);
    assert_eq!(value_a.to_vec().unwrap(), vec![3, 7]);
    assert_eq!(value_b.to_vec().unwrap(), vec![30.0, 70.0]);
}

#[test]
fn scan_by_key_accepts_borrowed_heterogeneous_soa_keys() {
    let policy = policy();
    let key_a = policy.to_device(&[1.0_f32, 1.0, 2.0, 2.0, 2.0]).unwrap();
    let key_b = policy.to_device(&[10_u32, 10, 20, 20, 30]).unwrap();
    let values = policy.to_device(&[1_u32, 2, 3, 4, 5]).unwrap();

    let inclusive =
        inclusive_scan_by_key(zip(&key_a, &key_b), &values, MixedTupleEqual, Sum).unwrap();
    let inclusive = inclusive;
    assert_eq!(inclusive.to_vec().unwrap(), vec![1, 3, 3, 7, 5]);

    let exclusive =
        exclusive_scan_by_key(zip(&key_a, &key_b), &values, MixedTupleEqual, 0_u32, Sum).unwrap();
    let exclusive = exclusive;
    assert_eq!(exclusive.to_vec().unwrap(), vec![0, 1, 0, 3, 0]);
}

#[test]
fn unique_by_key_accepts_borrowed_heterogeneous_soa_keys() {
    let policy = policy();
    let key_a = policy.to_device(&[1.0_f32, 1.0, 2.0, 2.0, 2.0]).unwrap();
    let key_b = policy.to_device(&[10_u32, 10, 20, 20, 30]).unwrap();
    let values = policy.to_device(&[100_u32, 101, 200, 201, 300]).unwrap();

    let (keys, values) = unique_by_key(zip(&key_a, &key_b), values, MixedTupleEqual).unwrap();
    let (key_a, key_b) = keys;
    let values = values;

    assert_eq!(key_a.to_vec().unwrap(), vec![1.0, 2.0, 2.0]);
    assert_eq!(key_b.to_vec().unwrap(), vec![10, 20, 30]);
    assert_eq!(values.to_vec().unwrap(), vec![100, 200, 300]);
}

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

#[test]
fn sort_by_tuple_key_reports_value_length_mismatch() {
    let policy = policy();
    let key_a = policy.to_device(&[2.0_f32, 1.0, 3.0]).unwrap();
    let key_b = policy.to_device(&[20_u32, 10, 30]).unwrap();
    let values = policy.to_device(&[200_u32, 100]).unwrap();

    let err = sort_by_key(zip(&key_a, &key_b), &values, MixedTupleLess).unwrap_err();

    assert_eq!(
        err,
        massively::Error::LengthMismatch {
            input: 2,
            output: 3
        }
    );
}

#[test]
fn merge_by_tuple_key_reports_left_value_length_mismatch() {
    let policy = policy();
    let left_key_a = policy.to_device(&[1.0_f32, 2.0]).unwrap();
    let left_key_b = policy.to_device(&[10_u32, 20]).unwrap();
    let left_values = policy.to_device(&[100_u32]).unwrap();
    let right_key_a = policy.to_device(&[3.0_f32]).unwrap();
    let right_key_b = policy.to_device(&[30_u32]).unwrap();
    let right_values = policy.to_device(&[300_u32]).unwrap();

    let err = merge_by_key(
        zip(&left_key_a, &left_key_b),
        &left_values,
        zip(&right_key_a, &right_key_b),
        &right_values,
        MixedTupleLess,
    )
    .unwrap_err();

    assert_eq!(
        err,
        massively::Error::LengthMismatch {
            input: 1,
            output: 2
        }
    );
}

#[test]
fn merge_by_tuple_key_reports_right_value_length_mismatch() {
    let policy = policy();
    let left_key_a = policy.to_device(&[1.0_f32]).unwrap();
    let left_key_b = policy.to_device(&[10_u32]).unwrap();
    let left_values = policy.to_device(&[100_u32]).unwrap();
    let right_key_a = policy.to_device(&[2.0_f32, 3.0]).unwrap();
    let right_key_b = policy.to_device(&[20_u32, 30]).unwrap();
    let right_values = policy.to_device(&[200_u32]).unwrap();

    let err = merge_by_key(
        zip(&left_key_a, &left_key_b),
        &left_values,
        zip(&right_key_a, &right_key_b),
        &right_values,
        MixedTupleLess,
    )
    .unwrap_err();

    assert_eq!(
        err,
        massively::Error::LengthMismatch {
            input: 1,
            output: 2
        }
    );
}

#[test]
fn merge_by_tuple_key_reports_left_tuple_value_length_mismatch() {
    let policy = policy();
    let left_key_a = policy.to_device(&[1.0_f32, 2.0]).unwrap();
    let left_key_b = policy.to_device(&[10_u32, 20]).unwrap();
    let left_value_a = policy.to_device(&[100_u32]).unwrap();
    let left_value_b = policy.to_device(&[1000.0_f32]).unwrap();
    let right_key_a = policy.to_device(&[3.0_f32]).unwrap();
    let right_key_b = policy.to_device(&[30_u32]).unwrap();
    let right_value_a = policy.to_device(&[300_u32]).unwrap();
    let right_value_b = policy.to_device(&[3000.0_f32]).unwrap();

    let err = merge_by_key(
        zip(&left_key_a, &left_key_b),
        zip(&left_value_a, &left_value_b),
        zip(&right_key_a, &right_key_b),
        zip(&right_value_a, &right_value_b),
        MixedTupleLess,
    )
    .unwrap_err();

    assert_eq!(
        err,
        massively::Error::LengthMismatch {
            input: 1,
            output: 2
        }
    );
}

#[test]
fn merge_by_tuple_key_reports_right_tuple_value_length_mismatch() {
    let policy = policy();
    let left_key_a = policy.to_device(&[1.0_f32]).unwrap();
    let left_key_b = policy.to_device(&[10_u32]).unwrap();
    let left_value_a = policy.to_device(&[100_u32]).unwrap();
    let left_value_b = policy.to_device(&[1000.0_f32]).unwrap();
    let right_key_a = policy.to_device(&[2.0_f32, 3.0]).unwrap();
    let right_key_b = policy.to_device(&[20_u32, 30]).unwrap();
    let right_value_a = policy.to_device(&[200_u32]).unwrap();
    let right_value_b = policy.to_device(&[2000.0_f32]).unwrap();

    let err = merge_by_key(
        zip(&left_key_a, &left_key_b),
        zip(&left_value_a, &left_value_b),
        zip(&right_key_a, &right_key_b),
        zip(&right_value_a, &right_value_b),
        MixedTupleLess,
    )
    .unwrap_err();

    assert_eq!(
        err,
        massively::Error::LengthMismatch {
            input: 1,
            output: 2
        }
    );
}

#[test]
fn by_key_scalar_values_accept_borrowed_heterogeneous_soa12_keys() {
    let policy = policy();

    let sort_a = policy.to_device(&[3.0_f32, 1.0, 2.0, 0.0]).unwrap();
    let sort_b = policy.to_device(&[30_u32, 10, 20, 9]).unwrap();
    let sort_c = policy.to_device(&[3.1_f32, 1.1, 2.1, 0.1]).unwrap();
    let sort_d = policy.to_device(&[31_u32, 11, 21, 8]).unwrap();
    let sort_e = policy.to_device(&[3.2_f32, 1.2, 2.2, 0.2]).unwrap();
    let sort_f = policy.to_device(&[32_u32, 12, 22, 7]).unwrap();
    let sort_g = policy.to_device(&[3.3_f32, 1.3, 2.3, 0.3]).unwrap();
    let sort_h = policy.to_device(&[33_u32, 13, 23, 6]).unwrap();
    let sort_i = policy.to_device(&[3.4_f32, 1.4, 2.4, 0.4]).unwrap();
    let sort_j = policy.to_device(&[34_u32, 14, 24, 5]).unwrap();
    let sort_k = policy.to_device(&[3.0_f32, 1.0, 2.0, 0.0]).unwrap();
    let sort_l = policy.to_device(&[30_u32, 10, 20, 10]).unwrap();
    let sort_values = policy.to_device(&[300_u32, 100, 200, 90]).unwrap();

    let (keys, values) = sort_by_key(
        zip12(
            &sort_a, &sort_b, &sort_c, &sort_d, &sort_e, &sort_f, &sort_g, &sort_h, &sort_i,
            &sort_j, &sort_k, &sort_l,
        ),
        &sort_values,
        Tuple12MixedTailLess,
    )
    .unwrap();
    let (_, _, _, _, _, _, _, _, _, _, key_k, key_l) = keys;
    let values = values;
    assert_eq!(key_k.to_vec().unwrap(), vec![0.0, 1.0, 2.0, 3.0]);
    assert_eq!(key_l.to_vec().unwrap(), vec![10, 10, 20, 30]);
    assert_eq!(values.to_vec().unwrap(), vec![90, 100, 200, 300]);

    let left_a = policy.to_device(&[1.0_f32, 3.0]).unwrap();
    let left_b = policy.to_device(&[10_u32, 30]).unwrap();
    let left_c = policy.to_device(&[1.1_f32, 3.1]).unwrap();
    let left_d = policy.to_device(&[11_u32, 31]).unwrap();
    let left_e = policy.to_device(&[1.2_f32, 3.2]).unwrap();
    let left_f = policy.to_device(&[12_u32, 32]).unwrap();
    let left_g = policy.to_device(&[1.3_f32, 3.3]).unwrap();
    let left_h = policy.to_device(&[13_u32, 33]).unwrap();
    let left_i = policy.to_device(&[1.4_f32, 3.4]).unwrap();
    let left_j = policy.to_device(&[14_u32, 34]).unwrap();
    let left_k = policy.to_device(&[1.0_f32, 3.0]).unwrap();
    let left_l = policy.to_device(&[10_u32, 30]).unwrap();
    let left_values = policy.to_device(&[100_u32, 300]).unwrap();
    let right_a = policy.to_device(&[0.0_f32, 2.0]).unwrap();
    let right_b = policy.to_device(&[9_u32, 20]).unwrap();
    let right_c = policy.to_device(&[0.1_f32, 2.1]).unwrap();
    let right_d = policy.to_device(&[8_u32, 21]).unwrap();
    let right_e = policy.to_device(&[0.2_f32, 2.2]).unwrap();
    let right_f = policy.to_device(&[7_u32, 22]).unwrap();
    let right_g = policy.to_device(&[0.3_f32, 2.3]).unwrap();
    let right_h = policy.to_device(&[6_u32, 23]).unwrap();
    let right_i = policy.to_device(&[0.4_f32, 2.4]).unwrap();
    let right_j = policy.to_device(&[5_u32, 24]).unwrap();
    let right_k = policy.to_device(&[0.0_f32, 2.0]).unwrap();
    let right_l = policy.to_device(&[10_u32, 20]).unwrap();
    let right_values = policy.to_device(&[90_u32, 200]).unwrap();
    let (keys, values) = merge_by_key(
        zip12(
            &left_a, &left_b, &left_c, &left_d, &left_e, &left_f, &left_g, &left_h, &left_i,
            &left_j, &left_k, &left_l,
        ),
        &left_values,
        zip12(
            &right_a, &right_b, &right_c, &right_d, &right_e, &right_f, &right_g, &right_h,
            &right_i, &right_j, &right_k, &right_l,
        ),
        &right_values,
        Tuple12MixedTailLess,
    )
    .unwrap();
    let (_, _, _, _, _, _, _, _, _, _, key_k, key_l) = keys;
    let values = values;
    assert_eq!(key_k.to_vec().unwrap(), vec![0.0, 1.0, 2.0, 3.0]);
    assert_eq!(key_l.to_vec().unwrap(), vec![10, 10, 20, 30]);
    assert_eq!(values.to_vec().unwrap(), vec![90, 100, 200, 300]);

    let key_a = policy.to_device(&[1.0_f32, 1.0, 2.0, 2.0, 3.0]).unwrap();
    let key_b = policy.to_device(&[10_u32, 10, 20, 20, 30]).unwrap();
    let key_c = policy.to_device(&[1.1_f32, 1.1, 2.1, 2.1, 3.1]).unwrap();
    let key_d = policy.to_device(&[11_u32, 11, 21, 21, 31]).unwrap();
    let key_e = policy.to_device(&[1.2_f32, 1.2, 2.2, 2.2, 3.2]).unwrap();
    let key_f = policy.to_device(&[12_u32, 12, 22, 22, 32]).unwrap();
    let key_g = policy.to_device(&[1.3_f32, 1.3, 2.3, 2.3, 3.3]).unwrap();
    let key_h = policy.to_device(&[13_u32, 13, 23, 23, 33]).unwrap();
    let key_i = policy.to_device(&[1.4_f32, 1.4, 2.4, 2.4, 3.4]).unwrap();
    let key_j = policy.to_device(&[14_u32, 14, 24, 24, 34]).unwrap();
    let key_k = policy.to_device(&[1.5_f32, 1.5, 2.5, 2.5, 3.5]).unwrap();
    let key_l = policy.to_device(&[15_u32, 15, 25, 25, 35]).unwrap();
    let values = policy.to_device(&[1_u32, 2, 3, 4, 5]).unwrap();
    let keys = zip12(
        &key_a, &key_b, &key_c, &key_d, &key_e, &key_f, &key_g, &key_h, &key_i, &key_j, &key_k,
        &key_l,
    );

    let inclusive = inclusive_scan_by_key(keys, &values, Tuple12MixedEqual, Sum).unwrap();
    assert_eq!(inclusive.to_vec().unwrap(), vec![1, 3, 3, 7, 5]);

    let keys = zip12(
        &key_a, &key_b, &key_c, &key_d, &key_e, &key_f, &key_g, &key_h, &key_i, &key_j, &key_k,
        &key_l,
    );
    let exclusive = exclusive_scan_by_key(keys, &values, Tuple12MixedEqual, 0_u32, Sum).unwrap();
    assert_eq!(exclusive.to_vec().unwrap(), vec![0, 1, 0, 3, 0]);

    let keys = zip12(
        &key_a, &key_b, &key_c, &key_d, &key_e, &key_f, &key_g, &key_h, &key_i, &key_j, &key_k,
        &key_l,
    );
    let (keys, values) = reduce_by_key(keys, &values, Tuple12MixedEqual, 0_u32, Sum).unwrap();
    let (key_a_out, key_b_out, _, _, _, _, _, _, _, _, key_k_out, key_l_out) = keys;
    assert_eq!(key_a_out.to_vec().unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(key_b_out.to_vec().unwrap(), vec![10, 20, 30]);
    assert_eq!(key_k_out.to_vec().unwrap(), vec![1.5, 2.5, 3.5]);
    assert_eq!(key_l_out.to_vec().unwrap(), vec![15, 25, 35]);
    assert_eq!(values.to_vec().unwrap(), vec![3, 7, 5]);

    let unique_values = policy.to_device(&[100_u32, 101, 200, 201, 300]).unwrap();
    let keys = zip12(
        &key_a, &key_b, &key_c, &key_d, &key_e, &key_f, &key_g, &key_h, &key_i, &key_j, &key_k,
        &key_l,
    );
    let (keys, values) = unique_by_key(keys, unique_values, Tuple12MixedEqual).unwrap();
    let (key_a, key_b, _, _, _, _, _, _, _, _, key_k, key_l) = keys;
    assert_eq!(key_a.to_vec().unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(key_b.to_vec().unwrap(), vec![10, 20, 30]);
    assert_eq!(key_k.to_vec().unwrap(), vec![1.5, 2.5, 3.5]);
    assert_eq!(key_l.to_vec().unwrap(), vec![15, 25, 35]);
    assert_eq!(values.to_vec().unwrap(), vec![100, 200, 300]);
}

#[test]
fn by_key_soa2_values_accept_declared_heterogeneous_key_layouts() {
    let policy = policy();

    let sort_a = policy.to_device(&[3.0_f32, 1.0, 2.0, 0.0]).unwrap();
    let sort_b = policy.to_device(&[30_u32, 10, 20, 9]).unwrap();
    let sort_c = policy.to_device(&[3.1_f32, 1.1, 2.1, 0.1]).unwrap();
    let sort_d = policy.to_device(&[31_u32, 11, 21, 8]).unwrap();
    let sort_e = policy.to_device(&[3.2_f32, 1.2, 2.2, 0.2]).unwrap();
    let sort_f = policy.to_device(&[32_u32, 12, 22, 7]).unwrap();
    let sort_g = policy.to_device(&[3.3_f32, 1.3, 2.3, 0.3]).unwrap();
    let sort_h = policy.to_device(&[33_u32, 13, 23, 6]).unwrap();
    let sort_i = policy.to_device(&[3.4_f32, 1.4, 2.4, 0.4]).unwrap();
    let sort_j = policy.to_device(&[34_u32, 14, 24, 5]).unwrap();
    let sort_k = policy.to_device(&[3.0_f32, 1.0, 2.0, 0.0]).unwrap();
    let sort_l = policy.to_device(&[30_u32, 10, 20, 10]).unwrap();
    let sort_value_a = policy.to_device(&[300_u32, 100, 200, 90]).unwrap();
    let sort_value_b = policy.to_device(&[30.0_f32, 10.0, 20.0, 9.0]).unwrap();

    let (keys, values) = sort_by_key(
        zip12(
            &sort_a, &sort_b, &sort_c, &sort_d, &sort_e, &sort_f, &sort_g, &sort_h, &sort_i,
            &sort_j, &sort_k, &sort_l,
        ),
        zip(&sort_value_a, &sort_value_b),
        Tuple12MixedTailLess,
    )
    .unwrap();
    let (_, _, _, _, _, _, _, _, _, _, key_k_out, key_l_out) = keys;
    let (value_a_out, value_b_out) = values;
    assert_eq!(key_k_out.to_vec().unwrap(), vec![0.0, 1.0, 2.0, 3.0]);
    assert_eq!(key_l_out.to_vec().unwrap(), vec![10, 10, 20, 30]);
    assert_eq!(value_a_out.to_vec().unwrap(), vec![90, 100, 200, 300]);
    assert_eq!(value_b_out.to_vec().unwrap(), vec![9.0, 10.0, 20.0, 30.0]);

    let left_a = policy.to_device(&[1.0_f32, 3.0]).unwrap();
    let left_b = policy.to_device(&[10_u32, 30]).unwrap();
    let left_c = policy.to_device(&[1.1_f32, 3.1]).unwrap();
    let left_d = policy.to_device(&[11_u32, 31]).unwrap();
    let left_e = policy.to_device(&[1.2_f32, 3.2]).unwrap();
    let left_f = policy.to_device(&[12_u32, 32]).unwrap();
    let left_g = policy.to_device(&[1.3_f32, 3.3]).unwrap();
    let left_h = policy.to_device(&[13_u32, 33]).unwrap();
    let left_i = policy.to_device(&[1.4_f32, 3.4]).unwrap();
    let left_j = policy.to_device(&[14_u32, 34]).unwrap();
    let left_k = policy.to_device(&[1.0_f32, 3.0]).unwrap();
    let left_l = policy.to_device(&[10_u32, 30]).unwrap();
    let left_value_a = policy.to_device(&[100_u32, 300]).unwrap();
    let left_value_b = policy.to_device(&[10.0_f32, 30.0]).unwrap();
    let right_a = policy.to_device(&[0.0_f32, 2.0]).unwrap();
    let right_b = policy.to_device(&[9_u32, 20]).unwrap();
    let right_c = policy.to_device(&[0.1_f32, 2.1]).unwrap();
    let right_d = policy.to_device(&[8_u32, 21]).unwrap();
    let right_e = policy.to_device(&[0.2_f32, 2.2]).unwrap();
    let right_f = policy.to_device(&[7_u32, 22]).unwrap();
    let right_g = policy.to_device(&[0.3_f32, 2.3]).unwrap();
    let right_h = policy.to_device(&[6_u32, 23]).unwrap();
    let right_i = policy.to_device(&[0.4_f32, 2.4]).unwrap();
    let right_j = policy.to_device(&[5_u32, 24]).unwrap();
    let right_k = policy.to_device(&[0.0_f32, 2.0]).unwrap();
    let right_l = policy.to_device(&[10_u32, 20]).unwrap();
    let right_value_a = policy.to_device(&[90_u32, 200]).unwrap();
    let right_value_b = policy.to_device(&[9.0_f32, 20.0]).unwrap();
    let (keys, values) = merge_by_key(
        zip12(
            &left_a, &left_b, &left_c, &left_d, &left_e, &left_f, &left_g, &left_h, &left_i,
            &left_j, &left_k, &left_l,
        ),
        zip(&left_value_a, &left_value_b),
        zip12(
            &right_a, &right_b, &right_c, &right_d, &right_e, &right_f, &right_g, &right_h,
            &right_i, &right_j, &right_k, &right_l,
        ),
        zip(&right_value_a, &right_value_b),
        Tuple12MixedTailLess,
    )
    .unwrap();
    let (_, _, _, _, _, _, _, _, _, _, key_k_out, key_l_out) = keys;
    let (value_a_out, value_b_out) = values;
    assert_eq!(key_k_out.to_vec().unwrap(), vec![0.0, 1.0, 2.0, 3.0]);
    assert_eq!(key_l_out.to_vec().unwrap(), vec![10, 10, 20, 30]);
    assert_eq!(value_a_out.to_vec().unwrap(), vec![90, 100, 200, 300]);
    assert_eq!(value_b_out.to_vec().unwrap(), vec![9.0, 10.0, 20.0, 30.0]);

    let key_a = policy.to_device(&[1.0_f32, 1.0, 2.0, 2.0, 3.0]).unwrap();
    let key_b = policy.to_device(&[10_u32, 10, 20, 20, 30]).unwrap();
    let key_c = policy.to_device(&[1.1_f32, 1.1, 2.1, 2.1, 3.1]).unwrap();
    let key_d = policy.to_device(&[11_u32, 11, 21, 21, 31]).unwrap();
    let key_e = policy.to_device(&[1.2_f32, 1.2, 2.2, 2.2, 3.2]).unwrap();
    let key_f = policy.to_device(&[12_u32, 12, 22, 22, 32]).unwrap();
    let key_g = policy.to_device(&[1.3_f32, 1.3, 2.3, 2.3, 3.3]).unwrap();
    let key_h = policy.to_device(&[13_u32, 13, 23, 23, 33]).unwrap();
    let key_i = policy.to_device(&[1.4_f32, 1.4, 2.4, 2.4, 3.4]).unwrap();
    let key_j = policy.to_device(&[14_u32, 14, 24, 24, 34]).unwrap();
    let key_k = policy.to_device(&[1.5_f32, 1.5, 2.5, 2.5, 3.5]).unwrap();
    let key_l = policy.to_device(&[15_u32, 15, 25, 25, 35]).unwrap();
    let value_a = policy.to_device(&[1_u32, 2, 3, 4, 5]).unwrap();
    let value_b = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0, 5.0]).unwrap();

    let keys = zip12(
        &key_a, &key_b, &key_c, &key_d, &key_e, &key_f, &key_g, &key_h, &key_i, &key_j, &key_k,
        &key_l,
    );
    let inclusive =
        inclusive_scan_by_key(keys, zip(&value_a, &value_b), Tuple12MixedEqual, Sum).unwrap();
    let (value_a_out, value_b_out) = inclusive;
    assert_eq!(value_a_out.to_vec().unwrap(), vec![1, 3, 3, 7, 5]);
    assert_eq!(value_b_out.to_vec().unwrap(), vec![1.0, 3.0, 3.0, 7.0, 5.0]);

    let keys = zip12(
        &key_a, &key_b, &key_c, &key_d, &key_e, &key_f, &key_g, &key_h, &key_i, &key_j, &key_k,
        &key_l,
    );
    let exclusive = exclusive_scan_by_key(
        keys,
        zip(&value_a, &value_b),
        Tuple12MixedEqual,
        (0_u32, 0.0_f32),
        Sum,
    )
    .unwrap();
    let (value_a_out, value_b_out) = exclusive;
    assert_eq!(value_a_out.to_vec().unwrap(), vec![0, 1, 0, 3, 0]);
    assert_eq!(value_b_out.to_vec().unwrap(), vec![0.0, 1.0, 0.0, 3.0, 0.0]);

    let keys = zip12(
        &key_a, &key_b, &key_c, &key_d, &key_e, &key_f, &key_g, &key_h, &key_i, &key_j, &key_k,
        &key_l,
    );
    let (keys, values) = reduce_by_key(
        keys,
        zip(&value_a, &value_b),
        Tuple12MixedEqual,
        (0_u32, 0.0_f32),
        Sum,
    )
    .unwrap();
    let (key_a_out, key_b_out, _, _, _, _, _, _, _, _, key_k_out, key_l_out) = keys;
    let (value_a_out, value_b_out) = values;
    assert_eq!(key_a_out.to_vec().unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(key_b_out.to_vec().unwrap(), vec![10, 20, 30]);
    assert_eq!(key_k_out.to_vec().unwrap(), vec![1.5, 2.5, 3.5]);
    assert_eq!(key_l_out.to_vec().unwrap(), vec![15, 25, 35]);
    assert_eq!(value_a_out.to_vec().unwrap(), vec![3, 7, 5]);
    assert_eq!(value_b_out.to_vec().unwrap(), vec![3.0, 7.0, 5.0]);

    let values_a = policy.to_device(&[100_u32, 101, 200, 201, 300]).unwrap();
    let values_b = policy
        .to_device(&[10.0_f32, 10.1, 20.0, 20.1, 30.0])
        .unwrap();
    let keys = zip12(
        &key_a, &key_b, &key_c, &key_d, &key_e, &key_f, &key_g, &key_h, &key_i, &key_j, &key_k,
        &key_l,
    );
    let (keys, values) = unique_by_key(keys, zip(&values_a, &values_b), Tuple12MixedEqual).unwrap();
    let (key_a_out, key_b_out, _, _, _, _, _, _, _, _, key_k_out, key_l_out) = keys;
    let (value_a_out, value_b_out) = values;
    assert_eq!(key_a_out.to_vec().unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(key_b_out.to_vec().unwrap(), vec![10, 20, 30]);
    assert_eq!(key_k_out.to_vec().unwrap(), vec![1.5, 2.5, 3.5]);
    assert_eq!(key_l_out.to_vec().unwrap(), vec![15, 25, 35]);
    assert_eq!(value_a_out.to_vec().unwrap(), vec![100, 200, 300]);
    assert_eq!(value_b_out.to_vec().unwrap(), vec![10.0, 20.0, 30.0]);
}

#[test]
fn by_key_soa3_values_accept_borrowed_heterogeneous_soa12_keys() {
    let policy = policy();
    let key_a = policy.to_device(&[1.0_f32, 1.0, 2.0, 2.0, 3.0]).unwrap();
    let key_b = policy.to_device(&[10_u32, 10, 20, 20, 30]).unwrap();
    let key_c = policy.to_device(&[1.1_f32, 1.1, 2.1, 2.1, 3.1]).unwrap();
    let key_d = policy.to_device(&[11_u32, 11, 21, 21, 31]).unwrap();
    let key_e = policy.to_device(&[1.2_f32, 1.2, 2.2, 2.2, 3.2]).unwrap();
    let key_f = policy.to_device(&[12_u32, 12, 22, 22, 32]).unwrap();
    let key_g = policy.to_device(&[1.3_f32, 1.3, 2.3, 2.3, 3.3]).unwrap();
    let key_h = policy.to_device(&[13_u32, 13, 23, 23, 33]).unwrap();
    let key_i = policy.to_device(&[1.4_f32, 1.4, 2.4, 2.4, 3.4]).unwrap();
    let key_j = policy.to_device(&[14_u32, 14, 24, 24, 34]).unwrap();
    let key_k = policy.to_device(&[1.5_f32, 1.5, 2.5, 2.5, 3.5]).unwrap();
    let key_l = policy.to_device(&[15_u32, 15, 25, 25, 35]).unwrap();
    let value_a = policy.to_device(&[1_u32, 2, 3, 4, 5]).unwrap();
    let value_b = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let value_c = policy.to_device(&[10_u32, 20, 30, 40, 50]).unwrap();

    let keys = zip12(
        &key_a, &key_b, &key_c, &key_d, &key_e, &key_f, &key_g, &key_h, &key_i, &key_j, &key_k,
        &key_l,
    );
    let inclusive = inclusive_scan_by_key(
        keys,
        zip3(&value_a, &value_b, &value_c),
        Tuple12MixedEqual,
        Sum,
    )
    .unwrap();
    let (value_a_out, value_b_out, value_c_out) = inclusive;
    assert_eq!(value_a_out.to_vec().unwrap(), vec![1, 3, 3, 7, 5]);
    assert_eq!(value_b_out.to_vec().unwrap(), vec![1.0, 3.0, 3.0, 7.0, 5.0]);
    assert_eq!(value_c_out.to_vec().unwrap(), vec![10, 30, 30, 70, 50]);

    let keys = zip12(
        &key_a, &key_b, &key_c, &key_d, &key_e, &key_f, &key_g, &key_h, &key_i, &key_j, &key_k,
        &key_l,
    );
    let (keys, values) = reduce_by_key(
        keys,
        zip3(&value_a, &value_b, &value_c),
        Tuple12MixedEqual,
        (0_u32, 0.0_f32, 0_u32),
        Sum,
    )
    .unwrap();
    let (key_a_out, key_b_out, _, _, _, _, _, _, _, _, key_k_out, key_l_out) = keys;
    let (value_a_out, value_b_out, value_c_out) = values;
    assert_eq!(key_a_out.to_vec().unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(key_b_out.to_vec().unwrap(), vec![10, 20, 30]);
    assert_eq!(key_k_out.to_vec().unwrap(), vec![1.5, 2.5, 3.5]);
    assert_eq!(key_l_out.to_vec().unwrap(), vec![15, 25, 35]);
    assert_eq!(value_a_out.to_vec().unwrap(), vec![3, 7, 5]);
    assert_eq!(value_b_out.to_vec().unwrap(), vec![3.0, 7.0, 5.0]);
    assert_eq!(value_c_out.to_vec().unwrap(), vec![30, 70, 50]);
}

#[test]
fn by_key_algorithms_accept_heterogeneous_soa2_keys_and_soa2_values() {
    let policy = policy();
    let key_a = policy.to_device(&[2.0_f32, 1.0, 2.0, 1.0]).unwrap();
    let key_b = policy.to_device(&[20_u32, 10, 10, 20]).unwrap();
    let value_a = policy.to_device(&[20_u32, 10, 21, 12]).unwrap();
    let value_b = policy.to_device(&[200.0_f32, 100.0, 210.0, 120.0]).unwrap();

    let (keys, values) =
        sort_by_key(zip(&key_a, &key_b), zip(&value_a, &value_b), MixedTupleLess).unwrap();
    let (key_a, key_b) = keys;
    let (value_a, value_b) = values;
    assert_eq!(key_a.to_vec().unwrap(), vec![1.0, 1.0, 2.0, 2.0]);
    assert_eq!(key_b.to_vec().unwrap(), vec![10, 20, 10, 20]);
    assert_eq!(value_a.to_vec().unwrap(), vec![10, 12, 21, 20]);
    assert_eq!(
        value_b.to_vec().unwrap(),
        vec![100.0_f32, 120.0, 210.0, 200.0]
    );

    let left_key_a = policy.to_device(&[1.0_f32, 2.0]).unwrap();
    let left_key_b = policy.to_device(&[10_u32, 20]).unwrap();
    let left_value_a = policy.to_device(&[10_u32, 20]).unwrap();
    let left_value_b = policy.to_device(&[100.0_f32, 200.0]).unwrap();
    let right_key_a = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let right_key_b = policy.to_device(&[20_u32, 10, 30]).unwrap();
    let right_value_a = policy.to_device(&[12_u32, 21, 30]).unwrap();
    let right_value_b = policy.to_device(&[120.0_f32, 210.0, 300.0]).unwrap();

    let (keys, values) = merge_by_key(
        zip(&left_key_a, &left_key_b),
        zip(&left_value_a, &left_value_b),
        zip(&right_key_a, &right_key_b),
        zip(&right_value_a, &right_value_b),
        MixedTupleLess,
    )
    .unwrap();
    let (key_a, key_b) = keys;
    let (value_a, value_b) = values;
    assert_eq!(key_a.to_vec().unwrap(), vec![1.0, 1.0, 2.0, 2.0, 3.0]);
    assert_eq!(key_b.to_vec().unwrap(), vec![10, 20, 10, 20, 30]);
    assert_eq!(value_a.to_vec().unwrap(), vec![10, 12, 21, 20, 30]);
    assert_eq!(
        value_b.to_vec().unwrap(),
        vec![100.0_f32, 120.0, 210.0, 200.0, 300.0]
    );

    let key_a = policy.to_device(&[1.0_f32, 1.0, 2.0, 2.0, 2.0]).unwrap();
    let key_b = policy.to_device(&[10_u32, 10, 20, 20, 30]).unwrap();
    let value_a = policy.to_device(&[1_u32, 2, 3, 4, 5]).unwrap();
    let value_b = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0, 5.0]).unwrap();

    let (keys, values) = reduce_by_key(
        zip(&key_a, &key_b),
        zip(&value_a, &value_b),
        MixedTupleEqual,
        (0_u32, 0.0_f32),
        Sum,
    )
    .unwrap();
    let (key_a_out, key_b_out) = keys;
    let (value_a_out, value_b_out) = values;
    assert_eq!(key_a_out.to_vec().unwrap(), vec![1.0, 2.0, 2.0]);
    assert_eq!(key_b_out.to_vec().unwrap(), vec![10, 20, 30]);
    assert_eq!(value_a_out.to_vec().unwrap(), vec![3, 7, 5]);
    assert_eq!(value_b_out.to_vec().unwrap(), vec![3.0, 7.0, 5.0]);

    let inclusive = inclusive_scan_by_key(
        zip(&key_a, &key_b),
        zip(&value_a, &value_b),
        MixedTupleEqual,
        Sum,
    )
    .unwrap();
    let (value_a_out, value_b_out) = inclusive;
    assert_eq!(value_a_out.to_vec().unwrap(), vec![1, 3, 3, 7, 5]);
    assert_eq!(value_b_out.to_vec().unwrap(), vec![1.0, 3.0, 3.0, 7.0, 5.0]);

    let exclusive = exclusive_scan_by_key(
        zip(&key_a, &key_b),
        zip(&value_a, &value_b),
        MixedTupleEqual,
        (0_u32, 0.0_f32),
        Sum,
    )
    .unwrap();
    let (value_a_out, value_b_out) = exclusive;
    assert_eq!(value_a_out.to_vec().unwrap(), vec![0, 1, 0, 3, 0]);
    assert_eq!(value_b_out.to_vec().unwrap(), vec![0.0, 1.0, 0.0, 3.0, 0.0]);

    let values_a = policy.to_device(&[100_u32, 101, 200, 201, 300]).unwrap();
    let values_b = policy
        .to_device(&[10.0_f32, 10.1, 20.0, 20.1, 30.0])
        .unwrap();
    let (keys, values) = unique_by_key(
        zip(&key_a, &key_b),
        zip(&values_a, &values_b),
        MixedTupleEqual,
    )
    .unwrap();
    let (key_a, key_b) = keys;
    let (values_a, values_b) = values;
    assert_eq!(key_a.to_vec().unwrap(), vec![1.0, 2.0, 2.0]);
    assert_eq!(key_b.to_vec().unwrap(), vec![10, 20, 30]);
    assert_eq!(values_a.to_vec().unwrap(), vec![100, 200, 300]);
    assert_eq!(values_b.to_vec().unwrap(), vec![10.0, 20.0, 30.0]);
}

#[test]
fn by_key_algorithms_accept_heterogeneous_soa3_keys_for_sort_and_unique() {
    let policy = policy();
    let key_a = policy.to_device(&[2.0_f32, 1.0, 2.0, 1.0]).unwrap();
    let key_b = policy.to_device(&[20_u32, 10, 10, 20]).unwrap();
    let key_c = policy.to_device(&[0.2_f32, 0.1, 0.3, 0.4]).unwrap();
    let values = policy.to_device(&[200_u32, 100, 210, 120]).unwrap();

    let (keys, values) =
        sort_by_key(zip3(&key_a, &key_b, &key_c), &values, MixedTuple3Less).unwrap();
    let (key_a, key_b, key_c) = keys;
    let values = values;
    assert_eq!(key_a.to_vec().unwrap(), vec![1.0, 2.0, 1.0, 2.0]);
    assert_eq!(key_b.to_vec().unwrap(), vec![10, 10, 20, 20]);
    assert_eq!(key_c.to_vec().unwrap(), vec![0.1, 0.3, 0.4, 0.2]);
    assert_eq!(values.to_vec().unwrap(), vec![100, 210, 120, 200]);

    let left_key_a = policy.to_device(&[1.0_f32, 2.0]).unwrap();
    let left_key_b = policy.to_device(&[10_u32, 20]).unwrap();
    let left_key_c = policy.to_device(&[0.1_f32, 0.2]).unwrap();
    let left_values = policy.to_device(&[100_u32, 200]).unwrap();
    let right_key_a = policy.to_device(&[2.0_f32, 1.0, 3.0]).unwrap();
    let right_key_b = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let right_key_c = policy.to_device(&[0.3_f32, 0.4, 0.5]).unwrap();
    let right_values = policy.to_device(&[210_u32, 120, 300]).unwrap();

    let (keys, values) = merge_by_key(
        zip3(&left_key_a, &left_key_b, &left_key_c),
        &left_values,
        zip3(&right_key_a, &right_key_b, &right_key_c),
        &right_values,
        MixedTuple3Less,
    )
    .unwrap();
    let (key_a, key_b, key_c) = keys;
    let values = values;
    assert_eq!(key_a.to_vec().unwrap(), vec![1.0, 2.0, 1.0, 2.0, 3.0]);
    assert_eq!(key_b.to_vec().unwrap(), vec![10, 10, 20, 20, 30]);
    assert_eq!(key_c.to_vec().unwrap(), vec![0.1, 0.3, 0.4, 0.2, 0.5]);
    assert_eq!(values.to_vec().unwrap(), vec![100, 210, 120, 200, 300]);

    let key_a = policy.to_device(&[1.0_f32, 1.0, 2.0, 2.0, 2.0]).unwrap();
    let key_b = policy.to_device(&[10_u32, 10, 20, 20, 30]).unwrap();
    let key_c = policy.to_device(&[0.1_f32, 0.1, 0.2, 0.2, 0.3]).unwrap();
    let values = policy.to_device(&[1_u32, 2, 3, 4, 5]).unwrap();

    let (keys, values) = reduce_by_key(
        zip3(&key_a, &key_b, &key_c),
        &values,
        MixedTuple3Equal,
        0_u32,
        Sum,
    )
    .unwrap();
    let (key_a_out, key_b_out, key_c_out) = keys;
    let values = values;
    assert_eq!(key_a_out.to_vec().unwrap(), vec![1.0, 2.0, 2.0]);
    assert_eq!(key_b_out.to_vec().unwrap(), vec![10, 20, 30]);
    assert_eq!(key_c_out.to_vec().unwrap(), vec![0.1, 0.2, 0.3]);
    assert_eq!(values.to_vec().unwrap(), vec![3, 7, 5]);

    let scan_values = policy.to_device(&[1_u32, 2, 3, 4, 5]).unwrap();
    let inclusive = inclusive_scan_by_key(
        zip3(&key_a, &key_b, &key_c),
        &scan_values,
        MixedTuple3Equal,
        Sum,
    )
    .unwrap();
    let inclusive = inclusive;
    assert_eq!(inclusive.to_vec().unwrap(), vec![1, 3, 3, 7, 5]);

    let exclusive = exclusive_scan_by_key(
        zip3(&key_a, &key_b, &key_c),
        &scan_values,
        MixedTuple3Equal,
        0_u32,
        Sum,
    )
    .unwrap();
    let exclusive = exclusive;
    assert_eq!(exclusive.to_vec().unwrap(), vec![0, 1, 0, 3, 0]);

    let values = policy.to_device(&[100_u32, 101, 200, 201, 300]).unwrap();
    let (keys, values) =
        unique_by_key(zip3(&key_a, &key_b, &key_c), values, MixedTuple3Equal).unwrap();
    let (key_a, key_b, key_c) = keys;
    let values = values;
    assert_eq!(key_a.to_vec().unwrap(), vec![1.0, 2.0, 2.0]);
    assert_eq!(key_b.to_vec().unwrap(), vec![10, 20, 30]);
    assert_eq!(key_c.to_vec().unwrap(), vec![0.1, 0.2, 0.3]);
    assert_eq!(values.to_vec().unwrap(), vec![100, 200, 300]);
}

#[test]
fn by_key_algorithms_accept_declared_soa3_soa3_key_layouts_and_soa2_values() {
    let policy = policy();
    let key_a = policy.to_device(&[2.0_f32, 1.0, 2.0, 1.0]).unwrap();
    let key_b = policy.to_device(&[20_u32, 10, 10, 20]).unwrap();
    let key_c = policy.to_device(&[0.2_f32, 0.1, 0.3, 0.4]).unwrap();
    let value_a = policy.to_device(&[20_u32, 10, 21, 12]).unwrap();
    let value_b = policy.to_device(&[200.0_f32, 100.0, 210.0, 120.0]).unwrap();

    let (keys, values) = sort_by_key(
        zip3(&key_a, &key_b, &key_c),
        zip(&value_a, &value_b),
        MixedTuple3Less,
    )
    .unwrap();
    let (key_a, key_b, key_c) = keys;
    let (value_a, value_b) = values;
    assert_eq!(key_a.to_vec().unwrap(), vec![1.0, 2.0, 1.0, 2.0]);
    assert_eq!(key_b.to_vec().unwrap(), vec![10, 10, 20, 20]);
    assert_eq!(key_c.to_vec().unwrap(), vec![0.1, 0.3, 0.4, 0.2]);
    assert_eq!(value_a.to_vec().unwrap(), vec![10, 21, 12, 20]);
    assert_eq!(
        value_b.to_vec().unwrap(),
        vec![100.0_f32, 210.0, 120.0, 200.0]
    );

    let left_key_a = policy.to_device(&[1.0_f32, 2.0]).unwrap();
    let left_key_b = policy.to_device(&[10_u32, 20]).unwrap();
    let left_key_c = policy.to_device(&[0.1_f32, 0.2]).unwrap();
    let left_value_a = policy.to_device(&[10_u32, 20]).unwrap();
    let left_value_b = policy.to_device(&[100.0_f32, 200.0]).unwrap();
    let right_key_a = policy.to_device(&[2.0_f32, 1.0, 3.0]).unwrap();
    let right_key_b = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let right_key_c = policy.to_device(&[0.3_f32, 0.4, 0.5]).unwrap();
    let right_value_a = policy.to_device(&[21_u32, 12, 30]).unwrap();
    let right_value_b = policy.to_device(&[210.0_f32, 120.0, 300.0]).unwrap();

    let (keys, values) = merge_by_key(
        zip3(&left_key_a, &left_key_b, &left_key_c),
        zip(&left_value_a, &left_value_b),
        zip3(&right_key_a, &right_key_b, &right_key_c),
        zip(&right_value_a, &right_value_b),
        MixedTuple3Less,
    )
    .unwrap();
    let (key_a, key_b, key_c) = keys;
    let (value_a, value_b) = values;
    assert_eq!(key_a.to_vec().unwrap(), vec![1.0, 2.0, 1.0, 2.0, 3.0]);
    assert_eq!(key_b.to_vec().unwrap(), vec![10, 10, 20, 20, 30]);
    assert_eq!(key_c.to_vec().unwrap(), vec![0.1, 0.3, 0.4, 0.2, 0.5]);
    assert_eq!(value_a.to_vec().unwrap(), vec![10, 21, 12, 20, 30]);
    assert_eq!(
        value_b.to_vec().unwrap(),
        vec![100.0_f32, 210.0, 120.0, 200.0, 300.0]
    );

    let key_a = policy.to_device(&[1.0_f32, 1.0, 2.0, 2.0, 2.0]).unwrap();
    let key_b = policy.to_device(&[10_u32, 10, 20, 20, 30]).unwrap();
    let key_c = policy.to_device(&[0.1_f32, 0.1, 0.2, 0.2, 0.3]).unwrap();
    let value_a = policy.to_device(&[1_u32, 2, 3, 4, 5]).unwrap();
    let value_b = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0, 5.0]).unwrap();

    let (keys, values) = reduce_by_key(
        zip3(&key_a, &key_b, &key_c),
        zip(&value_a, &value_b),
        MixedTuple3Equal,
        (0_u32, 0.0_f32),
        Sum,
    )
    .unwrap();
    let (key_a_out, key_b_out, key_c_out) = keys;
    let (value_a_out, value_b_out) = values;
    assert_eq!(key_a_out.to_vec().unwrap(), vec![1.0, 2.0, 2.0]);
    assert_eq!(key_b_out.to_vec().unwrap(), vec![10, 20, 30]);
    assert_eq!(key_c_out.to_vec().unwrap(), vec![0.1, 0.2, 0.3]);
    assert_eq!(value_a_out.to_vec().unwrap(), vec![3, 7, 5]);
    assert_eq!(value_b_out.to_vec().unwrap(), vec![3.0, 7.0, 5.0]);

    let inclusive = inclusive_scan_by_key(
        zip3(&key_a, &key_b, &key_c),
        zip(&value_a, &value_b),
        MixedTuple3Equal,
        Sum,
    )
    .unwrap();
    let (value_a_out, value_b_out) = inclusive;
    assert_eq!(value_a_out.to_vec().unwrap(), vec![1, 3, 3, 7, 5]);
    assert_eq!(value_b_out.to_vec().unwrap(), vec![1.0, 3.0, 3.0, 7.0, 5.0]);

    let exclusive = exclusive_scan_by_key(
        zip3(&key_a, &key_b, &key_c),
        zip(&value_a, &value_b),
        MixedTuple3Equal,
        (0_u32, 0.0_f32),
        Sum,
    )
    .unwrap();
    let (value_a_out, value_b_out) = exclusive;
    assert_eq!(value_a_out.to_vec().unwrap(), vec![0, 1, 0, 3, 0]);
    assert_eq!(value_b_out.to_vec().unwrap(), vec![0.0, 1.0, 0.0, 3.0, 0.0]);

    let values_a = policy.to_device(&[100_u32, 101, 200, 201, 300]).unwrap();
    let values_b = policy
        .to_device(&[10.0_f32, 10.1, 20.0, 20.1, 30.0])
        .unwrap();
    let (keys, values) = unique_by_key(
        zip3(&key_a, &key_b, &key_c),
        zip(&values_a, &values_b),
        MixedTuple3Equal,
    )
    .unwrap();
    let (key_a, key_b, key_c) = keys;
    let (values_a, values_b) = values;
    assert_eq!(key_a.to_vec().unwrap(), vec![1.0, 2.0, 2.0]);
    assert_eq!(key_b.to_vec().unwrap(), vec![10, 20, 30]);
    assert_eq!(key_c.to_vec().unwrap(), vec![0.1, 0.2, 0.3]);
    assert_eq!(values_a.to_vec().unwrap(), vec![100, 200, 300]);
    assert_eq!(values_b.to_vec().unwrap(), vec![10.0, 20.0, 30.0]);
}

#[test]
fn by_key_algorithms_accept_declared_soa3_soa3_key_layouts_and_soa3_values() {
    let policy = policy();
    let key_a = policy.to_device(&[2.0_f32, 1.0, 2.0, 1.0]).unwrap();
    let key_b = policy.to_device(&[20_u32, 10, 10, 20]).unwrap();
    let key_c = policy.to_device(&[0.2_f32, 0.1, 0.3, 0.4]).unwrap();
    let value_a = policy.to_device(&[20_u32, 10, 21, 12]).unwrap();
    let value_b = policy.to_device(&[200.0_f32, 100.0, 210.0, 120.0]).unwrap();
    let value_c = policy.to_device(&[2_u32, 1, 3, 4]).unwrap();

    let (keys, values) = sort_by_key(
        zip3(&key_a, &key_b, &key_c),
        zip3(&value_a, &value_b, &value_c),
        MixedTuple3Less,
    )
    .unwrap();
    let (key_a, key_b, key_c) = keys;
    let (value_a, value_b, value_c) = values;
    assert_eq!(key_a.to_vec().unwrap(), vec![1.0, 2.0, 1.0, 2.0]);
    assert_eq!(key_b.to_vec().unwrap(), vec![10, 10, 20, 20]);
    assert_eq!(key_c.to_vec().unwrap(), vec![0.1, 0.3, 0.4, 0.2]);
    assert_eq!(value_a.to_vec().unwrap(), vec![10, 21, 12, 20]);
    assert_eq!(
        value_b.to_vec().unwrap(),
        vec![100.0_f32, 210.0, 120.0, 200.0]
    );
    assert_eq!(value_c.to_vec().unwrap(), vec![1, 3, 4, 2]);

    let left_key_a = policy.to_device(&[1.0_f32, 2.0]).unwrap();
    let left_key_b = policy.to_device(&[10_u32, 20]).unwrap();
    let left_key_c = policy.to_device(&[0.1_f32, 0.2]).unwrap();
    let left_value_a = policy.to_device(&[10_u32, 20]).unwrap();
    let left_value_b = policy.to_device(&[100.0_f32, 200.0]).unwrap();
    let left_value_c = policy.to_device(&[1_u32, 2]).unwrap();
    let right_key_a = policy.to_device(&[2.0_f32, 1.0, 3.0]).unwrap();
    let right_key_b = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let right_key_c = policy.to_device(&[0.3_f32, 0.4, 0.5]).unwrap();
    let right_value_a = policy.to_device(&[21_u32, 12, 30]).unwrap();
    let right_value_b = policy.to_device(&[210.0_f32, 120.0, 300.0]).unwrap();
    let right_value_c = policy.to_device(&[3_u32, 4, 5]).unwrap();

    let (keys, values) = merge_by_key(
        zip3(&left_key_a, &left_key_b, &left_key_c),
        zip3(&left_value_a, &left_value_b, &left_value_c),
        zip3(&right_key_a, &right_key_b, &right_key_c),
        zip3(&right_value_a, &right_value_b, &right_value_c),
        MixedTuple3Less,
    )
    .unwrap();
    let (key_a, key_b, key_c) = keys;
    let (value_a, value_b, value_c) = values;
    assert_eq!(key_a.to_vec().unwrap(), vec![1.0, 2.0, 1.0, 2.0, 3.0]);
    assert_eq!(key_b.to_vec().unwrap(), vec![10, 10, 20, 20, 30]);
    assert_eq!(key_c.to_vec().unwrap(), vec![0.1, 0.3, 0.4, 0.2, 0.5]);
    assert_eq!(value_a.to_vec().unwrap(), vec![10, 21, 12, 20, 30]);
    assert_eq!(
        value_b.to_vec().unwrap(),
        vec![100.0_f32, 210.0, 120.0, 200.0, 300.0]
    );
    assert_eq!(value_c.to_vec().unwrap(), vec![1, 3, 4, 2, 5]);

    let key_a = policy.to_device(&[1.0_f32, 1.0, 2.0, 2.0, 2.0]).unwrap();
    let key_b = policy.to_device(&[10_u32, 10, 20, 20, 30]).unwrap();
    let key_c = policy.to_device(&[0.1_f32, 0.1, 0.2, 0.2, 0.3]).unwrap();
    let value_a = policy.to_device(&[1_u32, 2, 3, 4, 5]).unwrap();
    let value_b = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let value_c = policy.to_device(&[10_u32, 20, 30, 40, 50]).unwrap();

    let (keys, values) = reduce_by_key(
        zip3(&key_a, &key_b, &key_c),
        zip3(&value_a, &value_b, &value_c),
        MixedTuple3Equal,
        (0_u32, 0.0_f32, 0_u32),
        Sum,
    )
    .unwrap();
    let (key_a_out, key_b_out, key_c_out) = keys;
    let (value_a_out, value_b_out, value_c_out) = values;
    assert_eq!(key_a_out.to_vec().unwrap(), vec![1.0, 2.0, 2.0]);
    assert_eq!(key_b_out.to_vec().unwrap(), vec![10, 20, 30]);
    assert_eq!(key_c_out.to_vec().unwrap(), vec![0.1, 0.2, 0.3]);
    assert_eq!(value_a_out.to_vec().unwrap(), vec![3, 7, 5]);
    assert_eq!(value_b_out.to_vec().unwrap(), vec![3.0, 7.0, 5.0]);
    assert_eq!(value_c_out.to_vec().unwrap(), vec![30, 70, 50]);

    let inclusive = inclusive_scan_by_key(
        zip3(&key_a, &key_b, &key_c),
        zip3(&value_a, &value_b, &value_c),
        MixedTuple3Equal,
        Sum,
    )
    .unwrap();
    let (value_a_out, value_b_out, value_c_out) = inclusive;
    assert_eq!(value_a_out.to_vec().unwrap(), vec![1, 3, 3, 7, 5]);
    assert_eq!(value_b_out.to_vec().unwrap(), vec![1.0, 3.0, 3.0, 7.0, 5.0]);
    assert_eq!(value_c_out.to_vec().unwrap(), vec![10, 30, 30, 70, 50]);

    let exclusive = exclusive_scan_by_key(
        zip3(&key_a, &key_b, &key_c),
        zip3(&value_a, &value_b, &value_c),
        MixedTuple3Equal,
        (0_u32, 0.0_f32, 0_u32),
        Sum,
    )
    .unwrap();
    let (value_a_out, value_b_out, value_c_out) = exclusive;
    assert_eq!(value_a_out.to_vec().unwrap(), vec![0, 1, 0, 3, 0]);
    assert_eq!(value_b_out.to_vec().unwrap(), vec![0.0, 1.0, 0.0, 3.0, 0.0]);
    assert_eq!(value_c_out.to_vec().unwrap(), vec![0, 10, 0, 30, 0]);

    let values_a = policy.to_device(&[100_u32, 101, 200, 201, 300]).unwrap();
    let values_b = policy
        .to_device(&[10.0_f32, 10.1, 20.0, 20.1, 30.0])
        .unwrap();
    let values_c = policy.to_device(&[1_u32, 2, 3, 4, 5]).unwrap();
    let (keys, values) = unique_by_key(
        zip3(&key_a, &key_b, &key_c),
        zip3(&values_a, &values_b, &values_c),
        MixedTuple3Equal,
    )
    .unwrap();
    let (key_a, key_b, key_c) = keys;
    let (values_a, values_b, values_c) = values;
    assert_eq!(key_a.to_vec().unwrap(), vec![1.0, 2.0, 2.0]);
    assert_eq!(key_b.to_vec().unwrap(), vec![10, 20, 30]);
    assert_eq!(key_c.to_vec().unwrap(), vec![0.1, 0.2, 0.3]);
    assert_eq!(values_a.to_vec().unwrap(), vec![100, 200, 300]);
    assert_eq!(values_b.to_vec().unwrap(), vec![10.0, 20.0, 30.0]);
    assert_eq!(values_c.to_vec().unwrap(), vec![1, 3, 5]);
}

#[test]
fn by_key_algorithms_accept_declared_soa2_soa2_key_layouts_and_soa3_values() {
    let policy = policy();
    let key_a = policy.to_device(&[2.0_f32, 1.0, 2.0, 1.0]).unwrap();
    let key_b = policy.to_device(&[20_u32, 10, 10, 20]).unwrap();
    let value_a = policy.to_device(&[20_u32, 10, 21, 12]).unwrap();
    let value_b = policy.to_device(&[200.0_f32, 100.0, 210.0, 120.0]).unwrap();
    let value_c = policy.to_device(&[2_u32, 1, 3, 4]).unwrap();

    let (keys, values) = sort_by_key(
        zip(&key_a, &key_b),
        zip3(&value_a, &value_b, &value_c),
        MixedTupleLess,
    )
    .unwrap();
    let (key_a, key_b) = keys;
    let (value_a, value_b, value_c) = values;
    assert_eq!(key_a.to_vec().unwrap(), vec![1.0, 1.0, 2.0, 2.0]);
    assert_eq!(key_b.to_vec().unwrap(), vec![10, 20, 10, 20]);
    assert_eq!(value_a.to_vec().unwrap(), vec![10, 12, 21, 20]);
    assert_eq!(
        value_b.to_vec().unwrap(),
        vec![100.0_f32, 120.0, 210.0, 200.0]
    );
    assert_eq!(value_c.to_vec().unwrap(), vec![1, 4, 3, 2]);

    let key_a = policy.to_device(&[1.0_f32, 1.0, 2.0, 2.0, 2.0]).unwrap();
    let key_b = policy.to_device(&[10_u32, 10, 20, 20, 30]).unwrap();
    let value_a = policy.to_device(&[1_u32, 2, 3, 4, 5]).unwrap();
    let value_b = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let value_c = policy.to_device(&[10_u32, 20, 30, 40, 50]).unwrap();

    let (keys, values) = reduce_by_key(
        zip(&key_a, &key_b),
        zip3(&value_a, &value_b, &value_c),
        MixedTupleEqual,
        (0_u32, 0.0_f32, 0_u32),
        Sum,
    )
    .unwrap();
    let (key_a_out, key_b_out) = keys;
    let (value_a_out, value_b_out, value_c_out) = values;
    assert_eq!(key_a_out.to_vec().unwrap(), vec![1.0, 2.0, 2.0]);
    assert_eq!(key_b_out.to_vec().unwrap(), vec![10, 20, 30]);
    assert_eq!(value_a_out.to_vec().unwrap(), vec![3, 7, 5]);
    assert_eq!(value_b_out.to_vec().unwrap(), vec![3.0, 7.0, 5.0]);
    assert_eq!(value_c_out.to_vec().unwrap(), vec![30, 70, 50]);

    let inclusive = inclusive_scan_by_key(
        zip(&key_a, &key_b),
        zip3(&value_a, &value_b, &value_c),
        MixedTupleEqual,
        Sum,
    )
    .unwrap();
    let (value_a_out, value_b_out, value_c_out) = inclusive;
    assert_eq!(value_a_out.to_vec().unwrap(), vec![1, 3, 3, 7, 5]);
    assert_eq!(value_b_out.to_vec().unwrap(), vec![1.0, 3.0, 3.0, 7.0, 5.0]);
    assert_eq!(value_c_out.to_vec().unwrap(), vec![10, 30, 30, 70, 50]);

    let exclusive = exclusive_scan_by_key(
        zip(&key_a, &key_b),
        zip3(&value_a, &value_b, &value_c),
        MixedTupleEqual,
        (0_u32, 0.0_f32, 0_u32),
        Sum,
    )
    .unwrap();
    let (value_a_out, value_b_out, value_c_out) = exclusive;
    assert_eq!(value_a_out.to_vec().unwrap(), vec![0, 1, 0, 3, 0]);
    assert_eq!(value_b_out.to_vec().unwrap(), vec![0.0, 1.0, 0.0, 3.0, 0.0]);
    assert_eq!(value_c_out.to_vec().unwrap(), vec![0, 10, 0, 30, 0]);

    let values_a = policy.to_device(&[100_u32, 101, 200, 201, 300]).unwrap();
    let values_b = policy
        .to_device(&[10.0_f32, 10.1, 20.0, 20.1, 30.0])
        .unwrap();
    let values_c = policy.to_device(&[1_u32, 2, 3, 4, 5]).unwrap();
    let (keys, values) = unique_by_key(
        zip(&key_a, &key_b),
        zip3(&values_a, &values_b, &values_c),
        MixedTupleEqual,
    )
    .unwrap();
    let (key_a, key_b) = keys;
    let (values_a, values_b, values_c) = values;
    assert_eq!(key_a.to_vec().unwrap(), vec![1.0, 2.0, 2.0]);
    assert_eq!(key_b.to_vec().unwrap(), vec![10, 20, 30]);
    assert_eq!(values_a.to_vec().unwrap(), vec![100, 200, 300]);
    assert_eq!(values_b.to_vec().unwrap(), vec![10.0, 20.0, 30.0]);
    assert_eq!(values_c.to_vec().unwrap(), vec![1, 3, 5]);
}

#[test]
fn sort_by_key_accepts_borrowed_heterogeneous_soa2_keys_and_soa4_values() {
    let policy = policy();
    let key_a = policy.to_device(&[2.0_f32, 1.0, 2.0, 1.0]).unwrap();
    let key_b = policy.to_device(&[20_u32, 10, 10, 20]).unwrap();
    let value_a = policy.to_device(&[20_u32, 10, 21, 12]).unwrap();
    let value_b = policy.to_device(&[200.0_f32, 100.0, 210.0, 120.0]).unwrap();
    let value_c = policy.to_device(&[2_u32, 1, 3, 4]).unwrap();
    let value_d = policy.to_device(&[20.0_f32, 10.0, 30.0, 40.0]).unwrap();

    let (keys, values) = sort_by_key(
        zip(&key_a, &key_b),
        zip4(&value_a, &value_b, &value_c, &value_d),
        MixedTupleLess,
    )
    .unwrap();
    let (key_a, key_b) = keys;
    let (value_a, value_b, value_c, value_d) = values;
    assert_eq!(key_a.to_vec().unwrap(), vec![1.0, 1.0, 2.0, 2.0]);
    assert_eq!(key_b.to_vec().unwrap(), vec![10, 20, 10, 20]);
    assert_eq!(value_a.to_vec().unwrap(), vec![10, 12, 21, 20]);
    assert_eq!(
        value_b.to_vec().unwrap(),
        vec![100.0_f32, 120.0, 210.0, 200.0]
    );
    assert_eq!(value_c.to_vec().unwrap(), vec![1, 4, 3, 2]);
    assert_eq!(value_d.to_vec().unwrap(), vec![10.0, 40.0, 30.0, 20.0]);
}

#[test]
fn by_key_algorithms_accept_matrix_edges() {
    let policy = policy();

    let key_a = policy.to_device(&[2.0_f32, 1.0, 2.0, 1.0]).unwrap();
    let key_b = policy.to_device(&[20_u32, 10, 10, 20]).unwrap();
    let va = policy.to_device(&[20_u32, 10, 21, 12]).unwrap();
    let vb = policy.to_device(&[20.0_f32, 10.0, 21.0, 12.0]).unwrap();
    let vc = policy.to_device(&[200_u32, 100, 210, 120]).unwrap();
    let vd = policy.to_device(&[200.0_f32, 100.0, 210.0, 120.0]).unwrap();
    let ve = policy.to_device(&[2000_u32, 1000, 2100, 1200]).unwrap();
    let vf = policy
        .to_device(&[2000.0_f32, 1000.0, 2100.0, 1200.0])
        .unwrap();
    let vg = policy.to_device(&[3_u32, 1, 4, 2]).unwrap();
    let vh = policy.to_device(&[3.0_f32, 1.0, 4.0, 2.0]).unwrap();
    let vi = policy.to_device(&[30_u32, 10, 40, 20]).unwrap();
    let vj = policy.to_device(&[30.0_f32, 10.0, 40.0, 20.0]).unwrap();
    let vk = policy.to_device(&[300_u32, 100, 400, 200]).unwrap();
    let vl = policy.to_device(&[300.0_f32, 100.0, 400.0, 200.0]).unwrap();

    let (keys, values) = sort_by_key(
        zip(&key_a, &key_b),
        zip12(&va, &vb, &vc, &vd, &ve, &vf, &vg, &vh, &vi, &vj, &vk, &vl),
        MixedTupleLess,
    )
    .unwrap();
    let (key_a, key_b) = keys;
    let (va, _, _, _, _, _, _, _, _, _, _, vl) = values;
    assert_eq!(key_a.to_vec().unwrap(), vec![1.0, 1.0, 2.0, 2.0]);
    assert_eq!(key_b.to_vec().unwrap(), vec![10, 20, 10, 20]);
    assert_eq!(va.to_vec().unwrap(), vec![10, 12, 21, 20]);
    assert_eq!(vl.to_vec().unwrap(), vec![100.0, 200.0, 400.0, 300.0]);

    let la = policy.to_device(&[1.0_f32, 3.0]).unwrap();
    let lb = policy.to_device(&[10_u32, 30]).unwrap();
    let lc = policy.to_device(&[1.1_f32, 3.1]).unwrap();
    let ld = policy.to_device(&[11_u32, 31]).unwrap();
    let le = policy.to_device(&[1.2_f32, 3.2]).unwrap();
    let lf = policy.to_device(&[12_u32, 32]).unwrap();
    let lg = policy.to_device(&[1.3_f32, 3.3]).unwrap();
    let lh = policy.to_device(&[13_u32, 33]).unwrap();
    let li = policy.to_device(&[1.4_f32, 3.4]).unwrap();
    let lj = policy.to_device(&[14_u32, 34]).unwrap();
    let lk = policy.to_device(&[1.5_f32, 3.5]).unwrap();
    let ll = policy.to_device(&[15_u32, 35]).unwrap();
    let ra = policy.to_device(&[2.0_f32]).unwrap();
    let rb = policy.to_device(&[20_u32]).unwrap();
    let rc = policy.to_device(&[2.1_f32]).unwrap();
    let rd = policy.to_device(&[21_u32]).unwrap();
    let re = policy.to_device(&[2.2_f32]).unwrap();
    let rf = policy.to_device(&[22_u32]).unwrap();
    let rg = policy.to_device(&[2.3_f32]).unwrap();
    let rh = policy.to_device(&[23_u32]).unwrap();
    let ri = policy.to_device(&[2.4_f32]).unwrap();
    let rj = policy.to_device(&[24_u32]).unwrap();
    let rk = policy.to_device(&[2.5_f32]).unwrap();
    let rl = policy.to_device(&[25_u32]).unwrap();
    let lva = policy.to_device(&[10_u32, 30]).unwrap();
    let lvb = policy.to_device(&[10.0_f32, 30.0]).unwrap();
    let lvc = policy.to_device(&[100_u32, 300]).unwrap();
    let rva = policy.to_device(&[20_u32]).unwrap();
    let rvb = policy.to_device(&[20.0_f32]).unwrap();
    let rvc = policy.to_device(&[200_u32]).unwrap();

    let (keys, values) = merge_by_key(
        zip12(&la, &lb, &lc, &ld, &le, &lf, &lg, &lh, &li, &lj, &lk, &ll),
        zip3(&lva, &lvb, &lvc),
        zip12(&ra, &rb, &rc, &rd, &re, &rf, &rg, &rh, &ri, &rj, &rk, &rl),
        zip3(&rva, &rvb, &rvc),
        Tuple12MixedLess,
    )
    .unwrap();
    let (ka, kb, _, _, _, _, _, _, _, _, kk, kl) = keys;
    let (va, vb, vc) = values;
    assert_eq!(ka.to_vec().unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(kb.to_vec().unwrap(), vec![10, 20, 30]);
    assert_eq!(kk.to_vec().unwrap(), vec![1.5, 2.5, 3.5]);
    assert_eq!(kl.to_vec().unwrap(), vec![15, 25, 35]);
    assert_eq!(va.to_vec().unwrap(), vec![10, 20, 30]);
    assert_eq!(vb.to_vec().unwrap(), vec![10.0, 20.0, 30.0]);
    assert_eq!(vc.to_vec().unwrap(), vec![100, 200, 300]);

    let ua = policy.to_device(&[1.0_f32, 1.0, 2.0, 2.0]).unwrap();
    let ub = policy.to_device(&[10_u32, 10, 20, 20]).unwrap();
    let uc = policy.to_device(&[1.1_f32, 1.1, 2.1, 2.1]).unwrap();
    let ud = policy.to_device(&[11_u32, 11, 21, 21]).unwrap();
    let ue = policy.to_device(&[1.2_f32, 1.2, 2.2, 2.2]).unwrap();
    let uf = policy.to_device(&[12_u32, 12, 22, 22]).unwrap();
    let ug = policy.to_device(&[1.3_f32, 1.3, 2.3, 2.3]).unwrap();
    let uh = policy.to_device(&[13_u32, 13, 23, 23]).unwrap();
    let ui = policy.to_device(&[1.4_f32, 1.4, 2.4, 2.4]).unwrap();
    let uj = policy.to_device(&[14_u32, 14, 24, 24]).unwrap();
    let uk = policy.to_device(&[1.5_f32, 1.5, 2.5, 2.5]).unwrap();
    let ul = policy.to_device(&[15_u32, 15, 25, 25]).unwrap();
    let uva = policy.to_device(&[10_u32, 11, 20, 21]).unwrap();
    let uvb = policy.to_device(&[10.0_f32, 11.0, 20.0, 21.0]).unwrap();
    let uvc = policy.to_device(&[100_u32, 110, 200, 210]).unwrap();

    let (keys, values) = unique_by_key(
        zip12(&ua, &ub, &uc, &ud, &ue, &uf, &ug, &uh, &ui, &uj, &uk, &ul),
        zip3(&uva, &uvb, &uvc),
        Tuple12MixedEqual,
    )
    .unwrap();
    let (ka, kb, _, _, _, _, _, _, _, _, kk, kl) = keys;
    let (va, vb, vc) = values;
    assert_eq!(ka.to_vec().unwrap(), vec![1.0, 2.0]);
    assert_eq!(kb.to_vec().unwrap(), vec![10, 20]);
    assert_eq!(kk.to_vec().unwrap(), vec![1.5, 2.5]);
    assert_eq!(kl.to_vec().unwrap(), vec![15, 25]);
    assert_eq!(va.to_vec().unwrap(), vec![10, 20]);
    assert_eq!(vb.to_vec().unwrap(), vec![10.0, 20.0]);
    assert_eq!(vc.to_vec().unwrap(), vec![100, 200]);
}
