mod common;
use common::*;

#[test]
fn reduce_and_scan_accept_heterogeneous_columns_when_op_supports_each_item() {
    let policy = policy();
    let values = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let ids = policy.to_device(&[10_u32, 20, 30]).unwrap();

    let reduced = reduce(zip(&values, &ids), (0.0_f32, 0_u32), Sum).unwrap();
    assert_eq!(reduced, (6.0, 60));

    let inclusive = inclusive_scan(zip(&values, &ids), Sum).unwrap();
    let (inclusive_values, inclusive_ids) = inclusive;
    assert_eq!(inclusive_values.to_vec().unwrap(), vec![1.0, 3.0, 6.0]);
    assert_eq!(inclusive_ids.to_vec().unwrap(), vec![10, 30, 60]);

    let exclusive = exclusive_scan(zip(&values, &ids), (0.0_f32, 0_u32), Sum).unwrap();
    let (exclusive_values, exclusive_ids) = exclusive;
    assert_eq!(exclusive_values.to_vec().unwrap(), vec![0.0, 1.0, 3.0]);
    assert_eq!(exclusive_ids.to_vec().unwrap(), vec![0, 10, 30]);
}

#[test]
fn reduce_and_scan_accept_borrowed_heterogeneous_soas() {
    let policy = policy();
    let values = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let ids = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let reduced = reduce(zip(&values, &ids), (0.0_f32, 0_u32), Sum).unwrap();
    assert_eq!(reduced, (6.0, 60));

    let values = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let ids = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let inclusive = inclusive_scan(zip(&values, &ids), Sum).unwrap();
    let (values, ids) = inclusive;
    assert_eq!(values.to_vec().unwrap(), vec![1.0, 3.0, 6.0]);
    assert_eq!(ids.to_vec().unwrap(), vec![10, 30, 60]);

    let values = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let ids = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let exclusive = exclusive_scan(zip(&values, &ids), (0.0_f32, 0_u32), Sum).unwrap();
    let (values, ids) = exclusive;
    assert_eq!(values.to_vec().unwrap(), vec![0.0, 1.0, 3.0]);
    assert_eq!(ids.to_vec().unwrap(), vec![0, 10, 30]);

    let values = policy.to_device(&[1.0_f32, 3.0, 6.0]).unwrap();
    let ids = policy.to_device(&[10_u32, 30, 60]).unwrap();
    let diff = adjacent_difference(zip(&values, &ids), Sum).unwrap();
    let (values, ids) = diff;
    assert_eq!(values.to_vec().unwrap(), vec![1.0, 4.0, 9.0]);
    assert_eq!(ids.to_vec().unwrap(), vec![10, 40, 90]);
}

#[test]
fn reduce_and_scan_accept_one_component_borrowed_soas() {
    let policy = policy();
    let input = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();

    let sum = reduce(&input, 0.0, Sum).unwrap();
    assert_eq!(sum, 10.0);

    let inclusive = inclusive_scan(&input, Sum).unwrap();
    assert_eq!(inclusive.to_vec().unwrap(), vec![1.0, 3.0, 6.0, 10.0]);

    let exclusive = exclusive_scan(&input, 10.0, Sum).unwrap();
    assert_eq!(exclusive.to_vec().unwrap(), vec![10.0, 11.0, 13.0, 16.0]);
}

#[test]
fn inner_product_accepts_heterogeneous_soas() {
    let policy = policy();
    let left_a = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let left_b = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let right_a = policy.to_device(&[4.0_f32, 5.0, 6.0]).unwrap();
    let right_b = policy.to_device(&[40_u32, 50, 60]).unwrap();

    let pair = inner_product(
        zip(&left_a, &left_b),
        zip(&right_a, &right_b),
        (0.0_f32, 0_u32),
        Sum,
        Sum,
    )
    .unwrap();
    assert_eq!(pair, (21.0, 210));

    let a = policy.to_device(&[1.0_f32, 2.0]).unwrap();
    let b = policy.to_device(&[1_u32, 2]).unwrap();
    let c = policy.to_device(&[3.0_f32, 4.0]).unwrap();
    let d = policy.to_device(&[3_u32, 4]).unwrap();
    let e = policy.to_device(&[5.0_f32, 6.0]).unwrap();
    let f = policy.to_device(&[5_u32, 6]).unwrap();
    let g = policy.to_device(&[7.0_f32, 8.0]).unwrap();
    let h = policy.to_device(&[7_u32, 8]).unwrap();
    let i = policy.to_device(&[9.0_f32, 10.0]).unwrap();
    let j = policy.to_device(&[9_u32, 10]).unwrap();
    let k = policy.to_device(&[11.0_f32, 12.0]).unwrap();
    let l = policy.to_device(&[11_u32, 12]).unwrap();
    let ra = policy.to_device(&[2.0_f32, 3.0]).unwrap();
    let rb = policy.to_device(&[2_u32, 3]).unwrap();
    let rc = policy.to_device(&[4.0_f32, 5.0]).unwrap();
    let rd = policy.to_device(&[4_u32, 5]).unwrap();
    let re = policy.to_device(&[6.0_f32, 7.0]).unwrap();
    let rf = policy.to_device(&[6_u32, 7]).unwrap();
    let rg = policy.to_device(&[8.0_f32, 9.0]).unwrap();
    let rh = policy.to_device(&[8_u32, 9]).unwrap();
    let ri = policy.to_device(&[10.0_f32, 11.0]).unwrap();
    let rj = policy.to_device(&[10_u32, 11]).unwrap();
    let rk = policy.to_device(&[12.0_f32, 13.0]).unwrap();
    let rl = policy.to_device(&[12_u32, 13]).unwrap();

    let wide = inner_product(
        zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
        zip12(&ra, &rb, &rc, &rd, &re, &rf, &rg, &rh, &ri, &rj, &rk, &rl),
        (
            0.0_f32, 0_u32, 0.0_f32, 0_u32, 0.0_f32, 0_u32, 0.0_f32, 0_u32, 0.0_f32, 0_u32,
            0.0_f32, 0_u32,
        ),
        Sum,
        Sum,
    )
    .unwrap();
    assert_eq!(
        wide,
        (8.0, 8, 16.0, 16, 24.0, 24, 32.0, 32, 40.0, 40, 48.0, 48)
    );

    let lhs_a = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let lhs_b = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let rhs_a = policy.to_device(&[4.0_f32, 5.0, 6.0]).unwrap();
    let rhs_b = policy.to_device(&[40_u32, 50, 60]).unwrap();
    let zipped = inner_product(
        zip(&lhs_a, &lhs_b),
        zip(&rhs_a, &rhs_b),
        (0.0_f32, 0_u32),
        Sum,
        Sum,
    )
    .unwrap();
    assert_eq!(zipped, (21.0, 210));

    let mixed_left_a = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let mixed_left_b = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let mixed_right_a = policy.to_device(&[4.0_f32, 5.0, 6.0]).unwrap();
    let mixed_right_b = policy.to_device(&[40_u32, 50, 60]).unwrap();
    let mixed = inner_product(
        zip(&mixed_left_a, &mixed_left_b),
        zip(&mixed_right_a, &mixed_right_b),
        (0.0_f32, 0_u32),
        Sum,
        Sum,
    )
    .unwrap();
    assert_eq!(mixed, (21.0, 210));

    let mixed_left_a = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let mixed_left_b = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let mixed_right_a = policy.to_device(&[4.0_f32, 5.0, 6.0]).unwrap();
    let mixed_right_b = policy.to_device(&[40_u32, 50, 60]).unwrap();
    let mixed = inner_product(
        zip(&mixed_left_a, &mixed_left_b),
        zip(&mixed_right_a, &mixed_right_b),
        (0.0_f32, 0_u32),
        Sum,
        Sum,
    )
    .unwrap();
    assert_eq!(mixed, (21.0, 210));
}

#[test]
fn reduce_accepts_soa12() {
    let policy = policy();
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

    let sums = reduce(
        zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
        (
            0.0, 0_u32, 0.0, 0_u32, 0.0, 0_u32, 0.0, 0_u32, 0.0, 0_u32, 0.0, 0_u32,
        ),
        Sum,
    )
    .unwrap();

    assert_eq!(
        sums,
        (
            6.0, 60, 600.0, 6000, 15.0, 150, 1500.0, 15000, 24.0, 240, 2400.0, 24000
        )
    );
}

#[test]
fn scan_accepts_soa12() {
    let policy = policy();
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

    let inclusive =
        inclusive_scan(zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l), Sum).unwrap();
    let (a_out, b_out, c_out, d_out, e_out, f_out, g_out, h_out, i_out, j_out, k_out, l_out) =
        inclusive;
    assert_eq!(a_out.to_vec().unwrap(), vec![1.0, 3.0, 6.0]);
    assert_eq!(b_out.to_vec().unwrap(), vec![10, 30, 60]);
    assert_eq!(c_out.to_vec().unwrap(), vec![100.0, 300.0, 600.0]);
    assert_eq!(d_out.to_vec().unwrap(), vec![1000, 3000, 6000]);
    assert_eq!(e_out.to_vec().unwrap(), vec![4.0, 9.0, 15.0]);
    assert_eq!(f_out.to_vec().unwrap(), vec![40, 90, 150]);
    assert_eq!(g_out.to_vec().unwrap(), vec![400.0, 900.0, 1500.0]);
    assert_eq!(h_out.to_vec().unwrap(), vec![4000, 9000, 15000]);
    assert_eq!(i_out.to_vec().unwrap(), vec![7.0, 15.0, 24.0]);
    assert_eq!(j_out.to_vec().unwrap(), vec![70, 150, 240]);
    assert_eq!(k_out.to_vec().unwrap(), vec![700.0, 1500.0, 2400.0]);
    assert_eq!(l_out.to_vec().unwrap(), vec![7000, 15000, 24000]);

    let exclusive = exclusive_scan(
        zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
        (
            0.0, 10_u32, 100.0, 1000_u32, 4.0, 40_u32, 400.0, 4000_u32, 7.0, 70_u32, 700.0,
            7000_u32,
        ),
        Sum,
    )
    .unwrap();
    let (a_out, b_out, c_out, d_out, e_out, f_out, g_out, h_out, i_out, j_out, k_out, l_out) =
        exclusive;
    assert_eq!(a_out.to_vec().unwrap(), vec![0.0, 1.0, 3.0]);
    assert_eq!(b_out.to_vec().unwrap(), vec![10, 20, 40]);
    assert_eq!(c_out.to_vec().unwrap(), vec![100.0, 200.0, 400.0]);
    assert_eq!(d_out.to_vec().unwrap(), vec![1000, 2000, 4000]);
    assert_eq!(e_out.to_vec().unwrap(), vec![4.0, 8.0, 13.0]);
    assert_eq!(f_out.to_vec().unwrap(), vec![40, 80, 130]);
    assert_eq!(g_out.to_vec().unwrap(), vec![400.0, 800.0, 1300.0]);
    assert_eq!(h_out.to_vec().unwrap(), vec![4000, 8000, 13000]);
    assert_eq!(i_out.to_vec().unwrap(), vec![7.0, 14.0, 22.0]);
    assert_eq!(j_out.to_vec().unwrap(), vec![70, 140, 220]);
    assert_eq!(k_out.to_vec().unwrap(), vec![700.0, 1400.0, 2200.0]);
    assert_eq!(l_out.to_vec().unwrap(), vec![7000, 14000, 22000]);
}

#[test]
fn adjacent_difference_accepts_soa12() {
    let policy = policy();
    let a = policy.to_device(&[1.0_f32, 3.0, 6.0, 10.0]).unwrap();
    let b = policy.to_device(&[10_u32, 30, 60, 100]).unwrap();
    let c = policy.to_device(&[2.0_f32, 5.0, 9.0, 14.0]).unwrap();
    let d = policy.to_device(&[20_u32, 50, 90, 140]).unwrap();
    let e = policy.to_device(&[4.0_f32, 9.0, 15.0, 22.0]).unwrap();
    let f = policy.to_device(&[40_u32, 90, 150, 220]).unwrap();
    let g = policy.to_device(&[7.0_f32, 11.0, 18.0, 26.0]).unwrap();
    let h = policy.to_device(&[70_u32, 110, 180, 260]).unwrap();
    let i = policy.to_device(&[8.0_f32, 13.0, 21.0, 34.0]).unwrap();
    let j = policy.to_device(&[80_u32, 130, 210, 340]).unwrap();
    let k = policy.to_device(&[12.0_f32, 19.0, 31.0, 50.0]).unwrap();
    let l = policy.to_device(&[120_u32, 190, 310, 500]).unwrap();

    let output =
        adjacent_difference(zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l), Sum).unwrap();
    let (a_out, b_out, c_out, d_out, e_out, f_out, g_out, h_out, i_out, j_out, k_out, l_out) =
        output;

    assert_eq!(a_out.to_vec().unwrap(), vec![1.0, 4.0, 9.0, 16.0]);
    assert_eq!(b_out.to_vec().unwrap(), vec![10, 40, 90, 160]);
    assert_eq!(c_out.to_vec().unwrap(), vec![2.0, 7.0, 14.0, 23.0]);
    assert_eq!(d_out.to_vec().unwrap(), vec![20, 70, 140, 230]);
    assert_eq!(e_out.to_vec().unwrap(), vec![4.0, 13.0, 24.0, 37.0]);
    assert_eq!(f_out.to_vec().unwrap(), vec![40, 130, 240, 370]);
    assert_eq!(g_out.to_vec().unwrap(), vec![7.0, 18.0, 29.0, 44.0]);
    assert_eq!(h_out.to_vec().unwrap(), vec![70, 180, 290, 440]);
    assert_eq!(i_out.to_vec().unwrap(), vec![8.0, 21.0, 34.0, 55.0]);
    assert_eq!(j_out.to_vec().unwrap(), vec![80, 210, 340, 550]);
    assert_eq!(k_out.to_vec().unwrap(), vec![12.0, 31.0, 50.0, 81.0]);
    assert_eq!(l_out.to_vec().unwrap(), vec![120, 310, 500, 810]);
}

#[test]
fn scan_by_key_accepts_one_component_soa_values() {
    let policy = policy();
    let keys = policy.to_device(&[0_u32, 0, 1, 1, 1, 2]).unwrap();
    let values = policy
        .to_device(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap();

    let inclusive = inclusive_scan_by_key(&keys, &values, EqualU32, Sum).unwrap();
    assert_eq!(
        inclusive.to_vec().unwrap(),
        vec![1.0, 3.0, 3.0, 7.0, 12.0, 6.0]
    );

    let exclusive = exclusive_scan_by_key(&keys, &values, EqualU32, 0.0, Sum).unwrap();
    assert_eq!(
        exclusive.to_vec().unwrap(),
        vec![0.0, 1.0, 0.0, 3.0, 7.0, 0.0]
    );
}

#[test]
fn scan_by_key_accepts_tuple_soa_values() {
    let policy = policy();
    let keys = policy.to_device(&[0_u32, 0, 1]).unwrap();
    let x = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let y = policy.to_device(&[10_u32, 20, 30]).unwrap();

    let inclusive = inclusive_scan_by_key(&keys, zip(&x, &y), EqualU32, Sum).unwrap();
    let (x_out, y_out) = inclusive;
    assert_eq!(x_out.to_vec().unwrap(), vec![1.0, 3.0, 3.0]);
    assert_eq!(y_out.to_vec().unwrap(), vec![10, 30, 30]);

    let exclusive =
        exclusive_scan_by_key(&keys, zip(&x, &y), EqualU32, (0.0, 100_u32), Sum).unwrap();
    let (x_out, y_out) = exclusive;
    assert_eq!(x_out.to_vec().unwrap(), vec![0.0, 1.0, 0.0]);
    assert_eq!(y_out.to_vec().unwrap(), vec![100, 110, 100]);
}

#[test]
fn scan_by_key_accepts_soa12_values() {
    let policy = policy();
    let keys = policy.to_device(&[0_u32, 0, 1, 1, 1, 2]).unwrap();
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

    let inclusive = inclusive_scan_by_key(
        &keys,
        zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
        EqualU32,
        Sum,
    )
    .unwrap();
    let (a_out, b_out, c_out, d_out, e_out, f_out, g_out, h_out, i_out, j_out, k_out, l_out) =
        inclusive;
    assert_eq!(a_out.to_vec().unwrap(), vec![1.0, 3.0, 3.0, 7.0, 12.0, 6.0]);
    assert_eq!(b_out.to_vec().unwrap(), vec![10, 30, 30, 70, 120, 60]);
    assert_eq!(
        c_out.to_vec().unwrap(),
        vec![100.0, 300.0, 300.0, 700.0, 1200.0, 600.0]
    );
    assert_eq!(
        d_out.to_vec().unwrap(),
        vec![1000, 3000, 3000, 7000, 12000, 6000]
    );
    assert_eq!(
        e_out.to_vec().unwrap(),
        vec![7.0, 15.0, 9.0, 19.0, 30.0, 12.0]
    );
    assert_eq!(f_out.to_vec().unwrap(), vec![70, 150, 90, 190, 300, 120]);
    assert_eq!(
        g_out.to_vec().unwrap(),
        vec![700.0, 1500.0, 900.0, 1900.0, 3000.0, 1200.0]
    );
    assert_eq!(
        h_out.to_vec().unwrap(),
        vec![7000, 15000, 9000, 19000, 30000, 12000]
    );
    assert_eq!(
        i_out.to_vec().unwrap(),
        vec![13.0, 27.0, 15.0, 31.0, 48.0, 18.0]
    );
    assert_eq!(j_out.to_vec().unwrap(), vec![130, 270, 150, 310, 480, 180]);
    assert_eq!(
        k_out.to_vec().unwrap(),
        vec![1300.0, 2700.0, 1500.0, 3100.0, 4800.0, 1800.0]
    );
    assert_eq!(
        l_out.to_vec().unwrap(),
        vec![13000, 27000, 15000, 31000, 48000, 18000]
    );

    let exclusive = exclusive_scan_by_key(
        &keys,
        zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
        EqualU32,
        (
            0.0, 0_u32, 0.0, 0_u32, 0.0, 0_u32, 0.0, 0_u32, 0.0, 0_u32, 0.0, 0_u32,
        ),
        Sum,
    )
    .unwrap();
    let (a_out, b_out, c_out, d_out, e_out, f_out, g_out, h_out, i_out, j_out, k_out, l_out) =
        exclusive;
    assert_eq!(a_out.to_vec().unwrap(), vec![0.0, 1.0, 0.0, 3.0, 7.0, 0.0]);
    assert_eq!(b_out.to_vec().unwrap(), vec![0, 10, 0, 30, 70, 0]);
    assert_eq!(
        c_out.to_vec().unwrap(),
        vec![0.0, 100.0, 0.0, 300.0, 700.0, 0.0]
    );
    assert_eq!(d_out.to_vec().unwrap(), vec![0, 1000, 0, 3000, 7000, 0]);
    assert_eq!(e_out.to_vec().unwrap(), vec![0.0, 7.0, 0.0, 9.0, 19.0, 0.0]);
    assert_eq!(f_out.to_vec().unwrap(), vec![0, 70, 0, 90, 190, 0]);
    assert_eq!(
        g_out.to_vec().unwrap(),
        vec![0.0, 700.0, 0.0, 900.0, 1900.0, 0.0]
    );
    assert_eq!(h_out.to_vec().unwrap(), vec![0, 7000, 0, 9000, 19000, 0]);
    assert_eq!(
        i_out.to_vec().unwrap(),
        vec![0.0, 13.0, 0.0, 15.0, 31.0, 0.0]
    );
    assert_eq!(j_out.to_vec().unwrap(), vec![0, 130, 0, 150, 310, 0]);
    assert_eq!(
        k_out.to_vec().unwrap(),
        vec![0.0, 1300.0, 0.0, 1500.0, 3100.0, 0.0]
    );
    assert_eq!(l_out.to_vec().unwrap(), vec![0, 13000, 0, 15000, 31000, 0]);
}
