mod common;
use common::*;

#[test]
fn adjacent_find_accepts_borrowed_heterogeneous_soa12() {
    let policy = policy();
    let a = policy.to_device(&[1.0_f32, 2.0, 2.0, 3.0]).unwrap();
    let b = policy.to_device(&[10_u32, 20, 20, 30]).unwrap();
    let c = policy.to_device(&[100.0_f32, 200.0, 201.0, 300.0]).unwrap();
    let d = policy.to_device(&[1000_u32, 2000, 2001, 3000]).unwrap();
    let e = policy.to_device(&[4.0_f32, 5.0, 5.0, 6.0]).unwrap();
    let f = policy.to_device(&[40_u32, 50, 50, 60]).unwrap();
    let g = policy.to_device(&[400.0_f32, 500.0, 500.0, 600.0]).unwrap();
    let h = policy.to_device(&[4000_u32, 5000, 5000, 6000]).unwrap();
    let i = policy.to_device(&[7.0_f32, 8.0, 8.0, 9.0]).unwrap();
    let j = policy.to_device(&[70_u32, 80, 80, 90]).unwrap();
    let k = policy.to_device(&[700.0_f32, 800.0, 800.0, 900.0]).unwrap();
    let l = policy.to_device(&[7000_u32, 8000, 8000, 9000]).unwrap();

    let index = adjacent_find(
        zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
        Tuple12MixedEqual,
    )
    .unwrap();

    assert_eq!(index, Some(1));
}

#[test]
fn equal_and_mismatch_accept_borrowed_heterogeneous_soa12() {
    let policy = policy();
    let a = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let b = policy.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let c = policy.to_device(&[100.0_f32, 200.0, 300.0, 400.0]).unwrap();
    let d = policy.to_device(&[1000_u32, 2000, 3000, 4000]).unwrap();
    let e = policy.to_device(&[4.0_f32, 5.0, 6.0, 7.0]).unwrap();
    let f = policy.to_device(&[40_u32, 50, 60, 70]).unwrap();
    let g = policy.to_device(&[400.0_f32, 500.0, 600.0, 700.0]).unwrap();
    let h = policy.to_device(&[4000_u32, 5000, 6000, 7000]).unwrap();
    let i = policy.to_device(&[7.0_f32, 8.0, 9.0, 10.0]).unwrap();
    let j = policy.to_device(&[70_u32, 80, 90, 100]).unwrap();
    let k = policy
        .to_device(&[700.0_f32, 800.0, 900.0, 1000.0])
        .unwrap();
    let l = policy.to_device(&[7000_u32, 8000, 9000, 10000]).unwrap();

    let a2 = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let b2 = policy.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let c2 = policy.to_device(&[999.0_f32, 999.0, 999.0, 999.0]).unwrap();
    let d2 = policy.to_device(&[9_u32, 9, 9, 9]).unwrap();
    let e2 = policy.to_device(&[9.0_f32, 9.0, 9.0, 9.0]).unwrap();
    let f2 = policy.to_device(&[9_u32, 9, 9, 9]).unwrap();
    let g2 = policy.to_device(&[9.0_f32, 9.0, 9.0, 9.0]).unwrap();
    let h2 = policy.to_device(&[9_u32, 9, 9, 9]).unwrap();
    let i2 = policy.to_device(&[9.0_f32, 9.0, 9.0, 9.0]).unwrap();
    let j2 = policy.to_device(&[9_u32, 9, 9, 9]).unwrap();
    let k2 = policy
        .to_device(&[700.0_f32, 800.0, 900.0, 1000.0])
        .unwrap();
    let l2 = policy.to_device(&[7000_u32, 8000, 9000, 10000]).unwrap();

    assert!(
        equal(
            zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
            zip12(&a2, &b2, &c2, &d2, &e2, &f2, &g2, &h2, &i2, &j2, &k2, &l2),
            Tuple12MixedEqual,
        )
        .unwrap()
    );
    assert_eq!(
        mismatch(
            zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
            zip12(&a2, &b2, &c2, &d2, &e2, &f2, &g2, &h2, &i2, &j2, &k2, &l2),
            Tuple12MixedEqual,
        )
        .unwrap(),
        None
    );

    let k_diff = policy
        .to_device(&[700.0_f32, 800.0, -900.0, 1000.0])
        .unwrap();
    assert_eq!(
        mismatch(
            zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
            zip12(
                &a2, &b2, &c2, &d2, &e2, &f2, &g2, &h2, &i2, &j2, &k_diff, &l2
            ),
            Tuple12MixedEqual,
        )
        .unwrap(),
        Some(2)
    );

    let a_short = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b_short = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let c_short = policy.to_device(&[100.0_f32, 200.0, 300.0]).unwrap();
    let d_short = policy.to_device(&[1000_u32, 2000, 3000]).unwrap();
    let e_short = policy.to_device(&[4.0_f32, 5.0, 6.0]).unwrap();
    let f_short = policy.to_device(&[40_u32, 50, 60]).unwrap();
    let g_short = policy.to_device(&[400.0_f32, 500.0, 600.0]).unwrap();
    let h_short = policy.to_device(&[4000_u32, 5000, 6000]).unwrap();
    let i_short = policy.to_device(&[7.0_f32, 8.0, 9.0]).unwrap();
    let j_short = policy.to_device(&[70_u32, 80, 90]).unwrap();
    let k_short = policy.to_device(&[700.0_f32, 800.0, 900.0]).unwrap();
    let l_short = policy.to_device(&[7000_u32, 8000, 9000]).unwrap();

    assert!(
        !equal(
            zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
            zip12(
                &a_short, &b_short, &c_short, &d_short, &e_short, &f_short, &g_short, &h_short,
                &i_short, &j_short, &k_short, &l_short
            ),
            Tuple12MixedEqual,
        )
        .unwrap()
    );
    assert_eq!(
        mismatch(
            zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
            zip12(
                &a_short, &b_short, &c_short, &d_short, &e_short, &f_short, &g_short, &h_short,
                &i_short, &j_short, &k_short, &l_short
            ),
            Tuple12MixedEqual,
        )
        .unwrap(),
        Some(3)
    );
}

#[test]
fn sorted_queries_accept_borrowed_heterogeneous_soa12() {
    let policy = policy();
    let a = policy.to_device(&[9.0_f32, 1.0, 7.0, 3.0]).unwrap();
    let b = policy.to_device(&[90_u32, 10, 70, 30]).unwrap();
    let c = policy.to_device(&[0.0_f32, 0.0, 0.0, 0.0]).unwrap();
    let d = policy.to_device(&[0_u32, 0, 0, 0]).unwrap();
    let e = policy.to_device(&[0.0_f32, 0.0, 0.0, 0.0]).unwrap();
    let f = policy.to_device(&[0_u32, 0, 0, 0]).unwrap();
    let g = policy.to_device(&[0.0_f32, 0.0, 0.0, 0.0]).unwrap();
    let h = policy.to_device(&[0_u32, 0, 0, 0]).unwrap();
    let i = policy.to_device(&[0.0_f32, 0.0, 0.0, 0.0]).unwrap();
    let j = policy.to_device(&[0_u32, 0, 0, 0]).unwrap();
    let k = policy.to_device(&[4.0_f32, 3.0, 2.0, 1.0]).unwrap();
    let l = policy.to_device(&[10_u32, 20, 30, 40]).unwrap();

    assert!(
        is_sorted(
            zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
            Tuple12MixedTailLess,
        )
        .unwrap()
    );
    assert_eq!(
        is_sorted_until(
            zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
            Tuple12MixedTailLess,
        )
        .unwrap(),
        4
    );

    let l_unsorted = policy.to_device(&[10_u32, 30, 20, 40]).unwrap();
    assert!(
        !is_sorted(
            zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l_unsorted),
            Tuple12MixedTailLess,
        )
        .unwrap()
    );
    assert_eq!(
        is_sorted_until(
            zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l_unsorted),
            Tuple12MixedTailLess,
        )
        .unwrap(),
        2
    );
}

#[test]
fn pair_search_accepts_borrowed_heterogeneous_soa12_patterns() {
    let policy = policy();
    let a = policy.to_device(&[1.0_f32, 2.0, 3.0, 2.0, 3.0]).unwrap();
    let b = policy.to_device(&[10_u32, 20, 30, 20, 30]).unwrap();
    let c = policy
        .to_device(&[100.0_f32, 200.0, 300.0, 400.0, 500.0])
        .unwrap();
    let d = policy.to_device(&[1_u32, 2, 3, 4, 5]).unwrap();
    let e = policy.to_device(&[0.0_f32, 0.0, 0.0, 0.0, 0.0]).unwrap();
    let f = policy.to_device(&[0_u32, 0, 0, 0, 0]).unwrap();
    let g = policy.to_device(&[0.0_f32, 0.0, 0.0, 0.0, 0.0]).unwrap();
    let h = policy.to_device(&[0_u32, 0, 0, 0, 0]).unwrap();
    let i = policy.to_device(&[0.0_f32, 0.0, 0.0, 0.0, 0.0]).unwrap();
    let j = policy.to_device(&[0_u32, 0, 0, 0, 0]).unwrap();
    let k = policy
        .to_device(&[100.0_f32, 200.0, 300.0, 200.0, 300.0])
        .unwrap();
    let l = policy
        .to_device(&[1000_u32, 2000, 3000, 2000, 3000])
        .unwrap();

    let na = policy.to_device(&[9.0_f32, 3.0]).unwrap();
    let nb = policy.to_device(&[90_u32, 30]).unwrap();
    let nc = policy.to_device(&[0.0_f32, 0.0]).unwrap();
    let nd = policy.to_device(&[0_u32, 0]).unwrap();
    let ne = policy.to_device(&[0.0_f32, 0.0]).unwrap();
    let nf = policy.to_device(&[0_u32, 0]).unwrap();
    let ng = policy.to_device(&[0.0_f32, 0.0]).unwrap();
    let nh = policy.to_device(&[0_u32, 0]).unwrap();
    let ni = policy.to_device(&[0.0_f32, 0.0]).unwrap();
    let nj = policy.to_device(&[0_u32, 0]).unwrap();
    let nk = policy.to_device(&[900.0_f32, 300.0]).unwrap();
    let nl = policy.to_device(&[9000_u32, 3000]).unwrap();

    assert_eq!(
        find_first_of(
            zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
            zip12(&na, &nb, &nc, &nd, &ne, &nf, &ng, &nh, &ni, &nj, &nk, &nl),
            Tuple12MixedEqual,
        )
        .unwrap(),
        Some(2)
    );
}

#[test]
fn lexicographical_compare_accepts_borrowed_heterogeneous_soa12() {
    let policy = policy();
    let a = policy.to_device(&[9.0_f32, 9.0, 9.0]).unwrap();
    let b = policy.to_device(&[0_u32, 0, 0]).unwrap();
    let c = policy.to_device(&[0.0_f32, 0.0, 0.0]).unwrap();
    let d = policy.to_device(&[0_u32, 0, 0]).unwrap();
    let e = policy.to_device(&[0.0_f32, 0.0, 0.0]).unwrap();
    let f = policy.to_device(&[0_u32, 0, 0]).unwrap();
    let g = policy.to_device(&[0.0_f32, 0.0, 0.0]).unwrap();
    let h = policy.to_device(&[0_u32, 0, 0]).unwrap();
    let i = policy.to_device(&[0.0_f32, 0.0, 0.0]).unwrap();
    let j = policy.to_device(&[0_u32, 0, 0]).unwrap();
    let k = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let l = policy.to_device(&[10_u32, 20, 30]).unwrap();

    let ra = policy.to_device(&[1.0_f32, 1.0, 1.0]).unwrap();
    let rb = policy.to_device(&[0_u32, 0, 0]).unwrap();
    let rc = policy.to_device(&[0.0_f32, 0.0, 0.0]).unwrap();
    let rd = policy.to_device(&[0_u32, 0, 0]).unwrap();
    let re = policy.to_device(&[0.0_f32, 0.0, 0.0]).unwrap();
    let rf = policy.to_device(&[0_u32, 0, 0]).unwrap();
    let rg = policy.to_device(&[0.0_f32, 0.0, 0.0]).unwrap();
    let rh = policy.to_device(&[0_u32, 0, 0]).unwrap();
    let ri = policy.to_device(&[0.0_f32, 0.0, 0.0]).unwrap();
    let rj = policy.to_device(&[0_u32, 0, 0]).unwrap();
    let rk = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let rl = policy.to_device(&[10_u32, 25, 30]).unwrap();

    assert!(
        lexicographical_compare(
            zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
            zip12(&ra, &rb, &rc, &rd, &re, &rf, &rg, &rh, &ri, &rj, &rk, &rl),
            Tuple12MixedTailLess,
        )
        .unwrap()
    );
    assert!(
        !lexicographical_compare(
            zip12(&ra, &rb, &rc, &rd, &re, &rf, &rg, &rh, &ri, &rj, &rk, &rl),
            zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
            Tuple12MixedTailLess,
        )
        .unwrap()
    );

    let a_short = policy.to_device(&[9.0_f32, 9.0]).unwrap();
    let b_short = policy.to_device(&[0_u32, 0]).unwrap();
    let c_short = policy.to_device(&[0.0_f32, 0.0]).unwrap();
    let d_short = policy.to_device(&[0_u32, 0]).unwrap();
    let e_short = policy.to_device(&[0.0_f32, 0.0]).unwrap();
    let f_short = policy.to_device(&[0_u32, 0]).unwrap();
    let g_short = policy.to_device(&[0.0_f32, 0.0]).unwrap();
    let h_short = policy.to_device(&[0_u32, 0]).unwrap();
    let i_short = policy.to_device(&[0.0_f32, 0.0]).unwrap();
    let j_short = policy.to_device(&[0_u32, 0]).unwrap();
    let k_short = policy.to_device(&[1.0_f32, 2.0]).unwrap();
    let l_short = policy.to_device(&[10_u32, 20]).unwrap();

    assert!(
        lexicographical_compare(
            zip12(
                &a_short, &b_short, &c_short, &d_short, &e_short, &f_short, &g_short, &h_short,
                &i_short, &j_short, &k_short, &l_short
            ),
            zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
            Tuple12MixedTailLess,
        )
        .unwrap()
    );
}

#[test]
fn sorted_value_queries_accept_borrowed_heterogeneous_soa12() {
    let policy = policy();
    let a = policy.to_device(&[9.0_f32, 1.0, 7.0, 3.0]).unwrap();
    let b = policy.to_device(&[90_u32, 10, 70, 30]).unwrap();
    let c = policy.to_device(&[0.0_f32, 0.0, 0.0, 0.0]).unwrap();
    let d = policy.to_device(&[0_u32, 0, 0, 0]).unwrap();
    let e = policy.to_device(&[0.0_f32, 0.0, 0.0, 0.0]).unwrap();
    let f = policy.to_device(&[0_u32, 0, 0, 0]).unwrap();
    let g = policy.to_device(&[0.0_f32, 0.0, 0.0, 0.0]).unwrap();
    let h = policy.to_device(&[0_u32, 0, 0, 0]).unwrap();
    let i = policy.to_device(&[0.0_f32, 0.0, 0.0, 0.0]).unwrap();
    let j = policy.to_device(&[0_u32, 0, 0, 0]).unwrap();
    let k = policy.to_device(&[4.0_f32, 3.0, 2.0, 1.0]).unwrap();
    let l = policy.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let input = zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l);
    let value = (
        0.0_f32, 0_u32, 0.0_f32, 0_u32, 0.0_f32, 0_u32, 0.0_f32, 0_u32, 0.0_f32, 0_u32, 2.0_f32,
        30_u32,
    );

    assert_eq!(lower_bound(input, value, Tuple12MixedTailLess).unwrap(), 2);
    assert_eq!(
        upper_bound(
            zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
            value,
            Tuple12MixedTailLess,
        )
        .unwrap(),
        3
    );
    assert_eq!(
        equal_range(
            zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
            value,
            Tuple12MixedTailLess,
        )
        .unwrap(),
        (2, 3)
    );
}

#[test]
fn minmax_element_accepts_borrowed_heterogeneous_soa12() {
    let policy = policy();
    let a = policy.to_device(&[9.0_f32, 8.0, 7.0, 6.0]).unwrap();
    let b = policy.to_device(&[90_u32, 80, 70, 60]).unwrap();
    let c = policy.to_device(&[0.0_f32, 0.0, 0.0, 0.0]).unwrap();
    let d = policy.to_device(&[0_u32, 0, 0, 0]).unwrap();
    let e = policy.to_device(&[0.0_f32, 0.0, 0.0, 0.0]).unwrap();
    let f = policy.to_device(&[0_u32, 0, 0, 0]).unwrap();
    let g = policy.to_device(&[0.0_f32, 0.0, 0.0, 0.0]).unwrap();
    let h = policy.to_device(&[0_u32, 0, 0, 0]).unwrap();
    let i = policy.to_device(&[0.0_f32, 0.0, 0.0, 0.0]).unwrap();
    let j = policy.to_device(&[0_u32, 0, 0, 0]).unwrap();
    let k = policy.to_device(&[4.0_f32, 1.0, 9.0, 3.0]).unwrap();
    let l = policy.to_device(&[40_u32, 10, 90, 40]).unwrap();

    assert_eq!(
        min_element(
            zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
            Tuple12MixedTailLess,
        )
        .unwrap(),
        Some(1)
    );
    assert_eq!(
        max_element(
            zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
            Tuple12MixedTailLess,
        )
        .unwrap(),
        Some(2)
    );
    assert_eq!(
        minmax_element(
            zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
            Tuple12MixedTailLess,
        )
        .unwrap(),
        Some((1, 2))
    );
}
