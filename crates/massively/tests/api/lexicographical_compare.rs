use crate::common::*;

#[cfg(any())]
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
