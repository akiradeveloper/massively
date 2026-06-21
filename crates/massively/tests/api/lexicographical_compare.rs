use crate::common::*;

#[test]
fn lexicographical_compare_accepts_borrowed_tuple_columns() {
    let policy = policy();
    let left_a = policy.to_device(&[1.0_f32, 2.0]).unwrap();
    let left_b = policy.to_device(&[10_u32, 20]).unwrap();
    let right_a = policy.to_device(&[1.0_f32, 2.0]).unwrap();
    let right_b = policy.to_device(&[10_u32, 25]).unwrap();

    assert!(
        lexicographical_compare(
            (left_a.slice(..), left_b.slice(..)),
            (right_a.slice(..), right_b.slice(..)),
            MixedTupleLess
        )
        .unwrap()
    );
    assert!(
        !lexicographical_compare(
            (right_a.slice(..), right_b.slice(..)),
            (left_a.slice(..), left_b.slice(..)),
            MixedTupleLess
        )
        .unwrap()
    );
}

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
            zip12(
                a.slice(..),
                b.slice(..),
                c.slice(..),
                d.slice(..),
                e.slice(..),
                f.slice(..),
                g.slice(..),
                h.slice(..),
                i.slice(..),
                j.slice(..),
                k.slice(..),
                l.slice(..)
            ),
            zip12(
                ra.slice(..),
                rb.slice(..),
                rc.slice(..),
                rd.slice(..),
                re.slice(..),
                rf.slice(..),
                rg.slice(..),
                rh.slice(..),
                ri.slice(..),
                rj.slice(..),
                rk.slice(..),
                rl.slice(..)
            ),
            Tuple12MixedTailLess,
        )
        .unwrap()
    );
    assert!(
        !lexicographical_compare(
            zip12(
                ra.slice(..),
                rb.slice(..),
                rc.slice(..),
                rd.slice(..),
                re.slice(..),
                rf.slice(..),
                rg.slice(..),
                rh.slice(..),
                ri.slice(..),
                rj.slice(..),
                rk.slice(..),
                rl.slice(..)
            ),
            zip12(
                a.slice(..),
                b.slice(..),
                c.slice(..),
                d.slice(..),
                e.slice(..),
                f.slice(..),
                g.slice(..),
                h.slice(..),
                i.slice(..),
                j.slice(..),
                k.slice(..),
                l.slice(..)
            ),
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
                a_short.slice(..),
                b_short.slice(..),
                c_short.slice(..),
                d_short.slice(..),
                e_short.slice(..),
                f_short.slice(..),
                g_short.slice(..),
                h_short.slice(..),
                i_short.slice(..),
                j_short.slice(..),
                k_short.slice(..),
                l_short.slice(..)
            ),
            zip12(
                a.slice(..),
                b.slice(..),
                c.slice(..),
                d.slice(..),
                e.slice(..),
                f.slice(..),
                g.slice(..),
                h.slice(..),
                i.slice(..),
                j.slice(..),
                k.slice(..),
                l.slice(..)
            ),
            Tuple12MixedTailLess,
        )
        .unwrap()
    );
}
