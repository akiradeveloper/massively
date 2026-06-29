use crate::common::*;

#[test]
fn lexicographical_compare_accepts_borrowed_tuple_columns() {
    let exec = exec();
    let left_a = exec.to_device(&[1.0_f32, 2.0]).unwrap();
    let left_b = exec.to_device(&[10_u32, 20]).unwrap();
    let right_a = exec.to_device(&[1.0_f32, 2.0]).unwrap();
    let right_b = exec.to_device(&[10_u32, 25]).unwrap();

    assert!(
        lexicographical_compare(
            &exec,
            massively::SoA2(left_a.slice(..), left_b.slice(..)),
            massively::SoA2(right_a.slice(..), right_b.slice(..)),
            MixedTupleLess
        )
        .unwrap()
    );
    assert!(
        !lexicographical_compare(
            &exec,
            massively::SoA2(right_a.slice(..), right_b.slice(..)),
            massively::SoA2(left_a.slice(..), left_b.slice(..)),
            MixedTupleLess
        )
        .unwrap()
    );
}

#[cfg(any())]
#[cfg(any())]
#[test]
fn lexicographical_compare_accepts_borrowed_heterogeneous_soa12() {
    let exec = exec();
    let a = exec.to_device(&[9.0_f32, 9.0, 9.0]).unwrap();
    let b = exec.to_device(&[0_u32, 0, 0]).unwrap();
    let c = exec.to_device(&[0.0_f32, 0.0, 0.0]).unwrap();
    let d = exec.to_device(&[0_u32, 0, 0]).unwrap();
    let e = exec.to_device(&[0.0_f32, 0.0, 0.0]).unwrap();
    let f = exec.to_device(&[0_u32, 0, 0]).unwrap();
    let g = exec.to_device(&[0.0_f32, 0.0, 0.0]).unwrap();
    let h = exec.to_device(&[0_u32, 0, 0]).unwrap();
    let i = exec.to_device(&[0.0_f32, 0.0, 0.0]).unwrap();
    let j = exec.to_device(&[0_u32, 0, 0]).unwrap();
    let k = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let l = exec.to_device(&[10_u32, 20, 30]).unwrap();

    let ra = exec.to_device(&[1.0_f32, 1.0, 1.0]).unwrap();
    let rb = exec.to_device(&[0_u32, 0, 0]).unwrap();
    let rc = exec.to_device(&[0.0_f32, 0.0, 0.0]).unwrap();
    let rd = exec.to_device(&[0_u32, 0, 0]).unwrap();
    let re = exec.to_device(&[0.0_f32, 0.0, 0.0]).unwrap();
    let rf = exec.to_device(&[0_u32, 0, 0]).unwrap();
    let rg = exec.to_device(&[0.0_f32, 0.0, 0.0]).unwrap();
    let rh = exec.to_device(&[0_u32, 0, 0]).unwrap();
    let ri = exec.to_device(&[0.0_f32, 0.0, 0.0]).unwrap();
    let rj = exec.to_device(&[0_u32, 0, 0]).unwrap();
    let rk = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let rl = exec.to_device(&[10_u32, 25, 30]).unwrap();

    assert!(
        lexicographical_compare(
            &exec,
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
            &exec,
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

    let a_short = exec.to_device(&[9.0_f32, 9.0]).unwrap();
    let b_short = exec.to_device(&[0_u32, 0]).unwrap();
    let c_short = exec.to_device(&[0.0_f32, 0.0]).unwrap();
    let d_short = exec.to_device(&[0_u32, 0]).unwrap();
    let e_short = exec.to_device(&[0.0_f32, 0.0]).unwrap();
    let f_short = exec.to_device(&[0_u32, 0]).unwrap();
    let g_short = exec.to_device(&[0.0_f32, 0.0]).unwrap();
    let h_short = exec.to_device(&[0_u32, 0]).unwrap();
    let i_short = exec.to_device(&[0.0_f32, 0.0]).unwrap();
    let j_short = exec.to_device(&[0_u32, 0]).unwrap();
    let k_short = exec.to_device(&[1.0_f32, 2.0]).unwrap();
    let l_short = exec.to_device(&[10_u32, 20]).unwrap();

    assert!(
        lexicographical_compare(
            &exec,
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
