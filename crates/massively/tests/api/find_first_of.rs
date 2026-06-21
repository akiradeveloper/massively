use crate::common::*;

#[test]
fn find_first_of_accepts_borrowed_tuple_columns() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let needle_a = exec.to_device(&[9.0_f32, 3.0]).unwrap();
    let needle_b = exec.to_device(&[90_u32, 30]).unwrap();

    assert_eq!(
        find_first_of(
            &exec,
            (a.slice(..), b.slice(..)),
            (needle_a.slice(..), needle_b.slice(..)),
            MixedTupleEqual
        )
        .unwrap(),
        Some(2)
    );
}

#[cfg(any())]
#[test]
fn pair_search_accepts_borrowed_heterogeneous_soa12_patterns() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0, 2.0, 3.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 30, 20, 30]).unwrap();
    let c = exec
        .to_device(&[100.0_f32, 200.0, 300.0, 400.0, 500.0])
        .unwrap();
    let d = exec.to_device(&[1_u32, 2, 3, 4, 5]).unwrap();
    let e = exec.to_device(&[0.0_f32, 0.0, 0.0, 0.0, 0.0]).unwrap();
    let f = exec.to_device(&[0_u32, 0, 0, 0, 0]).unwrap();
    let g = exec.to_device(&[0.0_f32, 0.0, 0.0, 0.0, 0.0]).unwrap();
    let h = exec.to_device(&[0_u32, 0, 0, 0, 0]).unwrap();
    let i = exec.to_device(&[0.0_f32, 0.0, 0.0, 0.0, 0.0]).unwrap();
    let j = exec.to_device(&[0_u32, 0, 0, 0, 0]).unwrap();
    let k = exec
        .to_device(&[100.0_f32, 200.0, 300.0, 200.0, 300.0])
        .unwrap();
    let l = exec.to_device(&[1000_u32, 2000, 3000, 2000, 3000]).unwrap();

    let na = exec.to_device(&[9.0_f32, 3.0]).unwrap();
    let nb = exec.to_device(&[90_u32, 30]).unwrap();
    let nc = exec.to_device(&[0.0_f32, 0.0]).unwrap();
    let nd = exec.to_device(&[0_u32, 0]).unwrap();
    let ne = exec.to_device(&[0.0_f32, 0.0]).unwrap();
    let nf = exec.to_device(&[0_u32, 0]).unwrap();
    let ng = exec.to_device(&[0.0_f32, 0.0]).unwrap();
    let nh = exec.to_device(&[0_u32, 0]).unwrap();
    let ni = exec.to_device(&[0.0_f32, 0.0]).unwrap();
    let nj = exec.to_device(&[0_u32, 0]).unwrap();
    let nk = exec.to_device(&[900.0_f32, 300.0]).unwrap();
    let nl = exec.to_device(&[9000_u32, 3000]).unwrap();

    assert_eq!(
        find_first_of(
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
                na.slice(..),
                nb.slice(..),
                nc.slice(..),
                nd.slice(..),
                ne.slice(..),
                nf.slice(..),
                ng.slice(..),
                nh.slice(..),
                ni.slice(..),
                nj.slice(..),
                nk.slice(..),
                nl.slice(..)
            ),
            Tuple12MixedEqual,
        )
        .unwrap(),
        Some(2)
    );
}
