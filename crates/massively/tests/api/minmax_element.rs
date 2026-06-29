use crate::common::*;

#[cfg(any())]
#[cfg(any())]
#[test]
fn minmax_element_accepts_borrowed_heterogeneous_soa12() {
    let exec = exec();
    let a = exec.to_device(&[9.0_f32, 8.0, 7.0, 6.0]).unwrap();
    let b = exec.to_device(&[90_u32, 80, 70, 60]).unwrap();
    let c = exec.to_device(&[0.0_f32, 0.0, 0.0, 0.0]).unwrap();
    let d = exec.to_device(&[0_u32, 0, 0, 0]).unwrap();
    let e = exec.to_device(&[0.0_f32, 0.0, 0.0, 0.0]).unwrap();
    let f = exec.to_device(&[0_u32, 0, 0, 0]).unwrap();
    let g = exec.to_device(&[0.0_f32, 0.0, 0.0, 0.0]).unwrap();
    let h = exec.to_device(&[0_u32, 0, 0, 0]).unwrap();
    let i = exec.to_device(&[0.0_f32, 0.0, 0.0, 0.0]).unwrap();
    let j = exec.to_device(&[0_u32, 0, 0, 0]).unwrap();
    let k = exec.to_device(&[4.0_f32, 1.0, 9.0, 3.0]).unwrap();
    let l = exec.to_device(&[40_u32, 10, 90, 40]).unwrap();

    assert_eq!(
        min_element(
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
            Tuple12MixedTailLess,
        )
        .unwrap(),
        Some(1)
    );
    assert_eq!(
        max_element(
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
            Tuple12MixedTailLess,
        )
        .unwrap(),
        Some(2)
    );
    assert_eq!(
        minmax_element(
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
            Tuple12MixedTailLess,
        )
        .unwrap(),
        Some((1, 2))
    );
}
