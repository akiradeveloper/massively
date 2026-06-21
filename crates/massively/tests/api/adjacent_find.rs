use crate::common::*;

#[cfg(any())]
#[test]
fn adjacent_find_accepts_borrowed_heterogeneous_soa12() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0, 2.0, 3.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 20, 30]).unwrap();
    let c = exec.to_device(&[100.0_f32, 200.0, 201.0, 300.0]).unwrap();
    let d = exec.to_device(&[1000_u32, 2000, 2001, 3000]).unwrap();
    let e = exec.to_device(&[4.0_f32, 5.0, 5.0, 6.0]).unwrap();
    let f = exec.to_device(&[40_u32, 50, 50, 60]).unwrap();
    let g = exec.to_device(&[400.0_f32, 500.0, 500.0, 600.0]).unwrap();
    let h = exec.to_device(&[4000_u32, 5000, 5000, 6000]).unwrap();
    let i = exec.to_device(&[7.0_f32, 8.0, 8.0, 9.0]).unwrap();
    let j = exec.to_device(&[70_u32, 80, 80, 90]).unwrap();
    let k = exec.to_device(&[700.0_f32, 800.0, 800.0, 900.0]).unwrap();
    let l = exec.to_device(&[7000_u32, 8000, 8000, 9000]).unwrap();

    let index = adjacent_find(
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
            l.slice(..),
        ),
        Tuple12MixedEqual,
    )
    .unwrap();

    assert_eq!(index, Some(1));
}
