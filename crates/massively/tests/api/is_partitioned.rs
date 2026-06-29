use crate::common::*;

#[cfg(any())]
#[cfg(any())]
#[test]
fn is_partitioned_accepts_borrowed_heterogeneous_soa12() {
    let exec = exec();
    let a = exec.to_device(&[3.0_f32, 4.0, 2.0, 1.0, 0.0]).unwrap();
    let b = exec.to_device(&[30_u32, 40, 20, 10, 0]).unwrap();
    let c = exec
        .to_device(&[300.0_f32, 400.0, 200.0, 100.0, 0.0])
        .unwrap();
    let d = exec.to_device(&[3000_u32, 4000, 2000, 1000, 0]).unwrap();
    let e = exec.to_device(&[3.5_f32, 4.5, 2.5, 1.5, 0.5]).unwrap();
    let f = exec.to_device(&[35_u32, 45, 25, 15, 5]).unwrap();
    let g = exec
        .to_device(&[350.0_f32, 450.0, 250.0, 150.0, 50.0])
        .unwrap();
    let h = exec.to_device(&[3500_u32, 4500, 2500, 1500, 500]).unwrap();
    let i = exec.to_device(&[6.0_f32, 8.0, 4.0, 2.0, 0.0]).unwrap();
    let j = exec.to_device(&[60_u32, 80, 40, 20, 0]).unwrap();
    let k = exec
        .to_device(&[600.0_f32, 800.0, 400.0, 200.0, 0.0])
        .unwrap();
    let l = exec.to_device(&[6000_u32, 8000, 4000, 2000, 0]).unwrap();

    let input = zip12(
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
    );
    assert!(is_partitioned(&exec, input, Tuple12MixedFirstGreaterThanOne).unwrap());
}
