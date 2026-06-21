use crate::common::*;

#[cfg(any())]
#[test]
fn unique_accepts_borrowed_heterogeneous_soa12() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 1.0, 2.0, 2.0, 3.0]).unwrap();
    let b = exec.to_device(&[10_u32, 10, 20, 20, 30]).unwrap();
    let c = exec
        .to_device(&[100.0_f32, 101.0, 200.0, 201.0, 300.0])
        .unwrap();
    let d = exec.to_device(&[1000_u32, 1001, 2000, 2001, 3000]).unwrap();
    let e = exec.to_device(&[4.0_f32, 4.5, 5.0, 5.5, 6.0]).unwrap();
    let f = exec.to_device(&[40_u32, 41, 50, 51, 60]).unwrap();
    let g = exec
        .to_device(&[400.0_f32, 401.0, 500.0, 501.0, 600.0])
        .unwrap();
    let h = exec.to_device(&[4000_u32, 4001, 5000, 5001, 6000]).unwrap();
    let i = exec.to_device(&[7.0_f32, 7.5, 8.0, 8.5, 9.0]).unwrap();
    let j = exec.to_device(&[70_u32, 71, 80, 81, 90]).unwrap();
    let k = exec
        .to_device(&[700.0_f32, 700.0, 800.0, 800.0, 900.0])
        .unwrap();
    let l = exec.to_device(&[7000_u32, 7000, 8000, 8000, 9000]).unwrap();

    let output = unique(
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
    let (a, b, c, d, e, f, g, h, i, j, k, l) = output;

    assert_eq!(exec.to_host(&a).unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![10, 20, 30]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![100.0, 200.0, 300.0]);
    assert_eq!(exec.to_host(&d).unwrap(), vec![1000, 2000, 3000]);
    assert_eq!(exec.to_host(&e).unwrap(), vec![4.0, 5.0, 6.0]);
    assert_eq!(exec.to_host(&f).unwrap(), vec![40, 50, 60]);
    assert_eq!(exec.to_host(&g).unwrap(), vec![400.0, 500.0, 600.0]);
    assert_eq!(exec.to_host(&h).unwrap(), vec![4000, 5000, 6000]);
    assert_eq!(exec.to_host(&i).unwrap(), vec![7.0, 8.0, 9.0]);
    assert_eq!(exec.to_host(&j).unwrap(), vec![70, 80, 90]);
    assert_eq!(exec.to_host(&k).unwrap(), vec![700.0, 800.0, 900.0]);
    assert_eq!(exec.to_host(&l).unwrap(), vec![7000, 8000, 9000]);
}
