use crate::common::*;

#[test]
fn scatter_accepts_heterogeneous_columns() {
    let exec = exec();
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let ids = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let indices = exec.to_device(&[2_u32, 0, 3]).unwrap();

    let out_values = exec.to_device(&[0.0_f32; 4]).unwrap();
    let out_ids = exec.to_device(&[0_u32; 4]).unwrap();
    scatter(
        &exec,
        massively::SoA2(values.slice(..), ids.slice(..)),
        indices.slice(..),
        massively::SoA2(out_values.slice_mut(..), out_ids.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_values).unwrap(), vec![2.0, 0.0, 1.0, 3.0]);
    assert_eq!(exec.to_host(&out_ids).unwrap(), vec![20, 0, 10, 30]);
}

#[cfg(any())]
#[test]
fn scatter_accepts_soa12_values() {
    let exec = exec();
    let indices = exec.to_device(&[2_u32, 1, 0]).unwrap();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let c = exec.to_device(&[100.0_f32, 200.0, 300.0]).unwrap();
    let d = exec.to_device(&[1000_u32, 2000, 3000]).unwrap();
    let e = exec.to_device(&[4.0_f32, 5.0, 6.0]).unwrap();
    let f = exec.to_device(&[40_u32, 50, 60]).unwrap();
    let g = exec.to_device(&[400.0_f32, 500.0, 600.0]).unwrap();
    let h = exec.to_device(&[4000_u32, 5000, 6000]).unwrap();
    let i = exec.to_device(&[7.0_f32, 8.0, 9.0]).unwrap();
    let j = exec.to_device(&[70_u32, 80, 90]).unwrap();
    let k = exec.to_device(&[700.0_f32, 800.0, 900.0]).unwrap();
    let l = exec.to_device(&[7000_u32, 8000, 9000]).unwrap();

    let scattered = scatter(
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
        indices.slice(..),
        3,
        (
            0.0_f32, 0_u32, 0.0_f32, 0_u32, 0.0_f32, 0_u32, 0.0_f32, 0_u32, 0.0_f32, 0_u32,
            0.0_f32, 0_u32,
        ),
    )
    .unwrap();
    let (a_out, b_out, c_out, d_out, e_out, f_out, g_out, h_out, i_out, j_out, k_out, l_out) =
        scattered;
    assert_eq!(exec.to_host(&a_out).unwrap(), vec![3.0, 2.0, 1.0]);
    assert_eq!(exec.to_host(&b_out).unwrap(), vec![30, 20, 10]);
    assert_eq!(exec.to_host(&c_out).unwrap(), vec![300.0, 200.0, 100.0]);
    assert_eq!(exec.to_host(&d_out).unwrap(), vec![3000, 2000, 1000]);
    assert_eq!(exec.to_host(&e_out).unwrap(), vec![6.0, 5.0, 4.0]);
    assert_eq!(exec.to_host(&f_out).unwrap(), vec![60, 50, 40]);
    assert_eq!(exec.to_host(&g_out).unwrap(), vec![600.0, 500.0, 400.0]);
    assert_eq!(exec.to_host(&h_out).unwrap(), vec![6000, 5000, 4000]);
    assert_eq!(exec.to_host(&i_out).unwrap(), vec![9.0, 8.0, 7.0]);
    assert_eq!(exec.to_host(&j_out).unwrap(), vec![90, 80, 70]);
    assert_eq!(exec.to_host(&k_out).unwrap(), vec![900.0, 800.0, 700.0]);
    assert_eq!(exec.to_host(&l_out).unwrap(), vec![9000, 8000, 7000]);
}
