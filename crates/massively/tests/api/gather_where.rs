use crate::common::*;

#[test]
fn gather_where_accepts_soa12_values() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 0.0, 3.0]).unwrap();
    let b = exec.to_device(&[10_u32, 0, 30]).unwrap();
    let indices = exec.to_device(&[2_u32, 1, 0]).unwrap();
    let stencil = exec.to_device(&[1_u32, 0, 1]).unwrap();
    let mut out_a = exec.to_device(&[0.0_f32; 3]).unwrap();
    let mut out_b = exec.to_device(&[0_u32; 3]).unwrap();
    gather_where(
        &exec,
        massively::SoA2(a.slice(..), b.slice(..)),
        indices.slice(..),
        stencil.slice(..),
        massively::SoA2(out_a.slice_mut(..), out_b.slice_mut(..)),
    )
    .unwrap();
    assert_eq!(exec.to_host(&out_a).unwrap(), vec![3.0, 0.0, 1.0]);
    assert_eq!(exec.to_host(&out_b).unwrap(), vec![30, 0, 10]);
}

#[test]
fn gather_where_accepts_u32_stencil() {
    let exec = exec();
    let a = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let b = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let indices = exec.to_device(&[3_u32, 2, 1, 0]).unwrap();
    let stencil = exec.to_device(&[0_u32, 0, 1, 1]).unwrap();

    let mut out_a = exec.to_device(&[0_u32; 4]).unwrap();
    let mut out_b = exec.to_device(&[0.0_f32; 4]).unwrap();
    gather_where(
        &exec,
        massively::SoA2(a.slice(..), b.slice(..)),
        indices.slice(..),
        stencil.slice(..),
        massively::SoA2(out_a.slice_mut(..), out_b.slice_mut(..)),
    )
    .unwrap();
    assert_eq!(exec.to_host(&out_a).unwrap(), vec![0, 0, 20, 10]);
    assert_eq!(exec.to_host(&out_b).unwrap(), vec![0.0, 0.0, 2.0, 1.0]);
}
