use crate::common::*;

#[test]
fn scatter_where_accepts_soa12_values() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 0.0, 3.0]).unwrap();
    let b = exec.to_device(&[10_u32, 0, 30]).unwrap();
    let indices = exec.to_device(&[2_u32, 1, 0]).unwrap();
    let stencil = exec.to_device(&[1_u32, 0, 1]).unwrap();
    let out_a = exec.to_device(&[0.0_f32; 3]).unwrap();
    let out_b = exec.to_device(&[0_u32; 3]).unwrap();
    scatter_where(
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
fn scatter_where_accepts_u32_stencil() {
    let exec = exec();
    let a = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let b = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let indices = exec.to_device(&[3_u32, 2, 1, 0]).unwrap();
    let stencil = exec.to_device(&[0_u32, 0, 1, 1]).unwrap();

    let out_a = exec.to_device(&[0_u32; 4]).unwrap();
    let out_b = exec.to_device(&[0.0_f32; 4]).unwrap();
    scatter_where(
        &exec,
        massively::SoA2(a.slice(..), b.slice(..)),
        indices.slice(..),
        stencil.slice(..),
        massively::SoA2(out_a.slice_mut(..), out_b.slice_mut(..)),
    )
    .unwrap();
    assert_eq!(exec.to_host(&out_a).unwrap(), vec![40, 30, 0, 0]);
    assert_eq!(exec.to_host(&out_b).unwrap(), vec![4.0, 3.0, 0.0, 0.0]);
}
