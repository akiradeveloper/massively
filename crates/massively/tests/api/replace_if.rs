use crate::common::*;

#[test]
fn replace_if_accepts_three_tuple_columns() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 20, 30]).unwrap();
    let c = exec.to_device(&[1.0_f32, -1.0, 2.0, 3.0]).unwrap();
    let stencil = exec.to_device(&[0_u32, 1, 1, 0]).unwrap();

    let output = replace_if(
        &exec,
        massively::SoA3(a.slice(..), b.slice(..), c.slice(..)),
        (99.0_f32, 77_u32, -99.0_f32),
        stencil.slice(..),
    )
    .unwrap();
    let (a, b, c) = output;

    assert_eq!(exec.to_host(&a).unwrap(), vec![1.0, 99.0, 99.0, 4.0]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![10, 77, 77, 30]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![1.0, -99.0, -99.0, 3.0]);
}

#[test]
fn replace_if_accepts_u32_stencil() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let stencil = exec.to_device(&[0_u32, 0, 1, 1]).unwrap();

    let output = replace_if(
        &exec,
        massively::SoA2(a.slice(..), b.slice(..)),
        (-1.0_f32, 99_u32),
        stencil.slice(..),
    )
    .unwrap();
    let (a, b) = output;

    assert_eq!(exec.to_host(&a).unwrap(), vec![1.0, 2.0, -1.0, -1.0]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![10, 20, 99, 99]);
}
