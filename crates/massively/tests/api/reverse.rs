use crate::common::*;

#[test]
fn reverse_accepts_borrowed_tuple_columns() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let out_a = exec.to_device(&[0.0_f32; 3]).unwrap();
    let out_b = exec.to_device(&[0_u32; 3]).unwrap();

    reverse(
        &exec,
        massively::Zip2(a.slice(..), b.slice(..)),
        massively::Zip2(out_a.slice_mut(..), out_b.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_a).unwrap(), vec![3.0, 2.0, 1.0]);
    assert_eq!(exec.to_host(&out_b).unwrap(), vec![30, 20, 10]);
}

#[test]
fn reverse_accepts_borrowed_three_tuple_columns() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let c = exec.to_device(&[100.0_f32, 200.0, 300.0]).unwrap();
    let out_a = exec.to_device(&[0.0_f32; 3]).unwrap();
    let out_b = exec.to_device(&[0_u32; 3]).unwrap();
    let out_c = exec.to_device(&[0.0_f32; 3]).unwrap();

    reverse(
        &exec,
        massively::Zip3(a.slice(..), b.slice(..), c.slice(..)),
        massively::Zip3(
            out_a.slice_mut(..),
            out_b.slice_mut(..),
            out_c.slice_mut(..),
        ),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_a).unwrap(), vec![3.0, 2.0, 1.0]);
    assert_eq!(exec.to_host(&out_b).unwrap(), vec![30, 20, 10]);
    assert_eq!(exec.to_host(&out_c).unwrap(), vec![300.0, 200.0, 100.0]);
}
