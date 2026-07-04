use crate::common::*;

#[test]
fn adjacent_difference_accepts_three_tuple_columns() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 3.0, 6.0, 10.0]).unwrap();
    let b = exec.to_device(&[10_u32, 30, 60, 100]).unwrap();
    let c = exec.to_device(&[2.0_f32, 5.0, 9.0, 14.0]).unwrap();

    let a_out = exec.to_device(&[0.0_f32; 4]).unwrap();
    let b_out = exec.to_device(&[0_u32; 4]).unwrap();
    let c_out = exec.to_device(&[0.0_f32; 4]).unwrap();
    adjacent_difference(
        &exec,
        massively::SoA3(a.slice(..), b.slice(..), c.slice(..)),
        TupleSum,
        massively::SoA3(
            a_out.slice_mut(..),
            b_out.slice_mut(..),
            c_out.slice_mut(..),
        ),
    )
    .unwrap();

    assert_eq!(exec.to_host(&a_out).unwrap(), vec![1.0, 4.0, 9.0, 16.0]);
    assert_eq!(exec.to_host(&b_out).unwrap(), vec![10, 40, 90, 160]);
    assert_eq!(exec.to_host(&c_out).unwrap(), vec![2.0, 7.0, 14.0, 23.0]);
}
