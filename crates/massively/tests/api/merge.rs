use crate::common::*;

#[test]
fn merge_accepts_borrowed_tuple_columns() {
    let exec = exec();
    let left_a = exec.to_device(&[1.0_f32, 3.0]).unwrap();
    let left_b = exec.to_device(&[10_u32, 30]).unwrap();
    let right_a = exec.to_device(&[2.0_f32, 4.0]).unwrap();
    let right_b = exec.to_device(&[20_u32, 40]).unwrap();
    let out_a = exec.to_device(&[0.0_f32; 4]).unwrap();
    let out_b = exec.to_device(&[0_u32; 4]).unwrap();

    merge(
        &exec,
        massively::Zip2(left_a.slice(..), left_b.slice(..)),
        massively::Zip2(right_a.slice(..), right_b.slice(..)),
        MixedTupleLess,
        massively::Zip2(out_a.slice_mut(..), out_b.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_a).unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(exec.to_host(&out_b).unwrap(), vec![10, 20, 30, 40]);
}
