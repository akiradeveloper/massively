use crate::common::*;

#[test]
fn merge_accepts_borrowed_tuple_columns() {
    let exec = exec();
    let left_a = exec.to_device(&[1.0_f32, 3.0]).unwrap();
    let left_b = exec.to_device(&[10_u32, 30]).unwrap();
    let right_a = exec.to_device(&[2.0_f32, 4.0]).unwrap();
    let right_b = exec.to_device(&[20_u32, 40]).unwrap();

    let output = merge(
        &exec,
        (left_a.slice(..), left_b.slice(..)),
        (right_a.slice(..), right_b.slice(..)),
        MixedTupleLess,
    )
    .unwrap();
    let (a, b) = output;

    assert_eq!(exec.to_host(&a).unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![10, 20, 30, 40]);
}
