use crate::common::*;

#[test]
fn merge_accepts_borrowed_tuple_columns() {
    let policy = policy();
    let left_a = policy.to_device(&[1.0_f32, 3.0]).unwrap();
    let left_b = policy.to_device(&[10_u32, 30]).unwrap();
    let right_a = policy.to_device(&[2.0_f32, 4.0]).unwrap();
    let right_b = policy.to_device(&[20_u32, 40]).unwrap();

    let output = merge((&left_a, &left_b), (&right_a, &right_b), MixedTupleLess).unwrap();
    let (a, b) = output;

    assert_eq!(a.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(b.to_vec().unwrap(), vec![10, 20, 30, 40]);
}
