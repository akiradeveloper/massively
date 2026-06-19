use crate::common::*;

#[test]
fn adjacent_difference_accepts_three_tuple_columns() {
    let policy = policy();
    let a = policy.to_device(&[1.0_f32, 3.0, 6.0, 10.0]).unwrap();
    let b = policy.to_device(&[10_u32, 30, 60, 100]).unwrap();
    let c = policy.to_device(&[2.0_f32, 5.0, 9.0, 14.0]).unwrap();

    let output = adjacent_difference((&a, &b, &c), TupleSum).unwrap();
    let (a_out, b_out, c_out) = output;

    assert_eq!(a_out.to_vec().unwrap(), vec![1.0, 4.0, 9.0, 16.0]);
    assert_eq!(b_out.to_vec().unwrap(), vec![10, 40, 90, 160]);
    assert_eq!(c_out.to_vec().unwrap(), vec![2.0, 7.0, 14.0, 23.0]);
}
