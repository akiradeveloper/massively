use crate::common::*;

#[test]
fn inclusive_scan_accepts_tuple_columns() {
    let policy = policy();
    let a = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let output = inclusive_scan((&a, &b), TupleSum).unwrap();
    let (a, b) = output;
    assert_eq!(a.to_vec().unwrap(), vec![1.0, 3.0, 6.0]);
    assert_eq!(b.to_vec().unwrap(), vec![10, 30, 60]);
}

#[test]
fn inclusive_scan_accepts_single_column_as_tuple_item() {
    let policy = policy();
    let a = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();

    let (a,) = inclusive_scan((&a,), TupleSum).unwrap();

    assert_eq!(a.to_vec().unwrap(), vec![1.0, 3.0, 6.0]);
}

#[test]
fn inclusive_scan_accepts_three_column_tuple_item_op() {
    let policy = policy();
    let a = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let c = policy.to_device(&[100.0_f32, 200.0, 300.0]).unwrap();

    let (a, b, c) = inclusive_scan((&a, &b, &c), TupleSum).unwrap();

    assert_eq!(a.to_vec().unwrap(), vec![1.0, 3.0, 6.0]);
    assert_eq!(b.to_vec().unwrap(), vec![10, 30, 60]);
    assert_eq!(c.to_vec().unwrap(), vec![100.0, 300.0, 600.0]);
}
