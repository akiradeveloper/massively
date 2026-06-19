use crate::common::*;

#[test]
fn device_vec_views_as_one_component_miter() {
    let policy = policy();
    let input = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let soa = input;

    let output = soa;
    assert_eq!(output.to_vec().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn device_vec_is_soa1_without_zip() {
    let policy = policy();
    let input = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();

    let output = input;

    assert_eq!(output.to_vec().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn tuple_flattens_single_column_inputs() {
    let policy = policy();
    let left = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let right = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let indices = policy.to_device(&[0_u32, 1, 2]).unwrap();

    let (left, right) = gather((&left, &right), &indices).unwrap();

    assert_eq!(left.to_vec().unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(right.to_vec().unwrap(), vec![10, 20, 30]);
}

#[test]
fn tuple_materializes_heterogeneous_columns() {
    let policy = policy();
    let values = policy.to_device(&[1.5_f32, 2.5, 3.5]).unwrap();
    let ids = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let indices = policy.to_device(&[0_u32, 1, 2]).unwrap();

    let (values, ids) = gather((&values, &ids), &indices).unwrap();

    assert_eq!(values.to_vec().unwrap(), vec![1.5, 2.5, 3.5]);
    assert_eq!(ids.to_vec().unwrap(), vec![10, 20, 30]);
}

#[test]
fn tuple_gather_accepts_borrowed_heterogeneous_columns() {
    let policy = policy();
    let values = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let ids = policy.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let indices = policy.to_device(&[3_u32, 1, 0]).unwrap();

    let gathered = gather((&values, &ids), &indices).unwrap();
    let (values, ids) = gathered;

    assert_eq!(values.to_vec().unwrap(), vec![4.0, 2.0, 1.0]);
    assert_eq!(ids.to_vec().unwrap(), vec![40, 20, 10]);
}

#[test]
fn tuple_gather_accepts_heterogeneous_columns() {
    let policy = policy();
    let values = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let ids = policy.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let indices = policy.to_device(&[3_u32, 1, 0]).unwrap();

    let gathered = gather((&values, &ids), &indices).unwrap();
    let (values, ids) = gathered;

    assert_eq!(values.to_vec().unwrap(), vec![4.0, 2.0, 1.0]);
    assert_eq!(ids.to_vec().unwrap(), vec![40, 20, 10]);
}

#[test]
fn tuple_concatenates_borrowed_columns() {
    let policy = policy();
    let a = policy.to_device(&[1.0_f32, 2.0]).unwrap();
    let b = policy.to_device(&[10_u32, 20]).unwrap();
    let c = policy.to_device(&[100.0_f32, 200.0]).unwrap();

    let indices = policy.to_device(&[0_u32, 1]).unwrap();
    let (a, b, c) = gather((&a, &b, &c), &indices).unwrap();

    assert_eq!(a.to_vec().unwrap(), vec![1.0, 2.0]);
    assert_eq!(b.to_vec().unwrap(), vec![10, 20]);
    assert_eq!(c.to_vec().unwrap(), vec![100.0, 200.0]);
}

#[test]
fn tuple_concatenates_column_and_columns() {
    let policy = policy();
    let a = policy.to_device(&[1.0_f32, 2.0]).unwrap();
    let b = policy.to_device(&[10_u32, 20]).unwrap();
    let c = policy.to_device(&[100.0_f32, 200.0]).unwrap();

    let indices = policy.to_device(&[0_u32, 1]).unwrap();
    let (a, b, c) = gather((&a, &b, &c), &indices).unwrap();

    assert_eq!(a.to_vec().unwrap(), vec![1.0, 2.0]);
    assert_eq!(b.to_vec().unwrap(), vec![10, 20]);
    assert_eq!(c.to_vec().unwrap(), vec![100.0, 200.0]);
}
