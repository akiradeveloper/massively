use crate::common::*;

#[test]
fn device_vec_views_as_one_component_miter() {
    let exec = exec();
    let input = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let soa = input;

    let output = soa;
    assert_eq!(exec.to_host(&output).unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn device_vec_is_soa1_without_zip() {
    let exec = exec();
    let input = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();

    let output = input;

    assert_eq!(exec.to_host(&output).unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn tuple_flattens_single_column_inputs() {
    let exec = exec();
    let left = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let right = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let indices = exec.to_device(&[0_u32, 1, 2]).unwrap();

    let (left, right) = gather(
        &exec,
        massively::SoA2(left.slice(..), right.slice(..)),
        indices.slice(..),
    )
    .unwrap();

    assert_eq!(exec.to_host(&left).unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(exec.to_host(&right).unwrap(), vec![10, 20, 30]);
}

#[test]
fn tuple_materializes_heterogeneous_columns() {
    let exec = exec();
    let values = exec.to_device(&[1.5_f32, 2.5, 3.5]).unwrap();
    let ids = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let indices = exec.to_device(&[0_u32, 1, 2]).unwrap();

    let (values, ids) = gather(
        &exec,
        massively::SoA2(values.slice(..), ids.slice(..)),
        indices.slice(..),
    )
    .unwrap();

    assert_eq!(exec.to_host(&values).unwrap(), vec![1.5, 2.5, 3.5]);
    assert_eq!(exec.to_host(&ids).unwrap(), vec![10, 20, 30]);
}

#[test]
fn tuple_gather_accepts_borrowed_heterogeneous_columns() {
    let exec = exec();
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let ids = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let indices = exec.to_device(&[3_u32, 1, 0]).unwrap();

    let gathered = gather(
        &exec,
        massively::SoA2(values.slice(..), ids.slice(..)),
        indices.slice(..),
    )
    .unwrap();
    let (values, ids) = gathered;

    assert_eq!(exec.to_host(&values).unwrap(), vec![4.0, 2.0, 1.0]);
    assert_eq!(exec.to_host(&ids).unwrap(), vec![40, 20, 10]);
}

#[test]
fn tuple_gather_accepts_heterogeneous_columns() {
    let exec = exec();
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let ids = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let indices = exec.to_device(&[3_u32, 1, 0]).unwrap();

    let gathered = gather(
        &exec,
        massively::SoA2(values.slice(..), ids.slice(..)),
        indices.slice(..),
    )
    .unwrap();
    let (values, ids) = gathered;

    assert_eq!(exec.to_host(&values).unwrap(), vec![4.0, 2.0, 1.0]);
    assert_eq!(exec.to_host(&ids).unwrap(), vec![40, 20, 10]);
}

#[test]
fn tuple_concatenates_borrowed_columns() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20]).unwrap();
    let c = exec.to_device(&[100.0_f32, 200.0]).unwrap();

    let indices = exec.to_device(&[0_u32, 1]).unwrap();
    let (a, b, c) = gather(
        &exec,
        massively::SoA3(a.slice(..), b.slice(..), c.slice(..)),
        indices.slice(..),
    )
    .unwrap();

    assert_eq!(exec.to_host(&a).unwrap(), vec![1.0, 2.0]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![10, 20]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![100.0, 200.0]);
}

#[test]
fn tuple_concatenates_column_and_columns() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20]).unwrap();
    let c = exec.to_device(&[100.0_f32, 200.0]).unwrap();

    let indices = exec.to_device(&[0_u32, 1]).unwrap();
    let (a, b, c) = gather(
        &exec,
        massively::SoA3(a.slice(..), b.slice(..), c.slice(..)),
        indices.slice(..),
    )
    .unwrap();

    assert_eq!(exec.to_host(&a).unwrap(), vec![1.0, 2.0]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![10, 20]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![100.0, 200.0]);
}
