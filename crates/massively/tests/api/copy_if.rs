use crate::common::*;

#[test]
fn copy_if_accepts_u32_flags_for_heterogeneous_tuple_values() {
    let policy = policy();
    let values = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let tags = policy.to_device(&[10_u32, 20, 20, 30]).unwrap();
    let stencil = policy.to_device(&[0_u32, 1, 1, 0]).unwrap();

    let selected = copy_if((values.slice(..), tags.slice(..)), (stencil.slice(..),)).unwrap();
    let (values, tags) = selected;
    assert_eq!(values.to_vec().unwrap(), vec![2.0, 3.0]);
    assert_eq!(tags.to_vec().unwrap(), vec![20, 20]);
}

#[test]
fn copy_if_accepts_three_tuple_columns() {
    let policy = policy();
    let a = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let c = policy.to_device(&[100.0_f32, 200.0, 300.0]).unwrap();
    let stencil = policy.to_device(&[0_u32, 1, 1]).unwrap();

    let selected = copy_if(
        (a.slice(..), b.slice(..), c.slice(..)),
        (stencil.slice(..),),
    )
    .unwrap();
    let (a, b, c) = selected;
    assert_eq!(a.to_vec().unwrap(), vec![2.0, 3.0]);
    assert_eq!(b.to_vec().unwrap(), vec![20, 30]);
    assert_eq!(c.to_vec().unwrap(), vec![200.0, 300.0]);
}

#[test]
fn copy_if_accepts_u32_stencil() {
    let policy = policy();
    let values = policy.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let ids = policy.to_device(&[1_u32, 2, 3, 4]).unwrap();
    let stencil = policy.to_device(&[0_u32, 0, 1, 1]).unwrap();

    let selected = copy_if((values.slice(..), ids.slice(..)), (stencil.slice(..),)).unwrap();
    let (values, ids) = selected;
    assert_eq!(values.to_vec().unwrap(), vec![30, 40]);
    assert_eq!(ids.to_vec().unwrap(), vec![3, 4]);
}
