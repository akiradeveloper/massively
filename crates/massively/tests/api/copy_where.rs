use crate::common::*;

#[test]
fn copy_where_accepts_u32_flags_for_heterogeneous_tuple_values() {
    let exec = exec();
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let tags = exec.to_device(&[10_u32, 20, 20, 30]).unwrap();
    let stencil = exec.to_device(&[0_u32, 1, 1, 0]).unwrap();

    let selected = copy_where(
        &exec,
        massively::SoA2(values.slice(..), tags.slice(..)),
        stencil.slice(..),
    )
    .unwrap();
    let (values, tags) = selected;
    assert_eq!(exec.to_host(&values).unwrap(), vec![2.0, 3.0]);
    assert_eq!(exec.to_host(&tags).unwrap(), vec![20, 20]);
}

#[test]
fn copy_where_accepts_three_tuple_columns() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let c = exec.to_device(&[100.0_f32, 200.0, 300.0]).unwrap();
    let stencil = exec.to_device(&[0_u32, 1, 1]).unwrap();

    let selected = copy_where(
        &exec,
        massively::SoA3(a.slice(..), b.slice(..), c.slice(..)),
        stencil.slice(..),
    )
    .unwrap();
    let (a, b, c) = selected;
    assert_eq!(exec.to_host(&a).unwrap(), vec![2.0, 3.0]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![20, 30]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![200.0, 300.0]);
}

#[test]
fn copy_where_accepts_u32_stencil() {
    let exec = exec();
    let values = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let ids = exec.to_device(&[1_u32, 2, 3, 4]).unwrap();
    let stencil = exec.to_device(&[0_u32, 0, 1, 1]).unwrap();

    let selected = copy_where(
        &exec,
        massively::SoA2(values.slice(..), ids.slice(..)),
        stencil.slice(..),
    )
    .unwrap();
    let (values, ids) = selected;
    assert_eq!(exec.to_host(&values).unwrap(), vec![30, 40]);
    assert_eq!(exec.to_host(&ids).unwrap(), vec![3, 4]);
}

#[test]
fn copy_where_returns_empty_when_no_flags_are_selected() {
    let exec = exec();
    let values = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let stencil = exec.to_device(&[0_u32, 0, 0]).unwrap();

    let (selected,) =
        copy_where(&exec, massively::SoA1(values.slice(..)), stencil.slice(..)).unwrap();

    assert_eq!(exec.to_host(&selected).unwrap(), Vec::<u32>::new());
}

#[test]
fn copy_where_keeps_all_values_when_all_flags_are_selected() {
    let exec = exec();
    let values = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let stencil = exec.to_device(&[1_u32, 1, 1]).unwrap();

    let (selected,) =
        copy_where(&exec, massively::SoA1(values.slice(..)), stencil.slice(..)).unwrap();

    assert_eq!(exec.to_host(&selected).unwrap(), vec![10, 20, 30]);
}
