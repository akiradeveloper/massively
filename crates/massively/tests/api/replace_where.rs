use crate::common::*;

#[test]
fn replace_where_accepts_three_tuple_columns() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 20, 30]).unwrap();
    let c = exec.to_device(&[1.0_f32, -1.0, 2.0, 3.0]).unwrap();
    let stencil = exec.to_device(&[0_u32, 1, 1, 0]).unwrap();

    replace_where(
        &exec,
        (99.0_f32, 77_u32, -99.0_f32),
        stencil.slice(..),
        massively::SoA3(a.slice_mut(..), b.slice_mut(..), c.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&a).unwrap(), vec![1.0, 99.0, 99.0, 4.0]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![10, 77, 77, 30]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![1.0, -99.0, -99.0, 3.0]);
}

#[test]
fn replace_where_accepts_u32_stencil() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let stencil = exec.to_device(&[0_u32, 0, 1, 1]).unwrap();

    replace_where(
        &exec,
        (-1.0_f32, 99_u32),
        stencil.slice(..),
        massively::SoA2(a.slice_mut(..), b.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&a).unwrap(), vec![1.0, 2.0, -1.0, -1.0]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![10, 20, 99, 99]);
}

#[test]
fn replace_where_leaves_values_unchanged_when_no_flags_are_selected() {
    let exec = exec();
    let values = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let stencil = exec.to_device(&[0_u32, 0, 0]).unwrap();

    replace_where(
        &exec,
        (99_u32,),
        stencil.slice(..),
        massively::SoA1(values.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&values).unwrap(), vec![10, 20, 30]);
}

#[test]
fn replace_where_replaces_all_values_when_all_flags_are_selected() {
    let exec = exec();
    let values = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let stencil = exec.to_device(&[1_u32, 1, 1]).unwrap();

    replace_where(
        &exec,
        (99_u32,),
        stencil.slice(..),
        massively::SoA1(values.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&values).unwrap(), vec![99, 99, 99]);
}

#[test]
fn replace_where_accepts_sliced_output() {
    let exec = exec();
    let values = exec.to_device(&[1_u32, 10, 20, 30, 5]).unwrap();
    let stencil = exec.to_device(&[0_u32, 1, 1]).unwrap();

    replace_where(
        &exec,
        (99_u32,),
        stencil.slice(..),
        massively::SoA1(values.slice_mut(1..4)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&values).unwrap(), vec![1, 10, 99, 99, 5]);
}
