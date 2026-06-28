use crate::common::*;

#[test]
fn remove_where_accepts_heterogeneous_tuple_stencil() {
    let exec = exec();
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let tags = exec.to_device(&[10_u32, 20, 20, 30]).unwrap();
    let stencil = exec.to_device(&[0_u32, 1, 1, 0]).unwrap();

    let removed = remove_where(
        &exec,
        massively::SoA2(values.slice(..), tags.slice(..)),
        stencil.slice(..),
    )
    .unwrap();
    let (values, tags) = removed;
    assert_eq!(exec.to_host(&values).unwrap(), vec![1.0, 4.0]);
    assert_eq!(exec.to_host(&tags).unwrap(), vec![10, 30]);
}

#[test]
fn remove_where_keeps_all_values_when_no_flags_are_selected() {
    let exec = exec();
    let values = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let stencil = exec.to_device(&[0_u32, 0, 0]).unwrap();

    let (remaining,) =
        remove_where(&exec, massively::SoA1(values.slice(..)), stencil.slice(..)).unwrap();

    assert_eq!(exec.to_host(&remaining).unwrap(), vec![10, 20, 30]);
}

#[test]
fn remove_where_returns_empty_when_all_flags_are_selected() {
    let exec = exec();
    let values = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let stencil = exec.to_device(&[1_u32, 1, 1]).unwrap();

    let (remaining,) =
        remove_where(&exec, massively::SoA1(values.slice(..)), stencil.slice(..)).unwrap();

    assert_eq!(exec.to_host(&remaining).unwrap(), Vec::<u32>::new());
}
