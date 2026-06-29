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
    let massively::SoA2(values, tags) = removed;
    assert_eq!(exec.to_host(&values).unwrap(), vec![1.0, 4.0]);
    assert_eq!(exec.to_host(&tags).unwrap(), vec![10, 30]);
}

#[test]
fn remove_where_accepts_seven_tuple_columns() {
    let exec = exec();
    let a = exec.to_device(&[1_u32, 2, 3, 4, 5]).unwrap();
    let b = exec.to_device(&[11_u32, 12, 13, 14, 15]).unwrap();
    let c = exec.to_device(&[21_u32, 22, 23, 24, 25]).unwrap();
    let d = exec.to_device(&[31_u32, 32, 33, 34, 35]).unwrap();
    let e = exec.to_device(&[41_u32, 42, 43, 44, 45]).unwrap();
    let f = exec.to_device(&[51_u32, 52, 53, 54, 55]).unwrap();
    let g = exec.to_device(&[61_u32, 62, 63, 64, 65]).unwrap();
    let stencil = exec.to_device(&[0_u32, 1, 0, 1, 0]).unwrap();

    let remaining = remove_where(
        &exec,
        massively::SoA7(
            a.slice(..),
            b.slice(..),
            c.slice(..),
            d.slice(..),
            e.slice(..),
            f.slice(..),
            g.slice(..),
        ),
        stencil.slice(..),
    )
    .unwrap();
    let massively::SoA7(a, b, c, d, e, f, g) = remaining;
    assert_eq!(exec.to_host(&a).unwrap(), vec![1, 3, 5]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![11, 13, 15]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![21, 23, 25]);
    assert_eq!(exec.to_host(&d).unwrap(), vec![31, 33, 35]);
    assert_eq!(exec.to_host(&e).unwrap(), vec![41, 43, 45]);
    assert_eq!(exec.to_host(&f).unwrap(), vec![51, 53, 55]);
    assert_eq!(exec.to_host(&g).unwrap(), vec![61, 63, 65]);
}

#[test]
fn remove_where_keeps_all_values_when_no_flags_are_selected() {
    let exec = exec();
    let values = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let stencil = exec.to_device(&[0_u32, 0, 0]).unwrap();

    let massively::SoA1(remaining) =
        remove_where(&exec, massively::SoA1(values.slice(..)), stencil.slice(..)).unwrap();

    assert_eq!(exec.to_host(&remaining).unwrap(), vec![10, 20, 30]);
}

#[test]
fn remove_where_returns_empty_when_all_flags_are_selected() {
    let exec = exec();
    let values = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let stencil = exec.to_device(&[1_u32, 1, 1]).unwrap();

    let massively::SoA1(remaining) =
        remove_where(&exec, massively::SoA1(values.slice(..)), stencil.slice(..)).unwrap();

    assert_eq!(exec.to_host(&remaining).unwrap(), Vec::<u32>::new());
}
