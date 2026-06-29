use crate::common::*;

#[test]
fn fill_accepts_single_column() {
    let exec = exec();
    let values = exec.to_device(&[1_u32, 2, 3, 4]).unwrap();

    fill(&exec, (99_u32,), massively::SoA1(values.slice_mut(..))).unwrap();

    assert_eq!(exec.to_host(&values).unwrap(), vec![99, 99, 99, 99]);
}

#[test]
fn fill_accepts_tuple_columns() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let c = exec.to_device(&[-1.0_f32, -2.0, -3.0]).unwrap();

    fill(
        &exec,
        (9.0_f32, 99_u32, -9.0_f32),
        massively::SoA3(a.slice_mut(..), b.slice_mut(..), c.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&a).unwrap(), vec![9.0, 9.0, 9.0]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![99, 99, 99]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![-9.0, -9.0, -9.0]);
}

#[test]
fn fill_accepts_sliced_output() {
    let exec = exec();
    let values = exec.to_device(&[1_u32, 10, 20, 30, 5]).unwrap();

    fill(&exec, (99_u32,), massively::SoA1(values.slice_mut(1..4))).unwrap();

    assert_eq!(exec.to_host(&values).unwrap(), vec![1, 99, 99, 99, 5]);
}

#[test]
fn fill_accepts_empty_output() {
    let exec = exec();
    let values = exec.to_device(&[1_u32, 2, 3]).unwrap();

    fill(&exec, (99_u32,), massively::SoA1(values.slice_mut(1..1))).unwrap();

    assert_eq!(exec.to_host(&values).unwrap(), vec![1, 2, 3]);
}

#[test]
fn fill_accepts_seven_tuple_columns() {
    let exec = exec();
    let a = exec.to_device(&[1_u32, 2]).unwrap();
    let b = exec.to_device(&[11_u32, 12]).unwrap();
    let c = exec.to_device(&[21_u32, 22]).unwrap();
    let d = exec.to_device(&[31_u32, 32]).unwrap();
    let e = exec.to_device(&[41_u32, 42]).unwrap();
    let f = exec.to_device(&[51_u32, 52]).unwrap();
    let g = exec.to_device(&[61_u32, 62]).unwrap();

    fill(
        &exec,
        (
            101_u32, 102_u32, 103_u32, 104_u32, 105_u32, 106_u32, 107_u32,
        ),
        massively::SoA7(
            a.slice_mut(..),
            b.slice_mut(..),
            c.slice_mut(..),
            d.slice_mut(..),
            e.slice_mut(..),
            f.slice_mut(..),
            g.slice_mut(..),
        ),
    )
    .unwrap();

    assert_eq!(exec.to_host(&a).unwrap(), vec![101, 101]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![102, 102]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![103, 103]);
    assert_eq!(exec.to_host(&d).unwrap(), vec![104, 104]);
    assert_eq!(exec.to_host(&e).unwrap(), vec![105, 105]);
    assert_eq!(exec.to_host(&f).unwrap(), vec![106, 106]);
    assert_eq!(exec.to_host(&g).unwrap(), vec![107, 107]);
}
