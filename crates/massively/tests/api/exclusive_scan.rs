use crate::common::*;

#[test]
fn exclusive_scan_accepts_tuple_columns() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let output = exclusive_scan(
        &exec,
        massively::SoA2(a.slice(..), b.slice(..)),
        (0.0_f32, 0_u32),
        TupleSum,
    )
    .unwrap();
    let massively::SoA2(a, b) = output;
    assert_eq!(exec.to_host(&a).unwrap(), vec![0.0, 1.0, 3.0]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![0, 10, 30]);
}

#[test]
fn exclusive_scan_accepts_single_column_as_tuple_item() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();

    let massively::SoA1(a) =
        exclusive_scan(&exec, massively::SoA1(a.slice(..)), (0.0_f32,), TupleSum).unwrap();

    assert_eq!(exec.to_host(&a).unwrap(), vec![0.0, 1.0, 3.0]);
}

#[test]
fn exclusive_scan_accepts_three_column_tuple_item_op() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let c = exec.to_device(&[100.0_f32, 200.0, 300.0]).unwrap();

    let massively::SoA3(a, b, c) = exclusive_scan(
        &exec,
        massively::SoA3(a.slice(..), b.slice(..), c.slice(..)),
        (0.0_f32, 0_u32, 0.0_f32),
        TupleSum,
    )
    .unwrap();

    assert_eq!(exec.to_host(&a).unwrap(), vec![0.0, 1.0, 3.0]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![0, 10, 30]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![0.0, 100.0, 300.0]);
}

#[test]
fn exclusive_scan_accepts_seven_tuple_columns() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let c = exec.to_device(&[100.0_f32, 200.0, 300.0]).unwrap();
    let d = exec.to_device(&[1_u32, 2, 3]).unwrap();
    let e = exec.to_device(&[7.0_f32, 8.0, 9.0]).unwrap();
    let f = exec.to_device(&[4_u32, 5, 6]).unwrap();
    let g = exec.to_device(&[0.5_f32, 1.5, 2.5]).unwrap();

    let massively::SoA7(a, b, c, d, e, f, g) = exclusive_scan(
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
        (0.0_f32, 0_u32, 0.0_f32, 0_u32, 0.0_f32, 0_u32, 0.0_f32),
        TupleSum,
    )
    .unwrap();

    assert_eq!(exec.to_host(&a).unwrap(), vec![0.0, 1.0, 3.0]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![0, 10, 30]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![0.0, 100.0, 300.0]);
    assert_eq!(exec.to_host(&d).unwrap(), vec![0, 1, 3]);
    assert_eq!(exec.to_host(&e).unwrap(), vec![0.0, 7.0, 15.0]);
    assert_eq!(exec.to_host(&f).unwrap(), vec![0, 4, 9]);
    assert_eq!(exec.to_host(&g).unwrap(), vec![0.0, 0.5, 2.0]);
}
