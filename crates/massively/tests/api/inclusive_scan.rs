use crate::common::*;

#[test]
fn inclusive_scan_accepts_tuple_columns() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let out_a = exec.to_device(&[0.0_f32; 3]).unwrap();
    let out_b = exec.to_device(&[0_u32; 3]).unwrap();
    inclusive_scan(
        &exec,
        massively::SoA2(a.slice(..), b.slice(..)),
        TupleSum,
        massively::SoA2(out_a.slice_mut(..), out_b.slice_mut(..)),
    )
    .unwrap();
    assert_eq!(exec.to_host(&out_a).unwrap(), vec![1.0, 3.0, 6.0]);
    assert_eq!(exec.to_host(&out_b).unwrap(), vec![10, 30, 60]);
}

#[test]
fn inclusive_scan_accepts_tuple_max_u32() {
    let exec = exec();
    let values = exec.to_device(&[1_u32, 3, 2, 0, 5, 4]).unwrap();
    let output = exec.to_device(&[0_u32; 6]).unwrap();

    inclusive_scan(
        &exec,
        massively::SoA1(values.slice(..)),
        MaxU32,
        massively::SoA1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![1, 3, 3, 3, 5, 5]);
}

#[test]
fn inclusive_scan_accepts_single_column_as_tuple_item() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let out = exec.to_device(&[0.0_f32; 3]).unwrap();

    inclusive_scan(
        &exec,
        massively::SoA1(a.slice(..)),
        TupleSum,
        massively::SoA1(out.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out).unwrap(), vec![1.0, 3.0, 6.0]);
}

#[test]
fn inclusive_scan_accepts_three_column_tuple_item_op() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let c = exec.to_device(&[100.0_f32, 200.0, 300.0]).unwrap();
    let out_a = exec.to_device(&[0.0_f32; 3]).unwrap();
    let out_b = exec.to_device(&[0_u32; 3]).unwrap();
    let out_c = exec.to_device(&[0.0_f32; 3]).unwrap();

    inclusive_scan(
        &exec,
        massively::SoA3(a.slice(..), b.slice(..), c.slice(..)),
        TupleSum,
        massively::SoA3(
            out_a.slice_mut(..),
            out_b.slice_mut(..),
            out_c.slice_mut(..),
        ),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_a).unwrap(), vec![1.0, 3.0, 6.0]);
    assert_eq!(exec.to_host(&out_b).unwrap(), vec![10, 30, 60]);
    assert_eq!(exec.to_host(&out_c).unwrap(), vec![100.0, 300.0, 600.0]);
}

#[test]
fn inclusive_scan_accepts_seven_tuple_columns() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let c = exec.to_device(&[100.0_f32, 200.0, 300.0]).unwrap();
    let d = exec.to_device(&[1_u32, 2, 3]).unwrap();
    let e = exec.to_device(&[7.0_f32, 8.0, 9.0]).unwrap();
    let f = exec.to_device(&[4_u32, 5, 6]).unwrap();
    let g = exec.to_device(&[0.5_f32, 1.5, 2.5]).unwrap();
    let out_a = exec.to_device(&[0.0_f32; 3]).unwrap();
    let out_b = exec.to_device(&[0_u32; 3]).unwrap();
    let out_c = exec.to_device(&[0.0_f32; 3]).unwrap();
    let out_d = exec.to_device(&[0_u32; 3]).unwrap();
    let out_e = exec.to_device(&[0.0_f32; 3]).unwrap();
    let out_f = exec.to_device(&[0_u32; 3]).unwrap();
    let out_g = exec.to_device(&[0.0_f32; 3]).unwrap();

    inclusive_scan(
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
        TupleSum,
        massively::SoA7(
            out_a.slice_mut(..),
            out_b.slice_mut(..),
            out_c.slice_mut(..),
            out_d.slice_mut(..),
            out_e.slice_mut(..),
            out_f.slice_mut(..),
            out_g.slice_mut(..),
        ),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_a).unwrap(), vec![1.0, 3.0, 6.0]);
    assert_eq!(exec.to_host(&out_b).unwrap(), vec![10, 30, 60]);
    assert_eq!(exec.to_host(&out_c).unwrap(), vec![100.0, 300.0, 600.0]);
    assert_eq!(exec.to_host(&out_d).unwrap(), vec![1, 3, 6]);
    assert_eq!(exec.to_host(&out_e).unwrap(), vec![7.0, 15.0, 24.0]);
    assert_eq!(exec.to_host(&out_f).unwrap(), vec![4, 9, 15]);
    assert_eq!(exec.to_host(&out_g).unwrap(), vec![0.5, 2.0, 4.5]);
}
