use crate::common::*;

#[test]
fn exclusive_scan_accepts_tuple_columns() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let out_a = exec.to_device(&[0.0_f32; 3]).unwrap();
    let out_b = exec.to_device(&[0_u32; 3]).unwrap();
    exclusive_scan(
        &exec,
        massively::Zip2(a.slice(..), b.slice(..)),
        (0.0_f32, 0_u32),
        TupleSum,
        massively::Zip2(out_a.slice_mut(..), out_b.slice_mut(..)),
    )
    .unwrap();
    assert_eq!(exec.to_host(&out_a).unwrap(), vec![0.0, 1.0, 3.0]);
    assert_eq!(exec.to_host(&out_b).unwrap(), vec![0, 10, 30]);
}

#[test]
fn exclusive_scan_accepts_single_column_as_tuple_item() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let out = exec.to_device(&[0.0_f32; 3]).unwrap();

    exclusive_scan(
        &exec,
        massively::Zip1(a.slice(..)),
        (0.0_f32,),
        TupleSum,
        massively::Zip1(out.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out).unwrap(), vec![0.0, 1.0, 3.0]);
}

#[test]
fn exclusive_scan_accepts_three_column_tuple_item_op() {
    let exec = exec();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let c = exec.to_device(&[100.0_f32, 200.0, 300.0]).unwrap();
    let out_a = exec.to_device(&[0.0_f32; 3]).unwrap();
    let out_b = exec.to_device(&[0_u32; 3]).unwrap();
    let out_c = exec.to_device(&[0.0_f32; 3]).unwrap();

    exclusive_scan(
        &exec,
        massively::Zip3(a.slice(..), b.slice(..), c.slice(..)),
        (0.0_f32, 0_u32, 0.0_f32),
        TupleSum,
        massively::Zip3(
            out_a.slice_mut(..),
            out_b.slice_mut(..),
            out_c.slice_mut(..),
        ),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_a).unwrap(), vec![0.0, 1.0, 3.0]);
    assert_eq!(exec.to_host(&out_b).unwrap(), vec![0, 10, 30]);
    assert_eq!(exec.to_host(&out_c).unwrap(), vec![0.0, 100.0, 300.0]);
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
    let out_a = exec.to_device(&[0.0_f32; 3]).unwrap();
    let out_b = exec.to_device(&[0_u32; 3]).unwrap();
    let out_c = exec.to_device(&[0.0_f32; 3]).unwrap();
    let out_d = exec.to_device(&[0_u32; 3]).unwrap();
    let out_e = exec.to_device(&[0.0_f32; 3]).unwrap();
    let out_f = exec.to_device(&[0_u32; 3]).unwrap();
    let out_g = exec.to_device(&[0.0_f32; 3]).unwrap();

    exclusive_scan(
        &exec,
        massively::Zip7(
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
        massively::Zip7(
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

    assert_eq!(exec.to_host(&out_a).unwrap(), vec![0.0, 1.0, 3.0]);
    assert_eq!(exec.to_host(&out_b).unwrap(), vec![0, 10, 30]);
    assert_eq!(exec.to_host(&out_c).unwrap(), vec![0.0, 100.0, 300.0]);
    assert_eq!(exec.to_host(&out_d).unwrap(), vec![0, 1, 3]);
    assert_eq!(exec.to_host(&out_e).unwrap(), vec![0.0, 7.0, 15.0]);
    assert_eq!(exec.to_host(&out_f).unwrap(), vec![0, 4, 9]);
    assert_eq!(exec.to_host(&out_g).unwrap(), vec![0.0, 0.5, 2.0]);
}
