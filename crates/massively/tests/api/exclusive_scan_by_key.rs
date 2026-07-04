use crate::common::*;

#[test]
fn exclusive_scan_by_key_uses_supplied_key_equality() {
    let exec = exec();
    let keys = exec.to_device(&[0_u32, 0, 1]).unwrap();
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let ids = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let out_values = exec.to_device(&[0.0_f32; 3]).unwrap();
    let out_ids = exec.to_device(&[0_u32; 3]).unwrap();

    exclusive_scan_by_key(
        &exec,
        massively::SoA1(keys.slice(..)),
        massively::SoA2(values.slice(..), ids.slice(..)),
        NeverEqualU32,
        (100.0_f32, 1000_u32),
        TupleSum,
        massively::SoA2(out_values.slice_mut(..), out_ids.slice_mut(..)),
    )
    .unwrap();
    assert_eq!(
        exec.to_host(&out_values).unwrap(),
        vec![100.0, 100.0, 100.0]
    );
    assert_eq!(exec.to_host(&out_ids).unwrap(), vec![1000, 1000, 1000]);
}

#[test]
fn exclusive_scan_by_key_handles_one_run() {
    let exec = exec();
    let keys = exec.to_device(&[0_u32, 0, 0, 0]).unwrap();
    let values = exec.to_device(&[1_u32, 2, 3, 4]).unwrap();
    let output = exec.to_device(&[0_u32; 4]).unwrap();

    exclusive_scan_by_key(
        &exec,
        massively::SoA1(keys.slice(..)),
        massively::SoA1(values.slice(..)),
        EqualU32,
        (0_u32,),
        Sum,
        massively::SoA1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![0, 1, 3, 6]);
}

#[test]
fn exclusive_scan_by_key_handles_all_same_key_long_run() {
    let exec = exec();
    let len = 512;
    let keys = vec![7_u32; len];
    let values = vec![1_u32; len];
    let expected = (0..len as u32).collect::<Vec<_>>();

    let keys = exec.to_device(&keys).unwrap();
    let values = exec.to_device(&values).unwrap();
    let output = exec.to_device(&vec![0_u32; len]).unwrap();
    exclusive_scan_by_key(
        &exec,
        massively::SoA1(keys.slice(..)),
        massively::SoA1(values.slice(..)),
        EqualU32,
        (0_u32,),
        Sum,
        massively::SoA1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), expected);
}

#[test]
fn exclusive_scan_by_key_handles_block_boundary_runs() {
    let exec = exec();
    let mut keys = vec![0_u32; 300];
    keys.extend(vec![1_u32; 20]);
    keys.extend(vec![0_u32; 10]);
    let values = vec![1_u32; keys.len()];
    let mut expected = (0..300_u32).collect::<Vec<_>>();
    expected.extend(0..20_u32);
    expected.extend(0..10_u32);

    let keys = exec.to_device(&keys).unwrap();
    let values = exec.to_device(&values).unwrap();
    let output = exec.to_device(&vec![0_u32; values.len() as usize]).unwrap();
    exclusive_scan_by_key(
        &exec,
        massively::SoA1(keys.slice(..)),
        massively::SoA1(values.slice(..)),
        EqualU32,
        (0_u32,),
        Sum,
        massively::SoA1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), expected);
}

#[test]
fn exclusive_scan_by_key_accepts_tuple_values() {
    let exec = exec();
    let keys = exec.to_device(&[0_u32, 0, 1, 1, 1, 2]).unwrap();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 30, 40, 50, 60]).unwrap();
    let c = exec
        .to_device(&[100.0_f32, 200.0, 300.0, 400.0, 500.0, 600.0])
        .unwrap();
    let out_a = exec.to_device(&[0.0_f32; 6]).unwrap();
    let out_b = exec.to_device(&[0_u32; 6]).unwrap();
    let out_c = exec.to_device(&[0.0_f32; 6]).unwrap();

    exclusive_scan_by_key(
        &exec,
        massively::SoA1(keys.slice(..)),
        massively::SoA3(a.slice(..), b.slice(..), c.slice(..)),
        EqualU32,
        (0.0_f32, 0_u32, 0.0_f32),
        TupleSum,
        massively::SoA3(
            out_a.slice_mut(..),
            out_b.slice_mut(..),
            out_c.slice_mut(..),
        ),
    )
    .unwrap();
    assert_eq!(
        exec.to_host(&out_a).unwrap(),
        vec![0.0, 1.0, 0.0, 3.0, 7.0, 0.0]
    );
    assert_eq!(exec.to_host(&out_b).unwrap(), vec![0, 10, 0, 30, 70, 0]);
    assert_eq!(
        exec.to_host(&out_c).unwrap(),
        vec![0.0, 100.0, 0.0, 300.0, 700.0, 0.0]
    );
}

#[test]
fn exclusive_scan_by_key_accepts_three_column_keys() {
    let exec = exec();
    let key_a = exec.to_device(&[1.0_f32, 1.0, 1.0, 1.0, 2.0, 2.0]).unwrap();
    let key_b = exec.to_device(&[0_u32, 0, 1, 1, 0, 0]).unwrap();
    let key_c = exec.to_device(&[5.0_f32, 5.0, 5.0, 6.0, 7.0, 7.0]).unwrap();
    let values = exec.to_device(&[1_u32, 2, 3, 4, 5, 6]).unwrap();
    let output = exec.to_device(&[0_u32; 6]).unwrap();

    exclusive_scan_by_key(
        &exec,
        massively::SoA3(key_a.slice(..), key_b.slice(..), key_c.slice(..)),
        massively::SoA1(values.slice(..)),
        MixedTuple3Equal,
        (0_u32,),
        Sum,
        massively::SoA1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![0, 1, 0, 0, 0, 5]);
}

#[test]
fn exclusive_scan_by_key_accepts_three_column_keys_and_tuple_values() {
    let exec = exec();
    let key_a = exec.to_device(&[1.0_f32, 1.0, 1.0, 1.0, 2.0, 2.0]).unwrap();
    let key_b = exec.to_device(&[0_u32, 0, 1, 1, 0, 0]).unwrap();
    let key_c = exec.to_device(&[5.0_f32, 5.0, 5.0, 6.0, 7.0, 7.0]).unwrap();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 30, 40, 50, 60]).unwrap();
    let c = exec
        .to_device(&[100.0_f32, 200.0, 300.0, 400.0, 500.0, 600.0])
        .unwrap();
    let out_a = exec.to_device(&[0.0_f32; 6]).unwrap();
    let out_b = exec.to_device(&[0_u32; 6]).unwrap();
    let out_c = exec.to_device(&[0.0_f32; 6]).unwrap();

    exclusive_scan_by_key(
        &exec,
        massively::SoA3(key_a.slice(..), key_b.slice(..), key_c.slice(..)),
        massively::SoA3(a.slice(..), b.slice(..), c.slice(..)),
        MixedTuple3Equal,
        (0.0_f32, 0_u32, 0.0_f32),
        TupleSum,
        massively::SoA3(
            out_a.slice_mut(..),
            out_b.slice_mut(..),
            out_c.slice_mut(..),
        ),
    )
    .unwrap();

    assert_eq!(
        exec.to_host(&out_a).unwrap(),
        vec![0.0, 1.0, 0.0, 0.0, 0.0, 5.0]
    );
    assert_eq!(exec.to_host(&out_b).unwrap(), vec![0, 10, 0, 0, 0, 50]);
    assert_eq!(
        exec.to_host(&out_c).unwrap(),
        vec![0.0, 100.0, 0.0, 0.0, 0.0, 500.0]
    );
}

#[test]
fn exclusive_scan_by_key_accepts_three_column_keys_and_seven_column_values() {
    let exec = exec();
    let key_a = exec.to_device(&[1.0_f32, 1.0, 1.0, 1.0, 2.0, 2.0]).unwrap();
    let key_b = exec.to_device(&[0_u32, 0, 1, 1, 0, 0]).unwrap();
    let key_c = exec.to_device(&[5.0_f32, 5.0, 5.0, 6.0, 7.0, 7.0]).unwrap();
    let a = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = exec.to_device(&[10_u32, 20, 30, 40, 50, 60]).unwrap();
    let c = exec
        .to_device(&[100.0_f32, 200.0, 300.0, 400.0, 500.0, 600.0])
        .unwrap();
    let d = exec.to_device(&[1_u32, 2, 3, 4, 5, 6]).unwrap();
    let e = exec
        .to_device(&[7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0])
        .unwrap();
    let f = exec.to_device(&[4_u32, 5, 6, 7, 8, 9]).unwrap();
    let g = exec.to_device(&[0.5_f32, 1.5, 2.5, 3.5, 4.5, 5.5]).unwrap();
    let out_a = exec.to_device(&[0.0_f32; 6]).unwrap();
    let out_b = exec.to_device(&[0_u32; 6]).unwrap();
    let out_c = exec.to_device(&[0.0_f32; 6]).unwrap();
    let out_d = exec.to_device(&[0_u32; 6]).unwrap();
    let out_e = exec.to_device(&[0.0_f32; 6]).unwrap();
    let out_f = exec.to_device(&[0_u32; 6]).unwrap();
    let out_g = exec.to_device(&[0.0_f32; 6]).unwrap();

    exclusive_scan_by_key(
        &exec,
        massively::SoA3(key_a.slice(..), key_b.slice(..), key_c.slice(..)),
        massively::SoA7(
            a.slice(..),
            b.slice(..),
            c.slice(..),
            d.slice(..),
            e.slice(..),
            f.slice(..),
            g.slice(..),
        ),
        MixedTuple3Equal,
        (0.0_f32, 0_u32, 0.0_f32, 0_u32, 0.0_f32, 0_u32, 0.0_f32),
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

    assert_eq!(
        exec.to_host(&out_a).unwrap(),
        vec![0.0, 1.0, 0.0, 0.0, 0.0, 5.0]
    );
    assert_eq!(exec.to_host(&out_b).unwrap(), vec![0, 10, 0, 0, 0, 50]);
    assert_eq!(
        exec.to_host(&out_c).unwrap(),
        vec![0.0, 100.0, 0.0, 0.0, 0.0, 500.0]
    );
    assert_eq!(exec.to_host(&out_d).unwrap(), vec![0, 1, 0, 0, 0, 5]);
    assert_eq!(
        exec.to_host(&out_e).unwrap(),
        vec![0.0, 7.0, 0.0, 0.0, 0.0, 11.0]
    );
    assert_eq!(exec.to_host(&out_f).unwrap(), vec![0, 4, 0, 0, 0, 8]);
    assert_eq!(
        exec.to_host(&out_g).unwrap(),
        vec![0.0, 0.5, 0.0, 0.0, 0.0, 4.5]
    );
}
