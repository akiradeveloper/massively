use crate::common::*;

#[test]
fn inclusive_scan_by_key_handles_block_boundary_runs() {
    let exec = exec();
    let mut keys = vec![0_u32; 300];
    keys.extend(vec![1_u32; 20]);
    keys.extend(vec![0_u32; 10]);
    let values = vec![1_u32; keys.len()];
    let mut expected = (1..=300).collect::<Vec<u32>>();
    expected.extend(1..=20);
    expected.extend(1..=10);

    let keys = exec.to_device(&keys).unwrap();
    let values = exec.to_device(&values).unwrap();
    let output = exec.to_device(&vec![0_u32; values.len() as usize]).unwrap();
    inclusive_scan_by_key(
        &exec,
        massively::SoA1(keys.slice(..)),
        massively::SoA1(values.slice(..)),
        EqualU32,
        Sum,
        massively::SoA1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), expected);
}

#[test]
fn inclusive_scan_by_key_handles_all_same_key_long_run() {
    let exec = exec();
    let len = 512;
    let keys = vec![7_u32; len];
    let values = vec![1_u32; len];
    let expected = (1..=len as u32).collect::<Vec<_>>();

    let keys = exec.to_device(&keys).unwrap();
    let values = exec.to_device(&values).unwrap();
    let output = exec.to_device(&vec![0_u32; len]).unwrap();
    inclusive_scan_by_key(
        &exec,
        massively::SoA1(keys.slice(..)),
        massively::SoA1(values.slice(..)),
        EqualU32,
        Sum,
        massively::SoA1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), expected);
}

#[test]
fn inclusive_scan_by_key_handles_run_length_128_patterns() {
    let exec = exec();
    let mut keys = Vec::new();
    let mut expected = Vec::new();
    for key in 0..3_u32 {
        keys.extend(std::iter::repeat(key).take(128));
        expected.extend(1..=128_u32);
    }
    let values = vec![1_u32; keys.len()];

    let keys = exec.to_device(&keys).unwrap();
    let values = exec.to_device(&values).unwrap();
    let output = exec.to_device(&vec![0_u32; values.len() as usize]).unwrap();
    inclusive_scan_by_key(
        &exec,
        massively::SoA1(keys.slice(..)),
        massively::SoA1(values.slice(..)),
        EqualU32,
        Sum,
        massively::SoA1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), expected);
}

#[test]
fn inclusive_scan_by_key_handles_singleton_runs() {
    let exec = exec();
    let keys = exec.to_device(&[0_u32, 1, 2, 3]).unwrap();
    let values = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let output = exec.to_device(&[0_u32; 4]).unwrap();

    inclusive_scan_by_key(
        &exec,
        massively::SoA1(keys.slice(..)),
        massively::SoA1(values.slice(..)),
        EqualU32,
        Sum,
        massively::SoA1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![10, 20, 30, 40]);
}

#[test]
fn inclusive_scan_by_key_handles_one_run() {
    let exec = exec();
    let keys = exec.to_device(&[0_u32, 0, 0, 0]).unwrap();
    let values = exec.to_device(&[1_u32, 2, 3, 4]).unwrap();
    let output = exec.to_device(&[0_u32; 4]).unwrap();

    inclusive_scan_by_key(
        &exec,
        massively::SoA1(keys.slice(..)),
        massively::SoA1(values.slice(..)),
        EqualU32,
        Sum,
        massively::SoA1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![1, 3, 6, 10]);
}

#[test]
fn inclusive_scan_by_key_accepts_tuple_values() {
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

    inclusive_scan_by_key(
        &exec,
        massively::SoA1(keys.slice(..)),
        massively::SoA3(a.slice(..), b.slice(..), c.slice(..)),
        EqualU32,
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
        vec![1.0, 3.0, 3.0, 7.0, 12.0, 6.0]
    );
    assert_eq!(exec.to_host(&out_b).unwrap(), vec![10, 30, 30, 70, 120, 60]);
    assert_eq!(
        exec.to_host(&out_c).unwrap(),
        vec![100.0, 300.0, 300.0, 700.0, 1200.0, 600.0]
    );
}

#[test]
fn inclusive_scan_by_key_accepts_three_column_keys() {
    let exec = exec();
    let key_a = exec.to_device(&[1.0_f32, 1.0, 1.0, 1.0, 2.0, 2.0]).unwrap();
    let key_b = exec.to_device(&[0_u32, 0, 1, 1, 0, 0]).unwrap();
    let key_c = exec.to_device(&[5.0_f32, 5.0, 5.0, 6.0, 7.0, 7.0]).unwrap();
    let values = exec.to_device(&[1_u32, 2, 3, 4, 5, 6]).unwrap();
    let output = exec.to_device(&[0_u32; 6]).unwrap();

    inclusive_scan_by_key(
        &exec,
        massively::SoA3(key_a.slice(..), key_b.slice(..), key_c.slice(..)),
        massively::SoA1(values.slice(..)),
        MixedTuple3Equal,
        Sum,
        massively::SoA1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![1, 3, 3, 4, 5, 11]);
}

#[test]
fn inclusive_scan_by_key_accepts_three_column_keys_and_tuple_values() {
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

    inclusive_scan_by_key(
        &exec,
        massively::SoA3(key_a.slice(..), key_b.slice(..), key_c.slice(..)),
        massively::SoA3(a.slice(..), b.slice(..), c.slice(..)),
        MixedTuple3Equal,
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
        vec![1.0, 3.0, 3.0, 4.0, 5.0, 11.0]
    );
    assert_eq!(exec.to_host(&out_b).unwrap(), vec![10, 30, 30, 40, 50, 110]);
    assert_eq!(
        exec.to_host(&out_c).unwrap(),
        vec![100.0, 300.0, 300.0, 400.0, 500.0, 1100.0]
    );
}

#[test]
fn inclusive_scan_by_key_accepts_three_column_keys_and_seven_column_values() {
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

    inclusive_scan_by_key(
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
        vec![1.0, 3.0, 3.0, 4.0, 5.0, 11.0]
    );
    assert_eq!(exec.to_host(&out_b).unwrap(), vec![10, 30, 30, 40, 50, 110]);
    assert_eq!(
        exec.to_host(&out_c).unwrap(),
        vec![100.0, 300.0, 300.0, 400.0, 500.0, 1100.0]
    );
    assert_eq!(exec.to_host(&out_d).unwrap(), vec![1, 3, 3, 4, 5, 11]);
    assert_eq!(
        exec.to_host(&out_e).unwrap(),
        vec![7.0, 15.0, 9.0, 10.0, 11.0, 23.0]
    );
    assert_eq!(exec.to_host(&out_f).unwrap(), vec![4, 9, 6, 7, 8, 17]);
    assert_eq!(
        exec.to_host(&out_g).unwrap(),
        vec![0.5, 2.0, 2.5, 3.5, 4.5, 10.0]
    );
}

#[test]
fn inclusive_scan_by_key_accepts_single_column_max_with_offset_slices() {
    let exec = exec();
    let keys = exec.to_device(&[99_u32, 0, 0, 0, 1, 1, 1, 88]).unwrap();
    let values = exec.to_device(&[99_u32, 1, 3, 2, 0, 5, 4, 88]).unwrap();
    let output = exec.to_device(&[0_u32; 6]).unwrap();

    inclusive_scan_by_key(
        &exec,
        massively::SoA1(keys.slice(1..7)),
        massively::SoA1(values.slice(1..7)),
        SameLowNibbleU32,
        MaxU32,
        massively::SoA1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![1, 3, 3, 0, 5, 5]);
}

#[test]
fn inclusive_scan_by_key_accepts_single_column_sum_with_same_low_nibble() {
    let exec = exec();
    let keys = exec.to_device(&[99_u32, 0, 0, 0, 1, 1, 1, 88]).unwrap();
    let values = exec.to_device(&[99_u32, 1, 3, 2, 0, 5, 4, 88]).unwrap();
    let output = exec.to_device(&[0_u32; 6]).unwrap();

    inclusive_scan_by_key(
        &exec,
        massively::SoA1(keys.slice(1..7)),
        massively::SoA1(values.slice(1..7)),
        SameLowNibbleU32,
        Sum,
        massively::SoA1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![1, 4, 6, 0, 5, 9]);
}
