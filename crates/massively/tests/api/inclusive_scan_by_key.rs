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
    let output = inclusive_scan_by_key(
        &exec,
        massively::SoA1(keys.slice(..)),
        massively::SoA1(values.slice(..)),
        EqualU32,
        Sum,
    )
    .unwrap();
    let (output,) = output;

    assert_eq!(exec.to_host(&output).unwrap(), expected);
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

    let output = inclusive_scan_by_key(
        &exec,
        massively::SoA1(keys.slice(..)),
        massively::SoA3(a.slice(..), b.slice(..), c.slice(..)),
        EqualU32,
        TupleSum,
    )
    .unwrap();
    let (a, b, c) = output;
    assert_eq!(
        exec.to_host(&a).unwrap(),
        vec![1.0, 3.0, 3.0, 7.0, 12.0, 6.0]
    );
    assert_eq!(exec.to_host(&b).unwrap(), vec![10, 30, 30, 70, 120, 60]);
    assert_eq!(
        exec.to_host(&c).unwrap(),
        vec![100.0, 300.0, 300.0, 700.0, 1200.0, 600.0]
    );
}

#[test]
fn inclusive_scan_by_key_accepts_single_column_max_with_offset_slices() {
    let exec = exec();
    let keys = exec.to_device(&[99_u32, 0, 0, 0, 1, 1, 1, 88]).unwrap();
    let values = exec.to_device(&[99_u32, 1, 3, 2, 0, 5, 4, 88]).unwrap();

    let (output,) = inclusive_scan_by_key(
        &exec,
        massively::SoA1(keys.slice(1..7)),
        massively::SoA1(values.slice(1..7)),
        SameLowNibbleU32,
        MaxU32,
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![1, 3, 3, 0, 5, 5]);
}

#[test]
fn inclusive_scan_by_key_accepts_single_column_sum_with_same_low_nibble() {
    let exec = exec();
    let keys = exec.to_device(&[99_u32, 0, 0, 0, 1, 1, 1, 88]).unwrap();
    let values = exec.to_device(&[99_u32, 1, 3, 2, 0, 5, 4, 88]).unwrap();

    let (output,) = inclusive_scan_by_key(
        &exec,
        massively::SoA1(keys.slice(1..7)),
        massively::SoA1(values.slice(1..7)),
        SameLowNibbleU32,
        Sum,
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![1, 4, 6, 0, 5, 9]);
}
