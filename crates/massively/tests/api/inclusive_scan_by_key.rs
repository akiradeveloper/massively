use crate::common::*;

#[cfg(any())]
#[test]
fn inclusive_scan_by_key_accepts_tuple_keys() {
    let policy = policy();
    let key_a = policy.to_device(&[1.0_f32, 1.0, 2.0, 2.0, 2.0]).unwrap();
    let key_b = policy.to_device(&[10_u32, 10, 20, 20, 30]).unwrap();
    let values = policy.to_device(&[1_u32, 2, 3, 4, 5]).unwrap();

    let output = inclusive_scan_by_key(
        (key_a.slice(..), key_b.slice(..)),
        (values.slice(..),),
        MixedTupleEqual,
        Sum,
    )
    .unwrap();
    let (output,) = output;
    assert_eq!(output.to_vec().unwrap(), vec![1, 3, 3, 7, 5]);
}

#[test]
fn inclusive_scan_by_key_handles_block_boundary_runs() {
    let policy = policy();
    let mut keys = vec![0_u32; 300];
    keys.extend(vec![1_u32; 20]);
    keys.extend(vec![0_u32; 10]);
    let values = vec![1_u32; keys.len()];
    let mut expected = (1..=300).collect::<Vec<u32>>();
    expected.extend(1..=20);
    expected.extend(1..=10);

    let keys = policy.to_device(&keys).unwrap();
    let values = policy.to_device(&values).unwrap();
    let output =
        inclusive_scan_by_key((keys.slice(..),), (values.slice(..),), EqualU32, Sum).unwrap();
    let (output,) = output;

    assert_eq!(output.to_vec().unwrap(), expected);
}

#[test]
fn inclusive_scan_by_key_accepts_tuple_values() {
    let policy = policy();
    let keys = policy.to_device(&[0_u32, 0, 1, 1, 1, 2]).unwrap();
    let a = policy
        .to_device(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap();
    let b = policy.to_device(&[10_u32, 20, 30, 40, 50, 60]).unwrap();
    let c = policy
        .to_device(&[100.0_f32, 200.0, 300.0, 400.0, 500.0, 600.0])
        .unwrap();

    let output = inclusive_scan_by_key(
        (keys.slice(..),),
        (a.slice(..), b.slice(..), c.slice(..)),
        EqualU32,
        TupleSum,
    )
    .unwrap();
    let (a, b, c) = output;
    assert_eq!(a.to_vec().unwrap(), vec![1.0, 3.0, 3.0, 7.0, 12.0, 6.0]);
    assert_eq!(b.to_vec().unwrap(), vec![10, 30, 30, 70, 120, 60]);
    assert_eq!(
        c.to_vec().unwrap(),
        vec![100.0, 300.0, 300.0, 700.0, 1200.0, 600.0]
    );
}
