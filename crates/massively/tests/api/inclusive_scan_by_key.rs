use crate::common::*;

#[test]
fn inclusive_scan_by_key_accepts_tuple_keys() {
    let policy = policy();
    let key_a = policy.to_device(&[1.0_f32, 1.0, 2.0, 2.0, 2.0]).unwrap();
    let key_b = policy.to_device(&[10_u32, 10, 20, 20, 30]).unwrap();
    let values = policy.to_device(&[1_u32, 2, 3, 4, 5]).unwrap();

    let output = inclusive_scan_by_key((&key_a, &key_b), &values, MixedTupleEqual, Sum).unwrap();
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
    let output = inclusive_scan_by_key(&keys, &values, EqualU32, Sum).unwrap();
    let (output,) = output;

    assert_eq!(output.to_vec().unwrap(), expected);
}

#[cfg(any())]
#[test]
fn inclusive_scan_by_key_accepts_soa12_values() {
    let policy = policy();
    let keys = policy.to_device(&[0_u32, 0, 1, 1, 1, 2]).unwrap();
    let a = policy
        .to_device(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap();
    let b = policy.to_device(&[10_u32, 20, 30, 40, 50, 60]).unwrap();
    let c = policy
        .to_device(&[100.0_f32, 200.0, 300.0, 400.0, 500.0, 600.0])
        .unwrap();
    let d = policy
        .to_device(&[1000_u32, 2000, 3000, 4000, 5000, 6000])
        .unwrap();
    let e = policy
        .to_device(&[7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0])
        .unwrap();
    let f = policy.to_device(&[70_u32, 80, 90, 100, 110, 120]).unwrap();
    let g = policy
        .to_device(&[700.0_f32, 800.0, 900.0, 1000.0, 1100.0, 1200.0])
        .unwrap();
    let h = policy
        .to_device(&[7000_u32, 8000, 9000, 10000, 11000, 12000])
        .unwrap();
    let i = policy
        .to_device(&[13.0_f32, 14.0, 15.0, 16.0, 17.0, 18.0])
        .unwrap();
    let j = policy
        .to_device(&[130_u32, 140, 150, 160, 170, 180])
        .unwrap();
    let k = policy
        .to_device(&[1300.0_f32, 1400.0, 1500.0, 1600.0, 1700.0, 1800.0])
        .unwrap();
    let l = policy
        .to_device(&[13000_u32, 14000, 15000, 16000, 17000, 18000])
        .unwrap();

    let output = inclusive_scan_by_key(
        &keys,
        zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
        EqualU32,
        Sum,
    )
    .unwrap();
    let (a, b, _, _, _, _, _, _, _, _, _, _) = output;
    assert_eq!(a.to_vec().unwrap(), vec![1.0, 3.0, 3.0, 7.0, 12.0, 6.0]);
    assert_eq!(b.to_vec().unwrap(), vec![10, 30, 30, 70, 120, 60]);
}
