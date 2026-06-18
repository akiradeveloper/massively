use crate::common::*;

#[test]
fn exclusive_scan_accepts_heterogeneous_soa() {
    let policy = policy();
    let a = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let output = exclusive_scan(zip(&a, &b), (0.0_f32, 0_u32), Sum).unwrap();
    let (a, b) = output;
    assert_eq!(a.to_vec().unwrap(), vec![0.0, 1.0, 3.0]);
    assert_eq!(b.to_vec().unwrap(), vec![0, 10, 30]);
}
