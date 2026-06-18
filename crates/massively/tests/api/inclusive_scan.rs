use crate::common::*;

#[test]
fn inclusive_scan_accepts_soa12() {
    let policy = policy();
    let a = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let output = inclusive_scan(zip(&a, &b), Sum).unwrap();
    let (a, b) = output;
    assert_eq!(a.to_vec().unwrap(), vec![1.0, 3.0, 6.0]);
    assert_eq!(b.to_vec().unwrap(), vec![10, 30, 60]);
}
