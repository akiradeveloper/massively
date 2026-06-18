use crate::common::*;

#[test]
fn reduce_accepts_soa12() {
    let policy = policy();
    let a = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let sum = reduce(zip(&a, &b), (0.0_f32, 0_u32), Sum).unwrap();
    assert_eq!(sum, (6.0, 60));
}
