use crate::common::*;

#[test]
fn equal_accepts_borrowed_heterogeneous_soa12() {
    let policy = policy();
    let a = policy.to_device(&[1.0_f32, 2.0]).unwrap();
    let b = policy.to_device(&[10_u32, 20]).unwrap();
    let c = policy.to_device(&[1.0_f32, 2.0]).unwrap();
    let d = policy.to_device(&[10_u32, 20]).unwrap();

    assert!(equal(zip(&a, &b), zip(&c, &d), MixedTupleEqual).unwrap());
}
