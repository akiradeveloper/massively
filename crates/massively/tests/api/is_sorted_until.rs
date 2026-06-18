use crate::common::*;

#[test]
fn is_sorted_until_accepts_borrowed_heterogeneous_soa12() {
    let policy = policy();
    let k = policy.to_device(&[1.0_f32, 3.0, 2.0, 4.0]).unwrap();
    let l = policy.to_device(&[10_u32, 30, 20, 40]).unwrap();
    assert_eq!(is_sorted_until(zip(&k, &l), MixedTupleLess).unwrap(), 2);
}
