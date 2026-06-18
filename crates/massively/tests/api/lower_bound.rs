use crate::common::*;

#[test]
fn lower_bound_accepts_borrowed_heterogeneous_soa12() {
    let policy = policy();
    let k = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let l = policy.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let input = zip(&k, &l);
    assert_eq!(
        lower_bound(input, (3.0_f32, 30_u32), MixedTupleLess).unwrap(),
        2
    );
}
