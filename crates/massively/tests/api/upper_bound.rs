use crate::common::*;

#[test]
fn upper_bound_accepts_borrowed_tuple_columns() {
    let policy = policy();
    let k = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let l = policy.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let input = (&k, &l);
    assert_eq!(
        upper_bound(input, (3.0_f32, 30_u32), MixedTupleLess).unwrap(),
        3
    );
}
