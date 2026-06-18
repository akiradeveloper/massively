use crate::common::*;

#[test]
fn find_if_accepts_heterogeneous_tuple_predicates() {
    let policy = policy();
    let values = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let tags = policy.to_device(&[10_u32, 20, 20, 30]).unwrap();

    let first = find_if(zip(&values, &tags), PairMixedTagIsTwenty).unwrap();
    assert_eq!(first, Some(1));
}
