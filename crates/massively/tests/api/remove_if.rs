use crate::common::*;

#[test]
fn remove_if_accepts_heterogeneous_tuple_predicates() {
    let policy = policy();
    let values = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let tags = policy.to_device(&[10_u32, 20, 20, 30]).unwrap();

    let removed = remove_if((&values, &tags), PairMixedTagIsTwenty).unwrap();
    let (values, tags) = removed;
    assert_eq!(values.to_vec().unwrap(), vec![1.0, 4.0]);
    assert_eq!(tags.to_vec().unwrap(), vec![10, 30]);
}
