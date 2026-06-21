use crate::common::*;

#[test]
fn remove_if_accepts_heterogeneous_tuple_predicates() {
    let exec = exec();
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let tags = exec.to_device(&[10_u32, 20, 20, 30]).unwrap();

    let removed = remove_if(
        &exec,
        (values.slice(..), tags.slice(..)),
        PairMixedTagIsTwenty,
    )
    .unwrap();
    let (values, tags) = removed;
    assert_eq!(exec.to_host(&values).unwrap(), vec![1.0, 4.0]);
    assert_eq!(exec.to_host(&tags).unwrap(), vec![10, 30]);
}
