use crate::common::*;

#[test]
fn find_if_accepts_heterogeneous_tuple_predicates() {
    let exec = exec();
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let tags = exec.to_device(&[10_u32, 20, 20, 30]).unwrap();

    let first = find_if(
        &exec,
        (values.slice(..), tags.slice(..)),
        PairMixedTagIsTwenty,
    )
    .unwrap();
    assert_eq!(first, Some(1));
}
