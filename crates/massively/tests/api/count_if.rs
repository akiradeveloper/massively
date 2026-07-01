use crate::common::*;

#[test]
fn count_if_accepts_heterogeneous_tuple_predicates() {
    let exec = exec();
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let tags = exec.to_device(&[10_u32, 20, 20, 30]).unwrap();

    let count = count_if(
        &exec,
        massively::SoA2(values.slice(..), tags.slice(..)),
        PairMixedTagIsTwenty,
        (),
    )
    .unwrap();
    assert_eq!(count, 2);
}
