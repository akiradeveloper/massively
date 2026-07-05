use crate::common::*;

#[test]
fn count_if_accepts_heterogeneous_tuple_predicates() {
    let exec = exec();
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let tags = exec.to_device(&[10_u32, 20, 20, 30]).unwrap();

    let count = count_if(
        &exec,
        massively::Zip2(values.slice(..), tags.slice(..)),
        PairMixedTagIsTwenty,
        (),
    )
    .unwrap();
    assert_eq!(count, 2);
}

#[test]
fn predicate_queries_accept_device_slice_scalar_items() {
    let exec = exec();
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let partitioned = exec.to_device(&[4.0_f32, 3.0, 2.0, 1.0]).unwrap();

    assert_eq!(
        count_if(&exec, values.slice(..), GreaterThanF32, 2.5_f32).unwrap(),
        2
    );
    assert!(!massively::all_of(&exec, values.slice(..), GreaterThanF32, 2.5_f32).unwrap());
    assert!(massively::any_of(&exec, values.slice(..), GreaterThanF32, 2.5_f32).unwrap());
    assert!(!massively::none_of(&exec, values.slice(..), GreaterThanF32, 2.5_f32).unwrap());
    assert_eq!(
        find_if(&exec, values.slice(..), GreaterThanF32, 2.5_f32).unwrap(),
        Some(2)
    );
    assert!(is_partitioned(&exec, partitioned.slice(..), GreaterThanF32, 2.5_f32).unwrap());
}
