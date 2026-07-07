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
    )
    .unwrap();
    assert_eq!(count, 2);
}

#[test]
fn predicate_queries_preserve_nested_zip_input_shape() {
    let exec = exec();
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let tags = exec.to_device(&[10_u32, 20, 20, 30]).unwrap();
    let bias = exec.to_device(&[1.0_f32, 1.0, -1.0, 1.0]).unwrap();

    let input = massively::Zip2(
        massively::Zip2(values.slice(..), tags.slice(..)),
        bias.slice(..),
    );
    assert_eq!(
        count_if(&exec, input, NestedTuple3MixedTagIsTwenty).unwrap(),
        1
    );

    let input = massively::Zip2(
        massively::Zip2(values.slice(..), tags.slice(..)),
        bias.slice(..),
    );
    assert!(!massively::all_of(&exec, input, NestedTuple3MixedTagIsTwenty).unwrap());

    let input = massively::Zip2(
        massively::Zip2(values.slice(..), tags.slice(..)),
        bias.slice(..),
    );
    assert!(massively::any_of(&exec, input, NestedTuple3MixedTagIsTwenty).unwrap());

    let input = massively::Zip2(
        massively::Zip2(values.slice(..), tags.slice(..)),
        bias.slice(..),
    );
    assert!(!massively::none_of(&exec, input, NestedTuple3MixedTagIsTwenty).unwrap());

    let input = massively::Zip2(
        massively::Zip2(values.slice(..), tags.slice(..)),
        bias.slice(..),
    );
    assert_eq!(
        find_if(&exec, input, NestedTuple3MixedTagIsTwenty).unwrap(),
        Some(1)
    );

    let values = exec.to_device(&[2.0_f32, 4.0, 1.0, 3.0]).unwrap();
    let tags = exec.to_device(&[20_u32, 20, 10, 20]).unwrap();
    let bias = exec.to_device(&[1.0_f32, 1.0, 1.0, -1.0]).unwrap();
    let input = massively::Zip2(
        massively::Zip2(values.slice(..), tags.slice(..)),
        bias.slice(..),
    );
    assert!(is_partitioned(&exec, input, NestedTuple3MixedTagIsTwenty).unwrap());
}

#[test]
fn predicate_queries_accept_device_slice_scalar_items() {
    let exec = exec();
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let partitioned = exec.to_device(&[4.0_f32, 3.0, 2.0, 1.0]).unwrap();
    let threshold = 2.5_f32;

    assert_eq!(
        count_if(
            &exec,
            massively::Zip2(
                values.slice(..),
                massively::lazy::constant(threshold).take(values.len())
            ),
            GreaterThanF32
        )
        .unwrap(),
        2
    );
    assert!(
        !massively::all_of(
            &exec,
            massively::Zip2(
                values.slice(..),
                massively::lazy::constant(threshold).take(values.len())
            ),
            GreaterThanF32
        )
        .unwrap()
    );
    assert!(
        massively::any_of(
            &exec,
            massively::Zip2(
                values.slice(..),
                massively::lazy::constant(threshold).take(values.len())
            ),
            GreaterThanF32
        )
        .unwrap()
    );
    assert!(
        !massively::none_of(
            &exec,
            massively::Zip2(
                values.slice(..),
                massively::lazy::constant(threshold).take(values.len())
            ),
            GreaterThanF32
        )
        .unwrap()
    );
    assert_eq!(
        find_if(
            &exec,
            massively::Zip2(
                values.slice(..),
                massively::lazy::constant(threshold).take(values.len())
            ),
            GreaterThanF32
        )
        .unwrap(),
        Some(2)
    );
    assert!(
        is_partitioned(
            &exec,
            massively::Zip2(
                partitioned.slice(..),
                massively::lazy::constant(threshold).take(partitioned.len())
            ),
            GreaterThanF32
        )
        .unwrap()
    );
}
