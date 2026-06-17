mod common;
use common::*;

#[test]
fn two_column_soa_algorithms_preserve_columns() {
    let policy = policy();
    let values = policy.to_device(&[-1.0_f32, 2.0, -3.0, 4.0]).unwrap();
    let payload = policy.to_device(&[10_u32, 20, 30, 40]).unwrap();

    let selected = copy_if(vzip(&values, &payload), PairMixedFirstPositive).unwrap();
    let (selected_values, selected_payload) = unzip(selected).unwrap();
    assert_eq!(selected_values.to_vec().unwrap(), vec![2.0, 4.0]);
    assert_eq!(selected_payload.to_vec().unwrap(), vec![20, 40]);

    let scan = inclusive_scan(vzip(&values, &payload), Sum).unwrap();
    let (value_scan, payload_scan) = unzip(scan).unwrap();
    assert_eq!(value_scan.to_vec().unwrap(), vec![-1.0, 1.0, -2.0, 2.0]);
    assert_eq!(payload_scan.to_vec().unwrap(), vec![10, 30, 60, 100]);
}

#[test]
fn three_column_soa_algorithms_preserve_columns() {
    let policy = policy();
    let a = policy.to_device(&[-1.0_f32, 2.0, -3.0, 4.0]).unwrap();
    let b = policy.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let c = policy.to_device(&[100.0_f32, 200.0, 300.0, 400.0]).unwrap();

    let selected = copy_if(vzip3(&a, &b, &c), Tuple3MixedFirstPositive).unwrap();
    let (a_out, b_out, c_out) = unzip(selected).unwrap();
    assert_eq!(a_out.to_vec().unwrap(), vec![2.0, 4.0]);
    assert_eq!(b_out.to_vec().unwrap(), vec![20, 40]);
    assert_eq!(c_out.to_vec().unwrap(), vec![200.0, 400.0]);

    let sum = reduce(vzip3(&a, &b, &c), (0.0, 0_u32, 0.0), Sum).unwrap();
    assert_eq!(sum, (2.0, 100, 1000.0));

    let indices = policy.to_device(&[3_u32, 1, 0]).unwrap();
    let gathered = gather(vzip3(&a, &b, &c), &indices).unwrap();
    let (a_out, b_out, c_out) = unzip(gathered).unwrap();
    assert_eq!(a_out.to_vec().unwrap(), vec![4.0, 2.0, -1.0]);
    assert_eq!(b_out.to_vec().unwrap(), vec![40, 20, 10]);
    assert_eq!(c_out.to_vec().unwrap(), vec![400.0, 200.0, 100.0]);
}

#[test]
fn selection_accepts_heterogeneous_tuple_predicates() {
    let policy = policy();
    let values = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let tags = policy.to_device(&[10_u32, 20, 20, 30]).unwrap();
    let bias = policy.to_device(&[1.0_f32, -1.0, 2.0, 3.0]).unwrap();

    let selected = copy_if(vzip(&values, &tags), PairMixedTagIsTwenty).unwrap();
    let (selected_values, selected_tags) = unzip(selected).unwrap();
    assert_eq!(selected_values.to_vec().unwrap(), vec![2.0, 3.0]);
    assert_eq!(selected_tags.to_vec().unwrap(), vec![20, 20]);

    let selected = copy_if(vzip3(&values, &tags, &bias), Tuple3MixedTagIsTwenty).unwrap();
    let (selected_values, selected_tags, selected_bias) = unzip(selected).unwrap();
    assert_eq!(selected_values.to_vec().unwrap(), vec![3.0]);
    assert_eq!(selected_tags.to_vec().unwrap(), vec![20]);
    assert_eq!(selected_bias.to_vec().unwrap(), vec![2.0]);

    let count = count_if(vzip(&values, &tags), PairMixedTagIsTwenty).unwrap();
    assert_eq!(count, 2);

    let first = find_if(vzip(&values, &tags), PairMixedTagIsTwenty).unwrap();
    assert_eq!(first, Some(1));

    let removed = remove_if(zip(values, tags), PairMixedTagIsTwenty).unwrap();
    let (removed_values, removed_tags) = unzip(removed).unwrap();
    assert_eq!(removed_values.to_vec().unwrap(), vec![1.0, 4.0]);
    assert_eq!(removed_tags.to_vec().unwrap(), vec![10, 30]);
}

#[test]
fn selection_and_index_algorithms_use_device_soa_boundaries() {
    let policy = policy();
    let input = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let tags = policy.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let doubled = transform(vzip(&input, &tags), PairScaleAndTag).unwrap();
    let (doubled, transformed_tags) = unzip(doubled).unwrap();

    assert_eq!(transformed_tags.to_vec().unwrap(), vec![11, 21, 31, 41]);

    let selected = unzip(copy_if(&doubled, GreaterThanFour).unwrap()).unwrap();
    assert_eq!(selected.to_vec().unwrap(), vec![6.0, 8.0]);

    let count = count_if(&doubled, GreaterThanFour).unwrap();
    assert_eq!(count, 2);

    let indices = policy.to_device(&[3_u32, 2, 1, 0]).unwrap();
    let gathered = unzip(gather(&doubled, &indices).unwrap()).unwrap();
    assert_eq!(gathered.to_vec().unwrap(), vec![8.0, 6.0, 4.0, 2.0]);

    let initial = policy.device_filled(4, 0.0_f32).unwrap();
    let scattered = unzip(scatter(&doubled, &indices, initial).unwrap()).unwrap();
    assert_eq!(scattered.to_vec().unwrap(), vec![8.0, 6.0, 4.0, 2.0]);

    let stencil = policy.to_device(&[1_u32, 0, 1, 0]).unwrap();
    let scatter_if = unzip(
        massively::scatter_if(
            &doubled,
            &indices,
            &stencil,
            policy.device_filled(4, 0.0_f32).unwrap(),
            NonZero,
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(scatter_if.to_vec().unwrap(), vec![0.0, 6.0, 0.0, 2.0]);
}

#[test]
fn selection_accepts_soa12() {
    let policy = policy();
    let a = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let c = policy.to_device(&[10.0_f32, 20.0, 30.0]).unwrap();
    let d = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let e = policy.to_device(&[10.0_f32, 20.0, 30.0]).unwrap();
    let f = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let g = policy.to_device(&[10.0_f32, 20.0, 30.0]).unwrap();
    let h = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let i = policy.to_device(&[10.0_f32, 20.0, 30.0]).unwrap();
    let j = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let k = policy.to_device(&[10.0_f32, 20.0, 30.0]).unwrap();
    let l = policy.to_device(&[100_u32, 200, 300]).unwrap();

    let selected = copy_if(
        vzip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
        Tuple12MixedFirstGreaterThanOne,
    )
    .unwrap();
    let (a_out, _b, _c, _d, _e, _f, _g, _h, _i, _j, _k, l_out) = unzip(selected).unwrap();
    assert_eq!(a_out.to_vec().unwrap(), vec![2.0, 3.0]);
    assert_eq!(l_out.to_vec().unwrap(), vec![200, 300]);

    let count = count_if(
        vzip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
        Tuple12MixedFirstGreaterThanOne,
    )
    .unwrap();
    assert_eq!(count, 2);

    let first = find_if(
        vzip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
        Tuple12MixedFirstGreaterThanOne,
    )
    .unwrap();
    assert_eq!(first, Some(1));

    let removed = remove_if(
        zip12(a, b, c, d, e, f, g, h, i, j, k, l),
        Tuple12MixedFirstGreaterThanOne,
    )
    .unwrap();
    let (a_out, _b, _c, _d, _e, _f, _g, _h, _i, _j, _k, l_out) = unzip(removed).unwrap();
    assert_eq!(a_out.to_vec().unwrap(), vec![1.0]);
    assert_eq!(l_out.to_vec().unwrap(), vec![100]);
}
