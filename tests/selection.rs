mod common;
use common::*;

#[test]
fn two_column_soa_algorithms_preserve_columns() {
    let policy = policy();
    let values = policy.to_device(&[-1.0_f32, 2.0, -3.0, 4.0]).unwrap();
    let payload = policy.to_device(&[10_u32, 20, 30, 40]).unwrap();

    let selected = copy_if(zip(&values, &payload), PairMixedFirstPositive).unwrap();
    let (selected_values, selected_payload) = selected;
    assert_eq!(selected_values.to_vec().unwrap(), vec![2.0, 4.0]);
    assert_eq!(selected_payload.to_vec().unwrap(), vec![20, 40]);

    let scan = inclusive_scan(zip(&values, &payload), Sum).unwrap();
    let (value_scan, payload_scan) = scan;
    assert_eq!(value_scan.to_vec().unwrap(), vec![-1.0, 1.0, -2.0, 2.0]);
    assert_eq!(payload_scan.to_vec().unwrap(), vec![10, 30, 60, 100]);
}

#[test]
fn three_column_soa_algorithms_preserve_columns() {
    let policy = policy();
    let a = policy.to_device(&[-1.0_f32, 2.0, -3.0, 4.0]).unwrap();
    let b = policy.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let c = policy.to_device(&[100.0_f32, 200.0, 300.0, 400.0]).unwrap();

    let selected = copy_if(zip3(&a, &b, &c), Tuple3MixedFirstPositive).unwrap();
    let (a_out, b_out, c_out) = selected;
    assert_eq!(a_out.to_vec().unwrap(), vec![2.0, 4.0]);
    assert_eq!(b_out.to_vec().unwrap(), vec![20, 40]);
    assert_eq!(c_out.to_vec().unwrap(), vec![200.0, 400.0]);

    let sum = reduce(zip3(&a, &b, &c), (0.0, 0_u32, 0.0), Sum).unwrap();
    assert_eq!(sum, (2.0, 100, 1000.0));

    let indices = policy.to_device(&[3_u32, 1, 0]).unwrap();
    let gathered = gather(zip3(&a, &b, &c), &indices).unwrap();
    let (a_out, b_out, c_out) = gathered;
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

    let selected = copy_if(zip(&values, &tags), PairMixedTagIsTwenty).unwrap();
    let (selected_values, selected_tags) = selected;
    assert_eq!(selected_values.to_vec().unwrap(), vec![2.0, 3.0]);
    assert_eq!(selected_tags.to_vec().unwrap(), vec![20, 20]);

    let selected = copy_if(zip3(&values, &tags, &bias), Tuple3MixedTagIsTwenty).unwrap();
    let (selected_values, selected_tags, selected_bias) = selected;
    assert_eq!(selected_values.to_vec().unwrap(), vec![3.0]);
    assert_eq!(selected_tags.to_vec().unwrap(), vec![20]);
    assert_eq!(selected_bias.to_vec().unwrap(), vec![2.0]);

    let count = count_if(zip(&values, &tags), PairMixedTagIsTwenty).unwrap();
    assert_eq!(count, 2);

    let first = find_if(zip(&values, &tags), PairMixedTagIsTwenty).unwrap();
    assert_eq!(first, Some(1));

    let removed = remove_if(zip(&values, &tags), PairMixedTagIsTwenty).unwrap();
    let (removed_values, removed_tags) = removed;
    assert_eq!(removed_values.to_vec().unwrap(), vec![1.0, 4.0]);
    assert_eq!(removed_tags.to_vec().unwrap(), vec![10, 30]);
}

#[test]
fn selection_and_index_algorithms_use_soa_boundaries() {
    let policy = policy();
    let input = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let tags = policy.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let doubled = transform(zip(&input, &tags), PairScaleAndTag).unwrap();
    let (doubled, transformed_tags) = doubled;

    assert_eq!(transformed_tags.to_vec().unwrap(), vec![11, 21, 31, 41]);

    let selected = copy_if(&doubled, GreaterThanFour).unwrap();
    assert_eq!(selected.to_vec().unwrap(), vec![6.0, 8.0]);

    let count = count_if(&doubled, GreaterThanFour).unwrap();
    assert_eq!(count, 2);

    let indices = policy.to_device(&[3_u32, 2, 1, 0]).unwrap();
    let gathered = gather(&doubled, &indices).unwrap();
    assert_eq!(gathered.to_vec().unwrap(), vec![8.0, 6.0, 4.0, 2.0]);

    let initial = policy.device_filled(4, 0.0_f32).unwrap();
    let scattered = scatter(&doubled, &indices, initial).unwrap();
    assert_eq!(scattered.to_vec().unwrap(), vec![8.0, 6.0, 4.0, 2.0]);

    let stencil = policy.to_device(&[1_u32, 0, 1, 0]).unwrap();
    let scatter_if = massively::scatter_if(
        &doubled,
        &indices,
        &stencil,
        policy.device_filled(4, 0.0_f32).unwrap(),
        NonZero,
    )
    .unwrap();
    assert_eq!(scatter_if.to_vec().unwrap(), vec![0.0, 6.0, 0.0, 2.0]);
}

#[test]
fn selection_accepts_soa12() {
    let policy = policy();
    let a = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let c = policy.to_device(&[100.0_f32, 200.0, 300.0]).unwrap();
    let d = policy.to_device(&[1000_u32, 2000, 3000]).unwrap();
    let e = policy.to_device(&[4.0_f32, 5.0, 6.0]).unwrap();
    let f = policy.to_device(&[40_u32, 50, 60]).unwrap();
    let g = policy.to_device(&[400.0_f32, 500.0, 600.0]).unwrap();
    let h = policy.to_device(&[4000_u32, 5000, 6000]).unwrap();
    let i = policy.to_device(&[7.0_f32, 8.0, 9.0]).unwrap();
    let j = policy.to_device(&[70_u32, 80, 90]).unwrap();
    let k = policy.to_device(&[700.0_f32, 800.0, 900.0]).unwrap();
    let l = policy.to_device(&[100_u32, 200, 300]).unwrap();

    let selected = copy_if(
        zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
        Tuple12MixedFirstGreaterThanOne,
    )
    .unwrap();
    let (a_out, b_out, c_out, d_out, e_out, f_out, g_out, h_out, i_out, j_out, k_out, l_out) =
        selected;
    assert_eq!(a_out.to_vec().unwrap(), vec![2.0, 3.0]);
    assert_eq!(b_out.to_vec().unwrap(), vec![20, 30]);
    assert_eq!(c_out.to_vec().unwrap(), vec![200.0, 300.0]);
    assert_eq!(d_out.to_vec().unwrap(), vec![2000, 3000]);
    assert_eq!(e_out.to_vec().unwrap(), vec![5.0, 6.0]);
    assert_eq!(f_out.to_vec().unwrap(), vec![50, 60]);
    assert_eq!(g_out.to_vec().unwrap(), vec![500.0, 600.0]);
    assert_eq!(h_out.to_vec().unwrap(), vec![5000, 6000]);
    assert_eq!(i_out.to_vec().unwrap(), vec![8.0, 9.0]);
    assert_eq!(j_out.to_vec().unwrap(), vec![80, 90]);
    assert_eq!(k_out.to_vec().unwrap(), vec![800.0, 900.0]);
    assert_eq!(l_out.to_vec().unwrap(), vec![200, 300]);

    let count = count_if(
        zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
        Tuple12MixedFirstGreaterThanOne,
    )
    .unwrap();
    assert_eq!(count, 2);

    let first = find_if(
        zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
        Tuple12MixedFirstGreaterThanOne,
    )
    .unwrap();
    assert_eq!(first, Some(1));

    let removed = remove_if(
        zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
        Tuple12MixedFirstGreaterThanOne,
    )
    .unwrap();
    let (a_out, b_out, c_out, d_out, e_out, f_out, g_out, h_out, i_out, j_out, k_out, l_out) =
        removed;
    assert_eq!(a_out.to_vec().unwrap(), vec![1.0]);
    assert_eq!(b_out.to_vec().unwrap(), vec![10]);
    assert_eq!(c_out.to_vec().unwrap(), vec![100.0]);
    assert_eq!(d_out.to_vec().unwrap(), vec![1000]);
    assert_eq!(e_out.to_vec().unwrap(), vec![4.0]);
    assert_eq!(f_out.to_vec().unwrap(), vec![40]);
    assert_eq!(g_out.to_vec().unwrap(), vec![400.0]);
    assert_eq!(h_out.to_vec().unwrap(), vec![4000]);
    assert_eq!(i_out.to_vec().unwrap(), vec![7.0]);
    assert_eq!(j_out.to_vec().unwrap(), vec![70]);
    assert_eq!(k_out.to_vec().unwrap(), vec![700.0]);
    assert_eq!(l_out.to_vec().unwrap(), vec![100]);
}

#[test]
fn selection_accepts_soa12_predicates_that_read_tail_columns() {
    let policy = policy();
    let a = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let b = policy.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let c = policy.to_device(&[100.0_f32, 200.0, 300.0, 400.0]).unwrap();
    let d = policy.to_device(&[1000_u32, 2000, 3000, 4000]).unwrap();
    let e = policy.to_device(&[4.0_f32, 5.0, 6.0, 7.0]).unwrap();
    let f = policy.to_device(&[40_u32, 50, 60, 70]).unwrap();
    let g = policy.to_device(&[400.0_f32, 500.0, 600.0, 700.0]).unwrap();
    let h = policy.to_device(&[4000_u32, 5000, 6000, 7000]).unwrap();
    let i = policy.to_device(&[7.0_f32, 8.0, 9.0, 10.0]).unwrap();
    let j = policy.to_device(&[70_u32, 80, 90, 100]).unwrap();
    let k = policy
        .to_device(&[700.0_f32, 800.0, 900.0, 1000.0])
        .unwrap();
    let l = policy.to_device(&[7000_u32, 8000, 9000, 10000]).unwrap();

    let selected = copy_if(
        zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
        Tuple12MixedTailPredicate,
    )
    .unwrap();
    let (a_out, b_out, c_out, d_out, e_out, f_out, g_out, h_out, i_out, j_out, k_out, l_out) =
        selected;
    assert_eq!(a_out.to_vec().unwrap(), vec![2.0, 4.0]);
    assert_eq!(b_out.to_vec().unwrap(), vec![20, 40]);
    assert_eq!(c_out.to_vec().unwrap(), vec![200.0, 400.0]);
    assert_eq!(d_out.to_vec().unwrap(), vec![2000, 4000]);
    assert_eq!(e_out.to_vec().unwrap(), vec![5.0, 7.0]);
    assert_eq!(f_out.to_vec().unwrap(), vec![50, 70]);
    assert_eq!(g_out.to_vec().unwrap(), vec![500.0, 700.0]);
    assert_eq!(h_out.to_vec().unwrap(), vec![5000, 7000]);
    assert_eq!(i_out.to_vec().unwrap(), vec![8.0, 10.0]);
    assert_eq!(j_out.to_vec().unwrap(), vec![80, 100]);
    assert_eq!(k_out.to_vec().unwrap(), vec![800.0, 1000.0]);
    assert_eq!(l_out.to_vec().unwrap(), vec![8000, 10000]);

    let count = count_if(
        zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
        Tuple12MixedTailPredicate,
    )
    .unwrap();
    assert_eq!(count, 2);

    let first = find_if(
        zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
        Tuple12MixedTailPredicate,
    )
    .unwrap();
    assert_eq!(first, Some(1));

    let removed = remove_if(
        zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
        Tuple12MixedTailPredicate,
    )
    .unwrap();
    let (a_out, b_out, c_out, d_out, e_out, f_out, g_out, h_out, i_out, j_out, k_out, l_out) =
        removed;
    assert_eq!(a_out.to_vec().unwrap(), vec![1.0, 3.0]);
    assert_eq!(b_out.to_vec().unwrap(), vec![10, 30]);
    assert_eq!(c_out.to_vec().unwrap(), vec![100.0, 300.0]);
    assert_eq!(d_out.to_vec().unwrap(), vec![1000, 3000]);
    assert_eq!(e_out.to_vec().unwrap(), vec![4.0, 6.0]);
    assert_eq!(f_out.to_vec().unwrap(), vec![40, 60]);
    assert_eq!(g_out.to_vec().unwrap(), vec![400.0, 600.0]);
    assert_eq!(h_out.to_vec().unwrap(), vec![4000, 6000]);
    assert_eq!(i_out.to_vec().unwrap(), vec![7.0, 9.0]);
    assert_eq!(j_out.to_vec().unwrap(), vec![70, 90]);
    assert_eq!(k_out.to_vec().unwrap(), vec![700.0, 900.0]);
    assert_eq!(l_out.to_vec().unwrap(), vec![7000, 9000]);
}

#[test]
fn partition_accepts_borrowed_heterogeneous_soa12() {
    let policy = policy();
    let a = policy.to_device(&[3.0_f32, 1.0, 4.0, 0.0, 2.0]).unwrap();
    let b = policy.to_device(&[30_u32, 10, 40, 0, 20]).unwrap();
    let c = policy
        .to_device(&[300.0_f32, 100.0, 400.0, 0.0, 200.0])
        .unwrap();
    let d = policy.to_device(&[3000_u32, 1000, 4000, 0, 2000]).unwrap();
    let e = policy.to_device(&[3.5_f32, 1.5, 4.5, 0.5, 2.5]).unwrap();
    let f = policy.to_device(&[35_u32, 15, 45, 5, 25]).unwrap();
    let g = policy
        .to_device(&[350.0_f32, 150.0, 450.0, 50.0, 250.0])
        .unwrap();
    let h = policy
        .to_device(&[3500_u32, 1500, 4500, 500, 2500])
        .unwrap();
    let i = policy.to_device(&[6.0_f32, 2.0, 8.0, 0.0, 4.0]).unwrap();
    let j = policy.to_device(&[60_u32, 20, 80, 0, 40]).unwrap();
    let k = policy
        .to_device(&[600.0_f32, 200.0, 800.0, 0.0, 400.0])
        .unwrap();
    let l = policy.to_device(&[6000_u32, 2000, 8000, 0, 4000]).unwrap();

    let output = partition(
        zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
        Tuple12MixedFirstGreaterThanOne,
    )
    .unwrap();
    let (a, b, c, d, e, f, g, h, i, j, k, l) = output;
    assert_eq!(a.to_vec().unwrap(), vec![3.0, 4.0, 2.0, 1.0, 0.0]);
    assert_eq!(b.to_vec().unwrap(), vec![30, 40, 20, 10, 0]);
    assert_eq!(c.to_vec().unwrap(), vec![300.0, 400.0, 200.0, 100.0, 0.0]);
    assert_eq!(d.to_vec().unwrap(), vec![3000, 4000, 2000, 1000, 0]);
    assert_eq!(e.to_vec().unwrap(), vec![3.5, 4.5, 2.5, 1.5, 0.5]);
    assert_eq!(f.to_vec().unwrap(), vec![35, 45, 25, 15, 5]);
    assert_eq!(g.to_vec().unwrap(), vec![350.0, 450.0, 250.0, 150.0, 50.0]);
    assert_eq!(h.to_vec().unwrap(), vec![3500, 4500, 2500, 1500, 500]);
    assert_eq!(i.to_vec().unwrap(), vec![6.0, 8.0, 4.0, 2.0, 0.0]);
    assert_eq!(j.to_vec().unwrap(), vec![60, 80, 40, 20, 0]);
    assert_eq!(k.to_vec().unwrap(), vec![600.0, 800.0, 400.0, 200.0, 0.0]);
    assert_eq!(l.to_vec().unwrap(), vec![6000, 8000, 4000, 2000, 0]);
}

#[test]
fn is_partitioned_accepts_borrowed_heterogeneous_soa12() {
    let policy = policy();
    let a = policy.to_device(&[3.0_f32, 4.0, 2.0, 1.0, 0.0]).unwrap();
    let b = policy.to_device(&[30_u32, 40, 20, 10, 0]).unwrap();
    let c = policy
        .to_device(&[300.0_f32, 400.0, 200.0, 100.0, 0.0])
        .unwrap();
    let d = policy.to_device(&[3000_u32, 4000, 2000, 1000, 0]).unwrap();
    let e = policy.to_device(&[3.5_f32, 4.5, 2.5, 1.5, 0.5]).unwrap();
    let f = policy.to_device(&[35_u32, 45, 25, 15, 5]).unwrap();
    let g = policy
        .to_device(&[350.0_f32, 450.0, 250.0, 150.0, 50.0])
        .unwrap();
    let h = policy
        .to_device(&[3500_u32, 4500, 2500, 1500, 500])
        .unwrap();
    let i = policy.to_device(&[6.0_f32, 8.0, 4.0, 2.0, 0.0]).unwrap();
    let j = policy.to_device(&[60_u32, 80, 40, 20, 0]).unwrap();
    let k = policy
        .to_device(&[600.0_f32, 800.0, 400.0, 200.0, 0.0])
        .unwrap();
    let l = policy.to_device(&[6000_u32, 8000, 4000, 2000, 0]).unwrap();

    let input = zip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l);
    assert!(is_partitioned(input, Tuple12MixedFirstGreaterThanOne).unwrap());
}
