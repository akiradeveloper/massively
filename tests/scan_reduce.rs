mod common;
use common::*;

#[test]
fn reduce_and_scan_accept_heterogeneous_columns_when_op_supports_each_item() {
    let policy = policy();
    let values = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let ids = policy.to_device(&[10_u32, 20, 30]).unwrap();

    let reduced = reduce(vzip(&values, &ids), (0.0_f32, 0_u32), Sum).unwrap();
    assert_eq!(reduced, (6.0, 60));

    let inclusive = inclusive_scan(vzip(&values, &ids), Sum).unwrap();
    let (inclusive_values, inclusive_ids) = unzip(inclusive).unwrap();
    assert_eq!(inclusive_values.to_vec().unwrap(), vec![1.0, 3.0, 6.0]);
    assert_eq!(inclusive_ids.to_vec().unwrap(), vec![10, 30, 60]);

    let exclusive = exclusive_scan(vzip(&values, &ids), (0.0_f32, 0_u32), Sum).unwrap();
    let (exclusive_values, exclusive_ids) = unzip(exclusive).unwrap();
    assert_eq!(exclusive_values.to_vec().unwrap(), vec![0.0, 1.0, 3.0]);
    assert_eq!(exclusive_ids.to_vec().unwrap(), vec![0, 10, 30]);
}

#[test]
fn reduce_and_scan_accept_one_component_device_soas() {
    let policy = policy();
    let input = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();

    let sum = reduce(&input, 0.0, Sum).unwrap();
    assert_eq!(sum, 10.0);

    let inclusive = unzip(inclusive_scan(&input, Sum).unwrap()).unwrap();
    assert_eq!(inclusive.to_vec().unwrap(), vec![1.0, 3.0, 6.0, 10.0]);

    let exclusive = unzip(exclusive_scan(&input, 10.0, Sum).unwrap()).unwrap();
    assert_eq!(exclusive.to_vec().unwrap(), vec![10.0, 11.0, 13.0, 16.0]);
}

#[test]
fn reduce_accepts_sova12() {
    let policy = policy();
    let a = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = policy.to_device(&[1_u32, 2, 3]).unwrap();
    let c = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let d = policy.to_device(&[1_u32, 2, 3]).unwrap();
    let e = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let f = policy.to_device(&[1_u32, 2, 3]).unwrap();
    let g = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let h = policy.to_device(&[1_u32, 2, 3]).unwrap();
    let i = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let j = policy.to_device(&[1_u32, 2, 3]).unwrap();
    let k = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let l = policy.to_device(&[1_u32, 2, 3]).unwrap();

    let sums = reduce(
        vzip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
        (
            0.0, 0_u32, 0.0, 0_u32, 0.0, 0_u32, 0.0, 0_u32, 0.0, 0_u32, 0.0, 0_u32,
        ),
        Sum,
    )
    .unwrap();

    assert_eq!(sums, (6.0, 6, 6.0, 6, 6.0, 6, 6.0, 6, 6.0, 6, 6.0, 6));
}

#[test]
fn scan_accepts_sova12() {
    let policy = policy();
    let a = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let b = policy.to_device(&[1_u32, 2, 3]).unwrap();
    let c = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let d = policy.to_device(&[1_u32, 2, 3]).unwrap();
    let e = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let f = policy.to_device(&[1_u32, 2, 3]).unwrap();
    let g = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let h = policy.to_device(&[1_u32, 2, 3]).unwrap();
    let i = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let j = policy.to_device(&[1_u32, 2, 3]).unwrap();
    let k = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let l = policy.to_device(&[1_u32, 2, 3]).unwrap();

    let inclusive =
        inclusive_scan(vzip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l), Sum).unwrap();
    let (_a, _b, _c, _d, _e, _f, _g, _h, _i, _j, _k, l_out) = unzip(inclusive).unwrap();
    assert_eq!(l_out.to_vec().unwrap(), vec![1, 3, 6]);

    let exclusive = exclusive_scan(
        vzip12(&a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l),
        (
            0.0, 0_u32, 0.0, 0_u32, 0.0, 0_u32, 0.0, 0_u32, 0.0, 0_u32, 0.0, 10_u32,
        ),
        Sum,
    )
    .unwrap();
    let (a_out, _b, _c, _d, _e, _f, _g, _h, _i, _j, _k, l_out) = unzip(exclusive).unwrap();
    assert_eq!(a_out.to_vec().unwrap(), vec![0.0, 1.0, 3.0]);
    assert_eq!(l_out.to_vec().unwrap(), vec![10, 11, 13]);
}

#[test]
fn scan_by_key_accepts_one_component_sova_values() {
    let policy = policy();
    let keys = policy.to_device(&[0_u32, 0, 1, 1, 1, 2]).unwrap();
    let values = policy
        .to_device(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap();

    let inclusive = unzip(inclusive_scan_by_key(&values, &keys, EqualU32, Sum).unwrap()).unwrap();
    assert_eq!(
        inclusive.to_vec().unwrap(),
        vec![1.0, 3.0, 3.0, 7.0, 12.0, 6.0]
    );

    let exclusive =
        unzip(exclusive_scan_by_key(&values, &keys, 0.0, EqualU32, Sum).unwrap()).unwrap();
    assert_eq!(
        exclusive.to_vec().unwrap(),
        vec![0.0, 1.0, 0.0, 3.0, 7.0, 0.0]
    );
}

#[test]
fn scan_by_key_accepts_tuple_sova_values() {
    let policy = policy();
    let keys = policy.to_device(&[0_u32, 0, 1]).unwrap();
    let x = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let y = policy.to_device(&[10_u32, 20, 30]).unwrap();

    let inclusive = inclusive_scan_by_key(vzip(&x, &y), &keys, EqualU32, Sum).unwrap();
    let (x_out, y_out) = unzip(inclusive).unwrap();
    assert_eq!(x_out.to_vec().unwrap(), vec![1.0, 3.0, 3.0]);
    assert_eq!(y_out.to_vec().unwrap(), vec![10, 30, 30]);

    let exclusive =
        exclusive_scan_by_key(vzip(&x, &y), &keys, (0.0, 100_u32), EqualU32, Sum).unwrap();
    let (x_out, y_out) = unzip(exclusive).unwrap();
    assert_eq!(x_out.to_vec().unwrap(), vec![0.0, 1.0, 0.0]);
    assert_eq!(y_out.to_vec().unwrap(), vec![100, 110, 100]);
}
