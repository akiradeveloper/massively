mod common;
use common::*;

#[test]
fn vzip_flattens_soa1_columns() {
    let policy = policy();
    let left = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let right = policy.to_device(&[10_u32, 20, 30]).unwrap();

    let split = transform(vzip(&left, &right), PairMixedSplit).unwrap();
    let (values, tags) = unzip(split).unwrap();

    assert_eq!(values.to_vec().unwrap(), vec![11.0, 12.0, 13.0]);
    assert_eq!(tags.to_vec().unwrap(), vec![11, 21, 31]);
}

#[test]
fn transform_vzip_output_unzips_to_storage() {
    let policy = policy();
    let values = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let tags = policy.to_device(&[10_u32, 20, 30]).unwrap();

    let output = transform(vzip(&values, &tags), PairMixedSplit).unwrap();
    let (values, tags) = unzip(output).unwrap();

    assert_eq!(values.to_vec().unwrap(), vec![11.0, 12.0, 13.0]);
    assert_eq!(tags.to_vec().unwrap(), vec![11, 21, 31]);
}

#[test]
fn transform_returns_device_storage() {
    let policy = policy();
    let left = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let right = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let split = transform(vzip(&left, &right), PairMixedSplit).unwrap();
    let (values, tags) = unzip(split).unwrap();

    assert_eq!(values.to_vec().unwrap(), vec![11.0, 12.0, 13.0]);
    assert_eq!(tags.to_vec().unwrap(), vec![11, 21, 31]);
}

#[test]
fn transform_tuple_output_maps_to_storage_output() {
    let policy = policy();
    let values = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let tags = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let bias = policy.to_device(&[100.0_f32, 200.0, 300.0]).unwrap();

    let split = transform(vzip3(&values, &tags, &bias), Tuple3MixedSplit).unwrap();
    let (values, flags, bias) = unzip(split).unwrap();
    assert_eq!(values.to_vec().unwrap(), vec![101.0, 202.0, 303.0]);
    assert_eq!(flags.to_vec().unwrap(), vec![11, 21, 31]);
    assert_eq!(bias.to_vec().unwrap(), vec![101.0, 202.0, 303.0]);
}

#[test]
fn scalar_transform_returns_soa1_storage() {
    let policy = policy();
    let input = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();

    let output = transform(&input, Double).unwrap();
    let output = unzip(output).unwrap();

    assert_eq!(output.to_vec().unwrap(), vec![2.0, 4.0, 6.0]);
}

#[test]
fn tuple_transform_uses_flat_sova_input() {
    let policy = policy();
    let lhs = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let rhs = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let bias = policy.to_device(&[100.0_f32, 200.0, 300.0]).unwrap();

    let output = transform(vzip3(&lhs, &rhs, &bias), Tuple3MixedSplit).unwrap();
    let (values, tags, adjusted_bias) = unzip(output).unwrap();

    assert_eq!(values.to_vec().unwrap(), vec![101.0, 202.0, 303.0]);
    assert_eq!(tags.to_vec().unwrap(), vec![11, 21, 31]);
    assert_eq!(adjusted_bias.to_vec().unwrap(), vec![101.0, 202.0, 303.0]);
}

#[test]
fn transform_accepts_heterogeneous_tuple_inputs() {
    let policy = policy();
    let values = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();
    let tags = policy.to_device(&[10_u32, 20, 30]).unwrap();
    let bias = policy.to_device(&[100.0_f32, 200.0, 300.0]).unwrap();
    let pair_output = transform(vzip(&values, &tags), PairMixedSplit).unwrap();
    let (pair_values, pair_tags) = unzip(pair_output).unwrap();
    assert_eq!(pair_values.to_vec().unwrap(), vec![11.0, 12.0, 13.0]);
    assert_eq!(pair_tags.to_vec().unwrap(), vec![11, 21, 31]);

    let tuple3_output = transform(vzip3(&values, &tags, &bias), Tuple3MixedSplit).unwrap();
    let (tuple_values, tuple_tags, tuple_bias) = unzip(tuple3_output).unwrap();
    assert_eq!(tuple_values.to_vec().unwrap(), vec![101.0, 202.0, 303.0]);
    assert_eq!(tuple_tags.to_vec().unwrap(), vec![11, 21, 31]);
    assert_eq!(tuple_bias.to_vec().unwrap(), vec![101.0, 202.0, 303.0]);
}
