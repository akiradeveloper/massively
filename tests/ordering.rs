mod common;
use common::*;

#[test]
fn sort_returns_a_device_soa_until_unzip() {
    let policy = policy();
    let x = policy.to_device(&[3.0_f32, 1.0, 2.0]).unwrap();

    let sorted = sort(x, Less).unwrap();
    let sorted = unzip(sorted).unwrap();

    assert_eq!(sorted.to_vec().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn tuple_sort_preserves_soa_components() {
    let policy = policy();
    let x = policy.to_device(&[3.0_f32, 1.0, 2.0]).unwrap();
    let y = policy.to_device(&[30_u32, 10, 20]).unwrap();

    let sorted = sort(zip(x, y), MixedTupleLess).unwrap();
    let (x, y) = unzip(sorted).unwrap();

    assert_eq!(x.to_vec().unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(y.to_vec().unwrap(), vec![10, 20, 30]);
}

#[test]
fn sort_accepts_heterogeneous_tuple_comparators_for_two_and_three_columns() {
    let policy = policy();
    let values = policy.to_device(&[2.0_f32, 1.0, 2.0, 3.0]).unwrap();
    let tags = policy.to_device(&[20_u32, 30, 10, 40]).unwrap();

    let sorted = sort(zip(values, tags), MixedTupleLess).unwrap();
    let (values, tags) = unzip(sorted).unwrap();
    assert_eq!(values.to_vec().unwrap(), vec![1.0, 2.0, 2.0, 3.0]);
    assert_eq!(tags.to_vec().unwrap(), vec![30, 10, 20, 40]);

    let values = policy.to_device(&[2.0_f32, 1.0, 4.0, 3.0]).unwrap();
    let tags = policy.to_device(&[20_u32, 10, 20, 10]).unwrap();
    let payload = policy.to_device(&[200.0_f32, 100.0, 400.0, 300.0]).unwrap();

    let sorted = sort(zip3(values, tags, payload), MixedTuple3Less).unwrap();
    let (values, tags, payload) = unzip(sorted).unwrap();
    assert_eq!(values.to_vec().unwrap(), vec![1.0, 3.0, 2.0, 4.0]);
    assert_eq!(tags.to_vec().unwrap(), vec![10, 10, 20, 20]);
    assert_eq!(payload.to_vec().unwrap(), vec![100.0, 300.0, 200.0, 400.0]);
}

#[test]
fn tuple_sort_accepts_wide_device_soas() {
    let policy = policy();
    let a = policy.to_device(&[3.0_f32, 1.0, 2.0]).unwrap();
    let b = policy.to_device(&[30.0_f32, 10.0, 20.0]).unwrap();
    let c = policy.to_device(&[300.0_f32, 100.0, 200.0]).unwrap();
    let d = policy.to_device(&[3000.0_f32, 1000.0, 2000.0]).unwrap();

    let sorted = sort(zip4(a, b, c, d), Tuple4Less).unwrap();
    let (a, b, c, d) = unzip(sorted).unwrap();

    assert_eq!(a.to_vec().unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(b.to_vec().unwrap(), vec![10.0, 20.0, 30.0]);
    assert_eq!(c.to_vec().unwrap(), vec![100.0, 200.0, 300.0]);
    assert_eq!(d.to_vec().unwrap(), vec![1000.0, 2000.0, 3000.0]);
}

#[test]
fn tuple_sort_accepts_soa12() {
    let policy = policy();
    let a = policy.to_device(&[3.0_f32, 1.0, 2.0]).unwrap();
    let b = policy.to_device(&[30.0_f32, 10.0, 20.0]).unwrap();
    let c = policy.to_device(&[30.0_f32, 10.0, 20.0]).unwrap();
    let d = policy.to_device(&[30.0_f32, 10.0, 20.0]).unwrap();
    let e = policy.to_device(&[30.0_f32, 10.0, 20.0]).unwrap();
    let f = policy.to_device(&[30.0_f32, 10.0, 20.0]).unwrap();
    let g = policy.to_device(&[30.0_f32, 10.0, 20.0]).unwrap();
    let h = policy.to_device(&[30.0_f32, 10.0, 20.0]).unwrap();
    let i = policy.to_device(&[30.0_f32, 10.0, 20.0]).unwrap();
    let j = policy.to_device(&[30.0_f32, 10.0, 20.0]).unwrap();
    let k = policy.to_device(&[30.0_f32, 10.0, 20.0]).unwrap();
    let l = policy.to_device(&[3000.0_f32, 1000.0, 2000.0]).unwrap();

    let sorted = sort(zip12(a, b, c, d, e, f, g, h, i, j, k, l), Tuple12Less).unwrap();
    let (a, _b, _c, _d, _e, _f, _g, _h, _i, _j, _k, l) = unzip(sorted).unwrap();

    assert_eq!(a.to_vec().unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(l.to_vec().unwrap(), vec![1000.0, 2000.0, 3000.0]);
}

#[test]
fn sort_by_key_accepts_wide_device_soas() {
    let policy = policy();
    let keys = policy.to_device(&[2_u32, 0, 1]).unwrap();
    let a = policy.to_device(&[20.0_f32, 0.0, 10.0]).unwrap();
    let b = policy.to_device(&[200_u32, 0, 100]).unwrap();
    let c = policy.to_device(&[2000.0_f32, 0.0, 1000.0]).unwrap();
    let d = policy.to_device(&[20000_u32, 0, 10000]).unwrap();

    let (keys, values) = sort_by_key(&keys, zip4(a, b, c, d), LessU32).unwrap();
    let keys = unzip(keys).unwrap();
    let (a, b, c, d) = unzip(values).unwrap();

    assert_eq!(keys.to_vec().unwrap(), vec![0, 1, 2]);
    assert_eq!(a.to_vec().unwrap(), vec![0.0, 10.0, 20.0]);
    assert_eq!(b.to_vec().unwrap(), vec![0, 100, 200]);
    assert_eq!(c.to_vec().unwrap(), vec![0.0, 1000.0, 2000.0]);
    assert_eq!(d.to_vec().unwrap(), vec![0, 10000, 20000]);
}

#[test]
fn sort_by_key_accepts_soa12_values() {
    let policy = policy();
    let keys = policy.to_device(&[2_u32, 0, 1]).unwrap();
    let a = policy.to_device(&[20.0_f32, 0.0, 10.0]).unwrap();
    let b = policy.to_device(&[20_u32, 0, 10]).unwrap();
    let c = policy.to_device(&[20.0_f32, 0.0, 10.0]).unwrap();
    let d = policy.to_device(&[20_u32, 0, 10]).unwrap();
    let e = policy.to_device(&[20.0_f32, 0.0, 10.0]).unwrap();
    let f = policy.to_device(&[20_u32, 0, 10]).unwrap();
    let g = policy.to_device(&[20.0_f32, 0.0, 10.0]).unwrap();
    let h = policy.to_device(&[20_u32, 0, 10]).unwrap();
    let i = policy.to_device(&[20.0_f32, 0.0, 10.0]).unwrap();
    let j = policy.to_device(&[20_u32, 0, 10]).unwrap();
    let k = policy.to_device(&[20.0_f32, 0.0, 10.0]).unwrap();
    let l = policy.to_device(&[20_u32, 0, 10]).unwrap();

    let (_keys, values) =
        sort_by_key(&keys, zip12(a, b, c, d, e, f, g, h, i, j, k, l), LessU32).unwrap();
    let (_a, _b, _c, _d, _e, _f, _g, _h, _i, _j, _k, l) = unzip(values).unwrap();

    assert_eq!(l.to_vec().unwrap(), vec![0, 10, 20]);
}
