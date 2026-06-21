use crate::common::*;

#[test]
fn device_slice_to_vec_uses_range() {
    let policy = policy();
    let input = policy.to_device(&[10_u32, 20, 30, 40, 50]).unwrap();

    assert_eq!(input.slice(1..4).to_vec().unwrap(), vec![20, 30, 40]);
    assert_eq!(input.slice(..2).to_vec().unwrap(), vec![10, 20]);
    assert_eq!(input.slice(3..).to_vec().unwrap(), vec![40, 50]);
    assert_eq!(input.slice(..).to_vec().unwrap(), vec![10, 20, 30, 40, 50]);
}

#[test]
fn transform_accepts_device_slice() {
    let policy = policy();
    let input = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();

    let (output,) = transform((input.slice(1..3),), Double).unwrap();

    assert_eq!(output.to_vec().unwrap(), vec![4.0, 6.0]);
}

#[test]
fn reduce_accepts_device_slice() {
    let policy = policy();
    let input = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();

    let sum = reduce((input.slice(1..),), (0.0_f32,), TupleSum).unwrap();

    assert_eq!(sum, (9.0,));
}

#[test]
fn inclusive_scan_accepts_device_slice() {
    let policy = policy();
    let input = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();

    let (output,) = inclusive_scan((input.slice(1..4),), TupleSum).unwrap();

    assert_eq!(output.to_vec().unwrap(), vec![2.0, 5.0, 9.0]);
}

#[test]
fn transform_accepts_multi_column_device_slices() {
    let policy = policy();
    let values = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let tags = policy.to_device(&[10_u32, 20, 30, 40]).unwrap();

    let (values, tags) = transform((values.slice(1..4), tags.slice(1..4)), PairMixedSplit).unwrap();

    assert_eq!(values.to_vec().unwrap(), vec![12.0, 13.0, 14.0]);
    assert_eq!(tags.to_vec().unwrap(), vec![21, 31, 41]);
}

#[test]
fn reduce_accepts_multi_column_device_slices() {
    let policy = policy();
    let values = policy.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let tags = policy.to_device(&[10_u32, 20, 30, 40]).unwrap();

    let sum = reduce(
        (values.slice(1..4), tags.slice(1..4)),
        (0.0_f32, 0_u32),
        TupleSum,
    )
    .unwrap();

    assert_eq!(sum, (9.0, 90));
}

#[test]
fn reverse_accepts_multi_column_device_slices() {
    let policy = policy();
    let values = policy.to_device(&[0.0_f32, 1.0, 2.0, 3.0, 99.0]).unwrap();
    let tags = policy.to_device(&[0_u32, 10, 20, 30, 99]).unwrap();

    let (values, tags) = reverse((values.slice(1..4), tags.slice(1..4))).unwrap();

    assert_eq!(values.to_vec().unwrap(), vec![3.0, 2.0, 1.0]);
    assert_eq!(tags.to_vec().unwrap(), vec![30, 20, 10]);
}

#[test]
fn sort_accepts_multi_column_device_slices() {
    let policy = policy();
    let values = policy.to_device(&[99.0_f32, 2.0, 1.0, 2.0, 88.0]).unwrap();
    let tags = policy.to_device(&[99_u32, 20, 30, 10, 88]).unwrap();

    let (values, tags) = sort((values.slice(1..4), tags.slice(1..4)), MixedTupleLess).unwrap();

    assert_eq!(values.to_vec().unwrap(), vec![1.0, 2.0, 2.0]);
    assert_eq!(tags.to_vec().unwrap(), vec![30, 10, 20]);
}

#[test]
fn sort_accepts_offset_device_slice() {
    let policy = policy();
    let values = policy
        .to_device(&[999.0_f32, 4.0, 1.0, 3.0, 2.0, 888.0])
        .unwrap();

    let (values,) = sort((values.slice(1..5),), Less).unwrap();

    assert_eq!(values.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn gather_accepts_device_slice_indices() {
    let policy = policy();
    let values = policy.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let indices = policy.to_device(&[99_u32, 3, 1, 0, 88]).unwrap();

    let (output,) = gather((values.slice(..),), (indices.slice(1..4),)).unwrap();

    assert_eq!(output.to_vec().unwrap(), vec![40, 20, 10]);
}

#[test]
fn equal_accepts_device_slices() {
    let policy = policy();
    let left = policy.to_device(&[0_u32, 10, 20, 30, 99]).unwrap();
    let right = policy.to_device(&[10_u32, 20, 30]).unwrap();

    assert!(equal((left.slice(1..4),), (right.slice(..),), EqualU32).unwrap());
}

#[test]
fn merge_accepts_device_slices() {
    let policy = policy();
    let left = policy.to_device(&[0_u32, 1, 3, 99]).unwrap();
    let right = policy.to_device(&[2_u32, 4, 88]).unwrap();

    let (output,) = merge((left.slice(1..3),), (right.slice(..2),), LessU32).unwrap();

    assert_eq!(output.to_vec().unwrap(), vec![1, 2, 3, 4]);
}

#[test]
fn inclusive_scan_by_key_accepts_device_slice_keys_and_values() {
    let policy = policy();
    let keys = policy.to_device(&[9_u32, 1, 1, 2, 2, 8]).unwrap();
    let values = policy.to_device(&[99_u32, 10, 20, 1, 2, 88]).unwrap();

    let (output,) =
        inclusive_scan_by_key((keys.slice(1..5),), (values.slice(1..5),), EqualU32, Sum).unwrap();

    assert_eq!(output.to_vec().unwrap(), vec![10, 30, 1, 3]);
}

#[test]
fn sort_by_key_accepts_device_slice_keys_and_values() {
    let policy = policy();
    let keys = policy.to_device(&[99_u32, 3, 1, 2, 88]).unwrap();
    let values = policy.to_device(&[99_u32, 30, 10, 20, 88]).unwrap();

    let ((keys,), (values,)) =
        sort_by_key((keys.slice(1..4),), (values.slice(1..4),), LessU32).unwrap();

    assert_eq!(keys.to_vec().unwrap(), vec![1, 2, 3]);
    assert_eq!(values.to_vec().unwrap(), vec![10, 20, 30]);
}

#[test]
fn sort_by_key_accepts_multi_column_device_slice_values() {
    let policy = policy();
    let keys = policy.to_device(&[99_u32, 3, 1, 2, 88]).unwrap();
    let values = policy
        .to_device(&[99.0_f32, 30.0, 10.0, 20.0, 88.0])
        .unwrap();
    let tags = policy.to_device(&[99_u32, 300, 100, 200, 88]).unwrap();

    let ((keys,), (values, tags)) = sort_by_key(
        (keys.slice(1..4),),
        (values.slice(1..4), tags.slice(1..4)),
        LessU32,
    )
    .unwrap();

    assert_eq!(keys.to_vec().unwrap(), vec![1, 2, 3]);
    assert_eq!(values.to_vec().unwrap(), vec![10.0, 20.0, 30.0]);
    assert_eq!(tags.to_vec().unwrap(), vec![100, 200, 300]);
}

#[test]
fn unique_by_key_accepts_multi_column_device_slice_values() {
    let policy = policy();
    let keys = policy.to_device(&[99_u32, 0, 0, 1, 1, 88]).unwrap();
    let values = policy
        .to_device(&[99.0_f32, 10.0, 20.0, 30.0, 40.0, 88.0])
        .unwrap();
    let tags = policy.to_device(&[99_u32, 100, 200, 300, 400, 88]).unwrap();

    let ((keys,), (values, tags)) = unique_by_key(
        (keys.slice(1..5),),
        (values.slice(1..5), tags.slice(1..5)),
        EqualU32,
    )
    .unwrap();

    assert_eq!(keys.to_vec().unwrap(), vec![0, 1]);
    assert_eq!(values.to_vec().unwrap(), vec![10.0, 30.0]);
    assert_eq!(tags.to_vec().unwrap(), vec![100, 300]);
}

#[test]
fn inclusive_scan_by_key_accepts_multi_column_device_slice_values() {
    let policy = policy();
    let keys = policy.to_device(&[99_u32, 0, 0, 1, 1, 88]).unwrap();
    let values = policy
        .to_device(&[99.0_f32, 1.0, 2.0, 3.0, 4.0, 88.0])
        .unwrap();
    let tags = policy.to_device(&[99_u32, 10, 20, 30, 40, 88]).unwrap();

    let (values, tags) = inclusive_scan_by_key(
        (keys.slice(1..5),),
        (values.slice(1..5), tags.slice(1..5)),
        EqualU32,
        TupleSum,
    )
    .unwrap();

    assert_eq!(values.to_vec().unwrap(), vec![1.0, 3.0, 3.0, 7.0]);
    assert_eq!(tags.to_vec().unwrap(), vec![10, 30, 30, 70]);
}

#[test]
fn exclusive_scan_by_key_accepts_multi_column_device_slice_values() {
    let policy = policy();
    let keys = policy.to_device(&[99_u32, 0, 0, 1, 1, 88]).unwrap();
    let values = policy
        .to_device(&[99.0_f32, 1.0, 2.0, 3.0, 4.0, 88.0])
        .unwrap();
    let tags = policy.to_device(&[99_u32, 10, 20, 30, 40, 88]).unwrap();

    let (values, tags) = exclusive_scan_by_key(
        (keys.slice(1..5),),
        (values.slice(1..5), tags.slice(1..5)),
        EqualU32,
        (0.0_f32, 0_u32),
        TupleSum,
    )
    .unwrap();

    assert_eq!(values.to_vec().unwrap(), vec![0.0, 1.0, 0.0, 3.0]);
    assert_eq!(tags.to_vec().unwrap(), vec![0, 10, 0, 30]);
}

#[test]
fn reduce_by_key_accepts_multi_column_device_slice_values() {
    let policy = policy();
    let keys = policy.to_device(&[99_u32, 0, 0, 1, 1, 88]).unwrap();
    let values = policy
        .to_device(&[99.0_f32, 1.0, 2.0, 3.0, 4.0, 88.0])
        .unwrap();
    let tags = policy.to_device(&[99_u32, 10, 20, 30, 40, 88]).unwrap();

    let ((keys,), (values, tags)) = reduce_by_key(
        (keys.slice(1..5),),
        (values.slice(1..5), tags.slice(1..5)),
        EqualU32,
        (0.0_f32, 0_u32),
        TupleSum,
    )
    .unwrap();

    assert_eq!(keys.to_vec().unwrap(), vec![0, 1]);
    assert_eq!(values.to_vec().unwrap(), vec![3.0, 7.0]);
    assert_eq!(tags.to_vec().unwrap(), vec![30, 70]);
}

#[test]
fn copy_if_accepts_device_slice_stencil() {
    let policy = policy();
    let values = policy.to_device(&[10_u32, 20, 30, 40, 50]).unwrap();
    let stencil = policy.to_device(&[0_u32, 1, 0, 1, 0]).unwrap();

    let (output,) = copy_if((values.slice(1..4),), (stencil.slice(1..4),)).unwrap();

    assert_eq!(output.to_vec().unwrap(), vec![20, 40]);
}

#[test]
fn remove_if_accepts_device_slice_input() {
    let policy = policy();
    let values = policy.to_device(&[0_u32, 10, 20, 30, 99]).unwrap();

    let (output,) = remove_if((values.slice(1..4),), U32IsTwenty).unwrap();

    assert_eq!(output.to_vec().unwrap(), vec![10, 30]);
}

#[test]
fn replace_if_accepts_device_slice_stencil() {
    let policy = policy();
    let values = policy.to_device(&[10_u32, 20, 30, 40, 50]).unwrap();
    let stencil = policy.to_device(&[0_u32, 1, 0, 1, 0]).unwrap();

    let (output,) = replace_if((values.slice(1..4),), (99_u32,), (stencil.slice(1..4),)).unwrap();

    assert_eq!(output.to_vec().unwrap(), vec![99, 30, 99]);
}

#[test]
fn scatter_if_accepts_device_slice_indices_and_stencil() {
    let policy = policy();
    let values = policy.to_device(&[99_u32, 10, 20, 30, 88]).unwrap();
    let indices = policy.to_device(&[99_u32, 2, 1, 0, 88]).unwrap();
    let stencil = policy.to_device(&[0_u32, 1, 0, 1, 0]).unwrap();

    let (output,) = scatter_if(
        (values.slice(1..4),),
        (indices.slice(1..4),),
        3,
        (0_u32,),
        (stencil.slice(1..4),),
    )
    .unwrap();

    assert_eq!(output.to_vec().unwrap(), vec![30, 0, 10]);
}

#[test]
fn transform_accepts_three_column_device_slices() {
    let policy = policy();
    let a = policy.to_device(&[0.0_f32, 1.0, 2.0, 3.0, 99.0]).unwrap();
    let b = policy.to_device(&[0_u32, 10, 20, 30, 99]).unwrap();
    let c = policy
        .to_device(&[0.0_f32, 100.0, 200.0, 300.0, 99.0])
        .unwrap();

    let (a, b, c) = transform(
        (a.slice(1..4), b.slice(1..4), c.slice(1..4)),
        Tuple3MixedSplit,
    )
    .unwrap();

    assert_eq!(a.to_vec().unwrap(), vec![101.0, 202.0, 303.0]);
    assert_eq!(b.to_vec().unwrap(), vec![11, 21, 31]);
    assert_eq!(c.to_vec().unwrap(), vec![101.0, 202.0, 303.0]);
}

#[test]
fn empty_device_slice_is_valid_input() {
    let policy = policy();
    let values = policy.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();

    let slice = values.slice(1..1);
    let (output,) = transform((slice,), Double).unwrap();
    let sum = reduce((slice,), (0.0_f32,), TupleSum).unwrap();

    assert!(slice.is_empty());
    assert_eq!(slice.to_vec().unwrap(), Vec::<f32>::new());
    assert_eq!(output.to_vec().unwrap(), Vec::<f32>::new());
    assert_eq!(sum, (0.0,));
}

#[test]
#[should_panic(expected = "slice end (4) is out of bounds for DeviceVec of length 3")]
fn device_slice_range_end_panics_when_out_of_bounds() {
    let policy = policy();
    let values = policy.to_device(&[1_u32, 2, 3]).unwrap();

    let _ = values.slice(..4);
}

#[test]
#[should_panic(expected = "slice start (3) is greater than slice end (2)")]
fn device_slice_range_panics_when_start_is_after_end() {
    let policy = policy();
    let values = policy.to_device(&[1_u32, 2, 3]).unwrap();

    let _ = values.slice(3..2);
}
