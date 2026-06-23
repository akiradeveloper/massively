use crate::common::*;

#[test]
fn executor_to_host_accepts_device_slice() {
    let exec = exec();
    let input = exec.to_device(&[10_u32, 20, 30, 40, 50]).unwrap();

    assert_eq!(exec.to_host(&input.slice(1..4)).unwrap(), vec![20, 30, 40]);
    assert_eq!(exec.to_host(&input.slice(..2)).unwrap(), vec![10, 20]);
    assert_eq!(exec.to_host(&input.slice(3..)).unwrap(), vec![40, 50]);
    assert_eq!(
        exec.to_host(&input.slice(..)).unwrap(),
        vec![10, 20, 30, 40, 50]
    );
}

#[test]
fn executor_to_host_rejects_other_executor_data() {
    let data_exec = exec();
    let other_exec = exec();
    let input = data_exec.to_device(&[10_u32, 20, 30]).unwrap();

    assert!(other_exec.to_host(&input).is_err());
    assert!(other_exec.to_host(&input.slice(..)).is_err());
}

#[test]
fn executor_filled_allocates_owned_device_vec() {
    let exec = exec();
    let input = exec.filled(4, 7_u32).unwrap();

    assert_eq!(exec.to_host(&input).unwrap(), vec![7, 7, 7, 7]);
}

#[test]
fn device_slice_can_be_sliced_again() {
    let exec = exec();
    let input = exec.to_device(&[10_u32, 20, 30, 40, 50, 60]).unwrap();
    let slice = input.slice(1..5);

    assert_eq!(exec.to_host(&slice.slice(1..3)).unwrap(), vec![30, 40]);
    assert_eq!(exec.to_host(&slice.slice(..2)).unwrap(), vec![20, 30]);
    assert_eq!(exec.to_host(&slice.slice(2..)).unwrap(), vec![40, 50]);
}

#[test]
fn device_slice_mut_can_be_sliced_as_read_only() {
    let exec = exec();
    let mut input = exec.to_device(&[10_u32, 20, 30, 40, 50, 60]).unwrap();
    let slice = input.slice_mut(1..5);

    assert_eq!(exec.to_host(&slice.slice(1..3)).unwrap(), vec![30, 40]);
}

#[test]
fn executor_copy_copies_between_device_slices() {
    let exec = exec();
    let input = exec.to_device(&[10_u32, 20, 30, 40, 50]).unwrap();
    let mut output = exec.to_device(&[1_u32, 2, 3, 4, 5, 6]).unwrap();

    exec.copy(input.slice(1..4), output.slice_mut(2..5))
        .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![1, 2, 20, 30, 40, 6]);
}

#[test]
fn executor_copy_accepts_nested_mutable_destination_slice() {
    let exec = exec();
    let input = exec.to_device(&[7_u32, 8, 9]).unwrap();
    let mut output = exec.to_device(&[0_u32, 1, 2, 3, 4, 5]).unwrap();

    {
        let mut middle = output.slice_mut(1..5);
        exec.copy(input.slice(..2), middle.slice_mut(1..3)).unwrap();
    }

    assert_eq!(exec.to_host(&output).unwrap(), vec![0, 1, 7, 8, 4, 5]);
}

#[test]
fn executor_copy_rejects_mismatched_lengths() {
    let exec = exec();
    let input = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let mut output = exec.to_device(&[0_u32, 0]).unwrap();

    assert!(exec.copy(input.slice(..), output.slice_mut(..)).is_err());
}

#[test]
fn executor_copy_rejects_other_executor_data() {
    let data_exec = exec();
    let other_exec = exec();
    let input = data_exec.to_device(&[10_u32, 20]).unwrap();
    let mut output = data_exec.to_device(&[0_u32, 0]).unwrap();

    assert!(
        other_exec
            .copy(input.slice(..), output.slice_mut(..))
            .is_err()
    );
}

#[test]
fn algorithms_reject_other_executor_data() {
    let data_exec = exec();
    let other_exec = exec();
    let input = data_exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();

    let result = transform::<_, _, (massively::DeviceVec<Wgpu, f32>,), _>(
        &other_exec,
        massively::SoA1(input.slice(..)),
        Double,
    );

    assert!(result.is_err());
}

#[test]
fn transform_accepts_device_slice() {
    let exec = exec();
    let input = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();

    let (output,) = transform(&exec, massively::SoA1(input.slice(1..3)), Double).unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![4.0, 6.0]);
}

#[test]
fn reduce_accepts_device_slice() {
    let exec = exec();
    let input = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();

    let sum = reduce(
        &exec,
        massively::SoA1(input.slice(1..)),
        (0.0_f32,),
        TupleSum,
    )
    .unwrap();

    assert_eq!(sum, (9.0,));
}

#[test]
fn inclusive_scan_accepts_device_slice() {
    let exec = exec();
    let input = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();

    let (output,) = inclusive_scan(&exec, massively::SoA1(input.slice(1..4)), TupleSum).unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![2.0, 5.0, 9.0]);
}

#[test]
fn transform_accepts_multi_column_device_slices() {
    let exec = exec();
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let tags = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();

    let (values, tags) = transform(
        &exec,
        massively::SoA2(values.slice(1..4), tags.slice(1..4)),
        PairMixedSplit,
    )
    .unwrap();

    assert_eq!(exec.to_host(&values).unwrap(), vec![12.0, 13.0, 14.0]);
    assert_eq!(exec.to_host(&tags).unwrap(), vec![21, 31, 41]);
}

#[test]
fn reduce_accepts_multi_column_device_slices() {
    let exec = exec();
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let tags = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();

    let sum = reduce(
        &exec,
        massively::SoA2(values.slice(1..4), tags.slice(1..4)),
        (0.0_f32, 0_u32),
        TupleSum,
    )
    .unwrap();

    assert_eq!(sum, (9.0, 90));
}

#[test]
fn reverse_accepts_multi_column_device_slices() {
    let exec = exec();
    let values = exec.to_device(&[0.0_f32, 1.0, 2.0, 3.0, 99.0]).unwrap();
    let tags = exec.to_device(&[0_u32, 10, 20, 30, 99]).unwrap();

    let (values, tags) =
        reverse(&exec, massively::SoA2(values.slice(1..4), tags.slice(1..4))).unwrap();

    assert_eq!(exec.to_host(&values).unwrap(), vec![3.0, 2.0, 1.0]);
    assert_eq!(exec.to_host(&tags).unwrap(), vec![30, 20, 10]);
}

#[test]
fn sort_accepts_multi_column_device_slices() {
    let exec = exec();
    let values = exec.to_device(&[99.0_f32, 2.0, 1.0, 2.0, 88.0]).unwrap();
    let tags = exec.to_device(&[99_u32, 20, 30, 10, 88]).unwrap();

    let (values, tags) = sort(
        &exec,
        massively::SoA2(values.slice(1..4), tags.slice(1..4)),
        MixedTupleLess,
    )
    .unwrap();

    assert_eq!(exec.to_host(&values).unwrap(), vec![1.0, 2.0, 2.0]);
    assert_eq!(exec.to_host(&tags).unwrap(), vec![30, 10, 20]);
}

#[test]
fn sort_accepts_offset_device_slice() {
    let exec = exec();
    let values = exec
        .to_device(&[999.0_f32, 4.0, 1.0, 3.0, 2.0, 888.0])
        .unwrap();

    let (values,) = sort(&exec, massively::SoA1(values.slice(1..5)), Less).unwrap();

    assert_eq!(exec.to_host(&values).unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn gather_accepts_device_slice_indices() {
    let exec = exec();
    let values = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let indices = exec.to_device(&[99_u32, 3, 1, 0, 88]).unwrap();

    let (output,) = gather(
        &exec,
        massively::SoA1(values.slice(..)),
        indices.slice(1..4),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![40, 20, 10]);
}

#[test]
fn gather_if_accepts_offset_device_slices() {
    let exec = exec();
    let values = exec.to_device(&[99_u32, 10, 20, 30, 40, 88]).unwrap();
    let indices = exec.to_device(&[77_u32, 3, 1, 0, 2, 66]).unwrap();
    let stencil = exec.to_device(&[0_u32, 1, 0, 1, 0, 0]).unwrap();

    let (output,) = gather_if(
        &exec,
        massively::SoA1(values.slice(1..5)),
        indices.slice(1..5),
        (0_u32,),
        stencil.slice(1..5),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![40, 0, 10, 0]);
}

#[test]
fn equal_accepts_device_slices() {
    let exec = exec();
    let left = exec.to_device(&[0_u32, 10, 20, 30, 99]).unwrap();
    let right = exec.to_device(&[10_u32, 20, 30]).unwrap();

    assert!(
        equal(
            &exec,
            massively::SoA1(left.slice(1..4)),
            massively::SoA1(right.slice(..)),
            EqualU32
        )
        .unwrap()
    );
}

#[test]
fn merge_accepts_device_slices() {
    let exec = exec();
    let left = exec.to_device(&[0_u32, 1, 3, 99]).unwrap();
    let right = exec.to_device(&[2_u32, 4, 88]).unwrap();

    let (output,) = merge(
        &exec,
        massively::SoA1(left.slice(1..3)),
        massively::SoA1(right.slice(..2)),
        LessU32,
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![1, 2, 3, 4]);
}

#[test]
fn merge_by_key_accepts_offset_device_slices_with_tuple_values() {
    let exec = exec();
    let left_keys = exec.to_device(&[99_u32, 1, 3, 88]).unwrap();
    let left_a = exec.to_device(&[999.0_f32, 100.0, 300.0, 888.0]).unwrap();
    let left_b = exec.to_device(&[999_u32, 10, 30, 888]).unwrap();
    let right_keys = exec.to_device(&[77_u32, 2, 4, 66]).unwrap();
    let right_a = exec.to_device(&[777.0_f32, 200.0, 400.0, 666.0]).unwrap();
    let right_b = exec.to_device(&[777_u32, 20, 40, 666]).unwrap();

    let ((keys,), (a, b)) = merge_by_key(
        &exec,
        massively::SoA1(left_keys.slice(1..3)),
        massively::SoA2(left_a.slice(1..3), left_b.slice(1..3)),
        massively::SoA1(right_keys.slice(1..3)),
        massively::SoA2(right_a.slice(1..3), right_b.slice(1..3)),
        LessU32,
    )
    .unwrap();

    assert_eq!(exec.to_host(&keys).unwrap(), vec![1, 2, 3, 4]);
    assert_eq!(exec.to_host(&a).unwrap(), vec![100.0, 200.0, 300.0, 400.0]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![10, 20, 30, 40]);
}

#[test]
fn tuple_set_algorithms_accept_offset_device_slices() {
    let exec = exec();
    let left_a = exec
        .to_device(&[99.0_f32, 1.0, 2.0, 2.0, 4.0, 88.0])
        .unwrap();
    let left_b = exec.to_device(&[99_u32, 10, 20, 21, 40, 88]).unwrap();
    let right_a = exec.to_device(&[77.0_f32, 2.0, 3.0, 4.0, 66.0]).unwrap();
    let right_b = exec.to_device(&[77_u32, 20, 30, 40, 66]).unwrap();

    let (union_a, union_b) = set_union(
        &exec,
        massively::SoA2(left_a.slice(1..5), left_b.slice(1..5)),
        massively::SoA2(right_a.slice(1..4), right_b.slice(1..4)),
        MixedTupleLess,
    )
    .unwrap();
    let (intersection_a, intersection_b) = set_intersection(
        &exec,
        massively::SoA2(left_a.slice(1..5), left_b.slice(1..5)),
        massively::SoA2(right_a.slice(1..4), right_b.slice(1..4)),
        MixedTupleLess,
    )
    .unwrap();
    let (difference_a, difference_b) = set_difference(
        &exec,
        massively::SoA2(left_a.slice(1..5), left_b.slice(1..5)),
        massively::SoA2(right_a.slice(1..4), right_b.slice(1..4)),
        MixedTupleLess,
    )
    .unwrap();

    assert_eq!(
        exec.to_host(&union_a).unwrap(),
        vec![1.0, 2.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(exec.to_host(&union_b).unwrap(), vec![10, 20, 21, 30, 40]);
    assert_eq!(exec.to_host(&intersection_a).unwrap(), vec![2.0, 4.0]);
    assert_eq!(exec.to_host(&intersection_b).unwrap(), vec![20, 40]);
    assert_eq!(exec.to_host(&difference_a).unwrap(), vec![1.0, 2.0]);
    assert_eq!(exec.to_host(&difference_b).unwrap(), vec![10, 21]);
}

#[test]
fn minmax_element_accepts_offset_device_slice() {
    let exec = exec();
    let values = exec
        .to_device(&[99.0_f32, 4.0, 1.0, 3.0, 5.0, 88.0])
        .unwrap();

    let result = minmax_element(&exec, massively::SoA1(values.slice(1..5)), Less).unwrap();

    assert_eq!(result, Some((1, 3)));
}

#[test]
fn tuple_minmax_element_accepts_offset_device_slices() {
    let exec = exec();
    let values = exec
        .to_device(&[99.0_f32, 3.0, 1.0, 3.0, 4.0, 88.0])
        .unwrap();
    let tags = exec.to_device(&[99_u32, 30, 10, 20, 40, 88]).unwrap();

    let result = minmax_element(
        &exec,
        massively::SoA2(values.slice(1..5), tags.slice(1..5)),
        MixedTupleLess,
    )
    .unwrap();

    assert_eq!(result, Some((1, 3)));
}

#[test]
fn inclusive_scan_by_key_accepts_device_slice_keys_and_values() {
    let exec = exec();
    let keys = exec.to_device(&[9_u32, 1, 1, 2, 2, 8]).unwrap();
    let values = exec.to_device(&[99_u32, 10, 20, 1, 2, 88]).unwrap();

    let (output,) = inclusive_scan_by_key(
        &exec,
        massively::SoA1(keys.slice(1..5)),
        massively::SoA1(values.slice(1..5)),
        EqualU32,
        Sum,
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![10, 30, 1, 3]);
}

#[test]
fn sort_by_key_accepts_device_slice_keys_and_values() {
    let exec = exec();
    let keys = exec.to_device(&[99_u32, 3, 1, 2, 88]).unwrap();
    let values = exec.to_device(&[99_u32, 30, 10, 20, 88]).unwrap();

    let ((keys,), (values,)) = sort_by_key(
        &exec,
        massively::SoA1(keys.slice(1..4)),
        massively::SoA1(values.slice(1..4)),
        LessU32,
    )
    .unwrap();

    assert_eq!(exec.to_host(&keys).unwrap(), vec![1, 2, 3]);
    assert_eq!(exec.to_host(&values).unwrap(), vec![10, 20, 30]);
}

#[test]
fn sort_by_key_accepts_multi_column_device_slice_values() {
    let exec = exec();
    let keys = exec.to_device(&[99_u32, 3, 1, 2, 88]).unwrap();
    let values = exec.to_device(&[99.0_f32, 30.0, 10.0, 20.0, 88.0]).unwrap();
    let tags = exec.to_device(&[99_u32, 300, 100, 200, 88]).unwrap();

    let ((keys,), (values, tags)) = sort_by_key(
        &exec,
        massively::SoA1(keys.slice(1..4)),
        massively::SoA2(values.slice(1..4), tags.slice(1..4)),
        LessU32,
    )
    .unwrap();

    assert_eq!(exec.to_host(&keys).unwrap(), vec![1, 2, 3]);
    assert_eq!(exec.to_host(&values).unwrap(), vec![10.0, 20.0, 30.0]);
    assert_eq!(exec.to_host(&tags).unwrap(), vec![100, 200, 300]);
}

#[test]
fn unique_by_key_accepts_multi_column_device_slice_values() {
    let exec = exec();
    let keys = exec.to_device(&[99_u32, 0, 0, 1, 1, 88]).unwrap();
    let values = exec
        .to_device(&[99.0_f32, 10.0, 20.0, 30.0, 40.0, 88.0])
        .unwrap();
    let tags = exec.to_device(&[99_u32, 100, 200, 300, 400, 88]).unwrap();

    let ((keys,), (values, tags)) = unique_by_key(
        &exec,
        massively::SoA1(keys.slice(1..5)),
        massively::SoA2(values.slice(1..5), tags.slice(1..5)),
        EqualU32,
    )
    .unwrap();

    assert_eq!(exec.to_host(&keys).unwrap(), vec![0, 1]);
    assert_eq!(exec.to_host(&values).unwrap(), vec![10.0, 30.0]);
    assert_eq!(exec.to_host(&tags).unwrap(), vec![100, 300]);
}

#[test]
fn inclusive_scan_by_key_accepts_multi_column_device_slice_values() {
    let exec = exec();
    let keys = exec.to_device(&[99_u32, 0, 0, 1, 1, 88]).unwrap();
    let values = exec
        .to_device(&[99.0_f32, 1.0, 2.0, 3.0, 4.0, 88.0])
        .unwrap();
    let tags = exec.to_device(&[99_u32, 10, 20, 30, 40, 88]).unwrap();

    let (values, tags) = inclusive_scan_by_key(
        &exec,
        massively::SoA1(keys.slice(1..5)),
        massively::SoA2(values.slice(1..5), tags.slice(1..5)),
        EqualU32,
        TupleSum,
    )
    .unwrap();

    assert_eq!(exec.to_host(&values).unwrap(), vec![1.0, 3.0, 3.0, 7.0]);
    assert_eq!(exec.to_host(&tags).unwrap(), vec![10, 30, 30, 70]);
}

#[test]
fn exclusive_scan_by_key_accepts_multi_column_device_slice_values() {
    let exec = exec();
    let keys = exec.to_device(&[99_u32, 0, 0, 1, 1, 88]).unwrap();
    let values = exec
        .to_device(&[99.0_f32, 1.0, 2.0, 3.0, 4.0, 88.0])
        .unwrap();
    let tags = exec.to_device(&[99_u32, 10, 20, 30, 40, 88]).unwrap();

    let (values, tags) = exclusive_scan_by_key(
        &exec,
        massively::SoA1(keys.slice(1..5)),
        massively::SoA2(values.slice(1..5), tags.slice(1..5)),
        EqualU32,
        (0.0_f32, 0_u32),
        TupleSum,
    )
    .unwrap();

    assert_eq!(exec.to_host(&values).unwrap(), vec![0.0, 1.0, 0.0, 3.0]);
    assert_eq!(exec.to_host(&tags).unwrap(), vec![0, 10, 0, 30]);
}

#[test]
fn reduce_by_key_accepts_multi_column_device_slice_values() {
    let exec = exec();
    let keys = exec.to_device(&[99_u32, 0, 0, 1, 1, 88]).unwrap();
    let values = exec
        .to_device(&[99.0_f32, 1.0, 2.0, 3.0, 4.0, 88.0])
        .unwrap();
    let tags = exec.to_device(&[99_u32, 10, 20, 30, 40, 88]).unwrap();

    let ((keys,), (values, tags)) = reduce_by_key(
        &exec,
        massively::SoA1(keys.slice(1..5)),
        massively::SoA2(values.slice(1..5), tags.slice(1..5)),
        EqualU32,
        (0.0_f32, 0_u32),
        TupleSum,
    )
    .unwrap();

    assert_eq!(exec.to_host(&keys).unwrap(), vec![0, 1]);
    assert_eq!(exec.to_host(&values).unwrap(), vec![3.0, 7.0]);
    assert_eq!(exec.to_host(&tags).unwrap(), vec![30, 70]);
}

#[test]
fn copy_if_accepts_device_slice_stencil() {
    let exec = exec();
    let values = exec.to_device(&[10_u32, 20, 30, 40, 50]).unwrap();
    let stencil = exec.to_device(&[0_u32, 1, 0, 1, 0]).unwrap();

    let (output,) = copy_if(
        &exec,
        massively::SoA1(values.slice(1..4)),
        stencil.slice(1..4),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![20, 40]);
}

#[test]
fn remove_if_accepts_device_slice_input() {
    let exec = exec();
    let values = exec.to_device(&[0_u32, 10, 20, 30, 99]).unwrap();

    let (output,) = remove_if(&exec, massively::SoA1(values.slice(1..4)), U32IsTwenty).unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![10, 30]);
}

#[test]
fn replace_if_accepts_device_slice_stencil() {
    let exec = exec();
    let values = exec.to_device(&[10_u32, 20, 30, 40, 50]).unwrap();
    let stencil = exec.to_device(&[0_u32, 1, 0, 1, 0]).unwrap();

    let (output,) = replace_if(
        &exec,
        massively::SoA1(values.slice(1..4)),
        (99_u32,),
        stencil.slice(1..4),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![99, 30, 99]);
}

#[test]
fn scatter_if_accepts_device_slice_indices_and_stencil() {
    let exec = exec();
    let values = exec.to_device(&[99_u32, 10, 20, 30, 88]).unwrap();
    let indices = exec.to_device(&[99_u32, 2, 1, 0, 88]).unwrap();
    let stencil = exec.to_device(&[0_u32, 1, 0, 1, 0]).unwrap();

    let (output,) = scatter_if(
        &exec,
        massively::SoA1(values.slice(1..4)),
        indices.slice(1..4),
        3,
        (0_u32,),
        stencil.slice(1..4),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![30, 0, 10]);
}

#[test]
fn transform_accepts_three_column_device_slices() {
    let exec = exec();
    let a = exec.to_device(&[0.0_f32, 1.0, 2.0, 3.0, 99.0]).unwrap();
    let b = exec.to_device(&[0_u32, 10, 20, 30, 99]).unwrap();
    let c = exec
        .to_device(&[0.0_f32, 100.0, 200.0, 300.0, 99.0])
        .unwrap();

    let (a, b, c) = transform(
        &exec,
        massively::SoA3(a.slice(1..4), b.slice(1..4), c.slice(1..4)),
        Tuple3MixedSplit,
    )
    .unwrap();

    assert_eq!(exec.to_host(&a).unwrap(), vec![101.0, 202.0, 303.0]);
    assert_eq!(exec.to_host(&b).unwrap(), vec![11, 21, 31]);
    assert_eq!(exec.to_host(&c).unwrap(), vec![101.0, 202.0, 303.0]);
}

#[test]
fn empty_device_slice_is_valid_input() {
    let exec = exec();
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();

    let slice = values.slice(1..1);
    let (output,) = transform(&exec, massively::SoA1(slice), Double).unwrap();
    let sum = reduce(&exec, massively::SoA1(slice), (0.0_f32,), TupleSum).unwrap();

    assert!(slice.is_empty());
    assert_eq!(exec.to_host(&slice).unwrap(), Vec::<f32>::new());
    assert_eq!(exec.to_host(&output).unwrap(), Vec::<f32>::new());
    assert_eq!(sum, (0.0,));
}

#[test]
#[should_panic(expected = "slice end (4) is out of bounds for DeviceVec of length 3")]
fn device_slice_range_end_panics_when_out_of_bounds() {
    let exec = exec();
    let values = exec.to_device(&[1_u32, 2, 3]).unwrap();

    let _ = values.slice(..4);
}

#[test]
#[should_panic(expected = "slice start (3) is greater than slice end (2)")]
fn device_slice_range_panics_when_start_is_after_end() {
    let exec = exec();
    let values = exec.to_device(&[1_u32, 2, 3]).unwrap();

    let _ = values.slice(3..2);
}
