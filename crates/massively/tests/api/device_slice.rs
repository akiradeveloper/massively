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
fn executor_constant_allocates_owned_device_vec() {
    let exec = exec();
    let input = exec.constant(4, 7_u32).unwrap();

    assert_eq!(exec.to_host(&input).unwrap(), vec![7, 7, 7, 7]);
}

#[test]
fn executor_counting_allocates_owned_device_vec() {
    let exec = exec();
    let input = exec.counting(5).unwrap();

    assert_eq!(exec.to_host(&input).unwrap(), vec![0, 1, 2, 3, 4]);
}

#[test]
fn executor_alloc_allocates_owned_soa_from_mitem() {
    let exec = exec();
    let single: massively::SoA1<massively::DeviceVec<_, u32>> = exec.alloc::<(u32,)>(4).unwrap();
    let pair: massively::SoA2<massively::DeviceVec<_, f32>, massively::DeviceVec<_, u32>> =
        exec.alloc::<(f32, u32)>(3).unwrap();

    assert_eq!(single.0.len(), 4);
    assert_eq!(pair.0.len(), 3);
}

fn scatter_into_allocated<R, Input>(
    exec: &massively::Executor<R>,
    source: Input,
    indices: massively::DeviceSlice<'_, R, massively::MIndex>,
    len: massively::MIndex,
) -> Result<<Input::Item as massively::MAlloc<R>>::Storage, massively::Error>
where
    R: Runtime,
    Input: massively::MIter<R>,
    Input::Item: massively::MAlloc<R>,
{
    let out = exec.alloc::<Input::Item>(len)?;
    scatter(exec, source, indices, out.slice_mut(..))?;
    Ok(out)
}

#[test]
fn executor_alloc_can_create_temporary_buffer_from_miter_item() {
    let exec = exec();
    let values = exec.to_device(&[10.0_f32, 20.0, 30.0]).unwrap();
    let ids = exec.to_device(&[1_u32, 2, 3]).unwrap();
    let indices = exec.to_device(&[2_u32, 0, 1]).unwrap();

    let massively::SoA2(out_values, out_ids) = scatter_into_allocated(
        &exec,
        massively::SoA2(values.slice(..), ids.slice(..)),
        indices.slice(..),
        3,
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_values).unwrap(), vec![20.0, 30.0, 10.0]);
    assert_eq!(exec.to_host(&out_ids).unwrap(), vec![2, 3, 1]);
}

fn assert_miter_can_be_sliced_twice<R, Input>(input: &Input)
where
    R: Runtime,
    Input: massively::MIter<R>,
{
    let slice = input.slice(..);
    let _slice = slice.slice(..);
}

fn assert_miter_mut_can_be_sliced_twice<R, Output>(output: &Output)
where
    R: Runtime,
    Output: massively::MIterMut<R>,
{
    let slice = output.slice(..);
    let _slice = slice.slice(..);

    let slice_mut = output.slice_mut(..);
    let slice = slice_mut.slice(..);
    let _slice = slice.slice(..);
    let _slice_mut = slice_mut.slice_mut(..);
}

fn assert_alloc_storage_can_be_sliced_repeatedly<R, Storage>(storage: &Storage)
where
    R: Runtime,
    Storage: massively::MAllocStorage<R>,
    for<'a> <Storage as massively::ToSlice>::Slice<'a>: massively::MIter<R, Item = Storage::Item>,
    for<'a> <Storage as massively::ToSliceMut>::SliceMut<'a>:
        massively::MIterMut<R, Item = Storage::Item>,
{
    let slice = storage.slice(..);
    let slice = slice.slice(..);
    let _slice = slice.slice(..);

    let slice_mut = storage.slice_mut(..);
    let _slice_mut = slice_mut.slice_mut(..);
}

#[test]
fn generic_slice_contracts_allow_repeated_slicing() {
    let exec = exec();
    let input = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let soa = massively::SoA1(exec.to_device(&[1_u32, 2, 3, 4]).unwrap());

    assert_miter_can_be_sliced_twice::<WgpuRuntime, _>(&input.slice(..));
    assert_miter_can_be_sliced_twice::<WgpuRuntime, _>(&soa.slice(..));

    assert_miter_mut_can_be_sliced_twice::<WgpuRuntime, _>(&soa.slice_mut(..));

    assert_alloc_storage_can_be_sliced_repeatedly::<WgpuRuntime, _>(&soa);
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
    let input = exec.to_device(&[10_u32, 20, 30, 40, 50, 60]).unwrap();
    let slice = input.slice_mut(1..5);

    assert_eq!(exec.to_host(&slice.slice(1..3)).unwrap(), vec![30, 40]);
}

#[test]
fn executor_copy_copies_between_device_slices() {
    let exec = exec();
    let input = exec.to_device(&[10_u32, 20, 30, 40, 50]).unwrap();
    let output = exec.to_device(&[1_u32, 2, 3, 4, 5, 6]).unwrap();

    exec.copy(input.slice(1..4), output.slice_mut(2..5))
        .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![1, 2, 20, 30, 40, 6]);
}

#[test]
fn device_vec_can_create_read_and_mut_slices_at_once() {
    let exec = exec();
    let values = exec.to_device(&[10_u32, 20, 30, 0, 0, 0]).unwrap();

    let input = values.slice(0..3);
    let output = values.slice_mut(3..6);
    exec.copy(input, output).unwrap();

    assert_eq!(exec.to_host(&values).unwrap(), vec![10, 20, 30, 10, 20, 30]);
}

#[test]
fn executor_copy_accepts_nested_mutable_destination_slice() {
    let exec = exec();
    let input = exec.to_device(&[7_u32, 8, 9]).unwrap();
    let output = exec.to_device(&[0_u32, 1, 2, 3, 4, 5]).unwrap();

    {
        let middle = output.slice_mut(1..5);
        exec.copy(input.slice(..2), middle.slice_mut(1..3)).unwrap();
    }

    assert_eq!(exec.to_host(&output).unwrap(), vec![0, 1, 7, 8, 4, 5]);
}

#[test]
fn executor_copy_rejects_mismatched_lengths() {
    let exec = exec();
    let input = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let output = exec.to_device(&[0_u32, 0]).unwrap();

    assert!(exec.copy(input.slice(..), output.slice_mut(..)).is_err());
}

#[test]
fn executor_copy_rejects_other_executor_data() {
    let data_exec = exec();
    let other_exec = exec();
    let input = data_exec.to_device(&[10_u32, 20]).unwrap();
    let output = data_exec.to_device(&[0_u32, 0]).unwrap();

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
    let output = data_exec.to_device(&[0.0_f32; 3]).unwrap();

    let result = transform(
        &other_exec,
        massively::SoA1(input.slice(..)),
        Double,
        (),
        massively::SoA1(output.slice_mut(..)),
    );

    assert!(result.is_err());
}

#[test]
fn transform_accepts_device_slice() {
    let exec = exec();
    let input = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();

    let output = exec.to_device(&[0.0_f32; 2]).unwrap();
    transform(
        &exec,
        input.slice(1..3),
        Double,
        (),
        massively::SoA1(output.slice_mut(..)),
    )
    .unwrap();

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

    let output = exec.to_device(&[0.0_f32; 3]).unwrap();
    inclusive_scan(
        &exec,
        massively::SoA1(input.slice(1..4)),
        TupleSum,
        massively::SoA1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![2.0, 5.0, 9.0]);
}

#[test]
fn transform_accepts_multi_column_device_slices() {
    let exec = exec();
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let tags = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();

    let out_values = exec.to_device(&[0.0_f32; 3]).unwrap();
    let out_tags = exec.to_device(&[0_u32; 3]).unwrap();
    transform(
        &exec,
        massively::SoA2(values.slice(1..4), tags.slice(1..4)),
        PairMixedSplit,
        (),
        massively::SoA2(out_values.slice_mut(..), out_tags.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_values).unwrap(), vec![12.0, 13.0, 14.0]);
    assert_eq!(exec.to_host(&out_tags).unwrap(), vec![21, 31, 41]);
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
fn direct_device_slice_reduce_reads_scalar_items() {
    let exec = exec();
    let values = exec.to_device(&[99.0_f32, 1.0, 2.0, 3.0, 88.0]).unwrap();

    let sum = reduce(&exec, values.slice(1..4), 0.0_f32, Sum).unwrap();

    assert_eq!(sum, 6.0);
}

#[test]
fn reverse_accepts_multi_column_device_slices() {
    let exec = exec();
    let values = exec.to_device(&[0.0_f32, 1.0, 2.0, 3.0, 99.0]).unwrap();
    let tags = exec.to_device(&[0_u32, 10, 20, 30, 99]).unwrap();

    let out_values = exec.to_device(&[0.0_f32; 3]).unwrap();
    let out_tags = exec.to_device(&[0_u32; 3]).unwrap();
    reverse(
        &exec,
        massively::SoA2(values.slice(1..4), tags.slice(1..4)),
        massively::SoA2(out_values.slice_mut(..), out_tags.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_values).unwrap(), vec![3.0, 2.0, 1.0]);
    assert_eq!(exec.to_host(&out_tags).unwrap(), vec![30, 20, 10]);
}

#[test]
fn sort_accepts_multi_column_device_slices() {
    let exec = exec();
    let values = exec.to_device(&[99.0_f32, 2.0, 1.0, 2.0, 88.0]).unwrap();
    let tags = exec.to_device(&[99_u32, 20, 30, 10, 88]).unwrap();

    let out_values = exec.to_device(&[0.0_f32; 3]).unwrap();
    let out_tags = exec.to_device(&[0_u32; 3]).unwrap();
    sort(
        &exec,
        massively::SoA2(values.slice(1..4), tags.slice(1..4)),
        MixedTupleLess,
        massively::SoA2(out_values.slice_mut(..), out_tags.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_values).unwrap(), vec![1.0, 2.0, 2.0]);
    assert_eq!(exec.to_host(&out_tags).unwrap(), vec![30, 10, 20]);
}

#[test]
fn sort_accepts_offset_device_slice() {
    let exec = exec();
    let values = exec
        .to_device(&[999.0_f32, 4.0, 1.0, 3.0, 2.0, 888.0])
        .unwrap();

    let output = exec.to_device(&[0.0_f32; 4]).unwrap();
    sort(
        &exec,
        massively::SoA1(values.slice(1..5)),
        Less,
        massively::SoA1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn gather_accepts_device_slice_indices() {
    let exec = exec();
    let values = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let indices = exec.to_device(&[99_u32, 3, 1, 0, 88]).unwrap();

    let output = exec.to_device(&[0_u32; 3]).unwrap();
    gather(
        &exec,
        massively::SoA1(values.slice(..)),
        indices.slice(1..4),
        massively::SoA1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![40, 20, 10]);
}

#[test]
fn gather_where_accepts_offset_device_slices() {
    let exec = exec();
    let values = exec.to_device(&[99_u32, 10, 20, 30, 40, 88]).unwrap();
    let indices = exec.to_device(&[77_u32, 3, 1, 0, 2, 66]).unwrap();
    let stencil = exec.to_device(&[0_u32, 1, 0, 1, 0, 0]).unwrap();

    let output = exec.to_device(&[0_u32; 4]).unwrap();
    gather_where(
        &exec,
        massively::SoA1(values.slice(1..5)),
        indices.slice(1..5),
        stencil.slice(1..5),
        massively::SoA1(output.slice_mut(..)),
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

    let output = exec.to_device(&[0_u32; 4]).unwrap();
    merge(
        &exec,
        massively::SoA1(left.slice(1..3)),
        massively::SoA1(right.slice(..2)),
        LessU32,
        massively::SoA1(output.slice_mut(..)),
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

    let out_keys = exec.to_device(&[0_u32; 4]).unwrap();
    let out_a = exec.to_device(&[0.0_f32; 4]).unwrap();
    let out_b = exec.to_device(&[0_u32; 4]).unwrap();
    merge_by_key(
        &exec,
        massively::SoA1(left_keys.slice(1..3)),
        massively::SoA2(left_a.slice(1..3), left_b.slice(1..3)),
        massively::SoA1(right_keys.slice(1..3)),
        massively::SoA2(right_a.slice(1..3), right_b.slice(1..3)),
        LessU32,
        massively::SoA1(out_keys.slice_mut(..)),
        massively::SoA2(out_a.slice_mut(..), out_b.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_keys).unwrap(), vec![1, 2, 3, 4]);
    assert_eq!(
        exec.to_host(&out_a).unwrap(),
        vec![100.0, 200.0, 300.0, 400.0]
    );
    assert_eq!(exec.to_host(&out_b).unwrap(), vec![10, 20, 30, 40]);
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

    let union_a = exec.to_device(&[0.0_f32; 7]).unwrap();
    let union_b = exec.to_device(&[0_u32; 7]).unwrap();
    let union_len = set_union(
        &exec,
        massively::SoA2(left_a.slice(1..5), left_b.slice(1..5)),
        massively::SoA2(right_a.slice(1..4), right_b.slice(1..4)),
        MixedTupleLess,
        massively::SoA2(union_a.slice_mut(..), union_b.slice_mut(..)),
    )
    .unwrap();
    let intersection_a = exec.to_device(&[0.0_f32; 4]).unwrap();
    let intersection_b = exec.to_device(&[0_u32; 4]).unwrap();
    let intersection_len = set_intersection(
        &exec,
        massively::SoA2(left_a.slice(1..5), left_b.slice(1..5)),
        massively::SoA2(right_a.slice(1..4), right_b.slice(1..4)),
        MixedTupleLess,
        massively::SoA2(intersection_a.slice_mut(..), intersection_b.slice_mut(..)),
    )
    .unwrap();
    let difference_a = exec.to_device(&[0.0_f32; 4]).unwrap();
    let difference_b = exec.to_device(&[0_u32; 4]).unwrap();
    let difference_len = set_difference(
        &exec,
        massively::SoA2(left_a.slice(1..5), left_b.slice(1..5)),
        massively::SoA2(right_a.slice(1..4), right_b.slice(1..4)),
        MixedTupleLess,
        massively::SoA2(difference_a.slice_mut(..), difference_b.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(
        exec.to_host(&union_a.slice(..union_len)).unwrap(),
        vec![1.0, 2.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(
        exec.to_host(&union_b.slice(..union_len)).unwrap(),
        vec![10, 20, 21, 30, 40]
    );
    assert_eq!(
        exec.to_host(&intersection_a.slice(..intersection_len))
            .unwrap(),
        vec![2.0, 4.0]
    );
    assert_eq!(
        exec.to_host(&intersection_b.slice(..intersection_len))
            .unwrap(),
        vec![20, 40]
    );
    assert_eq!(
        exec.to_host(&difference_a.slice(..difference_len)).unwrap(),
        vec![1.0, 2.0]
    );
    assert_eq!(
        exec.to_host(&difference_b.slice(..difference_len)).unwrap(),
        vec![10, 21]
    );
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

    let output = exec.to_device(&[0_u32; 4]).unwrap();
    inclusive_scan_by_key(
        &exec,
        massively::SoA1(keys.slice(1..5)),
        massively::SoA1(values.slice(1..5)),
        EqualU32,
        Sum,
        massively::SoA1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![10, 30, 1, 3]);
}

#[test]
fn sort_by_key_accepts_device_slice_keys_and_values() {
    let exec = exec();
    let keys = exec.to_device(&[99_u32, 3, 1, 2, 88]).unwrap();
    let values = exec.to_device(&[99_u32, 30, 10, 20, 88]).unwrap();

    let out_keys = exec.to_device(&[0_u32; 3]).unwrap();
    let out_values = exec.to_device(&[0_u32; 3]).unwrap();
    sort_by_key(
        &exec,
        massively::SoA1(keys.slice(1..4)),
        massively::SoA1(values.slice(1..4)),
        LessU32,
        massively::SoA1(out_keys.slice_mut(..)),
        massively::SoA1(out_values.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_keys).unwrap(), vec![1, 2, 3]);
    assert_eq!(exec.to_host(&out_values).unwrap(), vec![10, 20, 30]);
}

#[test]
fn sort_by_key_accepts_multi_column_device_slice_values() {
    let exec = exec();
    let keys = exec.to_device(&[99_u32, 3, 1, 2, 88]).unwrap();
    let values = exec.to_device(&[99.0_f32, 30.0, 10.0, 20.0, 88.0]).unwrap();
    let tags = exec.to_device(&[99_u32, 300, 100, 200, 88]).unwrap();

    let out_keys = exec.to_device(&[0_u32; 3]).unwrap();
    let out_values = exec.to_device(&[0.0_f32; 3]).unwrap();
    let out_tags = exec.to_device(&[0_u32; 3]).unwrap();
    sort_by_key(
        &exec,
        massively::SoA1(keys.slice(1..4)),
        massively::SoA2(values.slice(1..4), tags.slice(1..4)),
        LessU32,
        massively::SoA1(out_keys.slice_mut(..)),
        massively::SoA2(out_values.slice_mut(..), out_tags.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_keys).unwrap(), vec![1, 2, 3]);
    assert_eq!(exec.to_host(&out_values).unwrap(), vec![10.0, 20.0, 30.0]);
    assert_eq!(exec.to_host(&out_tags).unwrap(), vec![100, 200, 300]);
}

#[test]
fn unique_by_key_accepts_multi_column_device_slice_values() {
    let exec = exec();
    let keys = exec.to_device(&[99_u32, 0, 0, 1, 1, 88]).unwrap();
    let values = exec
        .to_device(&[99.0_f32, 10.0, 20.0, 30.0, 40.0, 88.0])
        .unwrap();
    let tags = exec.to_device(&[99_u32, 100, 200, 300, 400, 88]).unwrap();

    let out_keys = exec.to_device(&[0_u32; 4]).unwrap();
    let out_values = exec.to_device(&[0.0_f32; 4]).unwrap();
    let out_tags = exec.to_device(&[0_u32; 4]).unwrap();
    let len = unique_by_key(
        &exec,
        massively::SoA1(keys.slice(1..5)),
        massively::SoA2(values.slice(1..5), tags.slice(1..5)),
        EqualU32,
        massively::SoA1(out_keys.slice_mut(..)),
        massively::SoA2(out_values.slice_mut(..), out_tags.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_keys.slice(..len)).unwrap(), vec![0, 1]);
    assert_eq!(
        exec.to_host(&out_values.slice(..len)).unwrap(),
        vec![10.0, 30.0]
    );
    assert_eq!(
        exec.to_host(&out_tags.slice(..len)).unwrap(),
        vec![100, 300]
    );
}

#[test]
fn inclusive_scan_by_key_accepts_multi_column_device_slice_values() {
    let exec = exec();
    let keys = exec.to_device(&[99_u32, 0, 0, 1, 1, 88]).unwrap();
    let values = exec
        .to_device(&[99.0_f32, 1.0, 2.0, 3.0, 4.0, 88.0])
        .unwrap();
    let tags = exec.to_device(&[99_u32, 10, 20, 30, 40, 88]).unwrap();

    let out_values = exec.to_device(&[0.0_f32; 4]).unwrap();
    let out_tags = exec.to_device(&[0_u32; 4]).unwrap();
    inclusive_scan_by_key(
        &exec,
        massively::SoA1(keys.slice(1..5)),
        massively::SoA2(values.slice(1..5), tags.slice(1..5)),
        EqualU32,
        TupleSum,
        massively::SoA2(out_values.slice_mut(..), out_tags.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_values).unwrap(), vec![1.0, 3.0, 3.0, 7.0]);
    assert_eq!(exec.to_host(&out_tags).unwrap(), vec![10, 30, 30, 70]);
}

#[test]
fn exclusive_scan_by_key_accepts_multi_column_device_slice_values() {
    let exec = exec();
    let keys = exec.to_device(&[99_u32, 0, 0, 1, 1, 88]).unwrap();
    let values = exec
        .to_device(&[99.0_f32, 1.0, 2.0, 3.0, 4.0, 88.0])
        .unwrap();
    let tags = exec.to_device(&[99_u32, 10, 20, 30, 40, 88]).unwrap();

    let out_values = exec.to_device(&[0.0_f32; 4]).unwrap();
    let out_tags = exec.to_device(&[0_u32; 4]).unwrap();
    exclusive_scan_by_key(
        &exec,
        massively::SoA1(keys.slice(1..5)),
        massively::SoA2(values.slice(1..5), tags.slice(1..5)),
        EqualU32,
        (0.0_f32, 0_u32),
        TupleSum,
        massively::SoA2(out_values.slice_mut(..), out_tags.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_values).unwrap(), vec![0.0, 1.0, 0.0, 3.0]);
    assert_eq!(exec.to_host(&out_tags).unwrap(), vec![0, 10, 0, 30]);
}

#[test]
fn reduce_by_key_accepts_multi_column_device_slice_values() {
    let exec = exec();
    let keys = exec.to_device(&[99_u32, 0, 0, 1, 1, 88]).unwrap();
    let values = exec
        .to_device(&[99.0_f32, 1.0, 2.0, 3.0, 4.0, 88.0])
        .unwrap();
    let tags = exec.to_device(&[99_u32, 10, 20, 30, 40, 88]).unwrap();

    let out_keys = exec.to_device(&[0_u32; 4]).unwrap();
    let out_values = exec.to_device(&[0.0_f32; 4]).unwrap();
    let out_tags = exec.to_device(&[0_u32; 4]).unwrap();
    let len = reduce_by_key(
        &exec,
        massively::SoA1(keys.slice(1..5)),
        massively::SoA2(values.slice(1..5), tags.slice(1..5)),
        EqualU32,
        (0.0_f32, 0_u32),
        TupleSum,
        massively::SoA1(out_keys.slice_mut(..)),
        massively::SoA2(out_values.slice_mut(..), out_tags.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_keys.slice(..len)).unwrap(), vec![0, 1]);
    assert_eq!(
        exec.to_host(&out_values.slice(..len)).unwrap(),
        vec![3.0, 7.0]
    );
    assert_eq!(exec.to_host(&out_tags.slice(..len)).unwrap(), vec![30, 70]);
}

#[test]
fn copy_where_accepts_device_slice_stencil() {
    let exec = exec();
    let values = exec.to_device(&[10_u32, 20, 30, 40, 50]).unwrap();
    let stencil = exec.to_device(&[0_u32, 1, 0, 1, 0]).unwrap();

    let output = exec.to_device(&[0_u32; 3]).unwrap();
    let len = copy_where(
        &exec,
        massively::SoA1(values.slice(1..4)),
        stencil.slice(1..4),
        massively::SoA1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output.slice(..len)).unwrap(), vec![20, 40]);
}

#[test]
fn remove_where_accepts_device_slice_input_and_stencil() {
    let exec = exec();
    let values = exec.to_device(&[0_u32, 10, 20, 30, 99]).unwrap();
    let stencil = exec.to_device(&[0_u32, 0, 1, 0, 0]).unwrap();

    let output = exec.to_device(&[0_u32; 3]).unwrap();
    let len = remove_where(
        &exec,
        massively::SoA1(values.slice(1..4)),
        stencil.slice(1..4),
        massively::SoA1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output.slice(..len)).unwrap(), vec![10, 30]);
}

#[test]
fn replace_where_accepts_device_slice_stencil() {
    let exec = exec();
    let values = exec.to_device(&[10_u32, 20, 30, 40, 50]).unwrap();
    let stencil = exec.to_device(&[0_u32, 1, 0, 1, 0]).unwrap();

    replace_where(
        &exec,
        (99_u32,),
        stencil.slice(1..4),
        massively::SoA1(values.slice_mut(1..4)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&values.slice(1..4)).unwrap(), vec![99, 30, 99]);
}

#[test]
fn scatter_where_accepts_device_slice_indices_and_stencil() {
    let exec = exec();
    let values = exec.to_device(&[99_u32, 10, 20, 30, 88]).unwrap();
    let indices = exec.to_device(&[99_u32, 2, 1, 0, 88]).unwrap();
    let stencil = exec.to_device(&[0_u32, 1, 0, 1, 0]).unwrap();

    let output = exec.to_device(&[0_u32; 3]).unwrap();
    scatter_where(
        &exec,
        massively::SoA1(values.slice(1..4)),
        indices.slice(1..4),
        stencil.slice(1..4),
        massively::SoA1(output.slice_mut(..)),
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

    let out_a = exec.to_device(&[0.0_f32; 3]).unwrap();
    let out_b = exec.to_device(&[0_u32; 3]).unwrap();
    let out_c = exec.to_device(&[0.0_f32; 3]).unwrap();
    transform(
        &exec,
        massively::SoA3(a.slice(1..4), b.slice(1..4), c.slice(1..4)),
        Tuple3MixedSplit,
        (),
        massively::SoA3(
            out_a.slice_mut(..),
            out_b.slice_mut(..),
            out_c.slice_mut(..),
        ),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_a).unwrap(), vec![101.0, 202.0, 303.0]);
    assert_eq!(exec.to_host(&out_b).unwrap(), vec![11, 21, 31]);
    assert_eq!(exec.to_host(&out_c).unwrap(), vec![101.0, 202.0, 303.0]);
}

#[test]
fn empty_device_slice_is_valid_input() {
    let exec = exec();
    let values = exec.to_device(&[1.0_f32, 2.0, 3.0]).unwrap();

    let slice = values.slice(1..1);
    let output = exec.to_device(&[] as &[f32]).unwrap();
    transform(
        &exec,
        massively::SoA1(slice),
        Double,
        (),
        massively::SoA1(output.slice_mut(..)),
    )
    .unwrap();
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
