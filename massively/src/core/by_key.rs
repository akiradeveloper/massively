//! By-key algorithms split into key-control and value-apply phases.

use cubecl::prelude::*;

use crate::{
    DeviceVec, Error, Executor, ReadExpression, RowStorage,
    allocation::ScratchStorage,
    ordering::{AdjacentFlagInput, BinaryPredicateOp, UniqueHead, unique_head_flags},
    selection::{CopySelected, SelectionControl},
};

const BLOCK_SIZE: u32 = 256;

#[cubecl::cube(launch_unchecked)]
fn tail_indices_kernel(head_indices: &[u32], count: &[u32], source_len: &[u32], tails: &mut [u32]) {
    let rank = ABSOLUTE_POS as usize;
    if rank < count[0] as usize {
        tails[rank] = if rank + 1usize < count[0] as usize {
            head_indices[rank + 1usize] - 1u32
        } else {
            source_len[0] - 1u32
        };
    }
}

pub(crate) fn tail_control_from_heads<R: Runtime>(
    exec: &Executor<R>,
    heads: &SelectionControl<R>,
) -> Result<SelectionControl<R>, Error> {
    let count = heads.count();
    let mut tails = exec.alloc_row::<u32>(heads.len());
    tails.set_logical_extent(crate::extent::LogicalExtent::from_device(
        count,
        heads.len(),
    ));
    if heads.len() != 0 {
        let len_handle = heads.source_extent().materialize(exec)?;
        unsafe {
            tail_indices_kernel::launch_unchecked::<R>(
                exec.client(),
                crate::launch::cube_count_1d(heads.len().div_ceil(BLOCK_SIZE as usize))?,
                CubeDim::new_1d(BLOCK_SIZE),
                BufferArg::from_raw_parts(
                    heads.indices().handle.clone(),
                    heads.indices().capacity(),
                ),
                BufferArg::from_raw_parts(count.handle.clone(), 1),
                BufferArg::from_raw_parts(len_handle.handle.clone(), 1),
                BufferArg::from_raw_parts(tails.handle.clone(), tails.capacity()),
            );
        }
    }
    Ok(SelectionControl::from_indices(
        heads.len(),
        heads.source_extent(),
        tails,
        count.clone(),
    ))
}

/// Key-only phase producing segment head flags.
#[doc(hidden)]
pub trait SegmentKeyInput<R: Runtime, Equal>: ReadExpression + Sized {
    fn segment_heads(self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error>;
}

impl<R, Keys, Equal> SegmentKeyInput<R, Equal> for Keys
where
    R: Runtime,
    Keys: ReadExpression + AdjacentFlagInput<R, UniqueHead<Equal>>,
    Equal: BinaryPredicateOp<Keys::Item>,
{
    fn segment_heads(self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error> {
        unique_head_flags::<R, _, Equal>(exec, self)
    }
}

fn segmented_lowered_storage<R, Values, Op>(
    exec: &Executor<R>,
    values: Values,
    heads: &DeviceVec<R, u32>,
    init: Option<<Values::Item as ScratchStorage<R>>::Storage>,
    op: Op,
    mode: u8,
) -> Result<<Values::Item as ScratchStorage<R>>::Storage, Error>
where
    R: Runtime,
    Values: crate::core::facade::KernelInput<R>,
    Values::Item: ScratchStorage<R>,
    Op: crate::op::ReductionOp<Values::Item>,
{
    let values = crate::allocation::normalize_lowered_scratch(exec, values)?;
    crate::segmented::segmented_fixed(exec, &values, heads, init, op, mode)
}

/// Fixed-input implementation used by the public logical iterator facade.
pub(crate) fn inclusive_scan_by_key_lowered<R, Keys, Values, Equal, Op, Output>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    _equal: Equal,
    op: Op,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Keys: SegmentKeyInput<R, Equal>,
    Values: crate::core::facade::KernelInput<R>,
    Values::Item: ScratchStorage<R>,
    Op: crate::op::ReductionOp<Values::Item>,
    Output:
        crate::core::facade::KernelOutput<R> + crate::output::OutputExpression<Item = Values::Item>,
{
    let heads = keys.segment_heads(exec)?;
    let scanned = segmented_lowered_storage(exec, values, &heads, None, op, 0)?;
    let read = crate::read::FixedRead::new(scanned.read());
    crate::transform::materialize_fixed(exec, &read, &output)
}

/// Fixed-input implementation used by the public logical iterator facade.
pub(crate) fn exclusive_scan_by_key_lowered<R, Keys, Values, Equal, Op, Output>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    _equal: Equal,
    init: <Values::Item as ScratchStorage<R>>::Storage,
    op: Op,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Keys: SegmentKeyInput<R, Equal>,
    Values: crate::core::facade::KernelInput<R>,
    Values::Item: ScratchStorage<R>,
    Op: crate::op::ReductionOp<Values::Item>,
    Output:
        crate::core::facade::KernelOutput<R> + crate::output::OutputExpression<Item = Values::Item>,
{
    let heads = keys.segment_heads(exec)?;
    let scanned = segmented_lowered_storage(exec, values, &heads, Some(init), op, 1)?;
    let read = crate::read::FixedRead::new(scanned.read());
    crate::transform::materialize_fixed(exec, &read, &output)
}

/// Fixed-input implementation used by the public logical iterator facade.
pub(crate) fn reduce_by_key_lowered<R, Keys, Values, Equal, Op, KeyOutput, ValueOutput>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    _equal: Equal,
    init: <Values::Item as ScratchStorage<R>>::Storage,
    op: Op,
    key_output: KeyOutput,
    value_output: ValueOutput,
) -> Result<DeviceVec<R, u32>, Error>
where
    R: Runtime,
    Keys: crate::core::facade::KernelInput<R, Item = KeyOutput::Item> + SegmentKeyInput<R, Equal>,
    Values: crate::core::facade::KernelInput<R>,
    Values::Item: ScratchStorage<R>,
    Op: crate::op::ReductionOp<Values::Item>,
    KeyOutput: crate::core::facade::KernelOutput<R>,
    ValueOutput:
        crate::core::facade::KernelOutput<R> + crate::output::OutputExpression<Item = Values::Item>,
{
    let heads = keys.clone().segment_heads(exec)?;
    let reduced = segmented_lowered_storage(exec, values, &heads, Some(init), op, 2)?;
    let head_control = SelectionControl::from_flags(exec, heads)?;
    let tail_control = tail_control_from_heads(exec, &head_control)?;
    crate::indexed::IndexedCopyInput::indexed_copy_selected(
        keys,
        exec,
        head_control.indices().column(),
        None,
        Some(head_control.count()),
        true,
        key_output,
    )?;
    let reduced = crate::read::FixedRead::new(reduced.read());
    crate::indexed::IndexedCopyInput::indexed_copy_selected(
        reduced,
        exec,
        tail_control.indices().column(),
        None,
        Some(tail_control.count()),
        true,
        value_output,
    )?;
    Ok(head_control.count().clone())
}

/// Reduces values using an already-computed segment-head control.
pub(crate) fn reduce_values_by_heads_lowered<R, Values, Op, Output>(
    exec: &Executor<R>,
    values: Values,
    heads: &DeviceVec<R, u32>,
    head_control: &SelectionControl<R>,
    init: <Values::Item as ScratchStorage<R>>::Storage,
    op: Op,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Values: crate::core::facade::KernelInput<R>,
    Values::Item: ScratchStorage<R>,
    Op: crate::op::ReductionOp<Values::Item>,
    Output:
        crate::core::facade::KernelOutput<R> + crate::output::OutputExpression<Item = Values::Item>,
{
    let reduced = segmented_lowered_storage(exec, values, heads, Some(init), op, 2)?;
    let tail_control = tail_control_from_heads(exec, head_control)?;
    let reduced = crate::read::FixedRead::new(reduced.read());
    crate::indexed::IndexedCopyInput::indexed_copy_selected(
        reduced,
        exec,
        tail_control.indices().column(),
        None,
        Some(tail_control.count()),
        true,
        output,
    )
}

/// Key phase for stable adjacent-key deduplication.
///
/// The resulting selection control is applied only to values, avoiding both
/// key materialization and key-arity × value-arity dispatch.
#[doc(hidden)]
pub trait UniqueByKeyKeys<R: Runtime, Equal>: ReadExpression + Sized {
    fn unique_key_len(&self) -> Result<usize, Error>;
    fn unique_key_extent(&self) -> Result<crate::extent::LogicalExtent, Error>;
    fn unique_key_control(self, exec: &Executor<R>) -> Result<SelectionControl<R>, Error>;
}

impl<R, Keys, Equal> UniqueByKeyKeys<R, Equal> for Keys
where
    R: Runtime,
    Keys: ReadExpression
        + AdjacentFlagInput<R, UniqueHead<Equal>>
        + crate::reduce::StageRead<R, crate::read::Env0>,
    Equal: BinaryPredicateOp<Keys::Item>,
{
    fn unique_key_len(&self) -> Result<usize, Error> {
        crate::reduce::StageRead::logical_len(self)
    }

    fn unique_key_extent(&self) -> Result<crate::extent::LogicalExtent, Error> {
        crate::reduce::StageRead::logical_extent(self)
    }

    fn unique_key_control(self, exec: &Executor<R>) -> Result<SelectionControl<R>, Error> {
        let flags = unique_head_flags::<R, _, Equal>(exec, self)?;
        SelectionControl::from_flags(exec, flags)
    }
}

/// Keeps the first value of every adjacent equal-key run.
///
/// `value_output` is preallocated and may be larger than the returned logical
/// length.
pub(crate) fn unique_by_key<R, Keys, Values, Equal, ValueOutput>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    _equal: Equal,
    value_output: ValueOutput,
) -> Result<DeviceVec<R, u32>, Error>
where
    R: Runtime,
    Keys: UniqueByKeyKeys<R, Equal>,
    Values: CopySelected<R, ValueOutput>,
{
    let key_len = keys.unique_key_len()?;
    let value_len = values.source_len()?;
    if key_len != value_len {
        return Err(Error::LengthMismatch {
            left: key_len,
            right: value_len,
        });
    }
    keys.unique_key_extent()?.zipped(&values.source_extent()?)?;
    let control = keys.unique_key_control(exec)?;
    values.copy_selected(exec, &control, value_output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Counting, Permute, RowStorage, Transform, Zip};
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    type Three = (u32, u32, u32);
    struct EqualThree;

    #[cubecl::cube]
    impl BinaryPredicateOp<Three> for EqualThree {
        fn apply(lhs: Three, rhs: Three) -> crate::MBool {
            crate::op::mbool(lhs.0 == rhs.0 && lhs.1 == rhs.1 && lhs.2 == rhs.2)
        }
    }

    struct EqualU32;

    #[cubecl::cube]
    impl BinaryPredicateOp<u32> for EqualU32 {
        fn apply(lhs: u32, rhs: u32) -> crate::MBool {
            crate::op::mbool(lhs == rhs)
        }
    }

    type Seven = (u32, u32, u32, u32, u32, u32, u32);
    struct SumSeven;

    #[cubecl::cube]
    impl crate::op::ReductionOp<Seven> for SumSeven {
        fn apply(lhs: Seven, rhs: Seven) -> Seven {
            (
                lhs.0 + rhs.0,
                lhs.1 + rhs.1,
                lhs.2 + rhs.2,
                lhs.3 + rhs.3,
                lhs.4 + rhs.4,
                lhs.5 + rhs.5,
                lhs.6 + rhs.6,
            )
        }
    }

    #[test]
    fn unique_by_key_separates_three_key_and_seven_value_arities() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let k0 = exec.to_device(&[1_u32, 1, 1, 2, 2]);
        let k1 = exec.to_device(&[10_u32, 10, 11, 10, 10]);
        let k2 = exec.to_device(&[100_u32, 100, 100, 200, 200]);
        let values: Vec<_> = (0_u32..7)
            .map(|column| {
                exec.to_device(&[
                    column * 100 + 10,
                    column * 100 + 11,
                    column * 100 + 12,
                    column * 100 + 20,
                    column * 100 + 21,
                ])
            })
            .collect();
        let out_values: Vec<_> = (0..7).map(|_| exec.to_device(&[0_u32; 5])).collect();

        let keys = Zip::new(k0.column(), Zip::new(k1.column(), k2.column()));
        let value_input = Zip::new(
            values[0].column(),
            Zip::new(
                values[1].column(),
                Zip::new(
                    values[2].column(),
                    Zip::new(
                        values[3].column(),
                        Zip::new(
                            values[4].column(),
                            Zip::new(values[5].column(), values[6].column()),
                        ),
                    ),
                ),
            ),
        );
        let value_output = Zip::new(
            Zip::new(
                Zip::new(
                    Zip::new(
                        Zip::new(
                            Zip::new(out_values[0].slice_mut(..), out_values[1].slice_mut(..)),
                            out_values[2].slice_mut(..),
                        ),
                        out_values[3].slice_mut(..),
                    ),
                    out_values[4].slice_mut(..),
                ),
                out_values[5].slice_mut(..),
            ),
            out_values[6].slice_mut(..),
        );

        let count = unique_by_key(&exec, keys, value_input, EqualThree, value_output).unwrap();
        let count = exec.to_host(&count).unwrap()[0] as usize;
        assert_eq!(count, 3);
        for (column, output) in out_values.iter().enumerate() {
            let base = column as u32 * 100;
            assert_eq!(
                exec.to_host(output).unwrap()[..3],
                [base + 10, base + 12, base + 20]
            );
        }
    }

    #[test]
    fn unique_by_key_rejects_key_value_length_mismatch() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let keys = exec.to_device(&[1_u32, 1]);
        let values = exec.to_device(&[10_u32]);
        let out_values = exec.to_device(&[0_u32; 2]);
        assert!(matches!(
            unique_by_key(
                &exec,
                keys.column(),
                values.column(),
                EqualU32,
                out_values.slice_mut(..),
            ),
            Err(Error::LengthMismatch { left: 2, right: 1 })
        ));
    }

    #[test]
    fn segmented_by_key_algorithms_separate_eval8_keys_from_storage7_values() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let len = 600usize;
        let keys_host: Vec<u32> = (0..len)
            .map(|index| if index < 300 { 1 } else { 2 })
            .collect();
        let keys = exec.to_device(&keys_host);
        let columns: Vec<_> = (1_u32..=7)
            .map(|value| exec.to_device(&vec![value; len]))
            .collect();
        let values = || {
            let seven = Zip::new(
                columns[0].column(),
                Zip::new(
                    columns[1].column(),
                    Zip::new(
                        columns[2].column(),
                        Zip::new(
                            columns[3].column(),
                            Zip::new(
                                columns[4].column(),
                                Zip::new(columns[5].column(), columns[6].column()),
                            ),
                        ),
                    ),
                ),
            );
            Permute::new(seven, Counting::new(0, len))
        };

        let inclusive = exec.alloc_row::<Seven>(len);
        crate::api::algorithm::inclusive_scan_by_key_into(
            &exec,
            keys.column(),
            values(),
            EqualU32,
            SumSeven,
            inclusive.write(),
        )
        .unwrap();
        let (first, _, _, _, _, _, seventh) = crate::MStorage::into_columns(inclusive);
        let first = exec.to_host(&first).unwrap();
        let seventh = exec.to_host(&seventh).unwrap();
        assert_eq!((first[299], first[300], first[599]), (300, 1, 300));
        assert_eq!((seventh[299], seventh[300], seventh[599]), (2100, 7, 2100));

        let init: Seven = (10, 20, 30, 40, 50, 60, 70);
        let init = exec.value(init).unwrap();
        let exclusive = exec.alloc_row::<Seven>(len);
        crate::api::algorithm::exclusive_scan_by_key_into(
            &exec,
            keys.column(),
            values(),
            EqualU32,
            init.clone(),
            SumSeven,
            exclusive.write(),
        )
        .unwrap();
        let (first, _, _, _, _, _, _) = crate::MStorage::into_columns(exclusive);
        let first = exec.to_host(&first).unwrap();
        assert_eq!(
            (first[0], first[1], first[300], first[301]),
            (10, 11, 10, 11)
        );

        let key_output = exec.to_device(&vec![0_u32; len]);
        let value_output = exec.alloc_row::<Seven>(len);
        let count = crate::api::algorithm::reduce_by_key_into(
            &exec,
            keys.column(),
            values(),
            EqualU32,
            init,
            SumSeven,
            key_output.slice_mut(..),
            value_output.write(),
        )
        .unwrap();
        let count = count.read(&exec).unwrap();
        assert_eq!(count, 2);
        assert_eq!(exec.to_host(&key_output.slice(..2)).unwrap(), vec![1, 2]);
        let (first, _, _, _, _, _, seventh) = crate::MStorage::into_columns(value_output);
        assert_eq!(exec.to_host(&first.slice(..2)).unwrap(), vec![310, 310]);
        assert_eq!(exec.to_host(&seventh.slice(..2)).unwrap(), vec![2170, 2170]);
    }
}
