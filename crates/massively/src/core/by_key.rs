//! By-key algorithms split into key-control and value-apply phases.

use cubecl::prelude::*;

use crate::{
    CanonicalAlloc, CanonicalStorage, Counting, DeviceVec, Error, Executor, ReadExpression,
    allocation::{NormalizeInput, singleton},
    indexed::GatherInput,
    ordering::{AdjacentFlagInput, BinaryPredicateOp, UniqueHead, unique_head_flags},
    segmented::SegmentedStorage,
    selection::{CopySelected, FillOutput, SelectionControl},
    storage::WriteFrom,
};

const BLOCK_SIZE: u32 = 256;

#[cubecl::cube(launch_unchecked)]
fn tail_indices_kernel(head_indices: &[u32], source_len: &[u32], tails: &mut [u32]) {
    let rank = ABSOLUTE_POS as usize;
    if rank < head_indices.len() {
        tails[rank] = if rank + 1usize < head_indices.len() {
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
    let tails = exec.alloc_canonical::<u32>(count as usize);
    if count != 0u32 {
        let len =
            u32::try_from(heads.len()).map_err(|_| Error::LengthTooLarge { len: heads.len() })?;
        let len_handle = exec.client().create_from_slice(u32::as_bytes(&[len]));
        unsafe {
            tail_indices_kernel::launch_unchecked::<R>(
                exec.client(),
                crate::launch::cube_count_1d((count as usize).div_ceil(BLOCK_SIZE as usize))?,
                CubeDim::new_1d(BLOCK_SIZE),
                BufferArg::from_raw_parts(heads.indices().handle.clone(), heads.indices().len()),
                BufferArg::from_raw_parts(len_handle, 1),
                BufferArg::from_raw_parts(tails.handle.clone(), tails.len()),
            );
        }
    }
    Ok(SelectionControl::from_indices(heads.len(), tails, count))
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

/// Value-only phase for segmented scans.  Keys never appear in this trait.
#[doc(hidden)]
pub trait SegmentedValues<R: Runtime, Op>: NormalizeInput<R> {
    fn inclusive_storage(
        self,
        exec: &Executor<R>,
        heads: &DeviceVec<R, u32>,
    ) -> Result<Self::Storage, Error>;
    fn exclusive_storage(
        self,
        exec: &Executor<R>,
        heads: &DeviceVec<R, u32>,
        init: Self::Item,
    ) -> Result<Self::Storage, Error>;
    fn reduced_storage(
        self,
        exec: &Executor<R>,
        heads: &DeviceVec<R, u32>,
        init: Self::Item,
    ) -> Result<Self::Storage, Error>;
}

impl<R, Values, Op> SegmentedValues<R, Op> for Values
where
    R: Runtime,
    Values: NormalizeInput<R>,
    Values::Item: CanonicalAlloc<R, CanonicalStorage = Values::Storage>,
    Values::Storage: CanonicalStorage<R> + SegmentedStorage<R, Values::Item, Op>,
    <Values::Storage as CanonicalStorage<R>>::Item: WriteFrom<Values::Item>,
    <Values::Storage as CanonicalStorage<R>>::Write: FillOutput<R>,
    Op: crate::ReductionOp<Values::Item>,
{
    fn inclusive_storage(
        self,
        exec: &Executor<R>,
        heads: &DeviceVec<R, u32>,
    ) -> Result<Self::Storage, Error> {
        let values = self.normalize(exec)?;
        if values.segmented_len() != heads.len() {
            return Err(Error::LengthMismatch {
                left: values.segmented_len(),
                right: heads.len(),
            });
        }
        let output = exec.alloc_canonical::<Values::Item>(heads.len());
        values.segmented_inclusive(exec, heads, &output)?;
        Ok(output)
    }

    fn exclusive_storage(
        self,
        exec: &Executor<R>,
        heads: &DeviceVec<R, u32>,
        init: Self::Item,
    ) -> Result<Self::Storage, Error> {
        let inclusive = self.inclusive_storage(exec, heads)?;
        let initial = singleton::<R, Values::Item>(exec, init)?;
        let output = exec.alloc_canonical::<Values::Item>(heads.len());
        inclusive.segmented_exclusive(exec, heads, &initial, &output)?;
        Ok(output)
    }

    fn reduced_storage(
        self,
        exec: &Executor<R>,
        heads: &DeviceVec<R, u32>,
        init: Self::Item,
    ) -> Result<Self::Storage, Error> {
        let inclusive = self.inclusive_storage(exec, heads)?;
        let initial = singleton::<R, Values::Item>(exec, init)?;
        let output = exec.alloc_canonical::<Values::Item>(heads.len());
        inclusive.apply_init(exec, &initial, &output)?;
        Ok(output)
    }
}

pub(crate) fn segmented_inclusive_with<R, Values, Op>(
    exec: &Executor<R>,
    values: Values,
    heads: &DeviceVec<R, u32>,
    _op: Op,
) -> Result<Values::Storage, Error>
where
    R: Runtime,
    Values: SegmentedValues<R, Op>,
{
    values.inclusive_storage(exec, heads)
}

pub(crate) fn segmented_exclusive_with<R, Values, Op>(
    exec: &Executor<R>,
    values: Values,
    heads: &DeviceVec<R, u32>,
    init: Values::Item,
    _op: Op,
) -> Result<Values::Storage, Error>
where
    R: Runtime,
    Values: SegmentedValues<R, Op>,
{
    values.exclusive_storage(exec, heads, init)
}

pub(crate) fn segmented_reduced_with<R, Values, Op>(
    exec: &Executor<R>,
    values: Values,
    heads: &DeviceVec<R, u32>,
    init: Values::Item,
    _op: Op,
) -> Result<Values::Storage, Error>
where
    R: Runtime,
    Values: SegmentedValues<R, Op>,
{
    values.reduced_storage(exec, heads, init)
}

pub(crate) fn segment_heads_with<R, Keys, Equal>(
    exec: &Executor<R>,
    keys: Keys,
    _equal: Equal,
) -> Result<DeviceVec<R, u32>, Error>
where
    R: Runtime,
    Keys: SegmentKeyInput<R, Equal>,
{
    keys.segment_heads(exec)
}

/// Inclusive segmented scan over values grouped by adjacent equal keys.
pub(crate) fn inclusive_scan_by_key<R, Keys, Values, Equal, Op, Output>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    _equal: Equal,
    _op: Op,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Keys: SegmentKeyInput<R, Equal>,
    Values: SegmentedValues<R, Op>,
    Values::Storage: CanonicalStorage<R>,
    <Values::Storage as CanonicalStorage<R>>::Read: GatherInput<R, Counting, Output>,
{
    let heads = keys.segment_heads(exec)?;
    let scanned = values.inclusive_storage(exec, &heads)?;
    crate::indexed::gather_direct(exec, scanned.read(), Counting::new(0, heads.len()), output)
}

/// Exclusive segmented scan over values grouped by adjacent equal keys.
pub(crate) fn exclusive_scan_by_key<R, Keys, Values, Equal, Op, Output>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    _equal: Equal,
    init: Values::Item,
    _op: Op,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Keys: SegmentKeyInput<R, Equal>,
    Values: SegmentedValues<R, Op>,
    Values::Storage: CanonicalStorage<R>,
    <Values::Storage as CanonicalStorage<R>>::Read: GatherInput<R, Counting, Output>,
{
    let heads = keys.segment_heads(exec)?;
    let scanned = values.exclusive_storage(exec, &heads, init)?;
    crate::indexed::gather_direct(exec, scanned.read(), Counting::new(0, heads.len()), output)
}

/// Reduces each adjacent equal-key segment, emitting its first key and one
/// initialized value reduction.
pub(crate) fn reduce_by_key<R, Keys, Values, Equal, Op, KeyOutput, ValueOutput>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    _equal: Equal,
    init: Values::Item,
    _op: Op,
    key_output: KeyOutput,
    value_output: ValueOutput,
) -> Result<u32, Error>
where
    R: Runtime,
    Keys: Clone + SegmentKeyInput<R, Equal> + CopySelected<R, KeyOutput>,
    Values: SegmentedValues<R, Op>,
    Values::Storage: CanonicalStorage<R>,
    <Values::Storage as CanonicalStorage<R>>::Read: CopySelected<R, ValueOutput>,
{
    let heads = keys.clone().segment_heads(exec)?;
    let reduced = values.reduced_storage(exec, &heads, init)?;
    let head_control = SelectionControl::from_flags(exec, heads)?;
    let tail_control = tail_control_from_heads(exec, &head_control)?;
    let key_count = keys.copy_selected(exec, &head_control, key_output)?;
    let value_count = reduced
        .read()
        .copy_selected(exec, &tail_control, value_output)?;
    debug_assert_eq!(key_count, value_count);
    Ok(key_count)
}

/// Key phase for stable adjacent-key deduplication.
///
/// This trait mentions keys and key output only.  The resulting selection
/// control is applied to values by a separate trait, avoiding key-arity ×
/// value-arity dispatch.
#[doc(hidden)]
pub trait UniqueByKeyKeys<R: Runtime, Equal, KeyOutput>: ReadExpression + Sized {
    fn unique_key_len(&self) -> Result<usize, Error>;
    fn unique_key_control(
        self,
        exec: &Executor<R>,
        output: KeyOutput,
    ) -> Result<SelectionControl<R>, Error>;
}

impl<R, Keys, Equal, KeyOutput> UniqueByKeyKeys<R, Equal, KeyOutput> for Keys
where
    R: Runtime,
    Keys: ReadExpression
        + Clone
        + AdjacentFlagInput<R, UniqueHead<Equal>>
        + CopySelected<R, KeyOutput>,
    Equal: BinaryPredicateOp<Keys::Item>,
{
    fn unique_key_len(&self) -> Result<usize, Error> {
        self.source_len()
    }

    fn unique_key_control(
        self,
        exec: &Executor<R>,
        output: KeyOutput,
    ) -> Result<SelectionControl<R>, Error> {
        let flags = unique_head_flags::<R, _, Equal>(exec, self.clone())?;
        let control = SelectionControl::from_flags(exec, flags)?;
        self.copy_selected(exec, &control, output)?;
        Ok(control)
    }
}

/// Keeps the first key and value of every adjacent equal-key run.
///
/// `key_output` and `value_output` are preallocated and may be larger than the
/// returned logical length.
pub(crate) fn unique_by_key<R, Keys, Values, Equal, KeyOutput, ValueOutput>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    _equal: Equal,
    key_output: KeyOutput,
    value_output: ValueOutput,
) -> Result<u32, Error>
where
    R: Runtime,
    Keys: UniqueByKeyKeys<R, Equal, KeyOutput>,
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
    let control = keys.unique_key_control(exec, key_output)?;
    values.copy_selected(exec, &control, value_output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CanonicalStorage, Counting, Permute, Zip};
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    type Three = (u32, (u32, u32));
    struct EqualThree;

    #[cubecl::cube]
    impl BinaryPredicateOp<Three> for EqualThree {
        fn apply(lhs: Three, rhs: Three) -> bool {
            lhs.0 == rhs.0 && lhs.1.0 == rhs.1.0 && lhs.1.1 == rhs.1.1
        }
    }

    struct EqualU32;

    #[cubecl::cube]
    impl BinaryPredicateOp<u32> for EqualU32 {
        fn apply(lhs: u32, rhs: u32) -> bool {
            lhs == rhs
        }
    }

    type Seven = (u32, (u32, (u32, (u32, (u32, (u32, u32))))));
    struct SumSeven;

    #[cubecl::cube]
    impl crate::ReductionOp<Seven> for SumSeven {
        fn apply(lhs: Seven, rhs: Seven) -> Seven {
            (
                lhs.0 + rhs.0,
                (
                    lhs.1.0 + rhs.1.0,
                    (
                        lhs.1.1.0 + rhs.1.1.0,
                        (
                            lhs.1.1.1.0 + rhs.1.1.1.0,
                            (
                                lhs.1.1.1.1.0 + rhs.1.1.1.1.0,
                                (
                                    lhs.1.1.1.1.1.0 + rhs.1.1.1.1.1.0,
                                    lhs.1.1.1.1.1.1 + rhs.1.1.1.1.1.1,
                                ),
                            ),
                        ),
                    ),
                ),
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
        let out_keys: Vec<_> = (0..3).map(|_| exec.to_device(&[0_u32; 5])).collect();
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
        let key_output = Zip::new(
            Zip::new(out_keys[0].slice_mut(..), out_keys[1].slice_mut(..)),
            out_keys[2].slice_mut(..),
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

        let count = unique_by_key(
            &exec,
            keys,
            value_input,
            EqualThree,
            key_output,
            value_output,
        )
        .unwrap();
        assert_eq!(count, 3);
        assert_eq!(exec.to_host(&out_keys[0]).unwrap()[..3], [1, 1, 2]);
        assert_eq!(exec.to_host(&out_keys[1]).unwrap()[..3], [10, 11, 10]);
        assert_eq!(exec.to_host(&out_keys[2]).unwrap()[..3], [100, 100, 200]);
        for (column, output) in out_values.iter().enumerate() {
            let base = column as u32 * 100;
            assert_eq!(
                exec.to_host(output).unwrap()[..3],
                [base + 10, base + 12, base + 20]
            );
        }
    }

    #[test]
    fn unique_by_key_rejects_key_value_length_mismatch_before_control_build() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let keys = exec.to_device(&[1_u32, 1]);
        let values = exec.to_device(&[10_u32]);
        let out_keys = exec.to_device(&[0_u32; 2]);
        let out_values = exec.to_device(&[0_u32; 2]);
        assert_eq!(
            unique_by_key(
                &exec,
                keys.column(),
                values.column(),
                EqualU32,
                out_keys.slice_mut(..),
                out_values.slice_mut(..),
            ),
            Err(Error::LengthMismatch { left: 2, right: 1 })
        );
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

        let inclusive = exec.alloc_canonical::<Seven>(len);
        inclusive_scan_by_key(
            &exec,
            keys.column(),
            values(),
            EqualU32,
            SumSeven,
            inclusive.write(),
        )
        .unwrap();
        let first = exec.to_host(&inclusive.0.0.0.0.0.0).unwrap();
        let seventh = exec.to_host(&inclusive.1).unwrap();
        assert_eq!((first[299], first[300], first[599]), (300, 1, 300));
        assert_eq!((seventh[299], seventh[300], seventh[599]), (2100, 7, 2100));

        let init: Seven = (10, (20, (30, (40, (50, (60, 70))))));
        let exclusive = exec.alloc_canonical::<Seven>(len);
        exclusive_scan_by_key(
            &exec,
            keys.column(),
            values(),
            EqualU32,
            init,
            SumSeven,
            exclusive.write(),
        )
        .unwrap();
        let first = exec.to_host(&exclusive.0.0.0.0.0.0).unwrap();
        assert_eq!(
            (first[0], first[1], first[300], first[301]),
            (10, 11, 10, 11)
        );

        let key_output = exec.to_device(&vec![0_u32; len]);
        let value_output = exec.alloc_canonical::<Seven>(len);
        let count = reduce_by_key(
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
        assert_eq!(count, 2);
        assert_eq!(exec.to_host(&key_output.slice(..2)).unwrap(), vec![1, 2]);
        assert_eq!(
            exec.to_host(&value_output.0.0.0.0.0.0.slice(..2)).unwrap(),
            vec![310, 310]
        );
        assert_eq!(
            exec.to_host(&value_output.1.slice(..2)).unwrap(),
            vec![2170, 2170]
        );
    }
}
