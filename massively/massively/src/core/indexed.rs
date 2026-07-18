//! Indexed copy primitives with independently evaluated value and index inputs.

use cubecl::prelude::*;

use crate::{
    Error, Executor, ReadExpression, StorageLayout,
    eval::Eval13,
    output::OutputBindings,
    read::{Env0, LowerReadExpression, PaddedReadSlots},
    reduce::{StageRead, StagedBindings},
    storage::{Decompose, StorePadded12, StorePadded12Expand},
};

const BLOCK_SIZE: u32 = 256;

#[cubecl::cube(launch_unchecked, explicit_define)]
fn indexed_copy_a13<
    Item: CubeType + Send + Sync + 'static,
    L0: CubePrimitive,
    L1: CubePrimitive,
    L2: CubePrimitive,
    L3: CubePrimitive,
    L4: CubePrimitive,
    L5: CubePrimitive,
    L6: CubePrimitive,
    L7: CubePrimitive,
    L8: CubePrimitive,
    L9: CubePrimitive,
    L10: CubePrimitive,
    L11: CubePrimitive,
    L12: CubePrimitive,
    I0: CubePrimitive,
    I1: CubePrimitive,
    I2: CubePrimitive,
    I3: CubePrimitive,
    I4: CubePrimitive,
    I5: CubePrimitive,
    I6: CubePrimitive,
    I7: CubePrimitive,
    I8: CubePrimitive,
    I9: CubePrimitive,
    I10: CubePrimitive,
    I11: CubePrimitive,
    I12: CubePrimitive,
    O0: CubePrimitive,
    O1: CubePrimitive,
    O2: CubePrimitive,
    O3: CubePrimitive,
    O4: CubePrimitive,
    O5: CubePrimitive,
    O6: CubePrimitive,
    O7: CubePrimitive,
    O8: CubePrimitive,
    O9: CubePrimitive,
    O10: CubePrimitive,
    O11: CubePrimitive,
    Leaves: CubeType
        + Send
        + Sync
        + 'static
        + StorePadded12<
            O0 = O0,
            O1 = O1,
            O2 = O2,
            O3 = O3,
            O4 = O4,
            O5 = O5,
            O6 = O6,
            O7 = O7,
            O8 = O8,
            O9 = O9,
            O10 = O10,
            O11 = O11,
        >,
    SourceExpr: Eval13<Item, L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12>,
    IndexExpr: Eval13<usize, I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11, I12>,
    Layout: Decompose<Item, Leaves = Leaves>,
>(
    slot0: &[L0],
    slot1: &[L1],
    slot2: &[L2],
    slot3: &[L3],
    slot4: &[L4],
    slot5: &[L5],
    slot6: &[L6],
    slot7: &[L7],
    slot8: &[L8],
    slot9: &[L9],
    slot10: &[L10],
    slot11: &[L11],
    slot12: &[L12],
    read_offsets: &[u32],
    index_slot0: &[I0],
    index_slot1: &[I1],
    index_slot2: &[I2],
    index_slot3: &[I3],
    index_slot4: &[I4],
    index_slot5: &[I5],
    index_slot6: &[I6],
    index_slot7: &[I7],
    index_slot8: &[I8],
    index_slot9: &[I9],
    index_slot10: &[I10],
    index_slot11: &[I11],
    index_slot12: &[I12],
    index_offsets: &[u32],
    selection: &[u32],
    use_selection: &[u32],
    gather: &[u32],
    len: &[u32],
    out0: &mut [O0],
    out1: &mut [O1],
    out2: &mut [O2],
    out3: &mut [O3],
    out4: &mut [O4],
    out5: &mut [O5],
    out6: &mut [O6],
    out7: &mut [O7],
    out8: &mut [O8],
    out9: &mut [O9],
    out10: &mut [O10],
    out11: &mut [O11],
    write_offsets: &[u32],
) {
    let compact_position = ABSOLUTE_POS as usize;
    if compact_position < len[0] as usize {
        let index_position = if use_selection[0] != 0u32 {
            selection[compact_position] as usize
        } else {
            compact_position
        };
        let indexed = IndexExpr::eval13(
            index_slot0,
            index_slot1,
            index_slot2,
            index_slot3,
            index_slot4,
            index_slot5,
            index_slot6,
            index_slot7,
            index_slot8,
            index_slot9,
            index_slot10,
            index_slot11,
            index_slot12,
            index_offsets,
            index_position,
        );
        let source_position = if gather[0] != 0u32 {
            indexed
        } else if use_selection[0] != 0u32 {
            selection[compact_position] as usize
        } else {
            compact_position
        };
        let output_position = if gather[0] != 0u32 {
            compact_position
        } else {
            indexed
        };
        let source = SourceExpr::eval13(
            slot0,
            slot1,
            slot2,
            slot3,
            slot4,
            slot5,
            slot6,
            slot7,
            slot8,
            slot9,
            slot10,
            slot11,
            slot12,
            read_offsets,
            source_position,
        );
        Layout::decompose(source).store_padded(
            out0,
            out1,
            out2,
            out3,
            out4,
            out5,
            out6,
            out7,
            out8,
            out9,
            out10,
            out11,
            write_offsets,
            output_position,
        );
    }
}

/// Internal capability for an indexed gather or scatter.
#[doc(hidden)]
pub trait IndexedCopyInput<R: Runtime, Indices, Output>: ReadExpression + Sized {
    fn indexed_copy(
        self,
        exec: &Executor<R>,
        indices: Indices,
        gather: bool,
        output: Output,
    ) -> Result<(), Error> {
        self.indexed_copy_selected(exec, indices, None, gather, output)
    }

    fn indexed_copy_selected(
        self,
        exec: &Executor<R>,
        indices: Indices,
        selection: Option<&crate::DeviceVec<R, u32>>,
        gather: bool,
        output: Output,
    ) -> Result<(), Error>;
}

impl<R, Values, Indices, Output> IndexedCopyInput<R, Indices, Output> for Values
where
    R: Runtime,
    Values: ReadExpression + LowerReadExpression + StageRead<R, Env0>,
    Values::Item: StorageLayout,
    Indices: ReadExpression<Item = usize> + LowerReadExpression + StageRead<R, Env0>,
    <Values::Item as StorageLayout>::StorageLeaves: StorePadded12,
    <<Values::Item as StorageLayout>::StorageLeaves as CubeType>::ExpandType: StorePadded12Expand,
    Output: crate::output::OutputExpression<Item = Values::Item>
        + crate::output::LowerOutputExpression
        + crate::output::StageOutput<R, Env0>,
    Output::Slots:
        crate::output::PaddedOutputSlots<Leaves = <Values::Item as StorageLayout>::StorageLeaves>,
{
    fn indexed_copy_selected(
        self,
        exec: &Executor<R>,
        indices: Indices,
        selection: Option<&crate::DeviceVec<R, u32>>,
        gather: bool,
        output: Output,
    ) -> Result<(), Error> {
        let values_len = self.logical_len()?;
        let indices_len = indices.logical_len()?;
        let unselected_len = if gather { indices_len } else { values_len };
        if !gather && values_len != indices_len {
            return Err(Error::LengthMismatch {
                left: values_len,
                right: indices_len,
            });
        }
        let operation_len = selection.map_or(unselected_len, crate::DeviceVec::len);
        let output_len = output.logical_len()?;
        if output_len < operation_len {
            return Err(Error::OutputTooShort {
                input: operation_len,
                output: output_len,
            });
        }
        if operation_len == 0 {
            return Ok(());
        }

        let len = u32::try_from(operation_len)
            .map_err(|_| Error::LengthTooLarge { len: operation_len })?;
        let mut reads = StagedBindings::new();
        self.stage_at(exec.client(), exec.id(), &mut reads)?;
        reads.pad_to_thirteen(exec.client());
        let mut index_reads = StagedBindings::new();
        indices.stage_at(exec.client(), exec.id(), &mut index_reads)?;
        index_reads.pad_to_thirteen(exec.client());
        let mut writes = OutputBindings::new();
        output.stage_output(exec.id(), &mut writes)?;
        writes.pad_to_twelve(exec.client());

        let read_offsets = exec
            .client()
            .create_from_slice(u32::as_bytes(&reads.offsets));
        let index_offsets = exec
            .client()
            .create_from_slice(u32::as_bytes(&index_reads.offsets));
        let write_offsets = exec
            .client()
            .create_from_slice(u32::as_bytes(&writes.offsets));
        let gather = exec
            .client()
            .create_from_slice(u32::as_bytes(&[u32::from(gather)]));
        let use_selection = exec
            .client()
            .create_from_slice(u32::as_bytes(&[u32::from(selection.is_some())]));
        let selection = selection
            .map(|selection| selection.handle.clone())
            .unwrap_or_else(|| exec.client().create_from_slice(u32::as_bytes(&[0u32])));
        let len = exec.client().create_from_slice(u32::as_bytes(&[len]));

        unsafe {
            indexed_copy_a13::launch_unchecked::<
                Values::Item,
                <Values::Slots as PaddedReadSlots>::L0,
                <Values::Slots as PaddedReadSlots>::L1,
                <Values::Slots as PaddedReadSlots>::L2,
                <Values::Slots as PaddedReadSlots>::L3,
                <Values::Slots as PaddedReadSlots>::L4,
                <Values::Slots as PaddedReadSlots>::L5,
                <Values::Slots as PaddedReadSlots>::L6,
                <Values::Slots as PaddedReadSlots>::L7,
                <Values::Slots as PaddedReadSlots>::L8,
                <Values::Slots as PaddedReadSlots>::L9,
                <Values::Slots as PaddedReadSlots>::L10,
                <Values::Slots as PaddedReadSlots>::L11,
                <Values::Slots as PaddedReadSlots>::L12,
                <Indices::Slots as PaddedReadSlots>::L0,
                <Indices::Slots as PaddedReadSlots>::L1,
                <Indices::Slots as PaddedReadSlots>::L2,
                <Indices::Slots as PaddedReadSlots>::L3,
                <Indices::Slots as PaddedReadSlots>::L4,
                <Indices::Slots as PaddedReadSlots>::L5,
                <Indices::Slots as PaddedReadSlots>::L6,
                <Indices::Slots as PaddedReadSlots>::L7,
                <Indices::Slots as PaddedReadSlots>::L8,
                <Indices::Slots as PaddedReadSlots>::L9,
                <Indices::Slots as PaddedReadSlots>::L10,
                <Indices::Slots as PaddedReadSlots>::L11,
                <Indices::Slots as PaddedReadSlots>::L12,
                <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O0,
                <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O1,
                <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O2,
                <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O3,
                <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O4,
                <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O5,
                <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O6,
                <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O7,
                <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O8,
                <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O9,
                <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O10,
                <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O11,
                <Output::Item as StorageLayout>::StorageLeaves,
                Values::DeviceExpr,
                Indices::DeviceExpr,
                <Output::Item as StorageLayout>::DeviceLayout,
                R,
            >(
                exec.client(),
                crate::launch::cube_count_1d(operation_len.div_ceil(BLOCK_SIZE as usize))?,
                CubeDim::new_1d(BLOCK_SIZE),
                BufferArg::from_raw_parts(reads.slots[0].0.clone(), reads.slots[0].1),
                BufferArg::from_raw_parts(reads.slots[1].0.clone(), reads.slots[1].1),
                BufferArg::from_raw_parts(reads.slots[2].0.clone(), reads.slots[2].1),
                BufferArg::from_raw_parts(reads.slots[3].0.clone(), reads.slots[3].1),
                BufferArg::from_raw_parts(reads.slots[4].0.clone(), reads.slots[4].1),
                BufferArg::from_raw_parts(reads.slots[5].0.clone(), reads.slots[5].1),
                BufferArg::from_raw_parts(reads.slots[6].0.clone(), reads.slots[6].1),
                BufferArg::from_raw_parts(reads.slots[7].0.clone(), reads.slots[7].1),
                BufferArg::from_raw_parts(reads.slots[8].0.clone(), reads.slots[8].1),
                BufferArg::from_raw_parts(reads.slots[9].0.clone(), reads.slots[9].1),
                BufferArg::from_raw_parts(reads.slots[10].0.clone(), reads.slots[10].1),
                BufferArg::from_raw_parts(reads.slots[11].0.clone(), reads.slots[11].1),
                BufferArg::from_raw_parts(reads.slots[12].0.clone(), reads.slots[12].1),
                BufferArg::from_raw_parts(read_offsets, reads.offsets.len()),
                BufferArg::from_raw_parts(index_reads.slots[0].0.clone(), index_reads.slots[0].1),
                BufferArg::from_raw_parts(index_reads.slots[1].0.clone(), index_reads.slots[1].1),
                BufferArg::from_raw_parts(index_reads.slots[2].0.clone(), index_reads.slots[2].1),
                BufferArg::from_raw_parts(index_reads.slots[3].0.clone(), index_reads.slots[3].1),
                BufferArg::from_raw_parts(index_reads.slots[4].0.clone(), index_reads.slots[4].1),
                BufferArg::from_raw_parts(index_reads.slots[5].0.clone(), index_reads.slots[5].1),
                BufferArg::from_raw_parts(index_reads.slots[6].0.clone(), index_reads.slots[6].1),
                BufferArg::from_raw_parts(index_reads.slots[7].0.clone(), index_reads.slots[7].1),
                BufferArg::from_raw_parts(index_reads.slots[8].0.clone(), index_reads.slots[8].1),
                BufferArg::from_raw_parts(index_reads.slots[9].0.clone(), index_reads.slots[9].1),
                BufferArg::from_raw_parts(index_reads.slots[10].0.clone(), index_reads.slots[10].1),
                BufferArg::from_raw_parts(index_reads.slots[11].0.clone(), index_reads.slots[11].1),
                BufferArg::from_raw_parts(index_reads.slots[12].0.clone(), index_reads.slots[12].1),
                BufferArg::from_raw_parts(index_offsets, index_reads.offsets.len()),
                BufferArg::from_raw_parts(selection, operation_len.max(1)),
                BufferArg::from_raw_parts(use_selection, 1),
                BufferArg::from_raw_parts(gather, 1),
                BufferArg::from_raw_parts(len, 1),
                BufferArg::from_raw_parts(writes.slots[0].0.clone(), writes.slots[0].1),
                BufferArg::from_raw_parts(writes.slots[1].0.clone(), writes.slots[1].1),
                BufferArg::from_raw_parts(writes.slots[2].0.clone(), writes.slots[2].1),
                BufferArg::from_raw_parts(writes.slots[3].0.clone(), writes.slots[3].1),
                BufferArg::from_raw_parts(writes.slots[4].0.clone(), writes.slots[4].1),
                BufferArg::from_raw_parts(writes.slots[5].0.clone(), writes.slots[5].1),
                BufferArg::from_raw_parts(writes.slots[6].0.clone(), writes.slots[6].1),
                BufferArg::from_raw_parts(writes.slots[7].0.clone(), writes.slots[7].1),
                BufferArg::from_raw_parts(writes.slots[8].0.clone(), writes.slots[8].1),
                BufferArg::from_raw_parts(writes.slots[9].0.clone(), writes.slots[9].1),
                BufferArg::from_raw_parts(writes.slots[10].0.clone(), writes.slots[10].1),
                BufferArg::from_raw_parts(writes.slots[11].0.clone(), writes.slots[11].1),
                BufferArg::from_raw_parts(write_offsets, writes.offsets.len()),
            );
        }
        Ok(())
    }
}

/// Internal gather capability retained for selection and ordering controls.
#[doc(hidden)]
pub trait GatherInput<R: Runtime, Indices, Output>: ReadExpression + Sized {
    fn gather(self, exec: &Executor<R>, indices: Indices, output: Output) -> Result<(), Error>;
}

impl<R, Values, Indices, Output> GatherInput<R, Indices, Output> for Values
where
    R: Runtime,
    Values: IndexedCopyInput<R, Indices, Output>,
{
    fn gather(self, exec: &Executor<R>, indices: Indices, output: Output) -> Result<(), Error> {
        self.indexed_copy(exec, indices, true, output)
    }
}

pub(crate) fn gather_direct<R, Values, Indices, Output>(
    exec: &Executor<R>,
    values: Values,
    indices: Indices,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Values: GatherInput<R, Indices, Output>,
{
    values.gather(exec, indices, output)
}

/// Gathers through a physically stored `u32` index expression.
///
/// The conversion is part of the read expression, so the indexed kernel still
/// receives logical `usize` values.
pub(crate) fn gather_u32<R, Values, Indices, Output>(
    exec: &Executor<R>,
    values: Values,
    indices: Indices,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Values: GatherInput<R, crate::Transform<Indices, crate::op::U32ToUsize>, Output>,
{
    gather_direct(
        exec,
        values,
        crate::Transform::new(indices, crate::op::U32ToUsize),
        output,
    )
}

pub(crate) fn apply_permutation<R, Input, Output>(
    exec: &Executor<R>,
    input: Input,
    indices: crate::Column<u32>,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: GatherInput<R, crate::Transform<crate::Column<u32>, crate::op::U32ToUsize>, Output>,
{
    gather_u32(exec, input, indices, output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Counting, Permute, ReverseCounting, RowStorage, Transform, Zip, read::FixedRead};
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn gather_seven_columns_uses_independent_index_expression() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let inputs: Vec<_> = (0_u32..7)
            .map(|base| exec.to_device(&[base * 10 + 1, base * 10 + 2, base * 10 + 3]))
            .collect();
        let outputs: Vec<_> = (0..7).map(|_| exec.to_device(&[0_u32; 2])).collect();
        let indices = exec.to_device(&[2_u32, 0]);
        let values = Zip::new(
            inputs[0].column(),
            Zip::new(
                inputs[1].column(),
                Zip::new(
                    inputs[2].column(),
                    Zip::new(
                        inputs[3].column(),
                        Zip::new(
                            inputs[4].column(),
                            Zip::new(inputs[5].column(), inputs[6].column()),
                        ),
                    ),
                ),
            ),
        );
        let output = Zip::new(
            Zip::new(
                Zip::new(
                    Zip::new(
                        Zip::new(
                            Zip::new(outputs[0].slice_mut(..), outputs[1].slice_mut(..)),
                            outputs[2].slice_mut(..),
                        ),
                        outputs[3].slice_mut(..),
                    ),
                    outputs[4].slice_mut(..),
                ),
                outputs[5].slice_mut(..),
            ),
            outputs[6].slice_mut(..),
        );

        gather_u32(&exec, FixedRead::new(values), indices.column(), output).unwrap();
        for (column, output) in outputs.iter().enumerate() {
            assert_eq!(
                exec.to_host(output).unwrap(),
                vec![column as u32 * 10 + 3, column as u32 * 10 + 1]
            );
        }
    }

    #[test]
    fn gather_accepts_logical_usize_indices() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let values = exec.to_device(&[10_u32, 20, 30]);
        let encoded = exec.to_device(&[2_u32, 0]);
        let indices = crate::Transform::new(encoded.column(), crate::op::U32ToUsize);
        let output = exec.alloc::<u32>(2);

        gather_direct(&exec, values.column(), indices, output.slice_mut(..)).unwrap();

        assert_eq!(exec.to_host(&output).unwrap(), vec![30, 10]);
    }

    #[test]
    fn gather_accepts_lazy_indices_independently() {
        type Seven = (u32, u32, u32, u32, u32, u32, u32);
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let inputs: Vec<_> = (0_u32..7)
            .map(|base| exec.to_device(&[base * 10 + 1, base * 10 + 2, base * 10 + 3]))
            .collect();
        let seven = Zip::new(
            inputs[0].column(),
            Zip::new(
                inputs[1].column(),
                Zip::new(
                    inputs[2].column(),
                    Zip::new(
                        inputs[3].column(),
                        Zip::new(
                            inputs[4].column(),
                            Zip::new(inputs[5].column(), inputs[6].column()),
                        ),
                    ),
                ),
            ),
        );
        let values = Permute::new(
            seven,
            Transform::new(Counting::new(0, 3), crate::op::U32ToUsize),
        );
        let raw_indices = exec.to_device(&[2_u32, 0]);
        let indices = Transform::new(
            Permute::new(
                raw_indices.column(),
                Transform::new(Counting::new(0, 2), crate::op::U32ToUsize),
            ),
            crate::op::U32ToUsize,
        );
        let output = exec.alloc_row::<Seven>(2);

        gather_direct(&exec, FixedRead::new(values), indices, output.write()).unwrap();
        let (first, _, _, _, _, _, last) = crate::MStorage::into_columns(output);
        assert_eq!(exec.to_host(&first).unwrap(), vec![3, 1]);
        assert_eq!(exec.to_host(&last).unwrap(), vec![63, 61]);
    }

    #[test]
    fn reverse_counting_remains_a_valid_internal_index() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let values = exec.to_device(&[10_u32, 20, 30]);
        let output = exec.alloc::<u32>(3);

        gather_u32(
            &exec,
            values.column(),
            ReverseCounting::new(3),
            output.slice_mut(..),
        )
        .unwrap();

        assert_eq!(exec.to_host(&output).unwrap(), vec![30, 20, 10]);
    }
}
