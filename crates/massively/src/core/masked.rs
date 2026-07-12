//! Fixed-ABI indexed conditional copy primitive.

use cubecl::prelude::*;

use crate::{
    A13, Dispatch, Error, Executor, MStorageElement, ReadExpression, S12, StorageLayout,
    WritableFrom,
    eval::Eval13,
    output::{
        KernelOutputSlots, LowerOutputExpression, OutputBindings, OutputExpression,
        PaddedOutputSlots, StageOutput,
    },
    read::{Env0, Env12, Env13, KernelReadSlots, LowerReadExpression, PaddedReadSlots},
    reduce::{StageRead, StagedBindings},
    storage::{Decompose, StorePadded12, StorePadded12Expand},
};

const BLOCK_SIZE: u32 = 256;

#[cubecl::cube(launch_unchecked, explicit_define)]
fn masked_copy_a13<
    Source: CubeType + Send + Sync + 'static,
    Target: WritableFrom<Source> + Send + Sync + 'static,
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
    Expr: Eval13<Source, L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12>,
    Layout: Decompose<Target, Leaves = Leaves>,
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
    indices: &[u32],
    flags: &[u32],
    index_mode: &[u32],
    use_flags: &[u32],
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
    let position = ABSOLUTE_POS as usize;
    if position < len[0] as usize && (use_flags[0] == 0u32 || flags[position] != 0u32) {
        let source_position = if index_mode[0] == 2u32 {
            indices[position] as usize
        } else {
            position
        };
        let output_position = if index_mode[0] == 1u32 {
            indices[position] as usize
        } else {
            position
        };
        let source = Expr::eval13(
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
        Layout::decompose(Target::write_from(source)).store_padded(
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

#[doc(hidden)]
pub trait MaskedCopyDispatch<R, Source, Output, ReadSlots, WriteSlots>
where
    R: Runtime,
{
    fn run(
        exec: &Executor<R>,
        source: &Source,
        indices: Option<&crate::DeviceVec<R, u32>>,
        gather: bool,
        flags: Option<&crate::DeviceVec<R, u32>>,
        output: &Output,
    ) -> Result<(), Error>;
}

impl<
    R,
    Source,
    Output,
    Leaves,
    L0,
    L1,
    L2,
    L3,
    L4,
    L5,
    L6,
    L7,
    L8,
    L9,
    L10,
    L11,
    L12,
    O0,
    O1,
    O2,
    O3,
    O4,
    O5,
    O6,
    O7,
    O8,
    O9,
    O10,
    O11,
>
    MaskedCopyDispatch<
        R,
        Source,
        Output,
        Env13<L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12>,
        Env12<O0, O1, O2, O3, O4, O5, O6, O7, O8, O9, O10, O11>,
    > for Dispatch<A13, S12>
where
    R: Runtime,
    L0: MStorageElement,
    L1: MStorageElement,
    L2: MStorageElement,
    L3: MStorageElement,
    L4: MStorageElement,
    L5: MStorageElement,
    L6: MStorageElement,
    L7: MStorageElement,
    L8: MStorageElement,
    L9: MStorageElement,
    L10: MStorageElement,
    L11: MStorageElement,
    L12: MStorageElement,
    O0: MStorageElement,
    O1: MStorageElement,
    O2: MStorageElement,
    O3: MStorageElement,
    O4: MStorageElement,
    O5: MStorageElement,
    O6: MStorageElement,
    O7: MStorageElement,
    O8: MStorageElement,
    O9: MStorageElement,
    O10: MStorageElement,
    O11: MStorageElement,
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
    Source: ReadExpression + LowerReadExpression + StageRead<R, Env0>,
    Source::Slots: PaddedReadSlots<
            L0 = L0,
            L1 = L1,
            L2 = L2,
            L3 = L3,
            L4 = L4,
            L5 = L5,
            L6 = L6,
            L7 = L7,
            L8 = L8,
            L9 = L9,
            L10 = L10,
            L11 = L11,
            L12 = L12,
        >,
    Source::DeviceExpr: Eval13<Source::Item, L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12>,
    Output: OutputExpression + LowerOutputExpression + StageOutput<R, Env0>,
    Output::Slots: PaddedOutputSlots<Leaves = Leaves>,
    Output::Item: StorageLayout<StorageLeaves = Leaves> + WritableFrom<Source::Item>,
    <Output::Item as StorageLayout>::DeviceLayout: Decompose<Output::Item, Leaves = Leaves>,
{
    fn run(
        exec: &Executor<R>,
        source: &Source,
        indices: Option<&crate::DeviceVec<R, u32>>,
        gather: bool,
        flags: Option<&crate::DeviceVec<R, u32>>,
        output: &Output,
    ) -> Result<(), Error> {
        let source_len = source.logical_len()?;
        let output_len = output.logical_len()?;
        let operation_len = if gather {
            indices.map_or(source_len, crate::DeviceVec::len)
        } else {
            source_len
        };
        if !gather {
            if let Some(indices) = indices
                && operation_len != indices.len()
            {
                return Err(Error::LengthMismatch {
                    left: operation_len,
                    right: indices.len(),
                });
            }
        }
        if let Some(flags) = flags {
            if operation_len != flags.len() {
                return Err(Error::LengthMismatch {
                    left: operation_len,
                    right: flags.len(),
                });
            }
        }
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
        source.stage_at(exec.client(), exec.id(), &mut reads)?;
        reads.pad_to_thirteen(exec.client());
        let mut writes = OutputBindings::new();
        output.stage_output(exec.id(), &mut writes)?;
        writes.pad_to_twelve(exec.client());
        let read_offsets = exec
            .client()
            .create_from_slice(u32::as_bytes(&reads.offsets));
        let write_offsets = exec
            .client()
            .create_from_slice(u32::as_bytes(&writes.offsets));
        let len_handle = exec.client().create_from_slice(u32::as_bytes(&[len]));
        let index_mode = exec.client().create_from_slice(u32::as_bytes(&[if gather {
            2u32
        } else {
            u32::from(indices.is_some())
        }]));
        let use_flags = exec
            .client()
            .create_from_slice(u32::as_bytes(&[u32::from(flags.is_some())]));
        let dummy = exec.client().create_from_slice(u32::as_bytes(&[0u32]));
        let (indices_handle, indices_len) = indices
            .map(|values| (values.handle.clone(), values.len()))
            .unwrap_or((dummy.clone(), 1));
        let (flags_handle, flags_len) = flags
            .map(|values| (values.handle.clone(), values.len()))
            .unwrap_or((dummy, 1));
        unsafe {
            masked_copy_a13::launch_unchecked::<
                Source::Item,
                Output::Item,
                L0,
                L1,
                L2,
                L3,
                L4,
                L5,
                L6,
                L7,
                L8,
                L9,
                L10,
                L11,
                L12,
                O0,
                O1,
                O2,
                O3,
                O4,
                O5,
                O6,
                O7,
                O8,
                O9,
                O10,
                O11,
                Leaves,
                Source::DeviceExpr,
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
                BufferArg::from_raw_parts(indices_handle, indices_len),
                BufferArg::from_raw_parts(flags_handle, flags_len),
                BufferArg::from_raw_parts(index_mode, 1),
                BufferArg::from_raw_parts(use_flags, 1),
                BufferArg::from_raw_parts(len_handle, 1),
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

#[doc(hidden)]
pub trait MaskedCopyInput<R: Runtime, Output>: ReadExpression + Sized {
    fn masked_copy(
        self,
        exec: &Executor<R>,
        flags: &crate::DeviceVec<R, u32>,
        output: Output,
    ) -> Result<(), Error>;

    fn indexed_copy(
        self,
        exec: &Executor<R>,
        indices: &crate::DeviceVec<R, u32>,
        output: Output,
    ) -> Result<(), Error>;
}

impl<R, Source, Output> MaskedCopyInput<R, Output> for Source
where
    R: Runtime,
    Source: ReadExpression + LowerReadExpression + StageRead<R, Env0>,
    Output: OutputExpression + LowerOutputExpression + StageOutput<R, Env0>,
    Output::Slots: PaddedOutputSlots,
    Dispatch<A13, S12>: MaskedCopyDispatch<
            R,
            Source,
            Output,
            KernelReadSlots<Source::Slots>,
            KernelOutputSlots<Output::Slots>,
        >,
{
    fn masked_copy(
        self,
        exec: &Executor<R>,
        flags: &crate::DeviceVec<R, u32>,
        output: Output,
    ) -> Result<(), Error> {
        <Dispatch<A13, S12> as MaskedCopyDispatch<
            R,
            Source,
            Output,
            KernelReadSlots<Source::Slots>,
            KernelOutputSlots<Output::Slots>,
        >>::run(exec, &self, None, false, Some(flags), &output)
    }

    fn indexed_copy(
        self,
        exec: &Executor<R>,
        indices: &crate::DeviceVec<R, u32>,
        output: Output,
    ) -> Result<(), Error> {
        <Dispatch<A13, S12> as MaskedCopyDispatch<
            R,
            Source,
            Output,
            KernelReadSlots<Source::Slots>,
            KernelOutputSlots<Output::Slots>,
        >>::run(exec, &self, Some(indices), true, None, &output)
    }
}
