//! Conflict-free application of already reduced logical scatter proposals.

use cubecl::prelude::*;

use crate::{
    Error, Executor, ReadExpression, StorageLayout,
    eval::Eval13,
    op::ReductionOp,
    output::OutputBindings,
    read::{Env0, LowerReadExpression, PaddedReadSlots},
    reduce::{StageRead, StagedBindings},
    storage::{Decompose, LoadMutPadded12, Recompose, StorePadded12, StorePadded12Expand},
};

const BLOCK_SIZE: u32 = 256;

#[cubecl::cube(launch_unchecked, explicit_define)]
#[allow(clippy::too_many_arguments)]
fn scatter_combine_a13<
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
    Leaves: LoadMutPadded12<
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
        > + Send
        + Sync
        + 'static,
    SourceExpr: Eval13<Item, L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12>,
    IndexExpr: Eval13<usize, I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11, I12>,
    Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
    Op: ReductionOp<Item>,
>(
    source0: &[L0],
    source1: &[L1],
    source2: &[L2],
    source3: &[L3],
    source4: &[L4],
    source5: &[L5],
    source6: &[L6],
    source7: &[L7],
    source8: &[L8],
    source9: &[L9],
    source10: &[L10],
    source11: &[L11],
    source12: &[L12],
    source_offsets: &[u32],
    index0: &[I0],
    index1: &[I1],
    index2: &[I2],
    index3: &[I3],
    index4: &[I4],
    index5: &[I5],
    index6: &[I6],
    index7: &[I7],
    index8: &[I8],
    index9: &[I9],
    index10: &[I10],
    index11: &[I11],
    index12: &[I12],
    index_offsets: &[u32],
    positions: &[u32],
    len: &[u32],
    output0: &mut [O0],
    output1: &mut [O1],
    output2: &mut [O2],
    output3: &mut [O3],
    output4: &mut [O4],
    output5: &mut [O5],
    output6: &mut [O6],
    output7: &mut [O7],
    output8: &mut [O8],
    output9: &mut [O9],
    output10: &mut [O10],
    output11: &mut [O11],
    output_offsets: &[u32],
) {
    let position = ABSOLUTE_POS as usize;
    if position < len[0] as usize {
        let index_position = positions[position] as usize;
        let destination = IndexExpr::eval13(
            index0,
            index1,
            index2,
            index3,
            index4,
            index5,
            index6,
            index7,
            index8,
            index9,
            index10,
            index11,
            index12,
            index_offsets,
            index_position,
        );
        let proposal = SourceExpr::eval13(
            source0,
            source1,
            source2,
            source3,
            source4,
            source5,
            source6,
            source7,
            source8,
            source9,
            source10,
            source11,
            source12,
            source_offsets,
            position,
        );
        let previous = Layout::recompose(Leaves::load_mut_padded(
            output0,
            output1,
            output2,
            output3,
            output4,
            output5,
            output6,
            output7,
            output8,
            output9,
            output10,
            output11,
            output_offsets,
            destination,
        ));
        Layout::decompose(Op::apply(previous, proposal)).store_padded(
            output0,
            output1,
            output2,
            output3,
            output4,
            output5,
            output6,
            output7,
            output8,
            output9,
            output10,
            output11,
            output_offsets,
            destination,
        );
    }
}

/// Applies one conflict-free proposal per selected logical destination.
#[doc(hidden)]
pub trait ScatterCombineInput<R: Runtime, Indices, Output>: ReadExpression + Sized {
    fn scatter_combine<Op>(
        self,
        exec: &Executor<R>,
        indices: Indices,
        positions: &crate::DeviceVec<R, u32>,
        output: Output,
    ) -> Result<(), Error>
    where
        Op: ReductionOp<Self::Item>;
}

impl<R, Source, Indices, Output> ScatterCombineInput<R, Indices, Output> for Source
where
    R: Runtime,
    Source: ReadExpression + LowerReadExpression + StageRead<R, Env0>,
    Indices: ReadExpression<Item = usize> + LowerReadExpression + StageRead<R, Env0>,
    Output:
        crate::core::facade::KernelOutput<R> + crate::output::OutputExpression<Item = Source::Item>,
    Source::Item: StorageLayout,
    <Source::Item as StorageLayout>::StorageLeaves: LoadMutPadded12,
    <Source::Item as StorageLayout>::DeviceLayout: Decompose<Source::Item, Leaves = <Source::Item as StorageLayout>::StorageLeaves>
        + Recompose<Source::Item, Leaves = <Source::Item as StorageLayout>::StorageLeaves>,
{
    fn scatter_combine<Op>(
        self,
        exec: &Executor<R>,
        indices: Indices,
        positions: &crate::DeviceVec<R, u32>,
        output: Output,
    ) -> Result<(), Error>
    where
        Op: ReductionOp<Self::Item>,
    {
        let len = self.logical_len()?;
        if len != positions.len() {
            return Err(Error::LengthMismatch {
                left: len,
                right: positions.len(),
            });
        }
        if len == 0 {
            return Ok(());
        }

        let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
        let mut source_reads = StagedBindings::new();
        self.stage_at(exec.client(), exec.id(), &mut source_reads)?;
        source_reads.pad_to_thirteen(exec.client());
        let mut index_reads = StagedBindings::new();
        indices.stage_at(exec.client(), exec.id(), &mut index_reads)?;
        index_reads.pad_to_thirteen(exec.client());
        let mut writes = OutputBindings::new();
        output.stage_output(exec.id(), &mut writes)?;
        writes.pad_to_twelve(exec.client());

        let source_offsets = exec
            .client()
            .create_from_slice(u32::as_bytes(&source_reads.offsets));
        let index_offsets = exec
            .client()
            .create_from_slice(u32::as_bytes(&index_reads.offsets));
        let output_offsets = exec
            .client()
            .create_from_slice(u32::as_bytes(&writes.offsets));
        let len_handle = exec.client().create_from_slice(u32::as_bytes(&[len_u32]));

        unsafe {
            scatter_combine_a13::launch_unchecked::<
                Source::Item,
                <Source::Slots as PaddedReadSlots>::L0,
                <Source::Slots as PaddedReadSlots>::L1,
                <Source::Slots as PaddedReadSlots>::L2,
                <Source::Slots as PaddedReadSlots>::L3,
                <Source::Slots as PaddedReadSlots>::L4,
                <Source::Slots as PaddedReadSlots>::L5,
                <Source::Slots as PaddedReadSlots>::L6,
                <Source::Slots as PaddedReadSlots>::L7,
                <Source::Slots as PaddedReadSlots>::L8,
                <Source::Slots as PaddedReadSlots>::L9,
                <Source::Slots as PaddedReadSlots>::L10,
                <Source::Slots as PaddedReadSlots>::L11,
                <Source::Slots as PaddedReadSlots>::L12,
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
                <<Source::Item as StorageLayout>::StorageLeaves as StorePadded12>::O0,
                <<Source::Item as StorageLayout>::StorageLeaves as StorePadded12>::O1,
                <<Source::Item as StorageLayout>::StorageLeaves as StorePadded12>::O2,
                <<Source::Item as StorageLayout>::StorageLeaves as StorePadded12>::O3,
                <<Source::Item as StorageLayout>::StorageLeaves as StorePadded12>::O4,
                <<Source::Item as StorageLayout>::StorageLeaves as StorePadded12>::O5,
                <<Source::Item as StorageLayout>::StorageLeaves as StorePadded12>::O6,
                <<Source::Item as StorageLayout>::StorageLeaves as StorePadded12>::O7,
                <<Source::Item as StorageLayout>::StorageLeaves as StorePadded12>::O8,
                <<Source::Item as StorageLayout>::StorageLeaves as StorePadded12>::O9,
                <<Source::Item as StorageLayout>::StorageLeaves as StorePadded12>::O10,
                <<Source::Item as StorageLayout>::StorageLeaves as StorePadded12>::O11,
                <Source::Item as StorageLayout>::StorageLeaves,
                Source::DeviceExpr,
                Indices::DeviceExpr,
                <Source::Item as StorageLayout>::DeviceLayout,
                Op,
                R,
            >(
                exec.client(),
                crate::launch::cube_count_1d(len.div_ceil(BLOCK_SIZE as usize))?,
                CubeDim::new_1d(BLOCK_SIZE),
                BufferArg::from_raw_parts(source_reads.slots[0].0.clone(), source_reads.slots[0].1),
                BufferArg::from_raw_parts(source_reads.slots[1].0.clone(), source_reads.slots[1].1),
                BufferArg::from_raw_parts(source_reads.slots[2].0.clone(), source_reads.slots[2].1),
                BufferArg::from_raw_parts(source_reads.slots[3].0.clone(), source_reads.slots[3].1),
                BufferArg::from_raw_parts(source_reads.slots[4].0.clone(), source_reads.slots[4].1),
                BufferArg::from_raw_parts(source_reads.slots[5].0.clone(), source_reads.slots[5].1),
                BufferArg::from_raw_parts(source_reads.slots[6].0.clone(), source_reads.slots[6].1),
                BufferArg::from_raw_parts(source_reads.slots[7].0.clone(), source_reads.slots[7].1),
                BufferArg::from_raw_parts(source_reads.slots[8].0.clone(), source_reads.slots[8].1),
                BufferArg::from_raw_parts(source_reads.slots[9].0.clone(), source_reads.slots[9].1),
                BufferArg::from_raw_parts(
                    source_reads.slots[10].0.clone(),
                    source_reads.slots[10].1,
                ),
                BufferArg::from_raw_parts(
                    source_reads.slots[11].0.clone(),
                    source_reads.slots[11].1,
                ),
                BufferArg::from_raw_parts(
                    source_reads.slots[12].0.clone(),
                    source_reads.slots[12].1,
                ),
                BufferArg::from_raw_parts(source_offsets, source_reads.offsets.len()),
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
                BufferArg::from_raw_parts(positions.handle.clone(), positions.len()),
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
                BufferArg::from_raw_parts(output_offsets, writes.offsets.len()),
            );
        }
        Ok(())
    }
}

pub(crate) fn apply<R, Source, Indices, Output, Op>(
    exec: &Executor<R>,
    source: Source,
    indices: Indices,
    positions: &crate::DeviceVec<R, u32>,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Source: ScatterCombineInput<R, Indices, Output>,
    Op: ReductionOp<Source::Item>,
{
    source.scatter_combine::<Op>(exec, indices, positions, output)
}
