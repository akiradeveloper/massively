//! Conflict-free application of already reduced scatter proposals.

use cubecl::prelude::*;

use crate::{
    A13, DeviceVec, Dispatch, Error, Executor, MStorageElement, ReadExpression, S12, StorageLayout,
    eval::Eval13,
    op::ReductionOp,
    output::{
        KernelOutputSlots, LowerOutputExpression, OutputBindings, OutputExpression,
        PaddedOutputSlots, StageOutput,
    },
    read::{Env0, Env12, Env13, KernelReadSlots, LowerReadExpression, PaddedReadSlots},
    reduce::{StageRead, StagedBindings},
    storage::{Decompose, LoadMutPadded12, Recompose, StorePadded12Expand},
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
    Expr: Eval13<Item, L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12>,
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
    indices: &[u32],
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
        let destination = indices[position] as usize;
        let proposal = Expr::eval13(
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

#[doc(hidden)]
pub trait ScatterCombineDispatch<R, Source, Output, ReadSlots, WriteSlots>
where
    R: Runtime,
{
    fn run<Op>(
        exec: &Executor<R>,
        source: &Source,
        indices: &DeviceVec<R, u32>,
        output: &Output,
    ) -> Result<(), Error>
    where
        Op: ReductionOp<Source::Item>,
        Source: ReadExpression;
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
    ScatterCombineDispatch<
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
    Output: OutputExpression<Item = Source::Item> + LowerOutputExpression + StageOutput<R, Env0>,
    Output::Slots: PaddedOutputSlots<Leaves = Leaves>,
    Source::Item: StorageLayout<StorageLeaves = Leaves>,
    <Source::Item as StorageLayout>::DeviceLayout:
        Decompose<Source::Item, Leaves = Leaves> + Recompose<Source::Item, Leaves = Leaves>,
{
    fn run<Op>(
        exec: &Executor<R>,
        source: &Source,
        indices: &DeviceVec<R, u32>,
        output: &Output,
    ) -> Result<(), Error>
    where
        Op: ReductionOp<Source::Item>,
    {
        let len = source.logical_len()?;
        if indices.len() != len {
            return Err(Error::LengthMismatch {
                left: len,
                right: indices.len(),
            });
        }
        if len == 0 {
            return Ok(());
        }
        let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
        let mut reads = StagedBindings::new();
        source.stage_at(exec.client(), exec.id(), &mut reads)?;
        reads.pad_to_thirteen(exec.client());
        let mut writes = OutputBindings::new();
        output.stage_output(exec.id(), &mut writes)?;
        writes.pad_to_twelve(exec.client());
        let source_offsets = exec
            .client()
            .create_from_slice(u32::as_bytes(&reads.offsets));
        let output_offsets = exec
            .client()
            .create_from_slice(u32::as_bytes(&writes.offsets));
        let len_handle = exec.client().create_from_slice(u32::as_bytes(&[len_u32]));
        unsafe {
            scatter_combine_a13::launch_unchecked::<
                Source::Item,
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
                <Source::Item as StorageLayout>::DeviceLayout,
                Op,
                R,
            >(
                exec.client(),
                crate::launch::cube_count_1d(len.div_ceil(BLOCK_SIZE as usize))?,
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
                BufferArg::from_raw_parts(source_offsets, reads.offsets.len()),
                BufferArg::from_raw_parts(indices.handle.clone(), indices.len()),
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

#[doc(hidden)]
pub trait ScatterCombineInput<R: Runtime, Output>: ReadExpression + Sized {
    fn scatter_combine<Op>(
        self,
        exec: &Executor<R>,
        indices: &DeviceVec<R, u32>,
        output: Output,
    ) -> Result<(), Error>
    where
        Op: ReductionOp<Self::Item>;
}

impl<R, Source, Output> ScatterCombineInput<R, Output> for Source
where
    R: Runtime,
    Source: ReadExpression + LowerReadExpression + StageRead<R, Env0>,
    Output: OutputExpression<Item = Source::Item> + LowerOutputExpression + StageOutput<R, Env0>,
    Output::Slots: PaddedOutputSlots,
    Dispatch<A13, S12>: ScatterCombineDispatch<
            R,
            Source,
            Output,
            KernelReadSlots<Source::Slots>,
            KernelOutputSlots<Output::Slots>,
        >,
{
    fn scatter_combine<Op>(
        self,
        exec: &Executor<R>,
        indices: &DeviceVec<R, u32>,
        output: Output,
    ) -> Result<(), Error>
    where
        Op: ReductionOp<Self::Item>,
    {
        <Dispatch<A13, S12> as ScatterCombineDispatch<
            R,
            Source,
            Output,
            KernelReadSlots<Source::Slots>,
            KernelOutputSlots<Output::Slots>,
        >>::run::<Op>(exec, &self, indices, &output)
    }
}

pub fn apply<R, Source, Output, Op>(
    exec: &Executor<R>,
    source: Source,
    indices: &DeviceVec<R, u32>,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Source: ScatterCombineInput<R, Output>,
    Op: ReductionOp<Source::Item>,
{
    source.scatter_combine::<Op>(exec, indices, output)
}
