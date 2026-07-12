//! Conflict-free application of already reduced scatter proposals.

use cubecl::prelude::*;

use crate::{
    A13, DeviceVec, Dispatch, Error, Executor, MStorageElement, ReadExpression, StorageLayout,
    eval::Eval13,
    op::ReductionOp,
    output::{
        LowerOutputExpression, OutputBindings, OutputExpression, PaddedOutputSlots, StageOutput,
    },
    read::{Env0, Env13, LowerReadExpression},
    reduce::{StageRead, StagedBindings},
    storage::{
        Decompose, LoadMutLeaves1, LoadMutLeaves2, LoadMutLeaves3, LoadMutLeaves4, LoadMutLeaves5,
        LoadMutLeaves6, LoadMutLeaves7, LoadMutLeaves8, LoadMutLeaves9, LoadMutLeaves10,
        LoadMutLeaves11, LoadMutLeaves12, Recompose, StoreLeaves1, StoreLeaves2, StoreLeaves3,
        StoreLeaves4, StoreLeaves5, StoreLeaves6, StoreLeaves7, StoreLeaves8, StoreLeaves9,
        StoreLeaves10, StoreLeaves11, StoreLeaves12,
    },
};

const BLOCK_SIZE: u32 = 256;

macro_rules! define_scatter_combine_kernel {
    ($name:ident, $leaves:ty, $load:ident, $store:ident; [$($oty:ident:$output:ident),+]; [$($dummy:ident),*]) => {
        #[cubecl::cube(launch_unchecked, explicit_define)]
        #[allow(unused_variables, clippy::too_many_arguments)]
        fn $name<
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
            $($oty: CubePrimitive,)+
            Expr: Eval13<Item, L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12>,
            Layout: Decompose<Item, Leaves = $leaves> + Recompose<Item, Leaves = $leaves>,
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
            $($output: &mut [$oty],)+
            $($dummy: &mut [u32],)*
            output_offsets: &[u32],
        ) where
            $leaves: $load<$($oty),+> + $store<$($oty),+>,
        {
            let position = ABSOLUTE_POS as usize;
            if position < len[0] as usize {
                let destination = indices[position] as usize;
                let proposal = Expr::eval13(
                    source0, source1, source2, source3, source4, source5, source6,
                    source7, source8, source9, source10, source11, source12,
                    source_offsets, position,
                );
                let previous = Layout::recompose(<$leaves as $load<$($oty),+>>::load_mut(
                    $($output,)+ output_offsets, destination,
                ));
                <$leaves as $store<$($oty),+>>::store(
                    Layout::decompose(Op::apply(previous, proposal)),
                    $($output,)+ output_offsets, destination,
                );
            }
        }
    };
}

define_scatter_combine_kernel!(scatter_combine_s1, crate::storage::Last<O0>, LoadMutLeaves1, StoreLeaves1; [O0:output0]; [output1,output2,output3,output4,output5,output6,output7,output8,output9,output10,output11]);
define_scatter_combine_kernel!(scatter_combine_s2, crate::storage::More<O0, crate::storage::Last<O1>>, LoadMutLeaves2, StoreLeaves2; [O0:output0,O1:output1]; [output2,output3,output4,output5,output6,output7,output8,output9,output10,output11]);
define_scatter_combine_kernel!(scatter_combine_s3, crate::storage::More<O0, crate::storage::More<O1, crate::storage::Last<O2>>>, LoadMutLeaves3, StoreLeaves3; [O0:output0,O1:output1,O2:output2]; [output3,output4,output5,output6,output7,output8,output9,output10,output11]);
define_scatter_combine_kernel!(scatter_combine_s4, crate::storage::More<O0, crate::storage::More<O1, crate::storage::More<O2, crate::storage::Last<O3>>>>, LoadMutLeaves4, StoreLeaves4; [O0:output0,O1:output1,O2:output2,O3:output3]; [output4,output5,output6,output7,output8,output9,output10,output11]);
define_scatter_combine_kernel!(scatter_combine_s5, crate::storage::More<O0, crate::storage::More<O1, crate::storage::More<O2, crate::storage::More<O3, crate::storage::Last<O4>>>>>, LoadMutLeaves5, StoreLeaves5; [O0:output0,O1:output1,O2:output2,O3:output3,O4:output4]; [output5,output6,output7,output8,output9,output10,output11]);
define_scatter_combine_kernel!(scatter_combine_s6, crate::storage::More<O0, crate::storage::More<O1, crate::storage::More<O2, crate::storage::More<O3, crate::storage::More<O4, crate::storage::Last<O5>>>>>>, LoadMutLeaves6, StoreLeaves6; [O0:output0,O1:output1,O2:output2,O3:output3,O4:output4,O5:output5]; [output6,output7,output8,output9,output10,output11]);
define_scatter_combine_kernel!(scatter_combine_s7, crate::storage::More<O0, crate::storage::More<O1, crate::storage::More<O2, crate::storage::More<O3, crate::storage::More<O4, crate::storage::More<O5, crate::storage::Last<O6>>>>>>>, LoadMutLeaves7, StoreLeaves7; [O0:output0,O1:output1,O2:output2,O3:output3,O4:output4,O5:output5,O6:output6]; [output7,output8,output9,output10,output11]);
define_scatter_combine_kernel!(scatter_combine_s8, crate::storage::More<O0, crate::storage::More<O1, crate::storage::More<O2, crate::storage::More<O3, crate::storage::More<O4, crate::storage::More<O5, crate::storage::More<O6, crate::storage::Last<O7>>>>>>>>, LoadMutLeaves8, StoreLeaves8; [O0:output0,O1:output1,O2:output2,O3:output3,O4:output4,O5:output5,O6:output6,O7:output7]; [output8,output9,output10,output11]);
define_scatter_combine_kernel!(scatter_combine_s9, crate::storage::More<O0, crate::storage::More<O1, crate::storage::More<O2, crate::storage::More<O3, crate::storage::More<O4, crate::storage::More<O5, crate::storage::More<O6, crate::storage::More<O7, crate::storage::Last<O8>>>>>>>>>, LoadMutLeaves9, StoreLeaves9; [O0:output0,O1:output1,O2:output2,O3:output3,O4:output4,O5:output5,O6:output6,O7:output7,O8:output8]; [output9,output10,output11]);
define_scatter_combine_kernel!(scatter_combine_s10, crate::storage::More<O0, crate::storage::More<O1, crate::storage::More<O2, crate::storage::More<O3, crate::storage::More<O4, crate::storage::More<O5, crate::storage::More<O6, crate::storage::More<O7, crate::storage::More<O8, crate::storage::Last<O9>>>>>>>>>>, LoadMutLeaves10, StoreLeaves10; [O0:output0,O1:output1,O2:output2,O3:output3,O4:output4,O5:output5,O6:output6,O7:output7,O8:output8,O9:output9]; [output10,output11]);
define_scatter_combine_kernel!(scatter_combine_s11, crate::storage::More<O0, crate::storage::More<O1, crate::storage::More<O2, crate::storage::More<O3, crate::storage::More<O4, crate::storage::More<O5, crate::storage::More<O6, crate::storage::More<O7, crate::storage::More<O8, crate::storage::More<O9, crate::storage::Last<O10>>>>>>>>>>>, LoadMutLeaves11, StoreLeaves11; [O0:output0,O1:output1,O2:output2,O3:output3,O4:output4,O5:output5,O6:output6,O7:output7,O8:output8,O9:output9,O10:output10]; [output11]);
define_scatter_combine_kernel!(scatter_combine_s12, crate::storage::More<O0, crate::storage::More<O1, crate::storage::More<O2, crate::storage::More<O3, crate::storage::More<O4, crate::storage::More<O5, crate::storage::More<O6, crate::storage::More<O7, crate::storage::More<O8, crate::storage::More<O9, crate::storage::More<O10, crate::storage::Last<O11>>>>>>>>>>>>, LoadMutLeaves12, StoreLeaves12; [O0:output0,O1:output1,O2:output2,O3:output3,O4:output4,O5:output5,O6:output6,O7:output7,O8:output8,O9:output9,O10:output10,O11:output11]; []);

pub trait ScatterCombineDispatch<R, Source, Output>
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

macro_rules! impl_scatter_combine_dispatch {
    ($storage:ty, $kernel:ident, $leaves:ty, $load:ident, $write_env:ty; [$($oty:ident),+]; [$($padded:ty),+]) => {
        impl<R, Item, Source, Output, L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12, $($oty),+>
            ScatterCombineDispatch<R, Source, Output> for Dispatch<A13, $storage>
        where
            R: Runtime,
            Item: StorageLayout<StorageArity = $storage, StorageLeaves = $leaves>,
            Item::DeviceLayout: Decompose<Item, Leaves = $leaves> + Recompose<Item, Leaves = $leaves>,
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
            $($oty: MStorageElement,)+
            Source: ReadExpression<Item = Item>
                + LowerReadExpression<Slots = Env13<L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11,L12>>
                + StageRead<R, Env0>,
            Source::DeviceExpr: Eval13<Item,L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11,L12>,
            Output: OutputExpression<Item = Item, StorageArity = $storage>
                + LowerOutputExpression<Slots = $write_env>
                + StageOutput<R, Env0>,
            Output::Slots: PaddedOutputSlots<Leaves = $leaves>,
            $leaves: $load<$($oty),+>,
        {
            fn run<Op>(exec: &Executor<R>, source: &Source, indices: &DeviceVec<R, u32>, output: &Output) -> Result<(), Error>
            where
                Op: ReductionOp<Item>,
            {
                let len = source.logical_len()?;
                if indices.len() != len {
                    return Err(Error::LengthMismatch { left: len, right: indices.len() });
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
                let source_offsets = exec.client().create_from_slice(u32::as_bytes(&reads.offsets));
                let output_offsets = exec.client().create_from_slice(u32::as_bytes(&writes.offsets));
                let len_handle = exec.client().create_from_slice(u32::as_bytes(&[len_u32]));
                unsafe {
                    $kernel::launch_unchecked::<
                        Item,L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11,L12,
                        $($oty,)+
                        Source::DeviceExpr,Item::DeviceLayout,Op,R,
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
    };
}

impl_scatter_combine_dispatch!(crate::S1,scatter_combine_s1,crate::storage::Last<O0>,LoadMutLeaves1,crate::read::Env1<O0>; [O0]; [O0,u32,u32,u32,u32,u32,u32,u32,u32,u32,u32,u32]);
impl_scatter_combine_dispatch!(crate::S2,scatter_combine_s2,crate::storage::More<O0,crate::storage::Last<O1>>,LoadMutLeaves2,crate::read::Env2<O0,O1>; [O0,O1]; [O0,O1,u32,u32,u32,u32,u32,u32,u32,u32,u32,u32]);
impl_scatter_combine_dispatch!(crate::S3,scatter_combine_s3,crate::storage::More<O0,crate::storage::More<O1,crate::storage::Last<O2>>>,LoadMutLeaves3,crate::read::Env3<O0,O1,O2>; [O0,O1,O2]; [O0,O1,O2,u32,u32,u32,u32,u32,u32,u32,u32,u32]);
impl_scatter_combine_dispatch!(crate::S4,scatter_combine_s4,crate::storage::More<O0,crate::storage::More<O1,crate::storage::More<O2,crate::storage::Last<O3>>>>,LoadMutLeaves4,crate::read::Env4<O0,O1,O2,O3>; [O0,O1,O2,O3]; [O0,O1,O2,O3,u32,u32,u32,u32,u32,u32,u32,u32]);
impl_scatter_combine_dispatch!(crate::S5,scatter_combine_s5,crate::storage::More<O0,crate::storage::More<O1,crate::storage::More<O2,crate::storage::More<O3,crate::storage::Last<O4>>>>>,LoadMutLeaves5,crate::read::Env5<O0,O1,O2,O3,O4>; [O0,O1,O2,O3,O4]; [O0,O1,O2,O3,O4,u32,u32,u32,u32,u32,u32,u32]);
impl_scatter_combine_dispatch!(crate::S6,scatter_combine_s6,crate::storage::More<O0,crate::storage::More<O1,crate::storage::More<O2,crate::storage::More<O3,crate::storage::More<O4,crate::storage::Last<O5>>>>>>,LoadMutLeaves6,crate::read::Env6<O0,O1,O2,O3,O4,O5>; [O0,O1,O2,O3,O4,O5]; [O0,O1,O2,O3,O4,O5,u32,u32,u32,u32,u32,u32]);
impl_scatter_combine_dispatch!(crate::S7,scatter_combine_s7,crate::storage::More<O0,crate::storage::More<O1,crate::storage::More<O2,crate::storage::More<O3,crate::storage::More<O4,crate::storage::More<O5,crate::storage::Last<O6>>>>>>>,LoadMutLeaves7,crate::read::Env7<O0,O1,O2,O3,O4,O5,O6>; [O0,O1,O2,O3,O4,O5,O6]; [O0,O1,O2,O3,O4,O5,O6,u32,u32,u32,u32,u32]);
impl_scatter_combine_dispatch!(crate::S8,scatter_combine_s8,crate::storage::More<O0,crate::storage::More<O1,crate::storage::More<O2,crate::storage::More<O3,crate::storage::More<O4,crate::storage::More<O5,crate::storage::More<O6,crate::storage::Last<O7>>>>>>>>,LoadMutLeaves8,crate::read::Env8<O0,O1,O2,O3,O4,O5,O6,O7>; [O0,O1,O2,O3,O4,O5,O6,O7]; [O0,O1,O2,O3,O4,O5,O6,O7,u32,u32,u32,u32]);
impl_scatter_combine_dispatch!(crate::S9,scatter_combine_s9,crate::storage::More<O0,crate::storage::More<O1,crate::storage::More<O2,crate::storage::More<O3,crate::storage::More<O4,crate::storage::More<O5,crate::storage::More<O6,crate::storage::More<O7,crate::storage::Last<O8>>>>>>>>>,LoadMutLeaves9,crate::read::Env9<O0,O1,O2,O3,O4,O5,O6,O7,O8>; [O0,O1,O2,O3,O4,O5,O6,O7,O8]; [O0,O1,O2,O3,O4,O5,O6,O7,O8,u32,u32,u32]);
impl_scatter_combine_dispatch!(crate::S10,scatter_combine_s10,crate::storage::More<O0,crate::storage::More<O1,crate::storage::More<O2,crate::storage::More<O3,crate::storage::More<O4,crate::storage::More<O5,crate::storage::More<O6,crate::storage::More<O7,crate::storage::More<O8,crate::storage::Last<O9>>>>>>>>>>,LoadMutLeaves10,crate::read::Env10<O0,O1,O2,O3,O4,O5,O6,O7,O8,O9>; [O0,O1,O2,O3,O4,O5,O6,O7,O8,O9]; [O0,O1,O2,O3,O4,O5,O6,O7,O8,O9,u32,u32]);
impl_scatter_combine_dispatch!(crate::S11,scatter_combine_s11,crate::storage::More<O0,crate::storage::More<O1,crate::storage::More<O2,crate::storage::More<O3,crate::storage::More<O4,crate::storage::More<O5,crate::storage::More<O6,crate::storage::More<O7,crate::storage::More<O8,crate::storage::More<O9,crate::storage::Last<O10>>>>>>>>>>>,LoadMutLeaves11,crate::read::Env11<O0,O1,O2,O3,O4,O5,O6,O7,O8,O9,O10>; [O0,O1,O2,O3,O4,O5,O6,O7,O8,O9,O10]; [O0,O1,O2,O3,O4,O5,O6,O7,O8,O9,O10,u32]);
impl_scatter_combine_dispatch!(crate::S12,scatter_combine_s12,crate::storage::More<O0,crate::storage::More<O1,crate::storage::More<O2,crate::storage::More<O3,crate::storage::More<O4,crate::storage::More<O5,crate::storage::More<O6,crate::storage::More<O7,crate::storage::More<O8,crate::storage::More<O9,crate::storage::More<O10,crate::storage::Last<O11>>>>>>>>>>>>,LoadMutLeaves12,crate::read::Env12<O0,O1,O2,O3,O4,O5,O6,O7,O8,O9,O10,O11>; [O0,O1,O2,O3,O4,O5,O6,O7,O8,O9,O10,O11]; [O0,O1,O2,O3,O4,O5,O6,O7,O8,O9,O10,O11]);

pub fn apply<R, Source, Output, Op>(
    exec: &Executor<R>,
    source: Source,
    indices: &DeviceVec<R, u32>,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Source: ReadExpression<ReadArity = A13> + LowerReadExpression + StageRead<R, Env0>,
    Output: OutputExpression<Item = Source::Item> + LowerOutputExpression + StageOutput<R, Env0>,
    Op: ReductionOp<Source::Item>,
    Dispatch<A13, Output::StorageArity>: ScatterCombineDispatch<R, Source, Output>,
{
    <Dispatch<A13, Output::StorageArity> as ScatterCombineDispatch<R, Source, Output>>::run::<Op>(
        exec, &source, indices, &output,
    )
}
