//! Conflict-free application of already reduced scatter proposals.

use cubecl::prelude::*;

use crate::{
    DeviceVec, Dispatch, Error, Executor, MStorageElement, ReadExpression, S1, S2, S3, S4, S5, S6,
    S7, StorageLayout,
    op::ReductionOp,
    output::{LowerOutputExpression, OutputBindings, OutputExpression, StageOutput},
    read::{Env0, Env1, Env2, Env3, Env4, Env5, Env6, Env7, LowerReadExpression},
    reduce::{StageRead, StagedBindings},
    storage::{
        Decompose, Last, LoadLeaves2, LoadLeaves3, LoadLeaves4, LoadLeaves5, LoadLeaves6,
        LoadLeaves7, LoadMutLeaves2, LoadMutLeaves3, LoadMutLeaves4, LoadMutLeaves5,
        LoadMutLeaves6, LoadMutLeaves7, More, Recompose, StoreLeaves2, StoreLeaves3, StoreLeaves4,
        StoreLeaves5, StoreLeaves6, StoreLeaves7,
    },
};

const BLOCK_SIZE: u32 = 256;

#[cubecl::cube(launch_unchecked, explicit_define)]
fn scatter_combine_s1<
    Item: CubeType + Send + Sync + 'static,
    T: CubePrimitive,
    Layout: Decompose<Item, Leaves = Last<T>> + Recompose<Item, Leaves = Last<T>>,
    Op: ReductionOp<Item>,
>(
    source: &[T],
    source_offsets: &[u32],
    indices: &[u32],
    len: &[u32],
    output: &mut [T],
    output_offsets: &[u32],
) {
    let position = ABSOLUTE_POS as usize;
    if position < len[0] as usize {
        let destination = indices[position] as usize;
        let previous = Layout::recompose(Last::<T> {
            value: output[output_offsets[0] as usize + destination],
        });
        let proposal = Layout::recompose(Last::<T> {
            value: source[source_offsets[0] as usize + position],
        });
        output[output_offsets[0] as usize + destination] =
            Layout::decompose(Op::apply(previous, proposal)).value;
    }
}

macro_rules! define_scatter_combine_kernel {
    (
        $name:ident, $leaves:ty, $load:ident, $load_mut:ident, $store:ident;
        [$( $ty:ident:$source:ident:$output:ident ),+]
    ) => {
        #[cubecl::cube(launch_unchecked, explicit_define)]
        fn $name<
            Item: CubeType + Send + Sync + 'static,
            $( $ty: CubePrimitive, )+
            Layout: Decompose<Item, Leaves = $leaves> + Recompose<Item, Leaves = $leaves>,
            Op: ReductionOp<Item>,
        >(
            $( $source: &[$ty], )+
            source_offsets: &[u32],
            indices: &[u32],
            len: &[u32],
            $( $output: &mut [$ty], )+
            output_offsets: &[u32],
        ) {
            let position = ABSOLUTE_POS as usize;
            if position < len[0] as usize {
                let destination = indices[position] as usize;
                let proposal_leaves = <$leaves as $load<$( $ty ),+>>::load(
                    $( $source, )+ source_offsets, position,
                );
                let previous_leaves = <$leaves as $load_mut<$( $ty ),+>>::load_mut(
                    $( $output, )+ output_offsets, destination,
                );
                let combined = Layout::decompose(Op::apply(
                    Layout::recompose(previous_leaves),
                    Layout::recompose(proposal_leaves),
                ));
                <$leaves as $store<$( $ty ),+>>::store(
                    combined, $( $output, )+ output_offsets, destination,
                );
            }
        }
    };
}

define_scatter_combine_kernel!(scatter_combine_s2, More<T0, Last<T1>>, LoadLeaves2, LoadMutLeaves2, StoreLeaves2; [T0:source0:output0,T1:source1:output1]);
define_scatter_combine_kernel!(scatter_combine_s3, More<T0, More<T1, Last<T2>>>, LoadLeaves3, LoadMutLeaves3, StoreLeaves3; [T0:source0:output0,T1:source1:output1,T2:source2:output2]);
define_scatter_combine_kernel!(scatter_combine_s4, More<T0, More<T1, More<T2, Last<T3>>>>, LoadLeaves4, LoadMutLeaves4, StoreLeaves4; [T0:source0:output0,T1:source1:output1,T2:source2:output2,T3:source3:output3]);
define_scatter_combine_kernel!(scatter_combine_s5, More<T0, More<T1, More<T2, More<T3, Last<T4>>>>>, LoadLeaves5, LoadMutLeaves5, StoreLeaves5; [T0:source0:output0,T1:source1:output1,T2:source2:output2,T3:source3:output3,T4:source4:output4]);
define_scatter_combine_kernel!(scatter_combine_s6, More<T0, More<T1, More<T2, More<T3, More<T4, Last<T5>>>>>>, LoadLeaves6, LoadMutLeaves6, StoreLeaves6; [T0:source0:output0,T1:source1:output1,T2:source2:output2,T3:source3:output3,T4:source4:output4,T5:source5:output5]);
define_scatter_combine_kernel!(scatter_combine_s7, More<T0, More<T1, More<T2, More<T3, More<T4, More<T5, Last<T6>>>>>>>, LoadLeaves7, LoadMutLeaves7, StoreLeaves7; [T0:source0:output0,T1:source1:output1,T2:source2:output2,T3:source3:output3,T4:source4:output4,T5:source5:output5,T6:source6:output6]);

pub trait ScatterCombineDispatch<R, Source, Output, Slots>
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
    ($storage:ty,$kernel:ident,$leaves:ty; [$( $ty:ident:$index:literal ),+],$env:ty) => {
        impl<R, Item, Source, Output, $( $ty ),+>
            ScatterCombineDispatch<R, Source, Output, $env> for Dispatch<$storage, $storage>
        where
            R: Runtime,
            Item: StorageLayout<StorageArity = $storage, StorageLeaves = $leaves>,
            Item::DeviceLayout:
                Decompose<Item, Leaves = $leaves> + Recompose<Item, Leaves = $leaves>,
            $( $ty: MStorageElement, )+
            Source: ReadExpression<Item = Item>
                + LowerReadExpression<Slots = $env>
                + StageRead<R, Env0>,
            Output: OutputExpression<Item = Item, StorageArity = $storage>
                + LowerOutputExpression<Slots = $env>
                + StageOutput<R, Env0>,
        {
            fn run<Op>(
                exec: &Executor<R>,
                source: &Source,
                indices: &DeviceVec<R, u32>,
                output: &Output,
            ) -> Result<(), Error>
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
                let mut writes = OutputBindings::new();
                output.stage_output(exec.id(), &mut writes)?;
                let source_offsets = exec.client().create_from_slice(u32::as_bytes(&reads.offsets));
                let output_offsets = exec.client().create_from_slice(u32::as_bytes(&writes.offsets));
                let len_handle = exec.client().create_from_slice(u32::as_bytes(&[len_u32]));
                unsafe {
                    $kernel::launch_unchecked::<Item, $( $ty, )+ Item::DeviceLayout, Op, R>(
                        exec.client(),
                        crate::launch::cube_count_1d(len.div_ceil(BLOCK_SIZE as usize))?,
                        CubeDim::new_1d(BLOCK_SIZE),
                        $( BufferArg::from_raw_parts(reads.slots[$index].0.clone(), reads.slots[$index].1), )+
                        BufferArg::from_raw_parts(source_offsets, reads.offsets.len()),
                        BufferArg::from_raw_parts(indices.handle.clone(), indices.len()),
                        BufferArg::from_raw_parts(len_handle, 1),
                        $( BufferArg::from_raw_parts(writes.slots[$index].0.clone(), writes.slots[$index].1), )+
                        BufferArg::from_raw_parts(output_offsets, writes.offsets.len()),
                    );
                }
                Ok(())
            }
        }
    };
}

impl_scatter_combine_dispatch!(S1,scatter_combine_s1,Last<T0>; [T0:0],Env1<T0>);
impl_scatter_combine_dispatch!(S2,scatter_combine_s2,More<T0,Last<T1>>; [T0:0,T1:1],Env2<T0,T1>);
impl_scatter_combine_dispatch!(S3,scatter_combine_s3,More<T0,More<T1,Last<T2>>>; [T0:0,T1:1,T2:2],Env3<T0,T1,T2>);
impl_scatter_combine_dispatch!(S4,scatter_combine_s4,More<T0,More<T1,More<T2,Last<T3>>>>; [T0:0,T1:1,T2:2,T3:3],Env4<T0,T1,T2,T3>);
impl_scatter_combine_dispatch!(S5,scatter_combine_s5,More<T0,More<T1,More<T2,More<T3,Last<T4>>>>>; [T0:0,T1:1,T2:2,T3:3,T4:4],Env5<T0,T1,T2,T3,T4>);
impl_scatter_combine_dispatch!(S6,scatter_combine_s6,More<T0,More<T1,More<T2,More<T3,More<T4,Last<T5>>>>>>; [T0:0,T1:1,T2:2,T3:3,T4:4,T5:5],Env6<T0,T1,T2,T3,T4,T5>);
impl_scatter_combine_dispatch!(S7,scatter_combine_s7,More<T0,More<T1,More<T2,More<T3,More<T4,More<T5,Last<T6>>>>>>>; [T0:0,T1:1,T2:2,T3:3,T4:4,T5:5,T6:6],Env7<T0,T1,T2,T3,T4,T5,T6>);

pub fn apply<R, Source, Output, Op>(
    exec: &Executor<R>,
    source: Source,
    indices: &DeviceVec<R, u32>,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Source: ReadExpression + LowerReadExpression + StageRead<R, Env0>,
    Output: OutputExpression<Item = Source::Item>
        + LowerOutputExpression<Slots = Source::Slots>
        + StageOutput<R, Env0>,
    Op: ReductionOp<Source::Item>,
    Dispatch<<Source::Item as StorageLayout>::StorageArity, Output::StorageArity>:
        ScatterCombineDispatch<R, Source, Output, Source::Slots>,
{
    <Dispatch<
        <Source::Item as StorageLayout>::StorageArity,
        Output::StorageArity,
    > as ScatterCombineDispatch<
        R,
        Source,
        Output,
        Source::Slots,
    >>::run::<Op>(exec, &source, indices, &output)
}
