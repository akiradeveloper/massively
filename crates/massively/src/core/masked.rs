//! Storage-arity indexed conditional copy primitive.

use cubecl::prelude::*;

use crate::{
    A1, A2, A3, A4, A5, A6, A7, Dispatch, Error, Executor, MStorageElement, ReadExpression, S1, S2,
    S3, S4, S5, S6, S7,
    output::{LowerOutputExpression, OutputBindings, OutputExpression, StageOutput},
    read::{Env0, Env1, Env2, Env3, Env4, Env5, Env6, Env7, LowerReadExpression},
    reduce::{StageRead, StagedBindings},
};

const BLOCK_SIZE: u32 = 256;

macro_rules! define_masked_copy_kernel {
    ($name:ident; $( $ty:ident : $source:ident : $output:ident : $index:literal ),+ $(,)?) => {
        #[cubecl::cube(launch_unchecked)]
        fn $name<$( $ty: CubePrimitive ),+>(
            $( $source: &[$ty], )+
            source_offsets: &[u32],
            flags: &[u32],
            len: &[u32],
            $( $output: &mut [$ty], )+
            output_offsets: &[u32],
        ) {
            let position = ABSOLUTE_POS as usize;
            if position < len[0] as usize && flags[position] != 0 {
                $(
                    $output[output_offsets[$index] as usize + position] =
                        $source[source_offsets[$index] as usize + position];
                )+
            }
        }
    };
}

define_masked_copy_kernel!(masked_copy_s1; T0:source0:output0:0);
define_masked_copy_kernel!(masked_copy_s2; T0:source0:output0:0,T1:source1:output1:1);
define_masked_copy_kernel!(masked_copy_s3; T0:source0:output0:0,T1:source1:output1:1,T2:source2:output2:2);
define_masked_copy_kernel!(masked_copy_s4; T0:source0:output0:0,T1:source1:output1:1,T2:source2:output2:2,T3:source3:output3:3);
define_masked_copy_kernel!(masked_copy_s5; T0:source0:output0:0,T1:source1:output1:1,T2:source2:output2:2,T3:source3:output3:3,T4:source4:output4:4);
define_masked_copy_kernel!(masked_copy_s6; T0:source0:output0:0,T1:source1:output1:1,T2:source2:output2:2,T3:source3:output3:3,T4:source4:output4:4,T5:source5:output5:5);
define_masked_copy_kernel!(masked_copy_s7; T0:source0:output0:0,T1:source1:output1:1,T2:source2:output2:2,T3:source3:output3:3,T4:source4:output4:4,T5:source5:output5:5,T6:source6:output6:6);

#[doc(hidden)]
pub trait MaskedCopyDispatch<R, Source, Output, ReadSlots, WriteSlots>
where
    R: Runtime,
{
    fn run(
        exec: &Executor<R>,
        source: &Source,
        flags: &crate::DeviceVec<R, u32>,
        output: &Output,
    ) -> Result<(), Error>;
}

macro_rules! impl_masked_copy_dispatch {
    (
        $arity:ty, $storage:ty, $kernel:ident;
        [$( $ty:ident : $index:literal ),+], $env:ty
    ) => {
        impl<R, Source, Output, $( $ty ),+>
            MaskedCopyDispatch<R, Source, Output, $env, $env>
            for Dispatch<$arity, $storage>
        where
            R: Runtime,
            $( $ty: MStorageElement, )+
            Source: ReadExpression + LowerReadExpression<Slots = $env> + StageRead<R, Env0>,
            Output: OutputExpression<StorageArity = $storage>
                + LowerOutputExpression<Slots = $env>
                + StageOutput<R, Env0>,
        {
            fn run(
                exec: &Executor<R>,
                source: &Source,
                flags: &crate::DeviceVec<R, u32>,
                output: &Output,
            ) -> Result<(), Error> {
                let source_len = source.logical_len()?;
                let output_len = output.logical_len()?;
                if source_len != flags.len() {
                    return Err(Error::LengthMismatch { left: source_len, right: flags.len() });
                }
                if output_len != source_len {
                    return Err(Error::LengthMismatch { left: source_len, right: output_len });
                }
                if source_len == 0 { return Ok(()); }
                let len = u32::try_from(source_len)
                    .map_err(|_| Error::LengthTooLarge { len: source_len })?;
                let mut reads = StagedBindings::new();
                source.stage_at(exec.client(), exec.id(), &mut reads)?;
                let mut writes = OutputBindings::new();
                output.stage_output(exec.id(), &mut writes)?;
                let source_offsets = exec.client().create_from_slice(u32::as_bytes(&reads.offsets));
                let output_offsets = exec.client().create_from_slice(u32::as_bytes(&writes.offsets));
                let len_handle = exec.client().create_from_slice(u32::as_bytes(&[len]));
                unsafe {
                    $kernel::launch_unchecked::<$( $ty, )+ R>(
                        exec.client(),
                        crate::launch::cube_count_1d(source_len.div_ceil(BLOCK_SIZE as usize))?,
                        CubeDim::new_1d(BLOCK_SIZE),
                        $( BufferArg::from_raw_parts(reads.slots[$index].0.clone(), reads.slots[$index].1), )+
                        BufferArg::from_raw_parts(source_offsets, reads.offsets.len()),
                        BufferArg::from_raw_parts(flags.handle.clone(), flags.len()),
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

impl_masked_copy_dispatch!(A1,S1,masked_copy_s1; [T0:0],Env1<T0>);
impl_masked_copy_dispatch!(A2,S2,masked_copy_s2; [T0:0,T1:1],Env2<T0,T1>);
impl_masked_copy_dispatch!(A3,S3,masked_copy_s3; [T0:0,T1:1,T2:2],Env3<T0,T1,T2>);
impl_masked_copy_dispatch!(A4,S4,masked_copy_s4; [T0:0,T1:1,T2:2,T3:3],Env4<T0,T1,T2,T3>);
impl_masked_copy_dispatch!(A5,S5,masked_copy_s5; [T0:0,T1:1,T2:2,T3:3,T4:4],Env5<T0,T1,T2,T3,T4>);
impl_masked_copy_dispatch!(A6,S6,masked_copy_s6; [T0:0,T1:1,T2:2,T3:3,T4:4,T5:5],Env6<T0,T1,T2,T3,T4,T5>);
impl_masked_copy_dispatch!(A7,S7,masked_copy_s7; [T0:0,T1:1,T2:2,T3:3,T4:4,T5:5,T6:6],Env7<T0,T1,T2,T3,T4,T5,T6>);

#[doc(hidden)]
pub trait MaskedCopyInput<R: Runtime, Output>: ReadExpression + Sized {
    fn masked_copy(
        self,
        exec: &Executor<R>,
        flags: &crate::DeviceVec<R, u32>,
        output: Output,
    ) -> Result<(), Error>;
}

impl<R, Source, Output> MaskedCopyInput<R, Output> for Source
where
    R: Runtime,
    Source: ReadExpression + LowerReadExpression + StageRead<R, Env0>,
    Output: OutputExpression + LowerOutputExpression + StageOutput<R, Env0>,
    Dispatch<Source::ReadArity, Output::StorageArity>:
        MaskedCopyDispatch<R, Source, Output, Source::Slots, Output::Slots>,
{
    fn masked_copy(
        self,
        exec: &Executor<R>,
        flags: &crate::DeviceVec<R, u32>,
        output: Output,
    ) -> Result<(), Error> {
        <Dispatch<Source::ReadArity, Output::StorageArity> as MaskedCopyDispatch<
            R,
            Source,
            Output,
            Source::Slots,
            Output::Slots,
        >>::run(exec, &self, flags, &output)
    }
}
