//! Scatter primitives separated into semantic materialization and physical apply.

use cubecl::prelude::*;

use crate::{
    A1, A2, A3, A4, A5, A6, A7, DeviceVec, Dispatch, Error, Executor, MAlloc, MStorage,
    MStorageElement, ReadExpression, S1, S2, S3, S4, S5, S6, S7,
    output::{LowerOutputExpression, OutputBindings, OutputExpression, StageOutput},
    read::{Env0, Env1, Env2, Env3, Env4, Env5, Env6, Env7, LowerReadExpression},
    reduce::{StageRead, StagedBindings},
    selection::FlagInput,
    transform::{MaterializeDispatch, materialize},
};

const BLOCK_SIZE: u32 = 256;

macro_rules! define_scatter_kernel {
    ($name:ident; $( $ty:ident : $source:ident : $output:ident : $index:literal ),+ $(,)?) => {
        #[cubecl::cube(launch_unchecked)]
        fn $name<$( $ty: CubePrimitive ),+>(
            $( $source: &[$ty], )+
            source_offsets: &[u32],
            indices: &[u32],
            flags: &[u32],
            use_flags: &[u32],
            len: &[u32],
            $( $output: &mut [$ty], )+
            output_offsets: &[u32],
        ) {
            let position = ABSOLUTE_POS as usize;
            if position < len[0] as usize
                && (use_flags[0] == 0u32 || flags[position] != 0u32)
            {
                let destination = indices[position] as usize;
                $(
                    $output[output_offsets[$index] as usize + destination] =
                        $source[source_offsets[$index] as usize + position];
                )+
            }
        }
    };
}

define_scatter_kernel!(scatter_s1; T0:source0:output0:0);
define_scatter_kernel!(scatter_s2; T0:source0:output0:0,T1:source1:output1:1);
define_scatter_kernel!(scatter_s3; T0:source0:output0:0,T1:source1:output1:1,T2:source2:output2:2);
define_scatter_kernel!(scatter_s4; T0:source0:output0:0,T1:source1:output1:1,T2:source2:output2:2,T3:source3:output3:3);
define_scatter_kernel!(scatter_s5; T0:source0:output0:0,T1:source1:output1:1,T2:source2:output2:2,T3:source3:output3:3,T4:source4:output4:4);
define_scatter_kernel!(scatter_s6; T0:source0:output0:0,T1:source1:output1:1,T2:source2:output2:2,T3:source3:output3:3,T4:source4:output4:4,T5:source5:output5:5);
define_scatter_kernel!(scatter_s7; T0:source0:output0:0,T1:source1:output1:1,T2:source2:output2:2,T3:source3:output3:3,T4:source4:output4:4,T5:source5:output5:5,T6:source6:output6:6);

#[doc(hidden)]
pub trait ScatterDispatch<R, Source, Output, ReadSlots, WriteSlots>
where
    R: Runtime,
{
    fn run(
        exec: &Executor<R>,
        source: &Source,
        indices: &DeviceVec<R, u32>,
        flags: Option<&DeviceVec<R, u32>>,
        output: &Output,
    ) -> Result<(), Error>;
}

macro_rules! impl_scatter_dispatch {
    ($arity:ty,$storage:ty,$kernel:ident; [$( $ty:ident:$index:literal ),+],$env:ty) => {
        impl<R, Source, Output, $( $ty ),+>
            ScatterDispatch<R, Source, Output, $env, $env> for Dispatch<$arity, $storage>
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
                indices: &DeviceVec<R, u32>,
                flags: Option<&DeviceVec<R, u32>>,
                output: &Output,
            ) -> Result<(), Error> {
                let len = source.logical_len()?;
                if indices.len() != len {
                    return Err(Error::LengthMismatch { left: len, right: indices.len() });
                }
                if let Some(flags) = flags {
                    if flags.len() != len {
                        return Err(Error::LengthMismatch { left: len, right: flags.len() });
                    }
                }
                if len == 0 { return Ok(()); }
                let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
                let mut reads = StagedBindings::new();
                source.stage_at(exec.client(), exec.id(), &mut reads)?;
                let mut writes = OutputBindings::new();
                output.stage_output(exec.id(), &mut writes)?;
                let source_offsets = exec.client().create_from_slice(u32::as_bytes(&reads.offsets));
                let output_offsets = exec.client().create_from_slice(u32::as_bytes(&writes.offsets));
                let len_handle = exec.client().create_from_slice(u32::as_bytes(&[len_u32]));
                let use_flags = exec.client().create_from_slice(u32::as_bytes(&[u32::from(flags.is_some())]));
                let dummy_flags = exec.client().create_from_slice(u32::as_bytes(&[0u32]));
                let (flags_handle, flags_len) = flags
                    .map(|flags| (flags.handle.clone(), flags.len()))
                    .unwrap_or((dummy_flags, 1));
                unsafe {
                    $kernel::launch_unchecked::<$( $ty, )+ R>(
                        exec.client(),
                        crate::launch::cube_count_1d(len.div_ceil(BLOCK_SIZE as usize))?,
                        CubeDim::new_1d(BLOCK_SIZE),
                        $( BufferArg::from_raw_parts(reads.slots[$index].0.clone(), reads.slots[$index].1), )+
                        BufferArg::from_raw_parts(source_offsets, reads.offsets.len()),
                        BufferArg::from_raw_parts(indices.handle.clone(), indices.len()),
                        BufferArg::from_raw_parts(flags_handle, flags_len),
                        BufferArg::from_raw_parts(use_flags, 1),
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

impl_scatter_dispatch!(A1,S1,scatter_s1; [T0:0],Env1<T0>);
impl_scatter_dispatch!(A2,S2,scatter_s2; [T0:0,T1:1],Env2<T0,T1>);
impl_scatter_dispatch!(A3,S3,scatter_s3; [T0:0,T1:1,T2:2],Env3<T0,T1,T2>);
impl_scatter_dispatch!(A4,S4,scatter_s4; [T0:0,T1:1,T2:2,T3:3],Env4<T0,T1,T2,T3>);
impl_scatter_dispatch!(A5,S5,scatter_s5; [T0:0,T1:1,T2:2,T3:3,T4:4],Env5<T0,T1,T2,T3,T4>);
impl_scatter_dispatch!(A6,S6,scatter_s6; [T0:0,T1:1,T2:2,T3:3,T4:4,T5:5],Env6<T0,T1,T2,T3,T4,T5>);
impl_scatter_dispatch!(A7,S7,scatter_s7; [T0:0,T1:1,T2:2,T3:3,T4:4,T5:5,T6:6],Env7<T0,T1,T2,T3,T4,T5,T6>);

#[doc(hidden)]
pub trait ScatterStorage<R: Runtime, Output>: ReadExpression + Sized {
    fn scatter_storage(
        self,
        exec: &Executor<R>,
        indices: &DeviceVec<R, u32>,
        flags: Option<&DeviceVec<R, u32>>,
        output: Output,
    ) -> Result<(), Error>;
}

impl<R, Source, Output> ScatterStorage<R, Output> for Source
where
    R: Runtime,
    Source: ReadExpression + LowerReadExpression + StageRead<R, Env0>,
    Output: OutputExpression + LowerOutputExpression + StageOutput<R, Env0>,
    Dispatch<Source::ReadArity, Output::StorageArity>:
        ScatterDispatch<R, Source, Output, Source::Slots, Output::Slots>,
{
    fn scatter_storage(
        self,
        exec: &Executor<R>,
        indices: &DeviceVec<R, u32>,
        flags: Option<&DeviceVec<R, u32>>,
        output: Output,
    ) -> Result<(), Error> {
        <Dispatch<Source::ReadArity, Output::StorageArity> as ScatterDispatch<
            R,
            Source,
            Output,
            Source::Slots,
            Output::Slots,
        >>::run(exec, &self, indices, flags, &output)
    }
}

#[doc(hidden)]
pub trait ScatterInput<R: Runtime, Indices, Output>: ReadExpression + Sized {
    fn scatter_input(
        self,
        exec: &Executor<R>,
        indices: Indices,
        flags: Option<&DeviceVec<R, u32>>,
        output: Output,
    ) -> Result<(), Error>;
}

impl<R, Values, Indices, Output> ScatterInput<R, Indices, Output> for Values
where
    R: Runtime,
    Values: ReadExpression + LowerReadExpression + StageRead<R, Env0>,
    Values::Item: MAlloc<R>,
    <Values::Item as MAlloc<R>>::Storage: MStorage<R>,
    <<Values::Item as MAlloc<R>>::Storage as MStorage<R>>::Write:
        LowerOutputExpression + StageOutput<R, Env0>,
    <<<Values::Item as MAlloc<R>>::Storage as MStorage<R>>::Write as OutputExpression>::Item:
        crate::WriteFrom<Values::Item>,
    Dispatch<Values::ReadArity, <<<Values::Item as MAlloc<R>>::Storage as MStorage<R>>::Write as OutputExpression>::StorageArity>:
        MaterializeDispatch<R, Values, <<Values::Item as MAlloc<R>>::Storage as MStorage<R>>::Write, Values::Slots, <<<Values::Item as MAlloc<R>>::Storage as MStorage<R>>::Write as LowerOutputExpression>::Slots>,
    <<Values::Item as MAlloc<R>>::Storage as MStorage<R>>::Read: ScatterStorage<R, Output>,
    Indices: FlagInput<R>,
{
    fn scatter_input(
        self,
        exec: &Executor<R>,
        indices: Indices,
        flags: Option<&DeviceVec<R, u32>>,
        output: Output,
    ) -> Result<(), Error> {
        let len = self.logical_len()?;
        let indices_len = indices.flag_len()?;
        if indices_len != len {
            return Err(Error::LengthMismatch { left: len, right: indices_len });
        }
        let temporary = exec.alloc::<Values::Item>(len);
        materialize(exec, self, temporary.write())?;
        let indices = indices.materialize_flags(exec)?;
        temporary.read().scatter_storage(exec, &indices, flags, output)
    }
}

/// Writes each input item to the output position given by its index.
pub(crate) fn scatter<R, Values, Indices, Output>(
    exec: &Executor<R>,
    values: Values,
    indices: Indices,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Values: ScatterInput<R, Indices, Output>,
{
    values.scatter_input(exec, indices, None, output)
}

/// Scatters rows whose stencil is nonzero, preserving other output rows.
pub(crate) fn scatter_where<R, Values, Indices, Stencil, Output>(
    exec: &Executor<R>,
    values: Values,
    indices: Indices,
    stencil: Stencil,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Values: ScatterInput<R, Indices, Output>,
    Stencil: FlagInput<R>,
{
    let flags = stencil.materialize_flags(exec)?;
    values.scatter_input(exec, indices, Some(&flags), output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Counting, Permute, Zip};
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn scatter_normalizes_eval8_source_then_applies_storage7() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let inputs: Vec<_> = (0_u32..7)
            .map(|base| exec.to_device(&[base * 10 + 1, base * 10 + 2, base * 10 + 3]))
            .collect();
        let outputs: Vec<_> = (0..7).map(|_| exec.to_device(&[0_u32; 4])).collect();
        let indices = exec.to_device(&[2_u32, 0, 3]);
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
        let values = Permute::new(seven, Counting::new(0, 3));
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

        scatter(&exec, values, indices.column(), output).unwrap();
        for (column, output) in outputs.iter().enumerate() {
            let base = column as u32 * 10;
            assert_eq!(
                exec.to_host(output).unwrap(),
                vec![base + 2, 0, base + 1, base + 3]
            );
        }
    }

    #[test]
    fn scatter_where_leaves_unselected_destinations_unchanged() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let values = exec.to_device(&[10_u32, 20, 30]);
        let indices = exec.to_device(&[2_u32, 0, 1]);
        let flags = exec.to_device(&[1_u32, 0, 1]);
        let output = exec.to_device(&[99_u32; 3]);
        scatter_where(
            &exec,
            values.column(),
            indices.column(),
            flags.column(),
            output.slice_mut(..),
        )
        .unwrap();
        assert_eq!(exec.to_host(&output).unwrap(), vec![99, 30, 10]);
    }
}
