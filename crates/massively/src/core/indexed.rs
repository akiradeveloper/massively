//! Indexed algorithms and permutation application.

use cubecl::prelude::*;

use crate::{
    Dispatch, Error, Executor, ReadExpression,
    masked::MaskedCopyDispatch,
    output::{
        KernelOutputSlots, LowerOutputExpression, OutputExpression, PaddedOutputSlots, StageOutput,
    },
    read::{Env0, KernelReadSlots, LowerReadExpression},
    reduce::StageRead,
};

/// Internal capability proving the combined value/index arity is supported.
#[doc(hidden)]
pub trait GatherInput<R: Runtime, Indices, Output>: ReadExpression + Sized {
    fn gather(self, exec: &Executor<R>, indices: Indices, output: Output) -> Result<(), Error>;
}

impl<R, Values, Indices, Output> GatherInput<R, Indices, Output> for Values
where
    R: Runtime,
    Values: ReadExpression + LowerReadExpression + StageRead<R, Env0>,
    Indices: crate::selection::FlagInput<R>,
    Output: OutputExpression + LowerOutputExpression + StageOutput<R, Env0>,
    Output::Slots: PaddedOutputSlots,
    Dispatch<crate::A13, crate::S12>: MaskedCopyDispatch<
            R,
            Values,
            Output,
            KernelReadSlots<Values::Slots>,
            KernelOutputSlots<Output::Slots>,
        >,
{
    fn gather(self, exec: &Executor<R>, indices: Indices, output: Output) -> Result<(), Error> {
        let indices = indices.materialize_flags(exec)?;
        <Dispatch<crate::A13, crate::S12> as MaskedCopyDispatch<
            R,
            Values,
            Output,
            KernelReadSlots<Values::Slots>,
            KernelOutputSlots<Output::Slots>,
        >>::run(exec, &self, Some(&indices), true, None, &output)
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

/// Fixed-ABI masked gather used by the public iterator facade.
pub(crate) fn gather_where_direct<R, Values, Indices, Stencil, Output>(
    exec: &Executor<R>,
    values: Values,
    indices: Indices,
    stencil: Stencil,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Values: ReadExpression + LowerReadExpression + StageRead<R, Env0>,
    Indices: crate::selection::FlagInput<R>,
    Stencil: crate::selection::FlagInput<R>,
    Output: OutputExpression + LowerOutputExpression + StageOutput<R, Env0>,
    Output::Slots: PaddedOutputSlots,
    Dispatch<crate::A13, crate::S12>: MaskedCopyDispatch<
            R,
            Values,
            Output,
            KernelReadSlots<Values::Slots>,
            KernelOutputSlots<Output::Slots>,
        >,
{
    let indices = indices.materialize_flags(exec)?;
    let flags = stencil.materialize_flags(exec)?;
    <Dispatch<crate::A13, crate::S12> as MaskedCopyDispatch<
        R,
        Values,
        Output,
        KernelReadSlots<Values::Slots>,
        KernelOutputSlots<Output::Slots>,
    >>::run(exec, &values, Some(&indices), true, Some(&flags), &output)
}

#[cfg(any())]
mod direct_permutation {
    use super::*;

    const PERMUTATION_BLOCK_SIZE: u32 = 256;

    macro_rules! define_padded_permutation_kernel {
    ($name:ident, $eval:ident, $method:ident; [$( $leaf:ident : $slot:ident ),+]) => {
        #[cubecl::cube(launch_unchecked, explicit_define)]
        fn $name<
            Source: CubeType + Send + Sync + 'static,
            Target: WritableFrom<Source> + Send + Sync + 'static,
            $( $leaf: CubePrimitive, )+
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
            Leaves: CubeType + Send + Sync + 'static
                + StorePadded12<
                    O0 = O0, O1 = O1, O2 = O2, O3 = O3, O4 = O4, O5 = O5,
                    O6 = O6, O7 = O7, O8 = O8, O9 = O9, O10 = O10, O11 = O11,
                >,
            Expr: $eval<Source, $( $leaf ),+>,
            Layout: Decompose<Target, Leaves = Leaves>,
        >(
            $( $slot: &[$leaf], )+
            read_offsets: &[u32],
            indices: &[u32],
            index_offset: &[u32],
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
            let index = ABSOLUTE_POS as usize;
            if index < len[0] as usize {
                let source_index = indices[index_offset[0] as usize + index] as usize;
                let source = Expr::$method($( $slot, )+ read_offsets, source_index);
                Layout::decompose(Target::write_from(source)).store_padded(
                    out0, out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11,
                    write_offsets, index,
                );
            }
        }
    };
}

    define_padded_permutation_kernel!(permutation_a1, Eval1, eval1; [L0:slot0]);
    define_padded_permutation_kernel!(permutation_a2, Eval2, eval2; [L0:slot0,L1:slot1]);
    define_padded_permutation_kernel!(permutation_a3, Eval3, eval3; [L0:slot0,L1:slot1,L2:slot2]);
    define_padded_permutation_kernel!(permutation_a4, Eval4, eval4; [L0:slot0,L1:slot1,L2:slot2,L3:slot3]);
    define_padded_permutation_kernel!(permutation_a5, Eval5, eval5; [L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4]);
    define_padded_permutation_kernel!(permutation_a6, Eval6, eval6; [L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5]);
    define_padded_permutation_kernel!(permutation_a7, Eval7, eval7; [L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5,L6:slot6]);
    define_padded_permutation_kernel!(permutation_a8, Eval8, eval8; [L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5,L6:slot6,L7:slot7]);
    define_padded_permutation_kernel!(permutation_a9, Eval9, eval9; [L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5,L6:slot6,L7:slot7,L8:slot8]);
    define_padded_permutation_kernel!(permutation_a10, Eval10, eval10; [L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5,L6:slot6,L7:slot7,L8:slot8,L9:slot9]);
    define_padded_permutation_kernel!(permutation_a11, Eval11, eval11; [L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5,L6:slot6,L7:slot7,L8:slot8,L9:slot9,L10:slot10]);
    define_padded_permutation_kernel!(permutation_a12, Eval12, eval12; [L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5,L6:slot6,L7:slot7,L8:slot8,L9:slot9,L10:slot10,L11:slot11]);
    define_padded_permutation_kernel!(permutation_a13, Eval13, eval13; [L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5,L6:slot6,L7:slot7,L8:slot8,L9:slot9,L10:slot10,L11:slot11,L12:slot12]);

    #[doc(hidden)]
    pub trait PermutationDispatch<R, Input, Output, ReadSlots, WriteSlots>
    where
        R: Runtime,
    {
        fn run(
            exec: &Executor<R>,
            input: &Input,
            indices: &crate::Column<u32>,
            output: &Output,
        ) -> Result<(), Error>;
    }

    macro_rules! impl_padded_permutation_dispatch {
    ($arity:ty, $eval:ident, $kernel:ident; [$( $leaf:ident : $read_index:literal ),+], $read_env:ty) => {
        impl<R, Input, Output, Source, Leaves, WriteSlots, $( $leaf, )+>
            PermutationDispatch<R, Input, Output, $read_env, WriteSlots>
            for Dispatch<$arity, S12>
        where
            R: Runtime,
            $( $leaf: MStorageElement, )+
            Source: StorageLayout + Send + Sync + 'static,
            Leaves: CubeType + Send + Sync + 'static + StorePadded12<
                O0 = WriteSlots::O0, O1 = WriteSlots::O1, O2 = WriteSlots::O2,
                O3 = WriteSlots::O3, O4 = WriteSlots::O4, O5 = WriteSlots::O5,
                O6 = WriteSlots::O6, O7 = WriteSlots::O7, O8 = WriteSlots::O8,
                O9 = WriteSlots::O9, O10 = WriteSlots::O10, O11 = WriteSlots::O11,
            >,
            WriteSlots: PaddedOutputSlots,
            Input: ReadExpression<Item = Source>
                + LowerReadExpression<Slots = $read_env>
                + StageRead<R, Env0>,
            Input::DeviceExpr: $eval<Source, $( $leaf ),+>,
            Output: OutputExpression
                + LowerOutputExpression<Slots = WriteSlots>
                + StageOutput<R, Env0>,
            Output::Item: StorageLayout<StorageLeaves = Leaves> + WritableFrom<Source>,
            <Output::Item as StorageLayout>::DeviceLayout:
                Decompose<Output::Item, Leaves = Leaves>,
        {
            fn run(
                exec: &Executor<R>,
                input: &Input,
                indices: &crate::Column<u32>,
                output: &Output,
            ) -> Result<(), Error> {
                let len = <crate::Column<u32> as StageRead<R, Env0>>::logical_len(indices)?;
                let output_len = output.logical_len()?;
                if output_len < len {
                    return Err(Error::OutputTooShort { input: len, output: output_len });
                }
                if len == 0 {
                    return Ok(());
                }
                let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
                let mut reads = StagedBindings::new();
                input.stage_at(exec.client(), exec.id(), &mut reads)?;
                let mut index_reads = StagedBindings::new();
                <crate::Column<u32> as StageRead<R, Env0>>::stage_at(
                    indices, exec.client(), exec.id(), &mut index_reads,
                )?;
                let mut writes = OutputBindings::new();
                output.stage_output(exec.id(), &mut writes)?;
                writes.pad_to_twelve(exec.client());
                let read_offsets = exec.client().create_from_slice(u32::as_bytes(&reads.offsets));
                let index_offset = exec.client().create_from_slice(u32::as_bytes(&index_reads.offsets));
                let write_offsets = exec.client().create_from_slice(u32::as_bytes(&writes.offsets));
                let len_handle = exec.client().create_from_slice(u32::as_bytes(&[len_u32]));
                unsafe {
                    $kernel::launch_unchecked::<
                        Source,
                        Output::Item,
                        $( $leaf, )+
                        WriteSlots::O0, WriteSlots::O1, WriteSlots::O2, WriteSlots::O3,
                        WriteSlots::O4, WriteSlots::O5, WriteSlots::O6, WriteSlots::O7,
                        WriteSlots::O8, WriteSlots::O9, WriteSlots::O10, WriteSlots::O11,
                        Leaves,
                        Input::DeviceExpr,
                        <Output::Item as StorageLayout>::DeviceLayout,
                        R,
                    >(
                        exec.client(),
                        crate::launch::cube_count_1d(len.div_ceil(PERMUTATION_BLOCK_SIZE as usize))?,
                        CubeDim::new_1d(PERMUTATION_BLOCK_SIZE),
                        $( BufferArg::from_raw_parts(reads.slots[$read_index].0.clone(), reads.slots[$read_index].1), )+
                        BufferArg::from_raw_parts(read_offsets, reads.offsets.len()),
                        BufferArg::from_raw_parts(index_reads.slots[0].0.clone(), index_reads.slots[0].1),
                        BufferArg::from_raw_parts(index_offset, 1),
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
    };
}

    impl_padded_permutation_dispatch!(A1, Eval1, permutation_a1; [L0:0], Env1<L0>);
    impl_padded_permutation_dispatch!(A2, Eval2, permutation_a2; [L0:0,L1:1], Env2<L0,L1>);
    impl_padded_permutation_dispatch!(A3, Eval3, permutation_a3; [L0:0,L1:1,L2:2], Env3<L0,L1,L2>);
    impl_padded_permutation_dispatch!(A4, Eval4, permutation_a4; [L0:0,L1:1,L2:2,L3:3], Env4<L0,L1,L2,L3>);
    impl_padded_permutation_dispatch!(A5, Eval5, permutation_a5; [L0:0,L1:1,L2:2,L3:3,L4:4], Env5<L0,L1,L2,L3,L4>);
    impl_padded_permutation_dispatch!(A6, Eval6, permutation_a6; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5], Env6<L0,L1,L2,L3,L4,L5>);
    impl_padded_permutation_dispatch!(A7, Eval7, permutation_a7; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5,L6:6], Env7<L0,L1,L2,L3,L4,L5,L6>);
    impl_padded_permutation_dispatch!(A8, Eval8, permutation_a8; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5,L6:6,L7:7], Env8<L0,L1,L2,L3,L4,L5,L6,L7>);
    impl_padded_permutation_dispatch!(A9, Eval9, permutation_a9; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5,L6:6,L7:7,L8:8], Env9<L0,L1,L2,L3,L4,L5,L6,L7,L8>);
    impl_padded_permutation_dispatch!(A10, Eval10, permutation_a10; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5,L6:6,L7:7,L8:8,L9:9], Env10<L0,L1,L2,L3,L4,L5,L6,L7,L8,L9>);
    impl_padded_permutation_dispatch!(A11, Eval11, permutation_a11; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5,L6:6,L7:7,L8:8,L9:9,L10:10], Env11<L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10>);
    impl_padded_permutation_dispatch!(A12, Eval12, permutation_a12; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5,L6:6,L7:7,L8:8,L9:9,L10:10,L11:11], Env12<L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11>);
    impl_padded_permutation_dispatch!(A13, Eval13, permutation_a13; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5,L6:6,L7:7,L8:8,L9:9,L10:10,L11:11,L12:12], Env13<L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11,L12>);
}

pub(crate) fn apply_permutation<R, Input, Output>(
    exec: &Executor<R>,
    input: Input,
    indices: crate::Column<u32>,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: GatherInput<R, crate::Column<u32>, Output>,
{
    gather_direct(exec, input, indices, output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CanonicalStorage, Counting, Permute, ReverseCounting, Zip, read::FixedRead};
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn gather_seven_columns_uses_eval8_and_reassociates_output() {
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

        gather_direct(&exec, FixedRead::new(values), indices.column(), output).unwrap();
        for (column, output) in outputs.iter().enumerate() {
            assert_eq!(
                exec.to_host(output).unwrap(),
                vec![column as u32 * 10 + 3, column as u32 * 10 + 1]
            );
        }
    }

    #[test]
    fn reverse_seven_columns_uses_reverse_counting_eval8() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let inputs: Vec<_> = (0_u32..7)
            .map(|base| exec.to_device(&[base * 10 + 1, base * 10 + 2, base * 10 + 3]))
            .collect();
        let outputs: Vec<_> = (0..7).map(|_| exec.to_device(&[0_u32; 3])).collect();
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

        gather_direct(
            &exec,
            FixedRead::new(values),
            ReverseCounting::new(3),
            output,
        )
        .unwrap();
        for (column, output) in outputs.iter().enumerate() {
            assert_eq!(
                exec.to_host(output).unwrap(),
                vec![
                    column as u32 * 10 + 3,
                    column as u32 * 10 + 2,
                    column as u32 * 10 + 1
                ]
            );
        }
    }

    #[test]
    fn gather_accepts_fixed_eval8_values_and_lazy_indices_independently() {
        type Seven = (u32, (u32, (u32, (u32, (u32, (u32, u32))))));
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
        let values = Permute::new(seven, Counting::new(0, 3));
        let raw_indices = exec.to_device(&[2_u32, 0]);
        let indices = Permute::new(raw_indices.column(), Counting::new(0, 2));
        let output = exec.alloc_canonical::<Seven>(2);

        gather_direct(&exec, FixedRead::new(values), indices, output.write()).unwrap();
        assert_eq!(exec.to_host(&output.0.0.0.0.0.0).unwrap(), vec![3, 1]);
        assert_eq!(exec.to_host(&output.1).unwrap(), vec![63, 61]);
    }

    #[test]
    fn gather_where_preserves_rows_with_zero_stencil() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let values = exec.to_device(&[10_u32, 20, 30, 40]);
        let indices = exec.to_device(&[3_u32, 2, 1, 0]);
        let stencil = exec.to_device(&[1_u32, 0, 1, 0]);
        let output = exec.to_device(&[100_u32, 200, 300, 400]);

        gather_where_direct(
            &exec,
            FixedRead::new(values.column()),
            indices.column(),
            stencil.column(),
            output.slice_mut(..),
        )
        .unwrap();
        assert_eq!(exec.to_host(&output).unwrap(), vec![40, 200, 20, 400]);
    }
}
