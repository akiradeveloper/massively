//! Indexed algorithms and permutation application.

use cubecl::prelude::*;

use crate::{
    A1, A2, A3, A4, A5, A6, A7, A8, CanonicalAlloc, CanonicalStorage, Dispatch, Error, Executor,
    MStorageElement, Permute, ReadExpression, ReverseCounting, S1, S2, S3, S4, S5, S6, S7,
    StorageLayout, WriteFrom,
    allocation::NormalizeInput,
    eval::{Eval1, Eval2, Eval3, Eval4, Eval5, Eval6, Eval7, Eval8},
    masked::MaskedCopyInput,
    output::{LowerOutputExpression, OutputBindings, OutputExpression, StageOutput},
    read::{Env0, Env1, Env2, Env3, Env4, Env5, Env6, Env7, Env8, LowerReadExpression},
    reduce::{StageRead, StagedBindings},
    storage::{
        Decompose, Last, More, StoreLeaves1, StoreLeaves1Expand, StoreLeaves2, StoreLeaves2Expand,
        StoreLeaves3, StoreLeaves3Expand, StoreLeaves4, StoreLeaves4Expand, StoreLeaves5,
        StoreLeaves5Expand, StoreLeaves6, StoreLeaves6Expand, StoreLeaves7, StoreLeaves7Expand,
    },
    transform::{MaterializeDispatch, materialize},
};

/// Internal capability proving the combined value/index arity is supported.
#[doc(hidden)]
pub trait GatherInput<R: Runtime, Indices, Output>: ReadExpression + Sized {
    fn gather(self, exec: &Executor<R>, indices: Indices, output: Output) -> Result<(), Error>;
}

impl<R, Values, Indices, Output> GatherInput<R, Indices, Output> for Values
where
    R: Runtime,
    Values: ReadExpression,
    Values::Item: StorageLayout,
    Indices: ReadExpression<Item = u32>,
    Permute<Values, Indices>:
        ReadExpression<Item = Values::Item> + LowerReadExpression + StageRead<R, Env0>,
    Output: OutputExpression + LowerOutputExpression + StageOutput<R, Env0>,
    Output::Item: crate::WriteFrom<Values::Item>,
    Dispatch<<Permute<Values, Indices> as ReadExpression>::ReadArity, Output::StorageArity>:
        MaterializeDispatch<
                R,
                Permute<Values, Indices>,
                Output,
                <Permute<Values, Indices> as LowerReadExpression>::Slots,
                Output::Slots,
            >,
{
    fn gather(self, exec: &Executor<R>, indices: Indices, output: Output) -> Result<(), Error> {
        materialize(exec, Permute::new(self, indices), output)
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

const PERMUTATION_BLOCK_SIZE: u32 = 256;

macro_rules! define_permutation_kernel {
    (
        $name:ident, $eval:ident, $method:ident;
        [$( $leaf:ident : $slot:ident ),+];
        [$( $output:ident : $out:ident ),+];
        $leaves:ty
    ) => {
        #[cubecl::cube(launch_unchecked, explicit_define)]
        fn $name<
            Source: CubeType + Send + Sync + 'static,
            Target: WriteFrom<Source> + Send + Sync + 'static,
            $( $leaf: CubePrimitive, )+
            $( $output: CubePrimitive, )+
            Expr: $eval<Source, $( $leaf ),+>,
            Layout: Decompose<Target, Leaves = $leaves>,
        >(
            $( $slot: &[$leaf], )+
            read_offsets: &[u32],
            indices: &[u32],
            index_offset: &[u32],
            len: &[u32],
            $( $out: &mut [$output], )+
            write_offsets: &[u32],
        ) {
            let index = ABSOLUTE_POS as usize;
            if index < len[0] as usize {
                let source_index = indices[index_offset[0] as usize + index] as usize;
                let source = Expr::$method($( $slot, )+ read_offsets, source_index);
                Layout::decompose(Target::write_from(source)).store(
                    $( $out, )+ write_offsets, index,
                );
            }
        }
    };
}

macro_rules! define_permutation_kernels_for_eval {
    (
        $eval:ident, $method:ident;
        [$( $leaf:ident : $slot:ident ),+];
        [$k1:ident, $k2:ident, $k3:ident, $k4:ident, $k5:ident, $k6:ident, $k7:ident]
    ) => {
        define_permutation_kernel!($k1,$eval,$method; [$($leaf:$slot),+]; [O0:out0]; Last<O0>);
        define_permutation_kernel!($k2,$eval,$method; [$($leaf:$slot),+]; [O0:out0,O1:out1]; More<O0,Last<O1>>);
        define_permutation_kernel!($k3,$eval,$method; [$($leaf:$slot),+]; [O0:out0,O1:out1,O2:out2]; More<O0,More<O1,Last<O2>>>);
        define_permutation_kernel!($k4,$eval,$method; [$($leaf:$slot),+]; [O0:out0,O1:out1,O2:out2,O3:out3]; More<O0,More<O1,More<O2,Last<O3>>>>);
        define_permutation_kernel!($k5,$eval,$method; [$($leaf:$slot),+]; [O0:out0,O1:out1,O2:out2,O3:out3,O4:out4]; More<O0,More<O1,More<O2,More<O3,Last<O4>>>>>);
        define_permutation_kernel!($k6,$eval,$method; [$($leaf:$slot),+]; [O0:out0,O1:out1,O2:out2,O3:out3,O4:out4,O5:out5]; More<O0,More<O1,More<O2,More<O3,More<O4,Last<O5>>>>>>);
        define_permutation_kernel!($k7,$eval,$method; [$($leaf:$slot),+]; [O0:out0,O1:out1,O2:out2,O3:out3,O4:out4,O5:out5,O6:out6]; More<O0,More<O1,More<O2,More<O3,More<O4,More<O5,Last<O6>>>>>>>);
    };
}

define_permutation_kernels_for_eval!(Eval1,eval1; [L0:slot0]; [permutation_a1_s1,permutation_a1_s2,permutation_a1_s3,permutation_a1_s4,permutation_a1_s5,permutation_a1_s6,permutation_a1_s7]);
define_permutation_kernels_for_eval!(Eval2,eval2; [L0:slot0,L1:slot1]; [permutation_a2_s1,permutation_a2_s2,permutation_a2_s3,permutation_a2_s4,permutation_a2_s5,permutation_a2_s6,permutation_a2_s7]);
define_permutation_kernels_for_eval!(Eval3,eval3; [L0:slot0,L1:slot1,L2:slot2]; [permutation_a3_s1,permutation_a3_s2,permutation_a3_s3,permutation_a3_s4,permutation_a3_s5,permutation_a3_s6,permutation_a3_s7]);
define_permutation_kernels_for_eval!(Eval4,eval4; [L0:slot0,L1:slot1,L2:slot2,L3:slot3]; [permutation_a4_s1,permutation_a4_s2,permutation_a4_s3,permutation_a4_s4,permutation_a4_s5,permutation_a4_s6,permutation_a4_s7]);
define_permutation_kernels_for_eval!(Eval5,eval5; [L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4]; [permutation_a5_s1,permutation_a5_s2,permutation_a5_s3,permutation_a5_s4,permutation_a5_s5,permutation_a5_s6,permutation_a5_s7]);
define_permutation_kernels_for_eval!(Eval6,eval6; [L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5]; [permutation_a6_s1,permutation_a6_s2,permutation_a6_s3,permutation_a6_s4,permutation_a6_s5,permutation_a6_s6,permutation_a6_s7]);
define_permutation_kernels_for_eval!(Eval7,eval7; [L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5,L6:slot6]; [permutation_a7_s1,permutation_a7_s2,permutation_a7_s3,permutation_a7_s4,permutation_a7_s5,permutation_a7_s6,permutation_a7_s7]);
define_permutation_kernels_for_eval!(Eval8,eval8; [L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5,L6:slot6,L7:slot7]; [permutation_a8_s1,permutation_a8_s2,permutation_a8_s3,permutation_a8_s4,permutation_a8_s5,permutation_a8_s6,permutation_a8_s7]);

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

macro_rules! impl_permutation_dispatch {
    (
        $arity:ty, $storage:ty, $eval:ident, $kernel:ident;
        [$( $leaf:ident : $read_index:literal ),+], $read_env:ty;
        [$( $output:ident : $write_index:literal ),+], $write_env:ty;
        $leaves:ty
    ) => {
        impl<R, Input, Output, Source, $( $leaf, )+ $( $output, )+>
            PermutationDispatch<R, Input, Output, $read_env, $write_env>
            for Dispatch<$arity, $storage>
        where
            R: Runtime,
            $( $leaf: MStorageElement, )+
            $( $output: MStorageElement, )+
            Source: StorageLayout + Send + Sync + 'static,
            Input: ReadExpression<Item = Source>
                + LowerReadExpression<Slots = $read_env>
                + StageRead<R, Env0>,
            Input::DeviceExpr: $eval<Source, $( $leaf ),+>,
            Output: OutputExpression<StorageArity = $storage>
                + LowerOutputExpression<Slots = $write_env>
                + StageOutput<R, Env0>,
            Output::Item: StorageLayout<StorageArity = $storage, StorageLeaves = $leaves>
                + WriteFrom<Source>,
            <Output::Item as StorageLayout>::DeviceLayout:
                Decompose<Output::Item, Leaves = $leaves>,
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
                    indices,
                    exec.client(),
                    exec.id(),
                    &mut index_reads,
                )?;
                let mut writes = OutputBindings::new();
                output.stage_output(exec.id(), &mut writes)?;
                let read_offsets = exec.client().create_from_slice(u32::as_bytes(&reads.offsets));
                let index_offset = exec.client().create_from_slice(u32::as_bytes(&index_reads.offsets));
                let write_offsets = exec.client().create_from_slice(u32::as_bytes(&writes.offsets));
                let len_handle = exec.client().create_from_slice(u32::as_bytes(&[len_u32]));
                unsafe {
                    $kernel::launch_unchecked::<
                        Source,
                        Output::Item,
                        $( $leaf, )+
                        $( $output, )+
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
                        $( BufferArg::from_raw_parts(writes.slots[$write_index].0.clone(), writes.slots[$write_index].1), )+
                        BufferArg::from_raw_parts(write_offsets, writes.offsets.len()),
                    );
                }
                Ok(())
            }
        }
    };
}

macro_rules! impl_all_permutation_storage_for_read {
    (
        $arity:ty, $eval:ident;
        [$( $leaf:ident : $read_index:literal ),+], $read_env:ty;
        [$k1:ident, $k2:ident, $k3:ident, $k4:ident, $k5:ident, $k6:ident, $k7:ident]
    ) => {
        impl_permutation_dispatch!($arity,S1,$eval,$k1; [$($leaf:$read_index),+],$read_env; [O0:0],Env1<O0>; Last<O0>);
        impl_permutation_dispatch!($arity,S2,$eval,$k2; [$($leaf:$read_index),+],$read_env; [O0:0,O1:1],Env2<O0,O1>; More<O0,Last<O1>>);
        impl_permutation_dispatch!($arity,S3,$eval,$k3; [$($leaf:$read_index),+],$read_env; [O0:0,O1:1,O2:2],Env3<O0,O1,O2>; More<O0,More<O1,Last<O2>>>);
        impl_permutation_dispatch!($arity,S4,$eval,$k4; [$($leaf:$read_index),+],$read_env; [O0:0,O1:1,O2:2,O3:3],Env4<O0,O1,O2,O3>; More<O0,More<O1,More<O2,Last<O3>>>>);
        impl_permutation_dispatch!($arity,S5,$eval,$k5; [$($leaf:$read_index),+],$read_env; [O0:0,O1:1,O2:2,O3:3,O4:4],Env5<O0,O1,O2,O3,O4>; More<O0,More<O1,More<O2,More<O3,Last<O4>>>>>);
        impl_permutation_dispatch!($arity,S6,$eval,$k6; [$($leaf:$read_index),+],$read_env; [O0:0,O1:1,O2:2,O3:3,O4:4,O5:5],Env6<O0,O1,O2,O3,O4,O5>; More<O0,More<O1,More<O2,More<O3,More<O4,Last<O5>>>>>>);
        impl_permutation_dispatch!($arity,S7,$eval,$k7; [$($leaf:$read_index),+],$read_env; [O0:0,O1:1,O2:2,O3:3,O4:4,O5:5,O6:6],Env7<O0,O1,O2,O3,O4,O5,O6>; More<O0,More<O1,More<O2,More<O3,More<O4,More<O5,Last<O6>>>>>>>);
    };
}

impl_all_permutation_storage_for_read!(A1,Eval1; [L0:0],Env1<L0>; [permutation_a1_s1,permutation_a1_s2,permutation_a1_s3,permutation_a1_s4,permutation_a1_s5,permutation_a1_s6,permutation_a1_s7]);
impl_all_permutation_storage_for_read!(A2,Eval2; [L0:0,L1:1],Env2<L0,L1>; [permutation_a2_s1,permutation_a2_s2,permutation_a2_s3,permutation_a2_s4,permutation_a2_s5,permutation_a2_s6,permutation_a2_s7]);
impl_all_permutation_storage_for_read!(A3,Eval3; [L0:0,L1:1,L2:2],Env3<L0,L1,L2>; [permutation_a3_s1,permutation_a3_s2,permutation_a3_s3,permutation_a3_s4,permutation_a3_s5,permutation_a3_s6,permutation_a3_s7]);
impl_all_permutation_storage_for_read!(A4,Eval4; [L0:0,L1:1,L2:2,L3:3],Env4<L0,L1,L2,L3>; [permutation_a4_s1,permutation_a4_s2,permutation_a4_s3,permutation_a4_s4,permutation_a4_s5,permutation_a4_s6,permutation_a4_s7]);
impl_all_permutation_storage_for_read!(A5,Eval5; [L0:0,L1:1,L2:2,L3:3,L4:4],Env5<L0,L1,L2,L3,L4>; [permutation_a5_s1,permutation_a5_s2,permutation_a5_s3,permutation_a5_s4,permutation_a5_s5,permutation_a5_s6,permutation_a5_s7]);
impl_all_permutation_storage_for_read!(A6,Eval6; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5],Env6<L0,L1,L2,L3,L4,L5>; [permutation_a6_s1,permutation_a6_s2,permutation_a6_s3,permutation_a6_s4,permutation_a6_s5,permutation_a6_s6,permutation_a6_s7]);
impl_all_permutation_storage_for_read!(A7,Eval7; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5,L6:6],Env7<L0,L1,L2,L3,L4,L5,L6>; [permutation_a7_s1,permutation_a7_s2,permutation_a7_s3,permutation_a7_s4,permutation_a7_s5,permutation_a7_s6,permutation_a7_s7]);
impl_all_permutation_storage_for_read!(A8,Eval8; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5,L6:6,L7:7],Env8<L0,L1,L2,L3,L4,L5,L6,L7>; [permutation_a8_s1,permutation_a8_s2,permutation_a8_s3,permutation_a8_s4,permutation_a8_s5,permutation_a8_s6,permutation_a8_s7]);

pub(crate) fn apply_permutation<R, Input, Output>(
    exec: &Executor<R>,
    input: Input,
    indices: crate::Column<u32>,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: ReadExpression + LowerReadExpression + StageRead<R, Env0>,
    Output: OutputExpression + LowerOutputExpression + StageOutput<R, Env0>,
    Dispatch<Input::ReadArity, Output::StorageArity>:
        PermutationDispatch<R, Input, Output, Input::Slots, Output::Slots>,
{
    <Dispatch<Input::ReadArity, Output::StorageArity> as PermutationDispatch<
        R,
        Input,
        Output,
        Input::Slots,
        Output::Slots,
    >>::run(exec, &input, &indices, &output)
}

/// Internal public-API capability that normalizes values and indices
/// independently before applying the permutation.
#[doc(hidden)]
pub trait GatherNormalized<R: Runtime, Indices, Output>: NormalizeInput<R> {
    fn gather_normalized(
        self,
        exec: &Executor<R>,
        indices: Indices,
        output: Output,
    ) -> Result<(), Error>;
}

impl<R, Values, Indices, Output> GatherNormalized<R, Indices, Output> for Values
where
    R: Runtime,
    Values: NormalizeInput<R>,
    Values::Storage: CanonicalStorage<R>,
    Indices: NormalizeInput<R> + ReadExpression<Item = u32>,
    Indices::Storage: CanonicalStorage<R>,
    <Indices::Storage as CanonicalStorage<R>>::Read: ReadExpression<Item = u32>,
    <Values::Storage as CanonicalStorage<R>>::Read:
        GatherInput<R, <Indices::Storage as CanonicalStorage<R>>::Read, Output>,
{
    fn gather_normalized(
        self,
        exec: &Executor<R>,
        indices: Indices,
        output: Output,
    ) -> Result<(), Error> {
        let values = self.normalize(exec)?;
        let indices = indices.normalize(exec)?;
        gather_direct(exec, values.read(), indices.read(), output)
    }
}

/// Gathers `values[indices[i]]` into preallocated output storage.
pub(crate) fn gather<R, Values, Indices, Output>(
    exec: &Executor<R>,
    values: Values,
    indices: Indices,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Values: GatherNormalized<R, Indices, Output>,
{
    values.gather_normalized(exec, indices, output)
}

/// Internal public-API capability for masked gather.
#[doc(hidden)]
pub trait GatherWhereInput<R: Runtime, Indices, Stencil, Output>: NormalizeInput<R> {
    fn gather_where_normalized(
        self,
        exec: &Executor<R>,
        indices: Indices,
        stencil: Stencil,
        output: Output,
    ) -> Result<(), Error>;
}

impl<R, Values, Indices, Stencil, Output> GatherWhereInput<R, Indices, Stencil, Output> for Values
where
    R: Runtime,
    Values: NormalizeInput<R>,
    Values::Item: CanonicalAlloc<R>,
    Values::Storage: CanonicalStorage<R>,
    <Values::Item as CanonicalAlloc<R>>::CanonicalStorage: CanonicalStorage<R>,
    Indices: NormalizeInput<R> + ReadExpression<Item = u32>,
    Indices::Storage: CanonicalStorage<R>,
    <Indices::Storage as CanonicalStorage<R>>::Read: ReadExpression<Item = u32>,
    <Values::Storage as CanonicalStorage<R>>::Read: GatherInput<
            R,
            <Indices::Storage as CanonicalStorage<R>>::Read,
            <<Values::Item as CanonicalAlloc<R>>::CanonicalStorage as CanonicalStorage<R>>::Write,
        >,
    <<Values::Item as CanonicalAlloc<R>>::CanonicalStorage as CanonicalStorage<R>>::Read:
        MaskedCopyInput<R, Output>,
    Stencil: crate::selection::FlagInput<R>,
    Output: OutputExpression,
{
    fn gather_where_normalized(
        self,
        exec: &Executor<R>,
        indices: Indices,
        stencil: Stencil,
        output: Output,
    ) -> Result<(), Error> {
        let stencil_len = stencil.flag_len()?;
        let output_len = output.logical_len()?;
        if stencil_len != output_len {
            return Err(Error::LengthMismatch {
                left: stencil_len,
                right: output_len,
            });
        }
        let values = self.normalize(exec)?;
        let indices = indices.normalize(exec)?;
        let gathered = exec.alloc_canonical::<Values::Item>(output_len);
        gather_direct(exec, values.read(), indices.read(), gathered.write())?;
        let flags = stencil.materialize_flags(exec)?;
        gathered.read().masked_copy(exec, &flags, output)
    }
}

/// Gathers only rows whose stencil is nonzero, preserving other output rows.
pub(crate) fn gather_where<R, Values, Indices, Stencil, Output>(
    exec: &Executor<R>,
    values: Values,
    indices: Indices,
    stencil: Stencil,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Values: GatherWhereInput<R, Indices, Stencil, Output>,
{
    values.gather_where_normalized(exec, indices, stencil, output)
}

/// Internal capability proving reverse permutation has a canonical evaluator.
#[doc(hidden)]
pub trait ReverseInput<R: Runtime, Output>: ReadExpression + Sized {
    fn reverse(self, exec: &Executor<R>, output: Output) -> Result<(), Error>;
}

impl<R, Values, Output> ReverseInput<R, Output> for Values
where
    R: Runtime,
    Values: ReadExpression + StageRead<R, Env0>,
    Values::Item: StorageLayout,
    Permute<Values, ReverseCounting>:
        ReadExpression<Item = Values::Item> + LowerReadExpression + StageRead<R, Env0>,
    Output: OutputExpression + LowerOutputExpression + StageOutput<R, Env0>,
    Output::Item: crate::WriteFrom<Values::Item>,
    Dispatch<<Permute<Values, ReverseCounting> as ReadExpression>::ReadArity, Output::StorageArity>:
        MaterializeDispatch<
                R,
                Permute<Values, ReverseCounting>,
                Output,
                <Permute<Values, ReverseCounting> as LowerReadExpression>::Slots,
                Output::Slots,
            >,
{
    fn reverse(self, exec: &Executor<R>, output: Output) -> Result<(), Error> {
        let len = self.logical_len()?;
        materialize(exec, Permute::new(self, ReverseCounting::new(len)), output)
    }
}

/// Reverses values into preallocated output storage.
pub(crate) fn reverse<R, Values, Output>(
    exec: &Executor<R>,
    values: Values,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Values: ReverseInput<R, Output>,
{
    values.reverse(exec, output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CanonicalStorage, Counting, Permute, Zip};
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

        gather(&exec, values, indices.column(), output).unwrap();
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

        reverse(&exec, values, output).unwrap();
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
    fn gather_normalizes_eval8_values_and_lazy_indices_independently() {
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

        gather(&exec, values, indices, output.write()).unwrap();
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

        gather_where(
            &exec,
            values.column(),
            indices.column(),
            stencil.column(),
            output.slice_mut(..),
        )
        .unwrap();
        assert_eq!(exec.to_host(&output).unwrap(), vec![40, 200, 20, 400]);
    }
}
