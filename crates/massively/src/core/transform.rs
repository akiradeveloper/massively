//! Materialization and unary transform dispatch.

use cubecl::prelude::*;

use crate::{
    A1, A2, A3, A4, A5, A6, A7, A8, Dispatch, Error, Executor, MStorageElement, ReadExpression, S1,
    S2, S3, S4, S5, S6, S7, StorageLayout, Transform, WriteFrom,
    eval::{Eval1, Eval2, Eval3, Eval4, Eval5, Eval6, Eval7, Eval8},
    op::UnaryOp,
    output::{LowerOutputExpression, OutputBindings, OutputExpression, StageOutput},
    read::{Env0, Env1, Env2, Env3, Env4, Env5, Env6, Env7, Env8, LowerReadExpression},
    reduce::{StageRead, StagedBindings},
    storage::{
        Decompose, Last, More, StoreLeaves1, StoreLeaves1Expand, StoreLeaves2, StoreLeaves2Expand,
        StoreLeaves3, StoreLeaves3Expand, StoreLeaves4, StoreLeaves4Expand, StoreLeaves5,
        StoreLeaves5Expand, StoreLeaves6, StoreLeaves6Expand, StoreLeaves7, StoreLeaves7Expand,
    },
};

const BLOCK_SIZE: u32 = 256;

macro_rules! define_materialize_kernel {
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
            len: &[u32],
            $( $out: &mut [$output], )+
            write_offsets: &[u32],
        ) {
            let index = ABSOLUTE_POS as usize;
            if index < len[0] as usize {
                let source = Expr::$method($( $slot, )+ read_offsets, index);
                let target = Target::write_from(source);
                let leaves = Layout::decompose(target);
                leaves.store($( $out, )+ write_offsets, index);
            }
        }
    };
}

macro_rules! define_materialize_kernels_for_eval {
    (
        $eval:ident, $method:ident;
        [$( $leaf:ident : $slot:ident ),+];
        [$k1:ident, $k2:ident, $k3:ident, $k4:ident, $k5:ident, $k6:ident, $k7:ident]
    ) => {
        define_materialize_kernel!($k1, $eval, $method; [$($leaf:$slot),+]; [O0:out0]; Last<O0>);
        define_materialize_kernel!($k2, $eval, $method; [$($leaf:$slot),+]; [O0:out0,O1:out1]; More<O0,Last<O1>>);
        define_materialize_kernel!($k3, $eval, $method; [$($leaf:$slot),+]; [O0:out0,O1:out1,O2:out2]; More<O0,More<O1,Last<O2>>>);
        define_materialize_kernel!($k4, $eval, $method; [$($leaf:$slot),+]; [O0:out0,O1:out1,O2:out2,O3:out3]; More<O0,More<O1,More<O2,Last<O3>>>>);
        define_materialize_kernel!($k5, $eval, $method; [$($leaf:$slot),+]; [O0:out0,O1:out1,O2:out2,O3:out3,O4:out4]; More<O0,More<O1,More<O2,More<O3,Last<O4>>>>>);
        define_materialize_kernel!($k6, $eval, $method; [$($leaf:$slot),+]; [O0:out0,O1:out1,O2:out2,O3:out3,O4:out4,O5:out5]; More<O0,More<O1,More<O2,More<O3,More<O4,Last<O5>>>>>>);
        define_materialize_kernel!($k7, $eval, $method; [$($leaf:$slot),+]; [O0:out0,O1:out1,O2:out2,O3:out3,O4:out4,O5:out5,O6:out6]; More<O0,More<O1,More<O2,More<O3,More<O4,More<O5,Last<O6>>>>>>>);
    };
}

define_materialize_kernels_for_eval!(Eval1, eval1; [L0:slot0]; [materialize_a1_s1, materialize_a1_s2, materialize_a1_s3, materialize_a1_s4, materialize_a1_s5, materialize_a1_s6, materialize_a1_s7]);
define_materialize_kernels_for_eval!(Eval2, eval2; [L0:slot0,L1:slot1]; [materialize_a2_s1, materialize_a2_s2, materialize_a2_s3, materialize_a2_s4, materialize_a2_s5, materialize_a2_s6, materialize_a2_s7]);
define_materialize_kernels_for_eval!(Eval3, eval3; [L0:slot0,L1:slot1,L2:slot2]; [materialize_a3_s1, materialize_a3_s2, materialize_a3_s3, materialize_a3_s4, materialize_a3_s5, materialize_a3_s6, materialize_a3_s7]);
define_materialize_kernels_for_eval!(Eval4, eval4; [L0:slot0,L1:slot1,L2:slot2,L3:slot3]; [materialize_a4_s1, materialize_a4_s2, materialize_a4_s3, materialize_a4_s4, materialize_a4_s5, materialize_a4_s6, materialize_a4_s7]);
define_materialize_kernels_for_eval!(Eval5, eval5; [L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4]; [materialize_a5_s1, materialize_a5_s2, materialize_a5_s3, materialize_a5_s4, materialize_a5_s5, materialize_a5_s6, materialize_a5_s7]);
define_materialize_kernels_for_eval!(Eval6, eval6; [L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5]; [materialize_a6_s1, materialize_a6_s2, materialize_a6_s3, materialize_a6_s4, materialize_a6_s5, materialize_a6_s6, materialize_a6_s7]);
define_materialize_kernels_for_eval!(Eval7, eval7; [L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5,L6:slot6]; [materialize_a7_s1, materialize_a7_s2, materialize_a7_s3, materialize_a7_s4, materialize_a7_s5, materialize_a7_s6, materialize_a7_s7]);
define_materialize_kernels_for_eval!(Eval8, eval8; [L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5,L6:slot6,L7:slot7]; [materialize_a8_s1, materialize_a8_s2, materialize_a8_s3, materialize_a8_s4, materialize_a8_s5, materialize_a8_s6, materialize_a8_s7]);

#[doc(hidden)]
pub trait MaterializeDispatch<R, Input, Output, ReadSlots, WriteSlots>
where
    R: Runtime,
{
    fn run(exec: &Executor<R>, input: &Input, output: &Output) -> Result<(), Error>;
}

macro_rules! impl_materialize_dispatch {
    (
        $arity:ty, $storage:ty, $eval:ident, $kernel:ident;
        [$( $leaf:ident : $read_index:literal ),+], $read_env:ty;
        [$( $output:ident : $write_index:literal ),+], $write_env:ty;
        $leaves:ty
    ) => {
        impl<R, Input, Output, Source, $( $leaf, )+ $( $output, )+>
            MaterializeDispatch<R, Input, Output, $read_env, $write_env>
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
            fn run(exec: &Executor<R>, input: &Input, output: &Output) -> Result<(), Error> {
                let input_len = input.logical_len()?;
                let output_len = output.logical_len()?;
                if output_len < input_len {
                    return Err(Error::OutputTooShort { input: input_len, output: output_len });
                }
                if input_len == 0 {
                    return Ok(());
                }
                let len = u32::try_from(input_len)
                    .map_err(|_| Error::LengthTooLarge { len: input_len })?;
                let mut reads = StagedBindings::new();
                input.stage_at(exec.client(), exec.id(), &mut reads)?;
                let mut writes = OutputBindings::new();
                output.stage_output(exec.id(), &mut writes)?;
                let read_offsets = exec.client().create_from_slice(u32::as_bytes(&reads.offsets));
                let write_offsets = exec.client().create_from_slice(u32::as_bytes(&writes.offsets));
                let len_handle = exec.client().create_from_slice(u32::as_bytes(&[len]));
                let cubes = crate::launch::cube_count_1d(
                    (len as usize).div_ceil(BLOCK_SIZE as usize),
                )?;
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
                        cubes,
                        CubeDim::new_1d(BLOCK_SIZE),
                        $( BufferArg::from_raw_parts(reads.slots[$read_index].0.clone(), reads.slots[$read_index].1), )+
                        BufferArg::from_raw_parts(read_offsets, reads.offsets.len()),
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

macro_rules! impl_all_storage_for_read {
    (
        $arity:ty, $eval:ident;
        [$( $leaf:ident : $read_index:literal ),+], $read_env:ty;
        [$k1:ident, $k2:ident, $k3:ident, $k4:ident, $k5:ident, $k6:ident, $k7:ident]
    ) => {
        impl_materialize_dispatch!($arity,S1,$eval,$k1; [$($leaf:$read_index),+],$read_env; [O0:0],Env1<O0>; Last<O0>);
        impl_materialize_dispatch!($arity,S2,$eval,$k2; [$($leaf:$read_index),+],$read_env; [O0:0,O1:1],Env2<O0,O1>; More<O0,Last<O1>>);
        impl_materialize_dispatch!($arity,S3,$eval,$k3; [$($leaf:$read_index),+],$read_env; [O0:0,O1:1,O2:2],Env3<O0,O1,O2>; More<O0,More<O1,Last<O2>>>);
        impl_materialize_dispatch!($arity,S4,$eval,$k4; [$($leaf:$read_index),+],$read_env; [O0:0,O1:1,O2:2,O3:3],Env4<O0,O1,O2,O3>; More<O0,More<O1,More<O2,Last<O3>>>>);
        impl_materialize_dispatch!($arity,S5,$eval,$k5; [$($leaf:$read_index),+],$read_env; [O0:0,O1:1,O2:2,O3:3,O4:4],Env5<O0,O1,O2,O3,O4>; More<O0,More<O1,More<O2,More<O3,Last<O4>>>>>);
        impl_materialize_dispatch!($arity,S6,$eval,$k6; [$($leaf:$read_index),+],$read_env; [O0:0,O1:1,O2:2,O3:3,O4:4,O5:5],Env6<O0,O1,O2,O3,O4,O5>; More<O0,More<O1,More<O2,More<O3,More<O4,Last<O5>>>>>>);
        impl_materialize_dispatch!($arity,S7,$eval,$k7; [$($leaf:$read_index),+],$read_env; [O0:0,O1:1,O2:2,O3:3,O4:4,O5:5,O6:6],Env7<O0,O1,O2,O3,O4,O5,O6>; More<O0,More<O1,More<O2,More<O3,More<O4,More<O5,Last<O6>>>>>>>);
    };
}

impl_all_storage_for_read!(A1,Eval1; [L0:0],Env1<L0>; [materialize_a1_s1,materialize_a1_s2,materialize_a1_s3,materialize_a1_s4,materialize_a1_s5,materialize_a1_s6,materialize_a1_s7]);
impl_all_storage_for_read!(A2,Eval2; [L0:0,L1:1],Env2<L0,L1>; [materialize_a2_s1,materialize_a2_s2,materialize_a2_s3,materialize_a2_s4,materialize_a2_s5,materialize_a2_s6,materialize_a2_s7]);
impl_all_storage_for_read!(A3,Eval3; [L0:0,L1:1,L2:2],Env3<L0,L1,L2>; [materialize_a3_s1,materialize_a3_s2,materialize_a3_s3,materialize_a3_s4,materialize_a3_s5,materialize_a3_s6,materialize_a3_s7]);
impl_all_storage_for_read!(A4,Eval4; [L0:0,L1:1,L2:2,L3:3],Env4<L0,L1,L2,L3>; [materialize_a4_s1,materialize_a4_s2,materialize_a4_s3,materialize_a4_s4,materialize_a4_s5,materialize_a4_s6,materialize_a4_s7]);
impl_all_storage_for_read!(A5,Eval5; [L0:0,L1:1,L2:2,L3:3,L4:4],Env5<L0,L1,L2,L3,L4>; [materialize_a5_s1,materialize_a5_s2,materialize_a5_s3,materialize_a5_s4,materialize_a5_s5,materialize_a5_s6,materialize_a5_s7]);
impl_all_storage_for_read!(A6,Eval6; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5],Env6<L0,L1,L2,L3,L4,L5>; [materialize_a6_s1,materialize_a6_s2,materialize_a6_s3,materialize_a6_s4,materialize_a6_s5,materialize_a6_s6,materialize_a6_s7]);
impl_all_storage_for_read!(A7,Eval7; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5,L6:6],Env7<L0,L1,L2,L3,L4,L5,L6>; [materialize_a7_s1,materialize_a7_s2,materialize_a7_s3,materialize_a7_s4,materialize_a7_s5,materialize_a7_s6,materialize_a7_s7]);
impl_all_storage_for_read!(A8,Eval8; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5,L6:6,L7:7],Env8<L0,L1,L2,L3,L4,L5,L6,L7>; [materialize_a8_s1,materialize_a8_s2,materialize_a8_s3,materialize_a8_s4,materialize_a8_s5,materialize_a8_s6,materialize_a8_s7]);

/// Evaluates a lazy read expression and writes it to a preallocated output tree.
pub(crate) fn materialize<R, Input, Output>(
    exec: &Executor<R>,
    input: Input,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: ReadExpression + LowerReadExpression + StageRead<R, Env0>,
    Input::Item: StorageLayout,
    Output: OutputExpression + LowerOutputExpression + StageOutput<R, Env0>,
    Output::Item: WriteFrom<Input::Item>,
    Dispatch<Input::ReadArity, Output::StorageArity>:
        MaterializeDispatch<R, Input, Output, Input::Slots, Output::Slots>,
{
    <Dispatch<Input::ReadArity, Output::StorageArity> as MaterializeDispatch<
        R,
        Input,
        Output,
        Input::Slots,
        Output::Slots,
    >>::run(exec, &input, &output)
}

/// Applies a unary operation and writes its result to preallocated storage.
pub(crate) fn transform<R, Input, Output, Op>(
    exec: &Executor<R>,
    input: Input,
    op: Op,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: ReadExpression,
    Op: UnaryOp<Input::Item>,
    Transform<Input, Op>:
        ReadExpression<Item = Op::Output> + LowerReadExpression + StageRead<R, Env0>,
    Op::Output: StorageLayout,
    Output: OutputExpression + LowerOutputExpression + StageOutput<R, Env0>,
    Output::Item: WriteFrom<<Op as UnaryOp<Input::Item>>::Output>,
    Dispatch<<Transform<Input, Op> as ReadExpression>::ReadArity, Output::StorageArity>:
        MaterializeDispatch<
                R,
                Transform<Input, Op>,
                Output,
                <Transform<Input, Op> as LowerReadExpression>::Slots,
                Output::Slots,
            >,
{
    materialize(exec, Transform::new(input, op), output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Counting, DeviceVec, Permute, Zip};
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    struct Double;

    #[cubecl::cube]
    impl UnaryOp<u32> for Double {
        type Output = u32;
        fn apply(input: u32) -> u32 {
            input * 2
        }
    }

    #[test]
    fn transform_a1_s1_writes_preallocated_slice() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let input: DeviceVec<_, u32> = exec.to_device(&[1, 2, 3, 4]);
        let output = exec.to_device(&[99_u32; 6]);

        transform(&exec, input.slice(1..4), Double, output.slice_mut(2..5)).unwrap();

        assert_eq!(exec.to_host(&output).unwrap(), vec![99, 99, 4, 6, 8, 99]);
    }

    #[test]
    fn materialize_a3_s3_reassociates_only_at_write_boundary() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let a = exec.to_device(&[1_u32, 2, 3]);
        let b = exec.to_device(&[10_f32, 20.0, 30.0]);
        let c = exec.to_device(&[-1_i32, -2, -3]);
        let out_a = exec.to_device(&[0_u32; 3]);
        let out_b = exec.to_device(&[0_f32; 3]);
        let out_c = exec.to_device(&[0_i32; 3]);

        let input = Zip::new(a.column(), Zip::new(b.column(), c.column()));
        let output = Zip::new(
            Zip::new(out_a.slice_mut(..), out_b.slice_mut(..)),
            out_c.slice_mut(..),
        );
        materialize(&exec, input, output).unwrap();

        assert_eq!(exec.to_host(&out_a).unwrap(), vec![1, 2, 3]);
        assert_eq!(exec.to_host(&out_b).unwrap(), vec![10.0, 20.0, 30.0]);
        assert_eq!(exec.to_host(&out_c).unwrap(), vec![-1, -2, -3]);
    }

    #[test]
    fn materialize_a8_s7_uses_the_canonical_evaluator_and_writer() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let columns: Vec<_> = (0_u32..7)
            .map(|column| exec.to_device(&[column * 10 + 1, column * 10 + 2, column * 10 + 3]))
            .collect();
        let outputs: Vec<_> = (0..7).map(|_| exec.to_device(&[0_u32; 3])).collect();
        let seven = Zip::new(
            columns[0].column(),
            Zip::new(
                columns[1].column(),
                Zip::new(
                    columns[2].column(),
                    Zip::new(
                        columns[3].column(),
                        Zip::new(
                            columns[4].column(),
                            Zip::new(columns[5].column(), columns[6].column()),
                        ),
                    ),
                ),
            ),
        );
        let input = Permute::new(seven, Counting::new(0, 3));
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

        materialize(&exec, input, output).unwrap();

        for (column, output) in outputs.iter().enumerate() {
            assert_eq!(
                exec.to_host(output).unwrap(),
                vec![
                    column as u32 * 10 + 1,
                    column as u32 * 10 + 2,
                    column as u32 * 10 + 3
                ]
            );
        }
    }
}
