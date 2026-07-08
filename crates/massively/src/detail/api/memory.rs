use crate::{
    detail::op::kernel::UnaryOp,
    device::{
        DeviceColumnView, DeviceVec, KernelColumn, KernelColumnAt, KernelColumnBindings, S0,
        StorageKernelColumn, Zip, Zip1, Zip2, Zip3,
    },
    error::Error,
    expr::{DeviceGpuExpr, LogicalDeviceExpr3, LogicalDeviceExpr7},
    kernels::*,
    policy::CubePolicy,
};
use cubecl::prelude::*;

fn transform_offset_handle<R: Runtime>(
    client: &ComputeClient<R>,
    offset: usize,
) -> Result<cubecl::server::Handle, Error> {
    let offset = u32::try_from(offset).map_err(|_| Error::LengthTooLarge { len: offset })?;
    Ok(client.create_from_slice(u32::as_bytes(&[offset])))
}

/// Storage shape used for a transformed device value.
#[doc(hidden)]
pub trait MItemStorage<R: Runtime>: CubeType {
    type Storage;
}

impl<R, A> MItemStorage<R> for (A,)
where
    R: Runtime,
    A: CubePrimitive + CubeElement,
{
    type Storage = Zip1<DeviceVec<R, A>>;
}

impl<R, A, B> MItemStorage<R> for (A, B)
where
    R: Runtime,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
{
    type Storage = Zip2<DeviceVec<R, A>, DeviceVec<R, B>>;
}

impl<R, A, B, C> MItemStorage<R> for (A, B, C)
where
    R: Runtime,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    C: CubePrimitive + CubeElement,
{
    type Storage = Zip3<DeviceVec<R, A>, DeviceVec<R, B>, DeviceVec<R, C>>;
}

macro_rules! impl_wide_mitem_storage {
    ($( $ty:ident ),+) => {
        impl<R, $( $ty ),+> MItemStorage<R> for ($( $ty, )+)
        where
            R: Runtime,
            $( $ty: CubePrimitive + CubeElement, )+
        {
            type Storage = ($( DeviceVec<R, $ty>, )+);
        }
    };
}

impl_wide_mitem_storage!(A, B, C, D);
impl_wide_mitem_storage!(A, B, C, D, E);
impl_wide_mitem_storage!(A, B, C, D, E, F);
impl_wide_mitem_storage!(A, B, C, D, E, F, G);

#[doc(hidden)]
pub trait TransformUnaryOutput<R, T, Op>: MItemStorage<R>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Op: UnaryOp<(T,), Output = Self>,
{
    fn run(
        policy: &CubePolicy<R>,
        input: DeviceColumnView<R, T>,
    ) -> Result<<Self as MItemStorage<R>>::Storage, Error>;
}

impl<R, T, A, Op> TransformUnaryOutput<R, T, Op> for (A,)
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    A: CubePrimitive + CubeElement,
    Op: UnaryOp<(T,), Output = (A,)>,
{
    fn run(
        policy: &CubePolicy<R>,
        input: DeviceColumnView<R, T>,
    ) -> Result<<Self as MItemStorage<R>>::Storage, Error> {
        let len = input.len();
        let client = policy.client();
        let output_a = client.empty(len * std::mem::size_of::<A>());
        if len != 0 {
            let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
            let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
            let offset_u32 = u32::try_from(input.offset)
                .map_err(|_| Error::LengthTooLarge { len: input.offset })?;
            let offset_handle = client.create_from_slice(u32::as_bytes(&[offset_u32]));
            let block_size = 256_u32;
            let block_count = len.div_ceil(block_size as usize);
            let block_count_u32 = u32::try_from(block_count)
                .map_err(|_| Error::LengthTooLarge { len: block_count })?;
            unsafe {
                transform_unary_tuple1_kernel::launch_unchecked::<T, A, Op, R>(
                    client,
                    crate::detail::launch::cube_count_1d(block_count_u32),
                    CubeDim::new_1d(block_size),
                    unsafe {
                        BufferArg::from_raw_parts(input.source.handle.clone(), input.source.len())
                    },
                    unsafe { BufferArg::from_raw_parts(offset_handle.clone(), 1) },
                    unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
                    unsafe { BufferArg::from_raw_parts(output_a.clone(), len) },
                );
            }
        }
        Ok(Zip1 {
            source: DeviceVec::from_handle(policy.id(), output_a, len),
        })
    }
}

impl<R, T, A, B, Op> TransformUnaryOutput<R, T, Op> for (A, B)
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    Op: UnaryOp<(T,), Output = (A, B)>,
{
    fn run(
        policy: &CubePolicy<R>,
        input: DeviceColumnView<R, T>,
    ) -> Result<<Self as MItemStorage<R>>::Storage, Error> {
        let len = input.len();
        let client = policy.client();
        let output_a = client.empty(len * std::mem::size_of::<A>());
        let output_b = client.empty(len * std::mem::size_of::<B>());
        if len != 0 {
            let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
            let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
            let offset_u32 = u32::try_from(input.offset)
                .map_err(|_| Error::LengthTooLarge { len: input.offset })?;
            let offset_handle = client.create_from_slice(u32::as_bytes(&[offset_u32]));
            let block_size = 256_u32;
            let block_count = len.div_ceil(block_size as usize);
            let block_count_u32 = u32::try_from(block_count)
                .map_err(|_| Error::LengthTooLarge { len: block_count })?;
            unsafe {
                transform_unary_tuple2_kernel::launch_unchecked::<T, A, B, Op, R>(
                    client,
                    crate::detail::launch::cube_count_1d(block_count_u32),
                    CubeDim::new_1d(block_size),
                    unsafe {
                        BufferArg::from_raw_parts(input.source.handle.clone(), input.source.len())
                    },
                    unsafe { BufferArg::from_raw_parts(offset_handle.clone(), 1) },
                    unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
                    unsafe { BufferArg::from_raw_parts(output_a.clone(), len) },
                    unsafe { BufferArg::from_raw_parts(output_b.clone(), len) },
                );
            }
        }
        Ok(Zip2 {
            left: DeviceVec::from_handle(policy.id(), output_a, len),
            right: DeviceVec::from_handle(policy.id(), output_b, len),
        })
    }
}

impl<R, T, A, B, C, Op> TransformUnaryOutput<R, T, Op> for (A, B, C)
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    C: CubePrimitive + CubeElement,
    Op: UnaryOp<(T,), Output = (A, B, C)>,
{
    fn run(
        policy: &CubePolicy<R>,
        input: DeviceColumnView<R, T>,
    ) -> Result<<Self as MItemStorage<R>>::Storage, Error> {
        let len = input.len();
        let client = policy.client();
        let output_a = client.empty(len * std::mem::size_of::<A>());
        let output_b = client.empty(len * std::mem::size_of::<B>());
        let output_c = client.empty(len * std::mem::size_of::<C>());
        if len != 0 {
            let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
            let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
            let offset_u32 = u32::try_from(input.offset)
                .map_err(|_| Error::LengthTooLarge { len: input.offset })?;
            let offset_handle = client.create_from_slice(u32::as_bytes(&[offset_u32]));
            let block_size = 256_u32;
            let block_count = len.div_ceil(block_size as usize);
            let block_count_u32 = u32::try_from(block_count)
                .map_err(|_| Error::LengthTooLarge { len: block_count })?;
            unsafe {
                transform_unary_tuple3_kernel::launch_unchecked::<T, A, B, C, Op, R>(
                    client,
                    crate::detail::launch::cube_count_1d(block_count_u32),
                    CubeDim::new_1d(block_size),
                    unsafe {
                        BufferArg::from_raw_parts(input.source.handle.clone(), input.source.len())
                    },
                    unsafe { BufferArg::from_raw_parts(offset_handle.clone(), 1) },
                    unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
                    unsafe { BufferArg::from_raw_parts(output_a.clone(), len) },
                    unsafe { BufferArg::from_raw_parts(output_b.clone(), len) },
                    unsafe { BufferArg::from_raw_parts(output_c.clone(), len) },
                );
            }
        }
        Ok(Zip3 {
            first: DeviceVec::from_handle(policy.id(), output_a, len),
            second: DeviceVec::from_handle(policy.id(), output_b, len),
            third: DeviceVec::from_handle(policy.id(), output_c, len),
        })
    }
}

macro_rules! impl_wide_transform_unary_output {
    (
        $kernel:ident,
        ($( $out_ty:ident : $out_handle:ident ),+)
    ) => {
        impl<R, T, $( $out_ty, )+ Op> TransformUnaryOutput<R, T, Op> for ($( $out_ty, )+)
        where
            R: Runtime,
            T: CubePrimitive + CubeElement,
            $( $out_ty: CubePrimitive + CubeElement, )+
            Op: UnaryOp<(T,), Output = ($( $out_ty, )+)>,
        {
            fn run(
                policy: &CubePolicy<R>,
                input: DeviceColumnView<R, T>,
            ) -> Result<<Self as MItemStorage<R>>::Storage, Error> {
                let len = input.len();
                let client = policy.client();
                $(
                    let $out_handle = client.empty(len * std::mem::size_of::<$out_ty>());
                )+
                if len != 0 {
                    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
                    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
                    let offset_handle = transform_offset_handle(client, input.offset)?;
                    let block_size = 256_u32;
                    let block_count = len.div_ceil(block_size as usize);
                    let block_count_u32 = u32::try_from(block_count)
                        .map_err(|_| Error::LengthTooLarge { len: block_count })?;
                    unsafe {
                        $kernel::launch_unchecked::<T, $( $out_ty, )+ Op, R>(
                            client,
                            crate::detail::launch::cube_count_1d(block_count_u32),
                            CubeDim::new_1d(block_size),
                            BufferArg::from_raw_parts(input.source.handle.clone(), input.source.len()),
                            BufferArg::from_raw_parts(offset_handle.clone(), 1),
                            BufferArg::from_raw_parts(len_handle.clone(), 1),
                            $(
                                BufferArg::from_raw_parts($out_handle.clone(), len),
                            )+
                        );
                    }
                }
                Ok(($( DeviceVec::from_handle(policy.id(), $out_handle, len), )+))
            }
        }
    };
}

impl_wide_transform_unary_output!(
    transform_tuple1_to_tuple4_kernel,
    (A: output_a, B: output_b, C: output_c, D: output_d)
);
impl_wide_transform_unary_output!(
    transform_tuple1_to_tuple5_kernel,
    (A: output_a, B: output_b, C: output_c, D: output_d, E: output_e)
);
impl_wide_transform_unary_output!(
    transform_tuple1_to_tuple6_kernel,
    (A: output_a, B: output_b, C: output_c, D: output_d, E: output_e, F: output_f)
);
impl_wide_transform_unary_output!(
    transform_tuple1_to_tuple7_kernel,
    (A: output_a, B: output_b, C: output_c, D: output_d, E: output_e, F: output_f, G: output_g)
);

#[doc(hidden)]
pub trait TransformLogical3Output<R, Input, LeafA, LeafB, LeafC, Expr, Op>:
    MItemStorage<R>
where
    R: Runtime,
    Input: CubeType + 'static + Send + Sync,
    LeafA: CubePrimitive + CubeElement,
    LeafB: CubePrimitive + CubeElement,
    LeafC: CubePrimitive + CubeElement,
    Expr: LogicalDeviceExpr3<Input, LeafA, LeafB, LeafC>,
    Op: UnaryOp<Input, Output = Self>,
{
    fn run_logical3(
        policy: &CubePolicy<R>,
        bindings: KernelColumnBindings,
        len: usize,
    ) -> Result<<Self as MItemStorage<R>>::Storage, Error>;
}

impl<R, Input, LeafA, LeafB, LeafC, Expr, OutA, Op>
    TransformLogical3Output<R, Input, LeafA, LeafB, LeafC, Expr, Op> for (OutA,)
where
    R: Runtime,
    Input: CubeType + 'static + Send + Sync,
    LeafA: CubePrimitive + CubeElement,
    LeafB: CubePrimitive + CubeElement,
    LeafC: CubePrimitive + CubeElement,
    Expr: LogicalDeviceExpr3<Input, LeafA, LeafB, LeafC>,
    OutA: CubePrimitive + CubeElement,
    Op: UnaryOp<Input, Output = (OutA,)>,
{
    fn run_logical3(
        policy: &CubePolicy<R>,
        bindings: KernelColumnBindings,
        len: usize,
    ) -> Result<<Self as MItemStorage<R>>::Storage, Error> {
        let client = policy.client();
        let output_a = client.empty(len * std::mem::size_of::<OutA>());
        if len != 0 {
            let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
            let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
            let offsets = bindings.slot_offsets_handle(client)?;
            let slot0 = bindings.slot_or_first(0);
            let slot1 = bindings.slot_or_first(1);
            let slot2 = bindings.slot_or_first(2);
            let block_size = 256_u32;
            let block_count = len.div_ceil(block_size as usize);
            let block_count_u32 = u32::try_from(block_count)
                .map_err(|_| Error::LengthTooLarge { len: block_count })?;
            unsafe {
                transform_logical3_to_tuple1_kernel::launch_unchecked::<
                    Input,
                    LeafA,
                    LeafB,
                    LeafC,
                    Expr,
                    OutA,
                    Op,
                    R,
                >(
                    client,
                    crate::detail::launch::cube_count_1d(block_count_u32),
                    CubeDim::new_1d(block_size),
                    BufferArg::from_raw_parts(slot0.0.clone(), slot0.1),
                    BufferArg::from_raw_parts(slot1.0.clone(), slot1.1),
                    BufferArg::from_raw_parts(slot2.0.clone(), slot2.1),
                    BufferArg::from_raw_parts(offsets.clone(), 4),
                    BufferArg::from_raw_parts(len_handle.clone(), 1),
                    BufferArg::from_raw_parts(output_a.clone(), len),
                );
            }
        }
        Ok(Zip1 {
            source: DeviceVec::from_handle(policy.id(), output_a, len),
        })
    }
}

impl<R, Input, LeafA, LeafB, LeafC, Expr, OutA, OutB, Op>
    TransformLogical3Output<R, Input, LeafA, LeafB, LeafC, Expr, Op> for (OutA, OutB)
where
    R: Runtime,
    Input: CubeType + 'static + Send + Sync,
    LeafA: CubePrimitive + CubeElement,
    LeafB: CubePrimitive + CubeElement,
    LeafC: CubePrimitive + CubeElement,
    Expr: LogicalDeviceExpr3<Input, LeafA, LeafB, LeafC>,
    OutA: CubePrimitive + CubeElement,
    OutB: CubePrimitive + CubeElement,
    Op: UnaryOp<Input, Output = (OutA, OutB)>,
{
    fn run_logical3(
        policy: &CubePolicy<R>,
        bindings: KernelColumnBindings,
        len: usize,
    ) -> Result<<Self as MItemStorage<R>>::Storage, Error> {
        let client = policy.client();
        let output_a = client.empty(len * std::mem::size_of::<OutA>());
        let output_b = client.empty(len * std::mem::size_of::<OutB>());
        if len != 0 {
            let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
            let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
            let offsets = bindings.slot_offsets_handle(client)?;
            let slot0 = bindings.slot_or_first(0);
            let slot1 = bindings.slot_or_first(1);
            let slot2 = bindings.slot_or_first(2);
            let block_size = 256_u32;
            let block_count = len.div_ceil(block_size as usize);
            let block_count_u32 = u32::try_from(block_count)
                .map_err(|_| Error::LengthTooLarge { len: block_count })?;
            unsafe {
                transform_logical3_to_tuple2_kernel::launch_unchecked::<
                    Input,
                    LeafA,
                    LeafB,
                    LeafC,
                    Expr,
                    OutA,
                    OutB,
                    Op,
                    R,
                >(
                    client,
                    crate::detail::launch::cube_count_1d(block_count_u32),
                    CubeDim::new_1d(block_size),
                    BufferArg::from_raw_parts(slot0.0.clone(), slot0.1),
                    BufferArg::from_raw_parts(slot1.0.clone(), slot1.1),
                    BufferArg::from_raw_parts(slot2.0.clone(), slot2.1),
                    BufferArg::from_raw_parts(offsets.clone(), 4),
                    BufferArg::from_raw_parts(len_handle.clone(), 1),
                    BufferArg::from_raw_parts(output_a.clone(), len),
                    BufferArg::from_raw_parts(output_b.clone(), len),
                );
            }
        }
        Ok(Zip2 {
            left: DeviceVec::from_handle(policy.id(), output_a, len),
            right: DeviceVec::from_handle(policy.id(), output_b, len),
        })
    }
}

impl<R, Input, LeafA, LeafB, LeafC, Expr, OutA, OutB, OutC, Op>
    TransformLogical3Output<R, Input, LeafA, LeafB, LeafC, Expr, Op> for (OutA, OutB, OutC)
where
    R: Runtime,
    Input: CubeType + 'static + Send + Sync,
    LeafA: CubePrimitive + CubeElement,
    LeafB: CubePrimitive + CubeElement,
    LeafC: CubePrimitive + CubeElement,
    Expr: LogicalDeviceExpr3<Input, LeafA, LeafB, LeafC>,
    OutA: CubePrimitive + CubeElement,
    OutB: CubePrimitive + CubeElement,
    OutC: CubePrimitive + CubeElement,
    Op: UnaryOp<Input, Output = (OutA, OutB, OutC)>,
{
    fn run_logical3(
        policy: &CubePolicy<R>,
        bindings: KernelColumnBindings,
        len: usize,
    ) -> Result<<Self as MItemStorage<R>>::Storage, Error> {
        let client = policy.client();
        let output_a = client.empty(len * std::mem::size_of::<OutA>());
        let output_b = client.empty(len * std::mem::size_of::<OutB>());
        let output_c = client.empty(len * std::mem::size_of::<OutC>());
        if len != 0 {
            let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
            let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
            let offsets = bindings.slot_offsets_handle(client)?;
            let slot0 = bindings.slot_or_first(0);
            let slot1 = bindings.slot_or_first(1);
            let slot2 = bindings.slot_or_first(2);
            let block_size = 256_u32;
            let block_count = len.div_ceil(block_size as usize);
            let block_count_u32 = u32::try_from(block_count)
                .map_err(|_| Error::LengthTooLarge { len: block_count })?;
            unsafe {
                transform_logical3_to_tuple3_kernel::launch_unchecked::<
                    Input,
                    LeafA,
                    LeafB,
                    LeafC,
                    Expr,
                    OutA,
                    OutB,
                    OutC,
                    Op,
                    R,
                >(
                    client,
                    crate::detail::launch::cube_count_1d(block_count_u32),
                    CubeDim::new_1d(block_size),
                    BufferArg::from_raw_parts(slot0.0.clone(), slot0.1),
                    BufferArg::from_raw_parts(slot1.0.clone(), slot1.1),
                    BufferArg::from_raw_parts(slot2.0.clone(), slot2.1),
                    BufferArg::from_raw_parts(offsets.clone(), 4),
                    BufferArg::from_raw_parts(len_handle.clone(), 1),
                    BufferArg::from_raw_parts(output_a.clone(), len),
                    BufferArg::from_raw_parts(output_b.clone(), len),
                    BufferArg::from_raw_parts(output_c.clone(), len),
                );
            }
        }
        Ok(Zip3 {
            first: DeviceVec::from_handle(policy.id(), output_a, len),
            second: DeviceVec::from_handle(policy.id(), output_b, len),
            third: DeviceVec::from_handle(policy.id(), output_c, len),
        })
    }
}

macro_rules! impl_transform_logical3_wide_output {
    (
        $kernel:ident,
        ($( $out_ty:ident : $out_handle:ident ),+),
        $storage:expr
    ) => {
        impl<R, Input, LeafA, LeafB, LeafC, Expr, $( $out_ty, )+ Op>
            TransformLogical3Output<R, Input, LeafA, LeafB, LeafC, Expr, Op> for ($( $out_ty, )+)
        where
            R: Runtime,
            Input: CubeType + 'static + Send + Sync,
            LeafA: CubePrimitive + CubeElement,
            LeafB: CubePrimitive + CubeElement,
            LeafC: CubePrimitive + CubeElement,
            Expr: LogicalDeviceExpr3<Input, LeafA, LeafB, LeafC>,
            $( $out_ty: CubePrimitive + CubeElement, )+
            Op: UnaryOp<Input, Output = ($( $out_ty, )+)>,
        {
            fn run_logical3(
                policy: &CubePolicy<R>,
                bindings: KernelColumnBindings,
                len: usize,
            ) -> Result<<Self as MItemStorage<R>>::Storage, Error> {
                let client = policy.client();
                $(
                    let $out_handle = client.empty(len * std::mem::size_of::<$out_ty>());
                )+
                if len != 0 {
                    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
                    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
                    let offsets = bindings.slot_offsets_handle(client)?;
                    let slot0 = bindings.slot_or_first(0);
                    let slot1 = bindings.slot_or_first(1);
                    let slot2 = bindings.slot_or_first(2);
                    let block_size = 256_u32;
                    let block_count = len.div_ceil(block_size as usize);
                    let block_count_u32 = u32::try_from(block_count)
                        .map_err(|_| Error::LengthTooLarge { len: block_count })?;
                    unsafe {
                        $kernel::launch_unchecked::<
                            Input,
                            LeafA,
                            LeafB,
                            LeafC,
                            Expr,
                            $( $out_ty, )+
                            Op,
                            R,
                        >(
                            client,
                            crate::detail::launch::cube_count_1d(block_count_u32),
                            CubeDim::new_1d(block_size),
                            BufferArg::from_raw_parts(slot0.0.clone(), slot0.1),
                            BufferArg::from_raw_parts(slot1.0.clone(), slot1.1),
                            BufferArg::from_raw_parts(slot2.0.clone(), slot2.1),
                            BufferArg::from_raw_parts(offsets.clone(), 4),
                            BufferArg::from_raw_parts(len_handle.clone(), 1),
                            $(
                                BufferArg::from_raw_parts($out_handle.clone(), len),
                            )+
                        );
                    }
                }
                Ok($storage(policy, len, $( $out_handle ),+))
            }
        }
    };
}

impl_transform_logical3_wide_output!(
    transform_logical3_to_tuple4_kernel,
    (OutA: output_a, OutB: output_b, OutC: output_c, OutD: output_d),
    logical7_tuple4_storage
);
impl_transform_logical3_wide_output!(
    transform_logical3_to_tuple5_kernel,
    (OutA: output_a, OutB: output_b, OutC: output_c, OutD: output_d, OutE: output_e),
    logical7_tuple5_storage
);
impl_transform_logical3_wide_output!(
    transform_logical3_to_tuple6_kernel,
    (OutA: output_a, OutB: output_b, OutC: output_c, OutD: output_d, OutE: output_e, OutF: output_f),
    logical7_tuple6_storage
);
impl_transform_logical3_wide_output!(
    transform_logical3_to_tuple7_kernel,
    (OutA: output_a, OutB: output_b, OutC: output_c, OutD: output_d, OutE: output_e, OutF: output_f, OutG: output_g),
    logical7_tuple7_storage
);

#[doc(hidden)]
pub trait TransformLogical7Output<
    R,
    Input,
    Leaf0,
    Leaf1,
    Leaf2,
    Leaf3,
    Leaf4,
    Leaf5,
    Leaf6,
    Leaf7,
    Expr,
    Op,
>: MItemStorage<R> where
    R: Runtime,
    Input: CubeType + 'static + Send + Sync,
    Leaf0: CubePrimitive + CubeElement,
    Leaf1: CubePrimitive + CubeElement,
    Leaf2: CubePrimitive + CubeElement,
    Leaf3: CubePrimitive + CubeElement,
    Leaf4: CubePrimitive + CubeElement,
    Leaf5: CubePrimitive + CubeElement,
    Leaf6: CubePrimitive + CubeElement,
    Leaf7: CubePrimitive + CubeElement,
    Expr: LogicalDeviceExpr7<Input, Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6, Leaf7>,
    Op: UnaryOp<Input, Output = Self>,
{
    fn run_logical7(
        policy: &CubePolicy<R>,
        bindings: KernelColumnBindings,
        len: usize,
    ) -> Result<<Self as MItemStorage<R>>::Storage, Error>;
}

macro_rules! impl_transform_logical7_output {
    (
        $kernel:ident,
        ($( $out_ty:ident : $out_handle:ident ),+),
        $storage:expr
    ) => {
        impl<R, Input, Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6, Leaf7, Expr, $( $out_ty, )+ Op>
            TransformLogical7Output<
                R,
                Input,
                Leaf0,
                Leaf1,
                Leaf2,
                Leaf3,
                Leaf4,
                Leaf5,
                Leaf6,
                Leaf7,
                Expr,
                Op,
            > for ($( $out_ty, )+)
        where
            R: Runtime,
            Input: CubeType + 'static + Send + Sync,
            Leaf0: CubePrimitive + CubeElement,
            Leaf1: CubePrimitive + CubeElement,
            Leaf2: CubePrimitive + CubeElement,
            Leaf3: CubePrimitive + CubeElement,
            Leaf4: CubePrimitive + CubeElement,
            Leaf5: CubePrimitive + CubeElement,
            Leaf6: CubePrimitive + CubeElement,
            Leaf7: CubePrimitive + CubeElement,
            Expr: LogicalDeviceExpr7<Input, Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6, Leaf7>,
            $( $out_ty: CubePrimitive + CubeElement, )+
            Op: UnaryOp<Input, Output = ($( $out_ty, )+)>,
        {
            fn run_logical7(
                policy: &CubePolicy<R>,
                bindings: KernelColumnBindings,
                len: usize,
            ) -> Result<<Self as MItemStorage<R>>::Storage, Error> {
                let client = policy.client();
                $(
                    let $out_handle = client.empty(len * std::mem::size_of::<$out_ty>());
                )+
                if len != 0 {
                    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
                    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
                    let offsets = bindings.slot_offsets8_handle(client)?;
                    let slot0 = bindings.slot_or_first(0);
                    let slot1 = bindings.slot_or_first(1);
                    let slot2 = bindings.slot_or_first(2);
                    let slot3 = bindings.slot_or_first(3);
                    let slot4 = bindings.slot_or_first(4);
                    let slot5 = bindings.slot_or_first(5);
                    let slot6 = bindings.slot_or_first(6);
                    let slot7 = bindings.slot_or_first(7);
                    let block_size = 256_u32;
                    let block_count = len.div_ceil(block_size as usize);
                    let block_count_u32 = u32::try_from(block_count)
                        .map_err(|_| Error::LengthTooLarge { len: block_count })?;
                    unsafe {
                        $kernel::launch_unchecked::<
                            Input,
                            Leaf0,
                            Leaf1,
                            Leaf2,
                            Leaf3,
                            Leaf4,
                            Leaf5,
                            Leaf6,
                            Leaf7,
                            Expr,
                            $( $out_ty, )+
                            Op,
                            R,
                        >(
                            client,
                            crate::detail::launch::cube_count_1d(block_count_u32),
                            CubeDim::new_1d(block_size),
                            BufferArg::from_raw_parts(slot0.0.clone(), slot0.1),
                            BufferArg::from_raw_parts(slot1.0.clone(), slot1.1),
                            BufferArg::from_raw_parts(slot2.0.clone(), slot2.1),
                            BufferArg::from_raw_parts(slot3.0.clone(), slot3.1),
                            BufferArg::from_raw_parts(slot4.0.clone(), slot4.1),
                            BufferArg::from_raw_parts(slot5.0.clone(), slot5.1),
                            BufferArg::from_raw_parts(slot6.0.clone(), slot6.1),
                            BufferArg::from_raw_parts(slot7.0.clone(), slot7.1),
                            BufferArg::from_raw_parts(offsets.clone(), 8),
                            BufferArg::from_raw_parts(len_handle.clone(), 1),
                            $(
                                BufferArg::from_raw_parts($out_handle.clone(), len),
                            )+
                        );
                    }
                }
                Ok($storage(policy, len, $( $out_handle ),+))
            }
        }
    };
}

fn logical7_zip1_storage<R: Runtime, A>(
    policy: &CubePolicy<R>,
    len: usize,
    output_a: cubecl::server::Handle,
) -> Zip1<DeviceVec<R, A>>
where
    A: CubePrimitive + CubeElement,
{
    Zip1 {
        source: DeviceVec::from_handle(policy.id(), output_a, len),
    }
}

fn logical7_zip2_storage<R: Runtime, A, B>(
    policy: &CubePolicy<R>,
    len: usize,
    output_a: cubecl::server::Handle,
    output_b: cubecl::server::Handle,
) -> Zip2<DeviceVec<R, A>, DeviceVec<R, B>>
where
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
{
    Zip2 {
        left: DeviceVec::from_handle(policy.id(), output_a, len),
        right: DeviceVec::from_handle(policy.id(), output_b, len),
    }
}

fn logical7_zip3_storage<R: Runtime, A, B, C>(
    policy: &CubePolicy<R>,
    len: usize,
    output_a: cubecl::server::Handle,
    output_b: cubecl::server::Handle,
    output_c: cubecl::server::Handle,
) -> Zip3<DeviceVec<R, A>, DeviceVec<R, B>, DeviceVec<R, C>>
where
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    C: CubePrimitive + CubeElement,
{
    Zip3 {
        first: DeviceVec::from_handle(policy.id(), output_a, len),
        second: DeviceVec::from_handle(policy.id(), output_b, len),
        third: DeviceVec::from_handle(policy.id(), output_c, len),
    }
}

macro_rules! define_logical7_tuple_storage {
    ($name:ident, $( $ty:ident : $handle:ident ),+) => {
        fn $name<R: Runtime, $( $ty ),+>(
            policy: &CubePolicy<R>,
            len: usize,
            $( $handle: cubecl::server::Handle ),+
        ) -> ($( DeviceVec<R, $ty>, )+)
        where
            $( $ty: CubePrimitive + CubeElement, )+
        {
            ($( DeviceVec::from_handle(policy.id(), $handle, len), )+)
        }
    };
}

define_logical7_tuple_storage!(logical7_tuple4_storage, A: output_a, B: output_b, C: output_c, D: output_d);
define_logical7_tuple_storage!(logical7_tuple5_storage, A: output_a, B: output_b, C: output_c, D: output_d, E: output_e);
define_logical7_tuple_storage!(logical7_tuple6_storage, A: output_a, B: output_b, C: output_c, D: output_d, E: output_e, F: output_f);
define_logical7_tuple_storage!(logical7_tuple7_storage, A: output_a, B: output_b, C: output_c, D: output_d, E: output_e, F: output_f, G: output_g);

impl_transform_logical7_output!(
    transform_logical7_to_tuple1_kernel,
    (OutA: output_a),
    logical7_zip1_storage
);
impl_transform_logical7_output!(
    transform_logical7_to_tuple2_kernel,
    (OutA: output_a, OutB: output_b),
    logical7_zip2_storage
);
impl_transform_logical7_output!(
    transform_logical7_to_tuple3_kernel,
    (OutA: output_a, OutB: output_b, OutC: output_c),
    logical7_zip3_storage
);
impl_transform_logical7_output!(
    transform_logical7_to_tuple4_kernel,
    (OutA: output_a, OutB: output_b, OutC: output_c, OutD: output_d),
    logical7_tuple4_storage
);
impl_transform_logical7_output!(
    transform_logical7_to_tuple5_kernel,
    (OutA: output_a, OutB: output_b, OutC: output_c, OutD: output_d, OutE: output_e),
    logical7_tuple5_storage
);
impl_transform_logical7_output!(
    transform_logical7_to_tuple6_kernel,
    (OutA: output_a, OutB: output_b, OutC: output_c, OutD: output_d, OutE: output_e, OutF: output_f),
    logical7_tuple6_storage
);
impl_transform_logical7_output!(
    transform_logical7_to_tuple7_kernel,
    (OutA: output_a, OutB: output_b, OutC: output_c, OutD: output_d, OutE: output_e, OutF: output_f, OutG: output_g),
    logical7_tuple7_storage
);

#[doc(hidden)]
pub trait SelectedLogical7Output<R, Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6, Leaf7, Expr>:
    MItemStorage<R> + Sized
where
    R: Runtime,
    Self: CubeType + 'static + Send + Sync,
    Leaf0: CubePrimitive + CubeElement,
    Leaf1: CubePrimitive + CubeElement,
    Leaf2: CubePrimitive + CubeElement,
    Leaf3: CubePrimitive + CubeElement,
    Leaf4: CubePrimitive + CubeElement,
    Leaf5: CubePrimitive + CubeElement,
    Leaf6: CubePrimitive + CubeElement,
    Leaf7: CubePrimitive + CubeElement,
    Expr: LogicalDeviceExpr7<Self, Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6, Leaf7>,
{
    fn run_selected_logical7(
        policy: &CubePolicy<R>,
        bindings: KernelColumnBindings,
        selected_rank: &crate::detail::control::SelectedRankControl,
        count: usize,
    ) -> Result<<Self as MItemStorage<R>>::Storage, Error>;
}

macro_rules! impl_selected_logical7_output {
    (
        $kernel:ident,
        ($( $out_ty:ident : $out_handle:ident ),+),
        $storage:expr
    ) => {
        impl<R, Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6, Leaf7, Expr, $( $out_ty, )+>
            SelectedLogical7Output<R, Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6, Leaf7, Expr>
            for ($( $out_ty, )+)
        where
            R: Runtime,
            Leaf0: CubePrimitive + CubeElement,
            Leaf1: CubePrimitive + CubeElement,
            Leaf2: CubePrimitive + CubeElement,
            Leaf3: CubePrimitive + CubeElement,
            Leaf4: CubePrimitive + CubeElement,
            Leaf5: CubePrimitive + CubeElement,
            Leaf6: CubePrimitive + CubeElement,
            Leaf7: CubePrimitive + CubeElement,
            Expr: LogicalDeviceExpr7<($( $out_ty, )+), Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6, Leaf7>,
            $( $out_ty: CubePrimitive + CubeElement, )+
        {
            fn run_selected_logical7(
                policy: &CubePolicy<R>,
                bindings: KernelColumnBindings,
                selected_rank: &crate::detail::control::SelectedRankControl,
                count: usize,
            ) -> Result<<Self as MItemStorage<R>>::Storage, Error> {
                let client = policy.client();
                $(
                    let $out_handle = if count == 0 {
                        policy.empty_handle()
                    } else {
                        client.empty(count * std::mem::size_of::<$out_ty>())
                    };
                )+
                if selected_rank.len != 0 && count != 0 {
                    let offsets = bindings.slot_offsets8_handle(client)?;
                    let slot0 = bindings.slot_or_first(0);
                    let slot1 = bindings.slot_or_first(1);
                    let slot2 = bindings.slot_or_first(2);
                    let slot3 = bindings.slot_or_first(3);
                    let slot4 = bindings.slot_or_first(4);
                    let slot5 = bindings.slot_or_first(5);
                    let slot6 = bindings.slot_or_first(6);
                    let slot7 = bindings.slot_or_first(7);
                    let block_size = 256_u32;
                    let block_count = selected_rank.len.div_ceil(block_size as usize);
                    let block_count_u32 = u32::try_from(block_count)
                        .map_err(|_| Error::LengthTooLarge { len: block_count })?;
                    unsafe {
                        $kernel::launch_unchecked::<
                            Leaf0,
                            Leaf1,
                            Leaf2,
                            Leaf3,
                            Leaf4,
                            Leaf5,
                            Leaf6,
                            Leaf7,
                            Expr,
                            $( $out_ty, )+
                            R,
                        >(
                            client,
                            crate::detail::launch::cube_count_1d(block_count_u32),
                            CubeDim::new_1d(block_size),
                            BufferArg::from_raw_parts(selected_rank.flag.clone(), selected_rank.len),
                            BufferArg::from_raw_parts(selected_rank.position.clone(), selected_rank.len),
                            BufferArg::from_raw_parts(slot0.0.clone(), slot0.1),
                            BufferArg::from_raw_parts(slot1.0.clone(), slot1.1),
                            BufferArg::from_raw_parts(slot2.0.clone(), slot2.1),
                            BufferArg::from_raw_parts(slot3.0.clone(), slot3.1),
                            BufferArg::from_raw_parts(slot4.0.clone(), slot4.1),
                            BufferArg::from_raw_parts(slot5.0.clone(), slot5.1),
                            BufferArg::from_raw_parts(slot6.0.clone(), slot6.1),
                            BufferArg::from_raw_parts(slot7.0.clone(), slot7.1),
                            BufferArg::from_raw_parts(offsets.clone(), 8),
                            $(
                                BufferArg::from_raw_parts($out_handle.clone(), count),
                            )+
                        );
                    }
                }
                Ok($storage(policy, count, $( $out_handle ),+))
            }
        }
    };
}

impl_selected_logical7_output!(
    selected_logical7_to_tuple1_kernel,
    (OutA: output_a),
    logical7_zip1_storage
);
impl_selected_logical7_output!(
    selected_logical7_to_tuple2_kernel,
    (OutA: output_a, OutB: output_b),
    logical7_zip2_storage
);
impl_selected_logical7_output!(
    selected_logical7_to_tuple3_kernel,
    (OutA: output_a, OutB: output_b, OutC: output_c),
    logical7_zip3_storage
);
impl_selected_logical7_output!(
    selected_logical7_to_tuple4_kernel,
    (OutA: output_a, OutB: output_b, OutC: output_c, OutD: output_d),
    logical7_tuple4_storage
);
impl_selected_logical7_output!(
    selected_logical7_to_tuple5_kernel,
    (OutA: output_a, OutB: output_b, OutC: output_c, OutD: output_d, OutE: output_e),
    logical7_tuple5_storage
);
impl_selected_logical7_output!(
    selected_logical7_to_tuple6_kernel,
    (OutA: output_a, OutB: output_b, OutC: output_c, OutD: output_d, OutE: output_e, OutF: output_f),
    logical7_tuple6_storage
);
impl_selected_logical7_output!(
    selected_logical7_to_tuple7_kernel,
    (OutA: output_a, OutB: output_b, OutC: output_c, OutD: output_d, OutE: output_e, OutF: output_f, OutG: output_g),
    logical7_tuple7_storage
);

#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub trait GatherLogical7Output<
    R,
    Leaf0,
    Leaf1,
    Leaf2,
    Leaf3,
    Leaf4,
    Leaf5,
    Leaf6,
    Leaf7,
    ValueExpr,
    IndexLeaf0,
    IndexLeaf1,
    IndexLeaf2,
    IndexLeaf3,
    IndexLeaf4,
    IndexLeaf5,
    IndexLeaf6,
    IndexLeaf7,
    IndexExpr,
>: MItemStorage<R> + Sized where
    R: Runtime,
    Self: CubeType + 'static + Send + Sync,
    Leaf0: CubePrimitive + CubeElement,
    Leaf1: CubePrimitive + CubeElement,
    Leaf2: CubePrimitive + CubeElement,
    Leaf3: CubePrimitive + CubeElement,
    Leaf4: CubePrimitive + CubeElement,
    Leaf5: CubePrimitive + CubeElement,
    Leaf6: CubePrimitive + CubeElement,
    Leaf7: CubePrimitive + CubeElement,
    IndexLeaf0: CubePrimitive + CubeElement,
    IndexLeaf1: CubePrimitive + CubeElement,
    IndexLeaf2: CubePrimitive + CubeElement,
    IndexLeaf3: CubePrimitive + CubeElement,
    IndexLeaf4: CubePrimitive + CubeElement,
    IndexLeaf5: CubePrimitive + CubeElement,
    IndexLeaf6: CubePrimitive + CubeElement,
    IndexLeaf7: CubePrimitive + CubeElement,
    ValueExpr: LogicalDeviceExpr7<Self, Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6, Leaf7>,
    IndexExpr: LogicalDeviceExpr7<
            crate::index::MIndex,
            IndexLeaf0,
            IndexLeaf1,
            IndexLeaf2,
            IndexLeaf3,
            IndexLeaf4,
            IndexLeaf5,
            IndexLeaf6,
            IndexLeaf7,
        >,
{
    fn run_gather_logical7(
        policy: &CubePolicy<R>,
        value_bindings: KernelColumnBindings,
        index_bindings: KernelColumnBindings,
        len: usize,
    ) -> Result<<Self as MItemStorage<R>>::Storage, Error>;
}

macro_rules! impl_gather_logical7_output {
    (
        $kernel:ident,
        ($( $out_ty:ident : $out_handle:ident ),+),
        $storage:expr
    ) => {
        impl<
                R,
                Leaf0,
                Leaf1,
                Leaf2,
                Leaf3,
                Leaf4,
                Leaf5,
                Leaf6,
                Leaf7,
                ValueExpr,
                IndexLeaf0,
                IndexLeaf1,
                IndexLeaf2,
                IndexLeaf3,
                IndexLeaf4,
                IndexLeaf5,
                IndexLeaf6,
                IndexLeaf7,
                IndexExpr,
                $( $out_ty, )+
            >
            GatherLogical7Output<
                R,
                Leaf0,
                Leaf1,
                Leaf2,
                Leaf3,
                Leaf4,
                Leaf5,
                Leaf6,
                Leaf7,
                ValueExpr,
                IndexLeaf0,
                IndexLeaf1,
                IndexLeaf2,
                IndexLeaf3,
                IndexLeaf4,
                IndexLeaf5,
                IndexLeaf6,
                IndexLeaf7,
                IndexExpr,
            > for ($( $out_ty, )+)
        where
            R: Runtime,
            Leaf0: CubePrimitive + CubeElement,
            Leaf1: CubePrimitive + CubeElement,
            Leaf2: CubePrimitive + CubeElement,
            Leaf3: CubePrimitive + CubeElement,
            Leaf4: CubePrimitive + CubeElement,
            Leaf5: CubePrimitive + CubeElement,
            Leaf6: CubePrimitive + CubeElement,
            Leaf7: CubePrimitive + CubeElement,
            IndexLeaf0: CubePrimitive + CubeElement,
            IndexLeaf1: CubePrimitive + CubeElement,
            IndexLeaf2: CubePrimitive + CubeElement,
            IndexLeaf3: CubePrimitive + CubeElement,
            IndexLeaf4: CubePrimitive + CubeElement,
            IndexLeaf5: CubePrimitive + CubeElement,
            IndexLeaf6: CubePrimitive + CubeElement,
            IndexLeaf7: CubePrimitive + CubeElement,
            ValueExpr: LogicalDeviceExpr7<($( $out_ty, )+), Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6, Leaf7>,
            IndexExpr: LogicalDeviceExpr7<
                crate::index::MIndex,
                IndexLeaf0,
                IndexLeaf1,
                IndexLeaf2,
                IndexLeaf3,
                IndexLeaf4,
                IndexLeaf5,
                IndexLeaf6,
                IndexLeaf7,
            >,
            $( $out_ty: CubePrimitive + CubeElement, )+
        {
            fn run_gather_logical7(
                policy: &CubePolicy<R>,
                value_bindings: KernelColumnBindings,
                index_bindings: KernelColumnBindings,
                len: usize,
            ) -> Result<<Self as MItemStorage<R>>::Storage, Error> {
                let client = policy.client();
                $(
                    let $out_handle = client.empty(len * std::mem::size_of::<$out_ty>());
                )+
                if len != 0 {
                    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
                    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
                    let value_offsets = value_bindings.slot_offsets8_handle(client)?;
                    let index_offsets = index_bindings.slot_offsets8_handle(client)?;
                    let value_slot0 = value_bindings.slot_or_first(0);
                    let value_slot1 = value_bindings.slot_or_first(1);
                    let value_slot2 = value_bindings.slot_or_first(2);
                    let value_slot3 = value_bindings.slot_or_first(3);
                    let value_slot4 = value_bindings.slot_or_first(4);
                    let value_slot5 = value_bindings.slot_or_first(5);
                    let value_slot6 = value_bindings.slot_or_first(6);
                    let value_slot7 = value_bindings.slot_or_first(7);
                    let index_slot0 = index_bindings.slot_or_first(0);
                    let index_slot1 = index_bindings.slot_or_first(1);
                    let index_slot2 = index_bindings.slot_or_first(2);
                    let index_slot3 = index_bindings.slot_or_first(3);
                    let index_slot4 = index_bindings.slot_or_first(4);
                    let index_slot5 = index_bindings.slot_or_first(5);
                    let index_slot6 = index_bindings.slot_or_first(6);
                    let index_slot7 = index_bindings.slot_or_first(7);
                    let block_size = 256_u32;
                    let block_count = len.div_ceil(block_size as usize);
                    let block_count_u32 = u32::try_from(block_count)
                        .map_err(|_| Error::LengthTooLarge { len: block_count })?;
                    unsafe {
                        $kernel::launch_unchecked::<
                            Leaf0,
                            Leaf1,
                            Leaf2,
                            Leaf3,
                            Leaf4,
                            Leaf5,
                            Leaf6,
                            Leaf7,
                            IndexLeaf0,
                            IndexLeaf1,
                            IndexLeaf2,
                            IndexLeaf3,
                            IndexLeaf4,
                            IndexLeaf5,
                            IndexLeaf6,
                            IndexLeaf7,
                            ValueExpr,
                            IndexExpr,
                            $( $out_ty, )+
                            R,
                        >(
                            client,
                            crate::detail::launch::cube_count_1d(block_count_u32),
                            CubeDim::new_1d(block_size),
                            BufferArg::from_raw_parts(value_slot0.0.clone(), value_slot0.1),
                            BufferArg::from_raw_parts(value_slot1.0.clone(), value_slot1.1),
                            BufferArg::from_raw_parts(value_slot2.0.clone(), value_slot2.1),
                            BufferArg::from_raw_parts(value_slot3.0.clone(), value_slot3.1),
                            BufferArg::from_raw_parts(value_slot4.0.clone(), value_slot4.1),
                            BufferArg::from_raw_parts(value_slot5.0.clone(), value_slot5.1),
                            BufferArg::from_raw_parts(value_slot6.0.clone(), value_slot6.1),
                            BufferArg::from_raw_parts(value_slot7.0.clone(), value_slot7.1),
                            BufferArg::from_raw_parts(value_offsets.clone(), 8),
                            BufferArg::from_raw_parts(index_slot0.0.clone(), index_slot0.1),
                            BufferArg::from_raw_parts(index_slot1.0.clone(), index_slot1.1),
                            BufferArg::from_raw_parts(index_slot2.0.clone(), index_slot2.1),
                            BufferArg::from_raw_parts(index_slot3.0.clone(), index_slot3.1),
                            BufferArg::from_raw_parts(index_slot4.0.clone(), index_slot4.1),
                            BufferArg::from_raw_parts(index_slot5.0.clone(), index_slot5.1),
                            BufferArg::from_raw_parts(index_slot6.0.clone(), index_slot6.1),
                            BufferArg::from_raw_parts(index_slot7.0.clone(), index_slot7.1),
                            BufferArg::from_raw_parts(index_offsets.clone(), 8),
                            BufferArg::from_raw_parts(len_handle.clone(), 1),
                            $(
                                BufferArg::from_raw_parts($out_handle.clone(), len),
                            )+
                        );
                    }
                }
                Ok($storage(policy, len, $( $out_handle ),+))
            }
        }
    };
}

impl_gather_logical7_output!(
    gather_logical7_to_tuple1_kernel,
    (OutA: output_a),
    logical7_zip1_storage
);
impl_gather_logical7_output!(
    gather_logical7_to_tuple2_kernel,
    (OutA: output_a, OutB: output_b),
    logical7_zip2_storage
);
impl_gather_logical7_output!(
    gather_logical7_to_tuple3_kernel,
    (OutA: output_a, OutB: output_b, OutC: output_c),
    logical7_zip3_storage
);
impl_gather_logical7_output!(
    gather_logical7_to_tuple4_kernel,
    (OutA: output_a, OutB: output_b, OutC: output_c, OutD: output_d),
    logical7_tuple4_storage
);
impl_gather_logical7_output!(
    gather_logical7_to_tuple5_kernel,
    (OutA: output_a, OutB: output_b, OutC: output_c, OutD: output_d, OutE: output_e),
    logical7_tuple5_storage
);
impl_gather_logical7_output!(
    gather_logical7_to_tuple6_kernel,
    (OutA: output_a, OutB: output_b, OutC: output_c, OutD: output_d, OutE: output_e, OutF: output_f),
    logical7_tuple6_storage
);
impl_gather_logical7_output!(
    gather_logical7_to_tuple7_kernel,
    (OutA: output_a, OutB: output_b, OutC: output_c, OutD: output_d, OutE: output_e, OutF: output_f, OutG: output_g),
    logical7_tuple7_storage
);

macro_rules! impl_transform_tuple_output {
    (
        ($trait_name:ident < $first_in:ident : $first_arg:ident, $( $in_ty:ident : $arg:ident ),+ >),
        $kernel:ident,
        $zip:ident,
        ($( $out_ty:ident : $out_handle:ident : $out_field:ident ),+)
    ) => {
        impl<R, $first_in, $( $in_ty, )+ $( $out_ty, )+ Op>
            $trait_name<R, $first_in, $( $in_ty, )+ Op> for ($( $out_ty, )+)
        where
            R: Runtime,
            $first_in: CubePrimitive + CubeElement,
            $( $in_ty: CubePrimitive + CubeElement, )+
            $( $out_ty: CubePrimitive + CubeElement, )+
            Op: UnaryOp<($first_in, $( $in_ty, )+), Output = ($( $out_ty, )+)>,
        {
            fn run(
                policy: &crate::policy::CubePolicy<R>,
                $first_arg: DeviceColumnView<R, $first_in>,
                $( $arg: DeviceColumnView<R, $in_ty>, )+
            ) -> Result<<Self as MItemStorage<R>>::Storage, Error> {
                let len = $first_arg.len();
                let client = policy.client();
                $(
                    let $out_handle = client.empty(len * std::mem::size_of::<$out_ty>());
                )+
                if len != 0 {
                    let len_u32 = u32::try_from(len)
                        .map_err(|_| Error::LengthTooLarge { len })?;
                    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
                    let block_size = 256_u32;
                    let block_count = len.div_ceil(block_size as usize);
                    let block_count_u32 = u32::try_from(block_count)
                        .map_err(|_| Error::LengthTooLarge { len: block_count })?;
                    unsafe {
                        $kernel::launch_unchecked::<
                            $first_in, $( $in_ty, )+ $( $out_ty, )+ Op, R,
                        >(
                            client,
                            crate::detail::launch::cube_count_1d(block_count_u32),
                            CubeDim::new_1d(block_size),
                            unsafe { BufferArg::from_raw_parts($first_arg.source.handle.clone(), $first_arg.source.len()) },
                            $(
                                unsafe { BufferArg::from_raw_parts($arg.source.handle.clone(), $arg.source.len()) },
                            )+
                            unsafe {
                                BufferArg::from_raw_parts({
                                    let offset = u32::try_from($first_arg.offset)
                                        .map_err(|_| Error::LengthTooLarge { len: $first_arg.offset })?;
                                    client.create_from_slice(u32::as_bytes(&[offset]))
                                }, 1)
                            },
                            $(
                                unsafe {
                                    BufferArg::from_raw_parts({
                                        let offset = u32::try_from($arg.offset)
                                            .map_err(|_| Error::LengthTooLarge { len: $arg.offset })?;
                                        client.create_from_slice(u32::as_bytes(&[offset]))
                                    }, 1)
                                },
                            )+
                            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
                            $(
                                unsafe { BufferArg::from_raw_parts($out_handle.clone(), len) },
                            )+
                        );
                    }
                }
                Ok($zip {
                    $(
                        $out_field: DeviceVec::from_handle(policy.id(), $out_handle, len),
                    )+
                })
            }
        }
    };
}

macro_rules! impl_transform_tuple_output_arity {
    ($input:tt, 1, $kernel:ident) => {
        impl_transform_tuple_output!($input, $kernel, Zip1, (OutA: out_a: source));
    };
    ($input:tt, 2, $kernel:ident) => {
        impl_transform_tuple_output!(
            $input,
            $kernel,
            Zip2,
            (OutA: out_a: left, OutB: out_b: right)
        );
    };
    ($input:tt, 3, $kernel:ident) => {
        impl_transform_tuple_output!(
            $input,
            $kernel,
            Zip3,
            (OutA: out_a: first, OutB: out_b: second, OutC: out_c: third)
        );
    };
}

macro_rules! impl_transform_tuple_outputs {
    (
        $trait_name:ident < $first_in:ident : $first_arg:ident, $( $in_ty:ident : $arg:ident ),+ >,
        $( $arity:tt => $kernel:ident ),+ $(,)?
    ) => {
        impl_transform_tuple_outputs!(
            @inner
            ($trait_name < $first_in : $first_arg, $( $in_ty : $arg ),+ >),
            $( $arity => $kernel ),+
        );
    };
    (
        @inner
        $input:tt,
        $( $arity:tt => $kernel:ident ),+ $(,)?
    ) => {
        $(
            impl_transform_tuple_output_arity!(
                $input,
                $arity,
                $kernel
            );
        )+
    };
}

#[doc(hidden)]
pub trait TransformZip2Output<R, InA, InB, Op>: CubeType + MItemStorage<R>
where
    R: Runtime,
    InA: CubePrimitive + CubeElement,
    InB: CubePrimitive + CubeElement,
    Op: UnaryOp<(InA, InB), Output = Self>,
{
    fn run(
        policy: &crate::policy::CubePolicy<R>,
        left: DeviceColumnView<R, InA>,
        right: DeviceColumnView<R, InB>,
    ) -> Result<<Self as MItemStorage<R>>::Storage, Error>;
}

impl<R, InA, InB, OutA, Op> TransformZip2Output<R, InA, InB, Op> for (OutA,)
where
    R: Runtime,
    InA: CubePrimitive + CubeElement,
    InB: CubePrimitive + CubeElement,
    OutA: CubePrimitive + CubeElement,
    Op: UnaryOp<(InA, InB), Output = (OutA,)>,
{
    fn run(
        policy: &crate::policy::CubePolicy<R>,
        left: DeviceColumnView<R, InA>,
        right: DeviceColumnView<R, InB>,
    ) -> Result<<Self as MItemStorage<R>>::Storage, Error> {
        let len = left.len();
        let client = policy.client();
        let output_a = client.empty(len * std::mem::size_of::<OutA>());
        if len != 0 {
            let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
            let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
            let left_offset = transform_offset_handle(client, left.offset)?;
            let right_offset = transform_offset_handle(client, right.offset)?;
            let block_size = 256_u32;
            let block_count = len.div_ceil(block_size as usize);
            let block_count_u32 = u32::try_from(block_count)
                .map_err(|_| Error::LengthTooLarge { len: block_count })?;
            unsafe {
                transform_tuple2_to_tuple1_kernel::launch_unchecked::<InA, InB, OutA, Op, R>(
                    client,
                    crate::detail::launch::cube_count_1d(block_count_u32),
                    CubeDim::new_1d(block_size),
                    unsafe {
                        BufferArg::from_raw_parts(left.source.handle.clone(), left.source.len())
                    },
                    unsafe {
                        BufferArg::from_raw_parts(right.source.handle.clone(), right.source.len())
                    },
                    unsafe { BufferArg::from_raw_parts(left_offset.clone(), 1) },
                    unsafe { BufferArg::from_raw_parts(right_offset.clone(), 1) },
                    unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
                    unsafe { BufferArg::from_raw_parts(output_a.clone(), len) },
                );
            }
        }
        Ok(Zip1 {
            source: DeviceVec::from_handle(policy.id(), output_a, len),
        })
    }
}

impl<R, InA, InB, OutA, OutB, Op> TransformZip2Output<R, InA, InB, Op> for (OutA, OutB)
where
    R: Runtime,
    InA: CubePrimitive + CubeElement,
    InB: CubePrimitive + CubeElement,
    OutA: CubePrimitive + CubeElement,
    OutB: CubePrimitive + CubeElement,
    Op: UnaryOp<(InA, InB), Output = (OutA, OutB)>,
{
    fn run(
        policy: &crate::policy::CubePolicy<R>,
        left: DeviceColumnView<R, InA>,
        right: DeviceColumnView<R, InB>,
    ) -> Result<<Self as MItemStorage<R>>::Storage, Error> {
        let len = left.len();
        let client = policy.client();
        let output_a = client.empty(len * std::mem::size_of::<OutA>());
        let output_b = client.empty(len * std::mem::size_of::<OutB>());
        if len != 0 {
            let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
            let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
            let left_offset = transform_offset_handle(client, left.offset)?;
            let right_offset = transform_offset_handle(client, right.offset)?;
            let block_size = 256_u32;
            let block_count = len.div_ceil(block_size as usize);
            let block_count_u32 = u32::try_from(block_count)
                .map_err(|_| Error::LengthTooLarge { len: block_count })?;
            unsafe {
                transform_tuple2_to_tuple2_kernel::launch_unchecked::<InA, InB, OutA, OutB, Op, R>(
                    client,
                    crate::detail::launch::cube_count_1d(block_count_u32),
                    CubeDim::new_1d(block_size),
                    unsafe {
                        BufferArg::from_raw_parts(left.source.handle.clone(), left.source.len())
                    },
                    unsafe {
                        BufferArg::from_raw_parts(right.source.handle.clone(), right.source.len())
                    },
                    unsafe { BufferArg::from_raw_parts(left_offset.clone(), 1) },
                    unsafe { BufferArg::from_raw_parts(right_offset.clone(), 1) },
                    unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
                    unsafe { BufferArg::from_raw_parts(output_a.clone(), len) },
                    unsafe { BufferArg::from_raw_parts(output_b.clone(), len) },
                );
            }
        }
        Ok(Zip2 {
            left: DeviceVec::from_handle(policy.id(), output_a, len),
            right: DeviceVec::from_handle(policy.id(), output_b, len),
        })
    }
}

#[doc(hidden)]
pub trait TransformZip3Output<R, InA, InB, InC, Op>: CubeType + MItemStorage<R>
where
    R: Runtime,
    InA: CubePrimitive + CubeElement,
    InB: CubePrimitive + CubeElement,
    InC: CubePrimitive + CubeElement,
    Op: UnaryOp<(InA, InB, InC), Output = Self>,
{
    fn run(
        policy: &crate::policy::CubePolicy<R>,
        first: DeviceColumnView<R, InA>,
        second: DeviceColumnView<R, InB>,
        third: DeviceColumnView<R, InC>,
    ) -> Result<<Self as MItemStorage<R>>::Storage, Error>;
}

impl<R, InA, InB, InC, OutA, Op> TransformZip3Output<R, InA, InB, InC, Op> for (OutA,)
where
    R: Runtime,
    InA: CubePrimitive + CubeElement,
    InB: CubePrimitive + CubeElement,
    InC: CubePrimitive + CubeElement,
    OutA: CubePrimitive + CubeElement,
    Op: UnaryOp<(InA, InB, InC), Output = (OutA,)>,
{
    fn run(
        policy: &crate::policy::CubePolicy<R>,
        first: DeviceColumnView<R, InA>,
        second: DeviceColumnView<R, InB>,
        third: DeviceColumnView<R, InC>,
    ) -> Result<<Self as MItemStorage<R>>::Storage, Error> {
        let len = first.len();
        let client = policy.client();
        let output_a = client.empty(len * std::mem::size_of::<OutA>());
        if len != 0 {
            let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
            let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
            let first_offset = transform_offset_handle(client, first.offset)?;
            let second_offset = transform_offset_handle(client, second.offset)?;
            let third_offset = transform_offset_handle(client, third.offset)?;
            let block_size = 256_u32;
            let block_count = len.div_ceil(block_size as usize);
            let block_count_u32 = u32::try_from(block_count)
                .map_err(|_| Error::LengthTooLarge { len: block_count })?;
            unsafe {
                transform_tuple3_to_tuple1_kernel::launch_unchecked::<InA, InB, InC, OutA, Op, R>(
                    client,
                    crate::detail::launch::cube_count_1d(block_count_u32),
                    CubeDim::new_1d(block_size),
                    unsafe {
                        BufferArg::from_raw_parts(first.source.handle.clone(), first.source.len())
                    },
                    unsafe {
                        BufferArg::from_raw_parts(second.source.handle.clone(), second.source.len())
                    },
                    unsafe {
                        BufferArg::from_raw_parts(third.source.handle.clone(), third.source.len())
                    },
                    unsafe { BufferArg::from_raw_parts(first_offset.clone(), 1) },
                    unsafe { BufferArg::from_raw_parts(second_offset.clone(), 1) },
                    unsafe { BufferArg::from_raw_parts(third_offset.clone(), 1) },
                    unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
                    unsafe { BufferArg::from_raw_parts(output_a.clone(), len) },
                );
            }
        }
        Ok(Zip1 {
            source: DeviceVec::from_handle(policy.id(), output_a, len),
        })
    }
}

impl<R, InA, InB, InC, OutA, OutB, OutC, Op> TransformZip3Output<R, InA, InB, InC, Op>
    for (OutA, OutB, OutC)
where
    R: Runtime,
    InA: CubePrimitive + CubeElement,
    InB: CubePrimitive + CubeElement,
    InC: CubePrimitive + CubeElement,
    OutA: CubePrimitive + CubeElement,
    OutB: CubePrimitive + CubeElement,
    OutC: CubePrimitive + CubeElement,
    Op: UnaryOp<(InA, InB, InC), Output = (OutA, OutB, OutC)>,
{
    fn run(
        policy: &crate::policy::CubePolicy<R>,
        first: DeviceColumnView<R, InA>,
        second: DeviceColumnView<R, InB>,
        third: DeviceColumnView<R, InC>,
    ) -> Result<<Self as MItemStorage<R>>::Storage, Error> {
        let len = first.len();
        let client = policy.client();
        let output_a = client.empty(len * std::mem::size_of::<OutA>());
        let output_b = client.empty(len * std::mem::size_of::<OutB>());
        let output_c = client.empty(len * std::mem::size_of::<OutC>());
        if len != 0 {
            let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
            let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
            let first_offset = transform_offset_handle(client, first.offset)?;
            let second_offset = transform_offset_handle(client, second.offset)?;
            let third_offset = transform_offset_handle(client, third.offset)?;
            let block_size = 256_u32;
            let block_count = len.div_ceil(block_size as usize);
            let block_count_u32 = u32::try_from(block_count)
                .map_err(|_| Error::LengthTooLarge { len: block_count })?;
            unsafe {
                transform_tuple3_to_tuple3_kernel::launch_unchecked::<
                    InA,
                    InB,
                    InC,
                    OutA,
                    OutB,
                    OutC,
                    Op,
                    R,
                >(
                    client,
                    crate::detail::launch::cube_count_1d(block_count_u32),
                    CubeDim::new_1d(block_size),
                    unsafe {
                        BufferArg::from_raw_parts(first.source.handle.clone(), first.source.len())
                    },
                    unsafe {
                        BufferArg::from_raw_parts(second.source.handle.clone(), second.source.len())
                    },
                    unsafe {
                        BufferArg::from_raw_parts(third.source.handle.clone(), third.source.len())
                    },
                    unsafe { BufferArg::from_raw_parts(first_offset.clone(), 1) },
                    unsafe { BufferArg::from_raw_parts(second_offset.clone(), 1) },
                    unsafe { BufferArg::from_raw_parts(third_offset.clone(), 1) },
                    unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
                    unsafe { BufferArg::from_raw_parts(output_a.clone(), len) },
                    unsafe { BufferArg::from_raw_parts(output_b.clone(), len) },
                    unsafe { BufferArg::from_raw_parts(output_c.clone(), len) },
                );
            }
        }
        Ok(Zip3 {
            first: DeviceVec::from_handle(policy.id(), output_a, len),
            second: DeviceVec::from_handle(policy.id(), output_b, len),
            third: DeviceVec::from_handle(policy.id(), output_c, len),
        })
    }
}

impl_transform_tuple_outputs!(
    TransformZip2Output<A: a, B: b>,
    3 => transform_tuple2_to_tuple3_kernel,
);

macro_rules! define_transform_zip_output_trait {
    ($trait_name:ident < $( $in_ty:ident : $arg:ident ),+ >) => {
        #[doc(hidden)]
        pub trait $trait_name<R, $( $in_ty, )+ Op>: CubeType + MItemStorage<R>
        where
            R: Runtime,
            $( $in_ty: CubePrimitive + CubeElement, )+
            Op: UnaryOp<($( $in_ty, )+), Output = Self>,
        {
            fn run(
                policy: &crate::policy::CubePolicy<R>,
                $( $arg: DeviceColumnView<R, $in_ty>, )+
            ) -> Result<<Self as MItemStorage<R>>::Storage, Error>;
        }
    };
}

define_transform_zip_output_trait!(TransformZip4Output<InA: a, InB: b, InC: c, InD: d>);
define_transform_zip_output_trait!(TransformZip5Output<InA: a, InB: b, InC: c, InD: d, InE: e>);
define_transform_zip_output_trait!(
    TransformZip6Output<InA: a, InB: b, InC: c, InD: d, InE: e, InF: f>
);

impl_transform_tuple_outputs!(
    TransformZip4Output<A: a, B: b, C: c, D: d>,
    1 => transform_tuple4_to_tuple1_kernel,
);
impl_transform_tuple_outputs!(
    TransformZip5Output<A: a, B: b, C: c, D: d, E: e>,
    1 => transform_tuple5_to_tuple1_kernel,
);
impl_transform_tuple_outputs!(
    TransformZip6Output<A: a, B: b, C: c, D: d, E: e, F: f>,
    1 => transform_tuple6_to_tuple1_kernel,
);
impl_transform_tuple_outputs!(
    TransformZip4Output<A: a, B: b, C: c, D: d>,
    2 => transform_tuple4_to_tuple2_kernel,
    3 => transform_tuple4_to_tuple3_kernel,
);
impl_transform_tuple_outputs!(
    TransformZip5Output<A: a, B: b, C: c, D: d, E: e>,
    2 => transform_tuple5_to_tuple2_kernel,
    3 => transform_tuple5_to_tuple3_kernel,
);
impl_transform_tuple_outputs!(
    TransformZip6Output<A: a, B: b, C: c, D: d, E: e, F: f>,
    2 => transform_tuple6_to_tuple2_kernel,
    3 => transform_tuple6_to_tuple3_kernel,
);

macro_rules! impl_wide_transform_zip_output {
    (
        $trait_name:ident < $first_in:ident : $first_arg:ident $(, $in_ty:ident : $arg:ident )+ >,
        $kernel:ident,
        ($( $out_ty:ident : $out_handle:ident : $field:tt ),+)
    ) => {
        impl<R, $first_in, $( $in_ty, )+ $( $out_ty, )+ Op> $trait_name<R, $first_in, $( $in_ty, )+ Op>
            for ($( $out_ty, )+)
        where
            R: Runtime,
            $first_in: CubePrimitive + CubeElement,
            $( $in_ty: CubePrimitive + CubeElement, )+
            $( $out_ty: CubePrimitive + CubeElement, )+
            Op: UnaryOp<($first_in, $( $in_ty, )+), Output = ($( $out_ty, )+)>,
        {
            fn run(
                policy: &CubePolicy<R>,
                $first_arg: DeviceColumnView<R, $first_in>,
                $( $arg: DeviceColumnView<R, $in_ty>, )+
            ) -> Result<<Self as MItemStorage<R>>::Storage, Error> {
                let len = $first_arg.len();
                let client = policy.client();
                $(
                    let $out_handle = client.empty(len * std::mem::size_of::<$out_ty>());
                )+
                if len != 0 {
                    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
                    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
                    let $first_arg = (transform_offset_handle(client, $first_arg.offset)?, $first_arg);
                    $(
                        let $arg = (transform_offset_handle(client, $arg.offset)?, $arg);
                    )+
                    let block_size = 256_u32;
                    let block_count = len.div_ceil(block_size as usize);
                    let block_count_u32 = u32::try_from(block_count)
                        .map_err(|_| Error::LengthTooLarge { len: block_count })?;
                    unsafe {
                        $kernel::launch_unchecked::<$first_in, $( $in_ty, )+ $( $out_ty, )+ Op, R>(
                            client,
                            crate::detail::launch::cube_count_1d(block_count_u32),
                            CubeDim::new_1d(block_size),
                            BufferArg::from_raw_parts($first_arg.1.source.handle.clone(), $first_arg.1.source.len()),
                            $(
                                BufferArg::from_raw_parts($arg.1.source.handle.clone(), $arg.1.source.len()),
                            )+
                            BufferArg::from_raw_parts($first_arg.0.clone(), 1),
                            $(
                                BufferArg::from_raw_parts($arg.0.clone(), 1),
                            )+
                            BufferArg::from_raw_parts(len_handle.clone(), 1),
                            $(
                                BufferArg::from_raw_parts($out_handle.clone(), len),
                            )+
                        );
                    }
                }
                Ok(($( DeviceVec::from_handle(policy.id(), $out_handle, len), )+))
            }
        }
    };
}

impl_wide_transform_zip_output!(
    TransformZip2Output<A: a, B: b>,
    transform_tuple2_to_tuple4_kernel,
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3)
);
impl_wide_transform_zip_output!(
    TransformZip2Output<A: a, B: b>,
    transform_tuple2_to_tuple5_kernel,
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4)
);
impl_wide_transform_zip_output!(
    TransformZip2Output<A: a, B: b>,
    transform_tuple2_to_tuple6_kernel,
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4, OutF: output_f: 5)
);
impl_wide_transform_zip_output!(
    TransformZip2Output<A: a, B: b>,
    transform_tuple2_to_tuple7_kernel,
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4, OutF: output_f: 5, OutG: output_g: 6)
);
impl_transform_tuple_outputs!(
    TransformZip3Output<A: a, B: b, C: c>,
    2 => transform_tuple3_to_tuple2_kernel,
);
impl_wide_transform_zip_output!(
    TransformZip3Output<A: a, B: b, C: c>,
    transform_tuple3_to_tuple4_kernel,
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3)
);
impl_wide_transform_zip_output!(
    TransformZip3Output<A: a, B: b, C: c>,
    transform_tuple3_to_tuple5_kernel,
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4)
);
impl_wide_transform_zip_output!(
    TransformZip3Output<A: a, B: b, C: c>,
    transform_tuple3_to_tuple6_kernel,
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4, OutF: output_f: 5)
);
impl_wide_transform_zip_output!(
    TransformZip3Output<A: a, B: b, C: c>,
    transform_tuple3_to_tuple7_kernel,
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4, OutF: output_f: 5, OutG: output_g: 6)
);
impl_wide_transform_zip_output!(
    TransformZip4Output<A: a, B: b, C: c, D: d>,
    transform_tuple4_to_tuple4_kernel,
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3)
);
impl_wide_transform_zip_output!(
    TransformZip4Output<A: a, B: b, C: c, D: d>,
    transform_tuple4_to_tuple5_kernel,
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4)
);
impl_wide_transform_zip_output!(
    TransformZip4Output<A: a, B: b, C: c, D: d>,
    transform_tuple4_to_tuple6_kernel,
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4, OutF: output_f: 5)
);
impl_wide_transform_zip_output!(
    TransformZip4Output<A: a, B: b, C: c, D: d>,
    transform_tuple4_to_tuple7_kernel,
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4, OutF: output_f: 5, OutG: output_g: 6)
);
impl_wide_transform_zip_output!(
    TransformZip5Output<A: a, B: b, C: c, D: d, E: e>,
    transform_tuple5_to_tuple4_kernel,
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3)
);
impl_wide_transform_zip_output!(
    TransformZip5Output<A: a, B: b, C: c, D: d, E: e>,
    transform_tuple5_to_tuple5_kernel,
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4)
);
impl_wide_transform_zip_output!(
    TransformZip5Output<A: a, B: b, C: c, D: d, E: e>,
    transform_tuple5_to_tuple6_kernel,
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4, OutF: output_f: 5)
);
impl_wide_transform_zip_output!(
    TransformZip5Output<A: a, B: b, C: c, D: d, E: e>,
    transform_tuple5_to_tuple7_kernel,
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4, OutF: output_f: 5, OutG: output_g: 6)
);
impl_wide_transform_zip_output!(
    TransformZip6Output<A: a, B: b, C: c, D: d, E: e, F: f>,
    transform_tuple6_to_tuple4_kernel,
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3)
);
impl_wide_transform_zip_output!(
    TransformZip6Output<A: a, B: b, C: c, D: d, E: e, F: f>,
    transform_tuple6_to_tuple5_kernel,
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4)
);
impl_wide_transform_zip_output!(
    TransformZip6Output<A: a, B: b, C: c, D: d, E: e, F: f>,
    transform_tuple6_to_tuple6_kernel,
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4, OutF: output_f: 5)
);
impl_wide_transform_zip_output!(
    TransformZip6Output<A: a, B: b, C: c, D: d, E: e, F: f>,
    transform_tuple6_to_tuple7_kernel,
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4, OutF: output_f: 5, OutG: output_g: 6)
);

define_transform_zip_output_trait!(
    TransformZip7Output<InA: a, InB: b, InC: c, InD: d, InE: e, InF: f, InG: g>
);

macro_rules! impl_transform_zip7_output {
    (
        $kernel:ident,
        $return_expr:expr,
        ($( $out_ty:ident : $out_handle:ident : $out_field:tt ),+)
    ) => {
        impl<R, InA, InB, InC, InD, InE, InF, InG, $( $out_ty, )+ Op>
            TransformZip7Output<R, InA, InB, InC, InD, InE, InF, InG, Op>
            for ($( $out_ty, )+)
        where
            R: Runtime,
            InA: CubePrimitive + CubeElement,
            InB: CubePrimitive + CubeElement,
            InC: CubePrimitive + CubeElement,
            InD: CubePrimitive + CubeElement,
            InE: CubePrimitive + CubeElement,
            InF: CubePrimitive + CubeElement,
            InG: CubePrimitive + CubeElement,
            $( $out_ty: CubePrimitive + CubeElement, )+
            Op: UnaryOp<(InA, InB, InC, InD, InE, InF, InG), Output = ($( $out_ty, )+)>,
        {
            fn run(
                policy: &crate::policy::CubePolicy<R>,
                a: DeviceColumnView<R, InA>,
                b: DeviceColumnView<R, InB>,
                c: DeviceColumnView<R, InC>,
                d: DeviceColumnView<R, InD>,
                e: DeviceColumnView<R, InE>,
                f: DeviceColumnView<R, InF>,
                g: DeviceColumnView<R, InG>,
            ) -> Result<<Self as MItemStorage<R>>::Storage, Error> {
                let len = a.len();
                let client = policy.client();
                $(
                    let $out_handle = client.empty(len * std::mem::size_of::<$out_ty>());
                )+
                if len != 0 {
                    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
                    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
                    let a_offset = transform_offset_handle(client, a.offset)?;
                    let b_offset = transform_offset_handle(client, b.offset)?;
                    let c_offset = transform_offset_handle(client, c.offset)?;
                    let d_offset = transform_offset_handle(client, d.offset)?;
                    let e_offset = transform_offset_handle(client, e.offset)?;
                    let f_offset = transform_offset_handle(client, f.offset)?;
                    let g_offset = transform_offset_handle(client, g.offset)?;
                    let block_size = 256_u32;
                    let block_count = len.div_ceil(block_size as usize);
                    let block_count_u32 = u32::try_from(block_count)
                        .map_err(|_| Error::LengthTooLarge { len: block_count })?;
                    unsafe {
                        $kernel::launch_unchecked::<
                            InA,
                            InB,
                            InC,
                            InD,
                            InE,
                            InF,
                            InG,
                            $( $out_ty, )+
                            Op,
                            R,
                        >(
                            client,
                            crate::detail::launch::cube_count_1d(block_count_u32),
                            CubeDim::new_1d(block_size),
                            BufferArg::from_raw_parts(a.source.handle.clone(), a.source.len()),
                            BufferArg::from_raw_parts(b.source.handle.clone(), b.source.len()),
                            BufferArg::from_raw_parts(c.source.handle.clone(), c.source.len()),
                            BufferArg::from_raw_parts(d.source.handle.clone(), d.source.len()),
                            BufferArg::from_raw_parts(e.source.handle.clone(), e.source.len()),
                            BufferArg::from_raw_parts(f.source.handle.clone(), f.source.len()),
                            BufferArg::from_raw_parts(g.source.handle.clone(), g.source.len()),
                            BufferArg::from_raw_parts(a_offset.clone(), 1),
                            BufferArg::from_raw_parts(b_offset.clone(), 1),
                            BufferArg::from_raw_parts(c_offset.clone(), 1),
                            BufferArg::from_raw_parts(d_offset.clone(), 1),
                            BufferArg::from_raw_parts(e_offset.clone(), 1),
                            BufferArg::from_raw_parts(f_offset.clone(), 1),
                            BufferArg::from_raw_parts(g_offset.clone(), 1),
                            BufferArg::from_raw_parts(len_handle.clone(), 1),
                            $(
                                BufferArg::from_raw_parts($out_handle.clone(), len),
                            )+
                        );
                    }
                }
                Ok($return_expr(policy, len, $($out_handle,)+))
            }
        }
    };
}

impl_transform_zip7_output!(
    transform_tuple7_to_tuple1_kernel,
    |policy: &CubePolicy<R>, len, output_a| Zip1 {
        source: DeviceVec::from_handle(policy.id(), output_a, len),
    },
    (OutA: output_a: source)
);
impl_transform_zip7_output!(
    transform_tuple7_to_tuple2_kernel,
    |policy: &CubePolicy<R>, len, output_a, output_b| Zip2 {
        left: DeviceVec::from_handle(policy.id(), output_a, len),
        right: DeviceVec::from_handle(policy.id(), output_b, len),
    },
    (OutA: output_a: left, OutB: output_b: right)
);
impl_transform_zip7_output!(
    transform_tuple7_to_tuple3_kernel,
    |policy: &CubePolicy<R>, len, output_a, output_b, output_c| Zip3 {
        first: DeviceVec::from_handle(policy.id(), output_a, len),
        second: DeviceVec::from_handle(policy.id(), output_b, len),
        third: DeviceVec::from_handle(policy.id(), output_c, len),
    },
    (OutA: output_a: first, OutB: output_b: second, OutC: output_c: third)
);
impl_transform_zip7_output!(
    transform_tuple7_to_tuple4_kernel,
    |policy: &CubePolicy<R>, len, output_a, output_b, output_c, output_d| (
        DeviceVec::from_handle(policy.id(), output_a, len),
        DeviceVec::from_handle(policy.id(), output_b, len),
        DeviceVec::from_handle(policy.id(), output_c, len),
        DeviceVec::from_handle(policy.id(), output_d, len),
    ),
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3)
);
impl_transform_zip7_output!(
    transform_tuple7_to_tuple5_kernel,
    |policy: &CubePolicy<R>, len, output_a, output_b, output_c, output_d, output_e| (
        DeviceVec::from_handle(policy.id(), output_a, len),
        DeviceVec::from_handle(policy.id(), output_b, len),
        DeviceVec::from_handle(policy.id(), output_c, len),
        DeviceVec::from_handle(policy.id(), output_d, len),
        DeviceVec::from_handle(policy.id(), output_e, len),
    ),
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4)
);
impl_transform_zip7_output!(
    transform_tuple7_to_tuple6_kernel,
    |policy: &CubePolicy<R>, len, output_a, output_b, output_c, output_d, output_e, output_f| (
        DeviceVec::from_handle(policy.id(), output_a, len),
        DeviceVec::from_handle(policy.id(), output_b, len),
        DeviceVec::from_handle(policy.id(), output_c, len),
        DeviceVec::from_handle(policy.id(), output_d, len),
        DeviceVec::from_handle(policy.id(), output_e, len),
        DeviceVec::from_handle(policy.id(), output_f, len),
    ),
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4, OutF: output_f: 5)
);
impl_transform_zip7_output!(
    transform_tuple7_to_tuple7_kernel,
    |policy: &CubePolicy<R>, len, output_a, output_b, output_c, output_d, output_e, output_f, output_g| (
        DeviceVec::from_handle(policy.id(), output_a, len),
        DeviceVec::from_handle(policy.id(), output_b, len),
        DeviceVec::from_handle(policy.id(), output_c, len),
        DeviceVec::from_handle(policy.id(), output_d, len),
        DeviceVec::from_handle(policy.id(), output_e, len),
        DeviceVec::from_handle(policy.id(), output_f, len),
        DeviceVec::from_handle(policy.id(), output_g, len),
    ),
    (OutA: output_a: 0, OutB: output_b: 1, OutC: output_c: 2, OutD: output_d: 3, OutE: output_e: 4, OutF: output_f: 5, OutG: output_g: 6)
);

/// Internal output that can be materialized into public owned device values.
#[doc(hidden)]
pub trait MaterializeOutput {
    /// Runtime used by this output.
    type Runtime: Runtime;
    /// Public output produced by materializing this internal output.
    type Output;

    /// Materializes this internal output.
    fn materialize_output(self, policy: &CubePolicy<Self::Runtime>) -> Result<Self::Output, Error>;
}

impl<Left, Right> MaterializeOutput for Zip2<Left, Right>
where
    Self: Zip<Item = (Left::Item, Right::Item), Scalar = Left::Item>,
    Left: StorageKernelColumn + KernelColumnAt<S0>,
    Right: StorageKernelColumn<Runtime = Left::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<Left as KernelColumnAt<S0>>::Next>,
    Left::Item: CubePrimitive + CubeElement,
    Right::Item: CubePrimitive + CubeElement,
    Left::Expr: DeviceGpuExpr<Left::Item>,
    Right::Expr: DeviceGpuExpr<Right::Item>,
{
    type Runtime = Left::Runtime;
    type Output = (
        DeviceVec<Left::Runtime, Left::Item>,
        DeviceVec<Left::Runtime, Right::Item>,
    );

    fn materialize_output(self, policy: &CubePolicy<Self::Runtime>) -> Result<Self::Output, Error> {
        Zip::validate(&self)?;
        let left = crate::detail::apply::MaterializePayloadApply::collect_expr(policy, &self.left)?;
        let right =
            crate::detail::apply::MaterializePayloadApply::collect_expr(policy, &self.right)?;
        Ok((left, right))
    }
}

impl<Source> MaterializeOutput for Zip1<Source>
where
    Self: Zip<Item = (Source::Item,), Scalar = Source::Item>,
    Source: StorageKernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
{
    type Runtime = Source::Runtime;
    type Output = (DeviceVec<Source::Runtime, Source::Item>,);

    fn materialize_output(self, policy: &CubePolicy<Self::Runtime>) -> Result<Self::Output, Error> {
        Zip::validate(&self)?;
        let source =
            crate::detail::apply::MaterializePayloadApply::collect_expr(policy, &self.source)?;
        Ok((source,))
    }
}

impl<R, T> MaterializeOutput for DeviceVec<R, T>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    type Runtime = R;
    type Output = Self;

    fn materialize_output(
        self,
        _policy: &CubePolicy<Self::Runtime>,
    ) -> Result<Self::Output, Error> {
        Ok(self)
    }
}

macro_rules! impl_wide_device_vec_materialize_output {
    ($( $ty:ident : $field:tt ),+) => {
        impl<R, $( $ty ),+> MaterializeOutput for ($( DeviceVec<R, $ty>, )+)
        where
            R: Runtime,
            $( $ty: CubePrimitive + CubeElement, )+
        {
            type Runtime = R;
            type Output = ($( DeviceVec<R, $ty>, )+);

            fn materialize_output(
                self,
                _policy: &CubePolicy<Self::Runtime>,
            ) -> Result<Self::Output, Error> {
                Ok(($( self.$field, )+))
            }
        }
    };
}

impl_wide_device_vec_materialize_output!(A: 0, B: 1, C: 2, D: 3);
impl_wide_device_vec_materialize_output!(A: 0, B: 1, C: 2, D: 3, E: 4);
impl_wide_device_vec_materialize_output!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5);
impl_wide_device_vec_materialize_output!(A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6);

impl<First, Second, Third> MaterializeOutput for Zip3<First, Second, Third>
where
    Self: Zip<Item = (First::Item, Second::Item, Third::Item), Scalar = First::Item>,
    First: StorageKernelColumn + KernelColumnAt<S0>,
    Second: StorageKernelColumn<Runtime = First::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<First as KernelColumnAt<S0>>::Next>,
    Third: StorageKernelColumn<Runtime = First::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<Second as KernelColumnAt<<First as KernelColumnAt<S0>>::Next>>::Next>,
    First::Item: CubePrimitive + CubeElement,
    Second::Item: CubePrimitive + CubeElement,
    Third::Item: CubePrimitive + CubeElement,
    First::Expr: DeviceGpuExpr<First::Item>,
    Second::Expr: DeviceGpuExpr<Second::Item>,
    Third::Expr: DeviceGpuExpr<Third::Item>,
{
    type Runtime = First::Runtime;
    type Output = (
        DeviceVec<First::Runtime, First::Item>,
        DeviceVec<First::Runtime, Second::Item>,
        DeviceVec<First::Runtime, Third::Item>,
    );

    fn materialize_output(self, policy: &CubePolicy<Self::Runtime>) -> Result<Self::Output, Error> {
        Zip::validate(&self)?;
        let first =
            crate::detail::apply::MaterializePayloadApply::collect_expr(policy, &self.first)?;
        let second =
            crate::detail::apply::MaterializePayloadApply::collect_expr(policy, &self.second)?;
        let third =
            crate::detail::apply::MaterializePayloadApply::collect_expr(policy, &self.third)?;
        Ok((first, second, third))
    }
}

impl<Left, Right> MaterializeOutput for (Left, Right)
where
    Left: MaterializeOutput,
    Right: MaterializeOutput<Runtime = Left::Runtime>,
{
    type Runtime = Left::Runtime;
    type Output = (Left::Output, Right::Output);

    fn materialize_output(self, policy: &CubePolicy<Self::Runtime>) -> Result<Self::Output, Error> {
        Ok((
            self.0.materialize_output(policy)?,
            self.1.materialize_output(policy)?,
        ))
    }
}

pub(crate) fn materialize<Source>(
    policy: &CubePolicy<Source::Runtime>,
    source: Source,
) -> Result<<Source as MaterializeOutput>::Output, Error>
where
    Source: MaterializeOutput,
{
    source.materialize_output(policy)
}
