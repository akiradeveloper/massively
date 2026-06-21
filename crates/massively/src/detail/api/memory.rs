use crate::{
    device::{
        DeviceColumnView, DeviceVec, KernelColumn, KernelColumnAt, S0, SoA, SoA1, SoA2, SoA3,
        StorageKernelColumn,
    },
    error::Error,
    expr::DeviceGpuExpr,
    kernels::*,
    op::UnaryOp,
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
pub trait StorageOutput<R: Runtime>: CubeType {
    type Storage;
}

impl<R, A> StorageOutput<R> for (A,)
where
    R: Runtime,
    A: CubePrimitive + CubeElement,
{
    type Storage = SoA1<DeviceVec<R, A>>;
}

impl<R, A, B> StorageOutput<R> for (A, B)
where
    R: Runtime,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
{
    type Storage = SoA2<DeviceVec<R, A>, DeviceVec<R, B>>;
}

impl<R, A, B, C> StorageOutput<R> for (A, B, C)
where
    R: Runtime,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    C: CubePrimitive + CubeElement,
{
    type Storage = SoA3<DeviceVec<R, A>, DeviceVec<R, B>, DeviceVec<R, C>>;
}

#[doc(hidden)]
pub trait TransformUnaryOutput<R, T, Op>: StorageOutput<R>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Op: UnaryOp<(T,), Output = Self>,
{
    fn run(
        policy: &CubePolicy<R>,
        input: DeviceColumnView<R, T>,
    ) -> Result<<Self as StorageOutput<R>>::Storage, Error>;
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
    ) -> Result<<Self as StorageOutput<R>>::Storage, Error> {
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
                    CubeCount::Static(block_count_u32, 1, 1),
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
        Ok(SoA1 {
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
    ) -> Result<<Self as StorageOutput<R>>::Storage, Error> {
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
                    CubeCount::Static(block_count_u32, 1, 1),
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
        Ok(SoA2 {
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
    ) -> Result<<Self as StorageOutput<R>>::Storage, Error> {
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
                    CubeCount::Static(block_count_u32, 1, 1),
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
        Ok(SoA3 {
            first: DeviceVec::from_handle(policy.id(), output_a, len),
            second: DeviceVec::from_handle(policy.id(), output_b, len),
            third: DeviceVec::from_handle(policy.id(), output_c, len),
        })
    }
}

macro_rules! impl_transform_tuple_output {
    (
        ($trait_name:ident < $first_in:ident : $first_arg:ident, $( $in_ty:ident : $arg:ident ),+ >),
        $kernel:ident,
        $soa:ident,
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
            ) -> Result<<Self as StorageOutput<R>>::Storage, Error> {
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
                            CubeCount::Static(block_count_u32, 1, 1),
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
                Ok($soa {
                    $(
                        $out_field: DeviceVec::from_handle(policy.id(), $out_handle, len),
                    )+
                })
            }
        }
    };
}

macro_rules! impl_transform_tuple_output_arity {
    ($input:tt, 2, $kernel:ident) => {
        impl_transform_tuple_output!(
            $input,
            $kernel,
            SoA2,
            (OutA: out_a: left, OutB: out_b: right)
        );
    };
    ($input:tt, 3, $kernel:ident) => {
        impl_transform_tuple_output!(
            $input,
            $kernel,
            SoA3,
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
pub trait TransformSoA2Output<R, InA, InB, Op>: CubeType + StorageOutput<R>
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
    ) -> Result<<Self as StorageOutput<R>>::Storage, Error>;
}

impl<R, InA, InB, OutA, Op> TransformSoA2Output<R, InA, InB, Op> for (OutA,)
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
    ) -> Result<<Self as StorageOutput<R>>::Storage, Error> {
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
                    CubeCount::Static(block_count_u32, 1, 1),
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
        Ok(SoA1 {
            source: DeviceVec::from_handle(policy.id(), output_a, len),
        })
    }
}

impl<R, InA, InB, OutA, OutB, Op> TransformSoA2Output<R, InA, InB, Op> for (OutA, OutB)
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
    ) -> Result<<Self as StorageOutput<R>>::Storage, Error> {
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
                    CubeCount::Static(block_count_u32, 1, 1),
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
        Ok(SoA2 {
            left: DeviceVec::from_handle(policy.id(), output_a, len),
            right: DeviceVec::from_handle(policy.id(), output_b, len),
        })
    }
}

#[doc(hidden)]
pub trait TransformSoA3Output<R, InA, InB, InC, Op>: CubeType + StorageOutput<R>
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
    ) -> Result<<Self as StorageOutput<R>>::Storage, Error>;
}

impl<R, InA, InB, InC, OutA, Op> TransformSoA3Output<R, InA, InB, InC, Op> for (OutA,)
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
    ) -> Result<<Self as StorageOutput<R>>::Storage, Error> {
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
                    CubeCount::Static(block_count_u32, 1, 1),
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
        Ok(SoA1 {
            source: DeviceVec::from_handle(policy.id(), output_a, len),
        })
    }
}

impl<R, InA, InB, InC, OutA, OutB, OutC, Op> TransformSoA3Output<R, InA, InB, InC, Op>
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
    ) -> Result<<Self as StorageOutput<R>>::Storage, Error> {
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
                    CubeCount::Static(block_count_u32, 1, 1),
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
        Ok(SoA3 {
            first: DeviceVec::from_handle(policy.id(), output_a, len),
            second: DeviceVec::from_handle(policy.id(), output_b, len),
            third: DeviceVec::from_handle(policy.id(), output_c, len),
        })
    }
}

impl_transform_tuple_outputs!(
    TransformSoA2Output<A: a, B: b>,
    3 => transform_tuple2_to_tuple3_kernel,
);
impl_transform_tuple_outputs!(
    TransformSoA3Output<A: a, B: b, C: c>,
    2 => transform_tuple3_to_tuple2_kernel,
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

impl<Left, Right> MaterializeOutput for SoA2<Left, Right>
where
    Self: SoA<Item = (Left::Item, Right::Item), Scalar = Left::Item>,
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
        SoA::validate(&self)?;
        let left = super::device_expr_collect_with_policy(policy, &self.left)?;
        let right = super::device_expr_collect_with_policy(policy, &self.right)?;
        Ok((left, right))
    }
}

impl<Source> MaterializeOutput for SoA1<Source>
where
    Self: SoA<Item = (Source::Item,), Scalar = Source::Item>,
    Source: StorageKernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
{
    type Runtime = Source::Runtime;
    type Output = (DeviceVec<Source::Runtime, Source::Item>,);

    fn materialize_output(self, policy: &CubePolicy<Self::Runtime>) -> Result<Self::Output, Error> {
        SoA::validate(&self)?;
        let source = super::device_expr_collect_with_policy(policy, &self.source)?;
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

impl<First, Second, Third> MaterializeOutput for SoA3<First, Second, Third>
where
    Self: SoA<Item = (First::Item, Second::Item, Third::Item), Scalar = First::Item>,
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
        SoA::validate(&self)?;
        let first = super::device_expr_collect_with_policy(policy, &self.first)?;
        let second = super::device_expr_collect_with_policy(policy, &self.second)?;
        let third = super::device_expr_collect_with_policy(policy, &self.third)?;
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
