use crate::{
    device::{DeviceVec, KernelColumnBindings},
    error::Error,
    expr::{DeviceGpuExpr, GpuExpr, Input},
    kernels::*,
    op::{BinaryOp, BinaryPredicateOp, GpuOp},
    policy::CubePolicy,
    primitives::{range, workspace::Workspace},
};
use cubecl::prelude::*;

pub(crate) const BLOCK_SCAN_SIZE: u32 = 256;

pub(crate) fn inclusive_scan_u32<R: Runtime>(
    client: &ComputeClient<R>,
    input_handle: &cubecl::server::Handle,
    len: usize,
) -> Result<cubecl::server::Handle, Error> {
    if len == 0 {
        return Ok(crate::policy::empty_handle(client));
    }

    let output_handle = client.empty(len * std::mem::size_of::<u32>());
    let workspace = Workspace::from_client(client);
    let num_blocks = len.div_ceil(BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_values = [len_u32];
    let len_handle = client.create_from_slice(u32::as_bytes(&len_values));
    let block_sums_handle = workspace.alloc::<u32>(num_blocks);

    unsafe {
        u32_block_inclusive_scan_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SCAN_SIZE),
            ArrayArg::from_raw_parts::<u32>(input_handle, len, 1),
            ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
            ArrayArg::from_raw_parts::<u32>(&output_handle, len, 1),
            ArrayArg::from_raw_parts::<u32>(&block_sums_handle, num_blocks, 1),
        )
        .map_err(|err| Error::Launch {
            message: format!("{err:?}"),
        })?;
    }

    if num_blocks > 1 {
        let block_prefixes_handle =
            inclusive_scan_u32::<R>(client, &block_sums_handle, num_blocks)?;
        unsafe {
            u32_add_block_prefix_kernel::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_SCAN_SIZE),
                ArrayArg::from_raw_parts::<u32>(&block_prefixes_handle, num_blocks, 1),
                ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
                ArrayArg::from_raw_parts::<u32>(&output_handle, len, 1),
            )
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
        }
    }

    Ok(output_handle)
}

pub(crate) fn inclusive_scan_values<R, T, Op>(
    client: &ComputeClient<R>,
    input_handle: &cubecl::server::Handle,
    len: usize,
) -> Result<cubecl::server::Handle, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Op: BinaryOp<T>,
    Input<T>: GpuExpr<T>,
{
    if len == 0 {
        return Ok(crate::policy::empty_handle(client));
    }

    let output_handle = client.empty(len * std::mem::size_of::<T>());
    let workspace = Workspace::from_client(client);
    let num_blocks = len.div_ceil(BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_values = [len_u32];
    let len_handle = client.create_from_slice(u32::as_bytes(&len_values));
    let dummy_indices = [0u32];
    let dummy_index_handle = client.create_from_slice(u32::as_bytes(&dummy_indices));
    let block_sums_handle = workspace.alloc::<T>(num_blocks);

    unsafe {
        inclusive_scan_expr_block_kernel::launch_unchecked::<T, Input<T>, Op, R>(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SCAN_SIZE),
            ArrayArg::from_raw_parts::<T>(input_handle, len, 1),
            ArrayArg::from_raw_parts::<u32>(&dummy_index_handle, dummy_indices.len(), 1),
            ArrayArg::from_raw_parts::<T>(input_handle, len, 1),
            ArrayArg::from_raw_parts::<u32>(&dummy_index_handle, dummy_indices.len(), 1),
            ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
            ArrayArg::from_raw_parts::<T>(&output_handle, len, 1),
            ArrayArg::from_raw_parts::<T>(&block_sums_handle, num_blocks, 1),
        )
        .map_err(|err| Error::Launch {
            message: format!("{err:?}"),
        })?;
    }

    if num_blocks > 1 {
        let block_prefixes_handle =
            inclusive_scan_values::<R, T, Op>(client, &block_sums_handle, num_blocks)?;
        unsafe {
            scan_add_block_prefix_kernel::launch_unchecked::<T, Op, R>(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_SCAN_SIZE),
                ArrayArg::from_raw_parts::<T>(&block_prefixes_handle, num_blocks, 1),
                ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
                ArrayArg::from_raw_parts::<T>(&output_handle, len, 1),
            )
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
        }
    }

    Ok(output_handle)
}

pub(crate) fn inclusive_scan_by_key_device_vec<R, K, T, KeyEq, Op>(
    keys: &DeviceVec<R, K>,
    values: &DeviceVec<R, T>,
    _key_eq: GpuOp<KeyEq>,
    _op: GpuOp<Op>,
) -> Result<DeviceVec<R, T>, Error>
where
    R: Runtime,
    K: CubePrimitive + CubeElement,
    T: CubePrimitive + CubeElement,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
{
    super::ensure_same_len(values.len(), keys.len())?;

    let output_handle =
        inclusive_scan_by_key_handle::<R, K, T, KeyEq, Op>(values.policy(), keys, &values.handle)?;
    Ok(DeviceVec::from_handle(
        values.policy().clone(),
        output_handle,
        values.len(),
    ))
}

pub(crate) fn exclusive_scan_by_key_device_vec<R, K, T, KeyEq, Op>(
    keys: &DeviceVec<R, K>,
    values: &DeviceVec<R, T>,
    init: T,
    _key_eq: GpuOp<KeyEq>,
    _op: GpuOp<Op>,
) -> Result<DeviceVec<R, T>, Error>
where
    R: Runtime,
    K: CubePrimitive + CubeElement,
    T: CubePrimitive + CubeElement,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
{
    super::ensure_same_len(values.len(), keys.len())?;

    let client = values.policy().client();
    let inclusive_handle =
        inclusive_scan_by_key_handle::<R, K, T, KeyEq, Op>(values.policy(), keys, &values.handle)?;
    let output_handle =
        make_scan_by_key_exclusive::<R, K, T, KeyEq, Op>(client, keys, &inclusive_handle, init)?;
    Ok(DeviceVec::from_handle(
        values.policy().clone(),
        output_handle,
        values.len(),
    ))
}

macro_rules! define_scan_tuple_by_key_device_vec {
    (
        $inclusive_fn:ident,
        $exclusive_fn:ident,
        $handle_fn:ident,
        $exclusive_handle_fn:ident,
        $block_kernel:ident,
        $add_prefix_kernel:ident,
        $exclusive_kernel:ident,
        ( $( $ty:ident: $key:ident: $tail_handle:ident: $tail_vec:ident ),+ )
    ) => {
        pub(crate) fn $inclusive_fn<R, $( $ty ),+, T, KeyEq, Op>(
            $( $key: &DeviceVec<R, $ty>, )+
            values: &DeviceVec<R, T>,
            _key_eq: GpuOp<KeyEq>,
            _op: GpuOp<Op>,
        ) -> Result<DeviceVec<R, T>, Error>
        where
            R: Runtime,
            $( $ty: CubePrimitive + CubeElement, )+
            T: CubePrimitive + CubeElement,
            KeyEq: BinaryPredicateOp<($( $ty ),+)>,
            Op: BinaryOp<T>,
        {
            let len = values.len();
            $(
                super::ensure_same_len($key.len(), len)?;
            )+

            let output_handle = $handle_fn::<R, $( $ty, )+ T, KeyEq, Op>(
                values.policy(),
                $( $key, )+
                &values.handle,
            )?;
            Ok(DeviceVec::from_handle(
                values.policy().clone(),
                output_handle,
                values.len(),
            ))
        }

        pub(crate) fn $exclusive_fn<R, $( $ty ),+, T, KeyEq, Op>(
            $( $key: &DeviceVec<R, $ty>, )+
            values: &DeviceVec<R, T>,
            init: T,
            _key_eq: GpuOp<KeyEq>,
            _op: GpuOp<Op>,
        ) -> Result<DeviceVec<R, T>, Error>
        where
            R: Runtime,
            $( $ty: CubePrimitive + CubeElement, )+
            T: CubePrimitive + CubeElement,
            KeyEq: BinaryPredicateOp<($( $ty ),+)>,
            Op: BinaryOp<T>,
        {
            let len = values.len();
            $(
                super::ensure_same_len($key.len(), len)?;
            )+

            let client = values.policy().client();
            let inclusive_handle = $handle_fn::<R, $( $ty, )+ T, KeyEq, Op>(
                values.policy(),
                $( $key, )+
                &values.handle,
            )?;
            let output_handle = $exclusive_handle_fn::<R, $( $ty, )+ T, KeyEq, Op>(
                client,
                $( $key, )+
                &inclusive_handle,
                init,
            )?;
            Ok(DeviceVec::from_handle(
                values.policy().clone(),
                output_handle,
                values.len(),
            ))
        }

        pub(crate) fn $handle_fn<R, $( $ty ),+, T, KeyEq, Op>(
            policy: &CubePolicy<R>,
            $( $key: &DeviceVec<R, $ty>, )+
            value_handle: &cubecl::server::Handle,
        ) -> Result<cubecl::server::Handle, Error>
        where
            R: Runtime,
            $( $ty: CubePrimitive + CubeElement, )+
            T: CubePrimitive + CubeElement,
            KeyEq: BinaryPredicateOp<($( $ty ),+)>,
            Op: BinaryOp<T>,
        {
            let mut len = None;
            $(
                let key_len = $key.len();
                if let Some(expected) = len {
                    super::ensure_same_len(key_len, expected)?;
                } else {
                    len = Some(key_len);
                }
            )+
            let len = len.unwrap_or(0);

            let client = policy.client();
            if len == 0 {
                return Ok(policy.empty_handle());
            }
            if len == 1 {
                return range::copy_handle::<R, T>(policy, value_handle, len);
            }

            let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
            let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
            let num_blocks = len.div_ceil(BLOCK_SCAN_SIZE as usize);
            let num_blocks_u32 =
                u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
            let workspace = Workspace::new(policy);
            let output_handle = workspace.alloc::<T>(len);
            $(
                let $tail_handle = workspace.alloc::<$ty>(num_blocks);
            )+
            let block_tail_values = workspace.alloc::<T>(num_blocks);

            unsafe {
                $block_kernel::launch_unchecked::<$( $ty, )+ T, KeyEq, Op, R>(
                    client,
                    CubeCount::Static(num_blocks_u32, 1, 1),
                    CubeDim::new_1d(BLOCK_SCAN_SIZE),
                    $(
                        ArrayArg::from_raw_parts::<$ty>(&$key.handle, len, 1),
                    )+
                    ArrayArg::from_raw_parts::<T>(value_handle, len, 1),
                    ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
                    ArrayArg::from_raw_parts::<T>(&output_handle, len, 1),
                    $(
                        ArrayArg::from_raw_parts::<$ty>(&$tail_handle, num_blocks, 1),
                    )+
                    ArrayArg::from_raw_parts::<T>(&block_tail_values, num_blocks, 1),
                )
                .map_err(|err| Error::Launch {
                    message: format!("{err:?}"),
                })?;
            }

            if num_blocks > 1 {
                $(
                    let $tail_vec = DeviceVec::from_handle(
                        policy.clone(),
                        $tail_handle.clone(),
                        num_blocks,
                    );
                )+
                let block_prefixes = $handle_fn::<R, $( $ty, )+ T, KeyEq, Op>(
                    policy,
                    $( &$tail_vec, )+
                    &block_tail_values,
                )?;
                unsafe {
                    $add_prefix_kernel::launch_unchecked::<$( $ty, )+ T, KeyEq, Op, R>(
                        client,
                        CubeCount::Static(num_blocks_u32, 1, 1),
                        CubeDim::new_1d(BLOCK_SCAN_SIZE),
                        $(
                            ArrayArg::from_raw_parts::<$ty>(&$key.handle, len, 1),
                        )+
                        $(
                            ArrayArg::from_raw_parts::<$ty>(&$tail_handle, num_blocks, 1),
                        )+
                        ArrayArg::from_raw_parts::<T>(&block_prefixes, num_blocks, 1),
                        ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
                        ArrayArg::from_raw_parts::<T>(&output_handle, len, 1),
                    )
                    .map_err(|err| Error::Launch {
                        message: format!("{err:?}"),
                    })?;
                }
            }

            Ok(output_handle)
        }

        pub(crate) fn $exclusive_handle_fn<R, $( $ty ),+, T, KeyEq, Op>(
            client: &ComputeClient<R>,
            $( $key: &DeviceVec<R, $ty>, )+
            inclusive_handle: &cubecl::server::Handle,
            init: T,
        ) -> Result<cubecl::server::Handle, Error>
        where
            R: Runtime,
            $( $ty: CubePrimitive + CubeElement, )+
            T: CubePrimitive + CubeElement,
            KeyEq: BinaryPredicateOp<($( $ty ),+)>,
            Op: BinaryOp<T>,
        {
            let mut len = None;
            $(
                let key_len = $key.len();
                if let Some(expected) = len {
                    super::ensure_same_len(key_len, expected)?;
                } else {
                    len = Some(key_len);
                }
            )+
            let len = len.unwrap_or(0);
            if len == 0 {
                return Ok(crate::policy::empty_handle(client));
            }

            let output_handle = client.empty(len * std::mem::size_of::<T>());
            u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
            let num_blocks = len.div_ceil(BLOCK_SCAN_SIZE as usize);
            let num_blocks_u32 =
                u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
            let init_handle = client.create_from_slice(T::as_bytes(&[init]));
            unsafe {
                $exclusive_kernel::launch_unchecked::<$( $ty, )+ T, KeyEq, Op, R>(
                    client,
                    CubeCount::Static(num_blocks_u32, 1, 1),
                    CubeDim::new_1d(BLOCK_SCAN_SIZE),
                    $(
                        ArrayArg::from_raw_parts::<$ty>(&$key.handle, len, 1),
                    )+
                    ArrayArg::from_raw_parts::<T>(inclusive_handle, len, 1),
                    ArrayArg::from_raw_parts::<T>(&init_handle, 1, 1),
                    ArrayArg::from_raw_parts::<T>(&output_handle, len, 1),
                )
                .map_err(|err| Error::Launch {
                    message: format!("{err:?}"),
                })?;
            }

            Ok(output_handle)
        }
    };
}

define_scan_tuple_by_key_device_vec!(
    inclusive_scan_tuple2_by_key_device_vec,
    exclusive_scan_tuple2_by_key_device_vec,
    inclusive_scan_tuple2_by_key_handle,
    make_scan_tuple2_by_key_exclusive,
    scan_tuple2_by_key_block_kernel,
    scan_tuple2_by_key_add_block_prefix_kernel,
    scan_tuple2_by_key_make_exclusive_kernel,
    (A: key_a: block_tail_a: block_tail_a_vec, B: key_b: block_tail_b: block_tail_b_vec)
);
define_scan_tuple_by_key_device_vec!(
    inclusive_scan_tuple3_by_key_device_vec,
    exclusive_scan_tuple3_by_key_device_vec,
    inclusive_scan_tuple3_by_key_handle,
    make_scan_tuple3_by_key_exclusive,
    scan_tuple3_by_key_block_kernel,
    scan_tuple3_by_key_add_block_prefix_kernel,
    scan_tuple3_by_key_make_exclusive_kernel,
    (A: key_a: block_tail_a: block_tail_a_vec, B: key_b: block_tail_b: block_tail_b_vec, C: key_c: block_tail_c: block_tail_c_vec)
);
define_scan_tuple_by_key_device_vec!(
    inclusive_scan_tuple4_by_key_device_vec,
    exclusive_scan_tuple4_by_key_device_vec,
    inclusive_scan_tuple4_by_key_handle,
    make_scan_tuple4_by_key_exclusive,
    scan_tuple4_by_key_block_kernel,
    scan_tuple4_by_key_add_block_prefix_kernel,
    scan_tuple4_by_key_make_exclusive_kernel,
    (A: key_a: block_tail_a: block_tail_a_vec, B: key_b: block_tail_b: block_tail_b_vec, C: key_c: block_tail_c: block_tail_c_vec, D: key_d: block_tail_d: block_tail_d_vec)
);
define_scan_tuple_by_key_device_vec!(
    inclusive_scan_tuple5_by_key_device_vec,
    exclusive_scan_tuple5_by_key_device_vec,
    inclusive_scan_tuple5_by_key_handle,
    make_scan_tuple5_by_key_exclusive,
    scan_tuple5_by_key_block_kernel,
    scan_tuple5_by_key_add_block_prefix_kernel,
    scan_tuple5_by_key_make_exclusive_kernel,
    (A: key_a: block_tail_a: block_tail_a_vec, B: key_b: block_tail_b: block_tail_b_vec, C: key_c: block_tail_c: block_tail_c_vec, D: key_d: block_tail_d: block_tail_d_vec, E: key_e: block_tail_e: block_tail_e_vec)
);
define_scan_tuple_by_key_device_vec!(
    inclusive_scan_tuple6_by_key_device_vec,
    exclusive_scan_tuple6_by_key_device_vec,
    inclusive_scan_tuple6_by_key_handle,
    make_scan_tuple6_by_key_exclusive,
    scan_tuple6_by_key_block_kernel,
    scan_tuple6_by_key_add_block_prefix_kernel,
    scan_tuple6_by_key_make_exclusive_kernel,
    (A: key_a: block_tail_a: block_tail_a_vec, B: key_b: block_tail_b: block_tail_b_vec, C: key_c: block_tail_c: block_tail_c_vec, D: key_d: block_tail_d: block_tail_d_vec, E: key_e: block_tail_e: block_tail_e_vec, F: key_f: block_tail_f: block_tail_f_vec)
);
define_scan_tuple_by_key_device_vec!(
    inclusive_scan_tuple7_by_key_device_vec,
    exclusive_scan_tuple7_by_key_device_vec,
    inclusive_scan_tuple7_by_key_handle,
    make_scan_tuple7_by_key_exclusive,
    scan_tuple7_by_key_block_kernel,
    scan_tuple7_by_key_add_block_prefix_kernel,
    scan_tuple7_by_key_make_exclusive_kernel,
    (A: key_a: block_tail_a: block_tail_a_vec, B: key_b: block_tail_b: block_tail_b_vec, C: key_c: block_tail_c: block_tail_c_vec, D: key_d: block_tail_d: block_tail_d_vec, E: key_e: block_tail_e: block_tail_e_vec, F: key_f: block_tail_f: block_tail_f_vec, G: key_g: block_tail_g: block_tail_g_vec)
);
define_scan_tuple_by_key_device_vec!(
    inclusive_scan_tuple8_by_key_device_vec,
    exclusive_scan_tuple8_by_key_device_vec,
    inclusive_scan_tuple8_by_key_handle,
    make_scan_tuple8_by_key_exclusive,
    scan_tuple8_by_key_block_kernel,
    scan_tuple8_by_key_add_block_prefix_kernel,
    scan_tuple8_by_key_make_exclusive_kernel,
    (A: key_a: block_tail_a: block_tail_a_vec, B: key_b: block_tail_b: block_tail_b_vec, C: key_c: block_tail_c: block_tail_c_vec, D: key_d: block_tail_d: block_tail_d_vec, E: key_e: block_tail_e: block_tail_e_vec, F: key_f: block_tail_f: block_tail_f_vec, G: key_g: block_tail_g: block_tail_g_vec, I: key_h: block_tail_h: block_tail_h_vec)
);
define_scan_tuple_by_key_device_vec!(
    inclusive_scan_tuple9_by_key_device_vec,
    exclusive_scan_tuple9_by_key_device_vec,
    inclusive_scan_tuple9_by_key_handle,
    make_scan_tuple9_by_key_exclusive,
    scan_tuple9_by_key_block_kernel,
    scan_tuple9_by_key_add_block_prefix_kernel,
    scan_tuple9_by_key_make_exclusive_kernel,
    (A: key_a: block_tail_a: block_tail_a_vec, B: key_b: block_tail_b: block_tail_b_vec, C: key_c: block_tail_c: block_tail_c_vec, D: key_d: block_tail_d: block_tail_d_vec, E: key_e: block_tail_e: block_tail_e_vec, F: key_f: block_tail_f: block_tail_f_vec, G: key_g: block_tail_g: block_tail_g_vec, I: key_h: block_tail_h: block_tail_h_vec, J: key_i: block_tail_i: block_tail_i_vec)
);
define_scan_tuple_by_key_device_vec!(
    inclusive_scan_tuple10_by_key_device_vec,
    exclusive_scan_tuple10_by_key_device_vec,
    inclusive_scan_tuple10_by_key_handle,
    make_scan_tuple10_by_key_exclusive,
    scan_tuple10_by_key_block_kernel,
    scan_tuple10_by_key_add_block_prefix_kernel,
    scan_tuple10_by_key_make_exclusive_kernel,
    (A: key_a: block_tail_a: block_tail_a_vec, B: key_b: block_tail_b: block_tail_b_vec, C: key_c: block_tail_c: block_tail_c_vec, D: key_d: block_tail_d: block_tail_d_vec, E: key_e: block_tail_e: block_tail_e_vec, F: key_f: block_tail_f: block_tail_f_vec, G: key_g: block_tail_g: block_tail_g_vec, I: key_h: block_tail_h: block_tail_h_vec, J: key_i: block_tail_i: block_tail_i_vec, K: key_j: block_tail_j: block_tail_j_vec)
);
define_scan_tuple_by_key_device_vec!(
    inclusive_scan_tuple11_by_key_device_vec,
    exclusive_scan_tuple11_by_key_device_vec,
    inclusive_scan_tuple11_by_key_handle,
    make_scan_tuple11_by_key_exclusive,
    scan_tuple11_by_key_block_kernel,
    scan_tuple11_by_key_add_block_prefix_kernel,
    scan_tuple11_by_key_make_exclusive_kernel,
    (A: key_a: block_tail_a: block_tail_a_vec, B: key_b: block_tail_b: block_tail_b_vec, C: key_c: block_tail_c: block_tail_c_vec, D: key_d: block_tail_d: block_tail_d_vec, E: key_e: block_tail_e: block_tail_e_vec, F: key_f: block_tail_f: block_tail_f_vec, G: key_g: block_tail_g: block_tail_g_vec, I: key_h: block_tail_h: block_tail_h_vec, J: key_i: block_tail_i: block_tail_i_vec, K: key_j: block_tail_j: block_tail_j_vec, L: key_k: block_tail_k: block_tail_k_vec)
);
define_scan_tuple_by_key_device_vec!(
    inclusive_scan_tuple12_by_key_device_vec,
    exclusive_scan_tuple12_by_key_device_vec,
    inclusive_scan_tuple12_by_key_handle,
    make_scan_tuple12_by_key_exclusive,
    scan_tuple12_by_key_block_kernel,
    scan_tuple12_by_key_add_block_prefix_kernel,
    scan_tuple12_by_key_make_exclusive_kernel,
    (A: key_a: block_tail_a: block_tail_a_vec, B: key_b: block_tail_b: block_tail_b_vec, C: key_c: block_tail_c: block_tail_c_vec, D: key_d: block_tail_d: block_tail_d_vec, E: key_e: block_tail_e: block_tail_e_vec, F: key_f: block_tail_f: block_tail_f_vec, G: key_g: block_tail_g: block_tail_g_vec, I: key_h: block_tail_h: block_tail_h_vec, J: key_i: block_tail_i: block_tail_i_vec, K: key_j: block_tail_j: block_tail_j_vec, L: key_k: block_tail_k: block_tail_k_vec, M: key_l: block_tail_l: block_tail_l_vec)
);

pub(crate) fn inclusive_scan_device_expr<R, T, Expr, Op>(
    policy: &CubePolicy<R>,
    bindings: &KernelColumnBindings,
    len: usize,
) -> Result<cubecl::server::Handle, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Expr: DeviceGpuExpr<T>,
    Op: BinaryOp<T>,
    Input<T>: GpuExpr<T>,
{
    let client = policy.client();
    if len == 0 {
        return Ok(policy.empty_handle());
    }

    let output_handle = client.empty(len * std::mem::size_of::<T>());
    let workspace = Workspace::new(policy);
    let slot0 = bindings.slots.first().unwrap();
    let slot1 = bindings.slots.get(1).unwrap_or(slot0);
    let slot2 = bindings.slots.get(2).unwrap_or(slot0);
    let slot3 = bindings.slots.get(3).unwrap_or(slot0);
    let num_blocks = len.div_ceil(BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let block_sums_handle = workspace.alloc::<T>(num_blocks);

    unsafe {
        device_inclusive_scan_expr_block_kernel::launch_unchecked::<T, Expr, Op, R>(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SCAN_SIZE),
            ArrayArg::from_raw_parts::<T>(&slot0.0, slot0.1, 1),
            ArrayArg::from_raw_parts::<T>(&slot1.0, slot1.1, 1),
            ArrayArg::from_raw_parts::<T>(&slot2.0, slot2.1, 1),
            ArrayArg::from_raw_parts::<T>(&slot3.0, slot3.1, 1),
            ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
            ArrayArg::from_raw_parts::<T>(&output_handle, len, 1),
            ArrayArg::from_raw_parts::<T>(&block_sums_handle, num_blocks, 1),
        )
        .map_err(|err| Error::Launch {
            message: format!("{err:?}"),
        })?;
    }

    if num_blocks > 1 {
        let block_prefixes_handle =
            inclusive_scan_values::<R, T, Op>(client, &block_sums_handle, num_blocks)?;
        unsafe {
            scan_add_block_prefix_kernel::launch_unchecked::<T, Op, R>(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_SCAN_SIZE),
                ArrayArg::from_raw_parts::<T>(&block_prefixes_handle, num_blocks, 1),
                ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
                ArrayArg::from_raw_parts::<T>(&output_handle, len, 1),
            )
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
        }
    }

    Ok(output_handle)
}

pub(crate) fn exclusive_scan_device_expr<R, T, Expr, Op>(
    policy: &CubePolicy<R>,
    bindings: &KernelColumnBindings,
    len: usize,
    init: T,
) -> Result<cubecl::server::Handle, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Expr: DeviceGpuExpr<T>,
    Op: BinaryOp<T>,
    Input<T>: GpuExpr<T>,
{
    let inclusive_handle = inclusive_scan_device_expr::<R, T, Expr, Op>(policy, bindings, len)?;
    make_exclusive::<R, T, Op>(policy.client(), &inclusive_handle, len, init)
}

pub(crate) fn inclusive_scan_by_key_handle<R, K, T, KeyEq, Op>(
    policy: &CubePolicy<R>,
    keys: &DeviceVec<R, K>,
    value_handle: &cubecl::server::Handle,
) -> Result<cubecl::server::Handle, Error>
where
    R: Runtime,
    K: CubePrimitive + CubeElement,
    T: CubePrimitive + CubeElement,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
{
    let len = keys.len();
    let client = policy.client();
    if len == 0 {
        return Ok(policy.empty_handle());
    }
    if len == 1 {
        return range::copy_handle::<R, T>(policy, value_handle, len);
    }

    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let num_blocks = len.div_ceil(BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let workspace = Workspace::new(policy);
    let output_handle = workspace.alloc::<T>(len);
    let block_tail_keys = workspace.alloc::<K>(num_blocks);
    let block_tail_values = workspace.alloc::<T>(num_blocks);

    unsafe {
        scan_by_key_block_kernel::launch_unchecked::<K, T, KeyEq, Op, R>(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SCAN_SIZE),
            ArrayArg::from_raw_parts::<K>(&keys.handle, len, 1),
            ArrayArg::from_raw_parts::<T>(value_handle, len, 1),
            ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
            ArrayArg::from_raw_parts::<T>(&output_handle, len, 1),
            ArrayArg::from_raw_parts::<K>(&block_tail_keys, num_blocks, 1),
            ArrayArg::from_raw_parts::<T>(&block_tail_values, num_blocks, 1),
        )
        .map_err(|err| Error::Launch {
            message: format!("{err:?}"),
        })?;
    }

    if num_blocks > 1 {
        let block_tail_keys_vec =
            DeviceVec::from_handle(policy.clone(), block_tail_keys.clone(), num_blocks);
        let block_prefixes = inclusive_scan_by_key_handle::<R, K, T, KeyEq, Op>(
            policy,
            &block_tail_keys_vec,
            &block_tail_values,
        )?;
        unsafe {
            scan_by_key_add_block_prefix_kernel::launch_unchecked::<K, T, KeyEq, Op, R>(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_SCAN_SIZE),
                ArrayArg::from_raw_parts::<K>(&keys.handle, len, 1),
                ArrayArg::from_raw_parts::<K>(&block_tail_keys, num_blocks, 1),
                ArrayArg::from_raw_parts::<T>(&block_prefixes, num_blocks, 1),
                ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
                ArrayArg::from_raw_parts::<T>(&output_handle, len, 1),
            )
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
        }
    }

    Ok(output_handle)
}

fn make_scan_by_key_exclusive<R, K, T, KeyEq, Op>(
    client: &ComputeClient<R>,
    keys: &DeviceVec<R, K>,
    inclusive_handle: &cubecl::server::Handle,
    init: T,
) -> Result<cubecl::server::Handle, Error>
where
    R: Runtime,
    K: CubePrimitive + CubeElement,
    T: CubePrimitive + CubeElement,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<T>,
{
    let len = keys.len();
    if len == 0 {
        return Ok(crate::policy::empty_handle(client));
    }

    let output_handle = client.empty(len * std::mem::size_of::<T>());
    u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let num_blocks = len.div_ceil(BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let init_handle = client.create_from_slice(T::as_bytes(&[init]));
    unsafe {
        scan_by_key_make_exclusive_kernel::launch_unchecked::<K, T, KeyEq, Op, R>(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SCAN_SIZE),
            ArrayArg::from_raw_parts::<K>(&keys.handle, len, 1),
            ArrayArg::from_raw_parts::<T>(inclusive_handle, len, 1),
            ArrayArg::from_raw_parts::<T>(&init_handle, 1, 1),
            ArrayArg::from_raw_parts::<T>(&output_handle, len, 1),
        )
        .map_err(|err| Error::Launch {
            message: format!("{err:?}"),
        })?;
    }

    Ok(output_handle)
}

fn make_exclusive<R, T, Op>(
    client: &ComputeClient<R>,
    inclusive_handle: &cubecl::server::Handle,
    len: usize,
    init: T,
) -> Result<cubecl::server::Handle, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Op: BinaryOp<T>,
{
    if len == 0 {
        return Ok(crate::policy::empty_handle(client));
    }

    let output_handle = client.empty(len * std::mem::size_of::<T>());
    u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let num_blocks = len.div_ceil(BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let init_handle = client.create_from_slice(T::as_bytes(&[init]));
    unsafe {
        scan_make_exclusive_kernel::launch_unchecked::<T, Op, R>(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SCAN_SIZE),
            ArrayArg::from_raw_parts::<T>(inclusive_handle, len, 1),
            ArrayArg::from_raw_parts::<T>(&init_handle, 1, 1),
            ArrayArg::from_raw_parts::<T>(&output_handle, len, 1),
        )
        .map_err(|err| Error::Launch {
            message: format!("{err:?}"),
        })?;
    }

    Ok(output_handle)
}

pub(crate) fn read_u32_scalar<R: Runtime>(
    client: &ComputeClient<R>,
    handle: cubecl::server::Handle,
) -> u32 {
    let bytes = client.read_one(handle);
    u32::from_bytes(&bytes)[0]
}
