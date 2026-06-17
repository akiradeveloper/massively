use crate::{
    device::{DeviceVec, KernelColumnBindings},
    error::Error,
    expr::{DeviceGpuExpr, GpuExpr, Input},
    kernels::*,
    op::BinaryOp,
    policy::CubePolicy,
    primitives::{scan, select},
};
use cubecl::prelude::*;

pub(crate) const BLOCK_REDUCE_SIZE: u32 = 256;

pub(crate) fn reduce_input_handle<R, T, Op>(
    policy: &CubePolicy<R>,
    input_handle: cubecl::server::Handle,
    storage_len: usize,
    len: usize,
    init: T,
) -> Result<T, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Op: BinaryOp<T>,
    Input<T>: GpuExpr<T>,
{
    if len == 0 {
        return Ok(init);
    }

    let client = policy.client();
    let dummy_indices = [0_u32];
    let dummy_index_handle = client.create_from_slice(u32::as_bytes(&dummy_indices));
    let mut current_handle = input_handle;
    let mut current_len = len;
    let mut current_storage_len = storage_len;

    while current_len > 1 {
        let partial_len = current_len.div_ceil(BLOCK_REDUCE_SIZE as usize);
        let partial_len_u32 =
            u32::try_from(partial_len).map_err(|_| Error::LengthTooLarge { len: partial_len })?;
        let current_len_u32 =
            u32::try_from(current_len).map_err(|_| Error::LengthTooLarge { len: current_len })?;
        let len_values = [current_len_u32];
        let len_handle = client.create_from_slice(u32::as_bytes(&len_values));
        let partial_handle = client.empty(partial_len * std::mem::size_of::<T>());

        unsafe {
            reduce_expr_partials_kernel::launch_unchecked::<T, Input<T>, Op, R>(
                client,
                CubeCount::Static(partial_len_u32, 1, 1),
                CubeDim::new_1d(BLOCK_REDUCE_SIZE),
                ArrayArg::from_raw_parts::<T>(&current_handle, current_storage_len, 1),
                ArrayArg::from_raw_parts::<u32>(&dummy_index_handle, dummy_indices.len(), 1),
                ArrayArg::from_raw_parts::<T>(&current_handle, current_storage_len, 1),
                ArrayArg::from_raw_parts::<u32>(&dummy_index_handle, dummy_indices.len(), 1),
                ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
                ArrayArg::from_raw_parts::<T>(&partial_handle, partial_len, 1),
            )
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
        }

        current_handle = partial_handle;
        current_len = partial_len;
        current_storage_len = partial_len;
    }

    finalize_handle::<R, T, Op>(policy, current_handle, init)
}

pub(crate) fn reduce_device_expr<R, T, Expr, Op>(
    policy: &CubePolicy<R>,
    bindings: &KernelColumnBindings,
    len: usize,
    init: T,
) -> Result<T, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Expr: DeviceGpuExpr<T>,
    Op: BinaryOp<T>,
{
    if len == 0 {
        return Ok(init);
    }

    let client = policy.client();
    let slot0 = bindings.slots.first().unwrap();
    let slot1 = bindings.slots.get(1).unwrap_or(slot0);
    let slot2 = bindings.slots.get(2).unwrap_or(slot0);
    let slot3 = bindings.slots.get(3).unwrap_or(slot0);
    let partial_len = len.div_ceil(BLOCK_REDUCE_SIZE as usize);
    let partial_len_u32 =
        u32::try_from(partial_len).map_err(|_| Error::LengthTooLarge { len: partial_len })?;
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_values = [len_u32];
    let len_handle = client.create_from_slice(u32::as_bytes(&len_values));
    let partial_handle = client.empty(partial_len * std::mem::size_of::<T>());

    unsafe {
        device_reduce_expr_partials_kernel::launch_unchecked::<T, Expr, Op, R>(
            client,
            CubeCount::Static(partial_len_u32, 1, 1),
            CubeDim::new_1d(BLOCK_REDUCE_SIZE),
            ArrayArg::from_raw_parts::<T>(&slot0.0, slot0.1, 1),
            ArrayArg::from_raw_parts::<T>(&slot1.0, slot1.1, 1),
            ArrayArg::from_raw_parts::<T>(&slot2.0, slot2.1, 1),
            ArrayArg::from_raw_parts::<T>(&slot3.0, slot3.1, 1),
            ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
            ArrayArg::from_raw_parts::<T>(&partial_handle, partial_len, 1),
        )
        .map_err(|err| Error::Launch {
            message: format!("{err:?}"),
        })?;
    }

    reduce_input_handle::<R, T, Op>(policy, partial_handle, partial_len, partial_len, init)
}

pub(crate) fn reduce_by_key_handle<R, K, T, Op>(
    policy: &CubePolicy<R>,
    keys: &DeviceVec<R, K>,
    value_handle: cubecl::server::Handle,
    init: T,
) -> Result<(DeviceVec<R, K>, DeviceVec<R, T>), Error>
where
    R: Runtime,
    K: CubePrimitive + CubeElement + PartialEq,
    T: CubePrimitive + CubeElement,
    Op: BinaryOp<T>,
{
    let len = keys.len();
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let client = policy.client();
    if len == 0 {
        return Ok((
            DeviceVec::from_handle(policy.clone(), client.empty(0), 0),
            DeviceVec::from_handle(policy.clone(), client.empty(0), 0),
        ));
    }

    let init_handle = client.create_from_slice(T::as_bytes(&[init]));
    let inclusive_handle = scan::inclusive_scan_by_key_handle::<R, K, T, crate::op::Equal, Op>(
        policy,
        keys,
        &value_handle,
    )?;
    let flag_handle = client.empty(len * std::mem::size_of::<u32>());
    let reduced_value_handle = client.empty(len * std::mem::size_of::<T>());
    let num_blocks = len.div_ceil(BLOCK_REDUCE_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;

    unsafe {
        reduce_by_key_end_flags_kernel::launch_unchecked::<K, T, crate::op::Equal, Op, R>(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(BLOCK_REDUCE_SIZE),
            ArrayArg::from_raw_parts::<K>(&keys.handle, keys.len(), 1),
            ArrayArg::from_raw_parts::<T>(&inclusive_handle, len, 1),
            ArrayArg::from_raw_parts::<T>(&init_handle, 1, 1),
            ArrayArg::from_raw_parts::<u32>(&flag_handle, len, 1),
            ArrayArg::from_raw_parts::<T>(&reduced_value_handle, len, 1),
        )
        .map_err(|err| Error::Launch {
            message: format!("{err:?}"),
        })?;
    }

    select::compact_pair_from_flags(
        policy,
        policy,
        len,
        len_u32,
        flag_handle,
        keys.handle.clone(),
        reduced_value_handle,
    )
}

pub(crate) fn reduce_by_key_expr_handle<R, K, T, Expr, Op>(
    policy: &CubePolicy<R>,
    keys: &DeviceVec<R, K>,
    input_handle: cubecl::server::Handle,
    input_len: usize,
    rhs_handle: cubecl::server::Handle,
    rhs_len: usize,
    init: T,
) -> Result<(DeviceVec<R, K>, DeviceVec<R, T>), Error>
where
    R: Runtime,
    K: CubePrimitive + CubeElement + PartialEq,
    T: CubePrimitive + CubeElement,
    Expr: GpuExpr<T>,
    Op: BinaryOp<T>,
{
    let len = keys.len();
    let client = policy.client();
    let value_handle = client.empty(len * std::mem::size_of::<T>());
    if len != 0 {
        let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
        let block_size = 256_u32;
        let block_count = len.div_ceil(block_size as usize);
        let block_count_u32 =
            u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;
        let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
        let dummy_indices = [0_u32];
        let dummy_index_handle = client.create_from_slice(u32::as_bytes(&dummy_indices));
        unsafe {
            collect_expr_block_kernel::launch_unchecked::<T, Expr, R>(
                client,
                CubeCount::Static(block_count_u32, 1, 1),
                CubeDim::new_1d(block_size),
                ArrayArg::from_raw_parts::<T>(&value_handle, len, 1),
                ArrayArg::from_raw_parts::<T>(&input_handle, input_len, 1),
                ArrayArg::from_raw_parts::<u32>(&dummy_index_handle, dummy_indices.len(), 1),
                ArrayArg::from_raw_parts::<T>(&rhs_handle, rhs_len, 1),
                ArrayArg::from_raw_parts::<u32>(&dummy_index_handle, dummy_indices.len(), 1),
                ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
            )
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
        }
    }

    reduce_by_key_handle::<R, K, T, Op>(policy, keys, value_handle, init)
}

pub(crate) fn finalize_handle<R, T, Op>(
    policy: &CubePolicy<R>,
    partial_handle: cubecl::server::Handle,
    init: T,
) -> Result<T, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Op: BinaryOp<T>,
{
    let client = policy.client();
    let init_values = [init];
    let init_handle = client.create_from_slice(T::as_bytes(&init_values));
    let output_handle = client.empty(std::mem::size_of::<T>());

    unsafe {
        reduce_finalize_kernel::launch_unchecked::<T, Op, R>(
            client,
            CubeCount::new_single(),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts::<T>(&partial_handle, 1, 1),
            ArrayArg::from_raw_parts::<T>(&init_handle, 1, 1),
            ArrayArg::from_raw_parts::<T>(&output_handle, 1, 1),
        )
        .map_err(|err| Error::Launch {
            message: format!("{err:?}"),
        })?;
    }

    let bytes = client.read_one(output_handle);
    Ok(T::from_bytes(&bytes)[0].clone())
}
