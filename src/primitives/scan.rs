use crate::{
    device::{DeviceVec, KernelColumnBindings},
    error::Error,
    expr::{DeviceGpuExpr, GpuExpr, Input},
    kernels::*,
    op::{BinaryOp, BinaryPredicateOp, GpuOp},
    policy::CubePolicy,
};
use cubecl::prelude::*;

pub(crate) const BLOCK_SCAN_SIZE: u32 = 256;

pub(crate) fn inclusive_scan_u32<R: Runtime>(
    client: &ComputeClient<R>,
    input_handle: &cubecl::server::Handle,
    len: usize,
) -> Result<cubecl::server::Handle, Error> {
    let output_handle = client.empty(len * std::mem::size_of::<u32>());
    if len == 0 {
        return Ok(output_handle);
    }

    let num_blocks = len.div_ceil(BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_values = [len_u32];
    let len_handle = client.create_from_slice(u32::as_bytes(&len_values));
    let block_sums_handle = client.empty(num_blocks * std::mem::size_of::<u32>());

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
    let output_handle = client.empty(len * std::mem::size_of::<T>());
    if len == 0 {
        return Ok(output_handle);
    }

    let num_blocks = len.div_ceil(BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_values = [len_u32];
    let len_handle = client.create_from_slice(u32::as_bytes(&len_values));
    let dummy_indices = [0u32];
    let dummy_index_handle = client.create_from_slice(u32::as_bytes(&dummy_indices));
    let block_sums_handle = client.empty(num_blocks * std::mem::size_of::<T>());

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
    if keys.len() != values.len() {
        return Err(Error::LengthMismatch {
            input: values.len(),
            output: keys.len(),
        });
    }

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
    if keys.len() != values.len() {
        return Err(Error::LengthMismatch {
            input: values.len(),
            output: keys.len(),
        });
    }

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
    let output_handle = client.empty(len * std::mem::size_of::<T>());
    if len == 0 {
        return Ok(output_handle);
    }

    let slot0 = bindings.slots.first().unwrap();
    let slot1 = bindings.slots.get(1).unwrap_or(slot0);
    let slot2 = bindings.slots.get(2).unwrap_or(slot0);
    let slot3 = bindings.slots.get(3).unwrap_or(slot0);
    let num_blocks = len.div_ceil(BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let block_sums_handle = client.empty(num_blocks * std::mem::size_of::<T>());

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
        return Ok(client.empty(0));
    }
    if len == 1 {
        return Ok(value_handle.clone());
    }

    u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let num_blocks = len.div_ceil(BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let scratch_a = client.empty(len * std::mem::size_of::<T>());
    let scratch_b = client.empty(len * std::mem::size_of::<T>());
    let mut input_handle = value_handle.clone();
    let mut output_handle = scratch_a.clone();
    let mut next_uses_a = false;
    let mut offset = 1usize;
    while offset < len {
        let offset_u32 =
            u32::try_from(offset).map_err(|_| Error::LengthTooLarge { len: offset })?;
        let offset_handle = client.create_from_slice(u32::as_bytes(&[offset_u32]));
        unsafe {
            scan_by_key_pass_kernel::launch_unchecked::<K, T, KeyEq, Op, R>(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_SCAN_SIZE),
                ArrayArg::from_raw_parts::<K>(&keys.handle, len, 1),
                ArrayArg::from_raw_parts::<T>(&input_handle, len, 1),
                ArrayArg::from_raw_parts::<u32>(&offset_handle, 1, 1),
                ArrayArg::from_raw_parts::<T>(&output_handle, len, 1),
            )
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
        }
        input_handle = output_handle.clone();
        output_handle = if next_uses_a {
            scratch_a.clone()
        } else {
            scratch_b.clone()
        };
        next_uses_a = !next_uses_a;
        offset *= 2;
    }

    Ok(input_handle)
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
    let output_handle = client.empty(len * std::mem::size_of::<T>());
    if len == 0 {
        return Ok(output_handle);
    }

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
    let output_handle = client.empty(len * std::mem::size_of::<T>());
    if len == 0 {
        return Ok(output_handle);
    }

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
