use crate::{
    device::DeviceVec,
    error::Error,
    kernels::*,
    op::{BinaryPredicateOp, GpuOp},
    policy::CubePolicy,
    primitives::ensure_same_len,
};
use cubecl::prelude::*;

use super::BLOCK_ORDERING_SIZE;

#[derive(Clone)]
pub(crate) struct MergeByKeyControl {
    source_sides: cubecl::server::Handle,
    source_indices: cubecl::server::Handle,
    left_len: usize,
    right_len: usize,
    len: usize,
}

pub(crate) fn merge_with_policy<R, T, Less>(
    policy: &CubePolicy<R>,
    left: &DeviceVec<R, T>,
    right: &DeviceVec<R, T>,
    _less: GpuOp<Less>,
) -> Result<DeviceVec<R, T>, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Less: BinaryPredicateOp<T>,
{
    let len = left.len() + right.len();
    if left.is_empty() {
        return Ok(DeviceVec::from_handle(
            policy.id(),
            right.handle.clone(),
            right.len(),
        ));
    }
    if right.is_empty() {
        return Ok(DeviceVec::from_handle(
            policy.id(),
            left.handle.clone(),
            left.len(),
        ));
    }

    let num_blocks = len.div_ceil(BLOCK_ORDERING_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let client = policy.client();
    let output_handle = client.empty(len * std::mem::size_of::<T>());

    if len != 0 {
        unsafe {
            merge_path_kernel::launch_unchecked::<T, Less, R>(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_ORDERING_SIZE),
                unsafe { BufferArg::from_raw_parts(output_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(left.handle.clone(), left.len()) },
                unsafe { BufferArg::from_raw_parts(right.handle.clone(), right.len()) },
            );
        }
    }

    Ok(DeviceVec::from_handle(policy.id(), output_handle, len))
}

pub(crate) fn merge_by_key_with_policy<R, K, T, Less>(
    policy: &CubePolicy<R>,
    left_keys: &DeviceVec<R, K>,
    left_values: &DeviceVec<R, T>,
    right_keys: &DeviceVec<R, K>,
    right_values: &DeviceVec<R, T>,
    _less: GpuOp<Less>,
) -> Result<(DeviceVec<R, K>, DeviceVec<R, T>), Error>
where
    R: Runtime,
    K: CubePrimitive + CubeElement,
    T: CubePrimitive + CubeElement,
    Less: BinaryPredicateOp<K>,
{
    ensure_same_len(left_values.len(), left_keys.len())?;
    ensure_same_len(right_values.len(), right_keys.len())?;

    let (keys, control) =
        merge_by_key_control_with_policy::<R, K, Less>(policy, left_keys, right_keys)?;
    let values =
        merge_by_key_values_with_control_with_policy(policy, left_values, right_values, &control)?;
    Ok((keys, values))
}

pub(crate) fn merge_by_key_control_with_policy<R, K, Less>(
    policy: &CubePolicy<R>,
    left_keys: &DeviceVec<R, K>,
    right_keys: &DeviceVec<R, K>,
) -> Result<(DeviceVec<R, K>, MergeByKeyControl), Error>
where
    R: Runtime,
    K: CubePrimitive + CubeElement,
    Less: BinaryPredicateOp<K>,
{
    let len = left_keys.len() + right_keys.len();
    let num_blocks = len.div_ceil(BLOCK_ORDERING_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let client = policy.client();
    let out_key_handle = client.empty(len * std::mem::size_of::<K>());
    let source_sides = client.empty(len * std::mem::size_of::<u32>());
    let source_indices = client.empty(len * std::mem::size_of::<u32>());

    if len != 0 {
        unsafe {
            merge_by_key_control_path_kernel::launch_unchecked::<K, Less, R>(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_ORDERING_SIZE),
                unsafe { BufferArg::from_raw_parts(left_keys.handle.clone(), left_keys.len()) },
                unsafe { BufferArg::from_raw_parts(right_keys.handle.clone(), right_keys.len()) },
                unsafe { BufferArg::from_raw_parts(out_key_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(source_sides.clone(), len) },
                unsafe { BufferArg::from_raw_parts(source_indices.clone(), len) },
            );
        }
    }

    Ok((
        DeviceVec::from_handle(policy.id(), out_key_handle, len),
        MergeByKeyControl {
            source_sides,
            source_indices,
            left_len: left_keys.len(),
            right_len: right_keys.len(),
            len,
        },
    ))
}

pub(crate) fn merge_by_key_values_with_control_with_policy<R, T>(
    policy: &CubePolicy<R>,
    left_values: &DeviceVec<R, T>,
    right_values: &DeviceVec<R, T>,
    control: &MergeByKeyControl,
) -> Result<DeviceVec<R, T>, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    ensure_same_len(left_values.len(), control.left_len)?;
    ensure_same_len(right_values.len(), control.right_len)?;

    let len = control.len;
    let num_blocks = len.div_ceil(BLOCK_ORDERING_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let client = policy.client();
    let out_value_handle = client.empty(len * std::mem::size_of::<T>());

    if len != 0 {
        unsafe {
            merge_by_key_values_from_control_kernel::launch_unchecked::<T, R>(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_ORDERING_SIZE),
                unsafe { BufferArg::from_raw_parts(left_values.handle.clone(), left_values.len()) },
                unsafe {
                    BufferArg::from_raw_parts(right_values.handle.clone(), right_values.len())
                },
                unsafe { BufferArg::from_raw_parts(control.source_sides.clone(), len) },
                unsafe { BufferArg::from_raw_parts(control.source_indices.clone(), len) },
                unsafe { BufferArg::from_raw_parts(out_value_handle.clone(), len) },
            );
        }
    }

    Ok(DeviceVec::from_handle(policy.id(), out_value_handle, len))
}
