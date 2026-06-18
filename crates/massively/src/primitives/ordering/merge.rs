use crate::{
    device::DeviceVec,
    error::Error,
    kernels::*,
    op::{BinaryPredicateOp, GpuOp},
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

pub(crate) fn merge<R, T, Less>(
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
    let num_blocks = len.div_ceil(BLOCK_ORDERING_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let client = left.policy().client();
    let output_handle = client.empty(len * std::mem::size_of::<T>());

    if len != 0 {
        unsafe {
            merge_path_kernel::launch_unchecked::<T, Less, R>(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_ORDERING_SIZE),
                ArrayArg::from_raw_parts::<T>(&output_handle, len, 1),
                ArrayArg::from_raw_parts::<T>(&left.handle, left.len(), 1),
                ArrayArg::from_raw_parts::<T>(&right.handle, right.len(), 1),
            )
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
        }
    }

    Ok(DeviceVec::from_handle(
        left.policy().clone(),
        output_handle,
        len,
    ))
}

pub(crate) fn merge_by_key<R, K, T, Less>(
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

    let (keys, control) = merge_by_key_control::<R, K, Less>(left_keys, right_keys)?;
    let values = merge_by_key_values_with_control(left_values, right_values, &control)?;
    Ok((keys, values))
}

pub(crate) fn merge_by_key_control<R, K, Less>(
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
    let client = left_keys.policy().client();
    let out_key_handle = client.empty(len * std::mem::size_of::<K>());
    let source_sides = client.empty(len * std::mem::size_of::<u32>());
    let source_indices = client.empty(len * std::mem::size_of::<u32>());

    if len != 0 {
        unsafe {
            merge_by_key_control_path_kernel::launch_unchecked::<K, Less, R>(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_ORDERING_SIZE),
                ArrayArg::from_raw_parts::<K>(&left_keys.handle, left_keys.len(), 1),
                ArrayArg::from_raw_parts::<K>(&right_keys.handle, right_keys.len(), 1),
                ArrayArg::from_raw_parts::<K>(&out_key_handle, len, 1),
                ArrayArg::from_raw_parts::<u32>(&source_sides, len, 1),
                ArrayArg::from_raw_parts::<u32>(&source_indices, len, 1),
            )
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
        }
    }

    Ok((
        DeviceVec::from_handle(left_keys.policy().clone(), out_key_handle, len),
        MergeByKeyControl {
            source_sides,
            source_indices,
            left_len: left_keys.len(),
            right_len: right_keys.len(),
            len,
        },
    ))
}

pub(crate) fn merge_by_key_values_with_control<R, T>(
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
    let client = left_values.policy().client();
    let out_value_handle = client.empty(len * std::mem::size_of::<T>());

    if len != 0 {
        unsafe {
            merge_by_key_values_from_control_kernel::launch_unchecked::<T, R>(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_ORDERING_SIZE),
                ArrayArg::from_raw_parts::<T>(&left_values.handle, left_values.len(), 1),
                ArrayArg::from_raw_parts::<T>(&right_values.handle, right_values.len(), 1),
                ArrayArg::from_raw_parts::<u32>(&control.source_sides, len, 1),
                ArrayArg::from_raw_parts::<u32>(&control.source_indices, len, 1),
                ArrayArg::from_raw_parts::<T>(&out_value_handle, len, 1),
            )
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
        }
    }

    Ok(DeviceVec::from_handle(
        left_values.policy().clone(),
        out_value_handle,
        len,
    ))
}
