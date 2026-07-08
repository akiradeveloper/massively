use crate::{
    device::DeviceVec,
    error::Error,
    kernels::*,
    policy::CubePolicy,
    primitives::{ensure_same_len, scan::inclusive_scan_u32, workspace::Workspace},
};
use cubecl::prelude::*;

use super::{BLOCK_ORDERING_SIZE, RADIX_DIGITS};

#[allow(dead_code)]
pub(crate) fn radix_sort_u32<R>(
    policy: &CubePolicy<R>,
    input: &DeviceVec<R, u32>,
) -> Result<DeviceVec<R, u32>, Error>
where
    R: Runtime,
{
    let client = policy.client();
    let len = input.len();
    let num_blocks = len.div_ceil(BLOCK_ORDERING_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    if len <= 1 {
        return Ok(DeviceVec::from_handle(
            policy.id(),
            input.handle.clone(),
            len,
        ));
    }

    let workspace = Workspace::new(policy);
    let (scratch_a, scratch_b) = workspace.alloc_pair::<u32>(len);
    let mut input_handle = input.handle.clone();
    let mut output_handle = scratch_a.clone();
    let mut next_uses_a = false;
    let histogram_len = num_blocks * RADIX_DIGITS;
    let histogram_handle = workspace.alloc::<u32>(histogram_len);

    for shift in (0_u32..32).step_by(4) {
        let shift_handle = client.create_from_slice(u32::as_bytes(&[shift]));
        unsafe {
            radix_digit_histogram_u32_kernel::launch_unchecked::<R>(
                client,
                crate::detail::launch::cube_count_1d(num_blocks_u32),
                CubeDim::new_1d(RADIX_DIGITS as u32),
                unsafe { BufferArg::from_raw_parts(input_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(shift_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(histogram_handle.clone(), histogram_len) },
            );
        }

        let histogram_prefix_handle =
            inclusive_scan_u32::<R>(client, &histogram_handle, histogram_len)?;
        unsafe {
            radix_digit_scatter_u32_kernel::launch_unchecked::<R>(
                client,
                crate::detail::launch::cube_count_1d(num_blocks_u32),
                CubeDim::new_1d(BLOCK_ORDERING_SIZE),
                unsafe { BufferArg::from_raw_parts(input_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(shift_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(histogram_handle.clone(), histogram_len) },
                unsafe {
                    BufferArg::from_raw_parts(histogram_prefix_handle.clone(), histogram_len)
                },
                unsafe { BufferArg::from_raw_parts(output_handle.clone(), len) },
            );
        }

        input_handle = output_handle.clone();
        output_handle = if next_uses_a {
            scratch_a.clone()
        } else {
            scratch_b.clone()
        };
        next_uses_a = !next_uses_a;
    }

    Ok(DeviceVec::from_handle(policy.id(), input_handle, len))
}

#[allow(dead_code)]
pub(crate) fn radix_sort_by_key_u32<R, T>(
    policy: &CubePolicy<R>,
    keys: &DeviceVec<R, u32>,
    values: &DeviceVec<R, T>,
) -> Result<(DeviceVec<R, u32>, DeviceVec<R, T>), Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    ensure_same_len(values.len(), keys.len())?;

    let client = policy.client();
    let len = keys.len();
    let num_blocks = len.div_ceil(BLOCK_ORDERING_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    if len <= 1 {
        return Ok((
            DeviceVec::from_handle(policy.id(), keys.handle.clone(), len),
            DeviceVec::from_handle(policy.id(), values.handle.clone(), len),
        ));
    }

    let workspace = Workspace::new(policy);
    let (scratch_keys_a, scratch_keys_b) = workspace.alloc_pair::<u32>(len);
    let (scratch_values_a, scratch_values_b) = workspace.alloc_pair::<T>(len);
    let mut input_key_handle = keys.handle.clone();
    let mut input_value_handle = values.handle.clone();
    let mut output_key_handle = scratch_keys_a.clone();
    let mut output_value_handle = scratch_values_a.clone();
    let mut next_uses_a = false;
    let histogram_len = num_blocks * RADIX_DIGITS;
    let histogram_handle = workspace.alloc::<u32>(histogram_len);

    for shift in (0_u32..32).step_by(4) {
        let shift_handle = client.create_from_slice(u32::as_bytes(&[shift]));
        unsafe {
            radix_digit_histogram_u32_kernel::launch_unchecked::<R>(
                client,
                crate::detail::launch::cube_count_1d(num_blocks_u32),
                CubeDim::new_1d(RADIX_DIGITS as u32),
                unsafe { BufferArg::from_raw_parts(input_key_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(shift_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(histogram_handle.clone(), histogram_len) },
            );
        }

        let histogram_prefix_handle =
            inclusive_scan_u32::<R>(client, &histogram_handle, histogram_len)?;
        unsafe {
            radix_digit_scatter_by_key_u32_kernel::launch_unchecked::<T, R>(
                client,
                crate::detail::launch::cube_count_1d(num_blocks_u32),
                CubeDim::new_1d(BLOCK_ORDERING_SIZE),
                unsafe { BufferArg::from_raw_parts(input_key_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(input_value_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(shift_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(histogram_handle.clone(), histogram_len) },
                unsafe {
                    BufferArg::from_raw_parts(histogram_prefix_handle.clone(), histogram_len)
                },
                unsafe { BufferArg::from_raw_parts(output_key_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(output_value_handle.clone(), len) },
            );
        }

        input_key_handle = output_key_handle.clone();
        input_value_handle = output_value_handle.clone();
        if next_uses_a {
            output_key_handle = scratch_keys_a.clone();
            output_value_handle = scratch_values_a.clone();
        } else {
            output_key_handle = scratch_keys_b.clone();
            output_value_handle = scratch_values_b.clone();
        }
        next_uses_a = !next_uses_a;
    }

    Ok((
        DeviceVec::from_handle(policy.id(), input_key_handle, len),
        DeviceVec::from_handle(policy.id(), input_value_handle, len),
    ))
}
