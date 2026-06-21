use crate::{
    device::DeviceVec,
    error::Error,
    kernels::*,
    op::{BinaryPredicateOp, GpuOp},
    policy::CubePolicy,
    primitives::select,
};
use cubecl::prelude::*;

use super::BLOCK_ORDERING_SIZE;

pub(crate) fn set_union_with_policy<R, T, Less>(
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
    let right_only = set_difference_with_policy(policy, right, left, GpuOp::<Less>::new())?;
    super::merge_with_policy(policy, left, &right_only, GpuOp::<Less>::new())
}

pub(crate) fn set_intersection_with_policy<R, T, Less>(
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
    membership_compact::<R, T, Less>(policy, left, right, true)
}

pub(crate) fn set_difference_with_policy<R, T, Less>(
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
    membership_compact::<R, T, Less>(policy, left, right, false)
}

fn membership_compact<R, T, Less>(
    policy: &CubePolicy<R>,
    candidates: &DeviceVec<R, T>,
    sorted: &DeviceVec<R, T>,
    keep_present: bool,
) -> Result<DeviceVec<R, T>, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Less: BinaryPredicateOp<T>,
{
    let len_u32 = u32::try_from(candidates.len()).map_err(|_| Error::LengthTooLarge {
        len: candidates.len(),
    })?;
    if candidates.len() == 0 {
        return Ok(policy.empty_device_vec());
    }

    let client = policy.client();
    let num_blocks = candidates.len().div_ceil(BLOCK_ORDERING_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let keep_values = [if keep_present { 1_u32 } else { 0_u32 }];
    let keep_handle = client.create_from_slice(u32::as_bytes(&keep_values));
    let flag_handle = client.empty(candidates.len() * std::mem::size_of::<u32>());
    unsafe {
        sorted_membership_flags_kernel::launch_unchecked::<T, Less, R>(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(BLOCK_ORDERING_SIZE),
            unsafe { BufferArg::from_raw_parts(candidates.handle.clone(), candidates.len()) },
            unsafe { BufferArg::from_raw_parts(sorted.handle.clone(), sorted.len()) },
            unsafe { BufferArg::from_raw_parts(keep_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(flag_handle.clone(), candidates.len()) },
        );
    }

    let handles = select::handles_from_flags(
        policy,
        candidates.len(),
        len_u32,
        flag_handle,
        candidates.handle.clone(),
    )?;
    select::compact(policy, handles)
}
