use crate::{
    device::DeviceVec,
    error::Error,
    kernels::*,
    policy::CubePolicy,
    primitives::scan::{inclusive_scan_u32, read_u32_scalar},
};
use cubecl::prelude::*;

pub(crate) use crate::detail::control::{SelectionControl, SelectionHandles};

const BLOCK_SELECT_SIZE: u32 = 256;

fn select_block_count(len: usize) -> Result<u32, Error> {
    let block_count = len.div_ceil(BLOCK_SELECT_SIZE as usize);
    u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })
}

pub(crate) fn handles_from_flags<R>(
    policy: &CubePolicy<R>,
    len: usize,
    len_u32: u32,
    flag_handle: cubecl::server::Handle,
    value_handle: cubecl::server::Handle,
) -> Result<SelectionHandles, Error>
where
    R: Runtime,
{
    if len == 0 {
        return Ok(SelectionHandles::empty(policy.client()));
    }

    let client = policy.client();
    let position_handle = inclusive_scan_u32::<R>(client, &flag_handle, len)?;
    let count_handle = client.empty(std::mem::size_of::<u32>());

    unsafe {
        compact_count_kernel::launch_unchecked::<R>(
            client,
            CubeCount::new_single(),
            CubeDim::new_1d(1),
            unsafe { BufferArg::from_raw_parts(position_handle.clone(), len) },
            unsafe { BufferArg::from_raw_parts(count_handle.clone(), 1) },
        );
    }

    Ok(SelectionControl {
        flag: flag_handle,
        position: position_handle,
        count: count_handle,
        len,
        len_u32,
    }
    .for_value(value_handle))
}

pub(crate) fn compact<R, T>(
    policy: &CubePolicy<R>,
    handles: SelectionHandles,
) -> Result<DeviceVec<R, T>, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    let count = selected_count(policy, &handles)?;
    compact_with_count(policy, handles, count)
}

pub(crate) fn selected_count<R>(
    policy: &CubePolicy<R>,
    handles: &SelectionControl,
) -> Result<usize, Error>
where
    R: Runtime,
{
    if handles.len == 0 {
        return Ok(0);
    }
    Ok(read_u32_scalar::<R>(policy.client(), handles.count.clone())? as usize)
}

pub(crate) fn compact_with_count<R, T>(
    policy: &CubePolicy<R>,
    handles: SelectionHandles,
    count: usize,
) -> Result<DeviceVec<R, T>, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    if handles.len == 0 {
        return Ok(policy.empty_device_vec());
    }

    let client = policy.client();
    if count == 0 {
        return Ok(policy.empty_device_vec());
    }
    let output_handle = client.empty(count * std::mem::size_of::<T>());

    let block_count_u32 = select_block_count(handles.len)?;
    unsafe {
        compact_scatter_kernel::launch_unchecked::<T, R>(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SELECT_SIZE),
            unsafe { BufferArg::from_raw_parts(handles.flag.clone(), handles.len) },
            unsafe { BufferArg::from_raw_parts(handles.position.clone(), handles.len) },
            unsafe { BufferArg::from_raw_parts(handles.value.clone(), handles.len) },
            unsafe { BufferArg::from_raw_parts(output_handle.clone(), count) },
        );
    }

    Ok(DeviceVec::from_handle(policy.id(), output_handle, count))
}

#[allow(dead_code)]
pub(crate) fn handles_for_value(
    control: &SelectionControl,
    value_handle: cubecl::server::Handle,
) -> SelectionHandles {
    control.for_value(value_handle)
}
