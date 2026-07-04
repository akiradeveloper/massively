use crate::{
    device::DeviceVec,
    error::Error,
    kernels::*,
    policy::CubePolicy,
    primitives::scan::{inclusive_scan_u32, read_u32_scalar},
};
use cubecl::prelude::*;

pub(crate) use crate::detail::control::{MaskControl, SelectedRankControl, SplitRankControl};

const BLOCK_SELECT_SIZE: u32 = 256;

fn select_block_count(len: usize) -> Result<u32, Error> {
    let block_count = len.div_ceil(BLOCK_SELECT_SIZE as usize);
    u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })
}

pub(crate) fn selected_rank_from_flags<R>(
    policy: &CubePolicy<R>,
    len: usize,
    len_u32: u32,
    flag_handle: cubecl::server::Handle,
) -> Result<SelectedRankControl, Error>
where
    R: Runtime,
{
    if len == 0 {
        return Ok(SelectedRankControl::empty(policy.client()));
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

    Ok(SelectedRankControl::from_parts(
        flag_handle,
        position_handle,
        count_handle,
        len,
        len_u32,
    ))
}

#[allow(dead_code)]
pub(crate) fn mask_from_flags(
    flag_handle: cubecl::server::Handle,
    len: usize,
    len_u32: u32,
) -> MaskControl {
    MaskControl::from_flags(flag_handle, len, len_u32)
}

pub(crate) fn selected_count<R>(
    policy: &CubePolicy<R>,
    control: &SelectedRankControl,
) -> Result<usize, Error>
where
    R: Runtime,
{
    if control.len == 0 {
        return Ok(0);
    }
    Ok(read_u32_scalar::<R>(policy.client(), control.count.clone())? as usize)
}

#[allow(dead_code)]
pub(crate) fn split_rank_from_flags<R>(
    policy: &CubePolicy<R>,
    len: usize,
    len_u32: u32,
    flag_handle: cubecl::server::Handle,
) -> Result<(SplitRankControl, usize, usize), Error>
where
    R: Runtime,
{
    let selected = selected_rank_from_flags(policy, len, len_u32, flag_handle)?;
    split_rank_from_selected(policy, selected)
}

pub(crate) fn split_rank_from_selected<R>(
    policy: &CubePolicy<R>,
    selected: SelectedRankControl,
) -> Result<(SplitRankControl, usize, usize), Error>
where
    R: Runtime,
{
    if selected.len == 0 {
        return Ok((SplitRankControl::empty(policy.client()), 0, 0));
    }

    let selected_count = selected_count(policy, &selected)?;
    let rejected_count = selected.len - selected_count;
    let rejected_count_u32 = u32::try_from(rejected_count).map_err(|_| Error::LengthTooLarge {
        len: rejected_count,
    })?;
    let rejected_count_handle = policy
        .client()
        .create_from_slice(u32::as_bytes(&[rejected_count_u32]));
    Ok((
        SplitRankControl::from_selected_rank(selected, rejected_count_handle),
        selected_count,
        rejected_count,
    ))
}

#[allow(dead_code)]
pub(crate) fn split_selected_count<R>(
    policy: &CubePolicy<R>,
    control: &SplitRankControl,
) -> Result<usize, Error>
where
    R: Runtime,
{
    if control.len == 0 {
        return Ok(0);
    }
    Ok(read_u32_scalar::<R>(policy.client(), control.selected_count.clone())? as usize)
}

#[allow(dead_code)]
pub(crate) fn split_rejected_count<R>(
    policy: &CubePolicy<R>,
    control: &SplitRankControl,
) -> Result<usize, Error>
where
    R: Runtime,
{
    if control.len == 0 {
        return Ok(0);
    }
    Ok(read_u32_scalar::<R>(policy.client(), control.rejected_count.clone())? as usize)
}

pub(crate) fn compact_value_with_count<R, T>(
    policy: &CubePolicy<R>,
    control: &SelectedRankControl,
    value_handle: cubecl::server::Handle,
    count: usize,
) -> Result<DeviceVec<R, T>, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    if control.len == 0 {
        return Ok(policy.empty_device_vec());
    }

    let client = policy.client();
    if count == 0 {
        return Ok(policy.empty_device_vec());
    }
    let control_len = control.len;
    let output_handle = client.empty(count * std::mem::size_of::<T>());

    let block_count_u32 = select_block_count(control.len)?;
    unsafe {
        compact_scatter_kernel::launch_unchecked::<T, R>(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SELECT_SIZE),
            unsafe { BufferArg::from_raw_parts(control.flag.clone(), control_len) },
            unsafe { BufferArg::from_raw_parts(control.position.clone(), control_len) },
            unsafe { BufferArg::from_raw_parts(value_handle, control_len) },
            unsafe { BufferArg::from_raw_parts(output_handle.clone(), count) },
        );
    }

    Ok(DeviceVec::from_handle(policy.id(), output_handle, count))
}
