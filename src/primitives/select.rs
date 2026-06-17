use crate::{
    device::DeviceVec,
    error::Error,
    kernels::*,
    op::{BinaryPredicateOp, GpuOp, PredicateOp},
    policy::CubePolicy,
    primitives::scan::{inclusive_scan_u32, read_u32_scalar},
};
use cubecl::prelude::*;

const BLOCK_SELECT_SIZE: u32 = 256;

fn select_block_count(len: usize) -> Result<u32, Error> {
    let block_count = len.div_ceil(BLOCK_SELECT_SIZE as usize);
    u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })
}

#[derive(Clone)]
pub(crate) struct SelectionHandles {
    pub(crate) flag: cubecl::server::Handle,
    pub(crate) position: cubecl::server::Handle,
    pub(crate) value: cubecl::server::Handle,
    pub(crate) count: cubecl::server::Handle,
    pub(crate) len: usize,
    pub(crate) len_u32: u32,
}

impl SelectionHandles {
    fn empty<R: Runtime>(client: &ComputeClient<R>) -> Self {
        Self {
            flag: client.empty(0),
            position: client.empty(0),
            value: client.empty(0),
            count: client.empty(std::mem::size_of::<u32>()),
            len: 0,
            len_u32: 0,
        }
    }
}

pub(crate) fn partition<R, T, Pred>(
    input: &DeviceVec<R, T>,
    _pred: GpuOp<Pred>,
) -> Result<DeviceVec<R, T>, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Pred: PredicateOp<T>,
{
    let handles = predicate_handles::<R, T, Pred>(input, false)?;
    partition_from_handles(input.policy(), handles)
}

pub(crate) fn partition_copy<R, T, Pred>(
    input: &DeviceVec<R, T>,
    _pred: GpuOp<Pred>,
) -> Result<(DeviceVec<R, T>, DeviceVec<R, T>), Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Pred: PredicateOp<T>,
{
    let selected_handles = predicate_handles::<R, T, Pred>(input, false)?;
    if selected_handles.len == 0 {
        let empty_selected =
            DeviceVec::from_handle(input.policy().clone(), input.policy().client().empty(0), 0);
        let empty_rejected =
            DeviceVec::from_handle(input.policy().clone(), input.policy().client().empty(0), 0);
        return Ok((empty_selected, empty_rejected));
    }

    let client = input.policy().client();
    let block_count_u32 = select_block_count(selected_handles.len)?;
    let rejected_flag_handle = client.empty(selected_handles.len * std::mem::size_of::<u32>());
    unsafe {
        invert_flags_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SELECT_SIZE),
            ArrayArg::from_raw_parts::<u32>(&selected_handles.flag, selected_handles.len, 1),
            ArrayArg::from_raw_parts::<u32>(&rejected_flag_handle, selected_handles.len, 1),
        )
        .map_err(|err| Error::Launch {
            message: format!("{err:?}"),
        })?;
    }

    let rejected_handles = handles_from_flags(
        input.policy(),
        selected_handles.len,
        selected_handles.len_u32,
        rejected_flag_handle,
        selected_handles.value.clone(),
    )?;

    Ok((
        compact(input.policy(), selected_handles)?,
        compact(input.policy(), rejected_handles)?,
    ))
}

pub(crate) fn unique<R, T, Pred>(
    input: &DeviceVec<R, T>,
    _pred: GpuOp<Pred>,
) -> Result<DeviceVec<R, T>, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Pred: BinaryPredicateOp<T>,
{
    let len_u32 =
        u32::try_from(input.len()).map_err(|_| Error::LengthTooLarge { len: input.len() })?;
    if input.is_empty() {
        return Ok(DeviceVec::from_handle(
            input.policy().clone(),
            input.policy().client().empty(0),
            0,
        ));
    }

    let client = input.policy().client();
    let block_count_u32 = select_block_count(input.len())?;
    let flag_handle = client.empty(input.len() * std::mem::size_of::<u32>());

    unsafe {
        unique_flags_kernel::launch_unchecked::<T, Pred, R>(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SELECT_SIZE),
            ArrayArg::from_raw_parts::<T>(&input.handle, input.len(), 1),
            ArrayArg::from_raw_parts::<u32>(&flag_handle, input.len(), 1),
        )
        .map_err(|err| Error::Launch {
            message: format!("{err:?}"),
        })?;
    }

    let handles = handles_from_flags(
        input.policy(),
        input.len(),
        len_u32,
        flag_handle,
        input.handle.clone(),
    )?;
    compact(input.policy(), handles)
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
            ArrayArg::from_raw_parts::<u32>(&position_handle, len, 1),
            ArrayArg::from_raw_parts::<u32>(&count_handle, 1, 1),
        )
        .map_err(|err| Error::Launch {
            message: format!("{err:?}"),
        })?;
    }

    Ok(SelectionHandles {
        flag: flag_handle,
        position: position_handle,
        value: value_handle,
        count: count_handle,
        len,
        len_u32,
    })
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
    handles: &SelectionHandles,
) -> Result<usize, Error>
where
    R: Runtime,
{
    if handles.len == 0 {
        return Ok(0);
    }
    Ok(read_u32_scalar::<R>(policy.client(), handles.count.clone()) as usize)
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
        return Ok(DeviceVec::from_handle(
            policy.clone(),
            policy.client().empty(0),
            0,
        ));
    }

    let client = policy.client();
    let output_handle = client.empty(count * std::mem::size_of::<T>());

    if count != 0 {
        let block_count_u32 = select_block_count(handles.len)?;
        unsafe {
            compact_scatter_kernel::launch_unchecked::<T, R>(
                client,
                CubeCount::Static(block_count_u32, 1, 1),
                CubeDim::new_1d(BLOCK_SELECT_SIZE),
                ArrayArg::from_raw_parts::<u32>(&handles.flag, handles.len, 1),
                ArrayArg::from_raw_parts::<u32>(&handles.position, handles.len, 1),
                ArrayArg::from_raw_parts::<T>(&handles.value, handles.len, 1),
                ArrayArg::from_raw_parts::<T>(&output_handle, count, 1),
            )
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
        }
    }

    Ok(DeviceVec::from_handle(policy.clone(), output_handle, count))
}

pub(crate) fn handles_for_value(
    control: &SelectionHandles,
    value_handle: cubecl::server::Handle,
) -> SelectionHandles {
    SelectionHandles {
        flag: control.flag.clone(),
        position: control.position.clone(),
        value: value_handle,
        count: control.count.clone(),
        len: control.len,
        len_u32: control.len_u32,
    }
}

fn predicate_handles<R, T, Pred>(
    input: &DeviceVec<R, T>,
    invert: bool,
) -> Result<SelectionHandles, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Pred: PredicateOp<T>,
{
    let len_u32 =
        u32::try_from(input.len()).map_err(|_| Error::LengthTooLarge { len: input.len() })?;
    if input.is_empty() {
        return Ok(SelectionHandles::empty(input.policy().client()));
    }

    let client = input.policy().client();
    let block_count_u32 = select_block_count(input.len())?;
    let invert_values = [if invert { 1_u32 } else { 0_u32 }];
    let invert_handle = client.create_from_slice(u32::as_bytes(&invert_values));
    let flag_handle = client.empty(input.len() * std::mem::size_of::<u32>());

    unsafe {
        copy_if_flag_only_kernel::launch_unchecked::<T, Pred, R>(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SELECT_SIZE),
            ArrayArg::from_raw_parts::<T>(&input.handle, input.len(), 1),
            ArrayArg::from_raw_parts::<u32>(&invert_handle, 1, 1),
            ArrayArg::from_raw_parts::<u32>(&flag_handle, input.len(), 1),
        )
        .map_err(|err| Error::Launch {
            message: format!("{err:?}"),
        })?;
    }

    handles_from_flags(
        input.policy(),
        input.len(),
        len_u32,
        flag_handle,
        input.handle.clone(),
    )
}

pub(crate) fn partition_from_handles<R, T>(
    policy: &CubePolicy<R>,
    handles: SelectionHandles,
) -> Result<DeviceVec<R, T>, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    if handles.len == 0 {
        return Ok(DeviceVec::from_handle(
            policy.clone(),
            policy.client().empty(0),
            0,
        ));
    }

    let client = policy.client();
    let block_count_u32 = select_block_count(handles.len)?;
    let output_handle = client.empty(handles.len * std::mem::size_of::<T>());

    unsafe {
        partition_scatter_kernel::launch_unchecked::<T, R>(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SELECT_SIZE),
            ArrayArg::from_raw_parts::<u32>(&handles.flag, handles.len, 1),
            ArrayArg::from_raw_parts::<u32>(&handles.position, handles.len, 1),
            ArrayArg::from_raw_parts::<u32>(&handles.count, 1, 1),
            ArrayArg::from_raw_parts::<T>(&handles.value, handles.len, 1),
            ArrayArg::from_raw_parts::<T>(&output_handle, handles.len, 1),
        )
        .map_err(|err| Error::Launch {
            message: format!("{err:?}"),
        })?;
    }

    Ok(DeviceVec::from_handle(
        policy.clone(),
        output_handle,
        handles.len,
    ))
}
