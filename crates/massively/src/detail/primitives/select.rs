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
pub(crate) struct SelectionControl {
    pub(crate) flag: cubecl::server::Handle,
    pub(crate) position: cubecl::server::Handle,
    pub(crate) value: cubecl::server::Handle,
    pub(crate) count: cubecl::server::Handle,
    pub(crate) len: usize,
    pub(crate) len_u32: u32,
}

pub(crate) type SelectionHandles = SelectionControl;

impl SelectionControl {
    fn empty<R: Runtime>(client: &ComputeClient<R>) -> Self {
        Self {
            flag: crate::policy::empty_handle(client),
            position: crate::policy::empty_handle(client),
            value: crate::policy::empty_handle(client),
            count: client.empty(std::mem::size_of::<u32>()),
            len: 0,
            len_u32: 0,
        }
    }

    pub(crate) fn for_value(&self, value_handle: cubecl::server::Handle) -> Self {
        Self {
            flag: self.flag.clone(),
            position: self.position.clone(),
            value: value_handle,
            count: self.count.clone(),
            len: self.len,
            len_u32: self.len_u32,
        }
    }
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
        let empty_selected = DeviceVec::empty(input.policy().clone());
        let empty_rejected = DeviceVec::empty(input.policy().clone());
        return Ok((empty_selected, empty_rejected));
    }

    let selected_count = selected_count(input.policy(), &selected_handles)?;
    let rejected_count = selected_handles.len - selected_count;

    Ok((
        compact_with_count(input.policy(), selected_handles.clone(), selected_count)?,
        compact_rejected_with_count(input.policy(), selected_handles, rejected_count)?,
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
        return Ok(DeviceVec::empty(input.policy().clone()));
    }

    let client = input.policy().client();
    let block_count_u32 = select_block_count(input.len())?;
    let flag_handle = client.empty(input.len() * std::mem::size_of::<u32>());

    unsafe {
        unique_flags_kernel::launch_unchecked::<T, Pred, R>(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SELECT_SIZE),
            unsafe { BufferArg::from_raw_parts(input.handle.clone(), input.len()) },
            unsafe { BufferArg::from_raw_parts(flag_handle.clone(), input.len()) },
        );
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
) -> Result<SelectionControl, Error>
where
    R: Runtime,
{
    if len == 0 {
        return Ok(SelectionControl::empty(policy.client()));
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
        value: value_handle,
        count: count_handle,
        len,
        len_u32,
    })
}

pub(crate) fn compact<R, T>(
    policy: &CubePolicy<R>,
    handles: SelectionControl,
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
    handles: SelectionControl,
    count: usize,
) -> Result<DeviceVec<R, T>, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    if handles.len == 0 {
        return Ok(DeviceVec::empty(policy.clone()));
    }

    let client = policy.client();
    if count == 0 {
        return Ok(DeviceVec::empty(policy.clone()));
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

    Ok(DeviceVec::from_handle(policy.clone(), output_handle, count))
}

pub(crate) fn compact_rejected_with_count<R, T>(
    policy: &CubePolicy<R>,
    handles: SelectionControl,
    count: usize,
) -> Result<DeviceVec<R, T>, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    if handles.len == 0 || count == 0 {
        return Ok(DeviceVec::empty(policy.clone()));
    }

    let client = policy.client();
    let output_handle = client.empty(count * std::mem::size_of::<T>());
    let block_count_u32 = select_block_count(handles.len)?;

    unsafe {
        compact_rejected_scatter_kernel::launch_unchecked::<T, R>(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SELECT_SIZE),
            unsafe { BufferArg::from_raw_parts(handles.flag.clone(), handles.len) },
            unsafe { BufferArg::from_raw_parts(handles.position.clone(), handles.len) },
            unsafe { BufferArg::from_raw_parts(handles.value.clone(), handles.len) },
            unsafe { BufferArg::from_raw_parts(output_handle.clone(), count) },
        );
    }

    Ok(DeviceVec::from_handle(policy.clone(), output_handle, count))
}

pub(crate) fn compact_pair_with_count<R, A, B>(
    policy: &CubePolicy<R>,
    handles: &SelectionControl,
    first_value_handle: cubecl::server::Handle,
    second_value_handle: cubecl::server::Handle,
    count: usize,
) -> Result<(DeviceVec<R, A>, DeviceVec<R, B>), Error>
where
    R: Runtime,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
{
    if handles.len == 0 || count == 0 {
        return Ok((
            DeviceVec::empty(policy.clone()),
            DeviceVec::empty(policy.clone()),
        ));
    }

    let client = policy.client();
    let first_output_handle = client.empty(count * std::mem::size_of::<A>());
    let second_output_handle = client.empty(count * std::mem::size_of::<B>());

    let block_count_u32 = select_block_count(handles.len)?;
    unsafe {
        compact_scatter_pair_kernel::launch_unchecked::<A, B, R>(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SELECT_SIZE),
            unsafe { BufferArg::from_raw_parts(handles.flag.clone(), handles.len) },
            unsafe { BufferArg::from_raw_parts(handles.position.clone(), handles.len) },
            unsafe { BufferArg::from_raw_parts(first_value_handle.clone(), handles.len) },
            unsafe { BufferArg::from_raw_parts(second_value_handle.clone(), handles.len) },
            unsafe { BufferArg::from_raw_parts(first_output_handle.clone(), count) },
            unsafe { BufferArg::from_raw_parts(second_output_handle.clone(), count) },
        );
    }

    Ok((
        DeviceVec::from_handle(policy.clone(), first_output_handle, count),
        DeviceVec::from_handle(policy.clone(), second_output_handle, count),
    ))
}

pub(crate) fn handles_for_value(
    control: &SelectionControl,
    value_handle: cubecl::server::Handle,
) -> SelectionControl {
    control.for_value(value_handle)
}

fn predicate_handles<R, T, Pred>(
    input: &DeviceVec<R, T>,
    invert: bool,
) -> Result<SelectionControl, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Pred: PredicateOp<T>,
{
    let len_u32 =
        u32::try_from(input.len()).map_err(|_| Error::LengthTooLarge { len: input.len() })?;
    if input.is_empty() {
        return Ok(SelectionControl::empty(input.policy().client()));
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
            unsafe { BufferArg::from_raw_parts(input.handle.clone(), input.len()) },
            unsafe { BufferArg::from_raw_parts(invert_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(flag_handle.clone(), input.len()) },
        );
    }

    handles_from_flags(
        input.policy(),
        input.len(),
        len_u32,
        flag_handle,
        input.handle.clone(),
    )
}
