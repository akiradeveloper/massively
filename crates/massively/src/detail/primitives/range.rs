use crate::{device::DeviceVec, error::Error, kernels::*, policy::CubePolicy};
use cubecl::prelude::*;

const BLOCK_RANGE_SIZE: u32 = 256;

pub(crate) fn to_device<R, T>(policy: &CubePolicy<R>, input: &[T]) -> Result<DeviceVec<R, T>, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    u32::try_from(input.len()).map_err(|_| Error::LengthTooLarge { len: input.len() })?;
    if input.is_empty() {
        return Ok(policy.empty_device_vec());
    }
    let handle = policy.client().create_from_slice(T::as_bytes(input));
    Ok(DeviceVec::from_handle(policy.id(), handle, input.len()))
}

pub(crate) fn filled<R, T>(
    policy: &CubePolicy<R>,
    len: usize,
    value: T,
) -> Result<DeviceVec<R, T>, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    if len == 0 {
        return Ok(policy.empty_device_vec());
    }

    let client = policy.client();
    let output_handle = client.empty(len * std::mem::size_of::<T>());

    let block_count = len.div_ceil(BLOCK_RANGE_SIZE as usize);
    let block_count_u32 =
        u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;
    let values = [value];
    let lengths = [len_u32];
    let value_handle = client.create_from_slice(T::as_bytes(&values));
    let len_handle = client.create_from_slice(u32::as_bytes(&lengths));

    unsafe {
        fill_kernel::launch_unchecked::<T, R>(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_RANGE_SIZE),
            unsafe { BufferArg::from_raw_parts(value_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output_handle.clone(), len) },
        );
    }

    Ok(DeviceVec::from_handle(policy.id(), output_handle, len))
}

pub(crate) fn tabulate<R, T, Op>(
    policy: &CubePolicy<R>,
    len: usize,
    _op: Op,
) -> Result<DeviceVec<R, T>, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Op: crate::op::kernel::UnaryOp<u32, Output = T>,
{
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    if len == 0 {
        return Ok(policy.empty_device_vec());
    }

    let client = policy.client();
    let output_handle = client.empty(len * std::mem::size_of::<T>());

    let block_count = len.div_ceil(BLOCK_RANGE_SIZE as usize);
    let block_count_u32 =
        u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));

    unsafe {
        tabulate_kernel::launch_unchecked::<T, Op, R>(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_RANGE_SIZE),
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output_handle.clone(), len) },
        );
    }

    Ok(DeviceVec::from_handle(policy.id(), output_handle, len))
}

#[allow(dead_code)]
pub(crate) fn concat_device_with_policy<R, T>(
    policy: &CubePolicy<R>,
    left: &DeviceVec<R, T>,
    right: &DeviceVec<R, T>,
) -> Result<DeviceVec<R, T>, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    let len = left.len() + right.len();
    u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    if len == 0 {
        return Ok(policy.empty_device_vec());
    }

    let client = policy.client();
    let output_handle = client.empty(len * std::mem::size_of::<T>());

    let block_count = len.div_ceil(BLOCK_RANGE_SIZE as usize);
    let block_count_u32 =
        u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;
    unsafe {
        concat_kernel::launch_unchecked::<T, R>(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_RANGE_SIZE),
            unsafe { BufferArg::from_raw_parts(left.handle.clone(), left.len()) },
            unsafe { BufferArg::from_raw_parts(right.handle.clone(), right.len()) },
            unsafe { BufferArg::from_raw_parts(output_handle.clone(), len) },
        );
    }

    Ok(DeviceVec::from_handle(policy.id(), output_handle, len))
}

pub(crate) fn copy_handle<R, T>(
    policy: &CubePolicy<R>,
    input_handle: &cubecl::server::Handle,
    len: usize,
) -> Result<cubecl::server::Handle, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    if len == 0 {
        return Ok(policy.empty_handle());
    }

    let client = policy.client();
    let output_handle = client.empty(len * std::mem::size_of::<T>());

    let block_count = len.div_ceil(BLOCK_RANGE_SIZE as usize);
    let block_count_u32 =
        u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;
    unsafe {
        copy_kernel::launch_unchecked::<T, R>(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_RANGE_SIZE),
            unsafe { BufferArg::from_raw_parts(input_handle.clone(), len) },
            unsafe { BufferArg::from_raw_parts(output_handle.clone(), len) },
        );
    }

    Ok(output_handle)
}

pub(crate) fn copy_slice_with_policy<R, T>(
    policy: &CubePolicy<R>,
    input: &DeviceVec<R, T>,
    offset: usize,
    len: usize,
) -> Result<DeviceVec<R, T>, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    let end = offset
        .checked_add(len)
        .ok_or(Error::LengthTooLarge { len })?;
    if end > input.len() {
        return Err(Error::LengthMismatch {
            input: end,
            output: input.len(),
        });
    }
    let offset_u32 = u32::try_from(offset).map_err(|_| Error::LengthTooLarge { len: offset })?;
    u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    if len == 0 {
        return Ok(policy.empty_device_vec());
    }

    let client = policy.client();
    let output_handle = client.empty(len * std::mem::size_of::<T>());
    let offset_handle = client.create_from_slice(u32::as_bytes(&[offset_u32]));

    let block_count = len.div_ceil(BLOCK_RANGE_SIZE as usize);
    let block_count_u32 =
        u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;
    unsafe {
        copy_slice_kernel::launch_unchecked::<T, R>(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_RANGE_SIZE),
            unsafe { BufferArg::from_raw_parts(input.handle.clone(), input.len()) },
            unsafe { BufferArg::from_raw_parts(offset_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output_handle.clone(), len) },
        );
    }

    Ok(DeviceVec::from_handle(policy.id(), output_handle, len))
}

pub(crate) fn copy_slice_to_slice_with_policy<R, T>(
    policy: &CubePolicy<R>,
    input: &DeviceVec<R, T>,
    input_offset: usize,
    output: &DeviceVec<R, T>,
    output_offset: usize,
    len: usize,
) -> Result<(), Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    let input_end = input_offset
        .checked_add(len)
        .ok_or(Error::LengthTooLarge { len })?;
    let output_end = output_offset
        .checked_add(len)
        .ok_or(Error::LengthTooLarge { len })?;
    if input_end > input.len() {
        return Err(Error::LengthMismatch {
            input: input_end,
            output: input.len(),
        });
    }
    if output_end > output.len() {
        return Err(Error::LengthMismatch {
            input: output_end,
            output: output.len(),
        });
    }

    let input_offset_u32 =
        u32::try_from(input_offset).map_err(|_| Error::LengthTooLarge { len: input_offset })?;
    let output_offset_u32 =
        u32::try_from(output_offset).map_err(|_| Error::LengthTooLarge { len: output_offset })?;
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    if len == 0 {
        return Ok(());
    }

    let client = policy.client();
    let metadata_handle = client.create_from_slice(u32::as_bytes(&[
        input_offset_u32,
        output_offset_u32,
        len_u32,
    ]));

    let block_count = len.div_ceil(BLOCK_RANGE_SIZE as usize);
    let block_count_u32 =
        u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;
    unsafe {
        copy_slice_to_slice_kernel::launch_unchecked::<T, R>(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_RANGE_SIZE),
            unsafe { BufferArg::from_raw_parts(input.handle.clone(), input.len()) },
            unsafe { BufferArg::from_raw_parts(metadata_handle.clone(), 3) },
            unsafe { BufferArg::from_raw_parts(output.handle.clone(), output.len()) },
        );
    }

    Ok(())
}

pub(crate) fn indices_u32<R>(policy: &CubePolicy<R>, len: usize) -> Result<DeviceVec<R, u32>, Error>
where
    R: Runtime,
{
    u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    if len == 0 {
        return Ok(policy.empty_device_vec());
    }

    let client = policy.client();
    let output_handle = client.empty(len * std::mem::size_of::<u32>());

    let block_count = len.div_ceil(BLOCK_RANGE_SIZE as usize);
    let block_count_u32 =
        u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;
    unsafe {
        indices_u32_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_RANGE_SIZE),
            unsafe { BufferArg::from_raw_parts(output_handle.clone(), len) },
        );
    }

    Ok(DeviceVec::from_handle(policy.id(), output_handle, len))
}

#[allow(dead_code)]
pub(crate) fn gather_device_with_policy<R, T>(
    policy: &CubePolicy<R>,
    input: &DeviceVec<R, T>,
    indices: &DeviceVec<R, u32>,
) -> Result<DeviceVec<R, T>, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    u32::try_from(indices.len()).map_err(|_| Error::LengthTooLarge { len: indices.len() })?;
    if indices.len() == 0 {
        return Ok(policy.empty_device_vec());
    }

    let client = policy.client();
    let output_handle = client.empty(indices.len() * std::mem::size_of::<T>());

    let block_count = indices.len().div_ceil(BLOCK_RANGE_SIZE as usize);
    let block_count_u32 =
        u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;
    unsafe {
        gather_kernel::launch_unchecked::<T, R>(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_RANGE_SIZE),
            unsafe { BufferArg::from_raw_parts(output_handle.clone(), indices.len()) },
            unsafe { BufferArg::from_raw_parts(indices.handle.clone(), indices.len()) },
            unsafe { BufferArg::from_raw_parts(input.handle.clone(), input.len()) },
        );
    }

    Ok(DeviceVec::from_handle(
        policy.id(),
        output_handle,
        indices.len(),
    ))
}
