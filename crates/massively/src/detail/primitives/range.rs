use crate::{
    device::{DeviceColumnMutView, DeviceVec},
    error::Error,
    index::{IntoMIndex, MIndex, mindex_from_usize, usize_from_mindex},
    kernels::*,
    policy::CubePolicy,
};
use cubecl::prelude::*;

const BLOCK_RANGE_SIZE: u32 = 256;

pub(crate) fn to_device<R, T>(policy: &CubePolicy<R>, input: &[T]) -> Result<DeviceVec<R, T>, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    let len = mindex_from_usize(input.len())?;
    if input.is_empty() {
        return Ok(policy.empty_device_vec());
    }
    let handle = policy.client().create_from_slice(T::as_bytes(input));
    Ok(DeviceVec::from_handle(policy.id(), handle, len))
}

pub(crate) fn filled<R, T>(
    policy: &CubePolicy<R>,
    len: impl IntoMIndex,
    value: T,
) -> Result<DeviceVec<R, T>, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    let len = len.into_mindex();
    if len == 0 {
        return Ok(policy.empty_device_vec());
    }
    let len_usize = usize_from_mindex(len);

    let client = policy.client();
    let output_handle = client.empty(len_usize * std::mem::size_of::<T>());

    let block_count = len_usize.div_ceil(BLOCK_RANGE_SIZE as usize);
    let block_count_u32 =
        u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;
    let values = [value];
    let lengths = [len];
    let value_handle = client.create_from_slice(T::as_bytes(&values));
    let len_handle = client.create_from_slice(u32::as_bytes(&lengths));

    unsafe {
        fill_kernel::launch_unchecked::<T, R>(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_RANGE_SIZE),
            unsafe { BufferArg::from_raw_parts(value_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output_handle.clone(), len_usize) },
        );
    }

    Ok(DeviceVec::from_handle(policy.id(), output_handle, len))
}

pub(crate) fn fill_slice_with_policy<R, T>(
    policy: &CubePolicy<R>,
    value: T,
    output: &DeviceColumnMutView<R, T>,
) -> Result<(), Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    if output.len == 0 {
        return Ok(());
    }
    let output_len = output.len;
    let output_offset_u32 = mindex_from_usize(output.offset)?;
    let output_len_u32 = mindex_from_usize(output.len)?;

    let client = policy.client();
    let values = [value];
    let value_handle = client.create_from_slice(T::as_bytes(&values));
    let metadata_handle =
        client.create_from_slice(u32::as_bytes(&[output_offset_u32, output_len_u32]));

    let block_count = output_len.div_ceil(BLOCK_RANGE_SIZE as usize);
    let block_count_u32 =
        u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;
    unsafe {
        fill_slice_kernel::launch_unchecked::<T, R>(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_RANGE_SIZE),
            unsafe { BufferArg::from_raw_parts(value_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(metadata_handle.clone(), 2) },
            unsafe { BufferArg::from_raw_parts(output.source.handle.clone(), output.source.len()) },
        );
    }

    Ok(())
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
    let len = left
        .len()
        .checked_add(right.len())
        .ok_or(Error::LengthTooLarge { len: usize::MAX })?;
    if len == 0 {
        return Ok(policy.empty_device_vec());
    }
    let len_usize = len;

    let client = policy.client();
    let output_handle = client.empty(len_usize * std::mem::size_of::<T>());

    let block_count = len_usize.div_ceil(BLOCK_RANGE_SIZE as usize);
    let block_count_u32 =
        u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;
    unsafe {
        concat_kernel::launch_unchecked::<T, R>(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_RANGE_SIZE),
            unsafe { BufferArg::from_raw_parts(left.handle.clone(), left.len()) },
            unsafe { BufferArg::from_raw_parts(right.handle.clone(), right.len()) },
            unsafe { BufferArg::from_raw_parts(output_handle.clone(), len_usize) },
        );
    }

    Ok(DeviceVec::from_handle(policy.id(), output_handle, len))
}

pub(crate) fn copy_handle<R, T>(
    policy: &CubePolicy<R>,
    input_handle: &cubecl::server::Handle,
    len: impl IntoMIndex,
) -> Result<cubecl::server::Handle, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    let len = len.into_mindex();
    if len == 0 {
        return Ok(policy.empty_handle());
    }
    let len_usize = usize_from_mindex(len);

    let client = policy.client();
    let output_handle = client.empty(len_usize * std::mem::size_of::<T>());

    let block_count = len_usize.div_ceil(BLOCK_RANGE_SIZE as usize);
    let block_count_u32 =
        u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;
    unsafe {
        copy_kernel::launch_unchecked::<T, R>(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_RANGE_SIZE),
            unsafe { BufferArg::from_raw_parts(input_handle.clone(), len_usize) },
            unsafe { BufferArg::from_raw_parts(output_handle.clone(), len_usize) },
        );
    }

    Ok(output_handle)
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
        .ok_or(Error::LengthTooLarge { len: usize::MAX })?;
    let output_end = output_offset
        .checked_add(len)
        .ok_or(Error::LengthTooLarge { len: usize::MAX })?;
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

    if len == 0 {
        return Ok(());
    }
    let len_usize = len;

    let client = policy.client();
    let metadata_handle = client.create_from_slice(u32::as_bytes(&[
        mindex_from_usize(input_offset)?,
        mindex_from_usize(output_offset)?,
        mindex_from_usize(len)?,
    ]));

    let block_count = len_usize.div_ceil(BLOCK_RANGE_SIZE as usize);
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

pub(crate) fn indices_mindex<R>(
    policy: &CubePolicy<R>,
    len: impl IntoMIndex,
) -> Result<DeviceVec<R, MIndex>, Error>
where
    R: Runtime,
{
    let len = len.into_mindex();
    if len == 0 {
        return Ok(policy.empty_device_vec());
    }
    let len_usize = usize_from_mindex(len);

    let client = policy.client();
    let output_handle = client.empty(len_usize * std::mem::size_of::<u32>());

    let block_count = len_usize.div_ceil(BLOCK_RANGE_SIZE as usize);
    let block_count_u32 =
        u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;
    unsafe {
        indices_u32_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_RANGE_SIZE),
            unsafe { BufferArg::from_raw_parts(output_handle.clone(), len_usize) },
        );
    }

    Ok(DeviceVec::from_handle(policy.id(), output_handle, len))
}

#[allow(dead_code)]
pub(crate) fn gather_device_with_policy<R, T>(
    policy: &CubePolicy<R>,
    input: &DeviceVec<R, T>,
    indices: &DeviceVec<R, MIndex>,
) -> Result<DeviceVec<R, T>, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    if indices.len() == 0 {
        return Ok(policy.empty_device_vec());
    }
    let indices_len = indices.len();
    let input_len = input.len();

    let client = policy.client();
    let output_handle = client.empty(indices_len * std::mem::size_of::<T>());

    let block_count = indices_len.div_ceil(BLOCK_RANGE_SIZE as usize);
    let block_count_u32 =
        u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;
    unsafe {
        gather_kernel::launch_unchecked::<T, R>(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_RANGE_SIZE),
            unsafe { BufferArg::from_raw_parts(output_handle.clone(), indices_len) },
            unsafe { BufferArg::from_raw_parts(indices.handle.clone(), indices_len) },
            unsafe { BufferArg::from_raw_parts(input.handle.clone(), input_len) },
        );
    }

    Ok(DeviceVec::from_handle(
        policy.id(),
        output_handle,
        indices.len(),
    ))
}
