use crate::{device::DeviceVec, error::Error, kernels::*, policy::CubePolicy};
use cubecl::prelude::*;

const BLOCK_RANGE_SIZE: u32 = 256;

pub(crate) fn to_device<R, T>(policy: &CubePolicy<R>, input: &[T]) -> Result<DeviceVec<R, T>, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    u32::try_from(input.len()).map_err(|_| Error::LengthTooLarge { len: input.len() })?;
    let handle = policy.client().create_from_slice(T::as_bytes(input));
    Ok(DeviceVec::from_handle(policy.clone(), handle, input.len()))
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
    let client = policy.client();
    let output_handle = client.empty(len * std::mem::size_of::<T>());

    if len != 0 {
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
                ArrayArg::from_raw_parts::<T>(&value_handle, 1, 1),
                ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
                ArrayArg::from_raw_parts::<T>(&output_handle, len, 1),
            )
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
        }
    }

    Ok(DeviceVec::from_handle(policy.clone(), output_handle, len))
}

pub(crate) fn copy_device<R, T>(input: &DeviceVec<R, T>) -> Result<DeviceVec<R, T>, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    u32::try_from(input.len()).map_err(|_| Error::LengthTooLarge { len: input.len() })?;
    let client = input.policy().client();
    let output_handle = client.empty(input.len() * std::mem::size_of::<T>());

    if input.len() != 0 {
        let block_count = input.len().div_ceil(BLOCK_RANGE_SIZE as usize);
        let block_count_u32 =
            u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;
        unsafe {
            copy_kernel::launch_unchecked::<T, R>(
                client,
                CubeCount::Static(block_count_u32, 1, 1),
                CubeDim::new_1d(BLOCK_RANGE_SIZE),
                ArrayArg::from_raw_parts::<T>(&input.handle, input.len(), 1),
                ArrayArg::from_raw_parts::<T>(&output_handle, input.len(), 1),
            )
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
        }
    }

    Ok(DeviceVec::from_handle(
        input.policy().clone(),
        output_handle,
        input.len(),
    ))
}

pub(crate) fn indices_u32<R>(policy: &CubePolicy<R>, len: usize) -> Result<DeviceVec<R, u32>, Error>
where
    R: Runtime,
{
    u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let client = policy.client();
    let output_handle = client.empty(len * std::mem::size_of::<u32>());

    if len != 0 {
        let block_count = len.div_ceil(BLOCK_RANGE_SIZE as usize);
        let block_count_u32 =
            u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;
        unsafe {
            indices_u32_kernel::launch_unchecked::<R>(
                client,
                CubeCount::Static(block_count_u32, 1, 1),
                CubeDim::new_1d(BLOCK_RANGE_SIZE),
                ArrayArg::from_raw_parts::<u32>(&output_handle, len, 1),
            )
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
        }
    }

    Ok(DeviceVec::from_handle(policy.clone(), output_handle, len))
}

pub(crate) fn gather_device<R, T>(
    input: &DeviceVec<R, T>,
    indices: &DeviceVec<R, u32>,
) -> Result<DeviceVec<R, T>, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    u32::try_from(indices.len()).map_err(|_| Error::LengthTooLarge { len: indices.len() })?;
    let client = input.policy().client();
    let output_handle = client.empty(indices.len() * std::mem::size_of::<T>());

    if indices.len() != 0 {
        let block_count = indices.len().div_ceil(BLOCK_RANGE_SIZE as usize);
        let block_count_u32 =
            u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;
        unsafe {
            gather_kernel::launch_unchecked::<T, R>(
                client,
                CubeCount::Static(block_count_u32, 1, 1),
                CubeDim::new_1d(BLOCK_RANGE_SIZE),
                ArrayArg::from_raw_parts::<T>(&output_handle, indices.len(), 1),
                ArrayArg::from_raw_parts::<u32>(&indices.handle, indices.len(), 1),
                ArrayArg::from_raw_parts::<T>(&input.handle, input.len(), 1),
            )
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
        }
    }

    Ok(DeviceVec::from_handle(
        input.policy().clone(),
        output_handle,
        indices.len(),
    ))
}
