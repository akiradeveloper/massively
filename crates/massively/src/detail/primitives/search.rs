use crate::{
    detail::op::kernel::BinaryPredicateOp, device::DeviceVec, error::Error, index::MIndex,
    kernels::*, op::GpuOp, policy::CubePolicy, primitives::scan::read_u32_scalar,
};
use cubecl::prelude::*;

const BLOCK_SELECT_SIZE: u32 = 256;

#[allow(dead_code)]
pub(crate) fn minmax_element<R, T, Less>(
    policy: &CubePolicy<R>,
    input: &DeviceVec<R, T>,
    _less: GpuOp<Less>,
) -> Result<Option<(MIndex, MIndex)>, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Less: BinaryPredicateOp<T>,
{
    if input.len() == 0 {
        return Ok(None);
    }

    let client = policy.client();
    let mut current_count = input.len().div_ceil(BLOCK_SELECT_SIZE as usize);
    let mut current_count_u32 =
        u32::try_from(current_count).map_err(|_| Error::LengthTooLarge { len: current_count })?;
    let len_u32 =
        u32::try_from(input.len()).map_err(|_| Error::LengthTooLarge { len: input.len() })?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let mut current_handle = client.empty(current_count * 2 * std::mem::size_of::<u32>());

    unsafe {
        minmax_element_partials_kernel::launch_unchecked::<T, Less, R>(
            client,
            crate::detail::launch::cube_count_1d(current_count_u32),
            CubeDim::new_1d(BLOCK_SELECT_SIZE),
            unsafe { BufferArg::from_raw_parts(input.handle.clone(), input.len()) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(current_handle.clone(), current_count * 2) },
        );
    }

    while current_count > 1 {
        let next_count = current_count.div_ceil(BLOCK_SELECT_SIZE as usize);
        let next_count_u32 =
            u32::try_from(next_count).map_err(|_| Error::LengthTooLarge { len: next_count })?;
        let candidate_len_handle = client.create_from_slice(u32::as_bytes(&[current_count_u32]));
        let next_handle = client.empty(next_count * 2 * std::mem::size_of::<u32>());

        unsafe {
            minmax_index_partials_kernel::launch_unchecked::<T, Less, R>(
                client,
                crate::detail::launch::cube_count_1d(next_count_u32),
                CubeDim::new_1d(BLOCK_SELECT_SIZE),
                unsafe { BufferArg::from_raw_parts(input.handle.clone(), input.len()) },
                unsafe { BufferArg::from_raw_parts(current_handle.clone(), current_count * 2) },
                unsafe { BufferArg::from_raw_parts(candidate_len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(next_handle.clone(), next_count * 2) },
            );
        }

        current_handle = next_handle;
        current_count = next_count;
        current_count_u32 = next_count_u32;
    }

    let bytes = client
        .read_one(current_handle)
        .map_err(|err| Error::Launch {
            message: format!("{err:?}"),
        })?;
    let indices = u32::from_bytes(&bytes);
    Ok(Some((indices[0], indices[1])))
}

pub(crate) fn first_flag<R>(
    policy: &CubePolicy<R>,
    flag_handle: cubecl::server::Handle,
    storage_len: usize,
    logical_len: usize,
) -> Result<Option<MIndex>, Error>
where
    R: Runtime,
{
    flag_index(
        policy,
        flag_handle,
        storage_len,
        logical_len,
        FlagIndexKind::First,
    )
}

pub(crate) fn first_unset_flag<R>(
    policy: &CubePolicy<R>,
    flag_handle: cubecl::server::Handle,
    storage_len: usize,
    logical_len: usize,
) -> Result<Option<MIndex>, Error>
where
    R: Runtime,
{
    flag_index(
        policy,
        flag_handle,
        storage_len,
        logical_len,
        FlagIndexKind::FirstUnset,
    )
}

#[derive(Clone, Copy)]
enum FlagIndexKind {
    First,
    FirstUnset,
}

fn flag_index<R>(
    policy: &CubePolicy<R>,
    flag_handle: cubecl::server::Handle,
    storage_len: usize,
    logical_len: usize,
    kind: FlagIndexKind,
) -> Result<Option<MIndex>, Error>
where
    R: Runtime,
{
    if storage_len == 0 || logical_len == 0 {
        return Ok(None);
    }

    let logical_len_u32 =
        u32::try_from(logical_len).map_err(|_| Error::LengthTooLarge { len: logical_len })?;
    let client = policy.client();
    let logical_len_handle = client.create_from_slice(u32::as_bytes(&[logical_len_u32]));
    let sentinel_handle = client.create_from_slice(u32::as_bytes(&[logical_len_u32]));
    let mut current_count = logical_len.div_ceil(BLOCK_SELECT_SIZE as usize);
    let mut current_count_u32 =
        u32::try_from(current_count).map_err(|_| Error::LengthTooLarge { len: current_count })?;
    let mut current_handle = client.empty(current_count * std::mem::size_of::<u32>());

    unsafe {
        match kind {
            FlagIndexKind::First => first_flag_partials_kernel::launch_unchecked::<R>(
                client,
                crate::detail::launch::cube_count_1d(current_count_u32),
                CubeDim::new_1d(BLOCK_SELECT_SIZE),
                unsafe { BufferArg::from_raw_parts(flag_handle.clone(), storage_len) },
                unsafe { BufferArg::from_raw_parts(logical_len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(current_handle.clone(), current_count) },
            ),
            FlagIndexKind::FirstUnset => first_unset_flag_partials_kernel::launch_unchecked::<R>(
                client,
                crate::detail::launch::cube_count_1d(current_count_u32),
                CubeDim::new_1d(BLOCK_SELECT_SIZE),
                unsafe { BufferArg::from_raw_parts(flag_handle.clone(), storage_len) },
                unsafe { BufferArg::from_raw_parts(logical_len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(current_handle.clone(), current_count) },
            ),
        }
    };

    while current_count > 1 {
        let next_count = current_count.div_ceil(BLOCK_SELECT_SIZE as usize);
        let next_count_u32 =
            u32::try_from(next_count).map_err(|_| Error::LengthTooLarge { len: next_count })?;
        let candidate_len_handle = client.create_from_slice(u32::as_bytes(&[current_count_u32]));
        let next_handle = client.empty(next_count * std::mem::size_of::<u32>());

        unsafe {
            match kind {
                FlagIndexKind::First | FlagIndexKind::FirstUnset => {
                    first_index_partials_kernel::launch_unchecked::<R>(
                        client,
                        crate::detail::launch::cube_count_1d(next_count_u32),
                        CubeDim::new_1d(BLOCK_SELECT_SIZE),
                        unsafe { BufferArg::from_raw_parts(current_handle.clone(), current_count) },
                        unsafe { BufferArg::from_raw_parts(candidate_len_handle.clone(), 1) },
                        unsafe { BufferArg::from_raw_parts(sentinel_handle.clone(), 1) },
                        unsafe { BufferArg::from_raw_parts(next_handle.clone(), next_count) },
                    )
                }
            }
        };

        current_handle = next_handle;
        current_count = next_count;
        current_count_u32 = next_count_u32;
    }

    let index = read_u32_scalar::<R>(client, current_handle)?;
    if index == logical_len_u32 {
        Ok(None)
    } else {
        Ok(Some(index))
    }
}
