use crate::{
    device::DeviceVec,
    error::Error,
    kernels::*,
    op::{BinaryPredicateOp, GpuOp, PredicateOp},
    policy::CubePolicy,
    primitives::scan::read_u32_scalar,
};
use cubecl::prelude::*;

const BLOCK_SELECT_SIZE: u32 = 256;

fn search_block_count(len: usize) -> Result<u32, Error> {
    let block_count = len.div_ceil(BLOCK_SELECT_SIZE as usize);
    u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })
}

pub(crate) fn equal<R, T, Eq>(
    left: &DeviceVec<R, T>,
    right: &DeviceVec<R, T>,
    _eq: GpuOp<Eq>,
) -> Result<bool, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Eq: BinaryPredicateOp<T>,
{
    if left.len() != right.len() {
        return Ok(false);
    }
    Ok(mismatch(left, right, GpuOp::<Eq>::new())?.is_none())
}

pub(crate) fn adjacent_find<R, T, Pred>(
    input: &DeviceVec<R, T>,
    _pred: GpuOp<Pred>,
) -> Result<Option<usize>, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Pred: BinaryPredicateOp<T>,
{
    if input.len() < 2 {
        return Ok(None);
    }

    let block_count_u32 = search_block_count(input.len())?;
    let client = input.policy().client();
    let flag_handle = client.empty(input.len() * std::mem::size_of::<u32>());

    unsafe {
        adjacent_find_flags_kernel::launch_unchecked::<T, Pred, R>(
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

    first_flag(input.policy(), flag_handle, input.len(), input.len() - 1)
}

pub(crate) fn mismatch<R, T, Eq>(
    left: &DeviceVec<R, T>,
    right: &DeviceVec<R, T>,
    _eq: GpuOp<Eq>,
) -> Result<Option<usize>, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Eq: BinaryPredicateOp<T>,
{
    let min_len = left.len().min(right.len());
    if min_len == 0 {
        return if left.len() == right.len() {
            Ok(None)
        } else {
            Ok(Some(0))
        };
    }

    u32::try_from(min_len).map_err(|_| Error::LengthTooLarge { len: min_len })?;
    let block_count_u32 = search_block_count(min_len)?;
    let client = left.policy().client();
    let flag_handle = client.empty(min_len * std::mem::size_of::<u32>());

    unsafe {
        mismatch_flags_kernel::launch_unchecked::<T, Eq, R>(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SELECT_SIZE),
            ArrayArg::from_raw_parts::<T>(&left.handle, left.len(), 1),
            ArrayArg::from_raw_parts::<T>(&right.handle, right.len(), 1),
            ArrayArg::from_raw_parts::<u32>(&flag_handle, min_len, 1),
        )
        .map_err(|err| Error::Launch {
            message: format!("{err:?}"),
        })?;
    }

    if let Some(index) = first_flag(left.policy(), flag_handle, min_len, min_len)? {
        return Ok(Some(index));
    }

    if left.len() == right.len() {
        Ok(None)
    } else {
        Ok(Some(min_len))
    }
}

pub(crate) fn is_sorted_until<R, T, Less>(
    input: &DeviceVec<R, T>,
    _less: GpuOp<Less>,
) -> Result<usize, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Less: BinaryPredicateOp<T>,
{
    if input.len() <= 1 {
        return Ok(input.len());
    }

    u32::try_from(input.len()).map_err(|_| Error::LengthTooLarge { len: input.len() })?;
    let block_count_u32 = search_block_count(input.len())?;
    let client = input.policy().client();
    let flag_handle = client.empty(input.len() * std::mem::size_of::<u32>());

    unsafe {
        sorted_break_flags_kernel::launch_unchecked::<T, Less, R>(
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

    Ok(first_flag(input.policy(), flag_handle, input.len(), input.len())?.unwrap_or(input.len()))
}

pub(crate) fn find_first_of<R, T, Eq>(
    input: &DeviceVec<R, T>,
    needles: &DeviceVec<R, T>,
    _eq: GpuOp<Eq>,
) -> Result<Option<usize>, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Eq: BinaryPredicateOp<T>,
{
    if input.len() == 0 || needles.len() == 0 {
        return Ok(None);
    }

    u32::try_from(input.len()).map_err(|_| Error::LengthTooLarge { len: input.len() })?;
    let block_count_u32 = search_block_count(input.len())?;
    let client = input.policy().client();
    let flag_handle = client.empty(input.len() * std::mem::size_of::<u32>());

    unsafe {
        find_first_of_flags_kernel::launch_unchecked::<T, Eq, R>(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SELECT_SIZE),
            ArrayArg::from_raw_parts::<T>(&input.handle, input.len(), 1),
            ArrayArg::from_raw_parts::<T>(&needles.handle, needles.len(), 1),
            ArrayArg::from_raw_parts::<u32>(&flag_handle, input.len(), 1),
        )
        .map_err(|err| Error::Launch {
            message: format!("{err:?}"),
        })?;
    }

    first_flag(input.policy(), flag_handle, input.len(), input.len())
}

pub(crate) fn find_end<R, T, Eq>(
    input: &DeviceVec<R, T>,
    pattern: &DeviceVec<R, T>,
    _eq: GpuOp<Eq>,
) -> Result<Option<usize>, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Eq: BinaryPredicateOp<T>,
{
    if pattern.len() == 0 || pattern.len() > input.len() {
        return Ok(None);
    }

    let candidate_len = input.len() - pattern.len() + 1;
    u32::try_from(candidate_len).map_err(|_| Error::LengthTooLarge { len: candidate_len })?;
    let block_count_u32 = search_block_count(candidate_len)?;
    let client = input.policy().client();
    let flag_handle = client.empty(candidate_len * std::mem::size_of::<u32>());

    unsafe {
        subrange_match_flags_kernel::launch_unchecked::<T, Eq, R>(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SELECT_SIZE),
            ArrayArg::from_raw_parts::<T>(&input.handle, input.len(), 1),
            ArrayArg::from_raw_parts::<T>(&pattern.handle, pattern.len(), 1),
            ArrayArg::from_raw_parts::<u32>(&flag_handle, candidate_len, 1),
        )
        .map_err(|err| Error::Launch {
            message: format!("{err:?}"),
        })?;
    }

    last_flag(input.policy(), flag_handle, candidate_len, candidate_len)
}

pub(crate) fn lexicographical_compare<R, T, Less>(
    left: &DeviceVec<R, T>,
    right: &DeviceVec<R, T>,
    _less: GpuOp<Less>,
) -> Result<bool, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Less: BinaryPredicateOp<T>,
{
    let min_len = left.len().min(right.len());
    if min_len == 0 {
        return Ok(left.len() < right.len());
    }

    u32::try_from(min_len).map_err(|_| Error::LengthTooLarge { len: min_len })?;
    let block_count_u32 = search_block_count(min_len)?;
    let client = left.policy().client();
    let flag_handle = client.empty(min_len * std::mem::size_of::<u32>());

    unsafe {
        lexicographical_diff_flags_kernel::launch_unchecked::<T, Less, R>(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SELECT_SIZE),
            ArrayArg::from_raw_parts::<T>(&left.handle, left.len(), 1),
            ArrayArg::from_raw_parts::<T>(&right.handle, right.len(), 1),
            ArrayArg::from_raw_parts::<u32>(&flag_handle, min_len, 1),
        )
        .map_err(|err| Error::Launch {
            message: format!("{err:?}"),
        })?;
    }

    let Some(index) = first_flag(left.policy(), flag_handle, min_len, min_len)? else {
        return Ok(left.len() < right.len());
    };

    let index_handle = client.create_from_slice(u32::as_bytes(&[index as u32]));
    let output_handle = client.empty(std::mem::size_of::<u32>());
    unsafe {
        lexicographical_compare_at_kernel::launch_unchecked::<T, Less, R>(
            client,
            CubeCount::new_single(),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts::<T>(&left.handle, left.len(), 1),
            ArrayArg::from_raw_parts::<T>(&right.handle, right.len(), 1),
            ArrayArg::from_raw_parts::<u32>(&index_handle, 1, 1),
            ArrayArg::from_raw_parts::<u32>(&output_handle, 1, 1),
        )
        .map_err(|err| Error::Launch {
            message: format!("{err:?}"),
        })?;
    }

    Ok(read_u32_scalar::<R>(client, output_handle) != 0)
}

pub(crate) fn includes<R, T, Less>(
    left: &DeviceVec<R, T>,
    right: &DeviceVec<R, T>,
    _less: GpuOp<Less>,
) -> Result<bool, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Less: BinaryPredicateOp<T>,
{
    if right.len() == 0 {
        return Ok(true);
    }
    if left.len() == 0 {
        return Ok(false);
    }

    u32::try_from(right.len()).map_err(|_| Error::LengthTooLarge { len: right.len() })?;
    let block_count_u32 = search_block_count(right.len())?;
    let client = left.policy().client();
    let keep_handle = client.create_from_slice(u32::as_bytes(&[0_u32]));
    let missing_flag_handle = client.empty(right.len() * std::mem::size_of::<u32>());

    unsafe {
        sorted_membership_flags_kernel::launch_unchecked::<T, Less, R>(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SELECT_SIZE),
            ArrayArg::from_raw_parts::<T>(&right.handle, right.len(), 1),
            ArrayArg::from_raw_parts::<T>(&left.handle, left.len(), 1),
            ArrayArg::from_raw_parts::<u32>(&keep_handle, 1, 1),
            ArrayArg::from_raw_parts::<u32>(&missing_flag_handle, right.len(), 1),
        )
        .map_err(|err| Error::Launch {
            message: format!("{err:?}"),
        })?;
    }

    Ok(first_flag(left.policy(), missing_flag_handle, right.len(), right.len())?.is_none())
}

pub(crate) fn partition_point<R, T, Pred>(
    input: &DeviceVec<R, T>,
    _pred: GpuOp<Pred>,
) -> Result<usize, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Pred: PredicateOp<T>,
{
    if input.len() == 0 {
        return Ok(0);
    }

    u32::try_from(input.len()).map_err(|_| Error::LengthTooLarge { len: input.len() })?;
    let block_count_u32 = search_block_count(input.len())?;
    let client = input.policy().client();
    let flag_handle = client.empty(input.len() * std::mem::size_of::<u32>());

    unsafe {
        partition_point_flags_kernel::launch_unchecked::<T, Pred, R>(
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

    Ok(first_flag(input.policy(), flag_handle, input.len(), input.len())?.unwrap_or(input.len()))
}

pub(crate) fn is_partitioned<R, T, Pred>(
    input: &DeviceVec<R, T>,
    _pred: GpuOp<Pred>,
) -> Result<bool, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Pred: PredicateOp<T>,
{
    if input.len() <= 1 {
        return Ok(true);
    }

    let point = partition_point(input, GpuOp::<Pred>::new())?;
    if point == input.len() {
        return Ok(true);
    }

    u32::try_from(input.len()).map_err(|_| Error::LengthTooLarge { len: input.len() })?;
    let block_count_u32 = search_block_count(input.len())?;
    let client = input.policy().client();
    let point_handle = client.create_from_slice(u32::as_bytes(&[point as u32]));
    let flag_handle = client.empty(input.len() * std::mem::size_of::<u32>());

    unsafe {
        partition_tail_true_flags_kernel::launch_unchecked::<T, Pred, R>(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SELECT_SIZE),
            ArrayArg::from_raw_parts::<T>(&input.handle, input.len(), 1),
            ArrayArg::from_raw_parts::<u32>(&point_handle, 1, 1),
            ArrayArg::from_raw_parts::<u32>(&flag_handle, input.len(), 1),
        )
        .map_err(|err| Error::Launch {
            message: format!("{err:?}"),
        })?;
    }

    Ok(first_flag(input.policy(), flag_handle, input.len(), input.len())?.is_none())
}

pub(crate) fn lower_bound<R, T, Less>(
    input: &DeviceVec<R, T>,
    value: T,
    _less: GpuOp<Less>,
) -> Result<usize, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Less: BinaryPredicateOp<T>,
{
    sorted_bound_flags::<R, T, Less>(input, value, BoundKind::Lower)
}

pub(crate) fn upper_bound<R, T, Less>(
    input: &DeviceVec<R, T>,
    value: T,
    _less: GpuOp<Less>,
) -> Result<usize, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Less: BinaryPredicateOp<T>,
{
    sorted_bound_flags::<R, T, Less>(input, value, BoundKind::Upper)
}

pub(crate) fn binary_search<R, T, Less>(
    input: &DeviceVec<R, T>,
    value: T,
    _less: GpuOp<Less>,
) -> Result<bool, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Less: BinaryPredicateOp<T>,
{
    let index = lower_bound(input, value.clone(), GpuOp::<Less>::new())?;
    if index >= input.len() {
        return Ok(false);
    }

    let client = input.policy().client();
    let value_handle = client.create_from_slice(T::as_bytes(&[value]));
    let index_handle = client.create_from_slice(u32::as_bytes(&[index as u32]));
    let output_handle = client.empty(std::mem::size_of::<u32>());
    unsafe {
        binary_search_at_kernel::launch_unchecked::<T, Less, R>(
            client,
            CubeCount::new_single(),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts::<T>(&input.handle, input.len(), 1),
            ArrayArg::from_raw_parts::<T>(&value_handle, 1, 1),
            ArrayArg::from_raw_parts::<u32>(&index_handle, 1, 1),
            ArrayArg::from_raw_parts::<u32>(&output_handle, 1, 1),
        )
        .map_err(|err| Error::Launch {
            message: format!("{err:?}"),
        })?;
    }
    Ok(read_u32_scalar::<R>(client, output_handle) != 0)
}

pub(crate) fn minmax_element<R, T, Less>(
    input: &DeviceVec<R, T>,
    _less: GpuOp<Less>,
) -> Result<Option<(usize, usize)>, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Less: BinaryPredicateOp<T>,
{
    if input.len() == 0 {
        return Ok(None);
    }

    let client = input.policy().client();
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
            CubeCount::Static(current_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_SELECT_SIZE),
            ArrayArg::from_raw_parts::<T>(&input.handle, input.len(), 1),
            ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
            ArrayArg::from_raw_parts::<u32>(&current_handle, current_count * 2, 1),
        )
        .map_err(|err| Error::Launch {
            message: format!("{err:?}"),
        })?;
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
                CubeCount::Static(next_count_u32, 1, 1),
                CubeDim::new_1d(BLOCK_SELECT_SIZE),
                ArrayArg::from_raw_parts::<T>(&input.handle, input.len(), 1),
                ArrayArg::from_raw_parts::<u32>(&current_handle, current_count * 2, 1),
                ArrayArg::from_raw_parts::<u32>(&candidate_len_handle, 1, 1),
                ArrayArg::from_raw_parts::<u32>(&next_handle, next_count * 2, 1),
            )
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
        }

        current_handle = next_handle;
        current_count = next_count;
        current_count_u32 = next_count_u32;
    }

    let bytes = client.read_one(current_handle);
    let indices = u32::from_bytes(&bytes);
    Ok(Some((indices[0] as usize, indices[1] as usize)))
}

enum BoundKind {
    Lower,
    Upper,
}

fn sorted_bound_flags<R, T, Less>(
    input: &DeviceVec<R, T>,
    value: T,
    kind: BoundKind,
) -> Result<usize, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Less: BinaryPredicateOp<T>,
{
    if input.len() == 0 {
        return Ok(0);
    }

    u32::try_from(input.len()).map_err(|_| Error::LengthTooLarge { len: input.len() })?;
    let block_count_u32 = search_block_count(input.len())?;
    let client = input.policy().client();
    let value_handle = client.create_from_slice(T::as_bytes(&[value]));
    let flag_handle = client.empty(input.len() * std::mem::size_of::<u32>());

    unsafe {
        match kind {
            BoundKind::Lower => lower_bound_flags_kernel::launch_unchecked::<T, Less, R>(
                client,
                CubeCount::Static(block_count_u32, 1, 1),
                CubeDim::new_1d(BLOCK_SELECT_SIZE),
                ArrayArg::from_raw_parts::<T>(&input.handle, input.len(), 1),
                ArrayArg::from_raw_parts::<T>(&value_handle, 1, 1),
                ArrayArg::from_raw_parts::<u32>(&flag_handle, input.len(), 1),
            ),
            BoundKind::Upper => upper_bound_flags_kernel::launch_unchecked::<T, Less, R>(
                client,
                CubeCount::Static(block_count_u32, 1, 1),
                CubeDim::new_1d(BLOCK_SELECT_SIZE),
                ArrayArg::from_raw_parts::<T>(&input.handle, input.len(), 1),
                ArrayArg::from_raw_parts::<T>(&value_handle, 1, 1),
                ArrayArg::from_raw_parts::<u32>(&flag_handle, input.len(), 1),
            ),
        }
    }
    .map_err(|err| Error::Launch {
        message: format!("{err:?}"),
    })?;

    Ok(first_flag(input.policy(), flag_handle, input.len(), input.len())?.unwrap_or(input.len()))
}

pub(crate) fn first_flag<R>(
    policy: &CubePolicy<R>,
    flag_handle: cubecl::server::Handle,
    storage_len: usize,
    logical_len: usize,
) -> Result<Option<usize>, Error>
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
) -> Result<Option<usize>, Error>
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

pub(crate) fn last_flag<R>(
    policy: &CubePolicy<R>,
    flag_handle: cubecl::server::Handle,
    storage_len: usize,
    logical_len: usize,
) -> Result<Option<usize>, Error>
where
    R: Runtime,
{
    flag_index(
        policy,
        flag_handle,
        storage_len,
        logical_len,
        FlagIndexKind::Last,
    )
}

#[derive(Clone, Copy)]
enum FlagIndexKind {
    First,
    FirstUnset,
    Last,
}

fn flag_index<R>(
    policy: &CubePolicy<R>,
    flag_handle: cubecl::server::Handle,
    storage_len: usize,
    logical_len: usize,
    kind: FlagIndexKind,
) -> Result<Option<usize>, Error>
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
                CubeCount::Static(current_count_u32, 1, 1),
                CubeDim::new_1d(BLOCK_SELECT_SIZE),
                ArrayArg::from_raw_parts::<u32>(&flag_handle, storage_len, 1),
                ArrayArg::from_raw_parts::<u32>(&logical_len_handle, 1, 1),
                ArrayArg::from_raw_parts::<u32>(&current_handle, current_count, 1),
            ),
            FlagIndexKind::FirstUnset => first_unset_flag_partials_kernel::launch_unchecked::<R>(
                client,
                CubeCount::Static(current_count_u32, 1, 1),
                CubeDim::new_1d(BLOCK_SELECT_SIZE),
                ArrayArg::from_raw_parts::<u32>(&flag_handle, storage_len, 1),
                ArrayArg::from_raw_parts::<u32>(&logical_len_handle, 1, 1),
                ArrayArg::from_raw_parts::<u32>(&current_handle, current_count, 1),
            ),
            FlagIndexKind::Last => last_flag_partials_kernel::launch_unchecked::<R>(
                client,
                CubeCount::Static(current_count_u32, 1, 1),
                CubeDim::new_1d(BLOCK_SELECT_SIZE),
                ArrayArg::from_raw_parts::<u32>(&flag_handle, storage_len, 1),
                ArrayArg::from_raw_parts::<u32>(&logical_len_handle, 1, 1),
                ArrayArg::from_raw_parts::<u32>(&current_handle, current_count, 1),
            ),
        }
    }
    .map_err(|err| Error::Launch {
        message: format!("{err:?}"),
    })?;

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
                        CubeCount::Static(next_count_u32, 1, 1),
                        CubeDim::new_1d(BLOCK_SELECT_SIZE),
                        ArrayArg::from_raw_parts::<u32>(&current_handle, current_count, 1),
                        ArrayArg::from_raw_parts::<u32>(&candidate_len_handle, 1, 1),
                        ArrayArg::from_raw_parts::<u32>(&sentinel_handle, 1, 1),
                        ArrayArg::from_raw_parts::<u32>(&next_handle, next_count, 1),
                    )
                }
                FlagIndexKind::Last => last_index_partials_kernel::launch_unchecked::<R>(
                    client,
                    CubeCount::Static(next_count_u32, 1, 1),
                    CubeDim::new_1d(BLOCK_SELECT_SIZE),
                    ArrayArg::from_raw_parts::<u32>(&current_handle, current_count, 1),
                    ArrayArg::from_raw_parts::<u32>(&candidate_len_handle, 1, 1),
                    ArrayArg::from_raw_parts::<u32>(&sentinel_handle, 1, 1),
                    ArrayArg::from_raw_parts::<u32>(&next_handle, next_count, 1),
                ),
            }
        }
        .map_err(|err| Error::Launch {
            message: format!("{err:?}"),
        })?;

        current_handle = next_handle;
        current_count = next_count;
        current_count_u32 = next_count_u32;
    }

    let index = read_u32_scalar::<R>(client, current_handle) as usize;
    if index == logical_len {
        Ok(None)
    } else {
        Ok(Some(index))
    }
}
