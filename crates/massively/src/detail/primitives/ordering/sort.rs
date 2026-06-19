use crate::{
    device::DeviceVec,
    error::Error,
    kernels::*,
    op::{BinaryPredicateOp, GpuOp},
    primitives::{ensure_same_len, workspace::Workspace},
};
use cubecl::prelude::*;

use super::BLOCK_ORDERING_SIZE;

pub(crate) fn sort<R, T, Less>(
    input: &DeviceVec<R, T>,
    _less: GpuOp<Less>,
) -> Result<DeviceVec<R, T>, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Less: BinaryPredicateOp<T>,
{
    let num_blocks = input.len().div_ceil(BLOCK_ORDERING_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let client = input.policy().client();
    if input.len() <= 1 {
        return Ok(DeviceVec::from_handle(
            input.policy().clone(),
            input.handle.clone(),
            input.len(),
        ));
    }

    let workspace = Workspace::new(input.policy());
    let (scratch_a, scratch_b) = workspace.alloc_pair::<T>(input.len());
    let mut input_handle = input.handle.clone();
    let mut output_handle = scratch_a.clone();
    let mut next_uses_a = false;
    let mut width = 1usize;

    while width < input.len() {
        let width_u32 = u32::try_from(width).map_err(|_| Error::LengthTooLarge { len: width })?;
        let width_handle = client.create_from_slice(u32::as_bytes(&[width_u32]));
        unsafe {
            merge_sort_pass_kernel::launch_unchecked::<T, Less, R>(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_ORDERING_SIZE),
                unsafe { BufferArg::from_raw_parts(input_handle.clone(), input.len()) },
                unsafe { BufferArg::from_raw_parts(width_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(output_handle.clone(), input.len()) },
            );
        }

        input_handle = output_handle.clone();
        output_handle = if next_uses_a {
            scratch_a.clone()
        } else {
            scratch_b.clone()
        };
        next_uses_a = !next_uses_a;
        width *= 2;
    }

    Ok(DeviceVec::from_handle(
        input.policy().clone(),
        input_handle,
        input.len(),
    ))
}

pub(crate) fn sort_by_key<R, K, T, Less>(
    keys: &DeviceVec<R, K>,
    values: &DeviceVec<R, T>,
    _less: GpuOp<Less>,
) -> Result<(DeviceVec<R, K>, DeviceVec<R, T>), Error>
where
    R: Runtime,
    K: CubePrimitive + CubeElement,
    T: CubePrimitive + CubeElement,
    Less: BinaryPredicateOp<K>,
{
    ensure_same_len(values.len(), keys.len())?;

    let num_blocks = keys.len().div_ceil(BLOCK_ORDERING_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let client = keys.policy().client();
    if keys.len() <= 1 {
        return Ok((
            DeviceVec::from_handle(keys.policy().clone(), keys.handle.clone(), keys.len()),
            DeviceVec::from_handle(values.policy().clone(), values.handle.clone(), values.len()),
        ));
    }

    let workspace = Workspace::new(keys.policy());
    let (scratch_keys_a, scratch_keys_b) = workspace.alloc_pair::<K>(keys.len());
    let (scratch_values_a, scratch_values_b) = workspace.alloc_pair::<T>(values.len());
    let mut input_key_handle = keys.handle.clone();
    let mut input_value_handle = values.handle.clone();
    let mut output_key_handle = scratch_keys_a.clone();
    let mut output_value_handle = scratch_values_a.clone();
    let mut next_uses_a = false;
    let mut width = 1usize;

    while width < keys.len() {
        let width_u32 = u32::try_from(width).map_err(|_| Error::LengthTooLarge { len: width })?;
        let width_handle = client.create_from_slice(u32::as_bytes(&[width_u32]));
        unsafe {
            merge_sort_by_key_pass_kernel::launch_unchecked::<K, T, Less, R>(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_ORDERING_SIZE),
                unsafe { BufferArg::from_raw_parts(input_key_handle.clone(), keys.len()) },
                unsafe { BufferArg::from_raw_parts(input_value_handle.clone(), values.len()) },
                unsafe { BufferArg::from_raw_parts(width_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(output_key_handle.clone(), keys.len()) },
                unsafe { BufferArg::from_raw_parts(output_value_handle.clone(), values.len()) },
            );
        }

        input_key_handle = output_key_handle.clone();
        input_value_handle = output_value_handle.clone();
        if next_uses_a {
            output_key_handle = scratch_keys_a.clone();
            output_value_handle = scratch_values_a.clone();
        } else {
            output_key_handle = scratch_keys_b.clone();
            output_value_handle = scratch_values_b.clone();
        }
        next_uses_a = !next_uses_a;
        width *= 2;
    }

    Ok((
        DeviceVec::from_handle(keys.policy().clone(), input_key_handle, keys.len()),
        DeviceVec::from_handle(values.policy().clone(), input_value_handle, values.len()),
    ))
}

pub(crate) fn sort_tuple2<R, A, B, Less>(
    first: &DeviceVec<R, A>,
    second: &DeviceVec<R, B>,
    _less: GpuOp<Less>,
) -> Result<(DeviceVec<R, A>, DeviceVec<R, B>), Error>
where
    R: Runtime,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    Less: BinaryPredicateOp<(A, B)>,
{
    ensure_same_len(second.len(), first.len())?;

    let len = first.len();
    let client = first.policy().client();
    if len <= 1 {
        return Ok((
            DeviceVec::from_handle(first.policy().clone(), first.handle.clone(), len),
            DeviceVec::from_handle(second.policy().clone(), second.handle.clone(), len),
        ));
    }

    let num_blocks = len.div_ceil(BLOCK_ORDERING_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let workspace = Workspace::new(first.policy());
    let (scratch_first_a, scratch_first_b) = workspace.alloc_pair::<A>(len);
    let (scratch_second_a, scratch_second_b) = workspace.alloc_pair::<B>(len);
    let mut input_first_handle = first.handle.clone();
    let mut input_second_handle = second.handle.clone();
    let mut output_first_handle = scratch_first_a.clone();
    let mut output_second_handle = scratch_second_a.clone();
    let mut next_uses_a = false;
    let mut width = 1usize;

    while width < len {
        let width_u32 = u32::try_from(width).map_err(|_| Error::LengthTooLarge { len: width })?;
        let width_handle = client.create_from_slice(u32::as_bytes(&[width_u32]));
        unsafe {
            merge_sort_tuple2_pass_kernel::launch_unchecked::<A, B, Less, R>(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_ORDERING_SIZE),
                unsafe { BufferArg::from_raw_parts(input_first_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(input_second_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(width_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(output_first_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(output_second_handle.clone(), len) },
            );
        }

        input_first_handle = output_first_handle.clone();
        input_second_handle = output_second_handle.clone();
        if next_uses_a {
            output_first_handle = scratch_first_a.clone();
            output_second_handle = scratch_second_a.clone();
        } else {
            output_first_handle = scratch_first_b.clone();
            output_second_handle = scratch_second_b.clone();
        }
        next_uses_a = !next_uses_a;
        width *= 2;
    }

    Ok((
        DeviceVec::from_handle(first.policy().clone(), input_first_handle, len),
        DeviceVec::from_handle(second.policy().clone(), input_second_handle, len),
    ))
}

pub(crate) fn sort_tuple3<R, A, B, C, Less>(
    first: &DeviceVec<R, A>,
    second: &DeviceVec<R, B>,
    third: &DeviceVec<R, C>,
    _less: GpuOp<Less>,
) -> Result<(DeviceVec<R, A>, DeviceVec<R, B>, DeviceVec<R, C>), Error>
where
    R: Runtime,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    C: CubePrimitive + CubeElement,
    Less: BinaryPredicateOp<(A, B, C)>,
{
    ensure_same_len(second.len(), first.len())?;
    ensure_same_len(third.len(), first.len())?;

    let len = first.len();
    let client = first.policy().client();
    if len <= 1 {
        return Ok((
            DeviceVec::from_handle(first.policy().clone(), first.handle.clone(), len),
            DeviceVec::from_handle(second.policy().clone(), second.handle.clone(), len),
            DeviceVec::from_handle(third.policy().clone(), third.handle.clone(), len),
        ));
    }

    let num_blocks = len.div_ceil(BLOCK_ORDERING_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let workspace = Workspace::new(first.policy());
    let (scratch_first_a, scratch_first_b) = workspace.alloc_pair::<A>(len);
    let (scratch_second_a, scratch_second_b) = workspace.alloc_pair::<B>(len);
    let (scratch_third_a, scratch_third_b) = workspace.alloc_pair::<C>(len);
    let mut input_first_handle = first.handle.clone();
    let mut input_second_handle = second.handle.clone();
    let mut input_third_handle = third.handle.clone();
    let mut output_first_handle = scratch_first_a.clone();
    let mut output_second_handle = scratch_second_a.clone();
    let mut output_third_handle = scratch_third_a.clone();
    let mut next_uses_a = false;
    let mut width = 1usize;

    while width < len {
        let width_u32 = u32::try_from(width).map_err(|_| Error::LengthTooLarge { len: width })?;
        let width_handle = client.create_from_slice(u32::as_bytes(&[width_u32]));
        unsafe {
            merge_sort_tuple3_pass_kernel::launch_unchecked::<A, B, C, Less, R>(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_ORDERING_SIZE),
                unsafe { BufferArg::from_raw_parts(input_first_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(input_second_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(input_third_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(width_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(output_first_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(output_second_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(output_third_handle.clone(), len) },
            );
        }

        input_first_handle = output_first_handle.clone();
        input_second_handle = output_second_handle.clone();
        input_third_handle = output_third_handle.clone();
        if next_uses_a {
            output_first_handle = scratch_first_a.clone();
            output_second_handle = scratch_second_a.clone();
            output_third_handle = scratch_third_a.clone();
        } else {
            output_first_handle = scratch_first_b.clone();
            output_second_handle = scratch_second_b.clone();
            output_third_handle = scratch_third_b.clone();
        }
        next_uses_a = !next_uses_a;
        width *= 2;
    }

    Ok((
        DeviceVec::from_handle(first.policy().clone(), input_first_handle, len),
        DeviceVec::from_handle(second.policy().clone(), input_second_handle, len),
        DeviceVec::from_handle(third.policy().clone(), input_third_handle, len),
    ))
}

pub(crate) fn sort_tuple2_by_key<R, A, B, T, Less>(
    key_a: &DeviceVec<R, A>,
    key_b: &DeviceVec<R, B>,
    values: &DeviceVec<R, T>,
    _less: GpuOp<Less>,
) -> Result<(DeviceVec<R, A>, DeviceVec<R, B>, DeviceVec<R, T>), Error>
where
    R: Runtime,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    T: CubePrimitive + CubeElement,
    Less: BinaryPredicateOp<(A, B)>,
{
    ensure_same_len(key_b.len(), key_a.len())?;
    ensure_same_len(values.len(), key_a.len())?;

    let len = key_a.len();
    let client = key_a.policy().client();
    if len <= 1 {
        return Ok((
            DeviceVec::from_handle(key_a.policy().clone(), key_a.handle.clone(), len),
            DeviceVec::from_handle(key_b.policy().clone(), key_b.handle.clone(), len),
            DeviceVec::from_handle(values.policy().clone(), values.handle.clone(), len),
        ));
    }

    let num_blocks = len.div_ceil(BLOCK_ORDERING_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let workspace = Workspace::new(key_a.policy());
    let (scratch_a_a, scratch_a_b) = workspace.alloc_pair::<A>(len);
    let (scratch_b_a, scratch_b_b) = workspace.alloc_pair::<B>(len);
    let (scratch_values_a, scratch_values_b) = workspace.alloc_pair::<T>(len);
    let mut input_a_handle = key_a.handle.clone();
    let mut input_b_handle = key_b.handle.clone();
    let mut input_value_handle = values.handle.clone();
    let mut output_a_handle = scratch_a_a.clone();
    let mut output_b_handle = scratch_b_a.clone();
    let mut output_value_handle = scratch_values_a.clone();
    let mut next_uses_a = false;
    let mut width = 1usize;

    while width < len {
        let width_u32 = u32::try_from(width).map_err(|_| Error::LengthTooLarge { len: width })?;
        let width_handle = client.create_from_slice(u32::as_bytes(&[width_u32]));
        unsafe {
            merge_sort_tuple2_by_key_pass_kernel::launch_unchecked::<A, B, T, Less, R>(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_ORDERING_SIZE),
                unsafe { BufferArg::from_raw_parts(input_a_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(input_b_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(input_value_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(width_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(output_a_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(output_b_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(output_value_handle.clone(), len) },
            );
        }

        input_a_handle = output_a_handle.clone();
        input_b_handle = output_b_handle.clone();
        input_value_handle = output_value_handle.clone();
        if next_uses_a {
            output_a_handle = scratch_a_a.clone();
            output_b_handle = scratch_b_a.clone();
            output_value_handle = scratch_values_a.clone();
        } else {
            output_a_handle = scratch_a_b.clone();
            output_b_handle = scratch_b_b.clone();
            output_value_handle = scratch_values_b.clone();
        }
        next_uses_a = !next_uses_a;
        width *= 2;
    }

    Ok((
        DeviceVec::from_handle(key_a.policy().clone(), input_a_handle, len),
        DeviceVec::from_handle(key_b.policy().clone(), input_b_handle, len),
        DeviceVec::from_handle(values.policy().clone(), input_value_handle, len),
    ))
}

pub(crate) fn sort_tuple3_by_key<R, A, B, C, T, Less>(
    key_a: &DeviceVec<R, A>,
    key_b: &DeviceVec<R, B>,
    key_c: &DeviceVec<R, C>,
    values: &DeviceVec<R, T>,
    _less: GpuOp<Less>,
) -> Result<
    (
        DeviceVec<R, A>,
        DeviceVec<R, B>,
        DeviceVec<R, C>,
        DeviceVec<R, T>,
    ),
    Error,
>
where
    R: Runtime,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    C: CubePrimitive + CubeElement,
    T: CubePrimitive + CubeElement,
    Less: BinaryPredicateOp<(A, B, C)>,
{
    ensure_same_len(key_b.len(), key_a.len())?;
    ensure_same_len(key_c.len(), key_a.len())?;
    ensure_same_len(values.len(), key_a.len())?;

    let len = key_a.len();
    let client = key_a.policy().client();
    if len <= 1 {
        return Ok((
            DeviceVec::from_handle(key_a.policy().clone(), key_a.handle.clone(), len),
            DeviceVec::from_handle(key_b.policy().clone(), key_b.handle.clone(), len),
            DeviceVec::from_handle(key_c.policy().clone(), key_c.handle.clone(), len),
            DeviceVec::from_handle(values.policy().clone(), values.handle.clone(), len),
        ));
    }

    let num_blocks = len.div_ceil(BLOCK_ORDERING_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let workspace = Workspace::new(key_a.policy());
    let (scratch_a_a, scratch_a_b) = workspace.alloc_pair::<A>(len);
    let (scratch_b_a, scratch_b_b) = workspace.alloc_pair::<B>(len);
    let (scratch_c_a, scratch_c_b) = workspace.alloc_pair::<C>(len);
    let (scratch_values_a, scratch_values_b) = workspace.alloc_pair::<T>(len);
    let mut input_a_handle = key_a.handle.clone();
    let mut input_b_handle = key_b.handle.clone();
    let mut input_c_handle = key_c.handle.clone();
    let mut input_value_handle = values.handle.clone();
    let mut output_a_handle = scratch_a_a.clone();
    let mut output_b_handle = scratch_b_a.clone();
    let mut output_c_handle = scratch_c_a.clone();
    let mut output_value_handle = scratch_values_a.clone();
    let mut next_uses_a = false;
    let mut width = 1usize;

    while width < len {
        let width_u32 = u32::try_from(width).map_err(|_| Error::LengthTooLarge { len: width })?;
        let width_handle = client.create_from_slice(u32::as_bytes(&[width_u32]));
        unsafe {
            merge_sort_tuple3_by_key_pass_kernel::launch_unchecked::<A, B, C, T, Less, R>(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_ORDERING_SIZE),
                unsafe { BufferArg::from_raw_parts(input_a_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(input_b_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(input_c_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(input_value_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(width_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(output_a_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(output_b_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(output_c_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(output_value_handle.clone(), len) },
            );
        }

        input_a_handle = output_a_handle.clone();
        input_b_handle = output_b_handle.clone();
        input_c_handle = output_c_handle.clone();
        input_value_handle = output_value_handle.clone();
        if next_uses_a {
            output_a_handle = scratch_a_a.clone();
            output_b_handle = scratch_b_a.clone();
            output_c_handle = scratch_c_a.clone();
            output_value_handle = scratch_values_a.clone();
        } else {
            output_a_handle = scratch_a_b.clone();
            output_b_handle = scratch_b_b.clone();
            output_c_handle = scratch_c_b.clone();
            output_value_handle = scratch_values_b.clone();
        }
        next_uses_a = !next_uses_a;
        width *= 2;
    }

    Ok((
        DeviceVec::from_handle(key_a.policy().clone(), input_a_handle, len),
        DeviceVec::from_handle(key_b.policy().clone(), input_b_handle, len),
        DeviceVec::from_handle(key_c.policy().clone(), input_c_handle, len),
        DeviceVec::from_handle(values.policy().clone(), input_value_handle, len),
    ))
}
