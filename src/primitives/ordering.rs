use crate::{
    device::DeviceVec,
    error::Error,
    kernels::*,
    op::{BinaryPredicateOp, GpuOp},
    primitives::{scan::inclusive_scan_u32, select},
};
use cubecl::prelude::*;

const BLOCK_ORDERING_SIZE: u32 = 256;
#[allow(dead_code)]
const RADIX_DIGITS: usize = 16;

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

    let scratch_a = client.empty(input.len() * std::mem::size_of::<T>());
    let scratch_b = client.empty(input.len() * std::mem::size_of::<T>());
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
                ArrayArg::from_raw_parts::<T>(&input_handle, input.len(), 1),
                ArrayArg::from_raw_parts::<u32>(&width_handle, 1, 1),
                ArrayArg::from_raw_parts::<T>(&output_handle, input.len(), 1),
            )
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
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
    if keys.len() != values.len() {
        return Err(Error::LengthMismatch {
            input: values.len(),
            output: keys.len(),
        });
    }

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

    let scratch_keys_a = client.empty(keys.len() * std::mem::size_of::<K>());
    let scratch_values_a = client.empty(values.len() * std::mem::size_of::<T>());
    let scratch_keys_b = client.empty(keys.len() * std::mem::size_of::<K>());
    let scratch_values_b = client.empty(values.len() * std::mem::size_of::<T>());
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
                ArrayArg::from_raw_parts::<K>(&input_key_handle, keys.len(), 1),
                ArrayArg::from_raw_parts::<T>(&input_value_handle, values.len(), 1),
                ArrayArg::from_raw_parts::<u32>(&width_handle, 1, 1),
                ArrayArg::from_raw_parts::<K>(&output_key_handle, keys.len(), 1),
                ArrayArg::from_raw_parts::<T>(&output_value_handle, values.len(), 1),
            )
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
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
    if first.len() != second.len() {
        return Err(Error::LengthMismatch {
            input: second.len(),
            output: first.len(),
        });
    }

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
    let scratch_first_a = client.empty(len * std::mem::size_of::<A>());
    let scratch_second_a = client.empty(len * std::mem::size_of::<B>());
    let scratch_first_b = client.empty(len * std::mem::size_of::<A>());
    let scratch_second_b = client.empty(len * std::mem::size_of::<B>());
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
                ArrayArg::from_raw_parts::<A>(&input_first_handle, len, 1),
                ArrayArg::from_raw_parts::<B>(&input_second_handle, len, 1),
                ArrayArg::from_raw_parts::<u32>(&width_handle, 1, 1),
                ArrayArg::from_raw_parts::<A>(&output_first_handle, len, 1),
                ArrayArg::from_raw_parts::<B>(&output_second_handle, len, 1),
            )
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
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
    if first.len() != second.len() {
        return Err(Error::LengthMismatch {
            input: second.len(),
            output: first.len(),
        });
    }
    if first.len() != third.len() {
        return Err(Error::LengthMismatch {
            input: third.len(),
            output: first.len(),
        });
    }

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
    let scratch_first_a = client.empty(len * std::mem::size_of::<A>());
    let scratch_second_a = client.empty(len * std::mem::size_of::<B>());
    let scratch_third_a = client.empty(len * std::mem::size_of::<C>());
    let scratch_first_b = client.empty(len * std::mem::size_of::<A>());
    let scratch_second_b = client.empty(len * std::mem::size_of::<B>());
    let scratch_third_b = client.empty(len * std::mem::size_of::<C>());
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
                ArrayArg::from_raw_parts::<A>(&input_first_handle, len, 1),
                ArrayArg::from_raw_parts::<B>(&input_second_handle, len, 1),
                ArrayArg::from_raw_parts::<C>(&input_third_handle, len, 1),
                ArrayArg::from_raw_parts::<u32>(&width_handle, 1, 1),
                ArrayArg::from_raw_parts::<A>(&output_first_handle, len, 1),
                ArrayArg::from_raw_parts::<B>(&output_second_handle, len, 1),
                ArrayArg::from_raw_parts::<C>(&output_third_handle, len, 1),
            )
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
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

macro_rules! define_sort_tuple {
    (@vec_ty $field:ident) => {
        DeviceVec<R, T>
    };

    (
        $fn_name:ident,
        $kernel_name:ident,
        ($first:ident, $($field:ident),+),
        $tuple_ty:ty
    ) => {
        pub(crate) fn $fn_name<R, T, Less>(
            $first: &DeviceVec<R, T>,
            $($field: &DeviceVec<R, T>,)+
            _less: GpuOp<Less>,
        ) -> Result<(define_sort_tuple!(@vec_ty $first), $(define_sort_tuple!(@vec_ty $field)),+), Error>
        where
            R: Runtime,
            T: CubePrimitive + CubeElement,
            Less: BinaryPredicateOp<$tuple_ty>,
        {
            $(
                if $first.len() != $field.len() {
                    return Err(Error::LengthMismatch {
                        input: $field.len(),
                        output: $first.len(),
                    });
                }
            )+

            let len = $first.len();
            let client = $first.policy().client();
            if len <= 1 {
                return Ok((
                    DeviceVec::from_handle($first.policy().clone(), $first.handle.clone(), len),
                    $(
                        DeviceVec::from_handle($field.policy().clone(), $field.handle.clone(), len),
                    )+
                ));
            }

            let num_blocks = len.div_ceil(BLOCK_ORDERING_SIZE as usize);
            let num_blocks_u32 =
                u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
            let scratch_a = vec![
                {
                    let _ = &$first;
                    client.empty(len * std::mem::size_of::<T>())
                },
                $(
                    {
                        let _ = &$field;
                        client.empty(len * std::mem::size_of::<T>())
                    },
                )+
            ];
            let scratch_b = vec![
                {
                    let _ = &$first;
                    client.empty(len * std::mem::size_of::<T>())
                },
                $(
                    {
                        let _ = &$field;
                        client.empty(len * std::mem::size_of::<T>())
                    },
                )+
            ];
            let mut input_handles = vec![
                $first.handle.clone(),
                $(
                    $field.handle.clone(),
                )+
            ];
            let mut output_handles = scratch_a.clone();
            let mut next_uses_a = false;
            let mut width = 1usize;

            while width < len {
                let width_u32 =
                    u32::try_from(width).map_err(|_| Error::LengthTooLarge { len: width })?;
                let width_handle = client.create_from_slice(u32::as_bytes(&[width_u32]));
                let mut input_iter = input_handles.iter();
                let mut output_iter = output_handles.iter();
                unsafe {
                    $kernel_name::launch_unchecked::<T, Less, R>(
                        client,
                        CubeCount::Static(num_blocks_u32, 1, 1),
                        CubeDim::new_1d(BLOCK_ORDERING_SIZE),
                        ArrayArg::from_raw_parts::<T>(
                            input_iter.next().expect("tuple sort input handle"),
                            len,
                            1,
                        ),
                        $(
                            ArrayArg::from_raw_parts::<T>(
                                {
                                    let _ = &$field;
                                    input_iter.next().expect("tuple sort input handle")
                                },
                                len,
                                1,
                            ),
                        )+
                        ArrayArg::from_raw_parts::<u32>(&width_handle, 1, 1),
                        ArrayArg::from_raw_parts::<T>(
                            output_iter.next().expect("tuple sort output handle"),
                            len,
                            1,
                        ),
                        $(
                            ArrayArg::from_raw_parts::<T>(
                                {
                                    let _ = &$field;
                                    output_iter.next().expect("tuple sort output handle")
                                },
                                len,
                                1,
                            ),
                        )+
                    )
                    .map_err(|err| Error::Launch {
                        message: format!("{err:?}"),
                    })?;
                }

                input_handles = output_handles.clone();
                output_handles = if next_uses_a {
                    scratch_a.clone()
                } else {
                    scratch_b.clone()
                };
                next_uses_a = !next_uses_a;
                width *= 2;
            }

            let mut input_iter = input_handles.into_iter();
            Ok((
                DeviceVec::from_handle(
                    $first.policy().clone(),
                    input_iter.next().expect("tuple sort result handle"),
                    len,
                ),
                $(
                    DeviceVec::from_handle(
                        $field.policy().clone(),
                        input_iter.next().expect("tuple sort result handle"),
                        len,
                    ),
                )+
            ))
        }
    };
}

define_sort_tuple!(
    sort_tuple4,
    merge_sort_tuple4_pass_kernel,
    (first, second, third, fourth),
    (T, T, T, T)
);
define_sort_tuple!(
    sort_tuple5,
    merge_sort_tuple5_pass_kernel,
    (first, second, third, fourth, fifth),
    (T, T, T, T, T)
);
define_sort_tuple!(
    sort_tuple6,
    merge_sort_tuple6_pass_kernel,
    (first, second, third, fourth, fifth, sixth),
    (T, T, T, T, T, T)
);
define_sort_tuple!(
    sort_tuple7,
    merge_sort_tuple7_pass_kernel,
    (first, second, third, fourth, fifth, sixth, seventh),
    (T, T, T, T, T, T, T)
);
define_sort_tuple!(
    sort_tuple8,
    merge_sort_tuple8_pass_kernel,
    (first, second, third, fourth, fifth, sixth, seventh, eighth),
    (T, T, T, T, T, T, T, T)
);
define_sort_tuple!(
    sort_tuple9,
    merge_sort_tuple9_pass_kernel,
    (
        first, second, third, fourth, fifth, sixth, seventh, eighth, ninth
    ),
    (T, T, T, T, T, T, T, T, T)
);
define_sort_tuple!(
    sort_tuple10,
    merge_sort_tuple10_pass_kernel,
    (
        first, second, third, fourth, fifth, sixth, seventh, eighth, ninth, tenth
    ),
    (T, T, T, T, T, T, T, T, T, T)
);
define_sort_tuple!(
    sort_tuple11,
    merge_sort_tuple11_pass_kernel,
    (
        first, second, third, fourth, fifth, sixth, seventh, eighth, ninth, tenth, eleventh
    ),
    (T, T, T, T, T, T, T, T, T, T, T)
);
define_sort_tuple!(
    sort_tuple12,
    merge_sort_tuple12_pass_kernel,
    (
        first, second, third, fourth, fifth, sixth, seventh, eighth, ninth, tenth, eleventh,
        twelfth
    ),
    (T, T, T, T, T, T, T, T, T, T, T, T)
);

#[allow(dead_code)]
pub(crate) fn radix_sort_u32<R>(input: &DeviceVec<R, u32>) -> Result<DeviceVec<R, u32>, Error>
where
    R: Runtime,
{
    let client = input.policy().client();
    let len = input.len();
    let num_blocks = len.div_ceil(BLOCK_ORDERING_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    if len <= 1 {
        return Ok(DeviceVec::from_handle(
            input.policy().clone(),
            input.handle.clone(),
            len,
        ));
    }

    let scratch_a = client.empty(len * std::mem::size_of::<u32>());
    let scratch_b = client.empty(len * std::mem::size_of::<u32>());
    let mut input_handle = input.handle.clone();
    let mut output_handle = scratch_a.clone();
    let mut next_uses_a = false;
    let histogram_len = num_blocks * RADIX_DIGITS;
    let histogram_handle = client.empty(histogram_len * std::mem::size_of::<u32>());

    for shift in (0_u32..32).step_by(4) {
        let shift_handle = client.create_from_slice(u32::as_bytes(&[shift]));
        unsafe {
            radix_digit_histogram_u32_kernel::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(RADIX_DIGITS as u32),
                ArrayArg::from_raw_parts::<u32>(&input_handle, len, 1),
                ArrayArg::from_raw_parts::<u32>(&shift_handle, 1, 1),
                ArrayArg::from_raw_parts::<u32>(&histogram_handle, histogram_len, 1),
            )
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
        }

        let histogram_prefix_handle =
            inclusive_scan_u32::<R>(client, &histogram_handle, histogram_len)?;
        unsafe {
            radix_digit_scatter_u32_kernel::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_ORDERING_SIZE),
                ArrayArg::from_raw_parts::<u32>(&input_handle, len, 1),
                ArrayArg::from_raw_parts::<u32>(&shift_handle, 1, 1),
                ArrayArg::from_raw_parts::<u32>(&histogram_handle, histogram_len, 1),
                ArrayArg::from_raw_parts::<u32>(&histogram_prefix_handle, histogram_len, 1),
                ArrayArg::from_raw_parts::<u32>(&output_handle, len, 1),
            )
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
        }

        input_handle = output_handle.clone();
        output_handle = if next_uses_a {
            scratch_a.clone()
        } else {
            scratch_b.clone()
        };
        next_uses_a = !next_uses_a;
    }

    Ok(DeviceVec::from_handle(
        input.policy().clone(),
        input_handle,
        len,
    ))
}

#[allow(dead_code)]
pub(crate) fn radix_sort_by_key_u32<R, T>(
    keys: &DeviceVec<R, u32>,
    values: &DeviceVec<R, T>,
) -> Result<(DeviceVec<R, u32>, DeviceVec<R, T>), Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    if keys.len() != values.len() {
        return Err(Error::LengthMismatch {
            input: values.len(),
            output: keys.len(),
        });
    }

    let client = keys.policy().client();
    let len = keys.len();
    let num_blocks = len.div_ceil(BLOCK_ORDERING_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    if len <= 1 {
        return Ok((
            DeviceVec::from_handle(keys.policy().clone(), keys.handle.clone(), len),
            DeviceVec::from_handle(values.policy().clone(), values.handle.clone(), len),
        ));
    }

    let scratch_keys_a = client.empty(len * std::mem::size_of::<u32>());
    let scratch_values_a = client.empty(len * std::mem::size_of::<T>());
    let scratch_keys_b = client.empty(len * std::mem::size_of::<u32>());
    let scratch_values_b = client.empty(len * std::mem::size_of::<T>());
    let mut input_key_handle = keys.handle.clone();
    let mut input_value_handle = values.handle.clone();
    let mut output_key_handle = scratch_keys_a.clone();
    let mut output_value_handle = scratch_values_a.clone();
    let mut next_uses_a = false;
    let histogram_len = num_blocks * RADIX_DIGITS;
    let histogram_handle = client.empty(histogram_len * std::mem::size_of::<u32>());

    for shift in (0_u32..32).step_by(4) {
        let shift_handle = client.create_from_slice(u32::as_bytes(&[shift]));
        unsafe {
            radix_digit_histogram_u32_kernel::launch_unchecked::<R>(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(RADIX_DIGITS as u32),
                ArrayArg::from_raw_parts::<u32>(&input_key_handle, len, 1),
                ArrayArg::from_raw_parts::<u32>(&shift_handle, 1, 1),
                ArrayArg::from_raw_parts::<u32>(&histogram_handle, histogram_len, 1),
            )
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
        }

        let histogram_prefix_handle =
            inclusive_scan_u32::<R>(client, &histogram_handle, histogram_len)?;
        unsafe {
            radix_digit_scatter_by_key_u32_kernel::launch_unchecked::<T, R>(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_ORDERING_SIZE),
                ArrayArg::from_raw_parts::<u32>(&input_key_handle, len, 1),
                ArrayArg::from_raw_parts::<T>(&input_value_handle, len, 1),
                ArrayArg::from_raw_parts::<u32>(&shift_handle, 1, 1),
                ArrayArg::from_raw_parts::<u32>(&histogram_handle, histogram_len, 1),
                ArrayArg::from_raw_parts::<u32>(&histogram_prefix_handle, histogram_len, 1),
                ArrayArg::from_raw_parts::<u32>(&output_key_handle, len, 1),
                ArrayArg::from_raw_parts::<T>(&output_value_handle, len, 1),
            )
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
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
    }

    Ok((
        DeviceVec::from_handle(keys.policy().clone(), input_key_handle, len),
        DeviceVec::from_handle(values.policy().clone(), input_value_handle, len),
    ))
}

pub(crate) fn merge<R, T, Less>(
    left: &DeviceVec<R, T>,
    right: &DeviceVec<R, T>,
    _less: GpuOp<Less>,
) -> Result<DeviceVec<R, T>, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Less: BinaryPredicateOp<T>,
{
    let len = left.len() + right.len();
    let num_blocks = len.div_ceil(BLOCK_ORDERING_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let client = left.policy().client();
    let output_handle = client.empty(len * std::mem::size_of::<T>());

    if len != 0 {
        unsafe {
            merge_path_kernel::launch_unchecked::<T, Less, R>(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_ORDERING_SIZE),
                ArrayArg::from_raw_parts::<T>(&output_handle, len, 1),
                ArrayArg::from_raw_parts::<T>(&left.handle, left.len(), 1),
                ArrayArg::from_raw_parts::<T>(&right.handle, right.len(), 1),
            )
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
        }
    }

    Ok(DeviceVec::from_handle(
        left.policy().clone(),
        output_handle,
        len,
    ))
}

pub(crate) fn merge_by_key<R, K, T, Less>(
    left_keys: &DeviceVec<R, K>,
    left_values: &DeviceVec<R, T>,
    right_keys: &DeviceVec<R, K>,
    right_values: &DeviceVec<R, T>,
    _less: GpuOp<Less>,
) -> Result<(DeviceVec<R, K>, DeviceVec<R, T>), Error>
where
    R: Runtime,
    K: CubePrimitive + CubeElement,
    T: CubePrimitive + CubeElement,
    Less: BinaryPredicateOp<K>,
{
    if left_keys.len() != left_values.len() {
        return Err(Error::LengthMismatch {
            input: left_values.len(),
            output: left_keys.len(),
        });
    }
    if right_keys.len() != right_values.len() {
        return Err(Error::LengthMismatch {
            input: right_values.len(),
            output: right_keys.len(),
        });
    }

    let len = left_keys.len() + right_keys.len();
    let num_blocks = len.div_ceil(BLOCK_ORDERING_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let client = left_keys.policy().client();
    let out_key_handle = client.empty(len * std::mem::size_of::<K>());
    let out_value_handle = client.empty(len * std::mem::size_of::<T>());

    if len != 0 {
        unsafe {
            merge_by_key_path_kernel::launch_unchecked::<K, T, Less, R>(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_ORDERING_SIZE),
                ArrayArg::from_raw_parts::<K>(&left_keys.handle, left_keys.len(), 1),
                ArrayArg::from_raw_parts::<T>(&left_values.handle, left_values.len(), 1),
                ArrayArg::from_raw_parts::<K>(&right_keys.handle, right_keys.len(), 1),
                ArrayArg::from_raw_parts::<T>(&right_values.handle, right_values.len(), 1),
                ArrayArg::from_raw_parts::<K>(&out_key_handle, len, 1),
                ArrayArg::from_raw_parts::<T>(&out_value_handle, len, 1),
            )
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
        }
    }

    Ok((
        DeviceVec::from_handle(left_keys.policy().clone(), out_key_handle, len),
        DeviceVec::from_handle(left_values.policy().clone(), out_value_handle, len),
    ))
}

pub(crate) fn set_union<R, T, Less>(
    left: &DeviceVec<R, T>,
    right: &DeviceVec<R, T>,
    _less: GpuOp<Less>,
) -> Result<DeviceVec<R, T>, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Less: BinaryPredicateOp<T>,
{
    let right_only = set_difference(right, left, GpuOp::<Less>::new())?;
    merge(left, &right_only, GpuOp::<Less>::new())
}

pub(crate) fn set_intersection<R, T, Less>(
    left: &DeviceVec<R, T>,
    right: &DeviceVec<R, T>,
    _less: GpuOp<Less>,
) -> Result<DeviceVec<R, T>, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Less: BinaryPredicateOp<T>,
{
    membership_compact::<R, T, Less>(left, right, true)
}

pub(crate) fn set_difference<R, T, Less>(
    left: &DeviceVec<R, T>,
    right: &DeviceVec<R, T>,
    _less: GpuOp<Less>,
) -> Result<DeviceVec<R, T>, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Less: BinaryPredicateOp<T>,
{
    membership_compact::<R, T, Less>(left, right, false)
}

pub(crate) fn set_symmetric_difference<R, T, Less>(
    left: &DeviceVec<R, T>,
    right: &DeviceVec<R, T>,
    less: GpuOp<Less>,
) -> Result<DeviceVec<R, T>, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Less: BinaryPredicateOp<T>,
{
    let left_only = set_difference(left, right, GpuOp::<Less>::new())?;
    let right_only = set_difference(right, left, GpuOp::<Less>::new())?;
    merge(&left_only, &right_only, less)
}

fn membership_compact<R, T, Less>(
    candidates: &DeviceVec<R, T>,
    sorted: &DeviceVec<R, T>,
    keep_present: bool,
) -> Result<DeviceVec<R, T>, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Less: BinaryPredicateOp<T>,
{
    let len_u32 = u32::try_from(candidates.len()).map_err(|_| Error::LengthTooLarge {
        len: candidates.len(),
    })?;
    if candidates.len() == 0 {
        return Ok(DeviceVec::from_handle(
            candidates.policy().clone(),
            candidates.policy().client().empty(0),
            0,
        ));
    }

    let client = candidates.policy().client();
    let num_blocks = candidates.len().div_ceil(BLOCK_ORDERING_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let keep_values = [if keep_present { 1_u32 } else { 0_u32 }];
    let keep_handle = client.create_from_slice(u32::as_bytes(&keep_values));
    let flag_handle = client.empty(candidates.len() * std::mem::size_of::<u32>());
    unsafe {
        sorted_membership_flags_kernel::launch_unchecked::<T, Less, R>(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(BLOCK_ORDERING_SIZE),
            ArrayArg::from_raw_parts::<T>(&candidates.handle, candidates.len(), 1),
            ArrayArg::from_raw_parts::<T>(&sorted.handle, sorted.len(), 1),
            ArrayArg::from_raw_parts::<u32>(&keep_handle, 1, 1),
            ArrayArg::from_raw_parts::<u32>(&flag_handle, candidates.len(), 1),
        )
        .map_err(|err| Error::Launch {
            message: format!("{err:?}"),
        })?;
    }

    let handles = select::handles_from_flags(
        candidates.policy(),
        candidates.len(),
        len_u32,
        flag_handle,
        candidates.handle.clone(),
    )?;
    select::compact(candidates.policy(), handles)
}
