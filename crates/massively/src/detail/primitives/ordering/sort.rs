use crate::{
    device::{DeviceVec, KernelColumn, KernelColumnAt, ReadOnlyKernelColumn, S0},
    error::Error,
    expr::DeviceGpuExpr,
    kernels::*,
    op::{BinaryPredicateOp, GpuOp},
    policy::CubePolicy,
    primitives::{ensure_same_len, workspace::Workspace},
};
use cubecl::prelude::*;

use super::BLOCK_ORDERING_SIZE;

pub(crate) fn sort_input_with_policy<Source, Less>(
    policy: &CubePolicy<Source::Runtime>,
    input: &Source,
    _less: GpuOp<Less>,
) -> Result<DeviceVec<Source::Runtime, Source::Item>, Error>
where
    Source: ReadOnlyKernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Less: BinaryPredicateOp<Source::Item>,
{
    input.validate()?;
    let len = input.len();
    let num_blocks = len.div_ceil(BLOCK_ORDERING_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let client = policy.client();
    if len == 0 {
        return Ok(policy.empty_device_vec());
    }

    let workspace = Workspace::new(policy);
    let (scratch_a, scratch_b) = workspace.alloc_pair::<Source::Item>(len);
    let bindings = input.stage(policy)?;
    let slot_offsets = bindings.slot_offsets_handle(client)?;
    let (slot2, slot2_len) = bindings.slots.get(2).unwrap_or(&bindings.slots[0]);
    let (slot3, slot3_len) = bindings.slots.get(3).unwrap_or(&bindings.slots[0]);
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));

    unsafe {
        merge_sort_expr_first_pass_kernel::launch_unchecked::<
            Source::Item,
            Source::Expr,
            Less,
            Source::Runtime,
        >(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(BLOCK_ORDERING_SIZE),
            unsafe { BufferArg::from_raw_parts(bindings.input.clone(), bindings.input_len) },
            unsafe { BufferArg::from_raw_parts(bindings.rhs.clone(), bindings.rhs_len) },
            unsafe { BufferArg::from_raw_parts(slot2.clone(), *slot2_len) },
            unsafe { BufferArg::from_raw_parts(slot3.clone(), *slot3_len) },
            unsafe { BufferArg::from_raw_parts(slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(scratch_a.clone(), len) },
        );
    }

    let mut input_handle = scratch_a.clone();
    let mut output_handle = scratch_b.clone();
    let mut next_uses_a = true;
    let mut width = 2usize;

    while width < len {
        let width_u32 = u32::try_from(width).map_err(|_| Error::LengthTooLarge { len: width })?;
        let width_handle = client.create_from_slice(u32::as_bytes(&[width_u32]));
        unsafe {
            merge_sort_pass_kernel::launch_unchecked::<Source::Item, Less, Source::Runtime>(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_ORDERING_SIZE),
                unsafe { BufferArg::from_raw_parts(input_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(width_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(output_handle.clone(), len) },
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

    Ok(DeviceVec::from_handle(policy.id(), input_handle, len))
}

pub(crate) fn sort_by_key_input_with_policy<KeySource, ValueSource, Less>(
    policy: &CubePolicy<KeySource::Runtime>,
    keys: &KeySource,
    values: &ValueSource,
    _less: GpuOp<Less>,
) -> Result<
    (
        DeviceVec<KeySource::Runtime, KeySource::Item>,
        DeviceVec<KeySource::Runtime, ValueSource::Item>,
    ),
    Error,
>
where
    KeySource: KernelColumn + KernelColumnAt<S0>,
    ValueSource: KernelColumn<Runtime = KeySource::Runtime> + KernelColumnAt<S0>,
    KeySource::Item: CubePrimitive + CubeElement,
    ValueSource::Item: CubePrimitive + CubeElement,
    KeySource::Expr: DeviceGpuExpr<KeySource::Item>,
    ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
    Less: BinaryPredicateOp<KeySource::Item>,
{
    keys.validate()?;
    values.validate()?;
    ensure_same_len(values.len(), keys.len())?;

    let len = keys.len();
    if len == 0 {
        return Ok((policy.empty_device_vec(), policy.empty_device_vec()));
    }

    let client = policy.client();
    let num_blocks = len.div_ceil(BLOCK_ORDERING_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let workspace = Workspace::new(policy);
    let (scratch_keys_a, scratch_keys_b) = workspace.alloc_pair::<KeySource::Item>(len);
    let (scratch_values_a, scratch_values_b) = workspace.alloc_pair::<ValueSource::Item>(len);
    let key_bindings = keys.stage(policy)?;
    let value_bindings = values.stage(policy)?;
    let key_offsets = key_bindings.slot_offsets_handle(client)?;
    let value_offsets = value_bindings.slot_offsets_handle(client)?;
    let (key_slot2, key_slot2_len) = key_bindings.slots.get(2).unwrap_or(&key_bindings.slots[0]);
    let (key_slot3, key_slot3_len) = key_bindings.slots.get(3).unwrap_or(&key_bindings.slots[0]);
    let (value_slot2, value_slot2_len) = value_bindings
        .slots
        .get(2)
        .unwrap_or(&value_bindings.slots[0]);
    let (value_slot3, value_slot3_len) = value_bindings
        .slots
        .get(3)
        .unwrap_or(&value_bindings.slots[0]);
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));

    unsafe {
        merge_sort_by_key_expr_first_pass_kernel::launch_unchecked::<
            KeySource::Item,
            ValueSource::Item,
            KeySource::Expr,
            ValueSource::Expr,
            Less,
            KeySource::Runtime,
        >(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(BLOCK_ORDERING_SIZE),
            unsafe {
                BufferArg::from_raw_parts(key_bindings.input.clone(), key_bindings.input_len)
            },
            unsafe { BufferArg::from_raw_parts(key_bindings.rhs.clone(), key_bindings.rhs_len) },
            unsafe { BufferArg::from_raw_parts(key_slot2.clone(), *key_slot2_len) },
            unsafe { BufferArg::from_raw_parts(key_slot3.clone(), *key_slot3_len) },
            unsafe { BufferArg::from_raw_parts(key_offsets.clone(), 4) },
            unsafe {
                BufferArg::from_raw_parts(value_bindings.input.clone(), value_bindings.input_len)
            },
            unsafe {
                BufferArg::from_raw_parts(value_bindings.rhs.clone(), value_bindings.rhs_len)
            },
            unsafe { BufferArg::from_raw_parts(value_slot2.clone(), *value_slot2_len) },
            unsafe { BufferArg::from_raw_parts(value_slot3.clone(), *value_slot3_len) },
            unsafe { BufferArg::from_raw_parts(value_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(scratch_keys_a.clone(), len) },
            unsafe { BufferArg::from_raw_parts(scratch_values_a.clone(), len) },
        );
    }

    let mut input_key_handle = scratch_keys_a.clone();
    let mut input_value_handle = scratch_values_a.clone();
    let mut output_key_handle = scratch_keys_b.clone();
    let mut output_value_handle = scratch_values_b.clone();
    let mut next_uses_a = true;
    let mut width = 2usize;

    while width < len {
        let width_u32 = u32::try_from(width).map_err(|_| Error::LengthTooLarge { len: width })?;
        let width_handle = client.create_from_slice(u32::as_bytes(&[width_u32]));
        unsafe {
            merge_sort_by_key_pass_kernel::launch_unchecked::<
                KeySource::Item,
                ValueSource::Item,
                Less,
                KeySource::Runtime,
            >(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_ORDERING_SIZE),
                unsafe { BufferArg::from_raw_parts(input_key_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(input_value_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(width_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(output_key_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(output_value_handle.clone(), len) },
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
        DeviceVec::from_handle(policy.id(), input_key_handle, len),
        DeviceVec::from_handle(policy.id(), input_value_handle, len),
    ))
}

pub(crate) fn sort_tuple2<R, A, B, Less>(
    policy: &CubePolicy<R>,
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
    let client = policy.client();
    if len <= 1 {
        return Ok((
            DeviceVec::from_handle(policy.id(), first.handle.clone(), len),
            DeviceVec::from_handle(policy.id(), second.handle.clone(), len),
        ));
    }

    let num_blocks = len.div_ceil(BLOCK_ORDERING_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let workspace = Workspace::new(policy);
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
        DeviceVec::from_handle(policy.id(), input_first_handle, len),
        DeviceVec::from_handle(policy.id(), input_second_handle, len),
    ))
}

pub(crate) fn sort_tuple2_input<Left, Right, Less>(
    policy: &CubePolicy<Left::Runtime>,
    first: &Left,
    second: &Right,
    _less: GpuOp<Less>,
) -> Result<
    (
        DeviceVec<Left::Runtime, Left::Item>,
        DeviceVec<Left::Runtime, Right::Item>,
    ),
    Error,
>
where
    Left: KernelColumnAt<S0>,
    Right: KernelColumnAt<S0>,
    Left: KernelColumn,
    Right: KernelColumn<Runtime = Left::Runtime>,
    Left::Item: CubePrimitive + CubeElement,
    Right::Item: CubePrimitive + CubeElement,
    Left::Expr: DeviceGpuExpr<Left::Item>,
    Right::Expr: DeviceGpuExpr<Right::Item>,
    Less: BinaryPredicateOp<(Left::Item, Right::Item)>,
{
    first.validate()?;
    second.validate()?;
    ensure_same_len(second.len(), first.len())?;

    let len = first.len();
    if len == 0 {
        return Ok((policy.empty_device_vec(), policy.empty_device_vec()));
    }

    let client = policy.client();
    let num_blocks = len.div_ceil(BLOCK_ORDERING_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let workspace = Workspace::new(policy);
    let (scratch_first_a, scratch_first_b) = workspace.alloc_pair::<Left::Item>(len);
    let (scratch_second_a, scratch_second_b) = workspace.alloc_pair::<Right::Item>(len);
    let first_bindings = first.stage(policy)?;
    let second_bindings = second.stage(policy)?;
    let first_offsets = first_bindings.slot_offsets_handle(client)?;
    let second_offsets = second_bindings.slot_offsets_handle(client)?;
    let (first_slot2, first_slot2_len) = first_bindings
        .slots
        .get(2)
        .unwrap_or(&first_bindings.slots[0]);
    let (first_slot3, first_slot3_len) = first_bindings
        .slots
        .get(3)
        .unwrap_or(&first_bindings.slots[0]);
    let (second_slot2, second_slot2_len) = second_bindings
        .slots
        .get(2)
        .unwrap_or(&second_bindings.slots[0]);
    let (second_slot3, second_slot3_len) = second_bindings
        .slots
        .get(3)
        .unwrap_or(&second_bindings.slots[0]);
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));

    unsafe {
        merge_sort_tuple2_expr_first_pass_kernel::launch_unchecked::<
            Left::Item,
            Right::Item,
            Left::Expr,
            Right::Expr,
            Less,
            Left::Runtime,
        >(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(BLOCK_ORDERING_SIZE),
            unsafe {
                BufferArg::from_raw_parts(first_bindings.input.clone(), first_bindings.input_len)
            },
            unsafe {
                BufferArg::from_raw_parts(first_bindings.rhs.clone(), first_bindings.rhs_len)
            },
            unsafe { BufferArg::from_raw_parts(first_slot2.clone(), *first_slot2_len) },
            unsafe { BufferArg::from_raw_parts(first_slot3.clone(), *first_slot3_len) },
            unsafe { BufferArg::from_raw_parts(first_offsets.clone(), 4) },
            unsafe {
                BufferArg::from_raw_parts(second_bindings.input.clone(), second_bindings.input_len)
            },
            unsafe {
                BufferArg::from_raw_parts(second_bindings.rhs.clone(), second_bindings.rhs_len)
            },
            unsafe { BufferArg::from_raw_parts(second_slot2.clone(), *second_slot2_len) },
            unsafe { BufferArg::from_raw_parts(second_slot3.clone(), *second_slot3_len) },
            unsafe { BufferArg::from_raw_parts(second_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(scratch_first_a.clone(), len) },
            unsafe { BufferArg::from_raw_parts(scratch_second_a.clone(), len) },
        );
    }

    let mut input_first_handle = scratch_first_a.clone();
    let mut input_second_handle = scratch_second_a.clone();
    let mut output_first_handle = scratch_first_b.clone();
    let mut output_second_handle = scratch_second_b.clone();
    let mut next_uses_a = true;
    let mut width = 2usize;

    while width < len {
        let width_u32 = u32::try_from(width).map_err(|_| Error::LengthTooLarge { len: width })?;
        let width_handle = client.create_from_slice(u32::as_bytes(&[width_u32]));
        unsafe {
            merge_sort_tuple2_pass_kernel::launch_unchecked::<
                Left::Item,
                Right::Item,
                Less,
                Left::Runtime,
            >(
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
        DeviceVec::from_handle(policy.id(), input_first_handle, len),
        DeviceVec::from_handle(policy.id(), input_second_handle, len),
    ))
}

pub(crate) fn sort_tuple3<R, A, B, C, Less>(
    policy: &CubePolicy<R>,
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
    let client = policy.client();
    if len <= 1 {
        return Ok((
            DeviceVec::from_handle(policy.id(), first.handle.clone(), len),
            DeviceVec::from_handle(policy.id(), second.handle.clone(), len),
            DeviceVec::from_handle(policy.id(), third.handle.clone(), len),
        ));
    }

    let num_blocks = len.div_ceil(BLOCK_ORDERING_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let workspace = Workspace::new(policy);
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
        DeviceVec::from_handle(policy.id(), input_first_handle, len),
        DeviceVec::from_handle(policy.id(), input_second_handle, len),
        DeviceVec::from_handle(policy.id(), input_third_handle, len),
    ))
}

pub(crate) fn sort_tuple3_input<First, Second, Third, Less>(
    policy: &CubePolicy<First::Runtime>,
    first: &First,
    second: &Second,
    third: &Third,
    _less: GpuOp<Less>,
) -> Result<
    (
        DeviceVec<First::Runtime, First::Item>,
        DeviceVec<First::Runtime, Second::Item>,
        DeviceVec<First::Runtime, Third::Item>,
    ),
    Error,
>
where
    First: KernelColumn + KernelColumnAt<S0>,
    Second: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
    Third: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
    First::Item: CubePrimitive + CubeElement,
    Second::Item: CubePrimitive + CubeElement,
    Third::Item: CubePrimitive + CubeElement,
    First::Expr: DeviceGpuExpr<First::Item>,
    Second::Expr: DeviceGpuExpr<Second::Item>,
    Third::Expr: DeviceGpuExpr<Third::Item>,
    Less: BinaryPredicateOp<(First::Item, Second::Item, Third::Item)>,
{
    first.validate()?;
    second.validate()?;
    third.validate()?;
    ensure_same_len(second.len(), first.len())?;
    ensure_same_len(third.len(), first.len())?;

    let len = first.len();
    if len == 0 {
        return Ok((
            policy.empty_device_vec(),
            policy.empty_device_vec(),
            policy.empty_device_vec(),
        ));
    }

    let client = policy.client();
    let num_blocks = len.div_ceil(BLOCK_ORDERING_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let workspace = Workspace::new(policy);
    let (scratch_first_a, scratch_first_b) = workspace.alloc_pair::<First::Item>(len);
    let (scratch_second_a, scratch_second_b) = workspace.alloc_pair::<Second::Item>(len);
    let (scratch_third_a, scratch_third_b) = workspace.alloc_pair::<Third::Item>(len);
    let first_bindings = first.stage(policy)?;
    let second_bindings = second.stage(policy)?;
    let third_bindings = third.stage(policy)?;
    let first_offsets = first_bindings.slot_offsets_handle(client)?;
    let second_offsets = second_bindings.slot_offsets_handle(client)?;
    let third_offsets = third_bindings.slot_offsets_handle(client)?;
    let (first_slot2, first_slot2_len) = first_bindings
        .slots
        .get(2)
        .unwrap_or(&first_bindings.slots[0]);
    let (first_slot3, first_slot3_len) = first_bindings
        .slots
        .get(3)
        .unwrap_or(&first_bindings.slots[0]);
    let (second_slot2, second_slot2_len) = second_bindings
        .slots
        .get(2)
        .unwrap_or(&second_bindings.slots[0]);
    let (second_slot3, second_slot3_len) = second_bindings
        .slots
        .get(3)
        .unwrap_or(&second_bindings.slots[0]);
    let (third_slot2, third_slot2_len) = third_bindings
        .slots
        .get(2)
        .unwrap_or(&third_bindings.slots[0]);
    let (third_slot3, third_slot3_len) = third_bindings
        .slots
        .get(3)
        .unwrap_or(&third_bindings.slots[0]);
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));

    unsafe {
        merge_sort_tuple3_expr_first_pass_kernel::launch_unchecked::<
            First::Item,
            Second::Item,
            Third::Item,
            First::Expr,
            Second::Expr,
            Third::Expr,
            Less,
            First::Runtime,
        >(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(BLOCK_ORDERING_SIZE),
            unsafe {
                BufferArg::from_raw_parts(first_bindings.input.clone(), first_bindings.input_len)
            },
            unsafe {
                BufferArg::from_raw_parts(first_bindings.rhs.clone(), first_bindings.rhs_len)
            },
            unsafe { BufferArg::from_raw_parts(first_slot2.clone(), *first_slot2_len) },
            unsafe { BufferArg::from_raw_parts(first_slot3.clone(), *first_slot3_len) },
            unsafe { BufferArg::from_raw_parts(first_offsets.clone(), 4) },
            unsafe {
                BufferArg::from_raw_parts(second_bindings.input.clone(), second_bindings.input_len)
            },
            unsafe {
                BufferArg::from_raw_parts(second_bindings.rhs.clone(), second_bindings.rhs_len)
            },
            unsafe { BufferArg::from_raw_parts(second_slot2.clone(), *second_slot2_len) },
            unsafe { BufferArg::from_raw_parts(second_slot3.clone(), *second_slot3_len) },
            unsafe { BufferArg::from_raw_parts(second_offsets.clone(), 4) },
            unsafe {
                BufferArg::from_raw_parts(third_bindings.input.clone(), third_bindings.input_len)
            },
            unsafe {
                BufferArg::from_raw_parts(third_bindings.rhs.clone(), third_bindings.rhs_len)
            },
            unsafe { BufferArg::from_raw_parts(third_slot2.clone(), *third_slot2_len) },
            unsafe { BufferArg::from_raw_parts(third_slot3.clone(), *third_slot3_len) },
            unsafe { BufferArg::from_raw_parts(third_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(scratch_first_a.clone(), len) },
            unsafe { BufferArg::from_raw_parts(scratch_second_a.clone(), len) },
            unsafe { BufferArg::from_raw_parts(scratch_third_a.clone(), len) },
        );
    }

    let mut input_first_handle = scratch_first_a.clone();
    let mut input_second_handle = scratch_second_a.clone();
    let mut input_third_handle = scratch_third_a.clone();
    let mut output_first_handle = scratch_first_b.clone();
    let mut output_second_handle = scratch_second_b.clone();
    let mut output_third_handle = scratch_third_b.clone();
    let mut next_uses_a = true;
    let mut width = 2usize;

    while width < len {
        let width_u32 = u32::try_from(width).map_err(|_| Error::LengthTooLarge { len: width })?;
        let width_handle = client.create_from_slice(u32::as_bytes(&[width_u32]));
        unsafe {
            merge_sort_tuple3_pass_kernel::launch_unchecked::<
                First::Item,
                Second::Item,
                Third::Item,
                Less,
                First::Runtime,
            >(
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
        DeviceVec::from_handle(policy.id(), input_first_handle, len),
        DeviceVec::from_handle(policy.id(), input_second_handle, len),
        DeviceVec::from_handle(policy.id(), input_third_handle, len),
    ))
}

pub(crate) fn sort_tuple2_by_key<R, A, B, T, Less>(
    policy: &CubePolicy<R>,
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
    let client = policy.client();
    if len <= 1 {
        return Ok((
            DeviceVec::from_handle(policy.id(), key_a.handle.clone(), len),
            DeviceVec::from_handle(policy.id(), key_b.handle.clone(), len),
            DeviceVec::from_handle(policy.id(), values.handle.clone(), len),
        ));
    }

    let num_blocks = len.div_ceil(BLOCK_ORDERING_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let workspace = Workspace::new(policy);
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
        DeviceVec::from_handle(policy.id(), input_a_handle, len),
        DeviceVec::from_handle(policy.id(), input_b_handle, len),
        DeviceVec::from_handle(policy.id(), input_value_handle, len),
    ))
}

pub(crate) fn sort_tuple2_by_key_input<KeyA, KeyB, ValueSource, Less>(
    policy: &CubePolicy<KeyA::Runtime>,
    key_a: &KeyA,
    key_b: &KeyB,
    values: &ValueSource,
    _less: GpuOp<Less>,
) -> Result<
    (
        DeviceVec<KeyA::Runtime, KeyA::Item>,
        DeviceVec<KeyA::Runtime, KeyB::Item>,
        DeviceVec<KeyA::Runtime, ValueSource::Item>,
    ),
    Error,
>
where
    KeyA: KernelColumn + KernelColumnAt<S0>,
    KeyB: KernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    ValueSource: KernelColumn<Runtime = KeyA::Runtime> + KernelColumnAt<S0>,
    KeyA::Item: CubePrimitive + CubeElement,
    KeyB::Item: CubePrimitive + CubeElement,
    ValueSource::Item: CubePrimitive + CubeElement,
    KeyA::Expr: DeviceGpuExpr<KeyA::Item>,
    KeyB::Expr: DeviceGpuExpr<KeyB::Item>,
    ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
    Less: BinaryPredicateOp<(KeyA::Item, KeyB::Item)>,
{
    key_a.validate()?;
    key_b.validate()?;
    values.validate()?;
    ensure_same_len(key_b.len(), key_a.len())?;
    ensure_same_len(values.len(), key_a.len())?;

    let len = key_a.len();
    if len == 0 {
        return Ok((
            policy.empty_device_vec(),
            policy.empty_device_vec(),
            policy.empty_device_vec(),
        ));
    }

    let client = policy.client();
    let num_blocks = len.div_ceil(BLOCK_ORDERING_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let workspace = Workspace::new(policy);
    let (scratch_a_a, scratch_a_b) = workspace.alloc_pair::<KeyA::Item>(len);
    let (scratch_b_a, scratch_b_b) = workspace.alloc_pair::<KeyB::Item>(len);
    let (scratch_values_a, scratch_values_b) = workspace.alloc_pair::<ValueSource::Item>(len);
    let a_bindings = key_a.stage(policy)?;
    let b_bindings = key_b.stage(policy)?;
    let value_bindings = values.stage(policy)?;
    let a_offsets = a_bindings.slot_offsets_handle(client)?;
    let b_offsets = b_bindings.slot_offsets_handle(client)?;
    let value_offsets = value_bindings.slot_offsets_handle(client)?;
    let (a_slot2, a_slot2_len) = a_bindings.slots.get(2).unwrap_or(&a_bindings.slots[0]);
    let (a_slot3, a_slot3_len) = a_bindings.slots.get(3).unwrap_or(&a_bindings.slots[0]);
    let (b_slot2, b_slot2_len) = b_bindings.slots.get(2).unwrap_or(&b_bindings.slots[0]);
    let (b_slot3, b_slot3_len) = b_bindings.slots.get(3).unwrap_or(&b_bindings.slots[0]);
    let (value_slot2, value_slot2_len) = value_bindings
        .slots
        .get(2)
        .unwrap_or(&value_bindings.slots[0]);
    let (value_slot3, value_slot3_len) = value_bindings
        .slots
        .get(3)
        .unwrap_or(&value_bindings.slots[0]);
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));

    unsafe {
        merge_sort_tuple2_by_key_expr_first_pass_kernel::launch_unchecked::<
            KeyA::Item,
            KeyB::Item,
            ValueSource::Item,
            KeyA::Expr,
            KeyB::Expr,
            ValueSource::Expr,
            Less,
            KeyA::Runtime,
        >(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(BLOCK_ORDERING_SIZE),
            unsafe { BufferArg::from_raw_parts(a_bindings.input.clone(), a_bindings.input_len) },
            unsafe { BufferArg::from_raw_parts(a_bindings.rhs.clone(), a_bindings.rhs_len) },
            unsafe { BufferArg::from_raw_parts(a_slot2.clone(), *a_slot2_len) },
            unsafe { BufferArg::from_raw_parts(a_slot3.clone(), *a_slot3_len) },
            unsafe { BufferArg::from_raw_parts(a_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(b_bindings.input.clone(), b_bindings.input_len) },
            unsafe { BufferArg::from_raw_parts(b_bindings.rhs.clone(), b_bindings.rhs_len) },
            unsafe { BufferArg::from_raw_parts(b_slot2.clone(), *b_slot2_len) },
            unsafe { BufferArg::from_raw_parts(b_slot3.clone(), *b_slot3_len) },
            unsafe { BufferArg::from_raw_parts(b_offsets.clone(), 4) },
            unsafe {
                BufferArg::from_raw_parts(value_bindings.input.clone(), value_bindings.input_len)
            },
            unsafe {
                BufferArg::from_raw_parts(value_bindings.rhs.clone(), value_bindings.rhs_len)
            },
            unsafe { BufferArg::from_raw_parts(value_slot2.clone(), *value_slot2_len) },
            unsafe { BufferArg::from_raw_parts(value_slot3.clone(), *value_slot3_len) },
            unsafe { BufferArg::from_raw_parts(value_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(scratch_a_a.clone(), len) },
            unsafe { BufferArg::from_raw_parts(scratch_b_a.clone(), len) },
            unsafe { BufferArg::from_raw_parts(scratch_values_a.clone(), len) },
        );
    }

    let mut input_a_handle = scratch_a_a.clone();
    let mut input_b_handle = scratch_b_a.clone();
    let mut input_value_handle = scratch_values_a.clone();
    let mut output_a_handle = scratch_a_b.clone();
    let mut output_b_handle = scratch_b_b.clone();
    let mut output_value_handle = scratch_values_b.clone();
    let mut next_uses_a = true;
    let mut width = 2usize;

    while width < len {
        let width_u32 = u32::try_from(width).map_err(|_| Error::LengthTooLarge { len: width })?;
        let width_handle = client.create_from_slice(u32::as_bytes(&[width_u32]));
        unsafe {
            merge_sort_tuple2_by_key_pass_kernel::launch_unchecked::<
                KeyA::Item,
                KeyB::Item,
                ValueSource::Item,
                Less,
                KeyA::Runtime,
            >(
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
        DeviceVec::from_handle(policy.id(), input_a_handle, len),
        DeviceVec::from_handle(policy.id(), input_b_handle, len),
        DeviceVec::from_handle(policy.id(), input_value_handle, len),
    ))
}

pub(crate) fn sort_tuple3_by_key<R, A, B, C, T, Less>(
    policy: &CubePolicy<R>,
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
    let client = policy.client();
    if len <= 1 {
        return Ok((
            DeviceVec::from_handle(policy.id(), key_a.handle.clone(), len),
            DeviceVec::from_handle(policy.id(), key_b.handle.clone(), len),
            DeviceVec::from_handle(policy.id(), key_c.handle.clone(), len),
            DeviceVec::from_handle(policy.id(), values.handle.clone(), len),
        ));
    }

    let num_blocks = len.div_ceil(BLOCK_ORDERING_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let workspace = Workspace::new(policy);
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
        DeviceVec::from_handle(policy.id(), input_a_handle, len),
        DeviceVec::from_handle(policy.id(), input_b_handle, len),
        DeviceVec::from_handle(policy.id(), input_c_handle, len),
        DeviceVec::from_handle(policy.id(), input_value_handle, len),
    ))
}
