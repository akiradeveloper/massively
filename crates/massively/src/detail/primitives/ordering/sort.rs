use crate::{
    detail::op::kernel::BinaryPredicateOp,
    device::{DeviceColumnView, DeviceVec, KernelColumn, KernelColumnAt, ReadOnlyKernelColumn, S0},
    error::Error,
    expr::DeviceGpuExpr,
    index::MIndex,
    kernels::*,
    op::GpuOp,
    policy::CubePolicy,
    primitives::{ensure_same_len, workspace::Workspace},
};
use cubecl::prelude::*;

use super::BLOCK_ORDERING_SIZE;

const LARGE_SORT_RUN_LEN: usize = 78_125;

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
    let client = policy.client();
    if len == 0 {
        return Ok(policy.empty_device_vec());
    }
    if len > LARGE_SORT_RUN_LEN {
        let materialized =
            crate::detail::apply::MaterializePayloadApply::collect_expr(policy, input)?;
        return sort_materialized_in_runs::<Source::Runtime, Source::Item, Less>(
            policy,
            &materialized,
        );
    }
    let launch = crate::detail::launch::launch_1d(client, len, BLOCK_ORDERING_SIZE)?;

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
            launch.cube_count(),
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
                launch.cube_count(),
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

fn sort_materialized_in_runs<R, T, Less>(
    policy: &CubePolicy<R>,
    input: &DeviceVec<R, T>,
) -> Result<DeviceVec<R, T>, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Less: BinaryPredicateOp<T>,
{
    let len = input.len();
    if len <= LARGE_SORT_RUN_LEN {
        let view = DeviceColumnView::from_column(input);
        return sort_input_with_policy(policy, &view, GpuOp::<Less>::new());
    }

    let mut runs = Vec::new();
    let mut start = 0usize;
    while start < len {
        let run_len = (len - start).min(LARGE_SORT_RUN_LEN);
        let view = DeviceColumnView::from_slice(input, start, run_len);
        runs.push(sort_input_with_policy(policy, &view, GpuOp::<Less>::new())?);
        start += run_len;
    }
    sync_client(policy)?;

    merge_sorted_runs::<R, T, Less>(policy, runs)
}

fn merge_sorted_runs<R, T, Less>(
    policy: &CubePolicy<R>,
    mut runs: Vec<DeviceVec<R, T>>,
) -> Result<DeviceVec<R, T>, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Less: BinaryPredicateOp<T>,
{
    while runs.len() > 1 {
        let mut next = Vec::with_capacity(runs.len().div_ceil(2));
        let mut iter = runs.into_iter();
        while let Some(left) = iter.next() {
            let Some(right) = iter.next() else {
                next.push(left);
                break;
            };
            let left = DeviceColumnView::from_column(&left);
            let right = DeviceColumnView::from_column(&right);
            next.push(crate::detail::apply::MergeExprApply::apply_expr::<
                DeviceColumnView<R, T>,
                DeviceColumnView<R, T>,
                Less,
            >(policy, &left, &right)?);
        }
        sync_client(policy)?;
        runs = next;
    }

    Ok(runs.pop().unwrap_or_else(|| policy.empty_device_vec()))
}

fn sync_client<R: Runtime>(policy: &CubePolicy<R>) -> Result<(), Error> {
    futures_lite::future::block_on(policy.client().sync()).map_err(|err| Error::Launch {
        message: err.to_string(),
    })
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
    let launch = crate::detail::launch::launch_1d(client, len, BLOCK_ORDERING_SIZE)?;
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
            launch.cube_count(),
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
                launch.cube_count(),
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
    let launch = crate::detail::launch::launch_1d(client, len, BLOCK_ORDERING_SIZE)?;
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
            launch.cube_count(),
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
                launch.cube_count(),
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

#[allow(dead_code)]
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

    let launch = crate::detail::launch::launch_1d(client, len, BLOCK_ORDERING_SIZE)?;
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
                launch.cube_count(),
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
    let launch = crate::detail::launch::launch_1d(client, len, BLOCK_ORDERING_SIZE)?;
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
            launch.cube_count(),
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
                launch.cube_count(),
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

pub(crate) fn sort_tuple3_by_key_input_with_policy<First, Second, Third, Value, Less>(
    policy: &CubePolicy<First::Runtime>,
    first: &First,
    second: &Second,
    third: &Third,
    values: &Value,
    _less: GpuOp<Less>,
) -> Result<
    (
        DeviceVec<First::Runtime, First::Item>,
        DeviceVec<First::Runtime, Second::Item>,
        DeviceVec<First::Runtime, Third::Item>,
        DeviceVec<First::Runtime, Value::Item>,
    ),
    Error,
>
where
    First: KernelColumn + KernelColumnAt<S0>,
    Second: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
    Third: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
    Value: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
    First::Item: CubePrimitive + CubeElement,
    Second::Item: CubePrimitive + CubeElement,
    Third::Item: CubePrimitive + CubeElement,
    Value::Item: CubePrimitive + CubeElement,
    First::Expr: DeviceGpuExpr<First::Item>,
    Second::Expr: DeviceGpuExpr<Second::Item>,
    Third::Expr: DeviceGpuExpr<Third::Item>,
    Value::Expr: DeviceGpuExpr<Value::Item>,
    Less: BinaryPredicateOp<(First::Item, Second::Item, Third::Item)>,
{
    first.validate()?;
    second.validate()?;
    third.validate()?;
    values.validate()?;
    ensure_same_len(second.len(), first.len())?;
    ensure_same_len(third.len(), first.len())?;
    ensure_same_len(values.len(), first.len())?;

    let len = first.len();
    if len == 0 {
        return Ok((
            policy.empty_device_vec(),
            policy.empty_device_vec(),
            policy.empty_device_vec(),
            policy.empty_device_vec(),
        ));
    }

    let client = policy.client();
    let launch = crate::detail::launch::launch_1d(client, len, BLOCK_ORDERING_SIZE)?;
    let workspace = Workspace::new(policy);
    let (scratch_first_a, scratch_first_b) = workspace.alloc_pair::<First::Item>(len);
    let (scratch_second_a, scratch_second_b) = workspace.alloc_pair::<Second::Item>(len);
    let (scratch_third_a, scratch_third_b) = workspace.alloc_pair::<Third::Item>(len);
    let (scratch_values_a, scratch_values_b) = workspace.alloc_pair::<Value::Item>(len);
    let first_bindings = first.stage(policy)?;
    let second_bindings = second.stage(policy)?;
    let third_bindings = third.stage(policy)?;
    let value_bindings = values.stage(policy)?;
    let first_offsets = first_bindings.slot_offsets_handle(client)?;
    let second_offsets = second_bindings.slot_offsets_handle(client)?;
    let third_offsets = third_bindings.slot_offsets_handle(client)?;
    let value_offsets = value_bindings.slot_offsets_handle(client)?;
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
        merge_sort_tuple3_by_key_expr_first_pass_kernel::launch_unchecked::<
            First::Item,
            Second::Item,
            Third::Item,
            Value::Item,
            First::Expr,
            Second::Expr,
            Third::Expr,
            Value::Expr,
            Less,
            First::Runtime,
        >(
            client,
            launch.cube_count(),
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
            unsafe { BufferArg::from_raw_parts(scratch_first_a.clone(), len) },
            unsafe { BufferArg::from_raw_parts(scratch_second_a.clone(), len) },
            unsafe { BufferArg::from_raw_parts(scratch_third_a.clone(), len) },
            unsafe { BufferArg::from_raw_parts(scratch_values_a.clone(), len) },
        );
    }

    let mut input_first_handle = scratch_first_a.clone();
    let mut input_second_handle = scratch_second_a.clone();
    let mut input_third_handle = scratch_third_a.clone();
    let mut input_values_handle = scratch_values_a.clone();
    let mut output_first_handle = scratch_first_b.clone();
    let mut output_second_handle = scratch_second_b.clone();
    let mut output_third_handle = scratch_third_b.clone();
    let mut output_values_handle = scratch_values_b.clone();
    let mut next_uses_a = true;
    let mut width = 2usize;

    while width < len {
        let width_u32 = u32::try_from(width).map_err(|_| Error::LengthTooLarge { len: width })?;
        let width_handle = client.create_from_slice(u32::as_bytes(&[width_u32]));
        unsafe {
            merge_sort_tuple3_by_key_pass_kernel::launch_unchecked::<
                First::Item,
                Second::Item,
                Third::Item,
                Value::Item,
                Less,
                First::Runtime,
            >(
                client,
                launch.cube_count(),
                CubeDim::new_1d(BLOCK_ORDERING_SIZE),
                unsafe { BufferArg::from_raw_parts(input_first_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(input_second_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(input_third_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(input_values_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(width_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(output_first_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(output_second_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(output_third_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(output_values_handle.clone(), len) },
            );
        }

        input_first_handle = output_first_handle.clone();
        input_second_handle = output_second_handle.clone();
        input_third_handle = output_third_handle.clone();
        input_values_handle = output_values_handle.clone();
        if next_uses_a {
            output_first_handle = scratch_first_a.clone();
            output_second_handle = scratch_second_a.clone();
            output_third_handle = scratch_third_a.clone();
            output_values_handle = scratch_values_a.clone();
        } else {
            output_first_handle = scratch_first_b.clone();
            output_second_handle = scratch_second_b.clone();
            output_third_handle = scratch_third_b.clone();
            output_values_handle = scratch_values_b.clone();
        }
        next_uses_a = !next_uses_a;
        width *= 2;
    }

    Ok((
        DeviceVec::from_handle(policy.id(), input_first_handle, len),
        DeviceVec::from_handle(policy.id(), input_second_handle, len),
        DeviceVec::from_handle(policy.id(), input_third_handle, len),
        DeviceVec::from_handle(policy.id(), input_values_handle, len),
    ))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn sort_tuple7_indices_input<A, B, C, D, E, F, G, Less>(
    policy: &CubePolicy<A::Runtime>,
    a: &A,
    b: &B,
    c: &C,
    d: &D,
    e: &E,
    f: &F,
    g: &G,
    _less: GpuOp<Less>,
) -> Result<DeviceVec<A::Runtime, MIndex>, Error>
where
    A: KernelColumn + KernelColumnAt<S0>,
    B: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    D: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    E: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    F: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    G: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    A::Item: CubePrimitive + CubeElement,
    B::Item: CubePrimitive + CubeElement,
    C::Item: CubePrimitive + CubeElement,
    D::Item: CubePrimitive + CubeElement,
    E::Item: CubePrimitive + CubeElement,
    F::Item: CubePrimitive + CubeElement,
    G::Item: CubePrimitive + CubeElement,
    A::Expr: DeviceGpuExpr<A::Item>,
    B::Expr: DeviceGpuExpr<B::Item>,
    C::Expr: DeviceGpuExpr<C::Item>,
    D::Expr: DeviceGpuExpr<D::Item>,
    E::Expr: DeviceGpuExpr<E::Item>,
    F::Expr: DeviceGpuExpr<F::Item>,
    G::Expr: DeviceGpuExpr<G::Item>,
    Less: BinaryPredicateOp<(
        A::Item,
        B::Item,
        C::Item,
        D::Item,
        E::Item,
        F::Item,
        G::Item,
    )>,
{
    a.validate()?;
    b.validate()?;
    c.validate()?;
    d.validate()?;
    e.validate()?;
    f.validate()?;
    g.validate()?;
    ensure_same_len(b.len(), a.len())?;
    ensure_same_len(c.len(), a.len())?;
    ensure_same_len(d.len(), a.len())?;
    ensure_same_len(e.len(), a.len())?;
    ensure_same_len(f.len(), a.len())?;
    ensure_same_len(g.len(), a.len())?;

    let len = a.len();
    if len == 0 {
        return Ok(policy.empty_device_vec());
    }

    let client = policy.client();
    let launch = crate::detail::launch::launch_1d(client, len, BLOCK_ORDERING_SIZE)?;
    let workspace = Workspace::new(policy);
    let (scratch_a, scratch_b) = workspace.alloc_pair::<u32>(len);
    let a_bindings = a.stage(policy)?;
    let b_bindings = b.stage(policy)?;
    let c_bindings = c.stage(policy)?;
    let d_bindings = d.stage(policy)?;
    let e_bindings = e.stage(policy)?;
    let f_bindings = f.stage(policy)?;
    let g_bindings = g.stage(policy)?;
    let a_offsets = a_bindings.slot_offsets_handle(client)?;
    let b_offsets = b_bindings.slot_offsets_handle(client)?;
    let c_offsets = c_bindings.slot_offsets_handle(client)?;
    let d_offsets = d_bindings.slot_offsets_handle(client)?;
    let e_offsets = e_bindings.slot_offsets_handle(client)?;
    let f_offsets = f_bindings.slot_offsets_handle(client)?;
    let g_offsets = g_bindings.slot_offsets_handle(client)?;
    let a0 = a_bindings.slot_or_first(0);
    let a1 = a_bindings.slot_or_first(1);
    let a2 = a_bindings.slot_or_first(2);
    let a3 = a_bindings.slot_or_first(3);
    let b0 = b_bindings.slot_or_first(0);
    let b1 = b_bindings.slot_or_first(1);
    let b2 = b_bindings.slot_or_first(2);
    let b3 = b_bindings.slot_or_first(3);
    let c0 = c_bindings.slot_or_first(0);
    let c1 = c_bindings.slot_or_first(1);
    let c2 = c_bindings.slot_or_first(2);
    let c3 = c_bindings.slot_or_first(3);
    let d0 = d_bindings.slot_or_first(0);
    let d1 = d_bindings.slot_or_first(1);
    let d2 = d_bindings.slot_or_first(2);
    let d3 = d_bindings.slot_or_first(3);
    let e0 = e_bindings.slot_or_first(0);
    let e1 = e_bindings.slot_or_first(1);
    let e2 = e_bindings.slot_or_first(2);
    let e3 = e_bindings.slot_or_first(3);
    let f0 = f_bindings.slot_or_first(0);
    let f1 = f_bindings.slot_or_first(1);
    let f2 = f_bindings.slot_or_first(2);
    let f3 = f_bindings.slot_or_first(3);
    let g0 = g_bindings.slot_or_first(0);
    let g1 = g_bindings.slot_or_first(1);
    let g2 = g_bindings.slot_or_first(2);
    let g3 = g_bindings.slot_or_first(3);
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));

    unsafe {
        merge_sort_tuple7_indices_expr_first_pass_kernel::launch_unchecked::<
            A::Item,
            B::Item,
            C::Item,
            D::Item,
            E::Item,
            F::Item,
            G::Item,
            A::Expr,
            B::Expr,
            C::Expr,
            D::Expr,
            E::Expr,
            F::Expr,
            G::Expr,
            Less,
            A::Runtime,
        >(
            client,
            launch.cube_count(),
            CubeDim::new_1d(BLOCK_ORDERING_SIZE),
            unsafe { BufferArg::from_raw_parts(a0.0.clone(), a0.1) },
            unsafe { BufferArg::from_raw_parts(a1.0.clone(), a1.1) },
            unsafe { BufferArg::from_raw_parts(a2.0.clone(), a2.1) },
            unsafe { BufferArg::from_raw_parts(a3.0.clone(), a3.1) },
            unsafe { BufferArg::from_raw_parts(a_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(b0.0.clone(), b0.1) },
            unsafe { BufferArg::from_raw_parts(b1.0.clone(), b1.1) },
            unsafe { BufferArg::from_raw_parts(b2.0.clone(), b2.1) },
            unsafe { BufferArg::from_raw_parts(b3.0.clone(), b3.1) },
            unsafe { BufferArg::from_raw_parts(b_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(c0.0.clone(), c0.1) },
            unsafe { BufferArg::from_raw_parts(c1.0.clone(), c1.1) },
            unsafe { BufferArg::from_raw_parts(c2.0.clone(), c2.1) },
            unsafe { BufferArg::from_raw_parts(c3.0.clone(), c3.1) },
            unsafe { BufferArg::from_raw_parts(c_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(d0.0.clone(), d0.1) },
            unsafe { BufferArg::from_raw_parts(d1.0.clone(), d1.1) },
            unsafe { BufferArg::from_raw_parts(d2.0.clone(), d2.1) },
            unsafe { BufferArg::from_raw_parts(d3.0.clone(), d3.1) },
            unsafe { BufferArg::from_raw_parts(d_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(e0.0.clone(), e0.1) },
            unsafe { BufferArg::from_raw_parts(e1.0.clone(), e1.1) },
            unsafe { BufferArg::from_raw_parts(e2.0.clone(), e2.1) },
            unsafe { BufferArg::from_raw_parts(e3.0.clone(), e3.1) },
            unsafe { BufferArg::from_raw_parts(e_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(f0.0.clone(), f0.1) },
            unsafe { BufferArg::from_raw_parts(f1.0.clone(), f1.1) },
            unsafe { BufferArg::from_raw_parts(f2.0.clone(), f2.1) },
            unsafe { BufferArg::from_raw_parts(f3.0.clone(), f3.1) },
            unsafe { BufferArg::from_raw_parts(f_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(g0.0.clone(), g0.1) },
            unsafe { BufferArg::from_raw_parts(g1.0.clone(), g1.1) },
            unsafe { BufferArg::from_raw_parts(g2.0.clone(), g2.1) },
            unsafe { BufferArg::from_raw_parts(g3.0.clone(), g3.1) },
            unsafe { BufferArg::from_raw_parts(g_offsets.clone(), 4) },
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
            merge_sort_tuple7_indices_pass_kernel::launch_unchecked::<
                A::Item,
                B::Item,
                C::Item,
                D::Item,
                E::Item,
                F::Item,
                G::Item,
                A::Expr,
                B::Expr,
                C::Expr,
                D::Expr,
                E::Expr,
                F::Expr,
                G::Expr,
                Less,
                A::Runtime,
            >(
                client,
                launch.cube_count(),
                CubeDim::new_1d(BLOCK_ORDERING_SIZE),
                unsafe { BufferArg::from_raw_parts(a0.0.clone(), a0.1) },
                unsafe { BufferArg::from_raw_parts(a1.0.clone(), a1.1) },
                unsafe { BufferArg::from_raw_parts(a2.0.clone(), a2.1) },
                unsafe { BufferArg::from_raw_parts(a3.0.clone(), a3.1) },
                unsafe { BufferArg::from_raw_parts(a_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(b0.0.clone(), b0.1) },
                unsafe { BufferArg::from_raw_parts(b1.0.clone(), b1.1) },
                unsafe { BufferArg::from_raw_parts(b2.0.clone(), b2.1) },
                unsafe { BufferArg::from_raw_parts(b3.0.clone(), b3.1) },
                unsafe { BufferArg::from_raw_parts(b_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(c0.0.clone(), c0.1) },
                unsafe { BufferArg::from_raw_parts(c1.0.clone(), c1.1) },
                unsafe { BufferArg::from_raw_parts(c2.0.clone(), c2.1) },
                unsafe { BufferArg::from_raw_parts(c3.0.clone(), c3.1) },
                unsafe { BufferArg::from_raw_parts(c_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(d0.0.clone(), d0.1) },
                unsafe { BufferArg::from_raw_parts(d1.0.clone(), d1.1) },
                unsafe { BufferArg::from_raw_parts(d2.0.clone(), d2.1) },
                unsafe { BufferArg::from_raw_parts(d3.0.clone(), d3.1) },
                unsafe { BufferArg::from_raw_parts(d_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(e0.0.clone(), e0.1) },
                unsafe { BufferArg::from_raw_parts(e1.0.clone(), e1.1) },
                unsafe { BufferArg::from_raw_parts(e2.0.clone(), e2.1) },
                unsafe { BufferArg::from_raw_parts(e3.0.clone(), e3.1) },
                unsafe { BufferArg::from_raw_parts(e_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(f0.0.clone(), f0.1) },
                unsafe { BufferArg::from_raw_parts(f1.0.clone(), f1.1) },
                unsafe { BufferArg::from_raw_parts(f2.0.clone(), f2.1) },
                unsafe { BufferArg::from_raw_parts(f3.0.clone(), f3.1) },
                unsafe { BufferArg::from_raw_parts(f_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(g0.0.clone(), g0.1) },
                unsafe { BufferArg::from_raw_parts(g1.0.clone(), g1.1) },
                unsafe { BufferArg::from_raw_parts(g2.0.clone(), g2.1) },
                unsafe { BufferArg::from_raw_parts(g3.0.clone(), g3.1) },
                unsafe { BufferArg::from_raw_parts(g_offsets.clone(), 4) },
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
