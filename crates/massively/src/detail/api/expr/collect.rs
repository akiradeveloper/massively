use super::*;

pub(in crate::detail) fn device_expr_collect_with_policy<ExprSource>(
    policy: &crate::policy::CubePolicy<ExprSource::Runtime>,
    expr: &ExprSource,
) -> Result<DeviceVec<ExprSource::Runtime, ExprSource::Item>, Error>
where
    ExprSource: KernelColumn + KernelColumnAt<S0>,
    ExprSource::Runtime: Runtime,
    ExprSource::Item: CubePrimitive + CubeElement,
    ExprSource::Expr: DeviceGpuExpr<ExprSource::Item>,
{
    expr.validate()?;
    let len = expr.len();
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    if len == 0 {
        return Ok(policy.empty_device_vec());
    }

    let client = policy.client();
    let output_handle = client.empty(len * std::mem::size_of::<ExprSource::Item>());

    let bindings = expr.stage(policy)?;
    let slot0 = bindings.slots.first().unwrap();
    let slot1 = bindings.slots.get(1).unwrap_or(slot0);
    let slot2 = bindings.slots.get(2).unwrap_or(slot0);
    let slot3 = bindings.slots.get(3).unwrap_or(slot0);
    let launch = crate::detail::launch::launch_1d(client, len, BLOCK_API_EXPR_SIZE)?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let slot_offsets = bindings.slot_offsets_handle(client)?;

    unsafe {
        device_collect_expr_block_kernel::launch_unchecked::<
            ExprSource::Item,
            ExprSource::Expr,
            ExprSource::Runtime,
        >(
            client,
            launch.cube_count(),
            CubeDim::new_1d(BLOCK_API_EXPR_SIZE),
            unsafe { BufferArg::from_raw_parts(output_handle.clone(), len) },
            unsafe { BufferArg::from_raw_parts(slot0.0.clone(), slot0.1) },
            unsafe { BufferArg::from_raw_parts(slot1.0.clone(), slot1.1) },
            unsafe { BufferArg::from_raw_parts(slot2.0.clone(), slot2.1) },
            unsafe { BufferArg::from_raw_parts(slot3.0.clone(), slot3.1) },
            unsafe { BufferArg::from_raw_parts(slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
        );
    }

    Ok(DeviceVec::from_handle(policy.id(), output_handle, len))
}

pub fn device_expr_collect_into_with_policy<ExprSource>(
    policy: &crate::policy::CubePolicy<ExprSource::Runtime>,
    expr: &ExprSource,
    output: &DeviceColumnMutView<ExprSource::Runtime, ExprSource::Item>,
) -> Result<(), Error>
where
    ExprSource: KernelColumn + KernelColumnAt<S0>,
    ExprSource::Runtime: Runtime,
    ExprSource::Item: CubePrimitive + CubeElement,
    ExprSource::Expr: DeviceGpuExpr<ExprSource::Item>,
{
    expr.validate()?;
    let len = expr.len();
    ensure_same_len(len, output.len)?;
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    if len == 0 {
        return Ok(());
    }

    let client = policy.client();
    let bindings = expr.stage(policy)?;
    let slot0 = bindings.slots.first().unwrap();
    let slot1 = bindings.slots.get(1).unwrap_or(slot0);
    let slot2 = bindings.slots.get(2).unwrap_or(slot0);
    let slot3 = bindings.slots.get(3).unwrap_or(slot0);
    let launch = crate::detail::launch::launch_1d(client, len, BLOCK_API_EXPR_SIZE)?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let slot_offsets = bindings.slot_offsets_handle(client)?;
    let output_offset = offset_handle(client, output.offset)?;

    unsafe {
        device_collect_expr_into_block_kernel::launch_unchecked::<
            ExprSource::Item,
            ExprSource::Expr,
            ExprSource::Runtime,
        >(
            client,
            launch.cube_count(),
            CubeDim::new_1d(BLOCK_API_EXPR_SIZE),
            unsafe { BufferArg::from_raw_parts(slot0.0.clone(), slot0.1) },
            unsafe { BufferArg::from_raw_parts(slot1.0.clone(), slot1.1) },
            unsafe { BufferArg::from_raw_parts(slot2.0.clone(), slot2.1) },
            unsafe { BufferArg::from_raw_parts(slot3.0.clone(), slot3.1) },
            unsafe { BufferArg::from_raw_parts(slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output_offset.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output.source.handle.clone(), output.source.len()) },
        );
    }
    Ok(())
}

pub(in crate::detail) fn device_expr_reverse_collect<ExprSource>(
    policy: &crate::policy::CubePolicy<ExprSource::Runtime>,
    expr: &ExprSource,
) -> Result<DeviceVec<ExprSource::Runtime, ExprSource::Item>, Error>
where
    ExprSource: KernelColumn + KernelColumnAt<S0>,
    ExprSource::Runtime: Runtime,
    ExprSource::Item: CubePrimitive + CubeElement,
    ExprSource::Expr: DeviceGpuExpr<ExprSource::Item>,
{
    expr.validate()?;
    let len = expr.len();
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    if len == 0 {
        return Ok(policy.empty_device_vec());
    }

    let client = policy.client();
    let output_handle = client.empty(len * std::mem::size_of::<ExprSource::Item>());

    let bindings = expr.stage(policy)?;
    let slot0 = bindings.slots.first().unwrap();
    let slot1 = bindings.slots.get(1).unwrap_or(slot0);
    let slot2 = bindings.slots.get(2).unwrap_or(slot0);
    let slot3 = bindings.slots.get(3).unwrap_or(slot0);
    let launch = crate::detail::launch::launch_1d(client, len, BLOCK_API_EXPR_SIZE)?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let slot_offsets = bindings.slot_offsets_handle(client)?;

    unsafe {
        device_collect_expr_reverse_block_kernel::launch_unchecked::<
            ExprSource::Item,
            ExprSource::Expr,
            ExprSource::Runtime,
        >(
            client,
            launch.cube_count(),
            CubeDim::new_1d(BLOCK_API_EXPR_SIZE),
            unsafe { BufferArg::from_raw_parts(output_handle.clone(), len) },
            unsafe { BufferArg::from_raw_parts(slot0.0.clone(), slot0.1) },
            unsafe { BufferArg::from_raw_parts(slot1.0.clone(), slot1.1) },
            unsafe { BufferArg::from_raw_parts(slot2.0.clone(), slot2.1) },
            unsafe { BufferArg::from_raw_parts(slot3.0.clone(), slot3.1) },
            unsafe { BufferArg::from_raw_parts(slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
        );
    }

    Ok(DeviceVec::from_handle(policy.id(), output_handle, len))
}
