use super::*;

pub fn replace_where_into_with_policy<R, T, Stencil, Pred>(
    policy: &crate::policy::CubePolicy<R>,
    replacement: T,
    stencil: &Stencil,
    output: &DeviceColumnMutView<R, T>,
    _pred: Pred,
) -> Result<(), Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Stencil: SelectionStencil<Pred, Runtime = R>,
{
    ensure_same_len(stencil.len(), output.len)?;
    let flags = stencil.selection_handles_with_policy(policy, false)?;
    let len = stencil.len();
    if len == 0 {
        return Ok(());
    }

    let client = policy.client();
    let replacement_values = [replacement];
    let replacement_handle = client.create_from_slice(T::as_bytes(&replacement_values));
    let output_offset = offset_handle(client, output.offset)?;
    let block_count_u32 = api_expr_block_count(len)?;

    unsafe {
        replace_with_flags_into_kernel::launch_unchecked::<T, R>(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_API_EXPR_SIZE),
            unsafe { BufferArg::from_raw_parts(replacement_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(flags.flag.clone(), flags.len) },
            unsafe { BufferArg::from_raw_parts(output_offset.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output.source.handle.clone(), output.source.len()) },
        );
    }
    Ok(())
}

pub fn device_expr_copy_where_into_with_policy<ExprSource, Stencil, Pred>(
    policy: &crate::policy::CubePolicy<ExprSource::Runtime>,
    expr: &ExprSource,
    stencil: &Stencil,
    output: &DeviceColumnMutView<ExprSource::Runtime, ExprSource::Item>,
    _pred: Pred,
) -> Result<(), Error>
where
    ExprSource: KernelColumn + KernelColumnAt<S0>,
    ExprSource::Runtime: Runtime,
    ExprSource::Item: CubePrimitive + CubeElement,
    ExprSource::Expr: DeviceGpuExpr<ExprSource::Item>,
    Stencil: SelectionStencil<Pred, Runtime = ExprSource::Runtime>,
{
    expr.validate()?;
    ensure_same_len(expr.len(), stencil.len())?;
    ensure_same_len(expr.len(), output.len)?;
    let flags = stencil.selection_handles_with_policy(policy, false)?;
    let len = expr.len();
    if len == 0 {
        return Ok(());
    }

    let client = policy.client();
    let bindings = expr.stage(policy)?;
    let slot0 = bindings.slot_or_first(0);
    let slot1 = bindings.slot_or_first(1);
    let slot2 = bindings.slot_or_first(2);
    let slot3 = bindings.slot_or_first(3);
    let slot_offsets = bindings.slot_offsets_handle(client)?;
    let output_offset = offset_handle(client, output.offset)?;
    let block_count_u32 = api_expr_block_count(len)?;

    unsafe {
        copy_if_flags_into_kernel::launch_unchecked::<
            ExprSource::Item,
            ExprSource::Expr,
            ExprSource::Runtime,
        >(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_API_EXPR_SIZE),
            unsafe { BufferArg::from_raw_parts(slot0.0.clone(), slot0.1) },
            unsafe { BufferArg::from_raw_parts(slot1.0.clone(), slot1.1) },
            unsafe { BufferArg::from_raw_parts(slot2.0.clone(), slot2.1) },
            unsafe { BufferArg::from_raw_parts(slot3.0.clone(), slot3.1) },
            unsafe { BufferArg::from_raw_parts(slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(flags.flag.clone(), flags.len) },
            unsafe { BufferArg::from_raw_parts(output_offset.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output.source.handle.clone(), output.source.len()) },
        );
    }
    Ok(())
}

pub(in crate::detail) fn device_expr_copy_where_with_policy<ExprSource, Pred>(
    policy: &crate::policy::CubePolicy<ExprSource::Runtime>,
    expr: &ExprSource,
    invert: bool,
) -> Result<DeviceVec<ExprSource::Runtime, ExprSource::Item>, Error>
where
    ExprSource: KernelColumn + KernelColumnAt<S0>,
    ExprSource::Runtime: Runtime,
    ExprSource::Item: CubePrimitive + CubeElement,
    ExprSource::Expr: DeviceGpuExpr<ExprSource::Item> + GpuExpr<ExprSource::Item>,
    Pred: PredicateOp<ExprSource::Item>,
{
    let handles =
        device_expr_selection_handles_with_policy::<ExprSource, Pred>(policy, expr, invert)?;
    let count = select::selected_count(policy, &handles)?;
    device_expr_compact_with_selection_with_policy(policy, expr, &handles, count)
}

pub(in crate::detail) fn device_expr_compact_with_flags_with_policy<ExprSource>(
    policy: &crate::policy::CubePolicy<ExprSource::Runtime>,
    expr: &ExprSource,
    flag_handle: cubecl::server::Handle,
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

    let handles =
        select::handles_from_flags(policy, len, len_u32, flag_handle, policy.empty_handle())?;
    let count = select::selected_count(policy, &handles)?;
    if count == 0 {
        return Ok(policy.empty_device_vec());
    }

    let client = policy.client();
    let output_handle = client.empty(count * std::mem::size_of::<ExprSource::Item>());
    let bindings = expr.stage(policy)?;
    let slot_offsets = bindings.slot_offsets_handle(client)?;
    let slot0 = bindings.slots.first().unwrap();
    let slot1 = bindings.slots.get(1).unwrap_or(slot0);
    let slot2 = bindings.slots.get(2).unwrap_or(slot0);
    let slot3 = bindings.slots.get(3).unwrap_or(slot0);
    let block_count_u32 = api_expr_block_count(len)?;

    unsafe {
        compact_scatter_device_expr_kernel::launch_unchecked::<
            ExprSource::Item,
            ExprSource::Expr,
            ExprSource::Runtime,
        >(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_API_EXPR_SIZE),
            unsafe { BufferArg::from_raw_parts(handles.flag.clone(), handles.len) },
            unsafe { BufferArg::from_raw_parts(handles.position.clone(), handles.len) },
            unsafe { BufferArg::from_raw_parts(slot0.0.clone(), slot0.1) },
            unsafe { BufferArg::from_raw_parts(slot1.0.clone(), slot1.1) },
            unsafe { BufferArg::from_raw_parts(slot2.0.clone(), slot2.1) },
            unsafe { BufferArg::from_raw_parts(slot3.0.clone(), slot3.1) },
            unsafe { BufferArg::from_raw_parts(slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(output_handle.clone(), count) },
        );
    }

    Ok(DeviceVec::from_handle(policy.id(), output_handle, count))
}

pub(in crate::detail) fn device_expr_compact_with_selection_with_policy<ExprSource>(
    policy: &crate::policy::CubePolicy<ExprSource::Runtime>,
    expr: &ExprSource,
    handles: &select::SelectionControl,
    count: usize,
) -> Result<DeviceVec<ExprSource::Runtime, ExprSource::Item>, Error>
where
    ExprSource: KernelColumn + KernelColumnAt<S0>,
    ExprSource::Runtime: Runtime,
    ExprSource::Item: CubePrimitive + CubeElement,
    ExprSource::Expr: DeviceGpuExpr<ExprSource::Item>,
{
    expr.validate()?;
    ensure_same_len(expr.len(), handles.len)?;
    if handles.len == 0 || count == 0 {
        return Ok(policy.empty_device_vec());
    }

    let client = policy.client();
    let output_handle = client.empty(count * std::mem::size_of::<ExprSource::Item>());
    let bindings = expr.stage(policy)?;
    let slot_offsets = bindings.slot_offsets_handle(client)?;
    let slot0 = bindings.slots.first().unwrap();
    let slot1 = bindings.slots.get(1).unwrap_or(slot0);
    let slot2 = bindings.slots.get(2).unwrap_or(slot0);
    let slot3 = bindings.slots.get(3).unwrap_or(slot0);
    let block_count_u32 = api_expr_block_count(handles.len)?;

    unsafe {
        compact_scatter_device_expr_kernel::launch_unchecked::<
            ExprSource::Item,
            ExprSource::Expr,
            ExprSource::Runtime,
        >(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_API_EXPR_SIZE),
            unsafe { BufferArg::from_raw_parts(handles.flag.clone(), handles.len) },
            unsafe { BufferArg::from_raw_parts(handles.position.clone(), handles.len) },
            unsafe { BufferArg::from_raw_parts(slot0.0.clone(), slot0.1) },
            unsafe { BufferArg::from_raw_parts(slot1.0.clone(), slot1.1) },
            unsafe { BufferArg::from_raw_parts(slot2.0.clone(), slot2.1) },
            unsafe { BufferArg::from_raw_parts(slot3.0.clone(), slot3.1) },
            unsafe { BufferArg::from_raw_parts(slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(output_handle.clone(), count) },
        );
    }

    Ok(DeviceVec::from_handle(policy.id(), output_handle, count))
}

pub(in crate::detail) fn device_expr_compact_rejected_with_selection_with_policy<ExprSource>(
    policy: &crate::policy::CubePolicy<ExprSource::Runtime>,
    expr: &ExprSource,
    handles: &select::SelectionControl,
    count: usize,
) -> Result<DeviceVec<ExprSource::Runtime, ExprSource::Item>, Error>
where
    ExprSource: KernelColumn + KernelColumnAt<S0>,
    ExprSource::Runtime: Runtime,
    ExprSource::Item: CubePrimitive + CubeElement,
    ExprSource::Expr: DeviceGpuExpr<ExprSource::Item>,
{
    expr.validate()?;
    ensure_same_len(expr.len(), handles.len)?;
    if handles.len == 0 || count == 0 {
        return Ok(policy.empty_device_vec());
    }

    let client = policy.client();
    let output_handle = client.empty(count * std::mem::size_of::<ExprSource::Item>());
    let bindings = expr.stage(policy)?;
    let slot_offsets = bindings.slot_offsets_handle(client)?;
    let slot0 = bindings.slots.first().unwrap();
    let slot1 = bindings.slots.get(1).unwrap_or(slot0);
    let slot2 = bindings.slots.get(2).unwrap_or(slot0);
    let slot3 = bindings.slots.get(3).unwrap_or(slot0);
    let block_count_u32 = api_expr_block_count(handles.len)?;

    unsafe {
        compact_rejected_scatter_device_expr_kernel::launch_unchecked::<
            ExprSource::Item,
            ExprSource::Expr,
            ExprSource::Runtime,
        >(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_API_EXPR_SIZE),
            unsafe { BufferArg::from_raw_parts(handles.flag.clone(), handles.len) },
            unsafe { BufferArg::from_raw_parts(handles.position.clone(), handles.len) },
            unsafe { BufferArg::from_raw_parts(slot0.0.clone(), slot0.1) },
            unsafe { BufferArg::from_raw_parts(slot1.0.clone(), slot1.1) },
            unsafe { BufferArg::from_raw_parts(slot2.0.clone(), slot2.1) },
            unsafe { BufferArg::from_raw_parts(slot3.0.clone(), slot3.1) },
            unsafe { BufferArg::from_raw_parts(slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(output_handle.clone(), count) },
        );
    }

    Ok(DeviceVec::from_handle(policy.id(), output_handle, count))
}

pub(in crate::detail) fn device_expr_count_if_with_policy<ExprSource, Pred>(
    policy: &crate::policy::CubePolicy<ExprSource::Runtime>,
    expr: &ExprSource,
    invert: bool,
) -> Result<usize, Error>
where
    ExprSource: KernelColumn + KernelColumnAt<S0>,
    ExprSource::Runtime: Runtime,
    ExprSource::Item: CubePrimitive + CubeElement,
    ExprSource::Expr: DeviceGpuExpr<ExprSource::Item>,
    Pred: PredicateOp<ExprSource::Item>,
{
    let handles =
        device_expr_selection_handles_with_policy::<ExprSource, Pred>(policy, expr, invert)?;
    if handles.len == 0 {
        return Ok(0);
    }

    Ok(read_u32_scalar::<ExprSource::Runtime>(policy.client(), handles.count)? as usize)
}

pub(in crate::detail) fn device_expr_find_if_with_policy<ExprSource, Pred>(
    policy: &crate::policy::CubePolicy<ExprSource::Runtime>,
    expr: &ExprSource,
    invert: bool,
) -> Result<Option<usize>, Error>
where
    ExprSource: KernelColumn + KernelColumnAt<S0>,
    ExprSource::Runtime: Runtime,
    ExprSource::Item: CubePrimitive + CubeElement,
    ExprSource::Expr: GpuExpr<ExprSource::Item>,
    Pred: PredicateOp<ExprSource::Item>,
{
    let handles =
        device_expr_selection_handles_with_policy::<ExprSource, Pred>(policy, expr, invert)?;
    primitive_search::first_flag(policy, handles.flag, handles.len, expr.len())
}

pub(in crate::detail) fn device_expr_selection_handles_with_policy<ExprSource, Pred>(
    policy: &crate::policy::CubePolicy<ExprSource::Runtime>,
    expr: &ExprSource,
    invert: bool,
) -> Result<select::SelectionHandles, Error>
where
    ExprSource: KernelColumn + KernelColumnAt<S0>,
    ExprSource::Runtime: Runtime,
    ExprSource::Item: CubePrimitive + CubeElement,
    ExprSource::Expr: GpuExpr<ExprSource::Item>,
    Pred: PredicateOp<ExprSource::Item>,
{
    expr.validate()?;
    let len = expr.len();
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let client = policy.client();
    if len == 0 {
        return select::handles_from_flags(
            policy,
            0,
            0,
            policy.empty_handle(),
            policy.empty_handle(),
        );
    }

    let len_values = [len_u32];
    let invert_values = [if invert { 1_u32 } else { 0_u32 }];
    let len_handle = client.create_from_slice(u32::as_bytes(&len_values));
    let invert_handle = client.create_from_slice(u32::as_bytes(&invert_values));
    let flag_handle = client.empty(len * std::mem::size_of::<u32>());
    let bindings = expr.stage(policy)?;
    let slot0 = bindings.slot_or_first(0);
    let slot1 = bindings.slot_or_first(1);
    let slot2 = bindings.slot_or_first(2);
    let slot3 = bindings.slot_or_first(3);
    let slot_offsets = bindings.slot_offsets_handle(client)?;
    let block_count_u32 = api_expr_block_count(len)?;

    unsafe {
        copy_if_device_expr_flag_only_kernel::launch_unchecked::<
            ExprSource::Item,
            ExprSource::Expr,
            Pred,
            ExprSource::Runtime,
        >(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_API_EXPR_SIZE),
            unsafe { BufferArg::from_raw_parts(slot0.0.clone(), slot0.1) },
            unsafe { BufferArg::from_raw_parts(slot1.0.clone(), slot1.1) },
            unsafe { BufferArg::from_raw_parts(slot2.0.clone(), slot2.1) },
            unsafe { BufferArg::from_raw_parts(slot3.0.clone(), slot3.1) },
            unsafe { BufferArg::from_raw_parts(slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(invert_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(flag_handle.clone(), len) },
        );
    }

    select::handles_from_flags(policy, len, len_u32, flag_handle, policy.empty_handle())
}
