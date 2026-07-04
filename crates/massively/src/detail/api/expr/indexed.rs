use super::*;

pub(in crate::detail) fn device_expr_gather_with_policy<InputSource, IndexSource>(
    policy: &crate::policy::CubePolicy<InputSource::Runtime>,
    input: &InputSource,
    indices: &IndexSource,
) -> Result<DeviceVec<InputSource::Runtime, InputSource::Item>, Error>
where
    InputSource: KernelColumn + KernelColumnAt<S0>,
    InputSource::Runtime: Runtime,
    IndexSource: KernelColumn<Runtime = InputSource::Runtime, Item = u32> + KernelColumnAt<S0>,
    InputSource::Item: CubePrimitive + CubeElement,
    InputSource::Expr: GpuExpr<InputSource::Item>,
    IndexSource::Expr: GpuExpr<u32>,
{
    input.validate()?;
    indices.validate()?;
    let len = indices.len();
    u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    if len == 0 {
        return Ok(policy.empty_device_vec());
    }

    let client = policy.client();
    let output_handle = client.empty(len * std::mem::size_of::<InputSource::Item>());

    let block_count_u32 = api_expr_block_count(len)?;
    let input_bindings = input.stage(policy)?;
    let index_bindings = indices.stage(policy)?;
    let input_offset = offset_handle(client, input_bindings.input_offset)?;
    let input_rhs_offset = offset_handle(client, input_bindings.rhs_offset)?;
    let index_offset = offset_handle(client, index_bindings.input_offset)?;
    let index_rhs_offset = offset_handle(client, index_bindings.rhs_offset)?;

    unsafe {
        gather_device_expr_kernel::launch_unchecked::<
            InputSource::Item,
            InputSource::Expr,
            IndexSource::Expr,
            InputSource::Runtime,
        >(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_API_EXPR_SIZE),
            unsafe { BufferArg::from_raw_parts(output_handle.clone(), len) },
            unsafe {
                BufferArg::from_raw_parts(input_bindings.input.clone(), input_bindings.input_len)
            },
            unsafe { BufferArg::from_raw_parts(input_offset.clone(), 1) },
            unsafe {
                BufferArg::from_raw_parts(input_bindings.rhs.clone(), input_bindings.rhs_len)
            },
            unsafe { BufferArg::from_raw_parts(input_rhs_offset.clone(), 1) },
            unsafe {
                BufferArg::from_raw_parts(index_bindings.input.clone(), index_bindings.input_len)
            },
            unsafe { BufferArg::from_raw_parts(index_offset.clone(), 1) },
            unsafe {
                BufferArg::from_raw_parts(index_bindings.rhs.clone(), index_bindings.rhs_len)
            },
            unsafe { BufferArg::from_raw_parts(index_rhs_offset.clone(), 1) },
        );
    }

    Ok(DeviceVec::from_handle(policy.id(), output_handle, len))
}

pub fn device_expr_gather_into_with_policy<InputSource, IndexSource>(
    policy: &crate::policy::CubePolicy<InputSource::Runtime>,
    input: &InputSource,
    indices: &IndexSource,
    output: &DeviceColumnMutView<InputSource::Runtime, InputSource::Item>,
) -> Result<(), Error>
where
    InputSource: KernelColumn + KernelColumnAt<S0>,
    InputSource::Runtime: Runtime,
    IndexSource: KernelColumn<Runtime = InputSource::Runtime, Item = u32> + KernelColumnAt<S0>,
    InputSource::Item: CubePrimitive + CubeElement,
    InputSource::Expr: GpuExpr<InputSource::Item>,
    IndexSource::Expr: GpuExpr<u32>,
{
    input.validate()?;
    indices.validate()?;
    ensure_same_len(indices.len(), output.len)?;
    let len = indices.len();
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    if len == 0 {
        return Ok(());
    }

    let client = policy.client();
    let block_count_u32 = api_expr_block_count(len)?;
    let input_bindings = input.stage(policy)?;
    let index_bindings = indices.stage(policy)?;
    let input_offset = offset_handle(client, input_bindings.input_offset)?;
    let input_rhs_offset = offset_handle(client, input_bindings.rhs_offset)?;
    let index_offset = offset_handle(client, index_bindings.input_offset)?;
    let index_rhs_offset = offset_handle(client, index_bindings.rhs_offset)?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let output_offset = offset_handle(client, output.offset)?;

    unsafe {
        gather_device_expr_into_kernel::launch_unchecked::<
            InputSource::Item,
            InputSource::Expr,
            IndexSource::Expr,
            InputSource::Runtime,
        >(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_API_EXPR_SIZE),
            unsafe {
                BufferArg::from_raw_parts(input_bindings.input.clone(), input_bindings.input_len)
            },
            unsafe { BufferArg::from_raw_parts(input_offset.clone(), 1) },
            unsafe {
                BufferArg::from_raw_parts(input_bindings.rhs.clone(), input_bindings.rhs_len)
            },
            unsafe { BufferArg::from_raw_parts(input_rhs_offset.clone(), 1) },
            unsafe {
                BufferArg::from_raw_parts(index_bindings.input.clone(), index_bindings.input_len)
            },
            unsafe { BufferArg::from_raw_parts(index_offset.clone(), 1) },
            unsafe {
                BufferArg::from_raw_parts(index_bindings.rhs.clone(), index_bindings.rhs_len)
            },
            unsafe { BufferArg::from_raw_parts(index_rhs_offset.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output_offset.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output.source.handle.clone(), output.source.len()) },
        );
    }
    Ok(())
}

pub fn device_expr_scatter_into_with_policy<ValueSource, IndexSource>(
    policy: &crate::policy::CubePolicy<ValueSource::Runtime>,
    values: &ValueSource,
    indices: &IndexSource,
    output: &DeviceColumnMutView<ValueSource::Runtime, ValueSource::Item>,
) -> Result<(), Error>
where
    ValueSource: KernelColumn + KernelColumnAt<S0>,
    ValueSource::Runtime: Runtime,
    IndexSource: KernelColumn<Runtime = ValueSource::Runtime, Item = u32> + KernelColumnAt<S0>,
    ValueSource::Item: CubePrimitive + CubeElement,
    ValueSource::Expr: GpuExpr<ValueSource::Item>,
    IndexSource::Expr: GpuExpr<u32>,
{
    values.validate()?;
    indices.validate()?;
    ensure_same_len(values.len(), indices.len())?;
    let len = values.len();
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    if len == 0 {
        return Ok(());
    }

    let client = policy.client();
    let block_count_u32 = api_expr_block_count(len)?;
    let value_bindings = values.stage(policy)?;
    let index_bindings = indices.stage(policy)?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let value_offset = offset_handle(client, value_bindings.input_offset)?;
    let value_rhs_offset = offset_handle(client, value_bindings.rhs_offset)?;
    let index_offset = offset_handle(client, index_bindings.input_offset)?;
    let index_rhs_offset = offset_handle(client, index_bindings.rhs_offset)?;
    let output_offset = offset_handle(client, output.offset)?;

    unsafe {
        scatter_expr_into_kernel::launch_unchecked::<
            ValueSource::Item,
            ValueSource::Expr,
            IndexSource::Expr,
            ValueSource::Runtime,
        >(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_API_EXPR_SIZE),
            unsafe {
                BufferArg::from_raw_parts(value_bindings.input.clone(), value_bindings.input_len)
            },
            unsafe { BufferArg::from_raw_parts(value_offset.clone(), 1) },
            unsafe {
                BufferArg::from_raw_parts(value_bindings.rhs.clone(), value_bindings.rhs_len)
            },
            unsafe { BufferArg::from_raw_parts(value_rhs_offset.clone(), 1) },
            unsafe {
                BufferArg::from_raw_parts(index_bindings.input.clone(), index_bindings.input_len)
            },
            unsafe { BufferArg::from_raw_parts(index_offset.clone(), 1) },
            unsafe {
                BufferArg::from_raw_parts(index_bindings.rhs.clone(), index_bindings.rhs_len)
            },
            unsafe { BufferArg::from_raw_parts(index_rhs_offset.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output_offset.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output.source.handle.clone(), output.source.len()) },
        );
    }
    Ok(())
}

pub(crate) fn device_expr_gather_where_into_with_control<InputSource, IndexSource>(
    policy: &crate::policy::CubePolicy<InputSource::Runtime>,
    input: &InputSource,
    indices: &IndexSource,
    control: &select::MaskControl,
    output: &DeviceColumnMutView<InputSource::Runtime, InputSource::Item>,
) -> Result<(), Error>
where
    InputSource: KernelColumn + KernelColumnAt<S0>,
    InputSource::Runtime: Runtime,
    IndexSource: KernelColumn<Runtime = InputSource::Runtime, Item = u32> + KernelColumnAt<S0>,
    InputSource::Item: CubePrimitive + CubeElement,
    InputSource::Expr: DeviceGpuExpr<InputSource::Item>,
    IndexSource::Expr: DeviceGpuExpr<u32>,
{
    input.validate()?;
    indices.validate()?;
    ensure_same_len(indices.len(), control.len)?;
    ensure_same_len(indices.len(), output.len)?;
    let len = indices.len();
    if len == 0 {
        return Ok(());
    }

    let client = policy.client();
    let input_bindings = input.stage(policy)?;
    let index_bindings = indices.stage(policy)?;
    let input_slot0 = input_bindings.slot_or_first(0);
    let input_slot1 = input_bindings.slot_or_first(1);
    let input_slot2 = input_bindings.slot_or_first(2);
    let input_slot3 = input_bindings.slot_or_first(3);
    let index_slot0 = index_bindings.slot_or_first(0);
    let index_slot1 = index_bindings.slot_or_first(1);
    let index_slot2 = index_bindings.slot_or_first(2);
    let index_slot3 = index_bindings.slot_or_first(3);
    let input_slot_offsets = input_bindings.slot_offsets_handle(client)?;
    let index_slot_offsets = index_bindings.slot_offsets_handle(client)?;
    let output_offset = offset_handle(client, output.offset)?;
    let block_count_u32 = api_expr_block_count(len)?;

    unsafe {
        gather_if_flags_into_kernel::launch_unchecked::<
            InputSource::Item,
            InputSource::Expr,
            IndexSource::Expr,
            InputSource::Runtime,
        >(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_API_EXPR_SIZE),
            unsafe { BufferArg::from_raw_parts(input_slot0.0.clone(), input_slot0.1) },
            unsafe { BufferArg::from_raw_parts(input_slot1.0.clone(), input_slot1.1) },
            unsafe { BufferArg::from_raw_parts(input_slot2.0.clone(), input_slot2.1) },
            unsafe { BufferArg::from_raw_parts(input_slot3.0.clone(), input_slot3.1) },
            unsafe { BufferArg::from_raw_parts(input_slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(index_slot0.0.clone(), index_slot0.1) },
            unsafe { BufferArg::from_raw_parts(index_slot1.0.clone(), index_slot1.1) },
            unsafe { BufferArg::from_raw_parts(index_slot2.0.clone(), index_slot2.1) },
            unsafe { BufferArg::from_raw_parts(index_slot3.0.clone(), index_slot3.1) },
            unsafe { BufferArg::from_raw_parts(index_slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(control.flag.clone(), control.len) },
            unsafe { BufferArg::from_raw_parts(output_offset.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output.source.handle.clone(), output.source.len()) },
        );
    }
    Ok(())
}

#[allow(dead_code)]
pub fn device_expr_gather_where_into_with_policy<InputSource, IndexSource, Stencil, Pred>(
    policy: &crate::policy::CubePolicy<InputSource::Runtime>,
    input: &InputSource,
    indices: &IndexSource,
    stencil: &Stencil,
    output: &DeviceColumnMutView<InputSource::Runtime, InputSource::Item>,
    _pred: Pred,
) -> Result<(), Error>
where
    InputSource: KernelColumn + KernelColumnAt<S0>,
    InputSource::Runtime: Runtime,
    IndexSource: KernelColumn<Runtime = InputSource::Runtime, Item = u32> + KernelColumnAt<S0>,
    Stencil: SelectionStencil<Pred, Runtime = InputSource::Runtime>,
    InputSource::Item: CubePrimitive + CubeElement,
    InputSource::Expr: DeviceGpuExpr<InputSource::Item>,
    IndexSource::Expr: DeviceGpuExpr<u32>,
{
    let control = stencil.selection_flags_with_policy(policy, false)?;
    device_expr_gather_where_into_with_control(policy, input, indices, &control, output)
}

pub(crate) fn device_expr_scatter_where_into_with_control<ValueSource, IndexSource>(
    policy: &crate::policy::CubePolicy<ValueSource::Runtime>,
    values: &ValueSource,
    indices: &IndexSource,
    control: &select::MaskControl,
    output: &DeviceColumnMutView<ValueSource::Runtime, ValueSource::Item>,
) -> Result<(), Error>
where
    ValueSource: KernelColumn + KernelColumnAt<S0>,
    ValueSource::Runtime: Runtime,
    IndexSource: KernelColumn<Runtime = ValueSource::Runtime, Item = u32> + KernelColumnAt<S0>,
    ValueSource::Item: CubePrimitive + CubeElement,
    ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
    IndexSource::Expr: DeviceGpuExpr<u32>,
{
    values.validate()?;
    indices.validate()?;
    ensure_same_len(values.len(), indices.len())?;
    ensure_same_len(values.len(), control.len)?;
    let len = values.len();
    if len == 0 {
        return Ok(());
    }

    let client = policy.client();
    let value_bindings = values.stage(policy)?;
    let index_bindings = indices.stage(policy)?;
    let value_slot0 = value_bindings.slot_or_first(0);
    let value_slot1 = value_bindings.slot_or_first(1);
    let value_slot2 = value_bindings.slot_or_first(2);
    let value_slot3 = value_bindings.slot_or_first(3);
    let index_slot0 = index_bindings.slot_or_first(0);
    let index_slot1 = index_bindings.slot_or_first(1);
    let index_slot2 = index_bindings.slot_or_first(2);
    let index_slot3 = index_bindings.slot_or_first(3);
    let value_slot_offsets = value_bindings.slot_offsets_handle(client)?;
    let index_slot_offsets = index_bindings.slot_offsets_handle(client)?;
    let output_offset = offset_handle(client, output.offset)?;
    let block_count_u32 = api_expr_block_count(len)?;

    unsafe {
        scatter_if_flags_into_kernel::launch_unchecked::<
            ValueSource::Item,
            ValueSource::Expr,
            IndexSource::Expr,
            ValueSource::Runtime,
        >(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_API_EXPR_SIZE),
            unsafe { BufferArg::from_raw_parts(value_slot0.0.clone(), value_slot0.1) },
            unsafe { BufferArg::from_raw_parts(value_slot1.0.clone(), value_slot1.1) },
            unsafe { BufferArg::from_raw_parts(value_slot2.0.clone(), value_slot2.1) },
            unsafe { BufferArg::from_raw_parts(value_slot3.0.clone(), value_slot3.1) },
            unsafe { BufferArg::from_raw_parts(value_slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(index_slot0.0.clone(), index_slot0.1) },
            unsafe { BufferArg::from_raw_parts(index_slot1.0.clone(), index_slot1.1) },
            unsafe { BufferArg::from_raw_parts(index_slot2.0.clone(), index_slot2.1) },
            unsafe { BufferArg::from_raw_parts(index_slot3.0.clone(), index_slot3.1) },
            unsafe { BufferArg::from_raw_parts(index_slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(control.flag.clone(), control.len) },
            unsafe { BufferArg::from_raw_parts(output_offset.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output.source.handle.clone(), output.source.len()) },
        );
    }
    Ok(())
}

#[allow(dead_code)]
pub fn device_expr_scatter_where_into_with_policy<ValueSource, IndexSource, Stencil, Pred>(
    policy: &crate::policy::CubePolicy<ValueSource::Runtime>,
    values: &ValueSource,
    indices: &IndexSource,
    stencil: &Stencil,
    output: &DeviceColumnMutView<ValueSource::Runtime, ValueSource::Item>,
    _pred: Pred,
) -> Result<(), Error>
where
    ValueSource: KernelColumn + KernelColumnAt<S0>,
    ValueSource::Runtime: Runtime,
    IndexSource: KernelColumn<Runtime = ValueSource::Runtime, Item = u32> + KernelColumnAt<S0>,
    Stencil: SelectionStencil<Pred, Runtime = ValueSource::Runtime>,
    ValueSource::Item: CubePrimitive + CubeElement,
    ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
    IndexSource::Expr: DeviceGpuExpr<u32>,
{
    let control = stencil.selection_flags_with_policy(policy, false)?;
    device_expr_scatter_where_into_with_control(policy, values, indices, &control, output)
}
