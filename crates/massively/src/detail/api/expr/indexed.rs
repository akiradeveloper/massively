use super::*;

pub(in crate::detail) fn device_expr_gather_with_policy<InputSource, IndexSource>(
    policy: &crate::policy::CubePolicy<InputSource::Runtime>,
    input: &InputSource,
    indices: &IndexSource,
) -> Result<DeviceVec<InputSource::Runtime, InputSource::Item>, Error>
where
    InputSource: KernelColumn + KernelColumnAt<S0>,
    InputSource::Runtime: Runtime,
    IndexSource: crate::detail::read::KernelReadBoundMany<InputSource::Runtime, Item = MIndex>,
    InputSource::Item: CubePrimitive + CubeElement,
    InputSource::Expr: GpuExpr<InputSource::Item>,
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
    let mut index_bindings = KernelColumnBindings::empty(client);
    <IndexSource as crate::detail::read::KernelReadAtEnv<
        InputSource::Runtime,
        crate::detail::read::Env0,
    >>::stage_at_env(indices, &mut index_bindings)?;
    index_bindings.finish();
    let input_offset = offset_handle(client, input_bindings.input_offset)?;
    let input_rhs_offset = offset_handle(client, input_bindings.rhs_offset)?;
    let index_slot0 = index_bindings.slot_or_first(0);
    let index_slot1 = index_bindings.slot_or_first(1);
    let index_slot2 = index_bindings.slot_or_first(2);
    let index_slot3 = index_bindings.slot_or_first(3);
    let index_slot4 = index_bindings.slot_or_first(4);
    let index_slot5 = index_bindings.slot_or_first(5);
    let index_slot6 = index_bindings.slot_or_first(6);
    let index_slot7 = index_bindings.slot_or_first(7);
    let index_slot_offsets = index_bindings.slot_offsets8_handle(client)?;
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let output_offset = client.create_from_slice(u32::as_bytes(&[0_u32]));

    unsafe {
        gather_device_expr_index7_into_kernel::launch_unchecked::<
            InputSource::Item,
            InputSource::Expr,
            <IndexSource as crate::detail::read::KernelReadBoundMany<InputSource::Runtime>>::ExprAt,
            <IndexSource as crate::detail::read::KernelReadBoundMany<InputSource::Runtime>>::Leaf0,
            <IndexSource as crate::detail::read::KernelReadBoundMany<InputSource::Runtime>>::Leaf1,
            <IndexSource as crate::detail::read::KernelReadBoundMany<InputSource::Runtime>>::Leaf2,
            <IndexSource as crate::detail::read::KernelReadBoundMany<InputSource::Runtime>>::Leaf3,
            <IndexSource as crate::detail::read::KernelReadBoundMany<InputSource::Runtime>>::Leaf4,
            <IndexSource as crate::detail::read::KernelReadBoundMany<InputSource::Runtime>>::Leaf5,
            <IndexSource as crate::detail::read::KernelReadBoundMany<InputSource::Runtime>>::Leaf6,
            <IndexSource as crate::detail::read::KernelReadBoundMany<InputSource::Runtime>>::Leaf7,
            InputSource::Runtime,
        >(
            client,
            crate::detail::launch::cube_count_1d(block_count_u32),
            CubeDim::new_1d(BLOCK_API_EXPR_SIZE),
            unsafe {
                BufferArg::from_raw_parts(input_bindings.input.clone(), input_bindings.input_len)
            },
            unsafe { BufferArg::from_raw_parts(input_offset.clone(), 1) },
            unsafe {
                BufferArg::from_raw_parts(input_bindings.rhs.clone(), input_bindings.rhs_len)
            },
            unsafe { BufferArg::from_raw_parts(input_rhs_offset.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(index_slot0.0.clone(), index_slot0.1) },
            unsafe { BufferArg::from_raw_parts(index_slot1.0.clone(), index_slot1.1) },
            unsafe { BufferArg::from_raw_parts(index_slot2.0.clone(), index_slot2.1) },
            unsafe { BufferArg::from_raw_parts(index_slot3.0.clone(), index_slot3.1) },
            unsafe { BufferArg::from_raw_parts(index_slot4.0.clone(), index_slot4.1) },
            unsafe { BufferArg::from_raw_parts(index_slot5.0.clone(), index_slot5.1) },
            unsafe { BufferArg::from_raw_parts(index_slot6.0.clone(), index_slot6.1) },
            unsafe { BufferArg::from_raw_parts(index_slot7.0.clone(), index_slot7.1) },
            unsafe { BufferArg::from_raw_parts(index_slot_offsets.clone(), 8) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output_offset.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output_handle.clone(), len) },
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
    IndexSource: crate::detail::read::KernelReadBoundMany<InputSource::Runtime, Item = MIndex>,
    InputSource::Item: CubePrimitive + CubeElement,
    InputSource::Expr: GpuExpr<InputSource::Item>,
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
    let mut index_bindings = KernelColumnBindings::empty(client);
    <IndexSource as crate::detail::read::KernelReadAtEnv<
        InputSource::Runtime,
        crate::detail::read::Env0,
    >>::stage_at_env(indices, &mut index_bindings)?;
    index_bindings.finish();
    let input_offset = offset_handle(client, input_bindings.input_offset)?;
    let input_rhs_offset = offset_handle(client, input_bindings.rhs_offset)?;
    let index_slot0 = index_bindings.slot_or_first(0);
    let index_slot1 = index_bindings.slot_or_first(1);
    let index_slot2 = index_bindings.slot_or_first(2);
    let index_slot3 = index_bindings.slot_or_first(3);
    let index_slot4 = index_bindings.slot_or_first(4);
    let index_slot5 = index_bindings.slot_or_first(5);
    let index_slot6 = index_bindings.slot_or_first(6);
    let index_slot7 = index_bindings.slot_or_first(7);
    let index_slot_offsets = index_bindings.slot_offsets8_handle(client)?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let output_offset = offset_handle(client, output.offset)?;

    unsafe {
        gather_device_expr_index7_into_kernel::launch_unchecked::<
            InputSource::Item,
            InputSource::Expr,
            <IndexSource as crate::detail::read::KernelReadBoundMany<InputSource::Runtime>>::ExprAt,
            <IndexSource as crate::detail::read::KernelReadBoundMany<InputSource::Runtime>>::Leaf0,
            <IndexSource as crate::detail::read::KernelReadBoundMany<InputSource::Runtime>>::Leaf1,
            <IndexSource as crate::detail::read::KernelReadBoundMany<InputSource::Runtime>>::Leaf2,
            <IndexSource as crate::detail::read::KernelReadBoundMany<InputSource::Runtime>>::Leaf3,
            <IndexSource as crate::detail::read::KernelReadBoundMany<InputSource::Runtime>>::Leaf4,
            <IndexSource as crate::detail::read::KernelReadBoundMany<InputSource::Runtime>>::Leaf5,
            <IndexSource as crate::detail::read::KernelReadBoundMany<InputSource::Runtime>>::Leaf6,
            <IndexSource as crate::detail::read::KernelReadBoundMany<InputSource::Runtime>>::Leaf7,
            InputSource::Runtime,
        >(
            client,
            crate::detail::launch::cube_count_1d(block_count_u32),
            CubeDim::new_1d(BLOCK_API_EXPR_SIZE),
            unsafe {
                BufferArg::from_raw_parts(input_bindings.input.clone(), input_bindings.input_len)
            },
            unsafe { BufferArg::from_raw_parts(input_offset.clone(), 1) },
            unsafe {
                BufferArg::from_raw_parts(input_bindings.rhs.clone(), input_bindings.rhs_len)
            },
            unsafe { BufferArg::from_raw_parts(input_rhs_offset.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(index_slot0.0.clone(), index_slot0.1) },
            unsafe { BufferArg::from_raw_parts(index_slot1.0.clone(), index_slot1.1) },
            unsafe { BufferArg::from_raw_parts(index_slot2.0.clone(), index_slot2.1) },
            unsafe { BufferArg::from_raw_parts(index_slot3.0.clone(), index_slot3.1) },
            unsafe { BufferArg::from_raw_parts(index_slot4.0.clone(), index_slot4.1) },
            unsafe { BufferArg::from_raw_parts(index_slot5.0.clone(), index_slot5.1) },
            unsafe { BufferArg::from_raw_parts(index_slot6.0.clone(), index_slot6.1) },
            unsafe { BufferArg::from_raw_parts(index_slot7.0.clone(), index_slot7.1) },
            unsafe { BufferArg::from_raw_parts(index_slot_offsets.clone(), 8) },
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
    IndexSource: crate::detail::read::KernelReadBoundMany<ValueSource::Runtime, Item = MIndex>,
    ValueSource::Item: CubePrimitive + CubeElement,
    ValueSource::Expr: GpuExpr<ValueSource::Item>,
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
    let mut index_bindings = KernelColumnBindings::empty(client);
    <IndexSource as crate::detail::read::KernelReadAtEnv<
        ValueSource::Runtime,
        crate::detail::read::Env0,
    >>::stage_at_env(indices, &mut index_bindings)?;
    index_bindings.finish();
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let value_offset = offset_handle(client, value_bindings.input_offset)?;
    let value_rhs_offset = offset_handle(client, value_bindings.rhs_offset)?;
    let index_slot0 = index_bindings.slot_or_first(0);
    let index_slot1 = index_bindings.slot_or_first(1);
    let index_slot2 = index_bindings.slot_or_first(2);
    let index_slot3 = index_bindings.slot_or_first(3);
    let index_slot4 = index_bindings.slot_or_first(4);
    let index_slot5 = index_bindings.slot_or_first(5);
    let index_slot6 = index_bindings.slot_or_first(6);
    let index_slot7 = index_bindings.slot_or_first(7);
    let index_slot_offsets = index_bindings.slot_offsets8_handle(client)?;
    let output_offset = offset_handle(client, output.offset)?;

    unsafe {
        scatter_expr_index7_into_kernel::launch_unchecked::<
            ValueSource::Item,
            ValueSource::Expr,
            <IndexSource as crate::detail::read::KernelReadBoundMany<ValueSource::Runtime>>::ExprAt,
            <IndexSource as crate::detail::read::KernelReadBoundMany<ValueSource::Runtime>>::Leaf0,
            <IndexSource as crate::detail::read::KernelReadBoundMany<ValueSource::Runtime>>::Leaf1,
            <IndexSource as crate::detail::read::KernelReadBoundMany<ValueSource::Runtime>>::Leaf2,
            <IndexSource as crate::detail::read::KernelReadBoundMany<ValueSource::Runtime>>::Leaf3,
            <IndexSource as crate::detail::read::KernelReadBoundMany<ValueSource::Runtime>>::Leaf4,
            <IndexSource as crate::detail::read::KernelReadBoundMany<ValueSource::Runtime>>::Leaf5,
            <IndexSource as crate::detail::read::KernelReadBoundMany<ValueSource::Runtime>>::Leaf6,
            <IndexSource as crate::detail::read::KernelReadBoundMany<ValueSource::Runtime>>::Leaf7,
            ValueSource::Runtime,
        >(
            client,
            crate::detail::launch::cube_count_1d(block_count_u32),
            CubeDim::new_1d(BLOCK_API_EXPR_SIZE),
            unsafe {
                BufferArg::from_raw_parts(value_bindings.input.clone(), value_bindings.input_len)
            },
            unsafe { BufferArg::from_raw_parts(value_offset.clone(), 1) },
            unsafe {
                BufferArg::from_raw_parts(value_bindings.rhs.clone(), value_bindings.rhs_len)
            },
            unsafe { BufferArg::from_raw_parts(value_rhs_offset.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(index_slot0.0.clone(), index_slot0.1) },
            unsafe { BufferArg::from_raw_parts(index_slot1.0.clone(), index_slot1.1) },
            unsafe { BufferArg::from_raw_parts(index_slot2.0.clone(), index_slot2.1) },
            unsafe { BufferArg::from_raw_parts(index_slot3.0.clone(), index_slot3.1) },
            unsafe { BufferArg::from_raw_parts(index_slot4.0.clone(), index_slot4.1) },
            unsafe { BufferArg::from_raw_parts(index_slot5.0.clone(), index_slot5.1) },
            unsafe { BufferArg::from_raw_parts(index_slot6.0.clone(), index_slot6.1) },
            unsafe { BufferArg::from_raw_parts(index_slot7.0.clone(), index_slot7.1) },
            unsafe { BufferArg::from_raw_parts(index_slot_offsets.clone(), 8) },
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
    IndexSource: crate::detail::read::KernelReadBoundMany<InputSource::Runtime, Item = MIndex>,
    InputSource::Item: CubePrimitive + CubeElement,
    InputSource::Expr: DeviceGpuExpr<InputSource::Item>,
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
    let mut index_bindings = KernelColumnBindings::empty(client);
    <IndexSource as crate::detail::read::KernelReadAtEnv<
        InputSource::Runtime,
        crate::detail::read::Env0,
    >>::stage_at_env(indices, &mut index_bindings)?;
    index_bindings.finish();
    let input_slot0 = input_bindings.slot_or_first(0);
    let input_slot1 = input_bindings.slot_or_first(1);
    let input_slot2 = input_bindings.slot_or_first(2);
    let input_slot3 = input_bindings.slot_or_first(3);
    let index_slot0 = index_bindings.slot_or_first(0);
    let index_slot1 = index_bindings.slot_or_first(1);
    let index_slot2 = index_bindings.slot_or_first(2);
    let index_slot3 = index_bindings.slot_or_first(3);
    let index_slot4 = index_bindings.slot_or_first(4);
    let index_slot5 = index_bindings.slot_or_first(5);
    let index_slot6 = index_bindings.slot_or_first(6);
    let index_slot7 = index_bindings.slot_or_first(7);
    let input_slot_offsets = input_bindings.slot_offsets_handle(client)?;
    let index_slot_offsets = index_bindings.slot_offsets8_handle(client)?;
    let output_offset = offset_handle(client, output.offset)?;
    let block_count_u32 = api_expr_block_count(len)?;

    unsafe {
        gather_if_flags_index7_into_kernel::launch_unchecked::<
            InputSource::Item,
            InputSource::Expr,
            <IndexSource as crate::detail::read::KernelReadBoundMany<InputSource::Runtime>>::ExprAt,
            <IndexSource as crate::detail::read::KernelReadBoundMany<InputSource::Runtime>>::Leaf0,
            <IndexSource as crate::detail::read::KernelReadBoundMany<InputSource::Runtime>>::Leaf1,
            <IndexSource as crate::detail::read::KernelReadBoundMany<InputSource::Runtime>>::Leaf2,
            <IndexSource as crate::detail::read::KernelReadBoundMany<InputSource::Runtime>>::Leaf3,
            <IndexSource as crate::detail::read::KernelReadBoundMany<InputSource::Runtime>>::Leaf4,
            <IndexSource as crate::detail::read::KernelReadBoundMany<InputSource::Runtime>>::Leaf5,
            <IndexSource as crate::detail::read::KernelReadBoundMany<InputSource::Runtime>>::Leaf6,
            <IndexSource as crate::detail::read::KernelReadBoundMany<InputSource::Runtime>>::Leaf7,
            InputSource::Runtime,
        >(
            client,
            crate::detail::launch::cube_count_1d(block_count_u32),
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
            unsafe { BufferArg::from_raw_parts(index_slot4.0.clone(), index_slot4.1) },
            unsafe { BufferArg::from_raw_parts(index_slot5.0.clone(), index_slot5.1) },
            unsafe { BufferArg::from_raw_parts(index_slot6.0.clone(), index_slot6.1) },
            unsafe { BufferArg::from_raw_parts(index_slot7.0.clone(), index_slot7.1) },
            unsafe { BufferArg::from_raw_parts(index_slot_offsets.clone(), 8) },
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
    IndexSource: crate::detail::read::KernelReadBoundMany<InputSource::Runtime, Item = MIndex>,
    Stencil: SelectionStencil<Pred, Runtime = InputSource::Runtime>,
    InputSource::Item: CubePrimitive + CubeElement,
    InputSource::Expr: DeviceGpuExpr<InputSource::Item>,
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
    IndexSource: crate::detail::read::KernelReadBoundMany<ValueSource::Runtime, Item = MIndex>,
    ValueSource::Item: CubePrimitive + CubeElement,
    ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
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
    let mut index_bindings = KernelColumnBindings::empty(client);
    <IndexSource as crate::detail::read::KernelReadAtEnv<
        ValueSource::Runtime,
        crate::detail::read::Env0,
    >>::stage_at_env(indices, &mut index_bindings)?;
    index_bindings.finish();
    let value_slot0 = value_bindings.slot_or_first(0);
    let value_slot1 = value_bindings.slot_or_first(1);
    let value_slot2 = value_bindings.slot_or_first(2);
    let value_slot3 = value_bindings.slot_or_first(3);
    let index_slot0 = index_bindings.slot_or_first(0);
    let index_slot1 = index_bindings.slot_or_first(1);
    let index_slot2 = index_bindings.slot_or_first(2);
    let index_slot3 = index_bindings.slot_or_first(3);
    let index_slot4 = index_bindings.slot_or_first(4);
    let index_slot5 = index_bindings.slot_or_first(5);
    let index_slot6 = index_bindings.slot_or_first(6);
    let index_slot7 = index_bindings.slot_or_first(7);
    let value_slot_offsets = value_bindings.slot_offsets_handle(client)?;
    let index_slot_offsets = index_bindings.slot_offsets8_handle(client)?;
    let output_offset = offset_handle(client, output.offset)?;
    let block_count_u32 = api_expr_block_count(len)?;

    unsafe {
        scatter_if_flags_index7_into_kernel::launch_unchecked::<
            ValueSource::Item,
            ValueSource::Expr,
            <IndexSource as crate::detail::read::KernelReadBoundMany<ValueSource::Runtime>>::ExprAt,
            <IndexSource as crate::detail::read::KernelReadBoundMany<ValueSource::Runtime>>::Leaf0,
            <IndexSource as crate::detail::read::KernelReadBoundMany<ValueSource::Runtime>>::Leaf1,
            <IndexSource as crate::detail::read::KernelReadBoundMany<ValueSource::Runtime>>::Leaf2,
            <IndexSource as crate::detail::read::KernelReadBoundMany<ValueSource::Runtime>>::Leaf3,
            <IndexSource as crate::detail::read::KernelReadBoundMany<ValueSource::Runtime>>::Leaf4,
            <IndexSource as crate::detail::read::KernelReadBoundMany<ValueSource::Runtime>>::Leaf5,
            <IndexSource as crate::detail::read::KernelReadBoundMany<ValueSource::Runtime>>::Leaf6,
            <IndexSource as crate::detail::read::KernelReadBoundMany<ValueSource::Runtime>>::Leaf7,
            ValueSource::Runtime,
        >(
            client,
            crate::detail::launch::cube_count_1d(block_count_u32),
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
            unsafe { BufferArg::from_raw_parts(index_slot4.0.clone(), index_slot4.1) },
            unsafe { BufferArg::from_raw_parts(index_slot5.0.clone(), index_slot5.1) },
            unsafe { BufferArg::from_raw_parts(index_slot6.0.clone(), index_slot6.1) },
            unsafe { BufferArg::from_raw_parts(index_slot7.0.clone(), index_slot7.1) },
            unsafe { BufferArg::from_raw_parts(index_slot_offsets.clone(), 8) },
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
    IndexSource: crate::detail::read::KernelReadBoundMany<ValueSource::Runtime, Item = MIndex>,
    Stencil: SelectionStencil<Pred, Runtime = ValueSource::Runtime>,
    ValueSource::Item: CubePrimitive + CubeElement,
    ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
{
    let control = stencil.selection_flags_with_policy(policy, false)?;
    device_expr_scatter_where_into_with_control(policy, values, indices, &control, output)
}
