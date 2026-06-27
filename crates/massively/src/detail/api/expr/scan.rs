use super::*;

pub(in crate::detail) fn device_expr_adjacent_difference_with_policy<ExprSource, Op>(
    policy: &crate::policy::CubePolicy<ExprSource::Runtime>,
    expr: &ExprSource,
) -> Result<DeviceVec<ExprSource::Runtime, ExprSource::Item>, Error>
where
    ExprSource: KernelColumn + KernelColumnAt<S0>,
    ExprSource::Runtime: Runtime,
    ExprSource::Item: CubePrimitive + CubeElement,
    ExprSource::Expr: DeviceGpuExpr<ExprSource::Item>,
    Op: BinaryOp<ExprSource::Item>,
{
    expr.validate()?;
    let len = expr.len();
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let client = policy.client();
    let output_handle = client.empty(len * std::mem::size_of::<ExprSource::Item>());

    if len != 0 {
        let block_count_u32 = api_expr_block_count(len)?;
        let len_values = [len_u32];
        let bindings = expr.stage(policy)?;
        let slot0 = bindings.slots.first().unwrap();
        let slot1 = bindings.slots.get(1).unwrap_or(slot0);
        let slot2 = bindings.slots.get(2).unwrap_or(slot0);
        let slot3 = bindings.slots.get(3).unwrap_or(slot0);
        let slot_offsets = bindings.slot_offsets_handle(client)?;
        let len_handle = client.create_from_slice(u32::as_bytes(&len_values));
        unsafe {
            adjacent_difference_expr_kernel::launch_unchecked::<
                ExprSource::Item,
                ExprSource::Expr,
                Op,
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
                unsafe { BufferArg::from_raw_parts(output_handle.clone(), len) },
            );
        }
    }

    Ok(DeviceVec::from_handle(policy.id(), output_handle, len))
}
#[allow(dead_code)]
pub(in crate::detail) fn device_expr_inclusive_scan_by_key_expr_keys_with_policy<
    KeySource,
    ExprSource,
    KeyEq,
    Op,
>(
    policy: &crate::policy::CubePolicy<ExprSource::Runtime>,
    keys: &KeySource,
    expr: &ExprSource,
) -> Result<DeviceVec<ExprSource::Runtime, ExprSource::Item>, Error>
where
    KeySource: KernelColumn<Runtime = ExprSource::Runtime> + KernelColumnAt<S0>,
    ExprSource: KernelColumn + KernelColumnAt<S0>,
    ExprSource::Runtime: Runtime,
    KeySource::Item: CubePrimitive + CubeElement,
    KeySource::Expr: DeviceGpuExpr<KeySource::Item>,
    ExprSource::Item: CubePrimitive + CubeElement,
    ExprSource::Expr: DeviceGpuExpr<ExprSource::Item>,
    KeyEq: BinaryPredicateOp<KeySource::Item>,
    Op: BinaryOp<ExprSource::Item>,
{
    keys.validate()?;
    expr.validate()?;
    ensure_same_len(keys.len(), expr.len())?;

    let key_bindings = keys.stage(policy)?;
    let value_bindings = expr.stage(policy)?;
    primitive_scan::inclusive_scan_by_key_device_expr::<
        ExprSource::Runtime,
        KeySource::Item,
        ExprSource::Item,
        KeySource::Expr,
        ExprSource::Expr,
        KeyEq,
        Op,
    >(policy, &key_bindings, &value_bindings, expr.len())
}

#[allow(dead_code)]
pub(in crate::detail) fn device_expr_exclusive_scan_by_key_expr_keys_with_policy<
    KeySource,
    ExprSource,
    KeyEq,
    Op,
>(
    policy: &crate::policy::CubePolicy<ExprSource::Runtime>,
    keys: &KeySource,
    expr: &ExprSource,
    init: ExprSource::Item,
) -> Result<DeviceVec<ExprSource::Runtime, ExprSource::Item>, Error>
where
    KeySource: KernelColumn<Runtime = ExprSource::Runtime> + KernelColumnAt<S0>,
    ExprSource: KernelColumn + KernelColumnAt<S0>,
    ExprSource::Runtime: Runtime,
    KeySource::Item: CubePrimitive + CubeElement,
    KeySource::Expr: DeviceGpuExpr<KeySource::Item>,
    ExprSource::Item: CubePrimitive + CubeElement,
    ExprSource::Expr: DeviceGpuExpr<ExprSource::Item>,
    KeyEq: BinaryPredicateOp<KeySource::Item>,
    Op: BinaryOp<ExprSource::Item>,
{
    keys.validate()?;
    expr.validate()?;
    ensure_same_len(keys.len(), expr.len())?;

    let key_bindings = keys.stage(policy)?;
    let value_bindings = expr.stage(policy)?;
    primitive_scan::exclusive_scan_by_key_device_expr::<
        ExprSource::Runtime,
        KeySource::Item,
        ExprSource::Item,
        KeySource::Expr,
        ExprSource::Expr,
        KeyEq,
        Op,
    >(policy, &key_bindings, &value_bindings, expr.len(), init)
}
