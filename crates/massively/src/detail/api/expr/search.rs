use super::*;

pub(in crate::detail) fn device_expr_minmax_element_with_policy<ExprSource, Less>(
    policy: &crate::policy::CubePolicy<ExprSource::Runtime>,
    expr: &ExprSource,
) -> Result<Option<(usize, usize)>, Error>
where
    ExprSource: KernelColumn + KernelColumnAt<S0>,
    ExprSource::Runtime: Runtime,
    ExprSource::Item: CubePrimitive + CubeElement,
    ExprSource::Expr: DeviceGpuExpr<ExprSource::Item>,
    Less: BinaryPredicateOp<ExprSource::Item>,
{
    expr.validate()?;
    let len = expr.len();
    if len == 0 {
        return Ok(None);
    }

    let client = policy.client();
    let mut current_count = len.div_ceil(BLOCK_API_EXPR_SIZE as usize);
    let mut current_count_u32 =
        u32::try_from(current_count).map_err(|_| Error::LengthTooLarge { len: current_count })?;
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let mut current_handle = client.empty(current_count * 2 * std::mem::size_of::<u32>());
    let bindings = expr.stage(policy)?;
    let slot_offsets = bindings.slot_offsets_handle(client)?;
    let slot0 = bindings.slot_or_first(0);
    let slot1 = bindings.slot_or_first(1);
    let slot2 = bindings.slot_or_first(2);
    let slot3 = bindings.slot_or_first(3);

    unsafe {
        minmax_element_device_expr_partials_kernel::launch_unchecked::<
            ExprSource::Item,
            ExprSource::Expr,
            Less,
            ExprSource::Runtime,
        >(
            client,
            CubeCount::Static(current_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_API_EXPR_SIZE),
            unsafe { BufferArg::from_raw_parts(slot0.0.clone(), slot0.1) },
            unsafe { BufferArg::from_raw_parts(slot1.0.clone(), slot1.1) },
            unsafe { BufferArg::from_raw_parts(slot2.0.clone(), slot2.1) },
            unsafe { BufferArg::from_raw_parts(slot3.0.clone(), slot3.1) },
            unsafe { BufferArg::from_raw_parts(slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(current_handle.clone(), current_count * 2) },
        );
    }

    while current_count > 1 {
        let next_count = current_count.div_ceil(BLOCK_API_EXPR_SIZE as usize);
        let next_count_u32 =
            u32::try_from(next_count).map_err(|_| Error::LengthTooLarge { len: next_count })?;
        let candidate_len_handle = client.create_from_slice(u32::as_bytes(&[current_count_u32]));
        let next_handle = client.empty(next_count * 2 * std::mem::size_of::<u32>());

        unsafe {
            minmax_index_device_expr_partials_kernel::launch_unchecked::<
                ExprSource::Item,
                ExprSource::Expr,
                Less,
                ExprSource::Runtime,
            >(
                client,
                CubeCount::Static(next_count_u32, 1, 1),
                CubeDim::new_1d(BLOCK_API_EXPR_SIZE),
                unsafe { BufferArg::from_raw_parts(slot0.0.clone(), slot0.1) },
                unsafe { BufferArg::from_raw_parts(slot1.0.clone(), slot1.1) },
                unsafe { BufferArg::from_raw_parts(slot2.0.clone(), slot2.1) },
                unsafe { BufferArg::from_raw_parts(slot3.0.clone(), slot3.1) },
                unsafe { BufferArg::from_raw_parts(slot_offsets.clone(), 4) },
                unsafe { BufferArg::from_raw_parts(current_handle.clone(), current_count * 2) },
                unsafe { BufferArg::from_raw_parts(candidate_len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(next_handle.clone(), next_count * 2) },
            );
        }

        current_handle = next_handle;
        current_count = next_count;
        current_count_u32 = next_count_u32;
    }

    let bytes = client
        .read_one(current_handle)
        .map_err(|err| Error::Launch {
            message: format!("{err:?}"),
        })?;
    let indices = u32::from_bytes(&bytes);
    Ok(Some((indices[0] as usize, indices[1] as usize)))
}
