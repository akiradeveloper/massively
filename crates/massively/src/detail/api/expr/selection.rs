use super::*;

pub(crate) fn replace_where_into_with_control<R, T>(
    policy: &crate::policy::CubePolicy<R>,
    replacement: T,
    control: &select::MaskControl,
    output: &DeviceColumnMutView<R, T>,
) -> Result<(), Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    ensure_same_len(control.len, output.len)?;
    let len = control.len;
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
            unsafe { BufferArg::from_raw_parts(control.flag.clone(), control.len) },
            unsafe { BufferArg::from_raw_parts(output_offset.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output.source.handle.clone(), output.source.len()) },
        );
    }
    Ok(())
}

#[allow(dead_code)]
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
    let control = stencil.selection_flags_with_policy(policy, false)?;
    replace_where_into_with_control(policy, replacement, &control, output)
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
    let flags = stencil.selection_flags_with_policy(policy, false)?;
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

pub(in crate::detail) fn device_expr_compact_with_selection_with_policy<ExprSource>(
    policy: &crate::policy::CubePolicy<ExprSource::Runtime>,
    expr: &ExprSource,
    selected_rank: &select::SelectedRankControl,
    count: usize,
) -> Result<DeviceVec<ExprSource::Runtime, ExprSource::Item>, Error>
where
    ExprSource: KernelColumn + KernelColumnAt<S0>,
    ExprSource::Runtime: Runtime,
    ExprSource::Item: CubePrimitive + CubeElement,
    ExprSource::Expr: DeviceGpuExpr<ExprSource::Item>,
{
    expr.validate()?;
    ensure_same_len(expr.len(), selected_rank.len)?;
    if selected_rank.len == 0 || count == 0 {
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
    let output_offset = offset_handle(client, 0)?;
    let block_count_u32 = api_expr_block_count(selected_rank.len)?;

    unsafe {
        compact_scatter_device_expr_kernel::launch_unchecked::<
            ExprSource::Item,
            ExprSource::Expr,
            ExprSource::Runtime,
        >(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_API_EXPR_SIZE),
            unsafe { BufferArg::from_raw_parts(selected_rank.flag.clone(), selected_rank.len) },
            unsafe { BufferArg::from_raw_parts(selected_rank.position.clone(), selected_rank.len) },
            unsafe { BufferArg::from_raw_parts(slot0.0.clone(), slot0.1) },
            unsafe { BufferArg::from_raw_parts(slot1.0.clone(), slot1.1) },
            unsafe { BufferArg::from_raw_parts(slot2.0.clone(), slot2.1) },
            unsafe { BufferArg::from_raw_parts(slot3.0.clone(), slot3.1) },
            unsafe { BufferArg::from_raw_parts(slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(output_offset.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output_handle.clone(), count) },
        );
    }

    Ok(DeviceVec::from_handle(policy.id(), output_handle, count))
}

pub(in crate::detail) fn device_expr_compact_into_with_selection_with_policy<ExprSource>(
    policy: &crate::policy::CubePolicy<ExprSource::Runtime>,
    expr: &ExprSource,
    selected_rank: &select::SelectedRankControl,
    count: usize,
    output: &DeviceColumnMutView<ExprSource::Runtime, ExprSource::Item>,
) -> Result<(), Error>
where
    ExprSource: KernelColumn + KernelColumnAt<S0>,
    ExprSource::Runtime: Runtime,
    ExprSource::Item: CubePrimitive + CubeElement,
    ExprSource::Expr: DeviceGpuExpr<ExprSource::Item>,
{
    expr.validate()?;
    ensure_same_len(expr.len(), selected_rank.len)?;
    if count > output.len {
        return Err(Error::LengthMismatch {
            input: count,
            output: output.len,
        });
    }
    if selected_rank.len == 0 || count == 0 {
        return Ok(());
    }

    let client = policy.client();
    let bindings = expr.stage(policy)?;
    let slot_offsets = bindings.slot_offsets_handle(client)?;
    let slot0 = bindings.slots.first().unwrap();
    let slot1 = bindings.slots.get(1).unwrap_or(slot0);
    let slot2 = bindings.slots.get(2).unwrap_or(slot0);
    let slot3 = bindings.slots.get(3).unwrap_or(slot0);
    let output_offset = offset_handle(client, output.offset)?;
    let block_count_u32 = api_expr_block_count(selected_rank.len)?;

    unsafe {
        compact_scatter_device_expr_kernel::launch_unchecked::<
            ExprSource::Item,
            ExprSource::Expr,
            ExprSource::Runtime,
        >(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_API_EXPR_SIZE),
            unsafe { BufferArg::from_raw_parts(selected_rank.flag.clone(), selected_rank.len) },
            unsafe { BufferArg::from_raw_parts(selected_rank.position.clone(), selected_rank.len) },
            unsafe { BufferArg::from_raw_parts(slot0.0.clone(), slot0.1) },
            unsafe { BufferArg::from_raw_parts(slot1.0.clone(), slot1.1) },
            unsafe { BufferArg::from_raw_parts(slot2.0.clone(), slot2.1) },
            unsafe { BufferArg::from_raw_parts(slot3.0.clone(), slot3.1) },
            unsafe { BufferArg::from_raw_parts(slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(output_offset.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output.source.handle.clone(), output.source.len()) },
        );
    }

    Ok(())
}

pub(in crate::detail) fn device_expr_apply_selected2_with_policy<Left, Right>(
    policy: &crate::policy::CubePolicy<Left::Runtime>,
    left: &Left,
    right: &Right,
    selected_rank: &select::SelectedRankControl,
    count: usize,
) -> Result<
    (
        DeviceVec<Left::Runtime, Left::Item>,
        DeviceVec<Left::Runtime, Right::Item>,
    ),
    Error,
>
where
    Left: KernelColumn + KernelColumnAt<S0>,
    Right: KernelColumn<Runtime = Left::Runtime> + KernelColumnAt<S0>,
    Left::Runtime: Runtime,
    Left::Item: CubePrimitive + CubeElement,
    Right::Item: CubePrimitive + CubeElement,
    Left::Expr: DeviceGpuExpr<Left::Item>,
    Right::Expr: DeviceGpuExpr<Right::Item>,
{
    left.validate()?;
    right.validate()?;
    ensure_same_len(left.len(), right.len())?;
    ensure_same_len(left.len(), selected_rank.len)?;
    if selected_rank.len == 0 || count == 0 {
        return Ok((policy.empty_device_vec(), policy.empty_device_vec()));
    }

    let client = policy.client();
    let output_left = client.empty(count * std::mem::size_of::<Left::Item>());
    let output_right = client.empty(count * std::mem::size_of::<Right::Item>());
    let left_bindings = left.stage(policy)?;
    let right_bindings = right.stage(policy)?;
    let left_slot0 = left_bindings.slot_or_first(0);
    let left_slot1 = left_bindings.slot_or_first(1);
    let left_slot2 = left_bindings.slot_or_first(2);
    let left_slot3 = left_bindings.slot_or_first(3);
    let right_slot0 = right_bindings.slot_or_first(0);
    let right_slot1 = right_bindings.slot_or_first(1);
    let right_slot2 = right_bindings.slot_or_first(2);
    let right_slot3 = right_bindings.slot_or_first(3);
    let left_slot_offsets = left_bindings.slot_offsets_handle(client)?;
    let right_slot_offsets = right_bindings.slot_offsets_handle(client)?;
    let block_count_u32 = api_expr_block_count(selected_rank.len)?;

    unsafe {
        selected_apply_tuple2_device_expr_kernel::launch_unchecked::<
            Left::Item,
            Right::Item,
            Left::Expr,
            Right::Expr,
            Left::Runtime,
        >(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_API_EXPR_SIZE),
            unsafe { BufferArg::from_raw_parts(selected_rank.flag.clone(), selected_rank.len) },
            unsafe { BufferArg::from_raw_parts(selected_rank.position.clone(), selected_rank.len) },
            unsafe { BufferArg::from_raw_parts(left_slot0.0.clone(), left_slot0.1) },
            unsafe { BufferArg::from_raw_parts(left_slot1.0.clone(), left_slot1.1) },
            unsafe { BufferArg::from_raw_parts(left_slot2.0.clone(), left_slot2.1) },
            unsafe { BufferArg::from_raw_parts(left_slot3.0.clone(), left_slot3.1) },
            unsafe { BufferArg::from_raw_parts(left_slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(right_slot0.0.clone(), right_slot0.1) },
            unsafe { BufferArg::from_raw_parts(right_slot1.0.clone(), right_slot1.1) },
            unsafe { BufferArg::from_raw_parts(right_slot2.0.clone(), right_slot2.1) },
            unsafe { BufferArg::from_raw_parts(right_slot3.0.clone(), right_slot3.1) },
            unsafe { BufferArg::from_raw_parts(right_slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(output_left.clone(), count) },
            unsafe { BufferArg::from_raw_parts(output_right.clone(), count) },
        );
    }

    Ok((
        DeviceVec::from_handle(policy.id(), output_left, count),
        DeviceVec::from_handle(policy.id(), output_right, count),
    ))
}

pub(in crate::detail) fn device_expr_apply_selected3_with_policy<First, Second, Third>(
    policy: &crate::policy::CubePolicy<First::Runtime>,
    first: &First,
    second: &Second,
    third: &Third,
    selected_rank: &select::SelectedRankControl,
    count: usize,
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
    First::Runtime: Runtime,
    First::Item: CubePrimitive + CubeElement,
    Second::Item: CubePrimitive + CubeElement,
    Third::Item: CubePrimitive + CubeElement,
    First::Expr: DeviceGpuExpr<First::Item>,
    Second::Expr: DeviceGpuExpr<Second::Item>,
    Third::Expr: DeviceGpuExpr<Third::Item>,
{
    first.validate()?;
    second.validate()?;
    third.validate()?;
    ensure_same_len(first.len(), second.len())?;
    ensure_same_len(first.len(), third.len())?;
    ensure_same_len(first.len(), selected_rank.len)?;
    if selected_rank.len == 0 || count == 0 {
        return Ok((
            policy.empty_device_vec(),
            policy.empty_device_vec(),
            policy.empty_device_vec(),
        ));
    }

    let client = policy.client();
    let output_first = client.empty(count * std::mem::size_of::<First::Item>());
    let output_second = client.empty(count * std::mem::size_of::<Second::Item>());
    let output_third = client.empty(count * std::mem::size_of::<Third::Item>());
    let first_bindings = first.stage(policy)?;
    let second_bindings = second.stage(policy)?;
    let third_bindings = third.stage(policy)?;
    let first_slot0 = first_bindings.slot_or_first(0);
    let first_slot1 = first_bindings.slot_or_first(1);
    let first_slot2 = first_bindings.slot_or_first(2);
    let first_slot3 = first_bindings.slot_or_first(3);
    let second_slot0 = second_bindings.slot_or_first(0);
    let second_slot1 = second_bindings.slot_or_first(1);
    let second_slot2 = second_bindings.slot_or_first(2);
    let second_slot3 = second_bindings.slot_or_first(3);
    let third_slot0 = third_bindings.slot_or_first(0);
    let third_slot1 = third_bindings.slot_or_first(1);
    let third_slot2 = third_bindings.slot_or_first(2);
    let third_slot3 = third_bindings.slot_or_first(3);
    let first_slot_offsets = first_bindings.slot_offsets_handle(client)?;
    let second_slot_offsets = second_bindings.slot_offsets_handle(client)?;
    let third_slot_offsets = third_bindings.slot_offsets_handle(client)?;
    let block_count_u32 = api_expr_block_count(selected_rank.len)?;

    unsafe {
        selected_apply_tuple3_device_expr_kernel::launch_unchecked::<
            First::Item,
            Second::Item,
            Third::Item,
            First::Expr,
            Second::Expr,
            Third::Expr,
            First::Runtime,
        >(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_API_EXPR_SIZE),
            unsafe { BufferArg::from_raw_parts(selected_rank.flag.clone(), selected_rank.len) },
            unsafe { BufferArg::from_raw_parts(selected_rank.position.clone(), selected_rank.len) },
            unsafe { BufferArg::from_raw_parts(first_slot0.0.clone(), first_slot0.1) },
            unsafe { BufferArg::from_raw_parts(first_slot1.0.clone(), first_slot1.1) },
            unsafe { BufferArg::from_raw_parts(first_slot2.0.clone(), first_slot2.1) },
            unsafe { BufferArg::from_raw_parts(first_slot3.0.clone(), first_slot3.1) },
            unsafe { BufferArg::from_raw_parts(first_slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(second_slot0.0.clone(), second_slot0.1) },
            unsafe { BufferArg::from_raw_parts(second_slot1.0.clone(), second_slot1.1) },
            unsafe { BufferArg::from_raw_parts(second_slot2.0.clone(), second_slot2.1) },
            unsafe { BufferArg::from_raw_parts(second_slot3.0.clone(), second_slot3.1) },
            unsafe { BufferArg::from_raw_parts(second_slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(third_slot0.0.clone(), third_slot0.1) },
            unsafe { BufferArg::from_raw_parts(third_slot1.0.clone(), third_slot1.1) },
            unsafe { BufferArg::from_raw_parts(third_slot2.0.clone(), third_slot2.1) },
            unsafe { BufferArg::from_raw_parts(third_slot3.0.clone(), third_slot3.1) },
            unsafe { BufferArg::from_raw_parts(third_slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(output_first.clone(), count) },
            unsafe { BufferArg::from_raw_parts(output_second.clone(), count) },
            unsafe { BufferArg::from_raw_parts(output_third.clone(), count) },
        );
    }

    Ok((
        DeviceVec::from_handle(policy.id(), output_first, count),
        DeviceVec::from_handle(policy.id(), output_second, count),
        DeviceVec::from_handle(policy.id(), output_third, count),
    ))
}

macro_rules! define_device_expr_apply_selected_tuple_with_policy {
    (
        $fn_name:ident,
        $kernel:ident,
        ($first_ty:ident: $first_arg:ident:
            $first_bindings:ident,
            $first_slot0:ident, $first_slot1:ident, $first_slot2:ident, $first_slot3:ident,
            $first_offsets:ident => $first_output:ident
        $(,
            $ty:ident: $arg:ident:
                $bindings:ident,
                $slot0:ident, $slot1:ident, $slot2:ident, $slot3:ident,
                $offsets:ident => $output:ident
        )* $(,)?)
    ) => {
        pub(in crate::detail) fn $fn_name<$first_ty, $( $ty ),+>(
            policy: &crate::policy::CubePolicy<$first_ty::Runtime>,
            $first_arg: &$first_ty,
            $( $arg: &$ty, )+
            selected_rank: &select::SelectedRankControl,
            count: usize,
        ) -> Result<
            (
                DeviceVec<$first_ty::Runtime, $first_ty::Item>,
                $(
                    DeviceVec<$first_ty::Runtime, $ty::Item>,
                )+
            ),
            Error,
        >
        where
            $first_ty: KernelColumn + KernelColumnAt<S0>,
            $( $ty: KernelColumn<Runtime = $first_ty::Runtime> + KernelColumnAt<S0>, )+
            $first_ty::Runtime: Runtime,
            $first_ty::Item: CubePrimitive + CubeElement,
            $( $ty::Item: CubePrimitive + CubeElement, )+
            $first_ty::Expr: DeviceGpuExpr<$first_ty::Item>,
            $( $ty::Expr: DeviceGpuExpr<$ty::Item>, )+
        {
            $first_arg.validate()?;
            $( $arg.validate()?; )+
            $( ensure_same_len($first_arg.len(), $arg.len())?; )+
            ensure_same_len($first_arg.len(), selected_rank.len)?;
            if selected_rank.len == 0 || count == 0 {
                return Ok((
                    policy.empty_device_vec(),
                    $(
                        {
                            let _ = stringify!($ty);
                            policy.empty_device_vec()
                        },
                    )+
                ));
            }

            let client = policy.client();
            let $first_output = client.empty(count * std::mem::size_of::<$first_ty::Item>());
            $(
                let $output = client.empty(count * std::mem::size_of::<$ty::Item>());
            )+
            let $first_bindings = $first_arg.stage(policy)?;
            $(
                let $bindings = $arg.stage(policy)?;
            )+
            let $first_slot0 = $first_bindings.slot_or_first(0);
            let $first_slot1 = $first_bindings.slot_or_first(1);
            let $first_slot2 = $first_bindings.slot_or_first(2);
            let $first_slot3 = $first_bindings.slot_or_first(3);
            $(
                let $slot0 = $bindings.slot_or_first(0);
                let $slot1 = $bindings.slot_or_first(1);
                let $slot2 = $bindings.slot_or_first(2);
                let $slot3 = $bindings.slot_or_first(3);
            )+
            let $first_offsets = $first_bindings.slot_offsets_handle(client)?;
            $(
                let $offsets = $bindings.slot_offsets_handle(client)?;
            )+
            let block_count_u32 = api_expr_block_count(selected_rank.len)?;

            unsafe {
                $kernel::launch_unchecked::<
                    $first_ty::Item,
                    $( $ty::Item, )+
                    $first_ty::Expr,
                    $( $ty::Expr, )+
                    $first_ty::Runtime,
                >(
                    client,
                    CubeCount::Static(block_count_u32, 1, 1),
                    CubeDim::new_1d(BLOCK_API_EXPR_SIZE),
                    unsafe { BufferArg::from_raw_parts(selected_rank.flag.clone(), selected_rank.len) },
                    unsafe { BufferArg::from_raw_parts(selected_rank.position.clone(), selected_rank.len) },
                    unsafe { BufferArg::from_raw_parts($first_slot0.0.clone(), $first_slot0.1) },
                    unsafe { BufferArg::from_raw_parts($first_slot1.0.clone(), $first_slot1.1) },
                    unsafe { BufferArg::from_raw_parts($first_slot2.0.clone(), $first_slot2.1) },
                    unsafe { BufferArg::from_raw_parts($first_slot3.0.clone(), $first_slot3.1) },
                    unsafe { BufferArg::from_raw_parts($first_offsets.clone(), 4) },
                    $(
                        unsafe { BufferArg::from_raw_parts($slot0.0.clone(), $slot0.1) },
                        unsafe { BufferArg::from_raw_parts($slot1.0.clone(), $slot1.1) },
                        unsafe { BufferArg::from_raw_parts($slot2.0.clone(), $slot2.1) },
                        unsafe { BufferArg::from_raw_parts($slot3.0.clone(), $slot3.1) },
                        unsafe { BufferArg::from_raw_parts($offsets.clone(), 4) },
                    )+
                    unsafe { BufferArg::from_raw_parts($first_output.clone(), count) },
                    $(
                        unsafe { BufferArg::from_raw_parts($output.clone(), count) },
                    )+
                );
            }

            Ok((
                DeviceVec::from_handle(policy.id(), $first_output, count),
                $(
                    DeviceVec::from_handle(policy.id(), $output, count),
                )+
            ))
        }
    };
}

define_device_expr_apply_selected_tuple_with_policy!(
    device_expr_apply_selected4_with_policy,
    selected_apply_tuple4_device_expr_kernel,
    (A: a: a_bindings, a0, a1, a2, a3, a_offsets => out_a,
     B: b: b_bindings, b0, b1, b2, b3, b_offsets => out_b,
     C: c: c_bindings, c0, c1, c2, c3, c_offsets => out_c,
     D: d: d_bindings, d0, d1, d2, d3, d_offsets => out_d)
);
define_device_expr_apply_selected_tuple_with_policy!(
    device_expr_apply_selected5_with_policy,
    selected_apply_tuple5_device_expr_kernel,
    (A: a: a_bindings, a0, a1, a2, a3, a_offsets => out_a,
     B: b: b_bindings, b0, b1, b2, b3, b_offsets => out_b,
     C: c: c_bindings, c0, c1, c2, c3, c_offsets => out_c,
     D: d: d_bindings, d0, d1, d2, d3, d_offsets => out_d,
     E: e: e_bindings, e0, e1, e2, e3, e_offsets => out_e)
);
define_device_expr_apply_selected_tuple_with_policy!(
    device_expr_apply_selected6_with_policy,
    selected_apply_tuple6_device_expr_kernel,
    (A: a: a_bindings, a0, a1, a2, a3, a_offsets => out_a,
     B: b: b_bindings, b0, b1, b2, b3, b_offsets => out_b,
     C: c: c_bindings, c0, c1, c2, c3, c_offsets => out_c,
     D: d: d_bindings, d0, d1, d2, d3, d_offsets => out_d,
     E: e: e_bindings, e0, e1, e2, e3, e_offsets => out_e,
     F: f: f_bindings, f0, f1, f2, f3, f_offsets => out_f)
);
define_device_expr_apply_selected_tuple_with_policy!(
    device_expr_apply_selected7_with_policy,
    selected_apply_tuple7_device_expr_kernel,
    (A: a: a_bindings, a0, a1, a2, a3, a_offsets => out_a,
     B: b: b_bindings, b0, b1, b2, b3, b_offsets => out_b,
     C: c: c_bindings, c0, c1, c2, c3, c_offsets => out_c,
     D: d: d_bindings, d0, d1, d2, d3, d_offsets => out_d,
     E: e: e_bindings, e0, e1, e2, e3, e_offsets => out_e,
     F: f: f_bindings, f0, f1, f2, f3, f_offsets => out_f,
     G: g: g_bindings, g0, g1, g2, g3, g_offsets => out_g)
);

pub(in crate::detail) fn device_expr_compact_split_with_split_with_policy<ExprSource>(
    policy: &crate::policy::CubePolicy<ExprSource::Runtime>,
    expr: &ExprSource,
    control: &select::SplitRankControl,
    selected_count: usize,
    rejected_count: usize,
) -> Result<
    (
        DeviceVec<ExprSource::Runtime, ExprSource::Item>,
        DeviceVec<ExprSource::Runtime, ExprSource::Item>,
    ),
    Error,
>
where
    ExprSource: KernelColumn + KernelColumnAt<S0>,
    ExprSource::Runtime: Runtime,
    ExprSource::Item: CubePrimitive + CubeElement,
    ExprSource::Expr: DeviceGpuExpr<ExprSource::Item>,
{
    expr.validate()?;
    ensure_same_len(expr.len(), control.len)?;
    if control.len == 0 {
        return Ok((policy.empty_device_vec(), policy.empty_device_vec()));
    }

    let client = policy.client();
    let selected_handle = if selected_count == 0 {
        policy.empty_handle()
    } else {
        client.empty(selected_count * std::mem::size_of::<ExprSource::Item>())
    };
    let rejected_handle = if rejected_count == 0 {
        policy.empty_handle()
    } else {
        client.empty(rejected_count * std::mem::size_of::<ExprSource::Item>())
    };
    let bindings = expr.stage(policy)?;
    let slot_offsets = bindings.slot_offsets_handle(client)?;
    let slot0 = bindings.slots.first().unwrap();
    let slot1 = bindings.slots.get(1).unwrap_or(slot0);
    let slot2 = bindings.slots.get(2).unwrap_or(slot0);
    let slot3 = bindings.slots.get(3).unwrap_or(slot0);
    let block_count_u32 = api_expr_block_count(control.len)?;

    unsafe {
        compact_split_scatter_device_expr_kernel::launch_unchecked::<
            ExprSource::Item,
            ExprSource::Expr,
            ExprSource::Runtime,
        >(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_API_EXPR_SIZE),
            unsafe { BufferArg::from_raw_parts(control.flag.clone(), control.len) },
            unsafe { BufferArg::from_raw_parts(control.position.clone(), control.len) },
            unsafe { BufferArg::from_raw_parts(slot0.0.clone(), slot0.1) },
            unsafe { BufferArg::from_raw_parts(slot1.0.clone(), slot1.1) },
            unsafe { BufferArg::from_raw_parts(slot2.0.clone(), slot2.1) },
            unsafe { BufferArg::from_raw_parts(slot3.0.clone(), slot3.1) },
            unsafe { BufferArg::from_raw_parts(slot_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(selected_handle.clone(), selected_count) },
            unsafe { BufferArg::from_raw_parts(rejected_handle.clone(), rejected_count) },
        );
    }

    Ok((
        DeviceVec::from_handle(policy.id(), selected_handle, selected_count),
        DeviceVec::from_handle(policy.id(), rejected_handle, rejected_count),
    ))
}

macro_rules! define_device_expr_apply_split_tuple_with_policy {
    (
        $fn_name:ident,
        $kernel:ident,
        ($first_ty:ident: $first_arg:ident:
            $first_bindings:ident,
            $first_slot0:ident, $first_slot1:ident, $first_slot2:ident, $first_slot3:ident,
            $first_offsets:ident => $first_selected:ident, $first_rejected:ident
        $(,
            $ty:ident: $arg:ident:
                $bindings:ident,
                $slot0:ident, $slot1:ident, $slot2:ident, $slot3:ident,
                $offsets:ident => $selected:ident, $rejected:ident
        )* $(,)?)
    ) => {
        pub(in crate::detail) fn $fn_name<$first_ty, $( $ty ),+>(
            policy: &crate::policy::CubePolicy<$first_ty::Runtime>,
            $first_arg: &$first_ty,
            $( $arg: &$ty, )+
            control: &select::SplitRankControl,
            selected_count: usize,
            rejected_count: usize,
        ) -> Result<
            (
                (
                    DeviceVec<$first_ty::Runtime, $first_ty::Item>,
                    $(
                        DeviceVec<$first_ty::Runtime, $ty::Item>,
                    )+
                ),
                (
                    DeviceVec<$first_ty::Runtime, $first_ty::Item>,
                    $(
                        DeviceVec<$first_ty::Runtime, $ty::Item>,
                    )+
                ),
            ),
            Error,
        >
        where
            $first_ty: KernelColumn + KernelColumnAt<S0>,
            $( $ty: KernelColumn<Runtime = $first_ty::Runtime> + KernelColumnAt<S0>, )+
            $first_ty::Runtime: Runtime,
            $first_ty::Item: CubePrimitive + CubeElement,
            $( $ty::Item: CubePrimitive + CubeElement, )+
            $first_ty::Expr: DeviceGpuExpr<$first_ty::Item>,
            $( $ty::Expr: DeviceGpuExpr<$ty::Item>, )+
        {
            $first_arg.validate()?;
            $( $arg.validate()?; )+
            $( ensure_same_len($first_arg.len(), $arg.len())?; )+
            ensure_same_len($first_arg.len(), control.len)?;
            if control.len == 0 {
                return Ok((
                    (
                        policy.empty_device_vec(),
                        $(
                            {
                                let _ = stringify!($ty);
                                policy.empty_device_vec()
                            },
                        )+
                    ),
                    (
                        policy.empty_device_vec(),
                        $(
                            {
                                let _ = stringify!($ty);
                                policy.empty_device_vec()
                            },
                        )+
                    ),
                ));
            }

            let client = policy.client();
            let $first_selected = if selected_count == 0 {
                policy.empty_handle()
            } else {
                client.empty(selected_count * std::mem::size_of::<$first_ty::Item>())
            };
            let $first_rejected = if rejected_count == 0 {
                policy.empty_handle()
            } else {
                client.empty(rejected_count * std::mem::size_of::<$first_ty::Item>())
            };
            $(
                let $selected = if selected_count == 0 {
                    policy.empty_handle()
                } else {
                    client.empty(selected_count * std::mem::size_of::<$ty::Item>())
                };
                let $rejected = if rejected_count == 0 {
                    policy.empty_handle()
                } else {
                    client.empty(rejected_count * std::mem::size_of::<$ty::Item>())
                };
            )+
            let $first_bindings = $first_arg.stage(policy)?;
            $(
                let $bindings = $arg.stage(policy)?;
            )+
            let $first_slot0 = $first_bindings.slot_or_first(0);
            let $first_slot1 = $first_bindings.slot_or_first(1);
            let $first_slot2 = $first_bindings.slot_or_first(2);
            let $first_slot3 = $first_bindings.slot_or_first(3);
            $(
                let $slot0 = $bindings.slot_or_first(0);
                let $slot1 = $bindings.slot_or_first(1);
                let $slot2 = $bindings.slot_or_first(2);
                let $slot3 = $bindings.slot_or_first(3);
            )+
            let $first_offsets = $first_bindings.slot_offsets_handle(client)?;
            $(
                let $offsets = $bindings.slot_offsets_handle(client)?;
            )+
            let block_count_u32 = api_expr_block_count(control.len)?;

            unsafe {
                $kernel::launch_unchecked::<
                    $first_ty::Item,
                    $( $ty::Item, )+
                    $first_ty::Expr,
                    $( $ty::Expr, )+
                    $first_ty::Runtime,
                >(
                    client,
                    CubeCount::Static(block_count_u32, 1, 1),
                    CubeDim::new_1d(BLOCK_API_EXPR_SIZE),
                    unsafe { BufferArg::from_raw_parts(control.flag.clone(), control.len) },
                    unsafe { BufferArg::from_raw_parts(control.position.clone(), control.len) },
                    unsafe { BufferArg::from_raw_parts($first_slot0.0.clone(), $first_slot0.1) },
                    unsafe { BufferArg::from_raw_parts($first_slot1.0.clone(), $first_slot1.1) },
                    unsafe { BufferArg::from_raw_parts($first_slot2.0.clone(), $first_slot2.1) },
                    unsafe { BufferArg::from_raw_parts($first_slot3.0.clone(), $first_slot3.1) },
                    unsafe { BufferArg::from_raw_parts($first_offsets.clone(), 4) },
                    $(
                        unsafe { BufferArg::from_raw_parts($slot0.0.clone(), $slot0.1) },
                        unsafe { BufferArg::from_raw_parts($slot1.0.clone(), $slot1.1) },
                        unsafe { BufferArg::from_raw_parts($slot2.0.clone(), $slot2.1) },
                        unsafe { BufferArg::from_raw_parts($slot3.0.clone(), $slot3.1) },
                        unsafe { BufferArg::from_raw_parts($offsets.clone(), 4) },
                    )+
                    unsafe { BufferArg::from_raw_parts($first_selected.clone(), selected_count) },
                    unsafe { BufferArg::from_raw_parts($first_rejected.clone(), rejected_count) },
                    $(
                        unsafe { BufferArg::from_raw_parts($selected.clone(), selected_count) },
                        unsafe { BufferArg::from_raw_parts($rejected.clone(), rejected_count) },
                    )+
                );
            }

            Ok((
                (
                    DeviceVec::from_handle(policy.id(), $first_selected, selected_count),
                    $(
                        DeviceVec::from_handle(policy.id(), $selected, selected_count),
                    )+
                ),
                (
                    DeviceVec::from_handle(policy.id(), $first_rejected, rejected_count),
                    $(
                        DeviceVec::from_handle(policy.id(), $rejected, rejected_count),
                    )+
                ),
            ))
        }
    };
}

define_device_expr_apply_split_tuple_with_policy!(
    device_expr_apply_split2_with_policy,
    split_apply_tuple2_device_expr_kernel,
    (A: a: a_bindings, a0, a1, a2, a3, a_offsets => selected_a, rejected_a,
     B: b: b_bindings, b0, b1, b2, b3, b_offsets => selected_b, rejected_b)
);
define_device_expr_apply_split_tuple_with_policy!(
    device_expr_apply_split3_with_policy,
    split_apply_tuple3_device_expr_kernel,
    (A: a: a_bindings, a0, a1, a2, a3, a_offsets => selected_a, rejected_a,
     B: b: b_bindings, b0, b1, b2, b3, b_offsets => selected_b, rejected_b,
     C: c: c_bindings, c0, c1, c2, c3, c_offsets => selected_c, rejected_c)
);
define_device_expr_apply_split_tuple_with_policy!(
    device_expr_apply_split4_with_policy,
    split_apply_tuple4_device_expr_kernel,
    (A: a: a_bindings, a0, a1, a2, a3, a_offsets => selected_a, rejected_a,
     B: b: b_bindings, b0, b1, b2, b3, b_offsets => selected_b, rejected_b,
     C: c: c_bindings, c0, c1, c2, c3, c_offsets => selected_c, rejected_c,
     D: d: d_bindings, d0, d1, d2, d3, d_offsets => selected_d, rejected_d)
);
define_device_expr_apply_split_tuple_with_policy!(
    device_expr_apply_split5_with_policy,
    split_apply_tuple5_device_expr_kernel,
    (A: a: a_bindings, a0, a1, a2, a3, a_offsets => selected_a, rejected_a,
     B: b: b_bindings, b0, b1, b2, b3, b_offsets => selected_b, rejected_b,
     C: c: c_bindings, c0, c1, c2, c3, c_offsets => selected_c, rejected_c,
     D: d: d_bindings, d0, d1, d2, d3, d_offsets => selected_d, rejected_d,
     E: e: e_bindings, e0, e1, e2, e3, e_offsets => selected_e, rejected_e)
);
define_device_expr_apply_split_tuple_with_policy!(
    device_expr_apply_split6_with_policy,
    split_apply_tuple6_device_expr_kernel,
    (A: a: a_bindings, a0, a1, a2, a3, a_offsets => selected_a, rejected_a,
     B: b: b_bindings, b0, b1, b2, b3, b_offsets => selected_b, rejected_b,
     C: c: c_bindings, c0, c1, c2, c3, c_offsets => selected_c, rejected_c,
     D: d: d_bindings, d0, d1, d2, d3, d_offsets => selected_d, rejected_d,
     E: e: e_bindings, e0, e1, e2, e3, e_offsets => selected_e, rejected_e,
     F: f: f_bindings, f0, f1, f2, f3, f_offsets => selected_f, rejected_f)
);

pub(in crate::detail) fn device_expr_apply_split7_with_policy<A, B, C, D, E, F, G>(
    policy: &crate::policy::CubePolicy<A::Runtime>,
    a: &A,
    b: &B,
    c: &C,
    d: &D,
    e: &E,
    f: &F,
    g: &G,
    control: &select::SplitRankControl,
    selected_count: usize,
    rejected_count: usize,
) -> Result<
    (
        (
            DeviceVec<A::Runtime, A::Item>,
            DeviceVec<A::Runtime, B::Item>,
            DeviceVec<A::Runtime, C::Item>,
            DeviceVec<A::Runtime, D::Item>,
            DeviceVec<A::Runtime, E::Item>,
            DeviceVec<A::Runtime, F::Item>,
            DeviceVec<A::Runtime, G::Item>,
        ),
        (
            DeviceVec<A::Runtime, A::Item>,
            DeviceVec<A::Runtime, B::Item>,
            DeviceVec<A::Runtime, C::Item>,
            DeviceVec<A::Runtime, D::Item>,
            DeviceVec<A::Runtime, E::Item>,
            DeviceVec<A::Runtime, F::Item>,
            DeviceVec<A::Runtime, G::Item>,
        ),
    ),
    Error,
>
where
    A: KernelColumn + KernelColumnAt<S0>,
    B: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    D: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    E: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    F: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    G: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    A::Runtime: Runtime,
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
{
    let (
        (selected_a, selected_b, selected_c, selected_d),
        (rejected_a, rejected_b, rejected_c, rejected_d),
    ) = device_expr_apply_split4_with_policy(
        policy,
        a,
        b,
        c,
        d,
        control,
        selected_count,
        rejected_count,
    )?;
    let ((selected_e, selected_f, selected_g), (rejected_e, rejected_f, rejected_g)) =
        device_expr_apply_split3_with_policy(
            policy,
            e,
            f,
            g,
            control,
            selected_count,
            rejected_count,
        )?;

    Ok((
        (
            selected_a, selected_b, selected_c, selected_d, selected_e, selected_f, selected_g,
        ),
        (
            rejected_a, rejected_b, rejected_c, rejected_d, rejected_e, rejected_f, rejected_g,
        ),
    ))
}

pub(in crate::detail) fn device_expr_count_if_with_policy<ExprSource, Pred>(
    policy: &crate::policy::CubePolicy<ExprSource::Runtime>,
    expr: &ExprSource,
    invert: bool,
) -> Result<MIndex, Error>
where
    ExprSource: KernelColumn + KernelColumnAt<S0>,
    ExprSource::Runtime: Runtime,
    ExprSource::Item: CubePrimitive + CubeElement,
    ExprSource::Expr: DeviceGpuExpr<ExprSource::Item>,
    Pred: PredicateOp<ExprSource::Item>,
{
    let selected_rank =
        device_expr_selected_rank_with_policy::<ExprSource, Pred>(policy, expr, invert)?;
    if selected_rank.len == 0 {
        return Ok(0);
    }

    Ok(read_u32_scalar::<ExprSource::Runtime>(
        policy.client(),
        selected_rank.count.clone(),
    )?)
}

pub(in crate::detail) fn device_expr_find_if_with_policy<ExprSource, Pred>(
    policy: &crate::policy::CubePolicy<ExprSource::Runtime>,
    expr: &ExprSource,
    invert: bool,
) -> Result<Option<MIndex>, Error>
where
    ExprSource: KernelColumn + KernelColumnAt<S0>,
    ExprSource::Runtime: Runtime,
    ExprSource::Item: CubePrimitive + CubeElement,
    ExprSource::Expr: GpuExpr<ExprSource::Item>,
    Pred: PredicateOp<ExprSource::Item>,
{
    let mask = device_expr_selection_flags_with_policy::<ExprSource, Pred>(policy, expr, invert)?;
    primitive_search::first_flag(policy, mask.flag.clone(), mask.len, expr.len())
}

pub(in crate::detail) fn device_expr_selected_rank_with_policy<ExprSource, Pred>(
    policy: &crate::policy::CubePolicy<ExprSource::Runtime>,
    expr: &ExprSource,
    invert: bool,
) -> Result<select::SelectedRankControl, Error>
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
        return Ok(select::SelectedRankControl::empty(client));
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

    select::selected_rank_from_flags(policy, len, len_u32, flag_handle)
}

pub(in crate::detail) fn device_expr_selection_flags_with_policy<ExprSource, Pred>(
    policy: &crate::policy::CubePolicy<ExprSource::Runtime>,
    expr: &ExprSource,
    invert: bool,
) -> Result<select::MaskControl, Error>
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
        return Ok(select::MaskControl::empty(client));
    }

    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let invert_handle =
        client.create_from_slice(u32::as_bytes(&[if invert { 1_u32 } else { 0_u32 }]));
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

    Ok(select::MaskControl::from_flags(flag_handle, len, len_u32))
}
