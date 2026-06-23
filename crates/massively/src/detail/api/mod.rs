mod gather;
mod memory;
mod ordering;
mod reduce;
mod scan;
mod scatter;
mod search;
mod selection;
mod sequence;

pub use gather::{gather, gather_if};
pub use memory::{
    MItemStorage, MaterializeOutput, TransformSoA2Output, TransformSoA3Output, TransformUnaryOutput,
};
pub use ordering::{
    merge, merge_by_key, reverse, set_difference, set_intersection, set_union, sort, sort_by_key,
};
pub use reduce::{reduce, reduce_by_key};
pub use scan::{
    adjacent_difference, exclusive_scan, exclusive_scan_by_key, inclusive_scan,
    inclusive_scan_by_key,
};
pub use scatter::{scatter, scatter_if};
pub use search::{
    adjacent_find, equal, equal_range, find_first_of, is_sorted, is_sorted_until,
    lexicographical_compare, lower_bound, max_element, min_element, minmax_element, mismatch,
    upper_bound,
};
pub use selection::{
    all_of, any_of, copy_if, count_if, find_if, is_partitioned, none_of, partition, remove_if,
};
pub use sequence::{replace_if, unique, unique_by_key};

use crate::{
    detail::op::kernel::{BinaryOp, BinaryPredicateOp, PredicateOp},
    device::{DeviceVec, KernelColumn, KernelColumnAt, S0, SoAView2, SoAView3},
    error::{Error, ensure_same_len},
    expr::{DeviceGpuExpr, GpuExpr},
    kernels::*,
    primitives::{
        scan as primitive_scan, scan::read_u32_scalar, search as primitive_search, select,
    },
};
use cubecl::prelude::*;

const BLOCK_API_EXPR_SIZE: u32 = 256;

mod tuple_adapter;
pub use tuple_adapter::{Tuple1BinaryOp, Tuple1Less, Tuple1PredicateOp};

fn api_expr_block_count(len: usize) -> Result<u32, Error> {
    let block_count = len.div_ceil(BLOCK_API_EXPR_SIZE as usize);
    u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })
}

fn offset_handle<R: Runtime>(
    client: &ComputeClient<R>,
    offset: usize,
) -> Result<cubecl::server::Handle, Error> {
    let offset = u32::try_from(offset).map_err(|_| Error::LengthTooLarge { len: offset })?;
    Ok(client.create_from_slice(u32::as_bytes(&[offset])))
}

pub(super) fn device_expr_collect_with_policy<ExprSource>(
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
    let block_count_u32 = api_expr_block_count(len)?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let slot_offsets = bindings.slot_offsets_handle(client)?;

    unsafe {
        device_collect_expr_block_kernel::launch_unchecked::<
            ExprSource::Item,
            ExprSource::Expr,
            ExprSource::Runtime,
        >(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
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

pub(super) fn device_expr_reverse_collect<ExprSource>(
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
    let block_count_u32 = api_expr_block_count(len)?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let slot_offsets = bindings.slot_offsets_handle(client)?;

    unsafe {
        device_collect_expr_reverse_block_kernel::launch_unchecked::<
            ExprSource::Item,
            ExprSource::Expr,
            ExprSource::Runtime,
        >(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
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

pub(super) fn device_expr_gather_with_policy<InputSource, IndexSource>(
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

pub(super) fn device_expr_scatter_with_policy<ValueSource, IndexSource, InitialSource>(
    policy: &crate::policy::CubePolicy<ValueSource::Runtime>,
    values: &ValueSource,
    indices: &IndexSource,
    initial: &InitialSource,
) -> Result<DeviceVec<ValueSource::Runtime, ValueSource::Item>, Error>
where
    ValueSource: KernelColumn + KernelColumnAt<S0>,
    ValueSource::Runtime: Runtime,
    IndexSource: KernelColumn<Runtime = ValueSource::Runtime, Item = u32> + KernelColumnAt<S0>,
    InitialSource:
        KernelColumn<Runtime = ValueSource::Runtime, Item = ValueSource::Item> + KernelColumnAt<S0>,
    ValueSource::Item: CubePrimitive + CubeElement,
    ValueSource::Expr: GpuExpr<ValueSource::Item>,
    IndexSource::Expr: GpuExpr<u32>,
    InitialSource::Expr: DeviceGpuExpr<ValueSource::Item>,
{
    values.validate()?;
    indices.validate()?;
    ensure_same_len(values.len(), indices.len())?;

    let output = device_expr_collect_with_policy(policy, initial)?;
    let len = values.len();
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    if len == 0 {
        return Ok(output);
    }

    let client = policy.client();
    let block_count_u32 = api_expr_block_count(len)?;
    let value_bindings = values.stage(policy)?;
    let index_bindings = indices.stage(policy)?;
    let len_values = [len_u32];
    let len_handle = client.create_from_slice(u32::as_bytes(&len_values));
    let value_offset = offset_handle(client, value_bindings.input_offset)?;
    let value_rhs_offset = offset_handle(client, value_bindings.rhs_offset)?;
    let index_offset = offset_handle(client, index_bindings.input_offset)?;
    let index_rhs_offset = offset_handle(client, index_bindings.rhs_offset)?;

    unsafe {
        scatter_expr_kernel::launch_unchecked::<
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
            unsafe { BufferArg::from_raw_parts(output.handle.clone(), output.len) },
        );
    }

    Ok(output)
}

pub(super) fn device_expr_copy_if_with_policy<ExprSource, Pred>(
    policy: &crate::policy::CubePolicy<ExprSource::Runtime>,
    expr: &ExprSource,
    invert: bool,
) -> Result<DeviceVec<ExprSource::Runtime, ExprSource::Item>, Error>
where
    ExprSource: KernelColumn + KernelColumnAt<S0>,
    ExprSource::Runtime: Runtime,
    ExprSource::Item: CubePrimitive + CubeElement,
    ExprSource::Expr: GpuExpr<ExprSource::Item>,
    Pred: PredicateOp<ExprSource::Item>,
{
    let handles =
        device_expr_selection_handles_with_policy::<ExprSource, Pred>(policy, expr, invert)?;
    select::compact::<ExprSource::Runtime, ExprSource::Item>(policy, handles)
}

pub(super) fn device_expr_compact_with_flags_with_policy<ExprSource>(
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

pub(super) fn device_expr_compact_with_selection_with_policy<ExprSource>(
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

pub(super) fn device_expr_compact_rejected_with_selection_with_policy<ExprSource>(
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

pub(super) fn device_expr_count_if_with_policy<ExprSource, Pred>(
    policy: &crate::policy::CubePolicy<ExprSource::Runtime>,
    expr: &ExprSource,
    invert: bool,
) -> Result<usize, Error>
where
    ExprSource: KernelColumn + KernelColumnAt<S0>,
    ExprSource::Runtime: Runtime,
    ExprSource::Item: CubePrimitive + CubeElement,
    ExprSource::Expr: GpuExpr<ExprSource::Item>,
    Pred: PredicateOp<ExprSource::Item>,
{
    let handles =
        device_expr_selection_handles_with_policy::<ExprSource, Pred>(policy, expr, invert)?;
    if handles.len == 0 {
        return Ok(0);
    }

    Ok(read_u32_scalar::<ExprSource::Runtime>(policy.client(), handles.count)? as usize)
}

pub(super) fn device_expr_find_if_with_policy<ExprSource, Pred>(
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

pub(super) fn device_expr_selection_handles_with_policy<ExprSource, Pred>(
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
    let input_offset = offset_handle(client, bindings.input_offset)?;
    let rhs_offset = offset_handle(client, bindings.rhs_offset)?;
    let value_handle = expr.staged_value_handle(&bindings);
    let block_count_u32 = api_expr_block_count(len)?;

    let value_handle = if let Some(value_handle) = value_handle {
        unsafe {
            copy_if_expr_flag_only_kernel::launch_unchecked::<
                ExprSource::Item,
                ExprSource::Expr,
                Pred,
                ExprSource::Runtime,
            >(
                client,
                CubeCount::Static(block_count_u32, 1, 1),
                CubeDim::new_1d(BLOCK_API_EXPR_SIZE),
                unsafe { BufferArg::from_raw_parts(bindings.input.clone(), bindings.input_len) },
                unsafe { BufferArg::from_raw_parts(input_offset.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(bindings.rhs.clone(), bindings.rhs_len) },
                unsafe { BufferArg::from_raw_parts(rhs_offset.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(invert_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(flag_handle.clone(), len) },
            );
        }
        value_handle
    } else {
        let value_handle = client.empty(len * std::mem::size_of::<ExprSource::Item>());
        unsafe {
            copy_if_expr_flags_kernel::launch_unchecked::<
                ExprSource::Item,
                ExprSource::Expr,
                Pred,
                ExprSource::Runtime,
            >(
                client,
                CubeCount::Static(block_count_u32, 1, 1),
                CubeDim::new_1d(BLOCK_API_EXPR_SIZE),
                unsafe { BufferArg::from_raw_parts(bindings.input.clone(), bindings.input_len) },
                unsafe { BufferArg::from_raw_parts(input_offset.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(bindings.rhs.clone(), bindings.rhs_len) },
                unsafe { BufferArg::from_raw_parts(rhs_offset.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(invert_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(flag_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(value_handle.clone(), len) },
            );
        }
        value_handle
    };

    select::handles_from_flags(policy, len, len_u32, flag_handle, value_handle)
}

mod selection_control;
pub use selection_control::{PrecomputedSelection, SelectionStencil};

pub(super) fn device_expr_adjacent_difference_with_policy<ExprSource, Op>(
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

pub(super) fn device_expr_minmax_element_with_policy<ExprSource, Less>(
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

pub(super) fn device_expr_inclusive_scan_by_key_expr_keys_with_policy<
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

pub(super) fn device_expr_exclusive_scan_by_key_expr_keys_with_policy<
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
