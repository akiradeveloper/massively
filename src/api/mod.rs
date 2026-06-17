mod gather_scatter;
mod memory;
mod mutation;
mod ordering;
mod reduce;
mod scan;
mod search;
mod selection;

pub use gather_scatter::{gather, gather_if, scatter, scatter_if};
pub use memory::{transform, zip, zip3, zip4, zip5, zip6, zip7, zip8, zip9, zip10, zip11, zip12};
pub use mutation::{replace_if, unique, unique_by_key};
pub use ordering::{
    merge, merge_by_key, reverse, set_difference, set_intersection, set_union, sort, sort_by_key,
    stable_sort, stable_sort_by_key,
};
pub use reduce::{inner_product, reduce, reduce_by_key};
pub use scan::{
    adjacent_difference, exclusive_scan, exclusive_scan_by_key, inclusive_scan,
    inclusive_scan_by_key,
};
pub use search::{
    adjacent_find, equal, equal_range, find_first_of, is_sorted, is_sorted_until,
    lexicographical_compare, lower_bound, max_element, min_element, minmax_element, mismatch,
    upper_bound,
};
pub use selection::{
    all_of, any_of, copy_if, count_if, find_if, is_partitioned, none_of, partition, remove_if,
};

use crate::{
    device::{DeviceVec, KernelColumn, KernelColumnAt, S0},
    error::{Error, ensure_same_len},
    expr::{DeviceGpuExpr, GpuExpr, Input},
    kernels::*,
    op::{BinaryOp, BinaryPredicateOp, GpuOp, PredicateOp},
    primitives::{
        reduce as primitive_reduce, scan as primitive_scan, scan::read_u32_scalar,
        search as primitive_search, select,
    },
};
use cubecl::prelude::*;

const BLOCK_API_EXPR_SIZE: u32 = 256;

fn api_expr_block_count(len: usize) -> Result<u32, Error> {
    let block_count = len.div_ceil(BLOCK_API_EXPR_SIZE as usize);
    u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })
}

pub(super) fn device_expr_collect<ExprSource>(
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
    let client = expr.policy().client();
    let output_handle = client.empty(len * std::mem::size_of::<ExprSource::Item>());

    if len != 0 {
        let bindings = expr.stage()?;
        let slot0 = bindings.slots.first().unwrap();
        let slot1 = bindings.slots.get(1).unwrap_or(slot0);
        let slot2 = bindings.slots.get(2).unwrap_or(slot0);
        let slot3 = bindings.slots.get(3).unwrap_or(slot0);
        let block_count_u32 = api_expr_block_count(len)?;
        let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));

        unsafe {
            device_collect_expr_block_kernel::launch_unchecked::<
                ExprSource::Item,
                ExprSource::Expr,
                ExprSource::Runtime,
            >(
                client,
                CubeCount::Static(block_count_u32, 1, 1),
                CubeDim::new_1d(BLOCK_API_EXPR_SIZE),
                ArrayArg::from_raw_parts::<ExprSource::Item>(&output_handle, len, 1),
                ArrayArg::from_raw_parts::<ExprSource::Item>(&slot0.0, slot0.1, 1),
                ArrayArg::from_raw_parts::<ExprSource::Item>(&slot1.0, slot1.1, 1),
                ArrayArg::from_raw_parts::<ExprSource::Item>(&slot2.0, slot2.1, 1),
                ArrayArg::from_raw_parts::<ExprSource::Item>(&slot3.0, slot3.1, 1),
                ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
            )
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
        }
    }

    Ok(DeviceVec::from_handle(
        expr.policy().clone(),
        output_handle,
        len,
    ))
}

pub(super) fn device_expr_gather<InputSource, IndexSource>(
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
    let client = input.policy().client();
    let output_handle = client.empty(len * std::mem::size_of::<InputSource::Item>());

    if len != 0 {
        let block_count_u32 = api_expr_block_count(len)?;
        let input_bindings = input.stage()?;
        let index_bindings = indices.stage()?;
        let dummy_indices = [0_u32];
        let dummy_index_handle = client.create_from_slice(u32::as_bytes(&dummy_indices));

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
                ArrayArg::from_raw_parts::<InputSource::Item>(&output_handle, len, 1),
                ArrayArg::from_raw_parts::<InputSource::Item>(
                    &input_bindings.input,
                    input_bindings.input_len,
                    1,
                ),
                ArrayArg::from_raw_parts::<u32>(&dummy_index_handle, dummy_indices.len(), 1),
                ArrayArg::from_raw_parts::<InputSource::Item>(
                    &input_bindings.rhs,
                    input_bindings.rhs_len,
                    1,
                ),
                ArrayArg::from_raw_parts::<u32>(&dummy_index_handle, dummy_indices.len(), 1),
                ArrayArg::from_raw_parts::<u32>(&index_bindings.input, index_bindings.input_len, 1),
                ArrayArg::from_raw_parts::<u32>(&dummy_index_handle, dummy_indices.len(), 1),
                ArrayArg::from_raw_parts::<u32>(&index_bindings.rhs, index_bindings.rhs_len, 1),
                ArrayArg::from_raw_parts::<u32>(&dummy_index_handle, dummy_indices.len(), 1),
            )
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
        }
    }

    Ok(DeviceVec::from_handle(
        input.policy().clone(),
        output_handle,
        len,
    ))
}

pub(super) fn device_expr_scatter<ValueSource, IndexSource, InitialSource>(
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

    let output = device_expr_collect(initial)?;
    let len = values.len();
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    if len == 0 {
        return Ok(output);
    }

    let client = values.policy().client();
    let block_count_u32 = api_expr_block_count(len)?;
    let value_bindings = values.stage()?;
    let index_bindings = indices.stage()?;
    let len_values = [len_u32];
    let len_handle = client.create_from_slice(u32::as_bytes(&len_values));
    let dummy_indices = [0_u32];
    let dummy_index_handle = client.create_from_slice(u32::as_bytes(&dummy_indices));

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
            ArrayArg::from_raw_parts::<ValueSource::Item>(
                &value_bindings.input,
                value_bindings.input_len,
                1,
            ),
            ArrayArg::from_raw_parts::<u32>(&dummy_index_handle, dummy_indices.len(), 1),
            ArrayArg::from_raw_parts::<ValueSource::Item>(
                &value_bindings.rhs,
                value_bindings.rhs_len,
                1,
            ),
            ArrayArg::from_raw_parts::<u32>(&dummy_index_handle, dummy_indices.len(), 1),
            ArrayArg::from_raw_parts::<u32>(&index_bindings.input, index_bindings.input_len, 1),
            ArrayArg::from_raw_parts::<u32>(&dummy_index_handle, dummy_indices.len(), 1),
            ArrayArg::from_raw_parts::<u32>(&index_bindings.rhs, index_bindings.rhs_len, 1),
            ArrayArg::from_raw_parts::<u32>(&dummy_index_handle, dummy_indices.len(), 1),
            ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
            ArrayArg::from_raw_parts::<ValueSource::Item>(&output.handle, output.len, 1),
        )
        .map_err(|err| Error::Launch {
            message: format!("{err:?}"),
        })?;
    }

    Ok(output)
}

pub(super) fn device_expr_scatter_if<ValueSource, IndexSource, StencilSource, InitialSource, Pred>(
    values: &ValueSource,
    indices: &IndexSource,
    stencil: &StencilSource,
    initial: &InitialSource,
) -> Result<DeviceVec<ValueSource::Runtime, ValueSource::Item>, Error>
where
    ValueSource: KernelColumn + KernelColumnAt<S0>,
    ValueSource::Runtime: Runtime,
    IndexSource: KernelColumn<Runtime = ValueSource::Runtime, Item = u32> + KernelColumnAt<S0>,
    StencilSource: KernelColumn<Runtime = ValueSource::Runtime> + KernelColumnAt<S0>,
    InitialSource:
        KernelColumn<Runtime = ValueSource::Runtime, Item = ValueSource::Item> + KernelColumnAt<S0>,
    ValueSource::Item: CubePrimitive + CubeElement,
    StencilSource::Item: CubePrimitive + CubeElement,
    ValueSource::Expr: GpuExpr<ValueSource::Item>,
    IndexSource::Expr: GpuExpr<u32>,
    StencilSource::Expr: GpuExpr<StencilSource::Item>,
    InitialSource::Expr: DeviceGpuExpr<ValueSource::Item>,
    Pred: PredicateOp<StencilSource::Item>,
{
    values.validate()?;
    indices.validate()?;
    stencil.validate()?;
    ensure_same_len(values.len(), indices.len())?;
    ensure_same_len(values.len(), stencil.len())?;

    let output = device_expr_collect(initial)?;
    let len = values.len();
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    if len == 0 {
        return Ok(output);
    }

    let client = values.policy().client();
    let block_count_u32 = api_expr_block_count(len)?;
    let value_bindings = values.stage()?;
    let index_bindings = indices.stage()?;
    let stencil_bindings = stencil.stage()?;
    let len_values = [len_u32];
    let len_handle = client.create_from_slice(u32::as_bytes(&len_values));
    let dummy_indices = [0_u32];
    let dummy_index_handle = client.create_from_slice(u32::as_bytes(&dummy_indices));

    unsafe {
        scatter_if_expr_kernel::launch_unchecked::<
            ValueSource::Item,
            StencilSource::Item,
            ValueSource::Expr,
            IndexSource::Expr,
            StencilSource::Expr,
            Pred,
            ValueSource::Runtime,
        >(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_API_EXPR_SIZE),
            ArrayArg::from_raw_parts::<ValueSource::Item>(
                &value_bindings.input,
                value_bindings.input_len,
                1,
            ),
            ArrayArg::from_raw_parts::<u32>(&dummy_index_handle, dummy_indices.len(), 1),
            ArrayArg::from_raw_parts::<ValueSource::Item>(
                &value_bindings.rhs,
                value_bindings.rhs_len,
                1,
            ),
            ArrayArg::from_raw_parts::<u32>(&dummy_index_handle, dummy_indices.len(), 1),
            ArrayArg::from_raw_parts::<u32>(&index_bindings.input, index_bindings.input_len, 1),
            ArrayArg::from_raw_parts::<u32>(&dummy_index_handle, dummy_indices.len(), 1),
            ArrayArg::from_raw_parts::<u32>(&index_bindings.rhs, index_bindings.rhs_len, 1),
            ArrayArg::from_raw_parts::<u32>(&dummy_index_handle, dummy_indices.len(), 1),
            ArrayArg::from_raw_parts::<StencilSource::Item>(
                &stencil_bindings.input,
                stencil_bindings.input_len,
                1,
            ),
            ArrayArg::from_raw_parts::<u32>(&dummy_index_handle, dummy_indices.len(), 1),
            ArrayArg::from_raw_parts::<StencilSource::Item>(
                &stencil_bindings.rhs,
                stencil_bindings.rhs_len,
                1,
            ),
            ArrayArg::from_raw_parts::<u32>(&dummy_index_handle, dummy_indices.len(), 1),
            ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
            ArrayArg::from_raw_parts::<ValueSource::Item>(&output.handle, output.len, 1),
        )
        .map_err(|err| Error::Launch {
            message: format!("{err:?}"),
        })?;
    }

    Ok(output)
}

pub(super) fn device_expr_copy_if<ExprSource, Pred>(
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
    let handles = device_expr_selection_handles::<ExprSource, Pred>(expr, invert)?;
    select::compact::<ExprSource::Runtime, ExprSource::Item>(expr.policy(), handles)
}

pub(super) fn device_expr_count_if<ExprSource, Pred>(
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
    let handles = device_expr_selection_handles::<ExprSource, Pred>(expr, invert)?;
    if handles.len == 0 {
        return Ok(0);
    }

    Ok(read_u32_scalar::<ExprSource::Runtime>(expr.policy().client(), handles.count) as usize)
}

pub(super) fn device_expr_find_if<ExprSource, Pred>(
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
    let handles = device_expr_selection_handles::<ExprSource, Pred>(expr, invert)?;
    primitive_search::first_flag(expr.policy(), handles.flag, handles.len, expr.len())
}

fn device_expr_selection_handles<ExprSource, Pred>(
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
    let client = expr.policy().client();
    if len == 0 {
        return select::handles_from_flags(expr.policy(), 0, 0, client.empty(0), client.empty(0));
    }

    let dummy_indices = [0_u32];
    let len_values = [len_u32];
    let invert_values = [if invert { 1_u32 } else { 0_u32 }];
    let dummy_index_handle = client.create_from_slice(u32::as_bytes(&dummy_indices));
    let len_handle = client.create_from_slice(u32::as_bytes(&len_values));
    let invert_handle = client.create_from_slice(u32::as_bytes(&invert_values));
    let flag_handle = client.empty(len * std::mem::size_of::<u32>());
    let bindings = expr.stage()?;
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
                ArrayArg::from_raw_parts::<ExprSource::Item>(
                    &bindings.input,
                    bindings.input_len,
                    1,
                ),
                ArrayArg::from_raw_parts::<u32>(&dummy_index_handle, dummy_indices.len(), 1),
                ArrayArg::from_raw_parts::<ExprSource::Item>(&bindings.rhs, bindings.rhs_len, 1),
                ArrayArg::from_raw_parts::<u32>(&dummy_index_handle, dummy_indices.len(), 1),
                ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
                ArrayArg::from_raw_parts::<u32>(&invert_handle, 1, 1),
                ArrayArg::from_raw_parts::<u32>(&flag_handle, len, 1),
            )
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
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
                ArrayArg::from_raw_parts::<ExprSource::Item>(
                    &bindings.input,
                    bindings.input_len,
                    1,
                ),
                ArrayArg::from_raw_parts::<u32>(&dummy_index_handle, dummy_indices.len(), 1),
                ArrayArg::from_raw_parts::<ExprSource::Item>(&bindings.rhs, bindings.rhs_len, 1),
                ArrayArg::from_raw_parts::<u32>(&dummy_index_handle, dummy_indices.len(), 1),
                ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
                ArrayArg::from_raw_parts::<u32>(&invert_handle, 1, 1),
                ArrayArg::from_raw_parts::<u32>(&flag_handle, len, 1),
                ArrayArg::from_raw_parts::<ExprSource::Item>(&value_handle, len, 1),
            )
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
        }
        value_handle
    };

    select::handles_from_flags(expr.policy(), len, len_u32, flag_handle, value_handle)
}

pub(super) fn device_expr_reduce<ExprSource, Op>(
    expr: &ExprSource,
    init: ExprSource::Item,
) -> Result<ExprSource::Item, Error>
where
    ExprSource: KernelColumn + KernelColumnAt<S0>,
    ExprSource::Runtime: Runtime,
    ExprSource::Item: CubePrimitive + CubeElement,
    ExprSource::Expr: DeviceGpuExpr<ExprSource::Item>,
    Op: BinaryOp<ExprSource::Item>,
{
    expr.validate()?;
    if expr.len() == 0 {
        return Ok(init);
    }

    let bindings = expr.stage()?;
    primitive_reduce::reduce_device_expr::<
        ExprSource::Runtime,
        ExprSource::Item,
        ExprSource::Expr,
        Op,
    >(expr.policy(), &bindings, expr.len(), init)
}

pub(super) fn device_expr_inclusive_scan<ExprSource, Op>(
    expr: &ExprSource,
) -> Result<DeviceVec<ExprSource::Runtime, ExprSource::Item>, Error>
where
    ExprSource: KernelColumn + KernelColumnAt<S0>,
    ExprSource::Runtime: Runtime,
    ExprSource::Item: CubePrimitive + CubeElement,
    ExprSource::Expr: DeviceGpuExpr<ExprSource::Item>,
    Op: BinaryOp<ExprSource::Item>,
    Input<ExprSource::Item>: GpuExpr<ExprSource::Item>,
{
    expr.validate()?;
    let len = expr.len();
    let bindings = expr.stage()?;
    let output_handle = primitive_scan::inclusive_scan_device_expr::<
        ExprSource::Runtime,
        ExprSource::Item,
        ExprSource::Expr,
        Op,
    >(expr.policy(), &bindings, len)?;

    Ok(DeviceVec::from_handle(
        expr.policy().clone(),
        output_handle,
        len,
    ))
}

pub(super) fn device_expr_exclusive_scan<ExprSource, Op>(
    expr: &ExprSource,
    init: ExprSource::Item,
) -> Result<DeviceVec<ExprSource::Runtime, ExprSource::Item>, Error>
where
    ExprSource: KernelColumn + KernelColumnAt<S0>,
    ExprSource::Runtime: Runtime,
    ExprSource::Item: CubePrimitive + CubeElement,
    ExprSource::Expr: DeviceGpuExpr<ExprSource::Item>,
    Op: BinaryOp<ExprSource::Item>,
    Input<ExprSource::Item>: GpuExpr<ExprSource::Item>,
{
    expr.validate()?;
    let len = expr.len();
    let bindings = expr.stage()?;
    let output_handle = primitive_scan::exclusive_scan_device_expr::<
        ExprSource::Runtime,
        ExprSource::Item,
        ExprSource::Expr,
        Op,
    >(expr.policy(), &bindings, len, init)?;

    Ok(DeviceVec::from_handle(
        expr.policy().clone(),
        output_handle,
        len,
    ))
}

pub(super) fn device_expr_adjacent_difference<ExprSource, Op>(
    expr: &ExprSource,
) -> Result<DeviceVec<ExprSource::Runtime, ExprSource::Item>, Error>
where
    ExprSource: KernelColumn + KernelColumnAt<S0>,
    ExprSource::Runtime: Runtime,
    ExprSource::Item: CubePrimitive + CubeElement,
    ExprSource::Expr: GpuExpr<ExprSource::Item>,
    Op: BinaryOp<ExprSource::Item>,
{
    expr.validate()?;
    let len = expr.len();
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let client = expr.policy().client();
    let output_handle = client.empty(len * std::mem::size_of::<ExprSource::Item>());

    if len != 0 {
        let block_count_u32 = api_expr_block_count(len)?;
        let dummy_indices = [0_u32];
        let len_values = [len_u32];
        let bindings = expr.stage()?;
        let dummy_index_handle = client.create_from_slice(u32::as_bytes(&dummy_indices));
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
                ArrayArg::from_raw_parts::<ExprSource::Item>(&bindings.rhs, bindings.rhs_len, 1),
                ArrayArg::from_raw_parts::<u32>(&dummy_index_handle, dummy_indices.len(), 1),
                ArrayArg::from_raw_parts::<ExprSource::Item>(
                    &bindings.input,
                    bindings.input_len,
                    1,
                ),
                ArrayArg::from_raw_parts::<u32>(&dummy_index_handle, dummy_indices.len(), 1),
                ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
                ArrayArg::from_raw_parts::<ExprSource::Item>(&output_handle, len, 1),
            )
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
        }
    }

    Ok(DeviceVec::from_handle(
        expr.policy().clone(),
        output_handle,
        len,
    ))
}

pub(super) fn device_expr_minmax_element<ExprSource, Less>(
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
    let values = device_expr_collect(expr)?;
    primitive_search::minmax_element(&values, GpuOp::<Less>::new())
}

pub(super) fn device_expr_inclusive_scan_by_key<ExprSource, K, KeyEq, Op>(
    expr: &ExprSource,
    keys: &DeviceVec<ExprSource::Runtime, K>,
) -> Result<DeviceVec<ExprSource::Runtime, ExprSource::Item>, Error>
where
    ExprSource: KernelColumn + KernelColumnAt<S0>,
    ExprSource::Runtime: Runtime,
    ExprSource::Item: CubePrimitive + CubeElement,
    ExprSource::Expr: DeviceGpuExpr<ExprSource::Item>,
    K: CubePrimitive + CubeElement,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<ExprSource::Item>,
{
    expr.validate()?;
    ensure_same_len(expr.len(), keys.len)?;

    let values = device_expr_collect(expr)?;
    primitive_scan::inclusive_scan_by_key_device_vec(
        keys,
        &values,
        GpuOp::<KeyEq>::new(),
        GpuOp::<Op>::new(),
    )
}

pub(super) fn device_expr_exclusive_scan_by_key<ExprSource, K, KeyEq, Op>(
    expr: &ExprSource,
    keys: &DeviceVec<ExprSource::Runtime, K>,
    init: ExprSource::Item,
) -> Result<DeviceVec<ExprSource::Runtime, ExprSource::Item>, Error>
where
    ExprSource: KernelColumn + KernelColumnAt<S0>,
    ExprSource::Runtime: Runtime,
    ExprSource::Item: CubePrimitive + CubeElement,
    ExprSource::Expr: DeviceGpuExpr<ExprSource::Item>,
    K: CubePrimitive + CubeElement,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<ExprSource::Item>,
{
    expr.validate()?;
    ensure_same_len(expr.len(), keys.len)?;

    let values = device_expr_collect(expr)?;
    primitive_scan::exclusive_scan_by_key_device_vec(
        keys,
        &values,
        init,
        GpuOp::<KeyEq>::new(),
        GpuOp::<Op>::new(),
    )
}

pub(super) fn device_expr_reduce_by_key<ExprSource, K, KeyEq, Op>(
    expr: &ExprSource,
    keys: &DeviceVec<ExprSource::Runtime, K>,
    init: ExprSource::Item,
) -> Result<
    (
        DeviceVec<ExprSource::Runtime, K>,
        DeviceVec<ExprSource::Runtime, ExprSource::Item>,
    ),
    Error,
>
where
    ExprSource: KernelColumn + KernelColumnAt<S0>,
    ExprSource::Runtime: Runtime,
    ExprSource::Item: CubePrimitive + CubeElement,
    ExprSource::Expr: GpuExpr<ExprSource::Item>,
    K: CubePrimitive + CubeElement,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<ExprSource::Item>,
{
    expr.validate()?;
    ensure_same_len(expr.len(), keys.len)?;

    let bindings = expr.stage()?;
    primitive_reduce::reduce_by_key_expr_handle::<
        ExprSource::Runtime,
        K,
        ExprSource::Item,
        ExprSource::Expr,
        KeyEq,
        Op,
    >(
        expr.policy(),
        keys,
        bindings.input,
        bindings.input_len,
        bindings.rhs,
        bindings.rhs_len,
        init,
    )
}

pub(super) fn device_expr_reduce_by_key_with_control<ExprSource, K, KeyEq, Op>(
    expr: &ExprSource,
    keys: &DeviceVec<ExprSource::Runtime, K>,
    init: ExprSource::Item,
) -> Result<
    (
        DeviceVec<ExprSource::Runtime, K>,
        DeviceVec<ExprSource::Runtime, ExprSource::Item>,
        primitive_reduce::ReduceByKeyControl,
    ),
    Error,
>
where
    ExprSource: KernelColumn + KernelColumnAt<S0>,
    ExprSource::Runtime: Runtime,
    ExprSource::Item: CubePrimitive + CubeElement,
    ExprSource::Expr: GpuExpr<ExprSource::Item>,
    K: CubePrimitive + CubeElement,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<ExprSource::Item>,
{
    expr.validate()?;
    ensure_same_len(expr.len(), keys.len)?;

    let bindings = expr.stage()?;
    primitive_reduce::reduce_by_key_expr_handle_with_control::<
        ExprSource::Runtime,
        K,
        ExprSource::Item,
        ExprSource::Expr,
        KeyEq,
        Op,
    >(
        expr.policy(),
        keys,
        bindings.input,
        bindings.input_len,
        bindings.rhs,
        bindings.rhs_len,
        init,
    )
}

pub(super) fn device_expr_reduce_by_key_with_existing_control<ExprSource, K, KeyEq, Op>(
    expr: &ExprSource,
    keys: &DeviceVec<ExprSource::Runtime, K>,
    init: ExprSource::Item,
    control: &primitive_reduce::ReduceByKeyControl,
) -> Result<DeviceVec<ExprSource::Runtime, ExprSource::Item>, Error>
where
    ExprSource: KernelColumn + KernelColumnAt<S0>,
    ExprSource::Runtime: Runtime,
    ExprSource::Item: CubePrimitive + CubeElement,
    ExprSource::Expr: GpuExpr<ExprSource::Item>,
    K: CubePrimitive + CubeElement,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<ExprSource::Item>,
{
    expr.validate()?;
    ensure_same_len(expr.len(), keys.len)?;

    let bindings = expr.stage()?;
    primitive_reduce::reduce_by_key_expr_handle_with_existing_control::<
        ExprSource::Runtime,
        K,
        ExprSource::Item,
        ExprSource::Expr,
        KeyEq,
        Op,
    >(
        expr.policy(),
        keys,
        bindings.input,
        bindings.input_len,
        bindings.rhs,
        bindings.rhs_len,
        init,
        control,
    )
}
