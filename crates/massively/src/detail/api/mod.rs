mod gather;
mod memory;
mod ordering;
mod reduce;
mod scan;
mod scatter;
mod search;
mod selection;
mod sequence;

use std::marker::PhantomData;

pub use gather::{gather, gather_if};
pub use memory::{
    MaterializeOutput, StorageOutput, TransformSoA2Output, TransformSoA3Output,
    TransformUnaryOutput,
};
pub use ordering::{
    merge, merge_by_key, reverse, set_difference, set_intersection, set_union, sort, sort_by_key,
};
pub use reduce::{inner_product, reduce, reduce_by_key};
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
    device::{DeviceVec, KernelColumn, KernelColumnAt, S0, SoAView2, SoAView3},
    error::{Error, ensure_same_len},
    expr::{DeviceGpuExpr, GpuExpr},
    kernels::*,
    op::{BinaryOp, BinaryPredicateOp, GpuOp, PredicateOp},
    primitives::{
        reduce as primitive_reduce, scan as primitive_scan, scan::read_u32_scalar,
        search as primitive_search, select,
    },
};
use cubecl::prelude::*;

const BLOCK_API_EXPR_SIZE: u32 = 256;

#[doc(hidden)]
pub struct Tuple1Less<Less> {
    _less: PhantomData<fn() -> Less>,
}

impl<Less> Default for Tuple1Less<Less> {
    fn default() -> Self {
        Self { _less: PhantomData }
    }
}

#[cubecl::cube]
impl<T, Less> BinaryPredicateOp<T> for Tuple1Less<Less>
where
    T: CubePrimitive + CubeElement,
    Less: BinaryPredicateOp<(T,)>,
{
    fn apply(lhs: T, rhs: T) -> bool {
        Less::apply((lhs,), (rhs,))
    }
}

#[doc(hidden)]
pub struct Tuple1BinaryOp<Op> {
    _op: PhantomData<fn() -> Op>,
}

impl<Op> Default for Tuple1BinaryOp<Op> {
    fn default() -> Self {
        Self { _op: PhantomData }
    }
}

#[cubecl::cube]
impl<T, Op> BinaryOp<T> for Tuple1BinaryOp<Op>
where
    T: CubePrimitive + CubeElement,
    Op: BinaryOp<(T,)>,
{
    fn apply(lhs: T, rhs: T) -> T {
        Op::apply((lhs,), (rhs,)).0
    }
}

#[doc(hidden)]
pub struct Tuple1PredicateOp<Op> {
    _op: PhantomData<fn() -> Op>,
}

impl<Op> Default for Tuple1PredicateOp<Op> {
    fn default() -> Self {
        Self { _op: PhantomData }
    }
}

#[cubecl::cube]
impl<T, Op> PredicateOp<T> for Tuple1PredicateOp<Op>
where
    T: CubePrimitive + CubeElement,
    Op: PredicateOp<(T,)>,
{
    fn apply(input: T) -> bool {
        Op::apply((input,))
    }
}

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

#[doc(hidden)]
pub trait SelectionStencil<Pred> {
    type Runtime: Runtime;

    fn len(&self) -> usize;
    fn selection_handles_with_policy(
        &self,
        policy: &crate::policy::CubePolicy<Self::Runtime>,
        invert: bool,
    ) -> Result<select::SelectionHandles, Error>;
}

#[doc(hidden)]
pub struct PrecomputedSelection<R: Runtime> {
    len: usize,
    handles: select::SelectionHandles,
    _runtime: std::marker::PhantomData<R>,
}

impl<R: Runtime> PrecomputedSelection<R> {
    pub(crate) fn from_stencil_with_policy<Stencil, Pred>(
        policy: &crate::policy::CubePolicy<R>,
        stencil: &Stencil,
        invert: bool,
    ) -> Result<Self, Error>
    where
        Stencil: SelectionStencil<Pred, Runtime = R>,
    {
        Ok(Self {
            len: stencil.len(),
            handles: stencil.selection_handles_with_policy(policy, invert)?,
            _runtime: std::marker::PhantomData,
        })
    }
}

impl<R, Pred> SelectionStencil<Pred> for PrecomputedSelection<R>
where
    R: Runtime,
{
    type Runtime = R;

    fn len(&self) -> usize {
        self.len
    }

    fn selection_handles_with_policy(
        &self,
        _policy: &crate::policy::CubePolicy<Self::Runtime>,
        _invert: bool,
    ) -> Result<select::SelectionHandles, Error> {
        Ok(self.handles.clone())
    }
}

impl<Stencil, Pred> SelectionStencil<Pred> for Stencil
where
    Stencil: KernelColumn + KernelColumnAt<S0>,
    Stencil::Runtime: Runtime,
    Stencil::Item: CubePrimitive + CubeElement,
    Stencil::Expr: GpuExpr<Stencil::Item>,
    Pred: PredicateOp<Stencil::Item>,
{
    type Runtime = Stencil::Runtime;

    fn len(&self) -> usize {
        KernelColumn::len(self)
    }

    fn selection_handles_with_policy(
        &self,
        policy: &crate::policy::CubePolicy<Self::Runtime>,
        invert: bool,
    ) -> Result<select::SelectionHandles, Error> {
        device_expr_selection_handles_with_policy::<Stencil, Pred>(policy, self, invert)
    }
}

impl<Stencil, Pred> SelectionStencil<Pred> for (Stencil,)
where
    Stencil: KernelColumn + KernelColumnAt<S0>,
    Stencil::Runtime: Runtime,
    Stencil::Item: CubePrimitive + CubeElement,
    Stencil::Expr: GpuExpr<Stencil::Item>,
    Pred: PredicateOp<(Stencil::Item,)>,
{
    type Runtime = Stencil::Runtime;

    fn len(&self) -> usize {
        self.0.len()
    }

    fn selection_handles_with_policy(
        &self,
        policy: &crate::policy::CubePolicy<Self::Runtime>,
        invert: bool,
    ) -> Result<select::SelectionHandles, Error> {
        device_expr_selection_handles_with_policy::<Stencil, Tuple1PredicateOp<Pred>>(
            policy, &self.0, invert,
        )
    }
}

macro_rules! impl_tuple_selection_stencil {
    (
        $name:ident < $first:ident, $( $rest:ident ),+ > {
            $first_field:ident, $( $field:ident ),+
        },
        $kernel_name:ident
    ) => {
        impl<$first, $( $rest ),+, Pred> SelectionStencil<Pred>
            for $name<$first, $( $rest ),+>
        where
            $first: KernelColumn + KernelColumnAt<S0>,
            $(
                $rest: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>,
            )+
            <$first as KernelColumn>::Runtime: Runtime,
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
                <$rest as KernelColumn>::Expr: DeviceGpuExpr<<$rest as KernelColumn>::Item>,
            )+
            Pred: PredicateOp<(
                <$first as KernelColumn>::Item,
                $( <$rest as KernelColumn>::Item ),+
            )>,
        {
            type Runtime = <$first as KernelColumn>::Runtime;

            fn len(&self) -> usize {
                self.$first_field.len()
            }

            fn selection_handles_with_policy(
                &self,
                policy: &crate::policy::CubePolicy<Self::Runtime>,
                invert: bool,
            ) -> Result<select::SelectionHandles, Error> {
                self.$first_field.validate()?;
                $(
                    self.$field.validate()?;
                    ensure_same_len(self.$field.len(), self.$first_field.len())?;
                )+
                let $first_field = device_expr_collect_with_policy(policy, &self.$first_field)?;
                $(
                    let $field = device_expr_collect_with_policy(policy, &self.$field)?;
                )+
                let len = $first_field.len();
                let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
                let client = policy.client();
                let flag = client.empty(len * std::mem::size_of::<u32>());
                if len != 0 {
                    let block_count_u32 = api_expr_block_count(len)?;
                    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
                    let invert_values = [if invert { 1_u32 } else { 0_u32 }];
                    let invert_handle = client.create_from_slice(u32::as_bytes(&invert_values));
                    unsafe {
                        $kernel_name::launch_unchecked::<
                            <$first as KernelColumn>::Item,
                            $( <$rest as KernelColumn>::Item, )+
                            Pred,
                            <$first as KernelColumn>::Runtime,
                        >(
                            client,
                            CubeCount::Static(block_count_u32, 1, 1),
                            CubeDim::new_1d(BLOCK_API_EXPR_SIZE),
                            unsafe { BufferArg::from_raw_parts($first_field.handle.clone(), len) },
                            $(
                                unsafe { BufferArg::from_raw_parts($field.handle.clone(), len) },
                            )+
                            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
                            unsafe { BufferArg::from_raw_parts(invert_handle.clone(), 1) },
                            unsafe { BufferArg::from_raw_parts(flag.clone(), len) },
                        );
                    }
                }
                select::handles_from_flags(
                    policy,
                    len,
                    len_u32,
                    flag,
                    $first_field.handle.clone(),
                )
            }
        }
    };
}

impl_tuple_selection_stencil!(
    SoAView2<A, B> { left, right },
    tuple2_predicate_flags_kernel
);
impl_tuple_selection_stencil!(
    SoAView3<A, B, C> { first, second, third },
    tuple3_predicate_flags_kernel
);

impl<Left, Right, Pred> SelectionStencil<Pred> for (Left, Right)
where
    Left: Copy,
    Right: Copy,
    SoAView2<Left, Right>: SelectionStencil<Pred>,
{
    type Runtime = <SoAView2<Left, Right> as SelectionStencil<Pred>>::Runtime;

    fn len(&self) -> usize {
        <SoAView2<Left, Right> as SelectionStencil<Pred>>::len(&SoAView2 {
            left: self.0,
            right: self.1,
        })
    }

    fn selection_handles_with_policy(
        &self,
        policy: &crate::policy::CubePolicy<Self::Runtime>,
        invert: bool,
    ) -> Result<select::SelectionHandles, Error> {
        SoAView2 {
            left: self.0,
            right: self.1,
        }
        .selection_handles_with_policy(policy, invert)
    }
}

impl<First, Second, Third, Pred> SelectionStencil<Pred> for (First, Second, Third)
where
    First: Copy,
    Second: Copy,
    Third: Copy,
    SoAView3<First, Second, Third>: SelectionStencil<Pred>,
{
    type Runtime = <SoAView3<First, Second, Third> as SelectionStencil<Pred>>::Runtime;

    fn len(&self) -> usize {
        <SoAView3<First, Second, Third> as SelectionStencil<Pred>>::len(&SoAView3 {
            first: self.0,
            second: self.1,
            third: self.2,
        })
    }

    fn selection_handles_with_policy(
        &self,
        policy: &crate::policy::CubePolicy<Self::Runtime>,
        invert: bool,
    ) -> Result<select::SelectionHandles, Error> {
        SoAView3 {
            first: self.0,
            second: self.1,
            third: self.2,
        }
        .selection_handles_with_policy(policy, invert)
    }
}

pub(super) fn device_expr_reduce_with_policy<ExprSource, Op>(
    policy: &crate::policy::CubePolicy<ExprSource::Runtime>,
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

    let values = device_expr_collect_with_policy(policy, expr)?;
    primitive_reduce::reduce_input_handle::<ExprSource::Runtime, ExprSource::Item, Op>(
        policy,
        values.handle,
        values.len,
        values.len,
        init,
    )
}

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
    let values = device_expr_collect_with_policy(policy, expr)?;
    primitive_search::minmax_element(policy, &values, GpuOp::<Less>::new())
}

pub(super) fn device_expr_inclusive_scan_by_key_with_policy<ExprSource, K, KeyEq, Op>(
    policy: &crate::policy::CubePolicy<ExprSource::Runtime>,
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

    let values = device_expr_collect_with_policy(policy, expr)?;
    primitive_scan::inclusive_scan_by_key_device_vec(
        policy,
        keys,
        &values,
        GpuOp::<KeyEq>::new(),
        GpuOp::<Op>::new(),
    )
}

pub(super) fn device_expr_exclusive_scan_by_key_with_policy<ExprSource, K, KeyEq, Op>(
    policy: &crate::policy::CubePolicy<ExprSource::Runtime>,
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

    let values = device_expr_collect_with_policy(policy, expr)?;
    primitive_scan::exclusive_scan_by_key_device_vec(
        policy,
        keys,
        &values,
        init,
        GpuOp::<KeyEq>::new(),
        GpuOp::<Op>::new(),
    )
}

pub(super) fn device_expr_reduce_by_key_with_policy<ExprSource, K, KeyEq, Op>(
    policy: &crate::policy::CubePolicy<ExprSource::Runtime>,
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
    ExprSource::Expr: DeviceGpuExpr<ExprSource::Item>,
    K: CubePrimitive + CubeElement,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<ExprSource::Item>,
{
    expr.validate()?;
    ensure_same_len(expr.len(), keys.len)?;

    let values = device_expr_collect_with_policy(policy, expr)?;
    primitive_reduce::reduce_by_key_handle::<ExprSource::Runtime, K, ExprSource::Item, KeyEq, Op>(
        policy,
        keys,
        values.handle.clone(),
        init,
    )
}

pub(super) fn device_expr_reduce_by_key_with_control_with_policy<ExprSource, K, KeyEq, Op>(
    policy: &crate::policy::CubePolicy<ExprSource::Runtime>,
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
    ExprSource::Expr: DeviceGpuExpr<ExprSource::Item>,
    K: CubePrimitive + CubeElement,
    KeyEq: BinaryPredicateOp<K>,
    Op: BinaryOp<ExprSource::Item>,
{
    expr.validate()?;
    ensure_same_len(expr.len(), keys.len)?;

    let values = device_expr_collect_with_policy(policy, expr)?;
    primitive_reduce::reduce_by_key_handle_with_control::<
        ExprSource::Runtime,
        K,
        ExprSource::Item,
        KeyEq,
        Op,
    >(policy, keys, values.handle.clone(), init)
}

pub(super) fn device_expr_reduce_by_key_with_existing_control_with_policy<
    ExprSource,
    K,
    KeyEq,
    Op,
>(
    policy: &crate::policy::CubePolicy<ExprSource::Runtime>,
    expr: &ExprSource,
    keys: &DeviceVec<ExprSource::Runtime, K>,
    init: ExprSource::Item,
    control: &primitive_reduce::ReduceByKeyControl,
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

    let values = device_expr_collect_with_policy(policy, expr)?;
    primitive_reduce::reduce_by_key_handle_with_existing_control::<
        ExprSource::Runtime,
        K,
        ExprSource::Item,
        KeyEq,
        Op,
    >(policy, keys, values.handle.clone(), init, control)
}
