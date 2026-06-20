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
    if len == 0 {
        return Ok(DeviceVec::empty(expr.policy().clone()));
    }

    let client = expr.policy().client();
    let output_handle = client.empty(len * std::mem::size_of::<ExprSource::Item>());

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
            unsafe { BufferArg::from_raw_parts(output_handle.clone(), len) },
            unsafe { BufferArg::from_raw_parts(slot0.0.clone(), slot0.1) },
            unsafe { BufferArg::from_raw_parts(slot1.0.clone(), slot1.1) },
            unsafe { BufferArg::from_raw_parts(slot2.0.clone(), slot2.1) },
            unsafe { BufferArg::from_raw_parts(slot3.0.clone(), slot3.1) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
        );
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
    if len == 0 {
        return Ok(DeviceVec::empty(input.policy().clone()));
    }

    let client = input.policy().client();
    let output_handle = client.empty(len * std::mem::size_of::<InputSource::Item>());

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
            unsafe { BufferArg::from_raw_parts(output_handle.clone(), len) },
            unsafe {
                BufferArg::from_raw_parts(input_bindings.input.clone(), input_bindings.input_len)
            },
            unsafe { BufferArg::from_raw_parts(dummy_index_handle.clone(), dummy_indices.len()) },
            unsafe {
                BufferArg::from_raw_parts(input_bindings.rhs.clone(), input_bindings.rhs_len)
            },
            unsafe { BufferArg::from_raw_parts(dummy_index_handle.clone(), dummy_indices.len()) },
            unsafe {
                BufferArg::from_raw_parts(index_bindings.input.clone(), index_bindings.input_len)
            },
            unsafe { BufferArg::from_raw_parts(dummy_index_handle.clone(), dummy_indices.len()) },
            unsafe {
                BufferArg::from_raw_parts(index_bindings.rhs.clone(), index_bindings.rhs_len)
            },
            unsafe { BufferArg::from_raw_parts(dummy_index_handle.clone(), dummy_indices.len()) },
        );
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
            unsafe {
                BufferArg::from_raw_parts(value_bindings.input.clone(), value_bindings.input_len)
            },
            unsafe { BufferArg::from_raw_parts(dummy_index_handle.clone(), dummy_indices.len()) },
            unsafe {
                BufferArg::from_raw_parts(value_bindings.rhs.clone(), value_bindings.rhs_len)
            },
            unsafe { BufferArg::from_raw_parts(dummy_index_handle.clone(), dummy_indices.len()) },
            unsafe {
                BufferArg::from_raw_parts(index_bindings.input.clone(), index_bindings.input_len)
            },
            unsafe { BufferArg::from_raw_parts(dummy_index_handle.clone(), dummy_indices.len()) },
            unsafe {
                BufferArg::from_raw_parts(index_bindings.rhs.clone(), index_bindings.rhs_len)
            },
            unsafe { BufferArg::from_raw_parts(dummy_index_handle.clone(), dummy_indices.len()) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output.handle.clone(), output.len) },
        );
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

    Ok(read_u32_scalar::<ExprSource::Runtime>(expr.policy().client(), handles.count)? as usize)
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

pub(super) fn device_expr_selection_handles<ExprSource, Pred>(
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
        return select::handles_from_flags(
            expr.policy(),
            0,
            0,
            expr.policy().empty_handle(),
            expr.policy().empty_handle(),
        );
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
                unsafe { BufferArg::from_raw_parts(bindings.input.clone(), bindings.input_len) },
                unsafe {
                    BufferArg::from_raw_parts(dummy_index_handle.clone(), dummy_indices.len())
                },
                unsafe { BufferArg::from_raw_parts(bindings.rhs.clone(), bindings.rhs_len) },
                unsafe {
                    BufferArg::from_raw_parts(dummy_index_handle.clone(), dummy_indices.len())
                },
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
                unsafe {
                    BufferArg::from_raw_parts(dummy_index_handle.clone(), dummy_indices.len())
                },
                unsafe { BufferArg::from_raw_parts(bindings.rhs.clone(), bindings.rhs_len) },
                unsafe {
                    BufferArg::from_raw_parts(dummy_index_handle.clone(), dummy_indices.len())
                },
                unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(invert_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(flag_handle.clone(), len) },
                unsafe { BufferArg::from_raw_parts(value_handle.clone(), len) },
            );
        }
        value_handle
    };

    select::handles_from_flags(expr.policy(), len, len_u32, flag_handle, value_handle)
}

#[doc(hidden)]
pub trait SelectionStencil<Pred> {
    type Runtime: Runtime;

    fn len(&self) -> usize;
    fn selection_handles(&self, invert: bool) -> Result<select::SelectionHandles, Error>;
}

#[doc(hidden)]
pub struct PrecomputedSelection<R: Runtime> {
    len: usize,
    handles: select::SelectionHandles,
    _runtime: std::marker::PhantomData<R>,
}

impl<R: Runtime> PrecomputedSelection<R> {
    pub(crate) fn from_stencil<Stencil, Pred>(
        stencil: &Stencil,
        invert: bool,
    ) -> Result<Self, Error>
    where
        Stencil: SelectionStencil<Pred, Runtime = R>,
    {
        Ok(Self {
            len: stencil.len(),
            handles: stencil.selection_handles(invert)?,
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

    fn selection_handles(&self, _invert: bool) -> Result<select::SelectionHandles, Error> {
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

    fn selection_handles(&self, invert: bool) -> Result<select::SelectionHandles, Error> {
        device_expr_selection_handles::<Stencil, Pred>(self, invert)
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

    fn selection_handles(&self, invert: bool) -> Result<select::SelectionHandles, Error> {
        device_expr_selection_handles::<Stencil, Tuple1PredicateOp<Pred>>(&self.0, invert)
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

            fn selection_handles(&self, invert: bool) -> Result<select::SelectionHandles, Error> {
                self.$first_field.validate()?;
                $(
                    self.$field.validate()?;
                    ensure_same_len(self.$field.len(), self.$first_field.len())?;
                )+
                let $first_field = device_expr_collect(&self.$first_field)?;
                $(
                    let $field = device_expr_collect(&self.$field)?;
                )+
                let len = $first_field.len();
                let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
                let client = $first_field.policy().client();
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
                    $first_field.policy(),
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

    fn selection_handles(&self, invert: bool) -> Result<select::SelectionHandles, Error> {
        SoAView2 {
            left: self.0,
            right: self.1,
        }
        .selection_handles(invert)
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

    fn selection_handles(&self, invert: bool) -> Result<select::SelectionHandles, Error> {
        SoAView3 {
            first: self.0,
            second: self.1,
            third: self.2,
        }
        .selection_handles(invert)
    }
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

    let values = device_expr_collect(expr)?;
    primitive_reduce::reduce_input_handle::<ExprSource::Runtime, ExprSource::Item, Op>(
        expr.policy(),
        values.handle,
        values.len,
        values.len,
        init,
    )
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
                unsafe { BufferArg::from_raw_parts(bindings.rhs.clone(), bindings.rhs_len) },
                unsafe {
                    BufferArg::from_raw_parts(dummy_index_handle.clone(), dummy_indices.len())
                },
                unsafe { BufferArg::from_raw_parts(bindings.input.clone(), bindings.input_len) },
                unsafe {
                    BufferArg::from_raw_parts(dummy_index_handle.clone(), dummy_indices.len())
                },
                unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(output_handle.clone(), len) },
            );
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
