//! Public algorithm API implementation for `massively`.

use cubecl::prelude::Runtime;

use crate::detail::dispatch as sealed;
use crate::detail::op_adapter::{KernelOp, StencilFlag};
use crate::iter::{MIter, MIterMut};
use crate::op;
use crate::runtime::{DeviceSlice, Executor};
use crate::value::{MItem, MVec};

pub use crate::Error;

fn validate_input<R, Input>(exec: &Executor<R>, input: &Input) -> Result<(), Error>
where
    R: Runtime,
    Input: MIter<R>,
{
    <Input as sealed::MIterDispatch<R>>::validate_executor(input, exec)
}

fn validate_output<R, Output>(exec: &Executor<R>, output: &Output) -> Result<(), Error>
where
    R: Runtime,
    Output: MIterMut<R>,
{
    <Output as sealed::MIterMutDispatch<R>>::validate_executor(output, exec)
}

fn validate_device_slice<R, T>(
    exec: &Executor<R>,
    slice: &DeviceSlice<'_, R, T>,
) -> Result<(), Error>
where
    R: Runtime,
{
    exec.ensure_policy_id(slice.policy_id())
}

fn u32_stencil<R>(
    policy: &crate::detail::CubePolicy<R>,
    stencil: DeviceSlice<'_, R, u32>,
    invert: bool,
) -> Result<crate::detail::api::PrecomputedSelection<R>, Error>
where
    R: Runtime,
{
    crate::detail::api::PrecomputedSelection::from_stencil_with_policy::<_, KernelOp<R, StencilFlag>>(
        policy,
        &(stencil.column_view(),),
        invert,
    )
}

fn u32_stencil_flags<R>(
    policy: &crate::detail::CubePolicy<R>,
    stencil: DeviceSlice<'_, R, u32>,
    invert: bool,
) -> Result<crate::detail::api::PrecomputedSelection<R>, Error>
where
    R: Runtime,
{
    crate::detail::api::PrecomputedSelection::from_stencil_flags_with_policy::<
        _,
        KernelOp<R, StencilFlag>,
    >(policy, &(stencil.column_view(),), invert)
}

mod indexed;
mod ordering;
mod predicate;
mod reduce;
mod scan;
mod search;
mod selection;
mod set;
mod transform;
mod unique;

pub use indexed::{gather, gather_where, permute, scatter, scatter_where};
pub use ordering::{
    merge, merge_by_key, reverse, sort, sort_by_key, stable_sort, stable_sort_by_key,
};
pub use predicate::{all_of, any_of, count_if, find_if, is_partitioned, none_of, partition};
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
pub use selection::{copy_where, remove_where, replace_where};
pub use set::{set_difference, set_intersection, set_union};
pub use transform::{map, transform, transform_where};
pub use unique::{unique, unique_by_key};
