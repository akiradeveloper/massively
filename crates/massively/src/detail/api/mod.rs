mod expr;
mod memory;
mod ordering;
mod payload;
mod reduce;
mod scan;
mod search;
mod selection;
mod sequence;

#[allow(unused_imports)]
pub(super) use expr::{
    device_expr_adjacent_difference_with_policy, device_expr_collect_with_policy,
    device_expr_copy_where_with_policy, device_expr_count_if_with_policy,
    device_expr_exclusive_scan_by_key_expr_keys_with_policy, device_expr_find_if_with_policy,
    device_expr_gather_where_into_with_control, device_expr_gather_with_policy,
    device_expr_inclusive_scan_by_key_expr_keys_with_policy,
    device_expr_minmax_element_with_policy, device_expr_reverse_collect,
    device_expr_scatter_where_into_with_control, device_expr_selected_rank_with_policy,
    device_expr_selection_flags_with_policy, replace_where_into_with_control,
};
pub use expr::{
    device_expr_collect_into_with_policy, device_expr_copy_where_into_with_policy,
    device_expr_gather_into_with_policy, device_expr_scatter_into_with_policy,
};
pub use memory::{
    MItemStorage, MaterializeOutput, TransformSoA2Output, TransformSoA3Output, TransformSoA4Output,
    TransformSoA5Output, TransformSoA6Output, TransformSoA7Output, TransformUnaryOutput,
};
pub(super) use ordering::{
    device_expr_merge_by_key_control_with_policy,
    device_expr_merge_by_key_values_with_control_with_policy,
    device_expr_merge_tuple2_by_key_control_with_policy,
    device_expr_merge_tuple3_by_key_control_with_policy,
};
pub use ordering::{
    merge, merge_by_key, reverse, set_difference, set_intersection, set_union, sort, sort_by_key,
};
pub(super) use payload::{
    SelectedPayloadApply, SplitPayloadApply, device_expr_apply_selected_with_policy,
    device_value_apply_selected_with_policy,
};
pub use reduce::{reduce, reduce_by_key};
pub use scan::{
    adjacent_difference, exclusive_scan, exclusive_scan_by_key, inclusive_scan,
    inclusive_scan_by_key,
};
pub use search::{
    adjacent_find, equal, find_first_of, is_sorted, is_sorted_until, lexicographical_compare,
    lower_bound_many, max_element, min_element, minmax_element, mismatch, upper_bound_many,
};
pub use selection::{
    all_of, any_of, copy_where, count_if, find_if, is_partitioned, none_of, partition, remove_if,
};
pub use sequence::{replace_where, unique, unique_by_key};

use crate::{
    detail::op::kernel::{BinaryOp, BinaryPredicateOp, PredicateOp},
    device::{
        DeviceColumnMutView, DeviceVec, KernelColumn, KernelColumnAt, S0, SoAView2, SoAView3,
    },
    error::{Error, ensure_same_len},
    expr::{DeviceGpuExpr, GpuExpr},
    index::{MIndex, mindex_from_usize},
    kernels::*,
    primitives::{
        scan as primitive_scan, scan::read_u32_scalar, search as primitive_search, select,
    },
};
use cubecl::prelude::*;

const BLOCK_API_EXPR_SIZE: u32 = 256;

fn api_expr_block_count(len: usize) -> Result<u32, Error> {
    let block_count = len.div_ceil(BLOCK_API_EXPR_SIZE as usize);
    u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })
}

mod tuple_adapter;
pub use tuple_adapter::{
    Tuple1BinaryOp, Tuple1Less, Tuple1PredicateOp, Tuple2AsTuple3Less, Tuple4AsTuple7BinaryOp,
    Tuple4AsTuple7BinaryPredicateOp, Tuple4AsTuple7PredicateOp, Tuple5AsTuple7BinaryOp,
    Tuple5AsTuple7BinaryPredicateOp, Tuple5AsTuple7PredicateOp, Tuple6AsTuple7BinaryOp,
    Tuple6AsTuple7BinaryPredicateOp, Tuple6AsTuple7PredicateOp,
};

mod selection_control;
pub use selection_control::PrecomputedSelection;
pub(crate) use selection_control::SelectionStencil;
