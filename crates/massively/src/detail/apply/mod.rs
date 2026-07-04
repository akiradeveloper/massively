use crate::{
    detail::api::{
        SelectionStencil, device_expr_count_if_with_policy, device_expr_find_if_with_policy,
        device_expr_merge_by_key_values_with_control_with_policy,
        device_expr_minmax_element_with_policy, expr,
    },
    detail::op::kernel::{BinaryPredicateOp, PredicateOp},
    device::{DeviceColumnMutView, DeviceVec, KernelColumn, KernelColumnAt, S0},
    error::{Error, ensure_same_len},
    expr::{DeviceGpuExpr, GpuExpr},
    index::{MIndex, mindex_from_usize},
    primitives::{search as primitive_search, select},
};
use cubecl::prelude::*;

mod mask;
mod materialize;
mod merge;
mod ordering;
mod permutation;
mod query;
mod range;
mod reduce;
mod scan;
mod search;
mod selection;
mod transform;

pub(in crate::detail) use mask::{MaskWriteApply, MaskedIndexedExprApply};
pub(in crate::detail) use materialize::{
    FillWriteApply, MaterializePayloadApply, MaterializeWriteApply,
};
pub(in crate::detail) use merge::MergePayloadApply;
pub(in crate::detail) use ordering::{
    MergeByKeyControlApply, MergeExprApply, SetMembershipControlApply, SortApply, SortByKeyApply,
};
pub(in crate::detail) use permutation::{
    IndexedExprApply, IndexedWriteApply, PermutationPayloadApply,
};
pub(in crate::detail) use query::QueryApply;
pub(in crate::detail) use range::{ConcatPayloadApply, RangePayloadApply};
pub(in crate::detail) use reduce::{LinearReduceApply, SegmentedReduceApply};
pub(in crate::detail) use scan::{LinearScanApply, SegmentedScanApply};
pub(in crate::detail) use search::{
    SearchControlApply, SearchPayloadApply, TupleSearchPayloadApply,
};
pub(in crate::detail) use selection::{SelectedPayloadApply, SplitPayloadApply};
pub(in crate::detail) use transform::TransformPayloadApply;
