use super::*;

mod collect;
mod indexed;
mod scan;
mod search;
pub(in crate::detail::api) mod selection;

pub use collect::device_expr_collect_into_with_policy;
pub(in crate::detail) use collect::{device_expr_collect_with_policy, device_expr_reverse_collect};

pub use indexed::{device_expr_gather_into_with_policy, device_expr_scatter_into_with_policy};
pub(in crate::detail) use indexed::{
    device_expr_gather_where_into_with_control, device_expr_gather_with_policy,
    device_expr_scatter_where_into_with_control,
};

#[allow(unused_imports)]
pub(in crate::detail) use scan::{
    device_expr_adjacent_difference_with_policy,
    device_expr_exclusive_scan_by_key_expr_keys_with_policy,
    device_expr_inclusive_scan_by_key_expr_keys_with_policy,
};

pub(in crate::detail) use search::device_expr_minmax_element_with_policy;

pub use selection::device_expr_copy_where_into_with_policy;
pub(in crate::detail) use selection::{
    device_expr_count_if_with_policy, device_expr_find_if_with_policy,
    device_expr_selected_rank_with_policy, device_expr_selection_flags_with_policy,
    replace_where_into_with_control,
};

pub(super) fn offset_handle<R: Runtime>(
    client: &ComputeClient<R>,
    offset: usize,
) -> Result<cubecl::server::Handle, Error> {
    let offset = u32::try_from(offset).map_err(|_| Error::LengthTooLarge { len: offset })?;
    Ok(client.create_from_slice(u32::as_bytes(&[offset])))
}
