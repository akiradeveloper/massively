//! Parallel algorithm types, operators, and free-function APIs.

pub(crate) mod api;
pub mod op;
mod soa;
mod traits;

pub use api::{
    adjacent_difference, adjacent_find, all_of, any_of, copy_if, count_if, equal, equal_range,
    exclusive_scan, exclusive_scan_by_key, find_first_of, find_if, gather, gather_if,
    inclusive_scan, inclusive_scan_by_key, inner_product, is_partitioned, is_sorted,
    is_sorted_until, lexicographical_compare, lower_bound, max_element, merge, merge_by_key,
    min_element, minmax_element, mismatch, none_of, partition, reduce, reduce_by_key, remove_if,
    replace_if, reverse, scatter, scatter_if, set_difference, set_intersection, set_union, sort,
    sort_by_key, stable_sort, stable_sort_by_key, transform, unique, unique_by_key, upper_bound,
};
pub use soa::{SoA1, SoA2, SoA3};
pub use traits::{MItem, MIter, MVec};
