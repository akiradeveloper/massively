//! CPU reference implementations used by `massively` property tests.
//!
//! This crate is intentionally small and test-oriented. Implementations are
//! fixed to `u32` first so property tests can compare GPU algorithms against a
//! deterministic reference without carrying generic API complexity.

mod adjacent_difference;
mod adjacent_find;
mod all_of;
mod any_of;
mod copy_where;
mod count_if;
mod equal;
mod equal_range;
mod exclusive_scan;
mod exclusive_scan_by_key;
mod find_first_of;
mod find_if;
mod gather;
mod gather_where;
mod inclusive_scan;
mod inclusive_scan_by_key;
mod inner_product;
mod is_partitioned;
mod is_sorted;
mod is_sorted_until;
mod lexicographical_compare;
mod lower_bound;
mod max_element;
mod merge;
mod merge_by_key;
mod min_element;
mod minmax_element;
mod mismatch;
mod none_of;
mod partition;
mod reduce;
mod reduce_by_key;
mod remove_where;
mod replace_where;
mod reverse;
mod scatter;
mod scatter_where;
mod set_difference;
mod set_intersection;
mod set_union;
mod sort;
mod sort_by_key;
mod transform;
mod unique;
mod unique_by_key;
mod upper_bound;

pub use adjacent_difference::adjacent_difference;
pub use adjacent_find::adjacent_find;
pub use all_of::all_of;
pub use any_of::any_of;
pub use copy_where::copy_where;
pub use count_if::count_if;
pub use equal::equal;
pub use equal_range::equal_range;
pub use exclusive_scan::exclusive_scan;
pub use exclusive_scan_by_key::exclusive_scan_by_key;
pub use find_first_of::find_first_of;
pub use find_if::find_if;
pub use gather::gather;
pub use gather_where::gather_where;
pub use inclusive_scan::inclusive_scan;
pub use inclusive_scan_by_key::inclusive_scan_by_key;
pub use inner_product::inner_product;
pub use is_partitioned::is_partitioned;
pub use is_sorted::is_sorted;
pub use is_sorted_until::is_sorted_until;
pub use lexicographical_compare::lexicographical_compare;
pub use lower_bound::lower_bound;
pub use max_element::max_element;
pub use merge::merge;
pub use merge_by_key::merge_by_key;
pub use min_element::min_element;
pub use minmax_element::minmax_element;
pub use mismatch::mismatch;
pub use none_of::none_of;
pub use partition::partition;
pub use reduce::reduce;
pub use reduce_by_key::reduce_by_key;
pub use remove_where::remove_where;
pub use replace_where::replace_where;
pub use reverse::reverse;
pub use scatter::scatter;
pub use scatter_where::scatter_where;
pub use set_difference::set_difference;
pub use set_intersection::set_intersection;
pub use set_union::set_union;
pub use sort::sort;
pub use sort_by_key::sort_by_key;
pub use transform::transform;
pub use unique::unique;
pub use unique_by_key::unique_by_key;
pub use upper_bound::upper_bound;

/// Fixed unary operation used by `transform` property tests.
pub fn xor_mask(x: u32) -> u32 {
    x ^ 0x5a5a_5a5a
}

/// Fixed associative operation used by reduce and scan property tests.
pub fn max_op(lhs: u32, rhs: u32) -> u32 {
    lhs.max(rhs)
}

/// Fixed unary predicate used by selection-style property tests.
pub fn keep(value: u32) -> bool {
    (value & 1) == 0
}

/// Fixed equivalence relation used by equality-style property tests.
pub fn same_low_nibble(lhs: u32, rhs: u32) -> bool {
    (lhs & 0x0f) == (rhs & 0x0f)
}

/// Fixed strict weak ordering used by ordering-style property tests.
pub fn bucket_then_value_less(lhs: u32, rhs: u32) -> bool {
    let lhs_key = lhs & 0x0f;
    let rhs_key = rhs & 0x0f;
    lhs_key < rhs_key || (lhs_key == rhs_key && lhs < rhs)
}
