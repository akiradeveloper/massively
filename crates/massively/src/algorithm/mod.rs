//! Parallel algorithm free-function APIs.

pub(crate) mod api;

pub use crate::iter::{MIter, MIterMut, Zip1, Zip2, Zip3, Zip4, Zip5, Zip6, Zip7};
pub use crate::op;
pub use crate::value::MItem;
pub use api::{
    adjacent_difference, adjacent_find, all_of, any_of, copy_where, count_if, equal,
    exclusive_scan, exclusive_scan_by_key, fill, find_first_of, find_if, gather, gather_where,
    inclusive_scan, inclusive_scan_by_key, is_partitioned, is_sorted, is_sorted_until,
    lexicographical_compare, lower_bound, max_element, merge, merge_by_key, min_element,
    minmax_element, mismatch, none_of, partition, reduce, reduce_by_key, remove_where,
    replace_where, reverse, scatter, scatter_where, set_difference, set_intersection, set_union,
    sort, sort_by_key, transform, transform_where, unique, unique_by_key, upper_bound,
};
