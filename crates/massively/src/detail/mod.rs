#![allow(unused_unsafe)]

pub(crate) mod api;
pub(crate) mod control;
pub(crate) mod device;
pub(crate) mod dispatch;
pub(crate) mod expr;
pub(crate) mod impls;
pub(crate) mod kernels;
pub(crate) mod op_adapter;
pub(crate) mod policy;
pub(crate) mod primitives;
pub(crate) mod read;

pub(crate) use crate::op;
pub(crate) use api::{
    MItemStorage, MaterializeOutput, TransformSoA2Output, TransformSoA3Output, TransformSoA4Output,
    TransformSoA5Output, TransformSoA6Output, TransformSoA7Output, TransformUnaryOutput,
    adjacent_difference, adjacent_find, all_of, any_of, copy_where, count_if, equal,
    exclusive_scan, exclusive_scan_by_key, find_first_of, find_if, inclusive_scan,
    inclusive_scan_by_key, is_partitioned, is_sorted, is_sorted_until, lexicographical_compare,
    lower_bound_many, max_element, merge, merge_by_key, min_element, minmax_element, mismatch,
    none_of, partition, reduce, reduce_by_key, remove_if, replace_where, reverse, set_difference,
    set_intersection, set_union, sort, sort_by_key, unique, unique_by_key, upper_bound_many,
};
pub(crate) use device::DeviceVec;
pub(crate) use policy::CubePolicy;
