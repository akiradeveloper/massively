#![allow(unused_unsafe)]

pub(crate) mod api;
pub(crate) mod device;
pub(crate) mod error;
pub(crate) mod expr;
pub(crate) mod kernels;
pub mod op;
pub(crate) mod policy;
pub(crate) mod primitives;

pub(crate) use api::{
    MaterializeOutput, StorageOutput, TransformSoA2Output, TransformSoA3Output,
    TransformUnaryOutput, adjacent_difference, adjacent_find, all_of, any_of, copy_if, count_if,
    equal, equal_range, exclusive_scan, exclusive_scan_by_key, find_first_of, find_if, gather,
    gather_if, inclusive_scan, inclusive_scan_by_key, inner_product, is_partitioned, is_sorted,
    is_sorted_until, lexicographical_compare, lower_bound, max_element, merge, merge_by_key,
    min_element, minmax_element, mismatch, none_of, partition, reduce, reduce_by_key, remove_if,
    replace_if, reverse, scatter, scatter_if, set_difference, set_intersection, set_union, sort,
    sort_by_key, unique, unique_by_key, upper_bound,
};
pub(crate) use device::DeviceVec;
pub use error::Error;
pub(crate) use policy::CubePolicy;

#[cfg(feature = "cuda")]
pub(crate) use policy::CubeCuda;

#[cfg(feature = "hip")]
pub(crate) use policy::CubeHip;

#[cfg(feature = "wgpu")]
pub(crate) use policy::CubeWgpu;
