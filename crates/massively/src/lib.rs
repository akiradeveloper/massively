//! Multi-platform GPU parallel algorithms for Rust.
#![allow(private_interfaces)]
//!
//! `massively` is a Thrust-inspired algorithm layer on top of CubeCL.
//!
//! The crate is organized around two public layers:
//!
//! - [`runtime`] prepares the execution backend, owns host/device transfers, and
//!   manages device memory such as [`DeviceVec`], [`DeviceSlice`], and
//!   [`DeviceSliceMut`].
//! - [`algorithm`] provides Structure-of-Arrays inputs, massively item/vector
//!   traits, CubeCL-backed operation traits, and parallel algorithms such as
//!   [`transform`], [`reduce`], and [`sort`].
//!
//! User-defined operations are written as CubeCL cube traits. Low-level CubeCL
//! runtime, launch, and storage details remain internal implementation details.

pub mod algorithm;
mod detail;
mod error;
pub mod runtime;

pub use algorithm::op;
pub use algorithm::{
    MItem, MIter, MVec, SoA1, SoA2, SoA3, adjacent_difference, adjacent_find, all_of, any_of,
    copy_if, count_if, equal, equal_range, exclusive_scan, exclusive_scan_by_key, find_first_of,
    find_if, gather, gather_if, inclusive_scan, inclusive_scan_by_key, inner_product,
    is_partitioned, is_sorted, is_sorted_until, lexicographical_compare, lower_bound, max_element,
    merge, merge_by_key, min_element, minmax_element, mismatch, none_of, partition, reduce,
    reduce_by_key, remove_if, replace_if, reverse, scatter, scatter_if, set_difference,
    set_intersection, set_union, sort, sort_by_key, stable_sort, stable_sort_by_key, transform,
    unique, unique_by_key, upper_bound,
};
pub use error::Error;
#[cfg(feature = "wgpu")]
pub use runtime::Wgpu;
pub use runtime::{Backend, DeviceSlice, DeviceSliceMut, DeviceVec, Executor, Scalar};

/// Common facade traits and types.
///
/// Algorithm functions are intentionally not included; call them through the
/// `massively::` namespace.
pub mod prelude {
    pub use crate::{
        Backend, DeviceSlice, DeviceSliceMut, DeviceVec, Executor, MIter, MVec, SoA1, SoA2, SoA3,
    };
}

pub(crate) use detail::{device, expr, kernels, policy, primitives};
