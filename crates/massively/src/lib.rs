//! Multi-platform GPU parallel algorithms for Rust.
#![allow(private_interfaces)]
//!
//! `massively` is a Thrust-inspired algorithm layer on top of CubeCL.
//!
//! The crate is organized around three public layers:
//!
//! - [`runtime`] prepares a CubeCL runtime from a device, owns host/device
//!   transfers, and manages device memory such as [`DeviceVec`],
//!   [`DeviceSlice`], and [`DeviceSliceMut`].
//! - [`iter`] provides Structure-of-Arrays inputs and massively iterator
//!   traits.
//! - [`value`] provides massively item/vector traits.
//! - [`op`] provides CubeCL-backed operation traits.
//! - [`algorithm`] provides parallel algorithms such as [`transform`],
//!   [`reduce`], and [`sort`].
//! - [`util`] provides helper facilities such as GPU-side random columns.
//!
//! User-defined operations are written as CubeCL cube traits. Low-level CubeCL
//! launch and storage details remain internal implementation details.

pub mod algorithm;
mod detail;
mod error;
pub mod iter;
pub mod op;
pub mod runtime;
pub mod slice;
pub mod util;
pub mod value;

pub use algorithm::{
    adjacent_difference, adjacent_find, all_of, any_of, copy_where, count_if, equal, equal_range,
    exclusive_scan, exclusive_scan_by_key, find_first_of, find_if, gather, gather_where,
    inclusive_scan, inclusive_scan_by_key, inner_product, is_partitioned, is_sorted,
    is_sorted_until, lexicographical_compare, lower_bound, max_element, merge, merge_by_key,
    min_element, minmax_element, mismatch, none_of, partition, reduce, reduce_by_key, remove_where,
    replace_where, reverse, scatter, scatter_where, set_difference, set_intersection, set_union,
    sort, sort_by_key, stable_sort, stable_sort_by_key, transform, transform_where, unique,
    unique_by_key, upper_bound,
};
pub use error::Error;
pub use iter::{MIter, MIterMut, SoA1, SoA2, SoA3};
pub use runtime::{DeviceSlice, DeviceSliceMut, DeviceVec, Executor, Scalar};
pub use slice::MSlice;
pub use value::{MItem, MVec};

/// Common facade traits and types.
///
/// Algorithm functions are intentionally not included; call them through the
/// `massively::` namespace.
pub mod prelude {
    pub use crate::{
        DeviceSlice, DeviceSliceMut, DeviceVec, Executor, MIter, MIterMut, MSlice, MVec, SoA1,
        SoA2, SoA3,
    };
}

pub(crate) use detail::{device, expr, kernels, policy, primitives};
