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
//! - [`iter`] provides Zip inputs and massively iterator traits.
//! - [`value`] provides massively item traits.
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
mod index;
pub mod iter;
pub mod lazy;
pub mod op;
pub mod runtime;
pub mod util;
pub mod value;

pub use algorithm::{
    adjacent_difference, adjacent_find, all_of, any_of, copy_where, count_if, equal,
    exclusive_scan, exclusive_scan_by_key, fill, find_first_of, find_if, gather, gather_where,
    inclusive_scan, inclusive_scan_by_key, is_partitioned, is_sorted, is_sorted_until,
    lexicographical_compare, lower_bound, max_element, merge, merge_by_key, min_element,
    minmax_element, mismatch, none_of, partition, reduce, reduce_by_key, remove_where,
    replace_where, reverse, scatter, scatter_where, set_difference, set_intersection, set_union,
    sort, sort_by_key, transform, transform_where, unique, unique_by_key, upper_bound,
};
pub use error::Error;
pub use index::MIndex;
pub use iter::{MIter, MIterMut, MIterReduce, MStorage, Zip1, Zip2, Zip3, Zip4, Zip5, Zip6, Zip7};
pub use runtime::{DeviceSlice, DeviceSliceMut, DeviceVec, Executor};
pub use value::{MAlloc, MItem, MStorageElement};

/// Common facade traits and types.
///
/// Algorithm functions are intentionally not included; call them through the
/// `massively::` namespace.
pub mod prelude {
    pub use crate::{
        DeviceSlice, DeviceSliceMut, DeviceVec, Executor, MAlloc, MIndex, MItem, MIter, MIterMut,
        MIterReduce, MStorage, MStorageElement, Zip1, Zip2, Zip3, Zip4, Zip5, Zip6, Zip7,
    };
}

pub(crate) use detail::{device, expr, kernels, policy, primitives};
