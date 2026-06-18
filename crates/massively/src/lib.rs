//! Multi-platform GPU parallel algorithms for Rust.
#![allow(private_interfaces)]
//!
//! `massively` provides Thrust-style free-function algorithms over
//! device-resident data. Backends are CubeCL runtimes, so the same API can target
//! WGPU, CUDA, and HIP where the corresponding CubeCL backend is available.
//!
//! # Memory Boundaries
//!
//! Host/device transfers are explicit. Use [`CubePolicy::to_device`] to copy a
//! host slice to the device, and [`DeviceVec::to_vec`] to copy device storage
//! back to the host.
//!
//! # Data Model
//!
//! A [`DeviceVec`] is one owned GPU column. Algorithms read borrowed SoA inputs:
//! borrow one column with `&DeviceVec`, or combine several borrowed columns with
//! [`zip`].
//!
//! Algorithm outputs are owned device storage: a [`DeviceVec`] for one output
//! column, or a tuple of [`DeviceVec`] columns for multi-column output.
//!
//! # Example
//!
//! ```no_run
//! use massively::{CubeWgpu, reduce, transform, zip3};
//!
//! struct Sum;
//! #[cubecl::cube]
//! impl massively::op::BinaryOp<f32> for Sum {
//!     fn apply(lhs: f32, rhs: f32) -> f32 {
//!         lhs + rhs
//!     }
//! }
//!
//! struct KineticEnergy;
//! #[cubecl::cube]
//! impl massively::op::UnaryOp<(f32, f32, f32)> for KineticEnergy {
//!     type Output = f32;
//!
//!     fn apply(input: (f32, f32, f32)) -> f32 {
//!         0.5 * (input.0 * input.0 + input.1 * input.1 + input.2 * input.2)
//!     }
//! }
//!
//! # fn main() -> Result<(), massively::Error> {
//! let policy = CubeWgpu::new();
//! let vx = policy.to_device(&[1.0_f32, 0.0, 2.0])?;
//! let vy = policy.to_device(&[0.0_f32, 2.0, 0.0])?;
//! let vz = policy.to_device(&[0.0_f32, 0.0, 2.0])?;
//!
//! let energy = transform(zip3(&vx, &vy, &vz), KineticEnergy)?;
//! let total = reduce(&energy, 0.0, Sum)?;
//!
//! assert_eq!(energy.to_vec()?, vec![0.5, 2.0, 4.0]);
//! assert_eq!(total, 6.5);
//! # Ok(())
//! # }
//! ```

mod api;
mod device;
mod error;
mod expr;
mod kernels;
pub mod op;
mod policy;
mod primitives;

pub use api::{
    adjacent_difference, adjacent_find, all_of, any_of, copy_if, count_if, equal, equal_range,
    exclusive_scan, exclusive_scan_by_key, find_first_of, find_if, gather, gather_if,
    inclusive_scan, inclusive_scan_by_key, inner_product, is_partitioned, is_sorted,
    is_sorted_until, lexicographical_compare, lower_bound, max_element, merge, merge_by_key,
    min_element, minmax_element, mismatch, none_of, partition, reduce, reduce_by_key, remove_if,
    replace_if, reverse, scatter, scatter_if, set_difference, set_intersection, set_union, sort,
    sort_by_key, stable_sort, stable_sort_by_key, transform, unique, unique_by_key, upper_bound,
    zip, zip3, zip4, zip5, zip6, zip7, zip8, zip9, zip10, zip11, zip12,
};
pub use device::{DeviceVec, SoA};
pub use error::Error;
pub use policy::CubePolicy;

#[cfg(feature = "cuda")]
pub use policy::CubeCuda;

#[cfg(feature = "hip")]
pub use policy::CubeHip;

#[cfg(feature = "wgpu")]
pub use policy::CubeWgpu;

/// Prelude for SoA composition helpers.
pub mod prelude {
    pub use crate::{zip, zip3, zip4, zip5, zip6, zip7, zip8, zip9, zip10, zip11, zip12};
}
