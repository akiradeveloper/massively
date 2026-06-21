//! Multi-platform GPU parallel algorithms for Rust.
#![allow(private_interfaces)]
//!
//! `massively` provides Thrust-style free-function algorithms over
//! device-resident data. Backends are CubeCL runtimes, so the same API can target
//! WGPU, CUDA, and HIP where the corresponding CubeCL backend is available.
//!
//! # Memory Boundaries
//!
//! Host/device transfers are explicit. Use [`Policy::to_device`] to copy a host
//! slice to the device, and [`DeviceVec::to_vec`] to copy device storage back to
//! the host.
//!
//! # Data Model
//!
//! A [`DeviceVec`] is one owned GPU column. Algorithms read borrowed
//! [`DeviceSlice`] inputs: borrow one column as `(xs.slice(..),)`, or combine
//! several borrowed columns as a tuple such as
//! `(xs.slice(..), ys.slice(..), zs.slice(..))`.
//!
//! Algorithm outputs are owned device storage: a [`DeviceVec`] for one output
//! column, or a tuple of [`DeviceVec`] columns for multi-column output.
//!
//! # v0.6 By-Key Shape
//!
//! By-key algorithms accept one key column and one or more value columns. If an
//! application needs a compound key, build a single key column first and pass
//! that key to the by-key algorithm.
//!
//! Stencil algorithms such as [`copy_if`], [`replace_if`], [`gather_if`], and
//! [`scatter_if`] accept one `u32` flag column. A flag value of `0` is false;
//! any non-zero value is true. Predicate markers remain the API for
//! [`remove_if`], [`count_if`], [`find_if`], and partition-style queries.
//!
//! Tuple keys are intentionally not part of the v0.6 public by-key API:
//!
//! ```compile_fail
//! use cubecl::prelude::*;
//! use massively::{CubeWgpu, sort_by_key};
//!
//! struct TupleLess;
//!
//! #[cubecl::cube]
//! impl massively::op::BinaryPredicateOp<(u32, u32)> for TupleLess {
//!     fn apply(lhs: (u32, u32), rhs: (u32, u32)) -> bool {
//!         lhs.0 < rhs.0 || (lhs.0 == rhs.0 && lhs.1 < rhs.1)
//!     }
//! }
//!
//! fn main() -> Result<(), massively::Error> {
//!     let policy = CubeWgpu::cpu();
//!     let key_a = policy.to_device(&[1_u32, 2, 3])?;
//!     let key_b = policy.to_device(&[10_u32, 20, 30])?;
//!     let values = policy.to_device(&[100_u32, 200, 300])?;
//!
//!     let _ = sort_by_key::<CubeWgpu, _, _, _, _, _, _>(
//!         (key_a.slice(..), key_b.slice(..)),
//!         (values.slice(..),),
//!         TupleLess,
//!     )?;
//!
//!     Ok(())
//! }
//! ```

mod api;
mod detail;

pub use api::*;
pub use detail::op;

pub(crate) use detail::{device, error, expr, kernels, policy, primitives};
