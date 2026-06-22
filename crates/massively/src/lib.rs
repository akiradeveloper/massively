//! Multi-platform GPU parallel algorithms for Rust.
#![allow(private_interfaces)]
//!
//! `massively` provides Thrust-style free-function algorithms over
//! device-resident data. Backends are CubeCL runtimes, so the same API can target
//! WGPU, CUDA, and HIP where the corresponding CubeCL backend is available.
//!
//! # Memory Boundaries
//!
//! Host/device transfers are explicit. Use [`Executor::to_device`] to copy a host
//! slice to the device, and [`Executor::to_host`] to copy device storage back to
//! the host. Since v0.7, algorithms take `&Executor<B>` as their first argument;
//! device storage does not own the execution context used for launches or reads.
//!
//! # Data Model
//!
//! A [`DeviceVec`] is one owned GPU column. Algorithms read borrowed
//! [`DeviceSlice`] inputs: borrow one column as `SoA1(xs.slice(..))`, or combine
//! several borrowed columns as `SoA2(xs.slice(..), ys.slice(..))` or
//! `SoA3(xs.slice(..), ys.slice(..), zs.slice(..))`.
//!
//! Algorithm outputs are owned device storage: a [`DeviceVec`] for one output
//! column, or a tuple of [`DeviceVec`] columns for multi-column output.
//!
//! # v0.7 By-Key Shape
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
//! Tuple keys are intentionally not part of the v0.7 public by-key API:
//!
//! ```compile_fail
//! use cubecl::prelude::*;
//! use massively::{Executor, Wgpu, sort_by_key};
//!
//! struct TupleLess;
//!
//! #[cubecl::cube]
//! impl massively::op::PredicateOp2<Wgpu, (u32, u32)> for TupleLess {
//!     fn apply(lhs: (u32, u32), rhs: (u32, u32)) -> bool {
//!         lhs.0 < rhs.0 || (lhs.0 == rhs.0 && lhs.1 < rhs.1)
//!     }
//! }
//!
//! fn main() -> Result<(), massively::Error> {
//!     let exec = Executor::<Wgpu>::cpu();
//!     let key_a = exec.to_device(&[1_u32, 2, 3])?;
//!     let key_b = exec.to_device(&[10_u32, 20, 30])?;
//!     let values = exec.to_device(&[100_u32, 200, 300])?;
//!
//!     let _ = sort_by_key::<Wgpu, _, _, _, _, _, _>(
//!         &exec,
//!         massively::SoA2(key_a.slice(..), key_b.slice(..)),
//!         massively::SoA1(values.slice(..)),
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

/// Common facade traits and types.
///
/// Algorithm functions are intentionally not included; call them through the
/// `massively::` namespace.
pub mod prelude {
    pub use crate::api::{
        Backend, DeviceSlice, DeviceVec, Executor, MIter, MVec, SoA1, SoA2, SoA3,
    };
}

pub(crate) use detail::{device, error, expr, kernels, policy, primitives};
