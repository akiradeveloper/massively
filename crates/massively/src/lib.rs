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
//! A [`DeviceVec`] is one owned GPU column. Algorithms read borrowed MIter
//! inputs: borrow one column as `(&xs,)`, or combine several borrowed columns as
//! a tuple such as `(&xs, &ys, &zs)`.
//!
//! Algorithm outputs are owned device storage: a [`DeviceVec`] for one output
//! column, or a tuple of [`DeviceVec`] columns for multi-column output.

mod api;
mod detail;

pub use api::*;
pub use detail::op;

pub(crate) use detail::{device, error, expr, kernels, policy, primitives};
