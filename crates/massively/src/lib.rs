//! Multi-platform GPU parallel algorithms for Rust.
//!
//! Massively provides Thrust-style algorithms on top of CubeCL. Host/device transfers are
//! explicit, algorithm outputs are preallocated, and lazy expressions can be fused into the
//! consuming GPU kernel.
//!
//! # Quick start
//!
//! ```
//! use cubecl::prelude::*;
//! use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
//! use massively::{Executor, UnaryOp, transform};
//!
//! struct Double;
//!
//! #[cubecl::cube]
//! impl UnaryOp<u32> for Double {
//!     type Output = u32;
//!
//!     fn apply(value: u32) -> u32 {
//!         value * 2
//!     }
//! }
//!
//! let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
//! let input = exec.to_device(&[1_u32, 2, 3, 4]);
//! let output = exec.alloc::<u32>(input.len());
//!
//! transform(&exec, input.slice(..), Double, output.slice_mut(..)).unwrap();
//!
//! assert_eq!(exec.to_host(&output).unwrap(), vec![2, 4, 6, 8]);
//! ```
//!
//! # Core concepts
//!
//! - [`Executor`] owns the GPU runtime client and provides allocation and transfer methods.
//! - [`DeviceVec`], [`DeviceSlice`], and [`DeviceSliceMut`] are the owning and borrowed device
//!   containers used at API boundaries.
//! - [`MIter`] and [`MIterMut`] are the public iterator capabilities accepted by algorithms.
//! - [`zip2`] through [`zip7`] combine columns; [`tuple2`] through [`tuple7`] construct their
//!   corresponding item values.
//! - [`lazy`] provides allocation-free sources and adapters.
//! - [`op`] contains reusable GPU operations such as [`op::Identity`].

mod api;
mod core;

// Crate-private compatibility aliases keep the kernel core independent from
// the public module layout. They are not part of the external API.
pub(crate) use core::allocation::{CanonicalAlloc, CanonicalStorage};
pub(crate) use core::arity::{A1, A2, A3, A4, A5, A6, A7, A8};
pub(crate) use core::read::{
    Column, Constant, Counting, Permute, ReadExpression, ReverseCounting, Taken, Transform,
};
pub(crate) use core::reduce::Dispatch;
pub(crate) use core::storage::{S1, S2, S3, S4, S5, S6, S7, StorageLayout};
pub(crate) use core::transform::materialize;
pub(crate) use core::value::MStorageElement;
pub(crate) use core::{
    allocation, arg_reduce, arity, eval, indexed, launch, masked, merge, ordering, output,
    predicate, read, reduce, scan, search, segmented, selection, storage, transform, value,
};

pub use api::algorithm::*;
pub use api::iter::{
    MAlloc, MItem, MIter, MIterMut, MStorage, WriteFrom, Zip, zip2, zip3, zip4, zip5, zip6, zip7,
};
pub use api::op::{BinaryPredicateOp, PredicateOp, ReductionOp, UnaryOp};
pub use api::runtime::{DeviceSlice, DeviceSliceMut, DeviceVec, Executor};
pub use api::tuple::{
    Tuple2, Tuple3, Tuple4, Tuple5, Tuple6, Tuple7, flatten3, flatten4, flatten5, flatten6,
    flatten7, tuple2, tuple3, tuple4, tuple5, tuple6, tuple7,
};
pub use api::{Error, MIndex, lazy, op, util};
/// Common public data and operation types. Algorithms remain at crate root.
pub mod prelude {
    pub use crate::{
        DeviceSlice, DeviceSliceMut, DeviceVec, Executor, MAlloc, MIndex, MItem, MIter, MIterMut,
        MStorage, Tuple2, Tuple3, Tuple4, Tuple5, Tuple6, Tuple7, Zip, flatten3, flatten4,
        flatten5, flatten6, flatten7, tuple2, tuple3, tuple4, tuple5, tuple6, tuple7, zip2, zip3,
        zip4, zip5, zip6, zip7,
    };
}
