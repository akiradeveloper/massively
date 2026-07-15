//! Multi-platform GPU parallel algorithms for Rust.
//!
//! Massively provides Thrust-style algorithms on top of CubeCL. Host/device transfers are
//! explicit, algorithms return owned device storage when they naturally produce a new sequence,
//! and lazy expressions can be fused into the consuming GPU kernel. Operations whose semantics
//! require an existing destination, such as scatter, take that destination as an argument.
//!
//! # Quick start
//!
//! ```
//! use cubecl::prelude::*;
//! use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
//! use massively::{Executor, op::UnaryOp, vector::transform};
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
//! let output = transform(&exec, input.slice(..), Double).unwrap();
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
//! - [`zip2`] through [`zip12`] combine columns, while [`unzip2`] through [`unzip12`] recover the
//!   individual columns; [`tuple2`] through [`tuple12`] construct their corresponding item values.
//! - [`lazy`] provides allocation-free sources and adapters.
//! - [`op`] contains reusable GPU operations such as [`op::Identity`].

mod api;
mod core;
pub mod graph;
pub mod seg;
pub mod vector;

// Crate-private compatibility aliases keep the kernel core independent from
// the public module layout. They are not part of the external API.
pub(crate) use core::allocation::{CanonicalAlloc, CanonicalStorage};
#[doc(hidden)]
pub use core::allocation::{CanonicalLeaves, FoldCanonical};
pub(crate) use core::arity::{A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13};
pub(crate) use core::read::{
    Column, Constant, Counting, Permute, ReadExpression, ReverseCounting, Taken, Transform,
};
pub(crate) use core::reduce::Dispatch;
pub(crate) use core::storage::{S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, StorageLayout};
pub(crate) use core::transform::materialize;
pub(crate) use core::value::MStorageElement;
pub(crate) use core::{
    allocation, arg_reduce, arity, eval, indexed, launch, masked, merge, ordering, output,
    predicate, read, reduce, scan, search, segmented, selection, storage, transform, value,
};

pub use api::iter::{
    Allocable, Canonicalizable, MItem, MIter, MIterMut, MStorage, MVec, Materializable,
    WritableFrom, Zip, unzip2, unzip3, unzip4, unzip5, unzip6, unzip7, unzip8, unzip9, unzip10,
    unzip11, unzip12, zip2, zip3, zip4, zip5, zip6, zip7, zip8, zip9, zip10, zip11, zip12,
};
#[doc(hidden)]
pub use api::runtime::{DeviceSlice, DeviceSliceMut, DeviceVec, Executor};
pub use api::tuple::{
    Tuple2, Tuple3, Tuple4, Tuple5, Tuple6, Tuple7, Tuple8, Tuple9, Tuple10, Tuple11, Tuple12,
    flatten3, flatten4, flatten5, flatten6, flatten7, flatten8, flatten9, flatten10, flatten11,
    flatten12, tuple2, tuple3, tuple4, tuple5, tuple6, tuple7, tuple8, tuple9, tuple10, tuple11,
    tuple12,
};
pub use api::{Error, MIndex, lazy, op, util};
/// Common public data and operation types.
pub mod prelude {
    pub use crate::{
        Allocable, Canonicalizable, DeviceSlice, DeviceSliceMut, DeviceVec, Executor, MIndex,
        MItem, MIter, MIterMut, MStorage, MVec, Materializable, Tuple2, Tuple3, Tuple4, Tuple5,
        Tuple6, Tuple7, Tuple8, Tuple9, Tuple10, Tuple11, Tuple12, Zip, flatten3, flatten4,
        flatten5, flatten6, flatten7, flatten8, flatten9, flatten10, flatten11, flatten12, tuple2,
        tuple3, tuple4, tuple5, tuple6, tuple7, tuple8, tuple9, tuple10, tuple11, tuple12, unzip2,
        unzip3, unzip4, unzip5, unzip6, unzip7, unzip8, unzip9, unzip10, unzip11, unzip12, zip2,
        zip3, zip4, zip5, zip6, zip7, zip8, zip9, zip10, zip11, zip12,
    };
}
