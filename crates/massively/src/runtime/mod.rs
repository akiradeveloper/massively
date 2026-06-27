//! Runtime setup, device memory types, and memory initialization operations.

mod device_vec;
mod executor;
pub mod op;

pub use device_vec::{DeviceSlice, DeviceSliceMut, DeviceVec};
pub use executor::{Executor, Scalar, ToHost};
