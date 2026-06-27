//! Runtime setup and device memory types.

mod device_vec;
mod executor;

pub use device_vec::{DeviceSlice, DeviceSliceMut, DeviceVec};
pub use executor::{Executor, Scalar, ToHost};
