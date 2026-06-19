mod backends;

#[cfg(feature = "cuda")]
pub use backends::CubeCuda;
#[cfg(feature = "hip")]
pub use backends::CubeHip;
#[cfg(feature = "wgpu")]
pub use backends::CubeWgpu;

use cubecl::prelude::*;
use std::fmt;

use crate::{device::DeviceVec, primitives::range};

pub(crate) const EMPTY_HANDLE_BYTES: usize = 16;

pub(crate) fn empty_handle<R: Runtime>(client: &ComputeClient<R>) -> cubecl::server::Handle {
    client.empty(EMPTY_HANDLE_BYTES)
}

/// CubeCL execution policy.
///
/// This keeps the public shape close to Thrust execution policies while making
/// CubeCL the backend from the first implemented algorithm.
///
/// A policy owns the backend client used for allocations, transfers, and kernel
/// launches. Client code usually creates one policy and uses it to move host
/// slices into [`DeviceVec`] storage with [`CubePolicy::to_device`].
pub struct CubePolicy<R: Runtime> {
    pub(crate) client: ComputeClient<R>,
}

impl<R: Runtime> CubePolicy<R> {
    /// Creates a policy from a CubeCL device.
    pub fn from_device(device: &R::Device) -> Self {
        Self {
            client: R::client(device),
        }
    }

    /// Returns the underlying CubeCL client.
    pub fn client(&self) -> &ComputeClient<R> {
        &self.client
    }

    pub(crate) fn empty_handle(&self) -> cubecl::server::Handle {
        empty_handle(&self.client)
    }

    /// Copies a host slice to device-resident storage.
    ///
    /// This is an explicit host-to-device transfer boundary. Algorithms operate
    /// on the returned [`DeviceVec`] or on read-only borrows of it.
    pub fn to_device<T>(&self, input: &[T]) -> Result<DeviceVec<R, T>, crate::Error>
    where
        T: CubePrimitive + CubeElement,
    {
        range::to_device(self, input)
    }

    /// Creates a device vector filled with `value`.
    ///
    /// This allocates owned device storage and initializes it on the device.
    pub fn device_filled<T>(&self, len: usize, value: T) -> Result<DeviceVec<R, T>, crate::Error>
    where
        T: CubePrimitive + CubeElement,
    {
        range::filled(self, len, value)
    }
}

impl<R: Runtime> Clone for CubePolicy<R> {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
        }
    }
}

impl<R: Runtime> fmt::Debug for CubePolicy<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CubePolicy")
            .field("runtime", &R::name(&self.client))
            .finish_non_exhaustive()
    }
}
