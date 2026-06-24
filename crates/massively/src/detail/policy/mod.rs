use cubecl::prelude::*;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{device::DeviceVec, primitives::range};

pub(crate) const EMPTY_HANDLE_BYTES: usize = 16;

static NEXT_POLICY_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CubePolicyId(u64);

fn next_policy_id() -> CubePolicyId {
    CubePolicyId(NEXT_POLICY_ID.fetch_add(1, Ordering::Relaxed))
}

pub(crate) fn empty_handle<R: Runtime>(client: &ComputeClient<R>) -> cubecl::server::Handle {
    client.empty(EMPTY_HANDLE_BYTES)
}

struct CubePolicyInner<R: Runtime> {
    client: ComputeClient<R>,
    id: CubePolicyId,
}

/// CubeCL execution policy.
///
/// This keeps the public shape close to Thrust execution policies while making
/// CubeCL the backend from the first implemented algorithm.
///
/// A policy is a lightweight shared handle to the backend client and owner id
/// used for allocations, transfers, and kernel launches. Client code usually
/// creates one policy and uses it to move host slices into [`DeviceVec`] storage
/// with [`CubePolicy::to_device`].
pub struct CubePolicy<R: Runtime> {
    inner: Arc<CubePolicyInner<R>>,
}

impl<R: Runtime> CubePolicy<R> {
    /// Creates a policy from a CubeCL device.
    pub fn from_device(device: &R::Device) -> Self {
        Self {
            inner: Arc::new(CubePolicyInner {
                client: R::client(device),
                id: next_policy_id(),
            }),
        }
    }

    /// Returns the underlying CubeCL client.
    pub fn client(&self) -> &ComputeClient<R> {
        &self.inner.client
    }

    pub(crate) fn id(&self) -> CubePolicyId {
        self.inner.id
    }

    pub(crate) fn empty_handle(&self) -> cubecl::server::Handle {
        empty_handle(&self.inner.client)
    }

    pub(crate) fn empty_device_vec<T>(&self) -> DeviceVec<R, T> {
        DeviceVec::from_handle(self.id(), self.empty_handle(), 0)
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
            inner: self.inner.clone(),
        }
    }
}

impl<R: Runtime> fmt::Debug for CubePolicy<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CubePolicy")
            .field("runtime", &R::name(self.client()))
            .field("id", &self.id())
            .finish_non_exhaustive()
    }
}
