use crate::{error::Error, policy::CubePolicy};
use cubecl::prelude::*;
use std::marker::PhantomData;

/// Device-resident vector storage.
///
/// This is the ownership boundary for keeping data on the CubeCL device across
/// multiple algorithm calls. Host transfer happens explicitly through
/// [`CubePolicy::to_device`] and [`DeviceVec::to_vec`].
///
/// `DeviceVec<T>` is also the one-column owned SoA form used by consuming
/// algorithms. Borrow `&DeviceVec<T>` when passing it to read-only algorithms.
pub struct DeviceVec<R: Runtime, T> {
    pub(crate) policy: CubePolicy<R>,
    pub(crate) handle: cubecl::server::Handle,
    pub(crate) len: usize,
    _element: PhantomData<fn() -> T>,
}

impl<R, T> std::fmt::Debug for DeviceVec<R, T>
where
    R: Runtime,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceVec")
            .field("policy", &self.policy)
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

impl<R, T> DeviceVec<R, T>
where
    R: Runtime,
{
    pub(crate) fn from_handle(
        policy: CubePolicy<R>,
        handle: cubecl::server::Handle,
        len: usize,
    ) -> Self {
        Self {
            policy,
            handle,
            len,
            _element: PhantomData,
        }
    }

    /// Returns the number of elements in this device vector.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether this device vector has no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the execution policy associated with this device vector.
    pub fn policy(&self) -> &CubePolicy<R> {
        &self.policy
    }
}

impl<R, T> DeviceVec<R, T>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    /// Copies this device vector back to host memory.
    ///
    /// This is an explicit device-to-host transfer boundary. It synchronizes
    /// with the backend as needed to read the current device contents.
    pub fn to_vec(&self) -> Result<Vec<T>, Error> {
        if self.len == 0 {
            return Ok(Vec::new());
        }

        let bytes = self.policy.client().read_one(self.handle.clone());
        Ok(T::from_bytes(&bytes)[..self.len].to_vec())
    }
}
