use crate::{
    error::Error,
    index::{IntoMIndex, MIndex, usize_from_mindex},
    policy::{CubePolicy, CubePolicyId},
};
use cubecl::prelude::*;
use std::marker::PhantomData;

/// Device-resident vector storage.
///
/// This is the ownership boundary for keeping data on the CubeCL device across
/// multiple algorithm calls. The vector carries only storage identity; host
/// transfers and kernel launches use an explicitly supplied executor.
///
/// Algorithms take `DeviceSlice<T>` as a one-column Zip input and return
/// `DeviceVec<T>` as one-column owned output storage.
#[derive(Clone)]
pub struct DeviceVec<R: Runtime, T> {
    pub(crate) policy_id: CubePolicyId,
    pub(crate) handle: cubecl::server::Handle,
    pub(crate) len: MIndex,
    _element: PhantomData<fn() -> (R, T)>,
}

impl<R, T> std::fmt::Debug for DeviceVec<R, T>
where
    R: Runtime,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceVec")
            .field("policy_id", &self.policy_id)
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

impl<R, T> DeviceVec<R, T>
where
    R: Runtime,
{
    pub(crate) fn from_handle(
        policy_id: CubePolicyId,
        handle: cubecl::server::Handle,
        len: impl IntoMIndex,
    ) -> Self {
        let len = len.into_mindex();
        Self {
            policy_id,
            handle,
            len,
            _element: PhantomData,
        }
    }

    /// Returns the number of elements in this device vector.
    pub fn len(&self) -> usize {
        usize_from_mindex(self.len)
    }

    pub fn mindex_len(&self) -> MIndex {
        self.len
    }

    /// Returns whether this device vector has no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub(crate) fn policy_id(&self) -> CubePolicyId {
        self.policy_id
    }
}

impl<R, T> DeviceVec<R, T>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    pub(crate) fn read_to_host(&self, policy: &CubePolicy<R>) -> Result<Vec<T>, Error> {
        if self.len == 0 {
            return Ok(Vec::new());
        }

        let bytes = policy
            .client()
            .read_one(self.handle.clone())
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
        Ok(T::from_bytes(&bytes)[..usize_from_mindex(self.len)].to_vec())
    }
}
