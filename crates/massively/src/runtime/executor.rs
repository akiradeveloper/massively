//! Execution context shared by runtime memory and algorithms.

use std::marker::PhantomData;

use cubecl::prelude::{CubeElement, CubePrimitive, Runtime};

use crate::Error;
use crate::algorithm::api::sealed;
use crate::runtime::op::KernelTabulateOp;
use crate::runtime::{DeviceSlice, DeviceSliceMut, DeviceVec};

/// Scalar value that can be stored in one device column.
pub trait Scalar: CubePrimitive + CubeElement {}
impl<T> Scalar for T where T: CubePrimitive + CubeElement {}

/// Device-resident data that can be copied back to host memory by an executor.
pub trait ToHost<B: Runtime>:
    sealed::ToHostDispatch<B, Output = <Self as ToHost<B>>::Output>
{
    type Output;
}

impl<B, T> ToHost<B> for T
where
    B: Runtime,
    T: sealed::ToHostDispatch<B>,
{
    type Output = <T as sealed::ToHostDispatch<B>>::Output;
}

/// Execution context for a CubeCL runtime.
#[derive(Debug)]
pub struct Executor<B: Runtime> {
    pub(crate) inner: crate::detail::CubePolicy<B>,
    pub(crate) _backend: PhantomData<fn() -> B>,
}

impl<B: Runtime> Clone for Executor<B> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            _backend: PhantomData,
        }
    }
}

impl<B: Runtime> Executor<B> {
    /// Creates an executor for the given CubeCL device.
    pub fn new(device: B::Device) -> Self {
        Self::from_device(&device)
    }

    /// Creates an executor for the given CubeCL device reference.
    pub fn from_device(device: &B::Device) -> Self {
        Self {
            inner: crate::detail::CubePolicy::from_device(device),
            _backend: PhantomData,
        }
    }

    pub(crate) fn ensure_policy_id(
        &self,
        id: crate::detail::policy::CubePolicyId,
    ) -> Result<(), Error> {
        if self.inner.id() == id {
            Ok(())
        } else {
            Err(Error::Launch {
                message: "executor does not own this device data".to_string(),
            })
        }
    }

    pub(crate) fn policy(&self) -> &crate::detail::CubePolicy<B> {
        &self.inner
    }

    /// Copies host data to device-resident storage.
    pub fn to_device<T>(&self, input: &[T]) -> Result<DeviceVec<B, T>, Error>
    where
        T: Scalar,
    {
        Ok(DeviceVec::from_inner(self.inner.to_device(input)?))
    }

    /// Allocates device-resident storage and fills it with `value`.
    pub fn filled<T>(&self, len: usize, value: T) -> Result<DeviceVec<B, T>, Error>
    where
        T: Scalar,
    {
        Ok(DeviceVec::from_inner(self.inner.device_filled(len, value)?))
    }

    /// Allocates device-resident storage and initializes each element from its
    /// logical `u32` index.
    pub fn tabulate<T, Op>(&self, len: usize, _op: Op) -> Result<DeviceVec<B, T>, Error>
    where
        T: Scalar,
        Op: crate::runtime::op::TabulateOp<B, T>,
    {
        Ok(DeviceVec::from_inner(
            crate::detail::primitives::range::tabulate(
                self.policy(),
                len,
                KernelTabulateOp::<B, Op, T>::new(),
            )?,
        ))
    }

    /// Copies device-resident storage back to host memory.
    pub fn to_host<Input>(&self, input: &Input) -> Result<<Input as ToHost<B>>::Output, Error>
    where
        Input: ToHost<B>,
    {
        input.to_host_with(self)
    }

    /// Copies one device slice into another device slice.
    ///
    /// The slices must have the same length and belong to this executor.
    pub fn copy<T>(
        &self,
        from: DeviceSlice<'_, B, T>,
        to: DeviceSliceMut<'_, B, T>,
    ) -> Result<(), Error>
    where
        T: Scalar,
    {
        if from.len != to.len {
            return Err(Error::LengthMismatch {
                input: from.len,
                output: to.len,
            });
        }
        self.ensure_policy_id(from.source.inner.policy_id())?;
        self.ensure_policy_id(to.source.inner.policy_id())?;
        crate::detail::primitives::range::copy_slice_to_slice_with_policy(
            self.policy(),
            &from.source.inner,
            from.offset,
            &to.source.inner,
            to.offset,
            from.len,
        )
    }

    /// Waits until all previously submitted work for this executor is complete.
    pub fn sync(&self) -> Result<(), Error> {
        futures_lite::future::block_on(self.inner.client().sync()).map_err(|err| Error::Launch {
            message: err.to_string(),
        })
    }
}
