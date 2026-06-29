//! Execution context shared by runtime memory and algorithms.

use std::marker::PhantomData;

use cubecl::prelude::{CubeElement, CubePrimitive, Runtime};

use crate::Error;
use crate::detail::dispatch;
use crate::runtime::{DeviceSlice, DeviceSliceMut, DeviceVec};
use crate::value::MItem;

/// Scalar value that can be stored in one device column.
pub trait Scalar: CubePrimitive + CubeElement {}
impl<T> Scalar for T where T: CubePrimitive + CubeElement {}

/// Device-resident data that can be copied back to host memory by an executor.
pub trait ToHost<R: Runtime>:
    dispatch::ToHostDispatch<R, Output = <Self as ToHost<R>>::Output>
{
    type Output;
}

impl<R, T> ToHost<R> for T
where
    R: Runtime,
    T: dispatch::ToHostDispatch<R>,
{
    type Output = <T as dispatch::ToHostDispatch<R>>::Output;
}

/// Execution context for a CubeCL runtime.
#[derive(Debug)]
pub struct Executor<R: Runtime> {
    pub(crate) inner: crate::detail::CubePolicy<R>,
    pub(crate) _backend: PhantomData<fn() -> R>,
}

impl<R: Runtime> Clone for Executor<R> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            _backend: PhantomData,
        }
    }
}

impl<R: Runtime> Executor<R> {
    /// Creates an executor for the given CubeCL device.
    pub fn new(device: R::Device) -> Self {
        Self::from_device(&device)
    }

    /// Creates an executor for the given CubeCL device reference.
    pub fn from_device(device: &R::Device) -> Self {
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

    pub(crate) fn policy(&self) -> &crate::detail::CubePolicy<R> {
        &self.inner
    }

    /// Copies host data to device-resident storage.
    pub fn to_device<T>(&self, input: &[T]) -> Result<DeviceVec<R, T>, Error>
    where
        T: Scalar,
    {
        Ok(DeviceVec::from_inner(self.inner.to_device(input)?))
    }

    /// Allocates device-resident storage and fills it with `value`.
    pub fn constant<T>(&self, len: usize, value: T) -> Result<DeviceVec<R, T>, Error>
    where
        T: Scalar,
    {
        Ok(DeviceVec::from_inner(self.inner.device_filled(len, value)?))
    }

    /// Allocates uninitialized owned device storage for `Item`.
    ///
    /// This does not launch an initialization kernel. The returned columns are
    /// intended as temporary output buffers for algorithms that write every
    /// element before the data is read. Reading an allocated buffer before
    /// writing it produces unspecified values.
    pub fn alloc<Item>(&self, len: usize) -> Result<Item::Vec, Error>
    where
        Item: MItem<R>,
    {
        Item::alloc_vec(self, len)
    }

    /// Allocates a `u32` device vector containing `0..len`.
    pub fn tabulate(&self, len: usize) -> Result<DeviceVec<R, u32>, Error> {
        Ok(DeviceVec::from_inner(
            crate::detail::primitives::range::indices_u32(self.policy(), len)?,
        ))
    }

    /// Copies device-resident storage back to host memory.
    pub fn to_host<Input>(&self, input: &Input) -> Result<<Input as ToHost<R>>::Output, Error>
    where
        Input: ToHost<R>,
    {
        input.to_host_with(self)
    }

    /// Copies one device slice into another device slice.
    ///
    /// The slices must have the same length and belong to this executor.
    pub fn copy<T>(
        &self,
        from: DeviceSlice<'_, R, T>,
        to: DeviceSliceMut<'_, R, T>,
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
