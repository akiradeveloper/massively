use std::marker::PhantomData;
use std::ops::{Bound, RangeBounds};

use cubecl::prelude::Runtime;

use crate::Error;
use crate::detail::dispatch;
use crate::runtime::{Executor, Scalar};

/// Owned device column.
#[derive(Debug)]
pub struct DeviceVec<B: Runtime, T> {
    pub(crate) inner: crate::detail::DeviceVec<B, T>,
    pub(crate) _backend: PhantomData<fn() -> B>,
}

impl<B, T> DeviceVec<B, T>
where
    B: Runtime,
{
    pub(crate) fn from_inner(inner: crate::detail::DeviceVec<B, T>) -> Self {
        Self {
            inner,
            _backend: PhantomData,
        }
    }

    /// Returns the number of elements.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns whether this column is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns a read-only device slice for the given range.
    ///
    /// The range is checked like a Rust slice range and panics if it is out of
    /// bounds or if the start is greater than the end.
    pub fn slice<R>(&self, range: R) -> DeviceSlice<'_, B, T>
    where
        R: RangeBounds<usize>,
    {
        let (offset, len) = resolve_slice_range(self.len(), range);
        DeviceSlice {
            source: self,
            offset,
            len,
        }
    }

    /// Returns a mutable device slice for the given range.
    ///
    /// The range is checked like a Rust slice range and panics if it is out of
    /// bounds or if the start is greater than the end.
    pub fn slice_mut<R>(&mut self, range: R) -> DeviceSliceMut<'_, B, T>
    where
        R: RangeBounds<usize>,
    {
        let (offset, len) = resolve_slice_range(self.len(), range);
        DeviceSliceMut {
            source: self,
            offset,
            len,
        }
    }
}

impl<B, T> dispatch::ToHostDispatch<B> for DeviceVec<B, T>
where
    B: Runtime,
    T: Scalar,
{
    type Output = Vec<T>;

    fn to_host_with(&self, exec: &Executor<B>) -> Result<Self::Output, Error> {
        exec.ensure_policy_id(self.inner.policy_id())?;
        self.inner.read_to_host(exec.policy())
    }
}

/// Read-only view into a contiguous range of a [`DeviceVec`].
#[derive(Debug)]
pub struct DeviceSlice<'a, B: Runtime, T> {
    pub(crate) source: &'a DeviceVec<B, T>,
    pub(crate) offset: usize,
    pub(crate) len: usize,
}

impl<'a, B, T> Copy for DeviceSlice<'a, B, T> where B: Runtime {}

impl<'a, B, T> Clone for DeviceSlice<'a, B, T>
where
    B: Runtime,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, B, T> DeviceSlice<'a, B, T>
where
    B: Runtime,
{
    /// Returns the number of elements in this slice.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether this slice is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns a read-only device subslice for the given range.
    ///
    /// The range is checked against this slice, not against the original
    /// `DeviceVec`.
    pub fn slice<R>(&self, range: R) -> DeviceSlice<'_, B, T>
    where
        R: RangeBounds<usize>,
    {
        let (relative_offset, len) = resolve_slice_range(self.len, range);
        DeviceSlice {
            source: self.source,
            offset: self.offset + relative_offset,
            len,
        }
    }

    pub(crate) fn policy_id(&self) -> crate::detail::policy::CubePolicyId {
        self.source.inner.policy_id()
    }

    pub(crate) fn column_view(&self) -> crate::detail::device::DeviceColumnView<B, T>
    where
        T: Scalar,
    {
        crate::detail::device::DeviceColumnView::from_slice(
            &self.source.inner,
            self.offset,
            self.len,
        )
    }

    pub(crate) fn column_view_as<U>(&self) -> Option<crate::detail::device::DeviceColumnView<B, U>>
    where
        U: Scalar + 'static,
        T: 'static,
    {
        let source = self.source as &dyn std::any::Any;
        let source = source.downcast_ref::<DeviceVec<B, U>>()?;
        Some(crate::detail::device::DeviceColumnView::from_slice(
            &source.inner,
            self.offset,
            self.len,
        ))
    }
}

impl<'a, B, T> dispatch::ToHostDispatch<B> for DeviceSlice<'a, B, T>
where
    B: Runtime,
    T: Scalar,
{
    type Output = Vec<T>;

    fn to_host_with(&self, exec: &Executor<B>) -> Result<Self::Output, Error> {
        exec.ensure_policy_id(self.source.inner.policy_id())?;
        let mut values = self.source.inner.read_to_host(exec.policy())?;
        let end = self
            .offset
            .checked_add(self.len)
            .ok_or(Error::LengthTooLarge { len: self.len })?;
        values.drain(end..);
        values.drain(..self.offset);
        Ok(values)
    }
}

/// Mutable view into a contiguous range of a [`DeviceVec`].
#[derive(Debug)]
pub struct DeviceSliceMut<'a, B: Runtime, T> {
    pub(crate) source: &'a mut DeviceVec<B, T>,
    pub(crate) offset: usize,
    pub(crate) len: usize,
}

impl<'a, B, T> DeviceSliceMut<'a, B, T>
where
    B: Runtime,
{
    /// Returns the number of elements in this slice.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether this slice is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns a read-only device subslice for the given range.
    ///
    /// The range is checked against this slice, not against the original
    /// `DeviceVec`.
    pub fn slice<R>(&self, range: R) -> DeviceSlice<'_, B, T>
    where
        R: RangeBounds<usize>,
    {
        let (relative_offset, len) = resolve_slice_range(self.len, range);
        DeviceSlice {
            source: self.source,
            offset: self.offset + relative_offset,
            len,
        }
    }

    /// Returns a mutable device subslice for the given range.
    ///
    /// The range is checked against this slice, not against the original
    /// `DeviceVec`.
    pub fn slice_mut<R>(&mut self, range: R) -> DeviceSliceMut<'_, B, T>
    where
        R: RangeBounds<usize>,
    {
        let (relative_offset, len) = resolve_slice_range(self.len, range);
        DeviceSliceMut {
            source: self.source,
            offset: self.offset + relative_offset,
            len,
        }
    }
}

fn resolve_slice_range<R>(len: usize, range: R) -> (usize, usize)
where
    R: RangeBounds<usize>,
{
    let start = match range.start_bound() {
        Bound::Included(&start) => start,
        Bound::Excluded(&start) => start.checked_add(1).expect("slice start overflow"),
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        Bound::Included(&end) => end.checked_add(1).expect("slice end overflow"),
        Bound::Excluded(&end) => end,
        Bound::Unbounded => len,
    };
    assert!(
        start <= end,
        "slice start ({start}) is greater than slice end ({end})"
    );
    assert!(
        end <= len,
        "slice end ({end}) is out of bounds for DeviceVec of length {len}"
    );
    (start, end - start)
}
