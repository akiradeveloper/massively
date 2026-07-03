use std::marker::PhantomData;
use std::ops::{Bound, RangeBounds};

use cubecl::prelude::Runtime;

use crate::Error;
use crate::detail::dispatch;
use crate::index::{MIndex, usize_from_mindex};
use crate::runtime::{Executor, Scalar};

/// Owned device column.
#[derive(Debug)]
pub struct DeviceVec<R: Runtime, T> {
    pub(crate) inner: crate::detail::DeviceVec<R, T>,
    pub(crate) _backend: PhantomData<fn() -> R>,
}

impl<R, T> DeviceVec<R, T>
where
    R: Runtime,
{
    pub(crate) fn from_inner(inner: crate::detail::DeviceVec<R, T>) -> Self {
        Self {
            inner,
            _backend: PhantomData,
        }
    }

    /// Returns the number of elements.
    pub fn len(&self) -> MIndex {
        self.inner.mindex_len()
    }

    /// Returns whether this column is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns a read-only device slice for the given range.
    ///
    /// The range is checked like a Rust slice range and panics if it is out of
    /// bounds or if the start is greater than the end.
    pub fn slice<Range>(&self, range: Range) -> DeviceSlice<'_, R, T>
    where
        Range: RangeBounds<MIndex>,
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
    pub fn slice_mut<Range>(&self, range: Range) -> DeviceSliceMut<'_, R, T>
    where
        Range: RangeBounds<MIndex>,
    {
        let (offset, len) = resolve_slice_range(self.len(), range);
        DeviceSliceMut {
            source: self,
            offset,
            len,
        }
    }
}

impl<R, T> dispatch::ToHostDispatch<R> for DeviceVec<R, T>
where
    R: Runtime,
    T: Scalar,
{
    type Output = Vec<T>;

    fn to_host_with(&self, exec: &Executor<R>) -> Result<Self::Output, Error> {
        exec.ensure_policy_id(self.inner.policy_id())?;
        self.inner.read_to_host(exec.policy())
    }
}

/// Read-only view into a contiguous range of a [`DeviceVec`].
#[derive(Debug)]
pub struct DeviceSlice<'a, R: Runtime, T> {
    pub(crate) source: &'a DeviceVec<R, T>,
    pub(crate) offset: MIndex,
    pub(crate) len: MIndex,
}

impl<'a, R, T> Copy for DeviceSlice<'a, R, T> where R: Runtime {}

impl<'a, R, T> Clone for DeviceSlice<'a, R, T>
where
    R: Runtime,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, R, T> DeviceSlice<'a, R, T>
where
    R: Runtime,
{
    /// Returns the number of elements in this slice.
    pub fn len(&self) -> MIndex {
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
    pub fn slice<Range>(&self, range: Range) -> DeviceSlice<'_, R, T>
    where
        Range: RangeBounds<MIndex>,
    {
        let (relative_offset, len) = resolve_slice_range(self.len, range);
        DeviceSlice {
            source: self.source,
            offset: self
                .offset
                .checked_add(relative_offset)
                .expect("slice offset overflow"),
            len,
        }
    }

    pub(crate) fn policy_id(&self) -> crate::detail::policy::CubePolicyId {
        self.source.inner.policy_id()
    }

    pub(crate) fn column_view(&self) -> crate::detail::device::DeviceColumnView<R, T>
    where
        T: Scalar,
    {
        crate::detail::device::DeviceColumnView::from_slice(
            &self.source.inner,
            usize_from_mindex(self.offset),
            usize_from_mindex(self.len),
        )
    }
}

impl<'a, R, T> dispatch::ToHostDispatch<R> for DeviceSlice<'a, R, T>
where
    R: Runtime,
    T: Scalar,
{
    type Output = Vec<T>;

    fn to_host_with(&self, exec: &Executor<R>) -> Result<Self::Output, Error> {
        exec.ensure_policy_id(self.source.inner.policy_id())?;
        let mut values = self.source.inner.read_to_host(exec.policy())?;
        let end = self
            .offset
            .checked_add(self.len)
            .ok_or(Error::LengthTooLarge {
                len: usize_from_mindex(self.len),
            })?;
        let end = usize_from_mindex(end);
        let offset = usize_from_mindex(self.offset);
        values.drain(end..);
        values.drain(..offset);
        Ok(values)
    }
}

/// Mutable view into a contiguous range of a [`DeviceVec`].
#[derive(Debug)]
pub struct DeviceSliceMut<'a, R: Runtime, T> {
    pub(crate) source: &'a DeviceVec<R, T>,
    pub(crate) offset: MIndex,
    pub(crate) len: MIndex,
}

impl<'a, R, T> DeviceSliceMut<'a, R, T>
where
    R: Runtime,
{
    /// Returns the number of elements in this slice.
    pub fn len(&self) -> MIndex {
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
    pub fn slice<Range>(&self, range: Range) -> DeviceSlice<'_, R, T>
    where
        Range: RangeBounds<MIndex>,
    {
        let (relative_offset, len) = resolve_slice_range(self.len, range);
        DeviceSlice {
            source: self.source,
            offset: self
                .offset
                .checked_add(relative_offset)
                .expect("slice offset overflow"),
            len,
        }
    }

    /// Returns a mutable device subslice for the given range.
    ///
    /// The range is checked against this slice, not against the original
    /// `DeviceVec`.
    pub fn slice_mut<Range>(&self, range: Range) -> DeviceSliceMut<'_, R, T>
    where
        Range: RangeBounds<MIndex>,
    {
        let (relative_offset, len) = resolve_slice_range(self.len, range);
        DeviceSliceMut {
            source: self.source,
            offset: self
                .offset
                .checked_add(relative_offset)
                .expect("slice offset overflow"),
            len,
        }
    }
}

fn resolve_slice_range<Range>(len: MIndex, range: Range) -> (MIndex, MIndex)
where
    Range: RangeBounds<MIndex>,
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
