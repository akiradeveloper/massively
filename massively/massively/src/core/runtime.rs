//! Runtime executor and contiguous device storage.

use core::marker::PhantomData;
use core::sync::atomic::{AtomicU64, Ordering};
use cubecl::prelude::*;
use std::ops::RangeBounds;

use crate::{Column, Error, MStorageElement};

pub use crate::read::DeviceSlice;

static NEXT_EXECUTOR_ID: AtomicU64 = AtomicU64::new(1);

/// Execution context for one CubeCL runtime.
#[derive(Clone)]
pub struct Executor<R: Runtime> {
    client: ComputeClient<R>,
    id: u64,
}

impl<R: Runtime> Executor<R> {
    /// Creates an executor for one CubeCL device.
    ///
    /// # Examples
    ///
    /// ```
    /// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    /// use massively::Executor;
    ///
    /// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    /// let values = exec.to_device(&[1_u32, 2, 3]);
    ///
    /// assert_eq!(exec.to_host(&values).unwrap(), vec![1, 2, 3]);
    /// ```
    pub fn new(device: R::Device) -> Self {
        Self {
            client: R::client(&device),
            id: NEXT_EXECUTOR_ID.fetch_add(1, Ordering::Relaxed),
        }
    }

    #[doc(hidden)]
    pub fn client(&self) -> &ComputeClient<R> {
        &self.client
    }

    #[doc(hidden)]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Explicitly copies a host slice to device memory.
    ///
    /// # Examples
    ///
    /// ```
    /// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    /// use massively::Executor;
    ///
    /// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    /// let values = exec.to_device(&[10_u32, 20, 30]);
    ///
    /// assert_eq!(values.len(), 3);
    /// assert_eq!(exec.to_host(&values).unwrap(), vec![10, 20, 30]);
    /// ```
    pub fn to_device<T>(&self, input: &[T]) -> DeviceVec<R, T>
    where
        T: MStorageElement,
    {
        let handle = if input.is_empty() {
            self.client.empty(size_of::<T>().max(1))
        } else {
            self.client.create_from_slice(T::as_bytes(input))
        };
        DeviceVec {
            handle,
            len: input.len(),
            owner: self.id,
            _runtime: PhantomData,
        }
    }

    /// Allocates device storage that must be fully written before it is read.
    pub(crate) fn alloc_column<T>(&self, len: usize) -> DeviceVec<R, T>
    where
        T: MStorageElement,
    {
        DeviceVec {
            handle: self.client.empty(len.max(1) * size_of::<T>()),
            len,
            owner: self.id,
            _runtime: PhantomData,
        }
    }

    /// Explicitly copies a device vector to the host.
    ///
    /// `input` may be a [`DeviceVec`], [`DeviceSlice`], or [`DeviceSliceMut`].
    ///
    /// # Examples
    ///
    /// ```
    /// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    /// use massively::Executor;
    ///
    /// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    /// let values = exec.to_device(&[10_u32, 20, 30, 40]);
    ///
    /// assert_eq!(exec.to_host(&values.slice(1..3)).unwrap(), vec![20, 30]);
    /// ```
    pub fn to_host<Input>(&self, input: &Input) -> Result<Vec<Input::Element>, Error>
    where
        Input: DeviceRange,
    {
        if input.owner() != self.id {
            return Err(Error::ForeignExecutor);
        }
        if input.len() == 0 {
            return Ok(Vec::new());
        }
        let bytes = self
            .client
            .read_one(input.handle())
            .map_err(|err| Error::Launch {
                message: format!("{err:?}"),
            })?;
        let values = Input::Element::from_bytes(&bytes);
        let start = input.offset();
        let end = start + input.len();
        Ok(values[start..end].to_vec())
    }

    /// Waits for all work submitted through this executor to complete.
    ///
    /// # Examples
    ///
    /// ```
    /// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    /// use massively::{Executor, vector::fill};
    ///
    /// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    /// let output = fill(&exec, 4, 7_u32).unwrap();
    /// exec.sync().unwrap();
    /// ```
    pub fn sync(&self) -> Result<(), Error> {
        futures_lite::future::block_on(self.client.sync()).map_err(|err| Error::Launch {
            message: format!("{err:?}"),
        })
    }
}

/// Owned single-column device storage.
pub struct DeviceVec<R: Runtime, T> {
    pub(crate) handle: cubecl::server::Handle,
    len: usize,
    owner: u64,
    _runtime: PhantomData<fn() -> (R, T)>,
}

impl<R: Runtime, T> Clone for DeviceVec<R, T> {
    fn clone(&self) -> Self {
        Self {
            handle: self.handle.clone(),
            len: self.len,
            owner: self.owner,
            _runtime: PhantomData,
        }
    }
}

impl<R: Runtime, T> DeviceVec<R, T> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Shrinks the logical length without copying or reallocating device memory.
    ///
    /// The underlying allocation remains unchanged and returns to the runtime's
    /// memory pool at its original size when the final handle is dropped.
    pub fn truncate(&mut self, len: usize) {
        assert!(
            len <= self.len,
            "cannot truncate a DeviceVec of length {} to {len}",
            self.len,
        );
        self.len = len;
    }

    /// Creates an internal read-expression leaf over the whole allocation.
    pub(crate) fn column(&self) -> Column<T> {
        Column::from_handle(self.handle.clone(), self.len, 0, self.owner, self.len)
    }

    /// Returns a read-only view into this allocation.
    ///
    /// # Examples
    ///
    /// ```
    /// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    /// use massively::Executor;
    ///
    /// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    /// let values = exec.to_device(&[1_u32, 2, 3, 4]);
    /// let middle = values.slice(1..3);
    ///
    /// assert_eq!(exec.to_host(&middle).unwrap(), vec![2, 3]);
    /// ```
    pub fn slice<Range>(&self, range: Range) -> DeviceSlice<T>
    where
        Range: RangeBounds<usize>,
    {
        self.column().slice(range)
    }

    /// Returns a mutable view into this allocation.
    ///
    /// # Examples
    ///
    /// ```
    /// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    /// use massively::{Executor, lazy, vector::replace_where};
    ///
    /// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    /// let values = exec.to_device(&[1_u32, 2, 3, 4]);
    /// replace_where(&exec, 9, lazy::constant(1_u32).take(2), values.slice_mut(1..3)).unwrap();
    ///
    /// assert_eq!(exec.to_host(&values).unwrap(), vec![1, 9, 9, 4]);
    /// ```
    pub fn slice_mut<Range>(&self, range: Range) -> DeviceSliceMut<T>
    where
        Range: RangeBounds<usize>,
    {
        let (offset, len) = crate::read::resolve_slice_range(self.len, range);
        DeviceSliceMut {
            handle: self.handle.clone(),
            len,
            offset: offset as u32,
            owner: self.owner,
            buffer_len: self.len,
            _item: PhantomData,
        }
    }
}

#[doc(hidden)]
pub trait DeviceRange {
    type Element: MStorageElement;
    fn handle(&self) -> cubecl::server::Handle;
    fn len(&self) -> usize;
    fn offset(&self) -> usize;
    fn owner(&self) -> u64;
}

impl<R: Runtime, T: MStorageElement> DeviceRange for DeviceVec<R, T> {
    type Element = T;
    fn handle(&self) -> cubecl::server::Handle {
        self.handle.clone()
    }
    fn len(&self) -> usize {
        self.len
    }
    fn offset(&self) -> usize {
        0
    }
    fn owner(&self) -> u64 {
        self.owner
    }
}

impl<T: MStorageElement> DeviceRange for DeviceSlice<T> {
    type Element = T;
    fn handle(&self) -> cubecl::server::Handle {
        self.handle.clone().expect("bound device slice")
    }
    fn len(&self) -> usize {
        self.len
    }
    fn offset(&self) -> usize {
        self.offset as usize
    }
    fn owner(&self) -> u64 {
        self.owner.expect("bound device slice")
    }
}

/// Mutable contiguous output view. Cloning a view does not copy device data.
#[derive(Clone, Debug)]
pub struct DeviceSliceMut<T> {
    pub(crate) handle: cubecl::server::Handle,
    pub(crate) len: usize,
    pub(crate) offset: u32,
    pub(crate) owner: u64,
    pub(crate) buffer_len: usize,
    pub(crate) _item: PhantomData<fn() -> T>,
}

impl<T> DeviceSliceMut<T> {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns a read-only subview of this mutable view.
    ///
    /// # Examples
    ///
    /// ```
    /// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    /// use massively::Executor;
    ///
    /// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    /// let values = exec.to_device(&[1_u32, 2, 3, 4, 5]);
    /// let writable = values.slice_mut(1..5);
    /// let readable = writable.slice(1..3);
    ///
    /// assert_eq!(exec.to_host(&readable).unwrap(), vec![3, 4]);
    /// ```
    pub fn slice<Range>(&self, range: Range) -> DeviceSlice<T>
    where
        Range: RangeBounds<usize>,
    {
        let (offset, len) = crate::read::resolve_slice_range(self.len, range);
        Column::from_handle(
            self.handle.clone(),
            len,
            self.offset + offset as u32,
            self.owner,
            self.buffer_len,
        )
    }

    /// Returns a mutable subview of this mutable view.
    ///
    /// # Examples
    ///
    /// ```
    /// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    /// use massively::{Executor, lazy, vector::replace_where};
    ///
    /// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    /// let values = exec.to_device(&[1_u32, 2, 3, 4, 5]);
    /// let writable = values.slice_mut(1..5);
    /// replace_where(&exec, 9, lazy::constant(1_u32).take(2), writable.slice_mut(1..3)).unwrap();
    ///
    /// assert_eq!(exec.to_host(&values).unwrap(), vec![1, 2, 9, 9, 5]);
    /// ```
    pub fn slice_mut<Range>(&self, range: Range) -> Self
    where
        Range: RangeBounds<usize>,
    {
        let (offset, len) = crate::read::resolve_slice_range(self.len, range);
        Self {
            handle: self.handle.clone(),
            len,
            offset: self.offset + offset as u32,
            owner: self.owner,
            buffer_len: self.buffer_len,
            _item: PhantomData,
        }
    }
}

impl<T: MStorageElement> DeviceRange for DeviceSliceMut<T> {
    type Element = T;
    fn handle(&self) -> cubecl::server::Handle {
        self.handle.clone()
    }
    fn len(&self) -> usize {
        self.len
    }
    fn offset(&self) -> usize {
        self.offset as usize
    }
    fn owner(&self) -> u64 {
        self.owner
    }
}
