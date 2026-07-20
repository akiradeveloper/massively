//! Lazy iterator sources and adapters.
//!
//! Lazy expressions allocate no result buffer by themselves. They are evaluated by the algorithm
//! that consumes them.

#![allow(private_interfaces)]

use core::marker::PhantomData;
use std::ops::RangeBounds;

use cubecl::prelude::Runtime;

use crate::{Error, MIndex, MIter, MStorageElement, op::UnaryOp};
use crate::{core::facade as private, read::SliceExpression};

pub use crate::core::read::Taken;

/// Logical lazy permutation lowered only when an algorithm consumes it.
#[derive(Clone, Copy, Debug)]
pub struct Permute<Values, Indices> {
    values: Values,
    indices: Indices,
}

impl<Values, Indices> Permute<Values, Indices> {
    pub const fn new(values: Values, indices: Indices) -> Self {
        Self { values, indices }
    }
}

#[doc(hidden)]
impl<R, Values, Indices> MIter<R> for Permute<Values, Indices>
where
    R: Runtime,
    Values: MIter<R>,
    Indices: MIter<R, Item = MIndex>,
    crate::read::Permute<Values::Read, Indices::Read>:
        private::KernelInput<R, Item = Values::Item> + private::IterLength + SliceExpression,
{
    type Item = Values::Item;
    type Read = crate::read::Permute<Values::Read, Indices::Read>;
    type Slice = crate::read::Slice<R, Self::Read>;

    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice
    where
        Bounds: RangeBounds<usize>,
    {
        let input = self.clone().lower_read();
        let len = private::IterLength::logical_len(&input)
            .expect("cannot slice a lazy permutation with an invalid length");
        let (start, count) = crate::api::iter::resolve_iter_range(len, range);
        crate::read::Slice::new(input.slice_expression(start, count))
    }

    fn capacity(&self) -> Result<MIndex, Error> {
        self.indices.capacity()
    }

    fn logical_extent(&self) -> Result<crate::extent::LogicalExtent, Error> {
        self.indices.logical_extent()
    }

    fn lower_read(self) -> Self::Read {
        crate::read::Permute::new(self.values.lower_read(), self.indices.lower_read())
    }
}

/// Logical lazy reverse view lowered only when an algorithm consumes it.
#[derive(Clone, Copy, Debug)]
pub struct Reverse<Values> {
    values: Values,
}

impl<Values> Reverse<Values> {
    pub const fn new(values: Values) -> Self {
        Self { values }
    }
}

#[doc(hidden)]
impl<R, Values> MIter<R> for Reverse<Values>
where
    R: Runtime,
    Values: MIter<R>,
    crate::read::Reverse<Values::Read>:
        private::KernelInput<R, Item = Values::Item> + private::IterLength + SliceExpression,
{
    type Item = Values::Item;
    type Read = crate::read::Reverse<Values::Read>;
    type Slice = crate::read::Slice<R, Self::Read>;

    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice
    where
        Bounds: RangeBounds<usize>,
    {
        let input = self.clone().lower_read();
        let len = private::IterLength::logical_len(&input)
            .expect("cannot slice a lazy reverse view with an invalid length");
        let (start, count) = crate::api::iter::resolve_iter_range(len, range);
        crate::read::Slice::new(input.slice_expression(start, count))
    }

    fn capacity(&self) -> Result<MIndex, Error> {
        self.values.capacity()
    }

    fn logical_extent(&self) -> Result<crate::extent::LogicalExtent, Error> {
        self.values.logical_extent()
    }

    fn lower_read(self) -> Self::Read {
        crate::read::Reverse::new(self.values.lower_read())
    }
}

/// Logical lazy transform lowered only when an algorithm consumes it.
#[derive(Debug)]
pub struct Transform<Input, Op> {
    input: Input,
    _op: PhantomData<fn() -> Op>,
}

impl<Input: Clone, Op> Clone for Transform<Input, Op> {
    fn clone(&self) -> Self {
        Self {
            input: self.input.clone(),
            _op: PhantomData,
        }
    }
}

impl<Input: Copy, Op> Copy for Transform<Input, Op> {}

impl<Input, Op> Transform<Input, Op> {
    pub fn new(input: Input, _op: Op) -> Self {
        Self {
            input,
            _op: PhantomData,
        }
    }
}

#[doc(hidden)]
impl<R, Input, Op> MIter<R> for Transform<Input, Op>
where
    R: Runtime,
    Input: MIter<R>,
    Op: UnaryOp<Input::Item>,
    crate::read::Transform<Input::Read, Op>:
        private::KernelInput<R, Item = Op::Output> + private::IterLength + SliceExpression,
{
    type Item = Op::Output;
    type Read = crate::read::Transform<Input::Read, Op>;
    type Slice = crate::read::Slice<R, Self::Read>;

    fn slice<Bounds>(&self, range: Bounds) -> Self::Slice
    where
        Bounds: RangeBounds<usize>,
    {
        let input = self.clone().lower_read();
        let len = private::IterLength::logical_len(&input)
            .expect("cannot slice a lazy transform with an invalid length");
        let (start, count) = crate::api::iter::resolve_iter_range(len, range);
        crate::read::Slice::new(input.slice_expression(start, count))
    }

    fn capacity(&self) -> Result<MIndex, Error> {
        self.input.capacity()
    }

    fn logical_extent(&self) -> Result<crate::extent::LogicalExtent, Error> {
        self.input.logical_extent()
    }

    fn lower_read(self) -> Self::Read {
        crate::read::Transform::from_input(self.input.lower_read())
    }
}

/// An unbounded stream that repeats one value.
#[derive(Clone, Copy, Debug)]
pub struct Constant<T> {
    value: T,
}

impl<T> Constant<T> {
    /// Limits this source to `len` logical items.
    pub fn take(self, len: MIndex) -> Taken<Self> {
        Taken::new(self, len as usize)
    }
}

/// An unbounded stream of consecutive [`MIndex`] values.
#[derive(Clone, Copy, Debug)]
pub struct Counting {
    start: MIndex,
}

impl Counting {
    /// Limits this source to `len` logical items.
    pub fn take(self, len: MIndex) -> Taken<Self> {
        Taken::new(self, len as usize)
    }
}

impl<T> crate::read::TakenSource for Constant<T>
where
    T: MStorageElement,
{
    type Read = crate::read::Constant<T>;

    fn lower(&self, _offset: usize, len: usize) -> Self::Read {
        crate::read::Constant::new(self.value, len)
    }
}

impl crate::read::TakenSource for Counting {
    type Read = crate::read::Counting;

    fn lower(&self, offset: usize, len: usize) -> Self::Read {
        let offset = u32::try_from(offset).expect("counting offset exceeds u32");
        crate::read::Counting::new(
            self.start
                .checked_add(offset)
                .expect("counting start overflow"),
            len,
        )
    }
}

/// Creates an unbounded stream that repeats `value`.
///
/// Call [`.take(len)`](Constant::take) before passing it to an algorithm.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, lazy, op, vector::transform};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let repeated = lazy::constant(7_u32).take(3);
/// let output = transform(&exec, repeated, op::Identity).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![7, 7, 7]);
/// ```
pub fn constant<T>(value: T) -> Constant<T> {
    Constant { value }
}

/// Creates an unbounded stream of consecutive indices beginning at `start`.
///
/// Call [`.take(len)`](Counting::take) before passing it to an algorithm.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, lazy, vector::gather};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let values = exec.to_device(&[10_u32, 20, 30, 40]);
/// let indices = lazy::counting(1).take(3);
/// let output = gather(&exec, values.slice(..), indices).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![20, 30, 40]);
/// ```
pub fn counting(start: MIndex) -> Counting {
    Counting { start }
}

/// Lazily applies `op` whenever an algorithm reads an item.
///
/// This does not allocate an intermediate device buffer.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, lazy, op, vector::transform};
///
/// struct Double;
///
/// #[cubecl::cube]
/// impl op::UnaryOp<u32> for Double {
///     type Output = u32;
///
///     fn apply(value: u32) -> u32 {
///         value * 2
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[1_u32, 2, 3]);
/// let doubled = lazy::transform(input.slice(..), Double);
/// let output = transform(&exec, doubled, op::Identity).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![2, 4, 6]);
/// ```
pub fn transform<Input, Op>(input: Input, op: Op) -> Transform<Input, Op> {
    Transform::new(input, op)
}

/// Lazily reads `values[indices[i]]`.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, lazy, op, vector::transform};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let values = exec.to_device(&[10_u32, 20, 30]);
/// let indices = exec.to_device(&[2_u32, 0]);
/// let permuted = lazy::permute(values.slice(..), indices.slice(..));
/// let output = transform(&exec, permuted, op::Identity).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![30, 10]);
/// ```
pub fn permute<Values, Indices>(values: Values, indices: Indices) -> Permute<Values, Indices> {
    Permute::new(values, indices)
}

/// Lazily reads an input in reverse order.
///
/// This generates reverse indices as part of the consuming kernel and does
/// not allocate an intermediate index or value buffer.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, lazy, op, vector::transform};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[10_u32, 20, 30]);
/// let output = transform(&exec, lazy::reverse(input.slice(..)), op::Identity).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![30, 20, 10]);
/// ```
pub fn reverse<Values>(values: Values) -> Reverse<Values> {
    Reverse::new(values)
}

/// Wraps an input in a lazy identity transform.
///
/// This is useful in tests and when an explicit lazy transform node is required.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, lazy, op, vector::transform};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[1_u32, 2, 3]);
/// let output = transform(&exec, lazy::identity(input.slice(..)), op::Identity).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![1, 2, 3]);
/// ```
pub fn identity<Input>(input: Input) -> Transform<Input, crate::op::Identity> {
    Transform::new(input, crate::op::Identity)
}
