//! Lazy iterator sources and adapters.
//!
//! Lazy expressions allocate no result buffer by themselves. They are evaluated by the algorithm
//! that consumes them.

use crate::MStorageElement;

pub use crate::core::read::{Permute, Reverse, Taken, Transform};

/// An unbounded stream that repeats one value.
#[derive(Clone, Copy, Debug)]
pub struct Constant<T> {
    value: T,
}

impl<T> Constant<T> {
    /// Limits this source to `len` logical items.
    pub fn take(self, len: usize) -> Taken<Self> {
        Taken::new(self, len)
    }
}

/// An unbounded stream of consecutive [`usize`] values.
#[derive(Clone, Copy, Debug)]
pub struct Counting {
    start: usize,
}

impl Counting {
    /// Limits this source to `len` logical items.
    pub fn take(self, len: usize) -> Taken<Self> {
        Taken::new(self, len)
    }
}

/// An unbounded stream of consecutive `u32` values.
///
/// This is distinct from [`counting`], whose item type is the logical index
/// type `usize`.
#[derive(Clone, Copy, Debug)]
pub struct CountingU32 {
    start: u32,
}

impl CountingU32 {
    /// Limits this source to `len` logical items.
    pub fn take(self, len: usize) -> Taken<Self> {
        Taken::new(self, len)
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

impl crate::read::TakenSource for Constant<bool> {
    type Read = crate::read::Transform<crate::read::Constant<u32>, crate::op::U32ToBool>;

    fn lower(&self, _offset: usize, len: usize) -> Self::Read {
        crate::read::Transform::new(
            crate::read::Constant::new(u32::from(self.value), len),
            crate::op::U32ToBool,
        )
    }
}

impl crate::read::TakenSource for Counting {
    type Read = crate::read::Transform<crate::read::Counting, crate::op::U32ToUsize>;

    fn lower(&self, offset: usize, len: usize) -> Self::Read {
        let start = self
            .start
            .checked_add(offset)
            .expect("counting slice start overflow");
        crate::read::Transform::new(
            crate::read::Counting::new(
                u32::try_from(start).expect("counting value exceeds device u32 range"),
                len,
            ),
            crate::op::U32ToUsize,
        )
    }
}

impl crate::read::TakenSource for CountingU32 {
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

impl crate::read::TakenSource for Constant<usize> {
    type Read = crate::read::Transform<crate::read::Constant<u32>, crate::op::U32ToUsize>;

    fn lower(&self, _offset: usize, len: usize) -> Self::Read {
        crate::read::Transform::new(
            crate::read::Constant::new(
                u32::try_from(self.value).expect("constant value exceeds device u32 range"),
                len,
            ),
            crate::op::U32ToUsize,
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
pub fn counting(start: usize) -> Counting {
    Counting { start }
}

/// Creates an unbounded stream of consecutive `u32` values.
pub fn counting_u32(start: u32) -> CountingU32 {
    CountingU32 { start }
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
/// use massively::{
///     Executor, lazy,
///     op::{self, U32ToUsize},
///     vector::transform,
/// };
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let values = exec.to_device(&[10_u32, 20, 30]);
/// let indices = exec.to_device(&[2_u32, 0]);
/// let indices = lazy::transform(indices.slice(..), U32ToUsize);
/// let permuted = lazy::permute(values.slice(..), indices);
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
