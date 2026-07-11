//! Lazy iterator sources and adapters.
//!
//! Lazy expressions allocate no result buffer by themselves. They are evaluated by the algorithm
//! that consumes them.

use crate::{MIndex, MStorageElement};

pub use crate::core::read::{Permute, Reverse, Taken, Transform};

/// An unbounded stream that repeats one value.
#[derive(Clone, Copy, Debug)]
pub struct Constant<T> {
    value: T,
}

impl<T> Constant<T> {
    /// Limits this source to `len` logical items.
    pub fn take(self, len: MIndex) -> Taken<Self> {
        Taken::new(self, len)
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
        Taken::new(self, len)
    }
}

impl<T> crate::read::TakenSource for Constant<T>
where
    T: MStorageElement,
{
    type Read = crate::read::Constant<T>;

    fn lower(&self, _offset: MIndex, len: MIndex) -> Self::Read {
        crate::read::Constant::new(self.value, len as usize)
    }
}

impl crate::read::TakenSource for Counting {
    type Read = crate::read::Counting;

    fn lower(&self, offset: MIndex, len: MIndex) -> Self::Read {
        crate::read::Counting::new(
            self.start
                .checked_add(offset)
                .expect("counting slice start overflow"),
            len as usize,
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
/// use massively::{Executor, transform};
/// use massively::{lazy, op::Identity};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let output = exec.alloc::<u32>(3);
/// let repeated = lazy::constant(7_u32).take(3);
/// transform(&exec, repeated, Identity, output.slice_mut(..)).unwrap();
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
/// use massively::{Executor, transform};
/// use massively::{lazy, op::Identity};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let output = exec.alloc::<u32>(4);
/// let indices = lazy::counting(5).take(4);
/// transform(&exec, indices, Identity, output.slice_mut(..)).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![5, 6, 7, 8]);
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
/// use massively::{Executor, UnaryOp, transform};
/// use massively::{lazy, op::Identity};
///
/// struct Double;
///
/// #[cubecl::cube]
/// impl UnaryOp<u32> for Double {
///     type Output = u32;
///
///     fn apply(value: u32) -> u32 {
///         value * 2
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[1_u32, 2, 3]);
/// let output = exec.alloc::<u32>(input.len());
/// let doubled = lazy::transform(input.slice(..), Double);
/// transform(&exec, doubled, Identity, output.slice_mut(..)).unwrap();
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
/// use massively::{Executor, transform};
/// use massively::{lazy, op::Identity};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let values = exec.to_device(&[10_u32, 20, 30]);
/// let indices = exec.to_device(&[2_u32, 0]);
/// let output = exec.alloc::<u32>(indices.len());
/// let permuted = lazy::permute(values.slice(..), indices.slice(..));
/// transform(&exec, permuted, Identity, output.slice_mut(..)).unwrap();
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
/// use massively::{Executor, lazy, op::Identity, transform};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[10_u32, 20, 30]);
/// let output = exec.alloc::<u32>(input.len());
///
/// transform(
///     &exec,
///     lazy::reverse(input.slice(..)),
///     Identity,
///     output.slice_mut(..),
/// )
/// .unwrap();
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
/// use massively::{Executor, transform};
/// use massively::{lazy, op::Identity};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[1_u32, 2, 3]);
/// let output = exec.alloc::<u32>(input.len());
/// transform(
///     &exec,
///     lazy::identity(input.slice(..)),
///     Identity,
///     output.slice_mut(..),
/// )
/// .unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![1, 2, 3]);
/// ```
pub fn identity<Input>(input: Input) -> Transform<Input, crate::op::Identity> {
    Transform::new(input, crate::op::Identity)
}
