//! Operations embedded in read expressions.

use cubecl::prelude::CubeType;

use crate::StorageLayout;

/// Compile-time unary operation used by [`crate::read::Transform`].
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, vector::transform};
/// use massively::op::UnaryOp;
///
/// struct Square;
///
/// #[cubecl::cube]
/// impl UnaryOp<u32> for Square {
///     type Output = u32;
///
///     fn apply(value: u32) -> u32 {
///         value * value
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[2_u32, 3, 4]);
/// let output = transform(&exec, input.slice(..), Square).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![4, 9, 16]);
/// ```
#[cubecl::cube]
pub trait UnaryOp<Input: CubeType>: 'static + Send + Sync {
    type Output: StorageLayout;

    fn apply(input: Input) -> Self::Output;
}

/// Internal index-aware transform operation.
#[doc(hidden)]
#[cubecl::cube]
pub trait IndexedUnaryOp<Input: CubeType>: 'static + Send + Sync {
    type Output: StorageLayout;

    fn apply(input: Input, index: u32) -> Self::Output;
}

/// Internal index-aware operation over an adjacent input pair.
#[doc(hidden)]
#[cubecl::cube]
pub trait IndexedBinaryOp<Input: CubeType>: 'static + Send + Sync {
    type Output: StorageLayout;

    fn apply(previous: Input, current: Input, index: u32) -> Self::Output;
}

/// Identity operation.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, vector::transform};
/// use massively::op::Identity;
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[1_u32, 2, 3]);
/// let output = transform(&exec, input.slice(..), Identity).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![1, 2, 3]);
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct Identity;

#[cubecl::cube]
impl<Input: StorageLayout> UnaryOp<Input> for Identity {
    type Output = Input;

    fn apply(input: Input) -> Input {
        input
    }
}
