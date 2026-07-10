use cubecl::prelude::Runtime;

use crate::{Error, Executor, MIter, MIterMut, UnaryOp, WriteFrom};

/// Applies a unary operation and writes its result to preallocated storage.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, UnaryOp, transform};
///
/// struct AddOne;
///
/// #[cubecl::cube]
/// impl UnaryOp<u32> for AddOne {
///     type Output = u32;
///
///     fn apply(value: u32) -> u32 {
///         value + 1
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[1_u32, 2, 3]);
/// let output = exec.alloc::<u32>(input.len());
///
/// transform(&exec, input.slice(..), AddOne, output.slice_mut(..)).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![2, 3, 4]);
/// ```
pub fn transform<R, Input, Output, Op>(
    exec: &Executor<R>,
    input: Input,
    op: Op,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: MIter<R>,
    Output: MIterMut<R>,
    Op: UnaryOp<Input::Item>,
    Output::Item: WriteFrom<<Op as UnaryOp<Input::Item>>::Output>,
{
    input.transform_into(exec, op, output)
}
