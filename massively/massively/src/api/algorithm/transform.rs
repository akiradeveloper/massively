use cubecl::prelude::Runtime;

use crate::{
    Error, Executor, MIter, MIterMut, MStorage, MVec, ToCanonical, WritableFrom, op::UnaryOp,
};

/// Copies every input row into caller-provided storage.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, vector::copy};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[1_u32, 2, 3]);
/// let output = exec.alloc::<u32>(3);
///
/// copy(&exec, input.slice(..), output.slice_mut(..)).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![1, 2, 3]);
/// ```
pub fn copy<R, Input, Output>(exec: &Executor<R>, input: Input, output: Output) -> Result<(), Error>
where
    R: Runtime,
    Input: MIter<R>,
    Output: MIterMut<R>,
    Output::Item: WritableFrom<Input::Item>,
{
    transform_into(exec, input, crate::op::Identity, output)
}

/// Applies a unary operation and returns its result in owned device storage.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, op, vector::transform};
///
/// struct AddOne;
///
/// #[cubecl::cube]
/// impl op::UnaryOp<u32> for AddOne {
///     type Output = u32;
///
///     fn apply(value: u32) -> u32 {
///         value + 1
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[1_u32, 2, 3]);
/// let output = transform(&exec, input.slice(..), AddOne).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![2, 3, 4]);
/// ```
pub fn transform<R, Input, Op>(
    exec: &Executor<R>,
    input: Input,
    op: Op,
) -> Result<MVec<R, Op::Output>, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Op: UnaryOp<Input::Item>,
    Op::Output: ToCanonical<R>,
{
    let len = input.len()? as usize;
    let output = exec.alloc::<Op::Output>(len);
    transform_into(exec, input, op, output.slice_mut(..))?;
    Ok(output)
}

/// Applies a unary operation into caller-provided storage.
#[doc(hidden)]
pub(crate) fn transform_into<R, Input, Output, Op>(
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
    Output::Item: WritableFrom<<Op as UnaryOp<Input::Item>>::Output>,
{
    crate::transform::transform_fixed(
        exec,
        crate::api::iter::lower_fixed::<R, _>(input),
        op,
        output.lower_output(),
    )
}
