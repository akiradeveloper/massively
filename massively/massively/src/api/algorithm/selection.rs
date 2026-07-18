use cubecl::prelude::Runtime;

use crate::{
    Error, Executor, MItem, MIter, MIterMut, MStorage, MVec, op::PredicateOp, op::UnaryOp,
};

/// Copies rows whose stencil is true.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, lazy, op::U32ToBool, vector::copy_where};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[10_u32, 20, 30, 40]);
/// let stencil = exec.to_device(&[1_u32, 0, 1, 0]);
/// let stencil = lazy::transform(stencil.slice(..), U32ToBool);
/// let output = copy_where(&exec, input.slice(..), stencil).unwrap();
///
/// assert_eq!(output.len(), 2);
/// assert_eq!(exec.to_host(&output).unwrap(), vec![10, 30]);
/// ```
pub fn copy_where<R, Input, Item, Stencil>(
    exec: &Executor<R>,
    input: Input,
    stencil: Stencil,
) -> Result<MVec<R, Item>, Error>
where
    R: Runtime,
    Input: MIter<R, Item = Item>,
    Item: MItem<R>,
    Stencil: MIter<R, Item = bool>,
{
    let capacity = input.len()? as usize;
    let mut output = exec.alloc::<Item>(capacity);
    let len = copy_where_into(exec, input, stencil, output.slice_mut(..))?;
    output.truncate(len as usize);
    Ok(output)
}

/// Copies rows whose stencil is true into caller-provided storage.
#[doc(hidden)]
pub(crate) fn copy_where_into<R, Input, Stencil, Output>(
    exec: &Executor<R>,
    input: Input,
    stencil: Stencil,
    output: Output,
) -> Result<u32, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Input::Item: MItem<R>,
    Stencil: MIter<R, Item = bool>,
    Output: MIterMut<R, Item = Input::Item>,
{
    crate::selection::copy_where(
        exec,
        crate::api::iter::lower::<R, _>(input),
        crate::api::iter::lower::<R, _>(stencil),
        output.lower_output(),
    )
}

/// Copies rows selected by a stored `u32` stencil after converting it at the read boundary.
pub(crate) fn copy_where_raw_into<R, Input, Stencil, Output>(
    exec: &Executor<R>,
    input: Input,
    stencil: Stencil,
    output: Output,
) -> Result<u32, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Input::Item: MItem<R>,
    Stencil: MIter<R, Item = u32>,
    Output: MIterMut<R, Item = Input::Item>,
{
    crate::selection::copy_where(
        exec,
        crate::api::iter::lower::<R, _>(input),
        crate::Transform::new(
            crate::api::iter::lower::<R, _>(stencil),
            crate::op::U32ToBool,
        ),
        output.lower_output(),
    )
}

/// Copies rows whose stencil is false.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, lazy, op::U32ToBool, vector::remove_where};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[10_u32, 20, 30, 40]);
/// let stencil = exec.to_device(&[1_u32, 0, 1, 0]);
/// let stencil = lazy::transform(stencil.slice(..), U32ToBool);
/// let output = remove_where(&exec, input.slice(..), stencil).unwrap();
///
/// assert_eq!(output.len(), 2);
/// assert_eq!(exec.to_host(&output).unwrap(), vec![20, 40]);
/// ```
pub fn remove_where<R, Input, Item, Stencil>(
    exec: &Executor<R>,
    input: Input,
    stencil: Stencil,
) -> Result<MVec<R, Item>, Error>
where
    R: Runtime,
    Input: MIter<R, Item = Item>,
    Item: MItem<R>,
    Stencil: MIter<R, Item = bool>,
{
    let capacity = input.len()? as usize;
    let mut output = exec.alloc::<Item>(capacity);
    let len = remove_where_into(exec, input, stencil, output.slice_mut(..))?;
    output.truncate(len as usize);
    Ok(output)
}

/// Copies rows whose stencil is false into caller-provided storage.
#[doc(hidden)]
pub(crate) fn remove_where_into<R, Input, Stencil, Output>(
    exec: &Executor<R>,
    input: Input,
    stencil: Stencil,
    output: Output,
) -> Result<u32, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Input::Item: MItem<R>,
    Stencil: MIter<R, Item = bool>,
    Output: MIterMut<R, Item = Input::Item>,
{
    crate::selection::remove_where(
        exec,
        crate::api::iter::lower::<R, _>(input),
        crate::api::iter::lower::<R, _>(stencil),
        output.lower_output(),
    )
}

/// Copies rows rejected by a stored `u32` stencil after converting it at the read boundary.
pub(crate) fn remove_where_raw_into<R, Input, Stencil, Output>(
    exec: &Executor<R>,
    input: Input,
    stencil: Stencil,
    output: Output,
) -> Result<u32, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Input::Item: MItem<R>,
    Stencil: MIter<R, Item = u32>,
    Output: MIterMut<R, Item = Input::Item>,
{
    crate::selection::remove_where(
        exec,
        crate::api::iter::lower::<R, _>(input),
        crate::Transform::new(
            crate::api::iter::lower::<R, _>(stencil),
            crate::op::U32ToBool,
        ),
        output.lower_output(),
    )
}

/// Stably partitions passing items before failing items.
///
/// The return value is the boundary between the two partitions.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, op, vector::partition};
///
/// struct Even;
///
/// #[cubecl::cube]
/// impl op::PredicateOp<u32> for Even {
///     fn apply(value: u32) -> bool {
///         value % 2 == 0
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[1_u32, 2, 3, 4]);
/// let (output, boundary) = partition(&exec, input.slice(..), Even).unwrap();
///
/// assert_eq!(boundary, 2);
/// assert_eq!(exec.to_host(&output).unwrap(), vec![2, 4, 1, 3]);
/// ```
pub fn partition<R, Input, Item, Pred>(
    exec: &Executor<R>,
    input: Input,
    pred: Pred,
) -> Result<(MVec<R, Item>, usize), Error>
where
    R: Runtime,
    Input: MIter<R, Item = Item>,
    Item: MItem<R>,
    Pred: PredicateOp<Item>,
{
    let len = input.len()? as usize;
    let output = exec.alloc::<Item>(len);
    let boundary = partition_into(exec, input, pred, output.slice_mut(..))?;
    Ok((output, boundary as usize))
}

/// Stably partitions into caller-provided storage.
#[doc(hidden)]
pub(crate) fn partition_into<R, Input, Output, Pred>(
    exec: &Executor<R>,
    input: Input,
    pred: Pred,
    output: Output,
) -> Result<u32, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Input::Item: MItem<R>,
    Output: MIterMut<R, Item = Input::Item>,
    Pred: PredicateOp<Input::Item>,
{
    crate::selection::partition(
        exec,
        crate::api::iter::lower_fixed::<R, _>(input),
        pred,
        output.lower_output(),
    )
}

/// Fills every output item with one value.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, vector::fill};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let output = exec.alloc::<u32>(4);
/// fill(&exec, 7_u32, output.slice_mut(..)).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![7, 7, 7, 7]);
/// ```
pub fn fill<R, Output>(exec: &Executor<R>, value: Output::Item, output: Output) -> Result<(), Error>
where
    R: Runtime,
    Output: MIterMut<R>,
{
    fill_into(exec, value, output)
}

/// Fills caller-provided storage with one value.
#[doc(hidden)]
pub(crate) fn fill_into<R, Output>(
    exec: &Executor<R>,
    value: Output::Item,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Output: MIterMut<R>,
{
    output.fill_with(exec, value)
}

/// Replaces output items whose stencil is true.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, lazy, op::U32ToBool, vector::replace_where};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let stencil = exec.to_device(&[0_u32, 1, 0, 1]);
/// let stencil = lazy::transform(stencil.slice(..), U32ToBool);
/// let output = exec.to_device(&[10_u32, 20, 30, 40]);
///
/// replace_where(
///     &exec,
///     99,
///     stencil,
///     output.slice_mut(..),
/// )
/// .unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![10, 99, 30, 99]);
/// ```
pub fn replace_where<R, Stencil, Output>(
    exec: &Executor<R>,
    value: Output::Item,
    stencil: Stencil,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Stencil: MIter<R, Item = bool>,
    Output: MIterMut<R>,
{
    crate::selection::replace_where(
        exec,
        value,
        crate::api::iter::lower::<R, _>(stencil),
        output.lower_output(),
    )
}

/// Applies an operation where the stencil is true.
///
/// A false stencil leaves the corresponding output item unchanged.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{
///     Executor, lazy,
///     op::{self, U32ToBool},
///     vector::transform_where,
/// };
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
/// let stencil = exec.to_device(&[1_u32, 0, 1]);
/// let stencil = lazy::transform(stencil.slice(..), U32ToBool);
/// let output = exec.to_device(&[100_u32, 100, 100]);
///
/// transform_where(
///     &exec,
///     input.slice(..),
///     AddOne,
///     stencil,
///     output.slice_mut(..),
/// )
/// .unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![2, 100, 4]);
/// ```
pub fn transform_where<R, Input, Stencil, Output, Op>(
    exec: &Executor<R>,
    input: Input,
    op: Op,
    stencil: Stencil,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: MIter<R>,
    Stencil: MIter<R, Item = bool>,
    Output: MIterMut<R, Item = <Op as UnaryOp<Input::Item>>::Output>,
    Op: UnaryOp<Input::Item>,
    Op::Output: MItem<R>,
{
    crate::selection::transform_where(
        exec,
        crate::api::iter::lower::<R, _>(input),
        op,
        crate::api::iter::lower::<R, _>(stencil),
        output.lower_output(),
    )
}

/// Applies a selected transform from a stored `u32` stencil after converting it at the read boundary.
pub(crate) fn transform_where_raw<R, Input, Stencil, Output, Op>(
    exec: &Executor<R>,
    input: Input,
    op: Op,
    stencil: Stencil,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: MIter<R>,
    Stencil: MIter<R, Item = u32>,
    Output: MIterMut<R, Item = <Op as UnaryOp<Input::Item>>::Output>,
    Op: UnaryOp<Input::Item>,
    Op::Output: MItem<R>,
{
    crate::selection::transform_where(
        exec,
        crate::api::iter::lower::<R, _>(input),
        op,
        crate::Transform::new(
            crate::api::iter::lower::<R, _>(stencil),
            crate::op::U32ToBool,
        ),
        output.lower_output(),
    )
}
