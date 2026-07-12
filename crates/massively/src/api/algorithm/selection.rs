use cubecl::prelude::Runtime;

use crate::{
    Error, Executor, MCanonical, MIndex, MIter, MIterMut, MStorage, MVec, WriteFrom,
    op::PredicateOp, op::UnaryOp,
};

/// Copies rows whose stencil is nonzero.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, vector::copy_where};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[10_u32, 20, 30, 40]);
/// let stencil = exec.to_device(&[1_u32, 0, 1, 0]);
/// let output = copy_where(&exec, input.slice(..), stencil.slice(..)).unwrap();
///
/// assert_eq!(output.len(), 2);
/// assert_eq!(exec.to_host(&output).unwrap(), vec![10, 30]);
/// ```
pub fn copy_where<R, Input, Stencil>(
    exec: &Executor<R>,
    input: Input,
    stencil: Stencil,
) -> Result<MVec<R, Input::Item>, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Input::Item: MCanonical<R>,
    Stencil: MIter<R, Item = MIndex>,
{
    let capacity = input.len()? as usize;
    let mut output = exec.alloc_mvec::<Input::Item>(capacity);
    let len = copy_where_into(exec, input, stencil, output.slice_mut(..))?;
    output.truncate(len);
    Ok(output)
}

/// Copies rows whose stencil is nonzero into caller-provided storage.
#[doc(hidden)]
pub(crate) fn copy_where_into<R, Input, Stencil, Output>(
    exec: &Executor<R>,
    input: Input,
    stencil: Stencil,
    output: Output,
) -> Result<MIndex, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Stencil: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
    Output::Item: WriteFrom<Input::Item>,
{
    let stencil = stencil.materialize_u32(exec)?;
    input.select_with_flags(exec, stencil.column(), false, output)
}

/// Copies rows whose stencil is zero.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, vector::remove_where};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[10_u32, 20, 30, 40]);
/// let stencil = exec.to_device(&[1_u32, 0, 1, 0]);
/// let output = remove_where(&exec, input.slice(..), stencil.slice(..)).unwrap();
///
/// assert_eq!(output.len(), 2);
/// assert_eq!(exec.to_host(&output).unwrap(), vec![20, 40]);
/// ```
pub fn remove_where<R, Input, Stencil>(
    exec: &Executor<R>,
    input: Input,
    stencil: Stencil,
) -> Result<MVec<R, Input::Item>, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Input::Item: MCanonical<R>,
    Stencil: MIter<R, Item = MIndex>,
{
    let capacity = input.len()? as usize;
    let mut output = exec.alloc_mvec::<Input::Item>(capacity);
    let len = remove_where_into(exec, input, stencil, output.slice_mut(..))?;
    output.truncate(len);
    Ok(output)
}

/// Copies rows whose stencil is zero into caller-provided storage.
#[doc(hidden)]
pub(crate) fn remove_where_into<R, Input, Stencil, Output>(
    exec: &Executor<R>,
    input: Input,
    stencil: Stencil,
    output: Output,
) -> Result<MIndex, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Stencil: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
    Output::Item: WriteFrom<Input::Item>,
{
    let stencil = stencil.materialize_u32(exec)?;
    input.select_with_flags(exec, stencil.column(), true, output)
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
/// use massively::{Executor, op::PredicateOp, vector::partition};
///
/// struct Even;
///
/// #[cubecl::cube]
/// impl PredicateOp<u32> for Even {
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
pub fn partition<R, Input, Pred>(
    exec: &Executor<R>,
    input: Input,
    pred: Pred,
) -> Result<(MVec<R, Input::Item>, MIndex), Error>
where
    R: Runtime,
    Input: MIter<R>,
    Input::Item: MCanonical<R>,
    Pred: PredicateOp<Input::Item>,
{
    let len = input.len()? as usize;
    let output = exec.alloc_mvec::<Input::Item>(len);
    let boundary = partition_into(exec, input, pred, output.slice_mut(..))?;
    Ok((output, boundary))
}

/// Stably partitions into caller-provided storage.
#[doc(hidden)]
pub(crate) fn partition_into<R, Input, Output, Pred>(
    exec: &Executor<R>,
    input: Input,
    pred: Pred,
    output: Output,
) -> Result<MIndex, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Output: MIterMut<R>,
    Output::Item: WriteFrom<Input::Item>,
    Pred: PredicateOp<Input::Item>,
{
    input.partition_with(exec, pred, output)
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
/// let output = fill(&exec, 4, 7_u32).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![7, 7, 7, 7]);
/// ```
pub fn fill<R, Item>(exec: &Executor<R>, len: usize, value: Item) -> Result<MVec<R, Item>, Error>
where
    R: Runtime,
    Item: MCanonical<R>,
{
    let output = exec.alloc_mvec::<Item>(len);
    let value = <Item::Canonical as WriteFrom<Item>>::write_from(value);
    fill_into(exec, value, output.slice_mut(..))?;
    Ok(output)
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

/// Replaces output items whose stencil is nonzero.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, vector::replace_where};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let stencil = exec.to_device(&[0_u32, 1, 0, 1]);
/// let output = exec.to_device(&[10_u32, 20, 30, 40]);
///
/// replace_where(
///     &exec,
///     99,
///     stencil.slice(..),
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
    Stencil: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
{
    let stencil = stencil.materialize_u32(exec)?;
    output.replace_with_flags(exec, value, stencil.column())
}

/// Applies an operation where the stencil is nonzero.
///
/// A zero stencil leaves the corresponding output item unchanged.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, op::UnaryOp, vector::transform_where};
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
/// let stencil = exec.to_device(&[1_u32, 0, 1]);
/// let output = exec.to_device(&[100_u32, 100, 100]);
///
/// transform_where(
///     &exec,
///     input.slice(..),
///     AddOne,
///     stencil.slice(..),
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
    Stencil: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
    Op: UnaryOp<Input::Item>,
    Output::Item: WriteFrom<<Op as UnaryOp<Input::Item>>::Output>,
{
    let stencil = stencil.materialize_u32(exec)?;
    input.transform_where_with(exec, op, stencil.column(), output)
}
