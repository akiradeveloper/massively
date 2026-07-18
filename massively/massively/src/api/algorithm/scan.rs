#![allow(private_bounds)]

use cubecl::prelude::Runtime;

use crate::{Error, Executor, MItem, MIter, MIterMut, MStorage, MVec, op::ReductionOp};

/// Computes an inclusive scan and returns owned device storage.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, op, vector::inclusive_scan};
///
/// struct Add;
///
/// #[cubecl::cube]
/// impl op::ReductionOp<u32> for Add {
///     fn apply(lhs: u32, rhs: u32) -> u32 {
///         lhs + rhs
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[1_u32, 2, 3, 4]);
/// let output = inclusive_scan(&exec, input.slice(..), Add).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![1, 3, 6, 10]);
/// ```
pub fn inclusive_scan<R, Input, Item, Op>(
    exec: &Executor<R>,
    input: Input,
    op: Op,
) -> Result<MVec<R, Item>, Error>
where
    R: Runtime,
    Input: MIter<R, Item = Item>,
    Item: MItem<R>,
    Op: ReductionOp<Item>,
{
    let len = input.len()? as usize;
    let output = exec.alloc::<Item>(len);
    crate::scan::inclusive_scan(
        exec,
        crate::api::iter::lower_fixed::<R, _>(input),
        op,
        output.slice_mut(..).lower_output(),
    )?;
    Ok(output)
}

/// Computes an inclusive scan into caller-provided storage.
#[doc(hidden)]
pub(crate) fn inclusive_scan_into<R, Input, Output, Op>(
    exec: &Executor<R>,
    input: Input,
    op: Op,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: MIter<R>,
    Input::Item: crate::api::iter::MItem<R>,
    Output: MIterMut<R, Item = Input::Item>,
    Op: ReductionOp<Input::Item>,
{
    crate::scan::inclusive_scan(
        exec,
        crate::api::iter::lower_fixed::<R, _>(input),
        op,
        output.lower_output(),
    )
}

/// Computes adjacent reductions while preserving the first item.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, op, vector::adjacent_difference};
///
/// struct Add;
///
/// #[cubecl::cube]
/// impl op::ReductionOp<u32> for Add {
///     fn apply(lhs: u32, rhs: u32) -> u32 {
///         lhs + rhs
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[1_u32, 2, 3, 4]);
/// let output = adjacent_difference(&exec, input.slice(..), Add).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![1, 3, 5, 7]);
/// ```
pub fn adjacent_difference<R, Input, Item, Op>(
    exec: &Executor<R>,
    input: Input,
    op: Op,
) -> Result<MVec<R, Item>, Error>
where
    R: Runtime,
    Input: MIter<R, Item = Item>,
    Item: MItem<R>,
    Op: ReductionOp<Item>,
{
    let len = input.len()? as usize;
    let output = exec.alloc::<Item>(len);
    adjacent_difference_into(exec, input, op, output.slice_mut(..))?;
    Ok(output)
}

/// Computes adjacent reductions into caller-provided storage.
#[doc(hidden)]
pub(crate) fn adjacent_difference_into<R, Input, Output, Op>(
    exec: &Executor<R>,
    input: Input,
    op: Op,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: MIter<R>,
    Input::Item: MItem<R>,
    Output: MIterMut<R, Item = Input::Item>,
    Op: ReductionOp<Input::Item>,
{
    crate::scan::adjacent_difference(
        exec,
        crate::api::iter::lower_fixed::<R, _>(input),
        op,
        output.lower_output(),
    )
}

/// Computes an exclusive scan and returns owned device storage.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, op, vector::exclusive_scan};
///
/// struct Add;
///
/// #[cubecl::cube]
/// impl op::ReductionOp<u32> for Add {
///     fn apply(lhs: u32, rhs: u32) -> u32 {
///         lhs + rhs
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[1_u32, 2, 3, 4]);
/// let output = exclusive_scan(&exec, input.slice(..), 0, Add).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![0, 1, 3, 6]);
/// ```
pub fn exclusive_scan<R, Input, Item, Op>(
    exec: &Executor<R>,
    input: Input,
    init: Item,
    op: Op,
) -> Result<MVec<R, Item>, Error>
where
    R: Runtime,
    Input: MIter<R, Item = Item>,
    Item: MItem<R>,
    Op: ReductionOp<Item>,
{
    let len = input.len()? as usize;
    let output = exec.alloc::<Item>(len);
    crate::scan::exclusive_scan(
        exec,
        crate::api::iter::lower_fixed::<R, _>(input),
        init,
        op,
        output.slice_mut(..).lower_output(),
    )?;
    Ok(output)
}
