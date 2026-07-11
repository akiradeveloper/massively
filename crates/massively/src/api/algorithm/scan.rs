use cubecl::prelude::Runtime;

use crate::{Error, Executor, MIter, MIterMut, WriteFrom, op::ReductionOp};

/// Computes an inclusive scan into preallocated output storage.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, op::ReductionOp, vector::inclusive_scan};
///
/// struct Add;
///
/// #[cubecl::cube]
/// impl ReductionOp<u32> for Add {
///     fn apply(lhs: u32, rhs: u32) -> u32 {
///         lhs + rhs
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[1_u32, 2, 3, 4]);
/// let output = exec.alloc::<u32>(input.len());
///
/// inclusive_scan(&exec, input.slice(..), Add, output.slice_mut(..)).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![1, 3, 6, 10]);
/// ```
pub fn inclusive_scan<R, Input, Output, Op>(
    exec: &Executor<R>,
    input: Input,
    op: Op,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: MIter<R>,
    Output: MIterMut<R>,
    Output::Item: WriteFrom<Input::Item>,
    Op: ReductionOp<Input::Item>,
{
    input.inclusive_scan_with(exec, op, output)
}

/// Computes adjacent reductions while preserving the first item.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, op::ReductionOp, vector::adjacent_difference};
///
/// struct Add;
///
/// #[cubecl::cube]
/// impl ReductionOp<u32> for Add {
///     fn apply(lhs: u32, rhs: u32) -> u32 {
///         lhs + rhs
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[1_u32, 2, 3, 4]);
/// let output = exec.alloc::<u32>(input.len());
///
/// adjacent_difference(&exec, input.slice(..), Add, output.slice_mut(..)).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![1, 3, 5, 7]);
/// ```
pub fn adjacent_difference<R, Input, Output, Op>(
    exec: &Executor<R>,
    input: Input,
    op: Op,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: MIter<R>,
    Output: MIterMut<R>,
    Output::Item: WriteFrom<Input::Item>,
    Op: ReductionOp<Input::Item>,
{
    input.adjacent_difference_with(exec, op, output)
}

/// Computes an exclusive scan into preallocated output storage.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, op::ReductionOp, vector::exclusive_scan};
///
/// struct Add;
///
/// #[cubecl::cube]
/// impl ReductionOp<u32> for Add {
///     fn apply(lhs: u32, rhs: u32) -> u32 {
///         lhs + rhs
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[1_u32, 2, 3, 4]);
/// let output = exec.alloc::<u32>(input.len());
///
/// exclusive_scan(&exec, input.slice(..), 0, Add, output.slice_mut(..)).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![0, 1, 3, 6]);
/// ```
pub fn exclusive_scan<R, Input, Output, Op>(
    exec: &Executor<R>,
    input: Input,
    init: Input::Item,
    op: Op,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: MIter<R>,
    Output: MIterMut<R>,
    Output::Item: WriteFrom<Input::Item>,
    Op: ReductionOp<Input::Item>,
{
    input.exclusive_scan_with(exec, init, op, output)
}
