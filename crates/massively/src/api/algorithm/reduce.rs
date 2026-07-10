use cubecl::prelude::Runtime;

use crate::{Error, Executor, MIter, ReductionOp};

/// Reduces all input items, starting from `init`.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, ReductionOp, reduce};
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
///
/// let sum = reduce(&exec, input.slice(..), 0, Add).unwrap();
///
/// assert_eq!(sum, 10);
/// ```
pub fn reduce<R, Input, Op>(
    exec: &Executor<R>,
    input: Input,
    init: Input::Item,
    op: Op,
) -> Result<Input::Item, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Op: ReductionOp<Input::Item>,
{
    input.reduce_with(exec, init, op)
}
