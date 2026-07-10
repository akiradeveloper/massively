use cubecl::prelude::Runtime;

use crate::{BinaryPredicateOp, Error, Executor, MIndex, MIter, MIterMut, ReductionOp, WriteFrom};

/// Stably sorts keys and applies the same ordering to values.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{BinaryPredicateOp, Executor, sort_by_key};
///
/// struct Less;
///
/// #[cubecl::cube]
/// impl BinaryPredicateOp<u32> for Less {
///     fn apply(lhs: u32, rhs: u32) -> bool {
///         lhs < rhs
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let keys = exec.to_device(&[3_u32, 1, 2]);
/// let values = exec.to_device(&[30_u32, 10, 20]);
/// let key_output = exec.alloc::<u32>(keys.len());
/// let value_output = exec.alloc::<u32>(values.len());
///
/// sort_by_key(
///     &exec,
///     keys.slice(..),
///     values.slice(..),
///     Less,
///     key_output.slice_mut(..),
///     value_output.slice_mut(..),
/// )
/// .unwrap();
///
/// assert_eq!(exec.to_host(&key_output).unwrap(), vec![1, 2, 3]);
/// assert_eq!(exec.to_host(&value_output).unwrap(), vec![10, 20, 30]);
/// ```
pub fn sort_by_key<R, Keys, Values, Less, KeyOutput, ValueOutput>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    less: Less,
    key_output: KeyOutput,
    value_output: ValueOutput,
) -> Result<(), Error>
where
    R: Runtime,
    Keys: MIter<R>,
    Values: MIter<R>,
    Less: BinaryPredicateOp<Keys::Item>,
    KeyOutput: MIterMut<R>,
    KeyOutput::Item: WriteFrom<Keys::Item>,
    ValueOutput: MIterMut<R>,
    ValueOutput::Item: WriteFrom<Values::Item>,
{
    keys.sort_by_key_with(exec, values, less, key_output, value_output)
}

/// Computes an inclusive scan within each adjacent equal-key segment.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{BinaryPredicateOp, Executor, ReductionOp, inclusive_scan_by_key};
///
/// struct Equal;
/// struct Add;
///
/// #[cubecl::cube]
/// impl BinaryPredicateOp<u32> for Equal {
///     fn apply(lhs: u32, rhs: u32) -> bool {
///         lhs == rhs
///     }
/// }
///
/// #[cubecl::cube]
/// impl ReductionOp<u32> for Add {
///     fn apply(lhs: u32, rhs: u32) -> u32 {
///         lhs + rhs
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let keys = exec.to_device(&[1_u32, 1, 2, 2]);
/// let values = exec.to_device(&[10_u32, 20, 30, 40]);
/// let output = exec.alloc::<u32>(values.len());
///
/// inclusive_scan_by_key(
///     &exec,
///     keys.slice(..),
///     values.slice(..),
///     Equal,
///     Add,
///     output.slice_mut(..),
/// )
/// .unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![10, 30, 30, 70]);
/// ```
pub fn inclusive_scan_by_key<R, Keys, Values, Equal, Op, Output>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    equal: Equal,
    op: Op,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Keys: MIter<R>,
    Values: MIter<R>,
    Equal: BinaryPredicateOp<Keys::Item>,
    Op: ReductionOp<Values::Item>,
    Output: MIterMut<R>,
    Output::Item: WriteFrom<Values::Item>,
{
    keys.scan_by_key_with(exec, values, equal, None, op, output)
}

/// Computes an exclusive scan within each adjacent equal-key segment.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{BinaryPredicateOp, Executor, ReductionOp, exclusive_scan_by_key};
///
/// struct Equal;
/// struct Add;
///
/// #[cubecl::cube]
/// impl BinaryPredicateOp<u32> for Equal {
///     fn apply(lhs: u32, rhs: u32) -> bool {
///         lhs == rhs
///     }
/// }
///
/// #[cubecl::cube]
/// impl ReductionOp<u32> for Add {
///     fn apply(lhs: u32, rhs: u32) -> u32 {
///         lhs + rhs
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let keys = exec.to_device(&[1_u32, 1, 2, 2]);
/// let values = exec.to_device(&[10_u32, 20, 30, 40]);
/// let output = exec.alloc::<u32>(values.len());
///
/// exclusive_scan_by_key(
///     &exec,
///     keys.slice(..),
///     values.slice(..),
///     Equal,
///     0,
///     Add,
///     output.slice_mut(..),
/// )
/// .unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![0, 10, 0, 30]);
/// ```
pub fn exclusive_scan_by_key<R, Keys, Values, Equal, Op, Output>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    equal: Equal,
    init: Values::Item,
    op: Op,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Keys: MIter<R>,
    Values: MIter<R>,
    Equal: BinaryPredicateOp<Keys::Item>,
    Op: ReductionOp<Values::Item>,
    Output: MIterMut<R>,
    Output::Item: WriteFrom<Values::Item>,
{
    keys.scan_by_key_with(exec, values, equal, Some(init), op, output)
}

/// Reduces each adjacent equal-key segment.
///
/// The return value is the number of segments written at the beginning of both outputs.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{BinaryPredicateOp, Executor, ReductionOp, reduce_by_key};
///
/// struct Equal;
/// struct Add;
///
/// #[cubecl::cube]
/// impl BinaryPredicateOp<u32> for Equal {
///     fn apply(lhs: u32, rhs: u32) -> bool {
///         lhs == rhs
///     }
/// }
///
/// #[cubecl::cube]
/// impl ReductionOp<u32> for Add {
///     fn apply(lhs: u32, rhs: u32) -> u32 {
///         lhs + rhs
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let keys = exec.to_device(&[1_u32, 1, 2, 2]);
/// let values = exec.to_device(&[10_u32, 20, 30, 40]);
/// let key_output = exec.alloc::<u32>(keys.len());
/// let value_output = exec.alloc::<u32>(values.len());
///
/// let len = reduce_by_key(
///     &exec,
///     keys.slice(..),
///     values.slice(..),
///     Equal,
///     0,
///     Add,
///     key_output.slice_mut(..),
///     value_output.slice_mut(..),
/// )
/// .unwrap();
///
/// assert_eq!(len, 2);
/// assert_eq!(exec.to_host(&key_output.slice(..len as usize)).unwrap(), vec![1, 2]);
/// assert_eq!(exec.to_host(&value_output.slice(..len as usize)).unwrap(), vec![30, 70]);
/// ```
pub fn reduce_by_key<R, Keys, Values, Equal, Op, KeyOutput, ValueOutput>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    equal: Equal,
    init: Values::Item,
    op: Op,
    key_output: KeyOutput,
    value_output: ValueOutput,
) -> Result<MIndex, Error>
where
    R: Runtime,
    Keys: MIter<R>,
    Values: MIter<R>,
    Equal: BinaryPredicateOp<Keys::Item>,
    Op: ReductionOp<Values::Item>,
    KeyOutput: MIterMut<R>,
    KeyOutput::Item: WriteFrom<Keys::Item>,
    ValueOutput: MIterMut<R>,
    ValueOutput::Item: WriteFrom<Values::Item>,
{
    keys.reduce_by_key_with(exec, values, equal, init, op, key_output, value_output)
}

/// Keeps the first key and value of every adjacent equal-key run.
///
/// The return value is the number of runs written at the beginning of both outputs.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{BinaryPredicateOp, Executor, unique_by_key};
///
/// struct Equal;
///
/// #[cubecl::cube]
/// impl BinaryPredicateOp<u32> for Equal {
///     fn apply(lhs: u32, rhs: u32) -> bool {
///         lhs == rhs
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let keys = exec.to_device(&[1_u32, 1, 2, 2, 3]);
/// let values = exec.to_device(&[10_u32, 11, 20, 21, 30]);
/// let key_output = exec.alloc::<u32>(keys.len());
/// let value_output = exec.alloc::<u32>(values.len());
///
/// let len = unique_by_key(
///     &exec,
///     keys.slice(..),
///     values.slice(..),
///     Equal,
///     key_output.slice_mut(..),
///     value_output.slice_mut(..),
/// )
/// .unwrap();
///
/// assert_eq!(len, 3);
/// assert_eq!(
///     exec.to_host(&key_output.slice(..len as usize)).unwrap(),
///     vec![1, 2, 3],
/// );
/// assert_eq!(
///     exec.to_host(&value_output.slice(..len as usize)).unwrap(),
///     vec![10, 20, 30],
/// );
/// ```
pub fn unique_by_key<R, Keys, Values, Equal, KeyOutput, ValueOutput>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    equal: Equal,
    key_output: KeyOutput,
    value_output: ValueOutput,
) -> Result<MIndex, Error>
where
    R: Runtime,
    Keys: MIter<R>,
    Values: MIter<R>,
    Equal: BinaryPredicateOp<Keys::Item>,
    KeyOutput: MIterMut<R>,
    KeyOutput::Item: WriteFrom<Keys::Item>,
    ValueOutput: MIterMut<R>,
    ValueOutput::Item: WriteFrom<Values::Item>,
{
    keys.unique_by_key_with(exec, values, equal, key_output, value_output)
}
