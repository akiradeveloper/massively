use cubecl::prelude::Runtime;

use crate::{
    Error, Executor, MCanonical, MIndex, MIter, MIterMut, MStorage, MVec, WriteFrom,
    op::BinaryPredicateOp, op::ReductionOp,
};

/// Stably sorts keys and applies the same ordering to values.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op::BinaryPredicateOp, Executor, vector::sort_by_key};
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
/// let (key_output, value_output) = sort_by_key(
///     &exec,
///     keys.slice(..),
///     values.slice(..),
///     Less,
/// )
/// .unwrap();
///
/// assert_eq!(exec.to_host(&key_output).unwrap(), vec![1, 2, 3]);
/// assert_eq!(exec.to_host(&value_output).unwrap(), vec![10, 20, 30]);
/// ```
pub fn sort_by_key<R, Keys, Values, Less>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    less: Less,
) -> Result<(MVec<R, Keys::Item>, MVec<R, Values::Item>), Error>
where
    R: Runtime,
    Keys: MIter<R>,
    Keys::Item: MCanonical<R>,
    Values: MIter<R>,
    Values::Item: MCanonical<R>,
    Less: BinaryPredicateOp<Keys::Item>,
{
    let len = keys.len()? as usize;
    let key_output = exec.alloc_mvec::<Keys::Item>(len);
    let value_output = exec.alloc_mvec::<Values::Item>(len);
    sort_by_key_into(
        exec,
        keys,
        values,
        less,
        key_output.slice_mut(..),
        value_output.slice_mut(..),
    )?;
    Ok((key_output, value_output))
}

/// Stably sorts keys and values into caller-provided storage.
#[doc(hidden)]
pub(crate) fn sort_by_key_into<R, Keys, Values, Less, KeyOutput, ValueOutput>(
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
/// use massively::{op::BinaryPredicateOp, Executor, op::ReductionOp, vector::inclusive_scan_by_key};
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
/// let output = inclusive_scan_by_key(
///     &exec,
///     keys.slice(..),
///     values.slice(..),
///     Equal,
///     Add,
/// )
/// .unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![10, 30, 30, 70]);
/// ```
pub fn inclusive_scan_by_key<R, Keys, Values, Equal, Op>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    equal: Equal,
    op: Op,
) -> Result<MVec<R, Values::Item>, Error>
where
    R: Runtime,
    Keys: MIter<R>,
    Values: MIter<R>,
    Values::Item: MCanonical<R>,
    Equal: BinaryPredicateOp<Keys::Item>,
    Op: ReductionOp<Values::Item>,
{
    let len = values.len()? as usize;
    let output = exec.alloc_mvec::<Values::Item>(len);
    inclusive_scan_by_key_into(exec, keys, values, equal, op, output.slice_mut(..))?;
    Ok(output)
}

/// Computes an inclusive scan by key into caller-provided storage.
#[doc(hidden)]
pub(crate) fn inclusive_scan_by_key_into<R, Keys, Values, Equal, Op, Output>(
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
/// use massively::{op::BinaryPredicateOp, Executor, op::ReductionOp, vector::exclusive_scan_by_key};
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
/// let output = exclusive_scan_by_key(
///     &exec,
///     keys.slice(..),
///     values.slice(..),
///     Equal,
///     0,
///     Add,
/// )
/// .unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![0, 10, 0, 30]);
/// ```
pub fn exclusive_scan_by_key<R, Keys, Values, Equal, Op>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    equal: Equal,
    init: Values::Item,
    op: Op,
) -> Result<MVec<R, Values::Item>, Error>
where
    R: Runtime,
    Keys: MIter<R>,
    Values: MIter<R>,
    Values::Item: MCanonical<R>,
    Equal: BinaryPredicateOp<Keys::Item>,
    Op: ReductionOp<Values::Item>,
{
    let len = values.len()? as usize;
    let output = exec.alloc_mvec::<Values::Item>(len);
    exclusive_scan_by_key_into(exec, keys, values, equal, init, op, output.slice_mut(..))?;
    Ok(output)
}

/// Computes an exclusive scan by key into caller-provided storage.
#[doc(hidden)]
pub(crate) fn exclusive_scan_by_key_into<R, Keys, Values, Equal, Op, Output>(
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
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op::BinaryPredicateOp, Executor, op::ReductionOp, vector::reduce_by_key};
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
/// let (key_output, value_output) = reduce_by_key(
///     &exec,
///     keys.slice(..),
///     values.slice(..),
///     Equal,
///     0,
///     Add,
/// )
/// .unwrap();
///
/// assert_eq!(exec.to_host(&key_output).unwrap(), vec![1, 2]);
/// assert_eq!(exec.to_host(&value_output).unwrap(), vec![30, 70]);
/// ```
pub fn reduce_by_key<R, Keys, Values, Equal, Op>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    equal: Equal,
    init: Values::Item,
    op: Op,
) -> Result<(MVec<R, Keys::Item>, MVec<R, Values::Item>), Error>
where
    R: Runtime,
    Keys: MIter<R>,
    Keys::Item: MCanonical<R>,
    Values: MIter<R>,
    Values::Item: MCanonical<R>,
    Equal: BinaryPredicateOp<Keys::Item>,
    Op: ReductionOp<Values::Item>,
{
    let capacity = keys.len()? as usize;
    let mut key_output = exec.alloc_mvec::<Keys::Item>(capacity);
    let mut value_output = exec.alloc_mvec::<Values::Item>(capacity);
    let len = reduce_by_key_into(
        exec,
        keys,
        values,
        equal,
        init,
        op,
        key_output.slice_mut(..),
        value_output.slice_mut(..),
    )?;
    key_output.truncate(len);
    value_output.truncate(len);
    Ok((key_output, value_output))
}

/// Reduces by key into caller-provided storage.
#[doc(hidden)]
pub(crate) fn reduce_by_key_into<R, Keys, Values, Equal, Op, KeyOutput, ValueOutput>(
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
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op::BinaryPredicateOp, Executor, vector::unique_by_key};
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
/// let (key_output, value_output) = unique_by_key(
///     &exec,
///     keys.slice(..),
///     values.slice(..),
///     Equal,
/// )
/// .unwrap();
///
/// assert_eq!(
///     exec.to_host(&key_output).unwrap(),
///     vec![1, 2, 3],
/// );
/// assert_eq!(
///     exec.to_host(&value_output).unwrap(),
///     vec![10, 20, 30],
/// );
/// ```
pub fn unique_by_key<R, Keys, Values, Equal>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    equal: Equal,
) -> Result<(MVec<R, Keys::Item>, MVec<R, Values::Item>), Error>
where
    R: Runtime,
    Keys: MIter<R>,
    Keys::Item: MCanonical<R>,
    Values: MIter<R>,
    Values::Item: MCanonical<R>,
    Equal: BinaryPredicateOp<Keys::Item>,
{
    let capacity = keys.len()? as usize;
    let mut key_output = exec.alloc_mvec::<Keys::Item>(capacity);
    let mut value_output = exec.alloc_mvec::<Values::Item>(capacity);
    let len = unique_by_key_into(
        exec,
        keys,
        values,
        equal,
        key_output.slice_mut(..),
        value_output.slice_mut(..),
    )?;
    key_output.truncate(len);
    value_output.truncate(len);
    Ok((key_output, value_output))
}

/// Keeps unique key/value runs in caller-provided storage.
#[doc(hidden)]
pub(crate) fn unique_by_key_into<R, Keys, Values, Equal, KeyOutput, ValueOutput>(
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
