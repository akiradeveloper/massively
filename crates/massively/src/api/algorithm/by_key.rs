use cubecl::prelude::Runtime;

use crate::{
    Error, Executor, MIndex, MIter, MIterMut, MStorage, MVec, Materializable, WritableFrom,
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
pub fn sort_by_key<R, Keys, Values, KeyItem, ValueItem, Less>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    less: Less,
) -> Result<(MVec<R, KeyItem>, MVec<R, ValueItem>), Error>
where
    R: Runtime,
    Keys: MIter<R, Item = KeyItem>,
    KeyItem: Materializable<R>,
    Values: MIter<R, Item = ValueItem>,
    ValueItem: Materializable<R>,
    Less: BinaryPredicateOp<KeyItem>,
{
    let len = keys.len()? as usize;
    let key_output = exec.alloc_mvec::<KeyItem>(len);
    let value_output = exec.alloc_mvec::<ValueItem>(len);
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
    KeyOutput::Item: WritableFrom<Keys::Item>,
    ValueOutput: MIterMut<R>,
    ValueOutput::Item: WritableFrom<Values::Item>,
{
    let key_len = keys.len()?;
    let value_len = values.len()?;
    if key_len != value_len {
        return Err(Error::LengthMismatch {
            left: key_len as usize,
            right: value_len as usize,
        });
    }
    let keys = crate::api::iter::lower_fixed::<R, _>(keys);
    let values = crate::api::iter::lower_fixed::<R, _>(values);
    let permutation = crate::ordering::sort_control_with(exec, keys.clone(), less)?;
    crate::indexed::apply_permutation(exec, keys, permutation.column(), key_output.lower_output())?;
    crate::indexed::apply_permutation(
        exec,
        values,
        permutation.column(),
        value_output.lower_output(),
    )
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
    Values::Item: Materializable<R>,
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
    Values::Item: Materializable<R>,
    Equal: BinaryPredicateOp<Keys::Item>,
    Op: ReductionOp<Values::Item>,
    Output: MIterMut<R>,
    Output::Item: WritableFrom<Values::Item>,
{
    crate::core::by_key::inclusive_scan_by_key_lowered(
        exec,
        crate::api::iter::lower_fixed::<R, _>(keys),
        crate::api::iter::lower_fixed::<R, _>(values),
        equal,
        op,
        output.lower_output(),
    )
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
    Values::Item: Materializable<R>,
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
    Values::Item: Materializable<R>,
    Equal: BinaryPredicateOp<Keys::Item>,
    Op: ReductionOp<Values::Item>,
    Output: MIterMut<R>,
    Output::Item: WritableFrom<Values::Item>,
{
    crate::core::by_key::exclusive_scan_by_key_lowered(
        exec,
        crate::api::iter::lower_fixed::<R, _>(keys),
        crate::api::iter::lower_fixed::<R, _>(values),
        equal,
        init,
        op,
        output.lower_output(),
    )
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
pub fn reduce_by_key<R, Keys, Values, KeyItem, ValueItem, Equal, Op>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    equal: Equal,
    init: ValueItem,
    op: Op,
) -> Result<(MVec<R, KeyItem>, MVec<R, ValueItem>), Error>
where
    R: Runtime,
    Keys: MIter<R, Item = KeyItem>,
    KeyItem: Materializable<R>,
    Values: MIter<R, Item = ValueItem>,
    ValueItem: Materializable<R>,
    Equal: BinaryPredicateOp<KeyItem>,
    Op: ReductionOp<ValueItem>,
{
    let capacity = keys.len()? as usize;
    let mut key_output = exec.alloc_mvec::<KeyItem>(capacity);
    let mut value_output = exec.alloc_mvec::<ValueItem>(capacity);
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
    Values::Item: Materializable<R>,
    Equal: BinaryPredicateOp<Keys::Item>,
    Op: ReductionOp<Values::Item>,
    KeyOutput: MIterMut<R>,
    KeyOutput::Item: WritableFrom<Keys::Item>,
    ValueOutput: MIterMut<R>,
    ValueOutput::Item: WritableFrom<Values::Item>,
{
    crate::core::by_key::reduce_by_key_lowered(
        exec,
        crate::api::iter::lower_fixed::<R, _>(keys),
        crate::api::iter::lower_fixed::<R, _>(values),
        equal,
        init,
        op,
        key_output.lower_output(),
        value_output.lower_output(),
    )
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
pub fn unique_by_key<R, Keys, Values, KeyItem, ValueItem, Equal>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    equal: Equal,
) -> Result<(MVec<R, KeyItem>, MVec<R, ValueItem>), Error>
where
    R: Runtime,
    Keys: MIter<R, Item = KeyItem>,
    KeyItem: Materializable<R>,
    Values: MIter<R, Item = ValueItem>,
    ValueItem: Materializable<R>,
    Equal: BinaryPredicateOp<KeyItem>,
{
    let capacity = keys.len()? as usize;
    let mut key_output = exec.alloc_mvec::<KeyItem>(capacity);
    let mut value_output = exec.alloc_mvec::<ValueItem>(capacity);
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
    KeyOutput::Item: WritableFrom<Keys::Item>,
    ValueOutput: MIterMut<R>,
    ValueOutput::Item: WritableFrom<Values::Item>,
{
    crate::core::by_key::unique_by_key(
        exec,
        crate::api::iter::lower_fixed::<R, _>(keys),
        crate::api::iter::lower_fixed::<R, _>(values),
        equal,
        key_output.lower_output(),
        value_output.lower_output(),
    )
}
