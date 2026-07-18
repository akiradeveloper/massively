#![allow(private_bounds)]

use cubecl::prelude::Runtime;

use crate::{
    Error, Executor, MItem, MIter, MIterMut, MStorage, MVec, RadixKey, op::BinaryPredicateOp,
    op::ReductionOp,
};

/// Stably sorts keys and applies the same ordering to values.
///
/// Keys are compared directly and are not materialized into owned storage.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op, Executor, vector::sort_by_key};
///
/// struct Less;
///
/// #[cubecl::cube]
/// impl op::BinaryPredicateOp<u32> for Less {
///     fn apply(lhs: u32, rhs: u32) -> bool {
///         lhs < rhs
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let keys = exec.to_device(&[3_u32, 1, 2]);
/// let values = exec.to_device(&[30_u32, 10, 20]);
/// let output = sort_by_key(
///     &exec,
///     keys.slice(..),
///     values.slice(..),
///     Less,
/// )
/// .unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![10, 20, 30]);
/// ```
pub fn sort_by_key<R, Keys, Values, Less>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    less: Less,
) -> Result<MVec<R, Values::Item>, Error>
where
    R: Runtime,
    Keys: MIter<R>,
    Values: MIter<R>,
    Values::Item: MItem<R>,
    Less: BinaryPredicateOp<Keys::Item>,
{
    let len = keys.len()? as usize;
    let value_output = exec.alloc::<Values::Item>(len);
    sort_values_by_key_into(exec, keys, values, less, value_output.slice_mut(..))?;
    Ok(value_output)
}

/// Stably radix-sorts keys in ascending order and applies the same ordering to values.
///
/// Integer keys use their numeric order. Floating-point keys use the same total
/// ordering as `f32::total_cmp` and `f64::total_cmp`. Compound keys produced by
/// `zip2` through `zip12` are ordered lexicographically from left to right.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, vector::radix_sort_by_key};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let keys = exec.to_device(&[3_i32, -1, 2, -1]);
/// let values = exec.to_device(&[30_u32, 10, 20, 11]);
/// let output = radix_sort_by_key(&exec, keys.slice(..), values.slice(..)).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![10, 11, 20, 30]);
/// ```
pub fn radix_sort_by_key<R, Keys, Values>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
) -> Result<MVec<R, Values::Item>, Error>
where
    R: Runtime,
    Keys: MIter<R>,
    Keys::Item: RadixKey<R>,
    Values: MIter<R>,
    Values::Item: MItem<R>,
{
    let len = keys.len()?;
    let value_output = exec.alloc::<Values::Item>(len);
    radix_sort_values_by_key_into(exec, keys, values, value_output.slice_mut(..))?;
    Ok(value_output)
}

/// Stably radix-sorts values into caller-provided storage.
#[doc(hidden)]
pub(crate) fn radix_sort_values_by_key_into<R, Keys, Values, ValueOutput>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    value_output: ValueOutput,
) -> Result<(), Error>
where
    R: Runtime,
    Keys: MIter<R>,
    Keys::Item: RadixKey<R>,
    Values: MIter<R>,
    Values::Item: MItem<R>,
    ValueOutput: MIterMut<R, Item = Values::Item>,
{
    let key_len = keys.len()?;
    let value_len = values.len()?;
    if key_len != value_len {
        return Err(Error::LengthMismatch {
            left: key_len,
            right: value_len,
        });
    }
    let key_storage = crate::api::algorithm::transform(exec, keys, crate::op::Identity)?;
    let permutation = <Keys::Item as RadixKey<R>>::radix_permutation(exec, &key_storage, key_len)?;
    crate::indexed::apply_permutation(
        exec,
        crate::api::iter::lower_fixed::<R, _>(values),
        permutation.column(),
        value_output.lower_output(),
    )
}

/// Stably sorts values using a key-derived permutation without materializing keys.
#[doc(hidden)]
pub(crate) fn sort_values_by_key_into<R, Keys, Values, Less, ValueOutput>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    less: Less,
    value_output: ValueOutput,
) -> Result<(), Error>
where
    R: Runtime,
    Keys: MIter<R>,
    Values: MIter<R>,
    Values::Item: MItem<R>,
    Less: BinaryPredicateOp<Keys::Item>,
    ValueOutput: MIterMut<R, Item = Values::Item>,
{
    let key_len = keys.len()?;
    let value_len = values.len()?;
    if key_len != value_len {
        return Err(Error::LengthMismatch {
            left: key_len,
            right: value_len,
        });
    }
    let permutation = crate::ordering::sort_control_with(
        exec,
        crate::api::iter::lower_fixed::<R, _>(keys),
        less,
    )?;
    crate::indexed::apply_permutation(
        exec,
        crate::api::iter::lower_fixed::<R, _>(values),
        permutation.column(),
        value_output.lower_output(),
    )
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
    Keys::Item: crate::api::iter::SortAbi<R>,
    Values: MIter<R>,
    Values::Item: MItem<R>,
    Less: BinaryPredicateOp<Keys::Item>,
    KeyOutput: MIterMut<R, Item = Keys::Item>,
    ValueOutput: MIterMut<R, Item = Values::Item>,
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
    let ordering =
        crate::ordering::sort_keys_with_control(exec, keys, less, key_output.lower_output())?;
    crate::indexed::apply_permutation(
        exec,
        values,
        ordering.permutation().column(),
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
/// use massively::{op, Executor, vector::inclusive_scan_by_key};
///
/// struct Equal;
/// struct Add;
///
/// #[cubecl::cube]
/// impl op::BinaryPredicateOp<u32> for Equal {
///     fn apply(lhs: u32, rhs: u32) -> bool {
///         lhs == rhs
///     }
/// }
///
/// #[cubecl::cube]
/// impl op::ReductionOp<u32> for Add {
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
    Values::Item: MItem<R>,
    Equal: BinaryPredicateOp<Keys::Item>,
    Op: ReductionOp<Values::Item>,
{
    let len = values.len()? as usize;
    let output = exec.alloc::<Values::Item>(len);
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
    Values::Item: MItem<R>,
    Values::Item: crate::api::iter::ScratchAbi<R>,
    Equal: BinaryPredicateOp<Keys::Item>,
    Op: ReductionOp<Values::Item>,
    Output: MIterMut<R, Item = Values::Item>,
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
/// use massively::{op, Executor, vector::exclusive_scan_by_key};
///
/// struct Equal;
/// struct Add;
///
/// #[cubecl::cube]
/// impl op::BinaryPredicateOp<u32> for Equal {
///     fn apply(lhs: u32, rhs: u32) -> bool {
///         lhs == rhs
///     }
/// }
///
/// #[cubecl::cube]
/// impl op::ReductionOp<u32> for Add {
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
    Values::Item: MItem<R>,
    Equal: BinaryPredicateOp<Keys::Item>,
    Op: ReductionOp<Values::Item>,
{
    let len = values.len()? as usize;
    let output = exec.alloc::<Values::Item>(len);
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
    Values::Item: MItem<R> + crate::api::iter::ScratchAbi<R>,
    Equal: BinaryPredicateOp<Keys::Item>,
    Op: ReductionOp<Values::Item>,
    Output: MIterMut<R, Item = Values::Item>,
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
/// use massively::{op, Executor, vector::reduce_by_key};
///
/// struct Equal;
/// struct Add;
///
/// #[cubecl::cube]
/// impl op::BinaryPredicateOp<u32> for Equal {
///     fn apply(lhs: u32, rhs: u32) -> bool {
///         lhs == rhs
///     }
/// }
///
/// #[cubecl::cube]
/// impl op::ReductionOp<u32> for Add {
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
    KeyItem: MItem<R>,
    Values: MIter<R, Item = ValueItem>,
    ValueItem: MItem<R>,
    Equal: BinaryPredicateOp<KeyItem>,
    Op: ReductionOp<ValueItem>,
{
    let capacity = keys.len()? as usize;
    let mut key_output = exec.alloc::<KeyItem>(capacity);
    let mut value_output = exec.alloc::<ValueItem>(capacity);
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
    key_output.truncate(len as usize);
    value_output.truncate(len as usize);
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
) -> Result<u32, Error>
where
    R: Runtime,
    Keys: MIter<R>,
    Keys::Item: MItem<R>,
    Values: MIter<R>,
    Values::Item: MItem<R> + crate::api::iter::ScratchAbi<R>,
    Equal: BinaryPredicateOp<Keys::Item>,
    Op: ReductionOp<Values::Item>,
    KeyOutput: MIterMut<R, Item = Keys::Item>,
    ValueOutput: MIterMut<R, Item = Values::Item>,
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

/// Keeps the first value of every adjacent equal-key run.
///
/// Keys are compared directly and are not materialized into owned storage.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op, Executor, vector::unique_by_key};
///
/// struct Equal;
///
/// #[cubecl::cube]
/// impl op::BinaryPredicateOp<u32> for Equal {
///     fn apply(lhs: u32, rhs: u32) -> bool {
///         lhs == rhs
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let keys = exec.to_device(&[1_u32, 1, 2, 2, 3]);
/// let values = exec.to_device(&[10_u32, 11, 20, 21, 30]);
/// let output = unique_by_key(
///     &exec,
///     keys.slice(..),
///     values.slice(..),
///     Equal,
/// )
/// .unwrap();
///
/// assert_eq!(
///     exec.to_host(&output).unwrap(),
///     vec![10, 20, 30],
/// );
/// ```
pub fn unique_by_key<R, Keys, Values, Equal>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    equal: Equal,
) -> Result<MVec<R, Values::Item>, Error>
where
    R: Runtime,
    Keys: MIter<R>,
    Values: MIter<R>,
    Values::Item: MItem<R>,
    Equal: BinaryPredicateOp<Keys::Item>,
{
    let capacity = keys.len()? as usize;
    let mut value_output = exec.alloc::<Values::Item>(capacity);
    let len = unique_by_key_into(exec, keys, values, equal, value_output.slice_mut(..))?;
    value_output.truncate(len as usize);
    Ok(value_output)
}

/// Keeps the first value of each unique key run in caller-provided storage.
#[doc(hidden)]
pub(crate) fn unique_by_key_into<R, Keys, Values, Equal, ValueOutput>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    equal: Equal,
    value_output: ValueOutput,
) -> Result<u32, Error>
where
    R: Runtime,
    Keys: MIter<R>,
    Values: MIter<R>,
    Values::Item: MItem<R>,
    Equal: BinaryPredicateOp<Keys::Item>,
    ValueOutput: MIterMut<R, Item = Values::Item>,
{
    crate::core::by_key::unique_by_key(
        exec,
        crate::api::iter::lower_fixed::<R, _>(keys),
        crate::api::iter::lower_fixed::<R, _>(values),
        equal,
        value_output.lower_output(),
    )
}
