#![allow(private_bounds)]

use cubecl::prelude::Runtime;

use crate::{
    Error, Executor, MIter, MIterMut, MStorage, MVec, ToCanonical, WritableFrom,
    op::BinaryPredicateOp,
};

/// Stably merges two sorted ranges.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op, Executor, vector::merge};
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
/// let left = exec.to_device(&[1_u32, 3, 5]);
/// let right = exec.to_device(&[2_u32, 4, 6]);
/// let output = merge(&exec, left.slice(..), right.slice(..), Less).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![1, 2, 3, 4, 5, 6]);
/// ```
pub fn merge<R, Left, Right, Item, Less>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    less: Less,
) -> Result<MVec<R, Item>, Error>
where
    R: Runtime,
    Left: MIter<R, Item = Item>,
    Right: MIter<R, Item = Item>,
    Item: ToCanonical<R>,
    Less: BinaryPredicateOp<Item>,
{
    let left_len = left.len()? as usize;
    let right_len = right.len()? as usize;
    let len = left_len
        .checked_add(right_len)
        .ok_or(Error::LengthTooLarge { len: usize::MAX })?;
    let output = exec.alloc::<Item>(len);
    merge_into(exec, left, right, less, output.slice_mut(..))?;
    Ok(output)
}

/// Stably merges two sorted ranges into caller-provided storage.
#[doc(hidden)]
pub(crate) fn merge_into<R, Left, Right, Less, Output>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    less: Less,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Left: MIter<R>,
    Right: MIter<R, Item = Left::Item>,
    Left::Item: crate::api::iter::CanonicalAbi<R>,
    Less: BinaryPredicateOp<Left::Item>,
    Output: MIterMut<R>,
    Output::Item: WritableFrom<Left::Item>,
{
    let left =
        crate::allocation::normalize_lowered(exec, crate::api::iter::lower_fixed::<R, _>(left))?;
    let right =
        crate::allocation::normalize_lowered(exec, crate::api::iter::lower_fixed::<R, _>(right))?;
    let control = crate::merge::merge_control_fixed(
        exec,
        crate::read::FixedReassociate::<_, Left::Item>::new(crate::CanonicalStorage::read(&left)),
        crate::read::FixedReassociate::<_, Left::Item>::new(crate::CanonicalStorage::read(&right)),
        less,
    )?;
    crate::merge::apply_canonical(exec, &left, &right, &control, output.lower_output())
}

/// Stably merges key/value ranges using the ordering of the keys.
///
/// Keys are compared directly and are not materialized into owned storage.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op, Executor, vector::merge_by_key};
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
/// let left_keys = exec.to_device(&[1_u32, 3]);
/// let left_values = exec.to_device(&[10_u32, 30]);
/// let right_keys = exec.to_device(&[2_u32, 4]);
/// let right_values = exec.to_device(&[20_u32, 40]);
/// let output = merge_by_key(
///     &exec,
///     left_keys.slice(..),
///     left_values.slice(..),
///     right_keys.slice(..),
///     right_values.slice(..),
///     Less,
/// )
/// .unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![10, 20, 30, 40]);
/// ```
pub fn merge_by_key<R, LeftKeys, LeftValues, RightKeys, RightValues, Less>(
    exec: &Executor<R>,
    left_keys: LeftKeys,
    left_values: LeftValues,
    right_keys: RightKeys,
    right_values: RightValues,
    less: Less,
) -> Result<MVec<R, LeftValues::Item>, Error>
where
    R: Runtime,
    LeftKeys: MIter<R>,
    LeftValues: MIter<R>,
    RightKeys: MIter<R, Item = LeftKeys::Item>,
    RightValues: MIter<R, Item = LeftValues::Item>,
    LeftValues::Item: ToCanonical<R>,
    Less: BinaryPredicateOp<LeftKeys::Item>,
{
    let left_len = left_keys.len()? as usize;
    let right_len = right_keys.len()? as usize;
    let len = left_len
        .checked_add(right_len)
        .ok_or(Error::LengthTooLarge { len: usize::MAX })?;
    let value_output = exec.alloc::<LeftValues::Item>(len);
    merge_by_key_into(
        exec,
        left_keys,
        left_values,
        right_keys,
        right_values,
        less,
        value_output.slice_mut(..),
    )?;
    Ok(value_output)
}

/// Stably merges values into caller-provided storage using key-derived control.
#[doc(hidden)]
pub(crate) fn merge_by_key_into<
    R,
    LeftKeys,
    LeftValues,
    RightKeys,
    RightValues,
    Less,
    ValueOutput,
>(
    exec: &Executor<R>,
    left_keys: LeftKeys,
    left_values: LeftValues,
    right_keys: RightKeys,
    right_values: RightValues,
    less: Less,
    value_output: ValueOutput,
) -> Result<(), Error>
where
    R: Runtime,
    LeftKeys: MIter<R>,
    LeftValues: MIter<R>,
    RightKeys: MIter<R, Item = LeftKeys::Item>,
    RightValues: MIter<R, Item = LeftValues::Item>,
    LeftValues::Item: crate::api::iter::CanonicalAbi<R>,
    Less: BinaryPredicateOp<LeftKeys::Item>,
    ValueOutput: MIterMut<R>,
    ValueOutput::Item: WritableFrom<LeftValues::Item>,
{
    let control = crate::merge::merge_control_fixed(
        exec,
        crate::api::iter::lower_fixed::<R, _>(left_keys),
        crate::api::iter::lower_fixed::<R, _>(right_keys),
        less,
    )?;

    let left_values = crate::allocation::normalize_lowered(
        exec,
        crate::api::iter::lower_fixed::<R, _>(left_values),
    )?;
    let right_values = crate::allocation::normalize_lowered(
        exec,
        crate::api::iter::lower_fixed::<R, _>(right_values),
    )?;
    crate::merge::apply_canonical(
        exec,
        &left_values,
        &right_values,
        &control,
        value_output.lower_output(),
    )
}
