use cubecl::prelude::Runtime;

use crate::{
    Error, Executor, MCanonical, MIter, MIterMut, MStorage, MVec, WriteFrom, op::BinaryPredicateOp,
};

/// Stably merges two sorted ranges.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op::BinaryPredicateOp, Executor, vector::merge};
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
/// let left = exec.to_device(&[1_u32, 3, 5]);
/// let right = exec.to_device(&[2_u32, 4, 6]);
/// let output = merge(&exec, left.slice(..), right.slice(..), Less).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![1, 2, 3, 4, 5, 6]);
/// ```
pub fn merge<R, Left, Right, Less>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    less: Less,
) -> Result<MVec<R, Left::Item>, Error>
where
    R: Runtime,
    Left: MIter<R>,
    Left::Item: MCanonical<R>,
    Right: MIter<R, Item = Left::Item>,
    Less: BinaryPredicateOp<Left::Item>,
{
    let left_len = left.len()? as usize;
    let right_len = right.len()? as usize;
    let len = left_len
        .checked_add(right_len)
        .ok_or(Error::LengthTooLarge { len: usize::MAX })?;
    let output = exec.alloc_mvec::<Left::Item>(len);
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
    Less: BinaryPredicateOp<Left::Item>,
    Output: MIterMut<R>,
    Output::Item: WriteFrom<Left::Item>,
{
    left.merge_with(exec, right, less, output)
}

/// Stably merges key/value ranges using the ordering of the keys.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op::BinaryPredicateOp, Executor, vector::merge_by_key};
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
/// let left_keys = exec.to_device(&[1_u32, 3]);
/// let left_values = exec.to_device(&[10_u32, 30]);
/// let right_keys = exec.to_device(&[2_u32, 4]);
/// let right_values = exec.to_device(&[20_u32, 40]);
/// let (key_output, value_output) = merge_by_key(
///     &exec,
///     left_keys.slice(..),
///     left_values.slice(..),
///     right_keys.slice(..),
///     right_values.slice(..),
///     Less,
/// )
/// .unwrap();
///
/// assert_eq!(exec.to_host(&key_output).unwrap(), vec![1, 2, 3, 4]);
/// assert_eq!(exec.to_host(&value_output).unwrap(), vec![10, 20, 30, 40]);
/// ```
pub fn merge_by_key<R, LeftKeys, LeftValues, RightKeys, RightValues, Less>(
    exec: &Executor<R>,
    left_keys: LeftKeys,
    left_values: LeftValues,
    right_keys: RightKeys,
    right_values: RightValues,
    less: Less,
) -> Result<(MVec<R, LeftKeys::Item>, MVec<R, LeftValues::Item>), Error>
where
    R: Runtime,
    LeftKeys: MIter<R>,
    LeftKeys::Item: MCanonical<R>,
    LeftValues: MIter<R>,
    LeftValues::Item: MCanonical<R>,
    RightKeys: MIter<R, Item = LeftKeys::Item>,
    RightValues: MIter<R, Item = LeftValues::Item>,
    Less: BinaryPredicateOp<LeftKeys::Item>,
{
    let left_len = left_keys.len()? as usize;
    let right_len = right_keys.len()? as usize;
    let len = left_len
        .checked_add(right_len)
        .ok_or(Error::LengthTooLarge { len: usize::MAX })?;
    let key_output = exec.alloc_mvec::<LeftKeys::Item>(len);
    let value_output = exec.alloc_mvec::<LeftValues::Item>(len);
    merge_by_key_into(
        exec,
        left_keys,
        left_values,
        right_keys,
        right_values,
        less,
        key_output.slice_mut(..),
        value_output.slice_mut(..),
    )?;
    Ok((key_output, value_output))
}

/// Stably merges key/value ranges into caller-provided storage.
#[doc(hidden)]
pub(crate) fn merge_by_key_into<
    R,
    LeftKeys,
    LeftValues,
    RightKeys,
    RightValues,
    Less,
    KeyOutput,
    ValueOutput,
>(
    exec: &Executor<R>,
    left_keys: LeftKeys,
    left_values: LeftValues,
    right_keys: RightKeys,
    right_values: RightValues,
    less: Less,
    key_output: KeyOutput,
    value_output: ValueOutput,
) -> Result<(), Error>
where
    R: Runtime,
    LeftKeys: MIter<R>,
    LeftValues: MIter<R>,
    RightKeys: MIter<R, Item = LeftKeys::Item>,
    RightValues: MIter<R, Item = LeftValues::Item>,
    Less: BinaryPredicateOp<LeftKeys::Item>,
    KeyOutput: MIterMut<R>,
    KeyOutput::Item: WriteFrom<LeftKeys::Item>,
    ValueOutput: MIterMut<R>,
    ValueOutput::Item: WriteFrom<LeftValues::Item>,
{
    left_keys.merge_by_key_with(
        exec,
        left_values,
        right_keys,
        right_values,
        less,
        key_output,
        value_output,
    )
}
