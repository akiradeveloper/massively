use cubecl::prelude::Runtime;

use crate::{Error, Executor, MIndex, MIter, MIterMut, WriteFrom, op::BinaryPredicateOp};

/// Finds the first source item equal to any needle.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op::BinaryPredicateOp, Executor, vector::find_first_of};
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
/// let source = exec.to_device(&[1_u32, 2, 3, 4]);
/// let needles = exec.to_device(&[7_u32, 3]);
///
/// let index = find_first_of(&exec, source.slice(..), needles.slice(..), Equal).unwrap();
///
/// assert_eq!(index, Some(2));
/// ```
pub fn find_first_of<R, Source, Needles, Equal>(
    exec: &Executor<R>,
    source: Source,
    needles: Needles,
    equal: Equal,
) -> Result<Option<MIndex>, Error>
where
    R: Runtime,
    Source: MIter<R>,
    Needles: MIter<R, Item = Source::Item>,
    Equal: BinaryPredicateOp<Source::Item>,
{
    source.find_first_of_with(exec, needles, equal)
}

/// Finds the lower bound of each value.
///
/// `source` must be sorted according to `less`.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op::BinaryPredicateOp, Executor, vector::lower_bound};
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
/// let source = exec.to_device(&[1_u32, 3, 5, 7]);
/// let values = exec.to_device(&[0_u32, 3, 6, 8]);
/// let output = exec.alloc::<u32>(values.len());
///
/// lower_bound(
///     &exec,
///     source.slice(..),
///     values.slice(..),
///     Less,
///     output.slice_mut(..),
/// )
/// .unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![0, 1, 3, 4]);
/// ```
pub fn lower_bound<R, Source, Values, Output, Less>(
    exec: &Executor<R>,
    source: Source,
    values: Values,
    less: Less,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Source: MIter<R>,
    Values: MIter<R, Item = Source::Item>,
    Output: MIterMut<R>,
    Output::Item: WriteFrom<MIndex>,
    Less: BinaryPredicateOp<Source::Item>,
{
    source.bounds_with(exec, values, less, false, output)
}

/// Finds the upper bound of each value.
///
/// `source` must be sorted according to `less`.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op::BinaryPredicateOp, Executor, vector::upper_bound};
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
/// let source = exec.to_device(&[1_u32, 3, 5, 7]);
/// let values = exec.to_device(&[0_u32, 3, 6, 8]);
/// let output = exec.alloc::<u32>(values.len());
///
/// upper_bound(
///     &exec,
///     source.slice(..),
///     values.slice(..),
///     Less,
///     output.slice_mut(..),
/// )
/// .unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![0, 2, 3, 4]);
/// ```
pub fn upper_bound<R, Source, Values, Output, Less>(
    exec: &Executor<R>,
    source: Source,
    values: Values,
    less: Less,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Source: MIter<R>,
    Values: MIter<R, Item = Source::Item>,
    Output: MIterMut<R>,
    Output::Item: WriteFrom<MIndex>,
    Less: BinaryPredicateOp<Source::Item>,
{
    source.bounds_with(exec, values, less, true, output)
}

/// Returns whether two ranges contain equal items.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op::BinaryPredicateOp, Executor, vector::equal};
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
/// let left = exec.to_device(&[1_u32, 2, 3]);
/// let right = exec.to_device(&[1_u32, 2, 3]);
///
/// assert!(equal(&exec, left.slice(..), right.slice(..), Equal).unwrap());
/// ```
pub fn equal<R, Left, Right, Equal>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    equal: Equal,
) -> Result<bool, Error>
where
    R: Runtime,
    Left: MIter<R>,
    Right: MIter<R, Item = Left::Item>,
    Equal: BinaryPredicateOp<Left::Item>,
{
    left.equal_with(exec, right, equal)
}

/// Returns the first mismatch.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op::BinaryPredicateOp, Executor, vector::mismatch};
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
/// let left = exec.to_device(&[1_u32, 2, 3]);
/// let right = exec.to_device(&[1_u32, 4, 3]);
///
/// assert_eq!(mismatch(&exec, left.slice(..), right.slice(..), Equal).unwrap(), Some(1));
/// ```
pub fn mismatch<R, Left, Right, Equal>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    equal: Equal,
) -> Result<Option<MIndex>, Error>
where
    R: Runtime,
    Left: MIter<R>,
    Right: MIter<R, Item = Left::Item>,
    Equal: BinaryPredicateOp<Left::Item>,
{
    left.mismatch_with(exec, right, equal)
}

/// Lexicographically compares two ranges.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op::BinaryPredicateOp, Executor, vector::lexicographical_compare};
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
/// let left = exec.to_device(&[1_u32, 2, 3]);
/// let right = exec.to_device(&[1_u32, 3, 0]);
///
/// assert!(
///     lexicographical_compare(&exec, left.slice(..), right.slice(..), Less).unwrap()
/// );
/// ```
pub fn lexicographical_compare<R, Left, Right, Less>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    less: Less,
) -> Result<bool, Error>
where
    R: Runtime,
    Left: MIter<R>,
    Right: MIter<R, Item = Left::Item>,
    Less: BinaryPredicateOp<Left::Item>,
{
    left.lexicographical_with(exec, right, less)
}
