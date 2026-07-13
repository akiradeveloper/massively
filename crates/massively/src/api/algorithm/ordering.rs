use cubecl::prelude::Runtime;

use crate::{
    Error, Executor, MIndex, MIter, MIterMut, MStorage, MVec, Materializable, WritableFrom,
    op::BinaryPredicateOp,
};

/// Stably sorts an input and returns owned device storage.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op::BinaryPredicateOp, Executor, vector::sort};
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
/// let input = exec.to_device(&[3_u32, 1, 2]);
/// let output = sort(&exec, input.slice(..), Less).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![1, 2, 3]);
/// ```
pub fn sort<R, Input, Item, Less>(
    exec: &Executor<R>,
    input: Input,
    less: Less,
) -> Result<MVec<R, Item>, Error>
where
    R: Runtime,
    Input: MIter<R, Item = Item>,
    Item: Materializable<R>,
    Less: BinaryPredicateOp<Item>,
{
    let len = input.len()? as usize;
    let output = exec.alloc_mvec::<Item>(len);
    sort_into(exec, input, less, output.slice_mut(..))?;
    Ok(output)
}

/// Stably sorts an input into caller-provided storage.
#[doc(hidden)]
pub(crate) fn sort_into<R, Input, Output, Less>(
    exec: &Executor<R>,
    input: Input,
    less: Less,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: MIter<R>,
    Output: MIterMut<R>,
    Output::Item: WritableFrom<Input::Item>,
    Less: BinaryPredicateOp<Input::Item>,
{
    let input = crate::api::iter::lower_fixed::<R, _>(input);
    let permutation = crate::ordering::sort_control_with(exec, input.clone(), less)?;
    crate::indexed::apply_permutation(exec, input, permutation.column(), output.lower_output())
}

/// Finds the first accepted adjacent pair.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op::BinaryPredicateOp, Executor, vector::adjacent_find};
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
/// let input = exec.to_device(&[1_u32, 2, 2, 3]);
///
/// assert_eq!(adjacent_find(&exec, input.slice(..), Equal).unwrap(), Some(1));
/// ```
pub fn adjacent_find<R, Input, Equal>(
    exec: &Executor<R>,
    input: Input,
    equal: Equal,
) -> Result<Option<MIndex>, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Equal: BinaryPredicateOp<Input::Item>,
{
    crate::ordering::adjacent_find(exec, crate::api::iter::lower_fixed::<R, _>(input), equal)
}

/// Removes consecutive duplicates.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op::BinaryPredicateOp, Executor, vector::unique};
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
/// let input = exec.to_device(&[1_u32, 1, 2, 2, 3]);
/// let output = unique(&exec, input.slice(..), Equal).unwrap();
///
/// assert_eq!(output.len(), 3);
/// assert_eq!(exec.to_host(&output).unwrap(), vec![1, 2, 3]);
/// ```
pub fn unique<R, Input, Item, Equal>(
    exec: &Executor<R>,
    input: Input,
    equal: Equal,
) -> Result<MVec<R, Item>, Error>
where
    R: Runtime,
    Input: MIter<R, Item = Item>,
    Item: Materializable<R>,
    Equal: BinaryPredicateOp<Item>,
{
    let capacity = input.len()? as usize;
    let mut output = exec.alloc_mvec::<Item>(capacity);
    let len = unique_into(exec, input, equal, output.slice_mut(..))?;
    output.truncate(len);
    Ok(output)
}

/// Removes consecutive duplicates into caller-provided storage.
///
/// Returns the number of items written at the beginning of `output`.
#[doc(hidden)]
pub(crate) fn unique_into<R, Input, Output, Equal>(
    exec: &Executor<R>,
    input: Input,
    equal: Equal,
    output: Output,
) -> Result<MIndex, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Output: MIterMut<R>,
    Output::Item: WritableFrom<Input::Item>,
    Equal: BinaryPredicateOp<Input::Item>,
{
    crate::ordering::unique(
        exec,
        crate::api::iter::lower_fixed::<R, _>(input),
        equal,
        output.lower_output(),
    )
}

/// Returns the first index at which the input ceases to be sorted.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op::BinaryPredicateOp, Executor, vector::is_sorted_until};
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
/// let input = exec.to_device(&[1_u32, 2, 4, 3, 5]);
///
/// assert_eq!(is_sorted_until(&exec, input.slice(..), Less).unwrap(), 3);
/// ```
pub fn is_sorted_until<R, Input, Less>(
    exec: &Executor<R>,
    input: Input,
    less: Less,
) -> Result<MIndex, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Less: BinaryPredicateOp<Input::Item>,
{
    crate::ordering::is_sorted_until(exec, crate::api::iter::lower_fixed::<R, _>(input), less)
}

/// Returns whether the input is sorted.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op::BinaryPredicateOp, Executor, vector::is_sorted};
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
/// let input = exec.to_device(&[1_u32, 2, 3, 4]);
///
/// assert!(is_sorted(&exec, input.slice(..), Less).unwrap());
/// ```
pub fn is_sorted<R, Input, Less>(
    exec: &Executor<R>,
    input: Input,
    less: Less,
) -> Result<bool, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Less: BinaryPredicateOp<Input::Item>,
{
    crate::ordering::is_sorted(exec, crate::api::iter::lower_fixed::<R, _>(input), less)
}

macro_rules! extremum_api {
    ($name:ident, $output:ty, $doc:literal) => {
        #[doc = $doc]
        pub fn $name<R, Input, Less>(
            exec: &Executor<R>,
            input: Input,
            less: Less,
        ) -> Result<$output, Error>
        where
            R: Runtime,
            Input: MIter<R>,
            Less: BinaryPredicateOp<Input::Item>,
        {
            crate::ordering::$name(exec, crate::api::iter::lower_fixed::<R, _>(input), less)
        }
    };
}

extremum_api!(
    min_element,
    Option<MIndex>,
    r#"Returns the first minimum element index.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{op::BinaryPredicateOp, Executor, vector::min_element};

struct Less;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for Less {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs < rhs
    }
}

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let input = exec.to_device(&[3_u32, 1, 2, 1]);

assert_eq!(min_element(&exec, input.slice(..), Less).unwrap(), Some(1));
```
"#
);
extremum_api!(
    max_element,
    Option<MIndex>,
    r#"Returns the first maximum element index.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{op::BinaryPredicateOp, Executor, vector::max_element};

struct Less;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for Less {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs < rhs
    }
}

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let input = exec.to_device(&[3_u32, 1, 3, 2]);

assert_eq!(max_element(&exec, input.slice(..), Less).unwrap(), Some(0));
```
"#
);
extremum_api!(
    minmax_element,
    Option<(MIndex, MIndex)>,
    r#"Returns the minimum and maximum element indices.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{op::BinaryPredicateOp, Executor, vector::minmax_element};

struct Less;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for Less {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs < rhs
    }
}

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let input = exec.to_device(&[3_u32, 1, 4, 2]);

assert_eq!(minmax_element(&exec, input.slice(..), Less).unwrap(), Some((1, 2)));
```
"#
);
