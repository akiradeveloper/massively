#![allow(private_bounds)]

use cubecl::prelude::{CubeType, Runtime};

use crate::{Error, Executor, MAllocItem, MIter, MIterMut, MStorage, MVec, op::BinaryPredicateOp};

struct UniqueOperation<'a, R: Runtime, Input, Equal> {
    exec: &'a Executor<R>,
    input: Input,
    equal: Equal,
}

impl<R, Item, Input, Equal> crate::api::iter::OutputOperation<R, Item>
    for UniqueOperation<'_, R, Input, Equal>
where
    R: Runtime,
    Item: CubeType + Send + Sync + 'static,
    Input: MIter<R, Item = Item>,
    Equal: BinaryPredicateOp<Item>,
{
    type Result = Result<u32, Error>;

    fn run<Output>(self, output: Output) -> Self::Result
    where
        Item: crate::api::iter::KernelRow + crate::allocation::ScratchStorage<R>,
        Output: crate::api::iter::ConcreteOutput<R, Item>,
    {
        crate::ordering::unique(
            self.exec,
            crate::api::iter::lower_fixed::<R, _>(self.input),
            self.equal,
            output,
        )
    }
}

/// Stably sorts an input and returns owned device storage.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op, Executor, vector::sort};
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
    Item: MAllocItem<R>,
    Less: BinaryPredicateOp<Item>,
{
    let len = input.len()? as usize;
    let output = exec.alloc::<Item>(len);
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
    Input: MIter<R, Item = Output::Item>,
    Output: MIterMut<R>,
    Output::Item: MAllocItem<R>,
    Less: BinaryPredicateOp<Input::Item>,
{
    <<Output::Item as MAllocItem<R>>::Dispatch as crate::api::iter::ItemDispatch<R>>::sort_into(
        exec, input, less, output,
    )
}

/// Finds the first accepted adjacent pair.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op, Executor, vector::adjacent_find};
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
/// let input = exec.to_device(&[1_u32, 2, 2, 3]);
///
/// assert_eq!(adjacent_find(&exec, input.slice(..), Equal).unwrap(), Some(1));
/// ```
pub fn adjacent_find<R, Input, Equal>(
    exec: &Executor<R>,
    input: Input,
    equal: Equal,
) -> Result<Option<usize>, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Equal: BinaryPredicateOp<Input::Item>,
{
    crate::ordering::adjacent_find(exec, crate::api::iter::lower_fixed::<R, _>(input), equal)
        .map(|index| index.map(|index| index as usize))
}

/// Removes consecutive duplicates.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op, Executor, vector::unique};
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
    Item: MAllocItem<R>,
    Equal: BinaryPredicateOp<Item>,
{
    let capacity = input.len()? as usize;
    let mut output = exec.alloc::<Item>(capacity);
    let len = unique_into(exec, input, equal, output.slice_mut(..))?;
    output.truncate(len as usize);
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
) -> Result<u32, Error>
where
    R: Runtime,
    Input: MIter<R, Item = Output::Item>,
    Output: MIterMut<R>,
    Equal: BinaryPredicateOp<Input::Item>,
{
    output.run_output_operation(UniqueOperation { exec, input, equal })
}

/// Returns the first index at which the input ceases to be sorted.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op, Executor, vector::is_sorted_until};
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
/// let input = exec.to_device(&[1_u32, 2, 4, 3, 5]);
///
/// assert_eq!(is_sorted_until(&exec, input.slice(..), Less).unwrap(), 3);
/// ```
pub fn is_sorted_until<R, Input, Less>(
    exec: &Executor<R>,
    input: Input,
    less: Less,
) -> Result<usize, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Less: BinaryPredicateOp<Input::Item>,
{
    crate::ordering::is_sorted_until(exec, crate::api::iter::lower_fixed::<R, _>(input), less)
        .map(|index| index as usize)
}

/// Returns whether the input is sorted.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op, Executor, vector::is_sorted};
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
    ($name:ident, $output:ty, $map:expr, $doc:literal) => {
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
                .map($map)
        }
    };
}

extremum_api!(
    min_element,
    Option<usize>,
    |index: Option<u32>| index.map(|index| index as usize),
    r#"Returns the first minimum element index.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{op, Executor, vector::min_element};

struct Less;

#[cubecl::cube]
impl op::BinaryPredicateOp<u32> for Less {
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
    Option<usize>,
    |index: Option<u32>| index.map(|index| index as usize),
    r#"Returns the first maximum element index.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{op, Executor, vector::max_element};

struct Less;

#[cubecl::cube]
impl op::BinaryPredicateOp<u32> for Less {
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
    Option<(usize, usize)>,
    |indices: Option<(u32, u32)>| indices.map(|(min, max)| (min as usize, max as usize)),
    r#"Returns the minimum and maximum element indices.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{op, Executor, vector::minmax_element};

struct Less;

#[cubecl::cube]
impl op::BinaryPredicateOp<u32> for Less {
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
