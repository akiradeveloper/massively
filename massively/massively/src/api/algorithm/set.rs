#![allow(private_bounds)]

use cubecl::prelude::Runtime;

use crate::{Error, Executor, MItem, MIter, MIterMut, MStorage, MVec, op::BinaryPredicateOp};

macro_rules! set_api {
    ($name:ident, $into_name:ident, $mode:literal, $capacity:expr, $doc:literal) => {
        #[doc = $doc]
        pub fn $name<R, Left, Right, Item, Less>(
            exec: &Executor<R>,
            left: Left,
            right: Right,
            less: Less,
        ) -> Result<MVec<R, Item>, Error>
        where
            R: Runtime,
            Left: MIter<R, Item = Item>,
            Right: MIter<R, Item = Item>,
            Item: MItem<R>,
            Less: BinaryPredicateOp<Item>,
        {
            let left_len = left.len()? as usize;
            let right_len = right.len()? as usize;
            let capacity = ($capacity)(left_len, right_len)?;
            let mut output = exec.alloc::<Item>(capacity);
            let len = $into_name(exec, left, right, less, output.slice_mut(..))?;
            output.truncate(len as usize);
            Ok(output)
        }

        #[doc = concat!("Caller-provided output variant of [`", stringify!($name), "`].")]
        #[doc(hidden)]
        pub(crate) fn $into_name<R, Left, Right, Less, Output>(
            exec: &Executor<R>,
            left: Left,
            right: Right,
            less: Less,
            output: Output,
        ) -> Result<u32, Error>
        where
            R: Runtime,
            Left: MIter<R>,
            Right: MIter<R, Item = Left::Item>,
            Left::Item: crate::api::iter::MItem<R>,
            Less: BinaryPredicateOp<Left::Item>,
            Output: MIterMut<R, Item = Left::Item>,
        {
            let left = crate::allocation::normalize_lowered(
                exec,
                crate::api::iter::lower_fixed::<R, _>(left),
            )?;
            let right = crate::allocation::normalize_lowered(
                exec,
                crate::api::iter::lower_fixed::<R, _>(right),
            )?;
            crate::core::set::set_storage(exec, &left, &right, less, output.lower_output(), $mode)
        }
    };
}

set_api!(
    set_union,
    set_union_into,
    0,
    |left: usize, right: usize| left
        .checked_add(right)
        .ok_or(Error::LengthTooLarge { len: usize::MAX }),
    r#"Computes the multiset union of two sorted ranges.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{op, Executor, vector::set_union};

struct Less;

#[cubecl::cube]
impl op::BinaryPredicateOp<u32> for Less {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs < rhs
    }
}

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let left = exec.to_device(&[1_u32, 2, 2, 4]);
let right = exec.to_device(&[2_u32, 3, 4]);
let output = set_union(&exec, left.slice(..), right.slice(..), Less).unwrap();

assert_eq!(
    exec.to_host(&output).unwrap(),
    vec![1, 2, 2, 3, 4],
);
```
"#
);
set_api!(
    set_intersection,
    set_intersection_into,
    1,
    |left: usize, right: usize| Ok(left.min(right)),
    r#"Computes the multiset intersection of two sorted ranges.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{op, Executor, vector::set_intersection};

struct Less;

#[cubecl::cube]
impl op::BinaryPredicateOp<u32> for Less {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs < rhs
    }
}

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let left = exec.to_device(&[1_u32, 2, 2, 4]);
let right = exec.to_device(&[2_u32, 3, 4]);
let output = set_intersection(&exec, left.slice(..), right.slice(..), Less).unwrap();

assert_eq!(exec.to_host(&output).unwrap(), vec![2, 4]);
```
"#
);
set_api!(
    set_difference,
    set_difference_into,
    2,
    |left: usize, _right: usize| Ok(left),
    r#"Computes the multiset difference of two sorted ranges.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{op, Executor, vector::set_difference};

struct Less;

#[cubecl::cube]
impl op::BinaryPredicateOp<u32> for Less {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs < rhs
    }
}

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let left = exec.to_device(&[1_u32, 2, 2, 4]);
let right = exec.to_device(&[2_u32, 3, 4]);
let output = set_difference(&exec, left.slice(..), right.slice(..), Less).unwrap();

assert_eq!(exec.to_host(&output).unwrap(), vec![1, 2]);
```
"#
);
