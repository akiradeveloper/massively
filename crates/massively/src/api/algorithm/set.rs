use cubecl::prelude::Runtime;

use crate::{BinaryPredicateOp, Error, Executor, MIndex, MIter, MIterMut, WriteFrom};

macro_rules! set_api {
    ($name:ident, $mode:literal, $doc:literal) => {
        #[doc = $doc]
        pub fn $name<R, Left, Right, Less, Output>(
            exec: &Executor<R>,
            left: Left,
            right: Right,
            less: Less,
            output: Output,
        ) -> Result<MIndex, Error>
        where
            R: Runtime,
            Left: MIter<R>,
            Right: MIter<R, Item = Left::Item>,
            Less: BinaryPredicateOp<Left::Item>,
            Output: MIterMut<R>,
            Output::Item: WriteFrom<Left::Item>,
        {
            left.set_with(exec, right, less, output, $mode)
        }
    };
}

set_api!(
    set_union,
    0,
    r#"Computes the multiset union of two sorted ranges.

The return value is the number of items written at the beginning of `output`.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{BinaryPredicateOp, Executor, set_union};

struct Less;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for Less {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs < rhs
    }
}

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let left = exec.to_device(&[1_u32, 2, 2, 4]);
let right = exec.to_device(&[2_u32, 3, 4]);
let output = exec.alloc::<u32>(left.len() + right.len());

let len = set_union(
    &exec,
    left.slice(..),
    right.slice(..),
    Less,
    output.slice_mut(..),
)
.unwrap();

assert_eq!(len, 5);
assert_eq!(
    exec.to_host(&output.slice(..len as usize)).unwrap(),
    vec![1, 2, 2, 3, 4],
);
```
"#
);
set_api!(
    set_intersection,
    1,
    r#"Computes the multiset intersection of two sorted ranges.

The return value is the number of items written at the beginning of `output`.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{BinaryPredicateOp, Executor, set_intersection};

struct Less;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for Less {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs < rhs
    }
}

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let left = exec.to_device(&[1_u32, 2, 2, 4]);
let right = exec.to_device(&[2_u32, 3, 4]);
let output = exec.alloc::<u32>(left.len());

let len = set_intersection(
    &exec,
    left.slice(..),
    right.slice(..),
    Less,
    output.slice_mut(..),
)
.unwrap();

assert_eq!(len, 2);
assert_eq!(exec.to_host(&output.slice(..len as usize)).unwrap(), vec![2, 4]);
```
"#
);
set_api!(
    set_difference,
    2,
    r#"Computes the multiset difference of two sorted ranges.

The return value is the number of items written at the beginning of `output`.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{BinaryPredicateOp, Executor, set_difference};

struct Less;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for Less {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs < rhs
    }
}

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let left = exec.to_device(&[1_u32, 2, 2, 4]);
let right = exec.to_device(&[2_u32, 3, 4]);
let output = exec.alloc::<u32>(left.len());

let len = set_difference(
    &exec,
    left.slice(..),
    right.slice(..),
    Less,
    output.slice_mut(..),
)
.unwrap();

assert_eq!(len, 2);
assert_eq!(exec.to_host(&output.slice(..len as usize)).unwrap(), vec![1, 2]);
```
"#
);
