use cubecl::prelude::Runtime;

use crate::{Error, Executor, MIter, op::PredicateOp};

macro_rules! predicate_api {
    ($name:ident, $output:ty, $map:expr, $doc:literal) => {
        #[doc = $doc]
        pub fn $name<R, Input, Pred>(
            exec: &Executor<R>,
            input: Input,
            pred: Pred,
        ) -> Result<$output, Error>
        where
            R: Runtime,
            Input: MIter<R>,
            Pred: PredicateOp<Input::Item>,
        {
            crate::predicate::$name(exec, crate::api::iter::lower_fixed::<R, _>(input), pred)
                .map($map)
        }
    };
}

predicate_api!(
    count_if,
    usize,
    |count: u32| count as usize,
    r#"Counts items satisfying a predicate.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, op, vector::count_if};

struct Even;

#[cubecl::cube]
impl op::PredicateOp<u32> for Even {
    fn apply(value: u32) -> bool {
        value % 2 == 0
    }
}

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let input = exec.to_device(&[1_u32, 2, 3, 4]);

assert_eq!(count_if(&exec, input.slice(..), Even).unwrap(), 2);
```
"#
);
predicate_api!(
    all_of,
    bool,
    |result: bool| result,
    r#"Returns whether every item satisfies a predicate.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, op, vector::all_of};

struct Even;

#[cubecl::cube]
impl op::PredicateOp<u32> for Even {
    fn apply(value: u32) -> bool {
        value % 2 == 0
    }
}

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let input = exec.to_device(&[2_u32, 4, 6]);

assert!(all_of(&exec, input.slice(..), Even).unwrap());
```
"#
);
predicate_api!(
    any_of,
    bool,
    |result: bool| result,
    r#"Returns whether any item satisfies a predicate.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, op, vector::any_of};

struct Even;

#[cubecl::cube]
impl op::PredicateOp<u32> for Even {
    fn apply(value: u32) -> bool {
        value % 2 == 0
    }
}

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let input = exec.to_device(&[1_u32, 3, 4]);

assert!(any_of(&exec, input.slice(..), Even).unwrap());
```
"#
);
predicate_api!(
    none_of,
    bool,
    |result: bool| result,
    r#"Returns whether no item satisfies a predicate.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, op, vector::none_of};

struct Even;

#[cubecl::cube]
impl op::PredicateOp<u32> for Even {
    fn apply(value: u32) -> bool {
        value % 2 == 0
    }
}

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let input = exec.to_device(&[1_u32, 3, 5]);

assert!(none_of(&exec, input.slice(..), Even).unwrap());
```
"#
);
predicate_api!(
    find_if,
    Option<usize>,
    |index: Option<u32>| index.map(|index| index as usize),
    r#"Returns the first index satisfying a predicate.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, op, vector::find_if};

struct Even;

#[cubecl::cube]
impl op::PredicateOp<u32> for Even {
    fn apply(value: u32) -> bool {
        value % 2 == 0
    }
}

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let input = exec.to_device(&[1_u32, 3, 4, 6]);

assert_eq!(find_if(&exec, input.slice(..), Even).unwrap(), Some(2));
```
"#
);
predicate_api!(
    is_partitioned,
    bool,
    |result: bool| result,
    r#"Returns whether passing items precede failing items.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, op, vector::is_partitioned};

struct Even;

#[cubecl::cube]
impl op::PredicateOp<u32> for Even {
    fn apply(value: u32) -> bool {
        value % 2 == 0
    }
}

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let input = exec.to_device(&[2_u32, 4, 1, 3]);

assert!(is_partitioned(&exec, input.slice(..), Even).unwrap());
```
"#
);
