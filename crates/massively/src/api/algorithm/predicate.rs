use cubecl::prelude::Runtime;

use crate::{Error, Executor, MIndex, MIter, op::PredicateOp};

macro_rules! predicate_api {
    ($name:ident, $method:ident, $output:ty, $doc:literal) => {
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
            crate::predicate::$name(exec, crate::api::iter::lower::<R, _>(input), pred)
        }
    };
}

predicate_api!(
    count_if,
    count_if_with,
    MIndex,
    r#"Counts items satisfying a predicate.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, op::PredicateOp, vector::count_if};

struct Even;

#[cubecl::cube]
impl PredicateOp<u32> for Even {
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
    all_of_with,
    bool,
    r#"Returns whether every item satisfies a predicate.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, op::PredicateOp, vector::all_of};

struct Even;

#[cubecl::cube]
impl PredicateOp<u32> for Even {
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
    any_of_with,
    bool,
    r#"Returns whether any item satisfies a predicate.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, op::PredicateOp, vector::any_of};

struct Even;

#[cubecl::cube]
impl PredicateOp<u32> for Even {
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
    none_of_with,
    bool,
    r#"Returns whether no item satisfies a predicate.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, op::PredicateOp, vector::none_of};

struct Even;

#[cubecl::cube]
impl PredicateOp<u32> for Even {
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
    find_if_with,
    Option<MIndex>,
    r#"Returns the first index satisfying a predicate.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, op::PredicateOp, vector::find_if};

struct Even;

#[cubecl::cube]
impl PredicateOp<u32> for Even {
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
    is_partitioned_with,
    bool,
    r#"Returns whether passing items precede failing items.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, op::PredicateOp, vector::is_partitioned};

struct Even;

#[cubecl::cube]
impl PredicateOp<u32> for Even {
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
