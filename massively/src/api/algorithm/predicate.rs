use cubecl::prelude::Runtime;

use crate::{Error, Executor, MBool, MIndex, MIter, MVal, op::PredicateOp};

macro_rules! predicate_api {
    ($name:ident, $output:ty, $doc:literal) => {
        #[doc = $doc]
        pub fn $name<R, Input, Pred>(
            exec: &Executor<R>,
            input: Input,
            pred: Pred,
        ) -> Result<MVal<R, $output>, Error>
        where
            R: Runtime,
            Input: MIter<R>,
            Pred: PredicateOp<Input::Item>,
        {
            crate::predicate::$name(exec, crate::api::iter::lower_fixed::<R, _>(input), pred)
        }
    };
}

predicate_api!(
    count_if,
    MIndex,
    r#"Counts items satisfying a predicate.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, op, vector::count_if};

struct Even;

#[cubecl::cube]
impl op::PredicateOp<u32> for Even {
    fn apply(value: u32) -> massively::MBool {
        op::mbool(value % 2 == 0)
    }
}

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let input = exec.to_device(&[1_u32, 2, 3, 4]);

assert_eq!(count_if(&exec, input.slice(..), Even).unwrap().read(&exec).unwrap(), 2);
```
"#
);
predicate_api!(
    all_of,
    MBool,
    r#"Returns whether every item satisfies a predicate.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, op, vector::all_of};

struct Even;

#[cubecl::cube]
impl op::PredicateOp<u32> for Even {
    fn apply(value: u32) -> massively::MBool {
        op::mbool(value % 2 == 0)
    }
}

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let input = exec.to_device(&[2_u32, 4, 6]);

assert_eq!(all_of(&exec, input.slice(..), Even).unwrap().read(&exec).unwrap(), 1);
```
"#
);
predicate_api!(
    any_of,
    MBool,
    r#"Returns whether any item satisfies a predicate.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, op, vector::any_of};

struct Even;

#[cubecl::cube]
impl op::PredicateOp<u32> for Even {
    fn apply(value: u32) -> massively::MBool {
        op::mbool(value % 2 == 0)
    }
}

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let input = exec.to_device(&[1_u32, 3, 4]);

assert_eq!(any_of(&exec, input.slice(..), Even).unwrap().read(&exec).unwrap(), 1);
```
"#
);
predicate_api!(
    none_of,
    MBool,
    r#"Returns whether no item satisfies a predicate.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, op, vector::none_of};

struct Even;

#[cubecl::cube]
impl op::PredicateOp<u32> for Even {
    fn apply(value: u32) -> massively::MBool {
        op::mbool(value % 2 == 0)
    }
}

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let input = exec.to_device(&[1_u32, 3, 5]);

assert_eq!(none_of(&exec, input.slice(..), Even).unwrap().read(&exec).unwrap(), 1);
```
"#
);
predicate_api!(
    find_if,
    (MBool, MIndex),
    r#"Returns the first index satisfying a predicate.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, op, vector::find_if};

struct Even;

#[cubecl::cube]
impl op::PredicateOp<u32> for Even {
    fn apply(value: u32) -> massively::MBool {
        op::mbool(value % 2 == 0)
    }
}

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let input = exec.to_device(&[1_u32, 3, 4, 6]);

assert_eq!(find_if(&exec, input.slice(..), Even).unwrap().read(&exec).unwrap(), (1, 2));
```
"#
);
predicate_api!(
    is_partitioned,
    MBool,
    r#"Returns whether passing items precede failing items.

# Examples

```
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, op, vector::is_partitioned};

struct Even;

#[cubecl::cube]
impl op::PredicateOp<u32> for Even {
    fn apply(value: u32) -> massively::MBool {
        op::mbool(value % 2 == 0)
    }
}

let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
let input = exec.to_device(&[2_u32, 4, 1, 3]);

assert_eq!(is_partitioned(&exec, input.slice(..), Even).unwrap().read(&exec).unwrap(), 1);
```
"#
);
