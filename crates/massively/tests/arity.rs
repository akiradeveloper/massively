#![allow(unused_macros)]

use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::op::{BinaryPredicateOp, PredicateOp, ReductionOp, UnaryOp};
use massively::{DeviceVec, Executor};

type F32Vec = DeviceVec<WgpuRuntime, f32>;
type U32Vec = DeviceVec<WgpuRuntime, u32>;

fn exec() -> Executor<WgpuRuntime> {
    Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice)
}

fn column_f32(exec: &Executor<WgpuRuntime>) -> F32Vec {
    exec.to_device(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap()
}

fn column_u32(exec: &Executor<WgpuRuntime>) -> U32Vec {
    exec.to_device(&[1_u32, 2, 3, 4]).unwrap()
}

fn zeros_f32(exec: &Executor<WgpuRuntime>) -> F32Vec {
    exec.to_device(&[0.0_f32; 4]).unwrap()
}

fn zeros_u32(exec: &Executor<WgpuRuntime>) -> U32Vec {
    exec.to_device(&[0_u32; 4]).unwrap()
}

fn indices(exec: &Executor<WgpuRuntime>) -> U32Vec {
    exec.to_device(&[3_u32, 2, 1, 0]).unwrap()
}

fn stencil(exec: &Executor<WgpuRuntime>) -> U32Vec {
    exec.to_device(&[1_u32, 0, 1, 0]).unwrap()
}

macro_rules! soa_fn {
    ($values:ident, $output:ident, $owned:ty, $ctor:expr, $out_ctor:expr) => {
        fn $values(exec: &Executor<WgpuRuntime>) -> $owned {
            $ctor(exec)
        }

        fn $output(exec: &Executor<WgpuRuntime>) -> $owned {
            $out_ctor(exec)
        }
    };
}

soa_fn!(
    values1,
    output1,
    massively::SoA1<U32Vec>,
    |exec: &Executor<WgpuRuntime>| massively::SoA1(column_u32(exec)),
    |exec: &Executor<WgpuRuntime>| massively::SoA1(zeros_u32(exec))
);
soa_fn!(
    values2,
    output2,
    massively::SoA2<F32Vec, U32Vec>,
    |exec: &Executor<WgpuRuntime>| massively::SoA2(column_f32(exec), column_u32(exec)),
    |exec: &Executor<WgpuRuntime>| massively::SoA2(zeros_f32(exec), zeros_u32(exec))
);
soa_fn!(
    values3,
    output3,
    massively::SoA3<F32Vec, U32Vec, F32Vec>,
    |exec: &Executor<WgpuRuntime>| {
        massively::SoA3(column_f32(exec), column_u32(exec), column_f32(exec))
    },
    |exec: &Executor<WgpuRuntime>| {
        massively::SoA3(zeros_f32(exec), zeros_u32(exec), zeros_f32(exec))
    }
);
soa_fn!(
    values4,
    output4,
    massively::SoA4<F32Vec, U32Vec, F32Vec, U32Vec>,
    |exec: &Executor<WgpuRuntime>| {
        massively::SoA4(
            column_f32(exec),
            column_u32(exec),
            column_f32(exec),
            column_u32(exec),
        )
    },
    |exec: &Executor<WgpuRuntime>| {
        massively::SoA4(
            zeros_f32(exec),
            zeros_u32(exec),
            zeros_f32(exec),
            zeros_u32(exec),
        )
    }
);
soa_fn!(
    values5,
    output5,
    massively::SoA5<F32Vec, U32Vec, F32Vec, U32Vec, F32Vec>,
    |exec: &Executor<WgpuRuntime>| {
        massively::SoA5(
            column_f32(exec),
            column_u32(exec),
            column_f32(exec),
            column_u32(exec),
            column_f32(exec),
        )
    },
    |exec: &Executor<WgpuRuntime>| {
        massively::SoA5(
            zeros_f32(exec),
            zeros_u32(exec),
            zeros_f32(exec),
            zeros_u32(exec),
            zeros_f32(exec),
        )
    }
);
soa_fn!(
    values6,
    output6,
    massively::SoA6<F32Vec, U32Vec, F32Vec, U32Vec, F32Vec, U32Vec>,
    |exec: &Executor<WgpuRuntime>| {
        massively::SoA6(
            column_f32(exec),
            column_u32(exec),
            column_f32(exec),
            column_u32(exec),
            column_f32(exec),
            column_u32(exec),
        )
    },
    |exec: &Executor<WgpuRuntime>| {
        massively::SoA6(
            zeros_f32(exec),
            zeros_u32(exec),
            zeros_f32(exec),
            zeros_u32(exec),
            zeros_f32(exec),
            zeros_u32(exec),
        )
    }
);
soa_fn!(
    values7,
    output7,
    massively::SoA7<F32Vec, U32Vec, F32Vec, U32Vec, F32Vec, U32Vec, F32Vec>,
    |exec: &Executor<WgpuRuntime>| {
        massively::SoA7(
            column_f32(exec),
            column_u32(exec),
            column_f32(exec),
            column_u32(exec),
            column_f32(exec),
            column_u32(exec),
            column_f32(exec),
        )
    },
    |exec: &Executor<WgpuRuntime>| {
        massively::SoA7(
            zeros_f32(exec),
            zeros_u32(exec),
            zeros_f32(exec),
            zeros_u32(exec),
            zeros_f32(exec),
            zeros_u32(exec),
            zeros_f32(exec),
        )
    }
);

struct ArityEq;
struct ArityLess;
struct ArityPred;
struct ArityReduce;
struct ArityScalarToTuple1;
struct ArityScalarToTuple2;
struct ArityScalarToTuple3;
struct ArityScalarToTuple4;
struct ArityScalarToTuple5;
struct ArityScalarToTuple6;
struct ArityScalarToTuple7;
struct ArityTupleToScalar;

macro_rules! impl_arity_ops {
    ($($ty:ty),+) => {
        #[cubecl::cube]
        impl BinaryPredicateOp<WgpuRuntime, ($($ty,)+)> for ArityEq {
            fn apply(lhs: ($($ty,)+), rhs: ($($ty,)+)) -> bool {
                lhs.0 == rhs.0
            }
        }

        #[cubecl::cube]
        impl BinaryPredicateOp<WgpuRuntime, ($($ty,)+)> for ArityLess {
            fn apply(lhs: ($($ty,)+), rhs: ($($ty,)+)) -> bool {
                lhs.0 < rhs.0
            }
        }

        #[cubecl::cube]
        impl PredicateOp<WgpuRuntime, ($($ty,)+)> for ArityPred {
            type Env = ();

            fn apply(_env: (), _input: ($($ty,)+)) -> bool {
                true
            }
        }

        #[cubecl::cube]
        impl ReductionOp<WgpuRuntime, ($($ty,)+)> for ArityReduce {
            fn apply(lhs: ($($ty,)+), _rhs: ($($ty,)+)) -> ($($ty,)+) {
                lhs
            }
        }

    };
}

macro_rules! impl_tuple_to_scalar {
    ($($ty:ty),+) => {
        #[cubecl::cube]
        impl UnaryOp<WgpuRuntime, ($($ty,)+)> for ArityTupleToScalar {
            type Env = ();
            type Output = (u32,);

            fn apply(_env: (), _input: ($($ty,)+)) -> (u32,) {
                (1_u32,)
            }
        }
    };
}

macro_rules! impl_scalar_to_tuple {
    ($op:ident, ($($ty:ty),+), ($($out:expr),+)) => {
        #[cubecl::cube]
        impl UnaryOp<WgpuRuntime, (u32,)> for $op {
            type Env = ();
            type Output = ($($ty,)+);

            fn apply(_env: (), input: (u32,)) -> ($($ty,)+) {
                let _ = input;
                ($($out,)+)
            }
        }
    };
}

impl_arity_ops!(u32);
impl_arity_ops!(f32, u32);
impl_arity_ops!(f32, u32, f32);
impl_arity_ops!(f32, u32, f32, u32);
impl_arity_ops!(f32, u32, f32, u32, f32);
impl_arity_ops!(f32, u32, f32, u32, f32, u32);
impl_arity_ops!(f32, u32, f32, u32, f32, u32, f32);

impl_tuple_to_scalar!(u32);
impl_tuple_to_scalar!(f32, u32);
impl_tuple_to_scalar!(f32, u32, f32);
impl_tuple_to_scalar!(f32, u32, f32, u32);
impl_tuple_to_scalar!(f32, u32, f32, u32, f32);
impl_tuple_to_scalar!(f32, u32, f32, u32, f32, u32);
impl_tuple_to_scalar!(f32, u32, f32, u32, f32, u32, f32);

impl_scalar_to_tuple!(ArityScalarToTuple1, (u32), (1_u32));
impl_scalar_to_tuple!(ArityScalarToTuple2, (f32, u32), (1.0_f32, 1_u32));
impl_scalar_to_tuple!(
    ArityScalarToTuple3,
    (f32, u32, f32),
    (1.0_f32, 1_u32, 1.0_f32)
);
impl_scalar_to_tuple!(
    ArityScalarToTuple4,
    (f32, u32, f32, u32),
    (1.0_f32, 1_u32, 1.0_f32, 1_u32)
);
impl_scalar_to_tuple!(
    ArityScalarToTuple5,
    (f32, u32, f32, u32, f32),
    (1.0_f32, 1_u32, 1.0_f32, 1_u32, 1.0_f32)
);
impl_scalar_to_tuple!(
    ArityScalarToTuple6,
    (f32, u32, f32, u32, f32, u32),
    (1.0_f32, 1_u32, 1.0_f32, 1_u32, 1.0_f32, 1_u32)
);
impl_scalar_to_tuple!(
    ArityScalarToTuple7,
    (f32, u32, f32, u32, f32, u32, f32),
    (1.0_f32, 1_u32, 1.0_f32, 1_u32, 1.0_f32, 1_u32, 1.0_f32)
);

macro_rules! define_value_arity_tests {
    ($module:ident, $case:ident) => {
        mod $module {
            use super::*;

            #[test]
            fn arity_1() {
                $case!(values1, output1, massively::SoA1<U32Vec>, (0_u32,));
            }

            #[test]
            fn arity_2() {
                $case!(
                    values2,
                    output2,
                    massively::SoA2<F32Vec, U32Vec>,
                    (0.0_f32, 0_u32)
                );
            }

            #[test]
            fn arity_3() {
                $case!(
                    values3,
                    output3,
                    massively::SoA3<F32Vec, U32Vec, F32Vec>,
                    (0.0_f32, 0_u32, 0.0_f32)
                );
            }

            #[test]
            fn arity_4() {
                $case!(
                    values4,
                    output4,
                    massively::SoA4<F32Vec, U32Vec, F32Vec, U32Vec>,
                    (0.0_f32, 0_u32, 0.0_f32, 0_u32)
                );
            }

            #[test]
            fn arity_5() {
                $case!(
                    values5,
                    output5,
                    massively::SoA5<F32Vec, U32Vec, F32Vec, U32Vec, F32Vec>,
                    (0.0_f32, 0_u32, 0.0_f32, 0_u32, 0.0_f32)
                );
            }

            #[test]
            fn arity_6() {
                $case!(
                    values6,
                    output6,
                    massively::SoA6<F32Vec, U32Vec, F32Vec, U32Vec, F32Vec, U32Vec>,
                    (0.0_f32, 0_u32, 0.0_f32, 0_u32, 0.0_f32, 0_u32)
                );
            }

            #[test]
            fn arity_7() {
                $case!(
                    values7,
                    output7,
                    massively::SoA7<F32Vec, U32Vec, F32Vec, U32Vec, F32Vec, U32Vec, F32Vec>,
                    (
                        0.0_f32, 0_u32, 0.0_f32, 0_u32, 0.0_f32, 0_u32, 0.0_f32
                    )
                );
            }
        }
    };
}

macro_rules! define_output_arity_tests {
    ($module:ident, $case:ident) => {
        mod $module {
            use super::*;

            #[test]
            fn arity_1() {
                $case!(
                    values1,
                    output1,
                    massively::SoA1<U32Vec>,
                    ArityScalarToTuple1
                );
            }

            #[test]
            fn arity_2() {
                $case!(
                    values2,
                    output2,
                    massively::SoA2<F32Vec, U32Vec>,
                    ArityScalarToTuple2
                );
            }

            #[test]
            fn arity_3() {
                $case!(
                    values3,
                    output3,
                    massively::SoA3<F32Vec, U32Vec, F32Vec>,
                    ArityScalarToTuple3
                );
            }

            #[test]
            fn arity_4() {
                $case!(
                    values4,
                    output4,
                    massively::SoA4<F32Vec, U32Vec, F32Vec, U32Vec>,
                    ArityScalarToTuple4
                );
            }

            #[test]
            fn arity_5() {
                $case!(
                    values5,
                    output5,
                    massively::SoA5<F32Vec, U32Vec, F32Vec, U32Vec, F32Vec>,
                    ArityScalarToTuple5
                );
            }

            #[test]
            fn arity_6() {
                $case!(
                    values6,
                    output6,
                    massively::SoA6<F32Vec, U32Vec, F32Vec, U32Vec, F32Vec, U32Vec>,
                    ArityScalarToTuple6
                );
            }

            #[test]
            fn arity_7() {
                $case!(
                    values7,
                    output7,
                    massively::SoA7<F32Vec, U32Vec, F32Vec, U32Vec, F32Vec, U32Vec, F32Vec>,
                    ArityScalarToTuple7
                );
            }
        }
    };
}

macro_rules! define_key_arity_tests {
    ($module:ident, $case:ident) => {
        mod $module {
            use super::*;

            #[test]
            fn arity_1() {
                $case!(values1, massively::SoA1<U32Vec>);
            }

            #[test]
            fn arity_2() {
                $case!(values2, massively::SoA2<F32Vec, U32Vec>);
            }

            #[test]
            fn arity_3() {
                $case!(values3, massively::SoA3<F32Vec, U32Vec, F32Vec>);
            }
        }
    };
}

macro_rules! adjacent_difference_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let values = $values(&exec);
        let out = $output(&exec);
        massively::adjacent_difference(&exec, values.slice(..), ArityReduce, out.slice_mut(..))
            .unwrap();
    }};
}
define_value_arity_tests!(adjacent_difference_arity, adjacent_difference_case);

macro_rules! adjacent_find_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let values = $values(&exec);
        massively::adjacent_find(&exec, values.slice(..), ArityEq).unwrap();
    }};
}
define_value_arity_tests!(adjacent_find_arity, adjacent_find_case);

macro_rules! all_of_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let values = $values(&exec);
        massively::all_of(&exec, values.slice(..), ArityPred, ()).unwrap();
    }};
}
define_value_arity_tests!(all_of_arity, all_of_case);

macro_rules! any_of_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let values = $values(&exec);
        massively::any_of(&exec, values.slice(..), ArityPred, ()).unwrap();
    }};
}
define_value_arity_tests!(any_of_arity, any_of_case);

macro_rules! copy_where_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let values = $values(&exec);
        let out = $output(&exec);
        let stencil = stencil(&exec);
        massively::copy_where(
            &exec,
            values.slice(..),
            stencil.slice(..),
            out.slice_mut(..),
        )
        .unwrap();
    }};
}
define_value_arity_tests!(copy_where_arity, copy_where_case);

macro_rules! count_if_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let values = $values(&exec);
        massively::count_if(&exec, values.slice(..), ArityPred, ()).unwrap();
    }};
}
define_value_arity_tests!(count_if_arity, count_if_case);

macro_rules! equal_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let left = $values(&exec);
        let right = $values(&exec);
        massively::equal(&exec, left.slice(..), right.slice(..), ArityEq).unwrap();
    }};
}
define_value_arity_tests!(equal_arity, equal_case);

macro_rules! exclusive_scan_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let values = $values(&exec);
        let out = $output(&exec);
        massively::exclusive_scan(
            &exec,
            values.slice(..),
            $init,
            ArityReduce,
            out.slice_mut(..),
        )
        .unwrap();
    }};
}
define_value_arity_tests!(exclusive_scan_arity, exclusive_scan_case);

macro_rules! exclusive_scan_by_key_key_case {
    ($keys:ident, $owned:ty) => {{
        let exec = exec();
        let keys = $keys(&exec);
        let values = values1(&exec);
        let out = output1(&exec);
        massively::exclusive_scan_by_key(
            &exec,
            keys.slice(..),
            values.slice(..),
            ArityEq,
            (0_u32,),
            ArityReduce,
            out.slice_mut(..),
        )
        .unwrap();
    }};
}
define_key_arity_tests!(
    exclusive_scan_by_key_key_arity,
    exclusive_scan_by_key_key_case
);

macro_rules! exclusive_scan_by_key_value_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let keys = values1(&exec);
        let values = $values(&exec);
        let out = $output(&exec);
        massively::exclusive_scan_by_key(
            &exec,
            keys.slice(..),
            values.slice(..),
            ArityEq,
            $init,
            ArityReduce,
            out.slice_mut(..),
        )
        .unwrap();
    }};
}
define_value_arity_tests!(
    exclusive_scan_by_key_value_arity,
    exclusive_scan_by_key_value_case
);

macro_rules! fill_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let out = $output(&exec);
        massively::fill(&exec, $init, out.slice_mut(..)).unwrap();
    }};
}
define_value_arity_tests!(fill_arity, fill_case);

macro_rules! find_first_of_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let values = $values(&exec);
        let needles = $values(&exec);
        massively::find_first_of(&exec, values.slice(..), needles.slice(..), ArityEq).unwrap();
    }};
}
define_value_arity_tests!(find_first_of_arity, find_first_of_case);

macro_rules! find_if_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let values = $values(&exec);
        massively::find_if(&exec, values.slice(..), ArityPred, ()).unwrap();
    }};
}
define_value_arity_tests!(find_if_arity, find_if_case);

macro_rules! gather_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let values = $values(&exec);
        let out = $output(&exec);
        let indices = indices(&exec);
        massively::gather(
            &exec,
            values.slice(..),
            indices.slice(..),
            out.slice_mut(..),
        )
        .unwrap();
    }};
}
define_value_arity_tests!(gather_arity, gather_case);

macro_rules! gather_where_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let values = $values(&exec);
        let out = $output(&exec);
        let indices = indices(&exec);
        let stencil = stencil(&exec);
        massively::gather_where(
            &exec,
            values.slice(..),
            indices.slice(..),
            stencil.slice(..),
            out.slice_mut(..),
        )
        .unwrap();
    }};
}
define_value_arity_tests!(gather_where_arity, gather_where_case);

macro_rules! inclusive_scan_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let values = $values(&exec);
        let out = $output(&exec);
        massively::inclusive_scan(&exec, values.slice(..), ArityReduce, out.slice_mut(..)).unwrap();
    }};
}
define_value_arity_tests!(inclusive_scan_arity, inclusive_scan_case);

macro_rules! inclusive_scan_by_key_key_case {
    ($keys:ident, $owned:ty) => {{
        let exec = exec();
        let keys = $keys(&exec);
        let values = values1(&exec);
        let out = output1(&exec);
        massively::inclusive_scan_by_key(
            &exec,
            keys.slice(..),
            values.slice(..),
            ArityEq,
            ArityReduce,
            out.slice_mut(..),
        )
        .unwrap();
    }};
}
define_key_arity_tests!(
    inclusive_scan_by_key_key_arity,
    inclusive_scan_by_key_key_case
);

macro_rules! inclusive_scan_by_key_value_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let keys = values1(&exec);
        let values = $values(&exec);
        let out = $output(&exec);
        massively::inclusive_scan_by_key(
            &exec,
            keys.slice(..),
            values.slice(..),
            ArityEq,
            ArityReduce,
            out.slice_mut(..),
        )
        .unwrap();
    }};
}
define_value_arity_tests!(
    inclusive_scan_by_key_value_arity,
    inclusive_scan_by_key_value_case
);

macro_rules! is_partitioned_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let values = $values(&exec);
        massively::is_partitioned(&exec, values.slice(..), ArityPred, ()).unwrap();
    }};
}
define_value_arity_tests!(is_partitioned_arity, is_partitioned_case);

macro_rules! is_sorted_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let values = $values(&exec);
        massively::is_sorted(&exec, values.slice(..), ArityLess).unwrap();
    }};
}
define_value_arity_tests!(is_sorted_arity, is_sorted_case);

macro_rules! is_sorted_until_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let values = $values(&exec);
        massively::is_sorted_until(&exec, values.slice(..), ArityLess).unwrap();
    }};
}
define_value_arity_tests!(is_sorted_until_arity, is_sorted_until_case);

macro_rules! lexicographical_compare_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let left = $values(&exec);
        let right = $values(&exec);
        massively::lexicographical_compare(&exec, left.slice(..), right.slice(..), ArityLess)
            .unwrap();
    }};
}
define_value_arity_tests!(lexicographical_compare_arity, lexicographical_compare_case);

macro_rules! lower_bound_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let values = $values(&exec);
        let needles = $values(&exec);
        let out = zeros_u32(&exec);
        massively::lower_bound(
            &exec,
            values.slice(..),
            needles.slice(..),
            ArityLess,
            out.slice_mut(..),
        )
        .unwrap();
    }};
}
define_value_arity_tests!(lower_bound_arity, lower_bound_case);

macro_rules! map_input_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let values = $values(&exec);
        let out = output1(&exec);
        massively::transform(
            &exec,
            values.slice(..),
            ArityTupleToScalar,
            (),
            out.slice_mut(..),
        )
        .unwrap();
    }};
}
define_value_arity_tests!(map_input_arity, map_input_case);

macro_rules! map_output_case {
    ($values_fn:ident, $output:ident, $owned:ty, $op:ident) => {{
        let exec = exec();
        let _ = $values_fn;
        let values = values1(&exec);
        let out = $output(&exec);
        massively::transform(&exec, values.slice(..), $op, (), out.slice_mut(..)).unwrap();
    }};
}
define_output_arity_tests!(map_output_arity, map_output_case);

macro_rules! max_element_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let values = $values(&exec);
        massively::max_element(&exec, values.slice(..), ArityLess).unwrap();
    }};
}
define_value_arity_tests!(max_element_arity, max_element_case);

macro_rules! merge_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let left = $values(&exec);
        let right = $values(&exec);
        let out = $output(&exec);
        massively::merge(
            &exec,
            left.slice(0..2),
            right.slice(0..2),
            ArityLess,
            out.slice_mut(..),
        )
        .unwrap();
    }};
}
define_value_arity_tests!(merge_arity, merge_case);

macro_rules! merge_by_key_key_case {
    ($keys:ident, $owned:ty) => {{
        let exec = exec();
        let left_keys = $keys(&exec);
        let right_keys = $keys(&exec);
        let left_values = values1(&exec);
        let right_values = values1(&exec);
        let out_keys = $keys(&exec);
        let out_values = output1(&exec);
        massively::merge_by_key(
            &exec,
            left_keys.slice(0..2),
            left_values.slice(0..2),
            right_keys.slice(0..2),
            right_values.slice(0..2),
            ArityLess,
            out_keys.slice_mut(..),
            out_values.slice_mut(..),
        )
        .unwrap();
    }};
}
define_key_arity_tests!(merge_by_key_key_arity, merge_by_key_key_case);

macro_rules! merge_by_key_value_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let left_keys = values1(&exec);
        let right_keys = values1(&exec);
        let left_values = $values(&exec);
        let right_values = $values(&exec);
        let out_keys = output1(&exec);
        let out_values = $output(&exec);
        massively::merge_by_key(
            &exec,
            left_keys.slice(0..2),
            left_values.slice(0..2),
            right_keys.slice(0..2),
            right_values.slice(0..2),
            ArityLess,
            out_keys.slice_mut(..),
            out_values.slice_mut(..),
        )
        .unwrap();
    }};
}
define_value_arity_tests!(merge_by_key_value_arity, merge_by_key_value_case);

macro_rules! min_element_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let values = $values(&exec);
        massively::min_element(&exec, values.slice(..), ArityLess).unwrap();
    }};
}
define_value_arity_tests!(min_element_arity, min_element_case);

macro_rules! minmax_element_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let values = $values(&exec);
        massively::minmax_element(&exec, values.slice(..), ArityLess).unwrap();
    }};
}
define_value_arity_tests!(minmax_element_arity, minmax_element_case);

macro_rules! mismatch_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let left = $values(&exec);
        let right = $values(&exec);
        massively::mismatch(&exec, left.slice(..), right.slice(..), ArityEq).unwrap();
    }};
}
define_value_arity_tests!(mismatch_arity, mismatch_case);

macro_rules! none_of_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let values = $values(&exec);
        massively::none_of(&exec, values.slice(..), ArityPred, ()).unwrap();
    }};
}
define_value_arity_tests!(none_of_arity, none_of_case);

macro_rules! partition_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let values = $values(&exec);
        let out = $output(&exec);
        massively::partition(&exec, values.slice(..), ArityPred, (), out.slice_mut(..)).unwrap();
    }};
}
define_value_arity_tests!(partition_arity, partition_case);

macro_rules! permute_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let values = $values(&exec);
        let out = $output(&exec);
        let indices = indices(&exec);
        massively::gather(
            &exec,
            values.slice(..),
            indices.slice(..),
            out.slice_mut(..),
        )
        .unwrap();
    }};
}
define_value_arity_tests!(permute_arity, permute_case);

macro_rules! reduce_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let values = $values(&exec);
        massively::reduce(&exec, values.slice(..), $init, ArityReduce).unwrap();
    }};
}
define_value_arity_tests!(reduce_arity, reduce_case);

macro_rules! reduce_by_key_key_case {
    ($keys:ident, $owned:ty) => {{
        let exec = exec();
        let keys = $keys(&exec);
        let values = values1(&exec);
        let out_keys = $keys(&exec);
        let out_values = output1(&exec);
        massively::reduce_by_key(
            &exec,
            keys.slice(..),
            values.slice(..),
            ArityEq,
            (0_u32,),
            ArityReduce,
            out_keys.slice_mut(..),
            out_values.slice_mut(..),
        )
        .unwrap();
    }};
}
define_key_arity_tests!(reduce_by_key_key_arity, reduce_by_key_key_case);

macro_rules! reduce_by_key_value_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let keys = values1(&exec);
        let values = $values(&exec);
        let out_keys = output1(&exec);
        let out_values = $output(&exec);
        massively::reduce_by_key(
            &exec,
            keys.slice(..),
            values.slice(..),
            ArityEq,
            $init,
            ArityReduce,
            out_keys.slice_mut(..),
            out_values.slice_mut(..),
        )
        .unwrap();
    }};
}
define_value_arity_tests!(reduce_by_key_value_arity, reduce_by_key_value_case);

macro_rules! remove_where_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let values = $values(&exec);
        let out = $output(&exec);
        let stencil = stencil(&exec);
        massively::remove_where(
            &exec,
            values.slice(..),
            stencil.slice(..),
            out.slice_mut(..),
        )
        .unwrap();
    }};
}
define_value_arity_tests!(remove_where_arity, remove_where_case);

macro_rules! replace_where_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let out = $output(&exec);
        let stencil = stencil(&exec);
        massively::replace_where(&exec, $init, stencil.slice(..), out.slice_mut(..)).unwrap();
    }};
}
define_value_arity_tests!(replace_where_arity, replace_where_case);

macro_rules! reverse_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let values = $values(&exec);
        let out = $output(&exec);
        massively::reverse(&exec, values.slice(..), out.slice_mut(..)).unwrap();
    }};
}
define_value_arity_tests!(reverse_arity, reverse_case);

macro_rules! scatter_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let values = $values(&exec);
        let out = $output(&exec);
        let indices = indices(&exec);
        massively::scatter(
            &exec,
            values.slice(..),
            indices.slice(..),
            out.slice_mut(..),
        )
        .unwrap();
    }};
}
define_value_arity_tests!(scatter_arity, scatter_case);

macro_rules! scatter_where_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let values = $values(&exec);
        let out = $output(&exec);
        let indices = indices(&exec);
        let stencil = stencil(&exec);
        massively::scatter_where(
            &exec,
            values.slice(..),
            indices.slice(..),
            stencil.slice(..),
            out.slice_mut(..),
        )
        .unwrap();
    }};
}
define_value_arity_tests!(scatter_where_arity, scatter_where_case);

macro_rules! set_difference_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let left = $values(&exec);
        let right = $values(&exec);
        let out = $output(&exec);
        massively::set_difference(
            &exec,
            left.slice(..),
            right.slice(..),
            ArityLess,
            out.slice_mut(..),
        )
        .unwrap();
    }};
}
define_value_arity_tests!(set_difference_arity, set_difference_case);

macro_rules! set_intersection_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let left = $values(&exec);
        let right = $values(&exec);
        let out = $output(&exec);
        massively::set_intersection(
            &exec,
            left.slice(..),
            right.slice(..),
            ArityLess,
            out.slice_mut(..),
        )
        .unwrap();
    }};
}
define_value_arity_tests!(set_intersection_arity, set_intersection_case);

macro_rules! set_union_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let left = $values(&exec);
        let right = $values(&exec);
        let out = $output(&exec);
        massively::set_union(
            &exec,
            left.slice(0..2),
            right.slice(0..2),
            ArityLess,
            out.slice_mut(..),
        )
        .unwrap();
    }};
}
define_value_arity_tests!(set_union_arity, set_union_case);

macro_rules! sort_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let values = $values(&exec);
        let out = $output(&exec);
        massively::sort(&exec, values.slice(..), ArityLess, out.slice_mut(..)).unwrap();
    }};
}
define_value_arity_tests!(sort_arity, sort_case);

macro_rules! sort_by_key_key_case {
    ($keys:ident, $owned:ty) => {{
        let exec = exec();
        let keys = $keys(&exec);
        let values = values1(&exec);
        let out_keys = $keys(&exec);
        let out_values = output1(&exec);
        massively::sort_by_key(
            &exec,
            keys.slice(..),
            values.slice(..),
            ArityLess,
            out_keys.slice_mut(..),
            out_values.slice_mut(..),
        )
        .unwrap();
    }};
}
define_key_arity_tests!(sort_by_key_key_arity, sort_by_key_key_case);

macro_rules! sort_by_key_value_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let keys = values1(&exec);
        let values = $values(&exec);
        let out_keys = output1(&exec);
        let out_values = $output(&exec);
        massively::sort_by_key(
            &exec,
            keys.slice(..),
            values.slice(..),
            ArityLess,
            out_keys.slice_mut(..),
            out_values.slice_mut(..),
        )
        .unwrap();
    }};
}
define_value_arity_tests!(sort_by_key_value_arity, sort_by_key_value_case);

macro_rules! stable_sort_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let values = $values(&exec);
        let out = $output(&exec);
        massively::stable_sort(&exec, values.slice(..), ArityLess, out.slice_mut(..)).unwrap();
    }};
}
define_value_arity_tests!(stable_sort_arity, stable_sort_case);

macro_rules! stable_sort_by_key_key_case {
    ($keys:ident, $owned:ty) => {{
        let exec = exec();
        let keys = $keys(&exec);
        let values = values1(&exec);
        let out_keys = $keys(&exec);
        let out_values = output1(&exec);
        massively::stable_sort_by_key(
            &exec,
            keys.slice(..),
            values.slice(..),
            ArityLess,
            out_keys.slice_mut(..),
            out_values.slice_mut(..),
        )
        .unwrap();
    }};
}
define_key_arity_tests!(stable_sort_by_key_key_arity, stable_sort_by_key_key_case);

macro_rules! stable_sort_by_key_value_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let keys = values1(&exec);
        let values = $values(&exec);
        let out_keys = output1(&exec);
        let out_values = $output(&exec);
        massively::stable_sort_by_key(
            &exec,
            keys.slice(..),
            values.slice(..),
            ArityLess,
            out_keys.slice_mut(..),
            out_values.slice_mut(..),
        )
        .unwrap();
    }};
}
define_value_arity_tests!(
    stable_sort_by_key_value_arity,
    stable_sort_by_key_value_case
);

macro_rules! transform_input_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let values = $values(&exec);
        let out = output1(&exec);
        massively::transform(
            &exec,
            values.slice(..),
            ArityTupleToScalar,
            (),
            out.slice_mut(..),
        )
        .unwrap();
    }};
}
define_value_arity_tests!(transform_input_arity, transform_input_case);

macro_rules! transform_output_case {
    ($values_fn:ident, $output:ident, $owned:ty, $op:ident) => {{
        let exec = exec();
        let _ = $values_fn;
        let values = values1(&exec);
        let out = $output(&exec);
        massively::transform(&exec, values.slice(..), $op, (), out.slice_mut(..)).unwrap();
    }};
}
define_output_arity_tests!(transform_output_arity, transform_output_case);

macro_rules! transform_where_input_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let values = $values(&exec);
        let out = output1(&exec);
        let stencil = stencil(&exec);
        massively::transform_where(
            &exec,
            values.slice(..),
            ArityTupleToScalar,
            (),
            stencil.slice(..),
            out.slice_mut(..),
        )
        .unwrap();
    }};
}
define_value_arity_tests!(transform_where_input_arity, transform_where_input_case);

macro_rules! transform_where_output_case {
    ($values_fn:ident, $output:ident, $owned:ty, $op:ident) => {{
        let exec = exec();
        let _ = $values_fn;
        let values = values1(&exec);
        let out = $output(&exec);
        let stencil = stencil(&exec);
        massively::transform_where(
            &exec,
            values.slice(..),
            $op,
            (),
            stencil.slice(..),
            out.slice_mut(..),
        )
        .unwrap();
    }};
}
define_output_arity_tests!(transform_where_output_arity, transform_where_output_case);

macro_rules! unique_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let values = $values(&exec);
        let out = $output(&exec);
        massively::unique(&exec, values.slice(..), ArityEq, out.slice_mut(..)).unwrap();
    }};
}
define_value_arity_tests!(unique_arity, unique_case);

macro_rules! unique_by_key_key_case {
    ($keys:ident, $owned:ty) => {{
        let exec = exec();
        let keys = $keys(&exec);
        let values = values1(&exec);
        let out_keys = $keys(&exec);
        let out_values = output1(&exec);
        massively::unique_by_key(
            &exec,
            keys.slice(..),
            values.slice(..),
            ArityEq,
            out_keys.slice_mut(..),
            out_values.slice_mut(..),
        )
        .unwrap();
    }};
}
define_key_arity_tests!(unique_by_key_key_arity, unique_by_key_key_case);

macro_rules! unique_by_key_value_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let keys = values1(&exec);
        let values = $values(&exec);
        let out_keys = output1(&exec);
        let out_values = $output(&exec);
        massively::unique_by_key(
            &exec,
            keys.slice(..),
            values.slice(..),
            ArityEq,
            out_keys.slice_mut(..),
            out_values.slice_mut(..),
        )
        .unwrap();
    }};
}
define_value_arity_tests!(unique_by_key_value_arity, unique_by_key_value_case);

macro_rules! upper_bound_case {
    ($values:ident, $output:ident, $owned:ty, $init:expr) => {{
        let exec = exec();
        let values = $values(&exec);
        let needles = $values(&exec);
        let out = zeros_u32(&exec);
        massively::upper_bound(
            &exec,
            values.slice(..),
            needles.slice(..),
            ArityLess,
            out.slice_mut(..),
        )
        .unwrap();
    }};
}
define_value_arity_tests!(upper_bound_arity, upper_bound_case);
