use cubecl::frontend::PartialEqExpand;
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::op as gpu_op;
use massively::{Executor, MIndex, MIter};
use oracle::op as host_op;
use proptest::prelude::*;
use std::sync::{Mutex, MutexGuard};

type ApiRuntime = WgpuRuntime;
type ApiExecutor = Executor<ApiRuntime>;

const CASES: u32 = 24;
const MAX_LEN: usize = 64;
static GPU_LOCK: Mutex<()> = Mutex::new(());

fn mindex(value: usize) -> MIndex {
    value.try_into().unwrap()
}

fn opt_mindex(value: Option<usize>) -> Option<MIndex> {
    value.map(mindex)
}

fn opt_pair_mindex(value: Option<(usize, usize)>) -> Option<(MIndex, MIndex)> {
    value.map(|(left, right)| (mindex(left), mindex(right)))
}

struct MaxTuple;
struct KeepTuple;
struct EqTuple;
struct LessTuple;
struct LessU32;
struct Less2;
struct Less3;
struct ArityTupleToTuple1;
struct ArityTupleToTuple2;
struct ArityTupleToTuple3;
struct ArityTupleToTuple4;
struct ArityTupleToTuple5;
struct ArityTupleToTuple6;
struct ArityTupleToTuple7;
struct U32Flag;

#[cubecl::cube]
impl gpu_op::UnaryOp<ApiRuntime, u32> for U32Flag {
    type Output = bool;

    fn apply(input: u32) -> bool {
        input != 0
    }
}

macro_rules! impl_arity_tuple_to_tuple {
    (($($ty:ty),+), $input:ident => $seed:expr) => {
        #[cubecl::cube]
        impl gpu_op::UnaryOp<ApiRuntime, ($($ty,)+)> for ArityTupleToTuple1 {
            type Output = (u32,);

            fn apply($input: ($($ty,)+)) -> (u32,) {
                let seed = $seed;
                (seed ^ 0x5a5a_5a5a,)
            }
        }

        impl host_op::UnaryOp<($($ty,)+)> for ArityTupleToTuple1 {
            type Output = (u32,);

            fn apply($input: ($($ty,)+)) -> (u32,) {
                let seed = $seed;
                (seed ^ 0x5a5a_5a5a,)
            }
        }

        #[cubecl::cube]
        impl gpu_op::UnaryOp<ApiRuntime, ($($ty,)+)> for ArityTupleToTuple2 {
            type Output = (u32, u32);

            fn apply($input: ($($ty,)+)) -> (u32, u32) {
                let seed = $seed;
                (seed ^ 0x5a5a_5a5a, (seed << 1) ^ 0xa5a5_a5a5)
            }
        }

        impl host_op::UnaryOp<($($ty,)+)> for ArityTupleToTuple2 {
            type Output = (u32, u32);

            fn apply($input: ($($ty,)+)) -> (u32, u32) {
                let seed = $seed;
                (seed ^ 0x5a5a_5a5a, (seed << 1) ^ 0xa5a5_a5a5)
            }
        }

        #[cubecl::cube]
        impl gpu_op::UnaryOp<ApiRuntime, ($($ty,)+)> for ArityTupleToTuple3 {
            type Output = (u32, u32, u32);

            fn apply($input: ($($ty,)+)) -> (u32, u32, u32) {
                let seed = $seed;
                (
                    seed ^ 0x5a5a_5a5a,
                    (seed << 1) ^ 0xa5a5_a5a5,
                    (seed >> 1) ^ 0x3c3c_3c3c,
                )
            }
        }

        impl host_op::UnaryOp<($($ty,)+)> for ArityTupleToTuple3 {
            type Output = (u32, u32, u32);

            fn apply($input: ($($ty,)+)) -> (u32, u32, u32) {
                let seed = $seed;
                (
                    seed ^ 0x5a5a_5a5a,
                    (seed << 1) ^ 0xa5a5_a5a5,
                    (seed >> 1) ^ 0x3c3c_3c3c,
                )
            }
        }

        #[cubecl::cube]
        impl gpu_op::UnaryOp<ApiRuntime, ($($ty,)+)> for ArityTupleToTuple4 {
            type Output = (u32, u32, u32, u32);

            fn apply($input: ($($ty,)+)) -> (u32, u32, u32, u32) {
                let seed = $seed;
                (
                    seed ^ 0x5a5a_5a5a,
                    (seed << 1) ^ 0xa5a5_a5a5,
                    (seed >> 1) ^ 0x3c3c_3c3c,
                    (seed << 2) ^ 0xc3c3_c3c3,
                )
            }
        }

        impl host_op::UnaryOp<($($ty,)+)> for ArityTupleToTuple4 {
            type Output = (u32, u32, u32, u32);

            fn apply($input: ($($ty,)+)) -> (u32, u32, u32, u32) {
                let seed = $seed;
                (
                    seed ^ 0x5a5a_5a5a,
                    (seed << 1) ^ 0xa5a5_a5a5,
                    (seed >> 1) ^ 0x3c3c_3c3c,
                    (seed << 2) ^ 0xc3c3_c3c3,
                )
            }
        }

        #[cubecl::cube]
        impl gpu_op::UnaryOp<ApiRuntime, ($($ty,)+)> for ArityTupleToTuple5 {
            type Output = (u32, u32, u32, u32, u32);

            fn apply($input: ($($ty,)+)) -> (u32, u32, u32, u32, u32) {
                let seed = $seed;
                (
                    seed ^ 0x5a5a_5a5a,
                    (seed << 1) ^ 0xa5a5_a5a5,
                    (seed >> 1) ^ 0x3c3c_3c3c,
                    (seed << 2) ^ 0xc3c3_c3c3,
                    (seed >> 2) ^ 0x0f0f_0f0f,
                )
            }
        }

        impl host_op::UnaryOp<($($ty,)+)> for ArityTupleToTuple5 {
            type Output = (u32, u32, u32, u32, u32);

            fn apply($input: ($($ty,)+)) -> (u32, u32, u32, u32, u32) {
                let seed = $seed;
                (
                    seed ^ 0x5a5a_5a5a,
                    (seed << 1) ^ 0xa5a5_a5a5,
                    (seed >> 1) ^ 0x3c3c_3c3c,
                    (seed << 2) ^ 0xc3c3_c3c3,
                    (seed >> 2) ^ 0x0f0f_0f0f,
                )
            }
        }

        #[cubecl::cube]
        impl gpu_op::UnaryOp<ApiRuntime, ($($ty,)+)> for ArityTupleToTuple6 {
            type Output = (u32, u32, u32, u32, u32, u32);

            fn apply($input: ($($ty,)+)) -> (u32, u32, u32, u32, u32, u32) {
                let seed = $seed;
                (
                    seed ^ 0x5a5a_5a5a,
                    (seed << 1) ^ 0xa5a5_a5a5,
                    (seed >> 1) ^ 0x3c3c_3c3c,
                    (seed << 2) ^ 0xc3c3_c3c3,
                    (seed >> 2) ^ 0x0f0f_0f0f,
                    (seed << 3) ^ 0xf0f0_f0f0,
                )
            }
        }

        impl host_op::UnaryOp<($($ty,)+)> for ArityTupleToTuple6 {
            type Output = (u32, u32, u32, u32, u32, u32);

            fn apply($input: ($($ty,)+)) -> (u32, u32, u32, u32, u32, u32) {
                let seed = $seed;
                (
                    seed ^ 0x5a5a_5a5a,
                    (seed << 1) ^ 0xa5a5_a5a5,
                    (seed >> 1) ^ 0x3c3c_3c3c,
                    (seed << 2) ^ 0xc3c3_c3c3,
                    (seed >> 2) ^ 0x0f0f_0f0f,
                    (seed << 3) ^ 0xf0f0_f0f0,
                )
            }
        }

        #[cubecl::cube]
        impl gpu_op::UnaryOp<ApiRuntime, ($($ty,)+)> for ArityTupleToTuple7 {
            type Output = (u32, u32, u32, u32, u32, u32, u32);

            fn apply($input: ($($ty,)+)) -> (u32, u32, u32, u32, u32, u32, u32) {
                let seed = $seed;
                (
                    seed ^ 0x5a5a_5a5a,
                    (seed << 1) ^ 0xa5a5_a5a5,
                    (seed >> 1) ^ 0x3c3c_3c3c,
                    (seed << 2) ^ 0xc3c3_c3c3,
                    (seed >> 2) ^ 0x0f0f_0f0f,
                    (seed << 3) ^ 0xf0f0_f0f0,
                    (seed >> 3) ^ 0x9696_9696,
                )
            }
        }

        impl host_op::UnaryOp<($($ty,)+)> for ArityTupleToTuple7 {
            type Output = (u32, u32, u32, u32, u32, u32, u32);

            fn apply($input: ($($ty,)+)) -> (u32, u32, u32, u32, u32, u32, u32) {
                let seed = $seed;
                (
                    seed ^ 0x5a5a_5a5a,
                    (seed << 1) ^ 0xa5a5_a5a5,
                    (seed >> 1) ^ 0x3c3c_3c3c,
                    (seed << 2) ^ 0xc3c3_c3c3,
                    (seed >> 2) ^ 0x0f0f_0f0f,
                    (seed << 3) ^ 0xf0f0_f0f0,
                    (seed >> 3) ^ 0x9696_9696,
                )
            }
        }
    };
}

impl_arity_tuple_to_tuple!((u32), input => input.0);
impl_arity_tuple_to_tuple!((u32, u32), input => input.0 ^ (input.1 << 1));
impl_arity_tuple_to_tuple!((u32, u32, u32), input => input.0 ^ (input.1 << 1) ^ (input.2 << 2));
impl_arity_tuple_to_tuple!((u32, u32, u32, u32), input => input.0 ^ (input.1 << 1) ^ (input.2 << 2) ^ (input.3 << 3));
impl_arity_tuple_to_tuple!((u32, u32, u32, u32, u32), input => input.0 ^ (input.1 << 1) ^ (input.2 << 2) ^ (input.3 << 3) ^ (input.4 << 4));
impl_arity_tuple_to_tuple!((u32, u32, u32, u32, u32, u32), input => input.0 ^ (input.1 << 1) ^ (input.2 << 2) ^ (input.3 << 3) ^ (input.4 << 4) ^ (input.5 << 5));
impl_arity_tuple_to_tuple!((u32, u32, u32, u32, u32, u32, u32), input => input.0 ^ (input.1 << 1) ^ (input.2 << 2) ^ (input.3 << 3) ^ (input.4 << 4) ^ (input.5 << 5) ^ (input.6 << 6));

macro_rules! impl_tuple_ops {
    (($($ty:ty),+), ($($field:tt),+)) => {
        #[cubecl::cube]
        impl gpu_op::ReductionOp<ApiRuntime, ($($ty,)+)> for MaxTuple {
            fn apply(lhs: ($($ty,)+), rhs: ($($ty,)+)) -> ($($ty,)+) {
                ($(lhs.$field.max(rhs.$field),)+)
            }
        }

        impl host_op::ReductionOp<($($ty,)+)> for MaxTuple {
            fn apply(lhs: ($($ty,)+), rhs: ($($ty,)+)) -> ($($ty,)+) {
                ($(lhs.$field.max(rhs.$field),)+)
            }
        }

        #[cubecl::cube]
        impl gpu_op::BinaryPredicateOp<ApiRuntime, ($($ty,)+)> for EqTuple {
            fn apply(lhs: ($($ty,)+), rhs: ($($ty,)+)) -> bool {
                true $(&& lhs.$field == rhs.$field)+
            }
        }

        impl host_op::BinaryPredicateOp<($($ty,)+)> for EqTuple {
            fn apply(lhs: ($($ty,)+), rhs: ($($ty,)+)) -> bool {
                true $(&& lhs.$field == rhs.$field)+
            }
        }
    };
}

macro_rules! impl_keep_tuple {
    (($($ty:ty),+), $zero:expr) => {
        #[cubecl::cube]
        impl gpu_op::PredicateOp<ApiRuntime, ($($ty,)+)> for KeepTuple {

            fn apply(input: ($($ty,)+)) -> bool {
                input.0 > $zero
            }
        }

        impl host_op::PredicateOp<($($ty,)+)> for KeepTuple {

            fn apply(input: ($($ty,)+)) -> bool {
                input.0 > $zero
            }
        }
    };
}

macro_rules! impl_less_tuple {
    (($($ty:ty),+), ($($field:tt),+)) => {
        #[cubecl::cube]
        impl gpu_op::BinaryPredicateOp<ApiRuntime, ($($ty,)+)> for LessTuple {
            fn apply(lhs: ($($ty,)+), rhs: ($($ty,)+)) -> bool {
                lhs.0 < rhs.0
            }
        }

        impl host_op::BinaryPredicateOp<($($ty,)+)> for LessTuple {
            fn apply(lhs: ($($ty,)+), rhs: ($($ty,)+)) -> bool {
                lhs.0 < rhs.0
            }
        }
    };
}

impl_tuple_ops!((u32), (0));
impl_tuple_ops!((u32, u32), (0, 1));
impl_tuple_ops!((u32, u32, u32), (0, 1, 2));
impl_tuple_ops!((u32, u32, u32, u32), (0, 1, 2, 3));
impl_tuple_ops!((u32, u32, u32, u32, u32), (0, 1, 2, 3, 4));
impl_tuple_ops!((u32, u32, u32, u32, u32, u32), (0, 1, 2, 3, 4, 5));
impl_tuple_ops!((u32, u32, u32, u32, u32, u32, u32), (0, 1, 2, 3, 4, 5, 6));

impl_keep_tuple!((u32), 0_u32);
impl_keep_tuple!((u32, u32), 0_u32);
impl_keep_tuple!((u32, u32, u32), 0_u32);
impl_keep_tuple!((u32, u32, u32, u32), 0_u32);
impl_keep_tuple!((u32, u32, u32, u32, u32), 0_u32);
impl_keep_tuple!((u32, u32, u32, u32, u32, u32), 0_u32);
impl_keep_tuple!((u32, u32, u32, u32, u32, u32, u32), 0_u32);

impl_less_tuple!((u32), (0));
impl_less_tuple!((u32, u32), (0, 1));
impl_less_tuple!((u32, u32, u32), (0, 1, 2));
impl_less_tuple!((u32, u32, u32, u32), (0, 1, 2, 3));
impl_less_tuple!((u32, u32, u32, u32, u32), (0, 1, 2, 3, 4));
impl_less_tuple!((u32, u32, u32, u32, u32, u32), (0, 1, 2, 3, 4, 5));
impl_less_tuple!((u32, u32, u32, u32, u32, u32, u32), (0, 1, 2, 3, 4, 5, 6));

#[cubecl::cube]
impl gpu_op::BinaryPredicateOp<ApiRuntime, (u32,)> for LessU32 {
    fn apply(lhs: (u32,), rhs: (u32,)) -> bool {
        lhs.0 < rhs.0
    }
}

impl host_op::BinaryPredicateOp<(u32,)> for LessU32 {
    fn apply(lhs: (u32,), rhs: (u32,)) -> bool {
        lhs.0 < rhs.0
    }
}

#[cubecl::cube]
impl gpu_op::BinaryPredicateOp<ApiRuntime, (u32, u32)> for Less2 {
    fn apply(lhs: (u32, u32), rhs: (u32, u32)) -> bool {
        lhs.0 < rhs.0 || (lhs.0 == rhs.0 && lhs.1 < rhs.1)
    }
}

impl host_op::BinaryPredicateOp<(u32, u32)> for Less2 {
    fn apply(lhs: (u32, u32), rhs: (u32, u32)) -> bool {
        lhs.0 < rhs.0 || (lhs.0 == rhs.0 && lhs.1 < rhs.1)
    }
}

#[cubecl::cube]
impl gpu_op::BinaryPredicateOp<ApiRuntime, (u32, u32, u32)> for Less3 {
    fn apply(lhs: (u32, u32, u32), rhs: (u32, u32, u32)) -> bool {
        lhs.0 < rhs.0 || (lhs.0 == rhs.0 && (lhs.1 < rhs.1 || (lhs.1 == rhs.1 && lhs.2 < rhs.2)))
    }
}

impl host_op::BinaryPredicateOp<(u32, u32, u32)> for Less3 {
    fn apply(lhs: (u32, u32, u32), rhs: (u32, u32, u32)) -> bool {
        lhs.0 < rhs.0 || (lhs.0 == rhs.0 && (lhs.1 < rhs.1 || (lhs.1 == rhs.1 && lhs.2 < rhs.2)))
    }
}

fn exec() -> ApiExecutor {
    Executor::<ApiRuntime>::new(WgpuDevice::Cpu)
}

fn gpu_lock() -> MutexGuard<'static, ()> {
    GPU_LOCK.lock().unwrap_or_else(|err| err.into_inner())
}

fn lazify<Input>(input: Input) -> massively::lazy::Identity<Input>
where
    Input: MIter<ApiRuntime>,
{
    massively::lazy::identity(input)
}

fn unique_by<T, K, F>(input: &[T], first: F) -> bool
where
    K: PartialEq,
    F: Fn(&T) -> K,
{
    for i in 0..input.len() {
        for j in (i + 1)..input.len() {
            if first(&input[i]) == first(&input[j]) {
                return false;
            }
        }
    }
    true
}

trait OffsetFirst {
    fn offset_first(self, offset: u32) -> Self;
}

impl OffsetFirst for (u32,) {
    fn offset_first(self, offset: u32) -> Self {
        (self.0 + offset,)
    }
}

impl OffsetFirst for (u32, u32) {
    fn offset_first(self, offset: u32) -> Self {
        (self.0 + offset, self.1)
    }
}

impl OffsetFirst for (u32, u32, u32) {
    fn offset_first(self, offset: u32) -> Self {
        (self.0 + offset, self.1, self.2)
    }
}

impl OffsetFirst for (u32, u32, u32, u32) {
    fn offset_first(self, offset: u32) -> Self {
        (self.0 + offset, self.1, self.2, self.3)
    }
}

impl OffsetFirst for (u32, u32, u32, u32, u32) {
    fn offset_first(self, offset: u32) -> Self {
        (self.0 + offset, self.1, self.2, self.3, self.4)
    }
}

impl OffsetFirst for (u32, u32, u32, u32, u32, u32) {
    fn offset_first(self, offset: u32) -> Self {
        (self.0 + offset, self.1, self.2, self.3, self.4, self.5)
    }
}

impl OffsetFirst for (u32, u32, u32, u32, u32, u32, u32) {
    fn offset_first(self, offset: u32) -> Self {
        (
            self.0 + offset,
            self.1,
            self.2,
            self.3,
            self.4,
            self.5,
            self.6,
        )
    }
}

macro_rules! cols_to_aos {
    ($cols:expr, Zip1) => {{
        let (a,) = $cols;
        a.into_iter().map(|a| (a,)).collect::<Vec<_>>()
    }};
    ($cols:expr, Zip2) => {{
        let (a, b) = $cols;
        a.into_iter().zip(b).collect::<Vec<_>>()
    }};
    ($cols:expr, Zip3) => {{
        let (a, b, c) = $cols;
        a.into_iter()
            .zip(b)
            .zip(c)
            .map(|((a, b), c)| (a, b, c))
            .collect::<Vec<_>>()
    }};
    ($cols:expr, Zip4) => {{
        let (a, b, c, d) = $cols;
        a.into_iter()
            .zip(b)
            .zip(c)
            .zip(d)
            .map(|(((a, b), c), d)| (a, b, c, d))
            .collect::<Vec<_>>()
    }};
    ($cols:expr, Zip5) => {{
        let (a, b, c, d, e) = $cols;
        a.into_iter()
            .zip(b)
            .zip(c)
            .zip(d)
            .zip(e)
            .map(|((((a, b), c), d), e)| (a, b, c, d, e))
            .collect::<Vec<_>>()
    }};
    ($cols:expr, Zip6) => {{
        let (a, b, c, d, e, f) = $cols;
        a.into_iter()
            .zip(b)
            .zip(c)
            .zip(d)
            .zip(e)
            .zip(f)
            .map(|(((((a, b), c), d), e), f)| (a, b, c, d, e, f))
            .collect::<Vec<_>>()
    }};
    ($cols:expr, Zip7) => {{
        let (a, b, c, d, e, f, g) = $cols;
        a.into_iter()
            .zip(b)
            .zip(c)
            .zip(d)
            .zip(e)
            .zip(f)
            .zip(g)
            .map(|((((((a, b), c), d), e), f), g)| (a, b, c, d, e, f, g))
            .collect::<Vec<_>>()
    }};
}

macro_rules! make_zip {
    ($exec:expr, $input:expr, Zip1, ($a:ty)) => {{
        let c0: Vec<$a> = $input.iter().map(|row| row.0).collect();
        massively::Zip1($exec.to_device(&c0).unwrap())
    }};
    ($exec:expr, $input:expr, Zip2, ($a:ty, $b:ty)) => {{
        let c0: Vec<$a> = $input.iter().map(|row| row.0).collect();
        let c1: Vec<$b> = $input.iter().map(|row| row.1).collect();
        massively::Zip2($exec.to_device(&c0).unwrap(), $exec.to_device(&c1).unwrap())
    }};
    ($exec:expr, $input:expr, Zip3, ($a:ty, $b:ty, $c:ty)) => {{
        let c0: Vec<$a> = $input.iter().map(|row| row.0).collect();
        let c1: Vec<$b> = $input.iter().map(|row| row.1).collect();
        let c2: Vec<$c> = $input.iter().map(|row| row.2).collect();
        massively::Zip3(
            $exec.to_device(&c0).unwrap(),
            $exec.to_device(&c1).unwrap(),
            $exec.to_device(&c2).unwrap(),
        )
    }};
    ($exec:expr, $input:expr, Zip4, ($a:ty, $b:ty, $c:ty, $d:ty)) => {{
        let c0: Vec<$a> = $input.iter().map(|row| row.0).collect();
        let c1: Vec<$b> = $input.iter().map(|row| row.1).collect();
        let c2: Vec<$c> = $input.iter().map(|row| row.2).collect();
        let c3: Vec<$d> = $input.iter().map(|row| row.3).collect();
        massively::Zip4(
            $exec.to_device(&c0).unwrap(),
            $exec.to_device(&c1).unwrap(),
            $exec.to_device(&c2).unwrap(),
            $exec.to_device(&c3).unwrap(),
        )
    }};
    ($exec:expr, $input:expr, Zip5, ($a:ty, $b:ty, $c:ty, $d:ty, $e:ty)) => {{
        let c0: Vec<$a> = $input.iter().map(|row| row.0).collect();
        let c1: Vec<$b> = $input.iter().map(|row| row.1).collect();
        let c2: Vec<$c> = $input.iter().map(|row| row.2).collect();
        let c3: Vec<$d> = $input.iter().map(|row| row.3).collect();
        let c4: Vec<$e> = $input.iter().map(|row| row.4).collect();
        massively::Zip5(
            $exec.to_device(&c0).unwrap(),
            $exec.to_device(&c1).unwrap(),
            $exec.to_device(&c2).unwrap(),
            $exec.to_device(&c3).unwrap(),
            $exec.to_device(&c4).unwrap(),
        )
    }};
    ($exec:expr, $input:expr, Zip6, ($a:ty, $b:ty, $c:ty, $d:ty, $e:ty, $f:ty)) => {{
        let c0: Vec<$a> = $input.iter().map(|row| row.0).collect();
        let c1: Vec<$b> = $input.iter().map(|row| row.1).collect();
        let c2: Vec<$c> = $input.iter().map(|row| row.2).collect();
        let c3: Vec<$d> = $input.iter().map(|row| row.3).collect();
        let c4: Vec<$e> = $input.iter().map(|row| row.4).collect();
        let c5: Vec<$f> = $input.iter().map(|row| row.5).collect();
        massively::Zip6(
            $exec.to_device(&c0).unwrap(),
            $exec.to_device(&c1).unwrap(),
            $exec.to_device(&c2).unwrap(),
            $exec.to_device(&c3).unwrap(),
            $exec.to_device(&c4).unwrap(),
            $exec.to_device(&c5).unwrap(),
        )
    }};
    ($exec:expr, $input:expr, Zip7, ($a:ty, $b:ty, $c:ty, $d:ty, $e:ty, $f:ty, $g:ty)) => {{
        let c0: Vec<$a> = $input.iter().map(|row| row.0).collect();
        let c1: Vec<$b> = $input.iter().map(|row| row.1).collect();
        let c2: Vec<$c> = $input.iter().map(|row| row.2).collect();
        let c3: Vec<$d> = $input.iter().map(|row| row.3).collect();
        let c4: Vec<$e> = $input.iter().map(|row| row.4).collect();
        let c5: Vec<$f> = $input.iter().map(|row| row.5).collect();
        let c6: Vec<$g> = $input.iter().map(|row| row.6).collect();
        massively::Zip7(
            $exec.to_device(&c0).unwrap(),
            $exec.to_device(&c1).unwrap(),
            $exec.to_device(&c2).unwrap(),
            $exec.to_device(&c3).unwrap(),
            $exec.to_device(&c4).unwrap(),
            $exec.to_device(&c5).unwrap(),
            $exec.to_device(&c6).unwrap(),
        )
    }};
}

macro_rules! map_arity_case {
    ($input:expr, $input_zip:ident, ($($input_ty:ty),+), $output_zip:ident, ($($output_ty:ty),+), $init:expr, $op:ident) => {{
        let _ = $init;
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_input = make_zip!(&exec, &input, $input_zip, ($($input_ty),+));
        let gpu_output = make_zip!(&exec, &vec![$init; input.len()], $output_zip, ($($output_ty),+));
        massively::transform(&exec, lazify(gpu_input.slice(..)), $op, gpu_output.slice_mut(..)).unwrap();
        let gpu = cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $output_zip);
        let host = oracle::map(&input, $op);
        prop_assert_eq!(gpu, host);
    }};
}

macro_rules! transform_arity_case {
    ($input:expr, $input_zip:ident, ($($input_ty:ty),+), $output_zip:ident, ($($output_ty:ty),+), $init:expr, $op:ident) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let mut host = vec![$init; input.len()];
        let gpu_input = make_zip!(&exec, &input, $input_zip, ($($input_ty),+));
        let gpu_output = make_zip!(&exec, &host, $output_zip, ($($output_ty),+));
        massively::transform(
            &exec,
            lazify(gpu_input.slice(..)),
            $op,
            gpu_output.slice_mut(..),
        )
        .unwrap();
        oracle::transform(&input, $op, &mut host);
        let gpu = cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $output_zip);
        prop_assert_eq!(gpu, host);
    }};
}

macro_rules! transform_where_arity_case {
    ($input:expr, $input_zip:ident, ($($input_ty:ty),+), $output_zip:ident, ($($output_ty:ty),+), $init:expr, $op:ident) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let stencil = stencil_for!(input.len());
        let mut host = vec![$init; input.len()];
        let gpu_input = make_zip!(&exec, &input, $input_zip, ($($input_ty),+));
        let gpu_stencil = exec.to_device(&stencil).unwrap();
        let gpu_output = make_zip!(&exec, &host, $output_zip, ($($output_ty),+));
        massively::transform_where(
            &exec,
            lazify(gpu_input.slice(..)),
            $op,
            lazify(massively::lazy::transform(gpu_stencil.slice(..), U32Flag)),
            gpu_output.slice_mut(..),
        )
        .unwrap();
        oracle::transform_where(&input, $op, &stencil, &mut host);
        let gpu = cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $output_zip);
        prop_assert_eq!(gpu, host);
    }};
}

macro_rules! sort_by_key_only_case {
    ($pairs:expr, $key_zip:ident, ($($key_ty:ty),+), $value_zip:ident, ($($value_ty:ty),+), $less:ident) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let pairs = $pairs;
        let (keys, values): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
        let gpu_keys = make_zip!(&exec, &keys, $key_zip, ($($key_ty),+));
        let gpu_values = make_zip!(&exec, &values, $value_zip, ($($value_ty),+));
        let out_keys = make_zip!(&exec, &keys, $key_zip, ($($key_ty),+));
        let out_values = make_zip!(&exec, &values, $value_zip, ($($value_ty),+));
        massively::sort_by_key(&exec, lazify(gpu_keys.slice(..)), lazify(gpu_values.slice(..)), $less, out_keys.slice_mut(..), out_values.slice_mut(..)).unwrap();
        let (host_keys, host_values) = oracle::sort_by_key(&keys, &values, $less);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&out_keys).unwrap(), $key_zip), host_keys);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&out_values).unwrap(), $value_zip), host_values);
    }};
}

macro_rules! scan_by_key_case {
    (inclusive_scan_by_key, $pairs:expr, $key_zip:ident, ($($key_ty:ty),+), $value_zip:ident, ($($value_ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let pairs = $pairs;
        let (keys, values): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
        let gpu_keys = make_zip!(&exec, &keys, $key_zip, ($($key_ty),+));
        let gpu_values = make_zip!(&exec, &values, $value_zip, ($($value_ty),+));
        let gpu_output = make_zip!(&exec, &values, $value_zip, ($($value_ty),+));
        massively::inclusive_scan_by_key(
            &exec,
            lazify(gpu_keys.slice(..)),
            lazify(gpu_values.slice(..)),
            EqTuple,
            MaxTuple,
            gpu_output.slice_mut(..),
        )
        .unwrap();
        prop_assert_eq!(
            cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $value_zip),
            oracle::inclusive_scan_by_key(&keys, &values, EqTuple, MaxTuple)
        );
    }};
    (exclusive_scan_by_key, $pairs:expr, $key_zip:ident, ($($key_ty:ty),+), $value_zip:ident, ($($value_ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let pairs = $pairs;
        let (keys, values): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
        let gpu_keys = make_zip!(&exec, &keys, $key_zip, ($($key_ty),+));
        let gpu_values = make_zip!(&exec, &values, $value_zip, ($($value_ty),+));
        let gpu_output = make_zip!(&exec, &values, $value_zip, ($($value_ty),+));
        massively::exclusive_scan_by_key(
            &exec,
            lazify(gpu_keys.slice(..)),
            lazify(gpu_values.slice(..)),
            EqTuple,
            $init,
            MaxTuple,
            gpu_output.slice_mut(..),
        )
        .unwrap();
        prop_assert_eq!(
            cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $value_zip),
            oracle::exclusive_scan_by_key(&keys, &values, EqTuple, $init, MaxTuple)
        );
    }};
    (reduce_by_key, $pairs:expr, $key_zip:ident, ($($key_ty:ty),+), $value_zip:ident, ($($value_ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let pairs = $pairs;
        let (keys, values): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
        let gpu_keys = make_zip!(&exec, &keys, $key_zip, ($($key_ty),+));
        let gpu_values = make_zip!(&exec, &values, $value_zip, ($($value_ty),+));
        let gpu_out_keys = make_zip!(&exec, &keys, $key_zip, ($($key_ty),+));
        let gpu_out_values = make_zip!(&exec, &values, $value_zip, ($($value_ty),+));
        let len = massively::reduce_by_key(
            &exec,
            lazify(gpu_keys.slice(..)),
            lazify(gpu_values.slice(..)),
            EqTuple,
            $init,
            MaxTuple,
            gpu_out_keys.slice_mut(..),
            gpu_out_values.slice_mut(..),
        )
        .unwrap();
        let (host_keys, host_values) =
            oracle::reduce_by_key(&keys, &values, EqTuple, $init, MaxTuple);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_out_keys.slice(..len)).unwrap(), $key_zip), host_keys);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_out_values.slice(..len)).unwrap(), $value_zip), host_values);
    }};
    (unique_by_key, $pairs:expr, $key_zip:ident, ($($key_ty:ty),+), $value_zip:ident, ($($value_ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let pairs = $pairs;
        let (keys, values): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
        let gpu_keys = make_zip!(&exec, &keys, $key_zip, ($($key_ty),+));
        let gpu_values = make_zip!(&exec, &values, $value_zip, ($($value_ty),+));
        let gpu_out_keys = make_zip!(&exec, &keys, $key_zip, ($($key_ty),+));
        let gpu_out_values = make_zip!(&exec, &values, $value_zip, ($($value_ty),+));
        let len = massively::unique_by_key(
            &exec,
            lazify(gpu_keys.slice(..)),
            lazify(gpu_values.slice(..)),
            EqTuple,
            gpu_out_keys.slice_mut(..),
            gpu_out_values.slice_mut(..),
        )
        .unwrap();
        let (host_keys, host_values) = oracle::unique_by_key(&keys, &values, EqTuple);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_out_keys.slice(..len)).unwrap(), $key_zip), host_keys);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_out_values.slice(..len)).unwrap(), $value_zip), host_values);
    }};
}

macro_rules! merge_by_key_case {
    ($pairs:expr, $key_zip:ident, ($($key_ty:ty),+), $value_zip:ident, ($($value_ty:ty),+), $less:ident) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let pairs = $pairs;
        let mid = pairs.len() / 2;
        let (left_keys_unsorted, left_values_unsorted): (Vec<_>, Vec<_>) =
            pairs[..mid].iter().copied().unzip();
        let (right_keys_unsorted, right_values_unsorted): (Vec<_>, Vec<_>) =
            pairs[mid..].iter().copied().unzip();
        let (left_keys, left_values) =
            oracle::sort_by_key(&left_keys_unsorted, &left_values_unsorted, $less);
        let (right_keys, right_values) =
            oracle::sort_by_key(&right_keys_unsorted, &right_values_unsorted, $less);
        let gpu_left_keys = make_zip!(&exec, &left_keys, $key_zip, ($($key_ty),+));
        let gpu_right_keys = make_zip!(&exec, &right_keys, $key_zip, ($($key_ty),+));
        let gpu_left_values = make_zip!(&exec, &left_values, $value_zip, ($($value_ty),+));
        let gpu_right_values = make_zip!(&exec, &right_values, $value_zip, ($($value_ty),+));

        let gpu_keys = make_zip!(&exec, &pairs.iter().map(|pair| pair.0).collect::<Vec<_>>(), $key_zip, ($($key_ty),+));
        let gpu_values = make_zip!(&exec, &pairs.iter().map(|pair| pair.1).collect::<Vec<_>>(), $value_zip, ($($value_ty),+));
        massively::merge_by_key(
            &exec,
            lazify(gpu_left_keys.slice(..)),
            lazify(gpu_left_values.slice(..)),
            lazify(gpu_right_keys.slice(..)),
            lazify(gpu_right_values.slice(..)),
            $less,
            gpu_keys.slice_mut(..),
            gpu_values.slice_mut(..),
        )
        .unwrap();
        let (host_keys, host_values) = oracle::merge_by_key(
            &left_keys,
            &left_values,
            &right_keys,
            &right_values,
            $less,
        );
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_keys).unwrap(), $key_zip), host_keys);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_values).unwrap(), $value_zip), host_values);
    }};
}

macro_rules! stencil_for {
    ($len:expr) => {{
        (0..$len)
            .map(|i| if i % 2 == 0 { 1_u32 } else { 0_u32 })
            .collect::<Vec<_>>()
    }};
}

macro_rules! reverse_indices_for {
    ($len:expr) => {{ (0..$len).rev().map(|i| i as u32).collect::<Vec<_>>() }};
}

macro_rules! value_case {
    (reduce, $input:expr, $zip:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_input = make_zip!(&exec, &input, $zip, ($($ty),+));
        let gpu = massively::reduce(&exec, lazify(gpu_input.slice(..)), $init, MaxTuple).unwrap();
        let host = oracle::reduce(&input, $init, MaxTuple);
        prop_assert_eq!(gpu, host);
    }};
    (inclusive_scan, $input:expr, $zip:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_input = make_zip!(&exec, &input, $zip, ($($ty),+));
        let gpu_output = make_zip!(&exec, &input, $zip, ($($ty),+));
        massively::inclusive_scan(&exec, lazify(gpu_input.slice(..)), MaxTuple, gpu_output.slice_mut(..)).unwrap();
        let gpu = cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $zip);
        let host = oracle::inclusive_scan(&input, MaxTuple);
        prop_assert_eq!(gpu, host);
    }};
    (exclusive_scan, $input:expr, $zip:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_input = make_zip!(&exec, &input, $zip, ($($ty),+));
        let gpu_output = make_zip!(&exec, &input, $zip, ($($ty),+));
        massively::exclusive_scan(&exec, lazify(gpu_input.slice(..)), $init, MaxTuple, gpu_output.slice_mut(..)).unwrap();
        let gpu = cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $zip);
        let host = oracle::exclusive_scan(&input, $init, MaxTuple);
        prop_assert_eq!(gpu, host);
    }};
    (adjacent_difference, $input:expr, $zip:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_input = make_zip!(&exec, &input, $zip, ($($ty),+));
        let gpu_output = make_zip!(&exec, &input, $zip, ($($ty),+));
        massively::adjacent_difference(&exec, lazify(gpu_input.slice(..)), MaxTuple, gpu_output.slice_mut(..)).unwrap();
        let gpu = cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $zip);
        let host = oracle::adjacent_difference(&input, MaxTuple);
        prop_assert_eq!(gpu, host);
    }};
    (copy_where, $input:expr, $zip:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let stencil = stencil_for!(input.len());
        let gpu_input = make_zip!(&exec, &input, $zip, ($($ty),+));
        let gpu_stencil = exec.to_device(&stencil).unwrap();
        let gpu_output = make_zip!(&exec, &input, $zip, ($($ty),+));
        let len = massively::copy_where(&exec, lazify(gpu_input.slice(..)), lazify(massively::lazy::transform(gpu_stencil.slice(..), U32Flag)), gpu_output.slice_mut(..)).unwrap();
        let gpu = cols_to_aos!(exec.to_host(&gpu_output.slice(..len)).unwrap(), $zip);
        let host = oracle::copy_where(&input, &stencil);
        prop_assert_eq!(gpu, host);
    }};
    (remove_where, $input:expr, $zip:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let stencil = stencil_for!(input.len());
        let gpu_input = make_zip!(&exec, &input, $zip, ($($ty),+));
        let gpu_stencil = exec.to_device(&stencil).unwrap();
        let gpu_output = make_zip!(&exec, &input, $zip, ($($ty),+));
        let len = massively::remove_where(&exec, lazify(gpu_input.slice(..)), lazify(massively::lazy::transform(gpu_stencil.slice(..), U32Flag)), gpu_output.slice_mut(..)).unwrap();
        let gpu = cols_to_aos!(exec.to_host(&gpu_output.slice(..len)).unwrap(), $zip);
        let host = oracle::remove_where(&input, &stencil);
        prop_assert_eq!(gpu, host);
    }};
    (reverse, $input:expr, $zip:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_input = make_zip!(&exec, &input, $zip, ($($ty),+));
        let gpu_output = make_zip!(&exec, &input, $zip, ($($ty),+));
        massively::reverse(&exec, lazify(gpu_input.slice(..)), gpu_output.slice_mut(..)).unwrap();
        let gpu = cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $zip);
        let host = oracle::reverse(&input);
        prop_assert_eq!(gpu, host);
    }};
    (count_if, $input:expr, $zip:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_input = make_zip!(&exec, &input, $zip, ($($ty),+));
        prop_assert_eq!(
            massively::count_if(&exec, lazify(gpu_input.slice(..)), KeepTuple).unwrap(),
            mindex(oracle::count_if(&input, KeepTuple))
        );
    }};
    (all_of, $input:expr, $zip:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_input = make_zip!(&exec, &input, $zip, ($($ty),+));
        prop_assert_eq!(
            massively::all_of(&exec, lazify(gpu_input.slice(..)), KeepTuple).unwrap(),
            oracle::all_of(&input, KeepTuple)
        );
    }};
    (any_of, $input:expr, $zip:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_input = make_zip!(&exec, &input, $zip, ($($ty),+));
        prop_assert_eq!(
            massively::any_of(&exec, lazify(gpu_input.slice(..)), KeepTuple).unwrap(),
            oracle::any_of(&input, KeepTuple)
        );
    }};
    (none_of, $input:expr, $zip:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_input = make_zip!(&exec, &input, $zip, ($($ty),+));
        prop_assert_eq!(
            massively::none_of(&exec, lazify(gpu_input.slice(..)), KeepTuple).unwrap(),
            oracle::none_of(&input, KeepTuple)
        );
    }};
    (find_if, $input:expr, $zip:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_input = make_zip!(&exec, &input, $zip, ($($ty),+));
        prop_assert_eq!(
            massively::find_if(&exec, lazify(gpu_input.slice(..)), KeepTuple).unwrap(),
            opt_mindex(oracle::find_if(&input, KeepTuple))
        );
    }};
    (is_partitioned, $input:expr, $zip:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_input = make_zip!(&exec, &input, $zip, ($($ty),+));
        prop_assert_eq!(
            massively::is_partitioned(&exec, lazify(gpu_input.slice(..)), KeepTuple).unwrap(),
            oracle::is_partitioned(&input, KeepTuple)
        );
    }};
    (partition, $input:expr, $zip:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_input = make_zip!(&exec, &input, $zip, ($($ty),+));
        let gpu_output = make_zip!(&exec, &input, $zip, ($($ty),+));
        let split = massively::partition(&exec, lazify(gpu_input.slice(..)), KeepTuple, gpu_output.slice_mut(..)).unwrap();
        let (host_yes, host_no) = oracle::partition(&input, KeepTuple);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_output.slice(..split)).unwrap(), $zip), host_yes);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_output.slice(split..mindex(input.len()))).unwrap(), $zip), host_no);
    }};
    (predicate, $input:expr, $zip:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_input = make_zip!(&exec, &input, $zip, ($($ty),+));
        prop_assert_eq!(
            massively::count_if(&exec, lazify(gpu_input.slice(..)), KeepTuple).unwrap(),
            mindex(oracle::count_if(&input, KeepTuple))
        );
        prop_assert_eq!(
            massively::all_of(&exec, lazify(gpu_input.slice(..)), KeepTuple).unwrap(),
            oracle::all_of(&input, KeepTuple)
        );
        prop_assert_eq!(
            massively::any_of(&exec, lazify(gpu_input.slice(..)), KeepTuple).unwrap(),
            oracle::any_of(&input, KeepTuple)
        );
        prop_assert_eq!(
            massively::none_of(&exec, lazify(gpu_input.slice(..)), KeepTuple).unwrap(),
            oracle::none_of(&input, KeepTuple)
        );
        prop_assert_eq!(
            massively::find_if(&exec, lazify(gpu_input.slice(..)), KeepTuple).unwrap(),
            opt_mindex(oracle::find_if(&input, KeepTuple))
        );
        prop_assert_eq!(
            massively::is_partitioned(&exec, lazify(gpu_input.slice(..)), KeepTuple).unwrap(),
            oracle::is_partitioned(&input, KeepTuple)
        );
        let gpu_output = make_zip!(&exec, &input, $zip, ($($ty),+));
        let split = massively::partition(&exec, lazify(gpu_input.slice(..)), KeepTuple, gpu_output.slice_mut(..)).unwrap();
        let (host_yes, host_no) = oracle::partition(&input, KeepTuple);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_output.slice(..split)).unwrap(), $zip), host_yes);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_output.slice(split..mindex(input.len()))).unwrap(), $zip), host_no);
    }};
    (indexed, $input:expr, $zip:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let indices = reverse_indices_for!(input.len());
        let stencil = stencil_for!(input.len());
        let gpu_input = make_zip!(&exec, &input, $zip, ($($ty),+));
        let gpu_indices = exec.to_device(&indices).unwrap();
        let gpu_stencil = exec.to_device(&stencil).unwrap();

        let gpu_output = make_zip!(&exec, &input, $zip, ($($ty),+));
        massively::gather(&exec, lazify(gpu_input.slice(..)), lazify(gpu_indices.slice(..)), gpu_output.slice_mut(..)).unwrap();
        let mut host = input.clone();
        oracle::gather(&input, &indices, &mut host);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $zip), host);

        let gpu_output = make_zip!(&exec, &input, $zip, ($($ty),+));
        massively::gather(&exec, lazify(gpu_input.slice(..)), lazify(gpu_indices.slice(..)), gpu_output.slice_mut(..)).unwrap();
        let mut host = input.clone();
        oracle::gather(&input, &indices, &mut host);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $zip), host);

        let gpu_output = make_zip!(&exec, &input, $zip, ($($ty),+));
        massively::gather_where(
            &exec,
            lazify(gpu_input.slice(..)),
            lazify(gpu_indices.slice(..)),
            lazify(massively::lazy::transform(gpu_stencil.slice(..), U32Flag)),
            gpu_output.slice_mut(..),
        )
        .unwrap();
        let mut host = input.clone();
        oracle::gather_where(&input, &indices, &stencil, &mut host);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $zip), host);

        let gpu_output = make_zip!(&exec, &input, $zip, ($($ty),+));
        massively::scatter(&exec, lazify(gpu_input.slice(..)), lazify(gpu_indices.slice(..)), gpu_output.slice_mut(..)).unwrap();
        let mut host = input.clone();
        oracle::scatter(&input, &indices, &mut host);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $zip), host);

        let gpu_output = make_zip!(&exec, &input, $zip, ($($ty),+));
        massively::scatter_where(
            &exec,
            lazify(gpu_input.slice(..)),
            lazify(gpu_indices.slice(..)),
            lazify(massively::lazy::transform(gpu_stencil.slice(..), U32Flag)),
            gpu_output.slice_mut(..),
        )
        .unwrap();
        let mut host = input.clone();
        oracle::scatter_where(&input, &indices, &stencil, &mut host);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $zip), host);
    }};
    (permute, $input:expr, $zip:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let indices = reverse_indices_for!(input.len());
        let gpu_input = make_zip!(&exec, &input, $zip, ($($ty),+));
        let gpu_indices = exec.to_device(&indices).unwrap();
        let gpu_output = make_zip!(&exec, &input, $zip, ($($ty),+));
        massively::gather(&exec, lazify(gpu_input.slice(..)), lazify(gpu_indices.slice(..)), gpu_output.slice_mut(..)).unwrap();
        let mut host = input.clone();
        oracle::gather(&input, &indices, &mut host);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $zip), host);
    }};
    (gather, $input:expr, $zip:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let indices = reverse_indices_for!(input.len());
        let gpu_input = make_zip!(&exec, &input, $zip, ($($ty),+));
        let gpu_indices = exec.to_device(&indices).unwrap();
        let gpu_output = make_zip!(&exec, &input, $zip, ($($ty),+));
        massively::gather(&exec, lazify(gpu_input.slice(..)), lazify(gpu_indices.slice(..)), gpu_output.slice_mut(..)).unwrap();
        let mut host = input.clone();
        oracle::gather(&input, &indices, &mut host);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $zip), host);
    }};
    (gather_where, $input:expr, $zip:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let indices = reverse_indices_for!(input.len());
        let stencil = stencil_for!(input.len());
        let gpu_input = make_zip!(&exec, &input, $zip, ($($ty),+));
        let gpu_indices = exec.to_device(&indices).unwrap();
        let gpu_stencil = exec.to_device(&stencil).unwrap();
        let gpu_output = make_zip!(&exec, &input, $zip, ($($ty),+));
        massively::gather_where(
            &exec,
            lazify(gpu_input.slice(..)),
            lazify(gpu_indices.slice(..)),
            lazify(massively::lazy::transform(gpu_stencil.slice(..), U32Flag)),
            gpu_output.slice_mut(..),
        )
        .unwrap();
        let mut host = input.clone();
        oracle::gather_where(&input, &indices, &stencil, &mut host);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $zip), host);
    }};
    (scatter, $input:expr, $zip:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let indices = reverse_indices_for!(input.len());
        let gpu_input = make_zip!(&exec, &input, $zip, ($($ty),+));
        let gpu_indices = exec.to_device(&indices).unwrap();
        let gpu_output = make_zip!(&exec, &input, $zip, ($($ty),+));
        massively::scatter(&exec, lazify(gpu_input.slice(..)), lazify(gpu_indices.slice(..)), gpu_output.slice_mut(..)).unwrap();
        let mut host = input.clone();
        oracle::scatter(&input, &indices, &mut host);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $zip), host);
    }};
    (scatter_where, $input:expr, $zip:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let indices = reverse_indices_for!(input.len());
        let stencil = stencil_for!(input.len());
        let gpu_input = make_zip!(&exec, &input, $zip, ($($ty),+));
        let gpu_indices = exec.to_device(&indices).unwrap();
        let gpu_stencil = exec.to_device(&stencil).unwrap();
        let gpu_output = make_zip!(&exec, &input, $zip, ($($ty),+));
        massively::scatter_where(
            &exec,
            lazify(gpu_input.slice(..)),
            lazify(gpu_indices.slice(..)),
            lazify(massively::lazy::transform(gpu_stencil.slice(..), U32Flag)),
            gpu_output.slice_mut(..),
        )
        .unwrap();
        let mut host = input.clone();
        oracle::scatter_where(&input, &indices, &stencil, &mut host);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $zip), host);
    }};
    (search_eq, $input:expr, $zip:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let mut other = input.clone();
        other.reverse();
        let gpu_input = make_zip!(&exec, &input, $zip, ($($ty),+));
        let gpu_other = make_zip!(&exec, &other, $zip, ($($ty),+));
        prop_assert_eq!(
            massively::equal(&exec, lazify(gpu_input.slice(..)), lazify(gpu_input.slice(..)), EqTuple).unwrap(),
            oracle::equal(&input, &input, EqTuple)
        );
        prop_assert_eq!(
            massively::mismatch(&exec, lazify(gpu_input.slice(..)), lazify(gpu_other.slice(..)), EqTuple).unwrap(),
            opt_mindex(oracle::mismatch(&input, &other, EqTuple))
        );
        prop_assert_eq!(
            massively::adjacent_find(&exec, lazify(gpu_input.slice(..)), EqTuple).unwrap(),
            opt_mindex(oracle::adjacent_find(&input, EqTuple))
        );
        prop_assert_eq!(
            massively::find_first_of(&exec, lazify(gpu_input.slice(..)), lazify(gpu_other.slice(..)), EqTuple).unwrap(),
            opt_mindex(oracle::find_first_of(&input, &other, EqTuple))
        );
    }};
    (equal, $input:expr, $zip:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_input = make_zip!(&exec, &input, $zip, ($($ty),+));
        prop_assert_eq!(
            massively::equal(&exec, lazify(gpu_input.slice(..)), lazify(gpu_input.slice(..)), EqTuple).unwrap(),
            oracle::equal(&input, &input, EqTuple)
        );
    }};
    (mismatch, $input:expr, $zip:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let mut other = input.clone();
        other.reverse();
        let gpu_input = make_zip!(&exec, &input, $zip, ($($ty),+));
        let gpu_other = make_zip!(&exec, &other, $zip, ($($ty),+));
        prop_assert_eq!(
            massively::mismatch(&exec, lazify(gpu_input.slice(..)), lazify(gpu_other.slice(..)), EqTuple).unwrap(),
            opt_mindex(oracle::mismatch(&input, &other, EqTuple))
        );
    }};
    (adjacent_find, $input:expr, $zip:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_input = make_zip!(&exec, &input, $zip, ($($ty),+));
        prop_assert_eq!(
            massively::adjacent_find(&exec, lazify(gpu_input.slice(..)), EqTuple).unwrap(),
            opt_mindex(oracle::adjacent_find(&input, EqTuple))
        );
    }};
    (find_first_of, $input:expr, $zip:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let mut other = input.clone();
        other.reverse();
        let gpu_input = make_zip!(&exec, &input, $zip, ($($ty),+));
        let gpu_other = make_zip!(&exec, &other, $zip, ($($ty),+));
        prop_assert_eq!(
            massively::find_first_of(&exec, lazify(gpu_input.slice(..)), lazify(gpu_other.slice(..)), EqTuple).unwrap(),
            opt_mindex(oracle::find_first_of(&input, &other, EqTuple))
        );
    }};
    (mutating, $input:expr, $zip:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let stencil = stencil_for!(input.len());
        let replacement = $init;
        let gpu_stencil = exec.to_device(&stencil).unwrap();
        let gpu_input = make_zip!(&exec, &input, $zip, ($($ty),+));

        let gpu_output = make_zip!(&exec, &input, $zip, ($($ty),+));
        massively::fill(&exec, replacement, gpu_output.slice_mut(..)).unwrap();
        let mut host = input.clone();
        oracle::fill(replacement, &mut host);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $zip), host);

        let gpu_output = make_zip!(&exec, &input, $zip, ($($ty),+));
        massively::replace_where(
            &exec,
            replacement,
            lazify(massively::lazy::transform(gpu_stencil.slice(..), U32Flag)),
            gpu_output.slice_mut(..),
        )
        .unwrap();
        let mut host = input.clone();
        oracle::replace_where(replacement, &stencil, &mut host);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $zip), host);

        let gpu_output = make_zip!(&exec, &input, $zip, ($($ty),+));
        massively::transform_where(
            &exec,
            lazify(gpu_input.slice(..)),
            Identity,
            lazify(massively::lazy::transform(gpu_stencil.slice(..), U32Flag)),
            gpu_output.slice_mut(..),
        )
        .unwrap();
        let mut host = input.clone();
        oracle::transform_where(&input, Identity, &stencil, &mut host);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $zip), host);
    }};
    (fill, $input:expr, $zip:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_output = make_zip!(&exec, &input, $zip, ($($ty),+));
        massively::fill(&exec, $init, gpu_output.slice_mut(..)).unwrap();
        let mut host = input.clone();
        oracle::fill($init, &mut host);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $zip), host);
    }};
    (replace_where, $input:expr, $zip:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let stencil = stencil_for!(input.len());
        let gpu_stencil = exec.to_device(&stencil).unwrap();
        let gpu_output = make_zip!(&exec, &input, $zip, ($($ty),+));
        massively::replace_where(
            &exec,
            $init,
            lazify(massively::lazy::transform(gpu_stencil.slice(..), U32Flag)),
            gpu_output.slice_mut(..),
        )
        .unwrap();
        let mut host = input.clone();
        oracle::replace_where($init, &stencil, &mut host);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $zip), host);
    }};
    (transform_where, $input:expr, $zip:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let stencil = stencil_for!(input.len());
        let gpu_stencil = exec.to_device(&stencil).unwrap();
        let gpu_input = make_zip!(&exec, &input, $zip, ($($ty),+));
        let gpu_output = make_zip!(&exec, &input, $zip, ($($ty),+));
        massively::transform_where(
            &exec,
            lazify(gpu_input.slice(..)),
            Identity,
            lazify(massively::lazy::transform(gpu_stencil.slice(..), U32Flag)),
            gpu_output.slice_mut(..),
        )
        .unwrap();
        let mut host = input.clone();
        oracle::transform_where(&input, Identity, &stencil, &mut host);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $zip), host);
    }};
    (ordering, $input:expr, $zip:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let sorted = oracle::sort(&input, LessTuple);
        let mut other = sorted.clone();
        other.reverse();
        let right = oracle::sort(&other, LessTuple);
        let gpu_input = make_zip!(&exec, &input, $zip, ($($ty),+));
        let gpu_sorted = make_zip!(&exec, &sorted, $zip, ($($ty),+));
        let gpu_right = make_zip!(&exec, &right, $zip, ($($ty),+));

        let gpu_output = make_zip!(&exec, &input, $zip, ($($ty),+));
        massively::sort(&exec, lazify(gpu_input.slice(..)), LessTuple, gpu_output.slice_mut(..)).unwrap();
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $zip), sorted.clone());
        let merge_init = sorted.iter().chain(right.iter()).copied().collect::<Vec<_>>();
        let gpu_output = make_zip!(&exec, &merge_init, $zip, ($($ty),+));
        massively::merge(&exec, lazify(gpu_sorted.slice(..)), lazify(gpu_right.slice(..)), LessTuple, gpu_output.slice_mut(..)).unwrap();
        prop_assert_eq!(
            cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $zip),
            oracle::merge(&sorted, &right, LessTuple)
        );

        prop_assert_eq!(
            massively::is_sorted(&exec, lazify(gpu_sorted.slice(..)), LessTuple).unwrap(),
            oracle::is_sorted(&sorted, LessTuple)
        );
        prop_assert_eq!(
            massively::is_sorted_until(&exec, lazify(gpu_input.slice(..)), LessTuple).unwrap(),
            mindex(oracle::is_sorted_until(&input, LessTuple))
        );
        prop_assert_eq!(
            massively::lexicographical_compare(&exec, lazify(gpu_input.slice(..)), lazify(gpu_right.slice(..)), LessTuple).unwrap(),
            oracle::lexicographical_compare(&input, &right, LessTuple)
        );
        prop_assert_eq!(
            massively::min_element(&exec, lazify(gpu_input.slice(..)), LessTuple).unwrap(),
            opt_mindex(oracle::min_element(&input, LessTuple))
        );
        prop_assert_eq!(
            massively::max_element(&exec, lazify(gpu_input.slice(..)), LessTuple).unwrap(),
            opt_mindex(oracle::max_element(&input, LessTuple))
        );
        prop_assert_eq!(
            massively::minmax_element(&exec, lazify(gpu_input.slice(..)), LessTuple).unwrap(),
            opt_pair_mindex(oracle::minmax_element(&input, LessTuple))
        );

        let gpu_bounds = exec.to_device(&vec![0_u32; right.len()]).unwrap();
        massively::lower_bound(&exec, lazify(gpu_sorted.slice(..)), lazify(gpu_right.slice(..)), LessTuple, gpu_bounds.slice_mut(..)).unwrap();
        prop_assert_eq!(exec.to_host(&gpu_bounds).unwrap(), oracle::lower_bound(&sorted, &right, LessTuple));
        let gpu_bounds = exec.to_device(&vec![0_u32; right.len()]).unwrap();
        massively::upper_bound(&exec, lazify(gpu_sorted.slice(..)), lazify(gpu_right.slice(..)), LessTuple, gpu_bounds.slice_mut(..)).unwrap();
        prop_assert_eq!(exec.to_host(&gpu_bounds).unwrap(), oracle::upper_bound(&sorted, &right, LessTuple));

        let gpu_output = make_zip!(&exec, &sorted, $zip, ($($ty),+));
        let len = massively::unique(&exec, lazify(gpu_sorted.slice(..)), EqTuple, gpu_output.slice_mut(..)).unwrap();
        prop_assert_eq!(
            cols_to_aos!(exec.to_host(&gpu_output.slice(..len)).unwrap(), $zip),
            oracle::unique(&sorted, EqTuple)
        );
        let set_init = sorted.iter().chain(right.iter()).copied().collect::<Vec<_>>();
        let gpu_output = make_zip!(&exec, &set_init, $zip, ($($ty),+));
        let len = massively::set_union(&exec, lazify(gpu_sorted.slice(..)), lazify(gpu_right.slice(..)), LessTuple, gpu_output.slice_mut(..)).unwrap();
        prop_assert_eq!(
            cols_to_aos!(exec.to_host(&gpu_output.slice(..len)).unwrap(), $zip),
            oracle::set_union(&sorted, &right, LessTuple)
        );
        let gpu_output = make_zip!(&exec, &set_init, $zip, ($($ty),+));
        let len = massively::set_intersection(&exec, lazify(gpu_sorted.slice(..)), lazify(gpu_right.slice(..)), LessTuple, gpu_output.slice_mut(..)).unwrap();
        prop_assert_eq!(
            cols_to_aos!(exec.to_host(&gpu_output.slice(..len)).unwrap(), $zip),
            oracle::set_intersection(&sorted, &right, LessTuple)
        );
        let gpu_output = make_zip!(&exec, &sorted, $zip, ($($ty),+));
        let len = massively::set_difference(&exec, lazify(gpu_sorted.slice(..)), lazify(gpu_right.slice(..)), LessTuple, gpu_output.slice_mut(..)).unwrap();
        prop_assert_eq!(
            cols_to_aos!(exec.to_host(&gpu_output.slice(..len)).unwrap(), $zip),
            oracle::set_difference(&sorted, &right, LessTuple)
        );
    }};
    (sort, $input:expr, $zip:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_input = make_zip!(&exec, &input, $zip, ($($ty),+));
        let gpu_output = make_zip!(&exec, &input, $zip, ($($ty),+));
        massively::sort(&exec, lazify(gpu_input.slice(..)), LessTuple, gpu_output.slice_mut(..)).unwrap();
        prop_assert_eq!(
            cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $zip),
            oracle::sort(&input, LessTuple)
        );
    }};
    (merge, $input:expr, $zip:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let sorted = oracle::sort(&input, LessTuple);
        let other = input
            .iter()
            .copied()
            .map(|value| value.offset_first(8192))
            .collect::<Vec<_>>();
        let right = oracle::sort(&other, LessTuple);
        let gpu_sorted = make_zip!(&exec, &sorted, $zip, ($($ty),+));
        let gpu_right = make_zip!(&exec, &right, $zip, ($($ty),+));
        let merge_init = sorted.iter().chain(right.iter()).copied().collect::<Vec<_>>();
        let gpu_output = make_zip!(&exec, &merge_init, $zip, ($($ty),+));
        massively::merge(&exec, lazify(gpu_sorted.slice(..)), lazify(gpu_right.slice(..)), LessTuple, gpu_output.slice_mut(..)).unwrap();
        prop_assert_eq!(
            cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $zip),
            oracle::merge(&sorted, &right, LessTuple)
        );
    }};
    (is_sorted, $input:expr, $zip:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let sorted = oracle::sort(&input, LessTuple);
        let gpu_sorted = make_zip!(&exec, &sorted, $zip, ($($ty),+));
        prop_assert_eq!(
            massively::is_sorted(&exec, lazify(gpu_sorted.slice(..)), LessTuple).unwrap(),
            oracle::is_sorted(&sorted, LessTuple)
        );
    }};
    (is_sorted_until, $input:expr, $zip:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_input = make_zip!(&exec, &input, $zip, ($($ty),+));
        prop_assert_eq!(
            massively::is_sorted_until(&exec, lazify(gpu_input.slice(..)), LessTuple).unwrap(),
            mindex(oracle::is_sorted_until(&input, LessTuple))
        );
    }};
    (lexicographical_compare, $input:expr, $zip:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let sorted = oracle::sort(&input, LessTuple);
        let mut other = sorted.clone();
        other.reverse();
        let right = oracle::sort(&other, LessTuple);
        let gpu_input = make_zip!(&exec, &input, $zip, ($($ty),+));
        let gpu_right = make_zip!(&exec, &right, $zip, ($($ty),+));
        prop_assert_eq!(
            massively::lexicographical_compare(&exec, lazify(gpu_input.slice(..)), lazify(gpu_right.slice(..)), LessTuple).unwrap(),
            oracle::lexicographical_compare(&input, &right, LessTuple)
        );
    }};
    (min_element, $input:expr, $zip:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_input = make_zip!(&exec, &input, $zip, ($($ty),+));
        prop_assert_eq!(
            massively::min_element(&exec, lazify(gpu_input.slice(..)), LessTuple).unwrap(),
            opt_mindex(oracle::min_element(&input, LessTuple))
        );
    }};
    (max_element, $input:expr, $zip:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_input = make_zip!(&exec, &input, $zip, ($($ty),+));
        prop_assert_eq!(
            massively::max_element(&exec, lazify(gpu_input.slice(..)), LessTuple).unwrap(),
            opt_mindex(oracle::max_element(&input, LessTuple))
        );
    }};
    (minmax_element, $input:expr, $zip:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_input = make_zip!(&exec, &input, $zip, ($($ty),+));
        prop_assert_eq!(
            massively::minmax_element(&exec, lazify(gpu_input.slice(..)), LessTuple).unwrap(),
            opt_pair_mindex(oracle::minmax_element(&input, LessTuple))
        );
    }};
    (lower_bound, $input:expr, $zip:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let sorted = oracle::sort(&input, LessTuple);
        let mut other = sorted.clone();
        other.reverse();
        let right = oracle::sort(&other, LessTuple);
        let gpu_sorted = make_zip!(&exec, &sorted, $zip, ($($ty),+));
        let gpu_right = make_zip!(&exec, &right, $zip, ($($ty),+));
        let gpu_bounds = exec.to_device(&vec![0_u32; right.len()]).unwrap();
        massively::lower_bound(&exec, lazify(gpu_sorted.slice(..)), lazify(gpu_right.slice(..)), LessTuple, gpu_bounds.slice_mut(..)).unwrap();
        prop_assert_eq!(exec.to_host(&gpu_bounds).unwrap(), oracle::lower_bound(&sorted, &right, LessTuple));
    }};
    (upper_bound, $input:expr, $zip:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let sorted = oracle::sort(&input, LessTuple);
        let mut other = sorted.clone();
        other.reverse();
        let right = oracle::sort(&other, LessTuple);
        let gpu_sorted = make_zip!(&exec, &sorted, $zip, ($($ty),+));
        let gpu_right = make_zip!(&exec, &right, $zip, ($($ty),+));
        let gpu_bounds = exec.to_device(&vec![0_u32; right.len()]).unwrap();
        massively::upper_bound(&exec, lazify(gpu_sorted.slice(..)), lazify(gpu_right.slice(..)), LessTuple, gpu_bounds.slice_mut(..)).unwrap();
        prop_assert_eq!(exec.to_host(&gpu_bounds).unwrap(), oracle::upper_bound(&sorted, &right, LessTuple));
    }};
    (unique, $input:expr, $zip:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let sorted = oracle::sort(&input, LessTuple);
        let gpu_sorted = make_zip!(&exec, &sorted, $zip, ($($ty),+));
        let gpu_output = make_zip!(&exec, &sorted, $zip, ($($ty),+));
        let len = massively::unique(&exec, lazify(gpu_sorted.slice(..)), EqTuple, gpu_output.slice_mut(..)).unwrap();
        prop_assert_eq!(
            cols_to_aos!(exec.to_host(&gpu_output.slice(..len)).unwrap(), $zip),
            oracle::unique(&sorted, EqTuple)
        );
    }};
    (set_union, $input:expr, $zip:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let sorted = oracle::sort(&input, LessTuple);
        let mut other = sorted.clone();
        other.reverse();
        let right = oracle::sort(&other, LessTuple);
        let gpu_sorted = make_zip!(&exec, &sorted, $zip, ($($ty),+));
        let gpu_right = make_zip!(&exec, &right, $zip, ($($ty),+));
        let set_init = sorted.iter().chain(right.iter()).copied().collect::<Vec<_>>();
        let gpu_output = make_zip!(&exec, &set_init, $zip, ($($ty),+));
        let len = massively::set_union(&exec, lazify(gpu_sorted.slice(..)), lazify(gpu_right.slice(..)), LessTuple, gpu_output.slice_mut(..)).unwrap();
        prop_assert_eq!(
            cols_to_aos!(exec.to_host(&gpu_output.slice(..len)).unwrap(), $zip),
            oracle::set_union(&sorted, &right, LessTuple)
        );
    }};
    (set_intersection, $input:expr, $zip:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let sorted = oracle::sort(&input, LessTuple);
        let mut other = sorted.clone();
        other.reverse();
        let right = oracle::sort(&other, LessTuple);
        let gpu_sorted = make_zip!(&exec, &sorted, $zip, ($($ty),+));
        let gpu_right = make_zip!(&exec, &right, $zip, ($($ty),+));
        let set_init = sorted.iter().chain(right.iter()).copied().collect::<Vec<_>>();
        let gpu_output = make_zip!(&exec, &set_init, $zip, ($($ty),+));
        let len = massively::set_intersection(&exec, lazify(gpu_sorted.slice(..)), lazify(gpu_right.slice(..)), LessTuple, gpu_output.slice_mut(..)).unwrap();
        prop_assert_eq!(
            cols_to_aos!(exec.to_host(&gpu_output.slice(..len)).unwrap(), $zip),
            oracle::set_intersection(&sorted, &right, LessTuple)
        );
    }};
    (set_difference, $input:expr, $zip:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let sorted = oracle::sort(&input, LessTuple);
        let mut other = sorted.clone();
        other.reverse();
        let right = oracle::sort(&other, LessTuple);
        let gpu_sorted = make_zip!(&exec, &sorted, $zip, ($($ty),+));
        let gpu_right = make_zip!(&exec, &right, $zip, ($($ty),+));
        let gpu_output = make_zip!(&exec, &sorted, $zip, ($($ty),+));
        let len = massively::set_difference(&exec, lazify(gpu_sorted.slice(..)), lazify(gpu_right.slice(..)), LessTuple, gpu_output.slice_mut(..)).unwrap();
        prop_assert_eq!(
            cols_to_aos!(exec.to_host(&gpu_output.slice(..len)).unwrap(), $zip),
            oracle::set_difference(&sorted, &right, LessTuple)
        );
    }};
}

macro_rules! define_value_arity_module {
    (@unignored $module:ident, $case:ident) => {
        mod $module {
            use super::*;

            proptest! {
                #![proptest_config(ProptestConfig::with_cases(CASES))]

                #[test]
                fn arity_1(input in prop::collection::vec((0_u32..4096).prop_map(|v| (v,)), 0..MAX_LEN)) {
                    value_case!($case, input, Zip1, (u32), (0_u32,));
                }

                #[test]
                fn arity_2(input in prop::collection::vec((0_u32..4096, 0_u32..4096), 0..MAX_LEN)) {
                    value_case!($case, input, Zip2, (u32, u32), (0_u32, 0_u32));
                }

                #[test]
                fn arity_3(input in prop::collection::vec((0_u32..4096, 0_u32..4096, 0_u32..4096), 0..MAX_LEN)) {
                    value_case!($case, input, Zip3, (u32, u32, u32), (0_u32, 0_u32, 0_u32));
                }

                #[test]
                fn arity_4(input in prop::collection::vec((0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096), 0..MAX_LEN)) {
                    value_case!($case, input, Zip4, (u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32));
                }

                #[test]
                fn arity_5(input in prop::collection::vec((0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096), 0..MAX_LEN)) {
                    value_case!($case, input, Zip5, (u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32));
                }

                #[test]
                fn arity_6(input in prop::collection::vec((0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096), 0..MAX_LEN)) {
                    value_case!($case, input, Zip6, (u32, u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32));
                }

                #[test]
                fn arity_7(input in prop::collection::vec((0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096), 0..MAX_LEN)) {
                    value_case!($case, input, Zip7, (u32, u32, u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32));
                }
            }
        }
    };
}

define_value_arity_module!(@unignored reduce_arity, reduce);
define_value_arity_module!(@unignored inclusive_scan_arity, inclusive_scan);
define_value_arity_module!(@unignored exclusive_scan_arity, exclusive_scan);
define_value_arity_module!(@unignored adjacent_difference_arity, adjacent_difference);
define_value_arity_module!(@unignored copy_where_arity, copy_where);
define_value_arity_module!(@unignored remove_where_arity, remove_where);
define_value_arity_module!(@unignored reverse_arity, reverse);
define_value_arity_module!(@unignored count_if_arity, count_if);
define_value_arity_module!(@unignored all_of_arity, all_of);
define_value_arity_module!(@unignored any_of_arity, any_of);
define_value_arity_module!(@unignored none_of_arity, none_of);
define_value_arity_module!(@unignored find_if_arity, find_if);
define_value_arity_module!(@unignored is_partitioned_arity, is_partitioned);
define_value_arity_module!(@unignored partition_arity, partition);
define_value_arity_module!(@unignored permute_arity, permute);
define_value_arity_module!(@unignored gather_arity, gather);
define_value_arity_module!(@unignored gather_where_arity, gather_where);
define_value_arity_module!(@unignored scatter_arity, scatter);
define_value_arity_module!(@unignored scatter_where_arity, scatter_where);
define_value_arity_module!(@unignored equal_arity, equal);
define_value_arity_module!(@unignored mismatch_arity, mismatch);
define_value_arity_module!(@unignored adjacent_find_arity, adjacent_find);
define_value_arity_module!(@unignored find_first_of_arity, find_first_of);
define_value_arity_module!(@unignored fill_arity, fill);
define_value_arity_module!(@unignored replace_where_arity, replace_where);

macro_rules! define_ordering_arity_module {
    ($module:ident, $case:ident) => {
        mod $module {
            use super::*;

            proptest! {
                #![proptest_config(ProptestConfig::with_cases(CASES))]

                #[test]
                fn arity_1(input in prop::collection::vec((0_u32..4096).prop_map(|v| (v,)), 0..MAX_LEN).prop_filter("unique first column", |input| unique_by(input, |row| row.0))) {
                    value_case!($case, input, Zip1, (u32), (0_u32,));
                }

                #[test]
                fn arity_2(input in prop::collection::vec((0_u32..4096, 0_u32..4096), 0..MAX_LEN).prop_filter("unique first column", |input| unique_by(input, |row| row.0))) {
                    value_case!($case, input, Zip2, (u32, u32), (0_u32, 0_u32));
                }

                #[test]
                fn arity_3(input in prop::collection::vec((0_u32..4096, 0_u32..4096, 0_u32..4096), 0..MAX_LEN).prop_filter("unique first column", |input| unique_by(input, |row| row.0))) {
                    value_case!($case, input, Zip3, (u32, u32, u32), (0_u32, 0_u32, 0_u32));
                }

                #[test]
                fn arity_4(input in prop::collection::vec((0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096), 0..MAX_LEN).prop_filter("unique first column", |input| unique_by(input, |row| row.0))) {
                    value_case!($case, input, Zip4, (u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32));
                }

                #[test]
                fn arity_5(input in prop::collection::vec((0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096), 0..MAX_LEN).prop_filter("unique first column", |input| unique_by(input, |row| row.0))) {
                    value_case!($case, input, Zip5, (u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32));
                }

                #[test]
                fn arity_6(input in prop::collection::vec((0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096), 0..MAX_LEN).prop_filter("unique first column", |input| unique_by(input, |row| row.0))) {
                    value_case!($case, input, Zip6, (u32, u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32));
                }

                #[test]
                fn arity_7(input in prop::collection::vec((0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096), 0..MAX_LEN).prop_filter("unique first column", |input| unique_by(input, |row| row.0))) {
                    value_case!($case, input, Zip7, (u32, u32, u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32));
                }
            }
        }
    };
}

define_ordering_arity_module!(sort_arity, sort);
define_ordering_arity_module!(merge_arity, merge);
define_ordering_arity_module!(is_sorted_arity, is_sorted);
define_ordering_arity_module!(is_sorted_until_arity, is_sorted_until);
define_ordering_arity_module!(lexicographical_compare_arity, lexicographical_compare);
define_ordering_arity_module!(min_element_arity, min_element);
define_ordering_arity_module!(max_element_arity, max_element);
define_ordering_arity_module!(minmax_element_arity, minmax_element);
define_ordering_arity_module!(lower_bound_arity, lower_bound);
define_ordering_arity_module!(upper_bound_arity, upper_bound);
define_ordering_arity_module!(unique_arity, unique);
define_ordering_arity_module!(set_union_arity, set_union);
define_ordering_arity_module!(set_intersection_arity, set_intersection);
define_ordering_arity_module!(set_difference_arity, set_difference);

macro_rules! define_unary_arity_product_module {
    ($module:ident, $case:ident) => {
        mod $module {
            use super::*;

            proptest! {
                #![proptest_config(ProptestConfig::with_cases(CASES))]

                #[test]
                fn input_1_output_1(input in prop::collection::vec(any::<u32>().prop_map(|v| (v,)), 0..MAX_LEN)) {
                    $case!(input, Zip1, (u32), Zip1, (u32), (0_u32,), ArityTupleToTuple1);
                }

                #[test]
                fn input_1_output_2(input in prop::collection::vec(any::<u32>().prop_map(|v| (v,)), 0..MAX_LEN)) {
                    $case!(input, Zip1, (u32), Zip2, (u32, u32), (0_u32, 0_u32), ArityTupleToTuple2);
                }

                #[test]
                fn input_1_output_3(input in prop::collection::vec(any::<u32>().prop_map(|v| (v,)), 0..MAX_LEN)) {
                    $case!(input, Zip1, (u32), Zip3, (u32, u32, u32), (0_u32, 0_u32, 0_u32), ArityTupleToTuple3);
                }

                #[test]
                fn input_1_output_4(input in prop::collection::vec(any::<u32>().prop_map(|v| (v,)), 0..MAX_LEN)) {
                    $case!(input, Zip1, (u32), Zip4, (u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32), ArityTupleToTuple4);
                }

                #[test]
                fn input_1_output_5(input in prop::collection::vec(any::<u32>().prop_map(|v| (v,)), 0..MAX_LEN)) {
                    $case!(input, Zip1, (u32), Zip5, (u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32), ArityTupleToTuple5);
                }

                #[test]
                fn input_1_output_6(input in prop::collection::vec(any::<u32>().prop_map(|v| (v,)), 0..MAX_LEN)) {
                    $case!(input, Zip1, (u32), Zip6, (u32, u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32), ArityTupleToTuple6);
                }

                #[test]
                fn input_1_output_7(input in prop::collection::vec(any::<u32>().prop_map(|v| (v,)), 0..MAX_LEN)) {
                    $case!(input, Zip1, (u32), Zip7, (u32, u32, u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32), ArityTupleToTuple7);
                }

                #[test]
                fn input_2_output_1(input in prop::collection::vec((any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, Zip2, (u32, u32), Zip1, (u32), (0_u32,), ArityTupleToTuple1);
                }

                #[test]
                fn input_2_output_2(input in prop::collection::vec((any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, Zip2, (u32, u32), Zip2, (u32, u32), (0_u32, 0_u32), ArityTupleToTuple2);
                }

                #[test]
                fn input_2_output_3(input in prop::collection::vec((any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, Zip2, (u32, u32), Zip3, (u32, u32, u32), (0_u32, 0_u32, 0_u32), ArityTupleToTuple3);
                }

                #[test]
                fn input_2_output_4(input in prop::collection::vec((any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, Zip2, (u32, u32), Zip4, (u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32), ArityTupleToTuple4);
                }

                #[test]
                fn input_2_output_5(input in prop::collection::vec((any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, Zip2, (u32, u32), Zip5, (u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32), ArityTupleToTuple5);
                }

                #[test]
                fn input_2_output_6(input in prop::collection::vec((any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, Zip2, (u32, u32), Zip6, (u32, u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32), ArityTupleToTuple6);
                }

                #[test]
                fn input_2_output_7(input in prop::collection::vec((any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, Zip2, (u32, u32), Zip7, (u32, u32, u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32), ArityTupleToTuple7);
                }

                #[test]
                fn input_3_output_1(input in prop::collection::vec((any::<u32>(), any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, Zip3, (u32, u32, u32), Zip1, (u32), (0_u32,), ArityTupleToTuple1);
                }

                #[test]
                fn input_3_output_2(input in prop::collection::vec((any::<u32>(), any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, Zip3, (u32, u32, u32), Zip2, (u32, u32), (0_u32, 0_u32), ArityTupleToTuple2);
                }

                #[test]
                fn input_3_output_3(input in prop::collection::vec((any::<u32>(), any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, Zip3, (u32, u32, u32), Zip3, (u32, u32, u32), (0_u32, 0_u32, 0_u32), ArityTupleToTuple3);
                }

                #[test]
                fn input_3_output_4(input in prop::collection::vec((any::<u32>(), any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, Zip3, (u32, u32, u32), Zip4, (u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32), ArityTupleToTuple4);
                }

                #[test]
                fn input_3_output_5(input in prop::collection::vec((any::<u32>(), any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, Zip3, (u32, u32, u32), Zip5, (u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32), ArityTupleToTuple5);
                }

                #[test]
                fn input_3_output_6(input in prop::collection::vec((any::<u32>(), any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, Zip3, (u32, u32, u32), Zip6, (u32, u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32), ArityTupleToTuple6);
                }

                #[test]
                fn input_3_output_7(input in prop::collection::vec((any::<u32>(), any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, Zip3, (u32, u32, u32), Zip7, (u32, u32, u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32), ArityTupleToTuple7);
                }

                #[test]
                fn input_4_output_1(input in prop::collection::vec((any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, Zip4, (u32, u32, u32, u32), Zip1, (u32), (0_u32,), ArityTupleToTuple1);
                }

                #[test]
                fn input_4_output_2(input in prop::collection::vec((any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, Zip4, (u32, u32, u32, u32), Zip2, (u32, u32), (0_u32, 0_u32), ArityTupleToTuple2);
                }

                #[test]
                fn input_4_output_3(input in prop::collection::vec((any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, Zip4, (u32, u32, u32, u32), Zip3, (u32, u32, u32), (0_u32, 0_u32, 0_u32), ArityTupleToTuple3);
                }

                #[test]
                fn input_4_output_4(input in prop::collection::vec((any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, Zip4, (u32, u32, u32, u32), Zip4, (u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32), ArityTupleToTuple4);
                }

                #[test]
                fn input_4_output_5(input in prop::collection::vec((any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, Zip4, (u32, u32, u32, u32), Zip5, (u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32), ArityTupleToTuple5);
                }

                #[test]
                fn input_4_output_6(input in prop::collection::vec((any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, Zip4, (u32, u32, u32, u32), Zip6, (u32, u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32), ArityTupleToTuple6);
                }

                #[test]
                fn input_4_output_7(input in prop::collection::vec((any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, Zip4, (u32, u32, u32, u32), Zip7, (u32, u32, u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32), ArityTupleToTuple7);
                }

                #[test]
                fn input_5_output_1(input in prop::collection::vec((any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, Zip5, (u32, u32, u32, u32, u32), Zip1, (u32), (0_u32,), ArityTupleToTuple1);
                }

                #[test]
                fn input_5_output_2(input in prop::collection::vec((any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, Zip5, (u32, u32, u32, u32, u32), Zip2, (u32, u32), (0_u32, 0_u32), ArityTupleToTuple2);
                }

                #[test]
                fn input_5_output_3(input in prop::collection::vec((any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, Zip5, (u32, u32, u32, u32, u32), Zip3, (u32, u32, u32), (0_u32, 0_u32, 0_u32), ArityTupleToTuple3);
                }

                #[test]
                fn input_5_output_4(input in prop::collection::vec((any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, Zip5, (u32, u32, u32, u32, u32), Zip4, (u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32), ArityTupleToTuple4);
                }

                #[test]
                fn input_5_output_5(input in prop::collection::vec((any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, Zip5, (u32, u32, u32, u32, u32), Zip5, (u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32), ArityTupleToTuple5);
                }

                #[test]
                fn input_5_output_6(input in prop::collection::vec((any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, Zip5, (u32, u32, u32, u32, u32), Zip6, (u32, u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32), ArityTupleToTuple6);
                }

                #[test]
                fn input_5_output_7(input in prop::collection::vec((any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, Zip5, (u32, u32, u32, u32, u32), Zip7, (u32, u32, u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32), ArityTupleToTuple7);
                }

                #[test]
                fn input_6_output_1(input in prop::collection::vec((any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, Zip6, (u32, u32, u32, u32, u32, u32), Zip1, (u32), (0_u32,), ArityTupleToTuple1);
                }

                #[test]
                fn input_6_output_2(input in prop::collection::vec((any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, Zip6, (u32, u32, u32, u32, u32, u32), Zip2, (u32, u32), (0_u32, 0_u32), ArityTupleToTuple2);
                }

                #[test]
                fn input_6_output_3(input in prop::collection::vec((any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, Zip6, (u32, u32, u32, u32, u32, u32), Zip3, (u32, u32, u32), (0_u32, 0_u32, 0_u32), ArityTupleToTuple3);
                }

                #[test]
                fn input_6_output_4(input in prop::collection::vec((any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, Zip6, (u32, u32, u32, u32, u32, u32), Zip4, (u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32), ArityTupleToTuple4);
                }

                #[test]
                fn input_6_output_5(input in prop::collection::vec((any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, Zip6, (u32, u32, u32, u32, u32, u32), Zip5, (u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32), ArityTupleToTuple5);
                }

                #[test]
                fn input_6_output_6(input in prop::collection::vec((any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, Zip6, (u32, u32, u32, u32, u32, u32), Zip6, (u32, u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32), ArityTupleToTuple6);
                }

                #[test]
                fn input_6_output_7(input in prop::collection::vec((any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, Zip6, (u32, u32, u32, u32, u32, u32), Zip7, (u32, u32, u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32), ArityTupleToTuple7);
                }

                #[test]
                fn input_7_output_1(input in prop::collection::vec((any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, Zip7, (u32, u32, u32, u32, u32, u32, u32), Zip1, (u32), (0_u32,), ArityTupleToTuple1);
                }

                #[test]
                fn input_7_output_2(input in prop::collection::vec((any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, Zip7, (u32, u32, u32, u32, u32, u32, u32), Zip2, (u32, u32), (0_u32, 0_u32), ArityTupleToTuple2);
                }

                #[test]
                fn input_7_output_3(input in prop::collection::vec((any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, Zip7, (u32, u32, u32, u32, u32, u32, u32), Zip3, (u32, u32, u32), (0_u32, 0_u32, 0_u32), ArityTupleToTuple3);
                }

                #[test]
                fn input_7_output_4(input in prop::collection::vec((any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, Zip7, (u32, u32, u32, u32, u32, u32, u32), Zip4, (u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32), ArityTupleToTuple4);
                }

                #[test]
                fn input_7_output_5(input in prop::collection::vec((any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, Zip7, (u32, u32, u32, u32, u32, u32, u32), Zip5, (u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32), ArityTupleToTuple5);
                }

                #[test]
                fn input_7_output_6(input in prop::collection::vec((any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, Zip7, (u32, u32, u32, u32, u32, u32, u32), Zip6, (u32, u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32), ArityTupleToTuple6);
                }

                #[test]
                fn input_7_output_7(input in prop::collection::vec((any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, Zip7, (u32, u32, u32, u32, u32, u32, u32), Zip7, (u32, u32, u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32), ArityTupleToTuple7);
                }

            }
        }
    };
}

define_unary_arity_product_module!(map_arity, map_arity_case);
define_unary_arity_product_module!(transform_arity, transform_arity_case);
define_unary_arity_product_module!(transform_where_arity, transform_where_arity_case);

macro_rules! define_sort_by_key_arity_product_module {
    ($module:ident, $case:ident) => {
        mod $module {
            use super::*;

            proptest! {
                #![proptest_config(ProptestConfig::with_cases(CASES))]

                #[test]
                fn key_1_value_1(pairs in prop::collection::vec((any::<u32>().prop_map(|k| (k,)), any::<u32>().prop_map(|v| (v,))), 0..MAX_LEN)) {
                    sort_by_key_only_case!(pairs, Zip1, (u32), Zip1, (u32), LessU32);
                }

                #[test]
                fn key_1_value_2(pairs in prop::collection::vec((any::<u32>().prop_map(|k| (k,)), (any::<u32>(), any::<u32>())), 0..MAX_LEN)) {
                    sort_by_key_only_case!(pairs, Zip1, (u32), Zip2, (u32, u32), LessU32);
                }

                #[test]
                fn key_1_value_3(pairs in prop::collection::vec((any::<u32>().prop_map(|k| (k,)), (any::<u32>(), any::<u32>(), any::<u32>())), 0..MAX_LEN)) {
                    sort_by_key_only_case!(pairs, Zip1, (u32), Zip3, (u32, u32, u32), LessU32);
                }

                #[test]
                fn key_1_value_4(pairs in prop::collection::vec((any::<u32>().prop_map(|k| (k,)), (any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>())), 0..MAX_LEN)) {
                    sort_by_key_only_case!(pairs, Zip1, (u32), Zip4, (u32, u32, u32, u32), LessU32);
                }

                #[test]
                fn key_1_value_5(pairs in prop::collection::vec((any::<u32>().prop_map(|k| (k,)), (any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>())), 0..MAX_LEN)) {
                    sort_by_key_only_case!(pairs, Zip1, (u32), Zip5, (u32, u32, u32, u32, u32), LessU32);
                }

                #[test]
                fn key_1_value_6(pairs in prop::collection::vec((any::<u32>().prop_map(|k| (k,)), (any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>())), 0..MAX_LEN)) {
                    sort_by_key_only_case!(pairs, Zip1, (u32), Zip6, (u32, u32, u32, u32, u32, u32), LessU32);
                }

                #[test]
                fn key_1_value_7(pairs in prop::collection::vec((any::<u32>().prop_map(|k| (k,)), (any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>())), 0..MAX_LEN)) {
                    sort_by_key_only_case!(pairs, Zip1, (u32), Zip7, (u32, u32, u32, u32, u32, u32, u32), LessU32);
                }

                #[test]
                fn key_2_value_1(pairs in prop::collection::vec(((any::<u32>(), any::<u32>()), any::<u32>().prop_map(|v| (v,))), 0..MAX_LEN)) {
                    sort_by_key_only_case!(pairs, Zip2, (u32, u32), Zip1, (u32), Less2);
                }

                #[test]
                fn key_2_value_2(pairs in prop::collection::vec(((any::<u32>(), any::<u32>()), (any::<u32>(), any::<u32>())), 0..MAX_LEN)) {
                    sort_by_key_only_case!(pairs, Zip2, (u32, u32), Zip2, (u32, u32), Less2);
                }

                #[test]
                fn key_2_value_3(pairs in prop::collection::vec(((any::<u32>(), any::<u32>()), (any::<u32>(), any::<u32>(), any::<u32>())), 0..MAX_LEN)) {
                    sort_by_key_only_case!(pairs, Zip2, (u32, u32), Zip3, (u32, u32, u32), Less2);
                }

                #[test]
                fn key_2_value_4(pairs in prop::collection::vec(((any::<u32>(), any::<u32>()), (any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>())), 0..MAX_LEN)) {
                    sort_by_key_only_case!(pairs, Zip2, (u32, u32), Zip4, (u32, u32, u32, u32), Less2);
                }

                #[test]
                fn key_2_value_5(pairs in prop::collection::vec(((any::<u32>(), any::<u32>()), (any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>())), 0..MAX_LEN)) {
                    sort_by_key_only_case!(pairs, Zip2, (u32, u32), Zip5, (u32, u32, u32, u32, u32), Less2);
                }

                #[test]
                fn key_2_value_6(pairs in prop::collection::vec(((any::<u32>(), any::<u32>()), (any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>())), 0..MAX_LEN)) {
                    sort_by_key_only_case!(pairs, Zip2, (u32, u32), Zip6, (u32, u32, u32, u32, u32, u32), Less2);
                }

                #[test]
                fn key_2_value_7(pairs in prop::collection::vec(((any::<u32>(), any::<u32>()), (any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>())), 0..MAX_LEN)) {
                    sort_by_key_only_case!(pairs, Zip2, (u32, u32), Zip7, (u32, u32, u32, u32, u32, u32, u32), Less2);
                }

                #[test]
                fn key_3_value_1(pairs in prop::collection::vec(((any::<u32>(), any::<u32>(), any::<u32>()), any::<u32>().prop_map(|v| (v,))), 0..MAX_LEN)) {
                    sort_by_key_only_case!(pairs, Zip3, (u32, u32, u32), Zip1, (u32), Less3);
                }

                #[test]
                fn key_3_value_2(pairs in prop::collection::vec(((any::<u32>(), any::<u32>(), any::<u32>()), (any::<u32>(), any::<u32>())), 0..MAX_LEN)) {
                    sort_by_key_only_case!(pairs, Zip3, (u32, u32, u32), Zip2, (u32, u32), Less3);
                }

                #[test]
                fn key_3_value_3(pairs in prop::collection::vec(((any::<u32>(), any::<u32>(), any::<u32>()), (any::<u32>(), any::<u32>(), any::<u32>())), 0..MAX_LEN)) {
                    sort_by_key_only_case!(pairs, Zip3, (u32, u32, u32), Zip3, (u32, u32, u32), Less3);
                }

                #[test]
                fn key_3_value_4(pairs in prop::collection::vec(((any::<u32>(), any::<u32>(), any::<u32>()), (any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>())), 0..MAX_LEN)) {
                    sort_by_key_only_case!(pairs, Zip3, (u32, u32, u32), Zip4, (u32, u32, u32, u32), Less3);
                }

                #[test]
                fn key_3_value_5(pairs in prop::collection::vec(((any::<u32>(), any::<u32>(), any::<u32>()), (any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>())), 0..MAX_LEN)) {
                    sort_by_key_only_case!(pairs, Zip3, (u32, u32, u32), Zip5, (u32, u32, u32, u32, u32), Less3);
                }

                #[test]
                fn key_3_value_6(pairs in prop::collection::vec(((any::<u32>(), any::<u32>(), any::<u32>()), (any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>())), 0..MAX_LEN)) {
                    sort_by_key_only_case!(pairs, Zip3, (u32, u32, u32), Zip6, (u32, u32, u32, u32, u32, u32), Less3);
                }

                #[test]
                fn key_3_value_7(pairs in prop::collection::vec(((any::<u32>(), any::<u32>(), any::<u32>()), (any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>())), 0..MAX_LEN)) {
                    sort_by_key_only_case!(pairs, Zip3, (u32, u32, u32), Zip7, (u32, u32, u32, u32, u32, u32, u32), Less3);
                }

            }
        }
    };
}

macro_rules! define_scan_by_key_arity_product_module {
    ($module:ident, $case:ident) => {
        mod $module {
            use super::*;

            proptest! {
                #![proptest_config(ProptestConfig::with_cases(CASES))]

                #[test]
                fn key_1_value_1(pairs in prop::collection::vec(((0_u32..16).prop_map(|k| (k,)), (0_u32..4096).prop_map(|v| (v,))), 0..MAX_LEN)) {
                    scan_by_key_case!($case, pairs, Zip1, (u32), Zip1, (u32), (0_u32,));
                }

                #[test]
                fn key_1_value_2(pairs in prop::collection::vec(((0_u32..16).prop_map(|k| (k,)), (0_u32..4096, 0_u32..4096)), 0..MAX_LEN)) {
                    scan_by_key_case!($case, pairs, Zip1, (u32), Zip2, (u32, u32), (0_u32, 0_u32));
                }

                #[test]
                fn key_1_value_3(pairs in prop::collection::vec(((0_u32..16).prop_map(|k| (k,)), (0_u32..4096, 0_u32..4096, 0_u32..4096)), 0..MAX_LEN)) {
                    scan_by_key_case!($case, pairs, Zip1, (u32), Zip3, (u32, u32, u32), (0_u32, 0_u32, 0_u32));
                }

                #[test]
                fn key_1_value_4(pairs in prop::collection::vec(((0_u32..16).prop_map(|k| (k,)), (0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096)), 0..MAX_LEN)) {
                    scan_by_key_case!($case, pairs, Zip1, (u32), Zip4, (u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32));
                }

                #[test]
                fn key_1_value_5(pairs in prop::collection::vec(((0_u32..16).prop_map(|k| (k,)), (0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096)), 0..MAX_LEN)) {
                    scan_by_key_case!($case, pairs, Zip1, (u32), Zip5, (u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32));
                }

                #[test]
                fn key_1_value_6(pairs in prop::collection::vec(((0_u32..16).prop_map(|k| (k,)), (0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096)), 0..MAX_LEN)) {
                    scan_by_key_case!($case, pairs, Zip1, (u32), Zip6, (u32, u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32));
                }

                #[test]
                fn key_1_value_7(pairs in prop::collection::vec(((0_u32..16).prop_map(|k| (k,)), (0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096)), 0..MAX_LEN)) {
                    scan_by_key_case!($case, pairs, Zip1, (u32), Zip7, (u32, u32, u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32));
                }

                #[test]
                fn key_2_value_1(pairs in prop::collection::vec(((0_u32..16, 0_u32..16), (0_u32..4096).prop_map(|v| (v,))), 0..MAX_LEN)) {
                    scan_by_key_case!($case, pairs, Zip2, (u32, u32), Zip1, (u32), (0_u32,));
                }

                #[test]
                fn key_2_value_2(pairs in prop::collection::vec(((0_u32..16, 0_u32..16), (0_u32..4096, 0_u32..4096)), 0..MAX_LEN)) {
                    scan_by_key_case!($case, pairs, Zip2, (u32, u32), Zip2, (u32, u32), (0_u32, 0_u32));
                }

                #[test]
                fn key_2_value_3(pairs in prop::collection::vec(((0_u32..16, 0_u32..16), (0_u32..4096, 0_u32..4096, 0_u32..4096)), 0..MAX_LEN)) {
                    scan_by_key_case!($case, pairs, Zip2, (u32, u32), Zip3, (u32, u32, u32), (0_u32, 0_u32, 0_u32));
                }

                #[test]
                fn key_2_value_4(pairs in prop::collection::vec(((0_u32..16, 0_u32..16), (0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096)), 0..MAX_LEN)) {
                    scan_by_key_case!($case, pairs, Zip2, (u32, u32), Zip4, (u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32));
                }

                #[test]
                fn key_2_value_5(pairs in prop::collection::vec(((0_u32..16, 0_u32..16), (0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096)), 0..MAX_LEN)) {
                    scan_by_key_case!($case, pairs, Zip2, (u32, u32), Zip5, (u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32));
                }

                #[test]
                fn key_2_value_6(pairs in prop::collection::vec(((0_u32..16, 0_u32..16), (0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096)), 0..MAX_LEN)) {
                    scan_by_key_case!($case, pairs, Zip2, (u32, u32), Zip6, (u32, u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32));
                }

                #[test]
                fn key_2_value_7(pairs in prop::collection::vec(((0_u32..16, 0_u32..16), (0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096)), 0..MAX_LEN)) {
                    scan_by_key_case!($case, pairs, Zip2, (u32, u32), Zip7, (u32, u32, u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32));
                }

                #[test]
                fn key_3_value_1(pairs in prop::collection::vec(((0_u32..16, 0_u32..16, 0_u32..16), (0_u32..4096).prop_map(|v| (v,))), 0..MAX_LEN)) {
                    scan_by_key_case!($case, pairs, Zip3, (u32, u32, u32), Zip1, (u32), (0_u32,));
                }

                #[test]
                fn key_3_value_2(pairs in prop::collection::vec(((0_u32..16, 0_u32..16, 0_u32..16), (0_u32..4096, 0_u32..4096)), 0..MAX_LEN)) {
                    scan_by_key_case!($case, pairs, Zip3, (u32, u32, u32), Zip2, (u32, u32), (0_u32, 0_u32));
                }

                #[test]
                fn key_3_value_3(pairs in prop::collection::vec(((0_u32..16, 0_u32..16, 0_u32..16), (0_u32..4096, 0_u32..4096, 0_u32..4096)), 0..MAX_LEN)) {
                    scan_by_key_case!($case, pairs, Zip3, (u32, u32, u32), Zip3, (u32, u32, u32), (0_u32, 0_u32, 0_u32));
                }

                #[test]
                fn key_3_value_4(pairs in prop::collection::vec(((0_u32..16, 0_u32..16, 0_u32..16), (0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096)), 0..MAX_LEN)) {
                    scan_by_key_case!($case, pairs, Zip3, (u32, u32, u32), Zip4, (u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32));
                }

                #[test]
                fn key_3_value_5(pairs in prop::collection::vec(((0_u32..16, 0_u32..16, 0_u32..16), (0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096)), 0..MAX_LEN)) {
                    scan_by_key_case!($case, pairs, Zip3, (u32, u32, u32), Zip5, (u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32));
                }

                #[test]
                fn key_3_value_6(pairs in prop::collection::vec(((0_u32..16, 0_u32..16, 0_u32..16), (0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096)), 0..MAX_LEN)) {
                    scan_by_key_case!($case, pairs, Zip3, (u32, u32, u32), Zip6, (u32, u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32));
                }

                #[test]
                fn key_3_value_7(pairs in prop::collection::vec(((0_u32..16, 0_u32..16, 0_u32..16), (0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096)), 0..MAX_LEN)) {
                    scan_by_key_case!($case, pairs, Zip3, (u32, u32, u32), Zip7, (u32, u32, u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32));
                }

            }
        }
    };
}

macro_rules! define_merge_by_key_arity_product_module {
    ($module:ident, $case:ident) => {
        mod $module {
            use super::*;

            proptest! {
                #![proptest_config(ProptestConfig::with_cases(CASES))]

                #[test]
                fn key_1_value_1(pairs in prop::collection::vec(((0_u32..4096).prop_map(|k| (k,)), (0_u32..4096).prop_map(|v| (v,))), 0..MAX_LEN)) {
                    merge_by_key_case!(pairs, Zip1, (u32), Zip1, (u32), LessU32);
                }

                #[test]
                fn key_1_value_2(pairs in prop::collection::vec(((0_u32..4096).prop_map(|k| (k,)), (0_u32..4096, 0_u32..4096)), 0..MAX_LEN)) {
                    merge_by_key_case!(pairs, Zip1, (u32), Zip2, (u32, u32), LessU32);
                }

                #[test]
                fn key_1_value_3(pairs in prop::collection::vec(((0_u32..4096).prop_map(|k| (k,)), (0_u32..4096, 0_u32..4096, 0_u32..4096)), 0..MAX_LEN)) {
                    merge_by_key_case!(pairs, Zip1, (u32), Zip3, (u32, u32, u32), LessU32);
                }

                #[test]
                fn key_1_value_4(pairs in prop::collection::vec(((0_u32..4096).prop_map(|k| (k,)), (0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096)), 0..MAX_LEN)) {
                    merge_by_key_case!(pairs, Zip1, (u32), Zip4, (u32, u32, u32, u32), LessU32);
                }

                #[test]
                fn key_1_value_5(pairs in prop::collection::vec(((0_u32..4096).prop_map(|k| (k,)), (0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096)), 0..MAX_LEN)) {
                    merge_by_key_case!(pairs, Zip1, (u32), Zip5, (u32, u32, u32, u32, u32), LessU32);
                }

                #[test]
                fn key_1_value_6(pairs in prop::collection::vec(((0_u32..4096).prop_map(|k| (k,)), (0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096)), 0..MAX_LEN)) {
                    merge_by_key_case!(pairs, Zip1, (u32), Zip6, (u32, u32, u32, u32, u32, u32), LessU32);
                }

                #[test]
                fn key_1_value_7(pairs in prop::collection::vec(((0_u32..4096).prop_map(|k| (k,)), (0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096)), 0..MAX_LEN)) {
                    merge_by_key_case!(pairs, Zip1, (u32), Zip7, (u32, u32, u32, u32, u32, u32, u32), LessU32);
                }

                #[test]
                fn key_2_value_1(pairs in prop::collection::vec(((0_u32..4096, 0_u32..4096), (0_u32..4096).prop_map(|v| (v,))), 0..MAX_LEN)) {
                    merge_by_key_case!(pairs, Zip2, (u32, u32), Zip1, (u32), Less2);
                }

                #[test]
                fn key_2_value_2(pairs in prop::collection::vec(((0_u32..4096, 0_u32..4096), (0_u32..4096, 0_u32..4096)), 0..MAX_LEN)) {
                    merge_by_key_case!(pairs, Zip2, (u32, u32), Zip2, (u32, u32), Less2);
                }

                #[test]
                fn key_2_value_3(pairs in prop::collection::vec(((0_u32..4096, 0_u32..4096), (0_u32..4096, 0_u32..4096, 0_u32..4096)), 0..MAX_LEN)) {
                    merge_by_key_case!(pairs, Zip2, (u32, u32), Zip3, (u32, u32, u32), Less2);
                }

                #[test]
                fn key_2_value_4(pairs in prop::collection::vec(((0_u32..4096, 0_u32..4096), (0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096)), 0..MAX_LEN)) {
                    merge_by_key_case!(pairs, Zip2, (u32, u32), Zip4, (u32, u32, u32, u32), Less2);
                }

                #[test]
                fn key_2_value_5(pairs in prop::collection::vec(((0_u32..4096, 0_u32..4096), (0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096)), 0..MAX_LEN)) {
                    merge_by_key_case!(pairs, Zip2, (u32, u32), Zip5, (u32, u32, u32, u32, u32), Less2);
                }

                #[test]
                fn key_2_value_6(pairs in prop::collection::vec(((0_u32..4096, 0_u32..4096), (0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096)), 0..MAX_LEN)) {
                    merge_by_key_case!(pairs, Zip2, (u32, u32), Zip6, (u32, u32, u32, u32, u32, u32), Less2);
                }

                #[test]
                fn key_2_value_7(pairs in prop::collection::vec(((0_u32..4096, 0_u32..4096), (0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096)), 0..MAX_LEN)) {
                    merge_by_key_case!(pairs, Zip2, (u32, u32), Zip7, (u32, u32, u32, u32, u32, u32, u32), Less2);
                }

                #[test]
                fn key_3_value_1(pairs in prop::collection::vec(((0_u32..4096, 0_u32..4096, 0_u32..4096), (0_u32..4096).prop_map(|v| (v,))), 0..MAX_LEN)) {
                    merge_by_key_case!(pairs, Zip3, (u32, u32, u32), Zip1, (u32), Less3);
                }

                #[test]
                fn key_3_value_2(pairs in prop::collection::vec(((0_u32..4096, 0_u32..4096, 0_u32..4096), (0_u32..4096, 0_u32..4096)), 0..MAX_LEN)) {
                    merge_by_key_case!(pairs, Zip3, (u32, u32, u32), Zip2, (u32, u32), Less3);
                }

                #[test]
                fn key_3_value_3(pairs in prop::collection::vec(((0_u32..4096, 0_u32..4096, 0_u32..4096), (0_u32..4096, 0_u32..4096, 0_u32..4096)), 0..MAX_LEN)) {
                    merge_by_key_case!(pairs, Zip3, (u32, u32, u32), Zip3, (u32, u32, u32), Less3);
                }

                #[test]
                fn key_3_value_4(pairs in prop::collection::vec(((0_u32..4096, 0_u32..4096, 0_u32..4096), (0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096)), 0..MAX_LEN)) {
                    merge_by_key_case!(pairs, Zip3, (u32, u32, u32), Zip4, (u32, u32, u32, u32), Less3);
                }

                #[test]
                fn key_3_value_5(pairs in prop::collection::vec(((0_u32..4096, 0_u32..4096, 0_u32..4096), (0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096)), 0..MAX_LEN)) {
                    merge_by_key_case!(pairs, Zip3, (u32, u32, u32), Zip5, (u32, u32, u32, u32, u32), Less3);
                }

                #[test]
                fn key_3_value_6(pairs in prop::collection::vec(((0_u32..4096, 0_u32..4096, 0_u32..4096), (0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096)), 0..MAX_LEN)) {
                    merge_by_key_case!(pairs, Zip3, (u32, u32, u32), Zip6, (u32, u32, u32, u32, u32, u32), Less3);
                }

                #[test]
                fn key_3_value_7(pairs in prop::collection::vec(((0_u32..4096, 0_u32..4096, 0_u32..4096), (0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096)), 0..MAX_LEN)) {
                    merge_by_key_case!(pairs, Zip3, (u32, u32, u32), Zip7, (u32, u32, u32, u32, u32, u32, u32), Less3);
                }

            }
        }
    };
}

define_sort_by_key_arity_product_module!(sort_by_key_arity, sort_by_key);
define_scan_by_key_arity_product_module!(inclusive_scan_by_key_arity, inclusive_scan_by_key);
define_scan_by_key_arity_product_module!(exclusive_scan_by_key_arity, exclusive_scan_by_key);
define_scan_by_key_arity_product_module!(reduce_by_key_arity, reduce_by_key);
define_scan_by_key_arity_product_module!(unique_by_key_arity, unique_by_key);
define_merge_by_key_arity_product_module!(merge_by_key_arity, merge_by_key);
