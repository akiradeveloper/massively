use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::op as gpu_op;
use massively::{DeviceVec, Executor};
use oracle::op as host_op;
use proptest::prelude::*;
use std::sync::{Mutex, MutexGuard};

type ApiRuntime = WgpuRuntime;
type ApiExecutor = Executor<ApiRuntime>;

const CASES: u32 = 24;
const MAX_LEN: usize = 64;
static GPU_LOCK: Mutex<()> = Mutex::new(());

struct MaxTuple;
struct KeepTuple;
struct EqTuple;
struct LessTuple;
struct LessU32;
struct Less2;
struct Less3;
struct ArityTupleToScalar;
struct ArityScalarToTuple1;
struct ArityScalarToTuple2;
struct ArityScalarToTuple3;
struct ArityScalarToTuple4;
struct ArityScalarToTuple5;
struct ArityScalarToTuple6;
struct ArityScalarToTuple7;

macro_rules! impl_tuple_to_scalar {
    (($($ty:ty),+), $input:ident => $out:expr) => {
        #[cubecl::cube]
        impl gpu_op::UnaryOp<ApiRuntime, ($($ty,)+)> for ArityTupleToScalar {
            type Env = ();
            type Output = (u32,);

            fn apply(_env: (), $input: ($($ty,)+)) -> (u32,) {
                ($out,)
            }
        }

        impl host_op::UnaryOp<($($ty,)+)> for ArityTupleToScalar {
            type Env = ();
            type Output = (u32,);

            fn apply(_env: (), $input: ($($ty,)+)) -> (u32,) {
                ($out,)
            }
        }
    };
}

macro_rules! impl_scalar_to_tuple {
    ($op:ident, ($($ty:ty),+), $input:ident => ($($out:expr),+)) => {
        #[cubecl::cube]
        impl gpu_op::UnaryOp<ApiRuntime, (u32,)> for $op {
            type Env = ();
            type Output = ($($ty,)+);

            fn apply(_env: (), $input: (u32,)) -> ($($ty,)+) {
                ($($out,)+)
            }
        }

        impl host_op::UnaryOp<(u32,)> for $op {
            type Env = ();
            type Output = ($($ty,)+);

            fn apply(_env: (), $input: (u32,)) -> ($($ty,)+) {
                ($($out,)+)
            }
        }
    };
}

impl_tuple_to_scalar!((u32), input => input.0 ^ 0x5a5a_5a5a);
impl_tuple_to_scalar!((u32, u32), input => (input.0 ^ 0x5a5a_5a5a) ^ (input.1 << 1));
impl_tuple_to_scalar!(
    (u32, u32, u32),
    input =>
    (input.0 ^ 0x5a5a_5a5a) ^ (input.1 << 1) ^ (input.2 << 2)
);
impl_tuple_to_scalar!(
    (u32, u32, u32, u32),
    input =>
    (input.0 ^ 0x5a5a_5a5a) ^ (input.1 << 1) ^ (input.2 << 2) ^ (input.3 << 3)
);
impl_tuple_to_scalar!(
    (u32, u32, u32, u32, u32),
    input =>
    (input.0 ^ 0x5a5a_5a5a) ^ (input.1 << 1) ^ (input.2 << 2) ^ (input.3 << 3) ^ (input.4 << 4)
);
impl_tuple_to_scalar!(
    (u32, u32, u32, u32, u32, u32),
    input =>
    (input.0 ^ 0x5a5a_5a5a)
        ^ (input.1 << 1)
        ^ (input.2 << 2)
        ^ (input.3 << 3)
        ^ (input.4 << 4)
        ^ (input.5 << 5)
);
impl_tuple_to_scalar!(
    (u32, u32, u32, u32, u32, u32, u32),
    input =>
    (input.0 ^ 0x5a5a_5a5a)
        ^ (input.1 << 1)
        ^ (input.2 << 2)
        ^ (input.3 << 3)
        ^ (input.4 << 4)
        ^ (input.5 << 5)
        ^ (input.6 << 6)
);

impl_scalar_to_tuple!(ArityScalarToTuple1, (u32), input => (input.0 ^ 0x5a5a_5a5a));
impl_scalar_to_tuple!(
    ArityScalarToTuple2,
    (u32, u32),
    input => (input.0 ^ 0x5a5a_5a5a, (input.0 << 1) ^ 0xa5a5_a5a5)
);
impl_scalar_to_tuple!(
    ArityScalarToTuple3,
    (u32, u32, u32),
    input =>
    (
        input.0 ^ 0x5a5a_5a5a,
        (input.0 << 1) ^ 0xa5a5_a5a5,
        (input.0 >> 1) ^ 0x3c3c_3c3c
    )
);
impl_scalar_to_tuple!(
    ArityScalarToTuple4,
    (u32, u32, u32, u32),
    input =>
    (
        input.0 ^ 0x5a5a_5a5a,
        (input.0 << 1) ^ 0xa5a5_a5a5,
        (input.0 >> 1) ^ 0x3c3c_3c3c,
        (input.0 << 2) ^ 0xc3c3_c3c3
    )
);
impl_scalar_to_tuple!(
    ArityScalarToTuple5,
    (u32, u32, u32, u32, u32),
    input =>
    (
        input.0 ^ 0x5a5a_5a5a,
        (input.0 << 1) ^ 0xa5a5_a5a5,
        (input.0 >> 1) ^ 0x3c3c_3c3c,
        (input.0 << 2) ^ 0xc3c3_c3c3,
        (input.0 >> 2) ^ 0x0f0f_0f0f
    )
);
impl_scalar_to_tuple!(
    ArityScalarToTuple6,
    (u32, u32, u32, u32, u32, u32),
    input =>
    (
        input.0 ^ 0x5a5a_5a5a,
        (input.0 << 1) ^ 0xa5a5_a5a5,
        (input.0 >> 1) ^ 0x3c3c_3c3c,
        (input.0 << 2) ^ 0xc3c3_c3c3,
        (input.0 >> 2) ^ 0x0f0f_0f0f,
        (input.0 << 3) ^ 0xf0f0_f0f0
    )
);
impl_scalar_to_tuple!(
    ArityScalarToTuple7,
    (u32, u32, u32, u32, u32, u32, u32),
    input =>
    (
        input.0 ^ 0x5a5a_5a5a,
        (input.0 << 1) ^ 0xa5a5_a5a5,
        (input.0 >> 1) ^ 0x3c3c_3c3c,
        (input.0 << 2) ^ 0xc3c3_c3c3,
        (input.0 >> 2) ^ 0x0f0f_0f0f,
        (input.0 << 3) ^ 0xf0f0_f0f0,
        (input.0 >> 3) ^ 0x9696_9696
    )
);

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
            type Env = ();

            fn apply(_env: (), input: ($($ty,)+)) -> bool {
                input.0 > $zero
            }
        }

        impl host_op::PredicateOp<($($ty,)+)> for KeepTuple {
            type Env = ();

            fn apply(_env: (), input: ($($ty,)+)) -> bool {
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
    ($cols:expr, SoA1) => {{
        let (a,) = $cols;
        a.into_iter().map(|a| (a,)).collect::<Vec<_>>()
    }};
    ($cols:expr, SoA2) => {{
        let (a, b) = $cols;
        a.into_iter().zip(b).collect::<Vec<_>>()
    }};
    ($cols:expr, SoA3) => {{
        let (a, b, c) = $cols;
        a.into_iter()
            .zip(b)
            .zip(c)
            .map(|((a, b), c)| (a, b, c))
            .collect::<Vec<_>>()
    }};
    ($cols:expr, SoA4) => {{
        let (a, b, c, d) = $cols;
        a.into_iter()
            .zip(b)
            .zip(c)
            .zip(d)
            .map(|(((a, b), c), d)| (a, b, c, d))
            .collect::<Vec<_>>()
    }};
    ($cols:expr, SoA5) => {{
        let (a, b, c, d, e) = $cols;
        a.into_iter()
            .zip(b)
            .zip(c)
            .zip(d)
            .zip(e)
            .map(|((((a, b), c), d), e)| (a, b, c, d, e))
            .collect::<Vec<_>>()
    }};
    ($cols:expr, SoA6) => {{
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
    ($cols:expr, SoA7) => {{
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

macro_rules! make_soa {
    ($exec:expr, $input:expr, SoA1, ($a:ty)) => {{
        let c0: Vec<$a> = $input.iter().map(|row| row.0).collect();
        massively::SoA1($exec.to_device(&c0).unwrap())
    }};
    ($exec:expr, $input:expr, SoA2, ($a:ty, $b:ty)) => {{
        let c0: Vec<$a> = $input.iter().map(|row| row.0).collect();
        let c1: Vec<$b> = $input.iter().map(|row| row.1).collect();
        massively::SoA2($exec.to_device(&c0).unwrap(), $exec.to_device(&c1).unwrap())
    }};
    ($exec:expr, $input:expr, SoA3, ($a:ty, $b:ty, $c:ty)) => {{
        let c0: Vec<$a> = $input.iter().map(|row| row.0).collect();
        let c1: Vec<$b> = $input.iter().map(|row| row.1).collect();
        let c2: Vec<$c> = $input.iter().map(|row| row.2).collect();
        massively::SoA3(
            $exec.to_device(&c0).unwrap(),
            $exec.to_device(&c1).unwrap(),
            $exec.to_device(&c2).unwrap(),
        )
    }};
    ($exec:expr, $input:expr, SoA4, ($a:ty, $b:ty, $c:ty, $d:ty)) => {{
        let c0: Vec<$a> = $input.iter().map(|row| row.0).collect();
        let c1: Vec<$b> = $input.iter().map(|row| row.1).collect();
        let c2: Vec<$c> = $input.iter().map(|row| row.2).collect();
        let c3: Vec<$d> = $input.iter().map(|row| row.3).collect();
        massively::SoA4(
            $exec.to_device(&c0).unwrap(),
            $exec.to_device(&c1).unwrap(),
            $exec.to_device(&c2).unwrap(),
            $exec.to_device(&c3).unwrap(),
        )
    }};
    ($exec:expr, $input:expr, SoA5, ($a:ty, $b:ty, $c:ty, $d:ty, $e:ty)) => {{
        let c0: Vec<$a> = $input.iter().map(|row| row.0).collect();
        let c1: Vec<$b> = $input.iter().map(|row| row.1).collect();
        let c2: Vec<$c> = $input.iter().map(|row| row.2).collect();
        let c3: Vec<$d> = $input.iter().map(|row| row.3).collect();
        let c4: Vec<$e> = $input.iter().map(|row| row.4).collect();
        massively::SoA5(
            $exec.to_device(&c0).unwrap(),
            $exec.to_device(&c1).unwrap(),
            $exec.to_device(&c2).unwrap(),
            $exec.to_device(&c3).unwrap(),
            $exec.to_device(&c4).unwrap(),
        )
    }};
    ($exec:expr, $input:expr, SoA6, ($a:ty, $b:ty, $c:ty, $d:ty, $e:ty, $f:ty)) => {{
        let c0: Vec<$a> = $input.iter().map(|row| row.0).collect();
        let c1: Vec<$b> = $input.iter().map(|row| row.1).collect();
        let c2: Vec<$c> = $input.iter().map(|row| row.2).collect();
        let c3: Vec<$d> = $input.iter().map(|row| row.3).collect();
        let c4: Vec<$e> = $input.iter().map(|row| row.4).collect();
        let c5: Vec<$f> = $input.iter().map(|row| row.5).collect();
        massively::SoA6(
            $exec.to_device(&c0).unwrap(),
            $exec.to_device(&c1).unwrap(),
            $exec.to_device(&c2).unwrap(),
            $exec.to_device(&c3).unwrap(),
            $exec.to_device(&c4).unwrap(),
            $exec.to_device(&c5).unwrap(),
        )
    }};
    ($exec:expr, $input:expr, SoA7, ($a:ty, $b:ty, $c:ty, $d:ty, $e:ty, $f:ty, $g:ty)) => {{
        let c0: Vec<$a> = $input.iter().map(|row| row.0).collect();
        let c1: Vec<$b> = $input.iter().map(|row| row.1).collect();
        let c2: Vec<$c> = $input.iter().map(|row| row.2).collect();
        let c3: Vec<$d> = $input.iter().map(|row| row.3).collect();
        let c4: Vec<$e> = $input.iter().map(|row| row.4).collect();
        let c5: Vec<$f> = $input.iter().map(|row| row.5).collect();
        let c6: Vec<$g> = $input.iter().map(|row| row.6).collect();
        massively::SoA7(
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

macro_rules! owned_soa_type {
    (SoA1, ($a:ty)) => {
        massively::SoA1<DeviceVec<ApiRuntime, $a>>
    };
    (SoA2, ($a:ty, $b:ty)) => {
        massively::SoA2<DeviceVec<ApiRuntime, $a>, DeviceVec<ApiRuntime, $b>>
    };
    (SoA3, ($a:ty, $b:ty, $c:ty)) => {
        massively::SoA3<DeviceVec<ApiRuntime, $a>, DeviceVec<ApiRuntime, $b>, DeviceVec<ApiRuntime, $c>>
    };
    (SoA4, ($a:ty, $b:ty, $c:ty, $d:ty)) => {
        massively::SoA4<
            DeviceVec<ApiRuntime, $a>,
            DeviceVec<ApiRuntime, $b>,
            DeviceVec<ApiRuntime, $c>,
            DeviceVec<ApiRuntime, $d>,
        >
    };
    (SoA5, ($a:ty, $b:ty, $c:ty, $d:ty, $e:ty)) => {
        massively::SoA5<
            DeviceVec<ApiRuntime, $a>,
            DeviceVec<ApiRuntime, $b>,
            DeviceVec<ApiRuntime, $c>,
            DeviceVec<ApiRuntime, $d>,
            DeviceVec<ApiRuntime, $e>,
        >
    };
    (SoA6, ($a:ty, $b:ty, $c:ty, $d:ty, $e:ty, $f:ty)) => {
        massively::SoA6<
            DeviceVec<ApiRuntime, $a>,
            DeviceVec<ApiRuntime, $b>,
            DeviceVec<ApiRuntime, $c>,
            DeviceVec<ApiRuntime, $d>,
            DeviceVec<ApiRuntime, $e>,
            DeviceVec<ApiRuntime, $f>,
        >
    };
    (SoA7, ($a:ty, $b:ty, $c:ty, $d:ty, $e:ty, $f:ty, $g:ty)) => {
        massively::SoA7<
            DeviceVec<ApiRuntime, $a>,
            DeviceVec<ApiRuntime, $b>,
            DeviceVec<ApiRuntime, $c>,
            DeviceVec<ApiRuntime, $d>,
            DeviceVec<ApiRuntime, $e>,
            DeviceVec<ApiRuntime, $f>,
            DeviceVec<ApiRuntime, $g>,
        >
    };
}

macro_rules! map_input_case {
    ($input:expr, $soa:ident, ($($ty:ty),+)) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_input = make_soa!(&exec, &input, $soa, ($($ty),+));
        let gpu_output: massively::SoA1<DeviceVec<ApiRuntime, u32>> =
            massively::map(&exec, gpu_input.slice(..), ArityTupleToScalar, ()).unwrap();
        let gpu = cols_to_aos!(exec.to_host(&gpu_output).unwrap(), SoA1);
        let host = oracle::map(&input, ArityTupleToScalar, ());
        prop_assert_eq!(gpu, host);
    }};
}

macro_rules! map_output_case {
    ($input:expr, $soa:ident, ($($ty:ty),+), $init:expr, $op:ident) => {{
        let _ = $init;
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_input = make_soa!(&exec, &input, SoA1, (u32));
        let gpu_output: owned_soa_type!($soa, ($($ty),+)) =
            massively::map(&exec, gpu_input.slice(..), $op, ()).unwrap();
        let gpu = cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $soa);
        let host = oracle::map(&input, $op, ());
        prop_assert_eq!(gpu, host);
    }};
}

macro_rules! transform_input_case {
    ($input:expr, $soa:ident, ($($ty:ty),+)) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_input = make_soa!(&exec, &input, $soa, ($($ty),+));
        let gpu_output = make_soa!(&exec, &vec![(0_u32,); input.len()], SoA1, (u32));
        massively::transform(
            &exec,
            gpu_input.slice(..),
            ArityTupleToScalar,
            (),
            gpu_output.slice_mut(..),
        )
        .unwrap();
        let gpu = cols_to_aos!(exec.to_host(&gpu_output).unwrap(), SoA1);
        let mut host = vec![(0_u32,); input.len()];
        oracle::transform(&input, ArityTupleToScalar, (), &mut host);
        prop_assert_eq!(gpu, host);
    }};
}

macro_rules! transform_output_case {
    ($input:expr, $soa:ident, ($($ty:ty),+), $init:expr, $op:ident) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let mut host = vec![$init; input.len()];
        let gpu_input = make_soa!(&exec, &input, SoA1, (u32));
        let gpu_output = make_soa!(&exec, &host, $soa, ($($ty),+));
        massively::transform(
            &exec,
            gpu_input.slice(..),
            $op,
            (),
            gpu_output.slice_mut(..),
        )
        .unwrap();
        oracle::transform(&input, $op, (), &mut host);
        let gpu = cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $soa);
        prop_assert_eq!(gpu, host);
    }};
}

macro_rules! transform_where_input_case {
    ($input:expr, $soa:ident, ($($ty:ty),+)) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let stencil = stencil_for!(input.len());
        let gpu_input = make_soa!(&exec, &input, $soa, ($($ty),+));
        let gpu_stencil = exec.to_device(&stencil).unwrap();
        let gpu_output = make_soa!(&exec, &vec![(0_u32,); input.len()], SoA1, (u32));
        massively::transform_where(
            &exec,
            gpu_input.slice(..),
            ArityTupleToScalar,
            (),
            gpu_stencil.slice(..),
            gpu_output.slice_mut(..),
        )
        .unwrap();
        let gpu = cols_to_aos!(exec.to_host(&gpu_output).unwrap(), SoA1);
        let mut host = vec![(0_u32,); input.len()];
        oracle::transform_where(&input, ArityTupleToScalar, (), &stencil, &mut host);
        prop_assert_eq!(gpu, host);
    }};
}

macro_rules! transform_where_output_case {
    ($input:expr, $soa:ident, ($($ty:ty),+), $init:expr, $op:ident) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let stencil = stencil_for!(input.len());
        let mut host = vec![$init; input.len()];
        let gpu_input = make_soa!(&exec, &input, SoA1, (u32));
        let gpu_stencil = exec.to_device(&stencil).unwrap();
        let gpu_output = make_soa!(&exec, &host, $soa, ($($ty),+));
        massively::transform_where(
            &exec,
            gpu_input.slice(..),
            $op,
            (),
            gpu_stencil.slice(..),
            gpu_output.slice_mut(..),
        )
        .unwrap();
        oracle::transform_where(&input, $op, (), &stencil, &mut host);
        let gpu = cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $soa);
        prop_assert_eq!(gpu, host);
    }};
}

macro_rules! sort_by_key_only_case {
    ($pairs:expr, $key_soa:ident, ($($key_ty:ty),+), $value_soa:ident, ($($value_ty:ty),+), $less:ident) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let pairs = $pairs;
        let (keys, values): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
        let gpu_keys = make_soa!(&exec, &keys, $key_soa, ($($key_ty),+));
        let gpu_values = make_soa!(&exec, &values, $value_soa, ($($value_ty),+));
        let (out_keys, out_values) =
            massively::sort_by_key(&exec, gpu_keys.slice(..), gpu_values.slice(..), $less).unwrap();
        let (host_keys, host_values) = oracle::sort_by_key(&keys, &values, $less);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&out_keys).unwrap(), $key_soa), host_keys);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&out_values).unwrap(), $value_soa), host_values);
    }};
}

macro_rules! stable_sort_by_key_case {
    ($pairs:expr, $key_soa:ident, ($($key_ty:ty),+), $value_soa:ident, ($($value_ty:ty),+), $less:ident) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let pairs = $pairs;
        let (keys, values): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
        let gpu_keys = make_soa!(&exec, &keys, $key_soa, ($($key_ty),+));
        let gpu_values = make_soa!(&exec, &values, $value_soa, ($($value_ty),+));
        let (out_keys, out_values) =
            massively::stable_sort_by_key(&exec, gpu_keys.slice(..), gpu_values.slice(..), $less).unwrap();
        let (host_keys, host_values) = oracle::stable_sort_by_key(&keys, &values, $less);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&out_keys).unwrap(), $key_soa), host_keys);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&out_values).unwrap(), $value_soa), host_values);
    }};
}

macro_rules! scan_by_key_case {
    (inclusive_scan_by_key, $pairs:expr, $key_soa:ident, ($($key_ty:ty),+), $value_soa:ident, ($($value_ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let pairs = $pairs;
        let (keys, values): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
        let gpu_keys = make_soa!(&exec, &keys, $key_soa, ($($key_ty),+));
        let gpu_values = make_soa!(&exec, &values, $value_soa, ($($value_ty),+));
        let gpu_output = massively::inclusive_scan_by_key(
            &exec,
            gpu_keys.slice(..),
            gpu_values.slice(..),
            EqTuple,
            MaxTuple,
        )
        .unwrap();
        prop_assert_eq!(
            cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $value_soa),
            oracle::inclusive_scan_by_key(&keys, &values, EqTuple, MaxTuple)
        );
    }};
    (exclusive_scan_by_key, $pairs:expr, $key_soa:ident, ($($key_ty:ty),+), $value_soa:ident, ($($value_ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let pairs = $pairs;
        let (keys, values): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
        let gpu_keys = make_soa!(&exec, &keys, $key_soa, ($($key_ty),+));
        let gpu_values = make_soa!(&exec, &values, $value_soa, ($($value_ty),+));
        let gpu_output = massively::exclusive_scan_by_key(
            &exec,
            gpu_keys.slice(..),
            gpu_values.slice(..),
            EqTuple,
            $init,
            MaxTuple,
        )
        .unwrap();
        prop_assert_eq!(
            cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $value_soa),
            oracle::exclusive_scan_by_key(&keys, &values, EqTuple, $init, MaxTuple)
        );
    }};
    (reduce_by_key, $pairs:expr, $key_soa:ident, ($($key_ty:ty),+), $value_soa:ident, ($($value_ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let pairs = $pairs;
        let (keys, values): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
        let gpu_keys = make_soa!(&exec, &keys, $key_soa, ($($key_ty),+));
        let gpu_values = make_soa!(&exec, &values, $value_soa, ($($value_ty),+));
        let (gpu_out_keys, gpu_out_values) = massively::reduce_by_key(
            &exec,
            gpu_keys.slice(..),
            gpu_values.slice(..),
            EqTuple,
            $init,
            MaxTuple,
        )
        .unwrap();
        let (host_keys, host_values) =
            oracle::reduce_by_key(&keys, &values, EqTuple, $init, MaxTuple);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_out_keys).unwrap(), $key_soa), host_keys);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_out_values).unwrap(), $value_soa), host_values);
    }};
    (unique_by_key, $pairs:expr, $key_soa:ident, ($($key_ty:ty),+), $value_soa:ident, ($($value_ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let pairs = $pairs;
        let (keys, values): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
        let gpu_keys = make_soa!(&exec, &keys, $key_soa, ($($key_ty),+));
        let gpu_values = make_soa!(&exec, &values, $value_soa, ($($value_ty),+));
        let (gpu_out_keys, gpu_out_values) =
            massively::unique_by_key(&exec, gpu_keys.slice(..), gpu_values.slice(..), EqTuple)
                .unwrap();
        let (host_keys, host_values) = oracle::unique_by_key(&keys, &values, EqTuple);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_out_keys).unwrap(), $key_soa), host_keys);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_out_values).unwrap(), $value_soa), host_values);
    }};
}

macro_rules! merge_by_key_case {
    ($pairs:expr, $key_soa:ident, ($($key_ty:ty),+), $value_soa:ident, ($($value_ty:ty),+), $less:ident) => {{
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
        let gpu_left_keys = make_soa!(&exec, &left_keys, $key_soa, ($($key_ty),+));
        let gpu_right_keys = make_soa!(&exec, &right_keys, $key_soa, ($($key_ty),+));
        let gpu_left_values = make_soa!(&exec, &left_values, $value_soa, ($($value_ty),+));
        let gpu_right_values = make_soa!(&exec, &right_values, $value_soa, ($($value_ty),+));

        let (gpu_keys, gpu_values) = massively::merge_by_key(
            &exec,
            gpu_left_keys.slice(..),
            gpu_left_values.slice(..),
            gpu_right_keys.slice(..),
            gpu_right_values.slice(..),
            $less,
        )
        .unwrap();
        let (host_keys, host_values) = oracle::merge_by_key(
            &left_keys,
            &left_values,
            &right_keys,
            &right_values,
            $less,
        );
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_keys).unwrap(), $key_soa), host_keys);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_values).unwrap(), $value_soa), host_values);
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
    (reduce, $input:expr, $soa:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_input = make_soa!(&exec, &input, $soa, ($($ty),+));
        let gpu = massively::reduce(&exec, gpu_input.slice(..), $init, MaxTuple).unwrap();
        let host = oracle::reduce(&input, $init, MaxTuple);
        prop_assert_eq!(gpu, host);
    }};
    (inclusive_scan, $input:expr, $soa:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_input = make_soa!(&exec, &input, $soa, ($($ty),+));
        let gpu_output = massively::inclusive_scan(&exec, gpu_input.slice(..), MaxTuple).unwrap();
        let gpu = cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $soa);
        let host = oracle::inclusive_scan(&input, MaxTuple);
        prop_assert_eq!(gpu, host);
    }};
    (exclusive_scan, $input:expr, $soa:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_input = make_soa!(&exec, &input, $soa, ($($ty),+));
        let gpu_output = massively::exclusive_scan(&exec, gpu_input.slice(..), $init, MaxTuple).unwrap();
        let gpu = cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $soa);
        let host = oracle::exclusive_scan(&input, $init, MaxTuple);
        prop_assert_eq!(gpu, host);
    }};
    (adjacent_difference, $input:expr, $soa:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_input = make_soa!(&exec, &input, $soa, ($($ty),+));
        let gpu_output = massively::adjacent_difference(&exec, gpu_input.slice(..), MaxTuple).unwrap();
        let gpu = cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $soa);
        let host = oracle::adjacent_difference(&input, MaxTuple);
        prop_assert_eq!(gpu, host);
    }};
    (copy_where, $input:expr, $soa:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let stencil = stencil_for!(input.len());
        let gpu_input = make_soa!(&exec, &input, $soa, ($($ty),+));
        let gpu_stencil = exec.to_device(&stencil).unwrap();
        let gpu_output = massively::copy_where(&exec, gpu_input.slice(..), gpu_stencil.slice(..)).unwrap();
        let gpu = cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $soa);
        let host = oracle::copy_where(&input, &stencil);
        prop_assert_eq!(gpu, host);
    }};
    (remove_where, $input:expr, $soa:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let stencil = stencil_for!(input.len());
        let gpu_input = make_soa!(&exec, &input, $soa, ($($ty),+));
        let gpu_stencil = exec.to_device(&stencil).unwrap();
        let gpu_output = massively::remove_where(&exec, gpu_input.slice(..), gpu_stencil.slice(..)).unwrap();
        let gpu = cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $soa);
        let host = oracle::remove_where(&input, &stencil);
        prop_assert_eq!(gpu, host);
    }};
    (reverse, $input:expr, $soa:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_input = make_soa!(&exec, &input, $soa, ($($ty),+));
        let gpu_output = massively::reverse(&exec, gpu_input.slice(..)).unwrap();
        let gpu = cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $soa);
        let host = oracle::reverse(&input);
        prop_assert_eq!(gpu, host);
    }};
    (count_if, $input:expr, $soa:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_input = make_soa!(&exec, &input, $soa, ($($ty),+));
        prop_assert_eq!(
            massively::count_if(&exec, gpu_input.slice(..), KeepTuple, ()).unwrap(),
            oracle::count_if(&input, KeepTuple, ())
        );
    }};
    (all_of, $input:expr, $soa:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_input = make_soa!(&exec, &input, $soa, ($($ty),+));
        prop_assert_eq!(
            massively::all_of(&exec, gpu_input.slice(..), KeepTuple, ()).unwrap(),
            oracle::all_of(&input, KeepTuple, ())
        );
    }};
    (any_of, $input:expr, $soa:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_input = make_soa!(&exec, &input, $soa, ($($ty),+));
        prop_assert_eq!(
            massively::any_of(&exec, gpu_input.slice(..), KeepTuple, ()).unwrap(),
            oracle::any_of(&input, KeepTuple, ())
        );
    }};
    (none_of, $input:expr, $soa:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_input = make_soa!(&exec, &input, $soa, ($($ty),+));
        prop_assert_eq!(
            massively::none_of(&exec, gpu_input.slice(..), KeepTuple, ()).unwrap(),
            oracle::none_of(&input, KeepTuple, ())
        );
    }};
    (find_if, $input:expr, $soa:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_input = make_soa!(&exec, &input, $soa, ($($ty),+));
        prop_assert_eq!(
            massively::find_if(&exec, gpu_input.slice(..), KeepTuple, ()).unwrap(),
            oracle::find_if(&input, KeepTuple, ())
        );
    }};
    (is_partitioned, $input:expr, $soa:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_input = make_soa!(&exec, &input, $soa, ($($ty),+));
        prop_assert_eq!(
            massively::is_partitioned(&exec, gpu_input.slice(..), KeepTuple, ()).unwrap(),
            oracle::is_partitioned(&input, KeepTuple, ())
        );
    }};
    (partition, $input:expr, $soa:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_input = make_soa!(&exec, &input, $soa, ($($ty),+));
        let (gpu_yes, gpu_no) = massively::partition(&exec, gpu_input.slice(..), KeepTuple, ()).unwrap();
        let (host_yes, host_no) = oracle::partition(&input, KeepTuple, ());
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_yes).unwrap(), $soa), host_yes);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_no).unwrap(), $soa), host_no);
    }};
    (predicate, $input:expr, $soa:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_input = make_soa!(&exec, &input, $soa, ($($ty),+));
        prop_assert_eq!(
            massively::count_if(&exec, gpu_input.slice(..), KeepTuple, ()).unwrap(),
            oracle::count_if(&input, KeepTuple, ())
        );
        prop_assert_eq!(
            massively::all_of(&exec, gpu_input.slice(..), KeepTuple, ()).unwrap(),
            oracle::all_of(&input, KeepTuple, ())
        );
        prop_assert_eq!(
            massively::any_of(&exec, gpu_input.slice(..), KeepTuple, ()).unwrap(),
            oracle::any_of(&input, KeepTuple, ())
        );
        prop_assert_eq!(
            massively::none_of(&exec, gpu_input.slice(..), KeepTuple, ()).unwrap(),
            oracle::none_of(&input, KeepTuple, ())
        );
        prop_assert_eq!(
            massively::find_if(&exec, gpu_input.slice(..), KeepTuple, ()).unwrap(),
            oracle::find_if(&input, KeepTuple, ())
        );
        prop_assert_eq!(
            massively::is_partitioned(&exec, gpu_input.slice(..), KeepTuple, ()).unwrap(),
            oracle::is_partitioned(&input, KeepTuple, ())
        );
        let (gpu_yes, gpu_no) = massively::partition(&exec, gpu_input.slice(..), KeepTuple, ()).unwrap();
        let (host_yes, host_no) = oracle::partition(&input, KeepTuple, ());
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_yes).unwrap(), $soa), host_yes);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_no).unwrap(), $soa), host_no);
    }};
    (indexed, $input:expr, $soa:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let indices = reverse_indices_for!(input.len());
        let stencil = stencil_for!(input.len());
        let gpu_input = make_soa!(&exec, &input, $soa, ($($ty),+));
        let gpu_indices = exec.to_device(&indices).unwrap();
        let gpu_stencil = exec.to_device(&stencil).unwrap();

        let gpu_output: owned_soa_type!($soa, ($($ty),+)) =
            massively::permute(&exec, gpu_input.slice(..), gpu_indices.slice(..)).unwrap();
        let mut host = input.clone();
        oracle::gather(&input, &indices, &mut host);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $soa), host);

        let gpu_output = make_soa!(&exec, &input, $soa, ($($ty),+));
        massively::gather(&exec, gpu_input.slice(..), gpu_indices.slice(..), gpu_output.slice_mut(..)).unwrap();
        let mut host = input.clone();
        oracle::gather(&input, &indices, &mut host);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $soa), host);

        let gpu_output = make_soa!(&exec, &input, $soa, ($($ty),+));
        massively::gather_where(
            &exec,
            gpu_input.slice(..),
            gpu_indices.slice(..),
            gpu_stencil.slice(..),
            gpu_output.slice_mut(..),
        )
        .unwrap();
        let mut host = input.clone();
        oracle::gather_where(&input, &indices, &stencil, &mut host);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $soa), host);

        let gpu_output = make_soa!(&exec, &input, $soa, ($($ty),+));
        massively::scatter(&exec, gpu_input.slice(..), gpu_indices.slice(..), gpu_output.slice_mut(..)).unwrap();
        let mut host = input.clone();
        oracle::scatter(&input, &indices, &mut host);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $soa), host);

        let gpu_output = make_soa!(&exec, &input, $soa, ($($ty),+));
        massively::scatter_where(
            &exec,
            gpu_input.slice(..),
            gpu_indices.slice(..),
            gpu_stencil.slice(..),
            gpu_output.slice_mut(..),
        )
        .unwrap();
        let mut host = input.clone();
        oracle::scatter_where(&input, &indices, &stencil, &mut host);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $soa), host);
    }};
    (permute, $input:expr, $soa:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let indices = reverse_indices_for!(input.len());
        let gpu_input = make_soa!(&exec, &input, $soa, ($($ty),+));
        let gpu_indices = exec.to_device(&indices).unwrap();
        let gpu_output: owned_soa_type!($soa, ($($ty),+)) =
            massively::permute(&exec, gpu_input.slice(..), gpu_indices.slice(..)).unwrap();
        let mut host = input.clone();
        oracle::gather(&input, &indices, &mut host);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $soa), host);
    }};
    (gather, $input:expr, $soa:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let indices = reverse_indices_for!(input.len());
        let gpu_input = make_soa!(&exec, &input, $soa, ($($ty),+));
        let gpu_indices = exec.to_device(&indices).unwrap();
        let gpu_output = make_soa!(&exec, &input, $soa, ($($ty),+));
        massively::gather(&exec, gpu_input.slice(..), gpu_indices.slice(..), gpu_output.slice_mut(..)).unwrap();
        let mut host = input.clone();
        oracle::gather(&input, &indices, &mut host);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $soa), host);
    }};
    (gather_where, $input:expr, $soa:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let indices = reverse_indices_for!(input.len());
        let stencil = stencil_for!(input.len());
        let gpu_input = make_soa!(&exec, &input, $soa, ($($ty),+));
        let gpu_indices = exec.to_device(&indices).unwrap();
        let gpu_stencil = exec.to_device(&stencil).unwrap();
        let gpu_output = make_soa!(&exec, &input, $soa, ($($ty),+));
        massively::gather_where(
            &exec,
            gpu_input.slice(..),
            gpu_indices.slice(..),
            gpu_stencil.slice(..),
            gpu_output.slice_mut(..),
        )
        .unwrap();
        let mut host = input.clone();
        oracle::gather_where(&input, &indices, &stencil, &mut host);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $soa), host);
    }};
    (scatter, $input:expr, $soa:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let indices = reverse_indices_for!(input.len());
        let gpu_input = make_soa!(&exec, &input, $soa, ($($ty),+));
        let gpu_indices = exec.to_device(&indices).unwrap();
        let gpu_output = make_soa!(&exec, &input, $soa, ($($ty),+));
        massively::scatter(&exec, gpu_input.slice(..), gpu_indices.slice(..), gpu_output.slice_mut(..)).unwrap();
        let mut host = input.clone();
        oracle::scatter(&input, &indices, &mut host);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $soa), host);
    }};
    (scatter_where, $input:expr, $soa:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let indices = reverse_indices_for!(input.len());
        let stencil = stencil_for!(input.len());
        let gpu_input = make_soa!(&exec, &input, $soa, ($($ty),+));
        let gpu_indices = exec.to_device(&indices).unwrap();
        let gpu_stencil = exec.to_device(&stencil).unwrap();
        let gpu_output = make_soa!(&exec, &input, $soa, ($($ty),+));
        massively::scatter_where(
            &exec,
            gpu_input.slice(..),
            gpu_indices.slice(..),
            gpu_stencil.slice(..),
            gpu_output.slice_mut(..),
        )
        .unwrap();
        let mut host = input.clone();
        oracle::scatter_where(&input, &indices, &stencil, &mut host);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $soa), host);
    }};
    (search_eq, $input:expr, $soa:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let mut other = input.clone();
        other.reverse();
        let gpu_input = make_soa!(&exec, &input, $soa, ($($ty),+));
        let gpu_other = make_soa!(&exec, &other, $soa, ($($ty),+));
        prop_assert_eq!(
            massively::equal(&exec, gpu_input.slice(..), gpu_input.slice(..), EqTuple).unwrap(),
            oracle::equal(&input, &input, EqTuple)
        );
        prop_assert_eq!(
            massively::mismatch(&exec, gpu_input.slice(..), gpu_other.slice(..), EqTuple).unwrap(),
            oracle::mismatch(&input, &other, EqTuple)
        );
        prop_assert_eq!(
            massively::adjacent_find(&exec, gpu_input.slice(..), EqTuple).unwrap(),
            oracle::adjacent_find(&input, EqTuple)
        );
        prop_assert_eq!(
            massively::find_first_of(&exec, gpu_input.slice(..), gpu_other.slice(..), EqTuple).unwrap(),
            oracle::find_first_of(&input, &other, EqTuple)
        );
    }};
    (equal, $input:expr, $soa:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_input = make_soa!(&exec, &input, $soa, ($($ty),+));
        prop_assert_eq!(
            massively::equal(&exec, gpu_input.slice(..), gpu_input.slice(..), EqTuple).unwrap(),
            oracle::equal(&input, &input, EqTuple)
        );
    }};
    (mismatch, $input:expr, $soa:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let mut other = input.clone();
        other.reverse();
        let gpu_input = make_soa!(&exec, &input, $soa, ($($ty),+));
        let gpu_other = make_soa!(&exec, &other, $soa, ($($ty),+));
        prop_assert_eq!(
            massively::mismatch(&exec, gpu_input.slice(..), gpu_other.slice(..), EqTuple).unwrap(),
            oracle::mismatch(&input, &other, EqTuple)
        );
    }};
    (adjacent_find, $input:expr, $soa:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_input = make_soa!(&exec, &input, $soa, ($($ty),+));
        prop_assert_eq!(
            massively::adjacent_find(&exec, gpu_input.slice(..), EqTuple).unwrap(),
            oracle::adjacent_find(&input, EqTuple)
        );
    }};
    (find_first_of, $input:expr, $soa:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let mut other = input.clone();
        other.reverse();
        let gpu_input = make_soa!(&exec, &input, $soa, ($($ty),+));
        let gpu_other = make_soa!(&exec, &other, $soa, ($($ty),+));
        prop_assert_eq!(
            massively::find_first_of(&exec, gpu_input.slice(..), gpu_other.slice(..), EqTuple).unwrap(),
            oracle::find_first_of(&input, &other, EqTuple)
        );
    }};
    (mutating, $input:expr, $soa:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let stencil = stencil_for!(input.len());
        let replacement = $init;
        let gpu_stencil = exec.to_device(&stencil).unwrap();
        let gpu_input = make_soa!(&exec, &input, $soa, ($($ty),+));

        let gpu_output = make_soa!(&exec, &input, $soa, ($($ty),+));
        massively::fill(&exec, replacement, gpu_output.slice_mut(..)).unwrap();
        let mut host = input.clone();
        oracle::fill(replacement, &mut host);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $soa), host);

        let gpu_output = make_soa!(&exec, &input, $soa, ($($ty),+));
        massively::replace_where(
            &exec,
            replacement,
            gpu_stencil.slice(..),
            gpu_output.slice_mut(..),
        )
        .unwrap();
        let mut host = input.clone();
        oracle::replace_where(replacement, &stencil, &mut host);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $soa), host);

        let gpu_output = make_soa!(&exec, &input, $soa, ($($ty),+));
        massively::transform_where(
            &exec,
            gpu_input.slice(..),
            Identity,
            (),
            gpu_stencil.slice(..),
            gpu_output.slice_mut(..),
        )
        .unwrap();
        let mut host = input.clone();
        oracle::transform_where(&input, Identity, (), &stencil, &mut host);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $soa), host);
    }};
    (fill, $input:expr, $soa:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_output = make_soa!(&exec, &input, $soa, ($($ty),+));
        massively::fill(&exec, $init, gpu_output.slice_mut(..)).unwrap();
        let mut host = input.clone();
        oracle::fill($init, &mut host);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $soa), host);
    }};
    (replace_where, $input:expr, $soa:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let stencil = stencil_for!(input.len());
        let gpu_stencil = exec.to_device(&stencil).unwrap();
        let gpu_output = make_soa!(&exec, &input, $soa, ($($ty),+));
        massively::replace_where(
            &exec,
            $init,
            gpu_stencil.slice(..),
            gpu_output.slice_mut(..),
        )
        .unwrap();
        let mut host = input.clone();
        oracle::replace_where($init, &stencil, &mut host);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $soa), host);
    }};
    (transform_where, $input:expr, $soa:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let stencil = stencil_for!(input.len());
        let gpu_stencil = exec.to_device(&stencil).unwrap();
        let gpu_input = make_soa!(&exec, &input, $soa, ($($ty),+));
        let gpu_output = make_soa!(&exec, &input, $soa, ($($ty),+));
        massively::transform_where(
            &exec,
            gpu_input.slice(..),
            Identity,
            (),
            gpu_stencil.slice(..),
            gpu_output.slice_mut(..),
        )
        .unwrap();
        let mut host = input.clone();
        oracle::transform_where(&input, Identity, (), &stencil, &mut host);
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $soa), host);
    }};
    (ordering, $input:expr, $soa:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let sorted = oracle::sort(&input, LessTuple);
        let mut other = sorted.clone();
        other.reverse();
        let right = oracle::sort(&other, LessTuple);
        let gpu_input = make_soa!(&exec, &input, $soa, ($($ty),+));
        let gpu_sorted = make_soa!(&exec, &sorted, $soa, ($($ty),+));
        let gpu_right = make_soa!(&exec, &right, $soa, ($($ty),+));

        let gpu_output = massively::sort(&exec, gpu_input.slice(..), LessTuple).unwrap();
        prop_assert_eq!(cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $soa), sorted.clone());

        let gpu_output = massively::stable_sort(&exec, gpu_input.slice(..), LessTuple).unwrap();
        prop_assert_eq!(
            cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $soa),
            oracle::stable_sort(&input, LessTuple)
        );

        let gpu_output = massively::merge(&exec, gpu_sorted.slice(..), gpu_right.slice(..), LessTuple).unwrap();
        prop_assert_eq!(
            cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $soa),
            oracle::merge(&sorted, &right, LessTuple)
        );

        prop_assert_eq!(
            massively::is_sorted(&exec, gpu_sorted.slice(..), LessTuple).unwrap(),
            oracle::is_sorted(&sorted, LessTuple)
        );
        prop_assert_eq!(
            massively::is_sorted_until(&exec, gpu_input.slice(..), LessTuple).unwrap(),
            oracle::is_sorted_until(&input, LessTuple)
        );
        prop_assert_eq!(
            massively::lexicographical_compare(&exec, gpu_input.slice(..), gpu_right.slice(..), LessTuple).unwrap(),
            oracle::lexicographical_compare(&input, &right, LessTuple)
        );
        prop_assert_eq!(
            massively::min_element(&exec, gpu_input.slice(..), LessTuple).unwrap(),
            oracle::min_element(&input, LessTuple)
        );
        prop_assert_eq!(
            massively::max_element(&exec, gpu_input.slice(..), LessTuple).unwrap(),
            oracle::max_element(&input, LessTuple)
        );
        prop_assert_eq!(
            massively::minmax_element(&exec, gpu_input.slice(..), LessTuple).unwrap(),
            oracle::minmax_element(&input, LessTuple)
        );

        let gpu_bounds = massively::lower_bound(&exec, gpu_sorted.slice(..), gpu_right.slice(..), LessTuple).unwrap();
        prop_assert_eq!(exec.to_host(&gpu_bounds).unwrap(), oracle::lower_bound(&sorted, &right, LessTuple));
        let gpu_bounds = massively::upper_bound(&exec, gpu_sorted.slice(..), gpu_right.slice(..), LessTuple).unwrap();
        prop_assert_eq!(exec.to_host(&gpu_bounds).unwrap(), oracle::upper_bound(&sorted, &right, LessTuple));

        let gpu_output = massively::unique(&exec, gpu_sorted.slice(..), EqTuple).unwrap();
        prop_assert_eq!(
            cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $soa),
            oracle::unique(&sorted, EqTuple)
        );
        let gpu_output = massively::set_union(&exec, gpu_sorted.slice(..), gpu_right.slice(..), LessTuple).unwrap();
        prop_assert_eq!(
            cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $soa),
            oracle::set_union(&sorted, &right, LessTuple)
        );
        let gpu_output = massively::set_intersection(&exec, gpu_sorted.slice(..), gpu_right.slice(..), LessTuple).unwrap();
        prop_assert_eq!(
            cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $soa),
            oracle::set_intersection(&sorted, &right, LessTuple)
        );
        let gpu_output = massively::set_difference(&exec, gpu_sorted.slice(..), gpu_right.slice(..), LessTuple).unwrap();
        prop_assert_eq!(
            cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $soa),
            oracle::set_difference(&sorted, &right, LessTuple)
        );
    }};
    (sort, $input:expr, $soa:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_input = make_soa!(&exec, &input, $soa, ($($ty),+));
        let gpu_output = massively::sort(&exec, gpu_input.slice(..), LessTuple).unwrap();
        prop_assert_eq!(
            cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $soa),
            oracle::sort(&input, LessTuple)
        );
    }};
    (stable_sort, $input:expr, $soa:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_input = make_soa!(&exec, &input, $soa, ($($ty),+));
        let gpu_output = massively::stable_sort(&exec, gpu_input.slice(..), LessTuple).unwrap();
        prop_assert_eq!(
            cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $soa),
            oracle::stable_sort(&input, LessTuple)
        );
    }};
    (merge, $input:expr, $soa:ident, ($($ty:ty),+), $init:expr) => {{
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
        let gpu_sorted = make_soa!(&exec, &sorted, $soa, ($($ty),+));
        let gpu_right = make_soa!(&exec, &right, $soa, ($($ty),+));
        let gpu_output = massively::merge(&exec, gpu_sorted.slice(..), gpu_right.slice(..), LessTuple).unwrap();
        prop_assert_eq!(
            cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $soa),
            oracle::merge(&sorted, &right, LessTuple)
        );
    }};
    (is_sorted, $input:expr, $soa:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let sorted = oracle::sort(&input, LessTuple);
        let gpu_sorted = make_soa!(&exec, &sorted, $soa, ($($ty),+));
        prop_assert_eq!(
            massively::is_sorted(&exec, gpu_sorted.slice(..), LessTuple).unwrap(),
            oracle::is_sorted(&sorted, LessTuple)
        );
    }};
    (is_sorted_until, $input:expr, $soa:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_input = make_soa!(&exec, &input, $soa, ($($ty),+));
        prop_assert_eq!(
            massively::is_sorted_until(&exec, gpu_input.slice(..), LessTuple).unwrap(),
            oracle::is_sorted_until(&input, LessTuple)
        );
    }};
    (lexicographical_compare, $input:expr, $soa:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let sorted = oracle::sort(&input, LessTuple);
        let mut other = sorted.clone();
        other.reverse();
        let right = oracle::sort(&other, LessTuple);
        let gpu_input = make_soa!(&exec, &input, $soa, ($($ty),+));
        let gpu_right = make_soa!(&exec, &right, $soa, ($($ty),+));
        prop_assert_eq!(
            massively::lexicographical_compare(&exec, gpu_input.slice(..), gpu_right.slice(..), LessTuple).unwrap(),
            oracle::lexicographical_compare(&input, &right, LessTuple)
        );
    }};
    (min_element, $input:expr, $soa:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_input = make_soa!(&exec, &input, $soa, ($($ty),+));
        prop_assert_eq!(
            massively::min_element(&exec, gpu_input.slice(..), LessTuple).unwrap(),
            oracle::min_element(&input, LessTuple)
        );
    }};
    (max_element, $input:expr, $soa:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_input = make_soa!(&exec, &input, $soa, ($($ty),+));
        prop_assert_eq!(
            massively::max_element(&exec, gpu_input.slice(..), LessTuple).unwrap(),
            oracle::max_element(&input, LessTuple)
        );
    }};
    (minmax_element, $input:expr, $soa:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let gpu_input = make_soa!(&exec, &input, $soa, ($($ty),+));
        prop_assert_eq!(
            massively::minmax_element(&exec, gpu_input.slice(..), LessTuple).unwrap(),
            oracle::minmax_element(&input, LessTuple)
        );
    }};
    (lower_bound, $input:expr, $soa:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let sorted = oracle::sort(&input, LessTuple);
        let mut other = sorted.clone();
        other.reverse();
        let right = oracle::sort(&other, LessTuple);
        let gpu_sorted = make_soa!(&exec, &sorted, $soa, ($($ty),+));
        let gpu_right = make_soa!(&exec, &right, $soa, ($($ty),+));
        let gpu_bounds = massively::lower_bound(&exec, gpu_sorted.slice(..), gpu_right.slice(..), LessTuple).unwrap();
        prop_assert_eq!(exec.to_host(&gpu_bounds).unwrap(), oracle::lower_bound(&sorted, &right, LessTuple));
    }};
    (upper_bound, $input:expr, $soa:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let sorted = oracle::sort(&input, LessTuple);
        let mut other = sorted.clone();
        other.reverse();
        let right = oracle::sort(&other, LessTuple);
        let gpu_sorted = make_soa!(&exec, &sorted, $soa, ($($ty),+));
        let gpu_right = make_soa!(&exec, &right, $soa, ($($ty),+));
        let gpu_bounds = massively::upper_bound(&exec, gpu_sorted.slice(..), gpu_right.slice(..), LessTuple).unwrap();
        prop_assert_eq!(exec.to_host(&gpu_bounds).unwrap(), oracle::upper_bound(&sorted, &right, LessTuple));
    }};
    (unique, $input:expr, $soa:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let sorted = oracle::sort(&input, LessTuple);
        let gpu_sorted = make_soa!(&exec, &sorted, $soa, ($($ty),+));
        let gpu_output = massively::unique(&exec, gpu_sorted.slice(..), EqTuple).unwrap();
        prop_assert_eq!(
            cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $soa),
            oracle::unique(&sorted, EqTuple)
        );
    }};
    (set_union, $input:expr, $soa:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let sorted = oracle::sort(&input, LessTuple);
        let mut other = sorted.clone();
        other.reverse();
        let right = oracle::sort(&other, LessTuple);
        let gpu_sorted = make_soa!(&exec, &sorted, $soa, ($($ty),+));
        let gpu_right = make_soa!(&exec, &right, $soa, ($($ty),+));
        let gpu_output = massively::set_union(&exec, gpu_sorted.slice(..), gpu_right.slice(..), LessTuple).unwrap();
        prop_assert_eq!(
            cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $soa),
            oracle::set_union(&sorted, &right, LessTuple)
        );
    }};
    (set_intersection, $input:expr, $soa:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let sorted = oracle::sort(&input, LessTuple);
        let mut other = sorted.clone();
        other.reverse();
        let right = oracle::sort(&other, LessTuple);
        let gpu_sorted = make_soa!(&exec, &sorted, $soa, ($($ty),+));
        let gpu_right = make_soa!(&exec, &right, $soa, ($($ty),+));
        let gpu_output = massively::set_intersection(&exec, gpu_sorted.slice(..), gpu_right.slice(..), LessTuple).unwrap();
        prop_assert_eq!(
            cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $soa),
            oracle::set_intersection(&sorted, &right, LessTuple)
        );
    }};
    (set_difference, $input:expr, $soa:ident, ($($ty:ty),+), $init:expr) => {{
        let _guard = gpu_lock();
        let exec = exec();
        let input = $input;
        let sorted = oracle::sort(&input, LessTuple);
        let mut other = sorted.clone();
        other.reverse();
        let right = oracle::sort(&other, LessTuple);
        let gpu_sorted = make_soa!(&exec, &sorted, $soa, ($($ty),+));
        let gpu_right = make_soa!(&exec, &right, $soa, ($($ty),+));
        let gpu_output = massively::set_difference(&exec, gpu_sorted.slice(..), gpu_right.slice(..), LessTuple).unwrap();
        prop_assert_eq!(
            cols_to_aos!(exec.to_host(&gpu_output).unwrap(), $soa),
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
                    value_case!($case, input, SoA1, (u32), (0_u32,));
                }

                #[test]
                fn arity_2(input in prop::collection::vec((0_u32..4096, 0_u32..4096), 0..MAX_LEN)) {
                    value_case!($case, input, SoA2, (u32, u32), (0_u32, 0_u32));
                }

                #[test]
                fn arity_3(input in prop::collection::vec((0_u32..4096, 0_u32..4096, 0_u32..4096), 0..MAX_LEN)) {
                    value_case!($case, input, SoA3, (u32, u32, u32), (0_u32, 0_u32, 0_u32));
                }

                #[test]
                fn arity_4(input in prop::collection::vec((0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096), 0..MAX_LEN)) {
                    value_case!($case, input, SoA4, (u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32));
                }

                #[test]
                fn arity_5(input in prop::collection::vec((0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096), 0..MAX_LEN)) {
                    value_case!($case, input, SoA5, (u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32));
                }

                #[test]
                fn arity_6(input in prop::collection::vec((0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096), 0..MAX_LEN)) {
                    value_case!($case, input, SoA6, (u32, u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32));
                }

                #[test]
                fn arity_7(input in prop::collection::vec((0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096), 0..MAX_LEN)) {
                    value_case!($case, input, SoA7, (u32, u32, u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32));
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
                    value_case!($case, input, SoA1, (u32), (0_u32,));
                }

                #[test]
                fn arity_2(input in prop::collection::vec((0_u32..4096, 0_u32..4096), 0..MAX_LEN).prop_filter("unique first column", |input| unique_by(input, |row| row.0))) {
                    value_case!($case, input, SoA2, (u32, u32), (0_u32, 0_u32));
                }

                #[test]
                fn arity_3(input in prop::collection::vec((0_u32..4096, 0_u32..4096, 0_u32..4096), 0..MAX_LEN).prop_filter("unique first column", |input| unique_by(input, |row| row.0))) {
                    value_case!($case, input, SoA3, (u32, u32, u32), (0_u32, 0_u32, 0_u32));
                }

                #[test]
                fn arity_4(input in prop::collection::vec((0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096), 0..MAX_LEN).prop_filter("unique first column", |input| unique_by(input, |row| row.0))) {
                    value_case!($case, input, SoA4, (u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32));
                }

                #[test]
                fn arity_5(input in prop::collection::vec((0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096), 0..MAX_LEN).prop_filter("unique first column", |input| unique_by(input, |row| row.0))) {
                    value_case!($case, input, SoA5, (u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32));
                }

                #[test]
                fn arity_6(input in prop::collection::vec((0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096), 0..MAX_LEN).prop_filter("unique first column", |input| unique_by(input, |row| row.0))) {
                    value_case!($case, input, SoA6, (u32, u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32));
                }

                #[test]
                fn arity_7(input in prop::collection::vec((0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096), 0..MAX_LEN).prop_filter("unique first column", |input| unique_by(input, |row| row.0))) {
                    value_case!($case, input, SoA7, (u32, u32, u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32));
                }
            }
        }
    };
}

define_ordering_arity_module!(sort_arity, sort);
define_ordering_arity_module!(stable_sort_arity, stable_sort);
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

macro_rules! define_unary_input_arity_module {
    ($module:ident, $case:ident) => {
        mod $module {
            use super::*;

            proptest! {
                #![proptest_config(ProptestConfig::with_cases(CASES))]

                #[test]
                fn arity_1(input in prop::collection::vec(any::<u32>().prop_map(|v| (v,)), 0..MAX_LEN)) {
                    $case!(input, SoA1, (u32));
                }

                #[test]
                fn arity_2(input in prop::collection::vec((any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, SoA2, (u32, u32));
                }

                #[test]
                fn arity_3(input in prop::collection::vec((any::<u32>(), any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, SoA3, (u32, u32, u32));
                }

                #[test]
                fn arity_4(input in prop::collection::vec((any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, SoA4, (u32, u32, u32, u32));
                }

                #[test]
                fn arity_5(input in prop::collection::vec((any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, SoA5, (u32, u32, u32, u32, u32));
                }

                #[test]
                fn arity_6(input in prop::collection::vec((any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, SoA6, (u32, u32, u32, u32, u32, u32));
                }

                #[test]
                fn arity_7(input in prop::collection::vec((any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>()), 0..MAX_LEN)) {
                    $case!(input, SoA7, (u32, u32, u32, u32, u32, u32, u32));
                }
            }
        }
    };
}

macro_rules! define_unary_output_arity_module {
    ($module:ident, $case:ident) => {
        mod $module {
            use super::*;

            proptest! {
                #![proptest_config(ProptestConfig::with_cases(CASES))]

                #[test]
                fn arity_1(input in prop::collection::vec(any::<u32>().prop_map(|v| (v,)), 0..MAX_LEN)) {
                    $case!(input, SoA1, (u32), (0_u32,), ArityScalarToTuple1);
                }

                #[test]
                fn arity_2(input in prop::collection::vec(any::<u32>().prop_map(|v| (v,)), 0..MAX_LEN)) {
                    $case!(input, SoA2, (u32, u32), (0_u32, 0_u32), ArityScalarToTuple2);
                }

                #[test]
                fn arity_3(input in prop::collection::vec(any::<u32>().prop_map(|v| (v,)), 0..MAX_LEN)) {
                    $case!(input, SoA3, (u32, u32, u32), (0_u32, 0_u32, 0_u32), ArityScalarToTuple3);
                }

                #[test]
                fn arity_4(input in prop::collection::vec(any::<u32>().prop_map(|v| (v,)), 0..MAX_LEN)) {
                    $case!(input, SoA4, (u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32), ArityScalarToTuple4);
                }

                #[test]
                fn arity_5(input in prop::collection::vec(any::<u32>().prop_map(|v| (v,)), 0..MAX_LEN)) {
                    $case!(input, SoA5, (u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32), ArityScalarToTuple5);
                }

                #[test]
                fn arity_6(input in prop::collection::vec(any::<u32>().prop_map(|v| (v,)), 0..MAX_LEN)) {
                    $case!(input, SoA6, (u32, u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32), ArityScalarToTuple6);
                }

                #[test]
                fn arity_7(input in prop::collection::vec(any::<u32>().prop_map(|v| (v,)), 0..MAX_LEN)) {
                    $case!(input, SoA7, (u32, u32, u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32), ArityScalarToTuple7);
                }
            }
        }
    };
}

define_unary_input_arity_module!(map_input_arity, map_input_case);
define_unary_output_arity_module!(map_output_arity, map_output_case);
define_unary_input_arity_module!(transform_input_arity, transform_input_case);
define_unary_output_arity_module!(transform_output_arity, transform_output_case);
define_unary_input_arity_module!(transform_where_input_arity, transform_where_input_case);
define_unary_output_arity_module!(transform_where_output_arity, transform_where_output_case);

mod sort_by_key_key_arity {
    use super::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(CASES))]

        #[test]
        fn arity_1(pairs in prop::collection::vec((any::<u32>().prop_map(|k| (k,)), any::<u32>().prop_map(|v| (v,))), 0..MAX_LEN)) {
            sort_by_key_only_case!(pairs, SoA1, (u32), SoA1, (u32), LessU32);
        }

        #[test]
        fn arity_2(pairs in prop::collection::vec(((any::<u32>(), any::<u32>()), any::<u32>().prop_map(|v| (v,))), 0..MAX_LEN)) {
            sort_by_key_only_case!(pairs, SoA2, (u32, u32), SoA1, (u32), Less2);
        }

        #[test]
        fn arity_3(pairs in prop::collection::vec(((any::<u32>(), any::<u32>(), any::<u32>()), any::<u32>().prop_map(|v| (v,))), 0..MAX_LEN)) {
            sort_by_key_only_case!(pairs, SoA3, (u32, u32, u32), SoA1, (u32), Less3);
        }
    }
}

mod sort_by_key_value_arity {
    use super::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(CASES))]

        #[test]
        fn arity_1(pairs in prop::collection::vec((any::<u32>().prop_map(|k| (k,)), any::<u32>().prop_map(|v| (v,))), 0..MAX_LEN)) {
            sort_by_key_only_case!(pairs, SoA1, (u32), SoA1, (u32), LessU32);
        }

        #[test]
        fn arity_2(pairs in prop::collection::vec((any::<u32>().prop_map(|k| (k,)), (any::<u32>(), any::<u32>())), 0..MAX_LEN)) {
            sort_by_key_only_case!(pairs, SoA1, (u32), SoA2, (u32, u32), LessU32);
        }

        #[test]
        fn arity_3(pairs in prop::collection::vec((any::<u32>().prop_map(|k| (k,)), (any::<u32>(), any::<u32>(), any::<u32>())), 0..MAX_LEN)) {
            sort_by_key_only_case!(pairs, SoA1, (u32), SoA3, (u32, u32, u32), LessU32);
        }

        #[test]
        fn arity_4(pairs in prop::collection::vec((any::<u32>().prop_map(|k| (k,)), (any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>())), 0..MAX_LEN)) {
            sort_by_key_only_case!(pairs, SoA1, (u32), SoA4, (u32, u32, u32, u32), LessU32);
        }

        #[test]
        fn arity_5(pairs in prop::collection::vec((any::<u32>().prop_map(|k| (k,)), (any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>())), 0..MAX_LEN)) {
            sort_by_key_only_case!(pairs, SoA1, (u32), SoA5, (u32, u32, u32, u32, u32), LessU32);
        }

        #[test]
        fn arity_6(pairs in prop::collection::vec((any::<u32>().prop_map(|k| (k,)), (any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>())), 0..MAX_LEN)) {
            sort_by_key_only_case!(pairs, SoA1, (u32), SoA6, (u32, u32, u32, u32, u32, u32), LessU32);
        }

        #[test]
        fn arity_7(pairs in prop::collection::vec((any::<u32>().prop_map(|k| (k,)), (any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>())), 0..MAX_LEN)) {
            sort_by_key_only_case!(pairs, SoA1, (u32), SoA7, (u32, u32, u32, u32, u32, u32, u32), LessU32);
        }
    }
}

mod stable_sort_by_key_key_arity {
    use super::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(CASES))]

        #[test]
        fn arity_1(pairs in prop::collection::vec((any::<u32>().prop_map(|k| (k,)), any::<u32>().prop_map(|v| (v,))), 0..MAX_LEN)) {
            stable_sort_by_key_case!(pairs, SoA1, (u32), SoA1, (u32), LessU32);
        }

        #[test]
        fn arity_2(pairs in prop::collection::vec(((any::<u32>(), any::<u32>()), any::<u32>().prop_map(|v| (v,))), 0..MAX_LEN)) {
            stable_sort_by_key_case!(pairs, SoA2, (u32, u32), SoA1, (u32), Less2);
        }

        #[test]
        fn arity_3(pairs in prop::collection::vec(((any::<u32>(), any::<u32>(), any::<u32>()), any::<u32>().prop_map(|v| (v,))), 0..MAX_LEN)) {
            stable_sort_by_key_case!(pairs, SoA3, (u32, u32, u32), SoA1, (u32), Less3);
        }
    }
}

mod stable_sort_by_key_value_arity {
    use super::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(CASES))]

        #[test]
        fn arity_1(pairs in prop::collection::vec((any::<u32>().prop_map(|k| (k,)), any::<u32>().prop_map(|v| (v,))), 0..MAX_LEN)) {
            stable_sort_by_key_case!(pairs, SoA1, (u32), SoA1, (u32), LessU32);
        }

        #[test]
        fn arity_2(pairs in prop::collection::vec((any::<u32>().prop_map(|k| (k,)), (any::<u32>(), any::<u32>())), 0..MAX_LEN)) {
            stable_sort_by_key_case!(pairs, SoA1, (u32), SoA2, (u32, u32), LessU32);
        }

        #[test]
        fn arity_3(pairs in prop::collection::vec((any::<u32>().prop_map(|k| (k,)), (any::<u32>(), any::<u32>(), any::<u32>())), 0..MAX_LEN)) {
            stable_sort_by_key_case!(pairs, SoA1, (u32), SoA3, (u32, u32, u32), LessU32);
        }

        #[test]
        fn arity_4(pairs in prop::collection::vec((any::<u32>().prop_map(|k| (k,)), (any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>())), 0..MAX_LEN)) {
            stable_sort_by_key_case!(pairs, SoA1, (u32), SoA4, (u32, u32, u32, u32), LessU32);
        }

        #[test]
        fn arity_5(pairs in prop::collection::vec((any::<u32>().prop_map(|k| (k,)), (any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>())), 0..MAX_LEN)) {
            stable_sort_by_key_case!(pairs, SoA1, (u32), SoA5, (u32, u32, u32, u32, u32), LessU32);
        }

        #[test]
        fn arity_6(pairs in prop::collection::vec((any::<u32>().prop_map(|k| (k,)), (any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>())), 0..MAX_LEN)) {
            stable_sort_by_key_case!(pairs, SoA1, (u32), SoA6, (u32, u32, u32, u32, u32, u32), LessU32);
        }

        #[test]
        fn arity_7(pairs in prop::collection::vec((any::<u32>().prop_map(|k| (k,)), (any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>(), any::<u32>())), 0..MAX_LEN)) {
            stable_sort_by_key_case!(pairs, SoA1, (u32), SoA7, (u32, u32, u32, u32, u32, u32, u32), LessU32);
        }
    }
}

macro_rules! define_by_key_key_arity_module {
    ($module:ident, $case:ident) => {
        mod $module {
            use super::*;

            proptest! {
                #![proptest_config(ProptestConfig::with_cases(CASES))]

                #[test]
                fn arity_1(pairs in prop::collection::vec((0_u32..16).prop_map(|k| (k,)).prop_flat_map(|k| Just(k)), 0..MAX_LEN).prop_map(|keys| keys.into_iter().map(|key| (key, (key.0,))).collect::<Vec<_>>())) {
                    scan_by_key_case!($case, pairs, SoA1, (u32), SoA1, (u32), (0_u32,));
                }

                #[test]
                fn arity_2(pairs in prop::collection::vec((0_u32..16, 0_u32..16), 0..MAX_LEN).prop_map(|keys| keys.into_iter().map(|key| (key, (key.1,))).collect::<Vec<_>>())) {
                    scan_by_key_case!($case, pairs, SoA2, (u32, u32), SoA1, (u32), (0_u32,));
                }

                #[test]
                fn arity_3(pairs in prop::collection::vec((0_u32..16, 0_u32..16, 0_u32..16), 0..MAX_LEN).prop_map(|keys| keys.into_iter().map(|key| (key, (key.1,))).collect::<Vec<_>>())) {
                    scan_by_key_case!($case, pairs, SoA3, (u32, u32, u32), SoA1, (u32), (0_u32,));
                }
            }
        }
    };
}

macro_rules! define_by_key_value_arity_module {
    ($module:ident, $case:ident) => {
        mod $module {
            use super::*;

            proptest! {
                #![proptest_config(ProptestConfig::with_cases(CASES))]

                #[test]
                fn arity_1(pairs in prop::collection::vec((0_u32..16, 0_u32..4096), 0..MAX_LEN).prop_map(|pairs| pairs.into_iter().map(|(key, value)| ((key,), (value,))).collect::<Vec<_>>())) {
                    scan_by_key_case!($case, pairs, SoA1, (u32), SoA1, (u32), (0_u32,));
                }

                #[test]
                fn arity_2(pairs in prop::collection::vec((0_u32..16, (0_u32..4096, 0_u32..4096)), 0..MAX_LEN).prop_map(|pairs| pairs.into_iter().map(|(key, value)| ((key,), value)).collect::<Vec<_>>())) {
                    scan_by_key_case!($case, pairs, SoA1, (u32), SoA2, (u32, u32), (0_u32, 0_u32));
                }

                #[test]
                fn arity_3(pairs in prop::collection::vec((0_u32..16, (0_u32..4096, 0_u32..4096, 0_u32..4096)), 0..MAX_LEN).prop_map(|pairs| pairs.into_iter().map(|(key, value)| ((key,), value)).collect::<Vec<_>>())) {
                    scan_by_key_case!($case, pairs, SoA1, (u32), SoA3, (u32, u32, u32), (0_u32, 0_u32, 0_u32));
                }

                #[test]
                fn arity_4(pairs in prop::collection::vec((0_u32..16, (0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096)), 0..MAX_LEN).prop_map(|pairs| pairs.into_iter().map(|(key, value)| ((key,), value)).collect::<Vec<_>>())) {
                    scan_by_key_case!($case, pairs, SoA1, (u32), SoA4, (u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32));
                }

                #[test]
                fn arity_5(pairs in prop::collection::vec((0_u32..16, (0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096)), 0..MAX_LEN).prop_map(|pairs| pairs.into_iter().map(|(key, value)| ((key,), value)).collect::<Vec<_>>())) {
                    scan_by_key_case!($case, pairs, SoA1, (u32), SoA5, (u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32));
                }

                #[test]
                fn arity_6(pairs in prop::collection::vec((0_u32..16, (0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096)), 0..MAX_LEN).prop_map(|pairs| pairs.into_iter().map(|(key, value)| ((key,), value)).collect::<Vec<_>>())) {
                    scan_by_key_case!($case, pairs, SoA1, (u32), SoA6, (u32, u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32));
                }

                #[test]
                fn arity_7(pairs in prop::collection::vec((0_u32..16, (0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096)), 0..MAX_LEN).prop_map(|pairs| pairs.into_iter().map(|(key, value)| ((key,), value)).collect::<Vec<_>>())) {
                    scan_by_key_case!($case, pairs, SoA1, (u32), SoA7, (u32, u32, u32, u32, u32, u32, u32), (0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32, 0_u32));
                }
            }
        }
    };
}

define_by_key_key_arity_module!(inclusive_scan_by_key_key_arity, inclusive_scan_by_key);
define_by_key_key_arity_module!(exclusive_scan_by_key_key_arity, exclusive_scan_by_key);
define_by_key_key_arity_module!(reduce_by_key_key_arity, reduce_by_key);
define_by_key_key_arity_module!(unique_by_key_key_arity, unique_by_key);
define_by_key_value_arity_module!(inclusive_scan_by_key_value_arity, inclusive_scan_by_key);
define_by_key_value_arity_module!(exclusive_scan_by_key_value_arity, exclusive_scan_by_key);
define_by_key_value_arity_module!(reduce_by_key_value_arity, reduce_by_key);
define_by_key_value_arity_module!(unique_by_key_value_arity, unique_by_key);

mod merge_by_key_key_arity {
    use super::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(CASES))]

        #[test]
        fn arity_1(pairs in prop::collection::vec((0_u32..4096, 0_u32..4096), 0..MAX_LEN).prop_map(|pairs| pairs.into_iter().map(|(key, value)| ((key,), (value,))).collect::<Vec<_>>())) {
            merge_by_key_case!(pairs, SoA1, (u32), SoA1, (u32), LessU32);
        }

        #[test]
        fn arity_2(pairs in prop::collection::vec(((0_u32..4096, 0_u32..4096), 0_u32..4096), 0..MAX_LEN).prop_map(|pairs| pairs.into_iter().map(|(key, value)| (key, (value,))).collect::<Vec<_>>())) {
            merge_by_key_case!(pairs, SoA2, (u32, u32), SoA1, (u32), Less2);
        }

        #[test]
        fn arity_3(pairs in prop::collection::vec(((0_u32..4096, 0_u32..4096, 0_u32..4096), 0_u32..4096), 0..MAX_LEN).prop_map(|pairs| pairs.into_iter().map(|(key, value)| (key, (value,))).collect::<Vec<_>>())) {
            merge_by_key_case!(pairs, SoA3, (u32, u32, u32), SoA1, (u32), Less3);
        }
    }
}

mod merge_by_key_value_arity {
    use super::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(CASES))]

        #[test]
        fn arity_1(pairs in prop::collection::vec((0_u32..4096, 0_u32..4096), 0..MAX_LEN).prop_map(|pairs| pairs.into_iter().map(|(key, value)| ((key,), (value,))).collect::<Vec<_>>())) {
            merge_by_key_case!(pairs, SoA1, (u32), SoA1, (u32), LessU32);
        }

        #[test]
        fn arity_2(pairs in prop::collection::vec((0_u32..4096, (0_u32..4096, 0_u32..4096)), 0..MAX_LEN).prop_map(|pairs| pairs.into_iter().map(|(key, value)| ((key,), value)).collect::<Vec<_>>())) {
            merge_by_key_case!(pairs, SoA1, (u32), SoA2, (u32, u32), LessU32);
        }

        #[test]
        fn arity_3(pairs in prop::collection::vec((0_u32..4096, (0_u32..4096, 0_u32..4096, 0_u32..4096)), 0..MAX_LEN).prop_map(|pairs| pairs.into_iter().map(|(key, value)| ((key,), value)).collect::<Vec<_>>())) {
            merge_by_key_case!(pairs, SoA1, (u32), SoA3, (u32, u32, u32), LessU32);
        }

        #[test]
        fn arity_4(pairs in prop::collection::vec((0_u32..4096, (0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096)), 0..MAX_LEN).prop_map(|pairs| pairs.into_iter().map(|(key, value)| ((key,), value)).collect::<Vec<_>>())) {
            merge_by_key_case!(pairs, SoA1, (u32), SoA4, (u32, u32, u32, u32), LessU32);
        }

        #[test]
        fn arity_5(pairs in prop::collection::vec((0_u32..4096, (0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096)), 0..MAX_LEN).prop_map(|pairs| pairs.into_iter().map(|(key, value)| ((key,), value)).collect::<Vec<_>>())) {
            merge_by_key_case!(pairs, SoA1, (u32), SoA5, (u32, u32, u32, u32, u32), LessU32);
        }

        #[test]
        fn arity_6(pairs in prop::collection::vec((0_u32..4096, (0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096)), 0..MAX_LEN).prop_map(|pairs| pairs.into_iter().map(|(key, value)| ((key,), value)).collect::<Vec<_>>())) {
            merge_by_key_case!(pairs, SoA1, (u32), SoA6, (u32, u32, u32, u32, u32, u32), LessU32);
        }

        #[test]
        fn arity_7(pairs in prop::collection::vec((0_u32..4096, (0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096, 0_u32..4096)), 0..MAX_LEN).prop_map(|pairs| pairs.into_iter().map(|(key, value)| ((key,), value)).collect::<Vec<_>>())) {
            merge_by_key_case!(pairs, SoA1, (u32), SoA7, (u32, u32, u32, u32, u32, u32, u32), LessU32);
        }
    }
}
