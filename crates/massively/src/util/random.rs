//! Lazy GPU-side random value generation.

use cubecl::prelude::*;

use crate::{Error, MIndex, MIter, lazy};

fn validate_inclusive_range<T>(min: T, max: T) -> Result<(), Error>
where
    T: PartialOrd,
{
    if min <= max {
        Ok(())
    } else {
        Err(Error::Launch {
            message: "random distribution range must satisfy min <= max".to_string(),
        })
    }
}

/// Creates a lazy stream of deterministic uniformly distributed `u32` values in `[min, max]`.
pub fn uniform_u32(min: u32, max: u32, seed: u64) -> Result<UniformU32, Error> {
    validate_inclusive_range(min, max)?;
    Ok(UniformU32 { min, max, seed })
}

/// Creates a lazy stream of deterministic uniformly distributed `u64` values in `[min, max]`.
pub fn uniform_u64(min: u64, max: u64, seed: u64) -> Result<UniformU64, Error> {
    validate_inclusive_range(min, max)?;
    Ok(UniformU64 { min, max, seed })
}

/// Creates a lazy stream of deterministic uniformly distributed `f32` values in `[min, max]`.
pub fn uniform_f32(min: f32, max: f32, seed: u64) -> Result<UniformF32, Error> {
    validate_inclusive_range(min, max)?;
    Ok(UniformF32 { min, max, seed })
}

/// Creates a lazy stream of deterministic uniformly distributed `f64` values in `[min, max]`.
pub fn uniform_f64(min: f64, max: f64, seed: u64) -> Result<UniformF64, Error> {
    validate_inclusive_range(min, max)?;
    Ok(UniformF64 { min, max, seed })
}

/// Creates a lazy stream of deterministic approximately normally distributed `f32` values.
pub fn normal_f32(mean: f32, stddev: f32, seed: u64) -> NormalF32 {
    NormalF32 { mean, stddev, seed }
}

/// Creates a lazy stream of deterministic approximately normally distributed `f64` values.
pub fn normal_f64(mean: f64, stddev: f64, seed: u64) -> NormalF64 {
    NormalF64 { mean, stddev, seed }
}

/// Infinite random stream for `uniform_u32`.
#[derive(Clone, Copy, Debug)]
pub struct UniformU32 {
    min: u32,
    max: u32,
    seed: u64,
}

/// Infinite random stream for `uniform_u64`.
#[derive(Clone, Copy, Debug)]
pub struct UniformU64 {
    min: u64,
    max: u64,
    seed: u64,
}

/// Infinite random stream for `uniform_f32`.
#[derive(Clone, Copy, Debug)]
pub struct UniformF32 {
    min: f32,
    max: f32,
    seed: u64,
}

/// Infinite random stream for `uniform_f64`.
#[derive(Clone, Copy, Debug)]
pub struct UniformF64 {
    min: f64,
    max: f64,
    seed: u64,
}

/// Infinite random stream for `normal_f32`.
#[derive(Clone, Copy, Debug)]
pub struct NormalF32 {
    mean: f32,
    stddev: f32,
    seed: u64,
}

/// Infinite random stream for `normal_f64`.
#[derive(Clone, Copy, Debug)]
pub struct NormalF64 {
    mean: f64,
    stddev: f64,
    seed: u64,
}

macro_rules! impl_take {
    ($( $stream:ident ),+ $(,)?) => {
        $(
            impl $stream {
                /// Turns this lazy stream into a finite read-only iterator.
                pub fn take(self, len: MIndex) -> lazy::Taken<Self> {
                    lazy::Taken::new(self, len)
                }
            }
        )+
    };
}

impl_take!(
    UniformU32, UniformU64, UniformF32, UniformF64, NormalF32, NormalF64
);

#[doc(hidden)]
pub struct UniformU32Op;
#[doc(hidden)]
pub struct UniformU64Op;
#[doc(hidden)]
pub struct UniformF32Op;
#[doc(hidden)]
pub struct UniformF64Op;
#[doc(hidden)]
pub struct NormalF32Op;
#[doc(hidden)]
pub struct NormalF64Op;

fn counting_read<R>(
    len: MIndex,
    policy: &crate::detail::CubePolicy<R>,
) -> crate::detail::read::CountingRead
where
    R: Runtime,
{
    let handle = policy.client().create_from_slice(MIndex::as_bytes(&[0]));
    crate::detail::read::CountingRead::new(handle, len as usize)
}

macro_rules! impl_random_miter {
    ($stream:ident { $a:ident, $b:ident } => $op:ident, $item:ty) => {
        impl<R> MIter<R> for lazy::Taken<$stream>
        where
            R: Runtime,
            $item: lazy::ConstantItem<R>,
            u64: lazy::ConstantItem<R>,
            crate::detail::read::ZipRead4<
                crate::detail::read::CountingRead,
                <$item as lazy::ConstantItem<R>>::Read,
                <$item as lazy::ConstantItem<R>>::Read,
                <u64 as lazy::ConstantItem<R>>::Read,
            >: crate::detail::read::KernelReadBoundMany<R, Item = (MIndex, $item, $item, u64)>,
            crate::detail::read::TransformRead<
                crate::detail::read::ZipRead4<
                    crate::detail::read::CountingRead,
                    <$item as lazy::ConstantItem<R>>::Read,
                    <$item as lazy::ConstantItem<R>>::Read,
                    <u64 as lazy::ConstantItem<R>>::Read,
                >,
                $op,
            >: crate::detail::read::KernelReadBoundMany<R, Item = $item>,
        {
            type Item = $item;
            type Inner = ();
            type Read = crate::detail::read::TransformRead<
                crate::detail::read::ZipRead4<
                    crate::detail::read::CountingRead,
                    <$item as lazy::ConstantItem<R>>::Read,
                    <$item as lazy::ConstantItem<R>>::Read,
                    <u64 as lazy::ConstantItem<R>>::Read,
                >,
                $op,
            >;

            fn len(&self) -> MIndex {
                self.len
            }

            fn into_inner(self) -> Self::Inner {
                unreachable!("lazy random MIter has no storage inner")
            }

            fn lower_read_ref(
                &self,
                policy: &crate::detail::CubePolicy<R>,
            ) -> Result<Self::Read, Error> {
                let len = self.len;
                let input = crate::detail::read::ZipRead4::new(
                    counting_read::<R>(len, policy),
                    <$item as lazy::ConstantItem<R>>::lower_constant_read(
                        self.expr.$a,
                        len,
                        policy,
                    )?,
                    <$item as lazy::ConstantItem<R>>::lower_constant_read(
                        self.expr.$b,
                        len,
                        policy,
                    )?,
                    <u64 as lazy::ConstantItem<R>>::lower_constant_read(
                        self.expr.seed,
                        len,
                        policy,
                    )?,
                );
                Ok(crate::detail::read::TransformRead::new(input))
            }

            fn validate_executor(&self, _exec: &crate::runtime::Executor<R>) -> Result<(), Error> {
                Ok(())
            }
        }
    };
}

impl_random_miter!(UniformU32 { min, max } => UniformU32Op, u32);
impl_random_miter!(UniformU64 { min, max } => UniformU64Op, u64);
impl_random_miter!(UniformF32 { min, max } => UniformF32Op, f32);
impl_random_miter!(UniformF64 { min, max } => UniformF64Op, f64);
impl_random_miter!(NormalF32 { mean, stddev } => NormalF32Op, f32);
impl_random_miter!(NormalF64 { mean, stddev } => NormalF64Op, f64);

#[cube]
impl<R> crate::op::UnaryOp<R, (MIndex, u32, u32, u64)> for UniformU32Op
where
    R: Runtime,
{
    type Output = u32;

    fn apply(input: (MIndex, u32, u32, u64)) -> u32 {
        uniform_u32_value(input.1, input.2, input.3, input.0)
    }
}

#[cube]
impl<R> crate::op::UnaryOp<R, (MIndex, u64, u64, u64)> for UniformU64Op
where
    R: Runtime,
{
    type Output = u64;

    fn apply(input: (MIndex, u64, u64, u64)) -> u64 {
        uniform_u64_value(input.1, input.2, input.3, input.0)
    }
}

#[cube]
impl<R> crate::op::UnaryOp<R, (MIndex, f32, f32, u64)> for UniformF32Op
where
    R: Runtime,
{
    type Output = f32;

    fn apply(input: (MIndex, f32, f32, u64)) -> f32 {
        uniform_f32_value(input.1, input.2, input.3, input.0)
    }
}

#[cube]
impl<R> crate::op::UnaryOp<R, (MIndex, f64, f64, u64)> for UniformF64Op
where
    R: Runtime,
{
    type Output = f64;

    fn apply(input: (MIndex, f64, f64, u64)) -> f64 {
        uniform_f64_value(input.1, input.2, input.3, input.0)
    }
}

#[cube]
impl<R> crate::op::UnaryOp<R, (MIndex, f32, f32, u64)> for NormalF32Op
where
    R: Runtime,
{
    type Output = f32;

    fn apply(input: (MIndex, f32, f32, u64)) -> f32 {
        normal_f32_value(input.1, input.2, input.3, input.0)
    }
}

#[cube]
impl<R> crate::op::UnaryOp<R, (MIndex, f64, f64, u64)> for NormalF64Op
where
    R: Runtime,
{
    type Output = f64;

    fn apply(input: (MIndex, f64, f64, u64)) -> f64 {
        normal_f64_value(input.1, input.2, input.3, input.0)
    }
}

#[cube]
fn splitmix64(mut state: u64) -> u64 {
    state += 0x9e37_79b9_7f4a_7c15u64;
    let z = RuntimeCell::<u64>::new(state);
    z.store((z.read() ^ (z.read() >> 30u64)) * 0xbf58_476d_1ce4_e5b9u64);
    z.store((z.read() ^ (z.read() >> 27u64)) * 0x94d0_49bb_1331_11ebu64);
    z.read() ^ (z.read() >> 31u64)
}

#[cube]
fn random_u32_at(seed: u64, index: MIndex, stream: u64) -> u32 {
    splitmix64(seed + (index as u64) * 0x9e37_79b9_7f4a_7c15u64 + stream) as u32
}

#[cube]
fn random_u64_at(seed: u64, index: MIndex, stream: u64) -> u64 {
    splitmix64(seed + (index as u64) * 0x9e37_79b9_7f4a_7c15u64 + stream)
}

#[cube]
fn unit_f32(seed: u64, index: MIndex, stream: u64) -> f32 {
    ((random_u32_at(seed, index, stream) >> 8u32) as f32) * 0.00000005960464832810486063f32
}

#[cube]
fn unit_f64(seed: u64, index: MIndex, stream: u64) -> f64 {
    ((random_u64_at(seed, index, stream) >> 11u64) as f64) * 0.00000000000000011102230246251567f64
}

#[cube]
fn open_unit_f32(seed: u64, index: MIndex, stream: u64) -> f32 {
    (((random_u32_at(seed, index, stream) >> 8u32) as f32) + 0.5f32)
        * 0.00000005960464832810486063f32
}

#[cube]
fn uniform_f32_value(min: f32, max: f32, seed: u64, i: MIndex) -> f32 {
    min + unit_f32(seed, i, 0u64) * (max - min)
}

#[cube]
fn uniform_f64_value(min: f64, max: f64, seed: u64, i: MIndex) -> f64 {
    min + unit_f64(seed, i, 0u64) * (max - min)
}

#[cube]
fn uniform_u32_value(min: u32, max: u32, seed: u64, i: MIndex) -> u32 {
    let span = max - min;
    let value = random_u32_at(seed, i, 0u64);
    if span == 0xffff_ffffu32 {
        value
    } else {
        min + value % (span + 1u32)
    }
}

#[cube]
fn uniform_u64_value(min: u64, max: u64, seed: u64, i: MIndex) -> u64 {
    let span = max - min;
    let value = random_u64_at(seed, i, 0u64);
    if span == 0xffff_ffff_ffff_ffffu64 {
        value
    } else {
        min + value % (span + 1u64)
    }
}

#[cube]
fn normal_f32_value(mean: f32, stddev: f32, seed: u64, i: MIndex) -> f32 {
    let u1 = open_unit_f32(seed, i, 0u64);
    let u2 = open_unit_f32(seed, i, 1u64);
    let radius = (-2.0f32 * u1.ln()).sqrt();
    let angle = 6.28318530717958647692f32 * u2;
    mean + stddev * radius * angle.cos()
}

#[cube]
fn normal_f64_value(mean: f64, stddev: f64, seed: u64, i: MIndex) -> f64 {
    let u1 = open_unit_f32(seed, i, 0u64);
    let u2 = open_unit_f32(seed, i, 1u64);
    let radius = (-2.0f32 * u1.ln()).sqrt();
    let angle = 6.28318530717958647692f32 * u2;
    mean + stddev * ((radius * angle.cos()) as f64)
}
