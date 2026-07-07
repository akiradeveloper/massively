//! Lazy GPU-side random value generation.

use cubecl::prelude::*;

use crate::{Error, MIndex, MIter, Zip4, lazy};

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
pub fn uniform_u32<R>(
    n: MIndex,
    min: u32,
    max: u32,
    seed: u64,
) -> Result<impl MIter<R, Item = u32>, Error>
where
    R: Runtime,
{
    validate_inclusive_range(min, max)?;
    Ok(lazy::transform(
        Zip4(
            lazy::counting(0).take(n),
            lazy::constant(min).take(n),
            lazy::constant(max).take(n),
            lazy::constant(seed).take(n),
        ),
        UniformU32,
    ))
}

/// Creates a lazy stream of deterministic uniformly distributed `u64` values in `[min, max]`.
pub fn uniform_u64<R>(
    n: MIndex,
    min: u64,
    max: u64,
    seed: u64,
) -> Result<impl MIter<R, Item = u64>, Error>
where
    R: Runtime,
{
    validate_inclusive_range(min, max)?;
    Ok(lazy::transform(
        Zip4(
            lazy::counting(0).take(n),
            lazy::constant(min).take(n),
            lazy::constant(max).take(n),
            lazy::constant(seed).take(n),
        ),
        UniformU64,
    ))
}

/// Creates a lazy stream of deterministic uniformly distributed `f32` values in `[min, max]`.
pub fn uniform_f32<R>(
    n: MIndex,
    min: f32,
    max: f32,
    seed: u64,
) -> Result<impl MIter<R, Item = f32>, Error>
where
    R: Runtime,
{
    validate_inclusive_range(min, max)?;
    Ok(lazy::transform(
        Zip4(
            lazy::counting(0).take(n),
            lazy::constant(min).take(n),
            lazy::constant(max).take(n),
            lazy::constant(seed).take(n),
        ),
        UniformF32,
    ))
}

/// Creates a lazy stream of deterministic uniformly distributed `f64` values in `[min, max]`.
pub fn uniform_f64<R>(
    n: MIndex,
    min: f64,
    max: f64,
    seed: u64,
) -> Result<impl MIter<R, Item = f64>, Error>
where
    R: Runtime,
{
    validate_inclusive_range(min, max)?;
    Ok(lazy::transform(
        Zip4(
            lazy::counting(0).take(n),
            lazy::constant(min).take(n),
            lazy::constant(max).take(n),
            lazy::constant(seed).take(n),
        ),
        UniformF64,
    ))
}

/// Creates a lazy stream of deterministic approximately normally distributed `f32` values.
pub fn normal_f32<R>(n: MIndex, mean: f32, stddev: f32, seed: u64) -> impl MIter<R, Item = f32>
where
    R: Runtime,
{
    lazy::transform(
        Zip4(
            lazy::counting(0).take(n),
            lazy::constant(mean).take(n),
            lazy::constant(stddev).take(n),
            lazy::constant(seed).take(n),
        ),
        NormalF32,
    )
}

/// Creates a lazy stream of deterministic approximately normally distributed `f64` values.
pub fn normal_f64<R>(n: MIndex, mean: f64, stddev: f64, seed: u64) -> impl MIter<R, Item = f64>
where
    R: Runtime,
{
    lazy::transform(
        Zip4(
            lazy::counting(0).take(n),
            lazy::constant(mean).take(n),
            lazy::constant(stddev).take(n),
            lazy::constant(seed).take(n),
        ),
        NormalF64,
    )
}

struct UniformU32;
struct UniformU64;
struct UniformF32;
struct UniformF64;
struct NormalF32;
struct NormalF64;

#[cube]
impl<R> crate::op::UnaryOp<R, (MIndex, u32, u32, u64)> for UniformU32
where
    R: Runtime,
{
    type Output = u32;

    fn apply(input: (MIndex, u32, u32, u64)) -> u32 {
        uniform_u32_value(input.1, input.2, input.3, input.0)
    }
}

#[cube]
impl<R> crate::op::UnaryOp<R, (MIndex, u64, u64, u64)> for UniformU64
where
    R: Runtime,
{
    type Output = u64;

    fn apply(input: (MIndex, u64, u64, u64)) -> u64 {
        uniform_u64_value(input.1, input.2, input.3, input.0)
    }
}

#[cube]
impl<R> crate::op::UnaryOp<R, (MIndex, f32, f32, u64)> for UniformF32
where
    R: Runtime,
{
    type Output = f32;

    fn apply(input: (MIndex, f32, f32, u64)) -> f32 {
        uniform_f32_value(input.1, input.2, input.3, input.0)
    }
}

#[cube]
impl<R> crate::op::UnaryOp<R, (MIndex, f64, f64, u64)> for UniformF64
where
    R: Runtime,
{
    type Output = f64;

    fn apply(input: (MIndex, f64, f64, u64)) -> f64 {
        uniform_f64_value(input.1, input.2, input.3, input.0)
    }
}

#[cube]
impl<R> crate::op::UnaryOp<R, (MIndex, f32, f32, u64)> for NormalF32
where
    R: Runtime,
{
    type Output = f32;

    fn apply(input: (MIndex, f32, f32, u64)) -> f32 {
        normal_f32_value(input.1, input.2, input.3, input.0)
    }
}

#[cube]
impl<R> crate::op::UnaryOp<R, (MIndex, f64, f64, u64)> for NormalF64
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
