//! GPU-side random value generation.

use cubecl::prelude::*;

use crate::{DeviceVec, Error, Executor};

const BLOCK_RANDOM_SIZE: u32 = 256;

fn random_block_count(len: usize) -> Result<u32, Error> {
    let block_count = len.div_ceil(BLOCK_RANDOM_SIZE as usize);
    u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })
}

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

/// Generates `n` uniformly distributed `u32` values in `[min, max]`.
pub fn uniform_distribution_u32<R>(
    exec: &Executor<R>,
    n: usize,
    min: u32,
    max: u32,
    seed: u64,
) -> Result<DeviceVec<R, u32>, Error>
where
    R: Runtime,
{
    validate_inclusive_range(min, max)?;
    u32::try_from(n).map_err(|_| Error::LengthTooLarge { len: n })?;
    if n == 0 {
        return Ok(DeviceVec::from_inner(exec.policy().empty_device_vec()));
    }

    let client = exec.policy().client();
    let output = client.empty(n * core::mem::size_of::<u32>());
    let params = [min, max];
    let seed = [seed];
    let params = client.create_from_slice(u32::as_bytes(&params));
    let seed = client.create_from_slice(u64::as_bytes(&seed));

    unsafe {
        random_uniform_u32_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(random_block_count(n)?, 1, 1),
            CubeDim::new_1d(BLOCK_RANDOM_SIZE),
            BufferArg::from_raw_parts(params.clone(), 2),
            BufferArg::from_raw_parts(seed.clone(), 1),
            BufferArg::from_raw_parts(output.clone(), n),
        );
    }

    Ok(DeviceVec::from_inner(
        crate::detail::DeviceVec::from_handle(exec.policy().id(), output, n),
    ))
}

/// Generates `n` uniformly distributed `u64` values in `[min, max]`.
pub fn uniform_distribution_u64<R>(
    exec: &Executor<R>,
    n: usize,
    min: u64,
    max: u64,
    seed: u64,
) -> Result<DeviceVec<R, u64>, Error>
where
    R: Runtime,
{
    validate_inclusive_range(min, max)?;
    u32::try_from(n).map_err(|_| Error::LengthTooLarge { len: n })?;
    if n == 0 {
        return Ok(DeviceVec::from_inner(exec.policy().empty_device_vec()));
    }

    let client = exec.policy().client();
    let output = client.empty(n * core::mem::size_of::<u64>());
    let params = [min, max, seed];
    let params = client.create_from_slice(u64::as_bytes(&params));

    unsafe {
        random_uniform_u64_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(random_block_count(n)?, 1, 1),
            CubeDim::new_1d(BLOCK_RANDOM_SIZE),
            BufferArg::from_raw_parts(params.clone(), 3),
            BufferArg::from_raw_parts(output.clone(), n),
        );
    }

    Ok(DeviceVec::from_inner(
        crate::detail::DeviceVec::from_handle(exec.policy().id(), output, n),
    ))
}

/// Generates `n` uniformly distributed `f32` values in `[0, 1]`.
pub fn uniform_distribution_f32<R>(
    exec: &Executor<R>,
    n: usize,
    seed: u64,
) -> Result<DeviceVec<R, f32>, Error>
where
    R: Runtime,
{
    u32::try_from(n).map_err(|_| Error::LengthTooLarge { len: n })?;
    if n == 0 {
        return Ok(DeviceVec::from_inner(exec.policy().empty_device_vec()));
    }

    let client = exec.policy().client();
    let output = client.empty(n * core::mem::size_of::<f32>());
    let seed = [seed];
    let seed = client.create_from_slice(u64::as_bytes(&seed));

    unsafe {
        random_uniform_f32_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(random_block_count(n)?, 1, 1),
            CubeDim::new_1d(BLOCK_RANDOM_SIZE),
            BufferArg::from_raw_parts(seed.clone(), 1),
            BufferArg::from_raw_parts(output.clone(), n),
        );
    }

    Ok(DeviceVec::from_inner(
        crate::detail::DeviceVec::from_handle(exec.policy().id(), output, n),
    ))
}

/// Generates `n` uniformly distributed `f64` values in `[0, 1]`.
pub fn uniform_distribution_f64<R>(
    exec: &Executor<R>,
    n: usize,
    seed: u64,
) -> Result<DeviceVec<R, f64>, Error>
where
    R: Runtime,
{
    u32::try_from(n).map_err(|_| Error::LengthTooLarge { len: n })?;
    if n == 0 {
        return Ok(DeviceVec::from_inner(exec.policy().empty_device_vec()));
    }

    let client = exec.policy().client();
    let output = client.empty(n * core::mem::size_of::<f64>());
    let seed = [seed];
    let seed = client.create_from_slice(u64::as_bytes(&seed));

    unsafe {
        random_uniform_f64_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(random_block_count(n)?, 1, 1),
            CubeDim::new_1d(BLOCK_RANDOM_SIZE),
            BufferArg::from_raw_parts(seed.clone(), 1),
            BufferArg::from_raw_parts(output.clone(), n),
        );
    }

    Ok(DeviceVec::from_inner(
        crate::detail::DeviceVec::from_handle(exec.policy().id(), output, n),
    ))
}

/// Generates `n` approximately normally distributed `f32` values.
pub fn normal_distribution_f32<R>(
    exec: &Executor<R>,
    n: usize,
    mean: f32,
    stddev: f32,
    seed: u64,
) -> Result<DeviceVec<R, f32>, Error>
where
    R: Runtime,
{
    u32::try_from(n).map_err(|_| Error::LengthTooLarge { len: n })?;
    if n == 0 {
        return Ok(DeviceVec::from_inner(exec.policy().empty_device_vec()));
    }

    let client = exec.policy().client();
    let output = client.empty(n * core::mem::size_of::<f32>());
    let params = [mean, stddev];
    let seed = [seed];
    let params = client.create_from_slice(f32::as_bytes(&params));
    let seed = client.create_from_slice(u64::as_bytes(&seed));

    unsafe {
        random_normal_f32_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(random_block_count(n)?, 1, 1),
            CubeDim::new_1d(BLOCK_RANDOM_SIZE),
            BufferArg::from_raw_parts(params.clone(), 2),
            BufferArg::from_raw_parts(seed.clone(), 1),
            BufferArg::from_raw_parts(output.clone(), n),
        );
    }

    Ok(DeviceVec::from_inner(
        crate::detail::DeviceVec::from_handle(exec.policy().id(), output, n),
    ))
}

/// Generates `n` approximately normally distributed `f64` values.
pub fn normal_distribution_f64<R>(
    exec: &Executor<R>,
    n: usize,
    mean: f64,
    stddev: f64,
    seed: u64,
) -> Result<DeviceVec<R, f64>, Error>
where
    R: Runtime,
{
    u32::try_from(n).map_err(|_| Error::LengthTooLarge { len: n })?;
    if n == 0 {
        return Ok(DeviceVec::from_inner(exec.policy().empty_device_vec()));
    }

    let client = exec.policy().client();
    let output = client.empty(n * core::mem::size_of::<f64>());
    let params = [mean, stddev];
    let seed = [seed];
    let params = client.create_from_slice(f64::as_bytes(&params));
    let seed = client.create_from_slice(u64::as_bytes(&seed));

    unsafe {
        random_normal_f64_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(random_block_count(n)?, 1, 1),
            CubeDim::new_1d(BLOCK_RANDOM_SIZE),
            BufferArg::from_raw_parts(params.clone(), 2),
            BufferArg::from_raw_parts(seed.clone(), 1),
            BufferArg::from_raw_parts(output.clone(), n),
        );
    }

    Ok(DeviceVec::from_inner(
        crate::detail::DeviceVec::from_handle(exec.policy().id(), output, n),
    ))
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
fn random_u32_at(seed: u64, index: u32, stream: u64) -> u32 {
    splitmix64(seed + (index as u64) * 0x9e37_79b9_7f4a_7c15u64 + stream) as u32
}

#[cube]
fn random_u64_at(seed: u64, index: u32, stream: u64) -> u64 {
    splitmix64(seed + (index as u64) * 0x9e37_79b9_7f4a_7c15u64 + stream)
}

#[cube]
fn unit_f32(seed: u64, index: u32, stream: u64) -> f32 {
    ((random_u32_at(seed, index, stream) >> 8u32) as f32) * 0.00000005960464832810486063f32
}

#[cube]
fn unit_f64(seed: u64, index: u32, stream: u64) -> f64 {
    ((random_u64_at(seed, index, stream) >> 11u64) as f64) * 0.00000000000000011102230246251567f64
}

#[cube]
fn open_unit_f32(seed: u64, index: u32, stream: u64) -> f32 {
    (((random_u32_at(seed, index, stream) >> 8u32) as f32) + 0.5f32)
        * 0.00000005960464832810486063f32
}

#[cube]
fn open_unit_f64(seed: u64, index: u32, stream: u64) -> f64 {
    (((random_u64_at(seed, index, stream) >> 11u64) as f64) + 0.5f64)
        * 0.00000000000000011102230246251567f64
}

/// Returns a deterministic uniformly distributed `f32` value in `[0, 1]`.
#[cube]
pub fn uniform_f32(seed: u64, i: u32) -> f32 {
    unit_f32(seed, i, 0u64)
}

/// Returns a deterministic uniformly distributed `f64` value in `[0, 1]`.
#[cube]
pub fn uniform_f64(seed: u64, i: u32) -> f64 {
    unit_f64(seed, i, 0u64)
}

/// Returns a deterministic uniformly distributed `u32` value in `[min, max]`.
#[cube]
pub fn uniform_u32(min: u32, max: u32, seed: u64, i: u32) -> u32 {
    let span = max - min;
    let value = random_u32_at(seed, i, 0u64);
    if span == 0xffff_ffffu32 {
        value
    } else {
        min + value % (span + 1u32)
    }
}

/// Returns a deterministic uniformly distributed `u64` value in `[min, max]`.
#[cube]
pub fn uniform_u64(min: u64, max: u64, seed: u64, i: u32) -> u64 {
    let span = max - min;
    let value = random_u64_at(seed, i, 0u64);
    if span == 0xffff_ffff_ffff_ffffu64 {
        value
    } else {
        min + value % (span + 1u64)
    }
}

/// Returns a deterministic normally distributed `f32` value.
#[cube]
pub fn normal_f32(mean: f32, stddev: f32, seed: u64, i: u32) -> f32 {
    let u1 = open_unit_f32(seed, i, 0u64);
    let u2 = open_unit_f32(seed, i, 1u64);
    let radius = (-2.0f32 * u1.ln()).sqrt();
    let angle = 6.28318530717958647692f32 * u2;
    mean + stddev * radius * angle.cos()
}

/// Returns a deterministic normally distributed `f64` value.
#[cube]
pub fn normal_f64(mean: f64, stddev: f64, seed: u64, i: u32) -> f64 {
    let u1 = open_unit_f32(seed, i, 0u64);
    let u2 = open_unit_f32(seed, i, 1u64);
    let radius = (-2.0f32 * u1.ln()).sqrt();
    let angle = 6.28318530717958647692f32 * u2;
    mean + stddev * ((radius * angle.cos()) as f64)
}

#[cube(launch_unchecked, explicit_define)]
fn random_uniform_u32_kernel(params: &[u32], seed: &[u64], output: &mut [u32]) {
    let unit = (CUBE_POS as usize) * (CUBE_DIM as usize) + (UNIT_POS as usize);
    if unit < output.len() {
        output[unit] = uniform_u32(params[0], params[1], seed[0], unit as u32);
    }
}

#[cube(launch_unchecked, explicit_define)]
fn random_uniform_u64_kernel(params: &[u64], output: &mut [u64]) {
    let unit = (CUBE_POS as usize) * (CUBE_DIM as usize) + (UNIT_POS as usize);
    if unit < output.len() {
        output[unit] = uniform_u64(params[0], params[1], params[2], unit as u32);
    }
}

#[cube(launch_unchecked, explicit_define)]
fn random_uniform_f32_kernel(seed: &[u64], output: &mut [f32]) {
    let unit = (CUBE_POS as usize) * (CUBE_DIM as usize) + (UNIT_POS as usize);
    if unit < output.len() {
        output[unit] = uniform_f32(seed[0], unit as u32);
    }
}

#[cube(launch_unchecked, explicit_define)]
fn random_uniform_f64_kernel(seed: &[u64], output: &mut [f64]) {
    let unit = (CUBE_POS as usize) * (CUBE_DIM as usize) + (UNIT_POS as usize);
    if unit < output.len() {
        output[unit] = uniform_f64(seed[0], unit as u32);
    }
}

#[cube(launch_unchecked, explicit_define)]
fn random_normal_f32_kernel(params: &[f32], seed: &[u64], output: &mut [f32]) {
    let unit = (CUBE_POS as usize) * (CUBE_DIM as usize) + (UNIT_POS as usize);
    if unit < output.len() {
        output[unit] = normal_f32(params[0], params[1], seed[0], unit as u32);
    }
}

#[cube(launch_unchecked, explicit_define)]
fn random_normal_f64_kernel(params: &[f64], seed: &[u64], output: &mut [f64]) {
    let unit = (CUBE_POS as usize) * (CUBE_DIM as usize) + (UNIT_POS as usize);
    if unit < output.len() {
        output[unit] = normal_f64(params[0], params[1], seed[0], unit as u32);
    }
}
