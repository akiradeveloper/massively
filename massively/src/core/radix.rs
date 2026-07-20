//! Stable least-significant-digit radix ordering controls.

use cubecl::prelude::*;

use crate::{DeviceVec, Error, Executor, MStorageElement, Zip, launch::cube_count_1d};

const BLOCK_SIZE: u32 = 256;
const RADIX_BLOCK_ITEMS: usize = 256;
const RADIX_BITS: u32 = 4;
const RADIX_BUCKETS: usize = 1usize << RADIX_BITS;
const RADIX_MASK: u32 = RADIX_BUCKETS as u32 - 1u32;

/// Returns one four-bit digit of the order-preserving radix representation.
#[cubecl::cube]
trait RadixDigitOp<T: CubePrimitive>: 'static + Send + Sync {
    fn apply(value: T, shift: u32) -> u32;
}

macro_rules! unsigned_radix_op {
    ($name:ident, $ty:ty, $wide:ty) => {
        struct $name;

        #[cubecl::cube]
        impl RadixDigitOp<$ty> for $name {
            fn apply(value: $ty, shift: u32) -> u32 {
                (((value as $wide) >> (shift as $wide)) & (RADIX_MASK as $wide)) as u32
            }
        }
    };
}

macro_rules! signed_radix_op {
    ($name:ident, $ty:ty, $wide:ty, $mask:expr, $sign:expr) => {
        struct $name;

        #[cubecl::cube]
        impl RadixDigitOp<$ty> for $name {
            fn apply(value: $ty, shift: u32) -> u32 {
                let encoded = (((value as $wide) & $mask) ^ $sign) as $wide;
                ((encoded >> (shift as $wide)) & (RADIX_MASK as $wide)) as u32
            }
        }
    };
}

unsigned_radix_op!(U8RadixZero, u8, u32);
unsigned_radix_op!(U16RadixZero, u16, u32);
unsigned_radix_op!(U32RadixZero, u32, u32);
unsigned_radix_op!(U64RadixZero, u64, u64);
signed_radix_op!(I8RadixZero, i8, u32, 0xffu32, 0x80u32);
signed_radix_op!(I16RadixZero, i16, u32, 0xffffu32, 0x8000u32);
signed_radix_op!(I32RadixZero, i32, u32, 0xffff_ffffu32, 0x8000_0000u32);
signed_radix_op!(
    I64RadixZero,
    i64,
    u64,
    0xffff_ffff_ffff_ffffu64,
    0x8000_0000_0000_0000u64
);

struct F32RadixZero;

#[cubecl::cube]
impl RadixDigitOp<f32> for F32RadixZero {
    fn apply(value: f32, shift: u32) -> u32 {
        let raw = u32::reinterpret(value);
        let encoded = if (raw & 0x8000_0000u32) != 0u32 {
            !raw
        } else {
            raw ^ 0x8000_0000u32
        };
        (encoded >> shift) & RADIX_MASK
    }
}

struct F64RadixZero;

#[cubecl::cube]
impl RadixDigitOp<f64> for F64RadixZero {
    fn apply(value: f64, shift: u32) -> u32 {
        let raw = u64::reinterpret(value);
        let encoded = if (raw & 0x8000_0000_0000_0000u64) != 0u64 {
            !raw
        } else {
            raw ^ 0x8000_0000_0000_0000u64
        };
        ((encoded >> (shift as u64)) & (RADIX_MASK as u64)) as u32
    }
}

trait RadixScalar: MStorageElement {
    const BITS: u32;
    type DigitOp: RadixDigitOp<Self>;
}

macro_rules! impl_radix_scalar {
    ($($ty:ty => ($bits:expr, $op:ty)),+ $(,)?) => {
        $(
            impl RadixScalar for $ty {
                const BITS: u32 = $bits;
                type DigitOp = $op;
            }
        )+
    };
}

impl_radix_scalar!(
    u8 => (8, U8RadixZero),
    u16 => (16, U16RadixZero),
    u32 => (32, U32RadixZero),
    u64 => (64, U64RadixZero),
    i8 => (8, I8RadixZero),
    i16 => (16, I16RadixZero),
    i32 => (32, I32RadixZero),
    i64 => (64, I64RadixZero),
    f32 => (32, F32RadixZero),
    f64 => (64, F64RadixZero),
);

#[cubecl::cube(launch_unchecked, explicit_define)]
fn block_digit_sort_kernel<T: CubePrimitive, Op: RadixDigitOp<T>>(
    keys: &[T],
    permutation: &[u32],
    logical_len_buffer: &[u32],
    params: &[u32],
    block_permutation: &mut [u32],
    local_metadata: &mut [u32],
    histograms: &mut [u32],
) {
    // params = [shift, block count, identity-input flag]
    let block = CUBE_POS as usize;
    let start = block * RADIX_BLOCK_ITEMS;
    let logical_len = logical_len_buffer[0] as usize;
    let end = if start >= logical_len {
        start
    } else if start + RADIX_BLOCK_ITEMS < logical_len {
        start + RADIX_BLOCK_ITEMS
    } else {
        logical_len
    };
    let local = UNIT_POS as usize;
    let global = start + local;
    let valid = global < end;
    let mut permutation_a = Shared::<[u32]>::new_slice(RADIX_BLOCK_ITEMS);
    let mut permutation_b = Shared::<[u32]>::new_slice(RADIX_BLOCK_ITEMS);
    let mut digits_a = Shared::<[u32]>::new_slice(RADIX_BLOCK_ITEMS);
    let mut digits_b = Shared::<[u32]>::new_slice(RADIX_BLOCK_ITEMS);
    let mut plane_prefixes = Shared::<[u32]>::new_slice(RADIX_BLOCK_ITEMS);

    if valid {
        let source = if params[2] != 0u32 {
            global as u32
        } else {
            permutation[global]
        };
        permutation_a[local] = source;
        digits_a[local] = Op::apply(keys[source as usize], params[0]);
    } else {
        permutation_a[local] = 0u32;
        digits_a[local] = 0u32;
    }
    sync_cube();

    // Four stable binary partitions produce one stable four-bit digit sort.
    // Each partition uses all lanes instead of assigning an entire block to
    // one scalar worker.
    let source_a = RuntimeCell::<u32>::new(1u32);
    let bit = RuntimeCell::<u32>::new(0u32);
    while bit.read() < RADIX_BITS {
        let digit = if source_a.read() != 0u32 {
            digits_a[local]
        } else {
            digits_b[local]
        };
        let is_zero = RuntimeCell::<u32>::new(if valid && ((digit >> bit.read()) & 1u32) == 0u32 {
            1u32
        } else {
            0u32
        });
        let inclusive_zeros = RuntimeCell::<u32>::new(is_zero.read());
        let offset = RuntimeCell::<u32>::new(1u32);
        while offset.read() < PLANE_DIM {
            let left = plane_shuffle_up(inclusive_zeros.read(), offset.read());
            if UNIT_POS_PLANE >= offset.read() {
                inclusive_zeros.store(inclusive_zeros.read() + left);
            }
            offset.store(offset.read() * 2u32);
        }
        if UNIT_POS_PLANE + 1u32 == PLANE_DIM {
            plane_prefixes[PLANE_POS as usize] = inclusive_zeros.read();
        }
        sync_cube();

        let plane_count = (CUBE_DIM + PLANE_DIM - 1u32) / PLANE_DIM;
        if UNIT_POS == 0u32 {
            let prefix = RuntimeCell::<u32>::new(0u32);
            let plane = RuntimeCell::<u32>::new(0u32);
            while plane.read() < plane_count {
                let index = plane.read() as usize;
                let count = plane_prefixes[index];
                plane_prefixes[index] = prefix.read();
                prefix.store(prefix.read() + count);
                plane.store(plane.read() + 1u32);
            }
            plane_prefixes[plane_count as usize] = prefix.read();
        }
        sync_cube();

        if valid {
            let exclusive_zeros =
                plane_prefixes[PLANE_POS as usize] + inclusive_zeros.read() - is_zero.read();
            let zero_count = plane_prefixes[plane_count as usize];
            let destination = if is_zero.read() != 0u32 {
                exclusive_zeros
            } else {
                zero_count + local as u32 - exclusive_zeros
            } as usize;
            if source_a.read() != 0u32 {
                permutation_b[destination] = permutation_a[local];
                digits_b[destination] = digit;
            } else {
                permutation_a[destination] = permutation_b[local];
                digits_a[destination] = digit;
            }
        }
        sync_cube();
        source_a.store(1u32 - source_a.read());
        bit.store(bit.read() + 1u32);
    }

    let sorted_digit = if source_a.read() != 0u32 {
        digits_a[local]
    } else {
        digits_b[local]
    };
    if local < RADIX_BUCKETS {
        plane_prefixes[local] = (end - start) as u32;
        histograms[local * params[1] as usize + block] = 0u32;
    }
    sync_cube();
    if valid {
        let previous_digit = if local > 0usize {
            if source_a.read() != 0u32 {
                digits_a[local - 1usize]
            } else {
                digits_b[local - 1usize]
            }
        } else {
            RADIX_BUCKETS as u32
        };
        if sorted_digit != previous_digit {
            plane_prefixes[sorted_digit as usize] = local as u32;
        }
    }
    sync_cube();

    if valid {
        block_permutation[global] = if source_a.read() != 0u32 {
            permutation_a[local]
        } else {
            permutation_b[local]
        };
        local_metadata[global] = sorted_digit * RADIX_BLOCK_ITEMS as u32 + local as u32
            - plane_prefixes[sorted_digit as usize];
        let next_digit = if local + 1usize < end - start {
            if source_a.read() != 0u32 {
                digits_a[local + 1usize]
            } else {
                digits_b[local + 1usize]
            }
        } else {
            RADIX_BUCKETS as u32
        };
        if next_digit != sorted_digit {
            histograms[sorted_digit as usize * params[1] as usize + block] =
                local as u32 + 1u32 - plane_prefixes[sorted_digit as usize];
        }
    }
}

#[cubecl::cube(launch_unchecked)]
fn histogram_prefix_kernel(
    histograms: &[u32],
    params: &[u32],
    block_prefixes: &mut [u32],
    bucket_totals: &mut [u32],
) {
    // One independently scheduled workgroup scans each bucket so the prefix
    // phase continues to scale when the number of input blocks grows.
    let bucket = CUBE_POS as usize;
    let lane = UNIT_POS as usize;
    let blocks = params[1] as usize;
    let chunk_size = blocks.div_ceil(CUBE_DIM as usize);
    let chunk_start = if lane * chunk_size < blocks {
        lane * chunk_size
    } else {
        blocks
    };
    let chunk_end = if chunk_start + chunk_size < blocks {
        chunk_start + chunk_size
    } else {
        blocks
    };
    let chunk_sum = RuntimeCell::<u32>::new(0u32);
    let block = RuntimeCell::<usize>::new(chunk_start);
    while block.read() < chunk_end {
        chunk_sum.store(chunk_sum.read() + histograms[bucket * blocks + block.read()]);
        block.store(block.read() + 1usize);
    }
    let mut chunk_prefixes = Shared::<[u32]>::new_slice(BLOCK_SIZE as usize);
    chunk_prefixes[lane] = chunk_sum.read();
    sync_cube();

    if lane == 0usize {
        let prefix = RuntimeCell::<u32>::new(0u32);
        let current = RuntimeCell::<usize>::new(0usize);
        while current.read() < CUBE_DIM as usize {
            let count = chunk_prefixes[current.read()];
            chunk_prefixes[current.read()] = prefix.read();
            prefix.store(prefix.read() + count);
            current.store(current.read() + 1usize);
        }
        bucket_totals[bucket] = prefix.read();
    }
    sync_cube();

    let prefix = RuntimeCell::<u32>::new(chunk_prefixes[lane]);
    block.store(chunk_start);
    while block.read() < chunk_end {
        let index = bucket * blocks + block.read();
        block_prefixes[index] = prefix.read();
        prefix.store(prefix.read() + histograms[index]);
        block.store(block.read() + 1usize);
    }
}

#[cubecl::cube(launch_unchecked)]
fn bucket_offsets_kernel(bucket_totals: &[u32], bucket_offsets: &mut [u32]) {
    let offset = RuntimeCell::<u32>::new(0u32);
    let bucket = RuntimeCell::<usize>::new(0usize);
    while bucket.read() < RADIX_BUCKETS {
        bucket_offsets[bucket.read()] = offset.read();
        offset.store(offset.read() + bucket_totals[bucket.read()]);
        bucket.store(bucket.read() + 1usize);
    }
}

#[cubecl::cube(launch_unchecked)]
fn stable_digit_scatter_kernel(
    block_permutation: &[u32],
    local_metadata: &[u32],
    block_prefixes: &[u32],
    bucket_offsets: &[u32],
    logical_len: &[u32],
    params: &[u32],
    output: &mut [u32],
) {
    let index = ABSOLUTE_POS as usize;
    if index < logical_len[0] as usize {
        let digit = (local_metadata[index] / RADIX_BLOCK_ITEMS as u32) as usize;
        let rank = local_metadata[index] % RADIX_BLOCK_ITEMS as u32;
        let block = index / RADIX_BLOCK_ITEMS;
        let destination =
            bucket_offsets[digit] + block_prefixes[digit * params[1] as usize + block] + rank;
        output[destination as usize] = block_permutation[index];
    }
}

pub(crate) struct RadixControl<R: Runtime> {
    current: DeviceVec<R, u32>,
    scratch: DeviceVec<R, u32>,
    block_permutation: DeviceVec<R, u32>,
    local_metadata: DeviceVec<R, u32>,
    histograms: DeviceVec<R, u32>,
    block_prefixes: DeviceVec<R, u32>,
    bucket_totals: DeviceVec<R, u32>,
    bucket_offsets: DeviceVec<R, u32>,
    block_count: usize,
    initialized: bool,
}

impl<R: Runtime> RadixControl<R> {
    fn new(
        exec: &Executor<R>,
        len: usize,
        extent: crate::extent::LogicalExtent,
    ) -> Result<Self, Error> {
        let block_count = len.div_ceil(RADIX_BLOCK_ITEMS);
        let mut current = exec.alloc_row::<u32>(len);
        current.set_logical_extent(extent.clone());
        let mut scratch = exec.alloc_row::<u32>(len);
        scratch.set_logical_extent(extent);
        Ok(Self {
            current,
            scratch,
            block_permutation: exec.alloc_row::<u32>(len),
            local_metadata: exec.alloc_row::<u32>(len),
            histograms: exec.alloc_row::<u32>(RADIX_BUCKETS * block_count),
            block_prefixes: exec.alloc_row::<u32>(RADIX_BUCKETS * block_count),
            bucket_totals: exec.alloc_row::<u32>(RADIX_BUCKETS),
            bucket_offsets: exec.alloc_row::<u32>(RADIX_BUCKETS),
            block_count,
            initialized: false,
        })
    }

    fn pass<T: RadixScalar>(
        &mut self,
        exec: &Executor<R>,
        keys: &DeviceVec<R, T>,
        shift: u32,
    ) -> Result<(), Error> {
        let len = self.current.capacity();
        if len == 0 {
            return Ok(());
        }
        let logical_len = self.current.logical_extent().materialize(exec)?;
        let params_handle = exec.client().create_from_slice(u32::as_bytes(&[
            shift,
            u32::try_from(self.block_count).map_err(|_| Error::LengthTooLarge {
                len: self.block_count,
            })?,
            u32::from(!self.initialized),
        ]));
        let rank_count = cube_count_1d(self.block_count)?;
        let scatter_count = cube_count_1d(len.div_ceil(BLOCK_SIZE as usize))?;
        let prefix_lanes = self.block_count.min(BLOCK_SIZE as usize).max(1) as u32;
        unsafe {
            block_digit_sort_kernel::launch_unchecked::<T, T::DigitOp, R>(
                exec.client(),
                rank_count,
                CubeDim::new_1d(RADIX_BLOCK_ITEMS as u32),
                BufferArg::from_raw_parts(keys.handle.clone(), keys.capacity()),
                BufferArg::from_raw_parts(self.current.handle.clone(), len),
                BufferArg::from_raw_parts(logical_len.handle.clone(), 1),
                BufferArg::from_raw_parts(params_handle.clone(), 3),
                BufferArg::from_raw_parts(self.block_permutation.handle.clone(), len),
                BufferArg::from_raw_parts(self.local_metadata.handle.clone(), len),
                BufferArg::from_raw_parts(
                    self.histograms.handle.clone(),
                    self.histograms.capacity(),
                ),
            );
            histogram_prefix_kernel::launch_unchecked::<R>(
                exec.client(),
                cube_count_1d(RADIX_BUCKETS)?,
                CubeDim::new_1d(prefix_lanes),
                BufferArg::from_raw_parts(
                    self.histograms.handle.clone(),
                    self.histograms.capacity(),
                ),
                BufferArg::from_raw_parts(params_handle.clone(), 3),
                BufferArg::from_raw_parts(
                    self.block_prefixes.handle.clone(),
                    self.block_prefixes.capacity(),
                ),
                BufferArg::from_raw_parts(
                    self.bucket_totals.handle.clone(),
                    self.bucket_totals.capacity(),
                ),
            );
            bucket_offsets_kernel::launch_unchecked::<R>(
                exec.client(),
                CubeCount::new_single(),
                CubeDim::new_1d(1),
                BufferArg::from_raw_parts(
                    self.bucket_totals.handle.clone(),
                    self.bucket_totals.capacity(),
                ),
                BufferArg::from_raw_parts(
                    self.bucket_offsets.handle.clone(),
                    self.bucket_offsets.capacity(),
                ),
            );
            stable_digit_scatter_kernel::launch_unchecked::<R>(
                exec.client(),
                scatter_count,
                CubeDim::new_1d(BLOCK_SIZE),
                BufferArg::from_raw_parts(self.block_permutation.handle.clone(), len),
                BufferArg::from_raw_parts(self.local_metadata.handle.clone(), len),
                BufferArg::from_raw_parts(
                    self.block_prefixes.handle.clone(),
                    self.block_prefixes.capacity(),
                ),
                BufferArg::from_raw_parts(
                    self.bucket_offsets.handle.clone(),
                    self.bucket_offsets.capacity(),
                ),
                BufferArg::from_raw_parts(logical_len.handle.clone(), 1),
                BufferArg::from_raw_parts(params_handle, 3),
                BufferArg::from_raw_parts(self.scratch.handle.clone(), len),
            );
        }
        core::mem::swap(&mut self.current, &mut self.scratch);
        self.initialized = true;
        Ok(())
    }
}

/// Flat-row key storage that can contribute lexicographic radix passes.
///
/// Zip nodes process the right child first because each binary pass is stable;
/// this makes the leftmost physical column the primary key.
pub(crate) trait RadixStorage<R: Runtime> {
    fn radix_passes(&self, exec: &Executor<R>, control: &mut RadixControl<R>) -> Result<(), Error>;
}

impl<R, T> RadixStorage<R> for DeviceVec<R, T>
where
    R: Runtime,
    T: RadixScalar,
{
    fn radix_passes(&self, exec: &Executor<R>, control: &mut RadixControl<R>) -> Result<(), Error> {
        for shift in (0..T::BITS).step_by(RADIX_BITS as usize) {
            control.pass(exec, self, shift)?;
        }
        Ok(())
    }
}

impl<R, Left, Right> RadixStorage<R> for Zip<Left, Right>
where
    R: Runtime,
    Left: RadixStorage<R>,
    Right: RadixStorage<R>,
{
    fn radix_passes(&self, exec: &Executor<R>, control: &mut RadixControl<R>) -> Result<(), Error> {
        self.1.radix_passes(exec, control)?;
        self.0.radix_passes(exec, control)
    }
}

pub(crate) fn permutation<R, Keys>(
    exec: &Executor<R>,
    keys: &Keys,
    len: usize,
    extent: crate::extent::LogicalExtent,
) -> Result<DeviceVec<R, u32>, Error>
where
    R: Runtime,
    Keys: RadixStorage<R>,
{
    let mut control = RadixControl::new(exec, len, extent)?;
    keys.radix_passes(exec, &mut control)?;
    Ok(control.current)
}
