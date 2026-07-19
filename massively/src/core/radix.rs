//! Stable least-significant-digit radix ordering controls.

use cubecl::prelude::*;

use crate::{DeviceVec, Error, Executor, MStorageElement, Zip, launch::cube_count_1d};

const BLOCK_SIZE: u32 = 256;

/// Returns one when `value` has a zero in the requested bit of its
/// order-preserving radix representation.
#[cubecl::cube]
trait RadixZeroOp<T: CubePrimitive>: 'static + Send + Sync {
    fn apply(value: T, bit: u32) -> u32;
}

macro_rules! unsigned_radix_op {
    ($name:ident, $ty:ty, $wide:ty) => {
        struct $name;

        #[cubecl::cube]
        impl RadixZeroOp<$ty> for $name {
            fn apply(value: $ty, bit: u32) -> u32 {
                1u32 - ((((value as $wide) >> (bit as $wide)) & (1 as $wide)) as u32)
            }
        }
    };
}

macro_rules! signed_radix_op {
    ($name:ident, $ty:ty, $wide:ty, $mask:expr, $sign:expr) => {
        struct $name;

        #[cubecl::cube]
        impl RadixZeroOp<$ty> for $name {
            fn apply(value: $ty, bit: u32) -> u32 {
                let encoded = (((value as $wide) & $mask) ^ $sign) as $wide;
                1u32 - (((encoded >> (bit as $wide)) & (1 as $wide)) as u32)
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
impl RadixZeroOp<f32> for F32RadixZero {
    fn apply(value: f32, bit: u32) -> u32 {
        let raw = u32::reinterpret(value);
        let encoded = if (raw & 0x8000_0000u32) != 0u32 {
            !raw
        } else {
            raw ^ 0x8000_0000u32
        };
        1u32 - ((encoded >> bit) & 1u32)
    }
}

struct F64RadixZero;

#[cubecl::cube]
impl RadixZeroOp<f64> for F64RadixZero {
    fn apply(value: f64, bit: u32) -> u32 {
        let raw = u64::reinterpret(value);
        let encoded = if (raw & 0x8000_0000_0000_0000u64) != 0u64 {
            !raw
        } else {
            raw ^ 0x8000_0000_0000_0000u64
        };
        1u32 - (((encoded >> (bit as u64)) & 1u64) as u32)
    }
}

trait RadixScalar: MStorageElement {
    const BITS: u32;
    type ZeroOp: RadixZeroOp<Self>;
}

macro_rules! impl_radix_scalar {
    ($($ty:ty => ($bits:expr, $op:ty)),+ $(,)?) => {
        $(
            impl RadixScalar for $ty {
                const BITS: u32 = $bits;
                type ZeroOp = $op;
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

#[cubecl::cube(launch_unchecked)]
fn iota_kernel(len: &[u32], output: &mut [u32]) {
    let index = ABSOLUTE_POS as usize;
    if index < len[0] as usize {
        output[index] = index as u32;
    }
}

#[cubecl::cube(launch_unchecked, explicit_define)]
fn zero_flags_kernel<T: CubePrimitive, Op: RadixZeroOp<T>>(
    keys: &[T],
    permutation: &[u32],
    bit: &[u32],
    flags: &mut [u32],
) {
    let index = ABSOLUTE_POS as usize;
    if index < permutation.len() {
        let source = permutation[index] as usize;
        flags[index] = Op::apply(keys[source], bit[0]);
    }
}

#[cubecl::cube(launch_unchecked)]
fn stable_binary_scatter_kernel(
    input: &[u32],
    zero_flags: &[u32],
    zero_prefixes: &[u32],
    output: &mut [u32],
) {
    let index = ABSOLUTE_POS as usize;
    if index < input.len() {
        let is_zero = zero_flags[index];
        let zeros_before = zero_prefixes[index] - is_zero;
        let zero_count = zero_prefixes[zero_prefixes.len() - 1usize];
        let destination = if is_zero != 0u32 {
            zeros_before
        } else {
            zero_count + index as u32 - zeros_before
        };
        output[destination as usize] = input[index];
    }
}

pub(crate) struct RadixControl<R: Runtime> {
    current: DeviceVec<R, u32>,
    scratch: DeviceVec<R, u32>,
}

impl<R: Runtime> RadixControl<R> {
    fn new(exec: &Executor<R>, len: usize) -> Result<Self, Error> {
        let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
        let current = exec.alloc_row::<u32>(len);
        let scratch = exec.alloc_row::<u32>(len);
        if len != 0 {
            let len_handle = exec.client().create_from_slice(u32::as_bytes(&[len_u32]));
            unsafe {
                iota_kernel::launch_unchecked::<R>(
                    exec.client(),
                    cube_count_1d(len.div_ceil(BLOCK_SIZE as usize))?,
                    CubeDim::new_1d(BLOCK_SIZE),
                    BufferArg::from_raw_parts(len_handle, 1),
                    BufferArg::from_raw_parts(current.handle.clone(), len),
                );
            }
        }
        Ok(Self { current, scratch })
    }

    fn pass<T: RadixScalar>(
        &mut self,
        exec: &Executor<R>,
        keys: &DeviceVec<R, T>,
        bit: u32,
    ) -> Result<(), Error> {
        let len = self.current.len();
        if len == 0 {
            return Ok(());
        }
        let flags = exec.alloc_row::<u32>(len);
        let bit_handle = exec.client().create_from_slice(u32::as_bytes(&[bit]));
        let count = cube_count_1d(len.div_ceil(BLOCK_SIZE as usize))?;
        unsafe {
            zero_flags_kernel::launch_unchecked::<T, T::ZeroOp, R>(
                exec.client(),
                count.clone(),
                CubeDim::new_1d(BLOCK_SIZE),
                BufferArg::from_raw_parts(keys.handle.clone(), keys.len()),
                BufferArg::from_raw_parts(self.current.handle.clone(), len),
                BufferArg::from_raw_parts(bit_handle, 1),
                BufferArg::from_raw_parts(flags.handle.clone(), len),
            );
        }
        let prefixes = crate::scan::inclusive_scan_u32(exec, &flags)?;
        unsafe {
            stable_binary_scatter_kernel::launch_unchecked::<R>(
                exec.client(),
                count,
                CubeDim::new_1d(BLOCK_SIZE),
                BufferArg::from_raw_parts(self.current.handle.clone(), len),
                BufferArg::from_raw_parts(flags.handle.clone(), len),
                BufferArg::from_raw_parts(prefixes.handle.clone(), len),
                BufferArg::from_raw_parts(self.scratch.handle.clone(), len),
            );
        }
        core::mem::swap(&mut self.current, &mut self.scratch);
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
        for bit in 0..T::BITS {
            control.pass(exec, self, bit)?;
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
) -> Result<DeviceVec<R, u32>, Error>
where
    R: Runtime,
    Keys: RadixStorage<R>,
{
    let mut control = RadixControl::new(exec, len)?;
    keys.radix_passes(exec, &mut control)?;
    Ok(control.current)
}
