use std::any::Any;

use cubecl::prelude::*;

use crate::Error;
use crate::detail::dispatch::{self as sealed, array_from_inner};
use crate::detail::op_adapter::{
    KernelOp, KernelTuple1InnerProductOp, KernelTuple1Op, StencilFlag,
};
use crate::error::ensure_same_len;
use crate::iter::{MIter, MIterMut, SoA1, SoA2, SoA3, SoA4, SoA5, SoA6, SoA7};
use crate::op;
use crate::runtime::{DeviceSliceMut, DeviceVec, Executor, Scalar};
use crate::value::{MItem, MVec};

mod item;
mod iter;

pub(crate) fn end_flags_from_head_flags<R>(
    policy: &crate::detail::CubePolicy<R>,
    head_flags: cubecl::server::Handle,
    len: usize,
) -> Result<cubecl::server::Handle, Error>
where
    R: Runtime,
{
    if len == 0 {
        return Ok(policy.empty_handle());
    }

    let client = policy.client();
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let end_flags = client.empty(len * std::mem::size_of::<u32>());
    let num_blocks = len.div_ceil(crate::detail::primitives::scan::BLOCK_SCAN_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;

    unsafe {
        crate::kernels::head_flags_to_end_flags_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(num_blocks_u32, 1, 1),
            CubeDim::new_1d(crate::detail::primitives::scan::BLOCK_SCAN_SIZE),
            BufferArg::from_raw_parts(head_flags.clone(), len),
            BufferArg::from_raw_parts(len_handle.clone(), 1),
            BufferArg::from_raw_parts(end_flags.clone(), len),
        );
    }

    Ok(end_flags)
}
