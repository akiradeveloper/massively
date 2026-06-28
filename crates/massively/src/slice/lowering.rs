use cubecl::prelude::Runtime;

use crate::Error;
use crate::detail::impls::lower_mslice_column;
use crate::detail::op_adapter::{KernelOp, StencilFlag};
use crate::slice::MSlice;

pub(crate) fn u32_read<R, Slice>(
    policy: &crate::detail::CubePolicy<R>,
    slice: Slice,
) -> Result<crate::detail::device::DeviceColumnView<R, u32>, Error>
where
    R: Runtime,
    Slice: MSlice<R, Item = u32>,
{
    lower_mslice_column(slice, policy)
}

pub(crate) fn u32_stencil<R, Slice>(
    policy: &crate::detail::CubePolicy<R>,
    slice: Slice,
    _role: &str,
    invert: bool,
) -> Result<crate::detail::api::PrecomputedSelection<R>, Error>
where
    R: Runtime,
    Slice: MSlice<R, Item = u32>,
{
    let stencil = lower_mslice_column(slice, policy)?;
    crate::detail::api::PrecomputedSelection::from_stencil_with_policy::<_, KernelOp<R, StencilFlag>>(
        policy,
        &(stencil,),
        invert,
    )
}

pub(crate) fn u32_stencil_flags<R, Slice>(
    policy: &crate::detail::CubePolicy<R>,
    slice: Slice,
    _role: &str,
    invert: bool,
) -> Result<crate::detail::api::PrecomputedSelection<R>, Error>
where
    R: Runtime,
    Slice: MSlice<R, Item = u32>,
{
    let stencil = lower_mslice_column(slice, policy)?;
    crate::detail::api::PrecomputedSelection::from_stencil_flags_with_policy::<
        _,
        KernelOp<R, StencilFlag>,
    >(policy, &(stencil,), invert)
}
