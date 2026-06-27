use cubecl::prelude::Runtime;

use crate::Error;
use crate::detail::op_adapter::{KernelOp, StencilFlag};
use crate::slice::MSlice;

pub(crate) fn u32_view<B, Slice>(
    slice: &Slice,
    role: &str,
) -> Result<crate::detail::device::DeviceColumnView<B, u32>, Error>
where
    B: Runtime,
    Slice: MSlice<B, Item = u32>,
{
    slice.column_view::<u32>()?.ok_or_else(|| Error::Launch {
        message: format!("{role} must lower to one u32 device column"),
    })
}

pub(crate) fn u32_stencil<B, Slice>(
    policy: &crate::detail::CubePolicy<B>,
    slice: &Slice,
    role: &str,
    invert: bool,
) -> Result<crate::detail::api::PrecomputedSelection<B>, Error>
where
    B: Runtime,
    Slice: MSlice<B, Item = u32>,
{
    let stencil = u32_view::<B, Slice>(slice, role)?;
    crate::detail::api::PrecomputedSelection::from_stencil_with_policy::<_, KernelOp<B, StencilFlag>>(
        policy,
        &(stencil,),
        invert,
    )
}
