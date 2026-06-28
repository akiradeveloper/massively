use std::any::Any;

use cubecl::prelude::Runtime;

use crate::Error;
use crate::detail::dispatch::{self as sealed, array_from_inner, column_view_at};
use crate::detail::op_adapter::{
    KernelOp, KernelTuple1InnerProductOp, KernelTuple1Op, StencilFlag,
};
use crate::error::ensure_same_len;
use crate::iter::{MIter, MIterMut, SoA1, SoA2, SoA3};
use crate::op;
use crate::runtime::{DeviceSliceMut, DeviceVec, Executor, Scalar};
use crate::slice::MSlice;
use crate::value::{MItem, MVec};

mod item;
mod iter;

fn lower_mslice_column<R, S, T>(
    slice: S,
    policy: &crate::detail::CubePolicy<R>,
) -> Result<crate::detail::device::DeviceColumnView<R, T>, Error>
where
    R: Runtime,
    S: MSlice<R, Item = T>,
    T: Scalar + 'static,
{
    if let Some(view) = slice.column_view::<T>()? {
        return Ok(view);
    }
    let read = slice.into_read(policy)?;
    let column = crate::detail::api::device_expr_collect_with_policy(policy, &read)?;
    Ok(crate::detail::device::DeviceColumnView::from_column(
        &column,
    ))
}

fn lower_mslice_column_as<R, S, T, U>(
    slice: S,
    policy: &crate::detail::CubePolicy<R>,
) -> Result<Option<crate::detail::device::DeviceColumnView<R, U>>, Error>
where
    R: Runtime,
    S: MSlice<R, Item = T>,
    T: Scalar + 'static,
    U: Scalar + 'static,
{
    if std::any::TypeId::of::<T>() != std::any::TypeId::of::<U>() {
        return Ok(None);
    }
    if let Some(view) = slice.column_view::<U>()? {
        return Ok(Some(view));
    }
    let read = slice.into_read(policy)?;
    let column = crate::detail::api::device_expr_collect_with_policy(policy, &read)?;
    let typed = crate::detail::DeviceVec::<R, U>::from_handle(
        column.policy_id(),
        column.handle.clone(),
        column.len(),
    );
    Ok(Some(crate::detail::device::DeviceColumnView::from_column(
        &typed,
    )))
}
