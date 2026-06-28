//! Hidden helpers for Criterion diagnostics.
//!
//! This module is feature-gated so benchmark code can isolate internal phases
//! without promoting those phases to the public library API.

use cubecl::prelude::*;

use crate::{
    DeviceSlice, DeviceSliceMut, Error, Executor, Scalar,
    detail::{
        api::{self, PrecomputedSelection},
        device::DeviceColumnMutView,
        op_adapter::{KernelOp, StencilFlag},
        primitives::select,
    },
    slice::{MSlice, lowering},
};

pub struct SelectionDiagControl<R: Runtime> {
    inner: PrecomputedSelection<R>,
}

impl<R: Runtime> SelectionDiagControl<R> {
    pub fn len(&self) -> usize {
        self.inner.control().len
    }
}

pub fn selection_control_from_u32_stencil<R>(
    exec: &Executor<R>,
    stencil: DeviceSlice<'_, R, u32>,
) -> Result<SelectionDiagControl<R>, Error>
where
    R: Runtime,
{
    Ok(SelectionDiagControl {
        inner: lowering::u32_stencil(
            exec.policy(),
            stencil,
            "selection diagnostic stencil",
            false,
        )?,
    })
}

pub fn selection_flags_from_u32_stencil<R>(
    exec: &Executor<R>,
    stencil: DeviceSlice<'_, R, u32>,
) -> Result<SelectionDiagControl<R>, Error>
where
    R: Runtime,
{
    Ok(SelectionDiagControl {
        inner: lowering::u32_stencil_flags(
            exec.policy(),
            stencil,
            "selection diagnostic stencil",
            false,
        )?,
    })
}

pub fn selection_control_from_predicate<R, T, Pred>(
    exec: &Executor<R>,
    values: DeviceSlice<'_, R, T>,
    invert: bool,
) -> Result<SelectionDiagControl<R>, Error>
where
    R: Runtime,
    T: Scalar,
    Pred: crate::op::PredicateOp<R, (T,)>,
{
    let input = values.into_read(exec.policy())?;
    Ok(SelectionDiagControl {
        inner: PrecomputedSelection::from_stencil_with_policy::<_, KernelOp<R, Pred>>(
            exec.policy(),
            &(input,),
            invert,
        )?,
    })
}

pub fn selected_count<R>(
    exec: &Executor<R>,
    control: &SelectionDiagControl<R>,
) -> Result<usize, Error>
where
    R: Runtime,
{
    select::selected_count(exec.policy(), control.inner.control())
}

pub fn apply_copy_where_with_control<R, T>(
    exec: &Executor<R>,
    values: DeviceSlice<'_, R, T>,
    control: &SelectionDiagControl<R>,
    output: DeviceSliceMut<'_, R, T>,
) -> Result<(), Error>
where
    R: Runtime,
    T: Scalar,
{
    let input = values.into_read(exec.policy())?;
    let output = DeviceColumnMutView::from_slice(&output.source.inner, output.offset, output.len);
    api::device_expr_copy_where_into_with_policy::<_, _, KernelOp<R, StencilFlag>>(
        exec.policy(),
        &input,
        &control.inner,
        &output,
        KernelOp::<R, StencilFlag>::new(),
    )
}
