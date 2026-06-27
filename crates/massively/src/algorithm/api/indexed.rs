use super::*;

/// Gathers a massively iterator at index positions into `out`.
pub fn gather<B, Input, Indices, Output>(
    exec: &Executor<B>,
    source: Input,
    indices: Indices,
    out: Output,
) -> Result<(), Error>
where
    B: Runtime,
    Input: MIter<B>,
    Indices: MSlice<B, Item = u32>,
    Output: MIterMut<B, Item = Input::Item>,
{
    validate_input(exec, &source)?;
    validate_mslice(exec, &indices)?;
    validate_output(exec, &out)?;
    let indices = lowering::u32_view::<B, Indices>(&indices, "gather indices")?;
    <Input as sealed::MIterDispatch<B>>::gather_dispatch(source, exec.policy(), indices, out)
}

/// Gathers elements whose `u32` stencil flag is non-zero.
pub fn gather_where<B, Input, Indices, Stencil, Output>(
    exec: &Executor<B>,
    source: Input,
    indices: Indices,
    stencil: Stencil,
    out: Output,
) -> Result<(), Error>
where
    B: Runtime,
    Input: MIter<B>,
    Indices: MSlice<B, Item = u32>,
    Stencil: MSlice<B, Item = u32>,
    Output: MIterMut<B, Item = Input::Item>,
{
    validate_input(exec, &source)?;
    validate_mslice(exec, &indices)?;
    validate_mslice(exec, &stencil)?;
    validate_output(exec, &out)?;
    let indices = lowering::u32_view::<B, Indices>(&indices, "gather_where indices")?;
    let stencil = lowering::u32_stencil(exec.policy(), &stencil, "gather_where stencil", false)?;
    <Input as sealed::MIterDispatch<B>>::gather_where_dispatch(
        source,
        exec.policy(),
        indices,
        stencil,
        out,
    )
}

/// Scatters values into `out`.
pub fn scatter<B, Input, Indices, Output>(
    exec: &Executor<B>,
    source: Input,
    indices: Indices,
    out: Output,
) -> Result<(), Error>
where
    B: Runtime,
    Input: MIter<B>,
    Indices: MSlice<B, Item = u32>,
    Output: MIterMut<B, Item = Input::Item>,
{
    validate_input(exec, &source)?;
    validate_mslice(exec, &indices)?;
    validate_output(exec, &out)?;
    let indices = lowering::u32_view::<B, Indices>(&indices, "scatter indices")?;
    <Input as sealed::MIterDispatch<B>>::scatter_dispatch(source, exec.policy(), indices, out)
}

/// Scatters values whose `u32` stencil flag is non-zero into a newly allocated output.
pub fn scatter_where<B, Input, Indices, Stencil, Output>(
    exec: &Executor<B>,
    source: Input,
    indices: Indices,
    stencil: Stencil,
    out: Output,
) -> Result<(), Error>
where
    B: Runtime,
    Input: MIter<B>,
    Indices: MSlice<B, Item = u32>,
    Stencil: MSlice<B, Item = u32>,
    Output: MIterMut<B, Item = Input::Item>,
{
    validate_input(exec, &source)?;
    validate_mslice(exec, &indices)?;
    validate_mslice(exec, &stencil)?;
    validate_output(exec, &out)?;
    let indices = lowering::u32_view::<B, Indices>(&indices, "scatter_where indices")?;
    let stencil = lowering::u32_stencil(exec.policy(), &stencil, "scatter_where stencil", false)?;
    <Input as sealed::MIterDispatch<B>>::scatter_where_dispatch(
        source,
        exec.policy(),
        indices,
        stencil,
        out,
    )
}
