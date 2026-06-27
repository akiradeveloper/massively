use super::*;

/// Gathers a massively iterator at index positions into `out`.
pub fn gather<R, Input, Indices, Output>(
    exec: &Executor<R>,
    source: Input,
    indices: Indices,
    out: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: MIter<R>,
    Indices: MSlice<R, Item = u32>,
    Output: MIterMut<R, Item = Input::Item>,
{
    validate_input(exec, &source)?;
    validate_mslice(exec, &indices)?;
    validate_output(exec, &out)?;
    let indices = lowering::u32_read::<R, Indices>(exec.policy(), indices)?;
    <Input as sealed::MIterDispatch<R>>::gather_dispatch(source, exec.policy(), indices, out)
}

/// Gathers elements whose `u32` stencil flag is non-zero.
pub fn gather_where<R, Input, Indices, Stencil, Output>(
    exec: &Executor<R>,
    source: Input,
    indices: Indices,
    stencil: Stencil,
    out: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: MIter<R>,
    Indices: MSlice<R, Item = u32>,
    Stencil: MSlice<R, Item = u32>,
    Output: MIterMut<R, Item = Input::Item>,
{
    validate_input(exec, &source)?;
    validate_mslice(exec, &indices)?;
    validate_mslice(exec, &stencil)?;
    validate_output(exec, &out)?;
    let indices = lowering::u32_read::<R, Indices>(exec.policy(), indices)?;
    let stencil = lowering::u32_stencil(exec.policy(), stencil, "gather_where stencil", false)?;
    <Input as sealed::MIterDispatch<R>>::gather_where_dispatch(
        source,
        exec.policy(),
        indices,
        stencil,
        out,
    )
}

/// Scatters values into `out`.
pub fn scatter<R, Input, Indices, Output>(
    exec: &Executor<R>,
    source: Input,
    indices: Indices,
    out: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: MIter<R>,
    Indices: MSlice<R, Item = u32>,
    Output: MIterMut<R, Item = Input::Item>,
{
    validate_input(exec, &source)?;
    validate_mslice(exec, &indices)?;
    validate_output(exec, &out)?;
    let indices = lowering::u32_read::<R, Indices>(exec.policy(), indices)?;
    <Input as sealed::MIterDispatch<R>>::scatter_dispatch(source, exec.policy(), indices, out)
}

/// Scatters values whose `u32` stencil flag is non-zero into a newly allocated output.
pub fn scatter_where<R, Input, Indices, Stencil, Output>(
    exec: &Executor<R>,
    source: Input,
    indices: Indices,
    stencil: Stencil,
    out: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: MIter<R>,
    Indices: MSlice<R, Item = u32>,
    Stencil: MSlice<R, Item = u32>,
    Output: MIterMut<R, Item = Input::Item>,
{
    validate_input(exec, &source)?;
    validate_mslice(exec, &indices)?;
    validate_mslice(exec, &stencil)?;
    validate_output(exec, &out)?;
    let indices = lowering::u32_read::<R, Indices>(exec.policy(), indices)?;
    let stencil = lowering::u32_stencil(exec.policy(), stencil, "scatter_where stencil", false)?;
    <Input as sealed::MIterDispatch<R>>::scatter_where_dispatch(
        source,
        exec.policy(),
        indices,
        stencil,
        out,
    )
}
