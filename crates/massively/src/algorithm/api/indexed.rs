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
    Indices: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
    Input: MIter<R, Item = Output::Item>,
{
    validate_input(exec, &source)?;
    validate_input(exec, &indices)?;
    validate_output(exec, &out)?;
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
    Indices: MIter<R, Item = MIndex>,
    Stencil: MIter<R, Item = u32>,
    Output: MIterMut<R>,
    Input: MIter<R, Item = Output::Item>,
{
    validate_input(exec, &source)?;
    validate_input(exec, &indices)?;
    validate_input(exec, &stencil)?;
    validate_output(exec, &out)?;
    let stencil = <Stencil as sealed::MIterDispatch<R>>::stencil_selection_dispatch(
        stencil,
        exec.policy(),
        false,
        true,
    )?;
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
    Indices: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
    Input: MIter<R, Item = Output::Item>,
{
    validate_input(exec, &source)?;
    validate_input(exec, &indices)?;
    validate_output(exec, &out)?;
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
    Indices: MIter<R, Item = MIndex>,
    Stencil: MIter<R, Item = u32>,
    Output: MIterMut<R>,
    Input: MIter<R, Item = Output::Item>,
{
    validate_input(exec, &source)?;
    validate_input(exec, &indices)?;
    validate_input(exec, &stencil)?;
    validate_output(exec, &out)?;
    let stencil = <Stencil as sealed::MIterDispatch<R>>::stencil_selection_dispatch(
        stencil,
        exec.policy(),
        false,
        true,
    )?;
    <Input as sealed::MIterDispatch<R>>::scatter_where_dispatch(
        source,
        exec.policy(),
        indices,
        stencil,
        out,
    )
}
