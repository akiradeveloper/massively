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
    Indices: MIter<R, Item = crate::MIndex>,
    Output: MIterMut<R>,
    Input: MIter<R, Item = Output::Item>,
{
    validate_input(exec, &source)?;
    validate_input(exec, &indices)?;
    validate_output(exec, &out)?;
    source.gather_with_policy(exec.policy(), indices, out)
}

/// Gathers elements whose stencil flag is true.
pub fn gather_where<R, Input, Indices, Stencil, Output>(
    exec: &Executor<R>,
    source: Input,
    indices: Indices,
    stencil: Stencil,
    out: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Indices: MIter<R, Item = crate::MIndex>,
    Stencil: MIter<R, Item = bool>,
    Output: MIterMut<R>,
    Input: MIter<R, Item = Output::Item>,
{
    validate_input(exec, &source)?;
    validate_input(exec, &indices)?;
    validate_input(exec, &stencil)?;
    validate_output(exec, &out)?;
    let stencil = stencil.stencil_selection_with_policy(exec.policy(), false, true)?;
    source.gather_where_with_policy(exec.policy(), indices, stencil, out)
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
    Indices: MIter<R, Item = crate::MIndex>,
    Output: MIterMut<R>,
    Input: MIter<R, Item = Output::Item>,
{
    validate_input(exec, &source)?;
    validate_input(exec, &indices)?;
    validate_output(exec, &out)?;
    source.scatter_with_policy(exec.policy(), indices, out)
}

/// Scatters values whose stencil flag is true into a newly allocated output.
pub fn scatter_where<R, Input, Indices, Stencil, Output>(
    exec: &Executor<R>,
    source: Input,
    indices: Indices,
    stencil: Stencil,
    out: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Indices: MIter<R, Item = crate::MIndex>,
    Stencil: MIter<R, Item = bool>,
    Output: MIterMut<R>,
    Input: MIter<R, Item = Output::Item>,
{
    validate_input(exec, &source)?;
    validate_input(exec, &indices)?;
    validate_input(exec, &stencil)?;
    validate_output(exec, &out)?;
    let stencil = stencil.stencil_selection_with_policy(exec.policy(), false, true)?;
    source.scatter_where_with_policy(exec.policy(), indices, stencil, out)
}
