use super::*;
/// Copies elements whose `u32` stencil flag is non-zero.
pub fn copy_where<R, Input, Stencil, Output>(
    exec: &Executor<R>,
    source: Input,
    stencil: Stencil,
    out: Output,
) -> Result<MIndex, Error>
where
    R: Runtime,
    Stencil: MIter<R, Item = u32>,
    Output: MIterMut<R>,
    Input: MIter<R, Item = Output::Item>,
{
    validate_input(exec, &source)?;
    validate_input(exec, &stencil)?;
    validate_output(exec, &out)?;
    let stencil = stencil.stencil_selection_with_policy(exec.policy(), false, false)?;
    source.copy_selected_with_policy(exec.policy(), stencil, out)
}

/// Removes elements whose `u32` stencil flag is non-zero.
pub fn remove_where<R, Input, Stencil, Output>(
    exec: &Executor<R>,
    source: Input,
    stencil: Stencil,
    out: Output,
) -> Result<MIndex, Error>
where
    R: Runtime,
    Stencil: MIter<R, Item = u32>,
    Output: MIterMut<R>,
    Input: MIter<R, Item = Output::Item>,
{
    validate_input(exec, &source)?;
    validate_input(exec, &stencil)?;
    validate_output(exec, &out)?;
    let stencil = stencil.stencil_selection_with_policy(exec.policy(), true, false)?;
    source.copy_selected_with_policy(exec.policy(), stencil, out)
}

/// Replaces elements whose `u32` stencil flag is non-zero.
pub fn replace_where<R, Stencil, Output>(
    exec: &Executor<R>,
    replacement: Output::Item,
    stencil: Stencil,
    out: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Stencil: MIter<R, Item = u32>,
    Output: MIterMut<R>,
{
    validate_input(exec, &stencil)?;
    validate_output(exec, &out)?;
    let stencil = stencil.stencil_selection_with_policy(exec.policy(), false, true)?;
    out.replace_where_inner(exec.policy(), replacement, stencil)
}

/// Fills every element of an output range with `value`.
pub fn fill<R, Output>(exec: &Executor<R>, value: Output::Item, out: Output) -> Result<(), Error>
where
    R: Runtime,
    Output: MIterMut<R>,
{
    validate_output(exec, &out)?;
    out.fill_inner(exec.policy(), value)
}
