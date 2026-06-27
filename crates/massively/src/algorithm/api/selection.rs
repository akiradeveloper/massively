use super::*;

/// Copies elements whose `u32` stencil flag is non-zero.
pub fn copy_where<R, Input, Stencil, Output>(
    exec: &Executor<R>,
    source: Input,
    stencil: Stencil,
) -> Result<Output, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Stencil: MSlice<R, Item = u32>,
    Output: MVec<R, Item = Input::Item>,
{
    validate_input(exec, &source)?;
    validate_mslice(exec, &stencil)?;
    let stencil = lowering::u32_stencil(exec.policy(), stencil, "copy_where stencil", false)?;
    <Input as sealed::MIterDispatch<R>>::copy_where_dispatch(source, exec.policy(), stencil)
}

/// Removes elements whose `u32` stencil flag is non-zero.
pub fn remove_where<R, Input, Stencil, Output>(
    exec: &Executor<R>,
    source: Input,
    stencil: Stencil,
) -> Result<Output, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Stencil: MSlice<R, Item = u32>,
    Output: MVec<R, Item = Input::Item>,
{
    validate_input(exec, &source)?;
    validate_mslice(exec, &stencil)?;
    let stencil = lowering::u32_stencil(exec.policy(), stencil, "remove_where stencil", true)?;
    <Input as sealed::MIterDispatch<R>>::remove_where_dispatch(source, exec.policy(), stencil)
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
    Stencil: MSlice<R, Item = u32>,
    Output: MIterMut<R>,
{
    validate_mslice(exec, &stencil)?;
    validate_output(exec, &out)?;
    let stencil = lowering::u32_stencil(exec.policy(), stencil, "replace_where stencil", false)?;
    out.replace_where_inner(exec.policy(), replacement, stencil)
}
