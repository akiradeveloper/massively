use super::*;

/// Copies elements whose `u32` stencil flag is non-zero.
pub fn copy_where<B, Input, Stencil, Output>(
    exec: &Executor<B>,
    source: Input,
    stencil: Stencil,
) -> Result<Output, Error>
where
    B: Runtime,
    Input: MIter<B>,
    Stencil: MSlice<B, Item = u32>,
    Output: MVec<B, Item = Input::Item>,
{
    validate_input(exec, &source)?;
    validate_mslice(exec, &stencil)?;
    let stencil = lowering::u32_stencil(exec.policy(), &stencil, "copy_where stencil", false)?;
    <Input as sealed::MIterDispatch<B>>::copy_where_dispatch(source, exec.policy(), stencil)
}

/// Removes elements whose `u32` stencil flag is non-zero.
pub fn remove_where<B, Input, Stencil, Output>(
    exec: &Executor<B>,
    source: Input,
    stencil: Stencil,
) -> Result<Output, Error>
where
    B: Runtime,
    Input: MIter<B>,
    Stencil: MSlice<B, Item = u32>,
    Output: MVec<B, Item = Input::Item>,
{
    validate_input(exec, &source)?;
    validate_mslice(exec, &stencil)?;
    let stencil = lowering::u32_stencil(exec.policy(), &stencil, "remove_where stencil", true)?;
    <Input as sealed::MIterDispatch<B>>::remove_where_dispatch(source, exec.policy(), stencil)
}

/// Replaces elements whose `u32` stencil flag is non-zero.
pub fn replace_where<B, Stencil, Output>(
    exec: &Executor<B>,
    replacement: Output::Item,
    stencil: Stencil,
    out: Output,
) -> Result<(), Error>
where
    B: Runtime,
    Stencil: MSlice<B, Item = u32>,
    Output: MIterMut<B>,
{
    validate_mslice(exec, &stencil)?;
    validate_output(exec, &out)?;
    let stencil = lowering::u32_stencil(exec.policy(), &stencil, "replace_where stencil", false)?;
    out.replace_where_inner(exec.policy(), replacement, stencil)
}
