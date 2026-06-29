use super::*;

/// Copies elements whose `u32` stencil flag is non-zero.
pub fn copy_where<R, Input>(
    exec: &Executor<R>,
    source: Input,
    stencil: DeviceSlice<'_, R, u32>,
) -> Result<<Input::Item as MItem<R>>::Vec, Error>
where
    R: Runtime,
    Input: MIter<R>,
{
    validate_input(exec, &source)?;
    validate_device_slice(exec, &stencil)?;
    let stencil = u32_stencil(exec.policy(), stencil, false)?;
    <Input as sealed::MIterDispatch<R>>::copy_where_dispatch(source, exec.policy(), stencil)
}

/// Removes elements whose `u32` stencil flag is non-zero.
pub fn remove_where<R, Input>(
    exec: &Executor<R>,
    source: Input,
    stencil: DeviceSlice<'_, R, u32>,
) -> Result<<Input::Item as MItem<R>>::Vec, Error>
where
    R: Runtime,
    Input: MIter<R>,
{
    validate_input(exec, &source)?;
    validate_device_slice(exec, &stencil)?;
    let stencil = u32_stencil(exec.policy(), stencil, true)?;
    <Input as sealed::MIterDispatch<R>>::remove_where_dispatch(source, exec.policy(), stencil)
}

/// Replaces elements whose `u32` stencil flag is non-zero.
pub fn replace_where<R, Output>(
    exec: &Executor<R>,
    replacement: Output::Item,
    stencil: DeviceSlice<'_, R, u32>,
    out: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Output: MIterMut<R>,
{
    validate_device_slice(exec, &stencil)?;
    validate_output(exec, &out)?;
    let stencil = u32_stencil_flags(exec.policy(), stencil, false)?;
    out.replace_where_inner(exec.policy(), replacement, stencil)
}
