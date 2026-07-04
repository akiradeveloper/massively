use super::*;

/// Copies elements whose `u32` stencil flag is non-zero.
pub fn copy_where<R, Input, Output>(
    exec: &Executor<R>,
    source: Input,
    stencil: DeviceSlice<'_, R, u32>,
    out: Output,
) -> Result<MIndex, Error>
where
    R: Runtime,
    Output: MIterMut<R>,
    Input: MIter<R, Item = Output::Item>,
{
    validate_input(exec, &source)?;
    validate_device_slice(exec, &stencil)?;
    validate_output(exec, &out)?;
    let stencil = u32_stencil(exec.policy(), stencil, false)?;
    let owned: <Output::Item as MAlloc<R>>::Storage =
        <Input as sealed::MIterDispatch<R>>::copy_where_dispatch(source, exec.policy(), stencil)?;
    write_owned_prefix(exec.policy(), owned, out)
}

/// Removes elements whose `u32` stencil flag is non-zero.
pub fn remove_where<R, Input, Output>(
    exec: &Executor<R>,
    source: Input,
    stencil: DeviceSlice<'_, R, u32>,
    out: Output,
) -> Result<MIndex, Error>
where
    R: Runtime,
    Output: MIterMut<R>,
    Input: MIter<R, Item = Output::Item>,
{
    validate_input(exec, &source)?;
    validate_device_slice(exec, &stencil)?;
    validate_output(exec, &out)?;
    let stencil = u32_stencil(exec.policy(), stencil, true)?;
    let owned: <Output::Item as MAlloc<R>>::Storage =
        <Input as sealed::MIterDispatch<R>>::remove_where_dispatch(source, exec.policy(), stencil)?;
    write_owned_prefix(exec.policy(), owned, out)
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

/// Fills every element of an output range with `value`.
pub fn fill<R, Output>(exec: &Executor<R>, value: Output::Item, out: Output) -> Result<(), Error>
where
    R: Runtime,
    Output: MIterMut<R>,
{
    validate_output(exec, &out)?;
    out.fill_inner(exec.policy(), value)
}
