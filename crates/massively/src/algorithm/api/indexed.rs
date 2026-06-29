use super::*;

/// Gathers a massively iterator at index positions into `out`.
pub fn gather<R, Input, Output>(
    exec: &Executor<R>,
    source: Input,
    indices: DeviceSlice<'_, R, u32>,
    out: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: MIter<R>,
    Output: MIterMut<R, Item = Input::Item>,
{
    validate_input(exec, &source)?;
    validate_device_slice(exec, &indices)?;
    validate_output(exec, &out)?;
    let indices = indices.column_view();
    <Input as sealed::MIterDispatch<R>>::gather_dispatch(source, exec.policy(), indices, out)
}

/// Gathers a massively iterator at index positions into owned device storage.
pub fn permute<R, Input, Output>(
    exec: &Executor<R>,
    source: Input,
    indices: DeviceSlice<'_, R, u32>,
) -> Result<Output, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Output: MVec<R, Item = Input::Item>,
{
    validate_input(exec, &source)?;
    validate_device_slice(exec, &indices)?;
    let indices = indices.column_view();
    <Input as sealed::MIterDispatch<R>>::permute_dispatch(source, exec.policy(), indices)
}

/// Gathers elements whose `u32` stencil flag is non-zero.
pub fn gather_where<R, Input, Output>(
    exec: &Executor<R>,
    source: Input,
    indices: DeviceSlice<'_, R, u32>,
    stencil: DeviceSlice<'_, R, u32>,
    out: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: MIter<R>,
    Output: MIterMut<R, Item = Input::Item>,
{
    validate_input(exec, &source)?;
    validate_device_slice(exec, &indices)?;
    validate_device_slice(exec, &stencil)?;
    validate_output(exec, &out)?;
    let indices = indices.column_view();
    let stencil = u32_stencil_flags(exec.policy(), stencil, false)?;
    <Input as sealed::MIterDispatch<R>>::gather_where_dispatch(
        source,
        exec.policy(),
        indices,
        stencil,
        out,
    )
}

/// Scatters values into `out`.
pub fn scatter<R, Input, Output>(
    exec: &Executor<R>,
    source: Input,
    indices: DeviceSlice<'_, R, u32>,
    out: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: MIter<R>,
    Output: MIterMut<R, Item = Input::Item>,
{
    validate_input(exec, &source)?;
    validate_device_slice(exec, &indices)?;
    validate_output(exec, &out)?;
    let indices = indices.column_view();
    <Input as sealed::MIterDispatch<R>>::scatter_dispatch(source, exec.policy(), indices, out)
}

/// Scatters values whose `u32` stencil flag is non-zero into a newly allocated output.
pub fn scatter_where<R, Input, Output>(
    exec: &Executor<R>,
    source: Input,
    indices: DeviceSlice<'_, R, u32>,
    stencil: DeviceSlice<'_, R, u32>,
    out: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: MIter<R>,
    Output: MIterMut<R, Item = Input::Item>,
{
    validate_input(exec, &source)?;
    validate_device_slice(exec, &indices)?;
    validate_device_slice(exec, &stencil)?;
    validate_output(exec, &out)?;
    let indices = indices.column_view();
    let stencil = u32_stencil_flags(exec.policy(), stencil, false)?;
    <Input as sealed::MIterDispatch<R>>::scatter_where_dispatch(
        source,
        exec.policy(),
        indices,
        stencil,
        out,
    )
}
