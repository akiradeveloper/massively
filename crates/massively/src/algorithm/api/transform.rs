use super::*;

/// Applies a unary transform to a massively iterator.
pub fn transform<R, Input, Output, Op>(
    exec: &Executor<R>,
    source: Input,
    op: Op,
    env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
    out: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: MIter<R>,
    Output: MIterMut<R>,
    Op: op::UnaryOp<R, Input::Item, Output = Output::Item>,
{
    validate_input(exec, &source)?;
    validate_output(exec, &out)?;
    <Input as sealed::MIterDispatch<R>>::transform_dispatch(source, exec.policy(), op, env, out)
}

/// Applies a unary transform where the `u32` stencil flag is non-zero.
pub fn transform_where<R, Input, Output, Op>(
    exec: &Executor<R>,
    source: Input,
    op: Op,
    env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
    stencil: DeviceSlice<'_, R, u32>,
    out: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: MIter<R>,
    Output: MIterMut<R>,
    Op: op::UnaryOp<R, Input::Item, Output = Output::Item>,
{
    validate_input(exec, &source)?;
    validate_device_slice(exec, &stencil)?;
    validate_output(exec, &out)?;
    let stencil = u32_stencil(exec.policy(), stencil, false)?;
    <Input as sealed::MIterDispatch<R>>::transform_where_dispatch(
        source,
        exec.policy(),
        op,
        env,
        stencil,
        out,
    )
}
