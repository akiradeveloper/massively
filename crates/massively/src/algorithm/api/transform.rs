use super::*;

/// Applies a unary transform to a massively iterator.
pub fn transform<R, Input, Output, Op>(
    exec: &Executor<R>,
    source: Input,
    op: Op,
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
    <Input as sealed::MIterDispatch<R>>::transform_dispatch(source, exec.policy(), op, out)
}

/// Applies a unary transform where the `u32` stencil flag is non-zero.
pub fn transform_where<R, Input, Stencil, Output, Op>(
    exec: &Executor<R>,
    source: Input,
    op: Op,
    stencil: Stencil,
    out: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: MIter<R>,
    Stencil: MSlice<R, Item = u32>,
    Output: MIterMut<R>,
    Op: op::UnaryOp<R, Input::Item, Output = Output::Item>,
{
    validate_input(exec, &source)?;
    validate_mslice(exec, &stencil)?;
    validate_output(exec, &out)?;
    let stencil = lowering::u32_stencil(exec.policy(), stencil, "transform_where stencil", false)?;
    <Input as sealed::MIterDispatch<R>>::transform_where_dispatch(
        source,
        exec.policy(),
        op,
        stencil,
        out,
    )
}
