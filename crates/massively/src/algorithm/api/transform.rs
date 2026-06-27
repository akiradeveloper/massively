use super::*;

/// Applies a unary transform to a massively iterator.
pub fn transform<B, Input, Output, Op>(
    exec: &Executor<B>,
    source: Input,
    op: Op,
    out: Output,
) -> Result<(), Error>
where
    B: Runtime,
    Input: MIter<B>,
    Output: MIterMut<B>,
    Op: op::UnaryOp<B, Input::Item, Output = Output::Item>,
{
    validate_input(exec, &source)?;
    validate_output(exec, &out)?;
    <Input as sealed::MIterDispatch<B>>::transform_dispatch(source, exec.policy(), op, out)
}

/// Applies a unary transform where the `u32` stencil flag is non-zero.
pub fn transform_where<B, Input, Stencil, Output, Op>(
    exec: &Executor<B>,
    source: Input,
    op: Op,
    stencil: Stencil,
    out: Output,
) -> Result<(), Error>
where
    B: Runtime,
    Input: MIter<B>,
    Stencil: MSlice<B, Item = u32>,
    Output: MIterMut<B>,
    Op: op::UnaryOp<B, Input::Item, Output = Output::Item>,
{
    validate_input(exec, &source)?;
    validate_mslice(exec, &stencil)?;
    validate_output(exec, &out)?;
    let stencil = lowering::u32_stencil(exec.policy(), &stencil, "transform_where stencil", false)?;
    <Input as sealed::MIterDispatch<B>>::transform_where_dispatch(
        source,
        exec.policy(),
        op,
        stencil,
        out,
    )
}
