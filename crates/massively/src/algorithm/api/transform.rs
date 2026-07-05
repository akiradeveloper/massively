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
    source.transform_with_policy(exec.policy(), op, env, out)
}

/// Applies a unary transform where the `u32` stencil flag is non-zero.
pub fn transform_where<R, Input, Stencil, Output, Op>(
    exec: &Executor<R>,
    source: Input,
    op: Op,
    env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
    stencil: Stencil,
    out: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: MIter<R>,
    Stencil: MIter<R, Item = u32>,
    Output: MIterMut<R>,
    Op: op::UnaryOp<R, Input::Item, Output = Output::Item>,
{
    validate_input(exec, &source)?;
    validate_input(exec, &stencil)?;
    validate_output(exec, &out)?;
    let stencil = stencil.stencil_selection_with_policy(exec.policy(), false, false)?;
    source.transform_where_with_policy(exec.policy(), op, env, stencil, out)
}
