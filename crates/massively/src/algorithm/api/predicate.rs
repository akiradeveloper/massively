use super::*;

/// Returns whether all elements satisfy `pred`.
pub fn all_of<R, Input, Pred>(
    exec: &Executor<R>,
    source: Input,
    pred: Pred,
    env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
) -> Result<bool, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Pred: op::PredicateOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<R>>::all_of_dispatch(source, exec.policy(), pred, env)
}

/// Returns whether any element satisfies `pred`.
pub fn any_of<R, Input, Pred>(
    exec: &Executor<R>,
    source: Input,
    pred: Pred,
    env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
) -> Result<bool, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Pred: op::PredicateOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<R>>::any_of_dispatch(source, exec.policy(), pred, env)
}

/// Counts elements satisfying `pred`.
pub fn count_if<R, Input, Pred>(
    exec: &Executor<R>,
    source: Input,
    pred: Pred,
    env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
) -> Result<MIndex, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Pred: op::PredicateOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<R>>::count_if_dispatch(source, exec.policy(), pred, env)
}

/// Finds the first element satisfying `pred`.
pub fn find_if<R, Input, Pred>(
    exec: &Executor<R>,
    source: Input,
    pred: Pred,
    env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
) -> Result<Option<MIndex>, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Pred: op::PredicateOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<R>>::find_if_dispatch(source, exec.policy(), pred, env)
}

/// Returns whether input is partitioned by `pred`.
pub fn is_partitioned<R, Input, Pred>(
    exec: &Executor<R>,
    source: Input,
    pred: Pred,
    env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
) -> Result<bool, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Pred: op::PredicateOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<R>>::is_partitioned_dispatch(source, exec.policy(), pred, env)
}

/// Returns whether no elements satisfy `pred`.
pub fn none_of<R, Input, Pred>(
    exec: &Executor<R>,
    source: Input,
    pred: Pred,
    env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
) -> Result<bool, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Pred: op::PredicateOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<R>>::none_of_dispatch(source, exec.policy(), pred, env)
}

/// Partitions elements by `pred`.
pub fn partition<R, Input, Pred, Output>(
    exec: &Executor<R>,
    source: Input,
    pred: Pred,
    env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
    out: Output,
) -> Result<MIndex, Error>
where
    R: Runtime,
    Output: MIterMut<R>,
    Input: MIter<R, Item = Output::Item>,
    Pred: op::PredicateOp<R, Output::Item>,
{
    validate_input(exec, &source)?;
    validate_output(exec, &out)?;
    <Input as sealed::MIterDispatch<R>>::partition_into_dispatch(
        source,
        exec.policy(),
        pred,
        env,
        out,
    )
}
