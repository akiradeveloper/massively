use super::*;

/// Returns whether all elements satisfy `pred`.
pub fn all_of<R, Input, Pred>(exec: &Executor<R>, source: Input, pred: Pred) -> Result<bool, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Pred: op::PredicateOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<R>>::all_of_dispatch(source, exec.policy(), pred)
}

/// Returns whether any element satisfies `pred`.
pub fn any_of<R, Input, Pred>(exec: &Executor<R>, source: Input, pred: Pred) -> Result<bool, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Pred: op::PredicateOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<R>>::any_of_dispatch(source, exec.policy(), pred)
}

/// Counts elements satisfying `pred`.
pub fn count_if<R, Input, Pred>(
    exec: &Executor<R>,
    source: Input,
    pred: Pred,
) -> Result<usize, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Pred: op::PredicateOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<R>>::count_if_dispatch(source, exec.policy(), pred)
}

/// Finds the first element satisfying `pred`.
pub fn find_if<R, Input, Pred>(
    exec: &Executor<R>,
    source: Input,
    pred: Pred,
) -> Result<Option<usize>, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Pred: op::PredicateOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<R>>::find_if_dispatch(source, exec.policy(), pred)
}

/// Returns whether input is partitioned by `pred`.
pub fn is_partitioned<R, Input, Pred>(
    exec: &Executor<R>,
    source: Input,
    pred: Pred,
) -> Result<bool, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Pred: op::PredicateOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<R>>::is_partitioned_dispatch(source, exec.policy(), pred)
}

/// Returns whether no elements satisfy `pred`.
pub fn none_of<R, Input, Pred>(exec: &Executor<R>, source: Input, pred: Pred) -> Result<bool, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Pred: op::PredicateOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<R>>::none_of_dispatch(source, exec.policy(), pred)
}

/// Partitions elements by `pred`.
pub fn partition<R, Input, Output, Pred>(
    exec: &Executor<R>,
    source: Input,
    pred: Pred,
) -> Result<(Output, Output), Error>
where
    R: Runtime,
    Input: MIter<R>,
    Output: MVec<R, Item = Input::Item>,
    Pred: op::PredicateOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<R>>::partition_dispatch(source, exec.policy(), pred)
}
