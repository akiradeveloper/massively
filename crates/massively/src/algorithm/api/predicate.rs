use super::*;

/// Returns whether all elements satisfy `pred`.
pub fn all_of<B, Input, Pred>(exec: &Executor<B>, source: Input, pred: Pred) -> Result<bool, Error>
where
    B: Runtime,
    Input: MIter<B>,
    Pred: op::PredicateOp<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::all_of_dispatch(source, exec.policy(), pred)
}

/// Returns whether any element satisfies `pred`.
pub fn any_of<B, Input, Pred>(exec: &Executor<B>, source: Input, pred: Pred) -> Result<bool, Error>
where
    B: Runtime,
    Input: MIter<B>,
    Pred: op::PredicateOp<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::any_of_dispatch(source, exec.policy(), pred)
}

/// Counts elements satisfying `pred`.
pub fn count_if<B, Input, Pred>(
    exec: &Executor<B>,
    source: Input,
    pred: Pred,
) -> Result<usize, Error>
where
    B: Runtime,
    Input: MIter<B>,
    Pred: op::PredicateOp<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::count_if_dispatch(source, exec.policy(), pred)
}

/// Finds the first element satisfying `pred`.
pub fn find_if<B, Input, Pred>(
    exec: &Executor<B>,
    source: Input,
    pred: Pred,
) -> Result<Option<usize>, Error>
where
    B: Runtime,
    Input: MIter<B>,
    Pred: op::PredicateOp<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::find_if_dispatch(source, exec.policy(), pred)
}

/// Returns whether input is partitioned by `pred`.
pub fn is_partitioned<B, Input, Pred>(
    exec: &Executor<B>,
    source: Input,
    pred: Pred,
) -> Result<bool, Error>
where
    B: Runtime,
    Input: MIter<B>,
    Pred: op::PredicateOp<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::is_partitioned_dispatch(source, exec.policy(), pred)
}

/// Returns whether no elements satisfy `pred`.
pub fn none_of<B, Input, Pred>(exec: &Executor<B>, source: Input, pred: Pred) -> Result<bool, Error>
where
    B: Runtime,
    Input: MIter<B>,
    Pred: op::PredicateOp<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::none_of_dispatch(source, exec.policy(), pred)
}

/// Partitions elements by `pred`.
pub fn partition<B, Input, Output, Pred>(
    exec: &Executor<B>,
    source: Input,
    pred: Pred,
) -> Result<(Output, Output), Error>
where
    B: Runtime,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
    Pred: op::PredicateOp<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::partition_dispatch(source, exec.policy(), pred)
}
