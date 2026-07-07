use super::*;

/// Returns whether all elements satisfy `pred`.
pub fn all_of<R, Input, Pred>(exec: &Executor<R>, source: Input, pred: Pred) -> Result<bool, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Pred: op::PredicateOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    source.all_of_with_policy(exec.policy(), pred)
}

/// Returns whether any element satisfies `pred`.
pub fn any_of<R, Input, Pred>(exec: &Executor<R>, source: Input, pred: Pred) -> Result<bool, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Pred: op::PredicateOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    source.any_of_with_policy(exec.policy(), pred)
}

/// Counts elements satisfying `pred`.
pub fn count_if<R, Input, Pred>(
    exec: &Executor<R>,
    source: Input,
    pred: Pred,
) -> Result<MIndex, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Pred: op::PredicateOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    source.count_if_with_policy(exec.policy(), pred)
}

/// Finds the first element satisfying `pred`.
pub fn find_if<R, Input, Pred>(
    exec: &Executor<R>,
    source: Input,
    pred: Pred,
) -> Result<Option<MIndex>, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Pred: op::PredicateOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    source.find_if_with_policy(exec.policy(), pred)
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
    source.is_partitioned_with_policy(exec.policy(), pred)
}

/// Returns whether no elements satisfy `pred`.
pub fn none_of<R, Input, Pred>(exec: &Executor<R>, source: Input, pred: Pred) -> Result<bool, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Pred: op::PredicateOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    source.none_of_with_policy(exec.policy(), pred)
}

/// Partitions elements by `pred`.
pub fn partition<R, Input, Pred, Output>(
    exec: &Executor<R>,
    source: Input,
    pred: Pred,
    out: Output,
) -> Result<MIndex, Error>
where
    R: Runtime,
    Output: MIterMut<R>,
    Input: MIter<R, Item = Output::Item>,
    Pred: op::PredicateOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    validate_output(exec, &out)?;
    source.partition_with_policy(exec.policy(), pred, out)
}
