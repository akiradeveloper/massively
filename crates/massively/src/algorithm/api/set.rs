use super::*;

/// Computes the sorted set difference of two sorted inputs.
pub fn set_difference<R, Left, Right, Less, Output>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    less: Less,
    out: Output,
) -> Result<MIndex, Error>
where
    R: Runtime,
    Output: MIterMut<R>,
    Left: MIter<R, Item = Output::Item>,
    Right: MIter<R, Item = Left::Item>,
    Less: op::BinaryPredicateOp<R, Left::Item>,
{
    validate_input(exec, &left)?;
    validate_input(exec, &right)?;
    validate_output(exec, &out)?;
    left.set_difference_with_policy(exec.policy(), right, less, out)
}

/// Computes the sorted set intersection of two sorted inputs.
pub fn set_intersection<R, Left, Right, Less, Output>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    less: Less,
    out: Output,
) -> Result<MIndex, Error>
where
    R: Runtime,
    Output: MIterMut<R>,
    Left: MIter<R, Item = Output::Item>,
    Right: MIter<R, Item = Left::Item>,
    Less: op::BinaryPredicateOp<R, Left::Item>,
{
    validate_input(exec, &left)?;
    validate_input(exec, &right)?;
    validate_output(exec, &out)?;
    left.set_intersection_with_policy(exec.policy(), right, less, out)
}

/// Computes the sorted set union of two sorted inputs.
pub fn set_union<R, Left, Right, Less, Output>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    less: Less,
    out: Output,
) -> Result<MIndex, Error>
where
    R: Runtime,
    Output: MIterMut<R>,
    Left: MIter<R, Item = Output::Item>,
    Right: MIter<R, Item = Left::Item>,
    Less: op::BinaryPredicateOp<R, Left::Item>,
{
    validate_input(exec, &left)?;
    validate_input(exec, &right)?;
    validate_output(exec, &out)?;
    left.set_union_with_policy(exec.policy(), right, less, out)
}
