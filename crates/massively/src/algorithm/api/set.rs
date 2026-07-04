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
    Right: MIter<R, Item = Output::Item>,
    Less: op::BinaryPredicateOp<R, Output::Item>,
{
    validate_input(exec, &left)?;
    validate_input(exec, &right)?;
    validate_output(exec, &out)?;
    <Left as sealed::MIterDispatch<R>>::set_difference_into_dispatch(
        left,
        exec.policy(),
        right,
        less,
        out,
    )
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
    Right: MIter<R, Item = Output::Item>,
    Less: op::BinaryPredicateOp<R, Output::Item>,
{
    validate_input(exec, &left)?;
    validate_input(exec, &right)?;
    validate_output(exec, &out)?;
    <Left as sealed::MIterDispatch<R>>::set_intersection_into_dispatch(
        left,
        exec.policy(),
        right,
        less,
        out,
    )
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
    Right: MIter<R, Item = Output::Item>,
    Less: op::BinaryPredicateOp<R, Output::Item>,
{
    validate_input(exec, &left)?;
    validate_input(exec, &right)?;
    validate_output(exec, &out)?;
    <Left as sealed::MIterDispatch<R>>::set_union_into_dispatch(
        left,
        exec.policy(),
        right,
        less,
        out,
    )
}
