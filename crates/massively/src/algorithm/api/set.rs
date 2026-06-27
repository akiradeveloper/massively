use super::*;

/// Computes the sorted set difference of two sorted inputs.
pub fn set_difference<B, Left, Right, Output, Less>(
    exec: &Executor<B>,
    left: Left,
    right: Right,
    less: Less,
) -> Result<Output, Error>
where
    B: Runtime,
    Left: MIter<B>,
    Right: MIter<B, Item = Left::Item>,
    Output: MVec<B, Item = Left::Item>,
    Less: op::BinaryPredicateOp<B, Left::Item>,
{
    validate_input(exec, &left)?;
    validate_input(exec, &right)?;
    <Left as sealed::MIterDispatch<B>>::set_difference_dispatch(left, exec.policy(), right, less)
}

/// Computes the sorted set intersection of two sorted inputs.
pub fn set_intersection<B, Left, Right, Output, Less>(
    exec: &Executor<B>,
    left: Left,
    right: Right,
    less: Less,
) -> Result<Output, Error>
where
    B: Runtime,
    Left: MIter<B>,
    Right: MIter<B, Item = Left::Item>,
    Output: MVec<B, Item = Left::Item>,
    Less: op::BinaryPredicateOp<B, Left::Item>,
{
    validate_input(exec, &left)?;
    validate_input(exec, &right)?;
    <Left as sealed::MIterDispatch<B>>::set_intersection_dispatch(left, exec.policy(), right, less)
}

/// Computes the sorted set union of two sorted inputs.
pub fn set_union<B, Left, Right, Output, Less>(
    exec: &Executor<B>,
    left: Left,
    right: Right,
    less: Less,
) -> Result<Output, Error>
where
    B: Runtime,
    Left: MIter<B>,
    Right: MIter<B, Item = Left::Item>,
    Output: MVec<B, Item = Left::Item>,
    Less: op::BinaryPredicateOp<B, Left::Item>,
{
    validate_input(exec, &left)?;
    validate_input(exec, &right)?;
    <Left as sealed::MIterDispatch<B>>::set_union_dispatch(left, exec.policy(), right, less)
}
