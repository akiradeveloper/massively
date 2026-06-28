use super::*;

/// Computes the sorted set difference of two sorted inputs.
pub fn set_difference<R, Left, Right, Less>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    less: Less,
) -> Result<<Left::Item as MItem<R>>::Vec, Error>
where
    R: Runtime,
    Left: MIter<R>,
    Right: MIter<R, Item = Left::Item, Inner = Left::Inner>,
    Less: op::BinaryPredicateOp<R, Left::Item>,
{
    validate_input(exec, &left)?;
    validate_input(exec, &right)?;
    <Left as sealed::MIterDispatch<R>>::set_difference_dispatch(left, exec.policy(), right, less)
}

/// Computes the sorted set intersection of two sorted inputs.
pub fn set_intersection<R, Left, Right, Less>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    less: Less,
) -> Result<<Left::Item as MItem<R>>::Vec, Error>
where
    R: Runtime,
    Left: MIter<R>,
    Right: MIter<R, Item = Left::Item, Inner = Left::Inner>,
    Less: op::BinaryPredicateOp<R, Left::Item>,
{
    validate_input(exec, &left)?;
    validate_input(exec, &right)?;
    <Left as sealed::MIterDispatch<R>>::set_intersection_dispatch(left, exec.policy(), right, less)
}

/// Computes the sorted set union of two sorted inputs.
pub fn set_union<R, Left, Right, Less>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    less: Less,
) -> Result<<Left::Item as MItem<R>>::Vec, Error>
where
    R: Runtime,
    Left: MIter<R>,
    Right: MIter<R, Item = Left::Item, Inner = Left::Inner>,
    Less: op::BinaryPredicateOp<R, Left::Item>,
{
    validate_input(exec, &left)?;
    validate_input(exec, &right)?;
    <Left as sealed::MIterDispatch<R>>::set_union_dispatch(left, exec.policy(), right, less)
}
