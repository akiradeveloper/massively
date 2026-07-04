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
    let owned: <Output::Item as MAlloc<R>>::Storage =
        <Left as sealed::MIterDispatch<R>>::set_difference_dispatch(
            left,
            exec.policy(),
            right,
            less,
        )?;
    write_owned_prefix(exec.policy(), owned, out)
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
    let owned: <Output::Item as MAlloc<R>>::Storage =
        <Left as sealed::MIterDispatch<R>>::set_intersection_dispatch(
            left,
            exec.policy(),
            right,
            less,
        )?;
    write_owned_prefix(exec.policy(), owned, out)
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
    let owned: <Output::Item as MAlloc<R>>::Storage =
        <Left as sealed::MIterDispatch<R>>::set_union_dispatch(left, exec.policy(), right, less)?;
    write_owned_prefix(exec.policy(), owned, out)
}
