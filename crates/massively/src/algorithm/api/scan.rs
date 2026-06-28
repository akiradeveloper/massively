use super::*;

/// Computes adjacent differences.
pub fn adjacent_difference<R, Input, Op>(
    exec: &Executor<R>,
    source: Input,
    op: Op,
) -> Result<<Input::Item as MItem<R>>::Vec, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Op: op::ReductionOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<R>>::adjacent_difference_dispatch(source, exec.policy(), op)
}

/// Computes an exclusive scan.
pub fn exclusive_scan<R, Input, Op>(
    exec: &Executor<R>,
    source: Input,
    init: Input::Item,
    op: Op,
) -> Result<<Input::Item as MItem<R>>::Vec, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Op: op::ReductionOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<R>>::exclusive_scan_dispatch(source, exec.policy(), init, op)
}

/// Exclusive scan by key.
pub fn exclusive_scan_by_key<R, Keys, Values, KeyEq, Op>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    key_eq: KeyEq,
    init: Values::Item,
    op: Op,
) -> Result<<Values::Item as MItem<R>>::Vec, Error>
where
    R: Runtime,
    Keys: MIter<R>,
    Values: MIter<R>,
    KeyEq: op::BinaryPredicateOp<R, Keys::Item>,
    Op: op::ReductionOp<R, Values::Item>,
{
    validate_input(exec, &keys)?;
    validate_input(exec, &values)?;
    <Keys as sealed::MIterDispatch<R>>::exclusive_scan_by_key_dispatch(
        keys,
        exec.policy(),
        values,
        key_eq,
        init,
        op,
    )
}

/// Computes an inclusive scan.
pub fn inclusive_scan<R, Input, Op>(
    exec: &Executor<R>,
    source: Input,
    op: Op,
) -> Result<<Input::Item as MItem<R>>::Vec, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Op: op::ReductionOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<R>>::inclusive_scan_dispatch(source, exec.policy(), op)
}

/// Inclusive scan by key.
pub fn inclusive_scan_by_key<R, Keys, Values, KeyEq, Op>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    key_eq: KeyEq,
    op: Op,
) -> Result<<Values::Item as MItem<R>>::Vec, Error>
where
    R: Runtime,
    Keys: MIter<R>,
    Values: MIter<R>,
    KeyEq: op::BinaryPredicateOp<R, Keys::Item>,
    Op: op::ReductionOp<R, Values::Item>,
{
    validate_input(exec, &keys)?;
    validate_input(exec, &values)?;
    <Keys as sealed::MIterDispatch<R>>::inclusive_scan_by_key_dispatch(
        keys,
        exec.policy(),
        values,
        key_eq,
        op,
    )
}
