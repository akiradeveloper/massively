use super::*;

/// Computes adjacent differences.
pub fn adjacent_difference<R, Input, Output, Op>(
    exec: &Executor<R>,
    source: Input,
    op: Op,
) -> Result<Output, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Output: MVec<R, Item = Input::Item>,
    Op: op::ReductionOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<R>>::adjacent_difference_dispatch(source, exec.policy(), op)
}

/// Computes an exclusive scan.
pub fn exclusive_scan<R, Input, Output, Op>(
    exec: &Executor<R>,
    source: Input,
    init: Input::Item,
    op: Op,
) -> Result<Output, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Output: MVec<R, Item = Input::Item>,
    Op: op::ReductionOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<R>>::exclusive_scan_dispatch(source, exec.policy(), init, op)
}

/// Exclusive scan by key.
pub fn exclusive_scan_by_key<R, Keys, Values, KeyEq, Op, Output>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    key_eq: KeyEq,
    init: Values::Item,
    op: Op,
) -> Result<Output, Error>
where
    R: Runtime,
    Keys: MIter<R>,
    Values: MIter<R>,
    KeyEq: op::BinaryPredicateOp<R, Keys::Item>,
    Op: op::ReductionOp<R, Values::Item>,
    Output: MVec<R, Item = Values::Item>,
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
pub fn inclusive_scan<R, Input, Output, Op>(
    exec: &Executor<R>,
    source: Input,
    op: Op,
) -> Result<Output, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Output: MVec<R, Item = Input::Item>,
    Op: op::ReductionOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<R>>::inclusive_scan_dispatch(source, exec.policy(), op)
}

/// Inclusive scan by key.
pub fn inclusive_scan_by_key<R, Keys, Values, KeyEq, Op, Output>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    key_eq: KeyEq,
    op: Op,
) -> Result<Output, Error>
where
    R: Runtime,
    Keys: MIter<R>,
    Values: MIter<R>,
    KeyEq: op::BinaryPredicateOp<R, Keys::Item>,
    Op: op::ReductionOp<R, Values::Item>,
    Output: MVec<R, Item = Values::Item>,
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
