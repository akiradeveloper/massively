use super::*;

/// Computes adjacent differences.
pub fn adjacent_difference<B, Input, Output, Op>(
    exec: &Executor<B>,
    source: Input,
    op: Op,
) -> Result<Output, Error>
where
    B: Runtime,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
    Op: op::ReductionOp<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::adjacent_difference_dispatch(source, exec.policy(), op)
}

/// Computes an exclusive scan.
pub fn exclusive_scan<B, Input, Output, Op>(
    exec: &Executor<B>,
    source: Input,
    init: Input::Item,
    op: Op,
) -> Result<Output, Error>
where
    B: Runtime,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
    Op: op::ReductionOp<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::exclusive_scan_dispatch(source, exec.policy(), init, op)
}

/// Exclusive scan by key.
pub fn exclusive_scan_by_key<B, Keys, Values, KeyEq, Op, Output>(
    exec: &Executor<B>,
    keys: Keys,
    values: Values,
    key_eq: KeyEq,
    init: Values::Item,
    op: Op,
) -> Result<Output, Error>
where
    B: Runtime,
    Keys: MIter<B>,
    Values: MIter<B>,
    KeyEq: op::BinaryPredicateOp<B, Keys::Item>,
    Op: op::ReductionOp<B, Values::Item>,
    Output: MVec<B, Item = Values::Item>,
{
    validate_input(exec, &keys)?;
    validate_input(exec, &values)?;
    <Keys as sealed::MIterDispatch<B>>::exclusive_scan_by_key_dispatch(
        keys,
        exec.policy(),
        values,
        key_eq,
        init,
        op,
    )
}

/// Computes an inclusive scan.
pub fn inclusive_scan<B, Input, Output, Op>(
    exec: &Executor<B>,
    source: Input,
    op: Op,
) -> Result<Output, Error>
where
    B: Runtime,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
    Op: op::ReductionOp<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::inclusive_scan_dispatch(source, exec.policy(), op)
}

/// Inclusive scan by key.
pub fn inclusive_scan_by_key<B, Keys, Values, KeyEq, Op, Output>(
    exec: &Executor<B>,
    keys: Keys,
    values: Values,
    key_eq: KeyEq,
    op: Op,
) -> Result<Output, Error>
where
    B: Runtime,
    Keys: MIter<B>,
    Values: MIter<B>,
    KeyEq: op::BinaryPredicateOp<B, Keys::Item>,
    Op: op::ReductionOp<B, Values::Item>,
    Output: MVec<B, Item = Values::Item>,
{
    validate_input(exec, &keys)?;
    validate_input(exec, &values)?;
    <Keys as sealed::MIterDispatch<B>>::inclusive_scan_by_key_dispatch(
        keys,
        exec.policy(),
        values,
        key_eq,
        op,
    )
}
