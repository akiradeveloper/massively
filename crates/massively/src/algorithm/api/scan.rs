use super::*;

/// Computes adjacent differences.
pub fn adjacent_difference<R, Input, Op, Output>(
    exec: &Executor<R>,
    source: Input,
    op: Op,
    out: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Output: MIterMut<R>,
    Input: MIter<R, Item = Output::Item>,
    Op: op::ReductionOp<R, Output::Item>,
{
    validate_input(exec, &source)?;
    validate_output(exec, &out)?;
    <Input as sealed::MIterDispatch<R>>::adjacent_difference_into_dispatch(
        source,
        exec.policy(),
        op,
        out,
    )
}

/// Computes an exclusive scan.
pub fn exclusive_scan<R, Input, Op, Output>(
    exec: &Executor<R>,
    source: Input,
    init: Output::Item,
    op: Op,
    out: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Output: MIterMut<R>,
    Input: MIter<R, Item = Output::Item>,
    Op: op::ReductionOp<R, Output::Item>,
{
    validate_input(exec, &source)?;
    validate_output(exec, &out)?;
    <Input as sealed::MIterDispatch<R>>::exclusive_scan_into_dispatch(
        source,
        exec.policy(),
        init,
        op,
        out,
    )
}

/// Exclusive scan by key.
pub fn exclusive_scan_by_key<R, Keys, Values, KeyEq, Op, Output>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    key_eq: KeyEq,
    init: Output::Item,
    op: Op,
    out: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Keys: MIter<R>,
    Output: MIterMut<R>,
    Values: MIter<R, Item = Output::Item>,
    KeyEq: op::BinaryPredicateOp<R, Keys::Item>,
    Op: op::ReductionOp<R, Output::Item>,
{
    validate_input(exec, &keys)?;
    validate_input(exec, &values)?;
    validate_output(exec, &out)?;
    <Keys as sealed::MIterDispatch<R>>::exclusive_scan_by_key_into_dispatch(
        keys,
        exec.policy(),
        values,
        key_eq,
        init,
        op,
        out,
    )
}

/// Computes an inclusive scan.
pub fn inclusive_scan<R, Input, Op, Output>(
    exec: &Executor<R>,
    source: Input,
    op: Op,
    out: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Output: MIterMut<R>,
    Input: MIter<R, Item = Output::Item>,
    Op: op::ReductionOp<R, Output::Item>,
{
    validate_input(exec, &source)?;
    validate_output(exec, &out)?;
    <Input as sealed::MIterDispatch<R>>::inclusive_scan_into_dispatch(
        source,
        exec.policy(),
        op,
        out,
    )
}

/// Inclusive scan by key.
pub fn inclusive_scan_by_key<R, Keys, Values, KeyEq, Op, Output>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    key_eq: KeyEq,
    op: Op,
    out: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Keys: MIter<R>,
    Output: MIterMut<R>,
    Values: MIter<R, Item = Output::Item>,
    KeyEq: op::BinaryPredicateOp<R, Keys::Item>,
    Op: op::ReductionOp<R, Output::Item>,
{
    validate_input(exec, &keys)?;
    validate_input(exec, &values)?;
    validate_output(exec, &out)?;
    <Keys as sealed::MIterDispatch<R>>::inclusive_scan_by_key_into_dispatch(
        keys,
        exec.policy(),
        values,
        key_eq,
        op,
        out,
    )
}
