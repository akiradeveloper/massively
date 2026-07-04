use super::*;

/// Reduces a massively iterator to one host item.
pub fn reduce<R, Input, Op>(
    exec: &Executor<R>,
    source: Input,
    init: Input::Item,
    op: Op,
) -> Result<Input::Item, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Op: op::ReductionOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<R>>::reduce_dispatch(source, exec.policy(), init, op)
}

/// Reduces consecutive values with equal keys.
pub fn reduce_by_key<R, Keys, Values, KeyEq, Op, KeyOutput, ValueOutput>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    key_eq: KeyEq,
    init: ValueOutput::Item,
    op: Op,
    out_k: KeyOutput,
    out_v: ValueOutput,
) -> Result<MIndex, Error>
where
    R: Runtime,
    KeyOutput: MIterMut<R>,
    ValueOutput: MIterMut<R>,
    Keys: MIter<R, Item = KeyOutput::Item>,
    Values: MIter<R, Item = ValueOutput::Item>,
    KeyEq: op::BinaryPredicateOp<R, KeyOutput::Item>,
    Op: op::ReductionOp<R, ValueOutput::Item>,
{
    validate_input(exec, &keys)?;
    validate_input(exec, &values)?;
    validate_output(exec, &out_k)?;
    validate_output(exec, &out_v)?;
    let (reduced_keys, reduced_values): (
        <KeyOutput::Item as MAlloc<R>>::Storage,
        <ValueOutput::Item as MAlloc<R>>::Storage,
    ) = <Keys as sealed::MIterDispatch<R>>::reduce_by_key_dispatch(
        keys,
        exec.policy(),
        values,
        key_eq,
        init,
        op,
    )?;
    let len = reduced_keys.len();
    out_k.write_prefix_from_inner(exec.policy(), reduced_keys.into_inner())?;
    out_v.write_prefix_from_inner(exec.policy(), reduced_values.into_inner())?;
    Ok(len)
}
