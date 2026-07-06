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
    Input: MIterReduce<R, Op>,
    Op: op::ReductionOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    MIterReduce::reduce_value_with_policy(source, exec.policy(), init, op)
}

/// Reduces consecutive values with equal keys.
pub fn reduce_by_key<R, Keys, Values, KeyEq, Op, KeyOutput, ValueOutput>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    key_eq: KeyEq,
    init: Values::Item,
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
    KeyEq: op::BinaryPredicateOp<R, Keys::Item>,
    Op: op::ReductionOp<R, Values::Item>,
{
    validate_input(exec, &keys)?;
    validate_input(exec, &values)?;
    validate_output(exec, &out_k)?;
    validate_output(exec, &out_v)?;
    keys.reduce_by_key_with_policy(exec.policy(), values, key_eq, init, op, out_k, out_v)
}
