use super::*;

/// Removes consecutive duplicates under `pred`.
pub fn unique<R, Input, Pred, Output>(
    exec: &Executor<R>,
    source: Input,
    pred: Pred,
    out: Output,
) -> Result<MIndex, Error>
where
    R: Runtime,
    Output: MIterMut<R>,
    Input: MIter<R, Item = Output::Item>,
    Pred: op::BinaryPredicateOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    validate_output(exec, &out)?;
    <Input as sealed::MIterDispatch<R>>::unique_into_dispatch(source, exec.policy(), pred, out)
}

/// Removes consecutive duplicate keys and keeps their values.
pub fn unique_by_key<R, Keys, Values, Eq, KeyOutput, ValueOutput>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    eq: Eq,
    out_k: KeyOutput,
    out_v: ValueOutput,
) -> Result<MIndex, Error>
where
    R: Runtime,
    KeyOutput: MIterMut<R>,
    ValueOutput: MIterMut<R>,
    Keys: MIter<R, Item = KeyOutput::Item>,
    Values: MIter<R, Item = ValueOutput::Item>,
    Eq: op::BinaryPredicateOp<R, Keys::Item>,
{
    validate_input(exec, &keys)?;
    validate_input(exec, &values)?;
    validate_output(exec, &out_k)?;
    validate_output(exec, &out_v)?;
    <Keys as sealed::MIterDispatch<R>>::unique_by_key_into_dispatch(
        keys,
        exec.policy(),
        values,
        eq,
        out_k,
        out_v,
    )
}
