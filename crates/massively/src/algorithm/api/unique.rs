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
    Pred: op::BinaryPredicateOp<R, Output::Item>,
{
    validate_input(exec, &source)?;
    validate_output(exec, &out)?;
    let owned: <Output::Item as MAlloc<R>>::Storage =
        <Input as sealed::MIterDispatch<R>>::unique_dispatch(source, exec.policy(), pred)?;
    write_owned_prefix(exec.policy(), owned, out)
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
    Eq: op::BinaryPredicateOp<R, KeyOutput::Item>,
{
    validate_input(exec, &keys)?;
    validate_input(exec, &values)?;
    validate_output(exec, &out_k)?;
    validate_output(exec, &out_v)?;
    let (unique_keys, unique_values): (
        <KeyOutput::Item as MAlloc<R>>::Storage,
        <ValueOutput::Item as MAlloc<R>>::Storage,
    ) = <Keys as sealed::MIterDispatch<R>>::unique_by_key_dispatch(
        keys,
        exec.policy(),
        values,
        eq,
    )?;
    let len = unique_keys.len();
    out_k.write_prefix_from_inner(exec.policy(), unique_keys.into_inner())?;
    out_v.write_prefix_from_inner(exec.policy(), unique_values.into_inner())?;
    Ok(len)
}
