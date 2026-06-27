use super::*;

/// Removes consecutive duplicates under `pred`.
pub fn unique<B, Input, Output, Pred>(
    exec: &Executor<B>,
    source: Input,
    pred: Pred,
) -> Result<Output, Error>
where
    B: Runtime,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
    Pred: op::BinaryPredicateOp<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::unique_dispatch(source, exec.policy(), pred)
}

/// Removes consecutive duplicate keys and keeps their values.
pub fn unique_by_key<B, Keys, Values, Eq, KeyOutput, ValueOutput>(
    exec: &Executor<B>,
    keys: Keys,
    values: Values,
    eq: Eq,
) -> Result<(KeyOutput, ValueOutput), Error>
where
    B: Runtime,
    Keys: MIter<B>,
    Values: MIter<B>,
    Eq: op::BinaryPredicateOp<B, Keys::Item>,
    KeyOutput: MVec<B, Item = Keys::Item>,
    ValueOutput: MVec<B, Item = Values::Item>,
{
    validate_input(exec, &keys)?;
    validate_input(exec, &values)?;
    <Keys as sealed::MIterDispatch<B>>::unique_by_key_dispatch(keys, exec.policy(), values, eq)
}
