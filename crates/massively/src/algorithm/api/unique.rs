use super::*;

/// Removes consecutive duplicates under `pred`.
pub fn unique<R, Input, Pred>(
    exec: &Executor<R>,
    source: Input,
    pred: Pred,
) -> Result<<Input::Item as MItem<R>>::Vec, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Pred: op::BinaryPredicateOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<R>>::unique_dispatch(source, exec.policy(), pred)
}

/// Removes consecutive duplicate keys and keeps their values.
pub fn unique_by_key<R, Keys, Values, Eq>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    eq: Eq,
) -> Result<
    (
        <Keys::Item as MItem<R>>::Vec,
        <Values::Item as MItem<R>>::Vec,
    ),
    Error,
>
where
    R: Runtime,
    Keys: MIter<R>,
    Values: MIter<R>,
    Eq: op::BinaryPredicateOp<R, Keys::Item>,
{
    validate_input(exec, &keys)?;
    validate_input(exec, &values)?;
    <Keys as sealed::MIterDispatch<R>>::unique_by_key_dispatch(keys, exec.policy(), values, eq)
}
