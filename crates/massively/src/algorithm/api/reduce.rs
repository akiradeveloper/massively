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
pub fn reduce_by_key<R, Keys, Values, KeyEq, Op>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    key_eq: KeyEq,
    init: Values::Item,
    op: Op,
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
    KeyEq: op::BinaryPredicateOp<R, Keys::Item>,
    Op: op::ReductionOp<R, Values::Item>,
{
    validate_input(exec, &keys)?;
    validate_input(exec, &values)?;
    <Keys as sealed::MIterDispatch<R>>::reduce_by_key_dispatch(
        keys,
        exec.policy(),
        values,
        key_eq,
        init,
        op,
    )
}
