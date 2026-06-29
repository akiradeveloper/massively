use super::*;

/// Merges two sorted inputs.
pub fn merge<R, Left, Right, Less>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    less: Less,
) -> Result<<Left::Item as MItem<R>>::Vec, Error>
where
    R: Runtime,
    Left: MIter<R>,
    Right: MIter<R, Item = Left::Item>,
    Less: op::BinaryPredicateOp<R, Left::Item>,
{
    validate_input(exec, &left)?;
    validate_input(exec, &right)?;
    <Left as sealed::MIterDispatch<R>>::merge_dispatch(left, exec.policy(), right, less)
}

/// Merges two sorted key-value ranges by key.
pub fn merge_by_key<R, LeftKeys, LeftValues, RightKeys, RightValues, Less>(
    exec: &Executor<R>,
    left_keys: LeftKeys,
    left_values: LeftValues,
    right_keys: RightKeys,
    right_values: RightValues,
    less: Less,
) -> Result<
    (
        <LeftKeys::Item as MItem<R>>::Vec,
        <LeftValues::Item as MItem<R>>::Vec,
    ),
    Error,
>
where
    R: Runtime,
    LeftKeys: MIter<R>,
    RightKeys: MIter<R, Item = LeftKeys::Item>,
    LeftValues: MIter<R>,
    RightValues: MIter<R, Item = LeftValues::Item>,
    Less: op::BinaryPredicateOp<R, LeftKeys::Item>,
{
    validate_input(exec, &left_keys)?;
    validate_input(exec, &left_values)?;
    validate_input(exec, &right_keys)?;
    validate_input(exec, &right_values)?;
    <LeftKeys as sealed::MIterDispatch<R>>::merge_by_key_dispatch(
        left_keys,
        exec.policy(),
        right_keys,
        left_values,
        right_values,
        less,
    )
}

/// Reverses a massively iterator into an owned vector.
pub fn reverse<R, Input>(
    exec: &Executor<R>,
    source: Input,
) -> Result<<Input::Item as MItem<R>>::Vec, Error>
where
    R: Runtime,
    Input: MIter<R>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<R>>::reverse_dispatch(source, exec.policy())
}

/// Sorts a massively iterator into an owned vector.
pub fn sort<R, Input, Less>(
    exec: &Executor<R>,
    source: Input,
    less: Less,
) -> Result<<Input::Item as MItem<R>>::Vec, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Less: op::BinaryPredicateOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<R>>::sort_dispatch(source, exec.policy(), less)
}

/// Sorts key-value pairs by key.
pub fn sort_by_key<R, Keys, Values, Less>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    less: Less,
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
    Less: op::BinaryPredicateOp<R, Keys::Item>,
{
    validate_input(exec, &keys)?;
    validate_input(exec, &values)?;
    <Keys as sealed::MIterDispatch<R>>::sort_by_key_dispatch(keys, exec.policy(), values, less)
}

/// Stable sort. The current lower implementation is stable.
pub fn stable_sort<R, Input, Less>(
    exec: &Executor<R>,
    source: Input,
    less: Less,
) -> Result<<Input::Item as MItem<R>>::Vec, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Less: op::BinaryPredicateOp<R, Input::Item>,
{
    sort(exec, source, less)
}

/// Stable key-value sort. The current lower implementation is stable.
pub fn stable_sort_by_key<R, Keys, Values, Less>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    less: Less,
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
    Less: op::BinaryPredicateOp<R, Keys::Item>,
{
    sort_by_key(exec, keys, values, less)
}
