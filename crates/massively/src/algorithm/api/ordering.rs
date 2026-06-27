use super::*;

/// Merges two sorted inputs.
pub fn merge<B, Left, Right, Output, Less>(
    exec: &Executor<B>,
    left: Left,
    right: Right,
    less: Less,
) -> Result<Output, Error>
where
    B: Runtime,
    Left: MIter<B>,
    Right: MIter<B, Item = Left::Item>,
    Output: MVec<B, Item = Left::Item>,
    Less: op::BinaryPredicateOp<B, Left::Item>,
{
    validate_input(exec, &left)?;
    validate_input(exec, &right)?;
    <Left as sealed::MIterDispatch<B>>::merge_dispatch(left, exec.policy(), right, less)
}

/// Merges two sorted key-value ranges by key.
pub fn merge_by_key<B, LeftKeys, LeftValues, RightKeys, RightValues, Less, KeyOutput, ValueOutput>(
    exec: &Executor<B>,
    left_keys: LeftKeys,
    left_values: LeftValues,
    right_keys: RightKeys,
    right_values: RightValues,
    less: Less,
) -> Result<(KeyOutput, ValueOutput), Error>
where
    B: Runtime,
    LeftKeys: MIter<B>,
    RightKeys: MIter<B, Item = LeftKeys::Item>,
    LeftValues: MIter<B>,
    RightValues: MIter<B, Item = LeftValues::Item>,
    Less: op::BinaryPredicateOp<B, LeftKeys::Item>,
    KeyOutput: MVec<B, Item = LeftKeys::Item>,
    ValueOutput: MVec<B, Item = LeftValues::Item>,
{
    validate_input(exec, &left_keys)?;
    validate_input(exec, &left_values)?;
    validate_input(exec, &right_keys)?;
    validate_input(exec, &right_values)?;
    <LeftKeys as sealed::MIterDispatch<B>>::merge_by_key_dispatch(
        left_keys,
        exec.policy(),
        right_keys,
        left_values,
        right_values,
        less,
    )
}

/// Reverses a massively iterator into an owned vector.
pub fn reverse<B, Input, Output>(exec: &Executor<B>, source: Input) -> Result<Output, Error>
where
    B: Runtime,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::reverse_dispatch(source, exec.policy())
}

/// Sorts a massively iterator into an owned vector.
pub fn sort<B, Input, Output, Less>(
    exec: &Executor<B>,
    source: Input,
    less: Less,
) -> Result<Output, Error>
where
    B: Runtime,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
    Less: op::BinaryPredicateOp<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::sort_dispatch(source, exec.policy(), less)
}

/// Sorts key-value pairs by key.
pub fn sort_by_key<B, Keys, Values, Less, KeyOutput, ValueOutput>(
    exec: &Executor<B>,
    keys: Keys,
    values: Values,
    less: Less,
) -> Result<(KeyOutput, ValueOutput), Error>
where
    B: Runtime,
    Keys: MIter<B>,
    Values: MIter<B>,
    Less: op::BinaryPredicateOp<B, Keys::Item>,
    KeyOutput: MVec<B, Item = Keys::Item>,
    ValueOutput: MVec<B, Item = Values::Item>,
{
    validate_input(exec, &keys)?;
    validate_input(exec, &values)?;
    <Keys as sealed::MIterDispatch<B>>::sort_by_key_dispatch(keys, exec.policy(), values, less)
}

/// Stable sort. The current lower implementation is stable.
pub fn stable_sort<B, Input, Output, Less>(
    exec: &Executor<B>,
    source: Input,
    less: Less,
) -> Result<Output, Error>
where
    B: Runtime,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
    Less: op::BinaryPredicateOp<B, Input::Item>,
{
    sort(exec, source, less)
}

/// Stable key-value sort. The current lower implementation is stable.
pub fn stable_sort_by_key<B, Keys, Values, Less, KeyOutput, ValueOutput>(
    exec: &Executor<B>,
    keys: Keys,
    values: Values,
    less: Less,
) -> Result<(KeyOutput, ValueOutput), Error>
where
    B: Runtime,
    Keys: MIter<B>,
    Values: MIter<B>,
    Less: op::BinaryPredicateOp<B, Keys::Item>,
    KeyOutput: MVec<B, Item = Keys::Item>,
    ValueOutput: MVec<B, Item = Values::Item>,
{
    sort_by_key(exec, keys, values, less)
}
