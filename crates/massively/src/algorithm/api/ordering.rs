use super::*;

/// Merges two sorted inputs.
pub fn merge<R, Left, Right, Less, Output>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    less: Less,
    out: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Output: MIterMut<R>,
    Left: MIter<R, Item = Output::Item>,
    Right: MIter<R, Item = Left::Item>,
    Less: op::BinaryPredicateOp<R, Left::Item>,
{
    validate_input(exec, &left)?;
    validate_input(exec, &right)?;
    validate_output(exec, &out)?;
    <Left as sealed::MIterDispatch<R>>::merge_into_dispatch(left, exec.policy(), right, less, out)
}

/// Merges two sorted key-value ranges by key.
pub fn merge_by_key<R, LeftKeys, LeftValues, RightKeys, RightValues, Less, KeyOutput, ValueOutput>(
    exec: &Executor<R>,
    left_keys: LeftKeys,
    left_values: LeftValues,
    right_keys: RightKeys,
    right_values: RightValues,
    less: Less,
    out_k: KeyOutput,
    out_v: ValueOutput,
) -> Result<(), Error>
where
    R: Runtime,
    KeyOutput: MIterMut<R>,
    ValueOutput: MIterMut<R>,
    LeftKeys: MIter<R, Item = KeyOutput::Item>,
    RightKeys: MIter<R, Item = LeftKeys::Item>,
    LeftValues: MIter<R, Item = ValueOutput::Item>,
    RightValues: MIter<R, Item = LeftValues::Item>,
    Less: op::BinaryPredicateOp<R, LeftKeys::Item>,
{
    validate_input(exec, &left_keys)?;
    validate_input(exec, &left_values)?;
    validate_input(exec, &right_keys)?;
    validate_input(exec, &right_values)?;
    validate_output(exec, &out_k)?;
    validate_output(exec, &out_v)?;
    <LeftKeys as sealed::MIterDispatch<R>>::merge_by_key_into_dispatch(
        left_keys,
        exec.policy(),
        right_keys,
        left_values,
        right_values,
        less,
        out_k,
        out_v,
    )
}

/// Reverses a massively iterator.
pub fn reverse<R, Input, Output>(
    exec: &Executor<R>,
    source: Input,
    out: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Output: MIterMut<R>,
    Input: MIter<R, Item = Output::Item>,
{
    validate_input(exec, &source)?;
    validate_output(exec, &out)?;
    <Input as sealed::MIterDispatch<R>>::reverse_into_dispatch(source, exec.policy(), out)
}

/// Sorts a massively iterator.
pub fn sort<R, Input, Less, Output>(
    exec: &Executor<R>,
    source: Input,
    less: Less,
    out: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Output: MIterMut<R>,
    Input: MIter<R, Item = Output::Item>,
    Less: op::BinaryPredicateOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    validate_output(exec, &out)?;
    <Input as sealed::MIterDispatch<R>>::sort_into_dispatch(source, exec.policy(), less, out)
}

/// Sorts key-value pairs by key.
pub fn sort_by_key<R, Keys, Values, Less, KeyOutput, ValueOutput>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    less: Less,
    out_k: KeyOutput,
    out_v: ValueOutput,
) -> Result<(), Error>
where
    R: Runtime,
    KeyOutput: MIterMut<R>,
    ValueOutput: MIterMut<R>,
    Keys: MIter<R, Item = KeyOutput::Item>,
    Values: MIter<R, Item = ValueOutput::Item>,
    Less: op::BinaryPredicateOp<R, Keys::Item>,
{
    validate_input(exec, &keys)?;
    validate_input(exec, &values)?;
    validate_output(exec, &out_k)?;
    validate_output(exec, &out_v)?;
    <Keys as sealed::MIterDispatch<R>>::sort_by_key_into_dispatch(
        keys,
        exec.policy(),
        values,
        less,
        out_k,
        out_v,
    )
}

/// Stable sort. The current lower implementation is stable.
pub fn stable_sort<R, Input, Less, Output>(
    exec: &Executor<R>,
    source: Input,
    less: Less,
    out: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Output: MIterMut<R>,
    Input: MIter<R, Item = Output::Item>,
    Less: op::BinaryPredicateOp<R, Input::Item>,
{
    sort(exec, source, less, out)
}

/// Stable key-value sort. The current lower implementation is stable.
pub fn stable_sort_by_key<R, Keys, Values, Less, KeyOutput, ValueOutput>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    less: Less,
    out_k: KeyOutput,
    out_v: ValueOutput,
) -> Result<(), Error>
where
    R: Runtime,
    KeyOutput: MIterMut<R>,
    ValueOutput: MIterMut<R>,
    Keys: MIter<R, Item = KeyOutput::Item>,
    Values: MIter<R, Item = ValueOutput::Item>,
    Less: op::BinaryPredicateOp<R, Keys::Item>,
{
    sort_by_key(exec, keys, values, less, out_k, out_v)
}
