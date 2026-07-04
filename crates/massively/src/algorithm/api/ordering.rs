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
    Right: MIter<R, Item = Output::Item>,
    Less: op::BinaryPredicateOp<R, Output::Item>,
{
    validate_input(exec, &left)?;
    validate_input(exec, &right)?;
    validate_output(exec, &out)?;
    let owned: <Output::Item as MAlloc<R>>::Storage =
        <Left as sealed::MIterDispatch<R>>::merge_dispatch(left, exec.policy(), right, less)?;
    write_owned_output(exec.policy(), owned, out)
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
    RightKeys: MIter<R, Item = KeyOutput::Item>,
    LeftValues: MIter<R, Item = ValueOutput::Item>,
    RightValues: MIter<R, Item = ValueOutput::Item>,
    Less: op::BinaryPredicateOp<R, KeyOutput::Item>,
{
    validate_input(exec, &left_keys)?;
    validate_input(exec, &left_values)?;
    validate_input(exec, &right_keys)?;
    validate_input(exec, &right_values)?;
    validate_output(exec, &out_k)?;
    validate_output(exec, &out_v)?;
    let (keys, values): (
        <KeyOutput::Item as MAlloc<R>>::Storage,
        <ValueOutput::Item as MAlloc<R>>::Storage,
    ) = <LeftKeys as sealed::MIterDispatch<R>>::merge_by_key_dispatch(
        left_keys,
        exec.policy(),
        right_keys,
        left_values,
        right_values,
        less,
    )?;
    write_owned_output(exec.policy(), keys, out_k)?;
    write_owned_output(exec.policy(), values, out_v)
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
    let owned: <Output::Item as MAlloc<R>>::Storage =
        <Input as sealed::MIterDispatch<R>>::reverse_dispatch(source, exec.policy())?;
    write_owned_output(exec.policy(), owned, out)
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
    Less: op::BinaryPredicateOp<R, Output::Item>,
{
    validate_input(exec, &source)?;
    validate_output(exec, &out)?;
    let owned: <Output::Item as MAlloc<R>>::Storage =
        <Input as sealed::MIterDispatch<R>>::sort_dispatch(source, exec.policy(), less)?;
    write_owned_output(exec.policy(), owned, out)
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
    Less: op::BinaryPredicateOp<R, KeyOutput::Item>,
{
    validate_input(exec, &keys)?;
    validate_input(exec, &values)?;
    validate_output(exec, &out_k)?;
    validate_output(exec, &out_v)?;
    let (sorted_keys, sorted_values): (
        <KeyOutput::Item as MAlloc<R>>::Storage,
        <ValueOutput::Item as MAlloc<R>>::Storage,
    ) = <Keys as sealed::MIterDispatch<R>>::sort_by_key_dispatch(
        keys,
        exec.policy(),
        values,
        less,
    )?;
    write_owned_output(exec.policy(), sorted_keys, out_k)?;
    write_owned_output(exec.policy(), sorted_values, out_v)
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
    Less: op::BinaryPredicateOp<R, Output::Item>,
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
    Less: op::BinaryPredicateOp<R, KeyOutput::Item>,
{
    sort_by_key(exec, keys, values, less, out_k, out_v)
}
