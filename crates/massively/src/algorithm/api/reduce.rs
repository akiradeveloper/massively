use super::*;

/// Applies a binary transform over two inputs and reduces the result.
pub fn inner_product<R, Left, Right, ZipperOp, ReduceOp>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    transform_op: ZipperOp,
    init: ZipperOp::Output,
    reduce_op: ReduceOp,
) -> Result<ZipperOp::Output, Error>
where
    R: Runtime,
    Left: MIter<R>,
    Right: MIter<R>,
    ZipperOp: op::BinaryOp<R, Left::Item, Right::Item>,
    ReduceOp: op::ReductionOp<R, ZipperOp::Output>,
{
    validate_input(exec, &left)?;
    validate_input(exec, &right)?;
    <Left::Item as sealed::MItemDispatch<R>>::inner_product_with_right_item::<
        Left,
        Right,
        ZipperOp,
        ReduceOp,
        ZipperOp::Output,
    >(exec.policy(), left, right, transform_op, init, reduce_op)
}

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
pub fn reduce_by_key<R, Keys, Values, KeyEq, Op, KeyOutput, ValueOutput>(
    exec: &Executor<R>,
    keys: Keys,
    values: Values,
    key_eq: KeyEq,
    init: Values::Item,
    op: Op,
) -> Result<(KeyOutput, ValueOutput), Error>
where
    R: Runtime,
    Keys: MIter<R>,
    Values: MIter<R>,
    KeyEq: op::BinaryPredicateOp<R, Keys::Item>,
    Op: op::ReductionOp<R, Values::Item>,
    KeyOutput: MVec<R, Item = Keys::Item>,
    ValueOutput: MVec<R, Item = Values::Item>,
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
