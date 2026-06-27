use super::*;

/// Applies a binary transform over two inputs and reduces the result.
pub fn inner_product<B, Left, Right, ZipperOp, ReduceOp>(
    exec: &Executor<B>,
    left: Left,
    right: Right,
    transform_op: ZipperOp,
    init: ZipperOp::Output,
    reduce_op: ReduceOp,
) -> Result<ZipperOp::Output, Error>
where
    B: Runtime,
    Left: MIter<B>,
    Right: MIter<B>,
    ZipperOp: op::BinaryOp<B, Left::Item, Right::Item>,
    ReduceOp: op::ReductionOp<B, ZipperOp::Output>,
{
    validate_input(exec, &left)?;
    validate_input(exec, &right)?;
    <Left::Item as sealed::MItemDispatch<B>>::inner_product_with_right_item::<
        Left,
        Right,
        ZipperOp,
        ReduceOp,
        ZipperOp::Output,
    >(exec.policy(), left, right, transform_op, init, reduce_op)
}

/// Reduces a massively iterator to one host item.
pub fn reduce<B, Input, Op>(
    exec: &Executor<B>,
    source: Input,
    init: Input::Item,
    op: Op,
) -> Result<Input::Item, Error>
where
    B: Runtime,
    Input: MIter<B>,
    Op: op::ReductionOp<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::reduce_dispatch(source, exec.policy(), init, op)
}

/// Reduces consecutive values with equal keys.
pub fn reduce_by_key<B, Keys, Values, KeyEq, Op, KeyOutput, ValueOutput>(
    exec: &Executor<B>,
    keys: Keys,
    values: Values,
    key_eq: KeyEq,
    init: Values::Item,
    op: Op,
) -> Result<(KeyOutput, ValueOutput), Error>
where
    B: Runtime,
    Keys: MIter<B>,
    Values: MIter<B>,
    KeyEq: op::BinaryPredicateOp<B, Keys::Item>,
    Op: op::ReductionOp<B, Values::Item>,
    KeyOutput: MVec<B, Item = Keys::Item>,
    ValueOutput: MVec<B, Item = Values::Item>,
{
    validate_input(exec, &keys)?;
    validate_input(exec, &values)?;
    <Keys as sealed::MIterDispatch<B>>::reduce_by_key_dispatch(
        keys,
        exec.policy(),
        values,
        key_eq,
        init,
        op,
    )
}
