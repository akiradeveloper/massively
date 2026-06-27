use super::*;

/// Finds the first adjacent pair satisfying `pred`.
pub fn adjacent_find<B, Input, Pred>(
    exec: &Executor<B>,
    source: Input,
    pred: Pred,
) -> Result<Option<usize>, Error>
where
    B: Runtime,
    Input: MIter<B>,
    Pred: op::BinaryPredicateOp<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::adjacent_find_dispatch(source, exec.policy(), pred)
}

/// Returns whether two inputs are equal under `eq`.
pub fn equal<B, Left, Right, Eq>(
    exec: &Executor<B>,
    left: Left,
    right: Right,
    eq: Eq,
) -> Result<bool, Error>
where
    B: Runtime,
    Left: MIter<B>,
    Right: MIter<B, Item = Left::Item>,
    Eq: op::BinaryPredicateOp<B, Left::Item>,
{
    validate_input(exec, &left)?;
    validate_input(exec, &right)?;
    <Left as sealed::MIterDispatch<B>>::equal_dispatch(left, exec.policy(), right, eq)
}

/// Finds the equal range of `value` in a sorted input.
pub fn equal_range<B, Input, Less>(
    exec: &Executor<B>,
    source: Input,
    value: Input::Item,
    less: Less,
) -> Result<(usize, usize), Error>
where
    B: Runtime,
    Input: MIter<B>,
    Less: op::BinaryPredicateOp<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::equal_range_dispatch(source, exec.policy(), value, less)
}

/// Finds the first input element equal to any needle.
pub fn find_first_of<B, Input, Needles, Eq>(
    exec: &Executor<B>,
    source: Input,
    needles: Needles,
    eq: Eq,
) -> Result<Option<usize>, Error>
where
    B: Runtime,
    Input: MIter<B>,
    Needles: MIter<B, Item = Input::Item>,
    Eq: op::BinaryPredicateOp<B, Input::Item>,
{
    validate_input(exec, &source)?;
    validate_input(exec, &needles)?;
    <Input as sealed::MIterDispatch<B>>::find_first_of_dispatch(source, exec.policy(), needles, eq)
}

/// Returns whether input is sorted.
pub fn is_sorted<B, Input, Less>(
    exec: &Executor<B>,
    source: Input,
    less: Less,
) -> Result<bool, Error>
where
    B: Runtime,
    Input: MIter<B>,
    Less: op::BinaryPredicateOp<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::is_sorted_dispatch(source, exec.policy(), less)
}

/// Returns the first position where sorted order is broken.
pub fn is_sorted_until<B, Input, Less>(
    exec: &Executor<B>,
    source: Input,
    less: Less,
) -> Result<usize, Error>
where
    B: Runtime,
    Input: MIter<B>,
    Less: op::BinaryPredicateOp<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::is_sorted_until_dispatch(source, exec.policy(), less)
}

/// Lexicographically compares two inputs.
pub fn lexicographical_compare<B, Left, Right, Less>(
    exec: &Executor<B>,
    left: Left,
    right: Right,
    less: Less,
) -> Result<bool, Error>
where
    B: Runtime,
    Left: MIter<B>,
    Right: MIter<B, Item = Left::Item>,
    Less: op::BinaryPredicateOp<B, Left::Item>,
{
    validate_input(exec, &left)?;
    validate_input(exec, &right)?;
    <Left as sealed::MIterDispatch<B>>::lexicographical_compare_dispatch(
        left,
        exec.policy(),
        right,
        less,
    )
}

/// Finds the lower bound of `value` in a sorted input.
pub fn lower_bound<B, Input, Less>(
    exec: &Executor<B>,
    source: Input,
    value: Input::Item,
    less: Less,
) -> Result<usize, Error>
where
    B: Runtime,
    Input: MIter<B>,
    Less: op::BinaryPredicateOp<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::lower_bound_dispatch(source, exec.policy(), value, less)
}

/// Finds the maximum element index.
pub fn max_element<B, Input, Less>(
    exec: &Executor<B>,
    source: Input,
    less: Less,
) -> Result<Option<usize>, Error>
where
    B: Runtime,
    Input: MIter<B>,
    Less: op::BinaryPredicateOp<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::max_element_dispatch(source, exec.policy(), less)
}

/// Finds the minimum element index.
pub fn min_element<B, Input, Less>(
    exec: &Executor<B>,
    source: Input,
    less: Less,
) -> Result<Option<usize>, Error>
where
    B: Runtime,
    Input: MIter<B>,
    Less: op::BinaryPredicateOp<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::min_element_dispatch(source, exec.policy(), less)
}

/// Finds both minimum and maximum element indices.
pub fn minmax_element<B, Input, Less>(
    exec: &Executor<B>,
    source: Input,
    less: Less,
) -> Result<Option<(usize, usize)>, Error>
where
    B: Runtime,
    Input: MIter<B>,
    Less: op::BinaryPredicateOp<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::minmax_element_dispatch(source, exec.policy(), less)
}

/// Finds the first mismatch between two inputs.
pub fn mismatch<B, Left, Right, Eq>(
    exec: &Executor<B>,
    left: Left,
    right: Right,
    eq: Eq,
) -> Result<Option<usize>, Error>
where
    B: Runtime,
    Left: MIter<B>,
    Right: MIter<B, Item = Left::Item>,
    Eq: op::BinaryPredicateOp<B, Left::Item>,
{
    validate_input(exec, &left)?;
    validate_input(exec, &right)?;
    <Left as sealed::MIterDispatch<B>>::mismatch_dispatch(left, exec.policy(), right, eq)
}

/// Finds the upper bound of `value` in a sorted input.
pub fn upper_bound<B, Input, Less>(
    exec: &Executor<B>,
    source: Input,
    value: Input::Item,
    less: Less,
) -> Result<usize, Error>
where
    B: Runtime,
    Input: MIter<B>,
    Less: op::BinaryPredicateOp<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::upper_bound_dispatch(source, exec.policy(), value, less)
}
