use super::*;

/// Finds the first adjacent pair satisfying `pred`.
pub fn adjacent_find<R, Input, Pred>(
    exec: &Executor<R>,
    source: Input,
    pred: Pred,
) -> Result<Option<usize>, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Pred: op::BinaryPredicateOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<R>>::adjacent_find_dispatch(source, exec.policy(), pred)
}

/// Returns whether two inputs are equal under `eq`.
pub fn equal<R, Left, Right, Eq>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    eq: Eq,
) -> Result<bool, Error>
where
    R: Runtime,
    Left: MIter<R>,
    Right: MIter<R, Item = Left::Item>,
    Eq: op::BinaryPredicateOp<R, Left::Item>,
{
    validate_input(exec, &left)?;
    validate_input(exec, &right)?;
    <Left as sealed::MIterDispatch<R>>::equal_dispatch(left, exec.policy(), right, eq)
}

/// Finds the equal range of `value` in a sorted input.
pub fn equal_range<R, Input, Less>(
    exec: &Executor<R>,
    source: Input,
    value: Input::Item,
    less: Less,
) -> Result<(usize, usize), Error>
where
    R: Runtime,
    Input: MIter<R>,
    Less: op::BinaryPredicateOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<R>>::equal_range_dispatch(source, exec.policy(), value, less)
}

/// Finds the first input element equal to any needle.
pub fn find_first_of<R, Input, Needles, Eq>(
    exec: &Executor<R>,
    source: Input,
    needles: Needles,
    eq: Eq,
) -> Result<Option<usize>, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Needles: MIter<R, Item = Input::Item>,
    Eq: op::BinaryPredicateOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    validate_input(exec, &needles)?;
    <Input as sealed::MIterDispatch<R>>::find_first_of_dispatch(source, exec.policy(), needles, eq)
}

/// Returns whether input is sorted.
pub fn is_sorted<R, Input, Less>(
    exec: &Executor<R>,
    source: Input,
    less: Less,
) -> Result<bool, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Less: op::BinaryPredicateOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<R>>::is_sorted_dispatch(source, exec.policy(), less)
}

/// Returns the first position where sorted order is broken.
pub fn is_sorted_until<R, Input, Less>(
    exec: &Executor<R>,
    source: Input,
    less: Less,
) -> Result<usize, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Less: op::BinaryPredicateOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<R>>::is_sorted_until_dispatch(source, exec.policy(), less)
}

/// Lexicographically compares two inputs.
pub fn lexicographical_compare<R, Left, Right, Less>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    less: Less,
) -> Result<bool, Error>
where
    R: Runtime,
    Left: MIter<R>,
    Right: MIter<R, Item = Left::Item>,
    Less: op::BinaryPredicateOp<R, Left::Item>,
{
    validate_input(exec, &left)?;
    validate_input(exec, &right)?;
    <Left as sealed::MIterDispatch<R>>::lexicographical_compare_dispatch(
        left,
        exec.policy(),
        right,
        less,
    )
}

/// Finds the lower bound of `value` in a sorted input.
pub fn lower_bound<R, Input, Less>(
    exec: &Executor<R>,
    source: Input,
    value: Input::Item,
    less: Less,
) -> Result<usize, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Less: op::BinaryPredicateOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<R>>::lower_bound_dispatch(source, exec.policy(), value, less)
}

/// Finds the maximum element index.
pub fn max_element<R, Input, Less>(
    exec: &Executor<R>,
    source: Input,
    less: Less,
) -> Result<Option<usize>, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Less: op::BinaryPredicateOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<R>>::max_element_dispatch(source, exec.policy(), less)
}

/// Finds the minimum element index.
pub fn min_element<R, Input, Less>(
    exec: &Executor<R>,
    source: Input,
    less: Less,
) -> Result<Option<usize>, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Less: op::BinaryPredicateOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<R>>::min_element_dispatch(source, exec.policy(), less)
}

/// Finds both minimum and maximum element indices.
pub fn minmax_element<R, Input, Less>(
    exec: &Executor<R>,
    source: Input,
    less: Less,
) -> Result<Option<(usize, usize)>, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Less: op::BinaryPredicateOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<R>>::minmax_element_dispatch(source, exec.policy(), less)
}

/// Finds the first mismatch between two inputs.
pub fn mismatch<R, Left, Right, Eq>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    eq: Eq,
) -> Result<Option<usize>, Error>
where
    R: Runtime,
    Left: MIter<R>,
    Right: MIter<R, Item = Left::Item>,
    Eq: op::BinaryPredicateOp<R, Left::Item>,
{
    validate_input(exec, &left)?;
    validate_input(exec, &right)?;
    <Left as sealed::MIterDispatch<R>>::mismatch_dispatch(left, exec.policy(), right, eq)
}

/// Finds the upper bound of `value` in a sorted input.
pub fn upper_bound<R, Input, Less>(
    exec: &Executor<R>,
    source: Input,
    value: Input::Item,
    less: Less,
) -> Result<usize, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Less: op::BinaryPredicateOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<R>>::upper_bound_dispatch(source, exec.policy(), value, less)
}
