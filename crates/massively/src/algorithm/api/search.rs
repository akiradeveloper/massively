use super::*;

/// Finds the first adjacent pair satisfying `pred`.
pub fn adjacent_find<R, Input, Pred>(
    exec: &Executor<R>,
    source: Input,
    pred: Pred,
) -> Result<Option<MIndex>, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Pred: op::BinaryPredicateOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    source.adjacent_find_with_policy(exec.policy(), pred)
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
    left.equal_with_policy(exec.policy(), right, eq)
}

/// Finds the first input element equal to any needle.
pub fn find_first_of<R, Input, Needles, Eq>(
    exec: &Executor<R>,
    source: Input,
    needles: Needles,
    eq: Eq,
) -> Result<Option<MIndex>, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Needles: MIter<R, Item = Input::Item>,
    Eq: op::BinaryPredicateOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    validate_input(exec, &needles)?;
    source.find_first_of_with_policy(exec.policy(), needles, eq)
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
    source.is_sorted_with_policy(exec.policy(), less)
}

/// Returns the first position where sorted order is broken.
pub fn is_sorted_until<R, Input, Less>(
    exec: &Executor<R>,
    source: Input,
    less: Less,
) -> Result<MIndex, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Less: op::BinaryPredicateOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    source.is_sorted_until_with_policy(exec.policy(), less)
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
    left.lexicographical_compare_with_policy(exec.policy(), right, less)
}

/// Finds the lower bound of each value in a sorted input.
pub fn lower_bound<R, Input, Values, Less>(
    exec: &Executor<R>,
    source: Input,
    values: Values,
    less: Less,
    out: crate::runtime::DeviceSliceMut<'_, R, MIndex>,
) -> Result<(), Error>
where
    R: Runtime,
    Input: MIter<R>,
    Values: MIter<R, Item = Input::Item>,
    Less: op::BinaryPredicateOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    validate_input(exec, &values)?;
    exec.ensure_policy_id(out.source.inner.policy_id())?;
    let bounds = source.lower_bound_many_with_policy(exec.policy(), values, less)?;
    exec.copy(bounds.slice(..), out)
}

/// Finds the maximum element index.
pub fn max_element<R, Input, Less>(
    exec: &Executor<R>,
    source: Input,
    less: Less,
) -> Result<Option<MIndex>, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Less: op::BinaryPredicateOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    source.max_element_with_policy(exec.policy(), less)
}

/// Finds the minimum element index.
pub fn min_element<R, Input, Less>(
    exec: &Executor<R>,
    source: Input,
    less: Less,
) -> Result<Option<MIndex>, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Less: op::BinaryPredicateOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    source.min_element_with_policy(exec.policy(), less)
}

/// Finds both minimum and maximum element indices.
pub fn minmax_element<R, Input, Less>(
    exec: &Executor<R>,
    source: Input,
    less: Less,
) -> Result<Option<(MIndex, MIndex)>, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Less: op::BinaryPredicateOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    source.minmax_element_with_policy(exec.policy(), less)
}

/// Finds the first mismatch between two inputs.
pub fn mismatch<R, Left, Right, Eq>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    eq: Eq,
) -> Result<Option<MIndex>, Error>
where
    R: Runtime,
    Left: MIter<R>,
    Right: MIter<R, Item = Left::Item>,
    Eq: op::BinaryPredicateOp<R, Left::Item>,
{
    validate_input(exec, &left)?;
    validate_input(exec, &right)?;
    left.mismatch_with_policy(exec.policy(), right, eq)
}

/// Finds the upper bound of each value in a sorted input.
pub fn upper_bound<R, Input, Values, Less>(
    exec: &Executor<R>,
    source: Input,
    values: Values,
    less: Less,
    out: crate::runtime::DeviceSliceMut<'_, R, MIndex>,
) -> Result<(), Error>
where
    R: Runtime,
    Input: MIter<R>,
    Values: MIter<R, Item = Input::Item>,
    Less: op::BinaryPredicateOp<R, Input::Item>,
{
    validate_input(exec, &source)?;
    validate_input(exec, &values)?;
    exec.ensure_policy_id(out.source.inner.policy_id())?;
    let bounds = source.upper_bound_many_with_policy(exec.policy(), values, less)?;
    exec.copy(bounds.slice(..), out)
}
