//! Public algorithm API implementation for `massively`.
//!
//! Public algorithm API implementation for `massively`.

use std::any::Any;

use cubecl::prelude::Runtime;

use crate::algorithm::op;
use crate::algorithm::{MItem, MIter, MIterMut, MVec, SoA1, SoA2, SoA3};
use crate::error::ensure_same_len;
use crate::runtime::{DeviceSlice, DeviceSliceMut, DeviceVec, Executor, Scalar};

pub use crate::Error;

pub(crate) mod sealed;

fn array_from_inner<B, Item, Output>(inner: <Item as MItem<B>>::Inner) -> Output
where
    B: Runtime,
    Item: MItem<B>,
    Output: MVec<B, Item = Item>,
{
    <Output as MVec<B>>::from_inner(inner)
}

fn gather_index_inner<B, Indices>(
    _policy: &crate::detail::CubePolicy<B>,
    indices: &Indices,
) -> Result<crate::detail::device::DeviceColumnView<B, u32>, Error>
where
    B: Runtime,
    Indices: MIter<B, Item = (u32,)>,
{
    <Indices as sealed::MIterDispatch<B>>::column_view_inner::<u32>(indices)?.ok_or_else(|| {
        Error::Launch {
            message: "gather indices must be backed by one u32 DeviceVec or DeviceSlice"
                .to_string(),
        }
    })
}

fn column_view_at<B, Iter, T>(
    iter: &Iter,
    index: usize,
    algorithm: &str,
) -> Result<crate::detail::device::DeviceColumnView<B, T>, Error>
where
    B: Runtime,
    Iter: MIter<B>,
    T: Scalar + 'static,
{
    <Iter as sealed::MIterDispatch<B>>::column_view_by_index_inner::<T>(iter, index)?.ok_or_else(
        || Error::Launch {
            message: format!("{algorithm} is not supported for this iterator shape"),
        },
    )
}

fn validate_input<B, Input>(exec: &Executor<B>, input: &Input) -> Result<(), Error>
where
    B: Runtime,
    Input: MIter<B>,
{
    <Input as sealed::MIterDispatch<B>>::validate_executor(input, exec)
}

fn validate_output<B, Output>(exec: &Executor<B>, output: &Output) -> Result<(), Error>
where
    B: Runtime,
    Output: MIterMut<B>,
{
    <Output as sealed::MIterMutDispatch<B>>::validate_executor(output, exec)
}

fn validate_slice<B, T>(exec: &Executor<B>, slice: &DeviceSlice<'_, B, T>) -> Result<(), Error>
where
    B: Runtime,
{
    exec.ensure_policy_id(slice.source.inner.policy_id())
}

mod adapter;
mod impls;
/// Computes adjacent differences.
pub fn adjacent_difference<B, Input, Output, Op>(
    exec: &Executor<B>,
    source: Input,
    op: Op,
) -> Result<Output, Error>
where
    B: Runtime,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
    Op: op::ReductionOp<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::adjacent_difference_dispatch(source, exec.policy(), op)
}

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

/// Returns whether all elements satisfy `pred`.
pub fn all_of<B, Input, Pred>(exec: &Executor<B>, source: Input, pred: Pred) -> Result<bool, Error>
where
    B: Runtime,
    Input: MIter<B>,
    Pred: op::PredicateOp<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::all_of_dispatch(source, exec.policy(), pred)
}

/// Returns whether any element satisfies `pred`.
pub fn any_of<B, Input, Pred>(exec: &Executor<B>, source: Input, pred: Pred) -> Result<bool, Error>
where
    B: Runtime,
    Input: MIter<B>,
    Pred: op::PredicateOp<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::any_of_dispatch(source, exec.policy(), pred)
}

/// Copies elements whose `u32` stencil flag is non-zero.
pub fn copy_where<B, Input, Output>(
    exec: &Executor<B>,
    source: Input,
    stencil: DeviceSlice<'_, B, u32>,
) -> Result<Output, Error>
where
    B: Runtime,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
{
    validate_input(exec, &source)?;
    validate_slice(exec, &stencil)?;
    <Input as sealed::MIterDispatch<B>>::copy_where_dispatch(source, exec.policy(), SoA1(stencil))
}

/// Counts elements satisfying `pred`.
pub fn count_if<B, Input, Pred>(
    exec: &Executor<B>,
    source: Input,
    pred: Pred,
) -> Result<usize, Error>
where
    B: Runtime,
    Input: MIter<B>,
    Pred: op::PredicateOp<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::count_if_dispatch(source, exec.policy(), pred)
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

/// Computes an exclusive scan.
pub fn exclusive_scan<B, Input, Output, Op>(
    exec: &Executor<B>,
    source: Input,
    init: Input::Item,
    op: Op,
) -> Result<Output, Error>
where
    B: Runtime,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
    Op: op::ReductionOp<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::exclusive_scan_dispatch(source, exec.policy(), init, op)
}

/// Exclusive scan by key.
pub fn exclusive_scan_by_key<B, Keys, Values, KeyEq, Op, Output>(
    exec: &Executor<B>,
    keys: Keys,
    values: Values,
    key_eq: KeyEq,
    init: Values::Item,
    op: Op,
) -> Result<Output, Error>
where
    B: Runtime,
    Keys: MIter<B>,
    Values: MIter<B>,
    KeyEq: op::BinaryPredicateOp<B, Keys::Item>,
    Op: op::ReductionOp<B, Values::Item>,
    Output: MVec<B, Item = Values::Item>,
{
    validate_input(exec, &keys)?;
    validate_input(exec, &values)?;
    <Keys as sealed::MIterDispatch<B>>::exclusive_scan_by_key_dispatch(
        keys,
        exec.policy(),
        values,
        key_eq,
        init,
        op,
    )
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

/// Finds the first element satisfying `pred`.
pub fn find_if<B, Input, Pred>(
    exec: &Executor<B>,
    source: Input,
    pred: Pred,
) -> Result<Option<usize>, Error>
where
    B: Runtime,
    Input: MIter<B>,
    Pred: op::PredicateOp<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::find_if_dispatch(source, exec.policy(), pred)
}

/// Gathers a massively iterator at index positions into `out`.
pub fn gather<B, Input, Output>(
    exec: &Executor<B>,
    source: Input,
    indices: DeviceSlice<'_, B, u32>,
    out: Output,
) -> Result<(), Error>
where
    B: Runtime,
    Input: MIter<B>,
    Output: MIterMut<B, Item = Input::Item>,
{
    validate_input(exec, &source)?;
    validate_slice(exec, &indices)?;
    validate_output(exec, &out)?;
    <Input as sealed::MIterDispatch<B>>::gather_dispatch(source, exec.policy(), SoA1(indices), out)
}

/// Gathers elements whose `u32` stencil flag is non-zero.
pub fn gather_where<B, Input, Output>(
    exec: &Executor<B>,
    source: Input,
    indices: DeviceSlice<'_, B, u32>,
    stencil: DeviceSlice<'_, B, u32>,
    out: Output,
) -> Result<(), Error>
where
    B: Runtime,
    Input: MIter<B>,
    Output: MIterMut<B, Item = Input::Item>,
{
    validate_input(exec, &source)?;
    validate_slice(exec, &indices)?;
    validate_slice(exec, &stencil)?;
    validate_output(exec, &out)?;
    <Input as sealed::MIterDispatch<B>>::gather_where_dispatch(
        source,
        exec.policy(),
        SoA1(indices),
        SoA1(stencil),
        out,
    )
}

/// Computes an inclusive scan.
pub fn inclusive_scan<B, Input, Output, Op>(
    exec: &Executor<B>,
    source: Input,
    op: Op,
) -> Result<Output, Error>
where
    B: Runtime,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
    Op: op::ReductionOp<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::inclusive_scan_dispatch(source, exec.policy(), op)
}

/// Inclusive scan by key.
pub fn inclusive_scan_by_key<B, Keys, Values, KeyEq, Op, Output>(
    exec: &Executor<B>,
    keys: Keys,
    values: Values,
    key_eq: KeyEq,
    op: Op,
) -> Result<Output, Error>
where
    B: Runtime,
    Keys: MIter<B>,
    Values: MIter<B>,
    KeyEq: op::BinaryPredicateOp<B, Keys::Item>,
    Op: op::ReductionOp<B, Values::Item>,
    Output: MVec<B, Item = Values::Item>,
{
    validate_input(exec, &keys)?;
    validate_input(exec, &values)?;
    <Keys as sealed::MIterDispatch<B>>::inclusive_scan_by_key_dispatch(
        keys,
        exec.policy(),
        values,
        key_eq,
        op,
    )
}

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

/// Returns whether input is partitioned by `pred`.
pub fn is_partitioned<B, Input, Pred>(
    exec: &Executor<B>,
    source: Input,
    pred: Pred,
) -> Result<bool, Error>
where
    B: Runtime,
    Input: MIter<B>,
    Pred: op::PredicateOp<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::is_partitioned_dispatch(source, exec.policy(), pred)
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

/// Returns whether no elements satisfy `pred`.
pub fn none_of<B, Input, Pred>(exec: &Executor<B>, source: Input, pred: Pred) -> Result<bool, Error>
where
    B: Runtime,
    Input: MIter<B>,
    Pred: op::PredicateOp<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::none_of_dispatch(source, exec.policy(), pred)
}

/// Partitions elements by `pred`.
pub fn partition<B, Input, Output, Pred>(
    exec: &Executor<B>,
    source: Input,
    pred: Pred,
) -> Result<(Output, Output), Error>
where
    B: Runtime,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
    Pred: op::PredicateOp<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::partition_dispatch(source, exec.policy(), pred)
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

/// Removes elements whose `u32` stencil flag is non-zero.
pub fn remove_where<B, Input, Output>(
    exec: &Executor<B>,
    source: Input,
    stencil: DeviceSlice<'_, B, u32>,
) -> Result<Output, Error>
where
    B: Runtime,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
{
    validate_input(exec, &source)?;
    validate_slice(exec, &stencil)?;
    <Input as sealed::MIterDispatch<B>>::remove_where_dispatch(source, exec.policy(), SoA1(stencil))
}

/// Replaces elements whose `u32` stencil flag is non-zero.
pub fn replace_where<B, Output>(
    exec: &Executor<B>,
    replacement: Output::Item,
    stencil: DeviceSlice<'_, B, u32>,
    out: Output,
) -> Result<(), Error>
where
    B: Runtime,
    Output: MIterMut<B>,
{
    validate_slice(exec, &stencil)?;
    validate_output(exec, &out)?;
    out.replace_where_inner(exec.policy(), replacement, SoA1(stencil))
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

/// Scatters values into `out`.
pub fn scatter<B, Input, Output>(
    exec: &Executor<B>,
    source: Input,
    indices: DeviceSlice<'_, B, u32>,
    out: Output,
) -> Result<(), Error>
where
    B: Runtime,
    Input: MIter<B>,
    Output: MIterMut<B, Item = Input::Item>,
{
    validate_input(exec, &source)?;
    validate_slice(exec, &indices)?;
    validate_output(exec, &out)?;
    <Input as sealed::MIterDispatch<B>>::scatter_dispatch(source, exec.policy(), SoA1(indices), out)
}

/// Scatters values whose `u32` stencil flag is non-zero into a newly allocated output.
pub fn scatter_where<B, Input, Output>(
    exec: &Executor<B>,
    source: Input,
    indices: DeviceSlice<'_, B, u32>,
    stencil: DeviceSlice<'_, B, u32>,
    out: Output,
) -> Result<(), Error>
where
    B: Runtime,
    Input: MIter<B>,
    Output: MIterMut<B, Item = Input::Item>,
{
    validate_input(exec, &source)?;
    validate_slice(exec, &indices)?;
    validate_slice(exec, &stencil)?;
    validate_output(exec, &out)?;
    <Input as sealed::MIterDispatch<B>>::scatter_where_dispatch(
        source,
        exec.policy(),
        SoA1(indices),
        SoA1(stencil),
        out,
    )
}

/// Computes the sorted set difference of two sorted inputs.
pub fn set_difference<B, Left, Right, Output, Less>(
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
    <Left as sealed::MIterDispatch<B>>::set_difference_dispatch(left, exec.policy(), right, less)
}

/// Computes the sorted set intersection of two sorted inputs.
pub fn set_intersection<B, Left, Right, Output, Less>(
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
    <Left as sealed::MIterDispatch<B>>::set_intersection_dispatch(left, exec.policy(), right, less)
}

/// Computes the sorted set union of two sorted inputs.
pub fn set_union<B, Left, Right, Output, Less>(
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
    <Left as sealed::MIterDispatch<B>>::set_union_dispatch(left, exec.policy(), right, less)
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

/// Applies a unary transform to a massively iterator.
pub fn transform<B, Input, Output, Op>(
    exec: &Executor<B>,
    source: Input,
    op: Op,
    out: Output,
) -> Result<(), Error>
where
    B: Runtime,
    Input: MIter<B>,
    Output: MIterMut<B>,
    Op: op::UnaryOp<B, Input::Item, Output = Output::Item>,
{
    validate_input(exec, &source)?;
    validate_output(exec, &out)?;
    <Input as sealed::MIterDispatch<B>>::transform_dispatch(source, exec.policy(), op, out)
}

/// Applies a unary transform where the `u32` stencil flag is non-zero.
pub fn transform_where<B, Input, Output, Op>(
    exec: &Executor<B>,
    source: Input,
    op: Op,
    stencil: DeviceSlice<'_, B, u32>,
    out: Output,
) -> Result<(), Error>
where
    B: Runtime,
    Input: MIter<B>,
    Output: MIterMut<B>,
    Op: op::UnaryOp<B, Input::Item, Output = Output::Item>,
{
    validate_input(exec, &source)?;
    validate_slice(exec, &stencil)?;
    validate_output(exec, &out)?;
    <Input as sealed::MIterDispatch<B>>::transform_where_dispatch(
        source,
        exec.policy(),
        op,
        SoA1(stencil),
        out,
    )
}

/// Removes consecutive duplicates under `pred`.
pub fn unique<B, Input, Output, Pred>(
    exec: &Executor<B>,
    source: Input,
    pred: Pred,
) -> Result<Output, Error>
where
    B: Runtime,
    Input: MIter<B>,
    Output: MVec<B, Item = Input::Item>,
    Pred: op::BinaryPredicateOp<B, Input::Item>,
{
    validate_input(exec, &source)?;
    <Input as sealed::MIterDispatch<B>>::unique_dispatch(source, exec.policy(), pred)
}

/// Removes consecutive duplicate keys and keeps their values.
pub fn unique_by_key<B, Keys, Values, Eq, KeyOutput, ValueOutput>(
    exec: &Executor<B>,
    keys: Keys,
    values: Values,
    eq: Eq,
) -> Result<(KeyOutput, ValueOutput), Error>
where
    B: Runtime,
    Keys: MIter<B>,
    Values: MIter<B>,
    Eq: op::BinaryPredicateOp<B, Keys::Item>,
    KeyOutput: MVec<B, Item = Keys::Item>,
    ValueOutput: MVec<B, Item = Values::Item>,
{
    validate_input(exec, &keys)?;
    validate_input(exec, &values)?;
    <Keys as sealed::MIterDispatch<B>>::unique_by_key_dispatch(keys, exec.policy(), values, eq)
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
