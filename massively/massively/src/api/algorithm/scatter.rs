use core::marker::PhantomData;

use cubecl::prelude::*;

use crate::{
    CanonicalStorage, Error, Executor, MIter, MIterMut, StorageLayout, WritableFrom,
    op::{BinaryPredicateOp, ReductionOp, UnaryOp},
};

struct IndexLess;

#[cubecl::cube]
impl BinaryPredicateOp<usize> for IndexLess {
    fn apply(lhs: usize, rhs: usize) -> bool {
        lhs < rhs
    }
}

struct IndexEqual;

#[cubecl::cube]
impl BinaryPredicateOp<usize> for IndexEqual {
    fn apply(lhs: usize, rhs: usize) -> bool {
        lhs == rhs
    }
}

struct WriteValue<Target>(PhantomData<fn() -> Target>);

#[cubecl::cube]
impl<Source, Target> UnaryOp<Source> for WriteValue<Target>
where
    Source: CubeType + Send + Sync + 'static,
    Target: StorageLayout + WritableFrom<Source> + Send + Sync + 'static,
{
    type Output = Target;

    fn apply(value: Source) -> Target {
        Target::write_from(value)
    }
}

struct WriteReduction<Source, Op>(PhantomData<fn() -> (Source, Op)>);

#[cubecl::cube]
impl<Source, Target, Op> ReductionOp<Target> for WriteReduction<Source, Op>
where
    Source: CubeType + Send + Sync + 'static,
    Target: StorageLayout + WritableFrom<Source> + Send + Sync + 'static,
    Op: ReductionOp<Source>,
{
    fn apply(lhs: Target, rhs: Target) -> Target {
        Target::write_from(Op::apply(
            Target::read_source(lhs),
            Target::read_source(rhs),
        ))
    }
}

/// Writes each input item to the position given by its index.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, lazy, op::U32ToUsize, vector::scatter};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let values = exec.to_device(&[10_u32, 20, 30]);
/// let indices = exec.to_device(&[2_u32, 0, 1]);
/// let indices = lazy::transform(indices.slice(..), U32ToUsize);
/// let output = exec.alloc::<u32>(3);
///
/// scatter(
///     &exec,
///     values.slice(..),
///     indices,
///     output.slice_mut(..),
/// )
/// .unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![20, 30, 10]);
/// ```
pub fn scatter<R, Values, Indices, Output>(
    exec: &Executor<R>,
    values: Values,
    indices: Indices,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Values: MIter<R>,
    Indices: MIter<R, Item = usize>,
    Output: MIterMut<R>,
    Output::Item: WritableFrom<Values::Item>,
{
    crate::core::scatter::scatter(
        exec,
        crate::api::iter::lower::<R, _>(values),
        crate::api::iter::lower::<R, _>(indices),
        output.lower_output(),
    )
}

/// Scatters from stored `u32` indices after converting them at the read boundary.
pub(crate) fn scatter_raw<R, Values, Indices, Output>(
    exec: &Executor<R>,
    values: Values,
    indices: Indices,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Values: MIter<R>,
    Indices: MIter<R, Item = u32>,
    Output: MIterMut<R>,
    Output::Item: WritableFrom<Values::Item>,
{
    crate::core::scatter::scatter(
        exec,
        crate::api::iter::lower::<R, _>(values),
        crate::Transform::new(
            crate::api::iter::lower::<R, _>(indices),
            crate::op::U32ToUsize,
        ),
        output.lower_output(),
    )
}

/// Scatters selected rows while preserving other output rows.
///
/// A false stencil leaves the indexed destination unchanged.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{
///     Executor, lazy,
///     op::{U32ToBool, U32ToUsize},
///     vector::scatter_where,
/// };
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let values = exec.to_device(&[10_u32, 20, 30]);
/// let indices = exec.to_device(&[2_u32, 0, 1]);
/// let stencil = exec.to_device(&[1_u32, 0, 1]);
/// let indices = lazy::transform(indices.slice(..), U32ToUsize);
/// let stencil = lazy::transform(stencil.slice(..), U32ToBool);
/// let output = exec.to_device(&[99_u32, 99, 99]);
///
/// scatter_where(
///     &exec,
///     values.slice(..),
///     indices,
///     stencil,
///     output.slice_mut(..),
/// )
/// .unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![99, 30, 10]);
/// ```
pub fn scatter_where<R, Values, Indices, Stencil, Output>(
    exec: &Executor<R>,
    values: Values,
    indices: Indices,
    stencil: Stencil,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Values: MIter<R>,
    Indices: MIter<R, Item = usize>,
    Stencil: MIter<R, Item = bool>,
    Output: MIterMut<R>,
    Output::Item: WritableFrom<Values::Item>,
{
    crate::core::scatter::scatter_where(
        exec,
        crate::api::iter::lower::<R, _>(values),
        crate::api::iter::lower::<R, _>(indices),
        crate::api::iter::lower::<R, _>(stencil),
        output.lower_output(),
    )
}

/// Reduces colliding scatter proposals and combines each result with its destination.
///
/// `init` must be the identity of `op`. Proposals targeting the same index are reduced in an
/// unspecified order; consequently `op` must be associative and commutative. Destinations not
/// present in `indices` are left unchanged.
///
/// This implementation preserves semantic rows, including multi-column tuple rows. It sorts and
/// reduces proposals before the final write, so the last phase has exactly one writer per
/// destination and does not require an atomic implementation for every item type.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{
///     Executor, lazy,
///     op::{self, U32ToUsize},
///     vector::scatter_reduce,
/// };
///
/// struct Add;
///
/// #[cubecl::cube]
/// impl op::ReductionOp<u32> for Add {
///     fn apply(lhs: u32, rhs: u32) -> u32 { lhs + rhs }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let values = exec.to_device(&[2_u32, 3, 5, 7]);
/// let indices = exec.to_device(&[1_u32, 0, 1, 1]);
/// let indices = lazy::transform(indices.slice(..), U32ToUsize);
/// let output = exec.to_device(&[10_u32, 20, 30]);
///
/// scatter_reduce(
///     &exec,
///     values.slice(..),
///     indices,
///     0,
///     Add,
///     output.slice_mut(..),
/// )
/// .unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![13, 34, 30]);
/// ```
pub fn scatter_reduce<R, Values, Indices, Output, Op>(
    exec: &Executor<R>,
    values: Values,
    indices: Indices,
    init: Values::Item,
    op: Op,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Values: MIter<R>,
    Indices: MIter<R, Item = usize>,
    Output: MIterMut<R>,
    Output::Item: WritableFrom<Values::Item>,
    Op: ReductionOp<Values::Item>,
{
    let indices_len = indices.len()?;
    scatter_reduce_lowered(
        exec,
        values,
        indices_len,
        crate::api::iter::lower::<R, _>(indices),
        init,
        op,
        output,
    )
}

/// Scatter-reduces stored `u32` indices after converting them at the read boundary.
pub(crate) fn scatter_reduce_raw<R, Values, Indices, Output, Op>(
    exec: &Executor<R>,
    values: Values,
    indices: Indices,
    init: Values::Item,
    op: Op,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Values: MIter<R>,
    Indices: MIter<R, Item = u32>,
    Output: MIterMut<R>,
    Output::Item: WritableFrom<Values::Item>,
    Op: ReductionOp<Values::Item>,
{
    let indices_len = indices.len()?;
    let indices = crate::Transform::new(
        crate::api::iter::lower::<R, _>(indices),
        crate::op::U32ToUsize,
    );
    scatter_reduce_lowered(exec, values, indices_len, indices, init, op, output)
}

fn scatter_reduce_lowered<R, Values, Indices, Output, Op>(
    exec: &Executor<R>,
    values: Values,
    indices_len: usize,
    indices: Indices,
    init: Values::Item,
    op: Op,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Values: MIter<R>,
    Indices: crate::core::facade::KernelInput<R, Item = usize>,
    Output: MIterMut<R>,
    Output::Item: WritableFrom<Values::Item>,
    Op: ReductionOp<Values::Item>,
{
    let len = values.len()?;
    if len != indices_len {
        return Err(Error::LengthMismatch {
            left: len,
            right: indices_len,
        });
    }
    if len == 0 {
        return Ok(());
    }

    let len_usize = len;
    let sorted_values =
        <Output::Item as crate::allocation::ScratchStorage<R>>::alloc_scratch(exec, len_usize);
    let values = crate::read::Transform::new(
        crate::api::iter::lower_fixed::<R, _>(values),
        WriteValue::<Output::Item>(PhantomData),
    );
    let permutation = crate::ordering::sort_control_with(exec, indices.clone(), IndexLess)?;
    let sorted_output = crate::output::ReassociatedOutput::<
        _,
        Output::Item,
        <<Output::Item as crate::StorageLayout>::StorageLeaves as crate::output::OutputSlotLayout>::Slots,
    >::new(sorted_values.write());
    crate::indexed::gather_u32(exec, values, permutation.column(), sorted_output)?;

    let heads = crate::ordering::unique_head_flags_ordered::<R, _, IndexEqual>(
        exec,
        indices.clone(),
        &permutation,
    )?;
    let head_control = crate::selection::SelectionControl::from_flags(exec, heads.clone())?;
    let unique_positions = exec.alloc::<u32>(head_control.count() as usize);
    crate::indexed::gather_u32(
        exec,
        permutation.column(),
        head_control.indices().column(),
        unique_positions.slice_mut(..),
    )?;

    let sorted_values = crate::read::FixedReassociate::<_, Output::Item>::new(sorted_values.read());
    let reduced_values = <Output::Item as crate::allocation::ScratchStorage<R>>::alloc_scratch(
        exec,
        head_control.count() as usize,
    );
    let reduced_output = crate::output::ReassociatedOutput::<
        _,
        Output::Item,
        <<Output::Item as crate::StorageLayout>::StorageLeaves as crate::output::OutputSlotLayout>::Slots,
    >::new(reduced_values.write());
    crate::core::by_key::reduce_values_by_heads_lowered(
        exec,
        sorted_values,
        &heads,
        &head_control,
        Output::Item::write_from(init),
        WriteReduction::<Values::Item, Op>(PhantomData),
        reduced_output,
    )?;

    let _ = op;
    let reduced_values = crate::read::FixedReassociate::<_, Output::Item>::new(
        CanonicalStorage::read(&reduced_values),
    );
    crate::core::scatter_reduce::apply::<R, _, _, _, WriteReduction<Values::Item, Op>>(
        exec,
        reduced_values,
        indices,
        &unique_positions,
        output.lower_output(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zip2;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    use static_assertions::assert_not_impl_any;

    #[derive(CubeType, Clone, Copy)]
    struct StoredOnlyValue {
        value: u32,
    }

    struct StoredOnlyLayout;

    #[cubecl::cube]
    impl crate::storage::Decompose<StoredOnlyValue> for StoredOnlyLayout {
        type Leaves = crate::storage::Last<u32>;

        fn decompose(item: StoredOnlyValue) -> Self::Leaves {
            crate::storage::Last::new(item.value)
        }
    }

    #[cubecl::cube]
    impl crate::storage::Recompose<StoredOnlyValue> for StoredOnlyLayout {
        type Leaves = crate::storage::Last<u32>;

        fn recompose(leaves: Self::Leaves) -> StoredOnlyValue {
            StoredOnlyValue {
                value: leaves.value,
            }
        }
    }

    impl StorageLayout for StoredOnlyValue {
        type StorageArity = crate::S1;
        type StorageLeaves = crate::storage::Last<u32>;
        type DeviceLayout = StoredOnlyLayout;

        fn into_storage_leaves(self) -> Self::StorageLeaves {
            crate::storage::Last { value: self.value }
        }

        fn from_storage_leaves(leaves: Self::StorageLeaves) -> Self {
            Self {
                value: leaves.value,
            }
        }
    }

    struct Add;

    #[cubecl::cube]
    impl ReductionOp<u32> for Add {
        fn apply(lhs: u32, rhs: u32) -> u32 {
            lhs + rhs
        }
    }

    struct PairAdd;

    #[cubecl::cube]
    impl ReductionOp<(u32, u32)> for PairAdd {
        fn apply(lhs: (u32, u32), rhs: (u32, u32)) -> (u32, u32) {
            (lhs.0 + rhs.0, lhs.1 + rhs.1)
        }
    }

    struct RightAssociatedAdd;

    #[cubecl::cube]
    impl ReductionOp<(u32, (u32, u32))> for RightAssociatedAdd {
        fn apply(lhs: (u32, (u32, u32)), rhs: (u32, (u32, u32))) -> (u32, (u32, u32)) {
            (lhs.0 + rhs.0, (lhs.1.0 + rhs.1.0, lhs.1.1 + rhs.1.1))
        }
    }

    #[test]
    fn scatter_reduce_preserves_multi_column_rows() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let left = exec.to_device(&[1_u32, 2, 3]);
        let right = exec.to_device(&[10_u32, 20, 30]);
        let indices = exec.to_device(&[1_u32, 0, 1]);
        let output_left = exec.to_device(&[100_u32, 200]);
        let output_right = exec.to_device(&[1000_u32, 2000]);
        let indices = crate::lazy::transform(indices.slice(..), crate::op::U32ToUsize);

        scatter_reduce(
            &exec,
            zip2(left.slice(..), right.slice(..)),
            indices,
            (0, 0),
            PairAdd,
            zip2(output_left.slice_mut(..), output_right.slice_mut(..)),
        )
        .unwrap();

        assert_eq!(exec.to_host(&output_left).unwrap(), vec![102, 204]);
        assert_eq!(exec.to_host(&output_right).unwrap(), vec![1020, 2040]);
    }

    #[test]
    fn scatter_reduce_writes_to_differently_associated_output() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let first = exec.to_device(&[1_u32, 2, 3]);
        let second = exec.to_device(&[10_u32, 20, 30]);
        let third = exec.to_device(&[100_u32, 200, 300]);
        let indices = exec.to_device(&[1_u32, 0, 1]);
        let output_first = exec.to_device(&[100_u32, 200]);
        let output_second = exec.to_device(&[1000_u32, 2000]);
        let output_third = exec.to_device(&[10000_u32, 20000]);
        let indices = crate::lazy::transform(indices.slice(..), crate::op::U32ToUsize);

        scatter_reduce(
            &exec,
            zip2(first.slice(..), zip2(second.slice(..), third.slice(..))),
            indices,
            (0, (0, 0)),
            RightAssociatedAdd,
            zip2(
                zip2(output_first.slice_mut(..), output_second.slice_mut(..)),
                output_third.slice_mut(..),
            ),
        )
        .unwrap();

        assert_eq!(exec.to_host(&output_first).unwrap(), vec![102, 204]);
        assert_eq!(exec.to_host(&output_second).unwrap(), vec![1020, 2040]);
        assert_eq!(exec.to_host(&output_third).unwrap(), vec![10200, 20400]);
    }

    #[test]
    fn scatter_reduce_accepts_preallocated_only_output() {
        assert_not_impl_any!(StoredOnlyValue: crate::ToCanonical<WgpuRuntime>);

        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let values = exec.to_device(&[2_u32, 3, 5, 7]);
        let indices = exec.to_device(&[1_u32, 0, 1, 1]);
        let backing = exec.to_device(&[10_u32, 20, 30]);
        let output =
            crate::output::ReassociatedOutput::<_, StoredOnlyValue, crate::read::Env1<u32>>::new(
                backing.slice_mut(..),
            );
        let indices = crate::lazy::transform(indices.slice(..), crate::op::U32ToUsize);

        scatter_reduce(&exec, values.slice(..), indices, 0, Add, output).unwrap();

        assert_eq!(exec.to_host(&backing).unwrap(), vec![13, 34, 30]);
    }
}
