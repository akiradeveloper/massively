use cubecl::prelude::*;

use crate::{
    Error, Executor, MIter, MIterMut, RowStorage,
    op::{BinaryPredicateOp, ReductionOp},
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

struct ScatterOperation<'a, R: Runtime, Values, Indices> {
    exec: &'a Executor<R>,
    values: Values,
    indices: Indices,
}

impl<R, Item, Values, Indices> crate::api::iter::OutputOperation<R, Item>
    for ScatterOperation<'_, R, Values, Indices>
where
    R: Runtime,
    Item: CubeType + Send + Sync + 'static,
    Values: MIter<R, Item = Item>,
    Indices: MIter<R, Item = usize>,
{
    type Result = Result<(), Error>;

    fn run<Output>(self, output: Output) -> Self::Result
    where
        Item: crate::api::iter::KernelRow + crate::allocation::ScratchStorage<R>,
        Output: crate::api::iter::ConcreteOutput<R, Item>,
    {
        crate::core::scatter::scatter(
            self.exec,
            crate::api::iter::lower::<R, _>(self.values),
            crate::api::iter::lower::<R, _>(self.indices),
            output,
        )
    }
}

struct ScatterWhereOperation<'a, R: Runtime, Values, Indices, Stencil> {
    exec: &'a Executor<R>,
    values: Values,
    indices: Indices,
    stencil: Stencil,
}

impl<R, Item, Values, Indices, Stencil> crate::api::iter::OutputOperation<R, Item>
    for ScatterWhereOperation<'_, R, Values, Indices, Stencil>
where
    R: Runtime,
    Item: CubeType + Send + Sync + 'static,
    Values: MIter<R, Item = Item>,
    Indices: MIter<R, Item = usize>,
    Stencil: MIter<R, Item = bool>,
{
    type Result = Result<(), Error>;

    fn run<Output>(self, output: Output) -> Self::Result
    where
        Item: crate::api::iter::KernelRow + crate::allocation::ScratchStorage<R>,
        Output: crate::api::iter::ConcreteOutput<R, Item>,
    {
        crate::core::scatter::scatter_where(
            self.exec,
            crate::api::iter::lower::<R, _>(self.values),
            crate::api::iter::lower::<R, _>(self.indices),
            crate::api::iter::lower::<R, _>(self.stencil),
            output,
        )
    }
}

struct ScatterReduceOperation<'a, R: Runtime, Values, Indices, Item, Op> {
    exec: &'a Executor<R>,
    values: Values,
    indices: Indices,
    init: Item,
    op: Op,
}

impl<R, Item, Values, Indices, Op> crate::api::iter::OutputOperation<R, Item>
    for ScatterReduceOperation<'_, R, Values, Indices, Item, Op>
where
    R: Runtime,
    Item: CubeType + Send + Sync + 'static,
    Values: MIter<R, Item = Item>,
    Indices: MIter<R, Item = usize>,
    Op: ReductionOp<Item>,
{
    type Result = Result<(), Error>;

    fn run<Output>(self, output: Output) -> Self::Result
    where
        Item: crate::api::iter::KernelRow + crate::allocation::ScratchStorage<R>,
        Output: crate::api::iter::ConcreteOutput<R, Item>,
    {
        let len = self.values.len()?;
        let indices_len = self.indices.len()?;
        if len != indices_len {
            return Err(Error::LengthMismatch {
                left: len,
                right: indices_len,
            });
        }
        if len == 0 {
            return Ok(());
        }

        let indices = crate::api::iter::lower::<R, _>(self.indices);
        let sorted_values =
            <Item as crate::allocation::ScratchStorage<R>>::alloc_scratch(self.exec, len);
        let values = crate::api::iter::lower_fixed::<R, _>(self.values);
        let permutation =
            crate::ordering::sort_control_with(self.exec, indices.clone(), IndexLess)?;
        crate::indexed::gather_u32(
            self.exec,
            values,
            permutation.column(),
            sorted_values.write(),
        )?;

        let heads = crate::ordering::unique_head_flags_ordered::<R, _, IndexEqual>(
            self.exec,
            indices.clone(),
            &permutation,
        )?;
        let head_control =
            crate::selection::SelectionControl::from_flags(self.exec, heads.clone())?;
        let unique_positions = self.exec.alloc::<u32>(head_control.count() as usize);
        crate::indexed::gather_u32(
            self.exec,
            permutation.column(),
            head_control.indices().column(),
            unique_positions.slice_mut(..),
        )?;

        let sorted_values = crate::read::FixedRead::new(sorted_values.read());
        let reduced_values = <Item as crate::allocation::ScratchStorage<R>>::alloc_scratch(
            self.exec,
            head_control.count() as usize,
        );
        crate::core::by_key::reduce_values_by_heads_lowered(
            self.exec,
            sorted_values,
            &heads,
            &head_control,
            self.init,
            self.op,
            reduced_values.write(),
        )?;

        let reduced_values = crate::read::FixedRead::new(RowStorage::read(&reduced_values));
        crate::core::scatter_reduce::apply::<R, _, _, _, Op>(
            self.exec,
            reduced_values,
            indices,
            &unique_positions,
            output,
        )
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
    Values: MIter<R, Item = Output::Item>,
    Indices: MIter<R, Item = usize>,
    Output: MIterMut<R>,
{
    output.run_output_operation(ScatterOperation {
        exec,
        values,
        indices,
    })
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
    Values: MIter<R, Item = Output::Item>,
    Indices: MIter<R, Item = u32>,
    Output: MIterMut<R>,
{
    let indices = crate::Transform::new(
        crate::api::iter::lower::<R, _>(indices),
        crate::op::U32ToUsize,
    );
    output.run_output_operation(ScatterOperation {
        exec,
        values,
        indices,
    })
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
    Values: MIter<R, Item = Output::Item>,
    Indices: MIter<R, Item = usize>,
    Stencil: MIter<R, Item = bool>,
    Output: MIterMut<R>,
{
    output.run_output_operation(ScatterWhereOperation {
        exec,
        values,
        indices,
        stencil,
    })
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
    Values: MIter<R, Item = Output::Item>,
    Indices: MIter<R, Item = usize>,
    Output: MIterMut<R>,
    Op: ReductionOp<Values::Item>,
{
    output.run_output_operation(ScatterReduceOperation {
        exec,
        values,
        indices,
        init,
        op,
    })
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
    Values: MIter<R, Item = Output::Item>,
    Indices: MIter<R, Item = u32>,
    Output: MIterMut<R>,
    Op: ReductionOp<Values::Item>,
{
    let indices = crate::Transform::new(
        crate::api::iter::lower::<R, _>(indices),
        crate::op::U32ToUsize,
    );
    output.run_output_operation(ScatterReduceOperation {
        exec,
        values,
        indices,
        init,
        op,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{zip2, zip3};
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

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

    struct TripleAdd;

    #[cubecl::cube]
    impl ReductionOp<(u32, u32, u32)> for TripleAdd {
        fn apply(lhs: (u32, u32, u32), rhs: (u32, u32, u32)) -> (u32, u32, u32) {
            (lhs.0 + rhs.0, lhs.1 + rhs.1, lhs.2 + rhs.2)
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
    fn scatter_reduce_preserves_three_flat_columns() {
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
            zip3(first.slice(..), second.slice(..), third.slice(..)),
            indices,
            (0, 0, 0),
            TripleAdd,
            zip3(
                output_first.slice_mut(..),
                output_second.slice_mut(..),
                output_third.slice_mut(..),
            ),
        )
        .unwrap();

        assert_eq!(exec.to_host(&output_first).unwrap(), vec![102, 204]);
        assert_eq!(exec.to_host(&output_second).unwrap(), vec![1020, 2040]);
        assert_eq!(exec.to_host(&output_third).unwrap(), vec![10200, 20400]);
    }
}
