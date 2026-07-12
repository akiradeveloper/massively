use cubecl::prelude::*;

use crate::{
    Error, Executor, MAlloc, MIndex, MIter, MIterMut, MStorage, WriteFrom, op::BinaryPredicateOp,
    op::ReductionOp,
};

use super::{reduce_by_key_into, sort_by_key_into};

struct IndexLess;

#[cubecl::cube]
impl BinaryPredicateOp<MIndex> for IndexLess {
    fn apply(lhs: MIndex, rhs: MIndex) -> bool {
        lhs < rhs
    }
}

struct IndexEqual;

#[cubecl::cube]
impl BinaryPredicateOp<MIndex> for IndexEqual {
    fn apply(lhs: MIndex, rhs: MIndex) -> bool {
        lhs == rhs
    }
}

/// Writes each input item to the position given by its index.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, vector::scatter};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let values = exec.to_device(&[10_u32, 20, 30]);
/// let indices = exec.to_device(&[2_u32, 0, 1]);
/// let output = exec.alloc::<u32>(3);
///
/// scatter(
///     &exec,
///     values.slice(..),
///     indices.slice(..),
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
    Indices: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
    Output::Item: WriteFrom<Values::Item>,
{
    let indices = indices.materialize_u32(exec)?;
    values.indexed_with(exec, indices.column(), None, true, output)
}

/// Scatters selected rows while preserving other output rows.
///
/// A zero stencil leaves the indexed destination unchanged.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, vector::scatter_where};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let values = exec.to_device(&[10_u32, 20, 30]);
/// let indices = exec.to_device(&[2_u32, 0, 1]);
/// let stencil = exec.to_device(&[1_u32, 0, 1]);
/// let output = exec.to_device(&[99_u32, 99, 99]);
///
/// scatter_where(
///     &exec,
///     values.slice(..),
///     indices.slice(..),
///     stencil.slice(..),
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
    Indices: MIter<R, Item = MIndex>,
    Stencil: MIter<R, Item = MIndex>,
    Output: MIterMut<R>,
    Output::Item: WriteFrom<Values::Item>,
{
    let indices = indices.materialize_u32(exec)?;
    let stencil = stencil.materialize_u32(exec)?;
    values.indexed_with(exec, indices.column(), Some(stencil.column()), true, output)
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
/// use massively::{Executor, op::ReductionOp, vector::scatter_reduce};
///
/// struct Add;
///
/// #[cubecl::cube]
/// impl ReductionOp<u32> for Add {
///     fn apply(lhs: u32, rhs: u32) -> u32 { lhs + rhs }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let values = exec.to_device(&[2_u32, 3, 5, 7]);
/// let indices = exec.to_device(&[1_u32, 0, 1, 1]);
/// let output = exec.to_device(&[10_u32, 20, 30]);
///
/// scatter_reduce(
///     &exec,
///     values.slice(..),
///     indices.slice(..),
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
    Indices: MIter<R, Item = MIndex>,
    Values::Item: MAlloc<R>,
    Output: MIterMut<R, Item = Values::Item>,
    Op: ReductionOp<Values::Item>,
{
    let len = values.len()?;
    let indices_len = indices.len()?;
    if len != indices_len {
        return Err(Error::LengthMismatch {
            left: len as usize,
            right: indices_len as usize,
        });
    }
    if len == 0 {
        return Ok(());
    }

    let len_usize = len as usize;
    let sorted_indices = exec.alloc::<MIndex>(len_usize);
    let sorted_values = <Values::Item as MAlloc<R>>::alloc(exec, len_usize);
    sort_by_key_into(
        exec,
        indices,
        values,
        IndexLess,
        sorted_indices.slice_mut(..),
        sorted_values.slice_mut(..),
    )?;

    let unique_indices = exec.alloc::<MIndex>(len_usize);
    let reduced_values = <Values::Item as MAlloc<R>>::alloc(exec, len_usize);
    let unique_len = reduce_by_key_into(
        exec,
        sorted_indices.slice(..),
        sorted_values.slice(..),
        IndexEqual,
        init,
        op,
        unique_indices.slice_mut(..),
        reduced_values.slice_mut(..),
    )?;

    output.scatter_combine_with::<_, Op>(
        exec,
        reduced_values.slice(..unique_len),
        unique_indices
            .slice(..unique_len as usize)
            .materialize_u32(exec)?,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zip2;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    struct PairAdd;

    #[cubecl::cube]
    impl ReductionOp<(u32, u32)> for PairAdd {
        fn apply(lhs: (u32, u32), rhs: (u32, u32)) -> (u32, u32) {
            (lhs.0 + rhs.0, lhs.1 + rhs.1)
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

        scatter_reduce(
            &exec,
            zip2(left.slice(..), right.slice(..)),
            indices.slice(..),
            (0, 0),
            PairAdd,
            zip2(output_left.slice_mut(..), output_right.slice_mut(..)),
        )
        .unwrap();

        assert_eq!(exec.to_host(&output_left).unwrap(), vec![102, 204]);
        assert_eq!(exec.to_host(&output_right).unwrap(), vec![1020, 2040]);
    }
}
