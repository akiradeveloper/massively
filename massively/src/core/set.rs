//! Ordered set algorithms composed from bounds, merge, and stable selection.

use cubecl::prelude::*;

use crate::{
    DeviceVec, Error, Executor, ReadExpression,
    merge::{ConcatApply, MergeControl, MergeControlInput},
    output::SliceOutput,
    search::SortedBoundsInput,
    selection::{CopySelected, SelectionControl},
};

const BLOCK_SIZE: u32 = 256;

#[cubecl::cube]
trait OccurrenceRule: 'static + Send + Sync {
    fn keep(rank: u32, other_count: u32) -> u32;
}

struct UnionExtra;
struct IntersectionKeep;
struct DifferenceKeep;

#[cubecl::cube]
impl OccurrenceRule for UnionExtra {
    fn keep(rank: u32, other_count: u32) -> u32 {
        if rank >= other_count { 1u32 } else { 0u32 }
    }
}

#[cubecl::cube]
impl OccurrenceRule for IntersectionKeep {
    fn keep(rank: u32, other_count: u32) -> u32 {
        if rank < other_count { 1u32 } else { 0u32 }
    }
}

#[cubecl::cube]
impl OccurrenceRule for DifferenceKeep {
    fn keep(rank: u32, other_count: u32) -> u32 {
        if rank >= other_count { 1u32 } else { 0u32 }
    }
}

#[cubecl::cube(launch_unchecked, explicit_define)]
fn occurrence_flags_kernel<Rule: OccurrenceRule>(
    own_lower: &[u32],
    other_lower: &[u32],
    other_upper: &[u32],
    len: &[u32],
    flags: &mut [u32],
) {
    let index = ABSOLUTE_POS as usize;
    if index < len[0] as usize {
        let rank = index as u32 - own_lower[index];
        let other_count = other_upper[index] - other_lower[index];
        flags[index] = Rule::keep(rank, other_count);
    }
}

fn occurrence_flags<R, Rule>(
    exec: &Executor<R>,
    own_lower: &DeviceVec<R, u32>,
    other_lower: &DeviceVec<R, u32>,
    other_upper: &DeviceVec<R, u32>,
) -> Result<DeviceVec<R, u32>, Error>
where
    R: Runtime,
    Rule: OccurrenceRule,
{
    let len = own_lower.capacity();
    if other_lower.capacity() != len || other_upper.capacity() != len {
        return Err(Error::LengthMismatch {
            left: len,
            right: other_lower.capacity().min(other_upper.capacity()),
        });
    }
    let extent = own_lower
        .logical_extent()
        .zipped(&other_lower.logical_extent())?
        .zipped(&other_upper.logical_extent())?;
    let mut flags = exec.alloc_row::<u32>(len);
    flags.set_logical_extent(extent.clone());
    if len == 0 {
        return Ok(flags);
    }
    let len_handle = extent.materialize(exec)?;
    unsafe {
        occurrence_flags_kernel::launch_unchecked::<Rule, R>(
            exec.client(),
            crate::launch::cube_count_1d(len.div_ceil(BLOCK_SIZE as usize))?,
            CubeDim::new_1d(BLOCK_SIZE),
            BufferArg::from_raw_parts(own_lower.handle.clone(), len),
            BufferArg::from_raw_parts(other_lower.handle.clone(), len),
            BufferArg::from_raw_parts(other_upper.handle.clone(), len),
            BufferArg::from_raw_parts(len_handle.handle.clone(), 1),
            BufferArg::from_raw_parts(flags.handle.clone(), len),
        );
    }
    Ok(flags)
}

/// Runs an ordered multiset operation on two flat-row payloads.
///
/// `mode` is 0 for union, 1 for intersection, and 2 for left difference.
pub(crate) fn set_storage<R, Item, Less, Output>(
    exec: &Executor<R>,
    left: &<Item as crate::allocation::ScratchStorage<R>>::Storage,
    right: &<Item as crate::allocation::ScratchStorage<R>>::Storage,
    _less: Less,
    output: Output,
    mode: u8,
) -> Result<DeviceVec<R, u32>, Error>
where
    R: Runtime,
    Item: crate::allocation::ScratchStorage<R>,
    Less: crate::op::BinaryPredicateOp<Item>,
    Item::StorageLeaves: crate::core::facade::KernelValue,
    <Item as crate::allocation::ScratchStorage<R>>::Storage: crate::RowStorage<R>,
    Output: crate::core::facade::KernelOutput<R>
        + crate::output::OutputExpression<Item = Item>
        + SliceOutput,
{
    let left_read = crate::read::FixedRead::new(crate::RowStorage::read(left));
    let right_read = crate::read::FixedRead::new(crate::RowStorage::read(right));

    if mode == 0 {
        let merge =
            crate::merge::merge_control_fixed(exec, left_read.clone(), right_read.clone(), _less)?;
        let right_own_lower = crate::search::lower_bounds_typed::<R, _, _, Less>(
            exec,
            right_read.clone(),
            right_read.clone(),
        )?;
        let left_lower = crate::search::lower_bounds_typed::<R, _, _, Less>(
            exec,
            left_read.clone(),
            right_read.clone(),
        )?;
        let left_upper = crate::search::upper_bounds_typed::<R, _, _, Less>(
            exec,
            left_read.clone(),
            right_read.clone(),
        )?;
        let right_extra =
            occurrence_flags::<R, UnionExtra>(exec, &right_own_lower, &left_lower, &left_upper)?;

        let total = merge.left_capacity + merge.right_capacity;
        let conceptual_flags = exec.alloc_row::<u32>(total);
        crate::selection::fill(
            exec,
            1u32,
            conceptual_flags.slice_mut(..merge.left_capacity),
        )?;
        crate::materialize(
            exec,
            right_extra.column(),
            conceptual_flags.slice_mut(merge.left_capacity..),
        )?;
        let mut merged_flags = exec.alloc_row::<u32>(total);
        merged_flags.set_logical_extent(merge.permutation.logical_extent());
        crate::indexed::gather_direct(
            exec,
            conceptual_flags.column(),
            merge.permutation.column(),
            merged_flags.slice_mut(..),
        )?;
        let selection = SelectionControl::from_flags(exec, merged_flags)?;
        let count = selection.count();
        let zero = exec.value(0u32)?;
        let mut selected_permutation = exec.full(total, &zero)?;
        crate::indexed::IndexedCopyInput::indexed_copy_selected(
            merge.permutation.column(),
            exec,
            selection.indices().column(),
            None,
            Some(count),
            true,
            selected_permutation.slice_mut(..),
        )?;
        selected_permutation
            .set_logical_extent(crate::extent::LogicalExtent::from_device(count, total));
        let selected = MergeControl {
            permutation: selected_permutation,
            left_capacity: merge.left_capacity,
            right_capacity: merge.right_capacity,
            left_extent: merge.left_extent,
            right_extent: merge.right_extent,
        };
        crate::merge::apply_storage::<R, Item, _>(exec, left, right, &selected, output)?;
        return Ok(count.clone());
    }

    let left_lower = crate::search::lower_bounds_typed::<R, _, _, Less>(
        exec,
        left_read.clone(),
        left_read.clone(),
    )?;
    let right_lower = crate::search::lower_bounds_typed::<R, _, _, Less>(
        exec,
        right_read.clone(),
        left_read.clone(),
    )?;
    let right_upper =
        crate::search::upper_bounds_typed::<R, _, _, Less>(exec, right_read, left_read.clone())?;
    let flags = if mode == 1 {
        occurrence_flags::<R, IntersectionKeep>(exec, &left_lower, &right_lower, &right_upper)?
    } else {
        occurrence_flags::<R, DifferenceKeep>(exec, &left_lower, &right_lower, &right_upper)?
    };
    let control = SelectionControl::from_flags(exec, flags)?;
    left_read.copy_selected(exec, &control, output)
}

/// Multiset union of two sorted ranges.
pub(crate) fn set_union<R, Left, Right, Less, Output>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    _less: Less,
    output: Output,
) -> Result<DeviceVec<R, u32>, Error>
where
    R: Runtime,
    Left: Clone
        + MergeControlInput<R, Right, Less>
        + SortedBoundsInput<R, Right, Less>
        + ConcatApply<R, Right, Output>,
    Right: Clone + ReadExpression + SortedBoundsInput<R, Right, Less>,
    Output: SliceOutput,
{
    let merge = left.clone().merge_control(exec, right.clone())?;
    let right_own_lower = right.clone().lower_bounds(exec, right.clone())?;
    let left_lower = left.clone().lower_bounds(exec, right.clone())?;
    let left_upper = left.clone().upper_bounds(exec, right.clone())?;
    let right_extra =
        occurrence_flags::<R, UnionExtra>(exec, &right_own_lower, &left_lower, &left_upper)?;

    let total = merge.left_capacity + merge.right_capacity;
    let conceptual_flags = exec.alloc_row::<u32>(total);
    crate::selection::fill(
        exec,
        1u32,
        conceptual_flags.slice_mut(..merge.left_capacity),
    )?;
    crate::materialize(
        exec,
        right_extra.column(),
        conceptual_flags.slice_mut(merge.left_capacity..),
    )?;
    let mut merged_flags = exec.alloc_row::<u32>(total);
    merged_flags.set_logical_extent(merge.permutation.logical_extent());
    crate::indexed::gather_direct(
        exec,
        conceptual_flags.column(),
        merge.permutation.column(),
        merged_flags.slice_mut(..),
    )?;
    let selection = SelectionControl::from_flags(exec, merged_flags)?;
    let count = selection.count();
    let zero = exec.value(0u32)?;
    let mut selected_permutation = exec.full(total, &zero)?;
    crate::indexed::IndexedCopyInput::indexed_copy_selected(
        merge.permutation.column(),
        exec,
        selection.indices().column(),
        None,
        Some(count),
        true,
        selected_permutation.slice_mut(..),
    )?;
    selected_permutation
        .set_logical_extent(crate::extent::LogicalExtent::from_device(count, total));
    let selected = MergeControl {
        permutation: selected_permutation,
        left_capacity: merge.left_capacity,
        right_capacity: merge.right_capacity,
        left_extent: merge.left_extent,
        right_extent: merge.right_extent,
    };
    left.concat_apply(exec, right, &selected, output)?;
    Ok(count.clone())
}

/// Multiset intersection of two sorted ranges.
pub(crate) fn set_intersection<R, Left, Right, Less, Output>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    _less: Less,
    output: Output,
) -> Result<DeviceVec<R, u32>, Error>
where
    R: Runtime,
    Left: Clone + ReadExpression + SortedBoundsInput<R, Left, Less> + CopySelected<R, Output>,
    Right: Clone + SortedBoundsInput<R, Left, Less>,
{
    let left_lower = left.clone().lower_bounds(exec, left.clone())?;
    let right_lower = right.clone().lower_bounds(exec, left.clone())?;
    let right_upper = right.upper_bounds(exec, left.clone())?;
    let flags =
        occurrence_flags::<R, IntersectionKeep>(exec, &left_lower, &right_lower, &right_upper)?;
    let control = SelectionControl::from_flags(exec, flags)?;
    left.copy_selected(exec, &control, output)
}

/// Multiset difference `left - right` for sorted ranges.
pub(crate) fn set_difference<R, Left, Right, Less, Output>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    _less: Less,
    output: Output,
) -> Result<DeviceVec<R, u32>, Error>
where
    R: Runtime,
    Left: Clone + ReadExpression + SortedBoundsInput<R, Left, Less> + CopySelected<R, Output>,
    Right: Clone + SortedBoundsInput<R, Left, Less>,
{
    let left_lower = left.clone().lower_bounds(exec, left.clone())?;
    let right_lower = right.clone().lower_bounds(exec, left.clone())?;
    let right_upper = right.upper_bounds(exec, left.clone())?;
    let flags =
        occurrence_flags::<R, DifferenceKeep>(exec, &left_lower, &right_lower, &right_upper)?;
    let control = SelectionControl::from_flags(exec, flags)?;
    left.copy_selected(exec, &control, output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::BinaryPredicateOp;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    struct LessU32;

    #[cubecl::cube]
    impl BinaryPredicateOp<u32> for LessU32 {
        fn apply(lhs: u32, rhs: u32) -> crate::MBool {
            crate::op::mbool(lhs < rhs)
        }
    }

    #[test]
    fn ordered_sets_preserve_standard_multiplicity() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let left = exec.to_device(&[1_u32, 2, 2, 2, 4]);
        let right = exec.to_device(&[2_u32, 2, 3, 4, 4]);

        let union = exec.to_device(&[0_u32; 10]);
        let union_len = crate::api::algorithm::set_union_into(
            &exec,
            left.column(),
            right.column(),
            LessU32,
            union.slice_mut(..),
        )
        .unwrap();
        let union_len = union_len.read(&exec).unwrap() as usize;
        assert_eq!(
            exec.to_host(&union.slice(..union_len)).unwrap(),
            vec![1, 2, 2, 2, 3, 4, 4]
        );

        let intersection = exec.to_device(&[0_u32; 5]);
        let intersection_len = crate::api::algorithm::set_intersection_into(
            &exec,
            left.column(),
            right.column(),
            LessU32,
            intersection.slice_mut(..),
        )
        .unwrap();
        let intersection_len = intersection_len.read(&exec).unwrap() as usize;
        assert_eq!(
            exec.to_host(&intersection.slice(..intersection_len))
                .unwrap(),
            vec![2, 2, 4]
        );

        let difference = exec.to_device(&[0_u32; 5]);
        let difference_len = crate::api::algorithm::set_difference_into(
            &exec,
            left.column(),
            right.column(),
            LessU32,
            difference.slice_mut(..),
        )
        .unwrap();
        let difference_len = difference_len.read(&exec).unwrap() as usize;
        assert_eq!(
            exec.to_host(&difference.slice(..difference_len)).unwrap(),
            vec![1, 2]
        );
    }
}
