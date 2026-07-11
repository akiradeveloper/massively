use cubecl::prelude::*;

use crate::{
    BinaryPredicateOp, DeviceVec, Error, Executor, MIndex, MItem, MIter, MIterMut, ReductionOp,
    WriteFrom,
};

const BLOCK_SIZE: u32 = 256;

#[cubecl::cube(launch_unchecked)]
fn mark_segment_heads_kernel(offsets: &[u32], segment_count: &[u32], heads: &mut [u32]) {
    let segment = ABSOLUTE_POS as usize;
    if segment < segment_count[0] as usize {
        let start = offsets[segment] as usize;
        let end = offsets[segment + 1usize] as usize;
        if start < end {
            heads[start] = segment as u32 + 1u32;
        }
    }
}

#[cubecl::cube(launch_unchecked)]
fn reverse_indices_kernel(offsets: &[u32], ids: &[u32], len: &[u32], indices: &mut [u32]) {
    let index = ABSOLUTE_POS as usize;
    if index < len[0] as usize {
        let segment = ids[index] as usize - 1usize;
        indices[index] = offsets[segment] + offsets[segment + 1usize] - 1u32 - index as u32;
    }
}

#[cubecl::cube(launch_unchecked)]
fn merge_head_flags_kernel(heads: &[u32], len: &[u32], flags: &mut [u32]) {
    let index = ABSOLUTE_POS as usize;
    if index < len[0] as usize && heads[index] != 0u32 {
        flags[index] = 1u32;
    }
}

#[cubecl::cube(launch_unchecked)]
fn clear_head_flags_kernel(heads: &[u32], len: &[u32], flags: &mut [u32]) {
    let index = ABSOLUTE_POS as usize;
    if index < len[0] as usize && heads[index] != 0u32 {
        flags[index] = 0u32;
    }
}

#[cubecl::cube(launch_unchecked)]
fn sorted_until_candidates_kernel(
    offsets: &[u32],
    heads: &[u32],
    ids: &[u32],
    breaks: &[u32],
    len: &[u32],
    candidates: &mut [u32],
) {
    let index = ABSOLUTE_POS as usize;
    if index < len[0] as usize {
        let segment = ids[index] as usize - 1usize;
        candidates[index] = if heads[index] == 0u32 && breaks[index] != 0u32 {
            index as u32 - offsets[segment]
        } else {
            4_294_967_295u32
        };
    }
}

#[cubecl::cube(launch_unchecked)]
fn finish_sorted_until_kernel(
    offsets: &[u32],
    reduced: &[u32],
    segment_count: &[u32],
    output: &mut [u32],
) {
    let segment = ABSOLUTE_POS as usize;
    if segment < segment_count[0] as usize {
        let candidate = reduced[segment];
        output[segment] = if candidate == 4_294_967_295u32 {
            offsets[segment + 1usize] - offsets[segment]
        } else {
            candidate
        };
    }
}

#[cubecl::cube(launch_unchecked)]
fn selected_offsets_kernel(
    input_offsets: &[u32],
    positions: &[u32],
    offset_count: &[u32],
    output_offsets: &mut [u32],
) {
    let index = ABSOLUTE_POS as usize;
    if index < offset_count[0] as usize {
        let end = input_offsets[index] as usize;
        output_offsets[index] = if end == 0usize {
            0u32
        } else {
            positions[end - 1usize]
        };
    }
}

#[cubecl::cube(launch_unchecked)]
fn gathered_segment_lengths_kernel(
    input_offsets: &[u32],
    segment_indices: &[u32],
    lengths: &mut [u32],
) {
    let selection = ABSOLUTE_POS as usize;
    if selection < lengths.len() {
        let segment = segment_indices[selection] as usize;
        lengths[selection] = input_offsets[segment + 1usize] - input_offsets[segment];
    }
}

#[cubecl::cube(launch_unchecked)]
fn gathered_offsets_kernel(positions: &[u32], output_offsets: &mut [u32]) {
    let selection = ABSOLUTE_POS as usize;
    if selection < positions.len() {
        if selection == 0usize {
            output_offsets[0] = 0u32;
        }
        output_offsets[selection + 1usize] = positions[selection];
    }
}

#[cubecl::cube(launch_unchecked)]
fn gathered_source_indices_kernel(
    input_offsets: &[u32],
    segment_indices: &[u32],
    output_offsets: &[u32],
    output_segment_ids: &[u32],
    source_indices: &mut [u32],
) {
    let output_index = ABSOLUTE_POS as usize;
    if output_index < source_indices.len() {
        let output_segment = output_segment_ids[output_index] as usize - 1usize;
        let input_segment = segment_indices[output_segment] as usize;
        source_indices[output_index] =
            input_offsets[input_segment] + output_index as u32 - output_offsets[output_segment];
    }
}

pub(crate) struct MaxU32;

#[cubecl::cube]
impl ReductionOp<u32> for MaxU32 {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        u32::max(lhs, rhs)
    }
}

pub(crate) struct MinU32;

#[cubecl::cube]
impl ReductionOp<u32> for MinU32 {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        u32::min(lhs, rhs)
    }
}

pub(crate) struct SumU32;

#[cubecl::cube]
impl ReductionOp<u32> for SumU32 {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        lhs + rhs
    }
}

pub(crate) struct EqualU32;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for EqualU32 {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs == rhs
    }
}

pub(crate) struct LessU32;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for LessU32 {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs < rhs
    }
}

pub(crate) struct SegmentControl<R: Runtime> {
    pub(crate) offsets: DeviceVec<R, u32>,
    pub(crate) heads: DeviceVec<R, u32>,
    pub(crate) ids: DeviceVec<R, u32>,
    pub(crate) segment_count: usize,
    pub(crate) value_len: usize,
}

impl<R: Runtime> SegmentControl<R> {
    pub(crate) fn new<Offsets>(
        exec: &Executor<R>,
        offsets: Offsets,
        value_len: usize,
    ) -> Result<Self, Error>
    where
        Offsets: MIter<R, Item = MIndex>,
    {
        let offsets = offsets.materialize_u32(exec)?;
        Self::from_materialized(exec, offsets, value_len)
    }

    pub(crate) fn from_materialized(
        exec: &Executor<R>,
        offsets: DeviceVec<R, u32>,
        value_len: usize,
    ) -> Result<Self, Error> {
        let Some(segment_count) = offsets.len().checked_sub(1) else {
            return Err(Error::LengthMismatch { left: 1, right: 0 });
        };
        let heads = exec.full(value_len, 0u32)?;

        if segment_count != 0 && value_len != 0 {
            let segment_count_u32 = u32::try_from(segment_count)
                .map_err(|_| Error::LengthTooLarge { len: segment_count })?;
            let segment_count_handle = exec
                .client()
                .create_from_slice(u32::as_bytes(&[segment_count_u32]));
            unsafe {
                mark_segment_heads_kernel::launch_unchecked::<R>(
                    exec.client(),
                    crate::launch::cube_count_1d(segment_count.div_ceil(BLOCK_SIZE as usize))?,
                    CubeDim::new_1d(BLOCK_SIZE),
                    BufferArg::from_raw_parts(offsets.handle.clone(), offsets.len()),
                    BufferArg::from_raw_parts(segment_count_handle, 1),
                    BufferArg::from_raw_parts(heads.handle.clone(), heads.len()),
                );
            }
        }

        let ids = exec.alloc::<u32>(value_len);
        if value_len != 0 {
            crate::inclusive_scan(exec, heads.slice(..), MaxU32, ids.slice_mut(..))?;
        }

        Ok(Self {
            offsets,
            heads,
            ids,
            segment_count,
            value_len,
        })
    }

    pub(crate) fn reverse_indices(&self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error> {
        let indices = exec.alloc::<u32>(self.value_len);
        if self.value_len == 0 {
            return Ok(indices);
        }
        let len = u32::try_from(self.value_len).map_err(|_| Error::LengthTooLarge {
            len: self.value_len,
        })?;
        let len_handle = exec.client().create_from_slice(u32::as_bytes(&[len]));
        unsafe {
            reverse_indices_kernel::launch_unchecked::<R>(
                exec.client(),
                crate::launch::cube_count_1d(self.value_len.div_ceil(BLOCK_SIZE as usize))?,
                CubeDim::new_1d(BLOCK_SIZE),
                BufferArg::from_raw_parts(self.offsets.handle.clone(), self.offsets.len()),
                BufferArg::from_raw_parts(self.ids.handle.clone(), self.ids.len()),
                BufferArg::from_raw_parts(len_handle, 1),
                BufferArg::from_raw_parts(indices.handle.clone(), indices.len()),
            );
        }
        Ok(indices)
    }

    pub(crate) fn merge_heads(
        &self,
        exec: &Executor<R>,
        flags: &DeviceVec<R, u32>,
    ) -> Result<(), Error> {
        if flags.len() != self.value_len {
            return Err(Error::LengthMismatch {
                left: self.value_len,
                right: flags.len(),
            });
        }
        if self.value_len == 0 {
            return Ok(());
        }
        let len = u32::try_from(self.value_len).map_err(|_| Error::LengthTooLarge {
            len: self.value_len,
        })?;
        let len_handle = exec.client().create_from_slice(u32::as_bytes(&[len]));
        unsafe {
            merge_head_flags_kernel::launch_unchecked::<R>(
                exec.client(),
                crate::launch::cube_count_1d(self.value_len.div_ceil(BLOCK_SIZE as usize))?,
                CubeDim::new_1d(BLOCK_SIZE),
                BufferArg::from_raw_parts(self.heads.handle.clone(), self.heads.len()),
                BufferArg::from_raw_parts(len_handle, 1),
                BufferArg::from_raw_parts(flags.handle.clone(), flags.len()),
            );
        }
        Ok(())
    }

    pub(crate) fn clear_heads(
        &self,
        exec: &Executor<R>,
        flags: &DeviceVec<R, u32>,
    ) -> Result<(), Error> {
        if flags.len() != self.value_len {
            return Err(Error::LengthMismatch {
                left: self.value_len,
                right: flags.len(),
            });
        }
        if self.value_len == 0 {
            return Ok(());
        }
        let len = u32::try_from(self.value_len).map_err(|_| Error::LengthTooLarge {
            len: self.value_len,
        })?;
        let len_handle = exec.client().create_from_slice(u32::as_bytes(&[len]));
        unsafe {
            clear_head_flags_kernel::launch_unchecked::<R>(
                exec.client(),
                crate::launch::cube_count_1d(self.value_len.div_ceil(BLOCK_SIZE as usize))?,
                CubeDim::new_1d(BLOCK_SIZE),
                BufferArg::from_raw_parts(self.heads.handle.clone(), self.heads.len()),
                BufferArg::from_raw_parts(len_handle, 1),
                BufferArg::from_raw_parts(flags.handle.clone(), flags.len()),
            );
        }
        Ok(())
    }

    pub(crate) fn sorted_until_candidates(
        &self,
        exec: &Executor<R>,
        breaks: &DeviceVec<R, u32>,
    ) -> Result<DeviceVec<R, u32>, Error> {
        if breaks.len() != self.value_len {
            return Err(Error::LengthMismatch {
                left: self.value_len,
                right: breaks.len(),
            });
        }
        let candidates = exec.alloc::<u32>(self.value_len);
        if self.value_len == 0 {
            return Ok(candidates);
        }
        let len = u32::try_from(self.value_len).map_err(|_| Error::LengthTooLarge {
            len: self.value_len,
        })?;
        let len_handle = exec.client().create_from_slice(u32::as_bytes(&[len]));
        unsafe {
            sorted_until_candidates_kernel::launch_unchecked::<R>(
                exec.client(),
                crate::launch::cube_count_1d(self.value_len.div_ceil(BLOCK_SIZE as usize))?,
                CubeDim::new_1d(BLOCK_SIZE),
                BufferArg::from_raw_parts(self.offsets.handle.clone(), self.offsets.len()),
                BufferArg::from_raw_parts(self.heads.handle.clone(), self.heads.len()),
                BufferArg::from_raw_parts(self.ids.handle.clone(), self.ids.len()),
                BufferArg::from_raw_parts(breaks.handle.clone(), breaks.len()),
                BufferArg::from_raw_parts(len_handle, 1),
                BufferArg::from_raw_parts(candidates.handle.clone(), candidates.len()),
            );
        }
        Ok(candidates)
    }

    pub(crate) fn finish_sorted_until(
        &self,
        exec: &Executor<R>,
        reduced: &DeviceVec<R, u32>,
    ) -> Result<DeviceVec<R, u32>, Error> {
        if reduced.len() != self.segment_count {
            return Err(Error::LengthMismatch {
                left: self.segment_count,
                right: reduced.len(),
            });
        }
        let output = exec.alloc::<u32>(self.segment_count);
        if self.segment_count == 0 {
            return Ok(output);
        }
        let segment_count =
            u32::try_from(self.segment_count).map_err(|_| Error::LengthTooLarge {
                len: self.segment_count,
            })?;
        let segment_count_handle = exec
            .client()
            .create_from_slice(u32::as_bytes(&[segment_count]));
        unsafe {
            finish_sorted_until_kernel::launch_unchecked::<R>(
                exec.client(),
                crate::launch::cube_count_1d(self.segment_count.div_ceil(BLOCK_SIZE as usize))?,
                CubeDim::new_1d(BLOCK_SIZE),
                BufferArg::from_raw_parts(self.offsets.handle.clone(), self.offsets.len()),
                BufferArg::from_raw_parts(reduced.handle.clone(), reduced.len()),
                BufferArg::from_raw_parts(segment_count_handle, 1),
                BufferArg::from_raw_parts(output.handle.clone(), output.len()),
            );
        }
        Ok(output)
    }

    pub(crate) fn compact<Item, Output, OutputOffsets>(
        &self,
        exec: &Executor<R>,
        storage: &<Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
        flags: DeviceVec<R, u32>,
        output: Output,
        output_offsets: OutputOffsets,
    ) -> Result<MIndex, Error>
    where
        Item: MItem<R>,
        Output: MIterMut<R>,
        Output::Item: WriteFrom<Item>,
        OutputOffsets: MIterMut<R, Item = MIndex>,
    {
        let positions = crate::core::scan::inclusive_scan_u32(exec, &flags)?;
        let offset_count = self.segment_count + 1;
        let selected_offsets = if self.value_len == 0 {
            exec.full(offset_count, 0u32)?
        } else {
            let selected_offsets = exec.alloc::<u32>(offset_count);
            let offset_count_u32 = u32::try_from(offset_count)
                .map_err(|_| Error::LengthTooLarge { len: offset_count })?;
            let offset_count_handle = exec
                .client()
                .create_from_slice(u32::as_bytes(&[offset_count_u32]));
            unsafe {
                selected_offsets_kernel::launch_unchecked::<R>(
                    exec.client(),
                    crate::launch::cube_count_1d(offset_count.div_ceil(BLOCK_SIZE as usize))?,
                    CubeDim::new_1d(BLOCK_SIZE),
                    BufferArg::from_raw_parts(self.offsets.handle.clone(), self.offsets.len()),
                    BufferArg::from_raw_parts(positions.handle.clone(), positions.len()),
                    BufferArg::from_raw_parts(offset_count_handle, 1),
                    BufferArg::from_raw_parts(
                        selected_offsets.handle.clone(),
                        selected_offsets.len(),
                    ),
                );
            }
            selected_offsets
        };
        selected_offsets
            .slice(..)
            .transform_into(exec, crate::op::Identity, output_offsets)?;

        let selection = crate::selection::SelectionControl::from_positions(exec, positions, false)?;
        crate::core::facade::KernelWrite::select_storage_control(
            output.lower_write_from::<Item>(),
            exec,
            storage,
            &selection,
        )
    }
}

pub(crate) struct GatherControl<R: Runtime> {
    pub(crate) output_offsets: DeviceVec<R, u32>,
    pub(crate) source_indices: DeviceVec<R, u32>,
    pub(crate) output_len: MIndex,
}

impl<R: Runtime> GatherControl<R> {
    pub(crate) fn new<InputOffsets, Indices>(
        exec: &Executor<R>,
        input_offsets: InputOffsets,
        segment_indices: Indices,
    ) -> Result<Self, Error>
    where
        InputOffsets: MIter<R, Item = MIndex>,
        Indices: MIter<R, Item = MIndex>,
    {
        let input_offsets = input_offsets.materialize_u32(exec)?;
        if input_offsets.is_empty() {
            return Err(Error::LengthMismatch { left: 1, right: 0 });
        }
        let segment_indices = segment_indices.materialize_u32(exec)?;
        let selection_count = segment_indices.len();
        let lengths = exec.alloc::<u32>(selection_count);

        if selection_count != 0 {
            unsafe {
                gathered_segment_lengths_kernel::launch_unchecked::<R>(
                    exec.client(),
                    crate::launch::cube_count_1d(selection_count.div_ceil(BLOCK_SIZE as usize))?,
                    CubeDim::new_1d(BLOCK_SIZE),
                    BufferArg::from_raw_parts(input_offsets.handle.clone(), input_offsets.len()),
                    BufferArg::from_raw_parts(
                        segment_indices.handle.clone(),
                        segment_indices.len(),
                    ),
                    BufferArg::from_raw_parts(lengths.handle.clone(), lengths.len()),
                );
            }
        }

        let positions = crate::core::scan::inclusive_scan_u32(exec, &lengths)?;
        let output_len = crate::core::scan::last_u32(exec, &positions)?;
        let output_offset_count = selection_count
            .checked_add(1)
            .ok_or(Error::LengthTooLarge {
                len: selection_count,
            })?;
        let output_offsets = if selection_count == 0 {
            exec.full(1, 0u32)?
        } else {
            let output_offsets = exec.alloc::<u32>(output_offset_count);
            unsafe {
                gathered_offsets_kernel::launch_unchecked::<R>(
                    exec.client(),
                    crate::launch::cube_count_1d(selection_count.div_ceil(BLOCK_SIZE as usize))?,
                    CubeDim::new_1d(BLOCK_SIZE),
                    BufferArg::from_raw_parts(positions.handle.clone(), positions.len()),
                    BufferArg::from_raw_parts(output_offsets.handle.clone(), output_offsets.len()),
                );
            }
            output_offsets
        };

        let source_indices = exec.alloc::<u32>(output_len as usize);
        if output_len != 0 {
            let output_control = SegmentControl::from_materialized(
                exec,
                output_offsets.clone(),
                output_len as usize,
            )?;
            unsafe {
                gathered_source_indices_kernel::launch_unchecked::<R>(
                    exec.client(),
                    crate::launch::cube_count_1d(
                        (output_len as usize).div_ceil(BLOCK_SIZE as usize),
                    )?,
                    CubeDim::new_1d(BLOCK_SIZE),
                    BufferArg::from_raw_parts(input_offsets.handle.clone(), input_offsets.len()),
                    BufferArg::from_raw_parts(
                        segment_indices.handle.clone(),
                        segment_indices.len(),
                    ),
                    BufferArg::from_raw_parts(output_offsets.handle.clone(), output_offsets.len()),
                    BufferArg::from_raw_parts(
                        output_control.ids.handle.clone(),
                        output_control.ids.len(),
                    ),
                    BufferArg::from_raw_parts(source_indices.handle.clone(), source_indices.len()),
                );
            }
        }

        Ok(Self {
            output_offsets,
            source_indices,
            output_len,
        })
    }
}
