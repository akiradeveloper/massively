//! Private planning data for semantic edge traversals.

use cubecl::prelude::*;

use crate::{
    DeviceVec, Error, Executor, MIndex, MIter, MVal, op::UnaryOp, seg::control::SegmentControl,
};

const BLOCK_SIZE: u32 = 256;

#[cubecl::cube(launch_unchecked)]
fn traversal_lengths_kernel(input_offsets: &[u32], vertices: &[u32], lengths: &mut [u32]) {
    let selection = ABSOLUTE_POS as usize;
    if selection < lengths.len() {
        let vertex = vertices[selection] as usize;
        lengths[selection] = input_offsets[vertex + 1usize] - input_offsets[vertex];
    }
}

#[cubecl::cube(launch_unchecked)]
fn traversal_offsets_kernel(positions: &[u32], output_offsets: &mut [u32]) {
    let selection = ABSOLUTE_POS as usize;
    if selection < positions.len() {
        if selection == 0usize {
            output_offsets[0] = 0u32;
        }
        output_offsets[selection + 1usize] = positions[selection];
    }
}

#[cubecl::cube(launch_unchecked)]
fn traversal_context_kernel(
    input_offsets: &[u32],
    input_destinations: &[u32],
    vertices: &[u32],
    output_offsets: &[u32],
    output_segment_ids: &[u32],
    active_len: &[u32],
    sources: &mut [u32],
    destinations: &mut [u32],
    edges: &mut [u32],
) {
    let output_index = ABSOLUTE_POS as usize;
    if output_index < edges.len() && output_index < active_len[0] as usize {
        let output_segment = output_segment_ids[output_index] as usize - 1usize;
        let source = vertices[output_segment];
        let edge =
            input_offsets[source as usize] + output_index as u32 - output_offsets[output_segment];
        sources[output_index] = source;
        destinations[output_index] = input_destinations[edge as usize];
        edges[output_index] = edge;
    }
}

struct MinIndex;

#[cubecl::cube]
impl UnaryOp<(MIndex, MIndex)> for MinIndex {
    type Output = MIndex;

    fn apply(input: (MIndex, MIndex)) -> MIndex {
        u32::min(input.0, input.1)
    }
}

struct FitsCapacity;

#[cubecl::cube]
impl UnaryOp<(MIndex, MIndex)> for FitsCapacity {
    type Output = crate::MBool;

    fn apply(input: (MIndex, MIndex)) -> crate::MBool {
        crate::op::mbool(input.0 <= input.1)
    }
}

pub(crate) struct TraversalControl<R: Runtime> {
    pub(super) output_offsets: DeviceVec<R, u32>,
    pub(super) sources: DeviceVec<R, u32>,
    pub(super) destinations: DeviceVec<R, u32>,
    pub(super) edges: DeviceVec<R, u32>,
    /// Number of materialized edges, clamped to the caller-provided capacity.
    pub(super) output_len: MVal<R, MIndex>,
    /// Unclamped number of edges selected by the frontier.
    pub(super) required_len: MVal<R, MIndex>,
    pub(super) fits: MVal<R, crate::MBool>,
    pub(super) capacity: MIndex,
    pub(super) source_count: MIndex,
    pub(super) vertex_count: MIndex,
}

impl<R: Runtime> TraversalControl<R> {
    pub(super) fn new<Destinations, InputOffsets, Vertices>(
        exec: &Executor<R>,
        destinations: Destinations,
        input_offsets: InputOffsets,
        vertices: Vertices,
        max_edges: MIndex,
    ) -> Result<Self, Error>
    where
        Destinations: MIter<R, Item = MIndex>,
        InputOffsets: MIter<R, Item = MIndex>,
        Vertices: MIter<R, Item = MIndex>,
    {
        let destinations = crate::api::iter::materialize_u32(exec, destinations)?;
        let input_offsets = crate::api::iter::materialize_u32(exec, input_offsets)?;
        if input_offsets.capacity() == 0 {
            return Err(Error::LengthMismatch { left: 1, right: 0 });
        }
        let vertices = crate::api::iter::materialize_u32(exec, vertices)?;
        let source_count = vertices.capacity();
        let vertex_count = input_offsets.capacity() - 1;
        let source_count_index =
            u32::try_from(source_count).map_err(|_| Error::LengthTooLarge { len: source_count })?;
        let vertex_count_index =
            u32::try_from(vertex_count).map_err(|_| Error::LengthTooLarge { len: vertex_count })?;
        let lengths = exec.alloc::<u32>(source_count);

        if source_count != 0 {
            unsafe {
                traversal_lengths_kernel::launch_unchecked::<R>(
                    exec.client(),
                    crate::launch::cube_count_1d(source_count.div_ceil(BLOCK_SIZE as usize))?,
                    CubeDim::new_1d(BLOCK_SIZE),
                    BufferArg::from_raw_parts(
                        input_offsets.handle.clone(),
                        input_offsets.capacity(),
                    ),
                    BufferArg::from_raw_parts(vertices.handle.clone(), vertices.capacity()),
                    BufferArg::from_raw_parts(lengths.handle.clone(), lengths.capacity()),
                );
            }
        }

        let positions = crate::core::scan::inclusive_scan_u32(exec, &lengths)?;
        let required_len = MVal::from_storage(crate::core::scan::last_u32(exec, &positions)?)?;
        let capacity = exec.value(max_edges)?;
        let output_len = MVal::from_storage(crate::vector::transform(
            exec,
            crate::zip2(required_len.as_iter(), capacity.as_iter()),
            MinIndex,
        )?)?;
        let fits = MVal::from_storage(crate::vector::transform(
            exec,
            crate::zip2(required_len.as_iter(), capacity.as_iter()),
            FitsCapacity,
        )?)?;
        let output_offset_count = source_count
            .checked_add(1)
            .ok_or(Error::LengthTooLarge { len: source_count })?;
        let output_offsets = if source_count == 0 {
            let zero = exec.value(0u32)?;
            exec.full(1, &zero)?
        } else {
            let output_offsets = exec.alloc::<u32>(output_offset_count);
            unsafe {
                traversal_offsets_kernel::launch_unchecked::<R>(
                    exec.client(),
                    crate::launch::cube_count_1d(source_count.div_ceil(BLOCK_SIZE as usize))?,
                    CubeDim::new_1d(BLOCK_SIZE),
                    BufferArg::from_raw_parts(positions.handle.clone(), positions.capacity()),
                    BufferArg::from_raw_parts(
                        output_offsets.handle.clone(),
                        output_offsets.capacity(),
                    ),
                );
            }
            output_offsets
        };

        let zero = exec.value(0u32)?;
        let sources = exec.full(max_edges as usize, &zero)?;
        let output_destinations = exec.full(max_edges as usize, &zero)?;
        let edges = exec.full(max_edges as usize, &zero)?;
        if max_edges != 0 {
            let output_len_storage: &DeviceVec<R, u32> = output_len.scratch_storage();
            let output_control = SegmentControl::from_materialized(
                exec,
                output_offsets.clone(),
                max_edges as usize,
            )?;
            unsafe {
                traversal_context_kernel::launch_unchecked::<R>(
                    exec.client(),
                    crate::launch::cube_count_1d(
                        (max_edges as usize).div_ceil(BLOCK_SIZE as usize),
                    )?,
                    CubeDim::new_1d(BLOCK_SIZE),
                    BufferArg::from_raw_parts(
                        input_offsets.handle.clone(),
                        input_offsets.capacity(),
                    ),
                    BufferArg::from_raw_parts(destinations.handle.clone(), destinations.capacity()),
                    BufferArg::from_raw_parts(vertices.handle.clone(), vertices.capacity()),
                    BufferArg::from_raw_parts(
                        output_offsets.handle.clone(),
                        output_offsets.capacity(),
                    ),
                    BufferArg::from_raw_parts(
                        output_control.ids.handle.clone(),
                        output_control.ids.capacity(),
                    ),
                    BufferArg::from_raw_parts(output_len_storage.handle.clone(), 1),
                    BufferArg::from_raw_parts(sources.handle.clone(), sources.capacity()),
                    BufferArg::from_raw_parts(
                        output_destinations.handle.clone(),
                        output_destinations.capacity(),
                    ),
                    BufferArg::from_raw_parts(edges.handle.clone(), edges.capacity()),
                );
            }
        }

        Ok(Self {
            output_offsets,
            sources,
            destinations: output_destinations,
            edges,
            output_len,
            required_len,
            fits,
            capacity: max_edges,
            source_count: source_count_index,
            vertex_count: vertex_count_index,
        })
    }
}
