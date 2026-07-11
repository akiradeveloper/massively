//! Private planning data for semantic edge traversals.

use cubecl::prelude::*;

use crate::{DeviceVec, Error, Executor, MIndex, MIter, seg::control::SegmentControl};

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
    sources: &mut [u32],
    destinations: &mut [u32],
    edges: &mut [u32],
) {
    let output_index = ABSOLUTE_POS as usize;
    if output_index < edges.len() {
        let output_segment = output_segment_ids[output_index] as usize - 1usize;
        let source = vertices[output_segment];
        let edge =
            input_offsets[source as usize] + output_index as u32 - output_offsets[output_segment];
        sources[output_index] = source;
        destinations[output_index] = input_destinations[edge as usize];
        edges[output_index] = edge;
    }
}

pub(crate) struct TraversalControl<R: Runtime> {
    pub(super) output_offsets: DeviceVec<R, u32>,
    pub(super) sources: DeviceVec<R, u32>,
    pub(super) destinations: DeviceVec<R, u32>,
    pub(super) edges: DeviceVec<R, u32>,
    pub(super) output_len: MIndex,
}

impl<R: Runtime> TraversalControl<R> {
    pub(super) fn new<Destinations, InputOffsets, Vertices>(
        exec: &Executor<R>,
        destinations: Destinations,
        input_offsets: InputOffsets,
        vertices: Vertices,
    ) -> Result<Self, Error>
    where
        Destinations: MIter<R, Item = MIndex>,
        InputOffsets: MIter<R, Item = MIndex>,
        Vertices: MIter<R, Item = MIndex>,
    {
        let destinations = destinations.materialize_u32(exec)?;
        let input_offsets = input_offsets.materialize_u32(exec)?;
        if input_offsets.is_empty() {
            return Err(Error::LengthMismatch { left: 1, right: 0 });
        }
        let vertices = vertices.materialize_u32(exec)?;
        let vertex_count = vertices.len();
        let lengths = exec.alloc::<u32>(vertex_count);

        if vertex_count != 0 {
            unsafe {
                traversal_lengths_kernel::launch_unchecked::<R>(
                    exec.client(),
                    crate::launch::cube_count_1d(vertex_count.div_ceil(BLOCK_SIZE as usize))?,
                    CubeDim::new_1d(BLOCK_SIZE),
                    BufferArg::from_raw_parts(input_offsets.handle.clone(), input_offsets.len()),
                    BufferArg::from_raw_parts(vertices.handle.clone(), vertices.len()),
                    BufferArg::from_raw_parts(lengths.handle.clone(), lengths.len()),
                );
            }
        }

        let positions = crate::core::scan::inclusive_scan_u32(exec, &lengths)?;
        let output_len = crate::core::scan::last_u32(exec, &positions)?;
        let output_offset_count = vertex_count
            .checked_add(1)
            .ok_or(Error::LengthTooLarge { len: vertex_count })?;
        let output_offsets = if vertex_count == 0 {
            exec.full(1, 0u32)?
        } else {
            let output_offsets = exec.alloc::<u32>(output_offset_count);
            unsafe {
                traversal_offsets_kernel::launch_unchecked::<R>(
                    exec.client(),
                    crate::launch::cube_count_1d(vertex_count.div_ceil(BLOCK_SIZE as usize))?,
                    CubeDim::new_1d(BLOCK_SIZE),
                    BufferArg::from_raw_parts(positions.handle.clone(), positions.len()),
                    BufferArg::from_raw_parts(output_offsets.handle.clone(), output_offsets.len()),
                );
            }
            output_offsets
        };

        let sources = exec.alloc::<u32>(output_len as usize);
        let output_destinations = exec.alloc::<u32>(output_len as usize);
        let edges = exec.alloc::<u32>(output_len as usize);
        if output_len != 0 {
            let output_control = SegmentControl::from_materialized(
                exec,
                output_offsets.clone(),
                output_len as usize,
            )?;
            unsafe {
                traversal_context_kernel::launch_unchecked::<R>(
                    exec.client(),
                    crate::launch::cube_count_1d(
                        (output_len as usize).div_ceil(BLOCK_SIZE as usize),
                    )?,
                    CubeDim::new_1d(BLOCK_SIZE),
                    BufferArg::from_raw_parts(input_offsets.handle.clone(), input_offsets.len()),
                    BufferArg::from_raw_parts(destinations.handle.clone(), destinations.len()),
                    BufferArg::from_raw_parts(vertices.handle.clone(), vertices.len()),
                    BufferArg::from_raw_parts(output_offsets.handle.clone(), output_offsets.len()),
                    BufferArg::from_raw_parts(
                        output_control.ids.handle.clone(),
                        output_control.ids.len(),
                    ),
                    BufferArg::from_raw_parts(sources.handle.clone(), sources.len()),
                    BufferArg::from_raw_parts(
                        output_destinations.handle.clone(),
                        output_destinations.len(),
                    ),
                    BufferArg::from_raw_parts(edges.handle.clone(), edges.len()),
                );
            }
        }

        Ok(Self {
            output_offsets,
            sources,
            destinations: output_destinations,
            edges,
            output_len,
        })
    }
}
