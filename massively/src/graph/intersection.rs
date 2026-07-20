//! Batched adjacency-set operations.

use cubecl::prelude::*;

use crate::{Error, Executor, MIndex, MIter, MIterMut, MVec};

use super::Csr;

const BLOCK_SIZE: u32 = 256;

#[cubecl::cube(launch_unchecked)]
fn intersect_count_kernel(
    offsets: &[u32],
    destinations: &[u32],
    sources: &[u32],
    targets: &[u32],
    output: &mut [u32],
) {
    let pair = ABSOLUTE_POS as usize;
    if pair < output.len() {
        let source = sources[pair] as usize;
        let target = targets[pair] as usize;
        let left = RuntimeCell::<usize>::new(offsets[source] as usize);
        let left_end = offsets[source + 1usize] as usize;
        let right = RuntimeCell::<usize>::new(offsets[target] as usize);
        let right_end = offsets[target + 1usize] as usize;
        let count = RuntimeCell::<u32>::new(0u32);

        while left.read() < left_end && right.read() < right_end {
            let lhs = destinations[left.read()];
            let rhs = destinations[right.read()];
            if lhs < rhs {
                left.store(left.read() + 1usize);
            } else if rhs < lhs {
                right.store(right.read() + 1usize);
            } else {
                count.store(count.read() + 1u32);
                left.store(left.read() + 1usize);
                right.store(right.read() + 1usize);
            }
        }
        output[pair] = count.read();
    }
}

/// Counts each pair's common destinations in one batched GPU operation.
///
/// Every CSR row must be sorted by destination. One result is written per `(source, target)` pair.
/// The current lowering assigns one thread per pair; future schedulers may select warp-, block-,
/// binary-search-, or bitmap-based intersection without changing this contract.
pub fn intersect_count<R, Destinations, Offsets, Sources, Targets>(
    exec: &Executor<R>,
    graph: Csr<Destinations, Offsets>,
    sources: Sources,
    targets: Targets,
) -> Result<MVec<R, MIndex>, Error>
where
    R: Runtime,
    Destinations: MIter<R, Item = MIndex>,
    Offsets: MIter<R, Item = MIndex>,
    Sources: MIter<R, Item = MIndex>,
    Targets: MIter<R, Item = MIndex>,
{
    let pair_count = sources.capacity()? as usize;
    let output = exec.alloc::<u32>(pair_count);
    intersect_count_into(exec, graph, sources, targets, output.slice_mut(..))?;
    Ok(output)
}

/// Counts common destinations into caller-provided storage.
fn intersect_count_into<R, Destinations, Offsets, Sources, Targets, Output>(
    exec: &Executor<R>,
    graph: Csr<Destinations, Offsets>,
    sources: Sources,
    targets: Targets,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Destinations: MIter<R, Item = MIndex>,
    Offsets: MIter<R, Item = MIndex>,
    Sources: MIter<R, Item = MIndex>,
    Targets: MIter<R, Item = MIndex>,
    Output: MIterMut<R, Item = MIndex>,
{
    let pair_count = sources.capacity()?;
    let target_count = targets.capacity()?;
    let output_count = output.capacity()?;
    if pair_count != target_count {
        return Err(Error::LengthMismatch {
            left: pair_count as usize,
            right: target_count as usize,
        });
    }
    if pair_count != output_count {
        return Err(Error::LengthMismatch {
            left: pair_count as usize,
            right: output_count as usize,
        });
    }
    if pair_count == 0 {
        return Ok(());
    }

    let (destinations, offsets) = graph.into_parts();
    let destinations = crate::api::iter::materialize_u32(exec, destinations)?;
    let offsets = crate::api::iter::materialize_u32(exec, offsets)?;
    let sources = crate::api::iter::materialize_u32(exec, sources)?;
    let targets = crate::api::iter::materialize_u32(exec, targets)?;
    let counts = exec.alloc_column::<u32>(pair_count as usize);
    unsafe {
        intersect_count_kernel::launch_unchecked::<R>(
            exec.client(),
            crate::launch::cube_count_1d((pair_count as usize).div_ceil(BLOCK_SIZE as usize))?,
            CubeDim::new_1d(BLOCK_SIZE),
            BufferArg::from_raw_parts(offsets.handle.clone(), offsets.capacity()),
            BufferArg::from_raw_parts(destinations.handle.clone(), destinations.capacity()),
            BufferArg::from_raw_parts(sources.handle.clone(), sources.capacity()),
            BufferArg::from_raw_parts(targets.handle.clone(), targets.capacity()),
            BufferArg::from_raw_parts(counts.handle.clone(), counts.capacity()),
        );
    }
    crate::vector::transform_into(exec, counts.slice(..), crate::op::Identity, output)
}
