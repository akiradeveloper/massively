//! Boolean sparse matrix multiplication into resident CSR storage.

use cubecl::prelude::*;
use massively::{
    Executor, graph, lazy,
    op::{BinaryPredicateOp, Identity},
    vector,
};

use super::common::{self, DeviceCsr};

struct LessU32;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for LessU32 {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs < rhs
    }
}

struct EqualU32;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for EqualU32 {
    fn apply(lhs: u32, rhs: u32) -> bool {
        lhs == rhs
    }
}

pub fn solve<R: Runtime>(
    exec: &Executor<R>,
    lhs: &DeviceCsr<R>,
    rhs: &DeviceCsr<R>,
) -> common::Result<DeviceCsr<R>> {
    assert_eq!(lhs.vertex_count(), rhs.vertex_count());
    let n = lhs.vertex_count();

    let capacity = graph::traverse(exec, rhs.csr(), lhs.destinations().slice(..))?.edge_count();
    let mut destinations = exec.alloc::<u32>(capacity as usize);
    let offsets = exec.alloc::<u32>(n as usize + 1);
    let mut output_len = 0u32;

    for vertex in 0..n {
        vector::scatter(
            exec,
            lazy::constant(output_len).take(1),
            common::indices(lazy::constant(vertex).take(1)),
            offsets.slice_mut(..),
        )?;

        let bounds = exec.to_host(&lhs.offsets().slice(vertex as usize..vertex as usize + 2))?;
        let frontier = lhs
            .destinations()
            .slice(bounds[0] as usize..bounds[1] as usize);
        if frontier.is_empty() {
            continue;
        }

        let traversal = graph::traverse(exec, rhs.csr(), frontier)?;
        if traversal.edge_count() == 0 {
            continue;
        }
        let candidates = traversal
            .map(graph::destination_id(), Identity)
            .emit(exec)?;
        let sorted = vector::sort(exec, candidates.slice(..), LessU32)?;
        let row = vector::unique(exec, sorted.slice(..), EqualU32)?;
        let row_len = u32::try_from(row.len())
            .map_err(|_| massively::Error::LengthTooLarge { len: row.len() })?;
        vector::scatter(
            exec,
            row.slice(..),
            common::indices(common::counting_u32(output_len as usize, row_len as usize)),
            destinations.slice_mut(..),
        )?;
        output_len += row_len;
    }

    vector::scatter(
        exec,
        lazy::constant(output_len).take(1),
        common::indices(lazy::constant(n).take(1)),
        offsets.slice_mut(..),
    )?;
    destinations.truncate(output_len as usize);
    DeviceCsr::from_parts(destinations, offsets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn path_squared_contains_two_hop_pairs() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        let graph = DeviceCsr::from_host(&exec, &common::path_graph()).unwrap();
        let output = solve(&exec, &graph, &graph).unwrap();
        assert_eq!(exec.to_host(output.offsets()).unwrap(), vec![0, 2, 4, 6, 8]);
        assert_eq!(
            exec.to_host(output.destinations()).unwrap(),
            vec![0, 2, 1, 3, 0, 2, 1, 3]
        );
    }
}
