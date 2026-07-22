//! Triangle counting by batched intersection over every directed CSR edge.

use cubecl::prelude::*;
use massively::{Executor, graph, op::Identity, op::ReductionOp};

use super::common::{self, DeviceCsr};

struct SumU32;

#[cubecl::cube]
impl ReductionOp<u32> for SumU32 {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        lhs + rhs
    }
}

pub fn solve<R: Runtime>(exec: &Executor<R>, graph: &DeviceCsr<R>) -> common::Result<u32> {
    if graph.edge_count() == 0 {
        return Ok(0);
    }
    let sources = graph::traverse(
        exec,
        graph.csr(),
        common::counting_u32(0, graph.vertex_count() as usize),
        graph.edge_capacity()?,
    )?
    .map(graph::source_id(), Identity)
    .emit(exec)?;
    let sources = common::materialize_exact(exec, sources)?;
    let counts = graph::intersect_count(
        exec,
        graph.csr(),
        sources.slice(..),
        graph.destinations().slice(..),
    )?;
    Ok(massively::vector::reduce(exec, counts.slice(..), 0, SumU32)? / 6)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn shared_edge_forms_two_triangles() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        let graph = DeviceCsr::from_host(&exec, &common::sample_graph()).unwrap();
        assert_eq!(solve(&exec, &graph).unwrap(), 2);
    }
}
