//! CSR sparse matrix-vector multiplication as an edge expression reduced by row.

use cubecl::prelude::*;
use massively::{Executor, graph, op::ReductionOp, op::UnaryOp, zip2};

use super::common::{self, DeviceGraph, WeightedCsr};

struct MulF32;

#[cubecl::cube]
impl UnaryOp<(f32, f32)> for MulF32 {
    type Output = f32;

    fn apply(input: (f32, f32)) -> f32 {
        input.0 * input.1
    }
}

struct SumF32;

#[cubecl::cube]
impl ReductionOp<f32> for SumF32 {
    fn apply(lhs: f32, rhs: f32) -> f32 {
        lhs + rhs
    }
}

pub fn solve<R: Runtime>(
    exec: &Executor<R>,
    matrix: &WeightedCsr,
    vector: &[f32],
) -> common::Result<Vec<f32>> {
    assert_eq!(vector.len(), matrix.graph.vertex_count());
    let device_graph = DeviceGraph::new(exec, &matrix.graph);
    let frontier = common::all_vertices(exec, &matrix.graph);
    let weights = exec.to_device(&matrix.weights);
    let vector = exec.to_device(vector);
    let output = exec.alloc::<f32>(matrix.graph.vertex_count());
    graph::traverse(exec, device_graph.csr(), frontier.slice(..))?
        .map(
            zip2(
                graph::edge(weights.slice(..)),
                graph::destination(vector.slice(..)),
            ),
            MulF32,
        )
        .reduce_by_source(exec, 0.0, SumF32, output.slice_mut(..))?;
    exec.to_host(&output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn path_adjacency_multiplies_vector() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        let matrix = WeightedCsr::new(common::path_graph(), vec![1.0; 6]);
        assert_eq!(
            solve(&exec, &matrix, &[1.0, 2.0, 3.0, 4.0]).unwrap(),
            vec![2.0, 4.0, 6.0, 3.0]
        );
    }
}
