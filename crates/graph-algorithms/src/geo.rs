//! Iterative unknown-vertex geolocation by neighbor-coordinate averaging.

use cubecl::prelude::*;
use massively::{Executor, graph, op::Identity, op::ReductionOp, zip2};

use super::common::{self, CsrGraph, DeviceGraph};

struct SumCoordinates;

#[cubecl::cube]
impl ReductionOp<(f32, f32)> for SumCoordinates {
    fn apply(lhs: (f32, f32), rhs: (f32, f32)) -> (f32, f32) {
        (lhs.0 + rhs.0, lhs.1 + rhs.1)
    }
}

pub fn solve<R: Runtime>(
    exec: &Executor<R>,
    graph: &CsrGraph,
    initial: &[(f32, f32)],
    known: &[bool],
    iterations: usize,
) -> common::Result<Vec<(f32, f32)>> {
    assert_eq!(initial.len(), graph.vertex_count());
    assert_eq!(known.len(), graph.vertex_count());

    let degree = exec.to_host(&common::degrees(exec, graph)?)?;
    let device_graph = DeviceGraph::new(exec, graph);
    let frontier = common::all_vertices(exec, graph);
    let mut coordinates = initial.to_vec();

    for _ in 0..iterations {
        let xs = exec.to_device(&coordinates.iter().map(|value| value.0).collect::<Vec<_>>());
        let ys = exec.to_device(&coordinates.iter().map(|value| value.1).collect::<Vec<_>>());
        let sums = graph::traverse(exec, device_graph.csr(), frontier.slice(..))?
            .map(
                zip2(
                    graph::destination(xs.slice(..)),
                    graph::destination(ys.slice(..)),
                ),
                Identity,
            )
            .reduce_by_source(exec, (0.0, 0.0), SumCoordinates)?;

        let sum_x = exec.to_host(&sums.0)?;
        let sum_y = exec.to_host(&sums.1)?;
        for vertex in 0..coordinates.len() {
            if !known[vertex] && degree[vertex] != 0 {
                coordinates[vertex] = (
                    sum_x[vertex] / degree[vertex] as f32,
                    sum_y[vertex] / degree[vertex] as f32,
                );
            }
        }
    }

    Ok(coordinates)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn unknown_center_is_neighbor_mean() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        let graph = CsrGraph::new(vec![0, 2, 3, 4], vec![1, 2, 0, 0]);
        let result = solve(
            &exec,
            &graph,
            &[(0.0, 0.0), (0.0, 0.0), (2.0, 2.0)],
            &[false, true, true],
            1,
        )
        .unwrap();
        assert_eq!(result, vec![(1.0, 1.0), (0.0, 0.0), (2.0, 2.0)]);
    }
}
