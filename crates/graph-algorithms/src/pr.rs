//! PageRank as source expressions reduced by destination.

use cubecl::prelude::*;
use massively::{Executor, graph, op::ReductionOp, op::UnaryOp, zip2};

use super::common::{self, CsrGraph, DeviceGraph};

struct RankContribution;

#[cubecl::cube]
impl UnaryOp<(f32, u32)> for RankContribution {
    type Output = f32;

    fn apply(input: (f32, u32)) -> f32 {
        if input.1 == 0u32 {
            0.0f32
        } else {
            input.0 / input.1 as f32
        }
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
    graph: &CsrGraph,
    damping: f32,
    iterations: usize,
) -> common::Result<Vec<f32>> {
    let n = graph.vertex_count();
    let degree_gpu = common::degrees(exec, graph)?;
    let degree = exec.to_host(&degree_gpu)?;
    let device_graph = DeviceGraph::new(exec, graph);
    let frontier = common::all_vertices(exec, graph);
    let mut rank = vec![1.0 / n as f32; n];

    for _ in 0..iterations {
        let dangling = rank
            .iter()
            .zip(&degree)
            .filter(|(_, degree)| **degree == 0)
            .map(|(rank, _)| *rank)
            .sum::<f32>();
        let base = vec![(1.0 - damping + damping * dangling) / n as f32; n];
        let scaled_rank =
            exec.to_device(&rank.iter().map(|value| value * damping).collect::<Vec<_>>());
        let output = exec.to_device(&base);
        graph::traverse(exec, device_graph.csr(), frontier.slice(..))?
            .map(
                zip2(
                    graph::source(scaled_rank.slice(..)),
                    graph::source(degree_gpu.slice(..)),
                ),
                RankContribution,
            )
            .update_by_destination(exec, 0.0, SumF32, output.slice_mut(..))?;
        rank = exec.to_host(&output)?;
    }

    Ok(rank)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn symmetric_vertices_receive_equal_rank() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        let rank = solve(&exec, &common::sample_graph(), 0.85, 20).unwrap();
        assert!((rank.iter().sum::<f32>() - 1.0).abs() < 1e-4);
        assert!((rank[0] - rank[3]).abs() < 1e-5);
        assert!((rank[1] - rank[2]).abs() < 1e-5);
    }
}
