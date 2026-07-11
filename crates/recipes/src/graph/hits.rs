//! HITS as alternating source and destination reductions.

use cubecl::prelude::*;
use massively::{Executor, graph, op::Identity, op::ReductionOp};

use super::common::{self, CsrGraph, DeviceGraph};

struct SumF32;

#[cubecl::cube]
impl ReductionOp<f32> for SumF32 {
    fn apply(lhs: f32, rhs: f32) -> f32 {
        lhs + rhs
    }
}

fn normalize(values: &mut [f32]) {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm != 0.0 {
        for value in values {
            *value /= norm;
        }
    }
}

pub fn solve<R: Runtime>(
    exec: &Executor<R>,
    graph: &CsrGraph,
    iterations: usize,
) -> common::Result<(Vec<f32>, Vec<f32>)> {
    let n = graph.vertex_count();
    let device_graph = DeviceGraph::new(exec, graph);
    let frontier = common::all_vertices(exec, graph);
    let mut hubs = vec![1.0f32; n];
    let mut authorities = vec![1.0f32; n];

    for _ in 0..iterations {
        let hubs_gpu = exec.to_device(&hubs);
        let authority_gpu = exec.full(n, 0.0f32)?;
        graph::traverse(exec, device_graph.csr(), frontier.slice(..))?
            .map(graph::source(hubs_gpu.slice(..)), Identity)
            .reduce_by_destination(exec, 0.0, SumF32, authority_gpu.slice_mut(..))?;
        authorities = exec.to_host(&authority_gpu)?;
        normalize(&mut authorities);

        let authority_gpu = exec.to_device(&authorities);
        let hub_gpu = exec.alloc::<f32>(n);
        graph::traverse(exec, device_graph.csr(), frontier.slice(..))?
            .map(graph::destination(authority_gpu.slice(..)), Identity)
            .reduce_by_source(exec, 0.0, SumF32, hub_gpu.slice_mut(..))?;
        hubs = exec.to_host(&hub_gpu)?;
        normalize(&mut hubs);
    }

    Ok((hubs, authorities))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn vectors_are_normalized() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        let (hubs, authorities) = solve(&exec, &common::sample_graph(), 4).unwrap();
        let hub_norm = hubs.iter().map(|value| value * value).sum::<f32>().sqrt();
        let authority_norm = authorities
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        assert!((hub_norm - 1.0).abs() < 1e-5);
        assert!((authority_norm - 1.0).abs() < 1e-5);
    }
}
