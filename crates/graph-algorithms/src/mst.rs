//! Kruskal minimum spanning forest over resident weighted CSR storage.

use cubecl::prelude::*;
use massively::{
    DeviceVec, Executor, MVec, graph, lazy,
    op::{BinaryPredicateOp, Identity, UnaryOp},
    vector, zip2, zip3,
};

use super::common::{self, DeviceWeightedCsr};

struct LessF32;

#[cubecl::cube]
impl BinaryPredicateOp<f32> for LessF32 {
    fn apply(lhs: f32, rhs: f32) -> bool {
        lhs < rhs
    }
}

struct EqualPair;

#[cubecl::cube]
impl UnaryOp<(u32, u32)> for EqualPair {
    type Output = u32;

    fn apply(input: (u32, u32)) -> u32 {
        if input.0 == input.1 { 1u32 } else { 0u32 }
    }
}

fn read_u32<R: Runtime>(
    exec: &Executor<R>,
    values: &DeviceVec<R, u32>,
    index: usize,
) -> common::Result<u32> {
    Ok(exec.to_host(&values.slice(index..index + 1))?[0])
}

pub fn solve<R: Runtime>(
    exec: &Executor<R>,
    graph: &DeviceWeightedCsr<R, f32>,
) -> common::Result<MVec<R, ((u32, u32), f32)>> {
    let topology = graph.graph();
    let edge_count = topology.edge_count();
    let sources = graph::traverse(
        exec,
        topology.csr(),
        lazy::counting(0).take(topology.vertex_count()),
    )?
    .map(graph::source_id(), Identity)
    .emit(exec)?;
    let (_, edge_order) = vector::sort_by_key(
        exec,
        graph.weights().slice(..),
        lazy::counting(0).take(edge_count as u32),
        LessF32,
    )?;
    let components = vector::transform(
        exec,
        lazy::counting(0).take(topology.vertex_count()),
        Identity,
    )?;
    let selected = vector::fill(exec, edge_count, 0u32)?;

    for position in 0..edge_count {
        let edge = read_u32(exec, &edge_order, position)?;
        let source = read_u32(exec, &sources, edge as usize)?;
        let destination = read_u32(exec, topology.destinations(), edge as usize)?;
        let source_component = read_u32(exec, &components, source as usize)?;
        let destination_component = read_u32(exec, &components, destination as usize)?;
        if source_component == destination_component {
            continue;
        }

        let stencil = vector::transform(
            exec,
            zip2(
                components.slice(..),
                lazy::constant(destination_component).take(topology.vertex_count()),
            ),
            EqualPair,
        )?;
        vector::replace_where(
            exec,
            source_component,
            stencil.slice(..),
            components.slice_mut(..),
        )?;
        vector::scatter(
            exec,
            lazy::constant(1u32).take(1),
            lazy::constant(edge).take(1),
            selected.slice_mut(..),
        )?;
    }

    vector::copy_where(
        exec,
        zip3(
            sources.slice(..),
            topology.destinations().slice(..),
            graph.weights().slice(..),
        ),
        selected.slice(..),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WeightedCsr;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn weighted_path_is_its_own_tree() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        let graph = WeightedCsr::new(common::path_graph(), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
        let graph = DeviceWeightedCsr::from_host(&exec, &graph).unwrap();
        let tree = solve(&exec, &graph).unwrap();
        assert_eq!(tree.0.0.len(), 3);
        assert_eq!(exec.to_host(&tree.1).unwrap().iter().sum::<f32>(), 6.0);
    }
}
