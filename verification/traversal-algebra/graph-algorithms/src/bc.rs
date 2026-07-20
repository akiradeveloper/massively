//! Unweighted Brandes betweenness centrality with resident per-source state.

use cubecl::prelude::*;
use massively::{
    DeviceVec, Executor, graph, lazy,
    op::{ReductionOp, UnaryOp},
    vector, zip2, zip3, zip5,
};

use super::{
    bfs,
    common::{self, DeviceCsr},
};

struct IsDepth;

#[cubecl::cube]
impl UnaryOp<(u32, u32)> for IsDepth {
    type Output = u32;

    fn apply(input: (u32, u32)) -> u32 {
        if input.0 == input.1 { 1u32 } else { 0u32 }
    }
}

struct PathContribution;

#[cubecl::cube]
impl UnaryOp<(u32, f32, u32)> for PathContribution {
    type Output = f32;

    fn apply(input: (u32, f32, u32)) -> f32 {
        if input.0 == input.2 { input.1 } else { 0.0f32 }
    }
}

struct DependencyContribution;

#[cubecl::cube]
impl UnaryOp<(f32, f32, f32, u32, u32)> for DependencyContribution {
    type Output = f32;

    fn apply(input: (f32, f32, f32, u32, u32)) -> f32 {
        let source_paths = input.0;
        let destination_paths = input.1;
        let destination_dependency = input.2;
        let source_distance = input.3;
        let destination_distance = input.4;
        if destination_distance == source_distance + 1u32 && destination_paths != 0.0f32 {
            source_paths / destination_paths * (1.0f32 + destination_dependency)
        } else {
            0.0f32
        }
    }
}

struct AccumulateCentrality;

#[cubecl::cube]
impl UnaryOp<(f32, f32, u32, u32)> for AccumulateCentrality {
    type Output = f32;

    fn apply(input: (f32, f32, u32, u32)) -> f32 {
        if input.2 == input.3 {
            input.0
        } else {
            input.0 + input.1
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

fn vertices_at_depth<R: Runtime>(
    exec: &Executor<R>,
    distance: &DeviceVec<R, u32>,
    depth: u32,
) -> common::Result<DeviceVec<R, u32>> {
    let n = u32::try_from(distance.capacity()).map_err(|_| massively::Error::LengthTooLarge {
        len: distance.capacity(),
    })?;
    let stencil = vector::transform(
        exec,
        zip2(distance.slice(..), lazy::constant(depth).take(n)),
        IsDepth,
    )?;
    common::materialize_exact(
        exec,
        vector::copy_where(
            exec,
            common::counting_u32(0, n as usize),
            common::stencil(stencil.slice(..)),
        )?,
    )
}

pub fn solve<R: Runtime>(
    exec: &Executor<R>,
    graph: &DeviceCsr<R>,
) -> common::Result<DeviceVec<R, f32>> {
    let n = graph.vertex_count();
    let centrality = common::filled(exec, n as usize, 0.0f32)?;

    for source in 0..n {
        let distance = bfs::solve(exec, graph, source)?;
        let paths = common::filled(exec, n as usize, 0.0f32)?;
        vector::scatter(
            exec,
            lazy::constant(1.0f32).take(1),
            common::indices(lazy::constant(source).take(1)),
            paths.slice_mut(..),
        )?;

        let mut max_depth = 0u32;
        let zero = exec.value(0.0f32)?;
        for depth in 0..n {
            let frontier = vertices_at_depth(exec, &distance, depth)?;
            if frontier.capacity() == 0 {
                break;
            }
            max_depth = depth;
            graph::traverse(
                exec,
                graph.csr(),
                frontier.slice(..),
                graph.edge_capacity()?,
            )?
            .map(
                zip3(
                    graph::destination(distance.slice(..)),
                    graph::source(paths.slice(..)),
                    graph::source(lazy::constant(depth + 1).take(n)),
                ),
                PathContribution,
            )
            .update_by_destination(exec, zero.clone(), SumF32, paths.slice_mut(..))?;
        }

        let dependency = common::filled(exec, n as usize, 0.0f32)?;
        for depth in (0..=max_depth).rev() {
            let frontier = vertices_at_depth(exec, &distance, depth)?;
            let values = graph::traverse(
                exec,
                graph.csr(),
                frontier.slice(..),
                graph.edge_capacity()?,
            )?
            .map(
                zip5(
                    graph::source(paths.slice(..)),
                    graph::destination(paths.slice(..)),
                    graph::destination(dependency.slice(..)),
                    graph::source(distance.slice(..)),
                    graph::destination(distance.slice(..)),
                ),
                DependencyContribution,
            )
            .reduce_by_source(exec, zero.clone(), SumF32)?;
            vector::scatter(
                exec,
                values.slice(..),
                common::indices(frontier.slice(..)),
                dependency.slice_mut(..),
            )?;
        }

        let next = vector::transform(
            exec,
            zip2(
                zip2(centrality.slice(..), dependency.slice(..)),
                zip2(
                    common::counting_u32(0, n as usize),
                    lazy::constant(source).take(n),
                ),
            ),
            AccumulateCentrality,
        )?;
        vector::scatter(
            exec,
            next.slice(..),
            common::indices(common::counting_u32(0, n as usize)),
            centrality.slice_mut(..),
        )?;
    }

    Ok(centrality)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn middle_vertices_dominate_a_path() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        let graph = DeviceCsr::from_host(&exec, &common::path_graph()).unwrap();
        let output = solve(&exec, &graph).unwrap();
        common::assert_near(&exec.to_host(&output).unwrap(), &[0.0, 4.0, 4.0, 0.0], 1e-5);
    }
}
