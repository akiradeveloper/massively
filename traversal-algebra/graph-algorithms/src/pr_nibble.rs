//! Personalized-PageRank sweep clustering (PR-Nibble).
//!
//! Vertices are ordered by `ppr(v) / degree(v)`.  Every prefix is evaluated
//! by its undirected conductance, and the minimum-conductance nonempty prefix
//! is returned.  The input must use a symmetric, simple CSR representation.

use cubecl::prelude::*;
use massively::{
    DeviceVec, Executor, graph, lazy,
    op::{BinaryPredicateOp, ReductionOp, UnaryOp},
    vector, zip2, zip3,
};

use super::{
    common::{self, DeviceCsr},
    ppr,
};

struct RankPerDegree;

#[cubecl::cube]
impl UnaryOp<(f32, u32)> for RankPerDegree {
    type Output = f32;

    fn apply(input: (f32, u32)) -> f32 {
        if input.1 == 0u32 {
            -1.0f32
        } else {
            input.0 / input.1 as f32
        }
    }
}

struct SweepOrder;

#[cubecl::cube]
impl BinaryPredicateOp<(f32, u32)> for SweepOrder {
    fn apply(lhs: (f32, u32), rhs: (f32, u32)) -> bool {
        if lhs.0 != rhs.0 {
            lhs.0 > rhs.0
        } else {
            lhs.1 < rhs.1
        }
    }
}

struct EarlierNeighbor;

#[cubecl::cube]
impl UnaryOp<(u32, u32)> for EarlierNeighbor {
    type Output = u32;

    fn apply(input: (u32, u32)) -> u32 {
        if input.1 < input.0 { 1u32 } else { 0u32 }
    }
}

struct SumU32;

#[cubecl::cube]
impl ReductionOp<u32> for SumU32 {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        lhs + rhs
    }
}

struct CutDelta;

#[cubecl::cube]
impl UnaryOp<(u32, u32)> for CutDelta {
    type Output = f32;

    fn apply(input: (u32, u32)) -> f32 {
        input.0 as f32 - 2.0f32 * input.1 as f32
    }
}

struct DegreeAsF32;

#[cubecl::cube]
impl UnaryOp<u32> for DegreeAsF32 {
    type Output = f32;

    fn apply(input: u32) -> f32 {
        input as f32
    }
}

struct SumF32;

#[cubecl::cube]
impl ReductionOp<f32> for SumF32 {
    fn apply(lhs: f32, rhs: f32) -> f32 {
        lhs + rhs
    }
}

struct Conductance;

#[cubecl::cube]
impl UnaryOp<((f32, f32), f32)> for Conductance {
    type Output = f32;

    fn apply(input: ((f32, f32), f32)) -> f32 {
        let complement = input.1 - input.0.1;
        let denominator = f32::min(input.0.1, complement);
        if denominator > 0.0f32 {
            input.0.0 / denominator
        } else {
            f32::MAX
        }
    }
}

struct BestPrefix;

#[cubecl::cube]
impl BinaryPredicateOp<(f32, u32)> for BestPrefix {
    fn apply(lhs: (f32, u32), rhs: (f32, u32)) -> bool {
        if lhs.0 != rhs.0 {
            lhs.0 < rhs.0
        } else {
            lhs.1 < rhs.1
        }
    }
}

pub fn solve<R: Runtime>(
    exec: &Executor<R>,
    graph: &DeviceCsr<R>,
    source: u32,
    damping: f32,
    iterations: usize,
) -> common::Result<DeviceVec<R, u32>> {
    let n = graph.vertex_count();
    assert!(n != 0);
    assert!(source < n);

    let degree = common::resident_degrees(exec, graph)?;
    let source_degree = exec.to_host(&degree.slice(source as usize..source as usize + 1))?[0];
    if source_degree == 0 {
        return common::filled(exec, 1, source);
    }

    let rank = ppr::solve(exec, graph, source, damping, iterations)?;
    let score = vector::transform(exec, zip2(rank.slice(..), degree.slice(..)), RankPerDegree)?;
    let order = vector::sort_by_key(
        exec,
        zip2(score.slice(..), common::counting_u32(0, n as usize)),
        common::counting_u32(0, n as usize),
        SweepOrder,
    )?;

    let positions = common::filled(exec, n as usize, 0u32)?;
    vector::scatter(
        exec,
        common::counting_u32(0, n as usize),
        common::indices(order.slice(..)),
        positions.slice_mut(..),
    )?;

    let earlier = graph::traverse(exec, graph.csr(), common::counting_u32(0, n as usize))?
        .map(
            zip2(
                graph::source(positions.slice(..)),
                graph::destination(positions.slice(..)),
            ),
            EarlierNeighbor,
        )
        .reduce_by_source(exec, 0, SumU32)?;
    let ordered_degree = vector::gather(exec, degree.slice(..), common::indices(order.slice(..)))?;
    let ordered_earlier =
        vector::gather(exec, earlier.slice(..), common::indices(order.slice(..)))?;
    let cut_delta = vector::transform(
        exec,
        zip2(ordered_degree.slice(..), ordered_earlier.slice(..)),
        CutDelta,
    )?;
    let cut = vector::inclusive_scan(exec, cut_delta.slice(..), SumF32)?;
    let ordered_volume = vector::transform(exec, ordered_degree.slice(..), DegreeAsF32)?;
    let volume = vector::inclusive_scan(exec, ordered_volume.slice(..), SumF32)?;
    let total_volume = graph.edge_count() as f32;
    let conductance = vector::transform(
        exec,
        zip3(
            cut.slice(..),
            volume.slice(..),
            lazy::constant(total_volume).take(n as usize),
        ),
        Conductance,
    )?;
    let best = vector::min_element(
        exec,
        zip2(conductance.slice(..), common::counting_u32(0, n as usize)),
        BestPrefix,
    )?
    .expect("a nonempty graph has a nonempty sweep order");

    vector::transform(
        exec,
        order.slice(..best as usize + 1),
        massively::op::Identity,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CsrGraph;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn separates_two_dense_groups_at_their_bridge() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        let host = CsrGraph::new(
            vec![0, 2, 4, 7, 10, 12, 14],
            vec![1, 2, 0, 2, 0, 1, 3, 2, 4, 5, 3, 5, 3, 4],
        );
        let graph = DeviceCsr::from_host(&exec, &host).unwrap();
        let cluster = solve(&exec, &graph, 0, 0.85, 30).unwrap();
        let mut cluster = exec.to_host(&cluster).unwrap();
        cluster.sort_unstable();
        assert_eq!(cluster, vec![0, 1, 2]);
    }
}
