//! Deterministic multilevel Louvain community detection.
//!
//! Each level performs sequential local moves using the standard modularity-gain
//! ordering for an undirected weighted graph, then contracts communities by a
//! traversal, sort, and reduction.  The public unweighted input is interpreted
//! as a symmetric CSR graph with unit edge weights and no self-loops.

use cubecl::prelude::*;
use massively::{
    DeviceVec, Executor, graph, lazy,
    op::{BinaryPredicateOp, Identity, ReductionOp, UnaryOp},
    vector, zip2,
};

use super::common::{self, DeviceCsr, DeviceWeightedCsr};

struct SumU32;

#[cubecl::cube]
impl ReductionOp<u32> for SumU32 {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        lhs + rhs
    }
}

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

struct Different;

#[cubecl::cube]
impl UnaryOp<(u32, u32)> for Different {
    type Output = u32;

    fn apply(input: (u32, u32)) -> u32 {
        if input.0 != input.1 { 1u32 } else { 0u32 }
    }
}

struct GainScore;

#[cubecl::cube]
impl UnaryOp<(((u32, u32), u32), u32)> for GainScore {
    type Output = f32;

    fn apply(input: (((u32, u32), u32), u32)) -> f32 {
        input.0.0.0 as f32 - input.0.0.1 as f32 * input.0.1 as f32 / input.1 as f32
    }
}

struct CandidateLess;

#[cubecl::cube]
impl BinaryPredicateOp<(f32, u32)> for CandidateLess {
    fn apply(lhs: (f32, u32), rhs: (f32, u32)) -> bool {
        if lhs.0 != rhs.0 {
            lhs.0 < rhs.0
        } else {
            // For equal gain, the smaller community is the maximum item.
            lhs.1 > rhs.1
        }
    }
}

struct PairLess;

#[cubecl::cube]
impl BinaryPredicateOp<(u32, u32)> for PairLess {
    fn apply(lhs: (u32, u32), rhs: (u32, u32)) -> bool {
        if lhs.0 != rhs.0 {
            lhs.0 < rhs.0
        } else {
            lhs.1 < rhs.1
        }
    }
}

struct PairEqual;

#[cubecl::cube]
impl BinaryPredicateOp<(u32, u32)> for PairEqual {
    fn apply(lhs: (u32, u32), rhs: (u32, u32)) -> bool {
        lhs.0 == rhs.0 && lhs.1 == rhs.1
    }
}

fn read_u32<R: Runtime>(
    exec: &Executor<R>,
    values: &DeviceVec<R, u32>,
    index: usize,
) -> common::Result<u32> {
    Ok(exec.to_host(&values.slice(index..index + 1))?[0])
}

fn read_f32<R: Runtime>(
    exec: &Executor<R>,
    values: &DeviceVec<R, f32>,
    index: usize,
) -> common::Result<f32> {
    Ok(exec.to_host(&values.slice(index..index + 1))?[0])
}

fn strengths<R: Runtime>(
    exec: &Executor<R>,
    graph: &DeviceWeightedCsr<R, u32>,
) -> common::Result<DeviceVec<R, u32>> {
    graph::traverse(
        exec,
        graph.graph().csr(),
        lazy::counting(0).take(graph.graph().vertex_count()),
    )?
    .map(graph::edge(graph.weights().slice(..)), Identity)
    .reduce_by_source(exec, 0, SumU32)
}

fn local_move<R: Runtime>(
    exec: &Executor<R>,
    graph: &DeviceWeightedCsr<R, u32>,
    max_passes: usize,
) -> common::Result<DeviceVec<R, u32>> {
    let topology = graph.graph();
    let n = topology.vertex_count();
    let m2 = vector::reduce(exec, graph.weights().slice(..), 0, SumU32)?;
    let strength = strengths(exec, graph)?;
    let communities = vector::transform(exec, lazy::counting(0).take(n), Identity)?;
    let totals = vector::transform(exec, strength.slice(..), Identity)?;
    if m2 == 0 {
        return Ok(communities);
    }

    for _ in 0..max_passes {
        let mut moved = false;
        for vertex in 0..n {
            let vertex_index = vertex as usize;
            let current = read_u32(exec, &communities, vertex_index)?;
            let vertex_strength = read_u32(exec, &strength, vertex_index)?;
            let current_total = read_u32(exec, &totals, current as usize)?;
            vector::scatter(
                exec,
                lazy::constant(current_total - vertex_strength).take(1),
                lazy::constant(current).take(1),
                totals.slice_mut(..),
            )?;

            let bounds = exec.to_host(&topology.offsets().slice(vertex_index..vertex_index + 2))?;
            let destinations = topology
                .destinations()
                .slice(bounds[0] as usize..bounds[1] as usize);
            let edge_weights = graph
                .weights()
                .slice(bounds[0] as usize..bounds[1] as usize);
            let row_len = bounds[1] - bounds[0];
            let neighbor_communities =
                vector::gather(exec, communities.slice(..), destinations.clone())?;
            let non_self = vector::transform(
                exec,
                zip2(destinations, lazy::constant(vertex).take(row_len)),
                Different,
            )?;
            let neighbor_communities =
                vector::copy_where(exec, neighbor_communities.slice(..), non_self.slice(..))?;
            let neighbor_weights = vector::copy_where(exec, edge_weights, non_self.slice(..))?;
            let neighbor_count = u32::try_from(neighbor_communities.len()).map_err(|_| {
                massively::Error::LengthTooLarge {
                    len: neighbor_communities.len(),
                }
            })?;

            // The current community is always an option, even when the vertex
            // currently has no edge back into it.
            let candidate_capacity = neighbor_communities.len() + 1;
            let candidate_communities = exec.alloc::<u32>(candidate_capacity);
            let candidate_weights = exec.alloc::<u32>(candidate_capacity);
            vector::scatter(
                exec,
                neighbor_communities.slice(..),
                lazy::counting(0).take(neighbor_count),
                candidate_communities.slice_mut(..),
            )?;
            vector::scatter(
                exec,
                neighbor_weights.slice(..),
                lazy::counting(0).take(neighbor_count),
                candidate_weights.slice_mut(..),
            )?;
            vector::scatter(
                exec,
                lazy::constant(current).take(1),
                lazy::constant(neighbor_count).take(1),
                candidate_communities.slice_mut(..),
            )?;
            vector::scatter(
                exec,
                lazy::constant(0u32).take(1),
                lazy::constant(neighbor_count).take(1),
                candidate_weights.slice_mut(..),
            )?;
            let (sorted_communities, sorted_weights) = vector::sort_by_key(
                exec,
                candidate_communities.slice(..),
                candidate_weights.slice(..),
                LessU32,
            )?;
            let (unique_communities, incident_weights) = vector::reduce_by_key(
                exec,
                sorted_communities.slice(..),
                sorted_weights.slice(..),
                EqualU32,
                0,
                SumU32,
            )?;
            let candidate_totals =
                vector::gather(exec, totals.slice(..), unique_communities.slice(..))?;
            let candidate_count = u32::try_from(unique_communities.len()).map_err(|_| {
                massively::Error::LengthTooLarge {
                    len: unique_communities.len(),
                }
            })?;
            let scores = vector::transform(
                exec,
                massively::zip4(
                    incident_weights.slice(..),
                    lazy::constant(vertex_strength).take(candidate_count),
                    candidate_totals.slice(..),
                    lazy::constant(m2).take(candidate_count),
                ),
                GainScore,
            )?;
            let best_index = vector::max_element(
                exec,
                zip2(scores.slice(..), unique_communities.slice(..)),
                CandidateLess,
            )?
            .expect("the current community is always a candidate");
            let current_location = vector::lower_bound(
                exec,
                unique_communities.slice(..),
                lazy::constant(current).take(1),
                LessU32,
            )?;
            let current_index = read_u32(exec, &current_location, 0)? as usize;
            let best = read_u32(exec, &unique_communities, best_index as usize)?;
            let best_score = read_f32(exec, &scores, best_index as usize)?;
            let current_score = read_f32(exec, &scores, current_index)?;

            if best != current && best_score > current_score + 1.0e-6 {
                let best_total = read_u32(exec, &totals, best as usize)?;
                vector::scatter(
                    exec,
                    lazy::constant(best_total + vertex_strength).take(1),
                    lazy::constant(best).take(1),
                    totals.slice_mut(..),
                )?;
                vector::scatter(
                    exec,
                    lazy::constant(best).take(1),
                    lazy::constant(vertex).take(1),
                    communities.slice_mut(..),
                )?;
                moved = true;
            } else {
                vector::scatter(
                    exec,
                    lazy::constant(current_total).take(1),
                    lazy::constant(current).take(1),
                    totals.slice_mut(..),
                )?;
            }
        }
        if !moved {
            break;
        }
    }

    Ok(communities)
}

fn compact<R: Runtime>(
    exec: &Executor<R>,
    communities: &DeviceVec<R, u32>,
) -> common::Result<(DeviceVec<R, u32>, u32)> {
    let sorted = vector::sort(exec, communities.slice(..), LessU32)?;
    let unique = vector::unique(exec, sorted.slice(..), EqualU32)?;
    let count = u32::try_from(unique.len())
        .map_err(|_| massively::Error::LengthTooLarge { len: unique.len() })?;
    let labels = vector::lower_bound(exec, unique.slice(..), communities.slice(..), LessU32)?;
    Ok((labels, count))
}

fn contract<R: Runtime>(
    exec: &Executor<R>,
    graph: &DeviceWeightedCsr<R, u32>,
    labels: &DeviceVec<R, u32>,
    community_count: u32,
) -> common::Result<DeviceWeightedCsr<R, u32>> {
    let topology = graph.graph();
    let sources = graph::traverse(
        exec,
        topology.csr(),
        lazy::counting(0).take(topology.vertex_count()),
    )?
    .map(graph::source_id(), Identity)
    .emit(exec)?;
    let source_labels = vector::gather(exec, labels.slice(..), sources.slice(..))?;
    let destination_labels =
        vector::gather(exec, labels.slice(..), topology.destinations().slice(..))?;
    let (sorted_pairs, sorted_weights) = vector::sort_by_key(
        exec,
        zip2(source_labels.slice(..), destination_labels.slice(..)),
        graph.weights().slice(..),
        PairLess,
    )?;
    let (pairs, weights) = vector::reduce_by_key(
        exec,
        zip2(sorted_pairs.0.slice(..), sorted_pairs.1.slice(..)),
        sorted_weights.slice(..),
        PairEqual,
        0,
        SumU32,
    )?;
    let edge_count = weights.len() as u32;
    let (row_ids, row_counts) = vector::reduce_by_key(
        exec,
        pairs.0.slice(..),
        lazy::constant(1u32).take(edge_count),
        EqualU32,
        0,
        SumU32,
    )?;
    let counts = vector::fill(exec, community_count as usize, 0u32)?;
    vector::scatter(
        exec,
        row_counts.slice(..),
        row_ids.slice(..),
        counts.slice_mut(..),
    )?;
    let ends = vector::inclusive_scan(exec, counts.slice(..), SumU32)?;
    let offsets = vector::fill(exec, community_count as usize + 1, 0u32)?;
    vector::scatter(
        exec,
        ends.slice(..),
        lazy::counting(1).take(community_count),
        offsets.slice_mut(..),
    )?;
    let topology = DeviceCsr::from_parts(pairs.1, offsets)?;
    DeviceWeightedCsr::from_parts(topology, weights)
}

pub fn solve<R: Runtime>(
    exec: &Executor<R>,
    graph: &DeviceCsr<R>,
    max_passes: usize,
    max_levels: usize,
) -> common::Result<DeviceVec<R, u32>> {
    let n = graph.vertex_count();
    assert!(n != 0);
    let weights = vector::fill(exec, graph.edge_count(), 1u32)?;
    let mut level_graph = DeviceWeightedCsr::from_parts(graph.clone(), weights)?;
    let mut assignment = vector::transform(exec, lazy::counting(0).take(n), Identity)?;

    for _ in 0..max_levels {
        let communities = local_move(exec, &level_graph, max_passes)?;
        let (labels, community_count) = compact(exec, &communities)?;
        assignment = vector::gather(exec, labels.slice(..), assignment.slice(..))?;
        if community_count == level_graph.graph().vertex_count() {
            break;
        }
        level_graph = contract(exec, &level_graph, &labels, community_count)?;
    }

    Ok(assignment)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CsrGraph;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn separates_two_triangles_joined_by_one_bridge() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        let host = CsrGraph::new(
            vec![0, 2, 4, 7, 10, 12, 14],
            vec![1, 2, 0, 2, 0, 1, 3, 2, 4, 5, 3, 5, 3, 4],
        );
        let graph = DeviceCsr::from_host(&exec, &host).unwrap();
        let communities = solve(&exec, &graph, 20, 10).unwrap();
        assert_eq!(exec.to_host(&communities).unwrap(), vec![0, 0, 0, 1, 1, 1]);
    }
}
