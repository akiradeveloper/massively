//! Kruskal minimum spanning tree with GPU edge sorting.

use cubecl::prelude::*;
use massively::{Executor, op::BinaryPredicateOp};

use super::common::{self, WeightedCsr};

struct LessF32;

#[cubecl::cube]
impl BinaryPredicateOp<f32> for LessF32 {
    fn apply(lhs: f32, rhs: f32) -> bool {
        lhs < rhs
    }
}

struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl DisjointSet {
    fn new(len: usize) -> Self {
        Self {
            parent: (0..len).collect(),
            rank: vec![0; len],
        }
    }

    fn find(&mut self, value: usize) -> usize {
        if self.parent[value] != value {
            self.parent[value] = self.find(self.parent[value]);
        }
        self.parent[value]
    }

    fn union(&mut self, lhs: usize, rhs: usize) -> bool {
        let mut lhs = self.find(lhs);
        let mut rhs = self.find(rhs);
        if lhs == rhs {
            return false;
        }
        if self.rank[lhs] < self.rank[rhs] {
            std::mem::swap(&mut lhs, &mut rhs);
        }
        self.parent[rhs] = lhs;
        if self.rank[lhs] == self.rank[rhs] {
            self.rank[lhs] += 1;
        }
        true
    }
}

pub fn solve<R: Runtime>(
    exec: &Executor<R>,
    graph: &WeightedCsr,
) -> common::Result<Vec<(u32, u32, f32)>> {
    let mut edges = Vec::new();
    for source in 0..graph.graph.vertex_count() {
        let start = graph.graph.offsets[source] as usize;
        let end = graph.graph.offsets[source + 1] as usize;
        for edge in start..end {
            let destination = graph.graph.neighbors[edge];
            if source as u32 <= destination {
                edges.push((source as u32, destination, graph.weights[edge]));
            }
        }
    }

    let weights = exec.to_device(&edges.iter().map(|edge| edge.2).collect::<Vec<_>>());
    let ids = exec.to_device(&(0..edges.len() as u32).collect::<Vec<_>>());
    let (_sorted_weights, sorted_ids) =
        massively::vector::sort_by_key(exec, weights.slice(..), ids.slice(..), LessF32)?;

    let sorted_ids = exec.to_host(&sorted_ids)?;
    let mut sets = DisjointSet::new(graph.graph.vertex_count());
    let mut tree = Vec::new();
    for id in sorted_ids {
        let edge = edges[id as usize];
        if sets.union(edge.0 as usize, edge.1 as usize) {
            tree.push(edge);
        }
    }
    Ok(tree)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn weighted_path_is_its_own_tree() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        let graph = WeightedCsr::new(common::path_graph(), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
        let tree = solve(&exec, &graph).unwrap();
        assert_eq!(tree.len(), 3);
        assert_eq!(tree.iter().map(|edge| edge.2).sum::<f32>(), 6.0);
    }
}
