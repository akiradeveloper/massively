use cubecl::prelude::*;
use massively::{
    DeviceSlice, DeviceVec, Executor,
    graph::{self, Csr},
    op::ReductionOp,
    op::UnaryOp,
};

pub(crate) type Result<T> = std::result::Result<T, massively::Error>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CsrGraph {
    pub offsets: Vec<u32>,
    pub neighbors: Vec<u32>,
}

impl CsrGraph {
    pub fn new(offsets: Vec<u32>, neighbors: Vec<u32>) -> Self {
        assert!(
            !offsets.is_empty(),
            "CSR offsets must contain the initial zero"
        );
        assert_eq!(offsets[0], 0, "CSR offsets must start at zero");
        assert_eq!(offsets.last().copied().unwrap() as usize, neighbors.len());
        assert!(offsets.windows(2).all(|pair| pair[0] <= pair[1]));
        let vertices = offsets.len() - 1;
        assert!(neighbors.iter().all(|&vertex| (vertex as usize) < vertices));
        Self { offsets, neighbors }
    }

    pub fn vertex_count(&self) -> usize {
        self.offsets.len() - 1
    }

    pub fn row(&self, vertex: usize) -> &[u32] {
        &self.neighbors[self.offsets[vertex] as usize..self.offsets[vertex + 1] as usize]
    }

    pub(crate) fn edge_sources(&self) -> Vec<u32> {
        let mut sources = Vec::with_capacity(self.neighbors.len());
        for vertex in 0..self.vertex_count() {
            sources.extend(std::iter::repeat_n(vertex as u32, self.row(vertex).len()));
        }
        sources
    }
}

#[derive(Clone, Debug)]
pub struct WeightedCsr {
    pub graph: CsrGraph,
    pub weights: Vec<f32>,
}

impl WeightedCsr {
    pub fn new(graph: CsrGraph, weights: Vec<f32>) -> Self {
        assert_eq!(graph.neighbors.len(), weights.len());
        Self { graph, weights }
    }
}

pub(crate) struct DeviceGraph<R: Runtime> {
    destinations: DeviceVec<R, u32>,
    offsets: DeviceVec<R, u32>,
}

impl<R: Runtime> DeviceGraph<R> {
    pub(crate) fn new(exec: &Executor<R>, graph: &CsrGraph) -> Self {
        Self {
            destinations: exec.to_device(&graph.neighbors),
            offsets: exec.to_device(&graph.offsets),
        }
    }

    pub(crate) fn csr(&self) -> Csr<DeviceSlice<u32>, DeviceSlice<u32>> {
        Csr::new(self.destinations.slice(..), self.offsets.slice(..))
    }
}

struct SumU32;

#[cubecl::cube]
impl ReductionOp<u32> for SumU32 {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        lhs + rhs
    }
}

struct One;

#[cubecl::cube]
impl UnaryOp<u32> for One {
    type Output = u32;

    fn apply(_input: u32) -> u32 {
        1u32
    }
}

pub(crate) fn all_vertices<R: Runtime>(exec: &Executor<R>, graph: &CsrGraph) -> DeviceVec<R, u32> {
    exec.to_device(&(0..graph.vertex_count() as u32).collect::<Vec<_>>())
}

pub(crate) fn degrees<R: Runtime>(exec: &Executor<R>, graph: &CsrGraph) -> Result<Vec<u32>> {
    let device_graph = DeviceGraph::new(exec, graph);
    let frontier = all_vertices(exec, graph);
    let output = exec.alloc::<u32>(graph.vertex_count());
    graph::traverse(exec, device_graph.csr(), frontier.slice(..))?
        .map(graph::edge_id(), One)
        .reduce_by_source(exec, 0, SumU32, output.slice_mut(..))?;
    exec.to_host(&output)
}

#[cfg(test)]
pub(crate) fn sample_graph() -> CsrGraph {
    CsrGraph::new(vec![0, 2, 5, 8, 10], vec![1, 2, 0, 2, 3, 0, 1, 3, 1, 2])
}

#[cfg(test)]
pub(crate) fn path_graph() -> CsrGraph {
    CsrGraph::new(vec![0, 1, 3, 5, 6], vec![1, 0, 2, 1, 3, 2])
}

#[cfg(test)]
pub(crate) fn assert_near(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (*actual - *expected).abs() <= tolerance,
            "actual={actual}, expected={expected}"
        );
    }
}
