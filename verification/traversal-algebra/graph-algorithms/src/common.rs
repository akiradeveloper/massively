use cubecl::prelude::*;
use massively::{
    DeviceSlice, DeviceVec, Executor, MIndex, MStorage,
    graph::{self, Csr},
    lazy,
    op::ReductionOp,
    op::UnaryOp,
    vector, zip2, zip3,
};

pub(crate) type Result<T> = std::result::Result<T, massively::Error>;

/// Verifies the exact owned length expected by these host-controlled
/// reference algorithms.
pub(crate) fn materialize_exact<R, Storage>(
    _exec: &Executor<R>,
    storage: Storage,
) -> Result<Storage>
where
    R: Runtime,
    Storage: MStorage<R>,
{
    storage.len()?;
    Ok(storage)
}

pub(crate) fn materialize_exact_pair<R, Left, Right>(
    _exec: &Executor<R>,
    (left, right): (Left, Right),
) -> Result<(Left, Right)>
where
    R: Runtime,
    Left: MStorage<R>,
    Right: MStorage<R>,
{
    let len = left.len()?;
    let right_len = right.len()?;
    if right_len != len {
        return Err(massively::Error::LengthMismatch {
            left: len as usize,
            right: right_len as usize,
        });
    }
    Ok((left, right))
}

pub(crate) fn counting_u32(start: usize, len: usize) -> lazy::Taken<lazy::Counting> {
    lazy::counting(u32::try_from(start).expect("counting value exceeds u32"))
        .take(u32::try_from(len).expect("counting length exceeds u32"))
}

pub(crate) fn indices<Input>(input: Input) -> Input {
    input
}

pub(crate) fn stencil<Input>(input: Input) -> lazy::Map<Input, massively::op::NonZero> {
    lazy::map(input, massively::op::NonZero)
}

pub(crate) trait FillValue<R: Runtime>: Sized {
    fn filled(exec: &Executor<R>, len: usize, value: Self) -> Result<DeviceVec<R, Self>>;
}

impl<R: Runtime> FillValue<R> for u32 {
    fn filled(exec: &Executor<R>, len: usize, value: Self) -> Result<DeviceVec<R, Self>> {
        let output = exec.alloc::<u32>(len);
        vector::fill(exec, value, output.slice_mut(..))?;
        Ok(output)
    }
}

impl<R: Runtime> FillValue<R> for f32 {
    fn filled(exec: &Executor<R>, len: usize, value: Self) -> Result<DeviceVec<R, Self>> {
        let output = exec.alloc::<f32>(len);
        vector::fill(exec, value, output.slice_mut(..))?;
        Ok(output)
    }
}

pub(crate) fn filled<R, T>(exec: &Executor<R>, len: usize, value: T) -> Result<DeviceVec<R, T>>
where
    R: Runtime,
    T: FillValue<R>,
{
    T::filled(exec, len, value)
}

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

/// An owned CSR topology resident in device memory.
pub struct DeviceCsr<R: Runtime> {
    destinations: DeviceVec<R, u32>,
    offsets: DeviceVec<R, u32>,
    vertex_count: u32,
}

impl<R: Runtime> Clone for DeviceCsr<R> {
    fn clone(&self) -> Self {
        Self {
            destinations: self.destinations.clone(),
            offsets: self.offsets.clone(),
            vertex_count: self.vertex_count,
        }
    }
}

impl<R: Runtime> DeviceCsr<R> {
    /// Creates a device CSR from already-resident storage without copying it.
    pub fn from_parts(destinations: DeviceVec<R, u32>, offsets: DeviceVec<R, u32>) -> Result<Self> {
        let vertex_count = offsets
            .len()
            .checked_sub(1)
            .ok_or(massively::Error::LengthMismatch { left: 1, right: 0 })?;
        Ok(Self {
            destinations,
            offsets,
            vertex_count,
        })
    }

    /// Explicitly uploads a host CSR topology.
    pub fn from_host(exec: &Executor<R>, graph: &CsrGraph) -> Result<Self> {
        Self::from_parts(
            exec.to_device(&graph.neighbors),
            exec.to_device(&graph.offsets),
        )
    }

    /// Returns the number of vertices.
    pub const fn vertex_count(&self) -> u32 {
        self.vertex_count
    }

    /// Returns the number of directed CSR entries.
    pub fn edge_count(&self) -> usize {
        self.destinations.len() as usize
    }

    /// Host-known physical bound for traversing a duplicate-free frontier.
    pub fn edge_capacity(&self) -> Result<MIndex> {
        MIndex::try_from(self.edge_count()).map_err(|_| massively::Error::LengthTooLarge {
            len: self.edge_count(),
        })
    }

    /// Safe physical traversal bound when the frontier may contain duplicate
    /// vertices and no tighter maximum-degree bound is available.
    pub fn repeated_edge_capacity(&self, source_count: MIndex) -> Result<MIndex> {
        let capacity = self
            .edge_count()
            .checked_mul(source_count as usize)
            .ok_or(massively::Error::LengthTooLarge { len: usize::MAX })?;
        MIndex::try_from(capacity).map_err(|_| massively::Error::LengthTooLarge { len: capacity })
    }

    /// Returns the resident destinations.
    pub const fn destinations(&self) -> &DeviceVec<R, u32> {
        &self.destinations
    }

    /// Returns the resident offsets.
    pub const fn offsets(&self) -> &DeviceVec<R, u32> {
        &self.offsets
    }

    /// Borrows the topology as a traversal input.
    pub fn csr(&self) -> Csr<DeviceSlice<u32>, DeviceSlice<u32>> {
        Csr::new(self.destinations.slice(..), self.offsets.slice(..))
    }
}

/// A floating-point weighted CSR matrix resident in device memory.
pub struct DeviceWeightedCsr<R: Runtime, Weight = f32> {
    graph: DeviceCsr<R>,
    weights: DeviceVec<R, Weight>,
}

impl<R: Runtime, Weight> DeviceWeightedCsr<R, Weight> {
    /// Creates a resident weighted CSR from already-resident storage.
    pub fn from_parts(graph: DeviceCsr<R>, weights: DeviceVec<R, Weight>) -> Result<Self> {
        if graph.edge_count() != weights.len() as usize {
            return Err(massively::Error::LengthMismatch {
                left: graph.edge_count(),
                right: weights.len() as usize,
            });
        }
        Ok(Self { graph, weights })
    }

    pub const fn graph(&self) -> &DeviceCsr<R> {
        &self.graph
    }

    pub const fn weights(&self) -> &DeviceVec<R, Weight> {
        &self.weights
    }
}

impl<R: Runtime> DeviceWeightedCsr<R, f32> {
    /// Explicitly uploads a host floating-point weighted CSR matrix.
    pub fn from_host(exec: &Executor<R>, matrix: &WeightedCsr) -> Result<Self> {
        Self::from_parts(
            DeviceCsr::from_host(exec, &matrix.graph)?,
            exec.to_device(&matrix.weights),
        )
    }
}

impl<R: Runtime> DeviceWeightedCsr<R, u32> {
    /// Explicitly uploads a host CSR topology and integer edge weights.
    pub fn from_host_parts(exec: &Executor<R>, graph: &CsrGraph, weights: &[u32]) -> Result<Self> {
        Self::from_parts(DeviceCsr::from_host(exec, graph)?, exec.to_device(weights))
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

struct DanglingRank;

#[cubecl::cube]
impl UnaryOp<(f32, u32)> for DanglingRank {
    type Output = f32;

    fn apply(input: (f32, u32)) -> f32 {
        if input.1 == 0u32 { input.0 } else { 0.0f32 }
    }
}

struct RankContribution;

#[cubecl::cube]
impl UnaryOp<(f32, u32, f32)> for RankContribution {
    type Output = f32;

    fn apply(input: (f32, u32, f32)) -> f32 {
        if input.1 == 0u32 {
            0.0f32
        } else {
            input.0 * input.2 / input.1 as f32
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

pub(crate) fn resident_degrees<R: Runtime>(
    exec: &Executor<R>,
    graph: &DeviceCsr<R>,
) -> Result<DeviceVec<R, u32>> {
    graph::traverse(
        exec,
        graph.csr(),
        counting_u32(0, graph.vertex_count() as usize),
        graph.edge_capacity()?,
    )?
    .map(graph::edge_id(), One)
    .reduce_by_source(exec, 0, SumU32)
}

pub(crate) fn dangling_mass<R: Runtime>(
    exec: &Executor<R>,
    rank: &DeviceVec<R, f32>,
    degree: &DeviceVec<R, u32>,
) -> Result<f32> {
    vector::reduce(
        exec,
        lazy::map(zip2(rank.slice(..), degree.slice(..)), DanglingRank),
        0.0,
        SumF32,
    )
}

pub(crate) fn accumulate_rank<R: Runtime>(
    exec: &Executor<R>,
    graph: &DeviceCsr<R>,
    degree: &DeviceVec<R, u32>,
    rank: &DeviceVec<R, f32>,
    damping: f32,
    output: &DeviceVec<R, f32>,
) -> Result<()> {
    graph::traverse(
        exec,
        graph.csr(),
        counting_u32(0, graph.vertex_count() as usize),
        graph.edge_capacity()?,
    )?
    .map(
        zip3(
            graph::source(rank.slice(..)),
            graph::source(degree.slice(..)),
            graph::source(lazy::constant(damping).take(graph.vertex_count())),
        ),
        RankContribution,
    )
    .update_by_destination(exec, 0.0, SumF32, output.slice_mut(..))
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
