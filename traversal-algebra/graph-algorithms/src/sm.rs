//! Exact unlabelled subgraph matching for small query graphs.
//!
//! The algorithm enumerates injective query-to-data vertex assignments in
//! device memory and filters them by every directed query edge.  Extra edges
//! in the data graph are allowed, so this is non-induced subgraph isomorphism.
//! Its `O(|V_data|^|V_query|)` candidate space deliberately targets small
//! query graphs, as does the corresponding Gunrock application.

use cubecl::prelude::*;
use massively::{
    DeviceVec, Executor, MStorage, graph, lazy,
    op::{BinaryPredicateOp, Identity, UnaryOp},
    vector, zip2, zip3,
};

use super::common::{self, CsrGraph, DeviceCsr};

pub struct Matches<R: Runtime> {
    mappings: DeviceVec<R, u32>,
    match_count: u32,
    query_vertex_count: u32,
}

impl<R: Runtime> Matches<R> {
    /// Flattened mappings: `mappings[match * query_vertex_count + query_vertex]`.
    pub const fn mappings(&self) -> &DeviceVec<R, u32> {
        &self.mappings
    }

    pub const fn match_count(&self) -> u32 {
        self.match_count
    }

    pub const fn query_vertex_count(&self) -> u32 {
        self.query_vertex_count
    }
}

#[cubecl::cube]
fn assignment_digit(code: u32, base: u32, position: u32) -> u32 {
    let value = RuntimeCell::<u32>::new(code);
    let index = RuntimeCell::<u32>::new(0u32);
    while index.read() < position {
        value.store(value.read() / base);
        index.store(index.read() + 1u32);
    }
    value.read() % base
}

struct Injective;

#[cubecl::cube]
impl UnaryOp<((u32, u32), u32)> for Injective {
    type Output = u32;

    fn apply(input: ((u32, u32), u32)) -> u32 {
        let left = RuntimeCell::<u32>::new(0u32);
        let valid = RuntimeCell::<u32>::new(1u32);
        while left.read() < input.1 {
            let right = RuntimeCell::<u32>::new(left.read() + 1u32);
            while right.read() < input.1 {
                if assignment_digit(input.0.0, input.0.1, left.read())
                    == assignment_digit(input.0.0, input.0.1, right.read())
                {
                    valid.store(0u32);
                }
                right.store(right.read() + 1u32);
            }
            left.store(left.read() + 1u32);
        }
        valid.read()
    }
}

struct DecodePair;

#[cubecl::cube]
impl UnaryOp<(((u32, u32), u32), u32)> for DecodePair {
    type Output = (u32, u32);

    fn apply(input: (((u32, u32), u32), u32)) -> (u32, u32) {
        (
            assignment_digit(input.0.0.0, input.0.0.1, input.0.1),
            assignment_digit(input.0.0.0, input.0.0.1, input.1),
        )
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
impl UnaryOp<((u32, u32), (u32, u32))> for PairEqual {
    type Output = u32;

    fn apply(input: ((u32, u32), (u32, u32))) -> u32 {
        if input.0.0 == input.1.0 && input.0.1 == input.1.1 {
            1u32
        } else {
            0u32
        }
    }
}

struct Both;

#[cubecl::cube]
impl UnaryOp<(u32, u32)> for Both {
    type Output = u32;

    fn apply(input: (u32, u32)) -> u32 {
        if input.0 != 0u32 && input.1 != 0u32 {
            1u32
        } else {
            0u32
        }
    }
}

struct Divide;

#[cubecl::cube]
impl UnaryOp<(u32, u32)> for Divide {
    type Output = u32;

    fn apply(input: (u32, u32)) -> u32 {
        input.0 / input.1
    }
}

struct Modulo;

#[cubecl::cube]
impl UnaryOp<(u32, u32)> for Modulo {
    type Output = u32;

    fn apply(input: (u32, u32)) -> u32 {
        input.0 % input.1
    }
}

struct Decode;

#[cubecl::cube]
impl UnaryOp<((u32, u32), u32)> for Decode {
    type Output = u32;

    fn apply(input: ((u32, u32), u32)) -> u32 {
        assignment_digit(input.0.0, input.0.1, input.1)
    }
}

pub fn solve<R: Runtime>(
    exec: &Executor<R>,
    data: &DeviceCsr<R>,
    query: &CsrGraph,
) -> common::Result<Matches<R>> {
    let n = data.vertex_count();
    let k = u32::try_from(query.vertex_count()).map_err(|_| massively::Error::LengthTooLarge {
        len: query.vertex_count(),
    })?;
    assert!(n != 0);
    assert!(k != 0);
    if k > n {
        return Ok(Matches {
            mappings: common::filled(exec, 0, 0u32)?,
            match_count: 0,
            query_vertex_count: k,
        });
    }
    let candidate_count = n
        .checked_pow(k)
        .ok_or(massively::Error::LengthTooLarge { len: usize::MAX })?;

    let mut stencil = vector::transform(
        exec,
        zip3(
            common::counting_u32(0, candidate_count as usize),
            lazy::constant(n).take(candidate_count as usize),
            lazy::constant(k).take(candidate_count as usize),
        ),
        Injective,
    )?;

    let edge_pairs = graph::traverse(exec, data.csr(), common::counting_u32(0, n as usize))?
        .map(zip2(graph::source_id(), graph::destination_id()), Identity)
        .emit(exec)?;
    let sorted_pairs = vector::sort(exec, edge_pairs.slice(..), PairLess)?;
    let edge_count = sorted_pairs.len()?;
    let searchable_len =
        (edge_count as usize)
            .checked_add(1)
            .ok_or(massively::Error::LengthTooLarge {
                len: edge_count as usize,
            })?;
    let searchable = exec.alloc::<(u32, u32)>(searchable_len);
    vector::scatter(
        exec,
        sorted_pairs.slice(..),
        common::indices(common::counting_u32(0, edge_count as usize)),
        searchable.slice_mut(..),
    )?;
    vector::scatter(
        exec,
        zip2(
            lazy::constant(u32::MAX).take(1),
            lazy::constant(u32::MAX).take(1),
        ),
        lazy::constant(edge_count).take(1),
        searchable.slice_mut(..),
    )?;

    for query_source in 0..query.vertex_count() {
        for &query_destination in query.row(query_source) {
            let candidates = vector::transform(
                exec,
                massively::zip4(
                    common::counting_u32(0, candidate_count as usize),
                    lazy::constant(n).take(candidate_count as usize),
                    lazy::constant(query_source as u32).take(candidate_count as usize),
                    lazy::constant(query_destination).take(candidate_count as usize),
                ),
                DecodePair,
            )?;
            let locations =
                vector::lower_bound(exec, searchable.slice(..), candidates.slice(..), PairLess)?;
            let found = vector::gather(
                exec,
                searchable.slice(..),
                common::indices(locations.slice(..)),
            )?;
            let present = vector::transform(
                exec,
                zip2(
                    zip2(candidates.0.slice(..), candidates.1.slice(..)),
                    zip2(found.0.slice(..), found.1.slice(..)),
                ),
                PairEqual,
            )?;
            stencil = vector::transform(exec, zip2(stencil.slice(..), present.slice(..)), Both)?;
        }
    }

    let codes = vector::copy_where(
        exec,
        common::counting_u32(0, candidate_count as usize),
        common::stencil(stencil.slice(..)),
    )?;
    let match_count = u32::try_from(codes.len())
        .map_err(|_| massively::Error::LengthTooLarge { len: codes.len() })?;
    let mapping_count = match_count
        .checked_mul(k)
        .ok_or(massively::Error::LengthTooLarge { len: usize::MAX })?;
    let code_indices = vector::transform(
        exec,
        zip2(
            common::counting_u32(0, mapping_count as usize),
            lazy::constant(k).take(mapping_count as usize),
        ),
        Divide,
    )?;
    let positions = vector::transform(
        exec,
        zip2(
            common::counting_u32(0, mapping_count as usize),
            lazy::constant(k).take(mapping_count as usize),
        ),
        Modulo,
    )?;
    let repeated_codes = vector::gather(
        exec,
        codes.slice(..),
        common::indices(code_indices.slice(..)),
    )?;
    let mappings = vector::transform(
        exec,
        zip3(
            repeated_codes.slice(..),
            lazy::constant(n).take(mapping_count as usize),
            positions.slice(..),
        ),
        Decode,
    )?;

    Ok(Matches {
        mappings,
        match_count,
        query_vertex_count: k,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn finds_every_ordered_triangle_embedding() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
        let data = DeviceCsr::from_host(&exec, &common::sample_graph()).unwrap();
        let triangle = CsrGraph::new(vec![0, 2, 4, 6], vec![1, 2, 0, 2, 0, 1]);
        let matches = solve(&exec, &data, &triangle).unwrap();
        assert_eq!(matches.query_vertex_count(), 3);
        assert_eq!(matches.match_count(), 12);
        assert_eq!(matches.mappings().len(), 36);
    }
}
