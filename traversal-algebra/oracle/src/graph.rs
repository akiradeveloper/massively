//! Host-owned semantic twin of `massively::graph`.

#![allow(private_interfaces)]

use std::{fmt::Debug, marker::PhantomData, sync::Arc};

use crate::{Error, Result};

/// One traversed CSR edge and its public structural identifiers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EdgeContext {
    pub source: u32,
    pub destination: u32,
    pub edge: u32,
}

/// A host-owned compressed sparse row topology.
///
/// The argument order follows `massively::graph::Csr::new`: destinations
/// first, then offsets.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Csr {
    destinations: Arc<[u32]>,
    offsets: Arc<[u32]>,
}

impl Csr {
    pub fn new(destinations: impl Into<Arc<[u32]>>, offsets: impl Into<Arc<[u32]>>) -> Self {
        Self {
            destinations: destinations.into(),
            offsets: offsets.into(),
        }
    }

    pub fn destinations(&self) -> &[u32] {
        &self.destinations
    }

    pub fn offsets(&self) -> &[u32] {
        &self.offsets
    }

    pub fn vertex_count(&self) -> Result<usize> {
        self.validate()?;
        Ok(self.offsets.len() - 1)
    }

    pub fn edge_count(&self) -> Result<usize> {
        self.validate()?;
        Ok(self.destinations.len())
    }

    pub(crate) fn validate(&self) -> Result<()> {
        let Some(&first) = self.offsets.first() else {
            return Err(Error::InvalidInput(
                "CSR offsets must contain the initial zero".into(),
            ));
        };
        if first != 0 {
            return Err(Error::InvalidInput(
                "CSR offsets must begin with zero".into(),
            ));
        }
        if self.offsets.windows(2).any(|pair| pair[0] > pair[1]) {
            return Err(Error::InvalidInput(
                "CSR offsets must be nondecreasing".into(),
            ));
        }
        let expected_edges = u32::try_from(self.destinations.len())
            .map_err(|_| Error::InvalidInput("CSR destination count does not fit u32".into()))?;
        if self.offsets.last().copied() != Some(expected_edges) {
            return Err(Error::InvalidInput(format!(
                "final CSR offset {:?} does not equal destination count {}",
                self.offsets.last(),
                self.destinations.len()
            )));
        }
        let vertices = self.offsets.len() - 1;
        if self
            .destinations
            .iter()
            .any(|&destination| destination as usize >= vertices)
        {
            return Err(Error::InvalidInput(
                "CSR destination lies outside the vertex set".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
enum ScalarNode {
    SourceId,
    DestinationId,
    EdgeId,
    Source(Arc<[u32]>),
    Destination(Arc<[u32]>),
    Edge(Arc<[u32]>),
    Constant(u32),
    Add(Box<ScalarNode>, Box<ScalarNode>),
}

impl ScalarNode {
    fn validate(&self, vertices: usize, edges: usize) -> Result<()> {
        match self {
            Self::Source(values) | Self::Destination(values) if values.len() != vertices => {
                Err(Error::InvalidInput(format!(
                    "oracle vertex column length {} does not equal vertex count {vertices}",
                    values.len()
                )))
            }
            Self::Edge(values) if values.len() != edges => Err(Error::InvalidInput(format!(
                "oracle edge column length {} does not equal edge count {edges}",
                values.len()
            ))),
            Self::Add(left, right) => {
                left.validate(vertices, edges)?;
                right.validate(vertices, edges)
            }
            Self::SourceId
            | Self::DestinationId
            | Self::EdgeId
            | Self::Source(_)
            | Self::Destination(_)
            | Self::Edge(_)
            | Self::Constant(_) => Ok(()),
        }
    }

    fn evaluate(&self, context: EdgeContext) -> Result<u32> {
        match self {
            Self::SourceId => Ok(context.source),
            Self::DestinationId => Ok(context.destination),
            Self::EdgeId => Ok(context.edge),
            Self::Source(values) => Ok(values[context.source as usize]),
            Self::Destination(values) => Ok(values[context.destination as usize]),
            Self::Edge(values) => Ok(values[context.edge as usize]),
            Self::Constant(value) => Ok(*value),
            Self::Add(left, right) => left
                .evaluate(context)?
                .checked_add(right.evaluate(context)?)
                .ok_or(Error::ArithmeticOverflow),
        }
    }
}

/// A scalar edge-context expression.
#[derive(Clone, Debug)]
pub struct ScalarExpr(ScalarNode);

/// A private binary product expression whose public item schema is flat.
#[derive(Clone, Debug)]
pub struct Zip<Left, Right> {
    left: Left,
    right: Right,
}

pub fn zip2<Left, Right>(left: Left, right: Right) -> Zip<Left, Right> {
    Zip { left, right }
}

pub fn source_id() -> ScalarExpr {
    ScalarExpr(ScalarNode::SourceId)
}

pub fn destination_id() -> ScalarExpr {
    ScalarExpr(ScalarNode::DestinationId)
}

pub fn edge_id() -> ScalarExpr {
    ScalarExpr(ScalarNode::EdgeId)
}

pub fn source(values: impl Into<Arc<[u32]>>) -> ScalarExpr {
    ScalarExpr(ScalarNode::Source(values.into()))
}

pub fn destination(values: impl Into<Arc<[u32]>>) -> ScalarExpr {
    ScalarExpr(ScalarNode::Destination(values.into()))
}

pub fn edge(values: impl Into<Arc<[u32]>>) -> ScalarExpr {
    ScalarExpr(ScalarNode::Edge(values.into()))
}

mod private {
    pub trait Sealed {}
}

#[derive(Clone, Copy, Debug)]
#[doc(hidden)]
pub struct Last<T>(PhantomData<fn() -> T>);

#[derive(Clone, Copy, Debug)]
#[doc(hidden)]
pub struct More<Head, Tail>(PhantomData<fn() -> (Head, Tail)>);

#[doc(hidden)]
pub trait Concat<Rhs> {
    type Output;
}

impl<Head, Rhs> Concat<Rhs> for Last<Head> {
    type Output = More<Head, Rhs>;
}

impl<Head, Tail, Rhs> Concat<Rhs> for More<Head, Tail>
where
    Tail: Concat<Rhs>,
{
    type Output = More<Head, Tail::Output>;
}

#[doc(hidden)]
pub trait FlatLeaves {
    type Item: Clone + Debug + PartialEq + Eq;

    fn assemble(columns: Vec<Vec<u32>>) -> Result<Vec<Self::Item>>;
}

impl FlatLeaves for Last<u32> {
    type Item = u32;

    fn assemble(columns: Vec<Vec<u32>>) -> Result<Vec<Self::Item>> {
        let received = columns.len();
        let [column]: [Vec<u32>; 1] = columns.try_into().map_err(|_| {
            Error::Internal(format!(
                "scalar result expected one column, received {received}"
            ))
        })?;
        Ok(column)
    }
}

macro_rules! u32_for {
    ($column:ident) => {
        u32
    };
}

macro_rules! impl_flat_leaves {
    ($leaves:ty => ($first:ident, $( $column:ident ),+ $(,)?)) => {
        impl FlatLeaves for $leaves {
            type Item = (u32, $(u32_for!($column),)+);

            fn assemble(columns: Vec<Vec<u32>>) -> Result<Vec<Self::Item>> {
                let expected = 1usize $(+ { let _ = stringify!($column); 1usize })+;
                let received = columns.len();
                let [$first, $( $column, )+] = columns.try_into().map_err(|_| {
                    Error::Internal(format!(
                        "product result expected {expected} columns, received {received}"
                    ))
                })?;
                let len = $first.len();
                if $( $column.len() != len )||+ {
                    return Err(Error::Internal(
                        "product result columns have different lengths".into(),
                    ));
                }
                Ok((0..len)
                    .map(|index| ($first[index], $( $column[index], )+))
                    .collect())
            }
        }
    };
}

type Leaves2 = More<u32, Last<u32>>;
type Leaves3 = More<u32, Leaves2>;
type Leaves4 = More<u32, Leaves3>;
type Leaves5 = More<u32, Leaves4>;
type Leaves6 = More<u32, Leaves5>;
type Leaves7 = More<u32, Leaves6>;
type Leaves8 = More<u32, Leaves7>;
type Leaves9 = More<u32, Leaves8>;
type Leaves10 = More<u32, Leaves9>;
type Leaves11 = More<u32, Leaves10>;
type Leaves12 = More<u32, Leaves11>;

impl_flat_leaves!(Leaves2 => (c0, c1));
impl_flat_leaves!(Leaves3 => (c0, c1, c2));
impl_flat_leaves!(Leaves4 => (c0, c1, c2, c3));
impl_flat_leaves!(Leaves5 => (c0, c1, c2, c3, c4));
impl_flat_leaves!(Leaves6 => (c0, c1, c2, c3, c4, c5));
impl_flat_leaves!(Leaves7 => (c0, c1, c2, c3, c4, c5, c6));
impl_flat_leaves!(Leaves8 => (c0, c1, c2, c3, c4, c5, c6, c7));
impl_flat_leaves!(Leaves9 => (c0, c1, c2, c3, c4, c5, c6, c7, c8));
impl_flat_leaves!(Leaves10 => (c0, c1, c2, c3, c4, c5, c6, c7, c8, c9));
impl_flat_leaves!(Leaves11 => (c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10));
impl_flat_leaves!(Leaves12 => (c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11));

/// A typed CPU-oracle expression whose scalar leaves are evaluated together
/// and reassembled without terminal-by-arity implementations.
pub trait Expression: private::Sealed + Clone + Debug {
    type Item: Clone + Debug + PartialEq + Eq;
}

#[doc(hidden)]
pub trait ExpressionImpl: Expression {
    type Leaves: FlatLeaves<Item = Self::Item>;

    fn leaf_count(&self) -> usize;
    fn validate(&self, vertices: usize, edges: usize) -> Result<()>;
    fn evaluate(&self, context: EdgeContext, output: &mut [u32]) -> Result<()>;
    fn assemble(&self, columns: Vec<Vec<u32>>) -> Result<Vec<Self::Item>>;
}

impl private::Sealed for ScalarExpr {}

impl Expression for ScalarExpr {
    type Item = u32;
}

impl ExpressionImpl for ScalarExpr {
    type Leaves = Last<u32>;

    fn leaf_count(&self) -> usize {
        1
    }

    fn validate(&self, vertices: usize, edges: usize) -> Result<()> {
        self.0.validate(vertices, edges)
    }

    fn evaluate(&self, context: EdgeContext, output: &mut [u32]) -> Result<()> {
        let [value] = output else {
            return Err(Error::Internal(format!(
                "scalar expression expected one output leaf, received {}",
                output.len()
            )));
        };
        *value = self.0.evaluate(context)?;
        Ok(())
    }

    fn assemble(&self, columns: Vec<Vec<u32>>) -> Result<Vec<Self::Item>> {
        <Self::Leaves as FlatLeaves>::assemble(columns)
    }
}

impl<Left, Right> private::Sealed for Zip<Left, Right>
where
    Left: Expression,
    Right: Expression,
{
}

#[allow(private_bounds)]
impl<Left, Right> Expression for Zip<Left, Right>
where
    Left: ExpressionImpl,
    Right: ExpressionImpl,
    Left::Leaves: Concat<Right::Leaves>,
    <Left::Leaves as Concat<Right::Leaves>>::Output: FlatLeaves,
{
    type Item = <<Left::Leaves as Concat<Right::Leaves>>::Output as FlatLeaves>::Item;
}

impl<Left, Right> ExpressionImpl for Zip<Left, Right>
where
    Left: ExpressionImpl,
    Right: ExpressionImpl,
    Left::Leaves: Concat<Right::Leaves>,
    <Left::Leaves as Concat<Right::Leaves>>::Output: FlatLeaves,
{
    type Leaves = <Left::Leaves as Concat<Right::Leaves>>::Output;

    fn leaf_count(&self) -> usize {
        self.left.leaf_count() + self.right.leaf_count()
    }

    fn validate(&self, vertices: usize, edges: usize) -> Result<()> {
        self.left.validate(vertices, edges)?;
        self.right.validate(vertices, edges)
    }

    fn evaluate(&self, context: EdgeContext, output: &mut [u32]) -> Result<()> {
        let left_count = self.left.leaf_count();
        if output.len() != self.leaf_count() {
            return Err(Error::Internal(format!(
                "product expression expected {} output leaves, received {}",
                self.leaf_count(),
                output.len()
            )));
        }
        let (left, right) = output.split_at_mut(left_count);
        self.left.evaluate(context, left)?;
        self.right.evaluate(context, right)
    }

    fn assemble(&self, columns: Vec<Vec<u32>>) -> Result<Vec<Self::Item>> {
        if columns.len() != self.leaf_count() {
            return Err(Error::Internal(format!(
                "product result expected {} columns, received {}",
                self.leaf_count(),
                columns.len()
            )));
        }
        <Self::Leaves as FlatLeaves>::assemble(columns)
    }
}

pub mod op {
    /// Pointwise identity map.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct Identity;

    /// Maps every edge occurrence to one.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct One;

    /// Adds the two scalar leaves of a binary product.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct AddPair;

    /// Natural-number addition with identity zero.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct Add;
}

pub trait MapOp<Input> {
    type Output: Expression;

    fn apply(self, input: Input) -> Self::Output;
}

impl<Input> MapOp<Input> for op::Identity
where
    Input: Expression,
{
    type Output = Input;

    fn apply(self, input: Input) -> Self::Output {
        input
    }
}

impl<Input> MapOp<Input> for op::One
where
    Input: Expression,
{
    type Output = ScalarExpr;

    fn apply(self, _input: Input) -> Self::Output {
        ScalarExpr(ScalarNode::Constant(1))
    }
}

impl MapOp<Zip<ScalarExpr, ScalarExpr>> for op::AddPair {
    type Output = ScalarExpr;

    fn apply(self, input: Zip<ScalarExpr, ScalarExpr>) -> Self::Output {
        ScalarExpr(ScalarNode::Add(
            Box::new(input.left.0),
            Box::new(input.right.0),
        ))
    }
}

/// Selects the ordered outgoing edges of a frontier.
#[derive(Clone, Debug)]
pub struct Traversal {
    graph: Csr,
    frontier: Arc<[u32]>,
    edge_count: u32,
}

pub fn traverse(graph: Csr, frontier: impl Into<Arc<[u32]>>) -> Result<Traversal> {
    graph.validate()?;
    let frontier = frontier.into();
    let vertices = graph.offsets.len() - 1;
    let mut edge_count = 0_u32;
    for &source in frontier.iter() {
        if source as usize >= vertices {
            return Err(Error::InvalidInput(format!(
                "frontier source {source} lies outside {vertices} vertices"
            )));
        }
        let source = source as usize;
        edge_count = edge_count
            .checked_add(graph.offsets[source + 1] - graph.offsets[source])
            .ok_or_else(|| Error::InvalidInput("active edge count overflows u32".into()))?;
    }
    Ok(Traversal {
        graph,
        frontier,
        edge_count,
    })
}

impl Traversal {
    pub const fn edge_count(&self) -> u32 {
        self.edge_count
    }

    pub fn map<Input, Map>(self, expression: Input, map: Map) -> MappedTraversal<Map::Output>
    where
        Input: Expression,
        Map: MapOp<Input>,
    {
        MappedTraversal {
            traversal: self,
            expression: map.apply(expression),
        }
    }
}

#[derive(Clone, Debug)]
pub struct MappedTraversal<Expr> {
    traversal: Traversal,
    expression: Expr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Terminal {
    Emit,
    ReduceBySource,
    ReduceByDestination,
}

/// A fully constructed semantic observation.
#[derive(Clone, Debug)]
pub struct Query<Expr> {
    pub(crate) graph: Csr,
    pub(crate) frontier: Arc<[u32]>,
    pub(crate) expression: Expr,
    pub(crate) terminal: Terminal,
    _private: PhantomData<fn()>,
}

/// Typed public result shapes of the three proved traversal terminals.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Observation<Item> {
    Emitted(Vec<Item>),
    SourceReduced(Vec<Item>),
    DestinationReduced(Vec<Item>),
}

impl<Item> Observation<Item> {
    pub fn into_values(self) -> Vec<Item> {
        match self {
            Self::Emitted(values)
            | Self::SourceReduced(values)
            | Self::DestinationReduced(values) => values,
        }
    }
}

impl<Expr> MappedTraversal<Expr>
where
    Expr: Expression,
{
    pub fn emit(self) -> Query<Expr> {
        Query {
            graph: self.traversal.graph,
            frontier: self.traversal.frontier,
            expression: self.expression,
            terminal: Terminal::Emit,
            _private: PhantomData,
        }
    }
}

impl MappedTraversal<ScalarExpr> {
    pub fn reduce_by_source(self, init: u32, _reduce: op::Add) -> Result<Query<ScalarExpr>> {
        reduction_query(self, init, Terminal::ReduceBySource)
    }

    pub fn reduce_by_destination(self, init: u32, _reduce: op::Add) -> Result<Query<ScalarExpr>> {
        reduction_query(self, init, Terminal::ReduceByDestination)
    }
}

fn reduction_query(
    mapped: MappedTraversal<ScalarExpr>,
    init: u32,
    terminal: Terminal,
) -> Result<Query<ScalarExpr>> {
    if init != 0 {
        return Err(Error::InvalidInput(format!(
            "the proved natural-add terminal requires identity 0, received {init}"
        )));
    }
    Ok(Query {
        graph: mapped.traversal.graph,
        frontier: mapped.traversal.frontier,
        expression: mapped.expression,
        terminal,
        _private: PhantomData,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_and_counts_duplicate_frontier_occurrences() {
        let graph = Csr::new(vec![1, 0], vec![0, 1, 2]);
        let traversal = traverse(graph, vec![0, 0, 1]).unwrap();
        assert_eq!(traversal.edge_count(), 3);
    }

    #[test]
    fn nested_products_reassemble_as_flat_rows() {
        let expression = zip2(zip2(source_id(), destination_id()), edge_id());
        let rows = expression
            .assemble(vec![vec![0, 1], vec![1, 0], vec![4, 7]])
            .unwrap();
        assert_eq!(rows, vec![(0, 1, 4), (1, 0, 7)]);
    }
}
