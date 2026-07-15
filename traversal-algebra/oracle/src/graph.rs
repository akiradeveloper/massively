//! Host-owned semantic twin of `massively::graph`.

use std::{fmt::Debug, marker::PhantomData, sync::Arc};

use crate::{Error, Result, protocol::EncodedExpr};

/// One traversed CSR edge and its public structural identifiers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EdgeContext {
    pub source: u32,
    pub destination: u32,
    pub edge: u32,
}

/// A committed Lean-generated regression case.
#[derive(Clone, Copy, Debug)]
pub struct OracleCase {
    pub name: &'static str,
    pub offsets: &'static [u32],
    pub destinations: &'static [u32],
    pub frontier: &'static [u32],
    pub expected_edges: &'static [EdgeContext],
    pub expected_source_counts: &'static [u32],
    pub expected_destination_counts: &'static [u32],
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

/// A scalar edge-context expression.
#[derive(Clone, Debug)]
pub struct ScalarExpr(ScalarNode);

/// A recursively nested product expression.  This is the sole structural
/// representation used for every arity.
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

/// A typed oracle expression whose scalar leaves can be evaluated by Lean and
/// reassembled without an arity-specific implementation.
pub trait Expression: private::Sealed + Clone + Debug {
    type Item: Clone + Debug + PartialEq + Eq;
}

pub(crate) trait ExpressionImpl: Expression {
    fn encode_leaves(&self, output: &mut Vec<EncodedExpr>);
    fn leaf_count(&self) -> usize;
    fn assemble(&self, columns: &[Vec<u32>]) -> Result<Vec<Self::Item>>;
}

impl private::Sealed for ScalarExpr {}

impl Expression for ScalarExpr {
    type Item = u32;
}

impl ExpressionImpl for ScalarExpr {
    fn encode_leaves(&self, output: &mut Vec<EncodedExpr>) {
        output.push(EncodedExpr::from_node(&self.0));
    }

    fn leaf_count(&self) -> usize {
        1
    }

    fn assemble(&self, columns: &[Vec<u32>]) -> Result<Vec<Self::Item>> {
        let [column] = columns else {
            return Err(Error::Protocol(format!(
                "scalar result expected one column, received {}",
                columns.len()
            )));
        };
        Ok(column.clone())
    }
}

impl<Left, Right> private::Sealed for Zip<Left, Right>
where
    Left: Expression,
    Right: Expression,
{
}

impl<Left, Right> Expression for Zip<Left, Right>
where
    Left: Expression,
    Right: Expression,
{
    type Item = (Left::Item, Right::Item);
}

impl<Left, Right> ExpressionImpl for Zip<Left, Right>
where
    Left: ExpressionImpl,
    Right: ExpressionImpl,
{
    fn encode_leaves(&self, output: &mut Vec<EncodedExpr>) {
        self.left.encode_leaves(output);
        self.right.encode_leaves(output);
    }

    fn leaf_count(&self) -> usize {
        self.left.leaf_count() + self.right.leaf_count()
    }

    fn assemble(&self, columns: &[Vec<u32>]) -> Result<Vec<Self::Item>> {
        let left_count = self.left.leaf_count();
        if columns.len() != self.leaf_count() {
            return Err(Error::Protocol(format!(
                "product result expected {} columns, received {}",
                self.leaf_count(),
                columns.len()
            )));
        }
        let left = self.left.assemble(&columns[..left_count])?;
        let right = self.right.assemble(&columns[left_count..])?;
        if left.len() != right.len() {
            return Err(Error::Protocol(format!(
                "product result length mismatch: {} and {}",
                left.len(),
                right.len()
            )));
        }
        Ok(left.into_iter().zip(right).collect())
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

impl EncodedExpr {
    fn from_node(node: &ScalarNode) -> Self {
        let mut encoded = Self::default();
        encode_node(node, &mut encoded);
        encoded
    }
}

fn encode_node(node: &ScalarNode, encoded: &mut EncodedExpr) {
    match node {
        ScalarNode::SourceId => encoded.tokens.push("sid".into()),
        ScalarNode::DestinationId => encoded.tokens.push("did".into()),
        ScalarNode::EdgeId => encoded.tokens.push("eid".into()),
        ScalarNode::Source(values) => {
            let column = register_column(&mut encoded.vertex_columns, values);
            encoded.tokens.push(format!("src{column}"));
        }
        ScalarNode::Destination(values) => {
            let column = register_column(&mut encoded.vertex_columns, values);
            encoded.tokens.push(format!("dst{column}"));
        }
        ScalarNode::Edge(values) => {
            let column = register_column(&mut encoded.edge_columns, values);
            encoded.tokens.push(format!("edge{column}"));
        }
        ScalarNode::Constant(value) => encoded.tokens.push(format!("c{value}")),
        ScalarNode::Add(left, right) => {
            encoded.tokens.push("add".into());
            encode_node(left, encoded);
            encode_node(right, encoded);
        }
    }
}

fn register_column(columns: &mut Vec<Arc<[u32]>>, values: &Arc<[u32]>) -> usize {
    if let Some(index) = columns.iter().position(|column| column == values) {
        index
    } else {
        let index = columns.len();
        columns.push(values.clone());
        index
    }
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
    fn nested_products_reassemble_without_arity_cases() {
        let expression = zip2(zip2(source_id(), destination_id()), edge_id());
        let rows = expression
            .assemble(&[vec![0, 1], vec![1, 0], vec![4, 7]])
            .unwrap();
        assert_eq!(rows, vec![((0, 1), 4), ((1, 0), 7)]);
    }
}
