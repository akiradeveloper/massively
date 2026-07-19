use crate::{
    Error, Result,
    graph::{EdgeContext, Expression, ExpressionImpl, Observation, Query, Terminal},
};

/// Stateless, sequential reference evaluator for Traversal Algebra queries.
#[derive(Clone, Copy, Debug, Default)]
pub struct CpuOracle;

impl CpuOracle {
    pub const fn new() -> Self {
        Self
    }

    #[allow(private_bounds)]
    pub fn evaluate<Expr>(&self, query: Query<Expr>) -> Result<Vec<Expr::Item>>
    where
        Expr: Expression + ExpressionImpl,
    {
        Ok(self.observe(query)?.into_values())
    }

    #[allow(private_bounds)]
    pub fn observe<Expr>(&self, query: Query<Expr>) -> Result<Observation<Expr::Item>>
    where
        Expr: Expression + ExpressionImpl,
    {
        query.graph.validate()?;
        let vertices = query.graph.offsets().len() - 1;
        let edges = query.graph.destinations().len();
        query.expression.validate(vertices, edges)?;
        validate_frontier(&query)?;

        let columns = match query.terminal {
            Terminal::Emit => emit_columns(&query)?,
            Terminal::ReduceBySource => source_columns(&query)?,
            Terminal::ReduceByDestination => destination_columns(&query, vertices)?,
        };
        let values = query.expression.assemble(columns)?;
        Ok(match query.terminal {
            Terminal::Emit => Observation::Emitted(values),
            Terminal::ReduceBySource => Observation::SourceReduced(values),
            Terminal::ReduceByDestination => Observation::DestinationReduced(values),
        })
    }
}

fn validate_frontier<Expr>(query: &Query<Expr>) -> Result<()> {
    let vertices = query.graph.offsets().len() - 1;
    if let Some(source) = query
        .frontier
        .iter()
        .copied()
        .find(|&source| source as usize >= vertices)
    {
        return Err(Error::InvalidInput(format!(
            "frontier source {source} lies outside {vertices} vertices"
        )));
    }
    Ok(())
}

fn active_edge_count<Expr>(query: &Query<Expr>) -> usize {
    query
        .frontier
        .iter()
        .map(|&source| {
            let source = source as usize;
            (query.graph.offsets()[source + 1] - query.graph.offsets()[source]) as usize
        })
        .sum()
}

fn emit_columns<Expr>(query: &Query<Expr>) -> Result<Vec<Vec<u32>>>
where
    Expr: ExpressionImpl,
{
    let leaves = query.expression.leaf_count();
    let capacity = active_edge_count(query);
    let mut columns = (0..leaves)
        .map(|_| Vec::with_capacity(capacity))
        .collect::<Vec<_>>();
    let mut values = vec![0; leaves];
    for_each_edge(query, |context| {
        query.expression.evaluate(context, &mut values)?;
        for (column, &value) in columns.iter_mut().zip(&values) {
            column.push(value);
        }
        Ok(())
    })?;
    Ok(columns)
}

fn source_columns<Expr>(query: &Query<Expr>) -> Result<Vec<Vec<u32>>>
where
    Expr: ExpressionImpl,
{
    let leaves = query.expression.leaf_count();
    let mut columns = (0..leaves)
        .map(|_| Vec::with_capacity(query.frontier.len()))
        .collect::<Vec<_>>();
    let mut values = vec![0; leaves];
    let mut reduced = vec![0_u32; leaves];
    let offsets = query.graph.offsets();
    let destinations = query.graph.destinations();

    for &source in query.frontier.iter() {
        let source_index = source as usize;
        reduced.fill(0);
        for edge in offsets[source_index] as usize..offsets[source_index + 1] as usize {
            let context = EdgeContext {
                source,
                destination: destinations[edge],
                edge: edge as u32,
            };
            query.expression.evaluate(context, &mut values)?;
            add_leaves(&mut reduced, &values)?;
        }
        for (column, &value) in columns.iter_mut().zip(&reduced) {
            column.push(value);
        }
    }
    Ok(columns)
}

fn destination_columns<Expr>(query: &Query<Expr>, vertices: usize) -> Result<Vec<Vec<u32>>>
where
    Expr: ExpressionImpl,
{
    let leaves = query.expression.leaf_count();
    let mut columns = vec![vec![0_u32; vertices]; leaves];
    let mut values = vec![0; leaves];
    for_each_edge(query, |context| {
        query.expression.evaluate(context, &mut values)?;
        let destination = context.destination as usize;
        for (column, &value) in columns.iter_mut().zip(&values) {
            column[destination] = column[destination]
                .checked_add(value)
                .ok_or(Error::ArithmeticOverflow)?;
        }
        Ok(())
    })?;
    Ok(columns)
}

fn add_leaves(reduced: &mut [u32], values: &[u32]) -> Result<()> {
    for (accumulator, &value) in reduced.iter_mut().zip(values) {
        *accumulator = accumulator
            .checked_add(value)
            .ok_or(Error::ArithmeticOverflow)?;
    }
    Ok(())
}

fn for_each_edge<Expr>(
    query: &Query<Expr>,
    mut visit: impl FnMut(EdgeContext) -> Result<()>,
) -> Result<()> {
    let offsets = query.graph.offsets();
    let destinations = query.graph.destinations();
    for &source in query.frontier.iter() {
        let source_index = source as usize;
        for edge in offsets[source_index] as usize..offsets[source_index + 1] as usize {
            visit(EdgeContext {
                source,
                destination: destinations[edge],
                edge: edge as u32,
            })?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{self, op};

    fn graph() -> graph::Csr {
        graph::Csr::new(vec![1, 1, 0], vec![0, 2, 3])
    }

    #[test]
    fn observes_order_duplicates_and_all_terminal_shapes() {
        let oracle = CpuOracle::new();
        let contexts = oracle
            .observe(
                graph::traverse(graph(), vec![0, 0, 1])
                    .unwrap()
                    .map(
                        graph::zip2(
                            graph::zip2(graph::source_id(), graph::destination_id()),
                            graph::edge_id(),
                        ),
                        op::Identity,
                    )
                    .emit(),
            )
            .unwrap();
        assert_eq!(
            contexts,
            Observation::Emitted(vec![(0, 1, 0), (0, 1, 1), (0, 1, 0), (0, 1, 1), (1, 0, 2),])
        );

        let mapped = || {
            graph::traverse(graph(), vec![0, 0, 1])
                .unwrap()
                .map(graph::edge_id(), op::One)
        };
        assert_eq!(
            oracle
                .observe(mapped().reduce_by_source(0, op::Add).unwrap())
                .unwrap(),
            Observation::SourceReduced(vec![2, 2, 1])
        );
        assert_eq!(
            oracle
                .observe(mapped().reduce_by_destination(0, op::Add).unwrap())
                .unwrap(),
            Observation::DestinationReduced(vec![1, 4])
        );
    }

    #[test]
    fn evaluates_vertex_edge_and_pointwise_map_values() {
        let oracle = CpuOracle::new();
        let query = graph::traverse(graph(), vec![0, 1])
            .unwrap()
            .map(
                graph::zip2(graph::source(vec![10, 20]), graph::edge(vec![1, 2, 3])),
                op::AddPair,
            )
            .emit();
        assert_eq!(oracle.evaluate(query).unwrap(), vec![11, 12, 23]);
    }

    #[test]
    fn rejects_columns_with_the_wrong_domain() {
        let query = graph::traverse(graph(), vec![0])
            .unwrap()
            .map(graph::source(vec![1]), op::Identity)
            .emit();
        assert!(matches!(
            CpuOracle::new().evaluate(query),
            Err(Error::InvalidInput(_))
        ));
    }

    #[test]
    fn rejects_natural_number_overflow_instead_of_wrapping() {
        let query = graph::traverse(graph(), vec![0])
            .unwrap()
            .map(
                graph::zip2(graph::source(vec![u32::MAX, 0]), graph::edge(vec![1, 0, 0])),
                op::AddPair,
            )
            .emit();
        assert!(matches!(
            CpuOracle::new().evaluate(query),
            Err(Error::ArithmeticOverflow)
        ));
    }
}
