import TraversalAlgebra.Algebra
import TraversalAlgebra.Expression

namespace TraversalAlgebra

/-- Emit one mapped value for each selected edge. -/
def emit (graph : Graph) (frontier : Frontier) (map : EdgeContext → α) : List α :=
  (graph.traverse frontier).map map

/-- Emit a structural identifier expression. -/
def emitId (graph : Graph) (frontier : Frontier) (expression : IdExpr) : List Nat :=
  emit graph frontier expression.eval

/-- Sequential denotation of a reduction. Lawful parallel implementations may
reassociate this fold; destination reduction additionally requires
commutativity when collisions can be reordered. -/
def reduceEdges (reduction : Reduction α) (map : EdgeContext → α)
    (edges : List EdgeContext) : α :=
  edges.foldl (fun accumulator edge => reduction.combine accumulator (map edge))
    reduction.identity

/-- One reduction result per frontier occurrence, including empty adjacency
rows. This intentionally does not deduplicate repeated source vertices. -/
def reduceBySource (graph : Graph) (frontier : Frontier)
    (reduction : Reduction α) (map : EdgeContext → α) : List α :=
  frontier.map fun source => reduceEdges reduction map (graph.expand source)

/-- Reduction of all values targeting one destination vertex. -/
def reduceByDestinationAt (graph : Graph) (frontier : Frontier)
    (reduction : Reduction α) (map : EdgeContext → α) (destination : Nat) : α :=
  (graph.traverse frontier).foldl
    (fun accumulator edge =>
      if edge.destination = destination then
        reduction.combine accumulator (map edge)
      else
        accumulator)
    reduction.identity

/-- Dense destination-indexed reduction result. -/
def reduceByDestination (graph : Graph) (frontier : Frontier)
    (reduction : Reduction α) (map : EdgeContext → α) : List α :=
  (List.range graph.vertexCount).map
    (reduceByDestinationAt graph frontier reduction map)

end TraversalAlgebra
