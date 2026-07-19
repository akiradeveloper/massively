import TraversalAlgebra.Algebra
import TraversalAlgebra.VerifiedGraph

namespace TraversalAlgebra.Verified.TraversalAlgebra

/-- Type-safe emission: one mapped value for every traversed edge context. -/
def emit
    (graph : OrderedGraph n EdgePayload)
    (frontier : Frontier n)
    (map : EdgeContext n EdgePayload → Value) : List Value :=
  (graph.traverse frontier).map map

/-- Sequential denotation of one monoidal edge fold. -/
def reduceEdges
    (reduction : Reduction Value)
    (map : EdgeContext n EdgePayload → Value)
    (contexts : List (EdgeContext n EdgePayload)) : Value :=
  contexts.foldl
    (fun accumulator context => reduction.combine accumulator (map context))
    reduction.identity

/-- One reduction result per frontier occurrence, preserving order and
multiplicity. -/
def reduceBySource
    (graph : OrderedGraph n EdgePayload)
    (frontier : Frontier n)
    (reduction : Reduction Value)
    (map : EdgeContext n EdgePayload → Value) : List Value :=
  frontier.map fun source => reduceEdges reduction map (graph.expand source)

/-- One destination-filtered reduction over the flattened traversal. -/
def reduceByDestinationAt
    (graph : OrderedGraph n EdgePayload)
    (frontier : Frontier n)
    (reduction : Reduction Value)
    (map : EdgeContext n EdgePayload → Value)
    (destination : Fin n) : Value :=
  (graph.traverse frontier).foldl
    (fun accumulator context =>
      if context.destination = destination then
        reduction.combine accumulator (map context)
      else
        accumulator)
    reduction.identity

/-- Dense destination-indexed reduction. -/
def reduceByDestination
    (graph : OrderedGraph n EdgePayload)
    (frontier : Frontier n)
    (reduction : Reduction Value)
    (map : EdgeContext n EdgePayload → Value) : Store n Value :=
  fun destination =>
    reduceByDestinationAt graph frontier reduction map destination

end TraversalAlgebra.Verified.TraversalAlgebra
