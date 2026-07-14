import TraversalAlgebra.Semantics

namespace TraversalAlgebra.Oracle

/-- An input whose expected outputs are evaluated by Lean during generation. -/
structure Case where
  name : String
  graph : Graph
  frontier : Frontier
deriving Repr, DecidableEq

/-- Cases cover emptiness, row order, repeated frontier vertices, isolated
vertices, parallel edges, and self-loops. -/
def cases : List Case :=
  [ { name := "empty", graph := ⟨[]⟩, frontier := [] }
  , { name := "ordered_rows"
      graph := ⟨[[1, 2], [2], [0, 1]]⟩
      frontier := [2, 0] }
  , { name := "isolated_and_parallel_self_loops"
      graph := ⟨[[0, 0], [], [1]]⟩
      frontier := [1, 0, 1] }
  , { name := "duplicate_frontier"
      graph := ⟨[[1], [0]]⟩
      frontier := [0, 0, 1] }
  , { name := "parallel_edges_and_self_loop"
      graph := ⟨[[1, 1, 0], [0]]⟩
      frontier := [0] }
  ]

def edgeCountMap (_ : EdgeContext) : Nat := 1

def expectedEdges (case : Case) : List EdgeContext :=
  case.graph.traverse case.frontier

def expectedSourceCounts (case : Case) : List Nat :=
  reduceBySource case.graph case.frontier natAdd edgeCountMap

def expectedDestinationCounts (case : Case) : List Nat :=
  reduceByDestination case.graph case.frontier natAdd edgeCountMap

def isValid (case : Case) : Bool :=
  case.graph.isValid && isValidFrontier case.graph case.frontier

end TraversalAlgebra.Oracle
