namespace TraversalAlgebra

/-- A finite ordered directed multigraph represented by adjacency rows.

The position of a row is its source vertex. A row is an ordered list rather
than a set, so parallel edges and self-loops are represented without a special
case. -/
structure Graph where
  rows : List (List Nat)
deriving Repr, DecidableEq

namespace Graph

/-- The graph's finite vertex count. -/
def vertexCount (graph : Graph) : Nat :=
  graph.rows.length

/-- The adjacency row of `source`, or the empty row outside the vertex set. -/
def row (graph : Graph) (source : Nat) : List Nat :=
  graph.rows.getD source []

/-- Number of edges in rows preceding `source`. -/
def edgeBase (graph : Graph) (source : Nat) : Nat :=
  (graph.rows.take source).foldl (fun total row => total + row.length) 0

/-- Total number of directed edges, counting multiplicity. -/
def edgeCount (graph : Graph) : Nat :=
  graph.rows.foldl (fun total row => total + row.length) 0

/-- Every stored destination names a vertex of the graph. -/
def Valid (graph : Graph) : Prop :=
  ∀ row ∈ graph.rows, ∀ destination ∈ row, destination < graph.vertexCount

/-- A decidable validity check used by executable artifact generation. -/
def isValid (graph : Graph) : Bool :=
  graph.rows.all fun row => row.all fun destination => destination < graph.vertexCount

/-- CSR offsets corresponding to the ordered adjacency rows. -/
def csrOffsets (graph : Graph) : List Nat :=
  0 :: go 0 graph.rows
where
  go (offset : Nat) : List (List Nat) → List Nat
    | [] => []
    | row :: rows =>
        let next := offset + row.length
        next :: go next rows

/-- Flat CSR destination stream corresponding to the ordered adjacency rows. -/
def csrDestinations (graph : Graph) : List Nat :=
  graph.rows.flatten

end Graph

/-- A selected edge together with all three structural identifiers exposed by
Traversal Algebra. -/
structure EdgeContext where
  source : Nat
  destination : Nat
  edge : Nat
deriving Repr, DecidableEq

/-- A frontier is an ordered sequence. Repeated vertices are semantically
significant and therefore preserved. -/
abbrev Frontier := List Nat

/-- Every frontier entry names a vertex of the graph. -/
def ValidFrontier (graph : Graph) (frontier : Frontier) : Prop :=
  ∀ source ∈ frontier, source < graph.vertexCount

/-- Decidable frontier validity check used by executable artifact generation. -/
def isValidFrontier (graph : Graph) (frontier : Frontier) : Bool :=
  frontier.all fun source => source < graph.vertexCount

namespace Graph

/-- Recursive row expansion with an explicit local edge index. -/
def expandFrom (graph : Graph) (source : Nat) : Nat → List Nat → List EdgeContext
  | _, [] => []
  | index, destination :: destinations =>
      { source, destination, edge := graph.edgeBase source + index } ::
        expandFrom graph source (index + 1) destinations

/-- Select the ordered outgoing edges of one source vertex. -/
def expand (graph : Graph) (source : Nat) : List EdgeContext :=
  expandFrom graph source 0 (graph.row source)

/-- Select all outgoing edges of the frontier, preserving frontier order and
each adjacency row's order. -/
def traverse (graph : Graph) (frontier : Frontier) : List EdgeContext :=
  frontier.flatMap (graph.expand ·)

end Graph

end TraversalAlgebra
