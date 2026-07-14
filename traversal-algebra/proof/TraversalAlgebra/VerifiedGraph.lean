namespace TraversalAlgebra.Verified

/-- An outgoing edge in a finite graph. Vertex identifiers are valid by
construction; the payload can contain an edge identifier or arbitrary static
edge state. -/
structure Edge (n : Nat) (EdgePayload : Type u) where
  destination : Fin n
  payload : EdgePayload

/-- A finite ordered directed multigraph. Lists preserve adjacency order and
permit parallel edges and self-loops. -/
structure OrderedGraph (n : Nat) (EdgePayload : Type u) where
  outgoing : Fin n → List (Edge n EdgePayload)

/-- An ordered frontier. Multiplicity remains observable. -/
abbrev Frontier (n : Nat) := List (Fin n)

/-- One traversed edge with all data available to pull/map expressions. -/
structure EdgeContext (n : Nat) (EdgePayload : Type u) where
  source : Fin n
  destination : Fin n
  payload : EdgePayload

namespace OrderedGraph

/-- Expand one source vertex in adjacency order. -/
def expand (graph : OrderedGraph n EdgePayload) (source : Fin n) :
    List (EdgeContext n EdgePayload) :=
  (graph.outgoing source).map fun edge =>
    { source, destination := edge.destination, payload := edge.payload }

/-- Expand a frontier in frontier order and then adjacency order. -/
def traverse (graph : OrderedGraph n EdgePayload) :
    Frontier n → List (EdgeContext n EdgePayload)
  | [] => []
  | source :: frontier => graph.expand source ++ graph.traverse frontier

@[simp]
theorem traverse_nil (graph : OrderedGraph n EdgePayload) :
    graph.traverse [] = [] := rfl

@[simp]
theorem traverse_cons (graph : OrderedGraph n EdgePayload)
    (source : Fin n) (frontier : Frontier n) :
    graph.traverse (source :: frontier) =
      graph.expand source ++ graph.traverse frontier := rfl

@[simp]
theorem traverse_append (graph : OrderedGraph n EdgePayload)
    (left right : Frontier n) :
    graph.traverse (left ++ right) =
      graph.traverse left ++ graph.traverse right := by
  induction left with
  | nil => rfl
  | cons source left induction =>
      simp only [List.cons_append, traverse_cons, induction, List.append_assoc]

/-- Destination candidates constructed directly by the source machine's
nested frontier/adjacency-row scan. Order and multiplicity are preserved. -/
def nestedDestinations (graph : OrderedGraph n EdgePayload) :
    Frontier n → Frontier n
  | [] => []
  | source :: frontier =>
      (graph.outgoing source).map (fun edge => edge.destination) ++
        graph.nestedDestinations frontier

/-- Destination candidates obtained from the flattened TA traversal. -/
def traversalDestinations
    (graph : OrderedGraph n EdgePayload) (frontier : Frontier n) : Frontier n :=
  (graph.traverse frontier).map (fun context => context.destination)

/-- Flattening preserves the exact sparse-candidate order and multiplicity. -/
theorem traversalDestinations_eq_nestedDestinations
    (graph : OrderedGraph n EdgePayload) (frontier : Frontier n) :
    graph.traversalDestinations frontier = graph.nestedDestinations frontier := by
  induction frontier with
  | nil => rfl
  | cons source frontier induction =>
      have headCandidates :
          (graph.expand source).map (fun context => context.destination) =
            (graph.outgoing source).map (fun edge => edge.destination) := by
        unfold expand
        rw [List.map_map]
        apply List.map_congr_left
        intro edge membership
        rfl
      calc
        graph.traversalDestinations (source :: frontier) =
            (graph.outgoing source).map (fun edge => edge.destination) ++
              graph.traversalDestinations frontier := by
          simp only [traversalDestinations, traverse_cons, List.map_append,
            headCandidates]
        _ = graph.nestedDestinations (source :: frontier) := by
          rw [induction]
          rfl

end OrderedGraph

/-- Vertex-indexed storage used by both semantic machines. -/
abbrev Store (n : Nat) (Value : Type u) := Fin n → Value

/-- The observable state at a bulk-synchronous barrier. -/
structure Configuration (n : Nat) (VertexState : Type u) where
  state : Store n VertexState
  frontier : Frontier n

namespace Configuration

/-- The standard frontier-machine halting observation. Execution functions
remain total; a driver may stop at the first empty frontier. -/
def Halted (configuration : Configuration n VertexState) : Prop :=
  configuration.frontier = []

/-- Apply dense vertex updates and choose the next ordered frontier. The
source and target machines share this barrier operation; their substantive
difference is how they construct `inbox`. -/
def advance
    (update : Fin n → VertexState → Message → VertexState)
    (select : Frontier n →
      Store n VertexState → Store n VertexState → Frontier n)
    (candidates : Frontier n)
    (configuration : Configuration n VertexState)
    (inbox : Store n Message) : Configuration n VertexState :=
  let nextState := fun vertex =>
    update vertex (configuration.state vertex) (inbox vertex)
  { state := nextState
    frontier := select candidates configuration.state nextState }

end Configuration

end TraversalAlgebra.Verified
