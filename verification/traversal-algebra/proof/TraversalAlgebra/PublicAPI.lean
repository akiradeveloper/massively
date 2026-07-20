import TraversalAlgebra.TypedObservations

namespace TraversalAlgebra.Verified.PublicAPI

/-!
This file is the denotational boundary of the exported Rust traversal API.

Unlike the compositional Core TA grammar, the surface shape below deliberately
has exactly one map after an edge expression.  Edge expressions contain the
six public Rust leaves (`source`, `destination`, `edge`, and the three IDs) and
binary `zip`; literals and nested maps are not surface constructors.

The model uses finite identifiers rather than the Rust `u32` representation.
Index encoding, buffer validity, CubeCL code generation, and hardware execution
remain implementation-refinement obligations.
-/

/-- A valid CSR topology with `n` vertices and `edgeCount` stored edge slots.

The payload of each ordered graph edge is its CSR edge position.  Thus an edge
column is exactly a function from that position to its value.  Construction of
this finite model from concrete destination and offset buffers is intentionally
the next implementation-refinement boundary. -/
structure Csr (n edgeCount : Nat) where
  graph : OrderedGraph n (Fin edgeCount)
  edgePositions :
    (graph.traverse (Typed.allVertices n)).map (fun context => context.payload) =
      List.finRange edgeCount

namespace Csr

/-- Select one adjacency row from a public CSR topology. -/
def expand (csr : Csr n edgeCount) (source : Fin n) :
    List (EdgeContext n (Fin edgeCount)) :=
  csr.graph.expand source

/-- Select the ordered edge stream of a public CSR traversal. -/
def traverse (csr : Csr n edgeCount) (frontier : Frontier n) :
    List (EdgeContext n (Fin edgeCount)) :=
  csr.graph.traverse frontier

end Csr

/-- The closed surface grammar implemented by `massively::graph::EdgeExpr`.

Columns are independent, matching the Rust API: source and destination pulls
need not originate from one pre-bundled vertex state, and separate edge pulls
need not originate from one pre-bundled edge payload. -/
inductive EdgeExpr (n edgeCount : Nat) : Type → Type 1
  | source (values : Store n Value) : EdgeExpr n edgeCount Value
  | destination (values : Store n Value) : EdgeExpr n edgeCount Value
  | edge (values : Fin edgeCount → Value) : EdgeExpr n edgeCount Value
  | sourceId : EdgeExpr n edgeCount (Fin n)
  | destinationId : EdgeExpr n edgeCount (Fin n)
  | edgeId : EdgeExpr n edgeCount (Fin edgeCount)
  | zip : EdgeExpr n edgeCount Left → EdgeExpr n edgeCount Right →
      EdgeExpr n edgeCount (Left × Right)

namespace EdgeExpr

/-- Evaluate one public edge expression at one materialized traversal row. -/
def evaluate : {Value : Type} →
    EdgeExpr n edgeCount Value → EdgeContext n (Fin edgeCount) → Value
  | _, .source values, context => values context.source
  | _, .destination values, context => values context.destination
  | _, .edge values, context => values context.payload
  | _, .sourceId, context => context.source
  | _, .destinationId, context => context.destination
  | _, .edgeId, context => context.payload
  | _, .zip left right, context =>
      (evaluate left context, evaluate right context)

end EdgeExpr

/-- The result of the public `traverse` constructor. -/
structure Traversal (n edgeCount : Nat) where
  graph : Csr n edgeCount
  frontier : Frontier n

/-- The result of the single public `Traversal::map` call. -/
structure MappedTraversal
    (n edgeCount : Nat) (Input Output : Type) where
  traversal : Traversal n edgeCount
  expression : EdgeExpr n edgeCount Input
  map : Input → Output

/-- The semantic precondition of a public `ReductionOp` plus its `init` value.

Rust documents these laws but cannot derive them from a CubeCL trait
implementation.  They are therefore an explicit premise at the proof boundary,
not a claim that every Rust `ReductionOp` is lawful. -/
structure ReductionOp (Value : Type) where
  reduction : Reduction Value
  lawful : LawfulCommutativeReduction reduction

/-- Contract data for the public minimum-relaxation specialization. -/
structure MinimumOp (Value : Type) where
  operation : ReductionOp Value
  lowered : Value → Value → Bool
  untouched : ∀ old, lowered old operation.reduction.identity = false

/-- Denotation of the public `traverse` constructor on a valid finite CSR. -/
def traverse (graph : Csr n edgeCount) (frontier : Frontier n) :
    Traversal n edgeCount :=
  { graph, frontier }

namespace Traversal

/-- Attach the one public pointwise map operation to an edge expression. -/
def map
    (traversal : Traversal n edgeCount)
    (expression : EdgeExpr n edgeCount Input)
    (mapper : Input → Output) :
    MappedTraversal n edgeCount Input Output :=
  { traversal, expression, map := mapper }

end Traversal

namespace MappedTraversal

/-- The mapped value associated with one selected edge occurrence. -/
def mapAt
    (mapped : MappedTraversal n edgeCount Input Output)
    (context : EdgeContext n (Fin edgeCount)) : Output :=
  mapped.map (mapped.expression.evaluate context)

/-- Public `emit`: one ordered result per selected edge occurrence. -/
def emit (mapped : MappedTraversal n edgeCount Input Output) : List Output :=
  (mapped.traversal.graph.traverse mapped.traversal.frontier).map mapped.mapAt

/-- Public `reduce_by_source`: one result per frontier occurrence. -/
def reduceBySource
    (mapped : MappedTraversal n edgeCount Input Output)
    (operation : ReductionOp Output) : List Output :=
  mapped.traversal.frontier.map fun source =>
    (mapped.traversal.graph.expand source).foldl
      (fun accumulator context =>
        operation.reduction.combine accumulator (mapped.mapAt context))
      operation.reduction.identity

/-- Public `reduce_by_destination`: a complete dense vertex result. -/
def reduceByDestination
    (mapped : MappedTraversal n edgeCount Input Output)
    (operation : ReductionOp Output) : Store n Output :=
  fun destination =>
    (mapped.traversal.graph.traverse mapped.traversal.frontier).foldl
      (fun accumulator context =>
        if context.destination = destination then
          operation.reduction.combine accumulator (mapped.mapAt context)
        else
          accumulator)
      operation.reduction.identity

/-- Public `update_by_destination`: combine each reduced proposal with existing
destination state.  The identity law makes untouched destinations unchanged. -/
def updateByDestination
    (mapped : MappedTraversal n edgeCount Input Output)
    (operation : ReductionOp Output)
    (state : Store n Output) : Store n Output :=
  fun destination => operation.reduction.combine (state destination)
    (mapped.reduceByDestination operation destination)

/-- Public `relax_min_by_destination` contract.

The concrete Rust specialization uses `u32::min`, `u32::MAX`, and `<`.  Its
sort/reduce-by-key path emits each lowered destination once in canonical vertex
order, so the result is modeled as a dense filter rather than the duplicate-
preserving Core TA sparse policy. -/
def relaxMinByDestination
    (mapped : MappedTraversal n edgeCount Input Output)
    (minimum : MinimumOp Output)
    (state : Store n Output) : Configuration n Output :=
  let proposals := mapped.reduceByDestination minimum.operation
  let nextState := fun destination =>
    minimum.operation.reduction.combine (state destination) (proposals destination)
  { state := nextState
    frontier := (Typed.allVertices n).filter fun destination =>
      minimum.lowered (state destination) (proposals destination) }

/-- The public emission contract is exactly the type-safe TA terminal. -/
theorem emit_correct
    (mapped : MappedTraversal n edgeCount Input Output) :
    mapped.emit =
      TraversalAlgebra.emit mapped.traversal.graph.graph mapped.traversal.frontier
        mapped.mapAt := by
  rfl

/-- The public source-reduction contract is exactly the type-safe TA terminal. -/
theorem reduceBySource_correct
    (mapped : MappedTraversal n edgeCount Input Output)
    (operation : ReductionOp Output) :
    mapped.reduceBySource operation =
      TraversalAlgebra.reduceBySource mapped.traversal.graph.graph
        mapped.traversal.frontier operation.reduction mapped.mapAt := by
  rfl

/-- The public destination-reduction contract is exactly the type-safe TA
terminal, including its dense result shape. -/
theorem reduceByDestination_correct
    (mapped : MappedTraversal n edgeCount Input Output)
    (operation : ReductionOp Output) :
    mapped.reduceByDestination operation =
      TraversalAlgebra.reduceByDestination mapped.traversal.graph.graph
        mapped.traversal.frontier operation.reduction mapped.mapAt := by
  rfl

/-- The state component of minimum relaxation is the public generic
destination update specialized to the minimum operation. -/
theorem relaxMinByDestination_state
    (mapped : MappedTraversal n edgeCount Input Output)
    (minimum : MinimumOp Output)
    (state : Store n Output) :
    (mapped.relaxMinByDestination minimum state).state =
      mapped.updateByDestination minimum.operation state := rfl

/-- Minimum relaxation emits a canonical dense filter, not the
duplicate-preserving sparse candidate stream of Core TA. -/
theorem relaxMinByDestination_frontier
    (mapped : MappedTraversal n edgeCount Input Output)
    (minimum : MinimumOp Output)
    (state : Store n Output) :
    (mapped.relaxMinByDestination minimum state).frontier =
      (Typed.allVertices n).filter fun destination =>
        minimum.lowered (state destination)
          (mapped.reduceByDestination minimum.operation destination) := rfl

end MappedTraversal

end TraversalAlgebra.Verified.PublicAPI
