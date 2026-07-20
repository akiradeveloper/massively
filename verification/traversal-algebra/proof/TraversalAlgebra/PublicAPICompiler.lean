import TraversalAlgebra.PublicAPI
import TraversalAlgebra.TypedTraversalAlgebra

namespace TraversalAlgebra.Verified.Typed.PublicAPI

variable {n : Nat} {Base : Type} {denoteBase : Base → Type}
variable {signature : Signature n Base denoteBase}
variable {vertex edge message : ValueType Base}

/-!
This file lowers a closed Core TA superstep to denotational contracts of the
exported `massively::graph` and `massively::vector` operations.

`Program` is a proof artifact describing a host-side sequence of public calls;
it is not a claim that Rust exports a closed graph-program object.  In
particular, dense vertex update and frontier filtering are expressed through
the public vector basis because that is how Massively applications compose the
current APIs.
-/

namespace VectorBasis

/-- Dense public `vector::transform`, with a counting vertex column zipped into
the input so the mapper can observe the logical vertex identifier. -/
def transformStore
    (input : Store n Input)
    (mapper : Fin n → Input → Output) : Store n Output :=
  fun index => mapper index (input index)

/-- Stable public `vector::copy_where`. -/
def copyWhere (input : List Input) (selected : Input → Bool) : List Input :=
  input.filter selected

end VectorBasis

namespace GraphBasis

/-- Public
`traverse(...).map(edge-context, evaluator).reduce_by_destination(...)`.

The edge-context input is precisely the source ID, destination ID, source
state, destination state, and edge payload available to the current Core TA
term.  A Rust lowering constructs it from public pulls and `zipN`, then uses one
`UnaryOp` for `Term.evaluate`. -/
def reduceByDestination
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (frontier : Frontier n)
    (state : Store n (vertex.denote n denoteBase))
    (pullMap : Term signature (MessageContext vertex edge) message)
    (reduction : signature.Monoid message) :
    Store n (message.denote n denoteBase) :=
  fun destination =>
    (graph.traverse frontier).foldl
      (fun accumulator context =>
        if context.destination = destination then
          (signature.denoteMonoid reduction).combine accumulator
            (Term.evaluate (signature := signature)
              (MessageContext.fromEdge state context) pullMap)
        else
          accumulator)
      (signature.denoteMonoid reduction).identity

/-- Public `traverse(...).map(destination_id(), Identity).emit()`. -/
def emitDestinations
    (graph : OrderedGraph n EdgePayload)
    (frontier : Frontier n) : Frontier n :=
  (graph.traverse frontier).map (fun context => context.destination)

/-- The destination-ID emission contract is the ordered Core candidate stream. -/
theorem emitDestinations_correct
    (graph : OrderedGraph n EdgePayload)
    (frontier : Frontier n) :
    emitDestinations graph frontier = graph.traversalDestinations frontier := rfl

end GraphBasis

/-- A closed host schedule assembled only from public graph and vector
operation contracts. -/
structure Program
    (signature : Signature n Base denoteBase)
    (vertex edge message : ValueType Base) where
  pullMap : Term signature (MessageContext vertex edge) message
  reduction : signature.Monoid message
  vertexMap : Term signature (UpdateContext vertex message) vertex
  frontier : FrontierPolicy signature vertex

namespace Program

/-- Evaluate the public dense vector transform used for vertex update. -/
def updateState
    (program : Program signature vertex edge message)
    (oldState : Store n (vertex.denote n denoteBase))
    (inbox : Store n (message.denote n denoteBase)) :
    Store n (vertex.denote n denoteBase) :=
  VectorBasis.transformStore oldState fun vertexId oldValue =>
    Term.evaluate (signature := signature)
      (UpdateContext.environment vertexId oldValue (inbox vertexId))
      program.vertexMap

/-- Build the next frontier with public vector filtering.

Dense policy filters the counting vertex sequence.  Sparse policy first emits
the traversal's destination-ID stream and then filters it stably, preserving
candidate order and multiplicity exactly. -/
def selectFrontier
    (program : Program signature vertex edge message)
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (active : Frontier n)
    (oldState newState : Store n (vertex.denote n denoteBase)) : Frontier n :=
  match program.frontier with
  | .dense predicate =>
      VectorBasis.copyWhere (allVertices n) fun vertexId =>
        Term.evaluate (signature := signature)
          (SelectionContext.environment vertexId
            (oldState vertexId) (newState vertexId)) predicate
  | .sparsePreserve predicate =>
      VectorBasis.copyWhere (GraphBasis.emitDestinations graph active) fun vertexId =>
        Term.evaluate (signature := signature)
          (SelectionContext.environment vertexId
            (oldState vertexId) (newState vertexId)) predicate

/-- Public graph+vector API execution of one closed TA superstep. -/
def step
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : Program signature vertex edge message)
    (configuration : Configuration n (vertex.denote n denoteBase)) :
    Configuration n (vertex.denote n denoteBase) :=
  let inbox := GraphBasis.reduceByDestination graph configuration.frontier
    configuration.state program.pullMap program.reduction
  let nextState := program.updateState configuration.state inbox
  { state := nextState
    frontier := program.selectFrontier graph configuration.frontier
      configuration.state nextState }

/-- Finite host execution of the compiled public operation schedule. -/
def run
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : Program signature vertex edge message) :
    Nat → Configuration n (vertex.denote n denoteBase) →
      Configuration n (vertex.denote n denoteBase)
  | 0, configuration => configuration
  | fuel + 1, configuration =>
      run graph program fuel (step graph program configuration)

/-- Public vector filtering implements the Core frontier-policy denotation. -/
theorem selectFrontier_correct
    (program : Program signature vertex edge message)
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (active : Frontier n)
    (oldState newState : Store n (vertex.denote n denoteBase)) :
    program.selectFrontier graph active oldState newState =
      program.frontier.denote (graph.traversalDestinations active)
        oldState newState := by
  cases program with
  | mk pullMap reduction vertexMap frontier =>
      cases frontier <;> rfl

end Program

/-- Syntax-directed lowering from Core TA to the public graph+vector basis. -/
def compile
    (program : TraversalAlgebra.Program signature vertex edge message) :
    Program signature vertex edge message where
  pullMap := program.destinationPush.pullMap
  reduction := program.destinationPush.reduction
  vertexMap := program.vertexMap
  frontier := program.frontierCompact

/-- One Core TA superstep is implemented exactly by the compiled sequence of
public graph and vector operation contracts. -/
theorem compile_step_correct
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : TraversalAlgebra.Program signature vertex edge message)
    (configuration : Configuration n (vertex.denote n denoteBase)) :
    TraversalAlgebra.step graph program configuration =
      Program.step graph (compile program) configuration := by
  cases program with
  | mk destinationPush vertexMap frontierCompact =>
      cases destinationPush
      cases frontierCompact <;> rfl

/-- Compilation to public operation contracts preserves every finite Core TA
execution. -/
theorem compile_run_correct
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : TraversalAlgebra.Program signature vertex edge message)
    (fuel : Nat)
    (configuration : Configuration n (vertex.denote n denoteBase)) :
    TraversalAlgebra.run graph program fuel configuration =
      Program.run graph (compile program) fuel configuration := by
  induction fuel generalizing configuration with
  | zero => rfl
  | succ fuel induction =>
      simp only [TraversalAlgebra.run, Program.run]
      rw [compile_step_correct]
      exact induction _

/-- Direct compiler from typed Monoidal Frontier BSP to the current public
graph+vector API basis. -/
def compileBSP
    (program : MonoidalFrontierBSP.Program signature vertex edge message) :
    Program signature vertex edge message :=
  compile (TraversalAlgebra.encode program)

/-- Every typed Monoidal Frontier BSP program has an execution using only the
modeled public graph and vector operation contracts, with the complete state
and ordered frontier preserved after every finite number of supersteps. -/
theorem compileBSP_run_correct
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : MonoidalFrontierBSP.Program signature vertex edge message)
    (fuel : Nat)
    (configuration : Configuration n (vertex.denote n denoteBase)) :
    MonoidalFrontierBSP.run graph program fuel configuration =
      Program.run graph (compileBSP program) fuel configuration := by
  calc
    MonoidalFrontierBSP.run graph program fuel configuration =
        TraversalAlgebra.run graph (TraversalAlgebra.encode program)
          fuel configuration :=
      TraversalAlgebra.encode_run_correct graph program fuel configuration
    _ = Program.run graph (compile (TraversalAlgebra.encode program))
          fuel configuration :=
      compile_run_correct graph (TraversalAlgebra.encode program) fuel configuration
    _ = Program.run graph (compileBSP program) fuel configuration := rfl

end TraversalAlgebra.Verified.Typed.PublicAPI
