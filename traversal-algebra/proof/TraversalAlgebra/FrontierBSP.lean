import TraversalAlgebra.Algebra
import TraversalAlgebra.VerifiedGraph

namespace TraversalAlgebra.Verified.MonoidalFrontierBSP

/-- A semantic Monoidal Frontier BSP program.

The source machine is intentionally stated without traversal-algebra
operators. A superstep scans each active adjacency row directly, reduces
messages into dense destination inboxes, updates vertex state, and selects the
next ordered frontier. -/
structure Program (n : Nat) (VertexState : Type u)
    (EdgePayload : Type v) (Message : Type w) where
  message : Fin n → Fin n → VertexState → VertexState → EdgePayload → Message
  reduction : Reduction Message
  lawful : LawfulCommutativeReduction reduction
  update : Fin n → VertexState → Message → VertexState
  select : Frontier n → Store n VertexState → Store n VertexState → Frontier n

/-- Fold one active source row into the inbox of one destination. -/
def foldOutgoingAt
    (graph : OrderedGraph n EdgePayload)
    (program : Program n VertexState EdgePayload Message)
    (state : Store n VertexState)
    (destination source : Fin n)
    (initial : Message) : Message :=
  (graph.outgoing source).foldl
    (fun accumulator edge =>
      if edge.destination = destination then
        program.reduction.combine accumulator
          (program.message source edge.destination
            (state source) (state edge.destination) edge.payload)
      else
        accumulator)
    initial

/-- The source-machine inbox at one destination. This definition scans the
frontier and adjacency rows as nested loops; it does not call the target
machine's flattened traversal. -/
def inboxAt
    (graph : OrderedGraph n EdgePayload)
    (program : Program n VertexState EdgePayload Message)
    (state : Store n VertexState)
    (frontier : Frontier n)
    (destination : Fin n) : Message :=
  frontier.foldl
    (fun accumulator source =>
      foldOutgoingAt graph program state destination source accumulator)
    program.reduction.identity

/-- One bulk-synchronous source-machine transition. -/
def step
    (graph : OrderedGraph n EdgePayload)
    (program : Program n VertexState EdgePayload Message)
    (configuration : Configuration n VertexState) :
    Configuration n VertexState :=
  Configuration.advance program.update program.select
    (graph.nestedDestinations configuration.frontier) configuration
    (inboxAt graph program configuration.state configuration.frontier)

/-- A finite, step-indexed execution. -/
def run
    (graph : OrderedGraph n EdgePayload)
    (program : Program n VertexState EdgePayload Message) :
    Nat → Configuration n VertexState → Configuration n VertexState
  | 0, configuration => configuration
  | fuel + 1, configuration =>
      run graph program fuel (step graph program configuration)

end TraversalAlgebra.Verified.MonoidalFrontierBSP
