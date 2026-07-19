import TraversalAlgebra.TypedSimulation
import TraversalAlgebra.VerifiedTraversalAlgebra

namespace TraversalAlgebra.Verified.Typed.TraversalAlgebra

variable {n : Nat} {Base : Type} {denoteBase : Base → Type}
variable {signature : Signature n Base denoteBase}
variable {vertex edge message : ValueType Base}

/-- The closed destination terminal of the Traversal Algebra BSP fragment.

Operationally this is
`traverse(frontier).map(pullMap).reduceByDestination(reduction)`. The typed
message term may pull source/destination identifiers and state plus the edge
payload through `MessageContext`. -/
structure DestinationPush
    (signature : Signature n Base denoteBase)
    (vertex edge message : ValueType Base) where
  pullMap : Term signature (MessageContext vertex edge) message
  reduction : signature.Monoid message

namespace DestinationPush

/-- Denotation of the pull/map stage on one traversed edge. -/
def mapAt
    (push : DestinationPush signature vertex edge message)
    (state : Store n (vertex.denote n denoteBase))
    (context : EdgeContext n (edge.denote n denoteBase)) :
    message.denote n denoteBase :=
  Term.evaluate (signature := signature)
    (MessageContext.fromEdge state context) push.pullMap

/-- One destination-filtered push action. -/
def foldContextAt
    (push : DestinationPush signature vertex edge message)
    (state : Store n (vertex.denote n denoteBase))
    (destination : Fin n)
    (accumulator : message.denote n denoteBase)
    (context : EdgeContext n (edge.denote n denoteBase)) :
    message.denote n denoteBase :=
  if context.destination = destination then
    (signature.denoteMonoid push.reduction).combine accumulator
      (push.mapAt state context)
  else
    accumulator

/-- The dense destination reduction produced by Traversal Algebra. -/
def inboxAt
    (push : DestinationPush signature vertex edge message)
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (state : Store n (vertex.denote n denoteBase))
    (frontier : Frontier n)
    (destination : Fin n) : message.denote n denoteBase :=
  TraversalAlgebra.Verified.TraversalAlgebra.reduceByDestinationAt
    graph frontier (signature.denoteMonoid push.reduction)
      (push.mapAt state) destination

end DestinationPush

/-- Closed, barrier-delimited Traversal Algebra programs corresponding to one
Monoidal Frontier BSP superstep.

The graph phase is an explicit destination push. Its dense result is consumed
by a vertex map, after which a typed compact policy constructs the next
frontier. `emit` and source-indexed terminals are deliberately outside this
closed state-transition fragment because they have different observations;
`TypedObservations` proves their separate observer correspondence. -/
structure Program
    (signature : Signature n Base denoteBase)
    (vertex edge message : ValueType Base) where
  destinationPush : DestinationPush signature vertex edge message
  vertexMap : Term signature (UpdateContext vertex message) vertex
  frontierCompact : FrontierPolicy signature vertex

namespace Program

def updateAt
    (program : Program signature vertex edge message)
    (vertexId : Fin n)
    (oldState : vertex.denote n denoteBase)
    (inbox : message.denote n denoteBase) : vertex.denote n denoteBase :=
  Term.evaluate (signature := signature)
    (UpdateContext.environment vertexId oldState inbox) program.vertexMap

end Program

/-- Independent operational denotation of one closed Traversal Algebra
superstep. -/
def step
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : Program signature vertex edge message)
    (configuration : Configuration n (vertex.denote n denoteBase)) :
    Configuration n (vertex.denote n denoteBase) :=
  Configuration.advance program.updateAt program.frontierCompact.denote
    (graph.traversalDestinations configuration.frontier) configuration
    (program.destinationPush.inboxAt graph
      configuration.state configuration.frontier)

/-- Finite execution of the closed Traversal Algebra fragment. -/
def run
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : Program signature vertex edge message) :
    Nat → Configuration n (vertex.denote n denoteBase) →
      Configuration n (vertex.denote n denoteBase)
  | 0, configuration => configuration
  | fuel + 1, configuration =>
      run graph program fuel (step graph program configuration)

/-- Forget the explicit TA stage nesting into the earlier pull--map--push
normal form. This bridge is used only to reuse its independently proved
nested-fold simulation. -/
def toPullMapPush
    (program : Program signature vertex edge message) :
    PullMapPush.Program signature vertex edge message where
  edgeMap := program.destinationPush.pullMap
  reduction := program.destinationPush.reduction
  vertexUpdate := program.vertexMap
  nextFrontier := program.frontierCompact

/-- The independent TA destination terminal denotes the same inbox as the
pull--map--push normal form. -/
theorem inboxAt_toPullMapPush
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : Program signature vertex edge message)
    (state : Store n (vertex.denote n denoteBase))
    (frontier : Frontier n)
    (destination : Fin n) :
    program.destinationPush.inboxAt graph state frontier destination =
      TraversalAlgebra.Verified.PullMapPush.inboxAt graph
        ((toPullMapPush program).denoteAt state) frontier destination := rfl

/-- One TA superstep is exactly its pull--map--push normal-form denotation. -/
theorem step_toPullMapPush
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : Program signature vertex edge message)
    (configuration : Configuration n (vertex.denote n denoteBase)) :
    step graph program configuration =
      PullMapPush.step graph (toPullMapPush program) configuration := rfl

/-- Every finite TA execution agrees with its pull--map--push normal form. -/
theorem run_toPullMapPush
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : Program signature vertex edge message)
    (fuel : Nat)
    (configuration : Configuration n (vertex.denote n denoteBase)) :
    run graph program fuel configuration =
      PullMapPush.run graph (toPullMapPush program) fuel configuration := by
  induction fuel generalizing configuration with
  | zero => rfl
  | succ fuel induction =>
      simp only [run, PullMapPush.run]
      rw [step_toPullMapPush]
      exact induction _

/-- Compile a typed Monoidal Frontier BSP program into the closed TA fragment. -/
def encode
    (program : MonoidalFrontierBSP.Program
      signature vertex edge message) :
    Program signature vertex edge message where
  destinationPush := {
    pullMap := program.messageTerm
    reduction := program.reduction
  }
  vertexMap := program.updateTerm
  frontierCompact := program.frontier

/-- Interpret a closed TA program as a typed Monoidal Frontier BSP program. -/
def decode
    (program : Program signature vertex edge message) :
    MonoidalFrontierBSP.Program signature vertex edge message where
  messageTerm := program.destinationPush.pullMap
  reduction := program.destinationPush.reduction
  updateTerm := program.vertexMap
  frontier := program.frontierCompact

/-- Encoding followed by decoding is the identity on BSP syntax. -/
@[simp]
theorem decode_encode
    (program : MonoidalFrontierBSP.Program
      signature vertex edge message) :
    decode (encode program) = program := by
  cases program
  rfl

/-- Decoding followed by encoding is the identity on the closed TA syntax. -/
@[simp]
theorem encode_decode
    (program : Program signature vertex edge message) :
    encode (decode program) = program := by
  cases program with
  | mk destinationPush vertexMap frontierCompact =>
      cases destinationPush
      rfl

/-- TA encoding is injective on the typed BSP syntax. -/
theorem encode_injective
    (left right : MonoidalFrontierBSP.Program
      signature vertex edge message)
    (equality : encode left = encode right) : left = right := by
  have decoded := congrArg decode equality
  simpa only [decode_encode] using decoded

/-- Every program in the closed TA syntax is the encoding of a typed BSP
program. Together with `encode_injective`, this states the syntactic
bijection. The substantive result below additionally relates the independently
defined nested and flattened transition systems. -/
theorem encode_surjective
    (program : Program signature vertex edge message) :
    ∃ source : MonoidalFrontierBSP.Program
        signature vertex edge message,
      encode source = program :=
  ⟨decode program, encode_decode program⟩

/-- BSP compilation and TA encoding have the same pull--map--push normal
form. -/
theorem toPullMapPush_encode
    (program : MonoidalFrontierBSP.Program
      signature vertex edge message) :
    toPullMapPush (encode program) =
      TraversalAlgebra.Verified.Typed.compile program := rfl

/-- Decoding a TA program and compiling it recovers the TA normal form. -/
theorem compile_decode
    (program : Program signature vertex edge message) :
    TraversalAlgebra.Verified.Typed.compile (decode program) =
      toPullMapPush program := by
  cases program with
  | mk destinationPush vertexMap frontierCompact =>
      cases destinationPush
      rfl

/-- The key operation equation: BSP's independently defined nested message
fold is exactly TA's `traverse -> pull/map -> reduceByDestination` terminal at
every destination. -/
theorem encode_inbox_correct
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : MonoidalFrontierBSP.Program
      signature vertex edge message)
    (state : Store n (vertex.denote n denoteBase))
    (frontier : Frontier n)
    (destination : Fin n) :
    TraversalAlgebra.Verified.MonoidalFrontierBSP.inboxAt
        graph program.denote state frontier destination =
      (encode program).destinationPush.inboxAt
        graph state frontier destination := by
  calc
    TraversalAlgebra.Verified.MonoidalFrontierBSP.inboxAt
        graph program.denote state frontier destination =
      TraversalAlgebra.Verified.PullMapPush.inboxAt graph
        (TraversalAlgebra.Verified.compile program.denote state)
        frontier destination :=
      TraversalAlgebra.Verified.inbox_compile_correct
        graph program.denote state frontier destination
    _ = TraversalAlgebra.Verified.PullMapPush.inboxAt graph
        ((TraversalAlgebra.Verified.Typed.compile program).denoteAt state)
        frontier destination := by
      rw [TraversalAlgebra.Verified.Typed.compile_denoteAt]
    _ = TraversalAlgebra.Verified.PullMapPush.inboxAt graph
        ((toPullMapPush (encode program)).denoteAt state)
        frontier destination := by
      rw [toPullMapPush_encode]
    _ = (encode program).destinationPush.inboxAt
        graph state frontier destination :=
      (inboxAt_toPullMapPush graph (encode program) state
        frontier destination).symm

/-- One-step completeness: encoding any typed BSP program as TA preserves the
entire barrier configuration. -/
theorem encode_step_correct
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : MonoidalFrontierBSP.Program
      signature vertex edge message)
    (configuration : Configuration n (vertex.denote n denoteBase)) :
    MonoidalFrontierBSP.step graph program configuration =
      step graph (encode program) configuration := by
  calc
    MonoidalFrontierBSP.step graph program configuration =
        PullMapPush.step graph
          (TraversalAlgebra.Verified.Typed.compile program) configuration :=
      TraversalAlgebra.Verified.Typed.compile_step_correct
        graph program configuration
    _ = PullMapPush.step graph (toPullMapPush (encode program))
          configuration := by rw [toPullMapPush_encode]
    _ = step graph (encode program) configuration :=
      (step_toPullMapPush graph (encode program) configuration).symm

/-- One-step soundness: decoding any program in the closed TA fragment as BSP
preserves the entire barrier configuration. -/
theorem decode_step_correct
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : Program signature vertex edge message)
    (configuration : Configuration n (vertex.denote n denoteBase)) :
    step graph program configuration =
      MonoidalFrontierBSP.step graph (decode program) configuration := by
  calc
    step graph program configuration =
        step graph (encode (decode program)) configuration := by
      rw [encode_decode]
    _ = MonoidalFrontierBSP.step graph (decode program) configuration :=
      (encode_step_correct graph (decode program) configuration).symm

/-- Completeness direction: every typed Monoidal Frontier BSP program has a
closed TA program with the same complete configuration after every finite
number of supersteps. -/
theorem encode_run_correct
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : MonoidalFrontierBSP.Program
      signature vertex edge message)
    (fuel : Nat)
    (configuration : Configuration n (vertex.denote n denoteBase)) :
    MonoidalFrontierBSP.run graph program fuel configuration =
      run graph (encode program) fuel configuration := by
  induction fuel generalizing configuration with
  | zero => rfl
  | succ fuel induction =>
      simp only [MonoidalFrontierBSP.run, run]
      rw [encode_step_correct]
      exact induction _

/-- Soundness direction: every program in the closed TA fragment has a typed
Monoidal Frontier BSP interpretation with the same complete configuration
after every finite number of supersteps. -/
theorem decode_run_correct
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : Program signature vertex edge message)
    (fuel : Nat)
    (configuration : Configuration n (vertex.denote n denoteBase)) :
    run graph program fuel configuration =
      MonoidalFrontierBSP.run graph (decode program) fuel configuration := by
  induction fuel generalizing configuration with
  | zero => rfl
  | succ fuel induction =>
      simp only [run, MonoidalFrontierBSP.run]
      rw [decode_step_correct]
      exact induction _

/-- At every step index, BSP and its TA encoding agree on the empty-frontier
halting observation. -/
theorem encode_halted_iff
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : MonoidalFrontierBSP.Program
      signature vertex edge message)
    (fuel : Nat)
    (configuration : Configuration n (vertex.denote n denoteBase)) :
    Configuration.Halted
        (MonoidalFrontierBSP.run graph program fuel configuration) ↔
      Configuration.Halted
        (run graph (encode program) fuel configuration) := by
  rw [encode_run_correct]

/-- A BSP program reaches an empty frontier exactly when its TA encoding does. -/
theorem encode_terminates_iff
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : MonoidalFrontierBSP.Program
      signature vertex edge message)
    (configuration : Configuration n (vertex.denote n denoteBase)) :
    (∃ fuel, Configuration.Halted
      (MonoidalFrontierBSP.run graph program fuel configuration)) ↔
    (∃ fuel, Configuration.Halted
      (run graph (encode program) fuel configuration)) := by
  constructor
  · rintro ⟨fuel, halted⟩
    exact ⟨fuel, (encode_halted_iff graph program fuel configuration).mp halted⟩
  · rintro ⟨fuel, halted⟩
    exact ⟨fuel, (encode_halted_iff graph program fuel configuration).mpr halted⟩

/-- The same halting equivalence in the TA-to-BSP direction. -/
theorem decode_terminates_iff
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : Program signature vertex edge message)
    (configuration : Configuration n (vertex.denote n denoteBase)) :
    (∃ fuel, Configuration.Halted
      (run graph program fuel configuration)) ↔
    (∃ fuel, Configuration.Halted
      (MonoidalFrontierBSP.run graph (decode program) fuel configuration)) := by
  constructor
  · rintro ⟨fuel, halted⟩
    refine ⟨fuel, ?_⟩
    rw [← decode_run_correct]
    exact halted
  · rintro ⟨fuel, halted⟩
    refine ⟨fuel, ?_⟩
    rw [decode_run_correct]
    exact halted

end TraversalAlgebra.Verified.Typed.TraversalAlgebra
