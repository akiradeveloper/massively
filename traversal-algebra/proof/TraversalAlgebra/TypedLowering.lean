import TraversalAlgebra.SignatureLowering
import TraversalAlgebra.TypedSimulation

namespace TraversalAlgebra.Verified.Typed

variable {n : Nat} {Base : Type} {denoteBase : Base → Type}
variable {source middle target : Signature n Base denoteBase}
variable {vertex edge message : ValueType Base}

namespace FrontierPolicy

/-- Lower every scalar symbol used by a frontier policy. The graph-control
structure and dense/sparse candidate policy are unchanged. -/
def lower
    (policy : FrontierPolicy source vertex)
    (lowering : Signature.Lowering source target) :
    FrontierPolicy target vertex :=
  match policy with
  | .dense predicate => .dense (predicate.lower lowering)
  | .sparsePreserve predicate =>
      .sparsePreserve (predicate.lower lowering)

/-- Signature lowering preserves the selected frontier exactly. -/
@[simp]
theorem denote_lower
    (policy : FrontierPolicy source vertex)
    (lowering : Signature.Lowering source target)
    (candidates : Frontier n)
    (oldState newState : Store n (vertex.denote n denoteBase)) :
    (policy.lower lowering).denote candidates oldState newState =
      policy.denote candidates oldState newState := by
  cases policy with
  | dense predicate =>
      simp only [lower, denote, Term.evaluate_lower]
  | sparsePreserve predicate =>
      simp only [lower, denote, Term.evaluate_lower]

/-- Frontier-policy lowering respects the identity translation. -/
@[simp]
theorem lower_identity
    (policy : FrontierPolicy source vertex) :
    policy.lower (Signature.Lowering.identity source) = policy := by
  cases policy with
  | dense predicate => simp only [lower, Term.lower_identity]
  | sparsePreserve predicate => simp only [lower, Term.lower_identity]

/-- Frontier-policy lowering respects composition. -/
theorem lower_trans
    (policy : FrontierPolicy source vertex)
    (first : Signature.Lowering source middle)
    (second : Signature.Lowering middle target) :
    policy.lower (first.trans second) =
      (policy.lower first).lower second := by
  cases policy with
  | dense predicate => simp only [lower, Term.lower_trans]
  | sparsePreserve predicate => simp only [lower, Term.lower_trans]

end FrontierPolicy

namespace MonoidalFrontierBSP.Program

/-- Lower all scalar symbols in a source BSP program. -/
def lower
    (program : MonoidalFrontierBSP.Program source vertex edge message)
    (lowering : Signature.Lowering source target) :
    MonoidalFrontierBSP.Program target vertex edge message where
  messageTerm := program.messageTerm.lower lowering
  reduction := lowering.monoid program.reduction
  updateTerm := program.updateTerm.lower lowering
  frontier := program.frontier.lower lowering

/-- A lowered source program has exactly the source program's mathematical
denotation. -/
@[simp]
theorem denote_lower
    (program : MonoidalFrontierBSP.Program source vertex edge message)
    (lowering : Signature.Lowering source target) :
    (program.lower lowering).denote = program.denote := by
  cases program with
  | mk messageTerm reduction updateTerm frontier =>
      simp only [lower, Program.denote, Term.evaluate_lower,
        lowering.monoid_correct reduction]
      congr 1
      funext candidates oldState newState
      exact FrontierPolicy.denote_lower frontier lowering candidates
        oldState newState

/-- Source-program lowering respects the identity translation. -/
@[simp]
theorem lower_identity
    (program : MonoidalFrontierBSP.Program source vertex edge message) :
    program.lower (Signature.Lowering.identity source) = program := by
  cases program with
  | mk messageTerm reduction updateTerm frontier =>
      simp only [lower]
      rw [Term.lower_identity messageTerm, Term.lower_identity updateTerm,
        FrontierPolicy.lower_identity frontier]
      rfl

/-- Source-program lowering respects composition. -/
theorem lower_trans
    (program : MonoidalFrontierBSP.Program source vertex edge message)
    (first : Signature.Lowering source middle)
    (second : Signature.Lowering middle target) :
    program.lower (first.trans second) =
      (program.lower first).lower second := by
  cases program with
  | mk messageTerm reduction updateTerm frontier =>
      simp only [lower]
      rw [Term.lower_trans first second messageTerm,
        Term.lower_trans first second updateTerm,
        FrontierPolicy.lower_trans frontier first second]
      rfl

end MonoidalFrontierBSP.Program

namespace PullMapPush.Program

/-- Lower all scalar symbols in a compiled pull--map--push program. -/
def lower
    (program : PullMapPush.Program source vertex edge message)
    (lowering : Signature.Lowering source target) :
    PullMapPush.Program target vertex edge message where
  edgeMap := program.edgeMap.lower lowering
  reduction := lowering.monoid program.reduction
  vertexUpdate := program.vertexUpdate.lower lowering
  nextFrontier := program.nextFrontier.lower lowering

/-- A lowered target program denotes exactly the same barrier-specialized
pull--map--push plan. -/
@[simp]
theorem denoteAt_lower
    (program : PullMapPush.Program source vertex edge message)
    (lowering : Signature.Lowering source target)
    (state : Store n (vertex.denote n denoteBase)) :
    (program.lower lowering).denoteAt state = program.denoteAt state := by
  cases program with
  | mk edgeMap reduction vertexUpdate nextFrontier =>
      simp only [lower, Program.denoteAt, Term.evaluate_lower,
        lowering.monoid_correct reduction]
      congr 1
      funext candidates oldState newState
      exact FrontierPolicy.denote_lower
        nextFrontier lowering candidates oldState newState

/-- Target-program lowering respects the identity translation. -/
@[simp]
theorem lower_identity
    (program : PullMapPush.Program source vertex edge message) :
    program.lower (Signature.Lowering.identity source) = program := by
  cases program with
  | mk edgeMap reduction vertexUpdate nextFrontier =>
      simp only [lower]
      rw [Term.lower_identity edgeMap, Term.lower_identity vertexUpdate,
        FrontierPolicy.lower_identity nextFrontier]
      rfl

/-- Target-program lowering respects composition. -/
theorem lower_trans
    (program : PullMapPush.Program source vertex edge message)
    (first : Signature.Lowering source middle)
    (second : Signature.Lowering middle target) :
    program.lower (first.trans second) =
      (program.lower first).lower second := by
  cases program with
  | mk edgeMap reduction vertexUpdate nextFrontier =>
      simp only [lower]
      rw [Term.lower_trans first second edgeMap,
        Term.lower_trans first second vertexUpdate,
        FrontierPolicy.lower_trans nextFrontier first second]
      rfl

end PullMapPush.Program

namespace MonoidalFrontierBSP

/-- Signature lowering is observationally invisible for one source-machine
superstep. -/
@[simp]
theorem step_lower
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : Program source vertex edge message)
    (lowering : Signature.Lowering source target)
    (configuration : Configuration n (vertex.denote n denoteBase)) :
    step graph (program.lower lowering) configuration =
      step graph program configuration := by
  unfold step
  rw [Program.denote_lower]

/-- Signature lowering is observationally invisible for every finite source
execution. -/
@[simp]
theorem run_lower
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : Program source vertex edge message)
    (lowering : Signature.Lowering source target)
    (fuel : Nat)
    (configuration : Configuration n (vertex.denote n denoteBase)) :
    run graph (program.lower lowering) fuel configuration =
      run graph program fuel configuration := by
  induction fuel generalizing configuration with
  | zero => rfl
  | succ fuel induction =>
      simp only [run]
      rw [step_lower]
      exact induction _

end MonoidalFrontierBSP

namespace PullMapPush

/-- Signature lowering is observationally invisible for one target-machine
superstep. -/
@[simp]
theorem step_lower
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : Program source vertex edge message)
    (lowering : Signature.Lowering source target)
    (configuration : Configuration n (vertex.denote n denoteBase)) :
    step graph (program.lower lowering) configuration =
      step graph program configuration := by
  unfold step
  rw [Program.denoteAt_lower]

/-- Signature lowering is observationally invisible for every finite target
execution. -/
@[simp]
theorem run_lower
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : Program source vertex edge message)
    (lowering : Signature.Lowering source target)
    (fuel : Nat)
    (configuration : Configuration n (vertex.denote n denoteBase)) :
    run graph (program.lower lowering) fuel configuration =
      run graph program fuel configuration := by
  induction fuel generalizing configuration with
  | zero => rfl
  | succ fuel induction =>
      simp only [run]
      rw [step_lower]
      exact induction _

end PullMapPush

/-- Lowering scalar symbols before or after graph-program compilation gives
the same target syntax. This is the compiler/lowering naturality square. -/
theorem compile_lower
    (program : MonoidalFrontierBSP.Program source vertex edge message)
    (lowering : Signature.Lowering source target) :
    compile (program.lower lowering) =
      (compile program).lower lowering := rfl

/-- The complete source-to-target correctness square remains valid after
lowering into another certified signature. -/
theorem lower_compile_run_correct
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : MonoidalFrontierBSP.Program source vertex edge message)
    (lowering : Signature.Lowering source target)
    (fuel : Nat)
    (configuration : Configuration n (vertex.denote n denoteBase)) :
    MonoidalFrontierBSP.run graph (program.lower lowering)
        fuel configuration =
      PullMapPush.run graph ((compile program).lower lowering)
        fuel configuration := by
  rw [compile_run_correct, compile_lower]

/-- Executing the original source program is equal to executing its compiled
program after any certified target-signature translation. This transport
theorem is orthogonal to source/TA expressive equivalence. -/
theorem compile_lowered_run_correct
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : MonoidalFrontierBSP.Program source vertex edge message)
    (lowering : Signature.Lowering source target)
    (fuel : Nat)
    (configuration : Configuration n (vertex.denote n denoteBase)) :
    MonoidalFrontierBSP.run graph program fuel configuration =
      PullMapPush.run graph ((compile program).lower lowering)
        fuel configuration := by
  calc
    MonoidalFrontierBSP.run graph program fuel configuration =
        PullMapPush.run graph (compile program) fuel configuration :=
      compile_run_correct graph program fuel configuration
    _ = PullMapPush.run graph ((compile program).lower lowering)
          fuel configuration :=
      (PullMapPush.run_lower graph (compile program) lowering
        fuel configuration).symm

end TraversalAlgebra.Verified.Typed
