import TraversalAlgebra.TypedProgram

namespace TraversalAlgebra.Verified.Typed

variable {n : Nat} {Base : Type} {denoteBase : Base → Type}
variable {signature : Signature n Base denoteBase}
variable {vertex edge message : ValueType Base}

/-- The syntax-directed compiler commutes with denotation at every barrier
state. This is the bridge from the closed typed target program to the semantic
plan used by the flattening proof. -/
theorem compile_denoteAt
    (program : MonoidalFrontierBSP.Program signature vertex edge message)
    (state : Store n (vertex.denote n denoteBase)) :
    TraversalAlgebra.Verified.compile program.denote state =
      (compile program).denoteAt state := rfl

/-- A symbolic typed source program and its compiled symbolic typed target
have equal complete configurations after one superstep. -/
theorem compile_step_correct
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : MonoidalFrontierBSP.Program signature vertex edge message)
    (configuration : Configuration n (vertex.denote n denoteBase)) :
    MonoidalFrontierBSP.step graph program configuration =
      PullMapPush.step graph (compile program) configuration := by
  unfold MonoidalFrontierBSP.step PullMapPush.step
  rw [TraversalAlgebra.Verified.compile_step_correct]
  unfold TraversalAlgebra.Verified.compiledStep
  rw [compile_denoteAt]

/-- Compiler correctness for every finite number of supersteps. Unlike the
earlier semantic theorem, both sides here are interpretations of symbolic
typed syntax relative to the fixed signature. -/
theorem compile_run_correct
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : MonoidalFrontierBSP.Program signature vertex edge message)
    (fuel : Nat)
    (configuration : Configuration n (vertex.denote n denoteBase)) :
    MonoidalFrontierBSP.run graph program fuel configuration =
      PullMapPush.run graph (compile program) fuel configuration := by
  induction fuel generalizing configuration with
  | zero => rfl
  | succ fuel induction =>
      simp only [MonoidalFrontierBSP.run, PullMapPush.run]
      rw [compile_step_correct]
      exact induction _

/-- The typed source inbox agrees with every permutation of its compiled edge
stream. -/
theorem inbox_schedule_correct
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : MonoidalFrontierBSP.Program signature vertex edge message)
    (state : Store n (vertex.denote n denoteBase))
    (frontier : Frontier n)
    (destination : Fin n)
    (scheduled : List (EdgeContext n (edge.denote n denoteBase)))
    (permutation : (graph.traverse frontier).Perm scheduled) :
    TraversalAlgebra.Verified.MonoidalFrontierBSP.inboxAt
        graph program.denote state frontier destination =
      scheduled.foldl
        (TraversalAlgebra.Verified.PullMapPush.foldContextAt
          ((compile program).denoteAt state) destination)
        (signature.denoteMonoid program.reduction).identity := by
  rw [← compile_denoteAt]
  exact TraversalAlgebra.Verified.inbox_schedule_compile_correct
    graph program.denote state frontier destination scheduled permutation

end TraversalAlgebra.Verified.Typed
