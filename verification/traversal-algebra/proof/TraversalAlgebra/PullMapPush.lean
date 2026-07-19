import TraversalAlgebra.FrontierBSP

namespace TraversalAlgebra.Verified.PullMapPush

/-- A flat pull--map--push plan for one superstep. `map` is evaluated once per
traversed edge context, and values are pushed into destination inboxes. -/
structure Plan (n : Nat) (VertexState : Type u)
    (EdgePayload : Type v) (Message : Type w) where
  map : EdgeContext n EdgePayload → Message
  reduction : Reduction Message
  lawful : LawfulCommutativeReduction reduction
  update : Fin n → VertexState → Message → VertexState
  select : Frontier n → Store n VertexState → Store n VertexState → Frontier n

/-- One destination-filtered fold step over the flattened edge stream. -/
def foldContextAt
    (plan : Plan n VertexState EdgePayload Message)
    (destination : Fin n)
    (accumulator : Message)
    (context : EdgeContext n EdgePayload) : Message :=
  if context.destination = destination then
    plan.reduction.combine accumulator (plan.map context)
  else
    accumulator

/-- A fold whose per-item actions commute is invariant under permutation.
This elementary lemma makes the scheduling assumption explicit instead of
relying on an implementation-specific reduction order. -/
private theorem foldl_eq_of_perm
    {Item : Type u} {Accumulator : Type v}
    (step : Accumulator → Item → Accumulator)
    (commute : ∀ accumulator left right,
      step (step accumulator left) right =
        step (step accumulator right) left)
    {left right : List Item}
    (permutation : left.Perm right)
    (initial : Accumulator) :
    left.foldl step initial = right.foldl step initial := by
  induction permutation generalizing initial with
  | nil => rfl
  | cons item permutation induction =>
      simp only [List.foldl]
      exact induction _
  | swap left right tail =>
      simp only [List.foldl]
      rw [commute]
  | trans first second firstInduction secondInduction =>
      exact (firstInduction initial).trans (secondInduction initial)

/-- Destination-filtered monoidal pushes commute pairwise. -/
theorem foldContextAt_commutes
    (plan : Plan n VertexState EdgePayload Message)
    (destination : Fin n)
    (accumulator : Message)
    (left right : EdgeContext n EdgePayload) :
    foldContextAt plan destination
        (foldContextAt plan destination accumulator left) right =
      foldContextAt plan destination
        (foldContextAt plan destination accumulator right) left := by
  unfold foldContextAt
  by_cases leftMatches : left.destination = destination
  · by_cases rightMatches : right.destination = destination
    · simp only [leftMatches, rightMatches, if_true]
      calc
        plan.reduction.combine
            (plan.reduction.combine accumulator (plan.map left))
            (plan.map right) =
          plan.reduction.combine accumulator
            (plan.reduction.combine (plan.map left) (plan.map right)) :=
          plan.lawful.associative _ _ _
        _ = plan.reduction.combine accumulator
            (plan.reduction.combine (plan.map right) (plan.map left)) :=
          congrArg (plan.reduction.combine accumulator)
            (plan.lawful.commutative _ _)
        _ = plan.reduction.combine
            (plan.reduction.combine accumulator (plan.map right))
            (plan.map left) :=
          (plan.lawful.associative _ _ _).symm
    · simp [leftMatches, rightMatches]
  · by_cases rightMatches : right.destination = destination
    · simp [leftMatches, rightMatches]
    · simp [leftMatches, rightMatches]

/-- A destination push has the same value under every permutation of the
flattened edge stream. -/
theorem foldContextAt_perm
    (plan : Plan n VertexState EdgePayload Message)
    (destination : Fin n)
    {contexts scheduled : List (EdgeContext n EdgePayload)}
    (permutation : contexts.Perm scheduled)
    (initial : Message) :
    contexts.foldl (foldContextAt plan destination) initial =
      scheduled.foldl (foldContextAt plan destination) initial := by
  exact foldl_eq_of_perm (foldContextAt plan destination)
    (foldContextAt_commutes plan destination) permutation initial

/-- Destination push in the target semantics. -/
def inboxAt
    (graph : OrderedGraph n EdgePayload)
    (plan : Plan n VertexState EdgePayload Message)
    (frontier : Frontier n)
    (destination : Fin n) : Message :=
  (graph.traverse frontier).foldl (foldContextAt plan destination)
    plan.reduction.identity

/-- Any parallel schedule that is a permutation of the logical traversal
produces the same destination inbox. -/
theorem inboxAt_schedule_independent
    (graph : OrderedGraph n EdgePayload)
    (plan : Plan n VertexState EdgePayload Message)
    (frontier : Frontier n)
    (destination : Fin n)
    (scheduled : List (EdgeContext n EdgePayload))
    (permutation : (graph.traverse frontier).Perm scheduled) :
    inboxAt graph plan frontier destination =
      scheduled.foldl (foldContextAt plan destination)
        plan.reduction.identity := by
  unfold inboxAt
  exact foldContextAt_perm plan destination permutation _

/-- One target pull--map--push transition. -/
def step
    (graph : OrderedGraph n EdgePayload)
    (plan : Plan n VertexState EdgePayload Message)
    (configuration : Configuration n VertexState) :
    Configuration n VertexState :=
  Configuration.advance plan.update plan.select
    (graph.traversalDestinations configuration.frontier) configuration
    (inboxAt graph plan configuration.frontier)

end TraversalAlgebra.Verified.PullMapPush
