import TraversalAlgebra.PullMapPush

namespace TraversalAlgebra.Verified

open MonoidalFrontierBSP PullMapPush

/-- Compile one BSP superstep at a particular barrier state. Pulling source and
destination state is made explicit in the target edge map. -/
def compile
    (program : MonoidalFrontierBSP.Program n VertexState EdgePayload Message)
    (state : Store n VertexState) :
    PullMapPush.Plan n VertexState EdgePayload Message where
  map := fun context =>
    program.message context.source context.destination
      (state context.source) (state context.destination) context.payload
  reduction := program.reduction
  lawful := program.lawful
  update := program.update
  select := program.select

/-- A compiled target transition instantiates pulls at each barrier state. -/
def compiledStep
    (graph : OrderedGraph n EdgePayload)
    (program : MonoidalFrontierBSP.Program n VertexState EdgePayload Message)
    (configuration : Configuration n VertexState) :
    Configuration n VertexState :=
  PullMapPush.step graph (compile program configuration.state) configuration

/-- Finite execution of the compiled target. -/
def compiledRun
    (graph : OrderedGraph n EdgePayload)
    (program : MonoidalFrontierBSP.Program n VertexState EdgePayload Message) :
    Nat → Configuration n VertexState → Configuration n VertexState
  | 0, configuration => configuration
  | fuel + 1, configuration =>
      compiledRun graph program fuel (compiledStep graph program configuration)

/-- Flattening one adjacency row preserves the source machine's fold. -/
theorem fold_expand_compile
    (graph : OrderedGraph n EdgePayload)
    (program : MonoidalFrontierBSP.Program n VertexState EdgePayload Message)
    (state : Store n VertexState)
    (destination source : Fin n)
    (initial : Message) :
    (graph.expand source).foldl
        (PullMapPush.foldContextAt (compile program state) destination) initial =
      MonoidalFrontierBSP.foldOutgoingAt
        graph program state destination source initial := by
  unfold OrderedGraph.expand MonoidalFrontierBSP.foldOutgoingAt
  generalize graph.outgoing source = edges
  induction edges generalizing initial with
  | nil => rfl
  | cons edge edges induction =>
      simp only [List.map, List.foldl, PullMapPush.foldContextAt, compile]
      exact induction _

/-- Flattening the nested frontier/adjacency scan preserves every accumulator,
not only the monoid identity. -/
theorem fold_traverse_compile
    (graph : OrderedGraph n EdgePayload)
    (program : MonoidalFrontierBSP.Program n VertexState EdgePayload Message)
    (state : Store n VertexState)
    (destination : Fin n)
    (frontier : Frontier n)
    (initial : Message) :
    (graph.traverse frontier).foldl
        (PullMapPush.foldContextAt (compile program state) destination) initial =
      frontier.foldl
        (fun accumulator source =>
          MonoidalFrontierBSP.foldOutgoingAt
            graph program state destination source accumulator)
        initial := by
  induction frontier generalizing initial with
  | nil => rfl
  | cons source frontier induction =>
      simp only [OrderedGraph.traverse_cons, List.foldl_append]
      rw [fold_expand_compile]
      exact induction _

/-- The independently defined source inbox equals destination push in the
compiled pull--map--push plan. -/
theorem inbox_compile_correct
    (graph : OrderedGraph n EdgePayload)
    (program : MonoidalFrontierBSP.Program n VertexState EdgePayload Message)
    (state : Store n VertexState)
    (frontier : Frontier n)
    (destination : Fin n) :
    MonoidalFrontierBSP.inboxAt graph program state frontier destination =
      PullMapPush.inboxAt graph (compile program state) frontier destination := by
  symm
  exact fold_traverse_compile graph program state destination frontier
    program.reduction.identity

/-- The source inbox also agrees with every permutation of the compiled edge
stream, connecting the ordered source semantics to admissible parallel
schedules. -/
theorem inbox_schedule_compile_correct
    (graph : OrderedGraph n EdgePayload)
    (program : MonoidalFrontierBSP.Program n VertexState EdgePayload Message)
    (state : Store n VertexState)
    (frontier : Frontier n)
    (destination : Fin n)
    (scheduled : List (EdgeContext n EdgePayload))
    (permutation : (graph.traverse frontier).Perm scheduled) :
    MonoidalFrontierBSP.inboxAt graph program state frontier destination =
      scheduled.foldl
        (PullMapPush.foldContextAt (compile program state) destination)
        program.reduction.identity := by
  rw [inbox_compile_correct]
  exact PullMapPush.inboxAt_schedule_independent
    graph (compile program state) frontier destination scheduled permutation

/-- One-step forward simulation from Monoidal Frontier BSP to compiled
pull--map--push semantics. -/
theorem compile_step_correct
    (graph : OrderedGraph n EdgePayload)
    (program : MonoidalFrontierBSP.Program n VertexState EdgePayload Message)
    (configuration : Configuration n VertexState) :
    MonoidalFrontierBSP.step graph program configuration =
      compiledStep graph program configuration := by
  unfold MonoidalFrontierBSP.step compiledStep PullMapPush.step
  rw [graph.traversalDestinations_eq_nestedDestinations
    configuration.frontier]
  change Configuration.advance program.update program.select
      (graph.nestedDestinations configuration.frontier) configuration _ =
    Configuration.advance program.update program.select
      (graph.nestedDestinations configuration.frontier) configuration _
  congr 1
  funext destination
  exact inbox_compile_correct graph program configuration.state
    configuration.frontier destination

/-- Finite-execution forward simulation for every fuel and initial
configuration. -/
theorem compile_run_correct
    (graph : OrderedGraph n EdgePayload)
    (program : MonoidalFrontierBSP.Program n VertexState EdgePayload Message)
    (fuel : Nat)
    (configuration : Configuration n VertexState) :
    MonoidalFrontierBSP.run graph program fuel configuration =
      compiledRun graph program fuel configuration := by
  induction fuel generalizing configuration with
  | zero => rfl
  | succ fuel induction =>
      simp only [MonoidalFrontierBSP.run, compiledRun]
      rw [compile_step_correct]
      exact induction _

end TraversalAlgebra.Verified
