import TraversalAlgebra.TypedTraversalAlgebra

namespace TraversalAlgebra.Verified.Typed

variable {n : Nat} {Base : Type} {denoteBase : Base → Type}
variable {signature : Signature n Base denoteBase}
variable {vertex edge message : ValueType Base}

namespace Term

/-- Bind one typed argument exactly once for a unary term.

Unlike tree substitution, this sharing-preserving scalar fusion never copies
the input syntax when the unary body reads its variable more than once. -/
def share
    {context : List (ValueType Base)} {input output : ValueType Base}
    (outer : Term signature [input] output)
    (inner : Term signature context input) :
    Term signature context output := .let1 inner outer

/-- Denotation of a shared binding is ordinary functional composition. -/
theorem evaluate_share
    {context : List (ValueType Base)} {input output : ValueType Base}
    (outer : Term signature [input] output)
    (inner : Term signature context input)
    (environment : Environment n denoteBase context) :
    Term.evaluate environment (share outer inner) =
      Term.evaluate (.cons (Term.evaluate environment inner) .nil) outer := rfl

end Term

namespace TraversalAlgebra.Expression

/-- An independently compositional expression over the edge stream selected
by `traverse`.

The `read` leaf is intrinsically restricted to the five public pulls from the
current edge context and vertex store. `literal` is a scalar constant, `map`
is an additional pointwise scalar stage, and `zip` combines two pointwise columns.
Unlike the BSP record and the earlier TA normal form, this syntax admits
arbitrarily nested map/zip trees and contains no arbitrary edge callback or
pre-normalized edge term. -/
inductive Traversal
    (signature : Signature n Base denoteBase)
    (vertex edge : ValueType Base) : ValueType Base → Type
  | read {output : ValueType Base} :
      Variable (MessageContext vertex edge) output →
      Traversal signature vertex edge output
  | literal {output : ValueType Base} :
      signature.Literal output → Traversal signature vertex edge output
  | map {input output : ValueType Base} :
      Traversal signature vertex edge input →
      Term signature [input] output →
      Traversal signature vertex edge output
  | zip {left right : ValueType Base} :
      Traversal signature vertex edge left →
      Traversal signature vertex edge right →
      Traversal signature vertex edge (.product left right)

namespace Traversal

/-- Evaluate one compositional traversal expression at one traversed edge. -/
def evaluateAt
    {output : ValueType Base}
    (expression : Traversal signature vertex edge output)
    (state : Store n (vertex.denote n denoteBase))
    (context : EdgeContext n (edge.denote n denoteBase)) :
    output.denote n denoteBase :=
  match expression with
  | .read reference =>
      reference.lookup (MessageContext.fromEdge state context)
  | .literal symbol => signature.denoteLiteral symbol
  | .map input mapper =>
      Term.evaluate (.cons (evaluateAt input state context) .nil) mapper
  | .zip left right =>
      (evaluateAt left state context, evaluateAt right state context)

/-- Evaluate the complete pointwise edge stream. This definition invokes the
TA `traverse` operation directly and does not mention BSP or normalization. -/
def evaluate
    {output : ValueType Base}
    (expression : Traversal signature vertex edge output)
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (state : Store n (vertex.denote n denoteBase))
    (frontier : Frontier n) : List (output.denote n denoteBase) :=
  (graph.traverse frontier).map (evaluateAt expression state)

/-- Fuse an arbitrary traversal map/zip tree into one typed edge term. -/
def normalize
    {output : ValueType Base}
    (expression : Traversal signature vertex edge output) :
    Term signature (MessageContext vertex edge) output :=
  match expression with
  | .read reference => .read reference
  | .literal symbol => .literal symbol
  | .map input mapper => Term.share mapper (normalize input)
  | .zip left right => .pair (normalize left) (normalize right)

/-- Pointwise semantic preservation of traversal-expression normalization. -/
theorem evaluateAt_normalize
    {output : ValueType Base}
    (expression : Traversal signature vertex edge output)
    (state : Store n (vertex.denote n denoteBase))
    (context : EdgeContext n (edge.denote n denoteBase)) :
    Term.evaluate (MessageContext.fromEdge state context)
        (normalize expression) =
      evaluateAt expression state context := by
  induction expression with
  | read reference => rfl
  | literal symbol => rfl
  | map input mapper induction =>
      rw [normalize, Term.evaluate_share, induction]
      rfl
  | zip left right leftInduction rightInduction =>
      simp only [normalize, Term.evaluate, evaluateAt,
        leftInduction, rightInduction]

/-- Whole-stream semantic preservation follows pointwise. -/
theorem evaluate_normalize
    {output : ValueType Base}
    (expression : Traversal signature vertex edge output)
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (state : Store n (vertex.denote n denoteBase))
    (frontier : Frontier n) :
    (graph.traverse frontier).map
        (fun context => Term.evaluate
          (MessageContext.fromEdge state context) (normalize expression)) =
      evaluate expression graph state frontier := by
  unfold evaluate
  apply List.map_congr_left
  intro context membership
  exact evaluateAt_normalize expression state context

/-- Reify a single-edge normal-form term back into the structural TA
expression language. Primitive applications and shared bindings become unary
`map` nodes, while pairs become `zip` nodes. -/
def ofTerm
    {output : ValueType Base}
    (term : Term signature (MessageContext vertex edge) output) :
    Traversal signature vertex edge output :=
  match term with
  | .read reference => .read reference
  | .literal symbol => .literal symbol
  | .pair left right => .zip (ofTerm left) (ofTerm right)
  | .apply primitive argument =>
      .map (ofTerm argument) (.apply primitive (.read .here))
  | .let1 input body => .map (ofTerm input) body

/-- Reification evaluates exactly like the normal-form term. -/
theorem evaluateAt_ofTerm :
    {output : ValueType Base} →
    (term : Term signature (MessageContext vertex edge) output) →
    (state : Store n (vertex.denote n denoteBase)) →
    (context : EdgeContext n (edge.denote n denoteBase)) →
      evaluateAt (ofTerm term) state context =
        Term.evaluate (MessageContext.fromEdge state context) term
  | _, .read reference, _, _ => by
      simp only [ofTerm, evaluateAt, Term.evaluate]
  | _, .literal symbol, _, _ => by
      simp only [ofTerm, evaluateAt, Term.evaluate]
  | _, .pair left right, state, context => by
      simp only [ofTerm, evaluateAt, Term.evaluate,
        evaluateAt_ofTerm left state context,
        evaluateAt_ofTerm right state context]
  | _, .apply primitive argument, state, context => by
      simp only [ofTerm, evaluateAt, Term.evaluate,
        evaluateAt_ofTerm argument state context, Variable.lookup]
  | _, .let1 input body, state, context => by
      simp only [ofTerm, evaluateAt, Term.evaluate,
        evaluateAt_ofTerm input state context]

/-- Normalizing a reified term may introduce explicit sharing nodes, so the
round trip is semantic rather than syntactic. -/
theorem evaluate_normalize_ofTerm
    {output : ValueType Base}
    (term : Term signature (MessageContext vertex edge) output)
    (state : Store n (vertex.denote n denoteBase))
    (context : EdgeContext n (edge.denote n denoteBase)) :
    Term.evaluate (MessageContext.fromEdge state context)
        (normalize (ofTerm term)) =
      Term.evaluate (MessageContext.fromEdge state context) term := by
  calc
    Term.evaluate (MessageContext.fromEdge state context)
        (normalize (ofTerm term)) =
      evaluateAt (ofTerm term) state context :=
        evaluateAt_normalize (ofTerm term) state context
    _ = Term.evaluate (MessageContext.fromEdge state context) term :=
      evaluateAt_ofTerm term state context

/-- Structural source-identifier pull. -/
def sourceId : Traversal signature vertex edge .index :=
  .read MessageContext.sourceId

/-- Structural destination-identifier pull. -/
def destinationId : Traversal signature vertex edge .index :=
  .read MessageContext.destinationId

/-- Pull the current source-vertex state. -/
def sourceState : Traversal signature vertex edge vertex :=
  .read MessageContext.sourceState

/-- Pull the current destination-vertex state. -/
def destinationState : Traversal signature vertex edge vertex :=
  .read MessageContext.destinationState

/-- Pull the current edge payload. -/
def edgePayload : Traversal signature vertex edge edge :=
  .read MessageContext.edgePayload

end Traversal

/-- A destination-indexed TA terminal over a compositional traversal
expression. -/
inductive Destination
    (signature : Signature n Base denoteBase)
    (vertex edge : ValueType Base) : ValueType Base → Type
  | reduceByDestination {output : ValueType Base} :
      Traversal signature vertex edge output →
      signature.Monoid output →
      Destination signature vertex edge output

namespace Destination

/-- Independent denotation of one destination inbox. -/
def inboxAt
    (expression : Destination signature vertex edge message)
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (state : Store n (vertex.denote n denoteBase))
    (frontier : Frontier n)
    (destination : Fin n) : message.denote n denoteBase :=
  match expression with
  | .reduceByDestination traversal reduction =>
      TraversalAlgebra.Verified.TraversalAlgebra.reduceByDestinationAt
        graph frontier (signature.denoteMonoid reduction)
          (traversal.evaluateAt state) destination

/-- Normalize the traversal expression beneath the destination terminal. -/
def normalize
    (expression : Destination signature vertex edge message) :
    TraversalAlgebra.DestinationPush signature vertex edge message :=
  match expression with
  | .reduceByDestination traversal reduction =>
      { pullMap := traversal.normalize, reduction }

/-- Destination reduction is preserved by pointwise map/zip fusion. -/
theorem inboxAt_normalize
    (expression : Destination signature vertex edge message)
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (state : Store n (vertex.denote n denoteBase))
    (frontier : Frontier n)
    (destination : Fin n) :
    inboxAt expression graph state frontier destination =
      (normalize expression).inboxAt graph state frontier destination := by
  cases expression with
  | reduceByDestination traversal reduction =>
      unfold inboxAt normalize TraversalAlgebra.DestinationPush.inboxAt
      unfold TraversalAlgebra.DestinationPush.mapAt
      apply congrArg (fun mapper =>
        TraversalAlgebra.Verified.TraversalAlgebra.reduceByDestinationAt
          graph frontier (signature.denoteMonoid reduction) mapper destination)
      funext context
      exact (Traversal.evaluateAt_normalize traversal state context).symm

end Destination

/-- A destination reduction followed by the dense vertex-state map. -/
inductive VertexStage
    (signature : Signature n Base denoteBase)
    (vertex edge message : ValueType Base) : Type
  | updateByDestination :
      Destination signature vertex edge message →
      Term signature (UpdateContext vertex message) vertex →
      VertexStage signature vertex edge message

/-- Independent, nested syntax of the closed frontier-to-frontier TA
fragment. Its constructors are algebra operations rather than BSP fields. -/
inductive Program
    (signature : Signature n Base denoteBase)
    (vertex edge message : ValueType Base) : Type
  | compact :
      VertexStage signature vertex edge message →
      FrontierPolicy signature vertex →
      Program signature vertex edge message

namespace Program

/-- Operational semantics of one independent TA expression superstep. -/
def step
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : Program signature vertex edge message)
    (configuration : Configuration n (vertex.denote n denoteBase)) :
    Configuration n (vertex.denote n denoteBase) :=
  match program with
  | .compact (.updateByDestination destination update) frontier =>
      Configuration.advance
        (fun vertexId oldState inbox =>
          Term.evaluate
            (UpdateContext.environment vertexId oldState inbox) update)
        frontier.denote (graph.traversalDestinations configuration.frontier)
          configuration
        (destination.inboxAt graph configuration.state configuration.frontier)

/-- Finite execution of the independent TA expression language. -/
def run
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : Program signature vertex edge message) :
    Nat → Configuration n (vertex.denote n denoteBase) →
      Configuration n (vertex.denote n denoteBase)
  | 0, configuration => configuration
  | fuel + 1, configuration =>
      run graph program fuel (step graph program configuration)

/-- Normalize a nested TA expression into the previously verified single-push
normal form. -/
def normalize
    (program : Program signature vertex edge message) :
    TraversalAlgebra.Program signature vertex edge message :=
  match program with
  | .compact (.updateByDestination destination update) frontier =>
      { destinationPush := destination.normalize
        vertexMap := update
        frontierCompact := frontier }

/-- Reify one normal-form program as an independent operator tree. -/
def ofNormalForm
    (program : TraversalAlgebra.Program signature vertex edge message) :
    Program signature vertex edge message :=
  .compact
    (.updateByDestination
      (.reduceByDestination
        (Traversal.ofTerm program.destinationPush.pullMap)
        program.destinationPush.reduction)
      program.vertexMap)
    program.frontierCompact

/-- Reification of a normal form preserves one complete superstep. Explicit
sharing nodes may make the re-normalized syntax different. -/
theorem ofNormalForm_step_correct
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : TraversalAlgebra.Program signature vertex edge message)
    (configuration : Configuration n (vertex.denote n denoteBase)) :
    step graph (ofNormalForm program) configuration =
      TraversalAlgebra.step graph program configuration := by
  cases program with
  | mk destinationPush vertexMap frontierCompact =>
      cases destinationPush with
      | mk pullMap reduction =>
          simp only [step, ofNormalForm, TraversalAlgebra.step]
          have inboxEquality :
              (Destination.reduceByDestination
                (Traversal.ofTerm pullMap) reduction).inboxAt
                  graph configuration.state configuration.frontier =
                ({ pullMap := pullMap, reduction := reduction } :
                  TraversalAlgebra.DestinationPush
                    signature vertex edge message).inboxAt
                  graph configuration.state configuration.frontier := by
            funext destination
            unfold Destination.inboxAt
            unfold TraversalAlgebra.DestinationPush.inboxAt
            unfold TraversalAlgebra.DestinationPush.mapAt
            apply congrArg (fun mapper =>
              TraversalAlgebra.Verified.TraversalAlgebra.reduceByDestinationAt
                graph configuration.frontier
                  (signature.denoteMonoid reduction) mapper destination)
            funext context
            exact Traversal.evaluateAt_ofTerm pullMap
              configuration.state context
          rw [inboxEquality]
          rfl

/-- Reification preserves every finite normal-form execution. -/
theorem ofNormalForm_run_correct
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : TraversalAlgebra.Program signature vertex edge message)
    (fuel : Nat)
    (configuration : Configuration n (vertex.denote n denoteBase)) :
    run graph (ofNormalForm program) fuel configuration =
      TraversalAlgebra.run graph program fuel configuration := by
  induction fuel generalizing configuration with
  | zero => rfl
  | succ fuel induction =>
      simp only [run, TraversalAlgebra.run]
      rw [ofNormalForm_step_correct]
      exact induction _

/-- One-step semantics is preserved by normalization. -/
theorem normalize_step_correct
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : Program signature vertex edge message)
    (configuration : Configuration n (vertex.denote n denoteBase)) :
    step graph program configuration =
      TraversalAlgebra.step graph (normalize program) configuration := by
  cases program with
  | compact vertexStage frontier =>
      cases vertexStage with
      | updateByDestination destination update =>
          simp only [step, normalize, TraversalAlgebra.step]
          have inboxEquality :
              destination.inboxAt graph configuration.state
                  configuration.frontier =
                destination.normalize.inboxAt graph configuration.state
                  configuration.frontier := by
            funext destinationId
            exact Destination.inboxAt_normalize destination graph
              configuration.state configuration.frontier destinationId
          rw [inboxEquality]
          rfl

/-- Every finite execution is preserved by normalization. -/
theorem normalize_run_correct
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : Program signature vertex edge message)
    (fuel : Nat)
    (configuration : Configuration n (vertex.denote n denoteBase)) :
    run graph program fuel configuration =
      TraversalAlgebra.run graph (normalize program) fuel configuration := by
  induction fuel generalizing configuration with
  | zero => rfl
  | succ fuel induction =>
      simp only [run, TraversalAlgebra.run]
      rw [normalize_step_correct]
      exact induction _

/-- Normalization followed by reification preserves an arbitrary expression's
finite-run semantics, although it generally changes its syntax. -/
theorem reify_normalize_run_correct
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : Program signature vertex edge message)
    (fuel : Nat)
    (configuration : Configuration n (vertex.denote n denoteBase)) :
    run graph (ofNormalForm (normalize program)) fuel configuration =
      run graph program fuel configuration := by
  calc
    run graph (ofNormalForm (normalize program)) fuel configuration =
        TraversalAlgebra.run graph (normalize program) fuel configuration :=
      ofNormalForm_run_correct graph (normalize program) fuel configuration
    _ = run graph program fuel configuration :=
      (normalize_run_correct graph program fuel configuration).symm

/-- Compile an arbitrary closed TA expression to typed Monoidal Frontier BSP. -/
def toBSP
    (program : Program signature vertex edge message) :
    MonoidalFrontierBSP.Program signature vertex edge message :=
  TraversalAlgebra.decode (normalize program)

/-- Embed a typed Monoidal Frontier BSP program as a closed TA expression. -/
def ofBSP
    (program : MonoidalFrontierBSP.Program signature vertex edge message) :
    Program signature vertex edge message :=
  ofNormalForm (TraversalAlgebra.encode program)

/-- Soundness of normalization: every independent closed TA expression has a
typed BSP interpretation with identical complete finite executions. -/
theorem toBSP_run_correct
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : Program signature vertex edge message)
    (fuel : Nat)
    (configuration : Configuration n (vertex.denote n denoteBase)) :
    run graph program fuel configuration =
      MonoidalFrontierBSP.run graph (toBSP program) fuel configuration := by
  calc
    run graph program fuel configuration =
        TraversalAlgebra.run graph (normalize program) fuel configuration :=
      normalize_run_correct graph program fuel configuration
    _ = MonoidalFrontierBSP.run graph
        (TraversalAlgebra.decode (normalize program)) fuel configuration :=
      TraversalAlgebra.decode_run_correct graph (normalize program)
        fuel configuration
    _ = MonoidalFrontierBSP.run graph (toBSP program) fuel configuration := rfl

/-- Completeness: every typed BSP program has an independent closed TA
expression with identical complete finite executions. -/
theorem ofBSP_run_correct
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : MonoidalFrontierBSP.Program signature vertex edge message)
    (fuel : Nat)
    (configuration : Configuration n (vertex.denote n denoteBase)) :
    MonoidalFrontierBSP.run graph program fuel configuration =
      run graph (ofBSP program) fuel configuration := by
  calc
    MonoidalFrontierBSP.run graph program fuel configuration =
        TraversalAlgebra.run graph (TraversalAlgebra.encode program)
          fuel configuration :=
      TraversalAlgebra.encode_run_correct graph program fuel configuration
    _ = run graph (ofNormalForm (TraversalAlgebra.encode program))
        fuel configuration :=
      (ofNormalForm_run_correct graph (TraversalAlgebra.encode program)
        fuel configuration).symm
    _ = run graph (ofBSP program) fuel configuration := rfl

/-- The BSP round trip is semantic. Sharing-preserving normalization may add
`let1` nodes, so syntactic equality is neither required nor claimed. -/
theorem toBSP_ofBSP_run_correct
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : MonoidalFrontierBSP.Program signature vertex edge message)
    (fuel : Nat)
    (configuration : Configuration n (vertex.denote n denoteBase)) :
    MonoidalFrontierBSP.run graph (toBSP (ofBSP program))
        fuel configuration =
      MonoidalFrontierBSP.run graph program fuel configuration := by
  calc
    MonoidalFrontierBSP.run graph (toBSP (ofBSP program))
        fuel configuration =
      run graph (ofBSP program) fuel configuration :=
        (toBSP_run_correct graph (ofBSP program) fuel configuration).symm
    _ = MonoidalFrontierBSP.run graph program fuel configuration :=
      (ofBSP_run_correct graph program fuel configuration).symm

/-- Existential form of TA-to-BSP representability. -/
theorem ta_to_bsp_complete
    (program : Program signature vertex edge message) :
    ∃ source : MonoidalFrontierBSP.Program signature vertex edge message,
      ∀ (graph : OrderedGraph n (edge.denote n denoteBase))
        (fuel : Nat)
        (configuration : Configuration n (vertex.denote n denoteBase)),
        run graph program fuel configuration =
          MonoidalFrontierBSP.run graph source fuel configuration :=
  ⟨toBSP program, fun graph fuel configuration =>
    toBSP_run_correct graph program fuel configuration⟩

/-- Existential form of BSP-to-TA representability. -/
theorem bsp_to_ta_complete
    (source : MonoidalFrontierBSP.Program signature vertex edge message) :
    ∃ program : Program signature vertex edge message,
      ∀ (graph : OrderedGraph n (edge.denote n denoteBase))
        (fuel : Nat)
        (configuration : Configuration n (vertex.denote n denoteBase)),
        MonoidalFrontierBSP.run graph source fuel configuration =
          run graph program fuel configuration :=
  ⟨ofBSP source, fun graph fuel configuration =>
    ofBSP_run_correct graph source fuel configuration⟩

/-- Empty-frontier termination is preserved by normalization to BSP. -/
theorem toBSP_terminates_iff
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : Program signature vertex edge message)
    (configuration : Configuration n (vertex.denote n denoteBase)) :
    (∃ fuel, Configuration.Halted
      (run graph program fuel configuration)) ↔
    (∃ fuel, Configuration.Halted
      (MonoidalFrontierBSP.run graph (toBSP program) fuel configuration)) := by
  constructor
  · rintro ⟨fuel, halted⟩
    refine ⟨fuel, ?_⟩
    rw [← toBSP_run_correct]
    exact halted
  · rintro ⟨fuel, halted⟩
    refine ⟨fuel, ?_⟩
    rw [toBSP_run_correct]
    exact halted

/-- Empty-frontier termination is preserved by the BSP embedding. -/
theorem ofBSP_terminates_iff
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : MonoidalFrontierBSP.Program signature vertex edge message)
    (configuration : Configuration n (vertex.denote n denoteBase)) :
    (∃ fuel, Configuration.Halted
      (MonoidalFrontierBSP.run graph program fuel configuration)) ↔
    (∃ fuel, Configuration.Halted
      (run graph (ofBSP program) fuel configuration)) := by
  constructor
  · rintro ⟨fuel, halted⟩
    refine ⟨fuel, ?_⟩
    rw [← ofBSP_run_correct]
    exact halted
  · rintro ⟨fuel, halted⟩
    refine ⟨fuel, ?_⟩
    rw [ofBSP_run_correct]
    exact halted

end Program
end TraversalAlgebra.Expression
end TraversalAlgebra.Verified.Typed
