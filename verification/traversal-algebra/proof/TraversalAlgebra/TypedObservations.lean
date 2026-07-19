import TraversalAlgebra.TypedCost

namespace TraversalAlgebra.Verified.Typed.TraversalAlgebra.Observation

variable {n : Nat} {Base : Type} {denoteBase : Base → Type}
variable {signature : Signature n Base denoteBase}
variable {vertex edge output : ValueType Base}

open TraversalAlgebra.Expression

/-- Explicit observations of the three public traversal terminal shapes.
Emission and source reduction both return lists, but distinct constructors keep
their indexing contracts observable. -/
inductive Result (n : Nat) (Value : Type u) where
  | emitted : List Value → Result n Value
  | sourceReduced : List Value → Result n Value
  | destinationReduced : Store n Value → Result n Value

/-- Public TA terminals over the independent compositional edge expression. -/
inductive Terminal
    (signature : Signature n Base denoteBase)
    (vertex edge output : ValueType Base) : Type
  | emit : Traversal signature vertex edge output →
      Terminal signature vertex edge output
  | reduceBySource : Traversal signature vertex edge output →
      signature.Monoid output → Terminal signature vertex edge output
  | reduceByDestination : Traversal signature vertex edge output →
      signature.Monoid output → Terminal signature vertex edge output

namespace Terminal

/-- Direct TA denotation of every observation shape. -/
def observe
    (terminal : Terminal signature vertex edge output)
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (state : Store n (vertex.denote n denoteBase))
    (frontier : Frontier n) : Result n (output.denote n denoteBase) :=
  match terminal with
  | .emit traversal =>
      .emitted <| TraversalAlgebra.Verified.TraversalAlgebra.emit
        graph frontier (traversal.evaluateAt state)
  | .reduceBySource traversal reduction =>
      .sourceReduced <|
        TraversalAlgebra.Verified.TraversalAlgebra.reduceBySource
          graph frontier (signature.denoteMonoid reduction)
            (traversal.evaluateAt state)
  | .reduceByDestination traversal reduction =>
      .destinationReduced <|
        TraversalAlgebra.Verified.TraversalAlgebra.reduceByDestination
          graph frontier (signature.denoteMonoid reduction)
            (traversal.evaluateAt state)

end Terminal

namespace BSP

/-- Independent observer syntax over one typed BSP edge-message term. -/
inductive Terminal
    (signature : Signature n Base denoteBase)
    (vertex edge output : ValueType Base) : Type
  | emit : Term signature (MessageContext vertex edge) output →
      Terminal signature vertex edge output
  | reduceBySource : Term signature (MessageContext vertex edge) output →
      signature.Monoid output → Terminal signature vertex edge output
  | reduceByDestination :
      Term signature (MessageContext vertex edge) output →
      signature.Monoid output → Terminal signature vertex edge output

namespace Terminal

def mapAt
    (term : Term signature (MessageContext vertex edge) output)
    (state : Store n (vertex.denote n denoteBase))
    (context : EdgeContext n (edge.denote n denoteBase)) :
    output.denote n denoteBase :=
  Term.evaluate (MessageContext.fromEdge state context) term

/-- Source-machine emission scans frontier and adjacency rows directly. -/
def nestedEmit
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (term : Term signature (MessageContext vertex edge) output)
    (state : Store n (vertex.denote n denoteBase)) :
    Frontier n → List (output.denote n denoteBase)
  | [] => []
  | source :: frontier =>
      (graph.outgoing source).map (fun edgeValue =>
        Term.evaluate
          (MessageContext.environment source edgeValue.destination
            (state source) (state edgeValue.destination) edgeValue.payload)
          term) ++ nestedEmit graph term state frontier

/-- One source row reduced without invoking flattened traversal. -/
def foldSource
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (term : Term signature (MessageContext vertex edge) output)
    (reduction : signature.Monoid output)
    (state : Store n (vertex.denote n denoteBase))
    (source : Fin n) : output.denote n denoteBase :=
  (graph.outgoing source).foldl
    (fun accumulator edgeValue =>
      (signature.denoteMonoid reduction).combine accumulator
        (Term.evaluate
          (MessageContext.environment source edgeValue.destination
            (state source) (state edgeValue.destination) edgeValue.payload)
          term))
    (signature.denoteMonoid reduction).identity

/-- One nested destination inbox in the observer source machine. -/
def destinationAt
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (term : Term signature (MessageContext vertex edge) output)
    (reduction : signature.Monoid output)
    (state : Store n (vertex.denote n denoteBase))
    (frontier : Frontier n)
    (destination : Fin n) : output.denote n denoteBase :=
  frontier.foldl
    (fun accumulator source =>
      (graph.outgoing source).foldl
        (fun accumulator edgeValue =>
          if edgeValue.destination = destination then
            (signature.denoteMonoid reduction).combine accumulator
              (Term.evaluate
                (MessageContext.environment source edgeValue.destination
                  (state source) (state edgeValue.destination)
                    edgeValue.payload) term)
          else accumulator)
        accumulator)
    (signature.denoteMonoid reduction).identity

/-- Independent BSP-style denotation of all three observer shapes. -/
def observe
    (terminal : Terminal signature vertex edge output)
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (state : Store n (vertex.denote n denoteBase))
    (frontier : Frontier n) : Result n (output.denote n denoteBase) :=
  match terminal with
  | .emit term => .emitted (nestedEmit graph term state frontier)
  | .reduceBySource term reduction =>
      .sourceReduced (frontier.map (foldSource graph term reduction state))
  | .reduceByDestination term reduction =>
      .destinationReduced (destinationAt graph term reduction state frontier)

end Terminal
end BSP

namespace Proof

/-- One normalized row emits the same ordered values through flattened edge
contexts and through direct adjacency access. -/
theorem map_expand
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (term : Term signature (MessageContext vertex edge) output)
    (state : Store n (vertex.denote n denoteBase))
    (source : Fin n) :
    (graph.expand source).map (BSP.Terminal.mapAt term state) =
      (graph.outgoing source).map (fun edgeValue =>
        Term.evaluate
          (MessageContext.environment source edgeValue.destination
            (state source) (state edgeValue.destination) edgeValue.payload)
          term) := by
  unfold OrderedGraph.expand BSP.Terminal.mapAt
  rw [List.map_map]
  apply List.map_congr_left
  intro edgeValue membership
  rfl

/-- Flattened emission equals the independently nested source observation. -/
theorem map_traverse
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (term : Term signature (MessageContext vertex edge) output)
    (state : Store n (vertex.denote n denoteBase))
    (frontier : Frontier n) :
    (graph.traverse frontier).map (BSP.Terminal.mapAt term state) =
      BSP.Terminal.nestedEmit graph term state frontier := by
  induction frontier with
  | nil => rfl
  | cons source frontier induction =>
      simp only [OrderedGraph.traverse_cons, List.map_append,
        BSP.Terminal.nestedEmit, map_expand, induction]

/-- A mapped fold over one expanded row equals the direct source-row fold for
every initial accumulator. -/
theorem fold_expand
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (term : Term signature (MessageContext vertex edge) output)
    (reduction : signature.Monoid output)
    (state : Store n (vertex.denote n denoteBase))
    (source : Fin n)
    (initial : output.denote n denoteBase) :
    (graph.expand source).foldl
        (fun accumulator context =>
          (signature.denoteMonoid reduction).combine accumulator
            (BSP.Terminal.mapAt term state context)) initial =
      (graph.outgoing source).foldl
        (fun accumulator edgeValue =>
          (signature.denoteMonoid reduction).combine accumulator
            (Term.evaluate
              (MessageContext.environment source edgeValue.destination
                (state source) (state edgeValue.destination) edgeValue.payload)
              term)) initial := by
  unfold OrderedGraph.expand
  generalize graph.outgoing source = edges
  induction edges generalizing initial with
  | nil => rfl
  | cons edgeValue edges induction =>
      simp only [List.map, List.foldl, BSP.Terminal.mapAt]
      exact induction _

/-- Destination-filtered form of `fold_expand`. -/
theorem fold_expand_destination
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (term : Term signature (MessageContext vertex edge) output)
    (reduction : signature.Monoid output)
    (state : Store n (vertex.denote n denoteBase))
    (source destination : Fin n)
    (initial : output.denote n denoteBase) :
    (graph.expand source).foldl
        (fun accumulator context =>
          if context.destination = destination then
            (signature.denoteMonoid reduction).combine accumulator
              (BSP.Terminal.mapAt term state context)
          else accumulator) initial =
      (graph.outgoing source).foldl
        (fun accumulator edgeValue =>
          if edgeValue.destination = destination then
            (signature.denoteMonoid reduction).combine accumulator
              (Term.evaluate
                (MessageContext.environment source edgeValue.destination
                  (state source) (state edgeValue.destination)
                    edgeValue.payload) term)
          else accumulator) initial := by
  unfold OrderedGraph.expand
  generalize graph.outgoing source = edges
  induction edges generalizing initial with
  | nil => rfl
  | cons edgeValue edges induction =>
      simp only [List.map, List.foldl, BSP.Terminal.mapAt]
      exact induction _

/-- Flattened destination reduction equals the nested observer for every
destination. -/
theorem fold_traverse_destination
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (term : Term signature (MessageContext vertex edge) output)
    (reduction : signature.Monoid output)
    (state : Store n (vertex.denote n denoteBase))
    (frontier : Frontier n)
    (destination : Fin n) :
    TraversalAlgebra.Verified.TraversalAlgebra.reduceByDestinationAt
        graph frontier (signature.denoteMonoid reduction)
          (BSP.Terminal.mapAt term state) destination =
      BSP.Terminal.destinationAt
        graph term reduction state frontier destination := by
  unfold TraversalAlgebra.Verified.TraversalAlgebra.reduceByDestinationAt
  unfold BSP.Terminal.destinationAt
  generalize (signature.denoteMonoid reduction).identity = initial
  induction frontier generalizing initial with
  | nil => rfl
  | cons source frontier induction =>
      simp only [OrderedGraph.traverse_cons, List.foldl_append, List.foldl]
      rw [fold_expand_destination]
      exact induction _

/-- Generic correctness for emission given pointwise equality of edge maps. -/
theorem emit_correct
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (traversal : Traversal signature vertex edge output)
    (term : Term signature (MessageContext vertex edge) output)
    (state : Store n (vertex.denote n denoteBase))
    (frontier : Frontier n)
    (pointwise : ∀ context,
      traversal.evaluateAt state context = BSP.Terminal.mapAt term state context) :
    TraversalAlgebra.Verified.TraversalAlgebra.emit
        graph frontier (traversal.evaluateAt state) =
      BSP.Terminal.nestedEmit graph term state frontier := by
  unfold TraversalAlgebra.Verified.TraversalAlgebra.emit
  rw [show traversal.evaluateAt state = BSP.Terminal.mapAt term state by
    funext context
    exact pointwise context]
  exact map_traverse graph term state frontier

/-- Generic correctness for source reduction given pointwise map equality. -/
theorem source_correct
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (traversal : Traversal signature vertex edge output)
    (term : Term signature (MessageContext vertex edge) output)
    (reduction : signature.Monoid output)
    (state : Store n (vertex.denote n denoteBase))
    (frontier : Frontier n)
    (pointwise : ∀ context,
      traversal.evaluateAt state context = BSP.Terminal.mapAt term state context) :
    TraversalAlgebra.Verified.TraversalAlgebra.reduceBySource
        graph frontier (signature.denoteMonoid reduction)
          (traversal.evaluateAt state) =
      frontier.map (BSP.Terminal.foldSource graph term reduction state) := by
  unfold TraversalAlgebra.Verified.TraversalAlgebra.reduceBySource
  apply List.map_congr_left
  intro source membership
  unfold TraversalAlgebra.Verified.TraversalAlgebra.reduceEdges
  rw [show traversal.evaluateAt state = BSP.Terminal.mapAt term state by
    funext context
    exact pointwise context]
  exact fold_expand graph term reduction state source _

/-- Generic correctness for destination reduction given pointwise equality. -/
theorem destination_correct
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (traversal : Traversal signature vertex edge output)
    (term : Term signature (MessageContext vertex edge) output)
    (reduction : signature.Monoid output)
    (state : Store n (vertex.denote n denoteBase))
    (frontier : Frontier n)
    (pointwise : ∀ context,
      traversal.evaluateAt state context = BSP.Terminal.mapAt term state context) :
    TraversalAlgebra.Verified.TraversalAlgebra.reduceByDestination
        graph frontier (signature.denoteMonoid reduction)
          (traversal.evaluateAt state) =
      BSP.Terminal.destinationAt graph term reduction state frontier := by
  funext destination
  unfold TraversalAlgebra.Verified.TraversalAlgebra.reduceByDestination
  rw [show traversal.evaluateAt state = BSP.Terminal.mapAt term state by
    funext context
    exact pointwise context]
  exact fold_traverse_destination
    graph term reduction state frontier destination

end Proof

namespace Terminal

/-- Normalize a public TA observer into the independent BSP observer syntax. -/
def toBSP
    (terminal : Terminal signature vertex edge output) :
    BSP.Terminal signature vertex edge output :=
  match terminal with
  | .emit traversal => .emit traversal.normalize
  | .reduceBySource traversal reduction =>
      .reduceBySource traversal.normalize reduction
  | .reduceByDestination traversal reduction =>
      .reduceByDestination traversal.normalize reduction

/-- Every TA terminal observation is preserved by BSP normalization. -/
theorem toBSP_observe_correct
    (terminal : Terminal signature vertex edge output)
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (state : Store n (vertex.denote n denoteBase))
    (frontier : Frontier n) :
    terminal.observe graph state frontier =
      (terminal.toBSP).observe graph state frontier := by
  cases terminal with
  | emit traversal =>
      apply congrArg Result.emitted
      exact Proof.emit_correct graph traversal traversal.normalize state frontier
        (fun context =>
          (Traversal.evaluateAt_normalize traversal state context).symm)
  | reduceBySource traversal reduction =>
      apply congrArg Result.sourceReduced
      exact Proof.source_correct graph traversal traversal.normalize reduction
        state frontier (fun context =>
          (Traversal.evaluateAt_normalize traversal state context).symm)
  | reduceByDestination traversal reduction =>
      apply congrArg Result.destinationReduced
      exact Proof.destination_correct graph traversal traversal.normalize reduction
        state frontier (fun context =>
          (Traversal.evaluateAt_normalize traversal state context).symm)

end Terminal

namespace BSP.Terminal

/-- Reify every typed BSP observer as a public compositional TA terminal. -/
def toTA
    (terminal : Terminal signature vertex edge output) :
    Observation.Terminal signature vertex edge output :=
  match terminal with
  | .emit term => .emit (Traversal.ofTerm term)
  | .reduceBySource term reduction =>
      .reduceBySource (Traversal.ofTerm term) reduction
  | .reduceByDestination term reduction =>
      .reduceByDestination (Traversal.ofTerm term) reduction

/-- Every BSP observer has an exactly equivalent TA observation, including
ordered emission and source-occurrence multiplicity. -/
theorem toTA_observe_correct
    (terminal : Terminal signature vertex edge output)
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (state : Store n (vertex.denote n denoteBase))
    (frontier : Frontier n) :
    terminal.observe graph state frontier =
      terminal.toTA.observe graph state frontier := by
  cases terminal with
  | emit term =>
      apply congrArg Result.emitted
      symm
      exact Proof.emit_correct graph (Traversal.ofTerm term) term state frontier
        (fun context => Traversal.evaluateAt_ofTerm term state context)
  | reduceBySource term reduction =>
      apply congrArg Result.sourceReduced
      symm
      exact Proof.source_correct graph (Traversal.ofTerm term) term reduction
        state frontier
        (fun context => Traversal.evaluateAt_ofTerm term state context)
  | reduceByDestination term reduction =>
      apply congrArg Result.destinationReduced
      symm
      exact Proof.destination_correct graph (Traversal.ofTerm term) term reduction
        state frontier
        (fun context => Traversal.evaluateAt_ofTerm term state context)

end BSP.Terminal

namespace Cost

/-- Abstract work of a TA observation. Emission charges one output per edge;
reductions additionally charge one monoid action per edge occurrence. -/
def taWork
    (terminal : Terminal signature vertex edge output)
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (frontier : Frontier n) : Nat :=
  let edges := activeEdgeCount graph frontier
  match terminal with
  | .emit traversal => edges * traversal.work + edges
  | .reduceBySource traversal _ => edges * traversal.work + edges
  | .reduceByDestination traversal _ => edges * traversal.work + edges

/-- Work of the normalized BSP observation under the same unit-cost model. -/
def bspWork
    (terminal : BSP.Terminal signature vertex edge output)
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (frontier : Frontier n) : Nat :=
  let edges := activeEdgeCount graph frontier
  match terminal with
  | .emit term => edges * term.work + edges
  | .reduceBySource term _ => edges * term.work + edges
  | .reduceByDestination term _ => edges * term.work + edges

/-- Observation normalization preserves work exactly for all three terminals. -/
@[simp]
theorem toBSP_work
    (terminal : Terminal signature vertex edge output)
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (frontier : Frontier n) :
    bspWork terminal.toBSP graph frontier = taWork terminal graph frontier := by
  cases terminal with
  | emit traversal => simp only [bspWork, Terminal.toBSP, taWork,
      Traversal.normalize_work]
  | reduceBySource traversal reduction =>
      simp only [bspWork, Terminal.toBSP, taWork,
        Traversal.normalize_work]
  | reduceByDestination traversal reduction =>
      simp only [bspWork, Terminal.toBSP, taWork,
        Traversal.normalize_work]

end Cost
end TraversalAlgebra.Verified.Typed.TraversalAlgebra.Observation
