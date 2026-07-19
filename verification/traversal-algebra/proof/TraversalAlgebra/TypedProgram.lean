import TraversalAlgebra.Simulation
import TraversalAlgebra.TypedIR

namespace TraversalAlgebra.Verified.Typed

variable {n : Nat} {Base : Type} {denoteBase : Base → Type}
variable {signature : Signature n Base denoteBase}
variable {vertex edge message : ValueType Base}

/-- Inputs visible to an edge-message term, in the same order as the source
machine's semantic message function. -/
abbrev MessageContext (vertex edge : ValueType Base) : List (ValueType Base) :=
  [.index, .index, vertex, vertex, edge]

namespace MessageContext

def sourceId : Variable (MessageContext vertex edge) .index := .here

def destinationId : Variable (MessageContext vertex edge) .index :=
  .there .here

def sourceState : Variable (MessageContext vertex edge) vertex :=
  .there (.there .here)

def destinationState : Variable (MessageContext vertex edge) vertex :=
  .there (.there (.there .here))

def edgePayload : Variable (MessageContext vertex edge) edge :=
  .there (.there (.there (.there .here)))

def environment
    (source destination : Fin n)
    (sourceState destinationState : vertex.denote n denoteBase)
    (payload : edge.denote n denoteBase) :
    Environment n denoteBase (MessageContext vertex edge) :=
  .cons source
    (.cons destination
      (.cons sourceState (.cons destinationState (.cons payload .nil))))

def fromEdge
    (state : Store n (vertex.denote n denoteBase))
    (context : EdgeContext n (edge.denote n denoteBase)) :
    Environment n denoteBase (MessageContext vertex edge) :=
  environment context.source context.destination
    (state context.source) (state context.destination) context.payload

end MessageContext

/-- Inputs visible while updating one dense vertex. -/
abbrev UpdateContext (vertex message : ValueType Base) : List (ValueType Base) :=
  [.index, vertex, message]

namespace UpdateContext

def vertexId : Variable (UpdateContext vertex message) .index := .here

def oldState : Variable (UpdateContext vertex message) vertex :=
  .there .here

def inbox : Variable (UpdateContext vertex message) message :=
  .there (.there .here)

def environment
    (vertexId : Fin n)
    (oldState : vertex.denote n denoteBase)
    (inbox : message.denote n denoteBase) :
    Environment n denoteBase (UpdateContext vertex message) :=
  .cons vertexId (.cons oldState (.cons inbox .nil))

end UpdateContext

/-- Inputs visible to either next-frontier predicate. -/
abbrev SelectionContext (vertex : ValueType Base) : List (ValueType Base) :=
  [.index, vertex, vertex]

namespace SelectionContext

def vertexId : Variable (SelectionContext vertex) .index := .here

def oldState : Variable (SelectionContext vertex) vertex :=
  .there .here

def newState : Variable (SelectionContext vertex) vertex :=
  .there (.there .here)

def environment
    (vertexId : Fin n)
    (oldState newState : vertex.denote n denoteBase) :
    Environment n denoteBase (SelectionContext vertex) :=
  .cons vertexId (.cons oldState (.cons newState .nil))

end SelectionContext

/-- Canonical ascending enumeration of every valid vertex. -/
def allVertices : (n : Nat) → List (Fin n)
  | 0 => []
  | n + 1 =>
      (allVertices n).map (fun vertex => vertex.castSucc) ++ [Fin.last n]

@[simp]
theorem allVertices_length (n : Nat) : (allVertices n).length = n := by
  induction n with
  | zero => rfl
  | succ n induction => simp [allVertices, induction]

/-- Closed next-frontier policies with explicit candidate semantics.

`dense` ignores candidates and scans every valid vertex in canonical order.
`sparsePreserve` filters only destinations touched by the active traversal and
therefore preserves their exact order and multiplicity. -/
inductive FrontierPolicy
    (signature : Signature n Base denoteBase)
    (vertex : ValueType Base) : Type
  | dense : Term signature (SelectionContext vertex) .boolean →
      FrontierPolicy signature vertex
  | sparsePreserve : Term signature (SelectionContext vertex) .boolean →
      FrontierPolicy signature vertex

namespace FrontierPolicy

def denote
    (policy : FrontierPolicy signature vertex)
    (candidates : Frontier n)
    (oldState newState : Store n (vertex.denote n denoteBase)) : Frontier n :=
  match policy with
  | .dense predicate =>
      (allVertices n).filter fun vertexId =>
        Term.evaluate (signature := signature)
          (SelectionContext.environment vertexId
            (oldState vertexId) (newState vertexId)) predicate
  | .sparsePreserve predicate =>
      candidates.filter fun vertexId =>
        Term.evaluate (signature := signature)
          (SelectionContext.environment vertexId
            (oldState vertexId) (newState vertexId)) predicate

/-- Sparse selection is literally a stable filter of the candidate stream. -/
@[simp]
theorem denote_sparsePreserve
    (predicate : Term signature (SelectionContext vertex) .boolean)
    (candidates : Frontier n)
    (oldState newState : Store n (vertex.denote n denoteBase)) :
    (FrontierPolicy.sparsePreserve predicate).denote
        candidates oldState newState =
      candidates.filter fun vertexId =>
        Term.evaluate (signature := signature)
          (SelectionContext.environment vertexId
            (oldState vertexId) (newState vertexId)) predicate := rfl

/-- Hence sparse selection preserves relative order and cannot introduce a
candidate occurrence. -/
theorem sparsePreserve_sublist
    (predicate : Term signature (SelectionContext vertex) .boolean)
    (candidates : Frontier n)
    (oldState newState : Store n (vertex.denote n denoteBase)) :
    (FrontierPolicy.sparsePreserve predicate).denote
        candidates oldState newState |>.Sublist candidates := by
  exact List.filter_sublist

/-- Every accepted vertex keeps exactly its candidate multiplicity. -/
theorem sparsePreserve_count_of_selected
    (predicate : Term signature (SelectionContext vertex) .boolean)
    (candidates : Frontier n)
    (oldState newState : Store n (vertex.denote n denoteBase))
    (vertexId : Fin n)
    (selected : Term.evaluate (signature := signature)
      (SelectionContext.environment vertexId
        (oldState vertexId) (newState vertexId)) predicate = true) :
    List.count vertexId
        ((FrontierPolicy.sparsePreserve predicate).denote
          candidates oldState newState) =
      List.count vertexId candidates := by
  rw [denote_sparsePreserve]
  exact List.count_filter selected

/-- Every rejected vertex loses all of its candidate occurrences. -/
theorem sparsePreserve_count_of_rejected
    (predicate : Term signature (SelectionContext vertex) .boolean)
    (candidates : Frontier n)
    (oldState newState : Store n (vertex.denote n denoteBase))
    (vertexId : Fin n)
    (rejected : Term.evaluate (signature := signature)
      (SelectionContext.environment vertexId
        (oldState vertexId) (newState vertexId)) predicate = false) :
    List.count vertexId
        ((FrontierPolicy.sparsePreserve predicate).denote
          candidates oldState newState) = 0 := by
  apply List.count_eq_zero_of_not_mem
  simp only [denote_sparsePreserve, List.mem_filter, not_and]
  intro membership selected
  rw [rejected] at selected
  contradiction

/-- Multiplicity is explicit: an accepted vertex keeps exactly its candidate
occurrence count, while a rejected vertex has count zero. -/
theorem sparsePreserve_count
    (predicate : Term signature (SelectionContext vertex) .boolean)
    (candidates : Frontier n)
    (oldState newState : Store n (vertex.denote n denoteBase))
    (vertexId : Fin n) :
    List.count vertexId
        ((FrontierPolicy.sparsePreserve predicate).denote
          candidates oldState newState) =
      if (show Bool from Term.evaluate (signature := signature)
          (SelectionContext.environment vertexId
            (oldState vertexId) (newState vertexId)) predicate) = true then
        List.count vertexId candidates
      else 0 := by
  by_cases selected : (show Bool from Term.evaluate (signature := signature)
      (SelectionContext.environment vertexId
        (oldState vertexId) (newState vertexId)) predicate) = true
  · rw [sparsePreserve_count_of_selected predicate candidates
      oldState newState vertexId selected]
    simp [selected]
  · have rejected : (show Bool from Term.evaluate (signature := signature)
        (SelectionContext.environment vertexId
          (oldState vertexId) (newState vertexId)) predicate) = false := by
      cases evaluated : (show Bool from Term.evaluate (signature := signature)
        (SelectionContext.environment vertexId
          (oldState vertexId) (newState vertexId)) predicate) <;> simp_all
    rw [sparsePreserve_count_of_rejected predicate candidates
      oldState newState vertexId rejected]
    simp [rejected]

end FrontierPolicy

namespace MonoidalFrontierBSP

/-- A syntactic BSP program. Denotation functions are not stored directly in
the program: its fields are typed signature symbols or typed terms. -/
structure Program
    (signature : Signature n Base denoteBase)
    (vertex edge message : ValueType Base) where
  messageTerm : Term signature (MessageContext vertex edge) message
  reduction : signature.Monoid message
  updateTerm : Term signature (UpdateContext vertex message) vertex
  frontier : FrontierPolicy signature vertex

/-- Interpret a syntactic source program in the independent semantic BSP
machine. -/
def Program.denote
    (program : Program signature vertex edge message) :
    TraversalAlgebra.Verified.MonoidalFrontierBSP.Program n
      (vertex.denote n denoteBase)
      (edge.denote n denoteBase)
      (message.denote n denoteBase) where
  message := fun source destination sourceState destinationState payload =>
    Term.evaluate (signature := signature)
      (MessageContext.environment source destination
        sourceState destinationState payload) program.messageTerm
  reduction := signature.denoteMonoid program.reduction
  lawful := signature.lawfulMonoid program.reduction
  update := fun vertexId oldState inbox =>
    Term.evaluate (signature := signature)
      (UpdateContext.environment vertexId oldState inbox) program.updateTerm
  select := program.frontier.denote

def step
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : Program signature vertex edge message)
    (configuration : Configuration n (vertex.denote n denoteBase)) :
    Configuration n (vertex.denote n denoteBase) :=
  TraversalAlgebra.Verified.MonoidalFrontierBSP.step
    graph program.denote configuration

def run
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : Program signature vertex edge message) :
    Nat → Configuration n (vertex.denote n denoteBase) →
      Configuration n (vertex.denote n denoteBase)
  | 0, configuration => configuration
  | fuel + 1, configuration =>
      run graph program fuel (step graph program configuration)

end MonoidalFrontierBSP

namespace PullMapPush

/-- Closed target syntax for a barrier-delimited pull--map--push program. -/
structure Program
    (signature : Signature n Base denoteBase)
    (vertex edge message : ValueType Base) where
  edgeMap : Term signature (MessageContext vertex edge) message
  reduction : signature.Monoid message
  vertexUpdate : Term signature (UpdateContext vertex message) vertex
  nextFrontier : FrontierPolicy signature vertex

/-- Interpret the target syntax at one barrier state. -/
def Program.denoteAt
    (program : Program signature vertex edge message)
    (state : Store n (vertex.denote n denoteBase)) :
    TraversalAlgebra.Verified.PullMapPush.Plan n
      (vertex.denote n denoteBase)
      (edge.denote n denoteBase)
      (message.denote n denoteBase) where
  map := fun context =>
    Term.evaluate (signature := signature)
      (MessageContext.fromEdge state context) program.edgeMap
  reduction := signature.denoteMonoid program.reduction
  lawful := signature.lawfulMonoid program.reduction
  update := fun vertexId oldState inbox =>
    Term.evaluate (signature := signature)
      (UpdateContext.environment vertexId oldState inbox) program.vertexUpdate
  select := program.nextFrontier.denote

def step
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : Program signature vertex edge message)
    (configuration : Configuration n (vertex.denote n denoteBase)) :
    Configuration n (vertex.denote n denoteBase) :=
  TraversalAlgebra.Verified.PullMapPush.step graph
    (program.denoteAt configuration.state) configuration

def run
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : Program signature vertex edge message) :
    Nat → Configuration n (vertex.denote n denoteBase) →
      Configuration n (vertex.denote n denoteBase)
  | 0, configuration => configuration
  | fuel + 1, configuration =>
      run graph program fuel (step graph program configuration)

end PullMapPush

/-- Syntax-directed compilation. The graph-control structure changes from
nested BSP loops to pull--map--push; all typed scalar terms are preserved. -/
def compile
    (program : MonoidalFrontierBSP.Program signature vertex edge message) :
    PullMapPush.Program signature vertex edge message where
  edgeMap := program.messageTerm
  reduction := program.reduction
  vertexUpdate := program.updateTerm
  nextFrontier := program.frontier

end TraversalAlgebra.Verified.Typed
