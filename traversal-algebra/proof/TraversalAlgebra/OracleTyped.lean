import TraversalAlgebra.TypedObservations
import TraversalAlgebra.Oracle

namespace TraversalAlgebra.Oracle.Typed

open TraversalAlgebra.Verified
open TraversalAlgebra.Verified.Typed
open TraversalAlgebra.Verified.Typed.TraversalAlgebra

/-- A checked concrete CSR value together with its extensional graph witness. -/
structure CheckedCsr where
  offsets : List Nat
  destinations : List Nat
  graph : Graph
  offsets_eq : graph.csrOffsets = offsets
  destinations_eq : graph.csrDestinations = destinations
  valid : graph.Valid

theorem graph_isValid_iff (graph : Graph) : graph.isValid = true ↔ graph.Valid := by
  unfold Graph.isValid Graph.Valid
  rw [List.all_eq_true]
  constructor
  · intro validRows row rowMembership destination destinationMembership
    have validDestinations := validRows row rowMembership
    have validDestination := List.all_eq_true.mp validDestinations
      destination destinationMembership
    exact of_decide_eq_true validDestination
  · intro valid row rowMembership
    apply List.all_eq_true.mpr
    intro destination destinationMembership
    exact decide_eq_true (valid row rowMembership destination destinationMembership)

theorem frontier_isValid_iff (graph : Graph) (frontier : Frontier) :
    isValidFrontier graph frontier = true ↔ ValidFrontier graph frontier := by
  unfold isValidFrontier ValidFrontier
  rw [List.all_eq_true]
  constructor
  · intro valid source membership
    exact of_decide_eq_true (valid source membership)
  · intro valid source membership
    exact decide_eq_true (valid source membership)

/-- Checks raw CSR arrays and retains exact round-trip equations as proof data. -/
def checkCsr (offsets destinations : List Nat) : Except String CheckedCsr := do
  if offsets.isEmpty then
    throw "CSR offsets must contain the initial zero"
  if offsets.head? != some 0 then
    throw "CSR offsets must start at zero"
  let rows := (offsets.zip offsets.tail).map fun (start, stop) =>
    (destinations.drop start).take (stop - start)
  let graph : Graph := ⟨rows⟩
  if offsetsEq : graph.csrOffsets = offsets then
    if destinationsEq : graph.csrDestinations = destinations then
      if validEq : graph.isValid = true then
        pure {
          offsets
          destinations
          graph
          offsets_eq := offsetsEq
          destinations_eq := destinationsEq
          valid := (graph_isValid_iff graph).mp validEq
        }
      else
        throw "CSR contains a destination outside the vertex set"
    else
      throw "CSR offsets are not canonical for the destination stream"
  else
    throw "CSR offsets are not canonical for the destination stream"

/-- The checked concrete CSR arrays are exactly the extensional graph's CSR. -/
theorem checkedCsr_corresponds (csr : CheckedCsr) :
    csr.graph.csrOffsets = csr.offsets ∧
      csr.graph.csrDestinations = csr.destinations :=
  ⟨csr.offsets_eq, csr.destinations_eq⟩

/-- All host columns visible to one vertex occurrence. -/
abbrev VertexColumns := List Nat

/-- The global CSR edge identifier and all host edge columns. -/
structure EdgeColumns where
  id : Nat
  values : List Nat
deriving Repr, DecidableEq

private def edgeColumnAt (columns : List (List Nat)) (edge : Nat) : List Nat :=
  columns.map fun column => column.getD edge 0

private def verifiedRow
    (n base : Nat) (columns : List (List Nat)) :
    (row : List Nat) → (∀ destination ∈ row, destination < n) →
      List (Verified.Edge n EdgeColumns)
  | [], _ => []
  | destination :: destinations, valid =>
      { destination := ⟨destination, valid destination (by simp)⟩
        payload := { id := base, values := edgeColumnAt columns base } } ::
      verifiedRow n (base + 1) columns destinations
        (fun value membership => valid value (by simp [membership]))

/-- Converts the checked extensional graph into the graph used by the typed
semantics. CSR position is carried explicitly in the edge payload. -/
def toOrderedGraph (csr : CheckedCsr) (edgeColumns : List (List Nat)) :
    OrderedGraph csr.graph.vertexCount EdgeColumns where
  outgoing source :=
    let row := csr.graph.rows.get source
    verifiedRow csr.graph.vertexCount (csr.graph.edgeBase source) edgeColumns row
      (fun destination membership =>
        csr.valid row (List.get_mem csr.graph.rows source) destination membership)

private theorem verifiedRow_destinations
    (row : List Nat) (valid : ∀ destination ∈ row, destination < n) :
    (verifiedRow n base columns row valid).map
        (fun edge => edge.destination.val) = row := by
  induction row generalizing base with
  | nil => rfl
  | cons destination destinations induction =>
      simp only [verifiedRow, List.map]
      rw [induction]

private theorem verifiedRow_edgeIds
    (row : List Nat) (valid : ∀ destination ∈ row, destination < n) :
    (verifiedRow n base columns row valid).map
        (fun edge => edge.payload.id) = List.range' base row.length := by
  induction row generalizing base with
  | nil => rfl
  | cons destination destinations induction =>
      simp only [verifiedRow, List.map, List.length_cons, List.range'_succ]
      rw [induction]

private theorem graph_row_eq_get
    (graph : Graph) (source : Fin graph.vertexCount) :
    graph.row source = graph.rows.get source := by
  unfold Graph.row Graph.vertexCount List.getD
  simp

/-- Every typed adjacency row has exactly the destinations represented by the
checked concrete CSR row. -/
theorem toOrderedGraph_destinations
    (csr : CheckedCsr) (edgeColumns : List (List Nat))
    (source : Fin csr.graph.vertexCount) :
    ((toOrderedGraph csr edgeColumns).outgoing source).map
        (fun edge => edge.destination.val) = csr.graph.row source := by
  unfold toOrderedGraph
  rw [verifiedRow_destinations, graph_row_eq_get]

/-- Edge payload identifiers are precisely the contiguous global CSR
positions for the source row. -/
theorem toOrderedGraph_edgeIds
    (csr : CheckedCsr) (edgeColumns : List (List Nat))
    (source : Fin csr.graph.vertexCount) :
    ((toOrderedGraph csr edgeColumns).outgoing source).map
        (fun edge => edge.payload.id) =
      List.range' (csr.graph.edgeBase source) (csr.graph.row source).length := by
  unfold toOrderedGraph
  rw [verifiedRow_edgeIds, graph_row_eq_get]

/-- Converts checked natural identifiers into the intrinsically valid typed
frontier while preserving order and multiplicity. -/
def toTypedFrontier
    (graph : Graph) (frontier : Frontier) (valid : ValidFrontier graph frontier) :
    Verified.Frontier graph.vertexCount :=
  frontier.attach.map fun source =>
    ⟨source.val, valid source.val source.property⟩

@[simp]
theorem toTypedFrontier_nil
    (graph : Graph) (valid : ValidFrontier graph []) :
    toTypedFrontier graph [] valid = [] := rfl

@[simp]
theorem toTypedFrontier_cons
    (graph : Graph) (source : Nat) (frontier : Frontier)
    (valid : ValidFrontier graph (source :: frontier)) :
    toTypedFrontier graph (source :: frontier) valid =
      ⟨source, valid source (by simp)⟩ ::
        toTypedFrontier graph frontier
          (fun candidate membership => valid candidate (by simp [membership])) := by
  simp [toTypedFrontier]

@[simp]
theorem toTypedFrontier_length
    (graph : Graph) (frontier : Frontier)
    (valid : ValidFrontier graph frontier) :
    (toTypedFrontier graph frontier valid).length = frontier.length := by
  simp [toTypedFrontier]

private theorem expandFrom_length
    (graph : Graph) (source index : Nat) (row : List Nat) :
    (graph.expandFrom source index row).length = row.length := by
  induction row generalizing index with
  | nil => rfl
  | cons destination destinations induction =>
      simp only [Graph.expandFrom, List.length_cons, induction]

theorem graph_expand_length (graph : Graph) (source : Nat) :
    (graph.expand source).length = (graph.row source).length := by
  unfold Graph.expand
  exact expandFrom_length graph source 0 (graph.row source)

theorem toOrderedGraph_expand_length
    (csr : CheckedCsr) (edgeColumns : List (List Nat))
    (source : Fin csr.graph.vertexCount) :
    ((toOrderedGraph csr edgeColumns).expand source).length =
      (csr.graph.expand source.val).length := by
  unfold OrderedGraph.expand
  rw [List.length_map, graph_expand_length]
  have destinations := toOrderedGraph_destinations csr edgeColumns source
  simpa only [List.length_map] using congrArg List.length destinations

/-- Concrete and typed traversals have exactly the same active-edge count,
including duplicate frontier occurrences. -/
theorem activeEdgeCount_corresponds
    (csr : CheckedCsr) (edgeColumns : List (List Nat))
    (frontier : Frontier) (valid : ValidFrontier csr.graph frontier) :
    activeEdgeCount (toOrderedGraph csr edgeColumns)
        (toTypedFrontier csr.graph frontier valid) =
      (csr.graph.traverse frontier).length := by
  induction frontier with
  | nil => rfl
  | cons source frontier induction =>
      let restValid : ValidFrontier csr.graph frontier :=
        fun candidate membership => valid candidate (by simp [membership])
      rw [toTypedFrontier_cons]
      unfold activeEdgeCount
      rw [OrderedGraph.traverse_cons, List.length_append]
      simp only [Graph.traverse, List.flatMap_cons, List.length_append]
      rw [toOrderedGraph_expand_length]
      have tail := induction restValid
      unfold activeEdgeCount at tail
      rw [tail]
      simp [Graph.traverse]

def vertexStore (columns : List (List Nat)) :
    Store n VertexColumns :=
  fun vertex => columns.map fun column => column.getD vertex 0

inductive Base where
  | natural
  | vertexColumns
  | edgeColumns
deriving Repr, DecidableEq

def denoteBase : Base → Type
  | .natural => Nat
  | .vertexColumns => VertexColumns
  | .edgeColumns => EdgeColumns

abbrev natural : ValueType Base := .base .natural
abbrev vertexType : ValueType Base := .base .vertexColumns
abbrev edgeType : ValueType Base := .base .edgeColumns

inductive Literal : ValueType Base → Type
  | natural (value : Nat) : Literal natural

def denoteLiteral : {output : ValueType Base} →
    Literal output → output.denote n denoteBase
  | _, .natural value => value

inductive Primitive : ValueType Base → ValueType Base → Type
  | indexToNatural : Primitive .index natural
  | vertexColumn (column : Nat) : Primitive vertexType natural
  | edgeId : Primitive edgeType natural
  | edgeColumn (column : Nat) : Primitive edgeType natural
  | add : Primitive (.product natural natural) natural

def denotePrimitive : {input output : ValueType Base} →
    Primitive input output →
      input.denote n denoteBase → output.denote n denoteBase
  | _, _, .indexToNatural => fun value => value.val
  | _, _, .vertexColumn column => fun values => values.getD column 0
  | _, _, .edgeId => fun value => value.id
  | _, _, .edgeColumn column => fun value => value.values.getD column 0
  | _, _, .add => fun values =>
      Nat.add (show Nat from values.1) (show Nat from values.2)

inductive MonoidSymbol : ValueType Base → Type
  | naturalAdd : MonoidSymbol natural

def denoteMonoid : {value : ValueType Base} →
    MonoidSymbol value → Reduction (value.denote n denoteBase)
  | _, .naturalAdd => natAdd

theorem lawfulMonoid : {value : ValueType Base} →
    (monoid : MonoidSymbol value) →
      LawfulCommutativeReduction (denoteMonoid (n := n) monoid)
  | _, .naturalAdd => natAdd_lawful

def signature (n : Nat) : Signature n Base denoteBase where
  Literal := Literal
  denoteLiteral := denoteLiteral
  Primitive := Primitive
  denotePrimitive := denotePrimitive
  Monoid := MonoidSymbol
  denoteMonoid := denoteMonoid
  lawfulMonoid := lawfulMonoid

/-- Serializable scalar fragment shared by the Rust façade and Lean. -/
inductive Expr where
  | sourceId
  | destinationId
  | edgeId
  | source (column : Nat)
  | destination (column : Nat)
  | edge (column : Nat)
  | constant (value : Nat)
  | add (left right : Expr)
deriving Repr, DecidableEq

def Expr.referencesValid (vertexColumns edgeColumns : Nat) : Expr → Bool
  | .sourceId | .destinationId | .edgeId | .constant _ => true
  | .source column | .destination column => column < vertexColumns
  | .edge column => column < edgeColumns
  | .add left right =>
      left.referencesValid vertexColumns edgeColumns &&
        right.referencesValid vertexColumns edgeColumns

private def unary
    (primitive : Primitive input output) :
    Term (signature n) [input] output :=
  .apply primitive (.read .here)

/-- Compiles every protocol expression into the independently compositional,
intrinsically typed TA traversal syntax. -/
def Expr.compile (n : Nat) : Expr →
    Expression.Traversal (signature n) vertexType edgeType natural
  | .sourceId =>
      .map Expression.Traversal.sourceId (unary .indexToNatural)
  | .destinationId =>
      .map Expression.Traversal.destinationId (unary .indexToNatural)
  | .edgeId =>
      .map Expression.Traversal.edgePayload (unary .edgeId)
  | .source column =>
      .map Expression.Traversal.sourceState (unary (.vertexColumn column))
  | .destination column =>
      .map Expression.Traversal.destinationState
        (unary (.vertexColumn column))
  | .edge column =>
      .map Expression.Traversal.edgePayload (unary (.edgeColumn column))
  | .constant value => .literal (.natural value)
  | .add left right =>
      .map (.zip (left.compile n) (right.compile n)) (unary .add)

inductive Terminal where
  | emit
  | reduceBySource
  | reduceByDestination
deriving Repr, DecidableEq

def Terminal.compile (n : Nat) (terminal : Terminal) (expression : Expr) :
    Observation.Terminal (signature n) vertexType edgeType natural :=
  match terminal with
  | .emit => .emit (expression.compile n)
  | .reduceBySource => .reduceBySource (expression.compile n) .naturalAdd
  | .reduceByDestination =>
      .reduceByDestination (expression.compile n) .naturalAdd

def resultValues : Observation.Result n Nat → List Nat
  | .emitted values => values
  | .sourceReduced values => values
  | .destinationReduced values => allVertices n |>.map values

/-- The executable evaluator is defined through the proved typed terminal
observation, rather than through a parallel handwritten traversal. -/
def evaluate
    (csr : CheckedCsr)
    (frontier : Frontier)
    (frontierValid : ValidFrontier csr.graph frontier)
    (vertexColumns edgeColumns : List (List Nat))
    (terminal : Terminal)
    (expression : Expr) : List Nat :=
  resultValues <|
    (terminal.compile csr.graph.vertexCount expression).observe
      (toOrderedGraph csr edgeColumns)
      (vertexStore vertexColumns)
      (toTypedFrontier csr.graph frontier frontierValid)

/-- Main implementation-connection theorem: every result returned by the
executable CSR evaluator is exactly a public typed TA observation. -/
theorem evaluate_correct
    (csr : CheckedCsr)
    (frontier : Frontier)
    (frontierValid : ValidFrontier csr.graph frontier)
    (vertexColumns edgeColumns : List (List Nat))
    (terminal : Terminal)
    (expression : Expr) :
    evaluate csr frontier frontierValid vertexColumns edgeColumns terminal expression =
      resultValues
        ((terminal.compile csr.graph.vertexCount expression).observe
          (toOrderedGraph csr edgeColumns)
          (vertexStore vertexColumns)
          (toTypedFrontier csr.graph frontier frontierValid)) := rfl

/-- Size-indexed dense destination accumulator used by the executable CSR
lowering. The array size invariant makes every typed destination an in-bounds
write. -/
private structure DestinationAccumulator (n : Nat) where
  values : Array Nat
  size_eq : values.size = n

namespace DestinationAccumulator

def zero (n : Nat) : DestinationAccumulator n :=
  { values := Array.replicate n 0, size_eq := by simp }

def get (accumulator : DestinationAccumulator n) (destination : Fin n) : Nat :=
  accumulator.values[destination.val]'(by
    rw [accumulator.size_eq]
    exact destination.isLt)

def add
    (mapped : Verified.EdgeContext n EdgeColumns → Nat)
    (accumulator : DestinationAccumulator n)
    (context : Verified.EdgeContext n EdgeColumns) : DestinationAccumulator n :=
  let destination := context.destination
  let inBounds : destination.val < accumulator.values.size := by
    rw [accumulator.size_eq]
    exact destination.isLt
  let values := accumulator.values.set destination.val
    (accumulator.values[destination.val] + mapped context) inBounds
  { values
    size_eq := by rw [Array.size_set, accumulator.size_eq] }

theorem add_get
    (mapped : Verified.EdgeContext n EdgeColumns → Nat)
    (accumulator : DestinationAccumulator n)
    (context : Verified.EdgeContext n EdgeColumns)
    (destination : Fin n) :
    (add mapped accumulator context).get destination =
      if context.destination = destination then
        accumulator.get destination + mapped context
      else accumulator.get destination := by
  unfold add get
  simp only [Array.getElem_set]
  split
  · rename_i equalValues
    have equalFin : context.destination = destination :=
      Fin.ext equalValues
    simp [equalFin]
  · rename_i differentValues
    have differentFin : context.destination ≠ destination := by
      intro equalFin
      exact differentValues (congrArg Fin.val equalFin)
    simp [differentFin]

theorem fold_get
    (mapped : Verified.EdgeContext n EdgeColumns → Nat)
    (contexts : List (Verified.EdgeContext n EdgeColumns))
    (accumulator : DestinationAccumulator n)
    (destination : Fin n) :
    (contexts.foldl (add mapped) accumulator).get destination =
      contexts.foldl
        (fun value context =>
          if context.destination = destination then value + mapped context else value)
        (accumulator.get destination) := by
  induction contexts generalizing accumulator with
  | nil => rfl
  | cons context contexts induction =>
      simp only [List.foldl]
      rw [induction, add_get]

end DestinationAccumulator

/-- One-pass dense destination lowering. Unlike the extensional observer it
does not rescan the active edge stream once per vertex. -/
def evaluateDestinationCsr
    (csr : CheckedCsr)
    (frontier : Frontier)
    (frontierValid : ValidFrontier csr.graph frontier)
    (vertexColumns edgeColumns : List (List Nat))
    (expression : Expr) : List Nat :=
  let graph := toOrderedGraph csr edgeColumns
  let state := vertexStore vertexColumns
  let typedFrontier := toTypedFrontier csr.graph frontier frontierValid
  let mapped := (expression.compile csr.graph.vertexCount).evaluateAt state
  let accumulator := (graph.traverse typedFrontier).foldl
    (DestinationAccumulator.add mapped)
    (DestinationAccumulator.zero csr.graph.vertexCount)
  allVertices csr.graph.vertexCount |>.map accumulator.get

/-- The one-pass concrete CSR lowering is extensionally equal to the proved
typed destination terminal for every valid graph, expression, and frontier. -/
theorem evaluateDestinationCsr_correct
    (csr : CheckedCsr)
    (frontier : Frontier)
    (frontierValid : ValidFrontier csr.graph frontier)
    (vertexColumns edgeColumns : List (List Nat))
    (expression : Expr) :
    evaluateDestinationCsr csr frontier frontierValid vertexColumns edgeColumns expression =
      evaluate csr frontier frontierValid vertexColumns edgeColumns
        .reduceByDestination expression := by
  unfold evaluateDestinationCsr evaluate resultValues Terminal.compile
  unfold Observation.Terminal.observe
  unfold TraversalAlgebra.Verified.TraversalAlgebra.reduceByDestination
  apply List.map_congr_left
  intro destination membership
  unfold TraversalAlgebra.Verified.TraversalAlgebra.reduceByDestinationAt
  rw [DestinationAccumulator.fold_get]
  simp [signature, denoteMonoid, natAdd, DestinationAccumulator.zero,
    DestinationAccumulator.get, HAdd.hAdd] <;> rfl

/-- Executable evaluator selecting the proved one-pass CSR lowering for the
dense destination terminal. -/
def evaluateCsr
    (csr : CheckedCsr)
    (frontier : Frontier)
    (frontierValid : ValidFrontier csr.graph frontier)
    (vertexColumns edgeColumns : List (List Nat))
    (terminal : Terminal)
    (expression : Expr) : List Nat :=
  match terminal with
  | .reduceByDestination =>
      evaluateDestinationCsr csr frontier frontierValid vertexColumns edgeColumns expression
  | .emit | .reduceBySource =>
      evaluate csr frontier frontierValid vertexColumns edgeColumns terminal expression

theorem evaluateCsr_correct
    (csr : CheckedCsr)
    (frontier : Frontier)
    (frontierValid : ValidFrontier csr.graph frontier)
    (vertexColumns edgeColumns : List (List Nat))
    (terminal : Terminal)
    (expression : Expr) :
    evaluateCsr csr frontier frontierValid vertexColumns edgeColumns terminal expression =
      evaluate csr frontier frontierValid vertexColumns edgeColumns terminal expression := by
  cases terminal with
  | emit => rfl
  | reduceBySource => rfl
  | reduceByDestination =>
      exact evaluateDestinationCsr_correct csr frontier frontierValid
        vertexColumns edgeColumns expression

/-- Product emission can be evaluated leaf-wise without changing row order;
this justifies the Rust façade's arity-independent column reconstruction. -/
theorem emit_zip_values
    (left : Expression.Traversal (signature n) vertexType edgeType leftType)
    (right : Expression.Traversal (signature n) vertexType edgeType rightType)
    (graph : OrderedGraph n EdgeColumns)
    (state : Store n VertexColumns)
    (frontier : Verified.Frontier n) :
    Expression.Traversal.evaluate (.zip left right) graph state frontier =
      List.zip
        (Expression.Traversal.evaluate left graph state frontier)
        (Expression.Traversal.evaluate right graph state frontier) := by
  unfold Expression.Traversal.evaluate
  simp only [Expression.Traversal.evaluateAt]
  exact List.zip_map'.symm

end TraversalAlgebra.Oracle.Typed
