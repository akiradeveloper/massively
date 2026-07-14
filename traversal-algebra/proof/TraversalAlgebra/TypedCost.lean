import TraversalAlgebra.TypedTraversalNormalization

namespace TraversalAlgebra.Verified.Typed

variable {n : Nat} {Base : Type} {denoteBase : Base → Type}
variable {signature : Signature n Base denoteBase}
variable {vertex edge message : ValueType Base}

namespace Term

/-- Tree/DAG syntax nodes. `let1` counts once and refers to its input through a
typed variable, so repeated reads in the body do not copy the input syntax. -/
def nodeCount : {context : List (ValueType Base)} →
    {output : ValueType Base} → Term signature context output → Nat
  | _, _, .read _ => 1
  | _, _, .literal _ => 1
  | _, _, .pair left right => 1 + nodeCount left + nodeCount right
  | _, _, .apply _ argument => 1 + nodeCount argument
  | _, _, .let1 input body => 1 + nodeCount input + nodeCount body

/-- Unit-cost scalar work. Reads, literals, pair construction, and primitive
invocation cost one; `let1` itself is administrative and evaluates each child
once. Signature-specific weighted costs can refine this baseline. -/
def work : {context : List (ValueType Base)} →
    {output : ValueType Base} → Term signature context output → Nat
  | _, _, .read _ => 1
  | _, _, .literal _ => 1
  | _, _, .pair left right => work left + work right + 1
  | _, _, .apply _ argument => work argument + 1
  | _, _, .let1 input body => work input + work body

/-- Scalar dependence depth. Pair branches may run in parallel; a primitive
follows its argument, and a shared body follows its bound input. -/
def depth : {context : List (ValueType Base)} →
    {output : ValueType Base} → Term signature context output → Nat
  | _, _, .read _ => 1
  | _, _, .literal _ => 1
  | _, _, .pair left right => Nat.max (depth left) (depth right) + 1
  | _, _, .apply _ argument => depth argument + 1
  | _, _, .let1 input body => depth input + depth body

/-- A conservative peak count of live per-edge scalar temporaries. This does
not count full-stream columns. -/
def scalarTemporaries : {context : List (ValueType Base)} →
    {output : ValueType Base} → Term signature context output → Nat
  | _, _, .read _ => 0
  | _, _, .literal _ => 0
  | _, _, .pair left right =>
      Nat.max (scalarTemporaries left) (scalarTemporaries right + 1)
  | _, _, .apply _ argument => scalarTemporaries argument
  | _, _, .let1 input body =>
      Nat.max (scalarTemporaries input) (scalarTemporaries body + 1)

end Term

namespace TraversalAlgebra.Expression.Traversal

/-- Size of the independent traversal syntax, including scalar mapper terms. -/
def nodeCount : {output : ValueType Base} →
    Traversal signature vertex edge output → Nat
  | _, .read _ => 1
  | _, .literal _ => 1
  | _, .map input mapper => 1 + nodeCount input + mapper.nodeCount
  | _, .zip left right => 1 + nodeCount left + nodeCount right

/-- Unit-cost work to evaluate one edge-context element. -/
def work : {output : ValueType Base} →
    Traversal signature vertex edge output → Nat
  | _, .read _ => 1
  | _, .literal _ => 1
  | _, .map input mapper => work input + mapper.work
  | _, .zip left right => work left + work right + 1

/-- Dependence depth to evaluate one edge-context element. -/
def depth : {output : ValueType Base} →
    Traversal signature vertex edge output → Nat
  | _, .read _ => 1
  | _, .literal _ => 1
  | _, .map input mapper => depth input + mapper.depth
  | _, .zip left right => Nat.max (depth left) (depth right) + 1

/-- Peak live per-edge scalar temporaries in the compositional evaluator. -/
def scalarTemporaries : {output : ValueType Base} →
    Traversal signature vertex edge output → Nat
  | _, .read _ => 0
  | _, .literal _ => 0
  | _, .map input mapper =>
      Nat.max (scalarTemporaries input) (mapper.scalarTemporaries + 1)
  | _, .zip left right =>
      Nat.max (scalarTemporaries left) (scalarTemporaries right + 1)

/-- Full-stream passes in a deliberately unfused reference schedule. Every
pointwise map or zip could materialize one intermediate column; the fused
normal form has zero such mandatory passes. -/
def stagedMaterializations : {output : ValueType Base} →
    Traversal signature vertex edge output → Nat
  | _, .read _ => 0
  | _, .literal _ => 0
  | _, .map input _ => stagedMaterializations input + 1
  | _, .zip left right =>
      stagedMaterializations left + stagedMaterializations right + 1

/-- Total full-stream temporary columns in the same unfused reference
schedule. This is a volume count, not a peak-liveness claim. -/
abbrev stagedTemporaryColumns
    (expression : Traversal signature vertex edge output) : Nat :=
  stagedMaterializations expression

/-- Sharing-preserving normalization is exactly linear in source syntax size. -/
@[simp]
theorem normalize_nodeCount
    (expression : Traversal signature vertex edge output) :
    expression.normalize.nodeCount = expression.nodeCount := by
  induction expression with
  | read reference => rfl
  | literal symbol => rfl
  | map input mapper induction =>
      simp only [normalize, Term.share, Term.nodeCount, nodeCount, induction]
  | zip left right leftInduction rightInduction =>
      simp only [normalize, Term.nodeCount, nodeCount,
        leftInduction, rightInduction]

/-- Normalization neither duplicates nor drops scalar work. -/
@[simp]
theorem normalize_work
    (expression : Traversal signature vertex edge output) :
    expression.normalize.work = expression.work := by
  induction expression with
  | read reference => rfl
  | literal symbol => rfl
  | map input mapper induction =>
      simp only [normalize, Term.share, Term.work, work, induction]
  | zip left right leftInduction rightInduction =>
      simp only [normalize, Term.work, work, leftInduction, rightInduction]

/-- Normalization preserves scalar dependence depth exactly. -/
@[simp]
theorem normalize_depth
    (expression : Traversal signature vertex edge output) :
    expression.normalize.depth = expression.depth := by
  induction expression with
  | read reference => rfl
  | literal symbol => rfl
  | map input mapper induction =>
      simp only [normalize, Term.share, Term.depth, depth, induction]
  | zip left right leftInduction rightInduction =>
      simp only [normalize, Term.depth, depth, leftInduction, rightInduction]

/-- Normalization preserves the conservative per-edge scalar storage bound. -/
@[simp]
theorem normalize_scalarTemporaries
    (expression : Traversal signature vertex edge output) :
    expression.normalize.scalarTemporaries =
      expression.scalarTemporaries := by
  induction expression with
  | read reference => rfl
  | literal symbol => rfl
  | map input mapper induction =>
      simp only [normalize, Term.share, Term.scalarTemporaries,
        scalarTemporaries, induction]
  | zip left right leftInduction rightInduction =>
      simp only [normalize, Term.scalarTemporaries, scalarTemporaries,
        leftInduction, rightInduction]

end TraversalAlgebra.Expression.Traversal

namespace FrontierPolicy

/-- Syntax size of the current frontier policy. -/
def nodeCount (policy : FrontierPolicy signature vertex) : Nat :=
  match policy with
  | .dense predicate => predicate.nodeCount + 1
  | .sparsePreserve predicate => predicate.nodeCount + 1

/-- Predicate work for one superstep. Dense selection evaluates all vertices;
sparse selection evaluates exactly the candidate occurrences. -/
def work (policy : FrontierPolicy signature vertex)
    (candidateCount : Nat) : Nat :=
  match policy with
  | .dense predicate => n * predicate.work
  | .sparsePreserve predicate => candidateCount * predicate.work

/-- Frontier-predicate dependence depth. Candidate occurrences are mutually
parallel in this language-level model. -/
def depth (policy : FrontierPolicy signature vertex) : Nat :=
  match policy with
  | .dense predicate => predicate.depth
  | .sparsePreserve predicate => predicate.depth

/-- Per-vertex scalar temporary bound of frontier selection. -/
def scalarTemporaries (policy : FrontierPolicy signature vertex) : Nat :=
  match policy with
  | .dense predicate => predicate.scalarTemporaries
  | .sparsePreserve predicate => predicate.scalarTemporaries

/-- Sparse filtering has work proportional to candidate occurrences. -/
@[simp]
theorem work_sparsePreserve
    (predicate : Term signature (SelectionContext vertex) .boolean)
    (candidateCount : Nat) :
    (FrontierPolicy.sparsePreserve predicate).work candidateCount =
      candidateCount * predicate.work := rfl

/-- Whenever the candidate stream length is at most the vertex count, sparse
predicate work is no greater than dense predicate work. -/
theorem sparsePreserve_work_le_dense
    (predicate : Term signature (SelectionContext vertex) .boolean)
    (candidateCount : Nat)
    (bounded : candidateCount ≤ n) :
    (FrontierPolicy.sparsePreserve predicate).work candidateCount ≤
      (FrontierPolicy.dense predicate).work candidateCount := by
  exact Nat.mul_le_mul_right predicate.work bounded

end FrontierPolicy

/-- Parallel reduction-tree depth used by the language-level cost model.
Zero messages have zero depth; otherwise a balanced binary tree is bounded by
`log2(m) + 1`. -/
def reductionDepth (messageCount : Nat) : Nat :=
  if messageCount = 0 then 0 else Nat.log2 messageCount + 1

/-- Logical active-edge occurrences, including multiplicity in the frontier. -/
def activeEdgeCount
    (graph : OrderedGraph n EdgePayload) (frontier : Frontier n) : Nat :=
  (graph.traverse frontier).length

namespace TraversalAlgebra

/-- Syntax size of one sharing-aware single-push normal form. -/
def Program.nodeCount
    (program : Program signature vertex edge message) : Nat :=
  program.destinationPush.pullMap.nodeCount +
    program.vertexMap.nodeCount + program.frontierCompact.nodeCount + 3

/-- Unit work of one normal-form superstep. Destination combination is charged
once per active edge, updates once per vertex, and frontier selection according
to its dense or candidate-occurrence policy. -/
def Program.work
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : Program signature vertex edge message)
    (configuration : Configuration n (vertex.denote n denoteBase)) : Nat :=
  let edges := activeEdgeCount graph configuration.frontier
  edges * program.destinationPush.pullMap.work + edges +
    n * program.vertexMap.work + program.frontierCompact.work edges

/-- Critical-path bound of one normal-form superstep. -/
def Program.depth
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : Program signature vertex edge message)
    (configuration : Configuration n (vertex.denote n denoteBase)) : Nat :=
  let edges := activeEdgeCount graph configuration.frontier
  program.destinationPush.pullMap.depth + reductionDepth edges +
    program.vertexMap.depth + program.frontierCompact.depth

/-- Peak local scalar temporaries across the three normal-form phases. -/
def Program.scalarTemporaries
    (program : Program signature vertex edge message) : Nat :=
  Nat.max program.destinationPush.pullMap.scalarTemporaries
    (Nat.max program.vertexMap.scalarTemporaries
      program.frontierCompact.scalarTemporaries)

/-- A fused normal form requires no full-stream intermediate edge column. -/
def Program.fullStreamTemporaryValues
    (_graph : OrderedGraph n (edge.denote n denoteBase))
    (_program : Program signature vertex edge message)
    (_configuration : Configuration n (vertex.denote n denoteBase)) : Nat := 0

/-- A fused normal form requires no intermediate pointwise materialization
pass before its destination terminal. -/
def Program.materializations
    (_program : Program signature vertex edge message) : Nat := 0

end TraversalAlgebra

namespace TraversalAlgebra.Expression.Program

/-- Syntax size of one independent closed expression. -/
def nodeCount (program : Program signature vertex edge message) : Nat :=
  match program with
  | .compact (.updateByDestination
      (.reduceByDestination traversal _) update) frontier =>
      traversal.nodeCount + update.nodeCount + frontier.nodeCount + 3

/-- Unit work of one compositional superstep. -/
def work
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : Program signature vertex edge message)
    (configuration : Configuration n (vertex.denote n denoteBase)) : Nat :=
  match program with
  | .compact (.updateByDestination
      (.reduceByDestination traversal _) update) frontier =>
      let edges := activeEdgeCount graph configuration.frontier
      edges * traversal.work + edges + n * update.work + frontier.work edges

/-- Critical-path bound of one compositional superstep. -/
def depth
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : Program signature vertex edge message)
    (configuration : Configuration n (vertex.denote n denoteBase)) : Nat :=
  match program with
  | .compact (.updateByDestination
      (.reduceByDestination traversal _) update) frontier =>
      let edges := activeEdgeCount graph configuration.frontier
      traversal.depth + reductionDepth edges + update.depth + frontier.depth

/-- Peak local scalar temporaries across expression, update, and selection. -/
def scalarTemporaries
    (program : Program signature vertex edge message) : Nat :=
  match program with
  | .compact (.updateByDestination
      (.reduceByDestination traversal _) update) frontier =>
      Nat.max traversal.scalarTemporaries
        (Nat.max update.scalarTemporaries frontier.scalarTemporaries)

/-- Full-stream temporary-value volume of an unfused reference schedule. -/
def fullStreamTemporaryValues
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : Program signature vertex edge message)
    (configuration : Configuration n (vertex.denote n denoteBase)) : Nat :=
  match program with
  | .compact (.updateByDestination
      (.reduceByDestination traversal _) _) _ =>
      activeEdgeCount graph configuration.frontier *
        traversal.stagedTemporaryColumns

/-- Intermediate full-stream pointwise passes of an unfused schedule. -/
def materializations
    (program : Program signature vertex edge message) : Nat :=
  match program with
  | .compact (.updateByDestination
      (.reduceByDestination traversal _) _) _ =>
      traversal.stagedMaterializations

/-- Program normalization has exactly linear syntax size (factor one). -/
@[simp]
theorem normalize_nodeCount
    (program : Program signature vertex edge message) :
    (normalize program).nodeCount = program.nodeCount := by
  cases program with
  | compact vertexStage frontier =>
      cases vertexStage with
      | updateByDestination destination update =>
          cases destination with
          | reduceByDestination traversal reduction =>
              simp only [TraversalAlgebra.Program.nodeCount, normalize,
                Destination.normalize, nodeCount,
                Traversal.normalize_nodeCount]

/-- Program normalization preserves language-level work exactly. -/
@[simp]
theorem normalize_work
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : Program signature vertex edge message)
    (configuration : Configuration n (vertex.denote n denoteBase)) :
    (normalize program).work graph configuration =
      program.work graph configuration := by
  cases program with
  | compact vertexStage frontier =>
      cases vertexStage with
      | updateByDestination destination update =>
          cases destination with
          | reduceByDestination traversal reduction =>
              simp only [TraversalAlgebra.Program.work, normalize,
                Destination.normalize, work, Traversal.normalize_work]

/-- Program normalization preserves critical-path depth exactly. -/
@[simp]
theorem normalize_depth
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : Program signature vertex edge message)
    (configuration : Configuration n (vertex.denote n denoteBase)) :
    (normalize program).depth graph configuration =
      program.depth graph configuration := by
  cases program with
  | compact vertexStage frontier =>
      cases vertexStage with
      | updateByDestination destination update =>
          cases destination with
          | reduceByDestination traversal reduction =>
              simp only [TraversalAlgebra.Program.depth, normalize,
                Destination.normalize, depth, Traversal.normalize_depth]

/-- Program normalization preserves local scalar storage exactly. -/
@[simp]
theorem normalize_scalarTemporaries
    (program : Program signature vertex edge message) :
    (normalize program).scalarTemporaries = program.scalarTemporaries := by
  cases program with
  | compact vertexStage frontier =>
      cases vertexStage with
      | updateByDestination destination update =>
          cases destination with
          | reduceByDestination traversal reduction =>
              simp only [TraversalAlgebra.Program.scalarTemporaries, normalize,
                Destination.normalize, scalarTemporaries,
                Traversal.normalize_scalarTemporaries]

/-- Fused normalization never increases full-stream temporary storage. -/
theorem normalize_fullStreamTemporaryValues_le
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (program : Program signature vertex edge message)
    (configuration : Configuration n (vertex.denote n denoteBase)) :
    (normalize program).fullStreamTemporaryValues graph configuration ≤
      program.fullStreamTemporaryValues graph configuration := by
  simp only [TraversalAlgebra.Program.fullStreamTemporaryValues]
  exact Nat.zero_le _

/-- Fused normalization never increases intermediate materialization passes. -/
theorem normalize_materializations_le
    (program : Program signature vertex edge message) :
    (normalize program).materializations ≤ program.materializations := by
  simp only [TraversalAlgebra.Program.materializations]
  exact Nat.zero_le _

end TraversalAlgebra.Expression.Program
end TraversalAlgebra.Verified.Typed
