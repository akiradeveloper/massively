import TraversalAlgebra.CubeCLCost
import TraversalAlgebra.OracleTyped

namespace TraversalAlgebra.Oracle.Typed.CubeCL

open TraversalAlgebra.Verified
open TraversalAlgebra.Verified.Typed
open TraversalAlgebra.Verified.Typed.CubeCL

/-- Physical word widths used by the scalar oracle fragment.  Each protocol
expression selects one concrete vertex or edge column before entering the
CubeCL storage model. -/
def baseWords : Base → Nat
  | .natural | .vertexColumns | .edgeColumns => 1

/-- Model-level atomic natural addition together with its semantic equality to
the declared reduction. -/
def naturalAddAtomic (n : Nat) :
    AtomicImplementation (signature := signature n) (.naturalAdd) where
  apply := Nat.add
  correct := rfl
  operationCount := fun _ => 1

/-- Constructive evidence required by the selected abstract CubeCL target
instruction. -/
private def strategyEvidence
    (n : Nat)
    (strategy : DestinationStrategy)
    (terminal : Terminal)
    (expression : Expr) :
    strategy.Evidence (terminal.compile n expression) := by
  cases strategy with
  | sortReduce =>
      cases terminal <;> exact ()
  | atomic =>
      cases terminal with
      | emit => exact ()
      | reduceBySource => exact ()
      | reduceByDestination => exact naturalAddAtomic n

/-- Typed abstract CubeCL target program for one serializable oracle query. -/
def program
    (n : Nat)
    (strategy : DestinationStrategy)
    (terminal : Terminal)
    (expression : Expr) :
    Program (signature n)
      vertexType edgeType natural :=
  lower baseWords strategy
    (terminal.compile n expression)
      (strategyEvidence n strategy terminal expression)

/-- The same semantic program prefixed with the resource contract for
Massively's current materialized CSR traversal-control path. -/
def programWithMaterializedCsrControl
    (n : Nat)
    (strategy : DestinationStrategy)
    (terminal : Terminal)
    (expression : Expr) :
    Program (signature n)
      vertexType edgeType natural :=
  (program n strategy terminal expression)
    |>.withMaterializedCsrControl

/-- Executing the abstract CubeCL target yields exactly the public typed TA
observation. -/
theorem program_execute_correct
    (csr : CheckedCsr)
    (frontier : Frontier)
    (frontierValid : ValidFrontier csr.graph frontier)
    (strategy : DestinationStrategy)
    (terminal : Terminal)
    (expression : Expr) :
    (program csr.graph.vertexCount strategy terminal expression).execute
        (toOrderedGraph csr []) (vertexStore [])
        (toTypedFrontier csr.graph frontier frontierValid) =
      (terminal.compile csr.graph.vertexCount expression).observe
        (toOrderedGraph csr []) (vertexStore [])
        (toTypedFrontier csr.graph frontier frontierValid) := by
  unfold program
  exact lower_correct baseWords strategy
    (terminal.compile csr.graph.vertexCount expression)
    (toOrderedGraph csr []) (vertexStore [])
    (toTypedFrontier csr.graph frontier frontierValid)
    (strategyEvidence csr.graph.vertexCount strategy terminal expression)

/-- The target instruction depends only on expression syntax; its normalized
execution remains correct for every concrete vertex/edge column store. -/
theorem program_execute_correct_on_columns
    (csr : CheckedCsr)
    (frontier : Frontier)
    (frontierValid : ValidFrontier csr.graph frontier)
    (vertexColumns edgeColumns : List (List Nat))
    (strategy : DestinationStrategy)
    (terminal : Terminal)
    (expression : Expr) :
    (program csr.graph.vertexCount strategy terminal expression).execute
        (toOrderedGraph csr edgeColumns) (vertexStore vertexColumns)
        (toTypedFrontier csr.graph frontier frontierValid) =
      (terminal.compile csr.graph.vertexCount expression).observe
        (toOrderedGraph csr edgeColumns) (vertexStore vertexColumns)
        (toTypedFrontier csr.graph frontier frontierValid) := by
  unfold program
  exact lower_correct baseWords strategy
    (terminal.compile csr.graph.vertexCount expression)
    (toOrderedGraph csr edgeColumns) (vertexStore vertexColumns)
    (toTypedFrontier csr.graph frontier frontierValid)
    (strategyEvidence csr.graph.vertexCount strategy terminal expression)

/-- Materialized traversal control changes only cost, never the observation. -/
theorem programWithMaterializedCsrControl_execute_correct
    (csr : CheckedCsr)
    (frontier : Frontier)
    (frontierValid : ValidFrontier csr.graph frontier)
    (strategy : DestinationStrategy)
    (terminal : Terminal)
    (expression : Expr) :
    (programWithMaterializedCsrControl csr.graph.vertexCount strategy terminal expression).execute
        (toOrderedGraph csr []) (vertexStore [])
        (toTypedFrontier csr.graph frontier frontierValid) =
      (terminal.compile csr.graph.vertexCount expression).observe
        (toOrderedGraph csr []) (vertexStore [])
        (toTypedFrontier csr.graph frontier frontierValid) := by
  rw [programWithMaterializedCsrControl, Program.execute_withMaterializedCsrControl]
  exact program_execute_correct csr frontier frontierValid strategy terminal expression

/-- End-to-end semantic connection: the costed abstract CubeCL target has
exactly the same flattened result as the executable proved CSR oracle. -/
theorem programWithMaterializedCsrControl_result_correct
    (csr : CheckedCsr)
    (frontier : Frontier)
    (frontierValid : ValidFrontier csr.graph frontier)
    (vertexColumns edgeColumns : List (List Nat))
    (strategy : DestinationStrategy)
    (terminal : Terminal)
    (expression : Expr) :
    resultValues
        ((programWithMaterializedCsrControl csr.graph.vertexCount strategy terminal expression).execute
          (toOrderedGraph csr edgeColumns) (vertexStore vertexColumns)
          (toTypedFrontier csr.graph frontier frontierValid)) =
      evaluateCsr csr frontier frontierValid vertexColumns edgeColumns
        terminal expression := by
  change resultValues
      ((program csr.graph.vertexCount strategy terminal expression).execute
        (toOrderedGraph csr edgeColumns) (vertexStore vertexColumns)
        (toTypedFrontier csr.graph frontier frontierValid)) = _
  rw [program_execute_correct_on_columns]
  rw [evaluateCsr_correct]
  rfl

/-- Serializable resource certificate returned to implementation tests. -/
structure Certificate where
  vertices : Nat
  topologyEdges : Nat
  frontierOccurrences : Nat
  activeEdges : Nat
  expressionWork : Nat
  expressionDepth : Nat
  globalLoadWordsPerEdge : Nat
  outputWords : Nat
  strategy : DestinationStrategy
  fused : Cost
  materializedCsrControl : Cost
  withMaterializedCsrControl : Cost
deriving Repr, DecidableEq

private def terminalTraversal (n : Nat) (terminal : Terminal) (expression : Expr) :
    TraversalAlgebra.Expression.Traversal
      (signature n) vertexType edgeType natural :=
  match terminal.compile n expression with
  | .emit traversal => traversal
  | .reduceBySource traversal _ => traversal
  | .reduceByDestination traversal _ => traversal

/-- Compute the proof-backed symbolic CubeCL certificate for one checked CSR
query. -/
def certificate
    (machine : Machine)
    (csr : CheckedCsr)
    (frontier : Frontier)
    (frontierValid : ValidFrontier csr.graph frontier)
    (strategy : DestinationStrategy)
    (terminal : Terminal)
    (expression : Expr) : Certificate :=
  let graph := toOrderedGraph csr []
  let typedFrontier := toTypedFrontier csr.graph frontier frontierValid
  let traversal := terminalTraversal csr.graph.vertexCount terminal expression
  let fused := program csr.graph.vertexCount strategy terminal expression
  let withMaterializedCsrControl :=
    programWithMaterializedCsrControl csr.graph.vertexCount strategy terminal expression
  { vertices := csr.graph.vertexCount
    topologyEdges := graphEdgeCount graph
    frontierOccurrences := frontier.length
    activeEdges := activeEdgeCount graph typedFrontier
    expressionWork := traversal.work
    expressionDepth := traversal.depth
    globalLoadWordsPerEdge := traversalGlobalLoadWords baseWords traversal
    outputWords := 1
    strategy
    fused := fused.cost machine graph typedFrontier
    materializedCsrControl := (materializedCsrControlPlan graph typedFrontier).cost machine
    withMaterializedCsrControl :=
      withMaterializedCsrControl.cost machine graph typedFrontier }

/-- The executable certificate's active-edge field is exactly the concrete CSR
traversal length, not merely an estimate from graph dimensions. -/
theorem certificate_activeEdges
    (machine : Machine)
    (csr : CheckedCsr)
    (frontier : Frontier)
    (frontierValid : ValidFrontier csr.graph frontier)
    (strategy : DestinationStrategy)
    (terminal : Terminal)
    (expression : Expr) :
    (certificate machine csr frontier frontierValid strategy terminal expression).activeEdges =
      (csr.graph.traverse frontier).length := by
  unfold certificate
  exact activeEdgeCount_corresponds csr [] frontier frontierValid

/-- The scalar certificate exposes the universal fused-emission work theorem
without weakening it at the serialization boundary. -/
theorem certificate_emit_scalarWork
    (machine : Machine)
    (csr : CheckedCsr)
    (frontier : Frontier)
    (frontierValid : ValidFrontier csr.graph frontier)
    (expression : Expr) :
    let result := certificate machine csr frontier frontierValid
      .sortReduce .emit expression
    result.fused.scalarWork =
      result.activeEdges * (result.expressionWork + result.outputWords) := by
  change
    ((lowerEmit baseWords (expression.compile csr.graph.vertexCount)).cost machine
      (toOrderedGraph csr [])
      (toTypedFrontier csr.graph frontier frontierValid)).scalarWork =
    activeEdgeCount (toOrderedGraph csr [])
      (toTypedFrontier csr.graph frontier frontierValid) *
        ((expression.compile csr.graph.vertexCount).work +
          ValueType.words baseWords natural)
  exact lowerEmit_scalarWork machine baseWords
    (expression.compile csr.graph.vertexCount)
    (toOrderedGraph csr [])
    (toTypedFrontier csr.graph frontier frontierValid)

/-- Exact global-load traffic also survives certificate serialization. -/
theorem certificate_emit_globalLoads
    (machine : Machine)
    (csr : CheckedCsr)
    (frontier : Frontier)
    (frontierValid : ValidFrontier csr.graph frontier)
    (expression : Expr) :
    let result := certificate machine csr frontier frontierValid
      .sortReduce .emit expression
    result.fused.globalLoads =
      result.activeEdges * result.globalLoadWordsPerEdge := by
  change
    ((lowerEmit baseWords (expression.compile csr.graph.vertexCount)).cost machine
      (toOrderedGraph csr [])
      (toTypedFrontier csr.graph frontier frontierValid)).globalLoads =
    activeEdgeCount (toOrderedGraph csr [])
      (toTypedFrontier csr.graph frontier frontierValid) *
        traversalGlobalLoadWords baseWords
          (expression.compile csr.graph.vertexCount)
  exact lowerEmit_globalLoads machine baseWords
    (expression.compile csr.graph.vertexCount)
    (toOrderedGraph csr [])
    (toTypedFrontier csr.graph frontier frontierValid)

/-- Exact source-reduction work at the executable certificate boundary. -/
theorem certificate_source_scalarWork
    (machine : Machine)
    (csr : CheckedCsr)
    (frontier : Frontier)
    (frontierValid : ValidFrontier csr.graph frontier)
    (expression : Expr) :
    let result := certificate machine csr frontier frontierValid
      .sortReduce .reduceBySource expression
    result.fused.scalarWork =
      result.frontierOccurrences +
        result.activeEdges * (result.expressionWork + 1) := by
  change
    ((lowerSource baseWords (expression.compile csr.graph.vertexCount)
      (.naturalAdd : MonoidSymbol natural)).cost machine
        (toOrderedGraph csr [])
        (toTypedFrontier csr.graph frontier frontierValid)).scalarWork =
      frontier.length +
        activeEdgeCount (toOrderedGraph csr [])
          (toTypedFrontier csr.graph frontier frontierValid) *
            ((expression.compile csr.graph.vertexCount).work + 1)
  rw [lowerSource_scalarWork]
  simp
  rfl

/-- Exact linear work of the natural-add atomic destination certificate. -/
theorem certificate_destinationAtomic_scalarWork
    (machine : Machine)
    (csr : CheckedCsr)
    (frontier : Frontier)
    (frontierValid : ValidFrontier csr.graph frontier)
    (expression : Expr) :
    let result := certificate machine csr frontier frontierValid
      .atomic .reduceByDestination expression
    result.fused.scalarWork =
      result.vertices + result.activeEdges * (result.expressionWork + 1) := by
  change
    ((lowerDestinationAtomic baseWords
      (expression.compile csr.graph.vertexCount) .naturalAdd
      (naturalAddAtomic csr.graph.vertexCount)).cost machine
        (toOrderedGraph csr [])
        (toTypedFrontier csr.graph frontier frontierValid)).scalarWork =
      csr.graph.vertexCount +
        activeEdgeCount (toOrderedGraph csr [])
          (toTypedFrontier csr.graph frontier frontierValid) *
            ((expression.compile csr.graph.vertexCount).work + 1)
  exact lowerDestinationAtomic_scalarWork machine baseWords
    (expression.compile csr.graph.vertexCount) .naturalAdd
    (naturalAddAtomic csr.graph.vertexCount)
    (toOrderedGraph csr [])
    (toTypedFrontier csr.graph frontier frontierValid)

/-- Natural addition uses exactly one modeled scalar atomic per active edge. -/
theorem certificate_destinationAtomic_operations
    (machine : Machine)
    (csr : CheckedCsr)
    (frontier : Frontier)
    (frontierValid : ValidFrontier csr.graph frontier)
    (expression : Expr) :
    let result := certificate machine csr frontier frontierValid
      .atomic .reduceByDestination expression
    result.fused.atomicOperations = result.activeEdges := by
  change
    ((lowerDestinationAtomic baseWords
      (expression.compile csr.graph.vertexCount) .naturalAdd
      (naturalAddAtomic csr.graph.vertexCount)).cost machine
        (toOrderedGraph csr [])
        (toTypedFrontier csr.graph frontier frontierValid)).atomicOperations =
      activeEdgeCount (toOrderedGraph csr [])
        (toTypedFrontier csr.graph frontier frontierValid)
  rw [lowerDestinationAtomic_operations]
  simp [naturalAddAtomic]
  rfl

/-- Exact quasilinear work exposed by the general destination certificate. -/
theorem certificate_destinationSort_scalarWork
    (machine : Machine)
    (csr : CheckedCsr)
    (frontier : Frontier)
    (frontierValid : ValidFrontier csr.graph frontier)
    (expression : Expr) :
    let result := certificate machine csr frontier frontierValid
      .sortReduce .reduceByDestination expression
    result.fused.scalarWork =
      result.vertices + result.activeEdges * result.expressionWork +
        result.activeEdges * reductionDepth result.activeEdges +
        result.activeEdges := by
  change
    ((lowerDestinationSort baseWords
      (expression.compile csr.graph.vertexCount) .naturalAdd).cost machine
        (toOrderedGraph csr [])
        (toTypedFrontier csr.graph frontier frontierValid)).scalarWork =
      csr.graph.vertexCount +
        activeEdgeCount (toOrderedGraph csr [])
          (toTypedFrontier csr.graph frontier frontierValid) *
            (expression.compile csr.graph.vertexCount).work +
        activeEdgeCount (toOrderedGraph csr [])
          (toTypedFrontier csr.graph frontier frontierValid) *
            reductionDepth
              (activeEdgeCount (toOrderedGraph csr [])
                (toTypedFrontier csr.graph frontier frontierValid)) +
        activeEdgeCount (toOrderedGraph csr [])
          (toTypedFrontier csr.graph frontier frontierValid)
  exact lowerDestinationSort_scalarWork machine baseWords
    (expression.compile csr.graph.vertexCount) .naturalAdd
    (toOrderedGraph csr [])
    (toTypedFrontier csr.graph frontier frontierValid)

/-- Materialized CSR control is an exact prefix of the fused terminal plan in
the exported certificate. -/
theorem certificate_withMaterializedCsrControl
    (machine : Machine)
    (csr : CheckedCsr)
    (frontier : Frontier)
    (frontierValid : ValidFrontier csr.graph frontier)
    (strategy : DestinationStrategy)
    (terminal : Terminal)
    (expression : Expr) :
    let result := certificate machine csr frontier frontierValid
      strategy terminal expression
    result.withMaterializedCsrControl =
      result.materializedCsrControl.seq result.fused := by
  simp only [certificate]
  exact Program.cost_withMaterializedCsrControl machine
    (toOrderedGraph csr [])
    (toTypedFrontier csr.graph frontier frontierValid)
    (program csr.graph.vertexCount strategy terminal expression)

/-- The certificate is observationally erasable: evaluating with a certificate
returns exactly the existing proved CSR evaluator result. -/
def evaluateWithCertificate
    (machine : Machine)
    (csr : CheckedCsr)
    (frontier : Frontier)
    (frontierValid : ValidFrontier csr.graph frontier)
    (vertexColumns edgeColumns : List (List Nat))
    (strategy : DestinationStrategy)
    (terminal : Terminal)
    (expression : Expr) : List Nat × Certificate :=
  (evaluateCsr csr frontier frontierValid vertexColumns edgeColumns terminal expression,
   certificate machine csr frontier frontierValid strategy terminal expression)

@[simp]
theorem evaluateWithCertificate_result
    (machine : Machine)
    (csr : CheckedCsr)
    (frontier : Frontier)
    (frontierValid : ValidFrontier csr.graph frontier)
    (vertexColumns edgeColumns : List (List Nat))
    (strategy : DestinationStrategy)
    (terminal : Terminal)
    (expression : Expr) :
    (evaluateWithCertificate machine csr frontier frontierValid
      vertexColumns edgeColumns strategy terminal expression).1 =
      evaluateCsr csr frontier frontierValid vertexColumns edgeColumns
        terminal expression := rfl

end TraversalAlgebra.Oracle.Typed.CubeCL
