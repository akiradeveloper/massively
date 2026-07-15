import TraversalAlgebra.TypedObservations

namespace TraversalAlgebra.Verified.Typed.CubeCL

variable {n : Nat} {Base : Type} {denoteBase : Base → Type}
variable {signature : Signature n Base denoteBase}
variable {vertex edge output : ValueType Base}

open TraversalAlgebra.Expression
open TraversalAlgebra.Observation

/-- Backend-neutral parameters exposed by CubeCL's hierarchical execution
model.  Instruction latency and physical bandwidth deliberately remain outside
this model. -/
structure Machine where
  workgroupSize : Nat
  subgroupSize : Nat
deriving Repr, DecidableEq

namespace Machine

/-- A zero workgroup size is normalized to one, keeping the cost interpreter
total even for an invalid external machine description. -/
def normalizedWorkgroupSize (machine : Machine) : Nat :=
  Nat.max 1 machine.workgroupSize

def normalizedSubgroupSize (machine : Machine) : Nat :=
  Nat.max 1 machine.subgroupSize

/-- Number of one-dimensional workgroups dispatched for `logicalThreads`. -/
def workgroups (machine : Machine) (logicalThreads : Nat) : Nat :=
  if logicalThreads = 0 then 0
  else (logicalThreads - 1) / machine.normalizedWorkgroupSize + 1

/-- Physical work-items scheduled after workgroup padding. -/
def scheduledThreads (machine : Machine) (logicalThreads : Nat) : Nat :=
  machine.workgroups logicalThreads * machine.normalizedWorkgroupSize

/-- Subgroups resident in one padded workgroup. -/
def subgroupsPerWorkgroup (machine : Machine) : Nat :=
  (machine.normalizedWorkgroupSize - 1) / machine.normalizedSubgroupSize + 1

/-- Scheduled subgroups across the whole dispatch. -/
def scheduledSubgroups (machine : Machine) (logicalThreads : Nat) : Nat :=
  machine.workgroups logicalThreads * machine.subgroupsPerWorkgroup

@[simp]
theorem workgroups_zero (machine : Machine) : machine.workgroups 0 = 0 := by
  simp [workgroups]

@[simp]
theorem scheduledThreads_zero (machine : Machine) :
    machine.scheduledThreads 0 = 0 := by
  simp [scheduledThreads]

@[simp]
theorem scheduledSubgroups_zero (machine : Machine) :
    machine.scheduledSubgroups 0 = 0 := by
  simp [scheduledSubgroups]

end Machine

/-- Exact symbolic resources charged by the abstract CubeCL interpreter.

`allocatedWords` is allocation volume, not a peak-liveness assertion.  The
separate materialization counter records full-stream intermediate buffers. -/
structure Cost where
  logicalThreads : Nat := 0
  scheduledThreads : Nat := 0
  scheduledSubgroups : Nat := 0
  scalarWork : Nat := 0
  span : Nat := 0
  globalLoads : Nat := 0
  globalStores : Nat := 0
  hostReadWords : Nat := 0
  atomicOperations : Nat := 0
  barriers : Nat := 0
  launches : Nat := 0
  allocatedWords : Nat := 0
  materializations : Nat := 0
deriving Repr, DecidableEq

namespace Cost

def zero : Cost := {}

/-- Sequential composition of resource certificates.  Work, traffic, and
allocation volume add; kernel critical paths add because stages are separated
by a launch boundary. -/
def seq (left right : Cost) : Cost where
  logicalThreads := left.logicalThreads + right.logicalThreads
  scheduledThreads := left.scheduledThreads + right.scheduledThreads
  scheduledSubgroups := left.scheduledSubgroups + right.scheduledSubgroups
  scalarWork := left.scalarWork + right.scalarWork
  span := left.span + right.span
  globalLoads := left.globalLoads + right.globalLoads
  globalStores := left.globalStores + right.globalStores
  hostReadWords := left.hostReadWords + right.hostReadWords
  atomicOperations := left.atomicOperations + right.atomicOperations
  barriers := left.barriers + right.barriers
  launches := left.launches + right.launches
  allocatedWords := left.allocatedWords + right.allocatedWords
  materializations := left.materializations + right.materializations

@[simp]
theorem zero_seq (cost : Cost) : zero.seq cost = cost := by
  cases cost
  simp [zero, seq]

@[simp]
theorem seq_zero (cost : Cost) : cost.seq zero = cost := by
  cases cost
  simp [seq, zero]

theorem seq_assoc (first second third : Cost) :
    (first.seq second).seq third = first.seq (second.seq third) := by
  cases first
  cases second
  cases third
  simp [seq, Nat.add_assoc]

end Cost

/-- One abstract CubeCL kernel dispatch.  All counters except scheduled thread
padding and launch presence are already total kernel counts. -/
structure Kernel where
  logicalThreads : Nat
  scalarWork : Nat
  span : Nat
  globalLoads : Nat
  globalStores : Nat
  hostReadWords : Nat := 0
  atomicOperations : Nat := 0
  barriers : Nat := 0
  allocatedWords : Nat := 0
  materializations : Nat := 0
deriving Repr, DecidableEq

namespace Kernel

/-- Construct a regular pointwise kernel from per-logical-thread costs. -/
def pointwise
    (logicalThreads workPerThread spanPerThread loadsPerThread
      storesPerThread : Nat)
    (atomicPerThread := 0) (barriers := 0)
    (allocatedWords := 0) (materializations := 0) : Kernel where
  logicalThreads
  scalarWork := logicalThreads * workPerThread
  span := if logicalThreads = 0 then 0 else spanPerThread
  globalLoads := logicalThreads * loadsPerThread
  globalStores := logicalThreads * storesPerThread
  atomicOperations := logicalThreads * atomicPerThread
  barriers
  allocatedWords
  materializations

/-- Interpret one kernel under a CubeCL machine configuration. -/
def cost (machine : Machine) (kernel : Kernel) : Cost where
  logicalThreads := kernel.logicalThreads
  scheduledThreads := machine.scheduledThreads kernel.logicalThreads
  scheduledSubgroups := machine.scheduledSubgroups kernel.logicalThreads
  scalarWork := kernel.scalarWork
  span := if kernel.logicalThreads = 0 then 0 else kernel.span
  globalLoads := kernel.globalLoads
  globalStores := kernel.globalStores
  hostReadWords := kernel.hostReadWords
  atomicOperations := kernel.atomicOperations
  barriers := kernel.barriers
  launches := if kernel.logicalThreads = 0 then 0 else 1
  allocatedWords := kernel.allocatedWords
  materializations := kernel.materializations

@[simp]
theorem pointwise_scalarWork
    (machine : Machine) (threads work depth loads stores atomics barriers
      allocated materializations : Nat) :
    ((pointwise threads work depth loads stores atomics barriers allocated
      materializations).cost machine).scalarWork = threads * work := rfl

@[simp]
theorem pointwise_globalLoads
    (machine : Machine) (threads work depth loads stores atomics barriers
      allocated materializations : Nat) :
    ((pointwise threads work depth loads stores atomics barriers allocated
      materializations).cost machine).globalLoads = threads * loads := rfl

end Kernel

/-- A launch-ordered CubeCL execution plan. -/
abbrev Plan := List Kernel

namespace Plan

def cost (machine : Machine) : Plan → Cost
  | [] => .zero
  | kernel :: rest => (kernel.cost machine).seq (cost machine rest)

@[simp]
theorem cost_nil (machine : Machine) : cost machine [] = .zero := rfl

@[simp]
theorem cost_cons (machine : Machine) (kernel : Kernel) (rest : Plan) :
    cost machine (kernel :: rest) =
      (kernel.cost machine).seq (cost machine rest) := rfl

@[simp]
theorem cost_append (machine : Machine) (left right : Plan) :
    cost machine (left ++ right) =
      (cost machine left).seq (cost machine right) := by
  induction left with
  | nil => simp
  | cons kernel rest induction =>
      simp only [List.cons_append, cost_cons, induction, Cost.seq_assoc]

end Plan

namespace ValueType

/-- Number of scalar storage words in a recursively nested value.  The base
width function makes the model signature- and backend-parametric. -/
def words (baseWords : Base → Nat) : ValueType Base → Nat
  | .boolean => 1
  | .index => 1
  | .base name => baseWords name
  | .product left right => words baseWords left + words baseWords right

@[simp]
theorem words_product (baseWords : Base → Nat)
    (left right : ValueType Base) :
    words baseWords (.product left right) =
      words baseWords left + words baseWords right := rfl

end ValueType

/-- Global words read by one scalar-term invocation.  Reads in the body of a
`let1` are reads of its local shared value, so only the bound input contributes
global traffic. -/
def termGlobalLoadWords (baseWords : Base → Nat) :
    {context : List (ValueType Base)} → {result : ValueType Base} →
      Term signature context result → Nat
  | _, result, .read _ => ValueType.words baseWords result
  | _, _, .literal _ => 0
  | _, _, .pair left right =>
      termGlobalLoadWords baseWords left + termGlobalLoadWords baseWords right
  | _, _, .apply _ argument => termGlobalLoadWords baseWords argument
  | _, _, .let1 input _ => termGlobalLoadWords baseWords input

/-- Global words pulled for one active edge before normalization. -/
def traversalGlobalLoadWords (baseWords : Base → Nat) :
    {result : ValueType Base} →
      Traversal signature vertex edge result → Nat
  | _, .read (output := result) _ => ValueType.words baseWords result
  | _, .literal _ => 0
  | _, .map input _ => traversalGlobalLoadWords baseWords input
  | _, .zip left right =>
      traversalGlobalLoadWords baseWords left +
        traversalGlobalLoadWords baseWords right

/-- Sharing-aware normalization preserves global traffic per active edge. -/
@[simp]
theorem normalize_globalLoadWords
    (baseWords : Base → Nat)
    (expression : Traversal signature vertex edge output) :
    termGlobalLoadWords baseWords expression.normalize =
      traversalGlobalLoadWords baseWords expression := by
  induction expression with
  | read reference => rfl
  | literal symbol => rfl
  | map input mapper induction =>
      simp only [Traversal.normalize, Term.share, termGlobalLoadWords,
        traversalGlobalLoadWords, induction]
  | zip left right leftInduction rightInduction =>
      simp only [Traversal.normalize, termGlobalLoadWords,
        traversalGlobalLoadWords,
        leftInduction, rightInduction]

/-- Destination collision strategy selected by the CubeCL lowering.  Atomic
lowering is valid only when the concrete monoid has a proved atomic
implementation; the general path remains sort/reduce. -/
inductive DestinationStrategy where
  | sortReduce
  | atomic
deriving Repr, DecidableEq

/-- Semantic contract required from an abstract atomic target primitive.  A
backend-model witness identifies the binary action and proves that it is the
declared monoid combination; connecting it to emitted native instructions is
a later refinement boundary. -/
structure AtomicImplementation
    (reduction : signature.Monoid output) where
  apply : output.denote n denoteBase → output.denote n denoteBase →
    output.denote n denoteBase
  correct : apply = (signature.denoteMonoid reduction).combine
  /-- Number of scalar atomic instructions used for one logical combine under
  a chosen physical storage-width interpretation. -/
  operationCount : (Base → Nat) → Nat

namespace DestinationStrategy

/-- Evidence required by a lowering choice.  It lives in `Type`, rather than
being a Boolean/`Prop` capability flag, so atomic lowering must carry the
actual action and its refinement proof into the target instruction. -/
def Evidence
    (strategy : DestinationStrategy)
    (terminal : TraversalAlgebra.Observation.Terminal
      signature vertex edge output) : Type :=
  match strategy, terminal with
  | .atomic, .reduceByDestination _ reduction => AtomicImplementation reduction
  | _, _ => Unit

end DestinationStrategy

/-- Total graph-edge occurrences in the static topology. -/
def graphEdgeCount (graph : OrderedGraph n EdgePayload) : Nat :=
  activeEdgeCount graph (allVertices n)

private def materializeU32 (items : Nat) : Kernel :=
  Kernel.pointwise items 1 1 1 1 0 0 items 1

/-- Resource contract for the hierarchical `u32` scan used by traversal
control.  The fixed-work constant abstracts subgroup shuffle details while the
dependence depth remains logarithmic. -/
private def scanU32 (items : Nat) : Kernel where
  logicalThreads := items
  scalarWork := 2 * items
  span := 2 * reductionDepth items
  globalLoads := 2 * items
  globalStores := 2 * items
  barriers := reductionDepth items
  allocatedWords := 2 * items
  materializations := 1

/-- Symbolic contract for the currently materialized CSR control path in
`massively::graph`: topology/frontier canonicalization, degree scan, segment
control, and explicit source/destination/edge context arrays.  Primitive scan
internals are summarized by `scanU32`; this is a CubeCL-level resource contract,
not a wall-clock model. -/
def materializedCsrControlPlan
    (graph : OrderedGraph n EdgePayload) (frontier : Frontier n) : Plan :=
  let topologyEdges := graphEdgeCount graph
  let sources := frontier.length
  let edges := activeEdgeCount graph frontier
  [materializeU32 topologyEdges,
   materializeU32 (n + 1),
   materializeU32 sources,
   Kernel.pointwise sources 1 1 3 1 0 0 sources 1,
   scanU32 sources,
   { logicalThreads := if sources = 0 then 0 else 1
     scalarWork := if sources = 0 then 0 else 1
     span := 1
     globalLoads := if sources = 0 then 0 else 1
     globalStores := if sources = 0 then 0 else 1
     hostReadWords := if sources = 0 then 0 else 1
     allocatedWords := if sources = 0 then 0 else 1 },
   Kernel.pointwise (sources + 1) 1 1 1 1 0 0 (sources + 1) 1,
   Kernel.pointwise edges 1 1 0 1 0 0 edges 1,
   Kernel.pointwise sources 1 1 1 1,
   scanU32 edges,
   Kernel.pointwise edges 8 1 5 3 0 0 (3 * edges) 3]

/-- Exact symbolic work charged by the materialized traversal-control
certificate.  In particular, topology canonicalization contributes
`graphEdgeCount graph + (n + 1)` even when the active frontier is sparse. -/
theorem materializedCsrControl_scalarWork
    (machine : Machine)
    (graph : OrderedGraph n EdgePayload)
    (frontier : Frontier n) :
    ((materializedCsrControlPlan graph frontier).cost machine).scalarWork =
      graphEdgeCount graph + (n + 1) +
      frontier.length + frontier.length + 2 * frontier.length +
      (if frontier.length = 0 then 0 else 1) +
      (frontier.length + 1) +
      activeEdgeCount graph frontier + frontier.length +
      2 * activeEdgeCount graph frontier +
      8 * activeEdgeCount graph frontier := by
  simp [materializedCsrControlPlan, materializeU32, scanU32,
    Plan.cost, Cost.seq, Cost.zero, Kernel.cost, Kernel.pointwise] <;> omega

/-- The current materialized control certificate performs one host-visible
word read to discover the active stream length exactly when the frontier is
nonempty. -/
theorem materializedCsrControl_hostReadWords
    (machine : Machine)
    (graph : OrderedGraph n EdgePayload)
    (frontier : Frontier n) :
    ((materializedCsrControlPlan graph frontier).cost machine).hostReadWords =
      if frontier.length = 0 then 0 else 1 := by
  simp [materializedCsrControlPlan, materializeU32, scanU32,
    Plan.cost, Cost.seq, Cost.zero, Kernel.cost, Kernel.pointwise]

/-- Closed linear upper bound for materialized traversal control.  The static
topology term is explicit, so this theorem does not disguise the current
whole-CSR canonicalization cost as active-edge-only work. -/
theorem materializedCsrControl_linearWork
    (machine : Machine)
    (graph : OrderedGraph n EdgePayload)
    (frontier : Frontier n) :
    ((materializedCsrControlPlan graph frontier).cost machine).scalarWork ≤
      graphEdgeCount graph + n + 6 * frontier.length +
        11 * activeEdgeCount graph frontier + 3 := by
  rw [materializedCsrControl_scalarWork]
  split <;> omega

/-- Destination reduction performed with the action carried by an abstract
atomic instruction. -/
private def atomicDestinationAt
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (term : Term signature (MessageContext vertex edge) output)
    (reduction : signature.Monoid output)
    (implementation : AtomicImplementation reduction)
    (state : Store n (vertex.denote n denoteBase))
    (frontier : Frontier n)
    (destination : Fin n) : output.denote n denoteBase :=
  frontier.foldl
    (fun accumulator source =>
      (graph.outgoing source).foldl
        (fun accumulator edgeValue =>
          if edgeValue.destination = destination then
            implementation.apply accumulator
              (Term.evaluate
                (MessageContext.environment source edgeValue.destination
                  (state source) (state edgeValue.destination)
                    edgeValue.payload) term)
          else accumulator)
        accumulator)
    (signature.denoteMonoid reduction).identity

/-- The target atomic action has exactly the declared destination-reduction
semantics. -/
private theorem atomicDestinationAt_correct
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (term : Term signature (MessageContext vertex edge) output)
    (reduction : signature.Monoid output)
    (implementation : AtomicImplementation reduction)
    (state : Store n (vertex.denote n denoteBase))
    (frontier : Frontier n)
    (destination : Fin n) :
    atomicDestinationAt graph term reduction implementation state frontier destination =
      TraversalAlgebra.Observation.BSP.Terminal.destinationAt
        graph term reduction state frontier destination := by
  unfold atomicDestinationAt
  unfold TraversalAlgebra.Observation.BSP.Terminal.destinationAt
  rw [implementation.correct]

/-- Typed instructions in the abstract CubeCL target.  There is no free-form
`code + plan` constructor: each semantic instruction determines its own
resource plan below, preventing an unrelated cheap certificate from being
attached to arbitrary observer code. -/
inductive Program
    (signature : Signature n Base denoteBase)
    (vertex edge output : ValueType Base) : Type
  | emit (baseWords : Base → Nat) :
      Traversal signature vertex edge output → Program signature vertex edge output
  | source (baseWords : Base → Nat) :
      Traversal signature vertex edge output → signature.Monoid output →
        Program signature vertex edge output
  | destinationSort (baseWords : Base → Nat) :
      Traversal signature vertex edge output → signature.Monoid output →
        Program signature vertex edge output
  | destinationAtomic (baseWords : Base → Nat)
      (traversal : Traversal signature vertex edge output)
      (reduction : signature.Monoid output) : AtomicImplementation reduction →
        Program signature vertex edge output
  | materializedCsrControl : Program signature vertex edge output →
      Program signature vertex edge output

namespace Program

def execute
    (program : Program signature vertex edge output)
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (state : Store n (vertex.denote n denoteBase))
    (frontier : Frontier n) :
      TraversalAlgebra.Observation.Result n (output.denote n denoteBase) :=
  match program with
  | .emit _ traversal =>
      (TraversalAlgebra.Observation.BSP.Terminal.emit traversal.normalize).observe
        graph state frontier
  | .source _ traversal reduction =>
      (TraversalAlgebra.Observation.BSP.Terminal.reduceBySource
        traversal.normalize reduction).observe graph state frontier
  | .destinationSort _ traversal reduction =>
      (TraversalAlgebra.Observation.BSP.Terminal.reduceByDestination
        traversal.normalize reduction).observe graph state frontier
  | .destinationAtomic _ traversal reduction implementation =>
      .destinationReduced fun destination =>
        atomicDestinationAt graph traversal.normalize reduction implementation
          state frontier destination
  | .materializedCsrControl inner => inner.execute graph state frontier

/-- Executing an atomic target instruction refines to the normalized BSP
destination observer because its carried action equals the monoid action. -/
theorem execute_destinationAtomic
    (baseWords : Base → Nat)
    (traversal : Traversal signature vertex edge output)
    (reduction : signature.Monoid output)
    (implementation : AtomicImplementation reduction)
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (state : Store n (vertex.denote n denoteBase))
    (frontier : Frontier n) :
    (Program.destinationAtomic baseWords traversal reduction implementation).execute
        graph state frontier =
      (TraversalAlgebra.Observation.BSP.Terminal.reduceByDestination
        traversal.normalize reduction).observe graph state frontier := by
  apply congrArg TraversalAlgebra.Observation.Result.destinationReduced
  funext destination
  exact atomicDestinationAt_correct graph traversal.normalize reduction
    implementation state frontier destination

/-- Prefix a semantic terminal program with the materialized CSR control path.
Execution is unchanged; only its implementation certificate grows. -/
def withMaterializedCsrControl
    (program : Program signature vertex edge output) :
      Program signature vertex edge output :=
  .materializedCsrControl program

@[simp]
theorem execute_withMaterializedCsrControl
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (state : Store n (vertex.denote n denoteBase))
    (frontier : Frontier n)
    (program : Program signature vertex edge output) :
    program.withMaterializedCsrControl.execute
        graph state frontier = program.execute graph state frontier := rfl

end Program

/-- Logical output occurrences of each public terminal. -/
def outputCount
    (terminal : TraversalAlgebra.Observation.Terminal signature vertex edge output)
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (frontier : Frontier n) : Nat :=
  match terminal with
  | .emit _ => activeEdgeCount graph frontier
  | .reduceBySource _ _ => frontier.length
  | .reduceByDestination _ _ => n

/-- Fused pointwise emission kernel. -/
private def emitPlan
    (baseWords : Base → Nat)
    (traversal : Traversal signature vertex edge output)
    (edges : Nat) : Plan :=
  [Kernel.pointwise edges
    (traversal.normalize.work + ValueType.words baseWords output)
    (traversal.normalize.depth + 1)
    (termGlobalLoadWords baseWords traversal.normalize)
    (ValueType.words baseWords output)
    0 0 (edges * ValueType.words baseWords output) 0]

/-- Fused segmented source reduction.  One logical monoid action is charged
per active edge and the final source-indexed result is written once. -/
private def sourcePlan
    (baseWords : Base → Nat)
    (traversal : Traversal signature vertex edge output)
    (edges sources : Nat) : Plan :=
  [{ logicalThreads := sources + edges
     scalarWork := sources + edges * (traversal.normalize.work + 1)
     span := traversal.normalize.depth + reductionDepth edges
     globalLoads := edges * termGlobalLoadWords baseWords traversal.normalize
     globalStores := sources * ValueType.words baseWords output
     allocatedWords := sources * ValueType.words baseWords output }]

/-- Single-pass destination plan for monoids with a certified atomic
implementation.  Dense initialization and the active-edge push are separate
CubeCL launches. -/
private def atomicDestinationPlan
    (baseWords : Base → Nat)
    (traversal : Traversal signature vertex edge output)
    (vertices edges atomicOperationsPerEdge : Nat) : Plan :=
  [Kernel.pointwise vertices 1 1 0 (ValueType.words baseWords output)
      0 0 (vertices * ValueType.words baseWords output) 0,
   Kernel.pointwise edges (traversal.normalize.work + 1)
      (traversal.normalize.depth + 1)
      (termGlobalLoadWords baseWords traversal.normalize)
      0 atomicOperationsPerEdge]

/-- General commutative-monoid destination plan.  `rounds` merge rounds make
the quasilinear sorting cost explicit rather than silently treating arbitrary
monoids as atomically reducible. -/
private def sortDestinationPlan
    (baseWords : Base → Nat)
    (traversal : Traversal signature vertex edge output)
    (vertices edges : Nat) : Plan :=
  let words := ValueType.words baseWords output
  let rounds := reductionDepth edges
  [Kernel.pointwise vertices 1 1 0 words
      0 0 (vertices * words) 0,
   Kernel.pointwise edges traversal.normalize.work traversal.normalize.depth
      (termGlobalLoadWords baseWords traversal.normalize) words
      0 0 (edges * words) 1,
   { logicalThreads := edges
     scalarWork := edges * rounds
     span := rounds
     globalLoads := edges * rounds * (words + 1)
     globalStores := edges * rounds * (words + 1)
     allocatedWords := edges * (2 * words + 2)
     materializations := 1 },
   { logicalThreads := edges
     scalarWork := edges
     span := reductionDepth edges
     globalLoads := edges * (words + 1)
     globalStores := Nat.min vertices edges * words }]

namespace Program

/-- Every target instruction determines its launch plan from the graph and
frontier on which it executes. -/
def plan
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (frontier : Frontier n) : Program signature vertex edge output → Plan
  | .emit baseWords traversal =>
      emitPlan baseWords traversal (activeEdgeCount graph frontier)
  | .source baseWords traversal _ =>
      sourcePlan baseWords traversal (activeEdgeCount graph frontier) frontier.length
  | .destinationSort baseWords traversal _ =>
      sortDestinationPlan baseWords traversal n (activeEdgeCount graph frontier)
  | .destinationAtomic baseWords traversal _ implementation =>
      atomicDestinationPlan baseWords traversal n (activeEdgeCount graph frontier)
        (implementation.operationCount baseWords)
  | .materializedCsrControl inner =>
      materializedCsrControlPlan graph frontier ++ inner.plan graph frontier

def cost
    (machine : Machine)
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (frontier : Frontier n)
    (program : Program signature vertex edge output) : Cost :=
  (program.plan graph frontier).cost machine

/-- Exact decomposition of a materialized-control target into its control and
terminal resources. -/
theorem cost_withMaterializedCsrControl
    (machine : Machine)
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (frontier : Frontier n)
    (program : Program signature vertex edge output) :
    program.withMaterializedCsrControl.cost machine graph frontier =
      ((materializedCsrControlPlan graph frontier).cost machine).seq
        (program.cost machine graph frontier) := by
  simp [withMaterializedCsrControl, cost, plan, Plan.cost_append]

end Program

/-- Lower a public compositional TA terminal to one typed abstract CubeCL
instruction.  Atomic destination lowering consumes concrete implementation
evidence; all other branches require only `Unit`. -/
def lower
    (baseWords : Base → Nat)
    (strategy : DestinationStrategy)
    (terminal : TraversalAlgebra.Observation.Terminal signature vertex edge output)
    (evidence : strategy.Evidence terminal) :
      Program signature vertex edge output :=
  match strategy, terminal with
  | _, .emit traversal => .emit baseWords traversal
  | _, .reduceBySource traversal reduction => .source baseWords traversal reduction
  | .sortReduce, .reduceByDestination traversal reduction =>
      .destinationSort baseWords traversal reduction
  | .atomic, .reduceByDestination traversal reduction =>
      .destinationAtomic baseWords traversal reduction evidence

/-- General emission lowering; destination atomic capability is irrelevant. -/
def lowerEmit
    (baseWords : Base → Nat)
    (traversal : Traversal signature vertex edge output) :
    Program signature vertex edge output :=
  lower baseWords .sortReduce (.emit traversal) ()

/-- General source-reduction lowering; destination atomic capability is
irrelevant. -/
def lowerSource
    (baseWords : Base → Nat)
    (traversal : Traversal signature vertex edge output)
    (reduction : signature.Monoid output) :
    Program signature vertex edge output :=
  lower baseWords .sortReduce (.reduceBySource traversal reduction) ()

/-- General destination lowering that works for every declared lawful
commutative monoid. -/
def lowerDestinationSort
    (baseWords : Base → Nat)
    (traversal : Traversal signature vertex edge output)
    (reduction : signature.Monoid output) :
    Program signature vertex edge output :=
  lower baseWords .sortReduce (.reduceByDestination traversal reduction) ()

/-- Single-pass atomic destination lowering, available only with a
backend-model witness for the concrete monoid symbol. -/
def lowerDestinationAtomic
    (baseWords : Base → Nat)
    (traversal : Traversal signature vertex edge output)
    (reduction : signature.Monoid output)
    (implementation : AtomicImplementation reduction) :
    Program signature vertex edge output :=
  lower baseWords .atomic (.reduceByDestination traversal reduction) implementation

/-- CubeCL lowering preserves every public observation, independently of the
chosen legal destination implementation strategy. -/
theorem lower_correct
    (baseWords : Base → Nat)
    (strategy : DestinationStrategy)
    (terminal : TraversalAlgebra.Observation.Terminal signature vertex edge output)
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (state : Store n (vertex.denote n denoteBase))
    (frontier : Frontier n)
    (evidence : strategy.Evidence terminal) :
    (lower baseWords strategy terminal evidence).execute
        graph state frontier =
      terminal.observe graph state frontier := by
  cases strategy with
  | sortReduce =>
      cases terminal with
      | emit traversal =>
          exact (Terminal.toBSP_observe_correct (.emit traversal)
            graph state frontier).symm
      | reduceBySource traversal reduction =>
          exact (Terminal.toBSP_observe_correct (.reduceBySource traversal reduction)
            graph state frontier).symm
      | reduceByDestination traversal reduction =>
          exact (Terminal.toBSP_observe_correct (.reduceByDestination traversal reduction)
            graph state frontier).symm
  | atomic =>
      cases terminal with
      | emit traversal =>
          exact (Terminal.toBSP_observe_correct (.emit traversal)
            graph state frontier).symm
      | reduceBySource traversal reduction =>
          exact (Terminal.toBSP_observe_correct (.reduceBySource traversal reduction)
            graph state frontier).symm
      | reduceByDestination traversal reduction =>
          calc
            (lower baseWords .atomic
                (.reduceByDestination traversal reduction) evidence).execute
                graph state frontier =
              ((.reduceByDestination traversal.normalize reduction :
                TraversalAlgebra.Observation.BSP.Terminal
                  signature vertex edge output).observe graph state frontier) :=
                Program.execute_destinationAtomic baseWords traversal reduction
                  evidence graph state frontier
            _ = (.reduceByDestination traversal reduction :
                TraversalAlgebra.Observation.Terminal
                  signature vertex edge output).observe graph state frontier :=
                (Terminal.toBSP_observe_correct
                  (.reduceByDestination traversal reduction)
                    graph state frontier).symm

/-- Erasing a resource certificate leaves the normalized target instruction's
observation unchanged. -/
theorem certificate_erasure
    (baseWords : Base → Nat)
    (strategy : DestinationStrategy)
    (terminal : TraversalAlgebra.Observation.Terminal signature vertex edge output)
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (state : Store n (vertex.denote n denoteBase))
    (frontier : Frontier n)
    (evidence : strategy.Evidence terminal) :
    (lower baseWords strategy terminal evidence).execute
        graph state frontier =
      terminal.observe graph state frontier :=
  lower_correct baseWords strategy terminal graph state frontier evidence

/-- Specialized semantic equation for emission. -/
@[simp]
theorem lowerEmit_correct
    (baseWords : Base → Nat)
    (traversal : Traversal signature vertex edge output)
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (state : Store n (vertex.denote n denoteBase))
    (frontier : Frontier n) :
    (lowerEmit baseWords traversal).execute graph state frontier =
      (TraversalAlgebra.Observation.Terminal.emit traversal).observe
        graph state frontier := by
  exact lower_correct baseWords .sortReduce
    (.emit traversal) graph state frontier ()

/-- Specialized semantic equation for source reduction. -/
@[simp]
theorem lowerSource_correct
    (baseWords : Base → Nat)
    (traversal : Traversal signature vertex edge output)
    (reduction : signature.Monoid output)
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (state : Store n (vertex.denote n denoteBase))
    (frontier : Frontier n) :
    (lowerSource baseWords traversal reduction).execute
        graph state frontier =
      (TraversalAlgebra.Observation.Terminal.reduceBySource
        traversal reduction).observe graph state frontier := by
  exact lower_correct baseWords .sortReduce
    (.reduceBySource traversal reduction) graph state frontier ()

/-- Specialized semantic equation for the general sort/reduce destination
path. -/
@[simp]
theorem lowerDestinationSort_correct
    (baseWords : Base → Nat)
    (traversal : Traversal signature vertex edge output)
    (reduction : signature.Monoid output)
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (state : Store n (vertex.denote n denoteBase))
    (frontier : Frontier n) :
    (lowerDestinationSort baseWords traversal reduction).execute
        graph state frontier =
      (TraversalAlgebra.Observation.Terminal.reduceByDestination
        traversal reduction).observe graph state frontier := by
  exact lower_correct baseWords .sortReduce
    (.reduceByDestination traversal reduction) graph state frontier ()

/-- Specialized semantic equation for an atomic-capable destination monoid. -/
@[simp]
theorem lowerDestinationAtomic_correct
    (baseWords : Base → Nat)
    (traversal : Traversal signature vertex edge output)
    (reduction : signature.Monoid output)
    (implementation : AtomicImplementation reduction)
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (state : Store n (vertex.denote n denoteBase))
    (frontier : Frontier n) :
    (lowerDestinationAtomic baseWords traversal reduction implementation).execute
      graph state frontier =
      (TraversalAlgebra.Observation.Terminal.reduceByDestination
        traversal reduction).observe graph state frontier := by
  exact lower_correct baseWords .atomic
    (.reduceByDestination traversal reduction) graph state frontier implementation

private theorem lowerEmit_cost_eq
    (machine : Machine)
    (baseWords : Base → Nat)
    (traversal : Traversal signature vertex edge output)
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (frontier : Frontier n) :
    (lowerEmit baseWords traversal).cost machine graph frontier =
      (emitPlan baseWords traversal (activeEdgeCount graph frontier)).cost machine := rfl

/-- Exact fused emission work, including one scalar store action per output
word. -/
theorem lowerEmit_scalarWork
    (machine : Machine)
    (baseWords : Base → Nat)
    (traversal : Traversal signature vertex edge output)
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (frontier : Frontier n) :
    ((lowerEmit baseWords traversal).cost machine graph frontier).scalarWork =
      activeEdgeCount graph frontier *
        (traversal.work + ValueType.words baseWords output) := by
  rw [lowerEmit_cost_eq]
  simp [emitPlan, Plan.cost, Cost.seq, Cost.zero, Kernel.cost,
    Kernel.pointwise, Traversal.normalize_work]

/-- Exact global-load traffic of fused emission. -/
theorem lowerEmit_globalLoads
    (machine : Machine)
    (baseWords : Base → Nat)
    (traversal : Traversal signature vertex edge output)
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (frontier : Frontier n) :
    ((lowerEmit baseWords traversal).cost machine graph frontier).globalLoads =
      activeEdgeCount graph frontier *
        traversalGlobalLoadWords baseWords traversal := by
  rw [lowerEmit_cost_eq]
  simp [emitPlan, Plan.cost, Cost.seq, Cost.zero, Kernel.cost,
    Kernel.pointwise, normalize_globalLoadWords]

/-- Exact output traffic of fused emission, compositional in recursive product
width. -/
theorem lowerEmit_globalStores
    (machine : Machine)
    (baseWords : Base → Nat)
    (traversal : Traversal signature vertex edge output)
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (frontier : Frontier n) :
    ((lowerEmit baseWords traversal).cost machine graph frontier).globalStores =
      activeEdgeCount graph frontier * ValueType.words baseWords output := by
  rw [lowerEmit_cost_eq]
  simp [emitPlan, Plan.cost, Cost.seq, Cost.zero, Kernel.cost,
    Kernel.pointwise]

/-- Fused emission allocates its observable result but no full-stream
intermediate expression column. -/
theorem lowerEmit_no_materialization
    (machine : Machine)
    (baseWords : Base → Nat)
    (traversal : Traversal signature vertex edge output)
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (frontier : Frontier n) :
    ((lowerEmit baseWords traversal).cost machine graph frontier).materializations = 0 := by
  rw [lowerEmit_cost_eq]
  simp [emitPlan, Plan.cost, Cost.seq, Cost.zero, Kernel.cost,
    Kernel.pointwise]

/-- Source reduction initializes one result per frontier occurrence, then
performs exactly one normalized expression evaluation and one monoid action
per active edge. -/
theorem lowerSource_scalarWork
    (machine : Machine)
    (baseWords : Base → Nat)
    (traversal : Traversal signature vertex edge output)
    (reduction : signature.Monoid output)
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (frontier : Frontier n) :
    ((lowerSource baseWords traversal reduction).cost machine graph frontier).scalarWork =
      frontier.length +
        activeEdgeCount graph frontier * (traversal.work + 1) := by
  simp [lowerSource, lower, Program.cost, Program.plan, sourcePlan, Plan.cost,
    Cost.seq, Cost.zero, Kernel.cost, Traversal.normalize_work]

/-- Atomic destination reduction is work-linear in vertices plus active edge
occurrences.  The capability witness prevents this theorem from being applied
to an arbitrary non-atomic monoid implementation. -/
theorem lowerDestinationAtomic_scalarWork
    (machine : Machine)
    (baseWords : Base → Nat)
    (traversal : Traversal signature vertex edge output)
    (reduction : signature.Monoid output)
    (implementation : AtomicImplementation reduction)
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (frontier : Frontier n) :
    ((lowerDestinationAtomic baseWords traversal reduction implementation).cost
      machine graph frontier).scalarWork =
      n + activeEdgeCount graph frontier * (traversal.work + 1) := by
  simp [lowerDestinationAtomic, lower, Program.cost, Program.plan,
    atomicDestinationPlan, Plan.cost, Cost.seq, Cost.zero, Kernel.cost,
    Kernel.pointwise, Traversal.normalize_work]

/-- Atomic operation count is exact for the carried backend-model witness.  A
multiword or packed backend states its per-combine count in that witness rather
than inheriting an unjustified one-atomic-per-word assumption. -/
theorem lowerDestinationAtomic_operations
    (machine : Machine)
    (baseWords : Base → Nat)
    (traversal : Traversal signature vertex edge output)
    (reduction : signature.Monoid output)
    (implementation : AtomicImplementation reduction)
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (frontier : Frontier n) :
    ((lowerDestinationAtomic baseWords traversal reduction implementation).cost
      machine graph frontier).atomicOperations =
      activeEdgeCount graph frontier * implementation.operationCount baseWords := by
  simp [lowerDestinationAtomic, lower, Program.cost, Program.plan,
    atomicDestinationPlan, Plan.cost, Cost.seq, Cost.zero, Kernel.cost,
    Kernel.pointwise]

/-- The general destination path exposes dense initialization and sorting
rounds explicitly: work is `vertices`, pointwise expression work,
`edges * rounds`, and one final reduction action per active edge. -/
theorem lowerDestinationSort_scalarWork
    (machine : Machine)
    (baseWords : Base → Nat)
    (traversal : Traversal signature vertex edge output)
    (reduction : signature.Monoid output)
    (graph : OrderedGraph n (edge.denote n denoteBase))
    (frontier : Frontier n) :
    ((lowerDestinationSort baseWords traversal reduction).cost
        machine graph frontier).scalarWork =
      n + activeEdgeCount graph frontier * traversal.work +
        activeEdgeCount graph frontier *
          reductionDepth (activeEdgeCount graph frontier) +
        activeEdgeCount graph frontier := by
  simp [lowerDestinationSort, lower, Program.cost, Program.plan,
    sortDestinationPlan, Plan.cost, Cost.seq, Cost.zero, Kernel.cost,
    Kernel.pointwise, Traversal.normalize_work, Nat.add_assoc]

end TraversalAlgebra.Verified.Typed.CubeCL
