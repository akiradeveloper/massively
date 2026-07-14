# Formal proof status

This is the detailed theorem and assumption ledger for formal reviewers. For a
plain-language account of what Traversal Algebra is, what has been proved, and
what has not been proved, start with the [project overview](../README.md).

This ledger separates machine-checked language results from executable
implementation validation. Declaration names are the stable interface between
the paper and the Lean development.

## Proved without project-specific axioms or admissions

### Graph and semantic machines

- `TraversalAlgebra.Verified.OrderedGraph` is a finite ordered directed
  multigraph with valid vertex references (`Fin n`). Parallel edges,
  self-loops, arbitrary payloads, and repeated frontier entries are allowed.
- `OrderedGraph.nestedDestinations` constructs destination candidates by the
  BSP source machine's nested frontier/adjacency scan.
  `traversalDestinations_eq_nestedDestinations` proves exact list equality
  with destinations projected from flattened TA traversal, including order and
  duplicate occurrences.
- `Verified.MonoidalFrontierBSP.Program` and `Verified.PullMapPush.Plan` are
  independent semantic source and target machines. Their frontier selectors
  receive the ordered candidate stream, old store, and new store.
- `fold_expand_compile`, `fold_traverse_compile`, and
  `inbox_compile_correct` prove that flattening preserves every destination
  inbox. `compile_step_correct` and `compile_run_correct` lift this to the
  complete store and ordered frontier for one step and every finite run.
- `foldContextAt_perm`, `inboxAt_schedule_independent`, and
  `inbox_schedule_compile_correct` prove invariance under every linear
  permutation of the logical edge stream, using the declared commutative
  monoid laws.
- `Verified.TraversalAlgebra.emit`, `reduceBySource`, and
  `reduceByDestination` define the three type-safe public TA terminals.

### Typed syntax and compiler

- `Typed.ValueType`, `Signature`, `Variable`, `Environment`, and `Term` define
  a signature-relative, intrinsically typed, first-order scalar IR. Recursive
  products represent arbitrary multi-column shapes. Program syntax contains
  symbols and typed terms, not per-program semantic callbacks.
- `Term.let1` is an explicit sharing constructor. Its input is evaluated once
  and its body reads the bound value from a singleton typed environment.
  `Term.share` constructs this node and `Term.evaluate_share` proves that its
  denotation is ordinary composition without syntax duplication.
- `Typed.MonoidalFrontierBSP.Program` and `Typed.PullMapPush.Program` are
  distinct source and target syntaxes. `Typed.compile_denoteAt` proves that
  syntax-directed compilation commutes with denotation.
  `Typed.compile_step_correct`, `compile_run_correct`, and
  `inbox_schedule_correct` prove typed one-step, every-finite-run, and
  schedule-permutation correctness.
- `FrontierPolicy.dense` scans all valid vertices in canonical order.
  `FrontierPolicy.sparsePreserve` stably filters the destination candidate
  stream.

### Independent TA expressions and sharing-aware normalization

- `Typed.TraversalAlgebra.Expression.Traversal` is an independently
  compositional TA grammar. Structural reads select only the five fields of
  `MessageContext`; literals, arbitrarily nested unary `map` stages, and
  pointwise `zip` trees are explicit. The grammar contains neither an
  arbitrary semantic callback nor a pre-normalized edge term.
- `Expression.Destination`, `VertexStage`, and `Program` form
  `compact(updateByDestination(reduceByDestination(traversal)))`. Their direct
  `step` and `run` use the type-safe TA terminal and do not call BSP or the
  normalizer.
- `Traversal.normalize` turns every traversal tree into one typed edge term.
  A map becomes `Term.share mapper (normalize input)`, so repeated reads by
  the mapper do not duplicate the normalized input. `evaluateAt_normalize`
  and `evaluate_normalize` prove pointwise and complete ordered-stream
  preservation.
- `Destination.inboxAt_normalize`, `Program.normalize_step_correct`, and
  `Program.normalize_run_correct` prove preservation of every destination
  inbox, the complete one-step configuration, and every finite execution.
- `Traversal.ofTerm` reifies variables as reads, pairs as zip nodes, and
  primitive applications or `let1` bindings as map nodes.
  `evaluateAt_ofTerm` and `evaluate_normalize_ofTerm` prove semantic
  reification. `Program.ofNormalForm_step_correct`,
  `ofNormalForm_run_correct`, and `reify_normalize_run_correct` lift this to
  programs. No false syntactic right-inverse claim is made.

### Monoidal Frontier BSP equivalence

- `Typed.TraversalAlgebra.DestinationPush` and `Program` are the sharing-aware
  single-push TA normal form. `encode` and `decode` translate between this
  normal form and typed Monoidal Frontier BSP.
- `decode_encode` and `encode_decode` prove a syntactic bijection for normal
  forms. `encode_inbox_correct`, `encode_step_correct`,
  `decode_step_correct`, `encode_run_correct`, and `decode_run_correct` prove
  the corresponding semantic equations.
- `Expression.Program.toBSP` normalizes and decodes any closed TA expression;
  `ofBSP` encodes and reifies a BSP program. `toBSP_run_correct` and
  `ofBSP_run_correct` prove both semantic directions for every finite run.
  `toBSP_ofBSP_run_correct` records the semantic BSP round trip after
  reification and sharing-aware renormalization.
- `ta_to_bsp_complete` and `bsp_to_ta_complete` state both directions as
  existential representability. `toBSP_terminates_iff` and
  `ofBSP_terminates_iff` preserve reaching an empty frontier.

### Quantitative normalization and sparse frontier

- `TypedCost` defines syntax-node count, unit scalar work, dependence depth,
  and a conservative per-edge scalar-temporary bound for terms and traversal
  expressions. Program work also charges active-edge monoid actions, dense
  vertex updates, and dense or sparse frontier predicates.
- `Traversal.normalize_nodeCount`, `normalize_work`, `normalize_depth`, and
  `normalize_scalarTemporaries` are exact equalities. Their program-level
  counterparts prove factor-one syntax growth and exact preservation of work,
  depth, and local scalar storage for every graph and configuration.
- `normalize_fullStreamTemporaryValues_le` and
  `normalize_materializations_le` prove that the fused normal form does not
  increase full-stream temporary volume or pointwise materialization passes
  relative to the defined unfused reference schedule.
- `sparsePreserve_sublist` proves stable-filter order. The selected, rejected,
  and combined `sparsePreserve_count*` theorems specify exact multiplicity.
  `work_sparsePreserve` gives work equal to candidate count times predicate
  work; `sparsePreserve_work_le_dense` proves the dense comparison under the
  explicit assumption that candidate count is at most vertex count.

### Explicit terminal observations

- `Typed.TraversalAlgebra.Observation.Result` distinguishes emitted lists,
  source-reduced lists, and destination-reduced stores.
  `Observation.Terminal.observe` gives the direct compositional TA semantics.
- `Observation.BSP.Terminal.observe` is independent: emission and reductions
  use nested frontier/adjacency recursion rather than the flattened TA
  terminal in their definitions.
- `Observation.Proof.map_traverse`, `source_correct`, and
  `destination_correct` prove the ordered nested/flattened equations.
  `Terminal.toBSP_observe_correct` and
  `BSP.Terminal.toTA_observe_correct` prove translations in both directions,
  preserving emission order, frontier/source occurrence multiplicity, and the
  complete destination store.
- `Observation.Cost.toBSP_work` proves exact TA-to-BSP work preservation under
  the same unit-cost model for all three terminals.

### Orthogonal signature transport

- `Typed.Signature.Lowering` maps literal, primitive, and monoid symbols while
  carrying exact denotational equations. `Lowering.identity` and
  `Lowering.trans` provide identity and composition.
- `Typed.Term.evaluate_lower` covers every term constructor, including
  `let1`. Dense and sparse policies and complete source/target programs retain
  denotation, one-step, and finite-run semantics under lowering.
- `Typed.compile_lower` proves exact compilation/lowering naturality;
  `Typed.lower_compile_run_correct` closes the finite-run square.
- `Typed.Example.pathProgram` and its non-identity alternate lowering witness
  that the typed fragment and transport discipline are inhabited.

All semantic theorems quantify universally over the relevant graph size,
types, graph, program, initial configuration, frontier, and step count. The
typed results additionally quantify over an arbitrary fixed base-type family,
signature, and well-typed syntax. They are proofs, not bounded tests.

## Assumptions and exact scope

- Graphs are finite, ordered, static during a run, and represented
  extensionally as adjacency rows.
- One message is generated per traversed edge occurrence. Destination
  collisions use a declared lawful commutative monoid.
- The broad semantic envelope permits arbitrary total message, update, and
  selection functions. Representability of every such callback by the typed
  syntax is intentionally not claimed.
- Vertex update is dense. Frontier selection is either a canonical dense scan
  or an exact stable filter of traversal destination candidates. Sparse
  candidates retain duplicates; sparse work is therefore not unconditionally
  bounded by dense work.
- Runs are indexed by a finite natural number. Reaching an empty frontier is
  preserved; convergence to a domain-specific fixed point is not inferred.
- The barrier equivalence covers structural pulls, literals, arbitrary finite
  typed `map`/`zip` trees, destination reduction, dense vertex update, and
  dense or sparse frontier selection. Emission and source reduction are
  covered by their explicit observation theorem rather than misrepresented as
  barrier-state transitions.
- The cost theorems concern the stated language-level unit model. Primitive
  symbols currently have unit invocation cost; `reductionDepth` is an abstract
  balanced-tree bound. These theorems do not predict a particular GPU kernel,
  memory allocator, or transfer schedule.
- Both languages use the same arbitrary but fixed signature. Signature
  transport and execution-platform lowering are not premises of equivalence.

## Not yet proved

- refinement from arbitrary parallel reduction-tree shapes to the
  permutation-invariant sequential denotation;
- signature- or backend-weighted primitive, transfer, allocation, residency,
  and concrete storage costs;
- correspondence between the verified extensional graph and a concrete CSR
  representation;
- coverage of the full intended algorithm surface, such as the Gunrock suite;
- universal correspondence with Rust, CubeCL, a device compiler, or hardware.

Consequently, `Expression.Program.toBSP_run_correct` and
`ofBSP_run_correct` establish bidirectional, signature-relative semantic
equivalence for the independently compositional closed barrier grammar.
Sharing-aware normalization additionally has factor-one syntax growth and
exact unit work/depth/local-storage preservation. Sparse frontier behavior and
all three terminal observation shapes now have separate universal theorems;
they are no longer entries in the open-proof list.

## Executable implementation validation

The native Lean oracle plus Rust `proptest` compare generated valid
CSR/frontier inputs with the public Massively GPU operations. This is strong
artifact-level evidence and produces shrinkable counterexamples, but remains
logically separate from the universal language theorems. Universal
Rust/CubeCL correspondence is deliberately not inferred from finite tests.
