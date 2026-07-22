# What is proved about Traversal Algebra

Traversal Algebra is the programming model behind `massively::graph`. A graph
program describes a data flow instead of prescribing a particular GPU kernel:

```text
frontier -> traverse edges -> pull values -> map -> reduce -> update -> next frontier
```

That separation is useful only if changing the execution plan does not change
the program's meaning or silently introduce an algorithmic cost explosion.
The Lean development makes those obligations precise and proves them for a
typed, finite frontier-computation fragment.

## Model covered by the proof

Graphs are finite, ordered, directed multigraphs. Parallel edges, self-loops,
edge payloads, and repeated vertices in a frontier are all represented rather
than ruled out as special cases. The Core TA grammar can read endpoint state
and edge payloads, combine arbitrary recursively nested product values, and
compose finite `map` and `zip` trees.

A separate public-API grammar matches the exported Rust shape exactly at its
denotational boundary: independent `source`, `destination`, and `edge` columns;
source, destination, and CSR-edge identifiers; recursive `zip`; one pointwise
`map`; and the public traversal terminals. Reduction laws are explicit premises
because Rust documents, but cannot type-check, associativity, commutativity,
and identity.

The model has three public result shapes:

- ordered emission, with one result per traversed edge occurrence;
- reduction by source, preserving each frontier occurrence;
- reduction by destination into a complete vertex store.

Destination collisions use an associative, commutative operation with an
identity. Vertex update is dense. The next frontier is either a canonical
dense scan or a stable filter of the ordered destination-candidate stream.

The proof compares independently defined semantics:

- a directly compositional Traversal Algebra expression grammar;
- a normalized single-push form used as a compilation target;
- Monoidal Frontier BSP, which evaluates nested frontier and adjacency rows
  instead of reusing Traversal Algebra's flattened evaluator;
- the Rust-shaped public traversal API and a host schedule composed from
  public graph and vector operation contracts.

This independence matters: equivalence is established by theorems, not by
giving the source and reference languages the same implementation.

## Main results

| Proved result | Engineering meaning |
| --- | --- |
| Finite-run semantic preservation | Normalizing or translating a program preserves the complete vertex store and ordered frontier after every finite number of steps. |
| Bidirectional expressiveness | Every closed program in the proved Traversal Algebra fragment has an equivalent typed Monoidal Frontier BSP program, and every program in that BSP fragment has an equivalent Traversal Algebra program. |
| Schedule-independent destination reduction | Every permutation of the logical edge-message schedule produces the same destination inbox when the declared reduction laws hold. |
| Public terminal preservation | Translation preserves emission order, repeated source/frontier occurrences, and the complete destination-reduced store. |
| Rust-shaped API correspondence | The six public edge-expression leaves, recursive zip, one map, and the three general terminals have independent denotations equal to the type-safe TA terminals. Stateful destination update and minimum relaxation have explicit public contracts. |
| Public API compilation | Every typed Monoidal Frontier BSP program compiles through Core TA to a finite host schedule of public graph destination reduction, destination-ID emission, dense vector map, and stable vector filtering, preserving the complete store and ordered frontier. |
| Sharing-aware normalization | Normalization preserves scalar work, dependency depth, and local temporary bounds exactly, with factor-one syntax growth. It does not increase full-stream temporary volume or materialization passes relative to the defined unfused schedule. |
| Exact sparse-frontier semantics | Stable filtering preserves candidate order and duplicate multiplicity. Its work is characterized exactly, including the condition under which it is no more expensive than a dense scan. |
| Meaning-preserving resource plans | Backend-neutral CubeCL plans have the same Traversal Algebra observation as the source terminal while exposing symbolic work, span, traffic, synchronization, launch, allocation, and materialization counts. |
| Signature-safe lowering | Replacing typed literals, primitives, and monoids with meaning-preserving lower-level counterparts commutes with compilation and preserves finite runs. |

“Bidirectional expressiveness” is intentionally scoped to the two formal
fragments above. It is not a claim of Turing completeness or of covering every
possible graph-processing language.

Reaching an empty frontier is preserved in both translation directions. The
proof does not infer that an arbitrary graph algorithm converges, nor does it
identify an empty frontier with a domain-specific fixed point unless the
program itself gives it that meaning.

## What the normalization result rules out

A naïve fusion pass can duplicate an expensive input whenever a mapper reads
it more than once. Traversal Algebra's normalizer introduces an explicit typed
sharing node, so the input is evaluated once and referenced from the mapper.
Lean proves that this transformation preserves the complete ordered edge
stream and then lifts that equality through destination reduction, update, and
frontier selection.

The quantitative theorem is stronger than “the output is the same.” Under the
defined unit-cost model, normalized and direct programs have equal scalar work,
dependence depth, and per-edge scalar storage. The normalized form also has no
more full-stream temporary values or pointwise materializations than the
defined unfused reference schedule. Recursive products make these statements
apply uniformly to multi-column values rather than to a fixed tuple arity.

## What the resource result says

The abstract CubeCL layer ties each executable target instruction to its own
resource plan. It is therefore impossible inside the model to attach an
unrelated cheap certificate to arbitrary observer code. Its fused terminal
instructions are optimization targets, not a claim that the current Rust
terminal path already emits those fused kernels.

For emission and source reduction, the proof gives exact symbolic scalar work
and global traffic formulas. Destination reduction has two explicit paths:

- an atomic path is available only with a witness that the backend atomic
  action implements the declared monoid operation, and then has one modeled
  atomic action per active edge;
- the general path exposes its sorting and reduction-depth term instead of
  assuming atomic support.

The separately modeled current materialized-CSR control contract includes
topology-wide work. Its proved unit-work bound is
`topologyEdges + vertices + 6*frontier + 11*activeEdges + 3`. This prevents a
sparse traversal from being advertised as active-edge-only while its control
path still scans or canonicalizes larger structures.

These formulas are symbolic operation and traffic counts. They do not assign
latencies to operations, model caches or coalescing, predict atomic contention,
or claim that a plan is optimal on a particular GPU.

## What a software engineer can rely on

Within the formal grammar and assumptions, the proof supports the following
design decisions:

- traversal expressions may be composed and normalized without changing
  their ordered meaning;
- a backend may reorder destination messages only when the reduction really
  satisfies the declared associative, commutative, and identity laws;
- dense and sparse frontier policies have explicit, different ordering and
  cost contracts;
- emission, source reduction, and destination reduction cannot be conflated
  merely because they consume the same traversal;
- closed Core TA supersteps need no imaginary Rust graph-program object: they
  can be expressed by composing the current graph terminals with public dense
  vector map and stable filtering contracts;
- lowering typed primitives or choosing atomic versus sort/reduce execution
  requires explicit semantic evidence rather than an unchecked optimization;
- cost discussions must include topology, frontier, and materialization terms
  represented by the selected plan.

## Boundary of the proof

The result does not prove that:

- the Rust implementation, emitted CubeCL IR, compiler, driver, or hardware
  refines the Lean model;
- a particular application such as PageRank or connected components computes
  its textbook domain result;
- arbitrary user callbacks are representable by the closed typed grammar;
- finite Lean identifiers and total denotational callbacks refine concrete
  `u32` indices, buffers, and generated CubeCL callbacks;
- every parallel reduction-tree implementation refines the sequential monoid
  denotation;
- the symbolic resource certificate can be recovered from generated kernels;
- finite-step equivalence implies convergence or real-world performance.

The [`proof/`](proof/) directory contains the universal mathematical results.
The independent [`oracle/`](oracle/) compares generated graph cases with the
public Massively GPU operations using a sequential CPU interpretation that
does not share traversal or reduction code. The
[`graph-algorithms/`](graph-algorithms/) crate adds CPU comparisons and
device-resident benchmarks for 19 complete algorithms. These are valuable
implementation evidence, but neither is a premise of the Lean proof or a
formal verification of Rust and CubeCL.

The exact theorem names, assumptions, and open refinement boundaries are
recorded in [`proof/STATUS.md`](proof/STATUS.md).
