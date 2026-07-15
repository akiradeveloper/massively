# Traversal Algebra implementation oracle

The Lean theorems prove statements about a mathematical model. This crate asks
the separate engineering question: does the current Rust/GPU implementation
behave like that model on generated inputs?

For each property case, `proptest` generates a valid graph, frontier, vertex
columns, and edge columns. Compiled Lean computes the expected result,
Massively computes the actual result, and the test compares them. Failures in
the small structural generator are reduced to smaller examples automatically;
seeded graph-family cases record all generation parameters for reproduction.

This crate deliberately contains no handwritten reference traversal. Lean
provides both oracle paths from the semantics in `../proof/`:

- `src/generated.rs` contains five named regression fixtures evaluated by Lean.
- the Docker-built `proof/.lake/build/bin/oracle` is a persistent, versioned
  protocol server. A graph is uploaded once and reused by typed terminal
  queries.

`src/graph.rs` is an independent host-owned semantic twin of
`massively::graph`: it exposes `Csr -> traverse -> map -> terminal` without
using Massively implementation types. Scalar leaves and recursive product
expressions become typed requests, and `LeanOracle` reconstructs product rows
without arity-specific cases.

The protocol expression is compiled in Lean to the intrinsically typed TA
syntax. `evaluate_correct` connects it directly to
`Observation.Terminal.observe`. Checked CSR inputs retain exact offset and
destination round-trip proofs; `toOrderedGraph_destinations` and
`toOrderedGraph_edgeIds` connect their rows to the verified graph. Dense
destination reduction uses a one-pass array lowering, with
`evaluateDestinationCsr_correct` proving equality to the public typed
terminal.

`LeanOracle::cubecl_certificate` exposes the separate proof-backed abstract
CubeCL resource model for scalar queries. A certificate contains workgroup and
subgroup padding, scalar work, span, global traffic, host-visible reads,
atomics, barriers, launches, allocation volume, and materialization counts for
the fused terminal target, the current materialized CSR-control prefix, and
their sequential composition. General destination monoids use sort/reduce;
the abstract atomic target carries a Lean witness that its natural-add action
equals the declared monoid action and states one modeled scalar atomic per
edge.

The integration test checks:

- the five committed regressions against the typed Lean evaluator;
- 256 mixed shrinkable and structured graph cases for ordered edge contexts
  and source/destination reductions;
- 32 additional cases for source, destination, and edge pulls, recursive zip,
  pointwise map, emission, and both reduction terminals;
- 32 generated cases checking CubeCL certificate dimensions, varied machine
  geometry, exact emission/source/destination formulas, atomic counts,
  materialized-control work, and full fieldwise cost composition;
- a deterministic 1,025-vertex scale graph, crossing the former 33-vertex and
  8-edges-per-row limits.

Structured families include directed multigraphs, skewed hubs with isolates,
bipartite graphs, boundary-crossing regular rows, parallel edges, self-loops,
sorted and unsorted rows, and several duplicate-frontier policies.

Regenerate and check the source from the repository root:

```text
just ta::check-generated
```

Run the comparison on the host GPU stack:

```text
just ta::oracle
```

Increase the randomized campaign when desired:

```text
TRAVERSAL_ALGEBRA_PROPTEST_CASES=4096 just ta::oracle
```

The semantic and scale layers are independently configurable:

```text
TRAVERSAL_ALGEBRA_SEMANTIC_CASES=256 just ta::oracle
TRAVERSAL_ALGEBRA_SCALE_VERTICES=4097 just ta::oracle
```

No finite randomized campaign proves universal correspondence. Its role is to
connect the formal executable semantics to the Rust/GPU implementation and to
find small counterexamples; universal claims remain Lean theorem obligations.
See the [Traversal Algebra overview](../README.md) for the plain-language proof
scope and the distinction between proof and implementation evidence.

The CubeCL counters are symbolic and backend-neutral. They do not predict
milliseconds, cache behavior, occupancy, or contention latency, and the
current artifact does not yet prove refinement from emitted CubeCL IR to the
certificate.
