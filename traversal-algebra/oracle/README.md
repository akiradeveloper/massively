# Traversal Algebra implementation oracle

This crate contains an independent sequential CPU reference for the public
Massively Traversal Algebra operations. Lean is not linked, compiled, or
executed by this crate.

For each property case, `proptest` generates a valid CSR graph, frontier,
vertex columns, and edge columns. The CPU oracle computes the expected result
from its own host arrays, Massively computes the actual result on the GPU, and
the test compares them.

## Reference semantics

The oracle deliberately uses direct loops rather than Massively primitives:

- traversal visits frontier occurrences in order and each selected CSR row in
  storage order;
- parallel edges, self-loops, and duplicate frontier entries are preserved;
- source, destination, edge, and structural-ID expressions are evaluated from
  raw host columns;
- emission appends one value per active edge;
- source reduction emits one fold per frontier occurrence;
- destination reduction uses a dense vertex accumulator;
- recursive products use one expression-tree evaluator, without
  terminal-by-arity implementations.

The current scalar reference fragment uses natural-number addition represented
as `u32`. An intermediate or terminal result that does not fit `u32` is rejected
instead of silently wrapping.

This implementation is intentionally independent, not formally verified.
Lean proves properties of the mathematical TA model; these property tests ask
the separate engineering question of whether the CPU and GPU implementations
agree on generated inputs.

## Coverage

The default campaign includes:

- 256 mixed shrinkable and structured cases for edge contexts and both
  reduction terminals;
- 32 cases for source, destination, and edge pulls, recursive products,
  pointwise maps, emission, and both reductions;
- a deterministic 65,537-vertex, 393,222-edge scale graph.

Structured families include directed multigraphs, skewed hubs with isolates,
bipartite graphs, boundary-crossing regular rows, parallel edges, self-loops,
sorted and unsorted rows, and duplicate-frontier policies.

Run the comparison from the repository root:

```text
just ta::oracle
```

Increase the randomized campaign or change the scale case when desired:

```text
TRAVERSAL_ALGEBRA_PROPTEST_CASES=4096 just ta::oracle
TRAVERSAL_ALGEBRA_SEMANTIC_CASES=256 just ta::oracle
TRAVERSAL_ALGEBRA_SCALE_VERTICES=262145 just ta::oracle
```

No finite randomized campaign proves universal correspondence. Its role is to
find small reproducible disagreements while the separate scale case exercises
large inputs.
