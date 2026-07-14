# Traversal Algebra

Traversal Algebra is Massively's new way to describe graph algorithms. Instead
of writing one GPU procedure for each algorithm, it describes the common flow
of frontier-based graph computation:

```text
frontier -> traverse edges -> pull values -> map -> push/reduce -> update state
```

An algorithm states what each stage means. Decisions such as how to expand the
frontier or resolve concurrent updates can remain implementation choices. This
separation is intended to make graph programs easier to compose, reason about,
and optimize as a family.

This directory provides mathematical evidence for that design and tests its
connection to the Rust/GPU implementation.

The latest prebuilt paper is available as [`paper.pdf`](paper.pdf). Its title
page records the UTC build time as the revision, so it can be read directly
without installing LaTeX or running Docker.

## The short version

- Traversal Algebra has a precise, typed mathematical definition.
- Within a clearly stated frontier-based fragment, Lean proves that Traversal
  Algebra and a reference model called Monoidal Frontier BSP express the same
  computations.
- Lean proves that converting compositional programs into their normalized
  form preserves results and does not introduce an algorithmic cost blow-up in
  the defined abstract cost model.
- A Lean-compiled oracle and Rust `proptest` compare the mathematical model
  with `massively::graph` over generated graphs on the host GPU stack.

The proof is universal inside its mathematical model. The implementation
comparison is deliberately separate: it is strong automated test evidence,
not a formal proof of Rust, CubeCL, a compiler, or GPU hardware.

## What has been proved?

The following is the plain-language interpretation of the formal results:

| Result | What it means |
| --- | --- |
| Meaning is preserved | Direct Traversal Algebra evaluation and its normalized or translated form produce the same state and frontier after any finite number of steps. |
| The chosen fragment is expressively complete | Every program in the closed Traversal Algebra fragment has an equivalent typed Monoidal Frontier BSP program, and every program in that BSP model has an equivalent Traversal Algebra program. |
| Legal scheduling does not change reductions | Reordering traversed edge messages does not change destination results when collisions use an associative, commutative reduction with an identity. |
| Normalization has no hidden algorithmic blow-up | Under the stated language-level model, normalization preserves abstract work, dependency depth, and local temporary bounds, and does not add full-stream materialization compared with the defined reference schedule. |
| The public result shapes agree | Ordered emission, reduction by source, and reduction by destination are preserved by the translations. |
| Sparse-frontier behavior is specified | Candidate order and duplicate occurrences are preserved, and sparse work has an exact bound with an explicit condition for comparison with dense work. |

Here, “complete” means equivalent in expressive power to the defined Monoidal
Frontier BSP fragment. It does not mean Turing complete, and it does not claim
to cover every imaginable graph language.

The cost result also has a precise boundary. It shows that the algebra's
normalization does not add overhead in its abstract unit-cost model. It does
not prove that every generated GPU kernel is fast, that the cost is globally
minimal, or that transfers and allocations are free.

## What this says about graph algorithms

The proof establishes the soundness and expressive scope of the programming
method. In practical terms, a graph algorithm can be written as compositions
of traversal, value selection, mapping, reduction, update, and frontier
selection without changing meaning when the proved transformations are
applied.

It does not, by itself, prove a domain statement such as “this particular
program computes textbook PageRank.” The separate
[`graph-algorithms/`](graph-algorithms/) artifact implements every application
that Gunrock's published table marks as supported by current Gunrock or as a
priority future port (16 of 16 in that scope). All 19 algorithms in the crate
are checked against independent CPU implementations on generated graphs and
have device-resident benchmarks. This is practical coverage evidence for
developers; it is not an additional premise of the mathematical proof or, by
itself, a performance comparison with Gunrock.

## How the evidence fits together

| Artifact | Role | Kind of assurance |
| --- | --- | --- |
| [`paper/`](paper/) | Explains the theory and arguments for human review. | Research presentation |
| [`proof/`](proof/) | Defines the languages and machine-checks the theorems in Lean 4. | Universal proof within the stated model |
| [`oracle/`](oracle/) | Generates bounded graphs, asks compiled Lean for expected results, and compares them with Massively. | Differential and property-test evidence for the implementation |
| [`graph-algorithms/`](graph-algorithms/) | Implements complete algorithms with CPU references and GPU benchmarks. | Practical coverage evidence for developers |

Most readers only need this README. The paper is for readers who want the
mathematics, while [`proof/STATUS.md`](proof/STATUS.md) is the exact theorem,
assumption, and open-boundary ledger.

## What is not claimed

- The Rust, CubeCL, compiler, and GPU stack are not formally verified.
- Real execution time, memory traffic, allocation cost, and host/device
  transfers are not predicted by the current abstract cost theorem.
- The proof does not establish the domain-level correctness of every complete
  graph algorithm in the repository.
- The model currently concerns finite, ordered, static graphs and finite runs.
- “Complete” is not a claim of unrestricted graph-language or Turing
  completeness.

These limits are intentional: they keep the public claim strong, precise, and
reproducible.

## Reproducing the artifact

Run these commands at the repository root:

```text
just ta::paper             rebuild paper.pdf in Docker
just ta::proof             check every Lean definition and theorem in Docker
just ta::generate          evaluate the Lean model and generate Rust fixtures
just ta::check-generated   verify that committed fixtures are current
just ta::oracle            compare Lean with Massively on the host GPU
just ta::algorithms        compare complete algorithms with CPU references
just ta::bench-graph       benchmark complete graph algorithms
just ta::check             run the complete artifact pipeline
```

The same recipes work without the `ta::` prefix from this directory. Docker
builds the paper, Lean proofs, native oracle, and generated Rust fixtures. The
comparison with Massively runs on the host so CubeCL can use its GPU stack.

By default, `proptest` checks 256 generated valid CSR/frontier inputs and
shrinks failures to smaller counterexamples. A larger campaign can be run with:

```text
TRAVERSAL_ALGEBRA_PROPTEST_CASES=4096 just ta::oracle
```

No finite test campaign replaces the Lean theorems. Its job is to check that
the implementation continues to behave like the proved executable model on a
broad, automatically generated set of inputs.
