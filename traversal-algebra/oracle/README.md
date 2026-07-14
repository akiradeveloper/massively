# Traversal Algebra implementation oracle

The Lean theorems prove statements about a mathematical model. This crate asks
the separate engineering question: does the current Rust/GPU implementation
behave like that model on generated inputs?

For each property case, `proptest` generates a valid bounded graph and frontier,
compiled Lean computes the expected result, Massively computes the actual
result, and the test compares them. Failures are reduced to smaller examples
automatically.

This crate deliberately contains no handwritten reference traversal. Lean
provides both oracle paths from the semantics in `../proof/`:

- `src/generated.rs` contains five named regression fixtures evaluated by Lean.
- the Docker-built `proof/.lake/build/bin/oracle` evaluates arbitrary valid CSR
  graphs and frontiers using the same Lean definitions.

The integration test first checks that the fixtures still equal the compiled
Lean oracle. It then checks 256 generated graph/frontier inputs by default and
compares edge contexts plus source and destination reductions.

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

No finite randomized campaign proves universal correspondence. Its role is to
connect the formal executable semantics to the Rust/GPU implementation and to
find small counterexamples; universal claims remain Lean theorem obligations.
See the [Traversal Algebra overview](../README.md) for the plain-language proof
scope and the distinction between proof and implementation evidence.
