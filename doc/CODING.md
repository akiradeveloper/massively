# Read This Before Coding

## Discussion

Before writing any code, discuss the task thoroughly and make sure you
understand what needs to be done and why.

## Implementation

### Do Not Optimize Specifically for Single Columns or Small Inputs

This library is intended for many columns and large datasets.

Therefore:

- Write general code that supports multiple columns. Macros are fine. Do not
  write ad hoc multi-column handling.
- Do not add optimizations that are specific to a single column.
- Do not add optimizations that are specific to small inputs.

### Prefer Simplicity

Among implementations that satisfy the requirements, choose the simplest one.
Do not do unnecessary work.

### Implement Algorithms by Combining Primitives

Parallel algorithms can be implemented by composing primitive algorithms such
as scan and compact.

There are two ways to implement an API: write a specialized implementation for
each algorithm, or express it as a composition of primitives. `massively` takes
the latter approach.

This keeps the implementation compact and makes improvements to each primitive
benefit the whole library.

Prefer a clean design over small optimizations.
Algorithms should be written in their simplest form.

### Avoid Arity Explosion

Some APIs take multiple parameters, such as keys and values. Suppose these are
`Param1` and `Param2`. If `Param1` is `Zip2` and `Param2` is `Zip3`, and the
implementation has to know about the `Zip2 x Zip3` combination directly, the
number of implementation cases grows multiplicatively. We call this "arity
explosion."

An implementation that suffers from arity explosion cannot scale to more key or
value columns, and compile times will also grow explosively.

Therefore, implement APIs in a way that avoids arity explosion.

For example, a by-key API can be decomposed into two stages: first build the
control structure from the keys, then apply that control structure to the
values.

## Tests

Each test area has a specific role.

- `massively/benches`: Measure single-column performance and use the results to
  guide internal implementation improvements. Because we do not add
  single-column-specific optimizations, improving the single-column foundation
  should also improve multi-column cases.
- `massively/tests`: Use simple data to verify that multi-column support works.
- `massively/tests/oracle.rs`: Property tests against the multi-column AoS CPU
  reference.
- `massively/tests/oracle_scale.rs`: Ignored scale property tests against the
  multi-column AoS CPU reference.
