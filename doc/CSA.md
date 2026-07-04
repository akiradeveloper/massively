# CSA Design

CSA is the internal algorithm structure used by massively.

```text
algorithm = control generation + apply/query/materialize
```

The public API should stay algorithm-shaped.  CSA is an implementation rule for
`detail`, not a user-facing vocabulary.

## Control

Control objects describe where data should move or which positions should be
observed.  They must not own payload handles.

Examples:

- `MaskControl`: flags-only control for fixed-position writes and queries.
- `SelectedRankControl`: selected flags, rank, and count for compaction.
- `SplitRankControl`: selected/rejected ranks for partition.
- `PermutationControl`: source indices for gather/permutation.
- `SegmentControl`: segment heads/ends for by-key algorithms.
- `MergeControl`: source side and source index for plain merge.
- `MergeByKeyControl`: source side and source index for merge-by-key values.
- `SearchControl`: flags/candidate positions for scalar search queries.
- `RangeControl`: range mappings such as reverse.

## Apply

Apply objects live under `detail/apply`.  They own payload movement, query
readback, materialization, or write boundaries.

Examples:

- `SelectedPayloadApply` and `SplitPayloadApply`
- `PermutationPayloadApply`, `IndexedExprApply`, and `IndexedWriteApply`
- `MergePayloadApply`
- `SegmentedScanApply` and `SegmentedReduceApply`
- `SearchPayloadApply` and `TupleSearchPayloadApply`
- `MaterializePayloadApply`, `MaterializeWriteApply`, and `FillWriteApply`
- `TransformPayloadApply`

Algorithm glue may choose the control and the apply object.  It should not
contain ad-hoc payload movement or arity-specific kernel launch code when a
typed apply boundary exists.

## Family Boundaries

Ordering algorithms are split by responsibility:

- `SortApply` owns direct sort primitive calls for arity 1-3.
- `SortByKeyApply` owns key sorting and stable index generation.
- Wide tuple `sort` materializes sorted indices as `OrderingControl`, then uses
  `PermutationPayloadApply` for payload movement.

Search algorithms are split into scalar query and vector payload paths:

- Scalar search queries build `SearchControl` from flags and read it through
  `QueryApply`.
- Single-column `lower_bound_many` and `upper_bound_many` use
  `SearchPayloadApply`.
- Tuple `lower_bound_many` and `upper_bound_many` use
  `TupleSearchPayloadApply`, which owns the arity-specific kernel launch,
  staging, and output construction.

Merge algorithms are split into order/control and payload movement:

- Plain `merge` builds `MergeControl` through `MergeControlApply`, then applies
  payload through `MergePayloadApply`.
- `merge_by_key` builds merged keys and `MergeByKeyControl` through
  `MergeByKeyControlApply`; values consume that control through
  `MergePayloadApply`.

Multi-column support should be centralized in typed apply boundaries or
macro-shaped dispatch boundaries.  Algorithm glue may select an apply object,
but arity-specific staging, padding, and payload movement should stay inside
the apply or primitive boundary that owns it.

## Raw Kernel Launches

Raw CubeCL launches are allowed in implementation boundaries:

- primitives;
- `detail/api/expr`;
- control-generation helpers;
- `detail/apply`;
- documented staging helpers.

They should not leak into high-level algorithm glue.  The
`csa_invariants` test suite keeps this boundary explicit.

## Performance Direction

CSA is a performance staging design.  The point is not only cleaner code, but
also making high-performance kernels easier to introduce:

- optimize a control once and reuse it across payload arities;
- optimize an apply once and reuse it across algorithms;
- avoid arity multiplication by separating key/control generation from value
  payload movement;
- keep multi-column paths structurally similar to single-column paths.

When a fused fast path is added, it should either remain inside an apply object
or expose the same control/apply shape so that the algorithm layer does not
fork into special cases.
