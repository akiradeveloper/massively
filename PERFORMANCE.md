# Performance

Reference execution times for the vector API at N = 10,000,000.
The table lists APIs covered by the dedicated performance benchmark.
These are approximate, machine-specific values rather than performance guarantees.

| Environment | Value |
|---|---|
| GPU | AMD Radeon 680M |
| Runtime | CubeCL WGPU default device |
| Rust | `rustc 1.96.0 (ac68faa20 2026-05-25)` |
| Revision | `eaea0688-dirty` |
| Measured | 2026-07-22 |

| API | Time |
|---|---:|
| `all_of` | 471 us |
| `any_of` | 400 us |
| `copy_where` | 5.71 ms |
| `count_if` | 411 us |
| `exclusive_scan` | 3.63 ms |
| `exclusive_scan_by_key` | 7.50 ms |
| `find_if` | 414 us |
| `gather` | 2.41 ms |
| `gather_where` | 7.72 ms |
| `inclusive_scan` | 3.54 ms |
| `inclusive_scan_by_key` | 5.42 ms |
| `is_partitioned` | 542 us |
| `lower_bound` | 65.7 ms |
| `max_element` | 263 us |
| `merge` | 6.20 ms |
| `merge_by_key` | 10.2 ms |
| `min_element` | 248 us |
| `minmax_element` | 626 us |
| `none_of` | 416 us |
| `partition` | 6.95 ms |
| `radix_sort_by_key` | 138 ms |
| `reduce` | 1.08 ms |
| `reduce_by_key` | 11.2 ms |
| `remove_where` | 5.89 ms |
| `scatter` | 2.36 ms |
| `scatter_reduce` | 54.4 ms |
| `scatter_where` | 6.60 ms |
| `set_difference` | 8.79 ms |
| `set_intersection` | 8.83 ms |
| `set_union` | 12.6 ms |
| `sort` | 43.2 ms |
| `sort_by_key` | 199 ms |
| `map` | 1.67 ms |
| `unique` | 6.37 ms |
| `unique_by_key` | 6.43 ms |
| `upper_bound` | 66.8 ms |

## Conditions

Stored inputs are already device-resident, and input construction and host-to-device transfer are excluded. API-internal output allocation and synchronization required to observe completion are included; caller-provided output buffers are allocated before timing. Times are Criterion point estimates after warm-up (slope when available, otherwise mean).

N is the length of each input range, so binary range algorithms process two inputs of N elements. The default element type is `f32` for value algorithms and `u32` for ordering, index, and key algorithms. Predicate and extremum queries use `lazy::counting`, whose item type is `MIndex`. Each `lower_bound` and `upper_bound` measurement performs N batched searches over an N-element source. Selection uses 50% u32 backing flags converted lazily to a bool stencil with `op::NonZero`, indexed operations use reverse indices, by-key algorithms use runs of eight equal keys, `scatter_reduce` maps four inputs to each output, and sorting uses deterministic shuffled keys.

Regenerate this file with:

```console
MASSIVELY_BENCH_DEVICE="<GPU model>" just performance
```
