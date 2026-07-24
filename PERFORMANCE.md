# Performance

Reference execution times for the vector API at N = 10,000,000.
The table lists APIs covered by the dedicated performance benchmark.
These are approximate, machine-specific values rather than performance guarantees.

| Environment | Value |
|---|---|
| GPU | AMD Radeon 680M |
| Runtime | CubeCL WGPU default device |
| Rust | `rustc 1.96.0 (ac68faa20 2026-05-25)` |
| Revision | `a957c839-dirty` |
| Measured | 2026-07-24 |

| API | Time |
|---|---:|
| `all_of` | 434 us |
| `any_of` | 418 us |
| `copy_where` | 6.47 ms |
| `count_if` | 428 us |
| `exclusive_scan` | 3.62 ms |
| `exclusive_scan_by_key` | 7.41 ms |
| `find_if` | 440 us |
| `gather` | 2.37 ms |
| `gather_where` | 7.83 ms |
| `inclusive_scan` | 3.58 ms |
| `inclusive_scan_by_key` | 5.54 ms |
| `is_partitioned` | 477 us |
| `lower_bound` | 66.7 ms |
| `map` | 1.64 ms |
| `max_element` | 267 us |
| `merge` | 6.23 ms |
| `merge_by_key` | 10.1 ms |
| `min_element` | 279 us |
| `minmax_element` | 613 us |
| `none_of` | 402 us |
| `partition` | 7.19 ms |
| `radix_sort_by_key` | 138 ms |
| `reduce` | 1.08 ms |
| `reduce_by_key` | 11.6 ms |
| `remove_where` | 6.46 ms |
| `scatter` | 2.41 ms |
| `scatter_reduce` | 54.4 ms |
| `scatter_where` | 6.80 ms |
| `set_difference` | 9.44 ms |
| `set_intersection` | 9.40 ms |
| `set_union` | 14.6 ms |
| `sort` | 42.8 ms |
| `sort_by_key` | 200 ms |
| `unique` | 6.58 ms |
| `unique_by_key` | 6.59 ms |
| `upper_bound` | 66.9 ms |

## Conditions

Stored inputs are already device-resident, and input construction and host-to-device transfer are excluded. API-internal output allocation and synchronization required to observe completion are included; caller-provided output buffers are allocated before timing. Times are Criterion point estimates after warm-up (slope when available, otherwise mean).

N is the length of each input range, so binary range algorithms process two inputs of N elements. The default element type is `f32` for value algorithms and `u32` for ordering, index, and key algorithms. Predicate and extremum queries use `lazy::counting`, whose item type is `MIndex`. Each `lower_bound` and `upper_bound` measurement performs N batched searches over an N-element source. Selection uses 50% u32 backing flags converted lazily to a bool stencil with `op::NonZero`, indexed operations use reverse indices, by-key algorithms use runs of eight equal keys, `scatter_reduce` maps four inputs to each output, and sorting uses deterministic shuffled keys.

Regenerate this file with:

```console
MASSIVELY_BENCH_DEVICE="<GPU model>" just performance
```
