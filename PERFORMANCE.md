# Performance

Reference execution times for the vector API at N = 10,000,000.
The table lists APIs covered by the dedicated performance benchmark.
These are approximate, machine-specific values rather than performance guarantees.

| Environment | Value |
|---|---|
| GPU | AMD Radeon 680M |
| Runtime | CubeCL WGPU default device |
| Rust | `rustc 1.96.0 (ac68faa20 2026-05-25)` |
| Revision | `c6424260` |
| Measured | 2026-07-20 |

| API | Time |
|---|---:|
| `all_of` | 875 us |
| `any_of` | 564 us |
| `copy_where` | 6.18 ms |
| `count_if` | 1.80 ms |
| `exclusive_scan` | 5.11 ms |
| `exclusive_scan_by_key` | 10.2 ms |
| `find_if` | 496 us |
| `gather` | 2.27 ms |
| `gather_where` | 8.12 ms |
| `inclusive_scan` | 3.57 ms |
| `inclusive_scan_by_key` | 8.26 ms |
| `is_partitioned` | 518 us |
| `lower_bound` | 70.8 ms |
| `max_element` | 346 us |
| `merge` | 39.2 ms |
| `min_element` | 289 us |
| `minmax_element` | 655 us |
| `none_of` | 504 us |
| `partition` | 8.30 ms |
| `radix_sort_by_key` | 133 ms |
| `reduce` | 1.22 ms |
| `reduce_by_key` | 14.4 ms |
| `remove_where` | 6.48 ms |
| `scatter` | 2.39 ms |
| `scatter_reduce` | 61.2 ms |
| `scatter_where` | 7.01 ms |
| `set_difference` | 20.4 ms |
| `set_intersection` | 21.3 ms |
| `set_union` | 54.7 ms |
| `sort` | 55.3 ms |
| `sort_by_key` | 357 ms |
| `transform` | 1.66 ms |
| `unique_by_key` | 6.72 ms |
| `upper_bound` | 72.0 ms |

## Conditions

Stored inputs are already device-resident, and input construction and host-to-device transfer are excluded. API-internal output allocation and synchronization required to observe completion are included; caller-provided output buffers are allocated before timing. Times are Criterion point estimates after warm-up (slope when available, otherwise mean).

N is the length of each input range, so binary range algorithms process two inputs of N elements. The default element type is `f32` for value algorithms and `u32` for ordering, index, and key algorithms. Predicate and extremum queries use `lazy::counting<usize>`. Selection uses a 50% stencil, indexed operations use reverse indices, by-key algorithms use runs of eight equal keys, `scatter_reduce` maps four inputs to each output, and sorting uses deterministic shuffled keys.

Regenerate this file with:

```console
MASSIVELY_BENCH_DEVICE="<GPU model>" just performance
```
