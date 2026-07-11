# massively recipes

Small algorithms composed from Massively primitives. The sources are split by
domain:

- `src/vector/`: vector and data-processing algorithms
- `src/graph/`: graph algorithms written with `massively::graph` traversal algebra

The graph recipes are runtime-generic `solve` functions. Their tests use the
WGPU CPU adapter, exercise the same GPU kernels, and serve as compact runnable
examples:

```sh
cargo nextest run -p massively --test graph_oracle
```

The oracle test generates valid sorted CSR graphs and compares all graph
recipes with independent CPU implementations.

## Vector recipes

- `monte_carlo_pi`
- `delayed_freight_by_route`
- `top_k_players`
- `histogram_u32`
- `unique_visitors_per_page`
- `warehouse_reorder_list`
- `race_lap_leaderboard`
- `merge_ranked_feeds`
- `fraudulent_transactions`
- `particle_survivors`
- `ticket_queue_offsets`
- `sparse_feature_gather`
- `ranked_slots_scatter`
- `common_customers`
- `allowed_users_after_banlist`
- `timestamp_window_query`
- `first_temperature_spike`
- `account_running_balance`
- `daily_inventory_delta`
- `first_blocked_keyword`
- `exam_pass_count`
- `price_minmax_range`
- `config_drift_index`
- `sorted_prefix_report`
- `price_change_series`
- `optional_feature_gather`
- `approved_rank_scatter`
- `duplicate_score_range`
- `merged_audience_union`
- `release_version_order`

## Graph recipes

- `bc`: betweenness centrality
- `bfs`: breadth-first search
- `color`: graph coloring
- `forman_ricci`: Forman–Ricci edge curvature
- `geo`: graph-based geolocation
- `hits`: hub and authority scores
- `kcore`: k-core decomposition
- `mst`: minimum spanning tree
- `ppr`: personalized PageRank
- `pr`: PageRank
- `spgemm`: Boolean sparse matrix multiplication
- `spmv`: sparse matrix-vector multiplication
- `sssp`: single-source shortest paths
- `tc`: triangle counting

These are composition and correctness recipes, not alternative public graph
APIs. Edge programs use source, destination, and edge expressions followed by
emit, source/destination reduction, destination-state update, or batched
adjacency intersection. Iterative algorithms keep convergence control on the
host while bulk traversal and state transitions run through Massively.
