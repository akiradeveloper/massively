# graph-algorithms

Complete graph algorithms composed from `massively::graph` traversal and
aggregation primitives.

The crate currently provides:

- betweenness centrality
- breadth-first search
- graph coloring
- Forman–Ricci edge curvature
- graph-based geolocation
- HITS
- k-core decomposition
- minimum spanning forest
- personalized PageRank
- PageRank
- Boolean sparse matrix multiplication
- sparse matrix-vector multiplication
- single-source shortest paths
- triangle counting

Generated CSR property tests compare every algorithm with an independent CPU
implementation:

```sh
cargo nextest run -p graph-algorithms --test oracle
```

The current benchmark suite measures each public `solve` function end to end,
including allocation, host/device transfer, host orchestration, and result
transfer. It establishes a baseline for removing those costs in later
device-resident implementations:

```sh
cargo bench -p graph-algorithms --bench algorithms
```

`massively` owns the graph primitives and their semantic tests. This crate owns
complete algorithm correctness and complete algorithm benchmarks.
