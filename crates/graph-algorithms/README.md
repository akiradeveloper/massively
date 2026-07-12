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

The benchmark suite measures only device-resident algorithm entry points.
CSR topology, weights, and input vectors are uploaded before timing. Each timed
iteration includes algorithm execution, pooled output allocation, and an
explicit synchronization, but no bulk host/device transfer. Every algorithm
entry point in this crate satisfies this contract. Algorithms whose exact
control flow is sequential may return individual control scalars to the host,
but their graph, state vectors, and output remain resident:

```sh
cargo bench -p graph-algorithms --bench algorithms
```

`massively` owns the graph primitives and their semantic tests. This crate owns
complete algorithm correctness and complete algorithm benchmarks.
