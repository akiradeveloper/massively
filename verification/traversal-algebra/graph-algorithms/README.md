# graph-algorithms

This crate is the practical coverage evidence for Massively's Traversal
Algebra. The mathematical result in the [Traversal Algebra artifact](../)
shows what the programming model can express. This crate answers the separate,
developer-facing question: can it implement a broad, recognizable set of real
graph problems?

The comparison target is the newer Essentials abstraction in Gunrock's
published
[Graph Algorithms table](https://gunrock.github.io/gunrock/gunrock.wiki/Graph-Algorithms.html),
because it represents Gunrock's current research frontier. In this document,
that column is therefore labeled simply **Gunrock**; the older abstraction is
not part of the comparison. As of 2026-07-15, Traversal Algebra implements
every application that the table marks as supported by current Gunrock, plus
all four applications marked as priority future ports (`🌟`). That is 16 of 16
rows in this comparison scope.

This is a coverage result: Traversal Algebra currently covers four applications
that the table still lists as future Gunrock work. It is not yet a claim that
Massively is faster than Gunrock; that requires a controlled benchmark
of both libraries on the same graphs, GPU, output semantics, and stopping
criteria.

## Gunrock comparison

This table includes only applications that the official table's Essentials
column marks as supported by current Gunrock or as a priority future port
(`🌟`). The **Gunrock** column reproduces that column's status. `✅` means this
crate has a complete entry point, an independent CPU oracle exercised by
generated property tests, and a device-resident benchmark.

| Application | File | Gunrock | Traversal Algebra |
| --- | --- | --- | --- |
| Betweenness Centrality | [bc](src/bc.rs) | v0.0.1 | ✅ |
| Breadth-First Search | [bfs](src/bfs.rs) | v0.0.1 | ✅ |
| Connected Components | [cc](src/cc.rs) | ❌🌟 | ✅ |
| Graph Coloring | [color](src/color.rs) | v0.0.1 | ✅ |
| Geolocation | [geo](src/geo.rs) | v0.0.1 | ✅ |
| Hyperlink-Induced Topic Search | [hits](src/hits.rs) | v0.0.1 | ✅ |
| K-Core Decomposition | [kcore](src/kcore.rs) | v0.0.1 | ✅ |
| Louvain Modularity | [louvain](src/louvain.rs) | ❌🌟 | ✅ |
| Minimum Spanning Tree | [mst](src/mst.rs) | v0.0.1 | ✅ |
| PageRank | [pr](src/pr.rs) | v0.0.1 | ✅ |
| Local Graph Clustering | [pr_nibble](src/pr_nibble.rs) | v0.0.1 | ✅ |
| Random Walk | [rw](src/rw.rs) | ❌🌟 | ✅ |
| Subgraph Matching | [sm](src/sm.rs) | ❌🌟 | ✅ |
| Sparse-Matrix Vector Multiplication | [spmv](src/spmv.rs) | v0.0.1 | ✅ |
| Single Source Shortest Path | [sssp](src/sssp.rs) | v0.0.1 | ✅ |
| Triangle Counting | [tc](src/tc.rs) | v0.0.1 | ✅ |

The crate also contains three complete applications not listed in that Gunrock
table: [Forman–Ricci curvature](src/forman_ricci.rs),
[personalized PageRank](src/ppr.rs), and [Boolean SpGEMM](src/spgemm.rs).

## What the new priority implementations mean

- `cc` returns the minimum vertex identifier in each connected component by
  sparse-frontier minimum-label relaxation.
- `louvain` performs deterministic standard modularity-gain local moves and repeated
  weighted community contraction, not label propagation under a Louvain name.
- `rw` implements batched uniform random walks, including multiple walks per
  vertex, deterministic seeded generation, explicit random-word injection for
  testing, and dead-end termination.
- `sm` performs exact, unlabelled, non-induced subgraph isomorphism and returns
  every ordered embedding. Its exhaustive candidate space is intended for
  small query graphs.
- `pr_nibble` computes personalized PageRank and then returns the
  minimum-conductance prefix of the degree-normalized sweep order.

The undirected algorithms expect symmetric CSR input. Adjacency rows used by
set operations must be sorted. These semantic boundaries are documented in the
corresponding modules and reproduced by their CPU references.

## Reproducing the evidence

Generated CSR property tests compare all 19 algorithms with independent CPU
implementations. The new subgraph-matching cases vary the query shape, and
random walks are compared from caller-supplied random words so the graph
semantics are tested independently of the RNG implementation.

```sh
cargo nextest run -p graph-algorithms --test oracle
```

The benchmark suite measures device-resident algorithm entry points. CSR
topology, weights, vectors, and random-walk choices are prepared outside the
timed region where applicable. Each iteration includes algorithm execution,
pooled output allocation, and explicit synchronization, but no bulk result
download. Sequential-control algorithms may return individual control scalars
while graph state and outputs remain resident.

```sh
cargo bench -p graph-algorithms --bench algorithms
```

The formal result concerns Traversal Algebra and its semantics-preserving
transformations; it does not automatically prove that each Rust program solves
its named textbook problem. The independent CPU oracles test that separate
claim. Conversely, implementing this broad suite is empirical coverage
evidence, not another premise of the mathematical completeness theorem.
