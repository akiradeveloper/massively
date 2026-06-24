# API Examples

This directory contains small runnable programs for public `massively` APIs.

- `algorithm/`: one example per `massively::algorithm` free-function API.
- `runtime/`: CubeCL runtime/device setup, host/device transfer, allocation,
  initialization, random generation, and memory-copy examples.

Run one example:

```sh
cargo run --example sort
cargo run --example runtime-random
cargo run --example runtime-tabulate
```

Algorithm example names match the Rust function names, for example:

- `sort.rs` -> `cargo run --example sort`
- `reduce_by_key.rs` -> `cargo run --example reduce-by-key`

`algorithm/common.rs` contains shared CubeCL operator marker types used by
algorithm examples.
