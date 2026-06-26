# API Examples

This directory contains small runnable programs for public `massively` APIs.

- `algorithm/`: one example per `massively::algorithm` free-function API.
- `runtime/`: CubeCL runtime/device setup, host/device transfer, allocation,
  initialization, and memory-copy examples.
- `util/`: utility examples such as random generation.

Run one example:

```sh
cargo run --example sort
cargo run --example util-random
cargo run --example runtime-tabulate
```

Algorithm example names match the Rust function names, for example:

- `sort.rs` -> `cargo run --example sort`
- `reduce_by_key.rs` -> `cargo run --example reduce-by-key`

`algorithm/common.rs` contains shared CubeCL operator marker types used by
algorithm examples.
