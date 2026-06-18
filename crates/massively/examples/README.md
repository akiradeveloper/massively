# API Examples

Each file in this directory is a small runnable program for one public
free-function API.

Run one example:

```sh
cargo run --example sort
```

The file names match the Rust function names, for example:

- `sort.rs` -> `cargo run --example sort`
- `reduce_by_key.rs` -> `cargo run --example reduce-by-key`

`common.rs` contains shared CubeCL operator marker types used by the examples.
