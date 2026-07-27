# scan-core

Shared, language-neutral Rust implementation used by the `scanr` and `scan-py` bindings.

During local development, both wrapper projects use this crate through a relative Cargo path.
After this directory is published as its own Git repository, replace the path dependencies with
a pinned Git tag or commit revision.

## Benchmarking

```sh
cargo run --release --example bench
```
