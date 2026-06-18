doc:
    cargo doc
    python3 -m http.server --directory target/doc 3000

bench:
    cargo bench > doc.ai/bench.log
