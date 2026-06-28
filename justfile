doc:
    cargo doc
    python3 -m http.server --directory target/doc 3000

bench:
    cargo bench -p massively > doc.ai/bench.log
