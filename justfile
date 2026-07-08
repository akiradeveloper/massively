doc:
    cargo doc -p massively --no-deps
    python3 -m http.server --directory target/doc 3000

bench:
    cargo bench -p massively > doc.ai/bench.log

test:
    cargo nextest run

test-scale:
    cargo nextest run -p oracle --test scale --run-ignored ignored-only --no-fail-fast
