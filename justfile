doc:
    cargo doc -p massively --no-deps
    python3 -m http.server --directory target/doc 3000

bench:
    cargo bench -p massively

test:
    cargo nextest run

test-scale:
    cargo nextest run -p massively --test oracle_scale --run-ignored ignored-only --no-fail-fast
