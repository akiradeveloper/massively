doc:
    cargo doc -p massively --no-deps
    python3 -m http.server --directory target/doc 3000

bench:
    cargo bench -p massively

test-api:
    cargo doc -p massively --no-deps
    bash scripts/check-public-api.sh

test: test-api
    cargo nextest run
    cargo test -p massively --doc

test-scale:
    cargo nextest run -p massively --test vector_oracle_scale --no-fail-fast
