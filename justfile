default:
    @just --list --list-submodules

mod ta 'traversal-algebra'

doc:
    cargo doc -p massively --no-deps
    python3 -m http.server --directory target/doc 3000

bench:
    cargo bench -p massively

test-api:
    cargo doc -p massively --no-deps
    bash scripts/check-public-api.sh

test: test-api ta::proof
    cargo nextest run
    cargo test -p massively --doc