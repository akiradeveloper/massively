#!/usr/bin/env bash
set -euo pipefail

forbidden='SegmentKeyInput|SegmentedValues|MStorage|GatherInput|ReadExpression|OutputExpression|StorageLayout|MaterializeDispatch|ReduceDispatch|ReadArity|StorageArity|Eval[1-8]|Column'

mapfile -d '' pages < <(
    find target/doc/massively -maxdepth 1 -type f \
        \( -name 'fn.*.html' -o -name 'trait.MIter.html' \
        -o -name 'trait.MIterMut.html' \) \
        -print0
)

if rg -n "$forbidden" "${pages[@]}"; then
    echo 'internal kernel constraints leaked into the public API documentation' >&2
    exit 1
fi
