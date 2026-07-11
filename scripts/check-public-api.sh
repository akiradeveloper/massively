#!/usr/bin/env bash
set -euo pipefail

forbidden='SegmentKeyInput|SegmentedValues|CanonicalAlloc|CanonicalStorage|AllocColumns|GatherInput|ReadExpression|OutputExpression|StorageLayout|MaterializeDispatch|ReduceDispatch|ReadArity|StorageArity|Eval[1-8]|Column'

mapfile -d '' pages < <(
    find target/doc/massively -type f \
        \( -name 'fn.*.html' -o -name 'trait.MAlloc.html' \
        -o -name 'trait.MIter.html' -o -name 'trait.MIterMut.html' \
        -o -name 'trait.MStorage.html' \) \
        -print0
)

if rg -n "$forbidden" "${pages[@]}"; then
    echo 'internal kernel constraints leaked into the public API documentation' >&2
    exit 1
fi

algorithms=(
    adjacent_difference adjacent_find all_of any_of copy_where count_if equal
    exclusive_scan exclusive_scan_by_key fill find_first_of find_if gather gather_where
    inclusive_scan inclusive_scan_by_key is_partitioned is_sorted is_sorted_until
    lexicographical_compare lower_bound max_element merge merge_by_key min_element
    minmax_element mismatch none_of partition reduce reduce_by_key remove_where
    replace_where reverse scatter scatter_reduce scatter_where set_difference
    set_intersection set_union sort sort_by_key transform transform_where unique
    unique_by_key upper_bound
)

for algorithm in "${algorithms[@]}"; do
    test -f "target/doc/massively/vector/fn.${algorithm}.html"
    test ! -e "target/doc/massively/fn.${algorithm}.html"
done

operations=(BinaryPredicateOp PredicateOp ReductionOp UnaryOp)

for operation in "${operations[@]}"; do
    test -f "target/doc/massively/op/trait.${operation}.html"
    test ! -e "target/doc/massively/trait.${operation}.html"
done
