#![allow(unused_imports)]

mod common;

#[path = "api/adjacent_difference.rs"]
mod adjacent_difference;
#[path = "api/adjacent_find.rs"]
mod adjacent_find;
#[path = "api/copy_if.rs"]
mod copy_if;
#[path = "api/count_if.rs"]
mod count_if;
#[path = "api/device_slice.rs"]
mod device_slice;
#[path = "api/equal.rs"]
mod equal;
#[path = "api/equal_range.rs"]
mod equal_range;
#[path = "api/exclusive_scan.rs"]
mod exclusive_scan;
#[path = "api/exclusive_scan_by_key.rs"]
mod exclusive_scan_by_key;
#[path = "api/find_first_of.rs"]
mod find_first_of;
#[path = "api/find_if.rs"]
mod find_if;
#[path = "api/gather.rs"]
mod gather;
#[path = "api/gather_if.rs"]
mod gather_if;
#[path = "api/inclusive_scan.rs"]
mod inclusive_scan;
#[path = "api/inclusive_scan_by_key.rs"]
mod inclusive_scan_by_key;
#[path = "api/inner_product.rs"]
mod inner_product;
#[path = "api/is_partitioned.rs"]
mod is_partitioned;
#[path = "api/is_sorted.rs"]
mod is_sorted;
#[path = "api/is_sorted_until.rs"]
mod is_sorted_until;
#[path = "api/lexicographical_compare.rs"]
mod lexicographical_compare;
#[path = "api/lower_bound.rs"]
mod lower_bound;
#[path = "api/merge.rs"]
mod merge;
#[path = "api/merge_by_key.rs"]
mod merge_by_key;
#[path = "api/minmax_element.rs"]
mod minmax_element;
#[path = "api/mismatch.rs"]
mod mismatch;
#[path = "api/partition.rs"]
mod partition;
#[path = "api/reduce.rs"]
mod reduce;
#[path = "api/reduce_by_key.rs"]
mod reduce_by_key;
#[path = "api/remove_if.rs"]
mod remove_if;
#[path = "api/replace_if.rs"]
mod replace_if;
#[path = "api/reverse.rs"]
mod reverse;
#[path = "api/scatter.rs"]
mod scatter;
#[path = "api/scatter_if.rs"]
mod scatter_if;
#[path = "api/set_difference.rs"]
mod set_difference;
#[path = "api/set_intersection.rs"]
mod set_intersection;
#[path = "api/set_union.rs"]
mod set_union;
#[path = "api/sort.rs"]
mod sort;
#[path = "api/sort_by_key.rs"]
mod sort_by_key;
#[path = "api/transform.rs"]
mod transform;
#[path = "api/unique.rs"]
mod unique;
#[path = "api/unique_by_key.rs"]
mod unique_by_key;
#[path = "api/upper_bound.rs"]
mod upper_bound;
#[path = "api/zip.rs"]
mod zip;

#[test]
fn public_api_is_available_from_massively() {
    let policy = massively::Policy::<massively::Wgpu>::cpu();
    let input = policy.to_device(&[1_u32, 2, 3]).unwrap();

    assert_eq!(input.to_vec().unwrap(), vec![1, 2, 3]);
}
