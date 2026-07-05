#![allow(unused_imports)]
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

mod common;

#[path = "api/adjacent_difference.rs"]
mod adjacent_difference;
#[path = "api/adjacent_find.rs"]
mod adjacent_find;
#[path = "api/copy_where.rs"]
mod copy_where;
#[path = "api/count_if.rs"]
mod count_if;
#[path = "api/device_slice.rs"]
mod device_slice;
#[path = "api/eager.rs"]
mod eager;
#[path = "api/equal.rs"]
mod equal;
#[path = "api/exclusive_scan.rs"]
mod exclusive_scan;
#[path = "api/exclusive_scan_by_key.rs"]
mod exclusive_scan_by_key;
#[path = "api/fill.rs"]
mod fill;
#[path = "api/find_first_of.rs"]
mod find_first_of;
#[path = "api/find_if.rs"]
mod find_if;
#[path = "api/gather.rs"]
mod gather;
#[path = "api/gather_where.rs"]
mod gather_where;
#[path = "api/inclusive_scan.rs"]
mod inclusive_scan;
#[path = "api/inclusive_scan_by_key.rs"]
mod inclusive_scan_by_key;
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
#[path = "api/random.rs"]
mod random;
#[path = "api/reduce.rs"]
mod reduce;
#[path = "api/reduce_by_key.rs"]
mod reduce_by_key;
#[path = "api/remove_where.rs"]
mod remove_where;
#[path = "api/replace_where.rs"]
mod replace_where;
#[path = "api/reverse.rs"]
mod reverse;
#[path = "api/scatter.rs"]
mod scatter;
#[path = "api/scatter_where.rs"]
mod scatter_where;
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
    let exec = massively::Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let input = exec.to_device(&[1_u32, 2, 3]).unwrap();
    let _: massively::runtime::DeviceSlice<'_, WgpuRuntime, u32> = input.slice(..);

    assert_eq!(exec.to_host(&input).unwrap(), vec![1, 2, 3]);

    let sum = massively::algorithm::reduce(
        &exec,
        massively::algorithm::Zip1(input.slice(..)),
        (0_u32,),
        common::Sum,
    )
    .unwrap();

    assert_eq!(sum, (6,));
}
