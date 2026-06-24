use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::op::{BinaryOp, BinaryPredicateOp, PredicateOp, ReductionOp, UnaryOp};
use massively::{
    DeviceVec, Executor as ApiExecutor, adjacent_difference as api_adjacent_difference,
    adjacent_find, all_of as api_all_of, any_of as api_any_of, copy_if as api_copy_if,
    count_if as api_count_if, equal, equal_range, exclusive_scan as api_exclusive_scan,
    exclusive_scan_by_key, find_first_of, find_if as api_find_if, gather as api_gather, gather_if,
    inclusive_scan as api_inclusive_scan, inclusive_scan_by_key, inner_product,
    is_partitioned as api_is_partitioned, is_sorted, is_sorted_until, lexicographical_compare,
    lower_bound, max_element, merge, merge_by_key, min_element, minmax_element, mismatch,
    none_of as api_none_of, partition as api_partition, reduce as api_reduce, reduce_by_key,
    remove_if as api_remove_if, replace_if as api_replace_if, reverse as api_reverse, scatter,
    scatter_if, set_difference, set_intersection, set_union, sort as api_sort, sort_by_key,
    stable_sort, stable_sort_by_key, transform as api_transform, unique as api_unique,
    unique_by_key, upper_bound,
};
use proptest::prelude::*;
use std::sync::{Mutex, MutexGuard};

type ApiRuntime = WgpuRuntime;

const CASES: u32 = 24;
const MAX_LEN: usize = 64;
static GPU_LOCK: Mutex<()> = Mutex::new(());

struct TransformMap;

#[cubecl::cube]
impl UnaryOp<ApiRuntime, (u32,)> for TransformMap {
    type Output = (u32,);

    fn apply(input: (u32,)) -> (u32,) {
        (input.0 / 3 + 17,)
    }
}

struct TupleMaxOp;

#[cubecl::cube]
impl ReductionOp<ApiRuntime, (u32,)> for TupleMaxOp {
    fn apply(lhs: (u32,), rhs: (u32,)) -> (u32,) {
        if lhs.0 > rhs.0 { lhs } else { rhs }
    }
}

struct TuplePairMax;

#[cubecl::cube]
impl BinaryOp<ApiRuntime, (u32,), (u32,)> for TuplePairMax {
    type Output = (u32,);

    fn apply(lhs: (u32,), rhs: (u32,)) -> (u32,) {
        if lhs.0 > rhs.0 { lhs } else { rhs }
    }
}

struct TupleKeep;

#[cubecl::cube]
impl PredicateOp<ApiRuntime, (u32,)> for TupleKeep {
    fn apply(input: (u32,)) -> bool {
        input.0 % 2 == 0
    }
}

struct TupleSameLowNibble;

#[cubecl::cube]
impl BinaryPredicateOp<ApiRuntime, (u32,)> for TupleSameLowNibble {
    fn apply(lhs: (u32,), rhs: (u32,)) -> bool {
        (lhs.0 % 16) == (rhs.0 % 16)
    }
}

struct TupleBucketThenValueLess;

#[cubecl::cube]
impl BinaryPredicateOp<ApiRuntime, (u32,)> for TupleBucketThenValueLess {
    fn apply(lhs: (u32,), rhs: (u32,)) -> bool {
        let lhs_key = lhs.0 % 16;
        let rhs_key = rhs.0 % 16;
        lhs_key < rhs_key || (lhs_key == rhs_key && lhs.0 < rhs.0)
    }
}

fn transform_map(input: &[u32]) -> Vec<u32> {
    input.iter().map(|value| value / 3 + 17).collect()
}

fn exec() -> ApiExecutor<ApiRuntime> {
    ApiExecutor::<ApiRuntime>::new(WgpuDevice::Cpu)
}

fn api_exec() -> ApiExecutor<ApiRuntime> {
    exec()
}

fn slice_range(input: &[u32]) -> std::ops::Range<usize> {
    1..input.len() + 1
}

fn padded_device(exec: &ApiExecutor<ApiRuntime>, input: &[u32]) -> DeviceVec<ApiRuntime, u32> {
    let mut padded = Vec::with_capacity(input.len() + 2);
    padded.push(0xface_feed);
    padded.extend_from_slice(input);
    padded.push(0xdead_beef);
    exec.to_device(&padded).unwrap()
}

fn gpu_lock() -> MutexGuard<'static, ()> {
    GPU_LOCK.lock().unwrap_or_else(|err| err.into_inner())
}

fn reverse_indices(len: usize) -> Vec<u32> {
    let mut indices = vec![0; len];
    for i in 0..len {
        indices[i] = (len - 1 - i) as u32;
    }
    indices
}

fn run_keys(len: usize) -> Vec<u32> {
    let mut keys = vec![0; len];
    for i in 0..len {
        keys[i] = (i / 3) as u32;
    }
    keys
}

fn unique_keys(len: usize) -> Vec<u32> {
    let mut keys = vec![0; len];
    for i in 0..len {
        keys[i] = (len - 1 - i) as u32;
    }
    keys
}

fn stencil_flags(input: &[u32]) -> Vec<u32> {
    input
        .iter()
        .map(|value| if oracle::keep(*value) { 1 } else { 0 })
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(CASES))]

    #[test]
    fn transform_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let exec = api_exec();
        let input_g = padded_device(&exec, &input);
        let (output_g,) = api_transform(&exec, massively::SoA1(input_g.slice(slice_range(&input))), TransformMap).unwrap();
        prop_assert_eq!(exec.to_host(&output_g).unwrap(), transform_map(&input));
    }

    #[test]
    fn reduce_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN), init in any::<u32>()) {
        let _guard = gpu_lock();
        let exec = api_exec();
        let input_g = padded_device(&exec, &input);
        prop_assert_eq!(api_reduce(&exec, massively::SoA1(input_g.slice(slice_range(&input))), (init,), TupleMaxOp).unwrap().0, oracle::reduce(&input, init));
    }

    #[test]
    fn inclusive_scan_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let exec = api_exec();
        let input_g = padded_device(&exec, &input);
        let (output_g,) = api_inclusive_scan(&exec, massively::SoA1(input_g.slice(slice_range(&input))), TupleMaxOp).unwrap();
        prop_assert_eq!(exec.to_host(&output_g).unwrap(), oracle::inclusive_scan(&input));
    }

    #[test]
    fn exclusive_scan_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN), init in any::<u32>()) {
        let _guard = gpu_lock();
        let exec = api_exec();
        let input_g = padded_device(&exec, &input);
        let (output_g,) = api_exclusive_scan(&exec, massively::SoA1(input_g.slice(slice_range(&input))), (init,), TupleMaxOp).unwrap();
        prop_assert_eq!(exec.to_host(&output_g).unwrap(), oracle::exclusive_scan(&input, init));
    }

    #[test]
    fn adjacent_difference_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let exec = api_exec();
        let input_g = padded_device(&exec, &input);
        let (output_g,) = api_adjacent_difference(&exec, massively::SoA1(input_g.slice(slice_range(&input))), TupleMaxOp).unwrap();
        prop_assert_eq!(exec.to_host(&output_g).unwrap(), oracle::adjacent_difference(&input));
    }

    #[test]
    fn copy_if_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let exec = api_exec();
        let stencil = stencil_flags(&input);
        let input_g = padded_device(&exec, &input);
        let stencil_g = padded_device(&exec, &stencil);
        let (output_g,) = api_copy_if(&exec, massively::SoA1(input_g.slice(slice_range(&input))), stencil_g.slice(slice_range(&stencil))).unwrap();
        prop_assert_eq!(exec.to_host(&output_g).unwrap(), oracle::copy_if(&input, &stencil));
    }

    #[test]
    fn remove_if_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let exec = api_exec();
        let input_g = padded_device(&exec, &input);
        let (output_g,) = api_remove_if(&exec, massively::SoA1(input_g.slice(slice_range(&input))), TupleKeep).unwrap();
        prop_assert_eq!(exec.to_host(&output_g).unwrap(), oracle::remove_if(&input));
    }

    #[test]
    fn partition_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let exec = api_exec();
        let input_g = padded_device(&exec, &input);
        let ((matching_g,), (failing_g,)) = api_partition(&exec, massively::SoA1(input_g.slice(slice_range(&input))), TupleKeep).unwrap();
        let (matching, failing) = oracle::partition(&input);
        prop_assert_eq!(exec.to_host(&matching_g).unwrap(), matching);
        prop_assert_eq!(exec.to_host(&failing_g).unwrap(), failing);
    }

    #[test]
    fn count_if_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let exec = api_exec();
        let input_g = padded_device(&exec, &input);
        prop_assert_eq!(api_count_if(&exec, massively::SoA1(input_g.slice(slice_range(&input))), TupleKeep).unwrap(), oracle::count_if(&input));
    }

    #[test]
    fn all_of_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let exec = api_exec();
        let input_g = padded_device(&exec, &input);
        prop_assert_eq!(api_all_of(&exec, massively::SoA1(input_g.slice(slice_range(&input))), TupleKeep).unwrap(), oracle::all_of(&input));
    }

    #[test]
    fn any_of_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let exec = api_exec();
        let input_g = padded_device(&exec, &input);
        prop_assert_eq!(api_any_of(&exec, massively::SoA1(input_g.slice(slice_range(&input))), TupleKeep).unwrap(), oracle::any_of(&input));
    }

    #[test]
    fn none_of_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let exec = api_exec();
        let input_g = padded_device(&exec, &input);
        prop_assert_eq!(api_none_of(&exec, massively::SoA1(input_g.slice(slice_range(&input))), TupleKeep).unwrap(), oracle::none_of(&input));
    }

    #[test]
    fn find_if_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let exec = api_exec();
        let input_g = padded_device(&exec, &input);
        prop_assert_eq!(api_find_if(&exec, massively::SoA1(input_g.slice(slice_range(&input))), TupleKeep).unwrap(), oracle::find_if(&input));
    }

    #[test]
    fn is_partitioned_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let exec = api_exec();
        let input_g = padded_device(&exec, &input);
        prop_assert_eq!(api_is_partitioned(&exec, massively::SoA1(input_g.slice(slice_range(&input))), TupleKeep).unwrap(), oracle::is_partitioned(&input));
    }

    #[test]
    fn replace_if_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN), replacement in any::<u32>()) {
        let _guard = gpu_lock();
        let exec = api_exec();
        let stencil = stencil_flags(&input);
        let input_g = padded_device(&exec, &input);
        let stencil_g = padded_device(&exec, &stencil);
        let (output_g,) = api_replace_if(&exec, massively::SoA1(input_g.slice(slice_range(&input))), (replacement,), stencil_g.slice(slice_range(&stencil))).unwrap();
        prop_assert_eq!(exec.to_host(&output_g).unwrap(), oracle::replace_if(&input, replacement, &stencil));
    }

    #[test]
    fn unique_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let exec = api_exec();
        let input_g = padded_device(&exec, &input);
        let (output_g,) = api_unique(&exec, massively::SoA1(input_g.slice(slice_range(&input))), TupleSameLowNibble).unwrap();
        prop_assert_eq!(exec.to_host(&output_g).unwrap(), oracle::unique(&input));
    }

    #[test]
    fn reverse_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let exec = api_exec();
        let input_g = padded_device(&exec, &input);
        let (output_g,) = api_reverse(&exec, massively::SoA1(input_g.slice(slice_range(&input)))).unwrap();
        prop_assert_eq!(exec.to_host(&output_g).unwrap(), oracle::reverse(&input));
    }

    #[test]
    fn gather_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let indices = reverse_indices(input.len());
        let exec = api_exec();
        let input_g = padded_device(&exec, &input);
        let indices_g = padded_device(&exec, &indices);
        let (output_g,) = api_gather(&exec, massively::SoA1(input_g.slice(slice_range(&input))), indices_g.slice(slice_range(&indices))).unwrap();
        prop_assert_eq!(exec.to_host(&output_g).unwrap(), oracle::gather(&input, &indices));
    }

    #[test]
    fn gather_if_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let indices = reverse_indices(input.len());
        let stencil = oracle::gather(&stencil_flags(&input), &indices);
        let exec = exec();
        let input_g = padded_device(&exec, &input);
        let indices_g = padded_device(&exec, &indices);
        let stencil_g = padded_device(&exec, &stencil);
        prop_assert_eq!(
            {
                let (output_g,) =
                    gather_if(&exec, massively::SoA1(input_g.slice(slice_range(&input))), indices_g.slice(slice_range(&indices)), (0_u32,), stencil_g.slice(slice_range(&stencil)))
                        .unwrap();
                exec.to_host(&output_g).unwrap()
            },
            oracle::gather_if(&input, &indices, &stencil)
        );
    }

    #[test]
    fn scatter_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let indices = reverse_indices(input.len());
        let default = 0xdead_beef;
        let exec = exec();
        let input_g = padded_device(&exec, &input);
        let indices_g = padded_device(&exec, &indices);
        let (output_g,) = scatter(&exec, massively::SoA1(input_g.slice(slice_range(&input))), indices_g.slice(slice_range(&indices)), input.len(), (default,)).unwrap();
        prop_assert_eq!(exec.to_host(&output_g).unwrap(), oracle::scatter(&input, &indices, input.len(), default));
    }

    #[test]
    fn scatter_if_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let indices = reverse_indices(input.len());
        let default = 0xdead_beef;
        let stencil = stencil_flags(&input);
        let exec = exec();
        let input_g = padded_device(&exec, &input);
        let indices_g = padded_device(&exec, &indices);
        let stencil_g = padded_device(&exec, &stencil);
        prop_assert_eq!(
            {
                let (output_g,) = scatter_if(&exec, massively::SoA1(input_g.slice(slice_range(&input))), indices_g.slice(slice_range(&indices)), input.len(), (default,), stencil_g.slice(slice_range(&stencil))).unwrap();
                exec.to_host(&output_g).unwrap()
            },
            oracle::scatter_if(&input, &indices, input.len(), default, &stencil)
        );
    }

    #[test]
    fn sort_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let exec = api_exec();
        let input_g = padded_device(&exec, &input);
        let (output_g,) = api_sort(&exec, massively::SoA1(input_g.slice(slice_range(&input))), TupleBucketThenValueLess).unwrap();
        prop_assert_eq!(exec.to_host(&output_g).unwrap(), oracle::sort(&input));
    }

    #[test]
    fn stable_sort_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let exec = exec();
        let input_g = padded_device(&exec, &input);
        let (output_g,) = stable_sort(&exec, massively::SoA1(input_g.slice(slice_range(&input))), TupleBucketThenValueLess).unwrap();
        prop_assert_eq!(exec.to_host(&output_g).unwrap(), oracle::sort(&input));
    }

    #[test]
    fn lower_bound_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN), value in any::<u32>()) {
        let _guard = gpu_lock();
        let sorted = oracle::sort(&input);
        let exec = exec();
        let sorted_g = padded_device(&exec, &sorted);
        prop_assert_eq!(lower_bound(&exec, massively::SoA1(sorted_g.slice(slice_range(&sorted))), (value,), TupleBucketThenValueLess).unwrap(), oracle::lower_bound(&sorted, value));
    }

    #[test]
    fn upper_bound_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN), value in any::<u32>()) {
        let _guard = gpu_lock();
        let sorted = oracle::sort(&input);
        let exec = exec();
        let sorted_g = padded_device(&exec, &sorted);
        prop_assert_eq!(upper_bound(&exec, massively::SoA1(sorted_g.slice(slice_range(&sorted))), (value,), TupleBucketThenValueLess).unwrap(), oracle::upper_bound(&sorted, value));
    }

    #[test]
    fn equal_range_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN), value in any::<u32>()) {
        let _guard = gpu_lock();
        let sorted = oracle::sort(&input);
        let exec = exec();
        let sorted_g = padded_device(&exec, &sorted);
        prop_assert_eq!(equal_range(&exec, massively::SoA1(sorted_g.slice(slice_range(&sorted))), (value,), TupleBucketThenValueLess).unwrap(), oracle::equal_range(&sorted, value));
    }

    #[test]
    fn is_sorted_until_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let sorted = oracle::sort(&input);
        let exec = exec();
        let sorted_g = padded_device(&exec, &sorted);
        prop_assert_eq!(is_sorted_until(&exec, massively::SoA1(sorted_g.slice(slice_range(&sorted))), TupleBucketThenValueLess).unwrap(), oracle::is_sorted_until(&sorted));
    }

    #[test]
    fn is_sorted_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let sorted = oracle::sort(&input);
        let exec = exec();
        let sorted_g = padded_device(&exec, &sorted);
        prop_assert_eq!(is_sorted(&exec, massively::SoA1(sorted_g.slice(slice_range(&sorted))), TupleBucketThenValueLess).unwrap(), oracle::is_sorted(&sorted));
    }

    #[test]
    fn merge_matches_oracle(left in prop::collection::vec(any::<u32>(), 0..MAX_LEN), right in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let left = oracle::sort(&left);
        let right = oracle::sort(&right);
        let exec = exec();
        let left_g = padded_device(&exec, &left);
        let right_g = padded_device(&exec, &right);
        let (output_g,) = merge(&exec, massively::SoA1(left_g.slice(slice_range(&left))), massively::SoA1(right_g.slice(slice_range(&right))), TupleBucketThenValueLess).unwrap();
        prop_assert_eq!(exec.to_host(&output_g).unwrap(), oracle::merge(&left, &right));
    }

    #[test]
    fn set_union_matches_oracle(left in prop::collection::vec(any::<u32>(), 0..MAX_LEN), right in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let left = oracle::sort(&left);
        let right = oracle::sort(&right);
        let exec = exec();
        let left_g = padded_device(&exec, &left);
        let right_g = padded_device(&exec, &right);
        let (output_g,) = set_union(&exec, massively::SoA1(left_g.slice(slice_range(&left))), massively::SoA1(right_g.slice(slice_range(&right))), TupleBucketThenValueLess).unwrap();
        prop_assert_eq!(exec.to_host(&output_g).unwrap(), oracle::set_union(&left, &right));
    }

    #[test]
    fn set_intersection_matches_oracle(left in prop::collection::vec(any::<u32>(), 0..MAX_LEN), right in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let left = oracle::sort(&left);
        let right = oracle::sort(&right);
        let exec = exec();
        let left_g = padded_device(&exec, &left);
        let right_g = padded_device(&exec, &right);
        let (output_g,) = set_intersection(&exec, massively::SoA1(left_g.slice(slice_range(&left))), massively::SoA1(right_g.slice(slice_range(&right))), TupleBucketThenValueLess).unwrap();
        prop_assert_eq!(exec.to_host(&output_g).unwrap(), oracle::set_intersection(&left, &right));
    }

    #[test]
    fn set_difference_matches_oracle(left in prop::collection::vec(any::<u32>(), 0..MAX_LEN), right in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let left = oracle::sort(&left);
        let right = oracle::sort(&right);
        let exec = exec();
        let left_g = padded_device(&exec, &left);
        let right_g = padded_device(&exec, &right);
        let (output_g,) = set_difference(&exec, massively::SoA1(left_g.slice(slice_range(&left))), massively::SoA1(right_g.slice(slice_range(&right))), TupleBucketThenValueLess).unwrap();
        prop_assert_eq!(exec.to_host(&output_g).unwrap(), oracle::set_difference(&left, &right));
    }

    #[test]
    fn adjacent_find_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let exec = exec();
        let input_g = padded_device(&exec, &input);
        prop_assert_eq!(adjacent_find(&exec, massively::SoA1(input_g.slice(slice_range(&input))), TupleSameLowNibble).unwrap(), oracle::adjacent_find(&input));
    }

    #[test]
    fn equal_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let right = oracle::transform(&input);
        let exec = exec();
        let input_g = padded_device(&exec, &input);
        let right_g = padded_device(&exec, &right);
        prop_assert_eq!(equal(&exec, massively::SoA1(input_g.slice(slice_range(&input))), massively::SoA1(right_g.slice(slice_range(&right))), TupleSameLowNibble).unwrap(), oracle::equal(&input, &right));
    }

    #[test]
    fn mismatch_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let right = oracle::transform(&input);
        let exec = exec();
        let input_g = padded_device(&exec, &input);
        let right_g = padded_device(&exec, &right);
        prop_assert_eq!(mismatch(&exec, massively::SoA1(input_g.slice(slice_range(&input))), massively::SoA1(right_g.slice(slice_range(&right))), TupleSameLowNibble).unwrap(), oracle::mismatch(&input, &right));
    }

    #[test]
    fn find_first_of_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let needles = if input.is_empty() {
            Vec::new()
        } else {
            vec![input[0], input[input.len() - 1]]
        };
        let exec = exec();
        let input_g = padded_device(&exec, &input);
        let needles_g = padded_device(&exec, &needles);
        prop_assert_eq!(find_first_of(&exec, massively::SoA1(input_g.slice(slice_range(&input))), massively::SoA1(needles_g.slice(slice_range(&needles))), TupleSameLowNibble).unwrap(), oracle::find_first_of(&input, &needles));
    }

    #[test]
    fn min_element_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let exec = exec();
        let input_g = padded_device(&exec, &input);
        prop_assert_eq!(min_element(&exec, massively::SoA1(input_g.slice(slice_range(&input))), TupleBucketThenValueLess).unwrap(), oracle::min_element(&input));
    }

    #[test]
    fn max_element_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let exec = exec();
        let input_g = padded_device(&exec, &input);
        prop_assert_eq!(max_element(&exec, massively::SoA1(input_g.slice(slice_range(&input))), TupleBucketThenValueLess).unwrap(), oracle::max_element(&input));
    }

    #[test]
    fn minmax_element_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let exec = exec();
        let input_g = padded_device(&exec, &input);
        prop_assert_eq!(minmax_element(&exec, massively::SoA1(input_g.slice(slice_range(&input))), TupleBucketThenValueLess).unwrap(), oracle::minmax_element(&input));
    }

    #[test]
    fn lexicographical_compare_matches_oracle(left in prop::collection::vec(any::<u32>(), 0..MAX_LEN), right in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let exec = exec();
        let left_g = padded_device(&exec, &left);
        let right_g = padded_device(&exec, &right);
        prop_assert_eq!(lexicographical_compare(&exec, massively::SoA1(left_g.slice(slice_range(&left))), massively::SoA1(right_g.slice(slice_range(&right))), TupleBucketThenValueLess).unwrap(), oracle::lexicographical_compare(&left, &right));
    }

    #[test]
    fn inner_product_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN), init in any::<u32>()) {
        let _guard = gpu_lock();
        let right = oracle::transform(&input);
        let exec = exec();
        let left_g = padded_device(&exec, &input);
        let right_g = padded_device(&exec, &right);
        prop_assert_eq!(
            inner_product(&exec,
                massively::SoA1(left_g.slice(slice_range(&input))),
                massively::SoA1(right_g.slice(slice_range(&right))),
                TuplePairMax,
                (init,),
                TupleMaxOp
            )
            .unwrap(),
            (oracle::inner_product(&input, &right, init),)
        );
    }

    #[test]
    fn inclusive_scan_by_key_matches_oracle(values in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let keys = run_keys(values.len());
        let exec = exec();
        let keys_g = padded_device(&exec, &keys);
        let values_g = padded_device(&exec, &values);
        let (output_g,) = inclusive_scan_by_key(&exec, massively::SoA1(keys_g.slice(slice_range(&keys))), massively::SoA1(values_g.slice(slice_range(&values))), TupleSameLowNibble, TupleMaxOp).unwrap();
        prop_assert_eq!(exec.to_host(&output_g).unwrap(), oracle::inclusive_scan_by_key(&keys, &values));
    }

    #[test]
    fn exclusive_scan_by_key_matches_oracle(values in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let keys = run_keys(values.len());
        let init = 0;
        let exec = exec();
        let keys_g = padded_device(&exec, &keys);
        let values_g = padded_device(&exec, &values);
        let (output_g,) = exclusive_scan_by_key(&exec, massively::SoA1(keys_g.slice(slice_range(&keys))), massively::SoA1(values_g.slice(slice_range(&values))), TupleSameLowNibble, (init,), TupleMaxOp).unwrap();
        prop_assert_eq!(exec.to_host(&output_g).unwrap(), oracle::exclusive_scan_by_key(&keys, &values, init));
    }

    #[test]
    fn reduce_by_key_matches_oracle(values in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let keys = run_keys(values.len());
        let init = 0;
        let exec = exec();
        let keys_g = padded_device(&exec, &keys);
        let values_g = padded_device(&exec, &values);
        let (expected_keys, expected_values) = oracle::reduce_by_key(&keys, &values, init);
        let ((actual_keys,), (actual_values,)) =
            reduce_by_key(&exec, massively::SoA1(keys_g.slice(slice_range(&keys))), massively::SoA1(values_g.slice(slice_range(&values))), TupleSameLowNibble, (init,), TupleMaxOp).unwrap();
        prop_assert_eq!(exec.to_host(&actual_keys).unwrap(), expected_keys);
        prop_assert_eq!(exec.to_host(&actual_values).unwrap(), expected_values);
    }

    #[test]
    fn unique_by_key_matches_oracle(values in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let keys = run_keys(values.len());
        let exec = exec();
        let keys_g = padded_device(&exec, &keys);
        let values_g = padded_device(&exec, &values);
        let (expected_keys, expected_values) = oracle::unique_by_key(&keys, &values);
        let ((actual_keys,), (actual_values,)) =
            unique_by_key(&exec, massively::SoA1(keys_g.slice(slice_range(&keys))), massively::SoA1(values_g.slice(slice_range(&values))), TupleSameLowNibble).unwrap();
        prop_assert_eq!(exec.to_host(&actual_keys).unwrap(), expected_keys);
        prop_assert_eq!(exec.to_host(&actual_values).unwrap(), expected_values);
    }

    #[test]
    fn sort_by_key_matches_oracle(values in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let keys = unique_keys(values.len());
        let exec = exec();
        let keys_g = padded_device(&exec, &keys);
        let values_g = padded_device(&exec, &values);
        let (expected_keys, expected_values) = oracle::sort_by_key(&keys, &values);
        let ((actual_keys,), (actual_values,)) =
            sort_by_key(&exec, massively::SoA1(keys_g.slice(slice_range(&keys))), massively::SoA1(values_g.slice(slice_range(&values))), TupleBucketThenValueLess).unwrap();
        prop_assert_eq!(exec.to_host(&actual_keys).unwrap(), expected_keys);
        prop_assert_eq!(exec.to_host(&actual_values).unwrap(), expected_values);
    }

    #[test]
    fn stable_sort_by_key_matches_oracle(values in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let keys = unique_keys(values.len());
        let exec = exec();
        let keys_g = padded_device(&exec, &keys);
        let values_g = padded_device(&exec, &values);
        let (expected_keys, expected_values) = oracle::sort_by_key(&keys, &values);
        let ((actual_keys,), (actual_values,)) =
            stable_sort_by_key(&exec, massively::SoA1(keys_g.slice(slice_range(&keys))), massively::SoA1(values_g.slice(slice_range(&values))), TupleBucketThenValueLess).unwrap();
        prop_assert_eq!(exec.to_host(&actual_keys).unwrap(), expected_keys);
        prop_assert_eq!(exec.to_host(&actual_values).unwrap(), expected_values);
    }

    #[test]
    fn merge_by_key_matches_oracle(values in prop::collection::vec(any::<u32>(), 2..MAX_LEN)) {
        let _guard = gpu_lock();
        let keys = unique_keys(values.len());
        let (keys, values) = oracle::sort_by_key(&keys, &values);
        let mid = keys.len() / 2;
        let left_keys = keys[..mid].to_vec();
        let left_values = values[..mid].to_vec();
        let right_keys = keys[mid..].to_vec();
        let right_values = values[mid..].to_vec();
        let exec = exec();
        let left_keys_g = padded_device(&exec, &left_keys);
        let left_values_g = padded_device(&exec, &left_values);
        let right_keys_g = padded_device(&exec, &right_keys);
        let right_values_g = padded_device(&exec, &right_values);
        let (expected_keys, expected_values) = oracle::merge_by_key(&left_keys, &left_values, &right_keys, &right_values);
        let ((actual_keys,), (actual_values,)) = merge_by_key(&exec,
            massively::SoA1(left_keys_g.slice(slice_range(&left_keys))),
            massively::SoA1(left_values_g.slice(slice_range(&left_values))),
            massively::SoA1(right_keys_g.slice(slice_range(&right_keys))),
            massively::SoA1(right_values_g.slice(slice_range(&right_values))),
            TupleBucketThenValueLess,
        )
        .unwrap();
        prop_assert_eq!(exec.to_host(&actual_keys).unwrap(), expected_keys);
        prop_assert_eq!(exec.to_host(&actual_values).unwrap(), expected_values);
    }
}
