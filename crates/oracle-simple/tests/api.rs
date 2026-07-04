use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::op::{BinaryPredicateOp, PredicateOp, ReductionOp, UnaryOp};
use massively::{
    DeviceVec, Executor as ApiExecutor, MIndex, adjacent_difference as api_adjacent_difference,
    adjacent_find, all_of as api_all_of, any_of as api_any_of, copy_where as api_copy_where,
    count_if as api_count_if, equal, exclusive_scan as api_exclusive_scan, exclusive_scan_by_key,
    fill as api_fill, find_first_of, find_if as api_find_if, gather as api_gather, gather_where,
    inclusive_scan as api_inclusive_scan, inclusive_scan_by_key,
    is_partitioned as api_is_partitioned, is_sorted, is_sorted_until, lexicographical_compare,
    lower_bound, max_element, merge, merge_by_key, min_element, minmax_element, mismatch,
    none_of as api_none_of, partition as api_partition, reduce as api_reduce, reduce_by_key,
    remove_where as api_remove_where, replace_where as api_replace_where, reverse as api_reverse,
    scatter, scatter_where, set_difference, set_intersection, set_union, sort as api_sort,
    sort_by_key, stable_sort, stable_sort_by_key, transform as api_transform, unique as api_unique,
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
    type Env = ();
    type Output = (u32,);

    fn apply(_env: (), input: (u32,)) -> (u32,) {
        (input.0 / 3 + 17,)
    }
}

struct TupleMaxOp;

#[cubecl::cube]
impl ReductionOp<ApiRuntime, (u32,)> for TupleMaxOp {
    fn apply(lhs: (u32,), rhs: (u32,)) -> (u32,) {
        (lhs.0.max(rhs.0),)
    }
}

struct TupleKeep;

#[cubecl::cube]
impl PredicateOp<ApiRuntime, (u32,)> for TupleKeep {
    type Env = ();

    fn apply(_env: (), input: (u32,)) -> bool {
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

fn mindex(value: usize) -> MIndex {
    value.try_into().unwrap()
}

fn opt_mindex(value: Option<usize>) -> Option<MIndex> {
    value.map(mindex)
}

fn opt_pair_mindex(value: Option<(usize, usize)>) -> Option<(MIndex, MIndex)> {
    value.map(|(left, right)| (mindex(left), mindex(right)))
}

fn slice_range(input: &[u32]) -> std::ops::Range<MIndex> {
    1..mindex(input.len() + 1)
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
        .map(|value| if oracle_simple::keep(*value) { 1 } else { 0 })
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(CASES))]

    #[test]
    fn transform_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let exec = api_exec();
        let input_g = padded_device(&exec, &input);
        let output_g = exec.to_device(&vec![0_u32; input.len()]).unwrap();
        api_transform(&exec, massively::SoA1(input_g.slice(slice_range(&input))), TransformMap, (), massively::SoA1(output_g.slice_mut(..))).unwrap();
        prop_assert_eq!(exec.to_host(&output_g).unwrap(), transform_map(&input));
    }

    #[test]
    fn reduce_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN), init in any::<u32>()) {
        let _guard = gpu_lock();
        let exec = api_exec();
        let input_g = padded_device(&exec, &input);
        prop_assert_eq!(api_reduce(&exec, massively::SoA1(input_g.slice(slice_range(&input))), (init,), TupleMaxOp).unwrap().0, oracle_simple::reduce(&input, init));
    }

    #[test]
    fn inclusive_scan_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let exec = api_exec();
        let input_g = padded_device(&exec, &input);
        let output_g = exec.to_device(&vec![0_u32; input.len()]).unwrap();
        api_inclusive_scan(&exec, massively::SoA1(input_g.slice(slice_range(&input))), TupleMaxOp, massively::SoA1(output_g.slice_mut(..))).unwrap();
        prop_assert_eq!(exec.to_host(&output_g).unwrap(), oracle_simple::inclusive_scan(&input));
    }

    #[test]
    fn exclusive_scan_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN), init in any::<u32>()) {
        let _guard = gpu_lock();
        let exec = api_exec();
        let input_g = padded_device(&exec, &input);
        let output_g = exec.to_device(&vec![0_u32; input.len()]).unwrap();
        api_exclusive_scan(&exec, massively::SoA1(input_g.slice(slice_range(&input))), (init,), TupleMaxOp, massively::SoA1(output_g.slice_mut(..))).unwrap();
        prop_assert_eq!(exec.to_host(&output_g).unwrap(), oracle_simple::exclusive_scan(&input, init));
    }

    #[test]
    fn adjacent_difference_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let exec = api_exec();
        let input_g = padded_device(&exec, &input);
        let output_g = exec.to_device(&vec![0_u32; input.len()]).unwrap();
        api_adjacent_difference(&exec, massively::SoA1(input_g.slice(slice_range(&input))), TupleMaxOp, massively::SoA1(output_g.slice_mut(..))).unwrap();
        prop_assert_eq!(exec.to_host(&output_g).unwrap(), oracle_simple::adjacent_difference(&input));
    }

    #[test]
    fn copy_where_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let exec = api_exec();
        let stencil = stencil_flags(&input);
        let input_g = padded_device(&exec, &input);
        let stencil_g = padded_device(&exec, &stencil);
        let output_g = exec.to_device(&vec![0_u32; input.len()]).unwrap();
        let len = api_copy_where(&exec, massively::SoA1(input_g.slice(slice_range(&input))), stencil_g.slice(slice_range(&stencil)), massively::SoA1(output_g.slice_mut(..))).unwrap();
        prop_assert_eq!(exec.to_host(&output_g.slice(..len)).unwrap(), oracle_simple::copy_where(&input, &stencil));
    }

    #[test]
    fn remove_where_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let exec = api_exec();
        let stencil = stencil_flags(&input);
        let input_g = padded_device(&exec, &input);
        let stencil_g = padded_device(&exec, &stencil);
        let output_g = exec.to_device(&vec![0_u32; input.len()]).unwrap();
        let len = api_remove_where(&exec, massively::SoA1(input_g.slice(slice_range(&input))), stencil_g.slice(slice_range(&stencil)), massively::SoA1(output_g.slice_mut(..))).unwrap();
        prop_assert_eq!(exec.to_host(&output_g.slice(..len)).unwrap(), oracle_simple::remove_where(&input, &stencil));
    }

    #[test]
    fn partition_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let exec = api_exec();
        let input_g = padded_device(&exec, &input);
        let output_g = exec.to_device(&vec![0_u32; input.len()]).unwrap();
        let split = api_partition(&exec, massively::SoA1(input_g.slice(slice_range(&input))), TupleKeep, (), massively::SoA1(output_g.slice_mut(..))).unwrap();
        let (matching, failing) = oracle_simple::partition(&input);
        prop_assert_eq!(exec.to_host(&output_g.slice(..split)).unwrap(), matching);
        prop_assert_eq!(exec.to_host(&output_g.slice(split..mindex(input.len()))).unwrap(), failing);
    }

    #[test]
    fn count_if_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let exec = api_exec();
        let input_g = padded_device(&exec, &input);
        prop_assert_eq!(api_count_if(&exec, massively::SoA1(input_g.slice(slice_range(&input))), TupleKeep, ()).unwrap(), mindex(oracle_simple::count_if(&input)));
    }

    #[test]
    fn all_of_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let exec = api_exec();
        let input_g = padded_device(&exec, &input);
        prop_assert_eq!(api_all_of(&exec, massively::SoA1(input_g.slice(slice_range(&input))), TupleKeep, ()).unwrap(), oracle_simple::all_of(&input));
    }

    #[test]
    fn any_of_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let exec = api_exec();
        let input_g = padded_device(&exec, &input);
        prop_assert_eq!(api_any_of(&exec, massively::SoA1(input_g.slice(slice_range(&input))), TupleKeep, ()).unwrap(), oracle_simple::any_of(&input));
    }

    #[test]
    fn none_of_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let exec = api_exec();
        let input_g = padded_device(&exec, &input);
        prop_assert_eq!(api_none_of(&exec, massively::SoA1(input_g.slice(slice_range(&input))), TupleKeep, ()).unwrap(), oracle_simple::none_of(&input));
    }

    #[test]
    fn find_if_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let exec = api_exec();
        let input_g = padded_device(&exec, &input);
        prop_assert_eq!(api_find_if(&exec, massively::SoA1(input_g.slice(slice_range(&input))), TupleKeep, ()).unwrap(), opt_mindex(oracle_simple::find_if(&input)));
    }

    #[test]
    fn is_partitioned_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let exec = api_exec();
        let input_g = padded_device(&exec, &input);
        prop_assert_eq!(api_is_partitioned(&exec, massively::SoA1(input_g.slice(slice_range(&input))), TupleKeep, ()).unwrap(), oracle_simple::is_partitioned(&input));
    }

    #[test]
    fn replace_where_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN), replacement in any::<u32>()) {
        let _guard = gpu_lock();
        let exec = api_exec();
        let stencil = stencil_flags(&input);
        let output_g = padded_device(&exec, &input);
        let stencil_g = padded_device(&exec, &stencil);
        api_replace_where(&exec, (replacement,), stencil_g.slice(slice_range(&stencil)), massively::SoA1(output_g.slice_mut(slice_range(&input)))).unwrap();
        prop_assert_eq!(exec.to_host(&output_g.slice(slice_range(&input))).unwrap(), oracle_simple::replace_where(&input, replacement, &stencil));
    }

    #[test]
    fn fill_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN), replacement in any::<u32>()) {
        let _guard = gpu_lock();
        let exec = api_exec();
        let output_g = padded_device(&exec, &input);
        let mut expected = input.clone();
        oracle_simple::fill(replacement, &mut expected);
        api_fill(&exec, (replacement,), massively::SoA1(output_g.slice_mut(slice_range(&input)))).unwrap();
        prop_assert_eq!(exec.to_host(&output_g.slice(slice_range(&input))).unwrap(), expected);
    }

    #[test]
    fn unique_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let exec = api_exec();
        let input_g = padded_device(&exec, &input);
        let output_g = exec.to_device(&vec![0_u32; input.len()]).unwrap();
        let len = api_unique(&exec, massively::SoA1(input_g.slice(slice_range(&input))), TupleSameLowNibble, massively::SoA1(output_g.slice_mut(..))).unwrap();
        prop_assert_eq!(exec.to_host(&output_g.slice(..len)).unwrap(), oracle_simple::unique(&input));
    }

    #[test]
    fn reverse_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let exec = api_exec();
        let input_g = padded_device(&exec, &input);
        let output_g = exec.to_device(&vec![0_u32; input.len()]).unwrap();
        api_reverse(&exec, massively::SoA1(input_g.slice(slice_range(&input))), massively::SoA1(output_g.slice_mut(..))).unwrap();
        prop_assert_eq!(exec.to_host(&output_g).unwrap(), oracle_simple::reverse(&input));
    }

    #[test]
    fn gather_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let indices = reverse_indices(input.len());
        let exec = api_exec();
        let input_g = padded_device(&exec, &input);
        let indices_g = padded_device(&exec, &indices);
        let output_g = exec.to_device(&vec![0_u32; indices.len()]).unwrap();
        api_gather(&exec, massively::SoA1(input_g.slice(slice_range(&input))), indices_g.slice(slice_range(&indices)), massively::SoA1(output_g.slice_mut(..))).unwrap();
        prop_assert_eq!(exec.to_host(&output_g).unwrap(), oracle_simple::gather(&input, &indices));
    }

    #[test]
    fn gather_where_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let indices = reverse_indices(input.len());
        let stencil = oracle_simple::gather(&stencil_flags(&input), &indices);
        let exec = exec();
        let input_g = padded_device(&exec, &input);
        let indices_g = padded_device(&exec, &indices);
        let stencil_g = padded_device(&exec, &stencil);
        prop_assert_eq!(
            {
                let output_g = exec.to_device(&vec![0_u32; indices.len()]).unwrap();
                gather_where(&exec, massively::SoA1(input_g.slice(slice_range(&input))), indices_g.slice(slice_range(&indices)), stencil_g.slice(slice_range(&stencil)), massively::SoA1(output_g.slice_mut(..)))
                    .unwrap();
                exec.to_host(&output_g).unwrap()
            },
            oracle_simple::gather_where(&input, &indices, &stencil)
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
        let output_g = exec.to_device(&vec![default; input.len()]).unwrap();
        scatter(&exec, massively::SoA1(input_g.slice(slice_range(&input))), indices_g.slice(slice_range(&indices)), massively::SoA1(output_g.slice_mut(..))).unwrap();
        prop_assert_eq!(exec.to_host(&output_g).unwrap(), oracle_simple::scatter(&input, &indices, input.len(), default));
    }

    #[test]
    fn scatter_where_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
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
                let output_g = exec.to_device(&vec![default; input.len()]).unwrap();
                scatter_where(&exec, massively::SoA1(input_g.slice(slice_range(&input))), indices_g.slice(slice_range(&indices)), stencil_g.slice(slice_range(&stencil)), massively::SoA1(output_g.slice_mut(..))).unwrap();
                exec.to_host(&output_g).unwrap()
            },
            oracle_simple::scatter_where(&input, &indices, input.len(), default, &stencil)
        );
    }

    #[test]
    fn sort_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let exec = api_exec();
        let input_g = padded_device(&exec, &input);
        let output_g = exec.to_device(&vec![0_u32; input.len()]).unwrap();
        api_sort(&exec, massively::SoA1(input_g.slice(slice_range(&input))), TupleBucketThenValueLess, massively::SoA1(output_g.slice_mut(..))).unwrap();
        prop_assert_eq!(exec.to_host(&output_g).unwrap(), oracle_simple::sort(&input));
    }

    #[test]
    fn stable_sort_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let exec = exec();
        let input_g = padded_device(&exec, &input);
        let output_g = exec.to_device(&vec![0_u32; input.len()]).unwrap();
        stable_sort(&exec, massively::SoA1(input_g.slice(slice_range(&input))), TupleBucketThenValueLess, massively::SoA1(output_g.slice_mut(..))).unwrap();
        prop_assert_eq!(exec.to_host(&output_g).unwrap(), oracle_simple::sort(&input));
    }

    #[test]
    fn lower_bound_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN), values in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let sorted = oracle_simple::sort(&input);
        let exec = exec();
        let sorted_g = padded_device(&exec, &sorted);
        let values_g = padded_device(&exec, &values);
        let output = exec.to_device(&vec![0_u32; values.len()]).unwrap();
        lower_bound(
            &exec,
            massively::SoA1(sorted_g.slice(slice_range(&sorted))),
            massively::SoA1(values_g.slice(slice_range(&values))),
            TupleBucketThenValueLess,
            output.slice_mut(..),
        ).unwrap();
        prop_assert_eq!(exec.to_host(&output).unwrap(), oracle_simple::lower_bound(&sorted, &values));
    }

    #[test]
    fn upper_bound_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN), values in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let sorted = oracle_simple::sort(&input);
        let exec = exec();
        let sorted_g = padded_device(&exec, &sorted);
        let values_g = padded_device(&exec, &values);
        let output = exec.to_device(&vec![0_u32; values.len()]).unwrap();
        upper_bound(
            &exec,
            massively::SoA1(sorted_g.slice(slice_range(&sorted))),
            massively::SoA1(values_g.slice(slice_range(&values))),
            TupleBucketThenValueLess,
            output.slice_mut(..),
        ).unwrap();
        prop_assert_eq!(exec.to_host(&output).unwrap(), oracle_simple::upper_bound(&sorted, &values));
    }

    #[test]
    fn is_sorted_until_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let sorted = oracle_simple::sort(&input);
        let exec = exec();
        let sorted_g = padded_device(&exec, &sorted);
        prop_assert_eq!(is_sorted_until(&exec, massively::SoA1(sorted_g.slice(slice_range(&sorted))), TupleBucketThenValueLess).unwrap(), mindex(oracle_simple::is_sorted_until(&sorted)));
    }

    #[test]
    fn is_sorted_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let sorted = oracle_simple::sort(&input);
        let exec = exec();
        let sorted_g = padded_device(&exec, &sorted);
        prop_assert_eq!(is_sorted(&exec, massively::SoA1(sorted_g.slice(slice_range(&sorted))), TupleBucketThenValueLess).unwrap(), oracle_simple::is_sorted(&sorted));
    }

    #[test]
    fn merge_matches_oracle(left in prop::collection::vec(any::<u32>(), 0..MAX_LEN), right in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let left = oracle_simple::sort(&left);
        let right = oracle_simple::sort(&right);
        let exec = exec();
        let left_g = padded_device(&exec, &left);
        let right_g = padded_device(&exec, &right);
        let output_g = exec.to_device(&vec![0_u32; left.len() + right.len()]).unwrap();
        merge(&exec, massively::SoA1(left_g.slice(slice_range(&left))), massively::SoA1(right_g.slice(slice_range(&right))), TupleBucketThenValueLess, massively::SoA1(output_g.slice_mut(..))).unwrap();
        prop_assert_eq!(exec.to_host(&output_g).unwrap(), oracle_simple::merge(&left, &right));
    }

    #[test]
    fn set_union_matches_oracle(left in prop::collection::vec(any::<u32>(), 0..MAX_LEN), right in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let left = oracle_simple::sort(&left);
        let right = oracle_simple::sort(&right);
        let exec = exec();
        let left_g = padded_device(&exec, &left);
        let right_g = padded_device(&exec, &right);
        let output_g = exec.to_device(&vec![0_u32; left.len() + right.len()]).unwrap();
        let len = set_union(&exec, massively::SoA1(left_g.slice(slice_range(&left))), massively::SoA1(right_g.slice(slice_range(&right))), TupleBucketThenValueLess, massively::SoA1(output_g.slice_mut(..))).unwrap();
        prop_assert_eq!(exec.to_host(&output_g.slice(..len)).unwrap(), oracle_simple::set_union(&left, &right));
    }

    #[test]
    fn set_intersection_matches_oracle(left in prop::collection::vec(any::<u32>(), 0..MAX_LEN), right in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let left = oracle_simple::sort(&left);
        let right = oracle_simple::sort(&right);
        let exec = exec();
        let left_g = padded_device(&exec, &left);
        let right_g = padded_device(&exec, &right);
        let output_g = exec.to_device(&vec![0_u32; left.len().min(right.len())]).unwrap();
        let len = set_intersection(&exec, massively::SoA1(left_g.slice(slice_range(&left))), massively::SoA1(right_g.slice(slice_range(&right))), TupleBucketThenValueLess, massively::SoA1(output_g.slice_mut(..))).unwrap();
        prop_assert_eq!(exec.to_host(&output_g.slice(..len)).unwrap(), oracle_simple::set_intersection(&left, &right));
    }

    #[test]
    fn set_difference_matches_oracle(left in prop::collection::vec(any::<u32>(), 0..MAX_LEN), right in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let left = oracle_simple::sort(&left);
        let right = oracle_simple::sort(&right);
        let exec = exec();
        let left_g = padded_device(&exec, &left);
        let right_g = padded_device(&exec, &right);
        let output_g = exec.to_device(&vec![0_u32; left.len()]).unwrap();
        let len = set_difference(&exec, massively::SoA1(left_g.slice(slice_range(&left))), massively::SoA1(right_g.slice(slice_range(&right))), TupleBucketThenValueLess, massively::SoA1(output_g.slice_mut(..))).unwrap();
        prop_assert_eq!(exec.to_host(&output_g.slice(..len)).unwrap(), oracle_simple::set_difference(&left, &right));
    }

    #[test]
    fn adjacent_find_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let exec = exec();
        let input_g = padded_device(&exec, &input);
        prop_assert_eq!(adjacent_find(&exec, massively::SoA1(input_g.slice(slice_range(&input))), TupleSameLowNibble).unwrap(), opt_mindex(oracle_simple::adjacent_find(&input)));
    }

    #[test]
    fn equal_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let right = oracle_simple::transform(&input);
        let exec = exec();
        let input_g = padded_device(&exec, &input);
        let right_g = padded_device(&exec, &right);
        prop_assert_eq!(equal(&exec, massively::SoA1(input_g.slice(slice_range(&input))), massively::SoA1(right_g.slice(slice_range(&right))), TupleSameLowNibble).unwrap(), oracle_simple::equal(&input, &right));
    }

    #[test]
    fn mismatch_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let right = oracle_simple::transform(&input);
        let exec = exec();
        let input_g = padded_device(&exec, &input);
        let right_g = padded_device(&exec, &right);
        prop_assert_eq!(mismatch(&exec, massively::SoA1(input_g.slice(slice_range(&input))), massively::SoA1(right_g.slice(slice_range(&right))), TupleSameLowNibble).unwrap(), opt_mindex(oracle_simple::mismatch(&input, &right)));
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
        prop_assert_eq!(find_first_of(&exec, massively::SoA1(input_g.slice(slice_range(&input))), massively::SoA1(needles_g.slice(slice_range(&needles))), TupleSameLowNibble).unwrap(), opt_mindex(oracle_simple::find_first_of(&input, &needles)));
    }

    #[test]
    fn min_element_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let exec = exec();
        let input_g = padded_device(&exec, &input);
        prop_assert_eq!(min_element(&exec, massively::SoA1(input_g.slice(slice_range(&input))), TupleBucketThenValueLess).unwrap(), opt_mindex(oracle_simple::min_element(&input)));
    }

    #[test]
    fn max_element_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let exec = exec();
        let input_g = padded_device(&exec, &input);
        prop_assert_eq!(max_element(&exec, massively::SoA1(input_g.slice(slice_range(&input))), TupleBucketThenValueLess).unwrap(), opt_mindex(oracle_simple::max_element(&input)));
    }

    #[test]
    fn minmax_element_matches_oracle(input in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let exec = exec();
        let input_g = padded_device(&exec, &input);
        prop_assert_eq!(minmax_element(&exec, massively::SoA1(input_g.slice(slice_range(&input))), TupleBucketThenValueLess).unwrap(), opt_pair_mindex(oracle_simple::minmax_element(&input)));
    }

    #[test]
    fn lexicographical_compare_matches_oracle(left in prop::collection::vec(any::<u32>(), 0..MAX_LEN), right in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let exec = exec();
        let left_g = padded_device(&exec, &left);
        let right_g = padded_device(&exec, &right);
        prop_assert_eq!(lexicographical_compare(&exec, massively::SoA1(left_g.slice(slice_range(&left))), massively::SoA1(right_g.slice(slice_range(&right))), TupleBucketThenValueLess).unwrap(), oracle_simple::lexicographical_compare(&left, &right));
    }

    #[test]
    fn inclusive_scan_by_key_matches_oracle(values in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let keys = run_keys(values.len());
        let exec = exec();
        let keys_g = padded_device(&exec, &keys);
        let values_g = padded_device(&exec, &values);
        let output_g = exec.to_device(&vec![0_u32; values.len()]).unwrap();
        inclusive_scan_by_key(&exec, massively::SoA1(keys_g.slice(slice_range(&keys))), massively::SoA1(values_g.slice(slice_range(&values))), TupleSameLowNibble, TupleMaxOp, massively::SoA1(output_g.slice_mut(..))).unwrap();
        prop_assert_eq!(exec.to_host(&output_g).unwrap(), oracle_simple::inclusive_scan_by_key(&keys, &values));
    }

    #[test]
    fn exclusive_scan_by_key_matches_oracle(values in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let keys = run_keys(values.len());
        let init = 0;
        let exec = exec();
        let keys_g = padded_device(&exec, &keys);
        let values_g = padded_device(&exec, &values);
        let output_g = exec.to_device(&vec![0_u32; values.len()]).unwrap();
        exclusive_scan_by_key(&exec, massively::SoA1(keys_g.slice(slice_range(&keys))), massively::SoA1(values_g.slice(slice_range(&values))), TupleSameLowNibble, (init,), TupleMaxOp, massively::SoA1(output_g.slice_mut(..))).unwrap();
        prop_assert_eq!(exec.to_host(&output_g).unwrap(), oracle_simple::exclusive_scan_by_key(&keys, &values, init));
    }

    #[test]
    fn reduce_by_key_matches_oracle(values in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let keys = run_keys(values.len());
        let init = 0;
        let exec = exec();
        let keys_g = padded_device(&exec, &keys);
        let values_g = padded_device(&exec, &values);
        let (expected_keys, expected_values) = oracle_simple::reduce_by_key(&keys, &values, init);
        let actual_keys = exec.to_device(&vec![0_u32; keys.len()]).unwrap();
        let actual_values = exec.to_device(&vec![0_u32; values.len()]).unwrap();
        let len = reduce_by_key(&exec, massively::SoA1(keys_g.slice(slice_range(&keys))), massively::SoA1(values_g.slice(slice_range(&values))), TupleSameLowNibble, (init,), TupleMaxOp, massively::SoA1(actual_keys.slice_mut(..)), massively::SoA1(actual_values.slice_mut(..))).unwrap();
        prop_assert_eq!(exec.to_host(&actual_keys.slice(..len)).unwrap(), expected_keys);
        prop_assert_eq!(exec.to_host(&actual_values.slice(..len)).unwrap(), expected_values);
    }

    #[test]
    fn unique_by_key_matches_oracle(values in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let keys = run_keys(values.len());
        let exec = exec();
        let keys_g = padded_device(&exec, &keys);
        let values_g = padded_device(&exec, &values);
        let (expected_keys, expected_values) = oracle_simple::unique_by_key(&keys, &values);
        let actual_keys = exec.to_device(&vec![0_u32; keys.len()]).unwrap();
        let actual_values = exec.to_device(&vec![0_u32; values.len()]).unwrap();
        let len = unique_by_key(&exec, massively::SoA1(keys_g.slice(slice_range(&keys))), massively::SoA1(values_g.slice(slice_range(&values))), TupleSameLowNibble, massively::SoA1(actual_keys.slice_mut(..)), massively::SoA1(actual_values.slice_mut(..))).unwrap();
        prop_assert_eq!(exec.to_host(&actual_keys.slice(..len)).unwrap(), expected_keys);
        prop_assert_eq!(exec.to_host(&actual_values.slice(..len)).unwrap(), expected_values);
    }

    #[test]
    fn sort_by_key_matches_oracle(values in prop::collection::vec(any::<u32>(), 0..MAX_LEN)) {
        let _guard = gpu_lock();
        let keys = unique_keys(values.len());
        let exec = exec();
        let keys_g = padded_device(&exec, &keys);
        let values_g = padded_device(&exec, &values);
        let (expected_keys, expected_values) = oracle_simple::sort_by_key(&keys, &values);
        let actual_keys = exec.to_device(&vec![0_u32; keys.len()]).unwrap();
        let actual_values = exec.to_device(&vec![0_u32; values.len()]).unwrap();
        sort_by_key(&exec, massively::SoA1(keys_g.slice(slice_range(&keys))), massively::SoA1(values_g.slice(slice_range(&values))), TupleBucketThenValueLess, massively::SoA1(actual_keys.slice_mut(..)), massively::SoA1(actual_values.slice_mut(..))).unwrap();
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
        let (expected_keys, expected_values) = oracle_simple::sort_by_key(&keys, &values);
        let actual_keys = exec.to_device(&vec![0_u32; keys.len()]).unwrap();
        let actual_values = exec.to_device(&vec![0_u32; values.len()]).unwrap();
        stable_sort_by_key(&exec, massively::SoA1(keys_g.slice(slice_range(&keys))), massively::SoA1(values_g.slice(slice_range(&values))), TupleBucketThenValueLess, massively::SoA1(actual_keys.slice_mut(..)), massively::SoA1(actual_values.slice_mut(..))).unwrap();
        prop_assert_eq!(exec.to_host(&actual_keys).unwrap(), expected_keys);
        prop_assert_eq!(exec.to_host(&actual_values).unwrap(), expected_values);
    }

    #[test]
    fn merge_by_key_matches_oracle(values in prop::collection::vec(any::<u32>(), 2..MAX_LEN)) {
        let _guard = gpu_lock();
        let keys = unique_keys(values.len());
        let (keys, values) = oracle_simple::sort_by_key(&keys, &values);
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
        let (expected_keys, expected_values) = oracle_simple::merge_by_key(&left_keys, &left_values, &right_keys, &right_values);
        let actual_keys = exec.to_device(&vec![0_u32; keys.len()]).unwrap();
        let actual_values = exec.to_device(&vec![0_u32; values.len()]).unwrap();
        merge_by_key(&exec,
            massively::SoA1(left_keys_g.slice(slice_range(&left_keys))),
            massively::SoA1(left_values_g.slice(slice_range(&left_values))),
            massively::SoA1(right_keys_g.slice(slice_range(&right_keys))),
            massively::SoA1(right_values_g.slice(slice_range(&right_values))),
            TupleBucketThenValueLess,
            massively::SoA1(actual_keys.slice_mut(..)),
            massively::SoA1(actual_values.slice_mut(..)),
        )
        .unwrap();
        prop_assert_eq!(exec.to_host(&actual_keys).unwrap(), expected_keys);
        prop_assert_eq!(exec.to_host(&actual_values).unwrap(), expected_values);
    }
}
