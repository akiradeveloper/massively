use cubecl::frontend::PartialEqExpand;
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::op as gpu_op;
use massively::{Executor, MIndex, MIter, Zip1};
use oracle::op as host_op;
use oracle_ref as oracle;
use proptest::prelude::*;
use proptest::test_runner::TestCaseError;
use std::fmt;
use std::ops::Deref;

type ScaleRuntime = WgpuRuntime;
type ScaleExecutor = Executor<ScaleRuntime>;

const CASES: u32 = 1;
const SCALE_LEN: usize = 10_000_000;
const PRIME_BLOCK_LEN: usize = 65_537 * 256;

struct ScaleVec(Vec<u32>);

impl fmt::Debug for ScaleVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScaleVec")
            .field("len", &self.0.len())
            .finish()
    }
}

impl Deref for ScaleVec {
    type Target = [u32];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

struct EqTuple;
struct Identity;
struct KeepTuple;
struct LessU32;
struct MaxTuple;
struct U32Flag;

#[cubecl::cube]
impl gpu_op::BinaryPredicateOp<ScaleRuntime, (u32,)> for EqTuple {
    fn apply(lhs: (u32,), rhs: (u32,)) -> bool {
        lhs.0 == rhs.0
    }
}

impl host_op::BinaryPredicateOp<(u32,)> for EqTuple {
    fn apply(lhs: (u32,), rhs: (u32,)) -> bool {
        lhs.0 == rhs.0
    }
}

#[cubecl::cube]
impl gpu_op::UnaryOp<ScaleRuntime, (u32,)> for Identity {
    type Output = (u32,);

    fn apply(input: (u32,)) -> (u32,) {
        input
    }
}

impl host_op::UnaryOp<(u32,)> for Identity {
    type Output = (u32,);

    fn apply(input: (u32,)) -> (u32,) {
        input
    }
}

#[cubecl::cube]
impl gpu_op::PredicateOp<ScaleRuntime, (u32,)> for KeepTuple {
    fn apply(input: (u32,)) -> bool {
        input.0 > 0
    }
}

impl host_op::PredicateOp<(u32,)> for KeepTuple {
    fn apply(input: (u32,)) -> bool {
        input.0 > 0
    }
}

#[cubecl::cube]
impl gpu_op::BinaryPredicateOp<ScaleRuntime, (u32,)> for LessU32 {
    fn apply(lhs: (u32,), rhs: (u32,)) -> bool {
        lhs.0 < rhs.0
    }
}

impl host_op::BinaryPredicateOp<(u32,)> for LessU32 {
    fn apply(lhs: (u32,), rhs: (u32,)) -> bool {
        lhs.0 < rhs.0
    }
}

#[cubecl::cube]
impl gpu_op::ReductionOp<ScaleRuntime, (u32,)> for MaxTuple {
    fn apply(lhs: (u32,), rhs: (u32,)) -> (u32,) {
        (lhs.0.max(rhs.0),)
    }
}

impl host_op::ReductionOp<(u32,)> for MaxTuple {
    fn apply(lhs: (u32,), rhs: (u32,)) -> (u32,) {
        (lhs.0.max(rhs.0),)
    }
}

#[cubecl::cube]
impl gpu_op::UnaryOp<ScaleRuntime, u32> for U32Flag {
    type Output = bool;

    fn apply(input: u32) -> bool {
        input != 0
    }
}

fn exec() -> ScaleExecutor {
    Executor::<ScaleRuntime>::new(WgpuDevice::DefaultDevice)
}

fn lazify<Input>(
    input: Input,
) -> massively::lazy::Transform<
    massively::lazy::Permute<Input, massively::lazy::Taken<massively::lazy::Counting>>,
    massively::op::Identity,
>
where
    Input: MIter<ScaleRuntime>,
{
    let len = input.len();
    massively::lazy::transform(
        massively::lazy::permute(input, massively::lazy::counting(0).take(len)),
        massively::op::Identity,
    )
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

fn scale_config() -> ProptestConfig {
    ProptestConfig {
        cases: CASES,
        failure_persistence: None,
        ..ProptestConfig::default()
    }
}

fn scale_vec() -> impl Strategy<Value = ScaleVec> {
    prop::collection::vec(any::<u32>(), SCALE_LEN..=SCALE_LEN)
        .prop_map(ScaleVec)
        .no_shrink()
}

#[test]
#[ignore = "scale"]
fn scale_prime_block_dispatch_guard() {
    let exec = exec();
    let input = (0..PRIME_BLOCK_LEN as u32).collect::<Vec<_>>();
    let input_g = exec.to_device(&input).unwrap();

    assert_eq!(
        massively::reduce(&exec, lazify(Zip1(input_g.slice(..))), (0_u32,), MaxTuple).unwrap(),
        ((PRIME_BLOCK_LEN - 1) as u32,)
    );

    assert_eq!(
        massively::minmax_element(&exec, lazify(Zip1(input_g.slice(..))), LessU32).unwrap(),
        Some((0, (PRIME_BLOCK_LEN - 1) as MIndex))
    );

    let output_g = exec.to_device(&vec![0_u32; input.len()]).unwrap();
    massively::inclusive_scan(
        &exec,
        lazify(Zip1(input_g.slice(..))),
        MaxTuple,
        Zip1(output_g.slice_mut(..)),
    )
    .unwrap();
    let output = exec.to_host(&output_g).unwrap();
    assert_eq!(output[0], 0);
    assert_eq!(output[PRIME_BLOCK_LEN - 1], (PRIME_BLOCK_LEN - 1) as u32);
}

fn aos(input: &[u32]) -> Vec<(u32,)> {
    input.iter().copied().map(|value| (value,)).collect()
}

fn col(input: &[(u32,)]) -> Vec<u32> {
    input.iter().map(|value| value.0).collect()
}

fn stencil_from(input: &[u32]) -> Vec<u32> {
    input.iter().map(|value| value & 1).collect()
}

fn indices_from(input: &[u32]) -> Vec<u32> {
    let len = u32::try_from(input.len()).unwrap();
    input.iter().map(|value| value % len).collect()
}

fn permutation_from(input: &[u32]) -> Vec<u32> {
    let mut indices = (0..u32::try_from(input.len()).unwrap()).collect::<Vec<_>>();
    indices.sort_by_key(|index| (input[*index as usize], *index));
    indices
}

fn assert_vec_eq(actual: &[u32], expected: &[u32]) -> Result<(), TestCaseError> {
    if actual.len() != expected.len() {
        return Err(TestCaseError::fail(format!(
            "length mismatch: actual_len={} expected_len={}",
            actual.len(),
            expected.len()
        )));
    }
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        if actual != expected {
            return Err(TestCaseError::fail(format!(
                "value mismatch: index={index} actual={actual} expected={expected}"
            )));
        }
    }
    Ok(())
}

fn assert_aos_eq(actual: &[u32], expected: &[(u32,)]) -> Result<(), TestCaseError> {
    if actual.len() != expected.len() {
        return Err(TestCaseError::fail(format!(
            "length mismatch: actual_len={} expected_len={}",
            actual.len(),
            expected.len()
        )));
    }
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        if *actual != expected.0 {
            return Err(TestCaseError::fail(format!(
                "value mismatch: index={index} actual={} expected={}",
                *actual, expected.0
            )));
        }
    }
    Ok(())
}

fn assert_eq_silent<T>(actual: T, expected: T) -> Result<(), TestCaseError>
where
    T: PartialEq + fmt::Debug,
{
    if actual == expected {
        Ok(())
    } else {
        Err(TestCaseError::fail(format!(
            "value mismatch: actual={actual:?} expected={expected:?}"
        )))
    }
}

macro_rules! scale_test {
    ($name:ident, $input:ident, $body:block) => {
        proptest! {
            #![proptest_config(scale_config())]

            #[test]
            #[ignore = "scale"]
            fn $name($input in scale_vec()) $body
        }
    };
}

macro_rules! scale_test2 {
    ($name:ident, $left:ident, $right:ident, $body:block) => {
        proptest! {
            #![proptest_config(scale_config())]

            #[test]
            #[ignore = "scale"]
            fn $name(($left, $right) in (scale_vec(), scale_vec())) $body
        }
    };
}

scale_test!(scale_map, input, {
    let exec = exec();
    let host_input = aos(&input);
    let input_g = exec.to_device(&input).unwrap();
    let output_g = exec.to_device(&vec![0_u32; input.len()]).unwrap();

    massively::transform(
        &exec,
        lazify(Zip1(input_g.slice(..))),
        Identity,
        Zip1(output_g.slice_mut(..)),
    )
    .unwrap();

    assert_aos_eq(
        &exec.to_host(&output_g).unwrap(),
        &oracle::map(&host_input, Identity),
    )?;
});

scale_test!(scale_transform, input, {
    let exec = exec();
    let host_input = aos(&input);
    let mut host_output = vec![(0_u32,); input.len()];
    let input_g = exec.to_device(&input).unwrap();
    let output_g = exec.to_device(&vec![0_u32; input.len()]).unwrap();

    massively::transform(
        &exec,
        lazify(Zip1(input_g.slice(..))),
        Identity,
        Zip1(output_g.slice_mut(..)),
    )
    .unwrap();
    oracle::transform(&host_input, Identity, &mut host_output);

    assert_aos_eq(&exec.to_host(&output_g).unwrap(), &host_output)?;
});

scale_test!(scale_transform_where, input, {
    let exec = exec();
    let host_input = aos(&input);
    let stencil = stencil_from(&input);
    let mut host_output = vec![(0_u32,); input.len()];
    let input_g = exec.to_device(&input).unwrap();
    let stencil_g = exec.to_device(&stencil).unwrap();
    let output_g = exec.to_device(&vec![0_u32; input.len()]).unwrap();

    massively::transform_where(
        &exec,
        lazify(Zip1(input_g.slice(..))),
        Identity,
        lazify(massively::lazy::transform(stencil_g.slice(..), U32Flag)),
        Zip1(output_g.slice_mut(..)),
    )
    .unwrap();
    oracle::transform_where(&host_input, Identity, &stencil, &mut host_output);

    assert_aos_eq(&exec.to_host(&output_g).unwrap(), &host_output)?;
});

scale_test!(scale_reduce, input, {
    let exec = exec();
    let host_input = aos(&input);
    let input_g = exec.to_device(&input).unwrap();

    assert_eq_silent(
        massively::reduce(&exec, lazify(Zip1(input_g.slice(..))), (0_u32,), MaxTuple).unwrap(),
        oracle::reduce(&host_input, (0_u32,), MaxTuple),
    )?;
});

scale_test!(scale_inclusive_scan, input, {
    let exec = exec();
    let host_input = aos(&input);
    let input_g = exec.to_device(&input).unwrap();
    let output_g = exec.to_device(&vec![0_u32; input.len()]).unwrap();

    massively::inclusive_scan(
        &exec,
        lazify(Zip1(input_g.slice(..))),
        MaxTuple,
        Zip1(output_g.slice_mut(..)),
    )
    .unwrap();

    assert_aos_eq(
        &exec.to_host(&output_g).unwrap(),
        &oracle::inclusive_scan(&host_input, MaxTuple),
    )?;
});

scale_test!(scale_exclusive_scan, input, {
    let exec = exec();
    let host_input = aos(&input);
    let input_g = exec.to_device(&input).unwrap();
    let output_g = exec.to_device(&vec![0_u32; input.len()]).unwrap();

    massively::exclusive_scan(
        &exec,
        lazify(Zip1(input_g.slice(..))),
        (0_u32,),
        MaxTuple,
        Zip1(output_g.slice_mut(..)),
    )
    .unwrap();

    assert_aos_eq(
        &exec.to_host(&output_g).unwrap(),
        &oracle::exclusive_scan(&host_input, (0_u32,), MaxTuple),
    )?;
});

scale_test!(scale_adjacent_difference, input, {
    let exec = exec();
    let host_input = aos(&input);
    let input_g = exec.to_device(&input).unwrap();
    let output_g = exec.to_device(&vec![0_u32; input.len()]).unwrap();

    massively::adjacent_difference(
        &exec,
        lazify(Zip1(input_g.slice(..))),
        MaxTuple,
        Zip1(output_g.slice_mut(..)),
    )
    .unwrap();

    assert_aos_eq(
        &exec.to_host(&output_g).unwrap(),
        &oracle::adjacent_difference(&host_input, MaxTuple),
    )?;
});

scale_test!(scale_copy_where, input, {
    let exec = exec();
    let host_input = aos(&input);
    let stencil = stencil_from(&input);
    let input_g = exec.to_device(&input).unwrap();
    let stencil_g = exec.to_device(&stencil).unwrap();
    let output_g = exec.to_device(&vec![0_u32; input.len()]).unwrap();

    let len = massively::copy_where(
        &exec,
        lazify(Zip1(input_g.slice(..))),
        lazify(massively::lazy::transform(stencil_g.slice(..), U32Flag)),
        Zip1(output_g.slice_mut(..)),
    )
    .unwrap();

    assert_aos_eq(
        &exec.to_host(&output_g.slice(..len)).unwrap(),
        &oracle::copy_where(&host_input, &stencil),
    )?;
});

scale_test!(scale_remove_where, input, {
    let exec = exec();
    let host_input = aos(&input);
    let stencil = stencil_from(&input);
    let input_g = exec.to_device(&input).unwrap();
    let stencil_g = exec.to_device(&stencil).unwrap();
    let output_g = exec.to_device(&vec![0_u32; input.len()]).unwrap();

    let len = massively::remove_where(
        &exec,
        lazify(Zip1(input_g.slice(..))),
        lazify(massively::lazy::transform(stencil_g.slice(..), U32Flag)),
        Zip1(output_g.slice_mut(..)),
    )
    .unwrap();

    assert_aos_eq(
        &exec.to_host(&output_g.slice(..len)).unwrap(),
        &oracle::remove_where(&host_input, &stencil),
    )?;
});

scale_test!(scale_partition, input, {
    let exec = exec();
    let host_input = aos(&input);
    let input_g = exec.to_device(&input).unwrap();
    let output_g = exec.to_device(&input).unwrap();

    let split = massively::partition(
        &exec,
        lazify(Zip1(input_g.slice(..))),
        KeepTuple,
        Zip1(output_g.slice_mut(..)),
    )
    .unwrap();
    let (host_yes, host_no) = oracle::partition(&host_input, KeepTuple);

    assert_aos_eq(&exec.to_host(&output_g.slice(..split)).unwrap(), &host_yes)?;
    assert_aos_eq(
        &exec
            .to_host(&output_g.slice(split..mindex(input.len())))
            .unwrap(),
        &host_no,
    )?;
});

scale_test!(scale_count_if, input, {
    let exec = exec();
    let host_input = aos(&input);
    let input_g = exec.to_device(&input).unwrap();

    assert_eq_silent(
        massively::count_if(&exec, lazify(Zip1(input_g.slice(..))), KeepTuple).unwrap(),
        mindex(oracle::count_if(&host_input, KeepTuple)),
    )?;
});

scale_test!(scale_all_of, input, {
    let exec = exec();
    let host_input = aos(&input);
    let input_g = exec.to_device(&input).unwrap();

    assert_eq_silent(
        massively::all_of(&exec, lazify(Zip1(input_g.slice(..))), KeepTuple).unwrap(),
        oracle::all_of(&host_input, KeepTuple),
    )?;
});

scale_test!(scale_any_of, input, {
    let exec = exec();
    let host_input = aos(&input);
    let input_g = exec.to_device(&input).unwrap();

    assert_eq_silent(
        massively::any_of(&exec, lazify(Zip1(input_g.slice(..))), KeepTuple).unwrap(),
        oracle::any_of(&host_input, KeepTuple),
    )?;
});

scale_test!(scale_none_of, input, {
    let exec = exec();
    let host_input = aos(&input);
    let input_g = exec.to_device(&input).unwrap();

    assert_eq_silent(
        massively::none_of(&exec, lazify(Zip1(input_g.slice(..))), KeepTuple).unwrap(),
        oracle::none_of(&host_input, KeepTuple),
    )?;
});

scale_test!(scale_find_if, input, {
    let exec = exec();
    let host_input = aos(&input);
    let input_g = exec.to_device(&input).unwrap();

    assert_eq_silent(
        massively::find_if(&exec, lazify(Zip1(input_g.slice(..))), KeepTuple).unwrap(),
        opt_mindex(oracle::find_if(&host_input, KeepTuple)),
    )?;
});

scale_test!(scale_is_partitioned, input, {
    let exec = exec();
    let host_input = aos(&input);
    let input_g = exec.to_device(&input).unwrap();

    assert_eq_silent(
        massively::is_partitioned(&exec, lazify(Zip1(input_g.slice(..))), KeepTuple).unwrap(),
        oracle::is_partitioned(&host_input, KeepTuple),
    )?;
});

scale_test2!(scale_gather, input, index_seed, {
    let exec = exec();
    let indices = indices_from(&index_seed);
    let mut host_output = aos(&input);
    let input_g = exec.to_device(&input).unwrap();
    let indices_g = exec.to_device(&indices).unwrap();
    let output_g = exec.to_device(&input).unwrap();

    massively::gather(
        &exec,
        lazify(Zip1(input_g.slice(..))),
        lazify(indices_g.slice(..)),
        Zip1(output_g.slice_mut(..)),
    )
    .unwrap();
    oracle::gather(&aos(&input), &indices, &mut host_output);

    assert_aos_eq(&exec.to_host(&output_g).unwrap(), &host_output)?;
});

scale_test2!(scale_gather_where, input, index_seed, {
    let exec = exec();
    let indices = indices_from(&index_seed);
    let stencil = stencil_from(&index_seed);
    let mut host_output = aos(&input);
    let input_g = exec.to_device(&input).unwrap();
    let indices_g = exec.to_device(&indices).unwrap();
    let stencil_g = exec.to_device(&stencil).unwrap();
    let output_g = exec.to_device(&input).unwrap();

    massively::gather_where(
        &exec,
        lazify(Zip1(input_g.slice(..))),
        lazify(indices_g.slice(..)),
        lazify(massively::lazy::transform(stencil_g.slice(..), U32Flag)),
        Zip1(output_g.slice_mut(..)),
    )
    .unwrap();
    oracle::gather_where(&aos(&input), &indices, &stencil, &mut host_output);

    assert_aos_eq(&exec.to_host(&output_g).unwrap(), &host_output)?;
});

scale_test2!(scale_scatter, input, index_seed, {
    let exec = exec();
    let indices = permutation_from(&index_seed);
    let mut host_output = aos(&input);
    let input_g = exec.to_device(&input).unwrap();
    let indices_g = exec.to_device(&indices).unwrap();
    let output_g = exec.to_device(&input).unwrap();

    massively::scatter(
        &exec,
        lazify(Zip1(input_g.slice(..))),
        lazify(indices_g.slice(..)),
        Zip1(output_g.slice_mut(..)),
    )
    .unwrap();
    oracle::scatter(&aos(&input), &indices, &mut host_output);

    assert_aos_eq(&exec.to_host(&output_g).unwrap(), &host_output)?;
});

scale_test2!(scale_scatter_where, input, index_seed, {
    let exec = exec();
    let indices = permutation_from(&index_seed);
    let stencil = stencil_from(&index_seed);
    let mut host_output = aos(&input);
    let input_g = exec.to_device(&input).unwrap();
    let indices_g = exec.to_device(&indices).unwrap();
    let stencil_g = exec.to_device(&stencil).unwrap();
    let output_g = exec.to_device(&input).unwrap();

    massively::scatter_where(
        &exec,
        lazify(Zip1(input_g.slice(..))),
        lazify(indices_g.slice(..)),
        lazify(massively::lazy::transform(stencil_g.slice(..), U32Flag)),
        Zip1(output_g.slice_mut(..)),
    )
    .unwrap();
    oracle::scatter_where(&aos(&input), &indices, &stencil, &mut host_output);

    assert_aos_eq(&exec.to_host(&output_g).unwrap(), &host_output)?;
});

scale_test!(scale_equal, input, {
    let exec = exec();
    let host_input = aos(&input);
    let input_g = exec.to_device(&input).unwrap();

    assert_eq_silent(
        massively::equal(
            &exec,
            lazify(Zip1(input_g.slice(..))),
            lazify(Zip1(input_g.slice(..))),
            EqTuple,
        )
        .unwrap(),
        oracle::equal(&host_input, &host_input, EqTuple),
    )?;
});

scale_test2!(scale_mismatch, input, other, {
    let exec = exec();
    let host_input = aos(&input);
    let host_other = aos(&other);
    let input_g = exec.to_device(&input).unwrap();
    let other_g = exec.to_device(&other).unwrap();

    assert_eq_silent(
        massively::mismatch(
            &exec,
            lazify(Zip1(input_g.slice(..))),
            lazify(Zip1(other_g.slice(..))),
            EqTuple,
        )
        .unwrap(),
        opt_mindex(oracle::mismatch(&host_input, &host_other, EqTuple)),
    )?;
});

scale_test!(scale_adjacent_find, input, {
    let exec = exec();
    let host_input = aos(&input);
    let input_g = exec.to_device(&input).unwrap();

    assert_eq_silent(
        massively::adjacent_find(&exec, lazify(Zip1(input_g.slice(..))), EqTuple).unwrap(),
        opt_mindex(oracle::adjacent_find(&host_input, EqTuple)),
    )?;
});

scale_test!(scale_find_first_of, input, {
    let exec = exec();
    let host_input = aos(&input);
    let input_g = exec.to_device(&input).unwrap();

    assert_eq_silent(
        massively::find_first_of(
            &exec,
            lazify(Zip1(input_g.slice(..))),
            lazify(Zip1(input_g.slice(..))),
            EqTuple,
        )
        .unwrap(),
        opt_mindex(oracle::find_first_of(&host_input, &host_input, EqTuple)),
    )?;
});

scale_test!(scale_fill, input, {
    let exec = exec();
    let mut host_output = aos(&input);
    let output_g = exec.to_device(&input).unwrap();

    massively::fill(&exec, (123_u32,), Zip1(output_g.slice_mut(..))).unwrap();
    oracle::fill((123_u32,), &mut host_output);

    assert_aos_eq(&exec.to_host(&output_g).unwrap(), &host_output)?;
});

scale_test!(scale_replace_where, input, {
    let exec = exec();
    let stencil = stencil_from(&input);
    let mut host_output = aos(&input);
    let stencil_g = exec.to_device(&stencil).unwrap();
    let output_g = exec.to_device(&input).unwrap();

    massively::replace_where(
        &exec,
        (123_u32,),
        lazify(massively::lazy::transform(stencil_g.slice(..), U32Flag)),
        Zip1(output_g.slice_mut(..)),
    )
    .unwrap();
    oracle::replace_where((123_u32,), &stencil, &mut host_output);

    assert_aos_eq(&exec.to_host(&output_g).unwrap(), &host_output)?;
});

scale_test!(scale_reverse, input, {
    let exec = exec();
    let host_input = aos(&input);
    let input_g = exec.to_device(&input).unwrap();
    let output_g = exec.to_device(&vec![0_u32; input.len()]).unwrap();

    massively::reverse(
        &exec,
        lazify(Zip1(input_g.slice(..))),
        Zip1(output_g.slice_mut(..)),
    )
    .unwrap();

    assert_aos_eq(
        &exec.to_host(&output_g).unwrap(),
        &oracle::reverse(&host_input),
    )?;
});

scale_test!(scale_sort, input, {
    let exec = exec();
    let host_input = aos(&input);
    let input_g = exec.to_device(&input).unwrap();
    let output_g = exec.to_device(&vec![0_u32; input.len()]).unwrap();

    massively::sort(
        &exec,
        lazify(Zip1(input_g.slice(..))),
        LessU32,
        Zip1(output_g.slice_mut(..)),
    )
    .unwrap();

    assert_aos_eq(
        &exec.to_host(&output_g).unwrap(),
        &oracle::sort(&host_input, LessU32),
    )?;
});

scale_test2!(scale_merge, left, right, {
    let exec = exec();
    let left_sorted = oracle::sort(&aos(&left), LessU32);
    let right_sorted = oracle::sort(&aos(&right), LessU32);
    let left_col = col(&left_sorted);
    let right_col = col(&right_sorted);
    let left_g = exec.to_device(&left_col).unwrap();
    let right_g = exec.to_device(&right_col).unwrap();
    let output_g = exec
        .to_device(&vec![0_u32; left_col.len() + right_col.len()])
        .unwrap();

    massively::merge(
        &exec,
        lazify(Zip1(left_g.slice(..))),
        lazify(Zip1(right_g.slice(..))),
        LessU32,
        Zip1(output_g.slice_mut(..)),
    )
    .unwrap();

    assert_aos_eq(
        &exec.to_host(&output_g).unwrap(),
        &oracle::merge(&left_sorted, &right_sorted, LessU32),
    )?;
});

scale_test!(scale_is_sorted, input, {
    let exec = exec();
    let sorted = oracle::sort(&aos(&input), LessU32);
    let sorted_col = col(&sorted);
    let sorted_g = exec.to_device(&sorted_col).unwrap();

    assert_eq_silent(
        massively::is_sorted(&exec, lazify(Zip1(sorted_g.slice(..))), LessU32).unwrap(),
        oracle::is_sorted(&sorted, LessU32),
    )?;
});

scale_test!(scale_is_sorted_until, input, {
    let exec = exec();
    let host_input = aos(&input);
    let input_g = exec.to_device(&input).unwrap();

    assert_eq_silent(
        massively::is_sorted_until(&exec, lazify(Zip1(input_g.slice(..))), LessU32).unwrap(),
        mindex(oracle::is_sorted_until(&host_input, LessU32)),
    )?;
});

scale_test2!(scale_lexicographical_compare, left, right, {
    let exec = exec();
    let host_left = aos(&left);
    let host_right = aos(&right);
    let left_g = exec.to_device(&left).unwrap();
    let right_g = exec.to_device(&right).unwrap();

    assert_eq_silent(
        massively::lexicographical_compare(
            &exec,
            lazify(Zip1(left_g.slice(..))),
            lazify(Zip1(right_g.slice(..))),
            LessU32,
        )
        .unwrap(),
        oracle::lexicographical_compare(&host_left, &host_right, LessU32),
    )?;
});

scale_test!(scale_min_element, input, {
    let exec = exec();
    let host_input = aos(&input);
    let input_g = exec.to_device(&input).unwrap();

    assert_eq_silent(
        massively::min_element(&exec, lazify(Zip1(input_g.slice(..))), LessU32).unwrap(),
        opt_mindex(oracle::min_element(&host_input, LessU32)),
    )?;
});

scale_test!(scale_max_element, input, {
    let exec = exec();
    let host_input = aos(&input);
    let input_g = exec.to_device(&input).unwrap();

    assert_eq_silent(
        massively::max_element(&exec, lazify(Zip1(input_g.slice(..))), LessU32).unwrap(),
        opt_mindex(oracle::max_element(&host_input, LessU32)),
    )?;
});

scale_test!(scale_minmax_element, input, {
    let exec = exec();
    let host_input = aos(&input);
    let input_g = exec.to_device(&input).unwrap();

    assert_eq_silent(
        massively::minmax_element(&exec, lazify(Zip1(input_g.slice(..))), LessU32).unwrap(),
        opt_pair_mindex(oracle::minmax_element(&host_input, LessU32)),
    )?;
});

scale_test2!(scale_lower_bound, source, values, {
    let exec = exec();
    let sorted = oracle::sort(&aos(&source), LessU32);
    let sorted_col = col(&sorted);
    let value_tuples = aos(&values);
    let sorted_g = exec.to_device(&sorted_col).unwrap();
    let values_g = exec.to_device(&values).unwrap();
    let output_g = exec.to_device(&vec![0_u32; values.len()]).unwrap();

    massively::lower_bound(
        &exec,
        lazify(Zip1(sorted_g.slice(..))),
        lazify(Zip1(values_g.slice(..))),
        LessU32,
        output_g.slice_mut(..),
    )
    .unwrap();

    assert_vec_eq(
        &exec.to_host(&output_g).unwrap(),
        &oracle::lower_bound(&sorted, &value_tuples, LessU32),
    )?;
});

scale_test2!(scale_upper_bound, source, values, {
    let exec = exec();
    let sorted = oracle::sort(&aos(&source), LessU32);
    let sorted_col = col(&sorted);
    let value_tuples = aos(&values);
    let sorted_g = exec.to_device(&sorted_col).unwrap();
    let values_g = exec.to_device(&values).unwrap();
    let output_g = exec.to_device(&vec![0_u32; values.len()]).unwrap();

    massively::upper_bound(
        &exec,
        lazify(Zip1(sorted_g.slice(..))),
        lazify(Zip1(values_g.slice(..))),
        LessU32,
        output_g.slice_mut(..),
    )
    .unwrap();

    assert_vec_eq(
        &exec.to_host(&output_g).unwrap(),
        &oracle::upper_bound(&sorted, &value_tuples, LessU32),
    )?;
});

scale_test!(scale_unique, input, {
    let exec = exec();
    let sorted = oracle::sort(&aos(&input), LessU32);
    let sorted_col = col(&sorted);
    let sorted_g = exec.to_device(&sorted_col).unwrap();
    let output_g = exec.to_device(&sorted_col).unwrap();

    let len = massively::unique(
        &exec,
        lazify(Zip1(sorted_g.slice(..))),
        EqTuple,
        Zip1(output_g.slice_mut(..)),
    )
    .unwrap();

    assert_aos_eq(
        &exec.to_host(&output_g.slice(..len)).unwrap(),
        &oracle::unique(&sorted, EqTuple),
    )?;
});

scale_test2!(scale_set_union, left, right, {
    let exec = exec();
    let left_sorted = oracle::sort(&aos(&left), LessU32);
    let right_sorted = oracle::sort(&aos(&right), LessU32);
    let left_col = col(&left_sorted);
    let right_col = col(&right_sorted);
    let left_g = exec.to_device(&left_col).unwrap();
    let right_g = exec.to_device(&right_col).unwrap();
    let output_g = exec
        .to_device(&vec![0_u32; left_col.len() + right_col.len()])
        .unwrap();

    let len = massively::set_union(
        &exec,
        lazify(Zip1(left_g.slice(..))),
        lazify(Zip1(right_g.slice(..))),
        LessU32,
        Zip1(output_g.slice_mut(..)),
    )
    .unwrap();

    assert_aos_eq(
        &exec.to_host(&output_g.slice(..len)).unwrap(),
        &oracle::set_union(&left_sorted, &right_sorted, LessU32),
    )?;
});

scale_test2!(scale_set_intersection, left, right, {
    let exec = exec();
    let left_sorted = oracle::sort(&aos(&left), LessU32);
    let right_sorted = oracle::sort(&aos(&right), LessU32);
    let left_col = col(&left_sorted);
    let right_col = col(&right_sorted);
    let left_g = exec.to_device(&left_col).unwrap();
    let right_g = exec.to_device(&right_col).unwrap();
    let output_g = exec
        .to_device(&vec![0_u32; left_col.len() + right_col.len()])
        .unwrap();

    let len = massively::set_intersection(
        &exec,
        lazify(Zip1(left_g.slice(..))),
        lazify(Zip1(right_g.slice(..))),
        LessU32,
        Zip1(output_g.slice_mut(..)),
    )
    .unwrap();

    assert_aos_eq(
        &exec.to_host(&output_g.slice(..len)).unwrap(),
        &oracle::set_intersection(&left_sorted, &right_sorted, LessU32),
    )?;
});

scale_test2!(scale_set_difference, left, right, {
    let exec = exec();
    let left_sorted = oracle::sort(&aos(&left), LessU32);
    let right_sorted = oracle::sort(&aos(&right), LessU32);
    let left_col = col(&left_sorted);
    let right_col = col(&right_sorted);
    let left_g = exec.to_device(&left_col).unwrap();
    let right_g = exec.to_device(&right_col).unwrap();
    let output_g = exec.to_device(&vec![0_u32; left_col.len()]).unwrap();

    let len = massively::set_difference(
        &exec,
        lazify(Zip1(left_g.slice(..))),
        lazify(Zip1(right_g.slice(..))),
        LessU32,
        Zip1(output_g.slice_mut(..)),
    )
    .unwrap();

    assert_aos_eq(
        &exec.to_host(&output_g.slice(..len)).unwrap(),
        &oracle::set_difference(&left_sorted, &right_sorted, LessU32),
    )?;
});

scale_test2!(scale_sort_by_key, keys, values, {
    let exec = exec();
    let key_tuples = aos(&keys);
    let value_tuples = aos(&values);
    let keys_g = exec.to_device(&keys).unwrap();
    let values_g = exec.to_device(&values).unwrap();
    let out_keys_g = exec.to_device(&vec![0_u32; keys.len()]).unwrap();
    let out_values_g = exec.to_device(&vec![0_u32; values.len()]).unwrap();

    massively::sort_by_key(
        &exec,
        lazify(Zip1(keys_g.slice(..))),
        lazify(Zip1(values_g.slice(..))),
        LessU32,
        Zip1(out_keys_g.slice_mut(..)),
        Zip1(out_values_g.slice_mut(..)),
    )
    .unwrap();
    let (host_keys, host_values) = oracle::sort_by_key(&key_tuples, &value_tuples, LessU32);

    assert_aos_eq(&exec.to_host(&out_keys_g).unwrap(), &host_keys)?;
    assert_aos_eq(&exec.to_host(&out_values_g).unwrap(), &host_values)?;
});

scale_test2!(scale_inclusive_scan_by_key, keys, values, {
    let exec = exec();
    let key_tuples = aos(&keys);
    let value_tuples = aos(&values);
    let keys_g = exec.to_device(&keys).unwrap();
    let values_g = exec.to_device(&values).unwrap();
    let output_g = exec.to_device(&vec![0_u32; values.len()]).unwrap();

    massively::inclusive_scan_by_key(
        &exec,
        lazify(Zip1(keys_g.slice(..))),
        lazify(Zip1(values_g.slice(..))),
        EqTuple,
        MaxTuple,
        Zip1(output_g.slice_mut(..)),
    )
    .unwrap();

    assert_aos_eq(
        &exec.to_host(&output_g).unwrap(),
        &oracle::inclusive_scan_by_key(&key_tuples, &value_tuples, EqTuple, MaxTuple),
    )?;
});

scale_test2!(scale_exclusive_scan_by_key, keys, values, {
    let exec = exec();
    let key_tuples = aos(&keys);
    let value_tuples = aos(&values);
    let keys_g = exec.to_device(&keys).unwrap();
    let values_g = exec.to_device(&values).unwrap();
    let output_g = exec.to_device(&vec![0_u32; values.len()]).unwrap();

    massively::exclusive_scan_by_key(
        &exec,
        lazify(Zip1(keys_g.slice(..))),
        lazify(Zip1(values_g.slice(..))),
        EqTuple,
        (0_u32,),
        MaxTuple,
        Zip1(output_g.slice_mut(..)),
    )
    .unwrap();

    assert_aos_eq(
        &exec.to_host(&output_g).unwrap(),
        &oracle::exclusive_scan_by_key(&key_tuples, &value_tuples, EqTuple, (0_u32,), MaxTuple),
    )?;
});

scale_test2!(scale_reduce_by_key, keys, values, {
    let exec = exec();
    let key_tuples = aos(&keys);
    let value_tuples = aos(&values);
    let keys_g = exec.to_device(&keys).unwrap();
    let values_g = exec.to_device(&values).unwrap();
    let out_keys_g = exec.to_device(&vec![0_u32; keys.len()]).unwrap();
    let out_values_g = exec.to_device(&vec![0_u32; values.len()]).unwrap();

    let len = massively::reduce_by_key(
        &exec,
        lazify(Zip1(keys_g.slice(..))),
        lazify(Zip1(values_g.slice(..))),
        EqTuple,
        (0_u32,),
        MaxTuple,
        Zip1(out_keys_g.slice_mut(..)),
        Zip1(out_values_g.slice_mut(..)),
    )
    .unwrap();
    let (host_keys, host_values) =
        oracle::reduce_by_key(&key_tuples, &value_tuples, EqTuple, (0_u32,), MaxTuple);

    assert_aos_eq(&exec.to_host(&out_keys_g.slice(..len)).unwrap(), &host_keys)?;
    assert_aos_eq(
        &exec.to_host(&out_values_g.slice(..len)).unwrap(),
        &host_values,
    )?;
});

scale_test2!(scale_unique_by_key, keys, values, {
    let exec = exec();
    let key_tuples = aos(&keys);
    let value_tuples = aos(&values);
    let keys_g = exec.to_device(&keys).unwrap();
    let values_g = exec.to_device(&values).unwrap();
    let out_keys_g = exec.to_device(&vec![0_u32; keys.len()]).unwrap();
    let out_values_g = exec.to_device(&vec![0_u32; values.len()]).unwrap();

    let len = massively::unique_by_key(
        &exec,
        lazify(Zip1(keys_g.slice(..))),
        lazify(Zip1(values_g.slice(..))),
        EqTuple,
        Zip1(out_keys_g.slice_mut(..)),
        Zip1(out_values_g.slice_mut(..)),
    )
    .unwrap();
    let (host_keys, host_values) = oracle::unique_by_key(&key_tuples, &value_tuples, EqTuple);

    assert_aos_eq(&exec.to_host(&out_keys_g.slice(..len)).unwrap(), &host_keys)?;
    assert_aos_eq(
        &exec.to_host(&out_values_g.slice(..len)).unwrap(),
        &host_values,
    )?;
});

scale_test2!(scale_merge_by_key, keys, values, {
    let exec = exec();
    let mid = keys.len() / 2;
    let left_key_tuples = aos(&keys[..mid]);
    let right_key_tuples = aos(&keys[mid..]);
    let left_value_tuples = aos(&values[..mid]);
    let right_value_tuples = aos(&values[mid..]);
    let (left_keys, left_values) =
        oracle::sort_by_key(&left_key_tuples, &left_value_tuples, LessU32);
    let (right_keys, right_values) =
        oracle::sort_by_key(&right_key_tuples, &right_value_tuples, LessU32);
    let left_key_col = col(&left_keys);
    let right_key_col = col(&right_keys);
    let left_value_col = col(&left_values);
    let right_value_col = col(&right_values);
    let left_keys_g = exec.to_device(&left_key_col).unwrap();
    let right_keys_g = exec.to_device(&right_key_col).unwrap();
    let left_values_g = exec.to_device(&left_value_col).unwrap();
    let right_values_g = exec.to_device(&right_value_col).unwrap();
    let out_keys_g = exec.to_device(&vec![0_u32; keys.len()]).unwrap();
    let out_values_g = exec.to_device(&vec![0_u32; values.len()]).unwrap();

    massively::merge_by_key(
        &exec,
        lazify(Zip1(left_keys_g.slice(..))),
        lazify(Zip1(left_values_g.slice(..))),
        lazify(Zip1(right_keys_g.slice(..))),
        lazify(Zip1(right_values_g.slice(..))),
        LessU32,
        Zip1(out_keys_g.slice_mut(..)),
        Zip1(out_values_g.slice_mut(..)),
    )
    .unwrap();
    let (host_keys, host_values) = oracle::merge_by_key(
        &left_keys,
        &left_values,
        &right_keys,
        &right_values,
        LessU32,
    );

    assert_aos_eq(&exec.to_host(&out_keys_g).unwrap(), &host_keys)?;
    assert_aos_eq(&exec.to_host(&out_values_g).unwrap(), &host_values)?;
});
