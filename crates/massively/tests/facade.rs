use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::algorithm::op::{BinaryPredicateOp, PredicateOp, ReductionOp, UnaryOp};
use massively::prelude::*;

#[test]
fn wide_zip_types_are_exported() {
    let _ = massively::Zip4(1_u32, 2_u32, 3_u32, 4_u32);
    let _ = massively::Zip5(1_u32, 2_u32, 3_u32, 4_u32, 5_u32);
    let _ = massively::Zip6(1_u32, 2_u32, 3_u32, 4_u32, 5_u32, 6_u32);
    let _ = massively::Zip7(1_u32, 2_u32, 3_u32, 4_u32, 5_u32, 6_u32, 7_u32);

    let _: massively::algorithm::Zip4<_, _, _, _> = (1_u32, 2_u32, 3_u32, 4_u32).into();
    let _: massively::algorithm::Zip5<_, _, _, _, _> = (1_u32, 2_u32, 3_u32, 4_u32, 5_u32).into();
    let _: massively::algorithm::Zip6<_, _, _, _, _, _> =
        (1_u32, 2_u32, 3_u32, 4_u32, 5_u32, 6_u32).into();
    let _: massively::algorithm::Zip7<_, _, _, _, _, _, _> =
        (1_u32, 2_u32, 3_u32, 4_u32, 5_u32, 6_u32, 7_u32).into();

    let _: Zip4<_, _, _, _> = (1_u32, 2_u32, 3_u32, 4_u32).into();
    let _: Zip5<_, _, _, _, _> = (1_u32, 2_u32, 3_u32, 4_u32, 5_u32).into();
    let _: Zip6<_, _, _, _, _, _> = (1_u32, 2_u32, 3_u32, 4_u32, 5_u32, 6_u32).into();
    let _: Zip7<_, _, _, _, _, _, _> = (1_u32, 2_u32, 3_u32, 4_u32, 5_u32, 6_u32, 7_u32).into();
}

#[test]
fn zip_types_are_exported() {
    let _: massively::Zip1<_> = massively::Zip1(1_u32);
    let _: massively::Zip2<_, _> = massively::Zip2(1_u32, 2_u32);
    let _: massively::Zip3<_, _, _> = massively::Zip3(1_u32, 2_u32, 3_u32);
    let _: massively::Zip4<_, _, _, _> = massively::Zip4(1_u32, 2_u32, 3_u32, 4_u32);
    let _: massively::Zip5<_, _, _, _, _> = massively::Zip5(1_u32, 2_u32, 3_u32, 4_u32, 5_u32);
    let _: massively::Zip6<_, _, _, _, _, _> =
        massively::Zip6(1_u32, 2_u32, 3_u32, 4_u32, 5_u32, 6_u32);
    let _: massively::Zip7<_, _, _, _, _, _, _> =
        massively::Zip7(1_u32, 2_u32, 3_u32, 4_u32, 5_u32, 6_u32, 7_u32);

    let _: massively::algorithm::Zip2<_, _> = massively::algorithm::Zip2(1_u32, 2_u32);
    let _: Zip2<_, _> = Zip2(1_u32, 2_u32);
}

struct AddOne;

#[cubecl::cube]
impl<R> UnaryOp<R, (u32,)> for AddOne
where
    R: Runtime,
{
    type Env = ();
    type Output = (u32,);

    fn apply(_env: (), input: (u32,)) -> (u32,) {
        (input.0 + 1,)
    }
}

struct Split;

#[cubecl::cube]
impl<R> UnaryOp<R, (u32,)> for Split
where
    R: Runtime,
{
    type Env = ();
    type Output = (u32, u32);

    fn apply(_env: (), input: (u32,)) -> (u32, u32) {
        (input.0, input.0 + 10)
    }
}

struct PairShift;

#[cubecl::cube]
impl<R> UnaryOp<R, (u32, u32)> for PairShift
where
    R: Runtime,
{
    type Env = ();
    type Output = (u32, u32);

    fn apply(_env: (), input: (u32, u32)) -> (u32, u32) {
        (input.0 + input.1, input.1 + 100)
    }
}

struct TripleShift;

#[cubecl::cube]
impl<R> UnaryOp<R, (u32, u32, u32)> for TripleShift
where
    R: Runtime,
{
    type Env = ();
    type Output = (u32, u32, u32);

    fn apply(_env: (), input: (u32, u32, u32)) -> (u32, u32, u32) {
        (input.0 + input.1, input.1 + input.2, input.2 + 1000)
    }
}

struct TupleU32Less;

#[cubecl::cube]
impl<R> BinaryPredicateOp<R, (u32,)> for TupleU32Less
where
    R: Runtime,
{
    fn apply(lhs: (u32,), rhs: (u32,)) -> bool {
        lhs.0 < rhs.0
    }
}

struct PairU32Less;

#[cubecl::cube]
impl<R> BinaryPredicateOp<R, (u32, u32)> for PairU32Less
where
    R: Runtime,
{
    fn apply(lhs: (u32, u32), rhs: (u32, u32)) -> bool {
        lhs.0 < rhs.0
    }
}

struct PairEqual;

#[cubecl::cube]
impl<R> BinaryPredicateOp<R, (u32, u32)> for PairEqual
where
    R: Runtime,
{
    fn apply(lhs: (u32, u32), rhs: (u32, u32)) -> bool {
        lhs.0 == rhs.0 && lhs.1 == rhs.1
    }
}

struct PairDifference;

#[cubecl::cube]
impl<R> ReductionOp<R, (u32, u32)> for PairDifference
where
    R: Runtime,
{
    fn apply(lhs: (u32, u32), rhs: (u32, u32)) -> (u32, u32) {
        (lhs.0 - rhs.0, lhs.1 - rhs.1)
    }
}

struct TripleU32Less;

#[cubecl::cube]
impl<R> BinaryPredicateOp<R, (u32, u32, u32)> for TripleU32Less
where
    R: Runtime,
{
    fn apply(lhs: (u32, u32, u32), rhs: (u32, u32, u32)) -> bool {
        lhs.0 < rhs.0
    }
}

struct PairFirstOdd;

#[cubecl::cube]
impl<R> PredicateOp<R, (u32, u32)> for PairFirstOdd
where
    R: Runtime,
{
    type Env = ();

    fn apply(_env: (), input: (u32, u32)) -> bool {
        input.0 % 2 == 1
    }
}

fn transform2<R, S1, S2, Op>(
    exec: &Executor<R>,
    source: S1,
    op: Op,
    out: S2,
) -> Result<(), massively::Error>
where
    R: Runtime,
    S1: MIter<R>,
    S2: MIterMut<R>,
    Op: UnaryOp<R, S1::Item, Output = S2::Item, Env = ()>,
{
    massively::transform(exec, source, op, (), out)
}

fn transform3<R, S1, S2, Op>(
    exec: &Executor<R>,
    source: S1,
    op: Op,
    out: S2,
) -> Result<(), massively::Error>
where
    R: Runtime,
    S1: MIter<R, Item = (u32,)>,
    S2: MIterMut<R, Item = (u32,)>,
    Op: UnaryOp<R, (u32,), Output = (u32,), Env = ()>,
{
    massively::transform(exec, source, op, (), out)
}

fn transform4<R, S1, S2, Op>(
    exec: &Executor<R>,
    source: S1,
    op: Op,
    out: S2,
) -> Result<(), massively::Error>
where
    R: Runtime,
    S1: MIter<R, Item = (u32, u32)>,
    S2: MIterMut<R, Item = (u32, u32)>,
    Op: UnaryOp<R, (u32, u32), Output = (u32, u32), Env = ()>,
{
    massively::transform(exec, source, op, (), out)
}

fn transform5<R, S1, S2, Op>(
    exec: &Executor<R>,
    source: S1,
    op: Op,
    out: S2,
) -> Result<(), massively::Error>
where
    R: Runtime,
    S1: MIter<R, Item = (u32, u32, u32)>,
    S2: MIterMut<R, Item = (u32, u32, u32)>,
    Op: UnaryOp<R, (u32, u32, u32), Output = (u32, u32, u32), Env = ()>,
{
    massively::transform(exec, source, op, (), out)
}

fn transform_without_op<R, S1, S2>(
    exec: &Executor<R>,
    source: S1,
    out: S2,
) -> Result<(), massively::Error>
where
    R: Runtime,
    S1: MIter<R, Item = (u32,)>,
    S2: MIterMut<R, Item = (u32,)>,
{
    massively::transform(exec, source, AddOne, (), out)
}

fn reverse2<R, S1, S2>(exec: &Executor<R>, source: S1, out: S2) -> Result<(), massively::Error>
where
    R: Runtime,
    S1: MIter<R>,
    S2: MIterMut<R, Item = S1::Item>,
{
    massively::reverse(exec, source, out)
}

fn sort2<R, S1, S2, Less>(
    exec: &Executor<R>,
    source: S1,
    less: Less,
    out: S2,
) -> Result<(), massively::Error>
where
    R: Runtime,
    S1: MIter<R>,
    S2: MIterMut<R, Item = S1::Item>,
    Less: BinaryPredicateOp<R, S1::Item>,
{
    massively::sort(exec, source, less, out)
}

fn minmax_element2<R, S1, Less>(
    exec: &Executor<R>,
    source: S1,
    less: Less,
) -> Result<Option<(massively::MIndex, massively::MIndex)>, massively::Error>
where
    R: Runtime,
    S1: MIter<R>,
    Less: BinaryPredicateOp<R, S1::Item>,
{
    massively::minmax_element(exec, source, less)
}

fn adjacent_find2<R, S1, Pred>(
    exec: &Executor<R>,
    source: S1,
    pred: Pred,
) -> Result<Option<massively::MIndex>, massively::Error>
where
    R: Runtime,
    S1: MIter<R>,
    Pred: BinaryPredicateOp<R, S1::Item>,
{
    massively::adjacent_find(exec, source, pred)
}

fn lower_bound2<R, S1, Values, Less>(
    exec: &Executor<R>,
    source: S1,
    values: Values,
    less: Less,
    out: DeviceSliceMut<'_, R, massively::MIndex>,
) -> Result<(), massively::Error>
where
    R: Runtime,
    S1: MIter<R>,
    Values: MIter<R, Item = S1::Item>,
    Less: BinaryPredicateOp<R, S1::Item>,
{
    massively::lower_bound(exec, source, values, less, out)
}

fn is_sorted2<R, S1, Less>(
    exec: &Executor<R>,
    source: S1,
    less: Less,
) -> Result<bool, massively::Error>
where
    R: Runtime,
    S1: MIter<R>,
    Less: BinaryPredicateOp<R, S1::Item>,
{
    massively::is_sorted(exec, source, less)
}

fn gather2<R, S1, S2>(
    exec: &Executor<R>,
    source: S1,
    indices: DeviceSlice<'_, R, massively::MIndex>,
    out: S2,
) -> Result<(), massively::Error>
where
    R: Runtime,
    S1: MIter<R>,
    S2: MIterMut<R, Item = S1::Item>,
{
    massively::gather(exec, source, indices, out)
}

fn copy_where2<R, S1, S2>(
    exec: &Executor<R>,
    source: S1,
    stencil: DeviceSlice<'_, R, u32>,
    out: S2,
) -> Result<massively::MIndex, massively::Error>
where
    R: Runtime,
    S1: MIter<R>,
    S2: MIterMut<R, Item = S1::Item>,
{
    massively::copy_where(exec, source, stencil, out)
}

fn replace_where2<R, S2>(
    exec: &Executor<R>,
    replacement: S2::Item,
    stencil: DeviceSlice<'_, R, u32>,
    out: S2,
) -> Result<(), massively::Error>
where
    R: Runtime,
    S2: MIterMut<R>,
{
    massively::replace_where(exec, replacement, stencil, out)
}

fn count_if2<R, S1, Pred>(
    exec: &Executor<R>,
    source: S1,
    pred: Pred,
) -> Result<massively::MIndex, massively::Error>
where
    R: Runtime,
    S1: MIter<R>,
    Pred: PredicateOp<R, S1::Item, Env = ()>,
{
    massively::count_if(exec, source, pred, ())
}

fn find_if2<R, S1, Pred>(
    exec: &Executor<R>,
    source: S1,
    pred: Pred,
) -> Result<Option<massively::MIndex>, massively::Error>
where
    R: Runtime,
    S1: MIter<R>,
    Pred: PredicateOp<R, S1::Item, Env = ()>,
{
    massively::find_if(exec, source, pred, ())
}

fn remove_where2<'a, R, S1, S2>(
    exec: &Executor<R>,
    source: S1,
    stencil: DeviceSlice<'a, R, u32>,
    out: S2,
) -> Result<massively::MIndex, massively::Error>
where
    R: Runtime,
    S1: MIter<R>,
    S2: MIterMut<R, Item = S1::Item>,
{
    massively::remove_where(exec, source, stencil, out)
}

fn partition2<R, S1, S2, Pred>(
    exec: &Executor<R>,
    source: S1,
    pred: Pred,
    out: S2,
) -> Result<massively::MIndex, massively::Error>
where
    R: Runtime,
    S1: MIter<R>,
    S2: MIterMut<R, Item = S1::Item>,
    Pred: PredicateOp<R, S1::Item, Env = ()>,
{
    massively::partition(exec, source, pred, (), out)
}

fn is_partitioned2<R, S1, Pred>(
    exec: &Executor<R>,
    source: S1,
    pred: Pred,
) -> Result<bool, massively::Error>
where
    R: Runtime,
    S1: MIter<R>,
    Pred: PredicateOp<R, S1::Item, Env = ()>,
{
    massively::is_partitioned(exec, source, pred, ())
}

fn unique2<R, S1, S2, Pred>(
    exec: &Executor<R>,
    source: S1,
    pred: Pred,
    out: S2,
) -> Result<massively::MIndex, massively::Error>
where
    R: Runtime,
    S1: MIter<R>,
    S2: MIterMut<R, Item = S1::Item>,
    Pred: BinaryPredicateOp<R, S1::Item>,
{
    massively::unique(exec, source, pred, out)
}

fn adjacent_difference2<R, S1, S2, Op>(
    exec: &Executor<R>,
    source: S1,
    op: Op,
    out: S2,
) -> Result<(), massively::Error>
where
    R: Runtime,
    S1: MIter<R>,
    S2: MIterMut<R, Item = S1::Item>,
    Op: ReductionOp<R, S1::Item>,
{
    massively::adjacent_difference(exec, source, op, out)
}

#[test]
fn transform2_wraps_tuple1_transform_without_cubecl_runtime_in_signature() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let input = exec.to_device(&[1_u32, 2, 3]).unwrap();

    let output = exec.to_device(&[0_u32; 3]).unwrap();
    transform2(
        &exec,
        Zip1(input.slice(..)),
        AddOne,
        Zip1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![2, 3, 4]);
}

#[test]
fn reverse2_wraps_reverse_with_slice_array_signature() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let input = exec.to_device(&[1_u32, 2, 3]).unwrap();
    let output = exec.to_device(&[0_u32; 3]).unwrap();

    reverse2(&exec, Zip1(input.slice(..)), Zip1(output.slice_mut(..))).unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![3, 2, 1]);
}

#[test]
fn sort2_wraps_sort_with_slice_array_signature() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let input = exec.to_device(&[3_u32, 1, 2]).unwrap();
    let output = exec.to_device(&[0_u32; 3]).unwrap();

    sort2(
        &exec,
        Zip1(input.slice(..)),
        TupleU32Less,
        Zip1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![1, 2, 3]);
}

#[test]
fn gather2_wraps_gather_with_slice_array_signature() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let input = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let indices = exec.to_device(&[2_u32, 0, 1]).unwrap();

    let output = exec.to_device(&[0_u32; 3]).unwrap();
    gather2(
        &exec,
        Zip1(input.slice(..)),
        indices.slice(..),
        Zip1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![30, 10, 20]);
}

#[test]
fn transform3_can_fix_concrete_input_and_output_types() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let input = exec.to_device(&[1_u32, 2, 3]).unwrap();

    let output = exec.to_device(&[0_u32; 3]).unwrap();
    transform3(
        &exec,
        Zip1(input.slice(..)),
        AddOne,
        Zip1(output.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![2, 3, 4]);
}

#[test]
fn transform_can_hide_op_inside_wrapper() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let input = exec.to_device(&[1_u32, 2, 3]).unwrap();

    let output = exec.to_device(&[0_u32; 3]).unwrap();
    transform_without_op(&exec, Zip1(input.slice(..)), Zip1(output.slice_mut(..))).unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![2, 3, 4]);
}

#[test]
fn transform4_can_fix_concrete_two_column_input_and_output_types() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let left = exec.to_device(&[1_u32, 2, 3]).unwrap();
    let right = exec.to_device(&[10_u32, 20, 30]).unwrap();

    let out_left = exec.to_device(&[0_u32; 3]).unwrap();
    let out_right = exec.to_device(&[0_u32; 3]).unwrap();
    transform4(
        &exec,
        Zip2(left.slice(..), right.slice(..)),
        PairShift,
        Zip2(out_left.slice_mut(..), out_right.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_left).unwrap(), vec![11, 22, 33]);
    assert_eq!(exec.to_host(&out_right).unwrap(), vec![110, 120, 130]);
}

#[test]
fn sort2_wraps_two_column_sort_with_slice_array_signature() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let left = exec.to_device(&[3_u32, 1, 2]).unwrap();
    let right = exec.to_device(&[30_u32, 10, 20]).unwrap();
    let out_left = exec.to_device(&[0_u32; 3]).unwrap();
    let out_right = exec.to_device(&[0_u32; 3]).unwrap();

    sort2(
        &exec,
        Zip2(left.slice(..), right.slice(..)),
        PairU32Less,
        Zip2(out_left.slice_mut(..), out_right.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_left).unwrap(), vec![1, 2, 3]);
    assert_eq!(exec.to_host(&out_right).unwrap(), vec![10, 20, 30]);
}

#[test]
fn transform5_can_fix_concrete_three_column_input_and_output_types() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let first = exec.to_device(&[1_u32, 2, 3]).unwrap();
    let second = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let third = exec.to_device(&[100_u32, 200, 300]).unwrap();

    let out_first = exec.to_device(&[0_u32; 3]).unwrap();
    let out_second = exec.to_device(&[0_u32; 3]).unwrap();
    let out_third = exec.to_device(&[0_u32; 3]).unwrap();
    transform5(
        &exec,
        Zip3(first.slice(..), second.slice(..), third.slice(..)),
        TripleShift,
        Zip3(
            out_first.slice_mut(..),
            out_second.slice_mut(..),
            out_third.slice_mut(..),
        ),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_first).unwrap(), vec![11, 22, 33]);
    assert_eq!(exec.to_host(&out_second).unwrap(), vec![110, 220, 330]);
    assert_eq!(exec.to_host(&out_third).unwrap(), vec![1100, 1200, 1300]);
}

#[test]
fn sort2_wraps_three_column_sort_with_slice_array_signature() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let first = exec.to_device(&[3_u32, 1, 2]).unwrap();
    let second = exec.to_device(&[30_u32, 10, 20]).unwrap();
    let third = exec.to_device(&[300_u32, 100, 200]).unwrap();
    let out_first = exec.to_device(&[0_u32; 3]).unwrap();
    let out_second = exec.to_device(&[0_u32; 3]).unwrap();
    let out_third = exec.to_device(&[0_u32; 3]).unwrap();

    sort2(
        &exec,
        Zip3(first.slice(..), second.slice(..), third.slice(..)),
        TripleU32Less,
        Zip3(
            out_first.slice_mut(..),
            out_second.slice_mut(..),
            out_third.slice_mut(..),
        ),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_first).unwrap(), vec![1, 2, 3]);
    assert_eq!(exec.to_host(&out_second).unwrap(), vec![10, 20, 30]);
    assert_eq!(exec.to_host(&out_third).unwrap(), vec![100, 200, 300]);
}

#[test]
fn minmax_element2_wraps_two_and_three_column_tuple_inputs() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let values = exec.to_device(&[2_u32, 1, 2, 3]).unwrap();
    let tags = exec.to_device(&[20_u32, 30, 10, 40]).unwrap();

    assert_eq!(
        minmax_element2(&exec, Zip2(values.slice(..), tags.slice(..)), PairU32Less).unwrap(),
        Some((1, 3))
    );

    let first = exec.to_device(&[2_u32, 1, 4, 3]).unwrap();
    let second = exec.to_device(&[20_u32, 10, 20, 10]).unwrap();
    let third = exec.to_device(&[200_u32, 100, 400, 300]).unwrap();

    assert_eq!(
        minmax_element2(
            &exec,
            Zip3(first.slice(..), second.slice(..), third.slice(..)),
            TripleU32Less
        )
        .unwrap(),
        Some((1, 2))
    );
}

#[test]
fn search_queries_wrap_two_column_tuple_inputs() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let left = exec.to_device(&[1_u32, 2, 2, 4]).unwrap();
    let right = exec.to_device(&[10_u32, 20, 20, 40]).unwrap();
    let query_left = exec.to_device(&[2_u32, 5]).unwrap();
    let query_right = exec.to_device(&[0_u32, 50]).unwrap();
    let lower = exec.to_device(&[0_u32; 2]).unwrap();

    assert_eq!(
        adjacent_find2(&exec, Zip2(left.slice(..), right.slice(..)), PairEqual).unwrap(),
        Some(1)
    );
    lower_bound2(
        &exec,
        Zip2(left.slice(..), right.slice(..)),
        Zip2(query_left.slice(..), query_right.slice(..)),
        PairU32Less,
        lower.slice_mut(..),
    )
    .unwrap();
    assert_eq!(exec.to_host(&lower).unwrap(), vec![1, 4]);
    assert!(is_sorted2(&exec, Zip2(left.slice(..), right.slice(..)), PairU32Less).unwrap());
}

#[test]
fn gather2_wraps_three_column_gather_with_slice_array_signature() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let first = exec.to_device(&[10_u32, 20, 30]).unwrap();
    let second = exec.to_device(&[100_u32, 200, 300]).unwrap();
    let third = exec.to_device(&[1000_u32, 2000, 3000]).unwrap();
    let indices = exec.to_device(&[2_u32, 0, 1]).unwrap();

    let out_first = exec.to_device(&[0_u32; 3]).unwrap();
    let out_second = exec.to_device(&[0_u32; 3]).unwrap();
    let out_third = exec.to_device(&[0_u32; 3]).unwrap();
    gather2(
        &exec,
        Zip3(first.slice(..), second.slice(..), third.slice(..)),
        indices.slice(..),
        Zip3(
            out_first.slice_mut(..),
            out_second.slice_mut(..),
            out_third.slice_mut(..),
        ),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_first).unwrap(), vec![30, 10, 20]);
    assert_eq!(exec.to_host(&out_second).unwrap(), vec![300, 100, 200]);
    assert_eq!(exec.to_host(&out_third).unwrap(), vec![3000, 1000, 2000]);
}

#[test]
fn transform2_wraps_tuple_output() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let input = exec.to_device(&[1_u32, 2, 3]).unwrap();

    let left = exec.to_device(&[0_u32; 3]).unwrap();
    let right = exec.to_device(&[0_u32; 3]).unwrap();
    transform2(
        &exec,
        Zip1(input.slice(..)),
        Split,
        Zip2(left.slice_mut(..), right.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&left).unwrap(), vec![1, 2, 3]);
    assert_eq!(exec.to_host(&right).unwrap(), vec![11, 12, 13]);
}

#[test]
fn copy_where2_wraps_two_column_copy_where_with_tuple_source() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let left = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let right = exec.to_device(&[100_u32, 200, 300, 400]).unwrap();
    let stencil = exec.to_device(&[1_u32, 0, 1, 0]).unwrap();
    let out_left = exec.to_device(&[0_u32; 4]).unwrap();
    let out_right = exec.to_device(&[0_u32; 4]).unwrap();

    let len = copy_where2(
        &exec,
        Zip2(left.slice(..), right.slice(..)),
        stencil.slice(..),
        Zip2(out_left.slice_mut(..), out_right.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(len, 2);
    assert_eq!(exec.to_host(&out_left.slice(..len)).unwrap(), vec![10, 30]);
    assert_eq!(
        exec.to_host(&out_right.slice(..len)).unwrap(),
        vec![100, 300]
    );
}

#[test]
fn replace_where2_wraps_two_column_replace_where_with_tuple_replacement() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let left = exec.to_device(&[10_u32, 20, 30, 40]).unwrap();
    let right = exec.to_device(&[100_u32, 200, 300, 400]).unwrap();
    let stencil = exec.to_device(&[1_u32, 0, 1, 0]).unwrap();

    replace_where2(
        &exec,
        (7_u32, 70_u32),
        stencil.slice(..),
        Zip2(left.slice_mut(..), right.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&left).unwrap(), vec![7, 20, 7, 40]);
    assert_eq!(exec.to_host(&right).unwrap(), vec![70, 200, 70, 400]);
}

#[test]
fn predicate_queries_wrap_two_column_tuple_predicates() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let left = exec.to_device(&[10_u32, 21, 30, 43]).unwrap();
    let right = exec.to_device(&[100_u32, 200, 300, 400]).unwrap();

    let count = count_if2(&exec, Zip2(left.slice(..), right.slice(..)), PairFirstOdd).unwrap();
    let found = find_if2(&exec, Zip2(left.slice(..), right.slice(..)), PairFirstOdd).unwrap();

    assert_eq!(count, 2);
    assert_eq!(found, Some(1));
}

#[test]
fn remove_where2_wraps_two_column_remove_where_with_stencil() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let left = exec.to_device(&[10_u32, 21, 30, 43]).unwrap();
    let right = exec.to_device(&[100_u32, 200, 300, 400]).unwrap();
    let stencil = exec.to_device(&[0_u32, 1, 0, 1]).unwrap();
    let out_left = exec.to_device(&[0_u32; 4]).unwrap();
    let out_right = exec.to_device(&[0_u32; 4]).unwrap();

    let len = remove_where2(
        &exec,
        Zip2(left.slice(..), right.slice(..)),
        stencil.slice(..),
        Zip2(out_left.slice_mut(..), out_right.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(len, 2);
    assert_eq!(exec.to_host(&out_left.slice(..len)).unwrap(), vec![10, 30]);
    assert_eq!(
        exec.to_host(&out_right.slice(..len)).unwrap(),
        vec![100, 300]
    );
}

#[test]
fn partition2_wraps_two_column_partition_with_tuple_predicate() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let left = exec.to_device(&[10_u32, 21, 30, 43]).unwrap();
    let right = exec.to_device(&[100_u32, 200, 300, 400]).unwrap();
    let out_left = exec.to_device(&[0_u32; 4]).unwrap();
    let out_right = exec.to_device(&[0_u32; 4]).unwrap();

    let split = partition2(
        &exec,
        Zip2(left.slice(..), right.slice(..)),
        PairFirstOdd,
        Zip2(out_left.slice_mut(..), out_right.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(split, 2);
    assert_eq!(
        exec.to_host(&out_left.slice(..split)).unwrap(),
        vec![21, 43]
    );
    assert_eq!(
        exec.to_host(&out_right.slice(..split)).unwrap(),
        vec![200, 400]
    );
    assert_eq!(
        exec.to_host(&out_left.slice(split..)).unwrap(),
        vec![10, 30]
    );
    assert_eq!(
        exec.to_host(&out_right.slice(split..)).unwrap(),
        vec![100, 300]
    );
}

#[test]
fn is_partitioned2_wraps_two_column_partition_query() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let partitioned_left = exec.to_device(&[21_u32, 43, 10, 30]).unwrap();
    let partitioned_right = exec.to_device(&[200_u32, 400, 100, 300]).unwrap();
    let mixed_left = exec.to_device(&[21_u32, 10, 43, 30]).unwrap();
    let mixed_right = exec.to_device(&[200_u32, 100, 400, 300]).unwrap();

    assert!(
        is_partitioned2(
            &exec,
            Zip2(partitioned_left.slice(..), partitioned_right.slice(..)),
            PairFirstOdd
        )
        .unwrap()
    );
    assert!(
        !is_partitioned2(
            &exec,
            Zip2(mixed_left.slice(..), mixed_right.slice(..)),
            PairFirstOdd
        )
        .unwrap()
    );
}

#[test]
fn unique2_wraps_two_column_unique_with_tuple_predicate() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let left = exec.to_device(&[1_u32, 1, 2, 2, 2, 3]).unwrap();
    let right = exec.to_device(&[10_u32, 10, 20, 21, 21, 30]).unwrap();
    let out_left = exec.to_device(&[0_u32; 6]).unwrap();
    let out_right = exec.to_device(&[0_u32; 6]).unwrap();

    let len = unique2(
        &exec,
        Zip2(left.slice(..), right.slice(..)),
        PairEqual,
        Zip2(out_left.slice_mut(..), out_right.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(len, 4);
    assert_eq!(
        exec.to_host(&out_left.slice(..len)).unwrap(),
        vec![1, 2, 2, 3]
    );
    assert_eq!(
        exec.to_host(&out_right.slice(..len)).unwrap(),
        vec![10, 20, 21, 30]
    );
}

#[test]
fn adjacent_difference2_wraps_two_column_tuple_op() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let left = exec.to_device(&[10_u32, 13, 20]).unwrap();
    let right = exec.to_device(&[100_u32, 107, 120]).unwrap();
    let out_left = exec.to_device(&[0_u32; 3]).unwrap();
    let out_right = exec.to_device(&[0_u32; 3]).unwrap();

    adjacent_difference2(
        &exec,
        Zip2(left.slice(..), right.slice(..)),
        PairDifference,
        Zip2(out_left.slice_mut(..), out_right.slice_mut(..)),
    )
    .unwrap();

    assert_eq!(exec.to_host(&out_left).unwrap(), vec![10, 3, 7]);
    assert_eq!(exec.to_host(&out_right).unwrap(), vec![100, 7, 13]);
}
