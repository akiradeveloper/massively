//! Predicate algorithms composed from materialization and reduction.

use core::marker::PhantomData;
use cubecl::prelude::*;

use crate::{
    DeviceSliceMut, DeviceVec, Dispatch, Error, Executor, MBool, MIndex, MVal, ReadExpression,
    Transform,
    op::{IndexedBinaryOp, IndexedUnaryOp, UnaryOp},
    output::StageOutput,
    read::{AdjacentIndexedTransform, Env0, Env1, IndexedTransform, LowerReadExpression},
    reduce::{ReduceDispatch, ReductionOp, StageRead, reduce},
    scan::{InclusiveScanDispatch, inclusive_scan},
    transform::{MaterializeDispatch, transform},
};

/// Compile-time predicate applied to one semantic input item.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, op, vector::count_if};
///
/// struct Positive;
///
/// #[cubecl::cube]
/// impl op::PredicateOp<i32> for Positive {
///     fn apply(value: i32) -> massively::MBool {
///         op::mbool(value > 0)
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[-1_i32, 2, 3]);
///
/// let count = count_if(&exec, input.slice(..), Positive).unwrap();
/// assert_eq!(count.read(&exec).unwrap(), 2);
/// ```
#[cubecl::cube]
pub trait PredicateOp<Input: CubeType>: 'static + Send + Sync {
    fn apply(input: Input) -> MBool;
}

#[cubecl::cube]
pub(crate) fn predicate<Input, Pred>(input: Input) -> bool
where
    Input: CubeType + 'static,
    Pred: PredicateOp<Input>,
{
    crate::op::is_true(Pred::apply(input))
}

#[doc(hidden)]
pub struct PredicateMap<Pred>(PhantomData<fn() -> Pred>);

#[cubecl::cube]
impl<Input, Pred> UnaryOp<Input> for PredicateMap<Pred>
where
    Input: CubeType + 'static,
    Pred: PredicateOp<Input>,
{
    type Output = MBool;

    fn apply(input: Input) -> MBool {
        crate::op::mbool(crate::op::is_true(Pred::apply(input)))
    }
}

struct SumU32;

#[cubecl::cube]
impl ReductionOp<u32> for SumU32 {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        lhs + rhs
    }
}

struct MinU32;

#[cubecl::cube]
impl ReductionOp<u32> for MinU32 {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        u32::min(lhs, rhs)
    }
}

struct CountIsAll;

#[cubecl::cube]
impl UnaryOp<(u32, u32)> for CountIsAll {
    type Output = MBool;

    fn apply(input: (u32, u32)) -> MBool {
        crate::op::mbool(input.0 == input.1)
    }
}

struct CountIsAny;

#[cubecl::cube]
impl UnaryOp<u32> for CountIsAny {
    type Output = MBool;

    fn apply(input: u32) -> MBool {
        crate::op::mbool(input != 0u32)
    }
}

struct CountIsNone;

#[cubecl::cube]
impl UnaryOp<u32> for CountIsNone {
    type Output = MBool;

    fn apply(input: u32) -> MBool {
        crate::op::mbool(input == 0u32)
    }
}

struct SentinelToOption;

#[cubecl::cube]
impl UnaryOp<u32> for SentinelToOption {
    type Output = (MBool, MIndex);

    fn apply(index: u32) -> Self::Output {
        (crate::op::mbool(index != u32::MAX), index)
    }
}

struct SentinelIsAbsent;

#[cubecl::cube]
impl UnaryOp<u32> for SentinelIsAbsent {
    type Output = MBool;

    fn apply(index: u32) -> MBool {
        crate::op::mbool(index == u32::MAX)
    }
}

struct FirstMatchingIndex<Pred>(PhantomData<fn() -> Pred>);

#[cubecl::cube]
impl<Input, Pred> IndexedUnaryOp<Input> for FirstMatchingIndex<Pred>
where
    Input: CubeType + 'static,
    Pred: PredicateOp<Input>,
{
    type Output = u32;

    fn apply(input: Input, index: u32) -> u32 {
        if crate::predicate::predicate::<Input, Pred>(input) {
            index
        } else {
            4_294_967_295u32
        }
    }
}

struct FirstPartitionViolation<Pred>(PhantomData<fn() -> Pred>);

#[cubecl::cube]
impl<Input, Pred> IndexedBinaryOp<Input> for FirstPartitionViolation<Pred>
where
    Input: CubeType + 'static,
    Pred: PredicateOp<Input>,
{
    type Output = u32;

    fn apply(previous: Input, current: Input, index: u32) -> u32 {
        if index != 0u32
            && !crate::predicate::predicate::<Input, Pred>(previous)
            && crate::predicate::predicate::<Input, Pred>(current)
        {
            index
        } else {
            4_294_967_295u32
        }
    }
}

#[cfg(test)]
struct PartitionViolation;

#[cfg(test)]
#[cubecl::cube]
impl UnaryOp<(u32, u32)> for PartitionViolation {
    type Output = u32;

    fn apply(input: (u32, u32)) -> u32 {
        (1u32 - input.0) * input.1
    }
}

/// Internal capability proving that the input has a supported predicate kernel.
#[doc(hidden)]
pub trait PredicateInput<R: Runtime, Pred>: ReadExpression + Sized {
    fn predicate_extent(&self) -> Result<crate::extent::LogicalExtent, Error>;
    fn predicate_count(self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error>;
    fn predicate_first(self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error>;
    fn predicate_is_partitioned(self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error>;
    fn predicate_positions(self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error>;
    fn predicate_flags(self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error>;
}

impl<R, Input, Pred> PredicateInput<R, Pred> for Input
where
    R: Runtime,
    Input: ReadExpression + StageRead<R, Env0>,
    Pred: PredicateOp<Input::Item>,
    Transform<Input, PredicateMap<Pred>>:
        ReadExpression<Item = u32> + LowerReadExpression + StageRead<R, Env0>,
    Dispatch<crate::A13, crate::S12>:
        MaterializeDispatch<
                R,
                Transform<Input, PredicateMap<Pred>>,
                DeviceSliceMut<u32>,
                crate::read::KernelReadSlots<
                    <Transform<Input, PredicateMap<Pred>> as LowerReadExpression>::Slots,
                >,
                crate::output::KernelOutputSlots<Env1<u32>>,
            >,
    IndexedTransform<Input, FirstMatchingIndex<Pred>>:
        ReadExpression<Item = u32> + LowerReadExpression + StageRead<R, Env0>,
    Dispatch<crate::A13, crate::S12>:
        ReduceDispatch<
                R,
                IndexedTransform<Input, FirstMatchingIndex<Pred>>,
                u32,
                MinU32,
                crate::read::KernelReadSlots<
                    <IndexedTransform<Input, FirstMatchingIndex<Pred>> as LowerReadExpression>::Slots,
                >,
                Storage = DeviceVec<R, u32>,
            >,
    AdjacentIndexedTransform<Input, FirstPartitionViolation<Pred>>:
        ReadExpression<Item = u32> + LowerReadExpression + StageRead<R, Env0>,
    Dispatch<crate::A13, crate::S12>: ReduceDispatch<
            R,
            AdjacentIndexedTransform<Input, FirstPartitionViolation<Pred>>,
            u32,
            MinU32,
            crate::read::KernelReadSlots<
                <AdjacentIndexedTransform<Input, FirstPartitionViolation<Pred>> as LowerReadExpression>::Slots,
            >,
            Storage = DeviceVec<R, u32>,
        >,
    Dispatch<crate::A13, crate::S12>:
        ReduceDispatch<
                R,
                Transform<Input, PredicateMap<Pred>>,
                u32,
                SumU32,
                crate::read::KernelReadSlots<
                    <Transform<Input, PredicateMap<Pred>> as LowerReadExpression>::Slots,
                >,
                Storage = DeviceVec<R, u32>,
            >,
    Dispatch<crate::A13, crate::S12>:
        InclusiveScanDispatch<
            R,
            Transform<Input, PredicateMap<Pred>>,
            DeviceSliceMut<u32>,
            u32,
            crate::read::KernelReadSlots<
                <Transform<Input, PredicateMap<Pred>> as LowerReadExpression>::Slots,
            >,
            crate::output::KernelOutputSlots<Env1<u32>>,
            SumU32,
        >,
    DeviceSliceMut<u32>: StageOutput<R, Env0>,
{
    fn predicate_extent(&self) -> Result<crate::extent::LogicalExtent, Error> {
        self.logical_extent()
    }

    fn predicate_count(self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error> {
        reduce(
            exec,
            Transform::new(self, PredicateMap::<Pred>(PhantomData)),
            exec.value(0u32)?.into_scratch_storage(),
            SumU32,
        )
    }

    fn predicate_first(self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error> {
        reduce(
            exec,
            IndexedTransform::new(self, FirstMatchingIndex::<Pred>(PhantomData)),
            exec.value(u32::MAX)?.into_scratch_storage(),
            MinU32,
        )
    }

    fn predicate_is_partitioned(self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error> {
        reduce(
            exec,
            AdjacentIndexedTransform::new(
                self,
                FirstPartitionViolation::<Pred>(PhantomData),
            ),
            exec.value(u32::MAX)?.into_scratch_storage(),
            MinU32,
        )
    }

    fn predicate_positions(self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error> {
        let len = self.logical_len()?;
        let extent = self.logical_extent()?;
        let mut positions = exec.alloc_row::<u32>(len);
        positions.set_logical_extent(extent);
        inclusive_scan(
            exec,
            Transform::new(self, PredicateMap::<Pred>(PhantomData)),
            SumU32,
            positions.slice_mut(..),
        )?;
        Ok(positions)
    }

    fn predicate_flags(self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error> {
        let len = self.logical_len()?;
        let extent = self.logical_extent()?;
        let mut flags = exec.alloc_row::<u32>(len);
        flags.set_logical_extent(extent);
        transform(
            exec,
            self,
            PredicateMap::<Pred>(PhantomData),
            flags.slice_mut(..),
        )?;
        Ok(flags)
    }
}

/// Counts elements satisfying `pred`.
pub(crate) fn count_if<R, Input, Pred>(
    exec: &Executor<R>,
    input: Input,
    _pred: Pred,
) -> Result<MVal<R, MIndex>, Error>
where
    R: Runtime,
    Input: PredicateInput<R, Pred>,
{
    MVal::from_storage(input.predicate_count(exec)?)
}

/// Returns whether every element satisfies `pred`.
pub(crate) fn all_of<R, Input, Pred>(
    exec: &Executor<R>,
    input: Input,
    pred: Pred,
) -> Result<MVal<R, MBool>, Error>
where
    R: Runtime,
    Input: PredicateInput<R, Pred>,
{
    let end = input.predicate_extent()?.materialize(exec)?;
    let count = count_if(exec, input, pred)?;
    let output = crate::vector::transform(
        exec,
        crate::zip2(count.as_iter(), end.slice(..)),
        CountIsAll,
    )?;
    MVal::from_storage(output)
}

/// Returns whether at least one element satisfies `pred`.
pub(crate) fn any_of<R, Input, Pred>(
    exec: &Executor<R>,
    input: Input,
    _pred: Pred,
) -> Result<MVal<R, MBool>, Error>
where
    R: Runtime,
    Input: PredicateInput<R, Pred>,
{
    MVal::from_storage(crate::vector::transform(
        exec,
        input.predicate_count(exec)?.slice(..),
        CountIsAny,
    )?)
}

/// Returns whether no element satisfies `pred`.
pub(crate) fn none_of<R, Input, Pred>(
    exec: &Executor<R>,
    input: Input,
    _pred: Pred,
) -> Result<MVal<R, MBool>, Error>
where
    R: Runtime,
    Input: PredicateInput<R, Pred>,
{
    MVal::from_storage(crate::vector::transform(
        exec,
        input.predicate_count(exec)?.slice(..),
        CountIsNone,
    )?)
}

/// Returns the first index satisfying `pred`.
pub(crate) fn find_if<R, Input, Pred>(
    exec: &Executor<R>,
    input: Input,
    _pred: Pred,
) -> Result<MVal<R, (MBool, MIndex)>, Error>
where
    R: Runtime,
    Input: PredicateInput<R, Pred>,
{
    MVal::from_storage(crate::vector::transform(
        exec,
        input.predicate_first(exec)?.slice(..),
        SentinelToOption,
    )?)
}

/// Returns whether all passing elements precede all failing elements.
pub(crate) fn is_partitioned<R, Input, Pred>(
    exec: &Executor<R>,
    input: Input,
    _pred: Pred,
) -> Result<MVal<R, MBool>, Error>
where
    R: Runtime,
    Input: PredicateInput<R, Pred>,
{
    MVal::from_storage(crate::vector::transform(
        exec,
        input.predicate_is_partitioned(exec)?.slice(..),
        SentinelIsAbsent,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Counting, Permute, Zip};
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    struct MatchTriple;

    #[cubecl::cube]
    impl PredicateOp<(u32, u32, u32)> for MatchTriple {
        fn apply(input: (u32, u32, u32)) -> crate::MBool {
            crate::op::mbool(input.0 + input.1 == input.2)
        }
    }

    #[test]
    fn predicate_queries_receive_flat_rows() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let a = exec.to_device(&[1_u32, 2, 3, 4]);
        let b = exec.to_device(&[10_u32, 20, 30, 40]);
        let c = exec.to_device(&[11_u32, 0, 33, 0]);

        let input = || Zip::new(a.column(), Zip::new(b.column(), c.column()));
        assert_eq!(
            count_if(&exec, input(), MatchTriple)
                .unwrap()
                .read(&exec)
                .unwrap(),
            2
        );
        assert_eq!(
            all_of(&exec, input(), MatchTriple)
                .unwrap()
                .read(&exec)
                .unwrap(),
            0
        );
        assert_eq!(
            any_of(&exec, input(), MatchTriple)
                .unwrap()
                .read(&exec)
                .unwrap(),
            1
        );
        assert_eq!(
            none_of(&exec, input(), MatchTriple)
                .unwrap()
                .read(&exec)
                .unwrap(),
            0
        );
        assert_eq!(
            find_if(&exec, input(), MatchTriple)
                .unwrap()
                .read(&exec)
                .unwrap(),
            (1, 0)
        );
        assert_eq!(
            is_partitioned(&exec, input(), MatchTriple)
                .unwrap()
                .read(&exec)
                .unwrap(),
            0
        );
    }

    struct LastLeafIsThree;

    struct IsEven;

    #[cubecl::cube]
    impl PredicateOp<u32> for IsEven {
        fn apply(input: u32) -> crate::MBool {
            crate::op::mbool(input % 2u32 == 0u32)
        }
    }

    #[test]
    fn partitioned_accepts_a_two_item_all_passing_range() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let input = exec.to_device(&[20_u32, 0]);
        let flags =
            <_ as PredicateInput<WgpuRuntime, IsEven>>::predicate_flags(input.column(), &exec)
                .unwrap();
        assert_eq!(exec.to_host(&flags).unwrap(), vec![1, 1]);
        let violations = exec.alloc_row::<u32>(1);
        transform(
            &exec,
            Zip::new(flags.slice(..1), flags.slice(1..)),
            PartitionViolation,
            violations.slice_mut(..),
        )
        .unwrap();
        assert_eq!(exec.to_host(&violations).unwrap(), vec![0]);
        let reduced = reduce(&exec, violations.column(), exec.to_device(&[0_u32]), SumU32).unwrap();
        assert_eq!(exec.to_host(&reduced).unwrap(), vec![0]);
        let reduced = reduce(
            &exec,
            Transform::new(
                Zip::new(flags.slice(..1), flags.slice(1..)),
                PartitionViolation,
            ),
            exec.to_device(&[0_u32]),
            SumU32,
        )
        .unwrap();
        assert_eq!(exec.to_host(&reduced).unwrap(), vec![0],);
        assert_eq!(
            is_partitioned(&exec, input.column(), IsEven)
                .unwrap()
                .read(&exec)
                .unwrap(),
            1
        );
    }

    type Seven = (u32, u32, u32, u32, u32, u32, u32);

    #[cubecl::cube]
    impl PredicateOp<Seven> for LastLeafIsThree {
        fn apply(input: Seven) -> crate::MBool {
            crate::op::mbool(input.6 == 3)
        }
    }

    #[test]
    fn predicate_materialization_uses_eval8() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let columns: Vec<_> = (0_u32..7)
            .map(|_| exec.to_device(&[1_u32, 2, 3, 4]))
            .collect();
        let seven = Zip::new(
            columns[0].column(),
            Zip::new(
                columns[1].column(),
                Zip::new(
                    columns[2].column(),
                    Zip::new(
                        columns[3].column(),
                        Zip::new(
                            columns[4].column(),
                            Zip::new(columns[5].column(), columns[6].column()),
                        ),
                    ),
                ),
            ),
        );
        let input = Permute::new(seven, Counting::new(0, 4));
        assert_eq!(
            count_if(&exec, input, LastLeafIsThree)
                .unwrap()
                .read(&exec)
                .unwrap(),
            1
        );
    }

    struct Never;

    #[cubecl::cube]
    impl PredicateOp<u32> for Never {
        fn apply(_input: u32) -> crate::MBool {
            0u32
        }
    }

    #[test]
    fn find_if_returns_none_without_a_numeric_sentinel() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let input = exec.to_device(&[1_u32, 2, 3, 4]);
        assert_eq!(
            find_if(&exec, input.column(), Never)
                .unwrap()
                .read(&exec)
                .unwrap(),
            (0, u32::MAX)
        );
    }
}
