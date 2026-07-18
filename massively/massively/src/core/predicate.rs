//! Predicate algorithms composed from materialization and reduction.

use core::marker::PhantomData;
use cubecl::prelude::*;

use crate::{
    DeviceSliceMut, DeviceVec, Dispatch, Error, Executor, ReadExpression, Transform,
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
///     fn apply(value: i32) -> bool {
///         value > 0
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[-1_i32, 2, 3]);
///
/// assert_eq!(count_if(&exec, input.slice(..), Positive).unwrap(), 2);
/// ```
#[cubecl::cube]
pub trait PredicateOp<Input: CubeType>: 'static + Send + Sync {
    fn apply(input: Input) -> bool;
}

#[doc(hidden)]
pub struct PredicateMap<Pred>(PhantomData<fn() -> Pred>);

#[cubecl::cube]
impl<Input, Pred> UnaryOp<Input> for PredicateMap<Pred>
where
    Input: CubeType + 'static,
    Pred: PredicateOp<Input>,
{
    type Output = u32;

    fn apply(input: Input) -> u32 {
        if Pred::apply(input) { 1u32 } else { 0u32 }
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

struct FirstMatchingIndex<Pred>(PhantomData<fn() -> Pred>);

#[cubecl::cube]
impl<Input, Pred> IndexedUnaryOp<Input> for FirstMatchingIndex<Pred>
where
    Input: CubeType + 'static,
    Pred: PredicateOp<Input>,
{
    type Output = u32;

    fn apply(input: Input, index: u32) -> u32 {
        if Pred::apply(input) {
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
        if index != 0u32 && !Pred::apply(previous) && Pred::apply(current) {
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
    fn predicate_len(&self) -> Result<usize, Error>;
    fn predicate_count(self, exec: &Executor<R>) -> Result<u32, Error>;
    fn predicate_first(self, exec: &Executor<R>) -> Result<Option<u32>, Error>;
    fn predicate_is_partitioned(self, exec: &Executor<R>) -> Result<bool, Error>;
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
    fn predicate_len(&self) -> Result<usize, Error> {
        self.logical_len()
    }

    fn predicate_count(self, exec: &Executor<R>) -> Result<u32, Error> {
        reduce(
            exec,
            Transform::new(self, PredicateMap::<Pred>(PhantomData)),
            0u32,
            SumU32,
        )
    }

    fn predicate_first(self, exec: &Executor<R>) -> Result<Option<u32>, Error> {
        let index = reduce(
            exec,
            IndexedTransform::new(self, FirstMatchingIndex::<Pred>(PhantomData)),
            u32::MAX,
            MinU32,
        )?;
        Ok((index != u32::MAX).then_some(index))
    }

    fn predicate_is_partitioned(self, exec: &Executor<R>) -> Result<bool, Error> {
        let index = reduce(
            exec,
            AdjacentIndexedTransform::new(
                self,
                FirstPartitionViolation::<Pred>(PhantomData),
            ),
            u32::MAX,
            MinU32,
        )?;
        Ok(index == u32::MAX)
    }

    fn predicate_positions(self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error> {
        let len = self.logical_len()?;
        let positions = exec.alloc_row::<u32>(len);
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
        let flags = exec.alloc_row::<u32>(len);
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
) -> Result<u32, Error>
where
    R: Runtime,
    Input: PredicateInput<R, Pred>,
{
    input.predicate_count(exec)
}

/// Returns whether every element satisfies `pred`.
pub(crate) fn all_of<R, Input, Pred>(
    exec: &Executor<R>,
    input: Input,
    pred: Pred,
) -> Result<bool, Error>
where
    R: Runtime,
    Input: PredicateInput<R, Pred>,
{
    let len = input.predicate_len()?;
    Ok(count_if(exec, input, pred)? as usize == len)
}

/// Returns whether at least one element satisfies `pred`.
pub(crate) fn any_of<R, Input, Pred>(
    exec: &Executor<R>,
    input: Input,
    _pred: Pred,
) -> Result<bool, Error>
where
    R: Runtime,
    Input: PredicateInput<R, Pred>,
{
    Ok(input.predicate_count(exec)? != 0)
}

/// Returns whether no element satisfies `pred`.
pub(crate) fn none_of<R, Input, Pred>(
    exec: &Executor<R>,
    input: Input,
    pred: Pred,
) -> Result<bool, Error>
where
    R: Runtime,
    Input: PredicateInput<R, Pred>,
{
    Ok(!any_of(exec, input, pred)?)
}

/// Returns the first index satisfying `pred`.
pub(crate) fn find_if<R, Input, Pred>(
    exec: &Executor<R>,
    input: Input,
    _pred: Pred,
) -> Result<Option<u32>, Error>
where
    R: Runtime,
    Input: PredicateInput<R, Pred>,
{
    input.predicate_first(exec)
}

/// Returns whether all passing elements precede all failing elements.
pub(crate) fn is_partitioned<R, Input, Pred>(
    exec: &Executor<R>,
    input: Input,
    _pred: Pred,
) -> Result<bool, Error>
where
    R: Runtime,
    Input: PredicateInput<R, Pred>,
{
    input.predicate_is_partitioned(exec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Counting, Permute, Zip};
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    struct MatchTriple;

    #[cubecl::cube]
    impl PredicateOp<(u32, u32, u32)> for MatchTriple {
        fn apply(input: (u32, u32, u32)) -> bool {
            input.0 + input.1 == input.2
        }
    }

    #[test]
    fn predicate_queries_receive_flat_rows() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let a = exec.to_device(&[1_u32, 2, 3, 4]);
        let b = exec.to_device(&[10_u32, 20, 30, 40]);
        let c = exec.to_device(&[11_u32, 0, 33, 0]);

        let input = || Zip::new(a.column(), Zip::new(b.column(), c.column()));
        assert_eq!(count_if(&exec, input(), MatchTriple).unwrap(), 2);
        assert!(!all_of(&exec, input(), MatchTriple).unwrap());
        assert!(any_of(&exec, input(), MatchTriple).unwrap());
        assert!(!none_of(&exec, input(), MatchTriple).unwrap());
        assert_eq!(find_if(&exec, input(), MatchTriple).unwrap(), Some(0));
        assert!(!is_partitioned(&exec, input(), MatchTriple).unwrap());
    }

    struct LastLeafIsThree;

    struct IsEven;

    #[cubecl::cube]
    impl PredicateOp<u32> for IsEven {
        fn apply(input: u32) -> bool {
            input % 2u32 == 0u32
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
        assert_eq!(reduce(&exec, violations.column(), 0, SumU32).unwrap(), 0);
        assert_eq!(
            reduce(
                &exec,
                Transform::new(
                    Zip::new(flags.slice(..1), flags.slice(1..)),
                    PartitionViolation,
                ),
                0,
                SumU32,
            )
            .unwrap(),
            0,
        );
        assert!(is_partitioned(&exec, input.column(), IsEven).unwrap());
    }

    type Seven = (u32, u32, u32, u32, u32, u32, u32);

    #[cubecl::cube]
    impl PredicateOp<Seven> for LastLeafIsThree {
        fn apply(input: Seven) -> bool {
            input.6 == 3
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
        let input = Permute::new(
            seven,
            crate::Transform::new(Counting::new(0, 4), crate::op::U32ToUsize),
        );
        assert_eq!(count_if(&exec, input, LastLeafIsThree).unwrap(), 1);
    }

    struct Never;

    #[cubecl::cube]
    impl PredicateOp<u32> for Never {
        fn apply(_input: u32) -> bool {
            false
        }
    }

    #[test]
    fn find_if_returns_none_without_a_numeric_sentinel() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let input = exec.to_device(&[1_u32, 2, 3, 4]);
        assert_eq!(find_if(&exec, input.column(), Never).unwrap(), None);
    }
}
