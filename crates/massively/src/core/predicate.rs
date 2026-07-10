//! Predicate algorithms composed from materialization and reduction.

use core::marker::PhantomData;
use cubecl::prelude::*;

use crate::{
    DeviceSliceMut, DeviceVec, Dispatch, Error, Executor, ReadExpression, S1, Transform, Zip,
    op::UnaryOp,
    output::StageOutput,
    read::{Env0, Env1, LowerReadExpression},
    reduce::{ReductionOp, StageRead, reduce},
    transform::{MaterializeDispatch, transform},
};

/// Compile-time predicate applied to one semantic input item.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, count_if};
/// use massively::op::PredicateOp;
///
/// struct Positive;
///
/// #[cubecl::cube]
/// impl PredicateOp<i32> for Positive {
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

struct PartitionViolation;

#[cubecl::cube]
impl UnaryOp<(u32, u32)> for PartitionViolation {
    type Output = u32;

    fn apply(input: (u32, u32)) -> u32 {
        (1u32 - input.0) * input.1
    }
}

/// Internal capability proving that the input has a canonical predicate kernel.
#[doc(hidden)]
pub trait PredicateInput<R: Runtime, Pred>: ReadExpression + Sized {
    fn predicate_len(&self) -> Result<usize, Error>;
    fn predicate_flags(self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error>;
}

impl<R, Input, Pred> PredicateInput<R, Pred> for Input
where
    R: Runtime,
    Input: ReadExpression + StageRead<R, Env0>,
    Pred: PredicateOp<Input::Item>,
    Transform<Input, PredicateMap<Pred>>:
        ReadExpression<Item = u32> + LowerReadExpression + StageRead<R, Env0>,
    Dispatch<<Transform<Input, PredicateMap<Pred>> as ReadExpression>::ReadArity, S1>:
        MaterializeDispatch<
                R,
                Transform<Input, PredicateMap<Pred>>,
                DeviceSliceMut<u32>,
                <Transform<Input, PredicateMap<Pred>> as LowerReadExpression>::Slots,
                Env1<u32>,
            >,
    DeviceSliceMut<u32>: StageOutput<R, Env0>,
{
    fn predicate_len(&self) -> Result<usize, Error> {
        self.logical_len()
    }

    fn predicate_flags(self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error> {
        let len = self.logical_len()?;
        let flags = exec.alloc::<u32>(len);
        transform(
            exec,
            self,
            PredicateMap::<Pred>(PhantomData),
            flags.slice_mut(..),
        )?;
        Ok(flags)
    }
}

fn count_flags<R: Runtime>(exec: &Executor<R>, flags: &DeviceVec<R, u32>) -> Result<u32, Error> {
    reduce(exec, flags.column(), 0, SumU32)
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
    let flags = input.predicate_flags(exec)?;
    count_flags(exec, &flags)
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
    let flags = input.predicate_flags(exec)?;
    Ok(count_flags(exec, &flags)? != 0)
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
    let flags = input.predicate_flags(exec)?;
    let control = crate::selection::SelectionControl::from_flags(exec, flags)?;
    control.first_index(exec)
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
    let flags = input.predicate_flags(exec)?;
    if flags.len() < 2 {
        return Ok(true);
    }
    let violations = Transform::new(
        Zip::new(flags.slice(..flags.len() - 1), flags.slice(1..)),
        PartitionViolation,
    );
    Ok(reduce(exec, violations, 0, SumU32)? == 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Counting, Permute};
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    struct NestedMatch;

    #[cubecl::cube]
    impl PredicateOp<(u32, (u32, u32))> for NestedMatch {
        fn apply(input: (u32, (u32, u32))) -> bool {
            input.0 + input.1.0 == input.1.1
        }
    }

    #[test]
    fn predicate_queries_preserve_nested_semantic_shape() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let a = exec.to_device(&[1_u32, 2, 3, 4]);
        let b = exec.to_device(&[10_u32, 20, 30, 40]);
        let c = exec.to_device(&[11_u32, 0, 33, 0]);

        let input = || Zip::new(a.column(), Zip::new(b.column(), c.column()));
        assert_eq!(count_if(&exec, input(), NestedMatch).unwrap(), 2);
        assert!(!all_of(&exec, input(), NestedMatch).unwrap());
        assert!(any_of(&exec, input(), NestedMatch).unwrap());
        assert!(!none_of(&exec, input(), NestedMatch).unwrap());
        assert_eq!(find_if(&exec, input(), NestedMatch).unwrap(), Some(0));
        assert!(!is_partitioned(&exec, input(), NestedMatch).unwrap());
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
        let violations = exec.alloc::<u32>(1);
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

    type Seven = (u32, (u32, (u32, (u32, (u32, (u32, u32))))));

    #[cubecl::cube]
    impl PredicateOp<Seven> for LastLeafIsThree {
        fn apply(input: Seven) -> bool {
            input.1.1.1.1.1.1 == 3
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
