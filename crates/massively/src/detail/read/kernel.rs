#![allow(dead_code)]

use cubecl::prelude::{BufferArg, CubeCount, CubeDim, CubeElement, CubeType, Runtime};

use crate::{
    Error,
    detail::{
        CubePolicy,
        api::PrecomputedSelection,
        control::{ScanByKeyControl, SegmentControl},
        device::{
            DeviceColumnView, DeviceVec, KernelColumn, KernelColumnAt, KernelColumnBindings, S0,
            S1, S2, S3, S4, S5, S6, ZipView1, ZipView2, ZipView3, ZipView4, ZipView5, ZipView6,
            ZipView7,
        },
        dispatch::MItemDispatch,
        op,
        op_adapter::{KernelOp, KernelScalarTuple1Op},
        primitives::select,
        write::MItemWriteDispatch,
    },
    error::ensure_same_len,
    index::mindex_from_usize,
    iter::MIterMut,
    value::{MAlloc, MItem, MStorageElement},
};

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LogicalIdentity;

#[cubecl::cube]
impl<R, Input> op::UnaryOp<R, Input> for LogicalIdentity
where
    R: Runtime,
    Input: MItem<R>,
{
    type Output = Input;

    fn apply(input: Input) -> Input {
        input
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ScalarToTuple1Identity;

#[cubecl::cube]
impl<R, Input> op::UnaryOp<R, Input> for ScalarToTuple1Identity
where
    R: Runtime,
    Input: MStorageElement + 'static,
{
    type Output = (Input,);

    fn apply(input: Input) -> (Input,) {
        (input,)
    }
}

pub(crate) trait ScanByKeyValueItem<R: Runtime>: MAlloc<R> {
    fn inclusive_scan_by_key_values_from_inner<Op, Output>(
        policy: &CubePolicy<R>,
        inner: <Self as MAlloc<R>>::Inner,
        control: &ScanByKeyControl<R>,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R, Item = Self>,
        Op: op::ReductionOp<R, Self>;

    fn exclusive_scan_by_key_values_from_inner<Op, Output>(
        policy: &CubePolicy<R>,
        inner: <Self as MAlloc<R>>::Inner,
        control: &ScanByKeyControl<R>,
        init: Self,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R, Item = Self>,
        Op: op::ReductionOp<R, Self>;
}

impl<R, A> ScanByKeyValueItem<R> for (A,)
where
    R: Runtime,
    A: MStorageElement + 'static,
    (A,): MAlloc<R, Inner = (DeviceVec<R, A>,)>,
{
    fn inclusive_scan_by_key_values_from_inner<Op, Output>(
        policy: &CubePolicy<R>,
        inner: <Self as MAlloc<R>>::Inner,
        control: &ScanByKeyControl<R>,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R, Item = Self>,
        Op: op::ReductionOp<R, Self>,
    {
        let _ = op;
        let apply = crate::detail::apply::SegmentedScanApply::new(control);
        let view = DeviceColumnView::from_column(&inner.0);
        let scanned =
            apply.inclusive_expr::<DeviceColumnView<R, A>, KernelOp<R, Op>>(policy, &view)?;
        output.write_from_inner(policy, (scanned,))
    }

    fn exclusive_scan_by_key_values_from_inner<Op, Output>(
        policy: &CubePolicy<R>,
        inner: <Self as MAlloc<R>>::Inner,
        control: &ScanByKeyControl<R>,
        init: Self,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R, Item = Self>,
        Op: op::ReductionOp<R, Self>,
    {
        let _ = op;
        let apply = crate::detail::apply::SegmentedScanApply::new(control);
        let view = DeviceColumnView::from_column(&inner.0);
        let scanned = apply
            .exclusive_expr::<DeviceColumnView<R, A>, KernelOp<R, Op>>(policy, &view, init.0)?;
        output.write_from_inner(policy, (scanned,))
    }
}

macro_rules! impl_scan_by_key_value_tuple2 {
    ($a:ident : $a_idx:tt, $b:ident : $b_idx:tt) => {
        impl<R, $a, $b> ScanByKeyValueItem<R> for ($a, $b)
        where
            R: Runtime,
            $a: MStorageElement + 'static,
            $b: MStorageElement + 'static,
            ($a, $b): MAlloc<R, Inner = (DeviceVec<R, $a>, DeviceVec<R, $b>)>,
        {
            fn inclusive_scan_by_key_values_from_inner<Op, Output>(
                policy: &CubePolicy<R>,
                inner: <Self as MAlloc<R>>::Inner,
                control: &ScanByKeyControl<R>,
                op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                Output: MIterMut<R, Item = Self>,
                Op: op::ReductionOp<R, Self>,
            {
                let _ = op;
                let apply = crate::detail::apply::SegmentedScanApply::new(control);
                let a = DeviceColumnView::from_column(&inner.$a_idx);
                let b = DeviceColumnView::from_column(&inner.$b_idx);
                type V<R, T> = DeviceColumnView<R, T>;
                let scanned =
                    apply.inclusive_expr2::<V<R, $a>, V<R, $b>, KernelOp<R, Op>>(policy, &a, &b)?;
                output.write_from_inner(policy, (scanned.left, scanned.right))
            }

            fn exclusive_scan_by_key_values_from_inner<Op, Output>(
                policy: &CubePolicy<R>,
                inner: <Self as MAlloc<R>>::Inner,
                control: &ScanByKeyControl<R>,
                init: Self,
                op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                Output: MIterMut<R, Item = Self>,
                Op: op::ReductionOp<R, Self>,
            {
                let _ = op;
                let apply = crate::detail::apply::SegmentedScanApply::new(control);
                let a = DeviceColumnView::from_column(&inner.$a_idx);
                let b = DeviceColumnView::from_column(&inner.$b_idx);
                type V<R, T> = DeviceColumnView<R, T>;
                let scanned = apply
                    .exclusive_expr2::<V<R, $a>, V<R, $b>, KernelOp<R, Op>>(policy, &a, &b, init)?;
                output.write_from_inner(policy, (scanned.left, scanned.right))
            }
        }
    };
}

macro_rules! impl_scan_by_key_value_tuple3 {
    ($a:ident : $a_idx:tt, $b:ident : $b_idx:tt, $c:ident : $c_idx:tt) => {
        impl<R, $a, $b, $c> ScanByKeyValueItem<R> for ($a, $b, $c)
        where
            R: Runtime,
            $a: MStorageElement + 'static,
            $b: MStorageElement + 'static,
            $c: MStorageElement + 'static,
            ($a, $b, $c): MAlloc<R, Inner = (DeviceVec<R, $a>, DeviceVec<R, $b>, DeviceVec<R, $c>)>,
        {
            fn inclusive_scan_by_key_values_from_inner<Op, Output>(
                policy: &CubePolicy<R>,
                inner: <Self as MAlloc<R>>::Inner,
                control: &ScanByKeyControl<R>,
                op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                Output: MIterMut<R, Item = Self>,
                Op: op::ReductionOp<R, Self>,
            {
                let _ = op;
                let apply = crate::detail::apply::SegmentedScanApply::new(control);
                let a = DeviceColumnView::from_column(&inner.$a_idx);
                let b = DeviceColumnView::from_column(&inner.$b_idx);
                let c = DeviceColumnView::from_column(&inner.$c_idx);
                type V<R, T> = DeviceColumnView<R, T>;
                let scanned = apply
                    .inclusive_expr3::<V<R, $a>, V<R, $b>, V<R, $c>, KernelOp<R, Op>>(
                        policy, &a, &b, &c,
                    )?;
                output.write_from_inner(policy, (scanned.first, scanned.second, scanned.third))
            }

            fn exclusive_scan_by_key_values_from_inner<Op, Output>(
                policy: &CubePolicy<R>,
                inner: <Self as MAlloc<R>>::Inner,
                control: &ScanByKeyControl<R>,
                init: Self,
                op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                Output: MIterMut<R, Item = Self>,
                Op: op::ReductionOp<R, Self>,
            {
                let _ = op;
                let apply = crate::detail::apply::SegmentedScanApply::new(control);
                let a = DeviceColumnView::from_column(&inner.$a_idx);
                let b = DeviceColumnView::from_column(&inner.$b_idx);
                let c = DeviceColumnView::from_column(&inner.$c_idx);
                type V<R, T> = DeviceColumnView<R, T>;
                let scanned = apply
                    .exclusive_expr3::<V<R, $a>, V<R, $b>, V<R, $c>, KernelOp<R, Op>>(
                        policy, &a, &b, &c, init,
                    )?;
                output.write_from_inner(policy, (scanned.first, scanned.second, scanned.third))
            }
        }
    };
}

macro_rules! impl_scan_by_key_value_tuple_wide {
    ($name_inclusive:ident, $name_exclusive:ident; $( $ty:ident : $idx:tt ),+) => {
        #[allow(non_snake_case)]
        impl<R, $( $ty ),+> ScanByKeyValueItem<R> for ($( $ty, )+)
        where
            R: Runtime,
            $( $ty: MStorageElement + 'static, )+
            ($( $ty, )+): MAlloc<R, Inner = ($( DeviceVec<R, $ty>, )+)>,
        {
            fn inclusive_scan_by_key_values_from_inner<Op, Output>(
                policy: &CubePolicy<R>,
                inner: <Self as MAlloc<R>>::Inner,
                control: &ScanByKeyControl<R>,
                op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                Output: MIterMut<R, Item = Self>,
                Op: op::ReductionOp<R, Self>,
            {
                let _ = op;
                let apply = crate::detail::apply::SegmentedScanApply::new(control);
                $(
                    let $ty = DeviceColumnView::from_column(&inner.$idx);
                )+
                let scanned = apply.$name_inclusive::<$( $ty, )+ KernelOp<R, Op>>(
                    policy,
                    $( &$ty, )+
                )?;
                output.write_from_inner(policy, scanned)
            }

            fn exclusive_scan_by_key_values_from_inner<Op, Output>(
                policy: &CubePolicy<R>,
                inner: <Self as MAlloc<R>>::Inner,
                control: &ScanByKeyControl<R>,
                init: Self,
                op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                Output: MIterMut<R, Item = Self>,
                Op: op::ReductionOp<R, Self>,
            {
                let _ = op;
                let apply = crate::detail::apply::SegmentedScanApply::new(control);
                $(
                    let $ty = DeviceColumnView::from_column(&inner.$idx);
                )+
                let scanned = apply.$name_exclusive::<$( $ty, )+ KernelOp<R, Op>>(
                    policy,
                    $( &$ty, )+
                    init,
                )?;
                output.write_from_inner(policy, scanned)
            }
        }
    };
}

impl_scan_by_key_value_tuple2!(A: 0, B: 1);
impl_scan_by_key_value_tuple3!(A: 0, B: 1, C: 2);
impl_scan_by_key_value_tuple_wide!(inclusive_views4, exclusive_views4; A: 0, B: 1, C: 2, D: 3);
impl_scan_by_key_value_tuple_wide!(inclusive_views5, exclusive_views5; A: 0, B: 1, C: 2, D: 3, E: 4);
impl_scan_by_key_value_tuple_wide!(inclusive_views6, exclusive_views6; A: 0, B: 1, C: 2, D: 3, E: 4, F: 5);
impl_scan_by_key_value_tuple_wide!(inclusive_views7, exclusive_views7; A: 0, B: 1, C: 2, D: 3, E: 4, F: 5, G: 6);

/// Internal logical read expression lowered from `MIter`.
///
/// This is deliberately not a public API concept. Public code sees `MIter`;
/// kernels see a read tree whose leaves are storage-backed columns.
#[doc(hidden)]
pub trait KernelRead<R: Runtime>: Sized {
    type Item: CubeType + crate::expr::LogicalItemPack7 + 'static;

    fn len(&self) -> usize;
    fn validate(&self) -> Result<(), Error>;

    fn reduce_value_read<Op>(
        self,
        policy: &CubePolicy<R>,
        init: Self::Item,
        op: Op,
    ) -> Result<Self::Item, Error>
    where
        Self::Item: MItem<R> + Send + Sync,
        Self: KernelReadBoundMany<R>,
        Op: op::ReductionOp<R, Self::Item>,
    {
        let _ = op;
        reduce_logical7_bound_read::<R, Self, Op>(self, policy, init)
    }

    fn transform_read<Output, Op>(
        self,
        policy: &CubePolicy<R>,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R>,
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Output::Item: MAlloc<R> + MItemDispatch<R>,
        Op: op::UnaryOp<R, Self::Item, Output = Output::Item>,
    {
        let _ = (policy, op, output);
        Err(Error::Launch {
            message: "transform is not supported for this iterator shape".to_string(),
        })
    }

    fn transform_where_read<Output, Op>(
        self,
        policy: &CubePolicy<R>,
        op: Op,
        stencil: PrecomputedSelection<R>,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R>,
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Output::Item: MAlloc<R> + MItemDispatch<R>,
        Op: op::UnaryOp<R, Self::Item, Output = Output::Item>,
    {
        let _ = (policy, op, stencil, output);
        Err(Error::Launch {
            message: "transform_where is not supported for this iterator shape".to_string(),
        })
    }

    fn count_if_read<Pred>(self, policy: &CubePolicy<R>, pred: Pred) -> Result<crate::MIndex, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Pred: op::PredicateOp<R, Self::Item>,
    {
        let _ = (policy, pred);
        Err(Error::Launch {
            message: "count_if is not supported for this iterator shape".to_string(),
        })
    }

    fn all_of_read<Pred>(self, policy: &CubePolicy<R>, pred: Pred) -> Result<bool, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Pred: op::PredicateOp<R, Self::Item>,
    {
        let _ = (policy, pred);
        Err(Error::Launch {
            message: "all_of is not supported for this iterator shape".to_string(),
        })
    }

    fn any_of_read<Pred>(self, policy: &CubePolicy<R>, pred: Pred) -> Result<bool, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Pred: op::PredicateOp<R, Self::Item>,
    {
        let _ = (policy, pred);
        Err(Error::Launch {
            message: "any_of is not supported for this iterator shape".to_string(),
        })
    }

    fn none_of_read<Pred>(self, policy: &CubePolicy<R>, pred: Pred) -> Result<bool, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Pred: op::PredicateOp<R, Self::Item>,
    {
        let _ = (policy, pred);
        Err(Error::Launch {
            message: "none_of is not supported for this iterator shape".to_string(),
        })
    }

    fn find_if_read<Pred>(
        self,
        policy: &CubePolicy<R>,
        pred: Pred,
    ) -> Result<Option<crate::MIndex>, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Pred: op::PredicateOp<R, Self::Item>,
    {
        let _ = (policy, pred);
        Err(Error::Launch {
            message: "find_if is not supported for this iterator shape".to_string(),
        })
    }

    fn is_partitioned_read<Pred>(self, policy: &CubePolicy<R>, pred: Pred) -> Result<bool, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Pred: op::PredicateOp<R, Self::Item>,
    {
        let _ = (policy, pred);
        Err(Error::Launch {
            message: "is_partitioned is not supported for this iterator shape".to_string(),
        })
    }

    fn copy_selected_read<Output>(
        self,
        policy: &CubePolicy<R>,
        stencil: PrecomputedSelection<R>,
        output: Output,
    ) -> Result<crate::MIndex, Error>
    where
        Output: MIterMut<R, Item = Self::Item>,
        Self: KernelReadBoundMany<R>,
        Self::Item: MAlloc<R>,
    {
        <Self::Item as MItemWriteDispatch<R>>::copy_selected_from_read(
            policy, self, stencil, output,
        )
    }

    fn gather_read<Indices, Output>(
        self,
        policy: &CubePolicy<R>,
        indices: Indices,
        output: Output,
    ) -> Result<(), Error>
    where
        Indices: KernelReadBoundMany<R, Item = crate::MIndex>,
        Output: MIterMut<R, Item = Self::Item>,
        Self: KernelReadBoundMany<R>,
        Self::Item: MAlloc<R>,
    {
        <Self::Item as MItemWriteDispatch<R>>::gather_from_read(policy, self, indices, output)
    }

    fn gather_where_read<Indices, Output>(
        self,
        policy: &CubePolicy<R>,
        indices: Indices,
        stencil: PrecomputedSelection<R>,
        output: Output,
    ) -> Result<(), Error>
    where
        Indices: KernelReadBoundMany<R, Item = crate::MIndex>,
        Output: MIterMut<R, Item = Self::Item>,
        Self: KernelReadBoundMany<R>,
        Self::Item: MAlloc<R>,
    {
        <Self::Item as MItemWriteDispatch<R>>::gather_where_from_read(
            policy, self, indices, stencil, output,
        )
    }

    fn scatter_read<Indices, Output>(
        self,
        policy: &CubePolicy<R>,
        indices: Indices,
        output: Output,
    ) -> Result<(), Error>
    where
        Indices: KernelReadBoundMany<R, Item = crate::MIndex>,
        Output: MIterMut<R, Item = Self::Item>,
        Self: KernelReadBoundMany<R>,
        Self::Item: MAlloc<R>,
    {
        <Self::Item as MItemWriteDispatch<R>>::scatter_from_read(policy, self, indices, output)
    }

    fn scatter_where_read<Indices, Output>(
        self,
        policy: &CubePolicy<R>,
        indices: Indices,
        stencil: PrecomputedSelection<R>,
        output: Output,
    ) -> Result<(), Error>
    where
        Indices: KernelReadBoundMany<R, Item = crate::MIndex>,
        Output: MIterMut<R, Item = Self::Item>,
        Self: KernelReadBoundMany<R>,
        Self::Item: MAlloc<R>,
    {
        <Self::Item as MItemWriteDispatch<R>>::scatter_where_from_read(
            policy, self, indices, stencil, output,
        )
    }

    fn unique_read<Pred, Output>(
        self,
        policy: &CubePolicy<R>,
        pred: Pred,
        output: Output,
    ) -> Result<crate::MIndex, Error>
    where
        Output: MIterMut<R, Item = Self::Item>,
        Self: KernelReadBoundMany<R>,
        Self::Item: MAlloc<R>,
        Pred: op::BinaryPredicateOp<R, Self::Item>,
    {
        <Self::Item as MItemWriteDispatch<R>>::unique_from_read(policy, self, pred, output)
    }

    fn adjacent_difference_read<Op, Output>(
        self,
        policy: &CubePolicy<R>,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R, Item = Self::Item>,
        Self: KernelReadBoundMany<R>,
        Self::Item: MAlloc<R>,
        Op: op::ReductionOp<R, Self::Item>,
    {
        <Self::Item as MItemWriteDispatch<R>>::adjacent_difference_from_read(
            policy, self, op, output,
        )
    }

    fn inclusive_scan_read<Op, Output>(
        self,
        policy: &CubePolicy<R>,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R, Item = Self::Item>,
        Self: KernelReadBoundMany<R>,
        Self::Item: MAlloc<R>,
        Op: op::ReductionOp<R, Self::Item>,
    {
        <Self::Item as MItemWriteDispatch<R>>::inclusive_scan_from_read(policy, self, op, output)
    }

    fn exclusive_scan_read<Op, Output>(
        self,
        policy: &CubePolicy<R>,
        init: Self::Item,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R, Item = Self::Item>,
        Self: KernelReadBoundMany<R>,
        Self::Item: MAlloc<R>,
        Op: op::ReductionOp<R, Self::Item>,
    {
        <Self::Item as MItemWriteDispatch<R>>::exclusive_scan_from_read(
            policy, self, init, op, output,
        )
    }

    fn partition_read<Pred, Output>(
        self,
        policy: &CubePolicy<R>,
        pred: Pred,
        output: Output,
    ) -> Result<crate::MIndex, Error>
    where
        Output: MIterMut<R, Item = Self::Item>,
        Self: KernelReadBoundMany<R>,
        Self::Item: MAlloc<R>,
        Pred: op::PredicateOp<R, Self::Item>,
    {
        <Self::Item as MItemWriteDispatch<R>>::partition_from_read(policy, self, pred, output)
    }

    fn adjacent_find_read<Pred>(
        self,
        policy: &CubePolicy<R>,
        pred: Pred,
    ) -> Result<Option<crate::MIndex>, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Pred: op::BinaryPredicateOp<R, Self::Item>,
    {
        let _ = (policy, pred);
        Err(Error::Launch {
            message: "adjacent_find is not supported for this iterator shape".to_string(),
        })
    }

    fn min_element_read<Less>(
        self,
        policy: &CubePolicy<R>,
        _less: Less,
    ) -> Result<Option<crate::MIndex>, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Less: op::BinaryPredicateOp<R, Self::Item>,
    {
        let _ = (policy, _less);
        Err(Error::Launch {
            message: "min_element is not supported for this iterator shape".to_string(),
        })
    }

    fn max_element_read<Less>(
        self,
        policy: &CubePolicy<R>,
        _less: Less,
    ) -> Result<Option<crate::MIndex>, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Less: op::BinaryPredicateOp<R, Self::Item>,
    {
        let _ = (policy, _less);
        Err(Error::Launch {
            message: "max_element is not supported for this iterator shape".to_string(),
        })
    }

    fn minmax_element_read<Less>(
        self,
        policy: &CubePolicy<R>,
        _less: Less,
    ) -> Result<Option<(crate::MIndex, crate::MIndex)>, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Less: op::BinaryPredicateOp<R, Self::Item>,
    {
        let _ = (policy, _less);
        Err(Error::Launch {
            message: "minmax_element is not supported for this iterator shape".to_string(),
        })
    }

    fn is_sorted_read<Less>(self, policy: &CubePolicy<R>, less: Less) -> Result<bool, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Less: op::BinaryPredicateOp<R, Self::Item>,
    {
        let _ = (policy, less);
        Err(Error::Launch {
            message: "is_sorted is not supported for this iterator shape".to_string(),
        })
    }

    fn is_sorted_until_read<Less>(
        self,
        policy: &CubePolicy<R>,
        less: Less,
    ) -> Result<crate::MIndex, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Less: op::BinaryPredicateOp<R, Self::Item>,
    {
        let _ = (policy, less);
        Err(Error::Launch {
            message: "is_sorted_until is not supported for this iterator shape".to_string(),
        })
    }

    fn lower_bound_many_read<Values, Less>(
        self,
        policy: &CubePolicy<R>,
        values: Values,
        less: Less,
    ) -> Result<crate::runtime::DeviceVec<R, crate::MIndex>, Error>
    where
        Values: KernelRead<R, Item = Self::Item>,
        Self::Item: MItem<R>,
        Less: op::BinaryPredicateOp<R, Self::Item>,
    {
        let _ = (policy, values, less);
        Err(Error::Launch {
            message: "lower_bound is not supported for this iterator shape".to_string(),
        })
    }

    fn upper_bound_many_read<Values, Less>(
        self,
        policy: &CubePolicy<R>,
        values: Values,
        less: Less,
    ) -> Result<crate::runtime::DeviceVec<R, crate::MIndex>, Error>
    where
        Values: KernelRead<R, Item = Self::Item>,
        Self::Item: MItem<R>,
        Less: op::BinaryPredicateOp<R, Self::Item>,
    {
        let _ = (policy, values, less);
        Err(Error::Launch {
            message: "upper_bound is not supported for this iterator shape".to_string(),
        })
    }

    fn equal_read<Right, Op>(
        self,
        policy: &CubePolicy<R>,
        right: Right,
        op: Op,
    ) -> Result<bool, Error>
    where
        Right: KernelReadBoundMany<R, Item = Self::Item>,
        Self: KernelReadBoundMany<R>,
        Self::Item: MItem<R>,
        Op: op::BinaryPredicateOp<R, Self::Item>,
    {
        let _ = op;
        Ok(logical7_mismatch_read::<R, Self, Right, Op>(&self, policy, &right)?.is_none())
    }

    fn mismatch_read<Right, Op>(
        self,
        policy: &CubePolicy<R>,
        right: Right,
        op: Op,
    ) -> Result<Option<crate::MIndex>, Error>
    where
        Right: KernelReadBoundMany<R, Item = Self::Item>,
        Self: KernelReadBoundMany<R>,
        Self::Item: MItem<R>,
        Op: op::BinaryPredicateOp<R, Self::Item>,
    {
        let _ = op;
        logical7_mismatch_read::<R, Self, Right, Op>(&self, policy, &right)
    }

    fn find_first_of_read<Right, Op>(
        self,
        policy: &CubePolicy<R>,
        needles: Right,
        op: Op,
    ) -> Result<Option<crate::MIndex>, Error>
    where
        Right: KernelReadBoundMany<R, Item = Self::Item>,
        Self: KernelReadBoundMany<R>,
        Self::Item: MItem<R>,
        Op: op::BinaryPredicateOp<R, Self::Item>,
    {
        let _ = op;
        logical7_find_first_of_read::<R, Self, Right, Op>(&self, policy, &needles)
    }

    fn lexicographical_compare_read<Right, Op>(
        self,
        policy: &CubePolicy<R>,
        right: Right,
        op: Op,
    ) -> Result<bool, Error>
    where
        Right: KernelReadBoundMany<R, Item = Self::Item>,
        Self: KernelReadBoundMany<R>,
        Self::Item: MItem<R>,
        Op: op::BinaryPredicateOp<R, Self::Item>,
    {
        let _ = op;
        logical7_lexicographical_compare_read::<R, Self, Right, Op>(&self, policy, &right)
    }

    fn scan_by_key_control_read<KeyEq>(
        self,
        policy: &CubePolicy<R>,
    ) -> Result<ScanByKeyControl<R>, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        KeyEq: op::BinaryPredicateOp<R, Self::Item>,
    {
        let _ = policy;
        Err(Error::Launch {
            message: "scan_by_key control is not supported for this iterator shape".to_string(),
        })
    }

    fn inclusive_scan_by_key_values_read<KeyEq, Op, Output>(
        self,
        policy: &CubePolicy<R>,
        control: &ScanByKeyControl<R>,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R, Item = Self::Item>,
        Self: KernelReadBoundMany<R>,
        Self::Item: MAlloc<R> + MItemDispatch<R> + Send + Sync,
        Op: op::ReductionOp<R, Self::Item>,
    {
        let inner = materialize_logical7_read(self, policy)?;
        <Self::Item as MAlloc<R>>::inclusive_scan_by_key_values_from_inner::<KeyEq, Op, Output>(
            policy, inner, control, op, output,
        )
    }

    fn exclusive_scan_by_key_values_read<KeyEq, Op, Output>(
        self,
        policy: &CubePolicy<R>,
        control: &ScanByKeyControl<R>,
        init: Self::Item,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R, Item = Self::Item>,
        Self: KernelReadBoundMany<R>,
        Self::Item: MAlloc<R> + MItemDispatch<R> + Send + Sync,
        Op: op::ReductionOp<R, Self::Item>,
    {
        let inner = materialize_logical7_read(self, policy)?;
        <Self::Item as MAlloc<R>>::exclusive_scan_by_key_values_from_inner::<KeyEq, Op, Output>(
            policy, inner, control, init, op, output,
        )
    }

    fn inclusive_scan_by_key_read<Values, KeyEq, Op, Output>(
        self,
        policy: &CubePolicy<R>,
        values: Values,
        key_eq: KeyEq,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Values: KernelReadBoundMany<R>,
        Self: KernelReadBoundMany<R>,
        Output: MIterMut<R, Item = Values::Item>,
        Self::Item: MItem<R>,
        Values::Item: MItem<R> + MAlloc<R> + MItemDispatch<R> + Send + Sync,
        KeyEq: op::BinaryPredicateOp<R, Self::Item>,
        Op: op::ReductionOp<R, Values::Item>,
    {
        let _ = key_eq;
        let control = self.scan_by_key_control_read::<KeyEq>(policy)?;
        values.inclusive_scan_by_key_values_read::<KeyEq, Op, Output>(policy, &control, op, output)
    }

    fn exclusive_scan_by_key_read<Values, KeyEq, Op, Output>(
        self,
        policy: &CubePolicy<R>,
        values: Values,
        key_eq: KeyEq,
        init: Values::Item,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Values: KernelReadBoundMany<R>,
        Self: KernelReadBoundMany<R>,
        Output: MIterMut<R, Item = Values::Item>,
        Self::Item: MItem<R>,
        Values::Item: MItem<R> + MAlloc<R> + MItemDispatch<R> + Send + Sync,
        KeyEq: op::BinaryPredicateOp<R, Self::Item>,
        Op: op::ReductionOp<R, Values::Item>,
    {
        let _ = key_eq;
        let control = self.scan_by_key_control_read::<KeyEq>(policy)?;
        values.exclusive_scan_by_key_values_read::<KeyEq, Op, Output>(
            policy, &control, init, op, output,
        )
    }

    fn stage(&self, policy: &CubePolicy<R>) -> Result<KernelColumnBindings, Error>
    where
        Self: KernelReadAt<R, S0>,
    {
        let mut bindings = KernelColumnBindings::empty(policy.client());
        <Self as KernelReadAt<R, S0>>::stage_at(self, &mut bindings)?;
        bindings.finish();
        Ok(bindings)
    }
}

/// Slot-aware staging for a logical read expression.
#[doc(hidden)]
pub trait KernelReadAt<R: Runtime, Start> {
    type LogicalItem: CubeType + 'static;
    type ExprAt: crate::expr::LogicalDeviceExpr<Self::LogicalItem>;
    type Next;

    fn stage_at(&self, bindings: &mut KernelColumnBindings) -> Result<(), Error>;
}

#[doc(hidden)]
pub struct Env0;
#[doc(hidden)]
pub struct Env1<A>(std::marker::PhantomData<fn() -> A>);
#[doc(hidden)]
pub struct Env2<A, B>(std::marker::PhantomData<fn() -> (A, B)>);
#[doc(hidden)]
pub struct Env3<A, B, C>(std::marker::PhantomData<fn() -> (A, B, C)>);
#[doc(hidden)]
pub struct Env4<A, B, C, D>(std::marker::PhantomData<fn() -> (A, B, C, D)>);
#[doc(hidden)]
pub struct Env5<A, B, C, D, E>(std::marker::PhantomData<fn() -> (A, B, C, D, E)>);
#[doc(hidden)]
pub struct Env6<A, B, C, D, E, F>(std::marker::PhantomData<fn() -> (A, B, C, D, E, F)>);
#[doc(hidden)]
pub struct Env7<A, B, C, D, E, F, G>(std::marker::PhantomData<fn() -> (A, B, C, D, E, F, G)>);

#[doc(hidden)]
pub trait EnvLeaf7 {
    type Leaf0: MStorageElement;
    type Leaf1: MStorageElement;
    type Leaf2: MStorageElement;
    type Leaf3: MStorageElement;
    type Leaf4: MStorageElement;
    type Leaf5: MStorageElement;
    type Leaf6: MStorageElement;
}

impl<A> EnvLeaf7 for Env1<A>
where
    A: MStorageElement,
{
    type Leaf0 = A;
    type Leaf1 = A;
    type Leaf2 = A;
    type Leaf3 = A;
    type Leaf4 = A;
    type Leaf5 = A;
    type Leaf6 = A;
}

impl<A, B> EnvLeaf7 for Env2<A, B>
where
    A: MStorageElement,
    B: MStorageElement,
{
    type Leaf0 = A;
    type Leaf1 = B;
    type Leaf2 = A;
    type Leaf3 = A;
    type Leaf4 = A;
    type Leaf5 = A;
    type Leaf6 = A;
}

impl<A, B, C> EnvLeaf7 for Env3<A, B, C>
where
    A: MStorageElement,
    B: MStorageElement,
    C: MStorageElement,
{
    type Leaf0 = A;
    type Leaf1 = B;
    type Leaf2 = C;
    type Leaf3 = A;
    type Leaf4 = A;
    type Leaf5 = A;
    type Leaf6 = A;
}

impl<A, B, C, D> EnvLeaf7 for Env4<A, B, C, D>
where
    A: MStorageElement,
    B: MStorageElement,
    C: MStorageElement,
    D: MStorageElement,
{
    type Leaf0 = A;
    type Leaf1 = B;
    type Leaf2 = C;
    type Leaf3 = D;
    type Leaf4 = A;
    type Leaf5 = A;
    type Leaf6 = A;
}

impl<A, B, C, D, E> EnvLeaf7 for Env5<A, B, C, D, E>
where
    A: MStorageElement,
    B: MStorageElement,
    C: MStorageElement,
    D: MStorageElement,
    E: MStorageElement,
{
    type Leaf0 = A;
    type Leaf1 = B;
    type Leaf2 = C;
    type Leaf3 = D;
    type Leaf4 = E;
    type Leaf5 = A;
    type Leaf6 = A;
}

impl<A, B, C, D, E, F> EnvLeaf7 for Env6<A, B, C, D, E, F>
where
    A: MStorageElement,
    B: MStorageElement,
    C: MStorageElement,
    D: MStorageElement,
    E: MStorageElement,
    F: MStorageElement,
{
    type Leaf0 = A;
    type Leaf1 = B;
    type Leaf2 = C;
    type Leaf3 = D;
    type Leaf4 = E;
    type Leaf5 = F;
    type Leaf6 = A;
}

impl<A, B, C, D, E, F, G> EnvLeaf7 for Env7<A, B, C, D, E, F, G>
where
    A: MStorageElement,
    B: MStorageElement,
    C: MStorageElement,
    D: MStorageElement,
    E: MStorageElement,
    F: MStorageElement,
    G: MStorageElement,
{
    type Leaf0 = A;
    type Leaf1 = B;
    type Leaf2 = C;
    type Leaf3 = D;
    type Leaf4 = E;
    type Leaf5 = F;
    type Leaf6 = G;
}

#[doc(hidden)]
pub trait KernelReadAtEnv<R: Runtime, Env> {
    type LogicalItem: CubeType + 'static;
    type ExprAt: crate::expr::LogicalDeviceExpr<Self::LogicalItem>;
    type NextEnv: EnvLeaf7;

    fn stage_at_env(&self, bindings: &mut KernelColumnBindings) -> Result<(), Error>;
}

/// Logical subrange of a read expression.
#[doc(hidden)]
pub struct SliceRead<Read> {
    read: Read,
    start: usize,
    len: usize,
}

impl<Read> SliceRead<Read> {
    pub(crate) fn new(read: Read, start: usize, len: usize) -> Self {
        Self { read, start, len }
    }

    fn adjust_offsets_from(&self, bindings: &mut KernelColumnBindings, first_slot: usize) {
        for offset in &mut bindings.slot_offsets[first_slot..] {
            *offset += self.start;
        }
    }
}

impl<R, Read> KernelRead<R> for SliceRead<Read>
where
    R: Runtime,
    Read: KernelRead<R>,
{
    type Item = Read::Item;

    fn len(&self) -> usize {
        self.len
    }

    fn validate(&self) -> Result<(), Error> {
        self.read.validate()?;
        let end = self
            .start
            .checked_add(self.len)
            .ok_or(Error::LengthTooLarge { len: usize::MAX })?;
        if end > self.read.len() {
            return Err(Error::LengthMismatch {
                input: end,
                output: self.read.len(),
            });
        }
        Ok(())
    }
}

impl<R, Read, Start> KernelReadAt<R, Start> for SliceRead<Read>
where
    R: Runtime,
    Read: KernelReadAt<R, Start>,
{
    type LogicalItem = Read::LogicalItem;
    type ExprAt = Read::ExprAt;
    type Next = Read::Next;

    fn stage_at(&self, bindings: &mut KernelColumnBindings) -> Result<(), Error> {
        let first_slot = bindings.slot_offsets.len();
        self.read.stage_at(bindings)?;
        self.adjust_offsets_from(bindings, first_slot);
        Ok(())
    }
}

impl<R, Read, Env> KernelReadAtEnv<R, Env> for SliceRead<Read>
where
    R: Runtime,
    Read: KernelReadAtEnv<R, Env>,
{
    type LogicalItem = Read::LogicalItem;
    type ExprAt = Read::ExprAt;
    type NextEnv = Read::NextEnv;

    fn stage_at_env(&self, bindings: &mut KernelColumnBindings) -> Result<(), Error> {
        let first_slot = bindings.slot_offsets.len();
        self.read.stage_at_env(bindings)?;
        self.adjust_offsets_from(bindings, first_slot);
        Ok(())
    }
}

/// Leaf read expression for a scalar value broadcast over a logical range.
#[doc(hidden)]
pub struct ConstantRead<T> {
    handle: cubecl::server::Handle,
    len: usize,
    _item: std::marker::PhantomData<fn() -> T>,
}

impl<T> ConstantRead<T> {
    pub(crate) fn new(handle: cubecl::server::Handle, len: usize) -> Self {
        Self {
            handle,
            len,
            _item: std::marker::PhantomData,
        }
    }

    fn stage_slot(&self, bindings: &mut KernelColumnBindings) {
        bindings.push(self.handle.clone(), 1);
    }
}

impl<R, T> KernelRead<R> for ConstantRead<T>
where
    R: Runtime,
    T: MStorageElement + crate::expr::LogicalItemPack7 + 'static,
{
    type Item = T;

    fn len(&self) -> usize {
        self.len
    }

    fn validate(&self) -> Result<(), Error> {
        Ok(())
    }

    fn transform_read<Output, Op>(
        self,
        policy: &CubePolicy<R>,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R>,
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Output::Item: MAlloc<R> + MItemDispatch<R>,
        Op: op::UnaryOp<R, Self::Item, Output = Output::Item>,
    {
        transform_logical7_read(self, policy, op, output)
    }

    fn transform_where_read<Output, Op>(
        self,
        policy: &CubePolicy<R>,
        op: Op,
        stencil: PrecomputedSelection<R>,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R>,
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Output::Item: MAlloc<R> + MItemDispatch<R>,
        Op: op::UnaryOp<R, Self::Item, Output = Output::Item>,
    {
        transform_where_logical7_read(self, policy, op, stencil, output)
    }
}

/// Leaf read expression for `start + logical_index`.
#[doc(hidden)]
pub struct CountingRead {
    handle: cubecl::server::Handle,
    len: usize,
}

impl CountingRead {
    pub(crate) fn new(handle: cubecl::server::Handle, len: usize) -> Self {
        Self { handle, len }
    }

    fn stage_slot(&self, bindings: &mut KernelColumnBindings) {
        bindings.push(self.handle.clone(), 1);
    }
}

impl<R> KernelRead<R> for CountingRead
where
    R: Runtime,
{
    type Item = crate::MIndex;

    fn len(&self) -> usize {
        self.len
    }

    fn validate(&self) -> Result<(), Error> {
        Ok(())
    }

    fn transform_read<Output, Op>(
        self,
        policy: &CubePolicy<R>,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R>,
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Output::Item: MAlloc<R> + MItemDispatch<R>,
        Op: op::UnaryOp<R, Self::Item, Output = Output::Item>,
    {
        transform_logical7_read(self, policy, op, output)
    }

    fn transform_where_read<Output, Op>(
        self,
        policy: &CubePolicy<R>,
        op: Op,
        stencil: PrecomputedSelection<R>,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R>,
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Output::Item: MAlloc<R> + MItemDispatch<R>,
        Op: op::UnaryOp<R, Self::Item, Output = Output::Item>,
    {
        transform_where_logical7_read(self, policy, op, stencil, output)
    }
}

/// Lazy read expression for `values[indices[index]]`.
#[doc(hidden)]
pub struct GatherRead<Values, Indices> {
    values: Values,
    indices: Indices,
}

impl<Values, Indices> GatherRead<Values, Indices> {
    pub(crate) fn new(values: Values, indices: Indices) -> Self {
        Self { values, indices }
    }
}

impl<R, Values, Indices> KernelRead<R> for GatherRead<Values, Indices>
where
    R: Runtime,
    Values: KernelRead<R>,
    Indices: KernelRead<R, Item = crate::MIndex>,
{
    type Item = Values::Item;

    fn len(&self) -> usize {
        self.indices.len()
    }

    fn validate(&self) -> Result<(), Error> {
        self.values.validate()?;
        self.indices.validate()
    }

    fn transform_read<Output, Op>(
        self,
        policy: &CubePolicy<R>,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R>,
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Output::Item: MAlloc<R> + MItemDispatch<R>,
        Op: op::UnaryOp<R, Self::Item, Output = Output::Item>,
    {
        transform_logical7_read(self, policy, op, output)
    }

    fn transform_where_read<Output, Op>(
        self,
        policy: &CubePolicy<R>,
        op: Op,
        stencil: PrecomputedSelection<R>,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R>,
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Output::Item: MAlloc<R> + MItemDispatch<R>,
        Op: op::UnaryOp<R, Self::Item, Output = Output::Item>,
    {
        transform_where_logical7_read(self, policy, op, stencil, output)
    }
}

/// Lazy read expression for `op(input[index])`.
#[doc(hidden)]
pub struct TransformRead<Read, Op> {
    read: Read,
    _op: std::marker::PhantomData<fn() -> Op>,
}

impl<Read, Op> TransformRead<Read, Op> {
    pub(crate) fn new(read: Read) -> Self {
        Self {
            read,
            _op: std::marker::PhantomData,
        }
    }
}

impl<R, Read, Op> KernelRead<R> for TransformRead<Read, Op>
where
    R: Runtime,
    Read: KernelRead<R>,
    Read::Item: MItem<R>,
    Op: op::UnaryOp<R, Read::Item>,
    Op::Output: crate::expr::LogicalItemPack7,
{
    type Item = Op::Output;

    fn len(&self) -> usize {
        self.read.len()
    }

    fn validate(&self) -> Result<(), Error> {
        self.read.validate()
    }

    fn transform_read<Output, NextOp>(
        self,
        policy: &CubePolicy<R>,
        op: NextOp,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R>,
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Output::Item: MAlloc<R> + MItemDispatch<R>,
        NextOp: op::UnaryOp<R, Self::Item, Output = Output::Item>,
    {
        transform_logical7_read(self, policy, op, output)
    }

    fn transform_where_read<Output, NextOp>(
        self,
        policy: &CubePolicy<R>,
        op: NextOp,
        stencil: PrecomputedSelection<R>,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R>,
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Output::Item: MAlloc<R> + MItemDispatch<R>,
        NextOp: op::UnaryOp<R, Self::Item, Output = Output::Item>,
    {
        transform_where_logical7_read(self, policy, op, stencil, output)
    }
}

macro_rules! impl_lazy_read_at {
    ($slot:ty, $next:ty, $constant_expr:ty, $counting_expr:ty) => {
        impl<R, T> KernelReadAt<R, $slot> for ConstantRead<T>
        where
            R: Runtime,
            T: MStorageElement + 'static,
        {
            type LogicalItem = T;
            type ExprAt = $constant_expr;
            type Next = $next;

            fn stage_at(&self, bindings: &mut KernelColumnBindings) -> Result<(), Error> {
                self.stage_slot(bindings);
                Ok(())
            }
        }

        impl<R> KernelReadAt<R, $slot> for CountingRead
        where
            R: Runtime,
        {
            type LogicalItem = crate::MIndex;
            type ExprAt = $counting_expr;
            type Next = $next;

            fn stage_at(&self, bindings: &mut KernelColumnBindings) -> Result<(), Error> {
                self.stage_slot(bindings);
                Ok(())
            }
        }
    };
}

impl_lazy_read_at!(
    S0,
    S1,
    crate::expr::ConstantSlot0<T>,
    crate::expr::CountingSlot0
);
impl_lazy_read_at!(
    S1,
    S2,
    crate::expr::ConstantSlot1<T>,
    crate::expr::CountingSlot1
);
impl_lazy_read_at!(
    S2,
    S3,
    crate::expr::ConstantSlot2<T>,
    crate::expr::CountingSlot2
);
impl_lazy_read_at!(
    S3,
    S4,
    crate::expr::ConstantSlot3<T>,
    crate::expr::CountingSlot3
);
impl_lazy_read_at!(
    S4,
    S5,
    crate::expr::ConstantSlot4<T>,
    crate::expr::CountingSlot4
);
impl_lazy_read_at!(
    S5,
    S6,
    crate::expr::ConstantSlot5<T>,
    crate::expr::CountingSlot5
);
impl_lazy_read_at!(
    S6,
    crate::detail::device::S7,
    crate::expr::ConstantSlot6<T>,
    crate::expr::CountingSlot6
);

macro_rules! impl_lazy_read_at_env {
    (impl < $($env_ty:ident),* > $env:ty => $constant_next:ty, $counting_next:ty, $constant_expr:ty, $counting_expr:ty) => {
        impl<R, T, $($env_ty),*> KernelReadAtEnv<R, $env> for ConstantRead<T>
        where
            R: Runtime,
            T: MStorageElement + 'static,
            $($env_ty: MStorageElement + 'static,)*
            $constant_next: EnvLeaf7,
        {
            type LogicalItem = T;
            type ExprAt = $constant_expr;
            type NextEnv = $constant_next;

            fn stage_at_env(&self, bindings: &mut KernelColumnBindings) -> Result<(), Error> {
                self.stage_slot(bindings);
                Ok(())
            }
        }

        impl<R, $($env_ty),*> KernelReadAtEnv<R, $env> for CountingRead
        where
            R: Runtime,
            $($env_ty: MStorageElement + 'static,)*
            $counting_next: EnvLeaf7,
        {
            type LogicalItem = crate::MIndex;
            type ExprAt = $counting_expr;
            type NextEnv = $counting_next;

            fn stage_at_env(&self, bindings: &mut KernelColumnBindings) -> Result<(), Error> {
                self.stage_slot(bindings);
                Ok(())
            }
        }
    };
}

impl_lazy_read_at_env!(impl <> Env0 => Env1<T>, Env1<crate::MIndex>, crate::expr::ConstantSlot0<T>, crate::expr::CountingSlot0);
impl_lazy_read_at_env!(impl <A> Env1<A> => Env2<A, T>, Env2<A, crate::MIndex>, crate::expr::ConstantSlot1<T>, crate::expr::CountingSlot1);
impl_lazy_read_at_env!(impl <A, B> Env2<A, B> => Env3<A, B, T>, Env3<A, B, crate::MIndex>, crate::expr::ConstantSlot2<T>, crate::expr::CountingSlot2);
impl_lazy_read_at_env!(impl <A, B, C> Env3<A, B, C> => Env4<A, B, C, T>, Env4<A, B, C, crate::MIndex>, crate::expr::ConstantSlot3<T>, crate::expr::CountingSlot3);
impl_lazy_read_at_env!(impl <A, B, C, D> Env4<A, B, C, D> => Env5<A, B, C, D, T>, Env5<A, B, C, D, crate::MIndex>, crate::expr::ConstantSlot4<T>, crate::expr::CountingSlot4);
impl_lazy_read_at_env!(impl <A, B, C, D, E> Env5<A, B, C, D, E> => Env6<A, B, C, D, E, T>, Env6<A, B, C, D, E, crate::MIndex>, crate::expr::ConstantSlot5<T>, crate::expr::CountingSlot5);
impl_lazy_read_at_env!(impl <A, B, C, D, E, F> Env6<A, B, C, D, E, F> => Env7<A, B, C, D, E, F, T>, Env7<A, B, C, D, E, F, crate::MIndex>, crate::expr::ConstantSlot6<T>, crate::expr::CountingSlot6);

impl<R, Values, Indices, Start> KernelReadAt<R, Start> for GatherRead<Values, Indices>
where
    R: Runtime,
    Values: KernelReadAt<R, Start>,
    Indices: KernelReadAt<R, <Values as KernelReadAt<R, Start>>::Next, LogicalItem = crate::MIndex>,
{
    type LogicalItem = Values::LogicalItem;
    type ExprAt = crate::expr::GatherExpr<Values::ExprAt, Indices::ExprAt>;
    type Next = Indices::Next;

    fn stage_at(&self, bindings: &mut KernelColumnBindings) -> Result<(), Error> {
        self.values.stage_at(bindings)?;
        self.indices.stage_at(bindings)
    }
}

impl<R, Values, Indices, Env> KernelReadAtEnv<R, Env> for GatherRead<Values, Indices>
where
    R: Runtime,
    Values: KernelReadAtEnv<R, Env>,
    Indices: KernelReadAtEnv<
            R,
            <Values as KernelReadAtEnv<R, Env>>::NextEnv,
            LogicalItem = crate::MIndex,
        >,
{
    type LogicalItem = Values::LogicalItem;
    type ExprAt = crate::expr::GatherExpr<Values::ExprAt, Indices::ExprAt>;
    type NextEnv = Indices::NextEnv;

    fn stage_at_env(&self, bindings: &mut KernelColumnBindings) -> Result<(), Error> {
        self.values.stage_at_env(bindings)?;
        self.indices.stage_at_env(bindings)
    }
}

impl<R, Read, Op, Start> KernelReadAt<R, Start> for TransformRead<Read, Op>
where
    R: Runtime,
    Read: KernelReadAt<R, Start>,
    Read::LogicalItem: MItem<R>,
    Op: op::UnaryOp<R, Read::LogicalItem>,
{
    type LogicalItem = Op::Output;
    type ExprAt = crate::expr::TransformExpr<
        Read::ExprAt,
        Read::LogicalItem,
        crate::detail::op_adapter::KernelOp<R, Op>,
    >;
    type Next = Read::Next;

    fn stage_at(&self, bindings: &mut KernelColumnBindings) -> Result<(), Error> {
        self.read.stage_at(bindings)
    }
}

impl<R, Read, Op, Env> KernelReadAtEnv<R, Env> for TransformRead<Read, Op>
where
    R: Runtime,
    Read: KernelReadAtEnv<R, Env>,
    Read::LogicalItem: MItem<R>,
    Op: op::UnaryOp<R, Read::LogicalItem>,
{
    type LogicalItem = Op::Output;
    type ExprAt = crate::expr::TransformExpr<
        Read::ExprAt,
        Read::LogicalItem,
        crate::detail::op_adapter::KernelOp<R, Op>,
    >;
    type NextEnv = Read::NextEnv;

    fn stage_at_env(&self, bindings: &mut KernelColumnBindings) -> Result<(), Error> {
        self.read.stage_at_env(bindings)
    }
}

#[doc(hidden)]
pub trait KernelReadExpr7At<R: Runtime, Env, FinalEnv>: KernelReadAtEnv<R, Env>
where
    FinalEnv: EnvLeaf7,
    Self::ExprAt: crate::expr::LogicalDeviceExpr7<
            Self::LogicalItem,
            <FinalEnv as EnvLeaf7>::Leaf0,
            <FinalEnv as EnvLeaf7>::Leaf1,
            <FinalEnv as EnvLeaf7>::Leaf2,
            <FinalEnv as EnvLeaf7>::Leaf3,
            <FinalEnv as EnvLeaf7>::Leaf4,
            <FinalEnv as EnvLeaf7>::Leaf5,
            <FinalEnv as EnvLeaf7>::Leaf6,
        >,
{
}

impl<R, Env, FinalEnv, Read> KernelReadExpr7At<R, Env, FinalEnv> for Read
where
    R: Runtime,
    FinalEnv: EnvLeaf7,
    Read: KernelReadAtEnv<R, Env>,
    Read::ExprAt: crate::expr::LogicalDeviceExpr7<
            Read::LogicalItem,
            <FinalEnv as EnvLeaf7>::Leaf0,
            <FinalEnv as EnvLeaf7>::Leaf1,
            <FinalEnv as EnvLeaf7>::Leaf2,
            <FinalEnv as EnvLeaf7>::Leaf3,
            <FinalEnv as EnvLeaf7>::Leaf4,
            <FinalEnv as EnvLeaf7>::Leaf5,
            <FinalEnv as EnvLeaf7>::Leaf6,
        >,
{
}

/// Internal marker for read trees that can be staged as a seven-slot logical
/// expression without changing their logical item shape.
#[doc(hidden)]
pub trait KernelReadLogical7<R: Runtime>:
    KernelRead<R> + KernelReadAt<R, S0, LogicalItem = <Self as KernelRead<R>>::Item>
where
    <Self as KernelReadAt<R, S0>>::ExprAt: crate::expr::LogicalDeviceExpr7Shape<<Self as KernelRead<R>>::Item>
        + crate::expr::LogicalDeviceExpr7<
            <Self as KernelRead<R>>::Item,
            <<Self as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
                <Self as KernelRead<R>>::Item,
            >>::Leaf0,
            <<Self as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
                <Self as KernelRead<R>>::Item,
            >>::Leaf1,
            <<Self as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
                <Self as KernelRead<R>>::Item,
            >>::Leaf2,
            <<Self as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
                <Self as KernelRead<R>>::Item,
            >>::Leaf3,
            <<Self as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
                <Self as KernelRead<R>>::Item,
            >>::Leaf4,
            <<Self as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
                <Self as KernelRead<R>>::Item,
            >>::Leaf5,
            <<Self as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
                <Self as KernelRead<R>>::Item,
            >>::Leaf6,
        >,
    <<Self as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
        <Self as KernelRead<R>>::Item,
    >>::Leaf0: MStorageElement,
    <<Self as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
        <Self as KernelRead<R>>::Item,
    >>::Leaf1: MStorageElement,
    <<Self as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
        <Self as KernelRead<R>>::Item,
    >>::Leaf2: MStorageElement,
    <<Self as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
        <Self as KernelRead<R>>::Item,
    >>::Leaf3: MStorageElement,
    <<Self as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
        <Self as KernelRead<R>>::Item,
    >>::Leaf4: MStorageElement,
    <<Self as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
        <Self as KernelRead<R>>::Item,
    >>::Leaf5: MStorageElement,
    <<Self as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
        <Self as KernelRead<R>>::Item,
    >>::Leaf6: MStorageElement,
{
}

impl<R, Read> KernelReadLogical7<R> for Read
where
    R: Runtime,
    Read: KernelRead<R> + KernelReadAt<R, S0, LogicalItem = <Read as KernelRead<R>>::Item>,
    <Read as KernelReadAt<R, S0>>::ExprAt: crate::expr::LogicalDeviceExpr7Shape<<Read as KernelRead<R>>::Item>
        + crate::expr::LogicalDeviceExpr7<
            <Read as KernelRead<R>>::Item,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
                <Read as KernelRead<R>>::Item,
            >>::Leaf0,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
                <Read as KernelRead<R>>::Item,
            >>::Leaf1,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
                <Read as KernelRead<R>>::Item,
            >>::Leaf2,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
                <Read as KernelRead<R>>::Item,
            >>::Leaf3,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
                <Read as KernelRead<R>>::Item,
            >>::Leaf4,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
                <Read as KernelRead<R>>::Item,
            >>::Leaf5,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
                <Read as KernelRead<R>>::Item,
            >>::Leaf6,
        >,
    <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
        <Read as KernelRead<R>>::Item,
    >>::Leaf0: MStorageElement,
    <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
        <Read as KernelRead<R>>::Item,
    >>::Leaf1: MStorageElement,
    <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
        <Read as KernelRead<R>>::Item,
    >>::Leaf2: MStorageElement,
    <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
        <Read as KernelRead<R>>::Item,
    >>::Leaf3: MStorageElement,
    <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
        <Read as KernelRead<R>>::Item,
    >>::Leaf4: MStorageElement,
    <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
        <Read as KernelRead<R>>::Item,
    >>::Leaf5: MStorageElement,
    <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
        <Read as KernelRead<R>>::Item,
    >>::Leaf6: MStorageElement,
{
}

/// Internal sorted-search lowering for logical read trees.
#[doc(hidden)]
pub trait KernelReadBoundMany<R: Runtime>:
    KernelRead<R> + KernelReadAtEnv<R, Env0, LogicalItem = <Self as KernelRead<R>>::Item>
{
    type Leaf0: MStorageElement;
    type Leaf1: MStorageElement;
    type Leaf2: MStorageElement;
    type Leaf3: MStorageElement;
    type Leaf4: MStorageElement;
    type Leaf5: MStorageElement;
    type Leaf6: MStorageElement;
    type ExprAt: crate::expr::LogicalDeviceExpr7<
            <Self as KernelRead<R>>::Item,
            Self::Leaf0,
            Self::Leaf1,
            Self::Leaf2,
            Self::Leaf3,
            Self::Leaf4,
            Self::Leaf5,
            Self::Leaf6,
        >;

    fn lower_bound_many_logical<Values, Less>(
        self,
        policy: &CubePolicy<R>,
        values: Values,
        less: Less,
    ) -> Result<crate::runtime::DeviceVec<R, crate::MIndex>, Error>
    where
        Values: KernelReadBoundMany<R, Item = Self::Item>,
        Self::Item: MItem<R> + Send + Sync,
        Less: op::BinaryPredicateOp<R, Self::Item>;

    fn upper_bound_many_logical<Values, Less>(
        self,
        policy: &CubePolicy<R>,
        values: Values,
        less: Less,
    ) -> Result<crate::runtime::DeviceVec<R, crate::MIndex>, Error>
    where
        Values: KernelReadBoundMany<R, Item = Self::Item>,
        Self::Item: MItem<R> + Send + Sync,
        Less: op::BinaryPredicateOp<R, Self::Item>;
}

impl<R, Read> KernelReadBoundMany<R> for Read
where
    R: Runtime,
    Read: KernelRead<R> + KernelReadAtEnv<R, Env0, LogicalItem = <Read as KernelRead<R>>::Item>,
    <Read as KernelReadAtEnv<R, Env0>>::NextEnv: EnvLeaf7,
    <Read as KernelReadAtEnv<R, Env0>>::ExprAt: crate::expr::LogicalDeviceExpr7<
            <Read as KernelRead<R>>::Item,
            <<Read as KernelReadAtEnv<R, Env0>>::NextEnv as EnvLeaf7>::Leaf0,
            <<Read as KernelReadAtEnv<R, Env0>>::NextEnv as EnvLeaf7>::Leaf1,
            <<Read as KernelReadAtEnv<R, Env0>>::NextEnv as EnvLeaf7>::Leaf2,
            <<Read as KernelReadAtEnv<R, Env0>>::NextEnv as EnvLeaf7>::Leaf3,
            <<Read as KernelReadAtEnv<R, Env0>>::NextEnv as EnvLeaf7>::Leaf4,
            <<Read as KernelReadAtEnv<R, Env0>>::NextEnv as EnvLeaf7>::Leaf5,
            <<Read as KernelReadAtEnv<R, Env0>>::NextEnv as EnvLeaf7>::Leaf6,
        >,
{
    type Leaf0 = <<Read as KernelReadAtEnv<R, Env0>>::NextEnv as EnvLeaf7>::Leaf0;
    type Leaf1 = <<Read as KernelReadAtEnv<R, Env0>>::NextEnv as EnvLeaf7>::Leaf1;
    type Leaf2 = <<Read as KernelReadAtEnv<R, Env0>>::NextEnv as EnvLeaf7>::Leaf2;
    type Leaf3 = <<Read as KernelReadAtEnv<R, Env0>>::NextEnv as EnvLeaf7>::Leaf3;
    type Leaf4 = <<Read as KernelReadAtEnv<R, Env0>>::NextEnv as EnvLeaf7>::Leaf4;
    type Leaf5 = <<Read as KernelReadAtEnv<R, Env0>>::NextEnv as EnvLeaf7>::Leaf5;
    type Leaf6 = <<Read as KernelReadAtEnv<R, Env0>>::NextEnv as EnvLeaf7>::Leaf6;
    type ExprAt = <Read as KernelReadAtEnv<R, Env0>>::ExprAt;

    fn lower_bound_many_logical<Values, Less>(
        self,
        policy: &CubePolicy<R>,
        values: Values,
        less: Less,
    ) -> Result<crate::runtime::DeviceVec<R, crate::MIndex>, Error>
    where
        Values: KernelReadBoundMany<R, Item = Self::Item>,
        Self::Item: MItem<R> + Send + Sync,
        Less: op::BinaryPredicateOp<R, Self::Item>,
    {
        let _ = less;
        logical7_bound_many_read::<R, Self, Values, Less>(&self, policy, &values, false)
    }

    fn upper_bound_many_logical<Values, Less>(
        self,
        policy: &CubePolicy<R>,
        values: Values,
        less: Less,
    ) -> Result<crate::runtime::DeviceVec<R, crate::MIndex>, Error>
    where
        Values: KernelReadBoundMany<R, Item = Self::Item>,
        Self::Item: MItem<R> + Send + Sync,
        Less: op::BinaryPredicateOp<R, Self::Item>,
    {
        let _ = less;
        logical7_bound_many_read::<R, Self, Values, Less>(&self, policy, &values, true)
    }
}

fn transform_logical3_read<R, Read, Output, Op>(
    read: Read,
    policy: &CubePolicy<R>,
    op: Op,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Read: KernelRead<R> + KernelReadAt<R, S0, LogicalItem = <Read as KernelRead<R>>::Item>,
    <Read as KernelRead<R>>::Item: MItem<R> + Send + Sync,
    Output: MIterMut<R>,
    Output::Item: MAlloc<R> + MItemDispatch<R>,
    <Read as KernelReadAt<R, S0>>::ExprAt: crate::expr::LogicalDeviceExpr3Shape<<Read as KernelRead<R>>::Item>
        + crate::expr::LogicalDeviceExpr3<
            <Read as KernelRead<R>>::Item,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Read as KernelRead<R>>::Item,
            >>::LeafA,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Read as KernelRead<R>>::Item,
            >>::LeafB,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Read as KernelRead<R>>::Item,
            >>::LeafC,
        >,
    <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
        <Read as KernelRead<R>>::Item,
    >>::LeafA: MStorageElement,
    <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
        <Read as KernelRead<R>>::Item,
    >>::LeafB: MStorageElement,
    <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
        <Read as KernelRead<R>>::Item,
    >>::LeafC: MStorageElement,
    Op: op::UnaryOp<R, <Read as KernelRead<R>>::Item, Output = Output::Item>,
{
    let len = read.len();
    let mut bindings = KernelColumnBindings::empty(policy.client());
    <Read as KernelReadAt<R, S0>>::stage_at(&read, &mut bindings)?;
    bindings.finish();
    let inner = <Output::Item as MItemDispatch<R>>::transform_logical3::<
        <Read as KernelRead<R>>::Item,
        <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
            <Read as KernelRead<R>>::Item,
        >>::LeafA,
        <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
            <Read as KernelRead<R>>::Item,
        >>::LeafB,
        <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
            <Read as KernelRead<R>>::Item,
        >>::LeafC,
        <Read as KernelReadAt<R, S0>>::ExprAt,
        Op,
    >(policy, bindings, len, op)?;
    output.write_from_inner(policy, inner)
}

fn transform_where_logical3_read<R, Read, Output, Op>(
    read: Read,
    policy: &CubePolicy<R>,
    op: Op,
    stencil: PrecomputedSelection<R>,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Read: KernelRead<R> + KernelReadAt<R, S0, LogicalItem = <Read as KernelRead<R>>::Item>,
    <Read as KernelRead<R>>::Item: MItem<R> + Send + Sync,
    Output: MIterMut<R>,
    Output::Item: MAlloc<R> + MItemDispatch<R>,
    <Read as KernelReadAt<R, S0>>::ExprAt: crate::expr::LogicalDeviceExpr3Shape<<Read as KernelRead<R>>::Item>
        + crate::expr::LogicalDeviceExpr3<
            <Read as KernelRead<R>>::Item,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Read as KernelRead<R>>::Item,
            >>::LeafA,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Read as KernelRead<R>>::Item,
            >>::LeafB,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Read as KernelRead<R>>::Item,
            >>::LeafC,
        >,
    <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
        <Read as KernelRead<R>>::Item,
    >>::LeafA: MStorageElement,
    <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
        <Read as KernelRead<R>>::Item,
    >>::LeafB: MStorageElement,
    <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
        <Read as KernelRead<R>>::Item,
    >>::LeafC: MStorageElement,
    Op: op::UnaryOp<R, <Read as KernelRead<R>>::Item, Output = Output::Item>,
{
    let len = read.len();
    let mut bindings = KernelColumnBindings::empty(policy.client());
    <Read as KernelReadAt<R, S0>>::stage_at(&read, &mut bindings)?;
    bindings.finish();
    let inner = <Output::Item as MItemDispatch<R>>::transform_logical3::<
        <Read as KernelRead<R>>::Item,
        <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
            <Read as KernelRead<R>>::Item,
        >>::LeafA,
        <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
            <Read as KernelRead<R>>::Item,
        >>::LeafB,
        <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
            <Read as KernelRead<R>>::Item,
        >>::LeafC,
        <Read as KernelReadAt<R, S0>>::ExprAt,
        Op,
    >(policy, bindings, len, op)?;
    output.write_where_from_inner(policy, inner, stencil)
}

fn transform_logical7_read<R, Read, Output, Op>(
    read: Read,
    policy: &CubePolicy<R>,
    op: Op,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Read: KernelReadBoundMany<R>,
    <Read as KernelRead<R>>::Item: MItem<R> + Send + Sync,
    Output: MIterMut<R>,
    Output::Item: MAlloc<R> + MItemDispatch<R>,
    Op: op::UnaryOp<R, <Read as KernelRead<R>>::Item, Output = Output::Item>,
{
    let len = read.len();
    let mut bindings = KernelColumnBindings::empty(policy.client());
    <Read as KernelReadAtEnv<R, Env0>>::stage_at_env(&read, &mut bindings)?;
    bindings.finish();
    let inner = <Output::Item as MItemDispatch<R>>::transform_logical7::<
        <Read as KernelRead<R>>::Item,
        <Read as KernelReadBoundMany<R>>::Leaf0,
        <Read as KernelReadBoundMany<R>>::Leaf1,
        <Read as KernelReadBoundMany<R>>::Leaf2,
        <Read as KernelReadBoundMany<R>>::Leaf3,
        <Read as KernelReadBoundMany<R>>::Leaf4,
        <Read as KernelReadBoundMany<R>>::Leaf5,
        <Read as KernelReadBoundMany<R>>::Leaf6,
        <Read as KernelReadBoundMany<R>>::ExprAt,
        Op,
    >(policy, bindings, len, op)?;
    output.write_from_inner(policy, inner)
}

fn transform_where_logical7_read<R, Read, Output, Op>(
    read: Read,
    policy: &CubePolicy<R>,
    op: Op,
    stencil: PrecomputedSelection<R>,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Read: KernelReadBoundMany<R>,
    <Read as KernelRead<R>>::Item: MItem<R> + Send + Sync,
    Output: MIterMut<R>,
    Output::Item: MAlloc<R> + MItemDispatch<R>,
    Op: op::UnaryOp<R, <Read as KernelRead<R>>::Item, Output = Output::Item>,
{
    let len = read.len();
    let mut bindings = KernelColumnBindings::empty(policy.client());
    <Read as KernelReadAtEnv<R, Env0>>::stage_at_env(&read, &mut bindings)?;
    bindings.finish();
    let inner = <Output::Item as MItemDispatch<R>>::transform_logical7::<
        <Read as KernelRead<R>>::Item,
        <Read as KernelReadBoundMany<R>>::Leaf0,
        <Read as KernelReadBoundMany<R>>::Leaf1,
        <Read as KernelReadBoundMany<R>>::Leaf2,
        <Read as KernelReadBoundMany<R>>::Leaf3,
        <Read as KernelReadBoundMany<R>>::Leaf4,
        <Read as KernelReadBoundMany<R>>::Leaf5,
        <Read as KernelReadBoundMany<R>>::Leaf6,
        <Read as KernelReadBoundMany<R>>::ExprAt,
        Op,
    >(policy, bindings, len, op)?;
    output.write_where_from_inner(policy, inner, stencil)
}

pub(crate) fn copy_selected_logical7_read<R, Read, Output>(
    read: Read,
    policy: &CubePolicy<R>,
    stencil: PrecomputedSelection<R>,
    output: Output,
) -> Result<crate::MIndex, Error>
where
    R: Runtime,
    Read: KernelReadBoundMany<R>,
    <Read as KernelRead<R>>::Item: MAlloc<R>
        + crate::detail::MItemStorage<R>
        + crate::detail::SelectedLogical7Output<
            R,
            <Read as KernelReadBoundMany<R>>::Leaf0,
            <Read as KernelReadBoundMany<R>>::Leaf1,
            <Read as KernelReadBoundMany<R>>::Leaf2,
            <Read as KernelReadBoundMany<R>>::Leaf3,
            <Read as KernelReadBoundMany<R>>::Leaf4,
            <Read as KernelReadBoundMany<R>>::Leaf5,
            <Read as KernelReadBoundMany<R>>::Leaf6,
            <Read as KernelReadBoundMany<R>>::ExprAt,
        > + Send
        + Sync,
    <<Read as KernelRead<R>>::Item as crate::detail::MItemStorage<R>>::Storage:
        crate::detail::MaterializeOutput<
                Runtime = R,
                Output = <<Read as KernelRead<R>>::Item as MAlloc<R>>::Inner,
            >,
    Output: MIterMut<R, Item = <Read as KernelRead<R>>::Item>,
{
    let selected_rank = stencil.selected_rank();
    ensure_same_len(read.len(), selected_rank.len)?;
    let count = select::selected_count(policy, selected_rank)?;
    let mut bindings = KernelColumnBindings::empty(policy.client());
    <Read as KernelReadAtEnv<R, Env0>>::stage_at_env(&read, &mut bindings)?;
    bindings.finish();
    let storage = <<Read as KernelRead<R>>::Item as crate::detail::SelectedLogical7Output<
        R,
        <Read as KernelReadBoundMany<R>>::Leaf0,
        <Read as KernelReadBoundMany<R>>::Leaf1,
        <Read as KernelReadBoundMany<R>>::Leaf2,
        <Read as KernelReadBoundMany<R>>::Leaf3,
        <Read as KernelReadBoundMany<R>>::Leaf4,
        <Read as KernelReadBoundMany<R>>::Leaf5,
        <Read as KernelReadBoundMany<R>>::Leaf6,
        <Read as KernelReadBoundMany<R>>::ExprAt,
    >>::run_selected_logical7(policy, bindings, selected_rank, count)?;
    let inner = crate::detail::MaterializeOutput::materialize_output(storage, policy)?;
    output.write_prefix_from_inner(policy, inner)?;
    mindex_from_usize(count)
}

pub(crate) fn gather_logical7_read<R, Read, Indices, Output>(
    read: Read,
    policy: &CubePolicy<R>,
    indices: Indices,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Read: KernelReadBoundMany<R>,
    Indices: KernelReadBoundMany<R, Item = crate::MIndex>,
    <Read as KernelRead<R>>::Item: MAlloc<R>
        + crate::detail::MItemStorage<R>
        + crate::detail::GatherLogical7Output<
            R,
            <Read as KernelReadBoundMany<R>>::Leaf0,
            <Read as KernelReadBoundMany<R>>::Leaf1,
            <Read as KernelReadBoundMany<R>>::Leaf2,
            <Read as KernelReadBoundMany<R>>::Leaf3,
            <Read as KernelReadBoundMany<R>>::Leaf4,
            <Read as KernelReadBoundMany<R>>::Leaf5,
            <Read as KernelReadBoundMany<R>>::Leaf6,
            <Read as KernelReadBoundMany<R>>::ExprAt,
            <Indices as KernelReadBoundMany<R>>::Leaf0,
            <Indices as KernelReadBoundMany<R>>::Leaf1,
            <Indices as KernelReadBoundMany<R>>::Leaf2,
            <Indices as KernelReadBoundMany<R>>::Leaf3,
            <Indices as KernelReadBoundMany<R>>::Leaf4,
            <Indices as KernelReadBoundMany<R>>::Leaf5,
            <Indices as KernelReadBoundMany<R>>::Leaf6,
            <Indices as KernelReadBoundMany<R>>::ExprAt,
        > + Send
        + Sync,
    <<Read as KernelRead<R>>::Item as crate::detail::MItemStorage<R>>::Storage:
        crate::detail::MaterializeOutput<
                Runtime = R,
                Output = <<Read as KernelRead<R>>::Item as MAlloc<R>>::Inner,
            >,
    Output: MIterMut<R, Item = <Read as KernelRead<R>>::Item>,
{
    let len = indices.len();
    let mut value_bindings = KernelColumnBindings::empty(policy.client());
    <Read as KernelReadAtEnv<R, Env0>>::stage_at_env(&read, &mut value_bindings)?;
    value_bindings.finish();
    let mut index_bindings = KernelColumnBindings::empty(policy.client());
    <Indices as KernelReadAtEnv<R, Env0>>::stage_at_env(&indices, &mut index_bindings)?;
    index_bindings.finish();
    let storage = <<Read as KernelRead<R>>::Item as crate::detail::GatherLogical7Output<
        R,
        <Read as KernelReadBoundMany<R>>::Leaf0,
        <Read as KernelReadBoundMany<R>>::Leaf1,
        <Read as KernelReadBoundMany<R>>::Leaf2,
        <Read as KernelReadBoundMany<R>>::Leaf3,
        <Read as KernelReadBoundMany<R>>::Leaf4,
        <Read as KernelReadBoundMany<R>>::Leaf5,
        <Read as KernelReadBoundMany<R>>::Leaf6,
        <Read as KernelReadBoundMany<R>>::ExprAt,
        <Indices as KernelReadBoundMany<R>>::Leaf0,
        <Indices as KernelReadBoundMany<R>>::Leaf1,
        <Indices as KernelReadBoundMany<R>>::Leaf2,
        <Indices as KernelReadBoundMany<R>>::Leaf3,
        <Indices as KernelReadBoundMany<R>>::Leaf4,
        <Indices as KernelReadBoundMany<R>>::Leaf5,
        <Indices as KernelReadBoundMany<R>>::Leaf6,
        <Indices as KernelReadBoundMany<R>>::ExprAt,
    >>::run_gather_logical7(policy, value_bindings, index_bindings, len)?;
    let inner = crate::detail::MaterializeOutput::materialize_output(storage, policy)?;
    output.write_from_inner(policy, inner)
}

pub(crate) fn gather_where_logical7_read<R, Read, Indices, Output>(
    read: Read,
    policy: &CubePolicy<R>,
    indices: Indices,
    stencil: PrecomputedSelection<R>,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Read: KernelReadBoundMany<R>,
    Indices: KernelReadBoundMany<R, Item = crate::MIndex>,
    <Read as KernelRead<R>>::Item: MAlloc<R>
        + crate::detail::MItemStorage<R>
        + crate::detail::GatherLogical7Output<
            R,
            <Read as KernelReadBoundMany<R>>::Leaf0,
            <Read as KernelReadBoundMany<R>>::Leaf1,
            <Read as KernelReadBoundMany<R>>::Leaf2,
            <Read as KernelReadBoundMany<R>>::Leaf3,
            <Read as KernelReadBoundMany<R>>::Leaf4,
            <Read as KernelReadBoundMany<R>>::Leaf5,
            <Read as KernelReadBoundMany<R>>::Leaf6,
            <Read as KernelReadBoundMany<R>>::ExprAt,
            <Indices as KernelReadBoundMany<R>>::Leaf0,
            <Indices as KernelReadBoundMany<R>>::Leaf1,
            <Indices as KernelReadBoundMany<R>>::Leaf2,
            <Indices as KernelReadBoundMany<R>>::Leaf3,
            <Indices as KernelReadBoundMany<R>>::Leaf4,
            <Indices as KernelReadBoundMany<R>>::Leaf5,
            <Indices as KernelReadBoundMany<R>>::Leaf6,
            <Indices as KernelReadBoundMany<R>>::ExprAt,
        > + Send
        + Sync,
    <<Read as KernelRead<R>>::Item as crate::detail::MItemStorage<R>>::Storage:
        crate::detail::MaterializeOutput<
                Runtime = R,
                Output = <<Read as KernelRead<R>>::Item as MAlloc<R>>::Inner,
            >,
    Output: MIterMut<R, Item = <Read as KernelRead<R>>::Item>,
{
    ensure_same_len(indices.len(), stencil.selected_rank().len)?;
    let len = indices.len();
    let mut value_bindings = KernelColumnBindings::empty(policy.client());
    <Read as KernelReadAtEnv<R, Env0>>::stage_at_env(&read, &mut value_bindings)?;
    value_bindings.finish();
    let mut index_bindings = KernelColumnBindings::empty(policy.client());
    <Indices as KernelReadAtEnv<R, Env0>>::stage_at_env(&indices, &mut index_bindings)?;
    index_bindings.finish();
    let storage = <<Read as KernelRead<R>>::Item as crate::detail::GatherLogical7Output<
        R,
        <Read as KernelReadBoundMany<R>>::Leaf0,
        <Read as KernelReadBoundMany<R>>::Leaf1,
        <Read as KernelReadBoundMany<R>>::Leaf2,
        <Read as KernelReadBoundMany<R>>::Leaf3,
        <Read as KernelReadBoundMany<R>>::Leaf4,
        <Read as KernelReadBoundMany<R>>::Leaf5,
        <Read as KernelReadBoundMany<R>>::Leaf6,
        <Read as KernelReadBoundMany<R>>::ExprAt,
        <Indices as KernelReadBoundMany<R>>::Leaf0,
        <Indices as KernelReadBoundMany<R>>::Leaf1,
        <Indices as KernelReadBoundMany<R>>::Leaf2,
        <Indices as KernelReadBoundMany<R>>::Leaf3,
        <Indices as KernelReadBoundMany<R>>::Leaf4,
        <Indices as KernelReadBoundMany<R>>::Leaf5,
        <Indices as KernelReadBoundMany<R>>::Leaf6,
        <Indices as KernelReadBoundMany<R>>::ExprAt,
    >>::run_gather_logical7(policy, value_bindings, index_bindings, len)?;
    let inner = crate::detail::MaterializeOutput::materialize_output(storage, policy)?;
    output.write_where_from_inner(policy, inner, stencil)
}

pub(crate) fn materialize_logical3_read<R, Read>(
    read: Read,
    policy: &CubePolicy<R>,
) -> Result<<Read::Item as MAlloc<R>>::Inner, Error>
where
    R: Runtime,
    Read: KernelRead<R> + KernelReadAt<R, S0, LogicalItem = <Read as KernelRead<R>>::Item>,
    <Read as KernelRead<R>>::Item: MAlloc<R> + MItemDispatch<R> + Send + Sync,
    <Read as KernelReadAt<R, S0>>::ExprAt: crate::expr::LogicalDeviceExpr3Shape<<Read as KernelRead<R>>::Item>
        + crate::expr::LogicalDeviceExpr3<
            <Read as KernelRead<R>>::Item,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Read as KernelRead<R>>::Item,
            >>::LeafA,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Read as KernelRead<R>>::Item,
            >>::LeafB,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Read as KernelRead<R>>::Item,
            >>::LeafC,
        >,
    <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
        <Read as KernelRead<R>>::Item,
    >>::LeafA: MStorageElement,
    <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
        <Read as KernelRead<R>>::Item,
    >>::LeafB: MStorageElement,
    <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
        <Read as KernelRead<R>>::Item,
    >>::LeafC: MStorageElement,
{
    let len = read.len();
    let mut bindings = KernelColumnBindings::empty(policy.client());
    <Read as KernelReadAt<R, S0>>::stage_at(&read, &mut bindings)?;
    bindings.finish();
    <Read::Item as MItemDispatch<R>>::transform_logical3::<
        <Read as KernelRead<R>>::Item,
        <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
            <Read as KernelRead<R>>::Item,
        >>::LeafA,
        <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
            <Read as KernelRead<R>>::Item,
        >>::LeafB,
        <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
            <Read as KernelRead<R>>::Item,
        >>::LeafC,
        <Read as KernelReadAt<R, S0>>::ExprAt,
        LogicalIdentity,
    >(policy, bindings, len, LogicalIdentity)
}

pub(crate) fn materialize_logical7_read<R, Read>(
    read: Read,
    policy: &CubePolicy<R>,
) -> Result<<Read::Item as MAlloc<R>>::Inner, Error>
where
    R: Runtime,
    Read: KernelReadBoundMany<R>,
    <Read as KernelRead<R>>::Item: MAlloc<R> + MItemDispatch<R> + Send + Sync,
{
    let len = read.len();
    let mut bindings = KernelColumnBindings::empty(policy.client());
    <Read as KernelReadAtEnv<R, Env0>>::stage_at_env(&read, &mut bindings)?;
    bindings.finish();
    <Read::Item as MItemDispatch<R>>::transform_logical7::<
        <Read as KernelRead<R>>::Item,
        <Read as KernelReadBoundMany<R>>::Leaf0,
        <Read as KernelReadBoundMany<R>>::Leaf1,
        <Read as KernelReadBoundMany<R>>::Leaf2,
        <Read as KernelReadBoundMany<R>>::Leaf3,
        <Read as KernelReadBoundMany<R>>::Leaf4,
        <Read as KernelReadBoundMany<R>>::Leaf5,
        <Read as KernelReadBoundMany<R>>::Leaf6,
        <Read as KernelReadBoundMany<R>>::ExprAt,
        LogicalIdentity,
    >(policy, bindings, len, LogicalIdentity)
}

fn reduce_logical3_read<R, Read, Op>(
    read: Read,
    policy: &CubePolicy<R>,
    init: Read::Item,
) -> Result<Read::Item, Error>
where
    R: Runtime,
    Read: KernelRead<R> + KernelReadAt<R, S0, LogicalItem = <Read as KernelRead<R>>::Item>,
    <Read as KernelRead<R>>::Item: MItem<R> + Send + Sync,
    <Read as KernelReadAt<R, S0>>::ExprAt: crate::expr::LogicalDeviceExpr3Shape<<Read as KernelRead<R>>::Item>
        + crate::expr::LogicalDeviceExpr3<
            <Read as KernelRead<R>>::Item,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Read as KernelRead<R>>::Item,
            >>::LeafA,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Read as KernelRead<R>>::Item,
            >>::LeafB,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Read as KernelRead<R>>::Item,
            >>::LeafC,
        > + crate::expr::LogicalDevicePack3<
            <Read as KernelRead<R>>::Item,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Read as KernelRead<R>>::Item,
            >>::LeafA,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Read as KernelRead<R>>::Item,
            >>::LeafB,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Read as KernelRead<R>>::Item,
            >>::LeafC,
        > + crate::expr::LogicalHostPack3<
            <Read as KernelRead<R>>::Item,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Read as KernelRead<R>>::Item,
            >>::LeafA,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Read as KernelRead<R>>::Item,
            >>::LeafB,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Read as KernelRead<R>>::Item,
            >>::LeafC,
        >,
    <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
        <Read as KernelRead<R>>::Item,
    >>::LeafA: MStorageElement,
    <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
        <Read as KernelRead<R>>::Item,
    >>::LeafB: MStorageElement,
    <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
        <Read as KernelRead<R>>::Item,
    >>::LeafC: MStorageElement,
    Op: op::ReductionOp<R, <Read as KernelRead<R>>::Item>,
{
    let len = read.len();
    let mut bindings = KernelColumnBindings::empty(policy.client());
    <Read as KernelReadAt<R, S0>>::stage_at(&read, &mut bindings)?;
    bindings.finish();
    crate::detail::primitives::reduce::reduce_logical3_device_expr::<
        R,
        <Read as KernelRead<R>>::Item,
        <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
            <Read as KernelRead<R>>::Item,
        >>::LeafA,
        <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
            <Read as KernelRead<R>>::Item,
        >>::LeafB,
        <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
            <Read as KernelRead<R>>::Item,
        >>::LeafC,
        <Read as KernelReadAt<R, S0>>::ExprAt,
        <Read as KernelReadAt<R, S0>>::ExprAt,
        KernelOp<R, Op>,
    >(policy, &bindings, len, init)
}

fn reduce_logical7_read<R, Read, Op>(
    read: Read,
    policy: &CubePolicy<R>,
    init: Read::Item,
) -> Result<Read::Item, Error>
where
    R: Runtime,
    Read: KernelRead<R> + KernelReadAt<R, S0, LogicalItem = <Read as KernelRead<R>>::Item>,
    <Read as KernelRead<R>>::Item: MItem<R> + Send + Sync,
    <Read as KernelReadAt<R, S0>>::ExprAt: crate::expr::LogicalDeviceExpr7Shape<<Read as KernelRead<R>>::Item>
        + crate::expr::LogicalDeviceExpr7<
            <Read as KernelRead<R>>::Item,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
                <Read as KernelRead<R>>::Item,
            >>::Leaf0,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
                <Read as KernelRead<R>>::Item,
            >>::Leaf1,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
                <Read as KernelRead<R>>::Item,
            >>::Leaf2,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
                <Read as KernelRead<R>>::Item,
            >>::Leaf3,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
                <Read as KernelRead<R>>::Item,
            >>::Leaf4,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
                <Read as KernelRead<R>>::Item,
            >>::Leaf5,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
                <Read as KernelRead<R>>::Item,
            >>::Leaf6,
        > + crate::expr::LogicalDevicePack7<
            <Read as KernelRead<R>>::Item,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
                <Read as KernelRead<R>>::Item,
            >>::Leaf0,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
                <Read as KernelRead<R>>::Item,
            >>::Leaf1,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
                <Read as KernelRead<R>>::Item,
            >>::Leaf2,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
                <Read as KernelRead<R>>::Item,
            >>::Leaf3,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
                <Read as KernelRead<R>>::Item,
            >>::Leaf4,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
                <Read as KernelRead<R>>::Item,
            >>::Leaf5,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
                <Read as KernelRead<R>>::Item,
            >>::Leaf6,
        > + crate::expr::LogicalHostPack7<
            <Read as KernelRead<R>>::Item,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
                <Read as KernelRead<R>>::Item,
            >>::Leaf0,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
                <Read as KernelRead<R>>::Item,
            >>::Leaf1,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
                <Read as KernelRead<R>>::Item,
            >>::Leaf2,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
                <Read as KernelRead<R>>::Item,
            >>::Leaf3,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
                <Read as KernelRead<R>>::Item,
            >>::Leaf4,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
                <Read as KernelRead<R>>::Item,
            >>::Leaf5,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
                <Read as KernelRead<R>>::Item,
            >>::Leaf6,
        >,
    <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
        <Read as KernelRead<R>>::Item,
    >>::Leaf0: MStorageElement,
    <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
        <Read as KernelRead<R>>::Item,
    >>::Leaf1: MStorageElement,
    <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
        <Read as KernelRead<R>>::Item,
    >>::Leaf2: MStorageElement,
    <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
        <Read as KernelRead<R>>::Item,
    >>::Leaf3: MStorageElement,
    <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
        <Read as KernelRead<R>>::Item,
    >>::Leaf4: MStorageElement,
    <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
        <Read as KernelRead<R>>::Item,
    >>::Leaf5: MStorageElement,
    <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
        <Read as KernelRead<R>>::Item,
    >>::Leaf6: MStorageElement,
    Op: op::ReductionOp<R, <Read as KernelRead<R>>::Item>,
{
    let len = read.len();
    let mut bindings = KernelColumnBindings::empty(policy.client());
    <Read as KernelReadAt<R, S0>>::stage_at(&read, &mut bindings)?;
    bindings.finish();
    crate::detail::primitives::reduce::reduce_logical7_device_expr::<
        R,
        <Read as KernelRead<R>>::Item,
        <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
            <Read as KernelRead<R>>::Item,
        >>::Leaf0,
        <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
            <Read as KernelRead<R>>::Item,
        >>::Leaf1,
        <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
            <Read as KernelRead<R>>::Item,
        >>::Leaf2,
        <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
            <Read as KernelRead<R>>::Item,
        >>::Leaf3,
        <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
            <Read as KernelRead<R>>::Item,
        >>::Leaf4,
        <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
            <Read as KernelRead<R>>::Item,
        >>::Leaf5,
        <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr7Shape<
            <Read as KernelRead<R>>::Item,
        >>::Leaf6,
        <<Read as KernelRead<R>>::Item as crate::expr::LogicalItemPack7>::Leaf0,
        <<Read as KernelRead<R>>::Item as crate::expr::LogicalItemPack7>::Leaf1,
        <<Read as KernelRead<R>>::Item as crate::expr::LogicalItemPack7>::Leaf2,
        <<Read as KernelRead<R>>::Item as crate::expr::LogicalItemPack7>::Leaf3,
        <<Read as KernelRead<R>>::Item as crate::expr::LogicalItemPack7>::Leaf4,
        <<Read as KernelRead<R>>::Item as crate::expr::LogicalItemPack7>::Leaf5,
        <<Read as KernelRead<R>>::Item as crate::expr::LogicalItemPack7>::Leaf6,
        <Read as KernelReadAt<R, S0>>::ExprAt,
        <<Read as KernelRead<R>>::Item as crate::expr::LogicalItemPack7>::Pack,
        KernelOp<R, Op>,
    >(policy, &bindings, len, init)
}

fn reduce_logical7_bound_read<R, Read, Op>(
    read: Read,
    policy: &CubePolicy<R>,
    init: Read::Item,
) -> Result<Read::Item, Error>
where
    R: Runtime,
    Read: KernelReadBoundMany<R>,
    <Read as KernelRead<R>>::Item: MItem<R> + Send + Sync,
    Op: op::ReductionOp<R, <Read as KernelRead<R>>::Item>,
{
    let len = read.len();
    let mut bindings = KernelColumnBindings::empty(policy.client());
    <Read as KernelReadAtEnv<R, Env0>>::stage_at_env(&read, &mut bindings)?;
    bindings.finish();
    crate::detail::primitives::reduce::reduce_logical7_device_expr::<
        R,
        <Read as KernelRead<R>>::Item,
        <Read as KernelReadBoundMany<R>>::Leaf0,
        <Read as KernelReadBoundMany<R>>::Leaf1,
        <Read as KernelReadBoundMany<R>>::Leaf2,
        <Read as KernelReadBoundMany<R>>::Leaf3,
        <Read as KernelReadBoundMany<R>>::Leaf4,
        <Read as KernelReadBoundMany<R>>::Leaf5,
        <Read as KernelReadBoundMany<R>>::Leaf6,
        <<Read as KernelRead<R>>::Item as crate::expr::LogicalItemPack7>::Leaf0,
        <<Read as KernelRead<R>>::Item as crate::expr::LogicalItemPack7>::Leaf1,
        <<Read as KernelRead<R>>::Item as crate::expr::LogicalItemPack7>::Leaf2,
        <<Read as KernelRead<R>>::Item as crate::expr::LogicalItemPack7>::Leaf3,
        <<Read as KernelRead<R>>::Item as crate::expr::LogicalItemPack7>::Leaf4,
        <<Read as KernelRead<R>>::Item as crate::expr::LogicalItemPack7>::Leaf5,
        <<Read as KernelRead<R>>::Item as crate::expr::LogicalItemPack7>::Leaf6,
        <Read as KernelReadBoundMany<R>>::ExprAt,
        <<Read as KernelRead<R>>::Item as crate::expr::LogicalItemPack7>::Pack,
        KernelOp<R, Op>,
    >(policy, &bindings, len, init)
}

fn logical3_predicate_flags_read<R, Read, Pred>(
    read: &Read,
    policy: &CubePolicy<R>,
    invert: bool,
) -> Result<Option<cubecl::server::Handle>, Error>
where
    R: Runtime,
    Read: KernelRead<R> + KernelReadAt<R, S0, LogicalItem = <Read as KernelRead<R>>::Item>,
    <Read as KernelRead<R>>::Item: MItem<R> + Send + Sync,
    <Read as KernelReadAt<R, S0>>::ExprAt: crate::expr::LogicalDeviceExpr3Shape<<Read as KernelRead<R>>::Item>
        + crate::expr::LogicalDeviceExpr3<
            <Read as KernelRead<R>>::Item,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Read as KernelRead<R>>::Item,
            >>::LeafA,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Read as KernelRead<R>>::Item,
            >>::LeafB,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Read as KernelRead<R>>::Item,
            >>::LeafC,
        >,
    <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
        <Read as KernelRead<R>>::Item,
    >>::LeafA: MStorageElement,
    <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
        <Read as KernelRead<R>>::Item,
    >>::LeafB: MStorageElement,
    <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
        <Read as KernelRead<R>>::Item,
    >>::LeafC: MStorageElement,
    Pred: op::PredicateOp<R, <Read as KernelRead<R>>::Item>,
{
    let len = read.len();
    if len == 0 {
        return Ok(None);
    }

    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let client = policy.client();
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let invert_handle = client.create_from_slice(u32::as_bytes(&[if invert { 1_u32 } else { 0 }]));
    let flag_handle = client.empty(len * std::mem::size_of::<u32>());

    let mut bindings = KernelColumnBindings::empty(client);
    <Read as KernelReadAt<R, S0>>::stage_at(read, &mut bindings)?;
    bindings.finish();
    let offsets = bindings.slot_offsets_handle(client)?;
    let slot0 = bindings.slot_or_first(0);
    let slot1 = bindings.slot_or_first(1);
    let slot2 = bindings.slot_or_first(2);
    let block_size = 256_u32;
    let block_count = len.div_ceil(block_size as usize);
    let block_count_u32 =
        u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;

    unsafe {
        crate::kernels::logical3_predicate_flags_kernel::launch_unchecked::<
            <Read as KernelRead<R>>::Item,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Read as KernelRead<R>>::Item,
            >>::LeafA,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Read as KernelRead<R>>::Item,
            >>::LeafB,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Read as KernelRead<R>>::Item,
            >>::LeafC,
            <Read as KernelReadAt<R, S0>>::ExprAt,
            KernelOp<R, Pred>,
            R,
        >(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(block_size),
            unsafe { BufferArg::from_raw_parts(slot0.0.clone(), slot0.1) },
            unsafe { BufferArg::from_raw_parts(slot1.0.clone(), slot1.1) },
            unsafe { BufferArg::from_raw_parts(slot2.0.clone(), slot2.1) },
            unsafe { BufferArg::from_raw_parts(offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(invert_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(flag_handle.clone(), len) },
        );
    }

    Ok(Some(flag_handle))
}

pub(crate) fn logical7_predicate_flags_read<R, Read, Pred>(
    read: &Read,
    policy: &CubePolicy<R>,
    invert: bool,
) -> Result<Option<cubecl::server::Handle>, Error>
where
    R: Runtime,
    Read: KernelReadBoundMany<R>,
    <Read as KernelRead<R>>::Item: MItem<R> + Send + Sync,
    Pred: op::PredicateOp<R, <Read as KernelRead<R>>::Item>,
{
    let len = read.len();
    if len == 0 {
        return Ok(None);
    }

    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let client = policy.client();
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let invert_handle = client.create_from_slice(u32::as_bytes(&[if invert { 1_u32 } else { 0 }]));
    let flag_handle = client.empty(len * std::mem::size_of::<u32>());

    let mut bindings = KernelColumnBindings::empty(client);
    <Read as KernelReadAtEnv<R, Env0>>::stage_at_env(read, &mut bindings)?;
    bindings.finish();
    let offsets = bindings.slot_offsets7_handle(client)?;
    let slot0 = bindings.slot_or_first(0);
    let slot1 = bindings.slot_or_first(1);
    let slot2 = bindings.slot_or_first(2);
    let slot3 = bindings.slot_or_first(3);
    let slot4 = bindings.slot_or_first(4);
    let slot5 = bindings.slot_or_first(5);
    let slot6 = bindings.slot_or_first(6);
    let block_size = 256_u32;
    let block_count = len.div_ceil(block_size as usize);
    let block_count_u32 =
        u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;

    unsafe {
        crate::kernels::logical7_predicate_flags_kernel::launch_unchecked::<
            <Read as KernelRead<R>>::Item,
            <Read as KernelReadBoundMany<R>>::Leaf0,
            <Read as KernelReadBoundMany<R>>::Leaf1,
            <Read as KernelReadBoundMany<R>>::Leaf2,
            <Read as KernelReadBoundMany<R>>::Leaf3,
            <Read as KernelReadBoundMany<R>>::Leaf4,
            <Read as KernelReadBoundMany<R>>::Leaf5,
            <Read as KernelReadBoundMany<R>>::Leaf6,
            <Read as KernelReadBoundMany<R>>::ExprAt,
            KernelOp<R, Pred>,
            R,
        >(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(block_size),
            unsafe { BufferArg::from_raw_parts(slot0.0.clone(), slot0.1) },
            unsafe { BufferArg::from_raw_parts(slot1.0.clone(), slot1.1) },
            unsafe { BufferArg::from_raw_parts(slot2.0.clone(), slot2.1) },
            unsafe { BufferArg::from_raw_parts(slot3.0.clone(), slot3.1) },
            unsafe { BufferArg::from_raw_parts(slot4.0.clone(), slot4.1) },
            unsafe { BufferArg::from_raw_parts(slot5.0.clone(), slot5.1) },
            unsafe { BufferArg::from_raw_parts(slot6.0.clone(), slot6.1) },
            unsafe { BufferArg::from_raw_parts(offsets.clone(), 7) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(invert_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(flag_handle.clone(), len) },
        );
    }

    Ok(Some(flag_handle))
}

pub(crate) fn unique_logical7_flags_read<R, Read, Pred>(
    read: &Read,
    policy: &CubePolicy<R>,
) -> Result<Option<cubecl::server::Handle>, Error>
where
    R: Runtime,
    Read: KernelReadBoundMany<R>,
    <Read as KernelRead<R>>::Item: MItem<R> + Send + Sync,
    Pred: op::BinaryPredicateOp<R, <Read as KernelRead<R>>::Item>,
{
    let len = read.len();
    if len == 0 {
        return Ok(None);
    }

    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let client = policy.client();
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let flag_handle = client.empty(len * std::mem::size_of::<u32>());

    let mut bindings = KernelColumnBindings::empty(client);
    <Read as KernelReadAtEnv<R, Env0>>::stage_at_env(read, &mut bindings)?;
    bindings.finish();
    let offsets = bindings.slot_offsets7_handle(client)?;
    let slot0 = bindings.slot_or_first(0);
    let slot1 = bindings.slot_or_first(1);
    let slot2 = bindings.slot_or_first(2);
    let slot3 = bindings.slot_or_first(3);
    let slot4 = bindings.slot_or_first(4);
    let slot5 = bindings.slot_or_first(5);
    let slot6 = bindings.slot_or_first(6);
    let block_size = 256_u32;
    let block_count = len.div_ceil(block_size as usize);
    let block_count_u32 =
        u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;

    unsafe {
        crate::kernels::unique_logical7_flags_kernel::launch_unchecked::<
            <Read as KernelRead<R>>::Item,
            <Read as KernelReadBoundMany<R>>::Leaf0,
            <Read as KernelReadBoundMany<R>>::Leaf1,
            <Read as KernelReadBoundMany<R>>::Leaf2,
            <Read as KernelReadBoundMany<R>>::Leaf3,
            <Read as KernelReadBoundMany<R>>::Leaf4,
            <Read as KernelReadBoundMany<R>>::Leaf5,
            <Read as KernelReadBoundMany<R>>::Leaf6,
            <Read as KernelReadBoundMany<R>>::ExprAt,
            KernelOp<R, Pred>,
            R,
        >(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(block_size),
            unsafe { BufferArg::from_raw_parts(slot0.0.clone(), slot0.1) },
            unsafe { BufferArg::from_raw_parts(slot1.0.clone(), slot1.1) },
            unsafe { BufferArg::from_raw_parts(slot2.0.clone(), slot2.1) },
            unsafe { BufferArg::from_raw_parts(slot3.0.clone(), slot3.1) },
            unsafe { BufferArg::from_raw_parts(slot4.0.clone(), slot4.1) },
            unsafe { BufferArg::from_raw_parts(slot5.0.clone(), slot5.1) },
            unsafe { BufferArg::from_raw_parts(slot6.0.clone(), slot6.1) },
            unsafe { BufferArg::from_raw_parts(offsets.clone(), 7) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(flag_handle.clone(), len) },
        );
    }

    Ok(Some(flag_handle))
}

pub(crate) fn sort_logical7_indices_read<R, Read, Less>(
    read: &Read,
    policy: &CubePolicy<R>,
) -> Result<DeviceVec<R, crate::MIndex>, Error>
where
    R: Runtime,
    Read: KernelReadBoundMany<R>,
    <Read as KernelRead<R>>::Item: MItem<R> + Send + Sync,
    Less: op::BinaryPredicateOp<R, <Read as KernelRead<R>>::Item>,
{
    let len = read.len();
    if len == 0 {
        return Ok(policy.empty_device_vec());
    }

    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let client = policy.client();
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let indices_handle = client.empty(len * std::mem::size_of::<u32>());

    let mut bindings = KernelColumnBindings::empty(client);
    <Read as KernelReadAtEnv<R, Env0>>::stage_at_env(read, &mut bindings)?;
    bindings.finish();
    let offsets = bindings.slot_offsets7_handle(client)?;
    let slot0 = bindings.slot_or_first(0);
    let slot1 = bindings.slot_or_first(1);
    let slot2 = bindings.slot_or_first(2);
    let slot3 = bindings.slot_or_first(3);
    let slot4 = bindings.slot_or_first(4);
    let slot5 = bindings.slot_or_first(5);
    let slot6 = bindings.slot_or_first(6);
    let block_size = 256_u32;
    let block_count = len.div_ceil(block_size as usize);
    let block_count_u32 =
        u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;

    unsafe {
        crate::kernels::sort_logical7_indices_kernel::launch_unchecked::<
            <Read as KernelRead<R>>::Item,
            <Read as KernelReadBoundMany<R>>::Leaf0,
            <Read as KernelReadBoundMany<R>>::Leaf1,
            <Read as KernelReadBoundMany<R>>::Leaf2,
            <Read as KernelReadBoundMany<R>>::Leaf3,
            <Read as KernelReadBoundMany<R>>::Leaf4,
            <Read as KernelReadBoundMany<R>>::Leaf5,
            <Read as KernelReadBoundMany<R>>::Leaf6,
            <Read as KernelReadBoundMany<R>>::ExprAt,
            KernelOp<R, Less>,
            R,
        >(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(block_size),
            unsafe { BufferArg::from_raw_parts(slot0.0.clone(), slot0.1) },
            unsafe { BufferArg::from_raw_parts(slot1.0.clone(), slot1.1) },
            unsafe { BufferArg::from_raw_parts(slot2.0.clone(), slot2.1) },
            unsafe { BufferArg::from_raw_parts(slot3.0.clone(), slot3.1) },
            unsafe { BufferArg::from_raw_parts(slot4.0.clone(), slot4.1) },
            unsafe { BufferArg::from_raw_parts(slot5.0.clone(), slot5.1) },
            unsafe { BufferArg::from_raw_parts(slot6.0.clone(), slot6.1) },
            unsafe { BufferArg::from_raw_parts(offsets.clone(), 7) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(indices_handle.clone(), len) },
        );
    }

    Ok(DeviceVec::from_handle(policy.id(), indices_handle, len))
}

pub(crate) fn merge_by_key_logical7_indices_read<R, Left, Right, Less>(
    left: &Left,
    right: &Right,
    policy: &CubePolicy<R>,
) -> Result<(DeviceVec<R, crate::MIndex>, DeviceVec<R, crate::MIndex>), Error>
where
    R: Runtime,
    Left: KernelReadBoundMany<R>,
    Right: KernelReadBoundMany<R, Item = <Left as KernelRead<R>>::Item>,
    <Left as KernelRead<R>>::Item: MItem<R> + Send + Sync,
    Less: op::BinaryPredicateOp<R, <Left as KernelRead<R>>::Item>,
{
    let left_len = left.len();
    let right_len = right.len();
    if left_len == 0 {
        return Ok((
            policy.empty_device_vec(),
            crate::detail::primitives::range::indices_mindex(policy, right_len)?,
        ));
    }
    if right_len == 0 {
        return Ok((
            crate::detail::primitives::range::indices_mindex(policy, left_len)?,
            policy.empty_device_vec(),
        ));
    }

    let client = policy.client();
    let left_indices_handle = client.empty(left_len * std::mem::size_of::<u32>());
    let right_indices_handle = client.empty(right_len * std::mem::size_of::<u32>());

    let mut left_bindings = KernelColumnBindings::empty(client);
    <Left as KernelReadAtEnv<R, Env0>>::stage_at_env(left, &mut left_bindings)?;
    left_bindings.finish();
    let mut right_bindings = KernelColumnBindings::empty(client);
    <Right as KernelReadAtEnv<R, Env0>>::stage_at_env(right, &mut right_bindings)?;
    right_bindings.finish();

    let left_offsets = left_bindings.slot_offsets7_handle(client)?;
    let right_offsets = right_bindings.slot_offsets7_handle(client)?;
    let left_slot0 = left_bindings.slot_or_first(0);
    let left_slot1 = left_bindings.slot_or_first(1);
    let left_slot2 = left_bindings.slot_or_first(2);
    let left_slot3 = left_bindings.slot_or_first(3);
    let left_slot4 = left_bindings.slot_or_first(4);
    let left_slot5 = left_bindings.slot_or_first(5);
    let left_slot6 = left_bindings.slot_or_first(6);
    let right_slot0 = right_bindings.slot_or_first(0);
    let right_slot1 = right_bindings.slot_or_first(1);
    let right_slot2 = right_bindings.slot_or_first(2);
    let right_slot3 = right_bindings.slot_or_first(3);
    let right_slot4 = right_bindings.slot_or_first(4);
    let right_slot5 = right_bindings.slot_or_first(5);
    let right_slot6 = right_bindings.slot_or_first(6);

    let metadata = [
        u32::try_from(left_len).map_err(|_| Error::LengthTooLarge { len: left_len })?,
        u32::try_from(right_len).map_err(|_| Error::LengthTooLarge { len: right_len })?,
    ];
    let metadata_handle = client.create_from_slice(u32::as_bytes(&metadata));
    let total_len = left_len + right_len;
    let block_size = 256_u32;
    let block_count = total_len.div_ceil(block_size as usize);
    let block_count_u32 =
        u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;

    unsafe {
        crate::kernels::merge_by_key_logical7_indices_kernel::launch_unchecked::<
            <Left as KernelRead<R>>::Item,
            <Left as KernelReadBoundMany<R>>::Leaf0,
            <Left as KernelReadBoundMany<R>>::Leaf1,
            <Left as KernelReadBoundMany<R>>::Leaf2,
            <Left as KernelReadBoundMany<R>>::Leaf3,
            <Left as KernelReadBoundMany<R>>::Leaf4,
            <Left as KernelReadBoundMany<R>>::Leaf5,
            <Left as KernelReadBoundMany<R>>::Leaf6,
            <Right as KernelReadBoundMany<R>>::Leaf0,
            <Right as KernelReadBoundMany<R>>::Leaf1,
            <Right as KernelReadBoundMany<R>>::Leaf2,
            <Right as KernelReadBoundMany<R>>::Leaf3,
            <Right as KernelReadBoundMany<R>>::Leaf4,
            <Right as KernelReadBoundMany<R>>::Leaf5,
            <Right as KernelReadBoundMany<R>>::Leaf6,
            <Left as KernelReadBoundMany<R>>::ExprAt,
            <Right as KernelReadBoundMany<R>>::ExprAt,
            KernelOp<R, Less>,
            R,
        >(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(block_size),
            unsafe { BufferArg::from_raw_parts(left_slot0.0.clone(), left_slot0.1) },
            unsafe { BufferArg::from_raw_parts(left_slot1.0.clone(), left_slot1.1) },
            unsafe { BufferArg::from_raw_parts(left_slot2.0.clone(), left_slot2.1) },
            unsafe { BufferArg::from_raw_parts(left_slot3.0.clone(), left_slot3.1) },
            unsafe { BufferArg::from_raw_parts(left_slot4.0.clone(), left_slot4.1) },
            unsafe { BufferArg::from_raw_parts(left_slot5.0.clone(), left_slot5.1) },
            unsafe { BufferArg::from_raw_parts(left_slot6.0.clone(), left_slot6.1) },
            unsafe { BufferArg::from_raw_parts(left_offsets.clone(), 7) },
            unsafe { BufferArg::from_raw_parts(right_slot0.0.clone(), right_slot0.1) },
            unsafe { BufferArg::from_raw_parts(right_slot1.0.clone(), right_slot1.1) },
            unsafe { BufferArg::from_raw_parts(right_slot2.0.clone(), right_slot2.1) },
            unsafe { BufferArg::from_raw_parts(right_slot3.0.clone(), right_slot3.1) },
            unsafe { BufferArg::from_raw_parts(right_slot4.0.clone(), right_slot4.1) },
            unsafe { BufferArg::from_raw_parts(right_slot5.0.clone(), right_slot5.1) },
            unsafe { BufferArg::from_raw_parts(right_slot6.0.clone(), right_slot6.1) },
            unsafe { BufferArg::from_raw_parts(right_offsets.clone(), 7) },
            unsafe { BufferArg::from_raw_parts(metadata_handle.clone(), metadata.len()) },
            unsafe { BufferArg::from_raw_parts(left_indices_handle.clone(), left_len) },
            unsafe { BufferArg::from_raw_parts(right_indices_handle.clone(), right_len) },
        );
    }

    Ok((
        DeviceVec::from_handle(policy.id(), left_indices_handle, left_len),
        DeviceVec::from_handle(policy.id(), right_indices_handle, right_len),
    ))
}

pub(crate) fn set_membership_logical7_flags_read<R, Candidate, Sorted, Less>(
    candidate: &Candidate,
    sorted: &Sorted,
    policy: &CubePolicy<R>,
    keep_members: bool,
) -> Result<Option<cubecl::server::Handle>, Error>
where
    R: Runtime,
    Candidate: KernelReadBoundMany<R>,
    Sorted: KernelReadBoundMany<R, Item = <Candidate as KernelRead<R>>::Item>,
    <Candidate as KernelRead<R>>::Item: MItem<R> + Send + Sync,
    Less: op::BinaryPredicateOp<R, <Candidate as KernelRead<R>>::Item>,
{
    let candidate_len = candidate.len();
    if candidate_len == 0 {
        return Ok(None);
    }

    let sorted_len = sorted.len();
    let client = policy.client();
    let flag_handle = client.empty(candidate_len * std::mem::size_of::<u32>());

    let mut candidate_bindings = KernelColumnBindings::empty(client);
    <Candidate as KernelReadAtEnv<R, Env0>>::stage_at_env(candidate, &mut candidate_bindings)?;
    candidate_bindings.finish();
    let mut sorted_bindings = KernelColumnBindings::empty(client);
    <Sorted as KernelReadAtEnv<R, Env0>>::stage_at_env(sorted, &mut sorted_bindings)?;
    sorted_bindings.finish();

    let candidate_offsets = candidate_bindings.slot_offsets7_handle(client)?;
    let sorted_offsets = sorted_bindings.slot_offsets7_handle(client)?;
    let candidate_slot0 = candidate_bindings.slot_or_first(0);
    let candidate_slot1 = candidate_bindings.slot_or_first(1);
    let candidate_slot2 = candidate_bindings.slot_or_first(2);
    let candidate_slot3 = candidate_bindings.slot_or_first(3);
    let candidate_slot4 = candidate_bindings.slot_or_first(4);
    let candidate_slot5 = candidate_bindings.slot_or_first(5);
    let candidate_slot6 = candidate_bindings.slot_or_first(6);
    let sorted_slot0 = sorted_bindings.slot_or_first(0);
    let sorted_slot1 = sorted_bindings.slot_or_first(1);
    let sorted_slot2 = sorted_bindings.slot_or_first(2);
    let sorted_slot3 = sorted_bindings.slot_or_first(3);
    let sorted_slot4 = sorted_bindings.slot_or_first(4);
    let sorted_slot5 = sorted_bindings.slot_or_first(5);
    let sorted_slot6 = sorted_bindings.slot_or_first(6);
    let metadata = [
        u32::try_from(candidate_len).map_err(|_| Error::LengthTooLarge { len: candidate_len })?,
        u32::try_from(sorted_len).map_err(|_| Error::LengthTooLarge { len: sorted_len })?,
        u32::from(keep_members),
    ];
    let metadata_handle = client.create_from_slice(u32::as_bytes(&metadata));
    let block_size = 256_u32;
    let launch = crate::detail::launch::launch_1d(client, candidate_len, block_size)?;

    unsafe {
        crate::kernels::set_membership_logical7_flags_kernel::launch_unchecked::<
            <Candidate as KernelRead<R>>::Item,
            <Candidate as KernelReadBoundMany<R>>::Leaf0,
            <Candidate as KernelReadBoundMany<R>>::Leaf1,
            <Candidate as KernelReadBoundMany<R>>::Leaf2,
            <Candidate as KernelReadBoundMany<R>>::Leaf3,
            <Candidate as KernelReadBoundMany<R>>::Leaf4,
            <Candidate as KernelReadBoundMany<R>>::Leaf5,
            <Candidate as KernelReadBoundMany<R>>::Leaf6,
            <Sorted as KernelReadBoundMany<R>>::Leaf0,
            <Sorted as KernelReadBoundMany<R>>::Leaf1,
            <Sorted as KernelReadBoundMany<R>>::Leaf2,
            <Sorted as KernelReadBoundMany<R>>::Leaf3,
            <Sorted as KernelReadBoundMany<R>>::Leaf4,
            <Sorted as KernelReadBoundMany<R>>::Leaf5,
            <Sorted as KernelReadBoundMany<R>>::Leaf6,
            <Candidate as KernelReadBoundMany<R>>::ExprAt,
            <Sorted as KernelReadBoundMany<R>>::ExprAt,
            KernelOp<R, Less>,
            R,
        >(
            client,
            launch.cube_count(),
            CubeDim::new_1d(block_size),
            unsafe { BufferArg::from_raw_parts(candidate_slot0.0.clone(), candidate_slot0.1) },
            unsafe { BufferArg::from_raw_parts(candidate_slot1.0.clone(), candidate_slot1.1) },
            unsafe { BufferArg::from_raw_parts(candidate_slot2.0.clone(), candidate_slot2.1) },
            unsafe { BufferArg::from_raw_parts(candidate_slot3.0.clone(), candidate_slot3.1) },
            unsafe { BufferArg::from_raw_parts(candidate_slot4.0.clone(), candidate_slot4.1) },
            unsafe { BufferArg::from_raw_parts(candidate_slot5.0.clone(), candidate_slot5.1) },
            unsafe { BufferArg::from_raw_parts(candidate_slot6.0.clone(), candidate_slot6.1) },
            unsafe { BufferArg::from_raw_parts(candidate_offsets.clone(), 7) },
            unsafe { BufferArg::from_raw_parts(sorted_slot0.0.clone(), sorted_slot0.1) },
            unsafe { BufferArg::from_raw_parts(sorted_slot1.0.clone(), sorted_slot1.1) },
            unsafe { BufferArg::from_raw_parts(sorted_slot2.0.clone(), sorted_slot2.1) },
            unsafe { BufferArg::from_raw_parts(sorted_slot3.0.clone(), sorted_slot3.1) },
            unsafe { BufferArg::from_raw_parts(sorted_slot4.0.clone(), sorted_slot4.1) },
            unsafe { BufferArg::from_raw_parts(sorted_slot5.0.clone(), sorted_slot5.1) },
            unsafe { BufferArg::from_raw_parts(sorted_slot6.0.clone(), sorted_slot6.1) },
            unsafe { BufferArg::from_raw_parts(sorted_offsets.clone(), 7) },
            unsafe { BufferArg::from_raw_parts(metadata_handle.clone(), metadata.len()) },
            unsafe { BufferArg::from_raw_parts(flag_handle.clone(), candidate_len) },
        );
    }

    Ok(Some(flag_handle))
}

fn logical3_mismatch_read<R, Left, Right, Eq>(
    left: &Left,
    policy: &CubePolicy<R>,
    right: &Right,
) -> Result<Option<crate::MIndex>, Error>
where
    R: Runtime,
    Left: KernelRead<R> + KernelReadAt<R, S0, LogicalItem = <Left as KernelRead<R>>::Item>,
    Right: KernelRead<R, Item = <Left as KernelRead<R>>::Item>
        + KernelReadAt<R, S0, LogicalItem = <Left as KernelRead<R>>::Item>,
    <Left as KernelRead<R>>::Item: MItem<R> + Send + Sync,
    <Left as KernelReadAt<R, S0>>::ExprAt: crate::expr::LogicalDeviceExpr3Shape<<Left as KernelRead<R>>::Item>
        + crate::expr::LogicalDeviceExpr3<
            <Left as KernelRead<R>>::Item,
            <<Left as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Left as KernelRead<R>>::Item,
            >>::LeafA,
            <<Left as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Left as KernelRead<R>>::Item,
            >>::LeafB,
            <<Left as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Left as KernelRead<R>>::Item,
            >>::LeafC,
        >,
    <Right as KernelReadAt<R, S0>>::ExprAt: crate::expr::LogicalDeviceExpr3Shape<<Left as KernelRead<R>>::Item>
        + crate::expr::LogicalDeviceExpr3<
            <Left as KernelRead<R>>::Item,
            <<Right as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Left as KernelRead<R>>::Item,
            >>::LeafA,
            <<Right as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Left as KernelRead<R>>::Item,
            >>::LeafB,
            <<Right as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Left as KernelRead<R>>::Item,
            >>::LeafC,
        >,
    <<Left as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
        <Left as KernelRead<R>>::Item,
    >>::LeafA: MStorageElement,
    <<Left as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
        <Left as KernelRead<R>>::Item,
    >>::LeafB: MStorageElement,
    <<Left as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
        <Left as KernelRead<R>>::Item,
    >>::LeafC: MStorageElement,
    <<Right as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
        <Left as KernelRead<R>>::Item,
    >>::LeafA: MStorageElement,
    <<Right as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
        <Left as KernelRead<R>>::Item,
    >>::LeafB: MStorageElement,
    <<Right as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
        <Left as KernelRead<R>>::Item,
    >>::LeafC: MStorageElement,
    Eq: op::BinaryPredicateOp<R, <Left as KernelRead<R>>::Item>,
{
    let left_len = left.len();
    let right_len = right.len();
    let min_len = left_len.min(right_len);
    if min_len == 0 {
        return if left_len == right_len {
            Ok(None)
        } else {
            Ok(Some(0))
        };
    }

    let client = policy.client();
    let mut left_bindings = KernelColumnBindings::empty(client);
    <Left as KernelReadAt<R, S0>>::stage_at(left, &mut left_bindings)?;
    left_bindings.finish();
    let mut right_bindings = KernelColumnBindings::empty(client);
    <Right as KernelReadAt<R, S0>>::stage_at(right, &mut right_bindings)?;
    right_bindings.finish();
    let left_offsets = left_bindings.slot_offsets_handle(client)?;
    let right_offsets = right_bindings.slot_offsets_handle(client)?;
    let left_slot0 = left_bindings.slot_or_first(0);
    let left_slot1 = left_bindings.slot_or_first(1);
    let left_slot2 = left_bindings.slot_or_first(2);
    let right_slot0 = right_bindings.slot_or_first(0);
    let right_slot1 = right_bindings.slot_or_first(1);
    let right_slot2 = right_bindings.slot_or_first(2);
    let flags = client.empty(min_len * std::mem::size_of::<u32>());
    let block_size = 256_u32;
    let block_count = min_len.div_ceil(block_size as usize);
    let block_count_u32 =
        u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;
    unsafe {
        crate::kernels::logical3_mismatch_flags_kernel::launch_unchecked::<
            <Left as KernelRead<R>>::Item,
            <<Left as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Left as KernelRead<R>>::Item,
            >>::LeafA,
            <<Left as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Left as KernelRead<R>>::Item,
            >>::LeafB,
            <<Left as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Left as KernelRead<R>>::Item,
            >>::LeafC,
            <<Right as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Left as KernelRead<R>>::Item,
            >>::LeafA,
            <<Right as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Left as KernelRead<R>>::Item,
            >>::LeafB,
            <<Right as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Left as KernelRead<R>>::Item,
            >>::LeafC,
            <Left as KernelReadAt<R, S0>>::ExprAt,
            <Right as KernelReadAt<R, S0>>::ExprAt,
            KernelOp<R, Eq>,
            R,
        >(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(block_size),
            unsafe { BufferArg::from_raw_parts(left_slot0.0.clone(), left_slot0.1) },
            unsafe { BufferArg::from_raw_parts(left_slot1.0.clone(), left_slot1.1) },
            unsafe { BufferArg::from_raw_parts(left_slot2.0.clone(), left_slot2.1) },
            unsafe { BufferArg::from_raw_parts(left_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(right_slot0.0.clone(), right_slot0.1) },
            unsafe { BufferArg::from_raw_parts(right_slot1.0.clone(), right_slot1.1) },
            unsafe { BufferArg::from_raw_parts(right_slot2.0.clone(), right_slot2.1) },
            unsafe { BufferArg::from_raw_parts(right_offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(flags.clone(), min_len) },
        );
    }

    if let Some(index) =
        crate::detail::primitives::search::first_flag(policy, flags, min_len, min_len)?
    {
        return Ok(Some(index));
    }
    if left_len == right_len {
        Ok(None)
    } else {
        Ok(Some(mindex_from_usize(min_len)?))
    }
}

fn logical7_mismatch_read<R, Left, Right, Eq>(
    left: &Left,
    policy: &CubePolicy<R>,
    right: &Right,
) -> Result<Option<crate::MIndex>, Error>
where
    R: Runtime,
    Left: KernelReadBoundMany<R>,
    Right: KernelReadBoundMany<R, Item = <Left as KernelRead<R>>::Item>,
    <Left as KernelRead<R>>::Item: MItem<R> + Send + Sync,
    Eq: op::BinaryPredicateOp<R, <Left as KernelRead<R>>::Item>,
{
    let left_len = left.len();
    let right_len = right.len();
    let min_len = left_len.min(right_len);
    if min_len == 0 {
        return if left_len == right_len {
            Ok(None)
        } else {
            Ok(Some(0))
        };
    }

    let client = policy.client();
    let mut left_bindings = KernelColumnBindings::empty(client);
    <Left as KernelReadAtEnv<R, Env0>>::stage_at_env(left, &mut left_bindings)?;
    left_bindings.finish();
    let mut right_bindings = KernelColumnBindings::empty(client);
    <Right as KernelReadAtEnv<R, Env0>>::stage_at_env(right, &mut right_bindings)?;
    right_bindings.finish();

    let left_offsets = left_bindings.slot_offsets7_handle(client)?;
    let right_offsets = right_bindings.slot_offsets7_handle(client)?;
    let left_slot0 = left_bindings.slot_or_first(0);
    let left_slot1 = left_bindings.slot_or_first(1);
    let left_slot2 = left_bindings.slot_or_first(2);
    let left_slot3 = left_bindings.slot_or_first(3);
    let left_slot4 = left_bindings.slot_or_first(4);
    let left_slot5 = left_bindings.slot_or_first(5);
    let left_slot6 = left_bindings.slot_or_first(6);
    let right_slot0 = right_bindings.slot_or_first(0);
    let right_slot1 = right_bindings.slot_or_first(1);
    let right_slot2 = right_bindings.slot_or_first(2);
    let right_slot3 = right_bindings.slot_or_first(3);
    let right_slot4 = right_bindings.slot_or_first(4);
    let right_slot5 = right_bindings.slot_or_first(5);
    let right_slot6 = right_bindings.slot_or_first(6);
    let flags = client.empty(min_len * std::mem::size_of::<u32>());
    let block_size = 256_u32;
    let block_count = min_len.div_ceil(block_size as usize);
    let block_count_u32 =
        u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;

    unsafe {
        crate::kernels::logical7_mismatch_flags_kernel::launch_unchecked::<
            <Left as KernelRead<R>>::Item,
            <Left as KernelReadBoundMany<R>>::Leaf0,
            <Left as KernelReadBoundMany<R>>::Leaf1,
            <Left as KernelReadBoundMany<R>>::Leaf2,
            <Left as KernelReadBoundMany<R>>::Leaf3,
            <Left as KernelReadBoundMany<R>>::Leaf4,
            <Left as KernelReadBoundMany<R>>::Leaf5,
            <Left as KernelReadBoundMany<R>>::Leaf6,
            <Right as KernelReadBoundMany<R>>::Leaf0,
            <Right as KernelReadBoundMany<R>>::Leaf1,
            <Right as KernelReadBoundMany<R>>::Leaf2,
            <Right as KernelReadBoundMany<R>>::Leaf3,
            <Right as KernelReadBoundMany<R>>::Leaf4,
            <Right as KernelReadBoundMany<R>>::Leaf5,
            <Right as KernelReadBoundMany<R>>::Leaf6,
            <Left as KernelReadBoundMany<R>>::ExprAt,
            <Right as KernelReadBoundMany<R>>::ExprAt,
            KernelOp<R, Eq>,
            R,
        >(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(block_size),
            unsafe { BufferArg::from_raw_parts(left_slot0.0.clone(), left_slot0.1) },
            unsafe { BufferArg::from_raw_parts(left_slot1.0.clone(), left_slot1.1) },
            unsafe { BufferArg::from_raw_parts(left_slot2.0.clone(), left_slot2.1) },
            unsafe { BufferArg::from_raw_parts(left_slot3.0.clone(), left_slot3.1) },
            unsafe { BufferArg::from_raw_parts(left_slot4.0.clone(), left_slot4.1) },
            unsafe { BufferArg::from_raw_parts(left_slot5.0.clone(), left_slot5.1) },
            unsafe { BufferArg::from_raw_parts(left_slot6.0.clone(), left_slot6.1) },
            unsafe { BufferArg::from_raw_parts(left_offsets.clone(), 7) },
            unsafe { BufferArg::from_raw_parts(right_slot0.0.clone(), right_slot0.1) },
            unsafe { BufferArg::from_raw_parts(right_slot1.0.clone(), right_slot1.1) },
            unsafe { BufferArg::from_raw_parts(right_slot2.0.clone(), right_slot2.1) },
            unsafe { BufferArg::from_raw_parts(right_slot3.0.clone(), right_slot3.1) },
            unsafe { BufferArg::from_raw_parts(right_slot4.0.clone(), right_slot4.1) },
            unsafe { BufferArg::from_raw_parts(right_slot5.0.clone(), right_slot5.1) },
            unsafe { BufferArg::from_raw_parts(right_slot6.0.clone(), right_slot6.1) },
            unsafe { BufferArg::from_raw_parts(right_offsets.clone(), 7) },
            unsafe { BufferArg::from_raw_parts(flags.clone(), min_len) },
        );
    }

    if let Some(index) =
        crate::detail::primitives::search::first_flag(policy, flags, min_len, min_len)?
    {
        return Ok(Some(index));
    }
    if left_len == right_len {
        Ok(None)
    } else {
        Ok(Some(mindex_from_usize(min_len)?))
    }
}

fn logical7_find_first_of_read<R, Input, Needles, Eq>(
    input: &Input,
    policy: &CubePolicy<R>,
    needles: &Needles,
) -> Result<Option<crate::MIndex>, Error>
where
    R: Runtime,
    Input: KernelReadBoundMany<R>,
    Needles: KernelReadBoundMany<R, Item = <Input as KernelRead<R>>::Item>,
    <Input as KernelRead<R>>::Item: MItem<R> + Send + Sync,
    Eq: op::BinaryPredicateOp<R, <Input as KernelRead<R>>::Item>,
{
    let input_len = input.len();
    let needle_len = needles.len();
    if input_len == 0 || needle_len == 0 {
        return Ok(None);
    }

    let client = policy.client();
    let mut input_bindings = KernelColumnBindings::empty(client);
    <Input as KernelReadAtEnv<R, Env0>>::stage_at_env(input, &mut input_bindings)?;
    input_bindings.finish();
    let mut needle_bindings = KernelColumnBindings::empty(client);
    <Needles as KernelReadAtEnv<R, Env0>>::stage_at_env(needles, &mut needle_bindings)?;
    needle_bindings.finish();

    let input_offsets = input_bindings.slot_offsets7_handle(client)?;
    let needle_offsets = needle_bindings.slot_offsets7_handle(client)?;
    let input_slot0 = input_bindings.slot_or_first(0);
    let input_slot1 = input_bindings.slot_or_first(1);
    let input_slot2 = input_bindings.slot_or_first(2);
    let input_slot3 = input_bindings.slot_or_first(3);
    let input_slot4 = input_bindings.slot_or_first(4);
    let input_slot5 = input_bindings.slot_or_first(5);
    let input_slot6 = input_bindings.slot_or_first(6);
    let needle_slot0 = needle_bindings.slot_or_first(0);
    let needle_slot1 = needle_bindings.slot_or_first(1);
    let needle_slot2 = needle_bindings.slot_or_first(2);
    let needle_slot3 = needle_bindings.slot_or_first(3);
    let needle_slot4 = needle_bindings.slot_or_first(4);
    let needle_slot5 = needle_bindings.slot_or_first(5);
    let needle_slot6 = needle_bindings.slot_or_first(6);
    let flags = client.empty(input_len * std::mem::size_of::<u32>());
    let needle_len_u32 =
        u32::try_from(needle_len).map_err(|_| Error::LengthTooLarge { len: needle_len })?;
    let needle_len_handle = client.create_from_slice(u32::as_bytes(&[needle_len_u32]));
    let block_size = 256_u32;
    let block_count = input_len.div_ceil(block_size as usize);
    let block_count_u32 =
        u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;

    unsafe {
        crate::kernels::logical7_find_first_of_flags_kernel::launch_unchecked::<
            <Input as KernelRead<R>>::Item,
            <Input as KernelReadBoundMany<R>>::Leaf0,
            <Input as KernelReadBoundMany<R>>::Leaf1,
            <Input as KernelReadBoundMany<R>>::Leaf2,
            <Input as KernelReadBoundMany<R>>::Leaf3,
            <Input as KernelReadBoundMany<R>>::Leaf4,
            <Input as KernelReadBoundMany<R>>::Leaf5,
            <Input as KernelReadBoundMany<R>>::Leaf6,
            <Needles as KernelReadBoundMany<R>>::Leaf0,
            <Needles as KernelReadBoundMany<R>>::Leaf1,
            <Needles as KernelReadBoundMany<R>>::Leaf2,
            <Needles as KernelReadBoundMany<R>>::Leaf3,
            <Needles as KernelReadBoundMany<R>>::Leaf4,
            <Needles as KernelReadBoundMany<R>>::Leaf5,
            <Needles as KernelReadBoundMany<R>>::Leaf6,
            <Input as KernelReadBoundMany<R>>::ExprAt,
            <Needles as KernelReadBoundMany<R>>::ExprAt,
            KernelOp<R, Eq>,
            R,
        >(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(block_size),
            unsafe { BufferArg::from_raw_parts(input_slot0.0.clone(), input_slot0.1) },
            unsafe { BufferArg::from_raw_parts(input_slot1.0.clone(), input_slot1.1) },
            unsafe { BufferArg::from_raw_parts(input_slot2.0.clone(), input_slot2.1) },
            unsafe { BufferArg::from_raw_parts(input_slot3.0.clone(), input_slot3.1) },
            unsafe { BufferArg::from_raw_parts(input_slot4.0.clone(), input_slot4.1) },
            unsafe { BufferArg::from_raw_parts(input_slot5.0.clone(), input_slot5.1) },
            unsafe { BufferArg::from_raw_parts(input_slot6.0.clone(), input_slot6.1) },
            unsafe { BufferArg::from_raw_parts(input_offsets.clone(), 7) },
            unsafe { BufferArg::from_raw_parts(needle_slot0.0.clone(), needle_slot0.1) },
            unsafe { BufferArg::from_raw_parts(needle_slot1.0.clone(), needle_slot1.1) },
            unsafe { BufferArg::from_raw_parts(needle_slot2.0.clone(), needle_slot2.1) },
            unsafe { BufferArg::from_raw_parts(needle_slot3.0.clone(), needle_slot3.1) },
            unsafe { BufferArg::from_raw_parts(needle_slot4.0.clone(), needle_slot4.1) },
            unsafe { BufferArg::from_raw_parts(needle_slot5.0.clone(), needle_slot5.1) },
            unsafe { BufferArg::from_raw_parts(needle_slot6.0.clone(), needle_slot6.1) },
            unsafe { BufferArg::from_raw_parts(needle_offsets.clone(), 7) },
            unsafe { BufferArg::from_raw_parts(needle_len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(flags.clone(), input_len) },
        );
    }

    crate::detail::primitives::search::first_flag(policy, flags, input_len, input_len)
}

fn logical7_lexicographical_compare_read<R, Left, Right, Less>(
    left: &Left,
    policy: &CubePolicy<R>,
    right: &Right,
) -> Result<bool, Error>
where
    R: Runtime,
    Left: KernelReadBoundMany<R>,
    Right: KernelReadBoundMany<R, Item = <Left as KernelRead<R>>::Item>,
    <Left as KernelRead<R>>::Item: MItem<R> + Send + Sync,
    Less: op::BinaryPredicateOp<R, <Left as KernelRead<R>>::Item>,
{
    let left_len = left.len();
    let right_len = right.len();
    let min_len = left_len.min(right_len);
    if min_len == 0 {
        return Ok(left_len < right_len);
    }

    let client = policy.client();
    let mut left_bindings = KernelColumnBindings::empty(client);
    <Left as KernelReadAtEnv<R, Env0>>::stage_at_env(left, &mut left_bindings)?;
    left_bindings.finish();
    let mut right_bindings = KernelColumnBindings::empty(client);
    <Right as KernelReadAtEnv<R, Env0>>::stage_at_env(right, &mut right_bindings)?;
    right_bindings.finish();

    let left_offsets = left_bindings.slot_offsets7_handle(client)?;
    let right_offsets = right_bindings.slot_offsets7_handle(client)?;
    let left_slot0 = left_bindings.slot_or_first(0);
    let left_slot1 = left_bindings.slot_or_first(1);
    let left_slot2 = left_bindings.slot_or_first(2);
    let left_slot3 = left_bindings.slot_or_first(3);
    let left_slot4 = left_bindings.slot_or_first(4);
    let left_slot5 = left_bindings.slot_or_first(5);
    let left_slot6 = left_bindings.slot_or_first(6);
    let right_slot0 = right_bindings.slot_or_first(0);
    let right_slot1 = right_bindings.slot_or_first(1);
    let right_slot2 = right_bindings.slot_or_first(2);
    let right_slot3 = right_bindings.slot_or_first(3);
    let right_slot4 = right_bindings.slot_or_first(4);
    let right_slot5 = right_bindings.slot_or_first(5);
    let right_slot6 = right_bindings.slot_or_first(6);
    let flags = client.empty(min_len * std::mem::size_of::<u32>());
    let block_size = 256_u32;
    let block_count = min_len.div_ceil(block_size as usize);
    let block_count_u32 =
        u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;

    unsafe {
        crate::kernels::logical7_lexicographical_diff_flags_kernel::launch_unchecked::<
            <Left as KernelRead<R>>::Item,
            <Left as KernelReadBoundMany<R>>::Leaf0,
            <Left as KernelReadBoundMany<R>>::Leaf1,
            <Left as KernelReadBoundMany<R>>::Leaf2,
            <Left as KernelReadBoundMany<R>>::Leaf3,
            <Left as KernelReadBoundMany<R>>::Leaf4,
            <Left as KernelReadBoundMany<R>>::Leaf5,
            <Left as KernelReadBoundMany<R>>::Leaf6,
            <Right as KernelReadBoundMany<R>>::Leaf0,
            <Right as KernelReadBoundMany<R>>::Leaf1,
            <Right as KernelReadBoundMany<R>>::Leaf2,
            <Right as KernelReadBoundMany<R>>::Leaf3,
            <Right as KernelReadBoundMany<R>>::Leaf4,
            <Right as KernelReadBoundMany<R>>::Leaf5,
            <Right as KernelReadBoundMany<R>>::Leaf6,
            <Left as KernelReadBoundMany<R>>::ExprAt,
            <Right as KernelReadBoundMany<R>>::ExprAt,
            KernelOp<R, Less>,
            R,
        >(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(block_size),
            unsafe { BufferArg::from_raw_parts(left_slot0.0.clone(), left_slot0.1) },
            unsafe { BufferArg::from_raw_parts(left_slot1.0.clone(), left_slot1.1) },
            unsafe { BufferArg::from_raw_parts(left_slot2.0.clone(), left_slot2.1) },
            unsafe { BufferArg::from_raw_parts(left_slot3.0.clone(), left_slot3.1) },
            unsafe { BufferArg::from_raw_parts(left_slot4.0.clone(), left_slot4.1) },
            unsafe { BufferArg::from_raw_parts(left_slot5.0.clone(), left_slot5.1) },
            unsafe { BufferArg::from_raw_parts(left_slot6.0.clone(), left_slot6.1) },
            unsafe { BufferArg::from_raw_parts(left_offsets.clone(), 7) },
            unsafe { BufferArg::from_raw_parts(right_slot0.0.clone(), right_slot0.1) },
            unsafe { BufferArg::from_raw_parts(right_slot1.0.clone(), right_slot1.1) },
            unsafe { BufferArg::from_raw_parts(right_slot2.0.clone(), right_slot2.1) },
            unsafe { BufferArg::from_raw_parts(right_slot3.0.clone(), right_slot3.1) },
            unsafe { BufferArg::from_raw_parts(right_slot4.0.clone(), right_slot4.1) },
            unsafe { BufferArg::from_raw_parts(right_slot5.0.clone(), right_slot5.1) },
            unsafe { BufferArg::from_raw_parts(right_slot6.0.clone(), right_slot6.1) },
            unsafe { BufferArg::from_raw_parts(right_offsets.clone(), 7) },
            unsafe { BufferArg::from_raw_parts(flags.clone(), min_len) },
        );
    }

    let Some(index) =
        crate::detail::primitives::search::first_flag(policy, flags, min_len, min_len)?
    else {
        return Ok(left_len < right_len);
    };

    let index_u32 = u32::try_from(index).map_err(|_| Error::LengthTooLarge {
        len: usize::try_from(index).unwrap_or(usize::MAX),
    })?;
    let index_handle = client.create_from_slice(u32::as_bytes(&[index_u32]));
    let output = client.empty(std::mem::size_of::<u32>());
    unsafe {
        crate::kernels::logical7_lexicographical_compare_at_kernel::launch_unchecked::<
            <Left as KernelRead<R>>::Item,
            <Left as KernelReadBoundMany<R>>::Leaf0,
            <Left as KernelReadBoundMany<R>>::Leaf1,
            <Left as KernelReadBoundMany<R>>::Leaf2,
            <Left as KernelReadBoundMany<R>>::Leaf3,
            <Left as KernelReadBoundMany<R>>::Leaf4,
            <Left as KernelReadBoundMany<R>>::Leaf5,
            <Left as KernelReadBoundMany<R>>::Leaf6,
            <Right as KernelReadBoundMany<R>>::Leaf0,
            <Right as KernelReadBoundMany<R>>::Leaf1,
            <Right as KernelReadBoundMany<R>>::Leaf2,
            <Right as KernelReadBoundMany<R>>::Leaf3,
            <Right as KernelReadBoundMany<R>>::Leaf4,
            <Right as KernelReadBoundMany<R>>::Leaf5,
            <Right as KernelReadBoundMany<R>>::Leaf6,
            <Left as KernelReadBoundMany<R>>::ExprAt,
            <Right as KernelReadBoundMany<R>>::ExprAt,
            KernelOp<R, Less>,
            R,
        >(
            client,
            CubeCount::new_single(),
            CubeDim::new_1d(1),
            unsafe { BufferArg::from_raw_parts(left_slot0.0.clone(), left_slot0.1) },
            unsafe { BufferArg::from_raw_parts(left_slot1.0.clone(), left_slot1.1) },
            unsafe { BufferArg::from_raw_parts(left_slot2.0.clone(), left_slot2.1) },
            unsafe { BufferArg::from_raw_parts(left_slot3.0.clone(), left_slot3.1) },
            unsafe { BufferArg::from_raw_parts(left_slot4.0.clone(), left_slot4.1) },
            unsafe { BufferArg::from_raw_parts(left_slot5.0.clone(), left_slot5.1) },
            unsafe { BufferArg::from_raw_parts(left_slot6.0.clone(), left_slot6.1) },
            unsafe { BufferArg::from_raw_parts(left_offsets.clone(), 7) },
            unsafe { BufferArg::from_raw_parts(right_slot0.0.clone(), right_slot0.1) },
            unsafe { BufferArg::from_raw_parts(right_slot1.0.clone(), right_slot1.1) },
            unsafe { BufferArg::from_raw_parts(right_slot2.0.clone(), right_slot2.1) },
            unsafe { BufferArg::from_raw_parts(right_slot3.0.clone(), right_slot3.1) },
            unsafe { BufferArg::from_raw_parts(right_slot4.0.clone(), right_slot4.1) },
            unsafe { BufferArg::from_raw_parts(right_slot5.0.clone(), right_slot5.1) },
            unsafe { BufferArg::from_raw_parts(right_slot6.0.clone(), right_slot6.1) },
            unsafe { BufferArg::from_raw_parts(right_offsets.clone(), 7) },
            unsafe { BufferArg::from_raw_parts(index_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(output.clone(), 1) },
        );
    }
    Ok(crate::detail::primitives::scan::read_u32_scalar::<R>(client, output)? != 0)
}

fn logical3_adjacent_find_read<R, Read, Pred>(
    read: &Read,
    policy: &CubePolicy<R>,
) -> Result<Option<crate::MIndex>, Error>
where
    R: Runtime,
    Read: KernelRead<R> + KernelReadAt<R, S0, LogicalItem = <Read as KernelRead<R>>::Item>,
    <Read as KernelRead<R>>::Item: MItem<R> + Send + Sync,
    <Read as KernelReadAt<R, S0>>::ExprAt: crate::expr::LogicalDeviceExpr3Shape<<Read as KernelRead<R>>::Item>
        + crate::expr::LogicalDeviceExpr3<
            <Read as KernelRead<R>>::Item,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Read as KernelRead<R>>::Item,
            >>::LeafA,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Read as KernelRead<R>>::Item,
            >>::LeafB,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Read as KernelRead<R>>::Item,
            >>::LeafC,
        >,
    <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
        <Read as KernelRead<R>>::Item,
    >>::LeafA: MStorageElement,
    <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
        <Read as KernelRead<R>>::Item,
    >>::LeafB: MStorageElement,
    <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
        <Read as KernelRead<R>>::Item,
    >>::LeafC: MStorageElement,
    Pred: op::BinaryPredicateOp<R, <Read as KernelRead<R>>::Item>,
{
    let len = read.len();
    if len < 2 {
        return Ok(None);
    }

    let client = policy.client();
    let flag_len = len - 1;
    let flags = client.empty(flag_len * std::mem::size_of::<u32>());
    let mut bindings = KernelColumnBindings::empty(client);
    <Read as KernelReadAt<R, S0>>::stage_at(read, &mut bindings)?;
    bindings.finish();
    let offsets = bindings.slot_offsets_handle(client)?;
    let slot0 = bindings.slot_or_first(0);
    let slot1 = bindings.slot_or_first(1);
    let slot2 = bindings.slot_or_first(2);
    let block_size = 256_u32;
    let block_count = flag_len.div_ceil(block_size as usize);
    let block_count_u32 =
        u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;

    unsafe {
        crate::kernels::logical3_adjacent_find_flags_kernel::launch_unchecked::<
            <Read as KernelRead<R>>::Item,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Read as KernelRead<R>>::Item,
            >>::LeafA,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Read as KernelRead<R>>::Item,
            >>::LeafB,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Read as KernelRead<R>>::Item,
            >>::LeafC,
            <Read as KernelReadAt<R, S0>>::ExprAt,
            KernelOp<R, Pred>,
            R,
        >(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(block_size),
            unsafe { BufferArg::from_raw_parts(slot0.0.clone(), slot0.1) },
            unsafe { BufferArg::from_raw_parts(slot1.0.clone(), slot1.1) },
            unsafe { BufferArg::from_raw_parts(slot2.0.clone(), slot2.1) },
            unsafe { BufferArg::from_raw_parts(offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(flags.clone(), flag_len) },
        );
    }

    crate::detail::primitives::search::first_flag(policy, flags, flag_len, flag_len)
}

fn logical7_adjacent_find_read<R, Read, Pred>(
    read: &Read,
    policy: &CubePolicy<R>,
) -> Result<Option<crate::MIndex>, Error>
where
    R: Runtime,
    Read: KernelReadBoundMany<R>,
    <Read as KernelRead<R>>::Item: MItem<R> + Send + Sync,
    Pred: op::BinaryPredicateOp<R, <Read as KernelRead<R>>::Item>,
{
    let len = read.len();
    if len < 2 {
        return Ok(None);
    }

    let client = policy.client();
    let flag_len = len - 1;
    let flags = client.empty(flag_len * std::mem::size_of::<u32>());
    let mut bindings = KernelColumnBindings::empty(client);
    <Read as KernelReadAtEnv<R, Env0>>::stage_at_env(read, &mut bindings)?;
    bindings.finish();
    let offsets = bindings.slot_offsets7_handle(client)?;
    let slot0 = bindings.slot_or_first(0);
    let slot1 = bindings.slot_or_first(1);
    let slot2 = bindings.slot_or_first(2);
    let slot3 = bindings.slot_or_first(3);
    let slot4 = bindings.slot_or_first(4);
    let slot5 = bindings.slot_or_first(5);
    let slot6 = bindings.slot_or_first(6);
    let block_size = 256_u32;
    let block_count = flag_len.div_ceil(block_size as usize);
    let block_count_u32 =
        u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;

    unsafe {
        crate::kernels::logical7_adjacent_find_flags_kernel::launch_unchecked::<
            <Read as KernelRead<R>>::Item,
            <Read as KernelReadBoundMany<R>>::Leaf0,
            <Read as KernelReadBoundMany<R>>::Leaf1,
            <Read as KernelReadBoundMany<R>>::Leaf2,
            <Read as KernelReadBoundMany<R>>::Leaf3,
            <Read as KernelReadBoundMany<R>>::Leaf4,
            <Read as KernelReadBoundMany<R>>::Leaf5,
            <Read as KernelReadBoundMany<R>>::Leaf6,
            <Read as KernelReadBoundMany<R>>::ExprAt,
            KernelOp<R, Pred>,
            R,
        >(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(block_size),
            unsafe { BufferArg::from_raw_parts(slot0.0.clone(), slot0.1) },
            unsafe { BufferArg::from_raw_parts(slot1.0.clone(), slot1.1) },
            unsafe { BufferArg::from_raw_parts(slot2.0.clone(), slot2.1) },
            unsafe { BufferArg::from_raw_parts(slot3.0.clone(), slot3.1) },
            unsafe { BufferArg::from_raw_parts(slot4.0.clone(), slot4.1) },
            unsafe { BufferArg::from_raw_parts(slot5.0.clone(), slot5.1) },
            unsafe { BufferArg::from_raw_parts(slot6.0.clone(), slot6.1) },
            unsafe { BufferArg::from_raw_parts(offsets.clone(), 7) },
            unsafe { BufferArg::from_raw_parts(flags.clone(), flag_len) },
        );
    }

    crate::detail::primitives::search::first_flag(policy, flags, flag_len, flag_len)
}

fn logical3_sorted_break_read<R, Read, Less>(
    read: &Read,
    policy: &CubePolicy<R>,
) -> Result<Option<crate::MIndex>, Error>
where
    R: Runtime,
    Read: KernelRead<R> + KernelReadAt<R, S0, LogicalItem = <Read as KernelRead<R>>::Item>,
    <Read as KernelRead<R>>::Item: MItem<R> + Send + Sync,
    <Read as KernelReadAt<R, S0>>::ExprAt: crate::expr::LogicalDeviceExpr3Shape<<Read as KernelRead<R>>::Item>
        + crate::expr::LogicalDeviceExpr3<
            <Read as KernelRead<R>>::Item,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Read as KernelRead<R>>::Item,
            >>::LeafA,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Read as KernelRead<R>>::Item,
            >>::LeafB,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Read as KernelRead<R>>::Item,
            >>::LeafC,
        >,
    <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
        <Read as KernelRead<R>>::Item,
    >>::LeafA: MStorageElement,
    <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
        <Read as KernelRead<R>>::Item,
    >>::LeafB: MStorageElement,
    <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
        <Read as KernelRead<R>>::Item,
    >>::LeafC: MStorageElement,
    Less: op::BinaryPredicateOp<R, <Read as KernelRead<R>>::Item>,
{
    let len = read.len();
    if len < 2 {
        return Ok(None);
    }

    let client = policy.client();
    let flag_len = len - 1;
    let flags = client.empty(flag_len * std::mem::size_of::<u32>());
    let mut bindings = KernelColumnBindings::empty(client);
    <Read as KernelReadAt<R, S0>>::stage_at(read, &mut bindings)?;
    bindings.finish();
    let offsets = bindings.slot_offsets_handle(client)?;
    let slot0 = bindings.slot_or_first(0);
    let slot1 = bindings.slot_or_first(1);
    let slot2 = bindings.slot_or_first(2);
    let block_size = 256_u32;
    let block_count = flag_len.div_ceil(block_size as usize);
    let block_count_u32 =
        u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;

    unsafe {
        crate::kernels::logical3_sorted_break_flags_kernel::launch_unchecked::<
            <Read as KernelRead<R>>::Item,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Read as KernelRead<R>>::Item,
            >>::LeafA,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Read as KernelRead<R>>::Item,
            >>::LeafB,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Read as KernelRead<R>>::Item,
            >>::LeafC,
            <Read as KernelReadAt<R, S0>>::ExprAt,
            KernelOp<R, Less>,
            R,
        >(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(block_size),
            unsafe { BufferArg::from_raw_parts(slot0.0.clone(), slot0.1) },
            unsafe { BufferArg::from_raw_parts(slot1.0.clone(), slot1.1) },
            unsafe { BufferArg::from_raw_parts(slot2.0.clone(), slot2.1) },
            unsafe { BufferArg::from_raw_parts(offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(flags.clone(), flag_len) },
        );
    }

    crate::detail::primitives::search::first_flag(policy, flags, flag_len, flag_len)
}

fn logical7_sorted_break_read<R, Read, Less>(
    read: &Read,
    policy: &CubePolicy<R>,
) -> Result<Option<crate::MIndex>, Error>
where
    R: Runtime,
    Read: KernelReadBoundMany<R>,
    <Read as KernelRead<R>>::Item: MItem<R> + Send + Sync,
    Less: op::BinaryPredicateOp<R, <Read as KernelRead<R>>::Item>,
{
    let len = read.len();
    if len < 2 {
        return Ok(None);
    }

    let client = policy.client();
    let flag_len = len - 1;
    let flags = client.empty(flag_len * std::mem::size_of::<u32>());
    let mut bindings = KernelColumnBindings::empty(client);
    <Read as KernelReadAtEnv<R, Env0>>::stage_at_env(read, &mut bindings)?;
    bindings.finish();
    let offsets = bindings.slot_offsets7_handle(client)?;
    let slot0 = bindings.slot_or_first(0);
    let slot1 = bindings.slot_or_first(1);
    let slot2 = bindings.slot_or_first(2);
    let slot3 = bindings.slot_or_first(3);
    let slot4 = bindings.slot_or_first(4);
    let slot5 = bindings.slot_or_first(5);
    let slot6 = bindings.slot_or_first(6);
    let block_size = 256_u32;
    let block_count = flag_len.div_ceil(block_size as usize);
    let block_count_u32 =
        u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;

    unsafe {
        crate::kernels::logical7_sorted_break_flags_kernel::launch_unchecked::<
            <Read as KernelRead<R>>::Item,
            <Read as KernelReadBoundMany<R>>::Leaf0,
            <Read as KernelReadBoundMany<R>>::Leaf1,
            <Read as KernelReadBoundMany<R>>::Leaf2,
            <Read as KernelReadBoundMany<R>>::Leaf3,
            <Read as KernelReadBoundMany<R>>::Leaf4,
            <Read as KernelReadBoundMany<R>>::Leaf5,
            <Read as KernelReadBoundMany<R>>::Leaf6,
            <Read as KernelReadBoundMany<R>>::ExprAt,
            KernelOp<R, Less>,
            R,
        >(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(block_size),
            unsafe { BufferArg::from_raw_parts(slot0.0.clone(), slot0.1) },
            unsafe { BufferArg::from_raw_parts(slot1.0.clone(), slot1.1) },
            unsafe { BufferArg::from_raw_parts(slot2.0.clone(), slot2.1) },
            unsafe { BufferArg::from_raw_parts(slot3.0.clone(), slot3.1) },
            unsafe { BufferArg::from_raw_parts(slot4.0.clone(), slot4.1) },
            unsafe { BufferArg::from_raw_parts(slot5.0.clone(), slot5.1) },
            unsafe { BufferArg::from_raw_parts(slot6.0.clone(), slot6.1) },
            unsafe { BufferArg::from_raw_parts(offsets.clone(), 7) },
            unsafe { BufferArg::from_raw_parts(flags.clone(), flag_len) },
        );
    }

    crate::detail::primitives::search::first_flag(policy, flags, flag_len, flag_len)
}

fn logical3_minmax_read<R, Read, Less>(
    read: &Read,
    policy: &CubePolicy<R>,
) -> Result<Option<(crate::MIndex, crate::MIndex)>, Error>
where
    R: Runtime,
    Read: KernelRead<R> + KernelReadAt<R, S0, LogicalItem = <Read as KernelRead<R>>::Item>,
    <Read as KernelRead<R>>::Item: MItem<R> + Send + Sync,
    <Read as KernelReadAt<R, S0>>::ExprAt: crate::expr::LogicalDeviceExpr3Shape<<Read as KernelRead<R>>::Item>
        + crate::expr::LogicalDeviceExpr3<
            <Read as KernelRead<R>>::Item,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Read as KernelRead<R>>::Item,
            >>::LeafA,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Read as KernelRead<R>>::Item,
            >>::LeafB,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Read as KernelRead<R>>::Item,
            >>::LeafC,
        >,
    <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
        <Read as KernelRead<R>>::Item,
    >>::LeafA: MStorageElement,
    <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
        <Read as KernelRead<R>>::Item,
    >>::LeafB: MStorageElement,
    <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
        <Read as KernelRead<R>>::Item,
    >>::LeafC: MStorageElement,
    Less: op::BinaryPredicateOp<R, <Read as KernelRead<R>>::Item>,
{
    let len = read.len();
    if len == 0 {
        return Ok(None);
    }

    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let client = policy.client();
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let min_flags = client.empty(len * std::mem::size_of::<u32>());
    let max_flags = client.empty(len * std::mem::size_of::<u32>());

    let mut bindings = KernelColumnBindings::empty(client);
    <Read as KernelReadAt<R, S0>>::stage_at(read, &mut bindings)?;
    bindings.finish();
    let offsets = bindings.slot_offsets_handle(client)?;
    let slot0 = bindings.slot_or_first(0);
    let slot1 = bindings.slot_or_first(1);
    let slot2 = bindings.slot_or_first(2);
    let block_size = 256_u32;
    let block_count = len.div_ceil(block_size as usize);
    let block_count_u32 =
        u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;

    unsafe {
        crate::kernels::logical3_minmax_flags_kernel::launch_unchecked::<
            <Read as KernelRead<R>>::Item,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Read as KernelRead<R>>::Item,
            >>::LeafA,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Read as KernelRead<R>>::Item,
            >>::LeafB,
            <<Read as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Read as KernelRead<R>>::Item,
            >>::LeafC,
            <Read as KernelReadAt<R, S0>>::ExprAt,
            KernelOp<R, Less>,
            R,
        >(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(block_size),
            unsafe { BufferArg::from_raw_parts(slot0.0.clone(), slot0.1) },
            unsafe { BufferArg::from_raw_parts(slot1.0.clone(), slot1.1) },
            unsafe { BufferArg::from_raw_parts(slot2.0.clone(), slot2.1) },
            unsafe { BufferArg::from_raw_parts(offsets.clone(), 4) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(min_flags.clone(), len) },
            unsafe { BufferArg::from_raw_parts(max_flags.clone(), len) },
        );
    }

    let min = crate::detail::primitives::search::first_flag(policy, min_flags, len, len)?
        .expect("non-empty min flags must contain one minimum");
    let max = crate::detail::primitives::search::first_flag(policy, max_flags, len, len)?
        .expect("non-empty max flags must contain one maximum");
    Ok(Some((min, max)))
}

fn logical7_minmax_read<R, Read, Less>(
    read: &Read,
    policy: &CubePolicy<R>,
) -> Result<Option<(crate::MIndex, crate::MIndex)>, Error>
where
    R: Runtime,
    Read: KernelReadBoundMany<R>,
    <Read as KernelRead<R>>::Item: MItem<R> + Send + Sync,
    Less: op::BinaryPredicateOp<R, <Read as KernelRead<R>>::Item>,
{
    let len = read.len();
    if len == 0 {
        return Ok(None);
    }

    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let client = policy.client();
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));

    let mut bindings = KernelColumnBindings::empty(client);
    <Read as KernelReadAtEnv<R, Env0>>::stage_at_env(read, &mut bindings)?;
    bindings.finish();
    let offsets = bindings.slot_offsets7_handle(client)?;
    let slot0 = bindings.slot_or_first(0);
    let slot1 = bindings.slot_or_first(1);
    let slot2 = bindings.slot_or_first(2);
    let slot3 = bindings.slot_or_first(3);
    let slot4 = bindings.slot_or_first(4);
    let slot5 = bindings.slot_or_first(5);
    let slot6 = bindings.slot_or_first(6);
    let block_size = 256_u32;
    let mut current_count = len.div_ceil(block_size as usize);
    let mut current_count_u32 =
        u32::try_from(current_count).map_err(|_| Error::LengthTooLarge { len: current_count })?;
    let mut current_handle = client.empty(current_count * 2 * std::mem::size_of::<u32>());
    let launch = crate::detail::launch::launch_1d(client, len, block_size)?;

    unsafe {
        crate::kernels::logical7_minmax_partials_kernel::launch_unchecked::<
            <Read as KernelRead<R>>::Item,
            <Read as KernelReadBoundMany<R>>::Leaf0,
            <Read as KernelReadBoundMany<R>>::Leaf1,
            <Read as KernelReadBoundMany<R>>::Leaf2,
            <Read as KernelReadBoundMany<R>>::Leaf3,
            <Read as KernelReadBoundMany<R>>::Leaf4,
            <Read as KernelReadBoundMany<R>>::Leaf5,
            <Read as KernelReadBoundMany<R>>::Leaf6,
            <Read as KernelReadBoundMany<R>>::ExprAt,
            KernelOp<R, Less>,
            R,
        >(
            client,
            launch.cube_count(),
            CubeDim::new_1d(block_size),
            unsafe { BufferArg::from_raw_parts(slot0.0.clone(), slot0.1) },
            unsafe { BufferArg::from_raw_parts(slot1.0.clone(), slot1.1) },
            unsafe { BufferArg::from_raw_parts(slot2.0.clone(), slot2.1) },
            unsafe { BufferArg::from_raw_parts(slot3.0.clone(), slot3.1) },
            unsafe { BufferArg::from_raw_parts(slot4.0.clone(), slot4.1) },
            unsafe { BufferArg::from_raw_parts(slot5.0.clone(), slot5.1) },
            unsafe { BufferArg::from_raw_parts(slot6.0.clone(), slot6.1) },
            unsafe { BufferArg::from_raw_parts(offsets.clone(), 7) },
            unsafe { BufferArg::from_raw_parts(len_handle.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(current_handle.clone(), current_count * 2) },
        );
    }

    while current_count > 1 {
        let next_count = current_count.div_ceil(block_size as usize);
        let candidate_len_handle = client.create_from_slice(u32::as_bytes(&[current_count_u32]));
        let next_handle = client.empty(next_count * 2 * std::mem::size_of::<u32>());
        let launch = crate::detail::launch::launch_1d(client, current_count, block_size)?;

        unsafe {
            crate::kernels::logical7_minmax_index_partials_kernel::launch_unchecked::<
                <Read as KernelRead<R>>::Item,
                <Read as KernelReadBoundMany<R>>::Leaf0,
                <Read as KernelReadBoundMany<R>>::Leaf1,
                <Read as KernelReadBoundMany<R>>::Leaf2,
                <Read as KernelReadBoundMany<R>>::Leaf3,
                <Read as KernelReadBoundMany<R>>::Leaf4,
                <Read as KernelReadBoundMany<R>>::Leaf5,
                <Read as KernelReadBoundMany<R>>::Leaf6,
                <Read as KernelReadBoundMany<R>>::ExprAt,
                KernelOp<R, Less>,
                R,
            >(
                client,
                launch.cube_count(),
                CubeDim::new_1d(block_size),
                unsafe { BufferArg::from_raw_parts(slot0.0.clone(), slot0.1) },
                unsafe { BufferArg::from_raw_parts(slot1.0.clone(), slot1.1) },
                unsafe { BufferArg::from_raw_parts(slot2.0.clone(), slot2.1) },
                unsafe { BufferArg::from_raw_parts(slot3.0.clone(), slot3.1) },
                unsafe { BufferArg::from_raw_parts(slot4.0.clone(), slot4.1) },
                unsafe { BufferArg::from_raw_parts(slot5.0.clone(), slot5.1) },
                unsafe { BufferArg::from_raw_parts(slot6.0.clone(), slot6.1) },
                unsafe { BufferArg::from_raw_parts(offsets.clone(), 7) },
                unsafe { BufferArg::from_raw_parts(current_handle.clone(), current_count * 2) },
                unsafe { BufferArg::from_raw_parts(candidate_len_handle.clone(), 1) },
                unsafe { BufferArg::from_raw_parts(next_handle.clone(), next_count * 2) },
            );
        }

        current_handle = next_handle;
        current_count = next_count;
        current_count_u32 = u32::try_from(current_count)
            .map_err(|_| Error::LengthTooLarge { len: current_count })?;
    }

    let bytes = client
        .read_one(current_handle)
        .map_err(|err| Error::Launch {
            message: format!("{err:?}"),
        })?;
    let indices = u32::from_bytes(&bytes);
    Ok(Some((indices[0], indices[1])))
}

fn logical7_scan_by_key_control_read<R, Read, KeyEq>(
    read: &Read,
    policy: &CubePolicy<R>,
) -> Result<ScanByKeyControl<R>, Error>
where
    R: Runtime,
    Read: KernelReadBoundMany<R>,
    <Read as KernelRead<R>>::Item: MItem<R> + Send + Sync,
    KeyEq: op::BinaryPredicateOp<R, <Read as KernelRead<R>>::Item>,
{
    let len = read.len();
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let client = policy.client();
    let flags = if len == 0 {
        client.empty(0)
    } else {
        let flags = client.empty(len * std::mem::size_of::<u32>());
        let mut bindings = KernelColumnBindings::empty(client);
        <Read as KernelReadAtEnv<R, Env0>>::stage_at_env(read, &mut bindings)?;
        bindings.finish();
        let offsets = bindings.slot_offsets7_handle(client)?;
        let slot0 = bindings.slot_or_first(0);
        let slot1 = bindings.slot_or_first(1);
        let slot2 = bindings.slot_or_first(2);
        let slot3 = bindings.slot_or_first(3);
        let slot4 = bindings.slot_or_first(4);
        let slot5 = bindings.slot_or_first(5);
        let slot6 = bindings.slot_or_first(6);
        let block_size = 256_u32;
        let block_count = len.div_ceil(block_size as usize);
        let block_count_u32 =
            u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;
        unsafe {
            crate::kernels::logical7_scan_by_key_head_flags_kernel::launch_unchecked::<
                <Read as KernelRead<R>>::Item,
                <Read as KernelReadBoundMany<R>>::Leaf0,
                <Read as KernelReadBoundMany<R>>::Leaf1,
                <Read as KernelReadBoundMany<R>>::Leaf2,
                <Read as KernelReadBoundMany<R>>::Leaf3,
                <Read as KernelReadBoundMany<R>>::Leaf4,
                <Read as KernelReadBoundMany<R>>::Leaf5,
                <Read as KernelReadBoundMany<R>>::Leaf6,
                <Read as KernelReadBoundMany<R>>::ExprAt,
                KernelOp<R, KeyEq>,
                R,
            >(
                client,
                CubeCount::Static(block_count_u32, 1, 1),
                CubeDim::new_1d(block_size),
                unsafe { BufferArg::from_raw_parts(slot0.0.clone(), slot0.1) },
                unsafe { BufferArg::from_raw_parts(slot1.0.clone(), slot1.1) },
                unsafe { BufferArg::from_raw_parts(slot2.0.clone(), slot2.1) },
                unsafe { BufferArg::from_raw_parts(slot3.0.clone(), slot3.1) },
                unsafe { BufferArg::from_raw_parts(slot4.0.clone(), slot4.1) },
                unsafe { BufferArg::from_raw_parts(slot5.0.clone(), slot5.1) },
                unsafe { BufferArg::from_raw_parts(slot6.0.clone(), slot6.1) },
                unsafe { BufferArg::from_raw_parts(offsets.clone(), 7) },
                unsafe { BufferArg::from_raw_parts(flags.clone(), len) },
            );
        }
        flags
    };
    let segment = SegmentControl::from_head_flags(flags, len, len_u32);
    Ok(ScanByKeyControl::from_segment(&segment))
}

pub(crate) fn inclusive_scan_by_key_values_from_view<R, Item, View, KeyEq, Op, Output>(
    policy: &CubePolicy<R>,
    view: View,
    control: &ScanByKeyControl<R>,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Item: MAlloc<R>,
    View: crate::detail::read::KernelInclusiveScanByKeyValues<
            ScanByKeyControl<R>,
            KernelOp<R, KeyEq>,
            KernelOp<R, Op>,
            Runtime = R,
        >,
    <View as crate::detail::read::KernelInclusiveScanByKeyValues<
        ScanByKeyControl<R>,
        KernelOp<R, KeyEq>,
        KernelOp<R, Op>,
    >>::Output:
        crate::detail::api::MaterializeOutput<Runtime = R, Output = <Item as MAlloc<R>>::Inner>,
    Op: op::ReductionOp<R, Item>,
    Output: MIterMut<R, Item = Item>,
{
    let scanned = <View as crate::detail::read::KernelInclusiveScanByKeyValues<
        ScanByKeyControl<R>,
        KernelOp<R, KeyEq>,
        KernelOp<R, Op>,
    >>::inclusive_scan_by_key_values(view, policy, control)?;
    let inner = crate::detail::api::MaterializeOutput::materialize_output(scanned, policy)?;
    output.write_from_inner(policy, inner)
}

pub(crate) fn exclusive_scan_by_key_values_from_view<R, Item, View, Init, KeyEq, Op, Output>(
    policy: &CubePolicy<R>,
    view: View,
    control: &ScanByKeyControl<R>,
    init: Init,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Item: MAlloc<R>,
    View: crate::detail::read::KernelExclusiveScanByKeyValues<
            ScanByKeyControl<R>,
            KernelOp<R, KeyEq>,
            KernelOp<R, Op>,
            Runtime = R,
            Init = Init,
        >,
    <View as crate::detail::read::KernelExclusiveScanByKeyValues<
        ScanByKeyControl<R>,
        KernelOp<R, KeyEq>,
        KernelOp<R, Op>,
    >>::Output:
        crate::detail::api::MaterializeOutput<Runtime = R, Output = <Item as MAlloc<R>>::Inner>,
    Op: op::ReductionOp<R, Item>,
    Output: MIterMut<R, Item = Item>,
{
    let scanned = <View as crate::detail::read::KernelExclusiveScanByKeyValues<
        ScanByKeyControl<R>,
        KernelOp<R, KeyEq>,
        KernelOp<R, Op>,
    >>::exclusive_scan_by_key_values(view, policy, control, init)?;
    let inner = crate::detail::api::MaterializeOutput::materialize_output(scanned, policy)?;
    output.write_from_inner(policy, inner)
}

fn logical3_bound_many_read<R, Source, Values, Less>(
    source: &Source,
    policy: &CubePolicy<R>,
    values: &Values,
    upper: bool,
) -> Result<crate::runtime::DeviceVec<R, crate::MIndex>, Error>
where
    R: Runtime,
    Source: KernelRead<R> + KernelReadAt<R, S0, LogicalItem = <Source as KernelRead<R>>::Item>,
    Values: KernelRead<R, Item = <Source as KernelRead<R>>::Item>
        + KernelReadAt<R, S0, LogicalItem = <Source as KernelRead<R>>::Item>,
    <Source as KernelRead<R>>::Item: MItem<R> + Send + Sync,
    <Source as KernelReadAt<R, S0>>::ExprAt: crate::expr::LogicalDeviceExpr3Shape<<Source as KernelRead<R>>::Item>
        + crate::expr::LogicalDeviceExpr3<
            <Source as KernelRead<R>>::Item,
            <<Source as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Source as KernelRead<R>>::Item,
            >>::LeafA,
            <<Source as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Source as KernelRead<R>>::Item,
            >>::LeafB,
            <<Source as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Source as KernelRead<R>>::Item,
            >>::LeafC,
        >,
    <Values as KernelReadAt<R, S0>>::ExprAt: crate::expr::LogicalDeviceExpr3Shape<<Source as KernelRead<R>>::Item>
        + crate::expr::LogicalDeviceExpr3<
            <Source as KernelRead<R>>::Item,
            <<Values as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Source as KernelRead<R>>::Item,
            >>::LeafA,
            <<Values as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Source as KernelRead<R>>::Item,
            >>::LeafB,
            <<Values as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                <Source as KernelRead<R>>::Item,
            >>::LeafC,
        >,
    <<Source as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
        <Source as KernelRead<R>>::Item,
    >>::LeafA: MStorageElement,
    <<Source as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
        <Source as KernelRead<R>>::Item,
    >>::LeafB: MStorageElement,
    <<Source as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
        <Source as KernelRead<R>>::Item,
    >>::LeafC: MStorageElement,
    <<Values as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
        <Source as KernelRead<R>>::Item,
    >>::LeafA: MStorageElement,
    <<Values as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
        <Source as KernelRead<R>>::Item,
    >>::LeafB: MStorageElement,
    <<Values as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
        <Source as KernelRead<R>>::Item,
    >>::LeafC: MStorageElement,
    Less: op::BinaryPredicateOp<R, <Source as KernelRead<R>>::Item>,
{
    let value_len = values.len();
    let value_len_mindex = mindex_from_usize(value_len)?;
    if value_len == 0 {
        return Ok(crate::runtime::DeviceVec::from_inner(
            policy.empty_device_vec(),
        ));
    }

    let client = policy.client();
    let source_len_u32 =
        u32::try_from(source.len()).map_err(|_| Error::LengthTooLarge { len: source.len() })?;
    let value_len_u32 =
        u32::try_from(value_len).map_err(|_| Error::LengthTooLarge { len: value_len })?;
    let source_len_handle = client.create_from_slice(u32::as_bytes(&[source_len_u32]));
    let value_len_handle = client.create_from_slice(u32::as_bytes(&[value_len_u32]));
    let output = client.empty(value_len * std::mem::size_of::<u32>());

    let mut source_bindings = KernelColumnBindings::empty(client);
    <Source as KernelReadAt<R, S0>>::stage_at(source, &mut source_bindings)?;
    source_bindings.finish();
    let mut value_bindings = KernelColumnBindings::empty(client);
    <Values as KernelReadAt<R, S0>>::stage_at(values, &mut value_bindings)?;
    value_bindings.finish();
    let source_offsets = source_bindings.slot_offsets_handle(client)?;
    let value_offsets = value_bindings.slot_offsets_handle(client)?;
    let source0 = source_bindings.slot_or_first(0);
    let source1 = source_bindings.slot_or_first(1);
    let source2 = source_bindings.slot_or_first(2);
    let value0 = value_bindings.slot_or_first(0);
    let value1 = value_bindings.slot_or_first(1);
    let value2 = value_bindings.slot_or_first(2);
    let block_size = 256_u32;
    let block_count = value_len.div_ceil(block_size as usize);
    let block_count_u32 =
        u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;

    unsafe {
        if upper {
            crate::kernels::logical3_upper_bound_many_kernel::launch_unchecked::<
                <Source as KernelRead<R>>::Item,
                <<Source as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                    <Source as KernelRead<R>>::Item,
                >>::LeafA,
                <<Source as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                    <Source as KernelRead<R>>::Item,
                >>::LeafB,
                <<Source as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                    <Source as KernelRead<R>>::Item,
                >>::LeafC,
                <<Values as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                    <Source as KernelRead<R>>::Item,
                >>::LeafA,
                <<Values as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                    <Source as KernelRead<R>>::Item,
                >>::LeafB,
                <<Values as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                    <Source as KernelRead<R>>::Item,
                >>::LeafC,
                <Source as KernelReadAt<R, S0>>::ExprAt,
                <Values as KernelReadAt<R, S0>>::ExprAt,
                KernelOp<R, Less>,
                R,
            >(
                client,
                CubeCount::Static(block_count_u32, 1, 1),
                CubeDim::new_1d(block_size),
                BufferArg::from_raw_parts(source0.0.clone(), source0.1),
                BufferArg::from_raw_parts(source1.0.clone(), source1.1),
                BufferArg::from_raw_parts(source2.0.clone(), source2.1),
                BufferArg::from_raw_parts(source_offsets.clone(), 4),
                BufferArg::from_raw_parts(value0.0.clone(), value0.1),
                BufferArg::from_raw_parts(value1.0.clone(), value1.1),
                BufferArg::from_raw_parts(value2.0.clone(), value2.1),
                BufferArg::from_raw_parts(value_offsets.clone(), 4),
                BufferArg::from_raw_parts(source_len_handle.clone(), 1),
                BufferArg::from_raw_parts(value_len_handle.clone(), 1),
                BufferArg::from_raw_parts(output.clone(), value_len),
            );
        } else {
            crate::kernels::logical3_lower_bound_many_kernel::launch_unchecked::<
                <Source as KernelRead<R>>::Item,
                <<Source as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                    <Source as KernelRead<R>>::Item,
                >>::LeafA,
                <<Source as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                    <Source as KernelRead<R>>::Item,
                >>::LeafB,
                <<Source as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                    <Source as KernelRead<R>>::Item,
                >>::LeafC,
                <<Values as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                    <Source as KernelRead<R>>::Item,
                >>::LeafA,
                <<Values as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                    <Source as KernelRead<R>>::Item,
                >>::LeafB,
                <<Values as KernelReadAt<R, S0>>::ExprAt as crate::expr::LogicalDeviceExpr3Shape<
                    <Source as KernelRead<R>>::Item,
                >>::LeafC,
                <Source as KernelReadAt<R, S0>>::ExprAt,
                <Values as KernelReadAt<R, S0>>::ExprAt,
                KernelOp<R, Less>,
                R,
            >(
                client,
                CubeCount::Static(block_count_u32, 1, 1),
                CubeDim::new_1d(block_size),
                BufferArg::from_raw_parts(source0.0.clone(), source0.1),
                BufferArg::from_raw_parts(source1.0.clone(), source1.1),
                BufferArg::from_raw_parts(source2.0.clone(), source2.1),
                BufferArg::from_raw_parts(source_offsets.clone(), 4),
                BufferArg::from_raw_parts(value0.0.clone(), value0.1),
                BufferArg::from_raw_parts(value1.0.clone(), value1.1),
                BufferArg::from_raw_parts(value2.0.clone(), value2.1),
                BufferArg::from_raw_parts(value_offsets.clone(), 4),
                BufferArg::from_raw_parts(source_len_handle.clone(), 1),
                BufferArg::from_raw_parts(value_len_handle.clone(), 1),
                BufferArg::from_raw_parts(output.clone(), value_len),
            );
        }
    }

    Ok(crate::runtime::DeviceVec::from_inner(
        crate::detail::DeviceVec::from_handle(policy.id(), output, value_len_mindex),
    ))
}

fn logical7_bound_many_read<R, Source, Values, Less>(
    source: &Source,
    policy: &CubePolicy<R>,
    values: &Values,
    upper: bool,
) -> Result<crate::runtime::DeviceVec<R, crate::MIndex>, Error>
where
    R: Runtime,
    Source: KernelReadBoundMany<R>,
    Values: KernelReadBoundMany<R, Item = <Source as KernelRead<R>>::Item>,
    <Source as KernelRead<R>>::Item: MItem<R> + Send + Sync,
    Less: op::BinaryPredicateOp<R, <Source as KernelRead<R>>::Item>,
{
    let value_len = values.len();
    let value_len_mindex = mindex_from_usize(value_len)?;
    if value_len == 0 {
        return Ok(crate::runtime::DeviceVec::from_inner(
            policy.empty_device_vec(),
        ));
    }

    let client = policy.client();
    let source_len_u32 =
        u32::try_from(source.len()).map_err(|_| Error::LengthTooLarge { len: source.len() })?;
    let value_len_u32 =
        u32::try_from(value_len).map_err(|_| Error::LengthTooLarge { len: value_len })?;
    let source_len_handle = client.create_from_slice(u32::as_bytes(&[source_len_u32]));
    let value_len_handle = client.create_from_slice(u32::as_bytes(&[value_len_u32]));
    let output = client.empty(value_len * std::mem::size_of::<u32>());

    let mut source_bindings = KernelColumnBindings::empty(client);
    <Source as KernelReadAtEnv<R, Env0>>::stage_at_env(source, &mut source_bindings)?;
    source_bindings.finish();
    let mut value_bindings = KernelColumnBindings::empty(client);
    <Values as KernelReadAtEnv<R, Env0>>::stage_at_env(values, &mut value_bindings)?;
    value_bindings.finish();
    let source_offsets = source_bindings.slot_offsets7_handle(client)?;
    let value_offsets = value_bindings.slot_offsets7_handle(client)?;
    let source0 = source_bindings.slot_or_first(0);
    let source1 = source_bindings.slot_or_first(1);
    let source2 = source_bindings.slot_or_first(2);
    let source3 = source_bindings.slot_or_first(3);
    let source4 = source_bindings.slot_or_first(4);
    let source5 = source_bindings.slot_or_first(5);
    let source6 = source_bindings.slot_or_first(6);
    let value0 = value_bindings.slot_or_first(0);
    let value1 = value_bindings.slot_or_first(1);
    let value2 = value_bindings.slot_or_first(2);
    let value3 = value_bindings.slot_or_first(3);
    let value4 = value_bindings.slot_or_first(4);
    let value5 = value_bindings.slot_or_first(5);
    let value6 = value_bindings.slot_or_first(6);
    let block_size = 256_u32;
    let block_count = value_len.div_ceil(block_size as usize);
    let block_count_u32 =
        u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;

    unsafe {
        if upper {
            crate::kernels::logical7_upper_bound_many_kernel::launch_unchecked::<
                <Source as KernelRead<R>>::Item,
                <Source as KernelReadBoundMany<R>>::Leaf0,
                <Source as KernelReadBoundMany<R>>::Leaf1,
                <Source as KernelReadBoundMany<R>>::Leaf2,
                <Source as KernelReadBoundMany<R>>::Leaf3,
                <Source as KernelReadBoundMany<R>>::Leaf4,
                <Source as KernelReadBoundMany<R>>::Leaf5,
                <Source as KernelReadBoundMany<R>>::Leaf6,
                <Values as KernelReadBoundMany<R>>::Leaf0,
                <Values as KernelReadBoundMany<R>>::Leaf1,
                <Values as KernelReadBoundMany<R>>::Leaf2,
                <Values as KernelReadBoundMany<R>>::Leaf3,
                <Values as KernelReadBoundMany<R>>::Leaf4,
                <Values as KernelReadBoundMany<R>>::Leaf5,
                <Values as KernelReadBoundMany<R>>::Leaf6,
                <Source as KernelReadBoundMany<R>>::ExprAt,
                <Values as KernelReadBoundMany<R>>::ExprAt,
                KernelOp<R, Less>,
                R,
            >(
                client,
                CubeCount::Static(block_count_u32, 1, 1),
                CubeDim::new_1d(block_size),
                BufferArg::from_raw_parts(source0.0.clone(), source0.1),
                BufferArg::from_raw_parts(source1.0.clone(), source1.1),
                BufferArg::from_raw_parts(source2.0.clone(), source2.1),
                BufferArg::from_raw_parts(source3.0.clone(), source3.1),
                BufferArg::from_raw_parts(source4.0.clone(), source4.1),
                BufferArg::from_raw_parts(source5.0.clone(), source5.1),
                BufferArg::from_raw_parts(source6.0.clone(), source6.1),
                BufferArg::from_raw_parts(source_offsets.clone(), 7),
                BufferArg::from_raw_parts(value0.0.clone(), value0.1),
                BufferArg::from_raw_parts(value1.0.clone(), value1.1),
                BufferArg::from_raw_parts(value2.0.clone(), value2.1),
                BufferArg::from_raw_parts(value3.0.clone(), value3.1),
                BufferArg::from_raw_parts(value4.0.clone(), value4.1),
                BufferArg::from_raw_parts(value5.0.clone(), value5.1),
                BufferArg::from_raw_parts(value6.0.clone(), value6.1),
                BufferArg::from_raw_parts(value_offsets.clone(), 7),
                BufferArg::from_raw_parts(source_len_handle.clone(), 1),
                BufferArg::from_raw_parts(value_len_handle.clone(), 1),
                BufferArg::from_raw_parts(output.clone(), value_len),
            );
        } else {
            crate::kernels::logical7_lower_bound_many_kernel::launch_unchecked::<
                <Source as KernelRead<R>>::Item,
                <Source as KernelReadBoundMany<R>>::Leaf0,
                <Source as KernelReadBoundMany<R>>::Leaf1,
                <Source as KernelReadBoundMany<R>>::Leaf2,
                <Source as KernelReadBoundMany<R>>::Leaf3,
                <Source as KernelReadBoundMany<R>>::Leaf4,
                <Source as KernelReadBoundMany<R>>::Leaf5,
                <Source as KernelReadBoundMany<R>>::Leaf6,
                <Values as KernelReadBoundMany<R>>::Leaf0,
                <Values as KernelReadBoundMany<R>>::Leaf1,
                <Values as KernelReadBoundMany<R>>::Leaf2,
                <Values as KernelReadBoundMany<R>>::Leaf3,
                <Values as KernelReadBoundMany<R>>::Leaf4,
                <Values as KernelReadBoundMany<R>>::Leaf5,
                <Values as KernelReadBoundMany<R>>::Leaf6,
                <Source as KernelReadBoundMany<R>>::ExprAt,
                <Values as KernelReadBoundMany<R>>::ExprAt,
                KernelOp<R, Less>,
                R,
            >(
                client,
                CubeCount::Static(block_count_u32, 1, 1),
                CubeDim::new_1d(block_size),
                BufferArg::from_raw_parts(source0.0.clone(), source0.1),
                BufferArg::from_raw_parts(source1.0.clone(), source1.1),
                BufferArg::from_raw_parts(source2.0.clone(), source2.1),
                BufferArg::from_raw_parts(source3.0.clone(), source3.1),
                BufferArg::from_raw_parts(source4.0.clone(), source4.1),
                BufferArg::from_raw_parts(source5.0.clone(), source5.1),
                BufferArg::from_raw_parts(source6.0.clone(), source6.1),
                BufferArg::from_raw_parts(source_offsets.clone(), 7),
                BufferArg::from_raw_parts(value0.0.clone(), value0.1),
                BufferArg::from_raw_parts(value1.0.clone(), value1.1),
                BufferArg::from_raw_parts(value2.0.clone(), value2.1),
                BufferArg::from_raw_parts(value3.0.clone(), value3.1),
                BufferArg::from_raw_parts(value4.0.clone(), value4.1),
                BufferArg::from_raw_parts(value5.0.clone(), value5.1),
                BufferArg::from_raw_parts(value6.0.clone(), value6.1),
                BufferArg::from_raw_parts(value_offsets.clone(), 7),
                BufferArg::from_raw_parts(source_len_handle.clone(), 1),
                BufferArg::from_raw_parts(value_len_handle.clone(), 1),
                BufferArg::from_raw_parts(output.clone(), value_len),
            );
        }
    }

    Ok(crate::runtime::DeviceVec::from_inner(
        crate::detail::DeviceVec::from_handle(policy.id(), output, value_len_mindex),
    ))
}

/// Internal transform lowering for read expressions.
///
/// Implementations are intentionally structural: only read trees that can
/// present the exact logical kernel item have this ability. Nested zips are not
/// flattened here.
#[doc(hidden)]
pub trait KernelReadTransform<R: Runtime, Output, Op>: KernelRead<R>
where
    Output: MIterMut<R>,
    Self::Item: MItem<R>,
    Op: op::UnaryOp<R, Self::Item, Output = Output::Item>,
{
    fn transform_into(self, policy: &CubePolicy<R>, op: Op, output: Output) -> Result<(), Error>;

    fn transform_where_into(
        self,
        policy: &CubePolicy<R>,
        op: Op,
        stencil: PrecomputedSelection<R>,
        output: Output,
    ) -> Result<(), Error>;
}

/// Internal lowering for `u32` stencil iterators.
#[doc(hidden)]
pub trait KernelStencilSelection<R: Runtime>: KernelRead<R, Item = bool> {
    fn stencil_selection(
        self,
        policy: &CubePolicy<R>,
        invert: bool,
        flags_only: bool,
    ) -> Result<PrecomputedSelection<R>, Error>;
}

/// Internal selected-copy lowering for read expressions.
#[doc(hidden)]
pub trait KernelReadSelection<R: Runtime, Output>: KernelRead<R>
where
    Output: MIterMut<R, Item = Self::Item>,
    Self::Item: MAlloc<R>,
{
    fn copy_selected_into(
        self,
        policy: &CubePolicy<R>,
        stencil: PrecomputedSelection<R>,
        output: Output,
    ) -> Result<crate::MIndex, Error>;
}

/// Internal consecutive-unique lowering for read expressions.
#[doc(hidden)]
pub trait KernelReadUnique<R: Runtime, Output, Pred>: KernelRead<R>
where
    Output: MIterMut<R, Item = Self::Item>,
    Self::Item: MAlloc<R>,
    Pred: op::BinaryPredicateOp<R, Self::Item>,
{
    fn unique_into(
        self,
        policy: &CubePolicy<R>,
        pred: Pred,
        output: Output,
    ) -> Result<crate::MIndex, Error>;
}

/// Internal partition lowering for read expressions.
#[doc(hidden)]
pub trait KernelReadPartition<R: Runtime, Output, Pred>: KernelRead<R>
where
    Output: MIterMut<R, Item = Self::Item>,
    Self::Item: MAlloc<R>,
    Pred: op::PredicateOp<R, Self::Item>,
{
    fn partition_into(
        self,
        policy: &CubePolicy<R>,
        pred: Pred,
        output: Output,
    ) -> Result<crate::MIndex, Error>;
}

/// Internal sorted-set lowering for read expressions.
#[doc(hidden)]
pub trait KernelReadSet<R: Runtime, Right, Output, Less>: KernelRead<R>
where
    Right: KernelRead<R, Item = Self::Item>,
    Output: MIterMut<R, Item = Self::Item>,
    Self::Item: MAlloc<R>,
    Less: op::BinaryPredicateOp<R, Self::Item>,
{
    fn set_union_into(
        self,
        policy: &CubePolicy<R>,
        right: Right,
        less: Less,
        output: Output,
    ) -> Result<crate::MIndex, Error>;

    fn set_intersection_into(
        self,
        policy: &CubePolicy<R>,
        right: Right,
        less: Less,
        output: Output,
    ) -> Result<crate::MIndex, Error>;

    fn set_difference_into(
        self,
        policy: &CubePolicy<R>,
        right: Right,
        less: Less,
        output: Output,
    ) -> Result<crate::MIndex, Error>;
}

/// Internal reduction lowering for read expressions.
#[doc(hidden)]
pub trait KernelReadReduce<R: Runtime, Op>: KernelRead<R>
where
    Self::Item: MItem<R>,
    Op: op::ReductionOp<R, Self::Item>,
{
    fn reduce_value(
        self,
        policy: &CubePolicy<R>,
        init: Self::Item,
        op: Op,
    ) -> Result<Self::Item, Error>;
}

/// Internal predicate-query lowering for read expressions.
#[doc(hidden)]
pub trait KernelReadPredicateQuery<R: Runtime, Pred>: KernelRead<R>
where
    Self::Item: MItem<R>,
    Pred: op::PredicateOp<R, Self::Item>,
{
    fn count_if(self, policy: &CubePolicy<R>, pred: Pred) -> Result<crate::MIndex, Error>;

    fn all_of(self, policy: &CubePolicy<R>, pred: Pred) -> Result<bool, Error>;

    fn any_of(self, policy: &CubePolicy<R>, pred: Pred) -> Result<bool, Error>;

    fn none_of(self, policy: &CubePolicy<R>, pred: Pred) -> Result<bool, Error>;

    fn find_if(self, policy: &CubePolicy<R>, pred: Pred) -> Result<Option<crate::MIndex>, Error>;

    fn is_partitioned(self, policy: &CubePolicy<R>, pred: Pred) -> Result<bool, Error>;
}

/// Internal adjacent-difference lowering for read expressions.
#[doc(hidden)]
pub trait KernelReadAdjacentDifference<R: Runtime, Output, Op>: KernelRead<R>
where
    Output: MIterMut<R, Item = Self::Item>,
    Self::Item: MItem<R>,
    Op: op::ReductionOp<R, Self::Item>,
{
    fn adjacent_difference_into(
        self,
        policy: &CubePolicy<R>,
        op: Op,
        output: Output,
    ) -> Result<(), Error>;
}

/// Internal inclusive/exclusive scan lowering for read expressions.
#[doc(hidden)]
pub trait KernelReadScan<R: Runtime, Output, Op>: KernelRead<R>
where
    Output: MIterMut<R, Item = Self::Item>,
    Self::Item: MItem<R>,
    Op: op::ReductionOp<R, Self::Item>,
{
    fn inclusive_scan_into(
        self,
        policy: &CubePolicy<R>,
        op: Op,
        output: Output,
    ) -> Result<(), Error>;

    fn exclusive_scan_into(
        self,
        policy: &CubePolicy<R>,
        init: Self::Item,
        op: Op,
        output: Output,
    ) -> Result<(), Error>;
}

/// Internal scan-by-key lowering for read expressions.
#[doc(hidden)]
pub trait KernelReadScanByKey<R: Runtime, Values, Output, KeyEq, Op>: KernelRead<R>
where
    Values: KernelRead<R>,
    Output: MIterMut<R, Item = Values::Item>,
    Self::Item: MItem<R>,
    Values::Item: MItem<R>,
    KeyEq: op::BinaryPredicateOp<R, Self::Item>,
    Op: op::ReductionOp<R, Values::Item>,
{
    fn inclusive_scan_by_key_into(
        self,
        policy: &CubePolicy<R>,
        values: Values,
        key_eq: KeyEq,
        op: Op,
        output: Output,
    ) -> Result<(), Error>;

    fn exclusive_scan_by_key_into(
        self,
        policy: &CubePolicy<R>,
        values: Values,
        key_eq: KeyEq,
        init: Values::Item,
        op: Op,
        output: Output,
    ) -> Result<(), Error>;
}

/// Internal indexed gather/scatter lowering for read expressions.
#[doc(hidden)]
pub trait KernelReadIndexed<R: Runtime, Indices, Output>: KernelRead<R>
where
    Indices: KernelRead<R, Item = crate::MIndex>,
    Output: MIterMut<R, Item = Self::Item>,
{
    fn gather_into(
        self,
        policy: &CubePolicy<R>,
        indices: Indices,
        output: Output,
    ) -> Result<(), Error>;

    fn gather_where_into(
        self,
        policy: &CubePolicy<R>,
        indices: Indices,
        stencil: PrecomputedSelection<R>,
        output: Output,
    ) -> Result<(), Error>;

    fn scatter_into(
        self,
        policy: &CubePolicy<R>,
        indices: Indices,
        output: Output,
    ) -> Result<(), Error>;

    fn scatter_where_into(
        self,
        policy: &CubePolicy<R>,
        indices: Indices,
        stencil: PrecomputedSelection<R>,
        output: Output,
    ) -> Result<(), Error>;
}

/// Internal adjacent-find lowering for read expressions.
#[doc(hidden)]
pub trait KernelReadAdjacentFind<R: Runtime, Pred>: KernelRead<R>
where
    Self::Item: MItem<R>,
    Pred: op::BinaryPredicateOp<R, Self::Item>,
{
    fn adjacent_find(
        self,
        policy: &CubePolicy<R>,
        pred: Pred,
    ) -> Result<Option<crate::MIndex>, Error>;
}

/// Internal min/max lowering for read expressions.
#[doc(hidden)]
pub trait KernelReadMinMax<R: Runtime, Less>: KernelRead<R>
where
    Self::Item: MItem<R>,
    Less: op::BinaryPredicateOp<R, Self::Item>,
{
    fn min_element(
        self,
        policy: &CubePolicy<R>,
        less: Less,
    ) -> Result<Option<crate::MIndex>, Error>;

    fn max_element(
        self,
        policy: &CubePolicy<R>,
        less: Less,
    ) -> Result<Option<crate::MIndex>, Error>;

    fn minmax_element(
        self,
        policy: &CubePolicy<R>,
        less: Less,
    ) -> Result<Option<(crate::MIndex, crate::MIndex)>, Error>;
}

/// Internal sorted-search lowering for read expressions.
#[doc(hidden)]
pub trait KernelReadSortedSearch<R: Runtime, Values, Less>: KernelRead<R>
where
    Values: KernelRead<R, Item = Self::Item>,
    Self::Item: MItem<R>,
    Less: op::BinaryPredicateOp<R, Self::Item>,
{
    fn lower_bound_many(
        self,
        policy: &CubePolicy<R>,
        values: Values,
        less: Less,
    ) -> Result<crate::runtime::DeviceVec<R, crate::MIndex>, Error>;

    fn upper_bound_many(
        self,
        policy: &CubePolicy<R>,
        values: Values,
        less: Less,
    ) -> Result<crate::runtime::DeviceVec<R, crate::MIndex>, Error>;

    fn is_sorted_until(self, policy: &CubePolicy<R>, less: Less) -> Result<crate::MIndex, Error>;

    fn is_sorted(self, policy: &CubePolicy<R>, less: Less) -> Result<bool, Error>;
}

/// Internal pair-search lowering for read expressions.
#[doc(hidden)]
pub trait KernelReadPairSearch<R: Runtime, Right, Op>: KernelRead<R>
where
    Right: KernelRead<R, Item = Self::Item>,
    Self::Item: MItem<R>,
    Op: op::BinaryPredicateOp<R, Self::Item>,
{
    fn equal(self, policy: &CubePolicy<R>, right: Right, op: Op) -> Result<bool, Error>;

    fn mismatch(
        self,
        policy: &CubePolicy<R>,
        right: Right,
        op: Op,
    ) -> Result<Option<crate::MIndex>, Error>;

    fn find_first_of(
        self,
        policy: &CubePolicy<R>,
        needles: Right,
        op: Op,
    ) -> Result<Option<crate::MIndex>, Error>;

    fn lexicographical_compare(
        self,
        policy: &CubePolicy<R>,
        right: Right,
        op: Op,
    ) -> Result<bool, Error>;
}

trait KernelReadScanByKeyView<R: Runtime>: KernelRead<R> {
    type View;

    fn into_scan_by_key_view(self) -> Self::View;
}

trait KernelReadScanByKeyValuesView<R: Runtime>: KernelReadScanByKeyView<R> {
    type ExclusiveInit;

    fn into_exclusive_scan_by_key_init(init: Self::Item) -> Self::ExclusiveInit;
}

trait KernelReadSearchView<R: Runtime>: KernelRead<R> {
    type View;

    fn into_search_view(self) -> Self::View;
}

trait SliceSearchView {
    fn slice_search_view(self, start: usize, len: usize) -> Self;
}

impl<R, T> SliceSearchView for DeviceColumnView<R, T>
where
    R: Runtime,
{
    fn slice_search_view(mut self, start: usize, len: usize) -> Self {
        self.offset += start;
        self.len = len;
        self
    }
}

impl<R, T> SliceSearchView for (DeviceColumnView<R, T>,)
where
    R: Runtime,
{
    fn slice_search_view(self, start: usize, len: usize) -> Self {
        (self.0.slice_search_view(start, len),)
    }
}

impl<Source> SliceSearchView for ZipView1<Source>
where
    Source: SliceSearchView,
{
    fn slice_search_view(self, start: usize, len: usize) -> Self {
        ZipView1 {
            source: self.source.slice_search_view(start, len),
        }
    }
}

impl<Left, Right> SliceSearchView for ZipView2<Left, Right>
where
    Left: SliceSearchView,
    Right: SliceSearchView,
{
    fn slice_search_view(self, start: usize, len: usize) -> Self {
        ZipView2 {
            left: self.left.slice_search_view(start, len),
            right: self.right.slice_search_view(start, len),
        }
    }
}

impl<First, Second, Third> SliceSearchView for ZipView3<First, Second, Third>
where
    First: SliceSearchView,
    Second: SliceSearchView,
    Third: SliceSearchView,
{
    fn slice_search_view(self, start: usize, len: usize) -> Self {
        ZipView3 {
            first: self.first.slice_search_view(start, len),
            second: self.second.slice_search_view(start, len),
            third: self.third.slice_search_view(start, len),
        }
    }
}

macro_rules! impl_slice_search_view {
    ($name:ident < $( $ty:ident : $field:ident ),+ >) => {
        impl<$( $ty, )+> SliceSearchView for $name<$( $ty ),+>
        where
            $( $ty: SliceSearchView, )+
        {
            fn slice_search_view(self, start: usize, len: usize) -> Self {
                $name {
                    $( $field: self.$field.slice_search_view(start, len), )+
                }
            }
        }
    };
}

impl_slice_search_view!(ZipView4<A: a, B: b, C: c, D: d>);
impl_slice_search_view!(ZipView5<A: a, B: b, C: c, D: d, E: e>);
impl_slice_search_view!(ZipView6<A: a, B: b, C: c, D: d, E: e, F: f>);
impl_slice_search_view!(ZipView7<A: a, B: b, C: c, D: d, E: e, F: f, G: g>);

impl<R, Read> KernelReadSearchView<R> for SliceRead<Read>
where
    R: Runtime,
    Read: KernelReadSearchView<R>,
    Read::View: SliceSearchView,
{
    type View = Read::View;

    fn into_search_view(self) -> Self::View {
        self.read
            .into_search_view()
            .slice_search_view(self.start, self.len)
    }
}

/// Leaf read expression for one physical device column.
#[doc(hidden)]
pub struct ColumnRead<R: Runtime, T> {
    pub(crate) column: DeviceColumnView<R, T>,
}

impl<R, T> ColumnRead<R, T>
where
    R: Runtime,
{
    pub(crate) fn new(column: DeviceColumnView<R, T>) -> Self {
        Self { column }
    }
}

impl<R, T> KernelRead<R> for ColumnRead<R, T>
where
    R: Runtime,
    T: MStorageElement + 'static,
{
    type Item = T;

    fn len(&self) -> usize {
        KernelColumn::len(&self.column)
    }

    fn validate(&self) -> Result<(), Error> {
        Ok(())
    }

    fn reduce_value_read<Op>(
        self,
        policy: &CubePolicy<R>,
        init: Self::Item,
        _op: Op,
    ) -> Result<Self::Item, Error>
    where
        Self::Item: MItem<R>,
        Op: op::ReductionOp<R, Self::Item>,
    {
        let result = crate::detail::reduce(
            policy,
            (self.column,),
            (init,),
            KernelScalarTuple1Op::<R, Op>::new(),
        )?;
        Ok(result.0)
    }

    fn transform_read<Output, Op>(
        self,
        policy: &CubePolicy<R>,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R>,
        Self::Item: MItem<R>,
        Output::Item: MAlloc<R> + MItemDispatch<R>,
        Op: op::UnaryOp<R, Self::Item, Output = Output::Item>,
    {
        transform_logical7_read(self, policy, op, output)
    }

    fn transform_where_read<Output, Op>(
        self,
        policy: &CubePolicy<R>,
        op: Op,
        stencil: PrecomputedSelection<R>,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R>,
        Self::Item: MItem<R>,
        Output::Item: MAlloc<R> + MItemDispatch<R>,
        Op: op::UnaryOp<R, Self::Item, Output = Output::Item>,
    {
        transform_where_logical7_read(self, policy, op, stencil, output)
    }

    fn count_if_read<Pred>(
        self,
        policy: &CubePolicy<R>,
        _pred: Pred,
    ) -> Result<crate::MIndex, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Pred: op::PredicateOp<R, Self::Item>,
    {
        let len = self.len();
        let Some(flags) = logical7_predicate_flags_read::<R, _, Pred>(&self, policy, false)? else {
            return Ok(0);
        };
        let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
        let selected = crate::detail::primitives::select::selected_rank_from_flags(
            policy, len, len_u32, flags,
        )?;
        mindex_from_usize(crate::detail::primitives::select::selected_count(
            policy, &selected,
        )?)
    }

    fn all_of_read<Pred>(self, policy: &CubePolicy<R>, _pred: Pred) -> Result<bool, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Pred: op::PredicateOp<R, Self::Item>,
    {
        let len = self.len();
        let Some(flags) = logical7_predicate_flags_read::<R, _, Pred>(&self, policy, true)? else {
            return Ok(true);
        };
        Ok(crate::detail::primitives::search::first_flag(policy, flags, len, len)?.is_none())
    }

    fn any_of_read<Pred>(self, policy: &CubePolicy<R>, _pred: Pred) -> Result<bool, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Pred: op::PredicateOp<R, Self::Item>,
    {
        let len = self.len();
        let Some(flags) = logical7_predicate_flags_read::<R, _, Pred>(&self, policy, false)? else {
            return Ok(false);
        };
        Ok(crate::detail::primitives::search::first_flag(policy, flags, len, len)?.is_some())
    }

    fn none_of_read<Pred>(self, policy: &CubePolicy<R>, _pred: Pred) -> Result<bool, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Pred: op::PredicateOp<R, Self::Item>,
    {
        let len = self.len();
        let Some(flags) = logical7_predicate_flags_read::<R, _, Pred>(&self, policy, false)? else {
            return Ok(true);
        };
        Ok(crate::detail::primitives::search::first_flag(policy, flags, len, len)?.is_none())
    }

    fn find_if_read<Pred>(
        self,
        policy: &CubePolicy<R>,
        _pred: Pred,
    ) -> Result<Option<crate::MIndex>, Error>
    where
        Self::Item: MItem<R>,
        Pred: op::PredicateOp<R, Self::Item>,
    {
        let len = self.len();
        let Some(flags) = logical7_predicate_flags_read::<R, _, Pred>(&self, policy, false)? else {
            return Ok(None);
        };
        crate::detail::primitives::search::first_flag(policy, flags, len, len)
    }

    fn is_partitioned_read<Pred>(self, policy: &CubePolicy<R>, _pred: Pred) -> Result<bool, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Pred: op::PredicateOp<R, Self::Item>,
    {
        let len = self.len();
        let Some(flags) = logical7_predicate_flags_read::<R, _, Pred>(&self, policy, false)? else {
            return Ok(true);
        };
        let first_rejected =
            crate::detail::primitives::search::first_unset_flag(policy, flags.clone(), len, len)?
                .unwrap_or(mindex_from_usize(len)?);
        let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
        let selected = crate::detail::primitives::select::selected_rank_from_flags(
            policy, len, len_u32, flags,
        )?;
        let selected_count = crate::detail::primitives::select::selected_count(policy, &selected)?;
        Ok(mindex_from_usize(selected_count)? == first_rejected)
    }

    fn scan_by_key_control_read<KeyEq>(
        self,
        policy: &CubePolicy<R>,
    ) -> Result<ScanByKeyControl<R>, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        KeyEq: op::BinaryPredicateOp<R, Self::Item>,
    {
        logical7_scan_by_key_control_read::<R, Self, KeyEq>(&self, policy)
    }

    fn inclusive_scan_by_key_values_read<KeyEq, Op, Output>(
        self,
        policy: &CubePolicy<R>,
        control: &ScanByKeyControl<R>,
        _op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R, Item = Self::Item>,
        Self::Item: MAlloc<R>,
        Op: op::ReductionOp<R, Self::Item>,
    {
        let apply = crate::detail::apply::SegmentedScanApply::new(control);
        let out = <Output as crate::iter::MIterMut<R>>::column_mut_view_inner::<T>(&output)?
            .ok_or_else(|| Error::Launch {
                message: "inclusive_scan_by_key output must match input shape".to_string(),
            })?;
        apply.inclusive_expr_into::<DeviceColumnView<R, T>, KernelScalarTuple1Op<R, Op>>(
            policy,
            &self.column,
            &out,
        )
    }

    fn exclusive_scan_by_key_values_read<KeyEq, Op, Output>(
        self,
        policy: &CubePolicy<R>,
        control: &ScanByKeyControl<R>,
        init: Self::Item,
        _op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R, Item = Self::Item>,
        Self::Item: MAlloc<R>,
        Op: op::ReductionOp<R, Self::Item>,
    {
        let apply = crate::detail::apply::SegmentedScanApply::new(control);
        let out = <Output as crate::iter::MIterMut<R>>::column_mut_view_inner::<T>(&output)?
            .ok_or_else(|| Error::Launch {
                message: "exclusive_scan_by_key output must match input shape".to_string(),
            })?;
        apply.exclusive_expr_into::<DeviceColumnView<R, T>, KernelScalarTuple1Op<R, Op>>(
            policy,
            &self.column,
            init,
            &out,
        )
    }

    fn copy_selected_read<Output>(
        self,
        policy: &CubePolicy<R>,
        stencil: PrecomputedSelection<R>,
        output: Output,
    ) -> Result<crate::MIndex, Error>
    where
        Output: MIterMut<R, Item = Self::Item>,
        Self::Item: MAlloc<R>,
    {
        let selected_rank = stencil.selected_rank();
        let count = crate::detail::primitives::select::selected_count(policy, selected_rank)?;
        let out = <Output as crate::iter::MIterMut<R>>::column_mut_view_inner::<T>(&output)?
            .ok_or_else(|| Error::Launch {
                message: "copy_where output must match input shape".to_string(),
            })?;
        let payload_apply = crate::detail::apply::SelectedPayloadApply::new(selected_rank, count);
        payload_apply.apply_expr_into(policy, &self.column, &out)?;
        let len = mindex_from_usize(count)?;
        Ok(len)
    }

    fn adjacent_find_read<Pred>(
        self,
        policy: &CubePolicy<R>,
        _pred: Pred,
    ) -> Result<Option<crate::MIndex>, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Pred: op::BinaryPredicateOp<R, Self::Item>,
    {
        logical7_adjacent_find_read::<R, Self, Pred>(&self, policy)
    }

    fn min_element_read<Less>(
        self,
        policy: &CubePolicy<R>,
        _less: Less,
    ) -> Result<Option<crate::MIndex>, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Less: op::BinaryPredicateOp<R, Self::Item>,
    {
        Ok(logical7_minmax_read::<R, Self, Less>(&self, policy)?.map(|pair| pair.0))
    }

    fn max_element_read<Less>(
        self,
        policy: &CubePolicy<R>,
        _less: Less,
    ) -> Result<Option<crate::MIndex>, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Less: op::BinaryPredicateOp<R, Self::Item>,
    {
        Ok(logical7_minmax_read::<R, Self, Less>(&self, policy)?.map(|pair| pair.1))
    }

    fn minmax_element_read<Less>(
        self,
        policy: &CubePolicy<R>,
        _less: Less,
    ) -> Result<Option<(crate::MIndex, crate::MIndex)>, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Less: op::BinaryPredicateOp<R, Self::Item>,
    {
        logical7_minmax_read::<R, Self, Less>(&self, policy)
    }

    fn is_sorted_read<Less>(self, policy: &CubePolicy<R>, _less: Less) -> Result<bool, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Less: op::BinaryPredicateOp<R, Self::Item>,
    {
        Ok(logical7_sorted_break_read::<R, Self, Less>(&self, policy)?.is_none())
    }

    fn is_sorted_until_read<Less>(
        self,
        policy: &CubePolicy<R>,
        _less: Less,
    ) -> Result<crate::MIndex, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Less: op::BinaryPredicateOp<R, Self::Item>,
    {
        match logical7_sorted_break_read::<R, Self, Less>(&self, policy)? {
            Some(index) => Ok(index + 1),
            None => mindex_from_usize(self.len()),
        }
    }
}

impl<R, T, Start> KernelReadAt<R, Start> for ColumnRead<R, T>
where
    R: Runtime,
    T: cubecl::prelude::CubePrimitive + cubecl::prelude::CubeElement + 'static,
    DeviceColumnView<R, T>: KernelColumnAt<Start>,
    <DeviceColumnView<R, T> as KernelColumnAt<Start>>::ExprAt: crate::expr::LogicalDeviceExpr<T>,
{
    type LogicalItem = T;
    type ExprAt = <DeviceColumnView<R, T> as KernelColumnAt<Start>>::ExprAt;
    type Next = <DeviceColumnView<R, T> as KernelColumnAt<Start>>::Next;

    fn stage_at(&self, bindings: &mut KernelColumnBindings) -> Result<(), Error> {
        <DeviceColumnView<R, T> as KernelColumnAt<Start>>::stage_at(&self.column, bindings)
    }
}

macro_rules! impl_column_read_at_env {
    (impl < $($env_ty:ident),* > $env:ty => $slot:ty, $next:ty) => {
        impl<R, T, $($env_ty),*> KernelReadAtEnv<R, $env> for ColumnRead<R, T>
        where
            R: Runtime,
            T: MStorageElement + 'static,
            $($env_ty: MStorageElement + 'static,)*
            DeviceColumnView<R, T>: KernelColumnAt<$slot>,
            <DeviceColumnView<R, T> as KernelColumnAt<$slot>>::ExprAt:
                crate::expr::LogicalDeviceExpr<T>,
            $next: EnvLeaf7,
        {
            type LogicalItem = T;
            type ExprAt = <DeviceColumnView<R, T> as KernelColumnAt<$slot>>::ExprAt;
            type NextEnv = $next;

            fn stage_at_env(&self, bindings: &mut KernelColumnBindings) -> Result<(), Error> {
                <DeviceColumnView<R, T> as KernelColumnAt<$slot>>::stage_at(&self.column, bindings)
            }
        }
    };
}

impl_column_read_at_env!(impl <> Env0 => S0, Env1<T>);
impl_column_read_at_env!(impl <A> Env1<A> => S1, Env2<A, T>);
impl_column_read_at_env!(impl <A, B> Env2<A, B> => S2, Env3<A, B, T>);
impl_column_read_at_env!(impl <A, B, C> Env3<A, B, C> => S3, Env4<A, B, C, T>);
impl_column_read_at_env!(impl <A, B, C, D> Env4<A, B, C, D> => S4, Env5<A, B, C, D, T>);
impl_column_read_at_env!(impl <A, B, C, D, E> Env5<A, B, C, D, E> => S5, Env6<A, B, C, D, E, T>);
impl_column_read_at_env!(impl <A, B, C, D, E, F> Env6<A, B, C, D, E, F> => S6, Env7<A, B, C, D, E, F, T>);

impl<R, T> KernelReadScanByKeyView<R> for ColumnRead<R, T>
where
    R: Runtime,
    T: MStorageElement + 'static,
{
    type View = DeviceColumnView<R, T>;

    fn into_scan_by_key_view(self) -> Self::View {
        self.column
    }
}

impl<R, T> KernelReadSearchView<R> for ColumnRead<R, T>
where
    R: Runtime,
    T: MStorageElement + 'static,
{
    type View = DeviceColumnView<R, T>;

    fn into_search_view(self) -> Self::View {
        self.column
    }
}

impl<R, T> KernelReadScanByKeyValuesView<R> for ColumnRead<R, T>
where
    R: Runtime,
    T: MStorageElement + 'static,
{
    type ExclusiveInit = T;

    fn into_exclusive_scan_by_key_init(init: Self::Item) -> Self::ExclusiveInit {
        init
    }
}

impl<R, T, Output, Op> KernelReadTransform<R, Output, Op> for ColumnRead<R, T>
where
    R: Runtime,
    T: MStorageElement + 'static,
    Output: MIterMut<R>,
    Output::Item: MAlloc<R> + MItemDispatch<R>,
    Op: op::UnaryOp<R, T, Output = Output::Item>,
{
    fn transform_into(self, policy: &CubePolicy<R>, op: Op, output: Output) -> Result<(), Error> {
        let inner =
            <Output::Item as MItemDispatch<R>>::transform_scalar_input(policy, self.column, op)?;
        output.write_from_inner(policy, inner)
    }

    fn transform_where_into(
        self,
        policy: &CubePolicy<R>,
        op: Op,
        stencil: PrecomputedSelection<R>,
        output: Output,
    ) -> Result<(), Error> {
        let inner =
            <Output::Item as MItemDispatch<R>>::transform_scalar_input(policy, self.column, op)?;
        output.write_where_from_inner(policy, inner, stencil)
    }
}

impl<R, Read> KernelStencilSelection<R> for Read
where
    R: Runtime,
    Read: KernelReadBoundMany<R, Item = bool>,
{
    fn stencil_selection(
        self,
        policy: &CubePolicy<R>,
        invert: bool,
        flags_only: bool,
    ) -> Result<PrecomputedSelection<R>, Error> {
        let len = self.len();
        let Some(flags) =
            logical7_predicate_flags_read::<R, _, crate::detail::op_adapter::ScalarStencilFlag>(
                &self, policy, invert,
            )?
        else {
            return Ok(PrecomputedSelection::from_selected_rank(
                select::SelectedRankControl::empty(policy.client()),
            ));
        };
        let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
        if flags_only {
            Ok(PrecomputedSelection::from_mask(
                policy.client(),
                select::MaskControl::from_flags(flags, len, len_u32),
            ))
        } else {
            Ok(PrecomputedSelection::from_selected_rank(
                select::selected_rank_from_flags(policy, len, len_u32, flags)?,
            ))
        }
    }
}

impl<R, T, Pred> KernelReadPredicateQuery<R, Pred> for ColumnRead<R, T>
where
    R: Runtime,
    T: MStorageElement + 'static,
    Pred: op::PredicateOp<R, T>,
{
    fn count_if(self, policy: &CubePolicy<R>, _pred: Pred) -> Result<crate::MIndex, Error> {
        crate::detail::count_if(policy, self.column, KernelOp::<R, Pred>::new())
    }

    fn all_of(self, policy: &CubePolicy<R>, _pred: Pred) -> Result<bool, Error> {
        crate::detail::all_of(policy, self.column, KernelOp::<R, Pred>::new())
    }

    fn any_of(self, policy: &CubePolicy<R>, _pred: Pred) -> Result<bool, Error> {
        crate::detail::any_of(policy, self.column, KernelOp::<R, Pred>::new())
    }

    fn none_of(self, policy: &CubePolicy<R>, _pred: Pred) -> Result<bool, Error> {
        crate::detail::none_of(policy, self.column, KernelOp::<R, Pred>::new())
    }

    fn find_if(self, policy: &CubePolicy<R>, _pred: Pred) -> Result<Option<crate::MIndex>, Error> {
        crate::detail::find_if(policy, self.column, KernelOp::<R, Pred>::new())
    }

    fn is_partitioned(self, policy: &CubePolicy<R>, _pred: Pred) -> Result<bool, Error> {
        crate::detail::is_partitioned(policy, self.column, KernelOp::<R, Pred>::new())
    }
}

impl<R, T, Pred> KernelReadAdjacentFind<R, Pred> for ColumnRead<R, T>
where
    R: Runtime,
    T: MStorageElement + 'static,
    Pred: op::BinaryPredicateOp<R, T>,
{
    fn adjacent_find(
        self,
        policy: &CubePolicy<R>,
        _pred: Pred,
    ) -> Result<Option<crate::MIndex>, Error> {
        crate::detail::adjacent_find(policy, self.column, KernelOp::<R, Pred>::new())
    }
}

macro_rules! impl_kernel_read_zip {
    ($name:ident < $first:ident : $first_field:ident > => ($($item:ty),+)) => {
        #[doc(hidden)]
        pub struct $name<$first> {
            pub(crate) $first_field: $first,
        }

        impl<$first> $name<$first> {
            pub(crate) fn new($first_field: $first) -> Self {
                Self { $first_field }
            }
        }

        impl<R, $first> KernelRead<R> for $name<$first>
        where
            R: Runtime,
            $first: KernelRead<R>,
            <$first as KernelRead<R>>::Item: Send + Sync,
            ($($item,)+): CubeType + crate::expr::LogicalItemPack7,
        {
            type Item = ($($item,)+);

            fn len(&self) -> usize {
                self.$first_field.len()
            }

            fn validate(&self) -> Result<(), Error> {
                self.$first_field.validate()
            }

            fn reduce_value_read<Op>(
                self,
                policy: &CubePolicy<R>,
                init: Self::Item,
                op: Op,
            ) -> Result<Self::Item, Error>
            where
                Self::Item: MItem<R> + Send + Sync,
                Self: KernelReadBoundMany<R>,
                Op: op::ReductionOp<R, Self::Item>,
            {
                let _ = op;
                reduce_logical7_bound_read::<R, Self, Op>(self, policy, init)
            }

            fn count_if_read<Pred>(
                self,
                policy: &CubePolicy<R>,
                pred: Pred,
            ) -> Result<crate::MIndex, Error>
            where
                Self::Item: MItem<R>,
                Self: KernelReadBoundMany<R>,
                Pred: op::PredicateOp<R, Self::Item>,
            {
                let _ = pred;
                let len = self.len();
                let Some(flags) = logical7_predicate_flags_read::<R, _, Pred>(&self, policy, false)?
                else {
                    return Ok(0);
                };
                let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
                let selected = crate::detail::primitives::select::selected_rank_from_flags(
                    policy, len, len_u32, flags,
                )?;
                mindex_from_usize(crate::detail::primitives::select::selected_count(
                    policy, &selected,
                )?)
            }

            fn all_of_read<Pred>(
                self,
                policy: &CubePolicy<R>,
                pred: Pred,
            ) -> Result<bool, Error>
            where
                Self::Item: MItem<R>,
                Self: KernelReadBoundMany<R>,
                Pred: op::PredicateOp<R, Self::Item>,
            {
                let _ = pred;
                let len = self.len();
                let Some(flags) = logical7_predicate_flags_read::<R, _, Pred>(&self, policy, true)?
                else {
                    return Ok(true);
                };
                Ok(crate::detail::primitives::search::first_flag(policy, flags, len, len)?.is_none())
            }

            fn any_of_read<Pred>(
                self,
                policy: &CubePolicy<R>,
                pred: Pred,
            ) -> Result<bool, Error>
            where
                Self::Item: MItem<R>,
                Self: KernelReadBoundMany<R>,
                Pred: op::PredicateOp<R, Self::Item>,
            {
                let _ = pred;
                let len = self.len();
                let Some(flags) = logical7_predicate_flags_read::<R, _, Pred>(&self, policy, false)?
                else {
                    return Ok(false);
                };
                Ok(crate::detail::primitives::search::first_flag(policy, flags, len, len)?.is_some())
            }

            fn none_of_read<Pred>(
                self,
                policy: &CubePolicy<R>,
                pred: Pred,
            ) -> Result<bool, Error>
            where
                Self::Item: MItem<R>,
                Self: KernelReadBoundMany<R>,
                Pred: op::PredicateOp<R, Self::Item>,
            {
                let _ = pred;
                let len = self.len();
                let Some(flags) = logical7_predicate_flags_read::<R, _, Pred>(&self, policy, false)?
                else {
                    return Ok(true);
                };
                Ok(crate::detail::primitives::search::first_flag(policy, flags, len, len)?.is_none())
            }

            fn find_if_read<Pred>(
                self,
                policy: &CubePolicy<R>,
                pred: Pred,
            ) -> Result<Option<crate::MIndex>, Error>
            where
                Self::Item: MItem<R>,
                Self: KernelReadBoundMany<R>,
                Pred: op::PredicateOp<R, Self::Item>,
            {
                let _ = pred;
                let len = self.len();
                let Some(flags) = logical7_predicate_flags_read::<R, _, Pred>(&self, policy, false)?
                else {
                    return Ok(None);
                };
                crate::detail::primitives::search::first_flag(policy, flags, len, len)
            }

            fn is_partitioned_read<Pred>(
                self,
                policy: &CubePolicy<R>,
                pred: Pred,
            ) -> Result<bool, Error>
            where
                Self::Item: MItem<R>,
                Self: KernelReadBoundMany<R>,
                Pred: op::PredicateOp<R, Self::Item>,
            {
                let _ = pred;
                let len = self.len();
                let Some(flags) = logical7_predicate_flags_read::<R, _, Pred>(&self, policy, false)?
                else {
                    return Ok(true);
                };
                let first_rejected =
                    crate::detail::primitives::search::first_unset_flag(policy, flags.clone(), len, len)?
                        .unwrap_or(mindex_from_usize(len)?);
                let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
                let selected = crate::detail::primitives::select::selected_rank_from_flags(
                    policy, len, len_u32, flags,
                )?;
                let selected_count =
                    crate::detail::primitives::select::selected_count(policy, &selected)?;
                Ok(mindex_from_usize(selected_count)? == first_rejected)
            }

            fn scan_by_key_control_read<KeyEq>(
                self,
                policy: &CubePolicy<R>,
            ) -> Result<ScanByKeyControl<R>, Error>
            where
                Self::Item: MItem<R>,
                Self: KernelReadBoundMany<R>,
                KeyEq: op::BinaryPredicateOp<R, Self::Item>,
            {
                logical7_scan_by_key_control_read::<R, Self, KeyEq>(&self, policy)
            }

            fn adjacent_find_read<Pred>(
                self,
                policy: &CubePolicy<R>,
                pred: Pred,
            ) -> Result<Option<crate::MIndex>, Error>
            where
                Self::Item: MItem<R>,
                Self: KernelReadBoundMany<R>,
                Pred: op::BinaryPredicateOp<R, Self::Item>,
            {
                let _ = pred;
                logical7_adjacent_find_read::<R, Self, Pred>(&self, policy)
            }

            fn min_element_read<Less>(
                self,
                policy: &CubePolicy<R>,
                less: Less,
            ) -> Result<Option<crate::MIndex>, Error>
            where
                Self::Item: MItem<R>,
                Self: KernelReadBoundMany<R>,
                Less: op::BinaryPredicateOp<R, Self::Item>,
            {
                let _ = less;
                Ok(logical7_minmax_read::<R, Self, Less>(&self, policy)?.map(|pair| pair.0))
            }

            fn max_element_read<Less>(
                self,
                policy: &CubePolicy<R>,
                less: Less,
            ) -> Result<Option<crate::MIndex>, Error>
            where
                Self::Item: MItem<R>,
                Self: KernelReadBoundMany<R>,
                Less: op::BinaryPredicateOp<R, Self::Item>,
            {
                let _ = less;
                Ok(logical7_minmax_read::<R, Self, Less>(&self, policy)?.map(|pair| pair.1))
            }

            fn minmax_element_read<Less>(
                self,
                policy: &CubePolicy<R>,
                less: Less,
            ) -> Result<Option<(crate::MIndex, crate::MIndex)>, Error>
            where
                Self::Item: MItem<R>,
                Self: KernelReadBoundMany<R>,
                Less: op::BinaryPredicateOp<R, Self::Item>,
            {
                let _ = less;
                logical7_minmax_read::<R, Self, Less>(&self, policy)
            }

            fn is_sorted_read<Less>(
                self,
                policy: &CubePolicy<R>,
                less: Less,
            ) -> Result<bool, Error>
            where
                Self::Item: MItem<R>,
                Self: KernelReadBoundMany<R>,
                Less: op::BinaryPredicateOp<R, Self::Item>,
            {
                let _ = less;
                Ok(logical7_sorted_break_read::<R, Self, Less>(&self, policy)?.is_none())
            }

            fn is_sorted_until_read<Less>(
                self,
                policy: &CubePolicy<R>,
                less: Less,
            ) -> Result<crate::MIndex, Error>
            where
                Self::Item: MItem<R>,
                Self: KernelReadBoundMany<R>,
                Less: op::BinaryPredicateOp<R, Self::Item>,
            {
                let _ = less;
                match logical7_sorted_break_read::<R, Self, Less>(&self, policy)? {
                    Some(index) => Ok(index + 1),
                    None => mindex_from_usize(self.len()),
                }
            }

            fn transform_read<Output, Op>(
                self,
                policy: &CubePolicy<R>,
                op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                Output: MIterMut<R>,
                Self::Item: MItem<R>,
                Self: KernelReadBoundMany<R>,
                Output::Item: MAlloc<R> + MItemDispatch<R>,
                Op: op::UnaryOp<R, Self::Item, Output = Output::Item>,
            {
                transform_logical7_read(self, policy, op, output)
            }

            fn transform_where_read<Output, Op>(
                self,
                policy: &CubePolicy<R>,
                op: Op,
                stencil: PrecomputedSelection<R>,
                output: Output,
            ) -> Result<(), Error>
            where
                Output: MIterMut<R>,
                Self::Item: MItem<R>,
                Self: KernelReadBoundMany<R>,
                Output::Item: MAlloc<R> + MItemDispatch<R>,
                Op: op::UnaryOp<R, Self::Item, Output = Output::Item>,
            {
                transform_where_logical7_read(self, policy, op, stencil, output)
            }
        }
    };

    ($name:ident < $first:ident : $first_field:ident, $( $ty:ident : $field:ident ),* > => ($($item:ty),+)) => {
        #[doc(hidden)]
        pub struct $name<$first, $( $ty ),*> {
            pub(crate) $first_field: $first,
            $(
                pub(crate) $field: $ty,
            )*
        }

        impl<$first, $( $ty ),*> $name<$first, $( $ty ),*> {
            pub(crate) fn new($first_field: $first, $( $field: $ty ),*) -> Self {
                Self { $first_field, $( $field ),* }
            }
        }

        impl<R, $first, $( $ty ),*> KernelRead<R> for $name<$first, $( $ty ),*>
        where
            R: Runtime,
            $first: KernelRead<R>,
            $(
                $ty: KernelRead<R>,
            )*
            <$first as KernelRead<R>>::Item: Send + Sync,
            $(
                <$ty as KernelRead<R>>::Item: Send + Sync,
            )*
            ($($item,)+): CubeType + crate::expr::LogicalItemPack7,
        {
            type Item = ($($item,)+);

            fn len(&self) -> usize {
                self.$first_field.len()
            }

            fn validate(&self) -> Result<(), Error> {
                self.$first_field.validate()?;
                $(
                    self.$field.validate()?;
                    ensure_same_len(self.$field.len(), self.$first_field.len())?;
                )*
                Ok(())
            }

            fn reduce_value_read<Op>(
                self,
                policy: &CubePolicy<R>,
                init: Self::Item,
                op: Op,
            ) -> Result<Self::Item, Error>
            where
                Self::Item: MItem<R> + Send + Sync,
                Self: KernelReadBoundMany<R>,
                Op: op::ReductionOp<R, Self::Item>,
            {
                let _ = op;
                reduce_logical7_bound_read::<R, Self, Op>(self, policy, init)
            }

            fn count_if_read<Pred>(
                self,
                policy: &CubePolicy<R>,
                pred: Pred,
            ) -> Result<crate::MIndex, Error>
            where
                Self::Item: MItem<R>,
                Self: KernelReadBoundMany<R>,
                Pred: op::PredicateOp<R, Self::Item>,
            {
                let _ = pred;
                let len = self.len();
                let Some(flags) = logical7_predicate_flags_read::<R, _, Pred>(&self, policy, false)?
                else {
                    return Ok(0);
                };
                let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
                let selected = crate::detail::primitives::select::selected_rank_from_flags(
                    policy, len, len_u32, flags,
                )?;
                mindex_from_usize(crate::detail::primitives::select::selected_count(
                    policy, &selected,
                )?)
            }

            fn all_of_read<Pred>(
                self,
                policy: &CubePolicy<R>,
                pred: Pred,
            ) -> Result<bool, Error>
            where
                Self::Item: MItem<R>,
                Self: KernelReadBoundMany<R>,
                Pred: op::PredicateOp<R, Self::Item>,
            {
                let _ = pred;
                let len = self.len();
                let Some(flags) = logical7_predicate_flags_read::<R, _, Pred>(&self, policy, true)?
                else {
                    return Ok(true);
                };
                Ok(crate::detail::primitives::search::first_flag(policy, flags, len, len)?.is_none())
            }

            fn any_of_read<Pred>(
                self,
                policy: &CubePolicy<R>,
                pred: Pred,
            ) -> Result<bool, Error>
            where
                Self::Item: MItem<R>,
                Self: KernelReadBoundMany<R>,
                Pred: op::PredicateOp<R, Self::Item>,
            {
                let _ = pred;
                let len = self.len();
                let Some(flags) = logical7_predicate_flags_read::<R, _, Pred>(&self, policy, false)?
                else {
                    return Ok(false);
                };
                Ok(crate::detail::primitives::search::first_flag(policy, flags, len, len)?.is_some())
            }

            fn none_of_read<Pred>(
                self,
                policy: &CubePolicy<R>,
                pred: Pred,
            ) -> Result<bool, Error>
            where
                Self::Item: MItem<R>,
                Self: KernelReadBoundMany<R>,
                Pred: op::PredicateOp<R, Self::Item>,
            {
                let _ = pred;
                let len = self.len();
                let Some(flags) = logical7_predicate_flags_read::<R, _, Pred>(&self, policy, false)?
                else {
                    return Ok(true);
                };
                Ok(crate::detail::primitives::search::first_flag(policy, flags, len, len)?.is_none())
            }

            fn find_if_read<Pred>(
                self,
                policy: &CubePolicy<R>,
                pred: Pred,
            ) -> Result<Option<crate::MIndex>, Error>
            where
                Self::Item: MItem<R>,
                Self: KernelReadBoundMany<R>,
                Pred: op::PredicateOp<R, Self::Item>,
            {
                let _ = pred;
                let len = self.len();
                let Some(flags) = logical7_predicate_flags_read::<R, _, Pred>(&self, policy, false)?
                else {
                    return Ok(None);
                };
                crate::detail::primitives::search::first_flag(policy, flags, len, len)
            }

            fn is_partitioned_read<Pred>(
                self,
                policy: &CubePolicy<R>,
                pred: Pred,
            ) -> Result<bool, Error>
            where
                Self::Item: MItem<R>,
                Self: KernelReadBoundMany<R>,
                Pred: op::PredicateOp<R, Self::Item>,
            {
                let _ = pred;
                let len = self.len();
                let Some(flags) = logical7_predicate_flags_read::<R, _, Pred>(&self, policy, false)?
                else {
                    return Ok(true);
                };
                let first_rejected =
                    crate::detail::primitives::search::first_unset_flag(policy, flags.clone(), len, len)?
                        .unwrap_or(mindex_from_usize(len)?);
                let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
                let selected = crate::detail::primitives::select::selected_rank_from_flags(
                    policy, len, len_u32, flags,
                )?;
                let selected_count =
                    crate::detail::primitives::select::selected_count(policy, &selected)?;
                Ok(mindex_from_usize(selected_count)? == first_rejected)
            }

            fn scan_by_key_control_read<KeyEq>(
                self,
                policy: &CubePolicy<R>,
            ) -> Result<ScanByKeyControl<R>, Error>
            where
                Self::Item: MItem<R>,
                Self: KernelReadBoundMany<R>,
                KeyEq: op::BinaryPredicateOp<R, Self::Item>,
            {
                logical7_scan_by_key_control_read::<R, Self, KeyEq>(&self, policy)
            }

            fn adjacent_find_read<Pred>(
                self,
                policy: &CubePolicy<R>,
                pred: Pred,
            ) -> Result<Option<crate::MIndex>, Error>
            where
                Self::Item: MItem<R>,
                Self: KernelReadBoundMany<R>,
                Pred: op::BinaryPredicateOp<R, Self::Item>,
            {
                let _ = pred;
                logical7_adjacent_find_read::<R, Self, Pred>(&self, policy)
            }

            fn min_element_read<Less>(
                self,
                policy: &CubePolicy<R>,
                less: Less,
            ) -> Result<Option<crate::MIndex>, Error>
            where
                Self::Item: MItem<R>,
                Self: KernelReadBoundMany<R>,
                Less: op::BinaryPredicateOp<R, Self::Item>,
            {
                let _ = less;
                Ok(logical7_minmax_read::<R, Self, Less>(&self, policy)?.map(|pair| pair.0))
            }

            fn max_element_read<Less>(
                self,
                policy: &CubePolicy<R>,
                less: Less,
            ) -> Result<Option<crate::MIndex>, Error>
            where
                Self::Item: MItem<R>,
                Self: KernelReadBoundMany<R>,
                Less: op::BinaryPredicateOp<R, Self::Item>,
            {
                let _ = less;
                Ok(logical7_minmax_read::<R, Self, Less>(&self, policy)?.map(|pair| pair.1))
            }

            fn minmax_element_read<Less>(
                self,
                policy: &CubePolicy<R>,
                less: Less,
            ) -> Result<Option<(crate::MIndex, crate::MIndex)>, Error>
            where
                Self::Item: MItem<R>,
                Self: KernelReadBoundMany<R>,
                Less: op::BinaryPredicateOp<R, Self::Item>,
            {
                let _ = less;
                logical7_minmax_read::<R, Self, Less>(&self, policy)
            }

            fn is_sorted_read<Less>(
                self,
                policy: &CubePolicy<R>,
                less: Less,
            ) -> Result<bool, Error>
            where
                Self::Item: MItem<R>,
                Self: KernelReadBoundMany<R>,
                Less: op::BinaryPredicateOp<R, Self::Item>,
            {
                let _ = less;
                Ok(logical7_sorted_break_read::<R, Self, Less>(&self, policy)?.is_none())
            }

            fn is_sorted_until_read<Less>(
                self,
                policy: &CubePolicy<R>,
                less: Less,
            ) -> Result<crate::MIndex, Error>
            where
                Self::Item: MItem<R>,
                Self: KernelReadBoundMany<R>,
                Less: op::BinaryPredicateOp<R, Self::Item>,
            {
                let _ = less;
                match logical7_sorted_break_read::<R, Self, Less>(&self, policy)? {
                    Some(index) => Ok(index + 1),
                    None => mindex_from_usize(self.len()),
                }
            }

            fn transform_read<Output, Op>(
                self,
                policy: &CubePolicy<R>,
                op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                Output: MIterMut<R>,
                Self::Item: MItem<R>,
                Self: KernelReadBoundMany<R>,
                Output::Item: MAlloc<R> + MItemDispatch<R>,
                Op: op::UnaryOp<R, Self::Item, Output = Output::Item>,
            {
                transform_logical7_read(self, policy, op, output)
            }

            fn transform_where_read<Output, Op>(
                self,
                policy: &CubePolicy<R>,
                op: Op,
                stencil: PrecomputedSelection<R>,
                output: Output,
            ) -> Result<(), Error>
            where
                Output: MIterMut<R>,
                Self::Item: MItem<R>,
                Self: KernelReadBoundMany<R>,
                Output::Item: MAlloc<R> + MItemDispatch<R>,
                Op: op::UnaryOp<R, Self::Item, Output = Output::Item>,
            {
                transform_where_logical7_read(self, policy, op, stencil, output)
            }
        }
    };
}

impl_kernel_read_zip!(ZipRead1<A: a> => (A::Item));

#[doc(hidden)]
pub struct ZipRead2<A, B> {
    pub(crate) a: A,
    pub(crate) b: B,
}

impl<A, B> ZipRead2<A, B> {
    pub(crate) fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<R, A, B> KernelRead<R> for ZipRead2<A, B>
where
    R: Runtime,
    A: KernelRead<R>,
    B: KernelRead<R>,
    <A as KernelRead<R>>::Item: Send + Sync,
    <B as KernelRead<R>>::Item: Send + Sync,
    (<A as KernelRead<R>>::Item, <B as KernelRead<R>>::Item):
        CubeType + crate::expr::LogicalItemPack7,
{
    type Item = (<A as KernelRead<R>>::Item, <B as KernelRead<R>>::Item);

    fn len(&self) -> usize {
        self.a.len()
    }

    fn validate(&self) -> Result<(), Error> {
        self.a.validate()?;
        self.b.validate()?;
        ensure_same_len(self.b.len(), self.a.len())?;
        Ok(())
    }

    fn reduce_value_read<Op>(
        self,
        policy: &CubePolicy<R>,
        init: Self::Item,
        op: Op,
    ) -> Result<Self::Item, Error>
    where
        Self::Item: MItem<R> + Send + Sync,
        Self: KernelReadBoundMany<R>,
        Op: op::ReductionOp<R, Self::Item>,
    {
        let _ = op;
        reduce_logical7_bound_read::<R, Self, Op>(self, policy, init)
    }

    fn count_if_read<Pred>(self, policy: &CubePolicy<R>, pred: Pred) -> Result<crate::MIndex, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Pred: op::PredicateOp<R, Self::Item>,
    {
        let _ = pred;
        let len = self.len();
        let Some(flags) = logical7_predicate_flags_read::<R, _, Pred>(&self, policy, false)? else {
            return Ok(0);
        };
        let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
        let selected = crate::detail::primitives::select::selected_rank_from_flags(
            policy, len, len_u32, flags,
        )?;
        mindex_from_usize(crate::detail::primitives::select::selected_count(
            policy, &selected,
        )?)
    }

    fn all_of_read<Pred>(self, policy: &CubePolicy<R>, pred: Pred) -> Result<bool, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Pred: op::PredicateOp<R, Self::Item>,
    {
        let _ = pred;
        let len = self.len();
        let Some(flags) = logical7_predicate_flags_read::<R, _, Pred>(&self, policy, true)? else {
            return Ok(true);
        };
        Ok(crate::detail::primitives::search::first_flag(policy, flags, len, len)?.is_none())
    }

    fn any_of_read<Pred>(self, policy: &CubePolicy<R>, pred: Pred) -> Result<bool, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Pred: op::PredicateOp<R, Self::Item>,
    {
        let _ = pred;
        let len = self.len();
        let Some(flags) = logical7_predicate_flags_read::<R, _, Pred>(&self, policy, false)? else {
            return Ok(false);
        };
        Ok(crate::detail::primitives::search::first_flag(policy, flags, len, len)?.is_some())
    }

    fn none_of_read<Pred>(self, policy: &CubePolicy<R>, pred: Pred) -> Result<bool, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Pred: op::PredicateOp<R, Self::Item>,
    {
        let _ = pred;
        let len = self.len();
        let Some(flags) = logical7_predicate_flags_read::<R, _, Pred>(&self, policy, false)? else {
            return Ok(true);
        };
        Ok(crate::detail::primitives::search::first_flag(policy, flags, len, len)?.is_none())
    }

    fn find_if_read<Pred>(
        self,
        policy: &CubePolicy<R>,
        pred: Pred,
    ) -> Result<Option<crate::MIndex>, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Pred: op::PredicateOp<R, Self::Item>,
    {
        let _ = pred;
        let len = self.len();
        let Some(flags) = logical7_predicate_flags_read::<R, _, Pred>(&self, policy, false)? else {
            return Ok(None);
        };
        crate::detail::primitives::search::first_flag(policy, flags, len, len)
    }

    fn is_partitioned_read<Pred>(self, policy: &CubePolicy<R>, pred: Pred) -> Result<bool, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Pred: op::PredicateOp<R, Self::Item>,
    {
        let _ = pred;
        let len = self.len();
        let Some(flags) = logical7_predicate_flags_read::<R, _, Pred>(&self, policy, false)? else {
            return Ok(true);
        };
        let first_rejected =
            crate::detail::primitives::search::first_unset_flag(policy, flags.clone(), len, len)?
                .unwrap_or(mindex_from_usize(len)?);
        let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
        let selected = crate::detail::primitives::select::selected_rank_from_flags(
            policy, len, len_u32, flags,
        )?;
        let selected_count = crate::detail::primitives::select::selected_count(policy, &selected)?;
        Ok(mindex_from_usize(selected_count)? == first_rejected)
    }

    fn scan_by_key_control_read<KeyEq>(
        self,
        policy: &CubePolicy<R>,
    ) -> Result<ScanByKeyControl<R>, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        KeyEq: op::BinaryPredicateOp<R, Self::Item>,
    {
        logical7_scan_by_key_control_read::<R, Self, KeyEq>(&self, policy)
    }

    fn adjacent_find_read<Pred>(
        self,
        policy: &CubePolicy<R>,
        pred: Pred,
    ) -> Result<Option<crate::MIndex>, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Pred: op::BinaryPredicateOp<R, Self::Item>,
    {
        let _ = pred;
        logical7_adjacent_find_read::<R, Self, Pred>(&self, policy)
    }

    fn min_element_read<Less>(
        self,
        policy: &CubePolicy<R>,
        less: Less,
    ) -> Result<Option<crate::MIndex>, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Less: op::BinaryPredicateOp<R, Self::Item>,
    {
        let _ = less;
        Ok(logical7_minmax_read::<R, Self, Less>(&self, policy)?.map(|pair| pair.0))
    }

    fn max_element_read<Less>(
        self,
        policy: &CubePolicy<R>,
        less: Less,
    ) -> Result<Option<crate::MIndex>, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Less: op::BinaryPredicateOp<R, Self::Item>,
    {
        let _ = less;
        Ok(logical7_minmax_read::<R, Self, Less>(&self, policy)?.map(|pair| pair.1))
    }

    fn minmax_element_read<Less>(
        self,
        policy: &CubePolicy<R>,
        less: Less,
    ) -> Result<Option<(crate::MIndex, crate::MIndex)>, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Less: op::BinaryPredicateOp<R, Self::Item>,
    {
        let _ = less;
        logical7_minmax_read::<R, Self, Less>(&self, policy)
    }

    fn is_sorted_read<Less>(self, policy: &CubePolicy<R>, less: Less) -> Result<bool, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Less: op::BinaryPredicateOp<R, Self::Item>,
    {
        let _ = less;
        Ok(logical7_sorted_break_read::<R, Self, Less>(&self, policy)?.is_none())
    }

    fn is_sorted_until_read<Less>(
        self,
        policy: &CubePolicy<R>,
        less: Less,
    ) -> Result<crate::MIndex, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Less: op::BinaryPredicateOp<R, Self::Item>,
    {
        let _ = less;
        match logical7_sorted_break_read::<R, Self, Less>(&self, policy)? {
            Some(index) => Ok(index + 1),
            None => mindex_from_usize(self.len()),
        }
    }

    fn transform_read<Output, Op>(
        self,
        policy: &CubePolicy<R>,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R>,
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Output::Item: MAlloc<R> + MItemDispatch<R>,
        Op: op::UnaryOp<R, Self::Item, Output = Output::Item>,
    {
        transform_logical7_read(self, policy, op, output)
    }

    fn transform_where_read<Output, Op>(
        self,
        policy: &CubePolicy<R>,
        op: Op,
        stencil: PrecomputedSelection<R>,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R>,
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Output::Item: MAlloc<R> + MItemDispatch<R>,
        Op: op::UnaryOp<R, Self::Item, Output = Output::Item>,
    {
        transform_where_logical7_read(self, policy, op, stencil, output)
    }
}

#[doc(hidden)]
pub struct ZipRead3<A, B, C> {
    pub(crate) a: A,
    pub(crate) b: B,
    pub(crate) c: C,
}

impl<A, B, C> ZipRead3<A, B, C> {
    pub(crate) fn new(a: A, b: B, c: C) -> Self {
        Self { a, b, c }
    }
}

impl<R, A, B, C> KernelRead<R> for ZipRead3<A, B, C>
where
    R: Runtime,
    A: KernelRead<R>,
    B: KernelRead<R>,
    C: KernelRead<R>,
    <A as KernelRead<R>>::Item: Send + Sync,
    <B as KernelRead<R>>::Item: Send + Sync,
    <C as KernelRead<R>>::Item: Send + Sync,
    (
        <A as KernelRead<R>>::Item,
        <B as KernelRead<R>>::Item,
        <C as KernelRead<R>>::Item,
    ): CubeType + crate::expr::LogicalItemPack7,
{
    type Item = (
        <A as KernelRead<R>>::Item,
        <B as KernelRead<R>>::Item,
        <C as KernelRead<R>>::Item,
    );

    fn len(&self) -> usize {
        self.a.len()
    }

    fn validate(&self) -> Result<(), Error> {
        self.a.validate()?;
        self.b.validate()?;
        ensure_same_len(self.b.len(), self.a.len())?;
        self.c.validate()?;
        ensure_same_len(self.c.len(), self.a.len())?;
        Ok(())
    }

    fn reduce_value_read<Op>(
        self,
        policy: &CubePolicy<R>,
        init: Self::Item,
        op: Op,
    ) -> Result<Self::Item, Error>
    where
        Self::Item: MItem<R> + Send + Sync,
        Self: KernelReadBoundMany<R>,
        Op: op::ReductionOp<R, Self::Item>,
    {
        let _ = op;
        reduce_logical7_bound_read::<R, Self, Op>(self, policy, init)
    }

    fn count_if_read<Pred>(self, policy: &CubePolicy<R>, pred: Pred) -> Result<crate::MIndex, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Pred: op::PredicateOp<R, Self::Item>,
    {
        let _ = pred;
        let len = self.len();
        let Some(flags) = logical7_predicate_flags_read::<R, _, Pred>(&self, policy, false)? else {
            return Ok(0);
        };
        let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
        let selected = crate::detail::primitives::select::selected_rank_from_flags(
            policy, len, len_u32, flags,
        )?;
        mindex_from_usize(crate::detail::primitives::select::selected_count(
            policy, &selected,
        )?)
    }

    fn all_of_read<Pred>(self, policy: &CubePolicy<R>, pred: Pred) -> Result<bool, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Pred: op::PredicateOp<R, Self::Item>,
    {
        let _ = pred;
        let len = self.len();
        let Some(flags) = logical7_predicate_flags_read::<R, _, Pred>(&self, policy, true)? else {
            return Ok(true);
        };
        Ok(crate::detail::primitives::search::first_flag(policy, flags, len, len)?.is_none())
    }

    fn any_of_read<Pred>(self, policy: &CubePolicy<R>, pred: Pred) -> Result<bool, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Pred: op::PredicateOp<R, Self::Item>,
    {
        let _ = pred;
        let len = self.len();
        let Some(flags) = logical7_predicate_flags_read::<R, _, Pred>(&self, policy, false)? else {
            return Ok(false);
        };
        Ok(crate::detail::primitives::search::first_flag(policy, flags, len, len)?.is_some())
    }

    fn none_of_read<Pred>(self, policy: &CubePolicy<R>, pred: Pred) -> Result<bool, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Pred: op::PredicateOp<R, Self::Item>,
    {
        let _ = pred;
        let len = self.len();
        let Some(flags) = logical7_predicate_flags_read::<R, _, Pred>(&self, policy, false)? else {
            return Ok(true);
        };
        Ok(crate::detail::primitives::search::first_flag(policy, flags, len, len)?.is_none())
    }

    fn find_if_read<Pred>(
        self,
        policy: &CubePolicy<R>,
        pred: Pred,
    ) -> Result<Option<crate::MIndex>, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Pred: op::PredicateOp<R, Self::Item>,
    {
        let _ = pred;
        let len = self.len();
        let Some(flags) = logical7_predicate_flags_read::<R, _, Pred>(&self, policy, false)? else {
            return Ok(None);
        };
        crate::detail::primitives::search::first_flag(policy, flags, len, len)
    }

    fn is_partitioned_read<Pred>(self, policy: &CubePolicy<R>, pred: Pred) -> Result<bool, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Pred: op::PredicateOp<R, Self::Item>,
    {
        let _ = pred;
        let len = self.len();
        let Some(flags) = logical7_predicate_flags_read::<R, _, Pred>(&self, policy, false)? else {
            return Ok(true);
        };
        let first_rejected =
            crate::detail::primitives::search::first_unset_flag(policy, flags.clone(), len, len)?
                .unwrap_or(mindex_from_usize(len)?);
        let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
        let selected = crate::detail::primitives::select::selected_rank_from_flags(
            policy, len, len_u32, flags,
        )?;
        let selected_count = crate::detail::primitives::select::selected_count(policy, &selected)?;
        Ok(mindex_from_usize(selected_count)? == first_rejected)
    }

    fn scan_by_key_control_read<KeyEq>(
        self,
        policy: &CubePolicy<R>,
    ) -> Result<ScanByKeyControl<R>, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        KeyEq: op::BinaryPredicateOp<R, Self::Item>,
    {
        logical7_scan_by_key_control_read::<R, Self, KeyEq>(&self, policy)
    }

    fn adjacent_find_read<Pred>(
        self,
        policy: &CubePolicy<R>,
        pred: Pred,
    ) -> Result<Option<crate::MIndex>, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Pred: op::BinaryPredicateOp<R, Self::Item>,
    {
        let _ = pred;
        logical7_adjacent_find_read::<R, Self, Pred>(&self, policy)
    }

    fn min_element_read<Less>(
        self,
        policy: &CubePolicy<R>,
        less: Less,
    ) -> Result<Option<crate::MIndex>, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Less: op::BinaryPredicateOp<R, Self::Item>,
    {
        let _ = less;
        Ok(logical7_minmax_read::<R, Self, Less>(&self, policy)?.map(|pair| pair.0))
    }

    fn max_element_read<Less>(
        self,
        policy: &CubePolicy<R>,
        less: Less,
    ) -> Result<Option<crate::MIndex>, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Less: op::BinaryPredicateOp<R, Self::Item>,
    {
        let _ = less;
        Ok(logical7_minmax_read::<R, Self, Less>(&self, policy)?.map(|pair| pair.1))
    }

    fn minmax_element_read<Less>(
        self,
        policy: &CubePolicy<R>,
        less: Less,
    ) -> Result<Option<(crate::MIndex, crate::MIndex)>, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Less: op::BinaryPredicateOp<R, Self::Item>,
    {
        let _ = less;
        logical7_minmax_read::<R, Self, Less>(&self, policy)
    }

    fn is_sorted_read<Less>(self, policy: &CubePolicy<R>, less: Less) -> Result<bool, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Less: op::BinaryPredicateOp<R, Self::Item>,
    {
        let _ = less;
        Ok(logical7_sorted_break_read::<R, Self, Less>(&self, policy)?.is_none())
    }

    fn is_sorted_until_read<Less>(
        self,
        policy: &CubePolicy<R>,
        less: Less,
    ) -> Result<crate::MIndex, Error>
    where
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Less: op::BinaryPredicateOp<R, Self::Item>,
    {
        let _ = less;
        match logical7_sorted_break_read::<R, Self, Less>(&self, policy)? {
            Some(index) => Ok(index + 1),
            None => mindex_from_usize(self.len()),
        }
    }

    fn transform_read<Output, Op>(
        self,
        policy: &CubePolicy<R>,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R>,
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Output::Item: MAlloc<R> + MItemDispatch<R>,
        Op: op::UnaryOp<R, Self::Item, Output = Output::Item>,
    {
        transform_logical7_read(self, policy, op, output)
    }

    fn transform_where_read<Output, Op>(
        self,
        policy: &CubePolicy<R>,
        op: Op,
        stencil: PrecomputedSelection<R>,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: MIterMut<R>,
        Self::Item: MItem<R>,
        Self: KernelReadBoundMany<R>,
        Output::Item: MAlloc<R> + MItemDispatch<R>,
        Op: op::UnaryOp<R, Self::Item, Output = Output::Item>,
    {
        transform_where_logical7_read(self, policy, op, stencil, output)
    }
}

impl_kernel_read_zip!(ZipRead4<A: a, B: b, C: c, D: d> => (
    A::Item,
    B::Item,
    C::Item,
    D::Item
));
impl_kernel_read_zip!(ZipRead5<A: a, B: b, C: c, D: d, E: e> => (
    A::Item,
    B::Item,
    C::Item,
    D::Item,
    E::Item
));
impl_kernel_read_zip!(ZipRead6<A: a, B: b, C: c, D: d, E: e, F: f> => (
    A::Item,
    B::Item,
    C::Item,
    D::Item,
    E::Item,
    F::Item
));
impl_kernel_read_zip!(ZipRead7<A: a, B: b, C: c, D: d, E: e, F: f, G: g> => (
    A::Item,
    B::Item,
    C::Item,
    D::Item,
    E::Item,
    F::Item,
    G::Item
));

impl<R, A, Start> KernelReadAt<R, Start> for ZipRead1<A>
where
    R: Runtime,
    A: KernelReadAt<R, Start>,
{
    type LogicalItem = (A::LogicalItem,);
    type ExprAt = (A::ExprAt,);
    type Next = A::Next;

    fn stage_at(&self, bindings: &mut KernelColumnBindings) -> Result<(), Error> {
        self.a.stage_at(bindings)
    }
}

macro_rules! impl_kernel_read_at_zip {
    ($name:ident < $first:ident : $first_field:ident, $second:ident : $second_field:ident >) => {
        impl<R, $first, $second, Start> KernelReadAt<R, Start> for $name<$first, $second>
        where
            R: Runtime,
            $first: KernelReadAt<R, Start>,
            $second: KernelReadAt<R, <$first as KernelReadAt<R, Start>>::Next>,
        {
            type LogicalItem = (
                <$first as KernelReadAt<R, Start>>::LogicalItem,
                <$second as KernelReadAt<
                    R,
                    <$first as KernelReadAt<R, Start>>::Next,
                >>::LogicalItem,
            );
            type ExprAt = (
                <$first as KernelReadAt<R, Start>>::ExprAt,
                <$second as KernelReadAt<
                    R,
                    <$first as KernelReadAt<R, Start>>::Next,
                >>::ExprAt,
            );
            type Next = <$second as KernelReadAt<
                R,
                <$first as KernelReadAt<R, Start>>::Next,
            >>::Next;

            fn stage_at(&self, bindings: &mut KernelColumnBindings) -> Result<(), Error> {
                self.$first_field.stage_at(bindings)?;
                self.$second_field.stage_at(bindings)
            }
        }
    };

    (
        $name:ident <
            $first:ident : $first_field:ident,
            $second:ident : $second_field:ident,
            $third:ident : $third_field:ident
        >
    ) => {
        impl<R, $first, $second, $third, Start> KernelReadAt<R, Start>
            for $name<$first, $second, $third>
        where
            R: Runtime,
            $first: KernelReadAt<R, Start>,
            $second: KernelReadAt<R, <$first as KernelReadAt<R, Start>>::Next>,
            $third: KernelReadAt<
                R,
                <$second as KernelReadAt<
                    R,
                    <$first as KernelReadAt<R, Start>>::Next,
                >>::Next,
            >,
        {
            type LogicalItem = (
                <$first as KernelReadAt<R, Start>>::LogicalItem,
                <$second as KernelReadAt<
                    R,
                    <$first as KernelReadAt<R, Start>>::Next,
                >>::LogicalItem,
                <$third as KernelReadAt<
                    R,
                    <$second as KernelReadAt<
                        R,
                        <$first as KernelReadAt<R, Start>>::Next,
                    >>::Next,
                >>::LogicalItem,
            );
            type ExprAt = (
                <$first as KernelReadAt<R, Start>>::ExprAt,
                <$second as KernelReadAt<
                    R,
                    <$first as KernelReadAt<R, Start>>::Next,
                >>::ExprAt,
                <$third as KernelReadAt<
                    R,
                    <$second as KernelReadAt<
                        R,
                        <$first as KernelReadAt<R, Start>>::Next,
                    >>::Next,
                >>::ExprAt,
            );
            type Next = <$third as KernelReadAt<
                R,
                <$second as KernelReadAt<
                    R,
                    <$first as KernelReadAt<R, Start>>::Next,
                >>::Next,
            >>::Next;

            fn stage_at(&self, bindings: &mut KernelColumnBindings) -> Result<(), Error> {
                self.$first_field.stage_at(bindings)?;
                self.$second_field.stage_at(bindings)?;
                self.$third_field.stage_at(bindings)
            }
        }
    };

    (
        $name:ident <
            $first:ident : $first_field:ident,
            $second:ident : $second_field:ident,
            $third:ident : $third_field:ident,
            $fourth:ident : $fourth_field:ident
        >
    ) => {
        impl<R, $first, $second, $third, $fourth, Start> KernelReadAt<R, Start>
            for $name<$first, $second, $third, $fourth>
        where
            R: Runtime,
            $first: KernelReadAt<R, Start>,
            $second: KernelReadAt<R, <$first as KernelReadAt<R, Start>>::Next>,
            $third: KernelReadAt<
                R,
                <$second as KernelReadAt<
                    R,
                    <$first as KernelReadAt<R, Start>>::Next,
                >>::Next,
            >,
            $fourth: KernelReadAt<
                R,
                <$third as KernelReadAt<
                    R,
                    <$second as KernelReadAt<
                        R,
                        <$first as KernelReadAt<R, Start>>::Next,
                    >>::Next,
                >>::Next,
            >,
        {
            type LogicalItem = (
                <$first as KernelReadAt<R, Start>>::LogicalItem,
                <$second as KernelReadAt<
                    R,
                    <$first as KernelReadAt<R, Start>>::Next,
                >>::LogicalItem,
                <$third as KernelReadAt<
                    R,
                    <$second as KernelReadAt<
                        R,
                        <$first as KernelReadAt<R, Start>>::Next,
                    >>::Next,
                >>::LogicalItem,
                <$fourth as KernelReadAt<
                    R,
                    <$third as KernelReadAt<
                        R,
                        <$second as KernelReadAt<
                            R,
                            <$first as KernelReadAt<R, Start>>::Next,
                        >>::Next,
                    >>::Next,
                >>::LogicalItem,
            );
            type ExprAt = (
                <$first as KernelReadAt<R, Start>>::ExprAt,
                <$second as KernelReadAt<
                    R,
                    <$first as KernelReadAt<R, Start>>::Next,
                >>::ExprAt,
                <$third as KernelReadAt<
                    R,
                    <$second as KernelReadAt<
                        R,
                        <$first as KernelReadAt<R, Start>>::Next,
                    >>::Next,
                >>::ExprAt,
                <$fourth as KernelReadAt<
                    R,
                    <$third as KernelReadAt<
                        R,
                        <$second as KernelReadAt<
                            R,
                            <$first as KernelReadAt<R, Start>>::Next,
                        >>::Next,
                    >>::Next,
                >>::ExprAt,
            );
            type Next = <$fourth as KernelReadAt<
                R,
                <$third as KernelReadAt<
                    R,
                    <$second as KernelReadAt<
                        R,
                        <$first as KernelReadAt<R, Start>>::Next,
                    >>::Next,
                >>::Next,
            >>::Next;

            fn stage_at(&self, bindings: &mut KernelColumnBindings) -> Result<(), Error> {
                self.$first_field.stage_at(bindings)?;
                self.$second_field.stage_at(bindings)?;
                self.$third_field.stage_at(bindings)?;
                self.$fourth_field.stage_at(bindings)
            }
        }
    };

    (
        $name:ident <
            $first:ident : $first_field:ident,
            $second:ident : $second_field:ident,
            $third:ident : $third_field:ident,
            $fourth:ident : $fourth_field:ident,
            $fifth:ident : $fifth_field:ident
        >
    ) => {
        impl<R, $first, $second, $third, $fourth, $fifth, Start> KernelReadAt<R, Start>
            for $name<$first, $second, $third, $fourth, $fifth>
        where
            R: Runtime,
            $first: KernelReadAt<R, Start>,
            $second: KernelReadAt<R, <$first as KernelReadAt<R, Start>>::Next>,
            $third: KernelReadAt<
                R,
                <$second as KernelReadAt<
                    R,
                    <$first as KernelReadAt<R, Start>>::Next,
                >>::Next,
            >,
            $fourth: KernelReadAt<
                R,
                <$third as KernelReadAt<
                    R,
                    <$second as KernelReadAt<
                        R,
                        <$first as KernelReadAt<R, Start>>::Next,
                    >>::Next,
                >>::Next,
            >,
            $fifth: KernelReadAt<
                R,
                <$fourth as KernelReadAt<
                    R,
                    <$third as KernelReadAt<
                        R,
                        <$second as KernelReadAt<
                            R,
                            <$first as KernelReadAt<R, Start>>::Next,
                        >>::Next,
                    >>::Next,
                >>::Next,
            >,
        {
            type LogicalItem = (
                <$first as KernelReadAt<R, Start>>::LogicalItem,
                <$second as KernelReadAt<
                    R,
                    <$first as KernelReadAt<R, Start>>::Next,
                >>::LogicalItem,
                <$third as KernelReadAt<
                    R,
                    <$second as KernelReadAt<
                        R,
                        <$first as KernelReadAt<R, Start>>::Next,
                    >>::Next,
                >>::LogicalItem,
                <$fourth as KernelReadAt<
                    R,
                    <$third as KernelReadAt<
                        R,
                        <$second as KernelReadAt<
                            R,
                            <$first as KernelReadAt<R, Start>>::Next,
                        >>::Next,
                    >>::Next,
                >>::LogicalItem,
                <$fifth as KernelReadAt<
                    R,
                    <$fourth as KernelReadAt<
                        R,
                        <$third as KernelReadAt<
                            R,
                            <$second as KernelReadAt<
                                R,
                                <$first as KernelReadAt<R, Start>>::Next,
                            >>::Next,
                        >>::Next,
                    >>::Next,
                >>::LogicalItem,
            );
            type ExprAt = (
                <$first as KernelReadAt<R, Start>>::ExprAt,
                <$second as KernelReadAt<
                    R,
                    <$first as KernelReadAt<R, Start>>::Next,
                >>::ExprAt,
                <$third as KernelReadAt<
                    R,
                    <$second as KernelReadAt<
                        R,
                        <$first as KernelReadAt<R, Start>>::Next,
                    >>::Next,
                >>::ExprAt,
                <$fourth as KernelReadAt<
                    R,
                    <$third as KernelReadAt<
                        R,
                        <$second as KernelReadAt<
                            R,
                            <$first as KernelReadAt<R, Start>>::Next,
                        >>::Next,
                    >>::Next,
                >>::ExprAt,
                <$fifth as KernelReadAt<
                    R,
                    <$fourth as KernelReadAt<
                        R,
                        <$third as KernelReadAt<
                            R,
                            <$second as KernelReadAt<
                                R,
                                <$first as KernelReadAt<R, Start>>::Next,
                            >>::Next,
                        >>::Next,
                    >>::Next,
                >>::ExprAt,
            );
            type Next = <$fifth as KernelReadAt<
                R,
                <$fourth as KernelReadAt<
                    R,
                    <$third as KernelReadAt<
                        R,
                        <$second as KernelReadAt<
                            R,
                            <$first as KernelReadAt<R, Start>>::Next,
                        >>::Next,
                    >>::Next,
                >>::Next,
            >>::Next;

            fn stage_at(&self, bindings: &mut KernelColumnBindings) -> Result<(), Error> {
                self.$first_field.stage_at(bindings)?;
                self.$second_field.stage_at(bindings)?;
                self.$third_field.stage_at(bindings)?;
                self.$fourth_field.stage_at(bindings)?;
                self.$fifth_field.stage_at(bindings)
            }
        }
    };

    (
        $name:ident <
            $first:ident : $first_field:ident,
            $second:ident : $second_field:ident,
            $third:ident : $third_field:ident,
            $fourth:ident : $fourth_field:ident,
            $fifth:ident : $fifth_field:ident,
            $sixth:ident : $sixth_field:ident
        >
    ) => {
        impl<R, $first, $second, $third, $fourth, $fifth, $sixth, Start>
            KernelReadAt<R, Start> for $name<$first, $second, $third, $fourth, $fifth, $sixth>
        where
            R: Runtime,
            $first: KernelReadAt<R, Start>,
            $second: KernelReadAt<R, <$first as KernelReadAt<R, Start>>::Next>,
            $third: KernelReadAt<
                R,
                <$second as KernelReadAt<
                    R,
                    <$first as KernelReadAt<R, Start>>::Next,
                >>::Next,
            >,
            $fourth: KernelReadAt<
                R,
                <$third as KernelReadAt<
                    R,
                    <$second as KernelReadAt<
                        R,
                        <$first as KernelReadAt<R, Start>>::Next,
                    >>::Next,
                >>::Next,
            >,
            $fifth: KernelReadAt<
                R,
                <$fourth as KernelReadAt<
                    R,
                    <$third as KernelReadAt<
                        R,
                        <$second as KernelReadAt<
                            R,
                            <$first as KernelReadAt<R, Start>>::Next,
                        >>::Next,
                    >>::Next,
                >>::Next,
            >,
            $sixth: KernelReadAt<
                R,
                <$fifth as KernelReadAt<
                    R,
                    <$fourth as KernelReadAt<
                        R,
                        <$third as KernelReadAt<
                            R,
                            <$second as KernelReadAt<
                                R,
                                <$first as KernelReadAt<R, Start>>::Next,
                            >>::Next,
                        >>::Next,
                    >>::Next,
                >>::Next,
            >,
        {
            type LogicalItem = (
                <$first as KernelReadAt<R, Start>>::LogicalItem,
                <$second as KernelReadAt<
                    R,
                    <$first as KernelReadAt<R, Start>>::Next,
                >>::LogicalItem,
                <$third as KernelReadAt<
                    R,
                    <$second as KernelReadAt<
                        R,
                        <$first as KernelReadAt<R, Start>>::Next,
                    >>::Next,
                >>::LogicalItem,
                <$fourth as KernelReadAt<
                    R,
                    <$third as KernelReadAt<
                        R,
                        <$second as KernelReadAt<
                            R,
                            <$first as KernelReadAt<R, Start>>::Next,
                        >>::Next,
                    >>::Next,
                >>::LogicalItem,
                <$fifth as KernelReadAt<
                    R,
                    <$fourth as KernelReadAt<
                        R,
                        <$third as KernelReadAt<
                            R,
                            <$second as KernelReadAt<
                                R,
                                <$first as KernelReadAt<R, Start>>::Next,
                            >>::Next,
                        >>::Next,
                    >>::Next,
                >>::LogicalItem,
                <$sixth as KernelReadAt<
                    R,
                    <$fifth as KernelReadAt<
                        R,
                        <$fourth as KernelReadAt<
                            R,
                            <$third as KernelReadAt<
                                R,
                                <$second as KernelReadAt<
                                    R,
                                    <$first as KernelReadAt<R, Start>>::Next,
                                >>::Next,
                            >>::Next,
                        >>::Next,
                    >>::Next,
                >>::LogicalItem,
            );
            type ExprAt = (
                <$first as KernelReadAt<R, Start>>::ExprAt,
                <$second as KernelReadAt<
                    R,
                    <$first as KernelReadAt<R, Start>>::Next,
                >>::ExprAt,
                <$third as KernelReadAt<
                    R,
                    <$second as KernelReadAt<
                        R,
                        <$first as KernelReadAt<R, Start>>::Next,
                    >>::Next,
                >>::ExprAt,
                <$fourth as KernelReadAt<
                    R,
                    <$third as KernelReadAt<
                        R,
                        <$second as KernelReadAt<
                            R,
                            <$first as KernelReadAt<R, Start>>::Next,
                        >>::Next,
                    >>::Next,
                >>::ExprAt,
                <$fifth as KernelReadAt<
                    R,
                    <$fourth as KernelReadAt<
                        R,
                        <$third as KernelReadAt<
                            R,
                            <$second as KernelReadAt<
                                R,
                                <$first as KernelReadAt<R, Start>>::Next,
                            >>::Next,
                        >>::Next,
                    >>::Next,
                >>::ExprAt,
                <$sixth as KernelReadAt<
                    R,
                    <$fifth as KernelReadAt<
                        R,
                        <$fourth as KernelReadAt<
                            R,
                            <$third as KernelReadAt<
                                R,
                                <$second as KernelReadAt<
                                    R,
                                    <$first as KernelReadAt<R, Start>>::Next,
                                >>::Next,
                            >>::Next,
                        >>::Next,
                    >>::Next,
                >>::ExprAt,
            );
            type Next = <$sixth as KernelReadAt<
                R,
                <$fifth as KernelReadAt<
                    R,
                    <$fourth as KernelReadAt<
                        R,
                        <$third as KernelReadAt<
                            R,
                            <$second as KernelReadAt<
                                R,
                                <$first as KernelReadAt<R, Start>>::Next,
                            >>::Next,
                        >>::Next,
                    >>::Next,
                >>::Next,
            >>::Next;

            fn stage_at(&self, bindings: &mut KernelColumnBindings) -> Result<(), Error> {
                self.$first_field.stage_at(bindings)?;
                self.$second_field.stage_at(bindings)?;
                self.$third_field.stage_at(bindings)?;
                self.$fourth_field.stage_at(bindings)?;
                self.$fifth_field.stage_at(bindings)?;
                self.$sixth_field.stage_at(bindings)
            }
        }
    };

    (
        $name:ident <
            $first:ident : $first_field:ident,
            $second:ident : $second_field:ident,
            $third:ident : $third_field:ident,
            $fourth:ident : $fourth_field:ident,
            $fifth:ident : $fifth_field:ident,
            $sixth:ident : $sixth_field:ident,
            $seventh:ident : $seventh_field:ident
        >
    ) => {
        impl<R, $first, $second, $third, $fourth, $fifth, $sixth, $seventh, Start>
            KernelReadAt<R, Start>
            for $name<$first, $second, $third, $fourth, $fifth, $sixth, $seventh>
        where
            R: Runtime,
            $first: KernelReadAt<R, Start>,
            $second: KernelReadAt<R, <$first as KernelReadAt<R, Start>>::Next>,
            $third: KernelReadAt<
                R,
                <$second as KernelReadAt<
                    R,
                    <$first as KernelReadAt<R, Start>>::Next,
                >>::Next,
            >,
            $fourth: KernelReadAt<
                R,
                <$third as KernelReadAt<
                    R,
                    <$second as KernelReadAt<
                        R,
                        <$first as KernelReadAt<R, Start>>::Next,
                    >>::Next,
                >>::Next,
            >,
            $fifth: KernelReadAt<
                R,
                <$fourth as KernelReadAt<
                    R,
                    <$third as KernelReadAt<
                        R,
                        <$second as KernelReadAt<
                            R,
                            <$first as KernelReadAt<R, Start>>::Next,
                        >>::Next,
                    >>::Next,
                >>::Next,
            >,
            $sixth: KernelReadAt<
                R,
                <$fifth as KernelReadAt<
                    R,
                    <$fourth as KernelReadAt<
                        R,
                        <$third as KernelReadAt<
                            R,
                            <$second as KernelReadAt<
                                R,
                                <$first as KernelReadAt<R, Start>>::Next,
                            >>::Next,
                        >>::Next,
                    >>::Next,
                >>::Next,
            >,
            $seventh: KernelReadAt<
                R,
                <$sixth as KernelReadAt<
                    R,
                    <$fifth as KernelReadAt<
                        R,
                        <$fourth as KernelReadAt<
                            R,
                            <$third as KernelReadAt<
                                R,
                                <$second as KernelReadAt<
                                    R,
                                    <$first as KernelReadAt<R, Start>>::Next,
                                >>::Next,
                            >>::Next,
                        >>::Next,
                    >>::Next,
                >>::Next,
            >,
        {
            type LogicalItem = (
                <$first as KernelReadAt<R, Start>>::LogicalItem,
                <$second as KernelReadAt<
                    R,
                    <$first as KernelReadAt<R, Start>>::Next,
                >>::LogicalItem,
                <$third as KernelReadAt<
                    R,
                    <$second as KernelReadAt<
                        R,
                        <$first as KernelReadAt<R, Start>>::Next,
                    >>::Next,
                >>::LogicalItem,
                <$fourth as KernelReadAt<
                    R,
                    <$third as KernelReadAt<
                        R,
                        <$second as KernelReadAt<
                            R,
                            <$first as KernelReadAt<R, Start>>::Next,
                        >>::Next,
                    >>::Next,
                >>::LogicalItem,
                <$fifth as KernelReadAt<
                    R,
                    <$fourth as KernelReadAt<
                        R,
                        <$third as KernelReadAt<
                            R,
                            <$second as KernelReadAt<
                                R,
                                <$first as KernelReadAt<R, Start>>::Next,
                            >>::Next,
                        >>::Next,
                    >>::Next,
                >>::LogicalItem,
                <$sixth as KernelReadAt<
                    R,
                    <$fifth as KernelReadAt<
                        R,
                        <$fourth as KernelReadAt<
                            R,
                            <$third as KernelReadAt<
                                R,
                                <$second as KernelReadAt<
                                    R,
                                    <$first as KernelReadAt<R, Start>>::Next,
                                >>::Next,
                            >>::Next,
                        >>::Next,
                    >>::Next,
                >>::LogicalItem,
                <$seventh as KernelReadAt<
                    R,
                    <$sixth as KernelReadAt<
                        R,
                        <$fifth as KernelReadAt<
                            R,
                            <$fourth as KernelReadAt<
                                R,
                                <$third as KernelReadAt<
                                    R,
                                    <$second as KernelReadAt<
                                        R,
                                        <$first as KernelReadAt<R, Start>>::Next,
                                    >>::Next,
                                >>::Next,
                            >>::Next,
                        >>::Next,
                    >>::Next,
                >>::LogicalItem,
            );
            type ExprAt = (
                <$first as KernelReadAt<R, Start>>::ExprAt,
                <$second as KernelReadAt<
                    R,
                    <$first as KernelReadAt<R, Start>>::Next,
                >>::ExprAt,
                <$third as KernelReadAt<
                    R,
                    <$second as KernelReadAt<
                        R,
                        <$first as KernelReadAt<R, Start>>::Next,
                    >>::Next,
                >>::ExprAt,
                <$fourth as KernelReadAt<
                    R,
                    <$third as KernelReadAt<
                        R,
                        <$second as KernelReadAt<
                            R,
                            <$first as KernelReadAt<R, Start>>::Next,
                        >>::Next,
                    >>::Next,
                >>::ExprAt,
                <$fifth as KernelReadAt<
                    R,
                    <$fourth as KernelReadAt<
                        R,
                        <$third as KernelReadAt<
                            R,
                            <$second as KernelReadAt<
                                R,
                                <$first as KernelReadAt<R, Start>>::Next,
                            >>::Next,
                        >>::Next,
                    >>::Next,
                >>::ExprAt,
                <$sixth as KernelReadAt<
                    R,
                    <$fifth as KernelReadAt<
                        R,
                        <$fourth as KernelReadAt<
                            R,
                            <$third as KernelReadAt<
                                R,
                                <$second as KernelReadAt<
                                    R,
                                    <$first as KernelReadAt<R, Start>>::Next,
                                >>::Next,
                            >>::Next,
                        >>::Next,
                    >>::Next,
                >>::ExprAt,
                <$seventh as KernelReadAt<
                    R,
                    <$sixth as KernelReadAt<
                        R,
                        <$fifth as KernelReadAt<
                            R,
                            <$fourth as KernelReadAt<
                                R,
                                <$third as KernelReadAt<
                                    R,
                                    <$second as KernelReadAt<
                                        R,
                                        <$first as KernelReadAt<R, Start>>::Next,
                                    >>::Next,
                                >>::Next,
                            >>::Next,
                        >>::Next,
                    >>::Next,
                >>::ExprAt,
            );
            type Next = <$seventh as KernelReadAt<
                R,
                <$sixth as KernelReadAt<
                    R,
                    <$fifth as KernelReadAt<
                        R,
                        <$fourth as KernelReadAt<
                            R,
                            <$third as KernelReadAt<
                                R,
                                <$second as KernelReadAt<
                                    R,
                                    <$first as KernelReadAt<R, Start>>::Next,
                                >>::Next,
                            >>::Next,
                        >>::Next,
                    >>::Next,
                >>::Next,
            >>::Next;

            fn stage_at(&self, bindings: &mut KernelColumnBindings) -> Result<(), Error> {
                self.$first_field.stage_at(bindings)?;
                self.$second_field.stage_at(bindings)?;
                self.$third_field.stage_at(bindings)?;
                self.$fourth_field.stage_at(bindings)?;
                self.$fifth_field.stage_at(bindings)?;
                self.$sixth_field.stage_at(bindings)?;
                self.$seventh_field.stage_at(bindings)
            }
        }
    };
}

impl_kernel_read_at_zip!(ZipRead2<A: a, B: b>);
impl_kernel_read_at_zip!(ZipRead3<A: a, B: b, C: c>);
impl_kernel_read_at_zip!(ZipRead4<A: a, B: b, C: c, D: d>);
impl_kernel_read_at_zip!(ZipRead5<A: a, B: b, C: c, D: d, E: e>);
impl_kernel_read_at_zip!(ZipRead6<A: a, B: b, C: c, D: d, E: e, F: f>);
impl_kernel_read_at_zip!(ZipRead7<A: a, B: b, C: c, D: d, E: e, F: f, G: g>);

impl<R, A, Env> KernelReadAtEnv<R, Env> for ZipRead1<A>
where
    R: Runtime,
    A: KernelReadAtEnv<R, Env>,
{
    type LogicalItem = (A::LogicalItem,);
    type ExprAt = (A::ExprAt,);
    type NextEnv = A::NextEnv;

    fn stage_at_env(&self, bindings: &mut KernelColumnBindings) -> Result<(), Error> {
        self.a.stage_at_env(bindings)
    }
}

impl<R, A, B, Env> KernelReadAtEnv<R, Env> for ZipRead2<A, B>
where
    R: Runtime,
    A: KernelReadAtEnv<R, Env>,
    B: KernelReadAtEnv<R, A::NextEnv>,
{
    type LogicalItem = (A::LogicalItem, B::LogicalItem);
    type ExprAt = (A::ExprAt, B::ExprAt);
    type NextEnv = B::NextEnv;

    fn stage_at_env(&self, bindings: &mut KernelColumnBindings) -> Result<(), Error> {
        self.a.stage_at_env(bindings)?;
        self.b.stage_at_env(bindings)
    }
}

impl<R, A, B, C, Env> KernelReadAtEnv<R, Env> for ZipRead3<A, B, C>
where
    R: Runtime,
    A: KernelReadAtEnv<R, Env>,
    B: KernelReadAtEnv<R, A::NextEnv>,
    C: KernelReadAtEnv<R, B::NextEnv>,
{
    type LogicalItem = (A::LogicalItem, B::LogicalItem, C::LogicalItem);
    type ExprAt = (A::ExprAt, B::ExprAt, C::ExprAt);
    type NextEnv = C::NextEnv;

    fn stage_at_env(&self, bindings: &mut KernelColumnBindings) -> Result<(), Error> {
        self.a.stage_at_env(bindings)?;
        self.b.stage_at_env(bindings)?;
        self.c.stage_at_env(bindings)
    }
}

impl<R, A, B, C, D, Env> KernelReadAtEnv<R, Env> for ZipRead4<A, B, C, D>
where
    R: Runtime,
    A: KernelReadAtEnv<R, Env>,
    B: KernelReadAtEnv<R, A::NextEnv>,
    C: KernelReadAtEnv<R, B::NextEnv>,
    D: KernelReadAtEnv<R, C::NextEnv>,
{
    type LogicalItem = (
        A::LogicalItem,
        B::LogicalItem,
        C::LogicalItem,
        D::LogicalItem,
    );
    type ExprAt = (A::ExprAt, B::ExprAt, C::ExprAt, D::ExprAt);
    type NextEnv = D::NextEnv;

    fn stage_at_env(&self, bindings: &mut KernelColumnBindings) -> Result<(), Error> {
        self.a.stage_at_env(bindings)?;
        self.b.stage_at_env(bindings)?;
        self.c.stage_at_env(bindings)?;
        self.d.stage_at_env(bindings)
    }
}

impl<R, A, B, C, D, E, Env> KernelReadAtEnv<R, Env> for ZipRead5<A, B, C, D, E>
where
    R: Runtime,
    A: KernelReadAtEnv<R, Env>,
    B: KernelReadAtEnv<R, A::NextEnv>,
    C: KernelReadAtEnv<R, B::NextEnv>,
    D: KernelReadAtEnv<R, C::NextEnv>,
    E: KernelReadAtEnv<R, D::NextEnv>,
{
    type LogicalItem = (
        A::LogicalItem,
        B::LogicalItem,
        C::LogicalItem,
        D::LogicalItem,
        E::LogicalItem,
    );
    type ExprAt = (A::ExprAt, B::ExprAt, C::ExprAt, D::ExprAt, E::ExprAt);
    type NextEnv = E::NextEnv;

    fn stage_at_env(&self, bindings: &mut KernelColumnBindings) -> Result<(), Error> {
        self.a.stage_at_env(bindings)?;
        self.b.stage_at_env(bindings)?;
        self.c.stage_at_env(bindings)?;
        self.d.stage_at_env(bindings)?;
        self.e.stage_at_env(bindings)
    }
}

impl<R, A, B, C, D, E, F, Env> KernelReadAtEnv<R, Env> for ZipRead6<A, B, C, D, E, F>
where
    R: Runtime,
    A: KernelReadAtEnv<R, Env>,
    B: KernelReadAtEnv<R, A::NextEnv>,
    C: KernelReadAtEnv<R, B::NextEnv>,
    D: KernelReadAtEnv<R, C::NextEnv>,
    E: KernelReadAtEnv<R, D::NextEnv>,
    F: KernelReadAtEnv<R, E::NextEnv>,
{
    type LogicalItem = (
        A::LogicalItem,
        B::LogicalItem,
        C::LogicalItem,
        D::LogicalItem,
        E::LogicalItem,
        F::LogicalItem,
    );
    type ExprAt = (
        A::ExprAt,
        B::ExprAt,
        C::ExprAt,
        D::ExprAt,
        E::ExprAt,
        F::ExprAt,
    );
    type NextEnv = F::NextEnv;

    fn stage_at_env(&self, bindings: &mut KernelColumnBindings) -> Result<(), Error> {
        self.a.stage_at_env(bindings)?;
        self.b.stage_at_env(bindings)?;
        self.c.stage_at_env(bindings)?;
        self.d.stage_at_env(bindings)?;
        self.e.stage_at_env(bindings)?;
        self.f.stage_at_env(bindings)
    }
}

impl<R, A, B, C, D, E, F, G, Env> KernelReadAtEnv<R, Env> for ZipRead7<A, B, C, D, E, F, G>
where
    R: Runtime,
    A: KernelReadAtEnv<R, Env>,
    B: KernelReadAtEnv<R, A::NextEnv>,
    C: KernelReadAtEnv<R, B::NextEnv>,
    D: KernelReadAtEnv<R, C::NextEnv>,
    E: KernelReadAtEnv<R, D::NextEnv>,
    F: KernelReadAtEnv<R, E::NextEnv>,
    G: KernelReadAtEnv<R, F::NextEnv>,
{
    type LogicalItem = (
        A::LogicalItem,
        B::LogicalItem,
        C::LogicalItem,
        D::LogicalItem,
        E::LogicalItem,
        F::LogicalItem,
        G::LogicalItem,
    );
    type ExprAt = (
        A::ExprAt,
        B::ExprAt,
        C::ExprAt,
        D::ExprAt,
        E::ExprAt,
        F::ExprAt,
        G::ExprAt,
    );
    type NextEnv = G::NextEnv;

    fn stage_at_env(&self, bindings: &mut KernelColumnBindings) -> Result<(), Error> {
        self.a.stage_at_env(bindings)?;
        self.b.stage_at_env(bindings)?;
        self.c.stage_at_env(bindings)?;
        self.d.stage_at_env(bindings)?;
        self.e.stage_at_env(bindings)?;
        self.f.stage_at_env(bindings)?;
        self.g.stage_at_env(bindings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_zip_read_expr_preserves_logical_shape() {
        fn assert_shape<R>()
        where
            R: Runtime,
            ZipRead2<
                ZipRead2<ColumnRead<R, u32>, ColumnRead<R, f32>>,
                ColumnRead<R, i32>,
            >: KernelRead<R, Item = ((u32, f32), i32)> + KernelReadAt<R, S0>,
            <ZipRead2<
                ZipRead2<ColumnRead<R, u32>, ColumnRead<R, f32>>,
                ColumnRead<R, i32>,
            > as KernelReadAt<R, S0>>::ExprAt:
                crate::expr::LogicalDeviceExpr<((u32, f32), i32)>,
        {
        }

        assert_shape::<cubecl::wgpu::WgpuRuntime>();
    }

    #[test]
    fn zip7_read_expr_stages_all_logical_leaves() {
        fn assert_shape<R>()
        where
            R: Runtime,
            ZipRead7<
                ColumnRead<R, u8>,
                ColumnRead<R, u16>,
                ColumnRead<R, u32>,
                ColumnRead<R, u64>,
                ColumnRead<R, i8>,
                ColumnRead<R, i16>,
                ColumnRead<R, i32>,
            >: KernelRead<R, Item = (u8, u16, u32, u64, i8, i16, i32)> + KernelReadAt<R, S0>,
            <ZipRead7<
                ColumnRead<R, u8>,
                ColumnRead<R, u16>,
                ColumnRead<R, u32>,
                ColumnRead<R, u64>,
                ColumnRead<R, i8>,
                ColumnRead<R, i16>,
                ColumnRead<R, i32>,
            > as KernelReadAt<R, S0>>::ExprAt:
                crate::expr::LogicalDeviceExpr<(u8, u16, u32, u64, i8, i16, i32)>,
        {
        }

        assert_shape::<cubecl::wgpu::WgpuRuntime>();
    }
}

impl<R, A, Op> KernelReadReduce<R, Op> for ColumnRead<R, A>
where
    R: Runtime,
    A: MStorageElement + 'static,
    Op: op::ReductionOp<R, A>,
{
    fn reduce_value(
        self,
        policy: &CubePolicy<R>,
        init: Self::Item,
        _op: Op,
    ) -> Result<Self::Item, Error> {
        let result = crate::detail::reduce(
            policy,
            (self.column,),
            (init,),
            KernelScalarTuple1Op::<R, Op>::new(),
        )?;
        Ok(result.0)
    }
}

impl<R, A, Op> KernelReadReduce<R, Op> for ZipRead1<ColumnRead<R, A>>
where
    R: Runtime,
    A: MStorageElement + 'static,
    Op: op::ReductionOp<R, (A,)>,
{
    fn reduce_value(
        self,
        policy: &CubePolicy<R>,
        init: Self::Item,
        _op: Op,
    ) -> Result<Self::Item, Error> {
        crate::detail::reduce(policy, (self.a.column,), init, KernelOp::<R, Op>::new())
    }
}

macro_rules! impl_flat_zip_reduce_small {
    ($name:ident; $( $ty:ident : $field:ident ),+) => {
        impl<R, $( $ty, )+ Op> KernelReadReduce<R, Op>
            for $name<$( ColumnRead<R, $ty> ),+>
        where
            R: Runtime,
            $( $ty: MStorageElement + 'static, )+
            Op: op::ReductionOp<R, ($( $ty, )+)>,
        {
            fn reduce_value(
                self,
                policy: &CubePolicy<R>,
                init: Self::Item,
                _op: Op,
            ) -> Result<Self::Item, Error> {
                crate::detail::reduce(policy, ($( self.$field.column, )+), init, KernelOp::<R, Op>::new())
            }
        }
    };
}

impl_flat_zip_reduce_small!(ZipRead2; A: a, B: b);
impl_flat_zip_reduce_small!(ZipRead3; A: a, B: b, C: c);

macro_rules! impl_flat_zip_reduce_wide {
    ($name:ident, $method:ident; $( $ty:ident : $field:ident ),+) => {
        impl<R, $( $ty, )+ Op> KernelReadReduce<R, Op>
            for $name<$( ColumnRead<R, $ty> ),+>
        where
            R: Runtime,
            $( $ty: MStorageElement + 'static, )+
            Op: op::ReductionOp<R, ($( $ty, )+)>,
        {
            fn reduce_value(
                self,
                policy: &CubePolicy<R>,
                init: Self::Item,
                _op: Op,
            ) -> Result<Self::Item, Error> {
                crate::detail::apply::LinearReduceApply::$method::<
                    R,
                    $( $ty, )+
                    KernelOp<R, Op>,
                >(policy, $( &self.$field.column, )+ init)
            }
        }
    };
}

impl_flat_zip_reduce_wide!(ZipRead4, apply_views4; A: a, B: b, C: c, D: d);
impl_flat_zip_reduce_wide!(ZipRead5, apply_views5; A: a, B: b, C: c, D: d, E: e);
impl_flat_zip_reduce_wide!(ZipRead6, apply_views6; A: a, B: b, C: c, D: d, E: e, F: f);
impl_flat_zip_reduce_wide!(ZipRead7, apply_views7; A: a, B: b, C: c, D: d, E: e, F: f, G: g);

macro_rules! impl_flat_zip_set {
    ($name:ident, $view:ident, { $( $ty:ident : $field:ident => $view_field:ident ),+ }) => {
        impl<R, $( $ty, )+ Output, Less> KernelReadSet<
            R,
            $name<$( ColumnRead<R, $ty> ),+>,
            Output,
            Less,
        > for $name<$( ColumnRead<R, $ty> ),+>
        where
            R: Runtime,
            $( $ty: MStorageElement + 'static, )+
            Output: MIterMut<R, Item = ($( $ty, )+)>,
            Less: op::BinaryPredicateOp<R, ($( $ty, )+)>,
        {
            fn set_union_into(
                self,
                policy: &CubePolicy<R>,
                right: $name<$( ColumnRead<R, $ty> ),+>,
                _less: Less,
                output: Output,
            ) -> Result<crate::MIndex, Error> {
                let inner = crate::detail::set_union(
                    policy,
                    $view { $( $view_field: self.$field.column, )+ },
                    $view { $( $view_field: right.$field.column, )+ },
                    KernelOp::<R, Less>::new(),
                )?;
                let len = mindex_from_usize(inner.0.len())?;
                output.write_prefix_from_inner(policy, inner)?;
                Ok(len)
            }

            fn set_intersection_into(
                self,
                policy: &CubePolicy<R>,
                right: $name<$( ColumnRead<R, $ty> ),+>,
                _less: Less,
                output: Output,
            ) -> Result<crate::MIndex, Error> {
                let inner = crate::detail::set_intersection(
                    policy,
                    $view { $( $view_field: self.$field.column, )+ },
                    $view { $( $view_field: right.$field.column, )+ },
                    KernelOp::<R, Less>::new(),
                )?;
                let len = mindex_from_usize(inner.0.len())?;
                output.write_prefix_from_inner(policy, inner)?;
                Ok(len)
            }

            fn set_difference_into(
                self,
                policy: &CubePolicy<R>,
                right: $name<$( ColumnRead<R, $ty> ),+>,
                _less: Less,
                output: Output,
            ) -> Result<crate::MIndex, Error> {
                let inner = crate::detail::set_difference(
                    policy,
                    $view { $( $view_field: self.$field.column, )+ },
                    $view { $( $view_field: right.$field.column, )+ },
                    KernelOp::<R, Less>::new(),
                )?;
                let len = mindex_from_usize(inner.0.len())?;
                output.write_prefix_from_inner(policy, inner)?;
                Ok(len)
            }
        }
    };
}

macro_rules! impl_flat_zip_set_wide {
    (
        $name:ident,
        $membership:ident,
        $selected_apply:ident;
        $first_ty:ident : $first_field:ident
        $(, $ty:ident : $field:ident )+
    ) => {
        impl<R, $first_ty, $( $ty, )+ Output, Less> KernelReadSet<
            R,
            $name<ColumnRead<R, $first_ty>, $( ColumnRead<R, $ty> ),+>,
            Output,
            Less,
        > for $name<ColumnRead<R, $first_ty>, $( ColumnRead<R, $ty> ),+>
        where
            R: Runtime,
            $first_ty: MStorageElement + 'static,
            $( $ty: MStorageElement + 'static, )+
            Output: MIterMut<R, Item = ($first_ty, $( $ty, )+)>,
            Less: op::BinaryPredicateOp<R, ($first_ty, $( $ty, )+)>,
        {
            fn set_union_into(
                self,
                policy: &CubePolicy<R>,
                _right: $name<ColumnRead<R, $first_ty>, $( ColumnRead<R, $ty> ),+>,
                _less: Less,
                output: Output,
            ) -> Result<crate::MIndex, Error> {
                let len = self.$first_field.column.len();
                let inner = (
                    crate::detail::apply::MaterializePayloadApply::collect_expr(
                        policy,
                        &self.$first_field.column,
                    )?,
                    $(
                        crate::detail::apply::MaterializePayloadApply::collect_expr(
                            policy,
                            &self.$field.column,
                        )?,
                    )+
                );
                output.write_prefix_from_inner(policy, inner)?;
                mindex_from_usize(len)
            }

            fn set_intersection_into(
                self,
                policy: &CubePolicy<R>,
                right: $name<ColumnRead<R, $first_ty>, $( ColumnRead<R, $ty> ),+>,
                _less: Less,
                output: Output,
            ) -> Result<crate::MIndex, Error> {
                let len = self.$first_field.column.len();
                let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
                let flags =
                    crate::detail::apply::SetMembershipControlApply::$membership::<
                        DeviceColumnView<R, $first_ty>,
                        $( DeviceColumnView<R, $ty>, )+
                        DeviceColumnView<R, $first_ty>,
                        $( DeviceColumnView<R, $ty>, )+
                        KernelOp<R, Less>,
                    >(
                        policy,
                        &self.$first_field.column,
                        $( &self.$field.column, )+
                        &right.$first_field.column,
                        $( &right.$field.column, )+
                        true,
                    )?;
                let selected_rank =
                    crate::detail::primitives::select::selected_rank_from_flags(
                        policy, len, len_u32, flags,
                    )?;
                let count =
                    crate::detail::primitives::select::selected_count(policy, &selected_rank)?;
                let apply =
                    crate::detail::apply::SelectedPayloadApply::new(&selected_rank, count);
                let inner = apply.$selected_apply(
                    policy,
                    &self.$first_field.column,
                    $( &self.$field.column, )+
                )?;
                output.write_prefix_from_inner(policy, inner)?;
                mindex_from_usize(count)
            }

            fn set_difference_into(
                self,
                policy: &CubePolicy<R>,
                right: $name<ColumnRead<R, $first_ty>, $( ColumnRead<R, $ty> ),+>,
                _less: Less,
                output: Output,
            ) -> Result<crate::MIndex, Error> {
                let len = self.$first_field.column.len();
                let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
                let flags =
                    crate::detail::apply::SetMembershipControlApply::$membership::<
                        DeviceColumnView<R, $first_ty>,
                        $( DeviceColumnView<R, $ty>, )+
                        DeviceColumnView<R, $first_ty>,
                        $( DeviceColumnView<R, $ty>, )+
                        KernelOp<R, Less>,
                    >(
                        policy,
                        &self.$first_field.column,
                        $( &self.$field.column, )+
                        &right.$first_field.column,
                        $( &right.$field.column, )+
                        false,
                    )?;
                let selected_rank =
                    crate::detail::primitives::select::selected_rank_from_flags(
                        policy, len, len_u32, flags,
                    )?;
                let count =
                    crate::detail::primitives::select::selected_count(policy, &selected_rank)?;
                let apply =
                    crate::detail::apply::SelectedPayloadApply::new(&selected_rank, count);
                let inner = apply.$selected_apply(
                    policy,
                    &self.$first_field.column,
                    $( &self.$field.column, )+
                )?;
                output.write_prefix_from_inner(policy, inner)?;
                mindex_from_usize(count)
            }
        }
    };
}

impl<R, A, Output, Less> KernelReadSet<R, ZipRead1<ColumnRead<R, A>>, Output, Less>
    for ZipRead1<ColumnRead<R, A>>
where
    R: Runtime,
    A: MStorageElement + 'static,
    Output: MIterMut<R, Item = (A,)>,
    Less: op::BinaryPredicateOp<R, (A,)>,
{
    fn set_union_into(
        self,
        policy: &CubePolicy<R>,
        right: ZipRead1<ColumnRead<R, A>>,
        _less: Less,
        output: Output,
    ) -> Result<crate::MIndex, Error> {
        let inner = crate::detail::set_union(
            policy,
            ZipView1 {
                source: self.a.column,
            },
            ZipView1 {
                source: right.a.column,
            },
            crate::detail::api::Tuple1Less::<KernelOp<R, Less>>::default(),
        )?;
        let len = mindex_from_usize(inner.0.len())?;
        output.write_prefix_from_inner(policy, inner)?;
        Ok(len)
    }

    fn set_intersection_into(
        self,
        policy: &CubePolicy<R>,
        right: ZipRead1<ColumnRead<R, A>>,
        _less: Less,
        output: Output,
    ) -> Result<crate::MIndex, Error> {
        let inner = crate::detail::set_intersection(
            policy,
            ZipView1 {
                source: self.a.column,
            },
            ZipView1 {
                source: right.a.column,
            },
            crate::detail::api::Tuple1Less::<KernelOp<R, Less>>::default(),
        )?;
        let len = mindex_from_usize(inner.0.len())?;
        output.write_prefix_from_inner(policy, inner)?;
        Ok(len)
    }

    fn set_difference_into(
        self,
        policy: &CubePolicy<R>,
        right: ZipRead1<ColumnRead<R, A>>,
        _less: Less,
        output: Output,
    ) -> Result<crate::MIndex, Error> {
        let inner = crate::detail::set_difference(
            policy,
            ZipView1 {
                source: self.a.column,
            },
            ZipView1 {
                source: right.a.column,
            },
            crate::detail::api::Tuple1Less::<KernelOp<R, Less>>::default(),
        )?;
        let len = mindex_from_usize(inner.0.len())?;
        output.write_prefix_from_inner(policy, inner)?;
        Ok(len)
    }
}

impl_flat_zip_set!(ZipRead2, ZipView2, { A: a => left, B: b => right });
impl_flat_zip_set!(ZipRead3, ZipView3, { A: a => first, B: b => second, C: c => third });
impl_flat_zip_set_wide!(ZipRead4, tuple4_membership_expr_flags_with_policy, apply_expr4; A: a, B: b, C: c, D: d);
impl_flat_zip_set_wide!(ZipRead5, tuple5_membership_expr_flags_with_policy, apply_expr5; A: a, B: b, C: c, D: d, E: e);
impl_flat_zip_set_wide!(ZipRead6, tuple6_membership_expr_flags_with_policy, apply_expr6; A: a, B: b, C: c, D: d, E: e, F: f);
impl_flat_zip_set_wide!(ZipRead7, tuple7_membership_expr_flags_with_policy, apply_expr7; A: a, B: b, C: c, D: d, E: e, F: f, G: g);

impl<R, A, Output, Op> KernelReadTransform<R, Output, Op> for ZipRead1<ColumnRead<R, A>>
where
    R: Runtime,
    A: MStorageElement + 'static,
    Output: MIterMut<R>,
    Output::Item: MAlloc<R> + MItemDispatch<R>,
    Op: op::UnaryOp<R, (A,), Output = Output::Item>,
{
    fn transform_into(self, policy: &CubePolicy<R>, op: Op, output: Output) -> Result<(), Error> {
        let inner = <Output::Item as MItemDispatch<R>>::transform_unary(policy, self.a.column, op)?;
        output.write_from_inner(policy, inner)
    }

    fn transform_where_into(
        self,
        policy: &CubePolicy<R>,
        op: Op,
        stencil: PrecomputedSelection<R>,
        output: Output,
    ) -> Result<(), Error> {
        let inner = <Output::Item as MItemDispatch<R>>::transform_unary(policy, self.a.column, op)?;
        output.write_where_from_inner(policy, inner, stencil)
    }
}

macro_rules! impl_flat_zip_transform {
    ($name:ident, $method:ident; $( $ty:ident : $field:ident ),+) => {
        impl<R, $( $ty, )+ Output, Op> KernelReadTransform<R, Output, Op>
            for $name<$( ColumnRead<R, $ty> ),+>
        where
            R: Runtime,
            $( $ty: MStorageElement + 'static, )+
            Output: MIterMut<R>,
            Output::Item: MAlloc<R> + MItemDispatch<R>,
            Op: op::UnaryOp<R, ($( $ty, )+), Output = Output::Item>,
        {
            fn transform_into(
                self,
                policy: &CubePolicy<R>,
                op: Op,
                output: Output,
            ) -> Result<(), Error> {
                let inner = <Output::Item as MItemDispatch<R>>::$method(
                    policy,
                    $( self.$field.column, )+
                    op,
                )?;
                output.write_from_inner(policy, inner)
            }

            fn transform_where_into(
                self,
                policy: &CubePolicy<R>,
                op: Op,
                stencil: PrecomputedSelection<R>,
                output: Output,
            ) -> Result<(), Error> {
                let inner = <Output::Item as MItemDispatch<R>>::$method(
                    policy,
                    $( self.$field.column, )+
                    op,
                )?;
                output.write_where_from_inner(policy, inner, stencil)
            }
        }
    };
}

impl_flat_zip_transform!(ZipRead2, transform_binary; A: a, B: b);
impl_flat_zip_transform!(ZipRead3, transform_ternary; A: a, B: b, C: c);
impl_flat_zip_transform!(ZipRead4, transform_quaternary; A: a, B: b, C: c, D: d);
impl_flat_zip_transform!(ZipRead5, transform_quinary; A: a, B: b, C: c, D: d, E: e);
impl_flat_zip_transform!(ZipRead6, transform_senary; A: a, B: b, C: c, D: d, E: e, F: f);
impl_flat_zip_transform!(
    ZipRead7,
    transform_septenary;
    A: a,
    B: b,
    C: c,
    D: d,
    E: e,
    F: f,
    G: g
);

impl<R, A, Output> KernelReadSelection<R, Output> for ZipRead1<ColumnRead<R, A>>
where
    R: Runtime,
    A: MStorageElement + 'static,
    Output: MIterMut<R, Item = (A,)>,
{
    fn copy_selected_into(
        self,
        policy: &CubePolicy<R>,
        stencil: PrecomputedSelection<R>,
        output: Output,
    ) -> Result<crate::MIndex, Error> {
        let inner = crate::detail::copy_where(
            policy,
            (self.a.column,),
            stencil,
            KernelOp::<R, crate::detail::op_adapter::StencilFlag>::new(),
        )?;
        let len = mindex_from_usize(inner.0.len())?;
        output.write_prefix_from_inner(policy, inner)?;
        Ok(len)
    }
}

macro_rules! impl_flat_zip_copy_where {
    ($name:ident, $view:ident, { $( $ty:ident : $field:ident => $view_field:ident ),+ }) => {
        impl<R, $( $ty, )+ Output> KernelReadSelection<R, Output>
            for $name<$( ColumnRead<R, $ty> ),+>
        where
            R: Runtime,
            $( $ty: MStorageElement + 'static, )+
            Output: MIterMut<R, Item = ($( $ty, )+)>,
        {
            fn copy_selected_into(
                self,
                policy: &CubePolicy<R>,
                stencil: PrecomputedSelection<R>,
                output: Output,
            ) -> Result<crate::MIndex, Error> {
                let inner = crate::detail::copy_where(
                    policy,
                    $view { $( $view_field: self.$field.column, )+ },
                    stencil,
                    KernelOp::<R, crate::detail::op_adapter::StencilFlag>::new(),
                )?;
                let len = mindex_from_usize(inner.0.len())?;
                output.write_prefix_from_inner(policy, inner)?;
                Ok(len)
            }
        }
    };
}

impl_flat_zip_copy_where!(ZipRead2, ZipView2, { A: a => left, B: b => right });
impl_flat_zip_copy_where!(ZipRead3, ZipView3, { A: a => first, B: b => second, C: c => third });

macro_rules! impl_wide_zip_copy_where {
    ($name:ident, $apply:ident; $( $ty:ident : $field:ident ),+) => {
        impl<R, $( $ty, )+ Output> KernelReadSelection<R, Output>
            for $name<$( ColumnRead<R, $ty> ),+>
        where
            R: Runtime,
            $( $ty: MStorageElement + 'static, )+
            Output: MIterMut<R, Item = ($( $ty, )+)>,
        {
            fn copy_selected_into(
                self,
                policy: &CubePolicy<R>,
                stencil: PrecomputedSelection<R>,
                output: Output,
            ) -> Result<crate::MIndex, Error> {
                let selected_rank = stencil.selected_rank();
                let first_len = KernelColumn::len(&self.a.column);
                ensure_same_len(first_len, selected_rank.len)?;
                let count =
                    crate::detail::primitives::select::selected_count(policy, selected_rank)?;
                let payload_apply =
                    crate::detail::apply::SelectedPayloadApply::new(selected_rank, count);
                let inner = payload_apply.$apply(policy, $( &self.$field.column, )+)?;
                let len = mindex_from_usize(count)?;
                output.write_prefix_from_inner(policy, inner)?;
                Ok(len)
            }
        }
    };
}

impl_wide_zip_copy_where!(ZipRead4, apply_expr4; A: a, B: b, C: c, D: d);
impl_wide_zip_copy_where!(ZipRead5, apply_expr5; A: a, B: b, C: c, D: d, E: e);
impl_wide_zip_copy_where!(ZipRead6, apply_expr6; A: a, B: b, C: c, D: d, E: e, F: f);
impl_wide_zip_copy_where!(ZipRead7, apply_expr7; A: a, B: b, C: c, D: d, E: e, F: f, G: g);

impl<R, A, Output, Pred> KernelReadUnique<R, Output, Pred> for ZipRead1<ColumnRead<R, A>>
where
    R: Runtime,
    A: MStorageElement + 'static,
    Output: MIterMut<R, Item = (A,)>,
    Pred: op::BinaryPredicateOp<R, (A,)>,
{
    fn unique_into(
        self,
        policy: &CubePolicy<R>,
        _pred: Pred,
        output: Output,
    ) -> Result<crate::MIndex, Error> {
        let inner = crate::detail::unique(policy, (self.a.column,), KernelOp::<R, Pred>::new())?;
        let len = mindex_from_usize(inner.0.len())?;
        output.write_prefix_from_inner(policy, inner)?;
        Ok(len)
    }
}

macro_rules! impl_flat_zip_unique_small {
    ($name:ident, $view:ident, $apply:ident, { $( $ty:ident : $field:ident => $view_field:ident ),+ }) => {
        impl<R, $( $ty, )+ Output, Pred> KernelReadUnique<R, Output, Pred>
            for $name<$( ColumnRead<R, $ty> ),+>
        where
            R: Runtime,
            $( $ty: MStorageElement + 'static, )+
            Output: MIterMut<R, Item = ($( $ty, )+)>,
            Pred: op::BinaryPredicateOp<R, ($( $ty, )+)>,
        {
            fn unique_into(
                self,
                policy: &CubePolicy<R>,
                _pred: Pred,
                output: Output,
            ) -> Result<crate::MIndex, Error> {
                let inner = crate::detail::unique(
                    policy,
                    $view { $( $view_field: self.$field.column, )+ },
                    KernelOp::<R, Pred>::new(),
                )?;
                let len = mindex_from_usize(inner.0.len())?;
                output.write_prefix_from_inner(policy, inner)?;
                Ok(len)
            }
        }
    };
}

impl_flat_zip_unique_small!(ZipRead2, ZipView2, apply_expr2, { A: a => left, B: b => right });
impl_flat_zip_unique_small!(ZipRead3, ZipView3, apply_expr3, { A: a => first, B: b => second, C: c => third });

macro_rules! unique_selected_rank7 {
    ($policy:ident, $len:ident, $flags:expr) => {{
        let len_u32 = mindex_from_usize($len)?;
        crate::detail::primitives::select::selected_rank_from_flags($policy, $len, len_u32, $flags)
    }};
}

macro_rules! impl_flat_zip_unique_wide {
    (
        $name:ident, $adapter:ty, $apply:ident;
        $( $ty:ident : $field:ident ),+
        ; dummy $( $dummy:ident ),*
    ) => {
        impl<R, $( $ty, )+ Output, Pred> KernelReadUnique<R, Output, Pred>
            for $name<$( ColumnRead<R, $ty> ),+>
        where
            R: Runtime,
            $( $ty: MStorageElement + 'static, )+
            Output: MIterMut<R, Item = ($( $ty, )+)>,
            Pred: op::BinaryPredicateOp<R, ($( $ty, )+)>,
        {
            fn unique_into(
                self,
                policy: &CubePolicy<R>,
                _pred: Pred,
                output: Output,
            ) -> Result<crate::MIndex, Error> {
                let len = KernelColumn::len(&self.a.column);
                $(
                    let $dummy = crate::detail::primitives::range::indices_mindex(policy, len)?;
                    let $dummy = DeviceColumnView::from_column(&$dummy);
                )*
                let flags = crate::detail::read::unique_tuple7_flags_read::<
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    $adapter,
                >(
                    policy,
                    $( &self.$field.column, )+
                    $( &$dummy, )*
                )?;
                let selected_rank = unique_selected_rank7!(policy, len, flags)?;
                let count =
                    crate::detail::primitives::select::selected_count(policy, &selected_rank)?;
                let payload_apply =
                    crate::detail::apply::SelectedPayloadApply::new(&selected_rank, count);
                let inner = payload_apply.$apply(policy, $( &self.$field.column, )+)?;
                let len = mindex_from_usize(count)?;
                output.write_prefix_from_inner(policy, inner)?;
                Ok(len)
            }
        }
    };
}

impl_flat_zip_unique_wide!(
    ZipRead4,
    crate::detail::api::Tuple4AsTuple7BinaryPredicateOp<KernelOp<R, Pred>>,
    apply_expr4;
    A: a, B: b, C: c, D: d;
    dummy dummy_e, dummy_f, dummy_g
);
impl_flat_zip_unique_wide!(
    ZipRead5,
    crate::detail::api::Tuple5AsTuple7BinaryPredicateOp<KernelOp<R, Pred>>,
    apply_expr5;
    A: a, B: b, C: c, D: d, E: e;
    dummy dummy_f, dummy_g
);
impl_flat_zip_unique_wide!(
    ZipRead6,
    crate::detail::api::Tuple6AsTuple7BinaryPredicateOp<KernelOp<R, Pred>>,
    apply_expr6;
    A: a, B: b, C: c, D: d, E: e, F: f;
    dummy dummy_g
);
impl_flat_zip_unique_wide!(
    ZipRead7,
    KernelOp<R, Pred>,
    apply_expr7;
    A: a, B: b, C: c, D: d, E: e, F: f, G: g;
    dummy
);

impl<R, A, Indices, Output> KernelReadIndexed<R, Indices, Output> for ZipRead1<ColumnRead<R, A>>
where
    R: Runtime,
    A: MStorageElement + 'static,
    Indices: KernelReadBoundMany<R, Item = crate::MIndex>,
    Output: MIterMut<R, Item = (A,)>,
{
    fn gather_into(
        self,
        policy: &CubePolicy<R>,
        indices: Indices,
        output: Output,
    ) -> Result<(), Error> {
        let output = <Output as crate::iter::MIterMut<R>>::column_mut_view_inner::<A>(&output)?
            .ok_or_else(|| Error::Launch {
                message: "gather output must match input shape".to_string(),
            })?;
        crate::detail::apply::IndexedExprApply::gather_expr_into(
            policy,
            &self.a.column,
            &indices,
            &output,
        )
    }

    fn gather_where_into(
        self,
        policy: &CubePolicy<R>,
        indices: Indices,
        stencil: PrecomputedSelection<R>,
        output: Output,
    ) -> Result<(), Error> {
        let mask = stencil.mask();
        let output = <Output as crate::iter::MIterMut<R>>::column_mut_view_inner::<A>(&output)?
            .ok_or_else(|| Error::Launch {
                message: "gather_where output must match input shape".to_string(),
            })?;
        crate::detail::apply::MaskedIndexedExprApply::gather_where_expr_into(
            policy,
            &self.a.column,
            &indices,
            &mask,
            &output,
        )
    }

    fn scatter_into(
        self,
        policy: &CubePolicy<R>,
        indices: Indices,
        output: Output,
    ) -> Result<(), Error> {
        let output = <Output as crate::iter::MIterMut<R>>::column_mut_view_inner::<A>(&output)?
            .ok_or_else(|| Error::Launch {
                message: "scatter output must match input shape".to_string(),
            })?;
        crate::detail::apply::IndexedExprApply::scatter_expr_into(
            policy,
            &self.a.column,
            &indices,
            &output,
        )
    }

    fn scatter_where_into(
        self,
        policy: &CubePolicy<R>,
        indices: Indices,
        stencil: PrecomputedSelection<R>,
        output: Output,
    ) -> Result<(), Error> {
        let mask = stencil.mask();
        let output = <Output as crate::iter::MIterMut<R>>::column_mut_view_inner::<A>(&output)?
            .ok_or_else(|| Error::Launch {
                message: "scatter_where output must match input shape".to_string(),
            })?;
        crate::detail::apply::MaskedIndexedExprApply::scatter_where_expr_into(
            policy,
            &self.a.column,
            &indices,
            &mask,
            &output,
        )
    }
}

macro_rules! impl_flat_zip_indexed {
    ($name:ident; $( $ty:ident : $field:ident => $idx:tt ),+) => {
        impl<R, $( $ty, )+ Indices, Output> KernelReadIndexed<R, Indices, Output>
            for $name<$( ColumnRead<R, $ty> ),+>
        where
            R: Runtime,
            $( $ty: MStorageElement + 'static, )+
            Indices: KernelReadBoundMany<R, Item = crate::MIndex>,
            Output: MIterMut<R, Item = ($( $ty, )+)>,
        {
            fn gather_into(
                self,
                policy: &CubePolicy<R>,
                indices: Indices,
                output: Output,
            ) -> Result<(), Error> {
                $(
                    let out = <Output as crate::iter::MIterMut<R>>::column_mut_view_by_index_inner::<$ty>(&output, $idx,)?
                    .ok_or_else(|| Error::Launch {
                        message: "gather output must match input shape".to_string(),
                    })?;
                    crate::detail::apply::IndexedExprApply::gather_expr_into(
                        policy,
                        &self.$field.column,
                        &indices,
                        &out,
                    )?;
                )+
                Ok(())
            }

            fn gather_where_into(
                self,
                policy: &CubePolicy<R>,
                indices: Indices,
                stencil: PrecomputedSelection<R>,
                output: Output,
            ) -> Result<(), Error> {
                let mask = stencil.mask();
                $(
                    let out = <Output as crate::iter::MIterMut<R>>::column_mut_view_by_index_inner::<$ty>(&output, $idx,)?
                    .ok_or_else(|| Error::Launch {
                        message: "gather_where output must match input shape".to_string(),
                    })?;
                    crate::detail::apply::MaskedIndexedExprApply::gather_where_expr_into(
                        policy,
                        &self.$field.column,
                        &indices,
                        &mask,
                        &out,
                    )?;
                )+
                Ok(())
            }

            fn scatter_into(
                self,
                policy: &CubePolicy<R>,
                indices: Indices,
                output: Output,
            ) -> Result<(), Error> {
                $(
                    let out = <Output as crate::iter::MIterMut<R>>::column_mut_view_by_index_inner::<$ty>(&output, $idx,)?
                    .ok_or_else(|| Error::Launch {
                        message: "scatter output must match input shape".to_string(),
                    })?;
                    crate::detail::apply::IndexedExprApply::scatter_expr_into(
                        policy,
                        &self.$field.column,
                        &indices,
                        &out,
                    )?;
                )+
                Ok(())
            }

            fn scatter_where_into(
                self,
                policy: &CubePolicy<R>,
                indices: Indices,
                stencil: PrecomputedSelection<R>,
                output: Output,
            ) -> Result<(), Error> {
                let mask = stencil.mask();
                $(
                    let out = <Output as crate::iter::MIterMut<R>>::column_mut_view_by_index_inner::<$ty>(&output, $idx,)?
                    .ok_or_else(|| Error::Launch {
                        message: "scatter_where output must match input shape".to_string(),
                    })?;
                    crate::detail::apply::MaskedIndexedExprApply::scatter_where_expr_into(
                        policy,
                        &self.$field.column,
                        &indices,
                        &mask,
                        &out,
                    )?;
                )+
                Ok(())
            }
        }
    };
}

impl_flat_zip_indexed!(ZipRead2; A: a => 0, B: b => 1);
impl_flat_zip_indexed!(ZipRead3; A: a => 0, B: b => 1, C: c => 2);
impl_flat_zip_indexed!(ZipRead4; A: a => 0, B: b => 1, C: c => 2, D: d => 3);
impl_flat_zip_indexed!(ZipRead5; A: a => 0, B: b => 1, C: c => 2, D: d => 3, E: e => 4);
impl_flat_zip_indexed!(ZipRead6; A: a => 0, B: b => 1, C: c => 2, D: d => 3, E: e => 4, F: f => 5);
impl_flat_zip_indexed!(ZipRead7; A: a => 0, B: b => 1, C: c => 2, D: d => 3, E: e => 4, F: f => 5, G: g => 6);

impl<R, A, Output, Op> KernelReadAdjacentDifference<R, Output, Op> for ZipRead1<ColumnRead<R, A>>
where
    R: Runtime,
    A: MStorageElement + 'static,
    Output: MIterMut<R, Item = (A,)>,
    Op: op::ReductionOp<R, (A,)>,
{
    fn adjacent_difference_into(
        self,
        policy: &CubePolicy<R>,
        _op: Op,
        output: Output,
    ) -> Result<(), Error> {
        let inner =
            crate::detail::adjacent_difference(policy, (self.a.column,), KernelOp::<R, Op>::new())?;
        output.write_from_inner(policy, inner)
    }
}

macro_rules! impl_flat_zip_adjacent_difference {
    ($name:ident, $view:ident, { $( $ty:ident : $field:ident => $view_field:ident ),+ }) => {
        impl<R, $( $ty, )+ Output, Op> KernelReadAdjacentDifference<R, Output, Op>
            for $name<$( ColumnRead<R, $ty> ),+>
        where
            R: Runtime,
            $( $ty: MStorageElement + 'static, )+
            Output: MIterMut<R, Item = ($( $ty, )+)>,
            Op: op::ReductionOp<R, ($( $ty, )+)>,
        {
            fn adjacent_difference_into(
                self,
                policy: &CubePolicy<R>,
                _op: Op,
                output: Output,
            ) -> Result<(), Error> {
                let inner = crate::detail::adjacent_difference(
                    policy,
                    $view { $( $view_field: self.$field.column, )+ },
                    KernelOp::<R, Op>::new(),
                )?;
                output.write_from_inner(policy, inner)
            }
        }
    };
}

impl_flat_zip_adjacent_difference!(ZipRead2, ZipView2, { A: a => left, B: b => right });
impl_flat_zip_adjacent_difference!(ZipRead3, ZipView3, { A: a => first, B: b => second, C: c => third });

macro_rules! impl_wide_zip_adjacent_difference {
    ($name:ident, $apply:ident; $( $ty:ident : $field:ident ),+) => {
        impl<R, $( $ty, )+ Output, Op> KernelReadAdjacentDifference<R, Output, Op>
            for $name<$( ColumnRead<R, $ty> ),+>
        where
            R: Runtime,
            $( $ty: MStorageElement + 'static, )+
            Output: MIterMut<R, Item = ($( $ty, )+)>,
            Op: op::ReductionOp<R, ($( $ty, )+)>,
        {
            fn adjacent_difference_into(
                self,
                policy: &CubePolicy<R>,
                _op: Op,
                output: Output,
            ) -> Result<(), Error> {
                let inner = crate::detail::apply::LinearScanApply::$apply::<
                    R,
                    $( $ty, )+
                    KernelOp<R, Op>,
                >(
                    policy,
                    $( &self.$field.column, )+
                )?;
                output.write_from_inner(policy, inner)
            }
        }
    };
}

impl_wide_zip_adjacent_difference!(ZipRead4, adjacent_views4; A: a, B: b, C: c, D: d);
impl_wide_zip_adjacent_difference!(ZipRead5, adjacent_views5; A: a, B: b, C: c, D: d, E: e);
impl_wide_zip_adjacent_difference!(ZipRead6, adjacent_views6; A: a, B: b, C: c, D: d, E: e, F: f);
impl_wide_zip_adjacent_difference!(ZipRead7, adjacent_views7; A: a, B: b, C: c, D: d, E: e, F: f, G: g);

impl<R, A, Output, Op> KernelReadScan<R, Output, Op> for ZipRead1<ColumnRead<R, A>>
where
    R: Runtime,
    A: MStorageElement + 'static,
    Output: MIterMut<R, Item = (A,)>,
    Op: op::ReductionOp<R, (A,)>,
{
    fn inclusive_scan_into(
        self,
        policy: &CubePolicy<R>,
        _op: Op,
        output: Output,
    ) -> Result<(), Error> {
        let inner =
            crate::detail::inclusive_scan(policy, (self.a.column,), KernelOp::<R, Op>::new())?;
        output.write_from_inner(policy, inner)
    }

    fn exclusive_scan_into(
        self,
        policy: &CubePolicy<R>,
        init: Self::Item,
        _op: Op,
        output: Output,
    ) -> Result<(), Error> {
        let inner = crate::detail::exclusive_scan(
            policy,
            (self.a.column,),
            init,
            KernelOp::<R, Op>::new(),
        )?;
        output.write_from_inner(policy, inner)
    }
}

macro_rules! impl_flat_zip_scan {
    ($name:ident, $view:ident, { $( $ty:ident : $field:ident => $view_field:ident ),+ }) => {
        impl<R, $( $ty, )+ Output, Op> KernelReadScan<R, Output, Op>
            for $name<$( ColumnRead<R, $ty> ),+>
        where
            R: Runtime,
            $( $ty: MStorageElement + 'static, )+
            Output: MIterMut<R, Item = ($( $ty, )+)>,
            Op: op::ReductionOp<R, ($( $ty, )+)>,
        {
            fn inclusive_scan_into(
                self,
                policy: &CubePolicy<R>,
                _op: Op,
                output: Output,
            ) -> Result<(), Error> {
                let inner = crate::detail::inclusive_scan(
                    policy,
                    $view { $( $view_field: self.$field.column, )+ },
                    KernelOp::<R, Op>::new(),
                )?;
                output.write_from_inner(policy, inner)
            }

            fn exclusive_scan_into(
                self,
                policy: &CubePolicy<R>,
                init: Self::Item,
                _op: Op,
                output: Output,
            ) -> Result<(), Error> {
                let inner = crate::detail::exclusive_scan(
                    policy,
                    $view { $( $view_field: self.$field.column, )+ },
                    init,
                    KernelOp::<R, Op>::new(),
                )?;
                output.write_from_inner(policy, inner)
            }
        }
    };
}

impl_flat_zip_scan!(ZipRead2, ZipView2, { A: a => left, B: b => right });
impl_flat_zip_scan!(ZipRead3, ZipView3, { A: a => first, B: b => second, C: c => third });

macro_rules! impl_wide_zip_scan {
    ($name:ident, $inclusive:ident, $exclusive:ident; $( $ty:ident : $field:ident ),+) => {
        impl<R, $( $ty, )+ Output, Op> KernelReadScan<R, Output, Op>
            for $name<$( ColumnRead<R, $ty> ),+>
        where
            R: Runtime,
            $( $ty: MStorageElement + 'static, )+
            Output: MIterMut<R, Item = ($( $ty, )+)>,
            Op: op::ReductionOp<R, ($( $ty, )+)>,
        {
            fn inclusive_scan_into(
                self,
                policy: &CubePolicy<R>,
                _op: Op,
                output: Output,
            ) -> Result<(), Error> {
                let inner = crate::detail::apply::LinearScanApply::$inclusive::<
                    R,
                    $( $ty, )+
                    KernelOp<R, Op>,
                >(
                    policy,
                    $( &self.$field.column, )+
                )?;
                output.write_from_inner(policy, inner)
            }

            fn exclusive_scan_into(
                self,
                policy: &CubePolicy<R>,
                init: Self::Item,
                _op: Op,
                output: Output,
            ) -> Result<(), Error> {
                let inner = crate::detail::apply::LinearScanApply::$exclusive::<
                    R,
                    $( $ty, )+
                    KernelOp<R, Op>,
                >(
                    policy,
                    $( &self.$field.column, )+
                    init,
                )?;
                output.write_from_inner(policy, inner)
            }
        }
    };
}

impl_wide_zip_scan!(ZipRead4, inclusive_views4, exclusive_views4; A: a, B: b, C: c, D: d);
impl_wide_zip_scan!(ZipRead5, inclusive_views5, exclusive_views5; A: a, B: b, C: c, D: d, E: e);
impl_wide_zip_scan!(ZipRead6, inclusive_views6, exclusive_views6; A: a, B: b, C: c, D: d, E: e, F: f);
impl_wide_zip_scan!(ZipRead7, inclusive_views7, exclusive_views7; A: a, B: b, C: c, D: d, E: e, F: f, G: g);

impl<R, A> KernelReadScanByKeyView<R> for ZipRead1<ColumnRead<R, A>>
where
    R: Runtime,
    A: MStorageElement + 'static,
{
    type View = (DeviceColumnView<R, A>,);

    fn into_scan_by_key_view(self) -> Self::View {
        (self.a.column,)
    }
}

impl<R, A> KernelReadScanByKeyValuesView<R> for ZipRead1<ColumnRead<R, A>>
where
    R: Runtime,
    A: MStorageElement + 'static,
{
    type ExclusiveInit = A;

    fn into_exclusive_scan_by_key_init(init: Self::Item) -> Self::ExclusiveInit {
        init.0
    }
}

macro_rules! impl_flat_zip_scan_by_key_view {
    ($name:ident; $( $ty:ident : $field:ident ),+) => {
        impl<R, $( $ty, )+> KernelReadScanByKeyView<R>
            for $name<$( ColumnRead<R, $ty> ),+>
        where
            R: Runtime,
            $( $ty: MStorageElement + 'static, )+
        {
            type View = ($( DeviceColumnView<R, $ty>, )+);

            fn into_scan_by_key_view(self) -> Self::View {
                ($( self.$field.column, )+)
            }
        }

        impl<R, $( $ty, )+> KernelReadScanByKeyValuesView<R>
            for $name<$( ColumnRead<R, $ty> ),+>
        where
            R: Runtime,
            $( $ty: MStorageElement + 'static, )+
        {
            type ExclusiveInit = ($( $ty, )+);

            fn into_exclusive_scan_by_key_init(init: Self::Item) -> Self::ExclusiveInit {
                init
            }
        }
    };
}

impl_flat_zip_scan_by_key_view!(ZipRead2; A: a, B: b);
impl_flat_zip_scan_by_key_view!(ZipRead3; A: a, B: b, C: c);
impl_flat_zip_scan_by_key_view!(ZipRead4; A: a, B: b, C: c, D: d);
impl_flat_zip_scan_by_key_view!(ZipRead5; A: a, B: b, C: c, D: d, E: e);
impl_flat_zip_scan_by_key_view!(ZipRead6; A: a, B: b, C: c, D: d, E: e, F: f);
impl_flat_zip_scan_by_key_view!(ZipRead7; A: a, B: b, C: c, D: d, E: e, F: f, G: g);

impl<R, Keys, Values, Output, KeyEq, Op> KernelReadScanByKey<R, Values, Output, KeyEq, Op> for Keys
where
    R: Runtime,
    Keys: KernelReadScanByKeyView<R>,
    Values: KernelReadScanByKeyValuesView<R>,
    Output: MIterMut<R, Item = Values::Item>,
    Keys::Item: MItem<R>,
    Values::Item: MAlloc<R> + MItem<R>,
    KeyEq: op::BinaryPredicateOp<R, Keys::Item>,
    Op: op::ReductionOp<R, Values::Item>,
    Keys::View: crate::detail::read::KernelInclusiveScanByKeyCall<
            Values::View,
            KernelOp<R, KeyEq>,
            KernelOp<R, Op>,
            Runtime = R,
        > + crate::detail::read::KernelExclusiveScanByKeyCall<
            Values::View,
            KernelOp<R, KeyEq>,
            KernelOp<R, Op>,
            Runtime = R,
            Init = Values::ExclusiveInit,
        >,
    <Keys::View as crate::detail::read::KernelInclusiveScanByKeyCall<
        Values::View,
        KernelOp<R, KeyEq>,
        KernelOp<R, Op>,
    >>::Output:
        crate::detail::MaterializeOutput<Runtime = R, Output = <Values::Item as MAlloc<R>>::Inner>,
    <Keys::View as crate::detail::read::KernelExclusiveScanByKeyCall<
        Values::View,
        KernelOp<R, KeyEq>,
        KernelOp<R, Op>,
    >>::Output:
        crate::detail::MaterializeOutput<Runtime = R, Output = <Values::Item as MAlloc<R>>::Inner>,
{
    fn inclusive_scan_by_key_into(
        self,
        policy: &CubePolicy<R>,
        values: Values,
        _key_eq: KeyEq,
        _op: Op,
        output: Output,
    ) -> Result<(), Error> {
        let inner = crate::detail::inclusive_scan_by_key(
            policy,
            self.into_scan_by_key_view(),
            values.into_scan_by_key_view(),
            KernelOp::<R, KeyEq>::new(),
            KernelOp::<R, Op>::new(),
        )?;
        output.write_from_inner(policy, inner)
    }

    fn exclusive_scan_by_key_into(
        self,
        policy: &CubePolicy<R>,
        values: Values,
        _key_eq: KeyEq,
        init: Values::Item,
        _op: Op,
        output: Output,
    ) -> Result<(), Error> {
        let inner = crate::detail::exclusive_scan_by_key(
            policy,
            self.into_scan_by_key_view(),
            values.into_scan_by_key_view(),
            KernelOp::<R, KeyEq>::new(),
            Values::into_exclusive_scan_by_key_init(init),
            KernelOp::<R, Op>::new(),
        )?;
        output.write_from_inner(policy, inner)
    }
}

impl<R, A, Pred> KernelReadAdjacentFind<R, Pred> for ZipRead1<ColumnRead<R, A>>
where
    R: Runtime,
    A: MStorageElement + 'static,
    Pred: op::BinaryPredicateOp<R, (A,)>,
{
    fn adjacent_find(
        self,
        policy: &CubePolicy<R>,
        _pred: Pred,
    ) -> Result<Option<crate::MIndex>, Error> {
        crate::detail::adjacent_find(policy, (self.a.column,), KernelOp::<R, Pred>::new())
    }
}

macro_rules! impl_flat_zip_adjacent_find {
    ($name:ident, $view:ident, { $( $ty:ident : $field:ident => $view_field:ident ),+ }) => {
        impl<R, $( $ty, )+ Pred> KernelReadAdjacentFind<R, Pred>
            for $name<$( ColumnRead<R, $ty> ),+>
        where
            R: Runtime,
            $( $ty: MStorageElement + 'static, )+
            Pred: op::BinaryPredicateOp<R, ($( $ty, )+)>,
        {
            fn adjacent_find(
                self,
                policy: &CubePolicy<R>,
                _pred: Pred,
            ) -> Result<Option<crate::MIndex>, Error> {
                crate::detail::adjacent_find(
                    policy,
                    $view { $( $view_field: self.$field.column, )+ },
                    KernelOp::<R, Pred>::new(),
                )
            }
        }
    };
}

impl_flat_zip_adjacent_find!(ZipRead2, ZipView2, { A: a => left, B: b => right });
impl_flat_zip_adjacent_find!(ZipRead3, ZipView3, { A: a => first, B: b => second, C: c => third });
impl_flat_zip_adjacent_find!(ZipRead4, ZipView4, { A: a => a, B: b => b, C: c => c, D: d => d });
impl_flat_zip_adjacent_find!(ZipRead5, ZipView5, { A: a => a, B: b => b, C: c => c, D: d => d, E: e => e });
impl_flat_zip_adjacent_find!(ZipRead6, ZipView6, { A: a => a, B: b => b, C: c => c, D: d => d, E: e => e, F: f => f });
impl_flat_zip_adjacent_find!(ZipRead7, ZipView7, { A: a => a, B: b => b, C: c => c, D: d => d, E: e => e, F: f => f, G: g => g });

impl<R, A> KernelReadSearchView<R> for ZipRead1<ColumnRead<R, A>>
where
    R: Runtime,
    A: MStorageElement + 'static,
{
    type View = (DeviceColumnView<R, A>,);

    fn into_search_view(self) -> Self::View {
        (self.a.column,)
    }
}

macro_rules! impl_flat_zip_search_view {
    ($name:ident, $view:ident, { $( $ty:ident : $field:ident => $view_field:ident ),+ }) => {
        impl<R, $( $ty, )+> KernelReadSearchView<R>
            for $name<$( ColumnRead<R, $ty> ),+>
        where
            R: Runtime,
            $( $ty: MStorageElement + 'static, )+
        {
            type View = $view<$( DeviceColumnView<R, $ty> ),+>;

            fn into_search_view(self) -> Self::View {
                $view { $( $view_field: self.$field.column, )+ }
            }
        }
    };
}

impl_flat_zip_search_view!(ZipRead2, ZipView2, { A: a => left, B: b => right });
impl_flat_zip_search_view!(ZipRead3, ZipView3, { A: a => first, B: b => second, C: c => third });
impl_flat_zip_search_view!(ZipRead4, ZipView4, { A: a => a, B: b => b, C: c => c, D: d => d });
impl_flat_zip_search_view!(ZipRead5, ZipView5, { A: a => a, B: b => b, C: c => c, D: d => d, E: e => e });
impl_flat_zip_search_view!(ZipRead6, ZipView6, { A: a => a, B: b => b, C: c => c, D: d => d, E: e => e, F: f => f });
impl_flat_zip_search_view!(ZipRead7, ZipView7, { A: a => a, B: b => b, C: c => c, D: d => d, E: e => e, F: f => f, G: g => g });

impl<R, Input, Less> KernelReadMinMax<R, Less> for Input
where
    R: Runtime,
    Input: KernelReadSearchView<R>,
    Input::Item: MItem<R>,
    Less: op::BinaryPredicateOp<R, Input::Item>,
    Input::View: crate::detail::read::KernelMinMaxInput<KernelOp<R, Less>, Runtime = R>,
{
    fn min_element(
        self,
        policy: &CubePolicy<R>,
        less: Less,
    ) -> Result<Option<crate::MIndex>, Error> {
        let _ = less;
        crate::detail::min_element(policy, self.into_search_view(), KernelOp::<R, Less>::new())
    }

    fn max_element(
        self,
        policy: &CubePolicy<R>,
        less: Less,
    ) -> Result<Option<crate::MIndex>, Error> {
        let _ = less;
        crate::detail::max_element(policy, self.into_search_view(), KernelOp::<R, Less>::new())
    }

    fn minmax_element(
        self,
        policy: &CubePolicy<R>,
        less: Less,
    ) -> Result<Option<(crate::MIndex, crate::MIndex)>, Error> {
        let _ = less;
        crate::detail::minmax_element(policy, self.into_search_view(), KernelOp::<R, Less>::new())
    }
}

impl<R, Input, Values, Less> KernelReadSortedSearch<R, Values, Less> for Input
where
    R: Runtime,
    Input: KernelReadSearchView<R>,
    Values: KernelReadSearchView<R, Item = Input::Item>,
    Input::Item: MItem<R>,
    Less: op::BinaryPredicateOp<R, Input::Item>,
    Input::View: crate::detail::read::KernelSortedSearchManyInput<
            Values::View,
            KernelOp<R, Less>,
            Runtime = R,
        > + crate::detail::read::KernelSortedSearchInput<KernelOp<R, Less>, Runtime = R>,
{
    fn lower_bound_many(
        self,
        policy: &CubePolicy<R>,
        values: Values,
        _less: Less,
    ) -> Result<crate::runtime::DeviceVec<R, crate::MIndex>, Error> {
        let bounds = crate::detail::lower_bound_many(
            policy,
            self.into_search_view(),
            values.into_search_view(),
            KernelOp::<R, Less>::new(),
        )?;
        Ok(crate::runtime::DeviceVec::from_inner(bounds))
    }

    fn upper_bound_many(
        self,
        policy: &CubePolicy<R>,
        values: Values,
        _less: Less,
    ) -> Result<crate::runtime::DeviceVec<R, crate::MIndex>, Error> {
        let bounds = crate::detail::upper_bound_many(
            policy,
            self.into_search_view(),
            values.into_search_view(),
            KernelOp::<R, Less>::new(),
        )?;
        Ok(crate::runtime::DeviceVec::from_inner(bounds))
    }

    fn is_sorted_until(self, policy: &CubePolicy<R>, less: Less) -> Result<crate::MIndex, Error> {
        let _ = less;
        crate::detail::is_sorted_until(policy, self.into_search_view(), KernelOp::<R, Less>::new())
    }

    fn is_sorted(self, policy: &CubePolicy<R>, less: Less) -> Result<bool, Error> {
        let _ = less;
        crate::detail::is_sorted(policy, self.into_search_view(), KernelOp::<R, Less>::new())
    }
}

impl<R, Left, Right, Op> KernelReadPairSearch<R, Right, Op> for Left
where
    R: Runtime,
    Left: KernelReadSearchView<R>,
    Right: KernelReadSearchView<R, Item = Left::Item>,
    Left::Item: MItem<R>,
    Op: op::BinaryPredicateOp<R, Left::Item>,
    Left::View:
        crate::detail::read::KernelPairSearchInput<Right::View, KernelOp<R, Op>, Runtime = R>,
{
    fn equal(self, policy: &CubePolicy<R>, right: Right, op: Op) -> Result<bool, Error> {
        let _ = op;
        crate::detail::equal(
            policy,
            self.into_search_view(),
            right.into_search_view(),
            KernelOp::<R, Op>::new(),
        )
    }

    fn mismatch(
        self,
        policy: &CubePolicy<R>,
        right: Right,
        op: Op,
    ) -> Result<Option<crate::MIndex>, Error> {
        let _ = op;
        crate::detail::mismatch(
            policy,
            self.into_search_view(),
            right.into_search_view(),
            KernelOp::<R, Op>::new(),
        )
    }

    fn find_first_of(
        self,
        policy: &CubePolicy<R>,
        needles: Right,
        op: Op,
    ) -> Result<Option<crate::MIndex>, Error> {
        let _ = op;
        crate::detail::find_first_of(
            policy,
            self.into_search_view(),
            needles.into_search_view(),
            KernelOp::<R, Op>::new(),
        )
    }

    fn lexicographical_compare(
        self,
        policy: &CubePolicy<R>,
        right: Right,
        op: Op,
    ) -> Result<bool, Error> {
        let _ = op;
        crate::detail::lexicographical_compare(
            policy,
            self.into_search_view(),
            right.into_search_view(),
            KernelOp::<R, Op>::new(),
        )
    }
}

macro_rules! wide_predicate_rank {
    ($policy:ident; $pred:ty; ($ty0:ty, $ty1:ty, $ty2:ty, $ty3:ty); $a:expr, $b:expr, $c:expr, $d:expr) => {{
        let dummy_e = crate::detail::primitives::range::indices_mindex($policy, $a.len)?;
        let dummy_f = crate::detail::primitives::range::indices_mindex($policy, $a.len)?;
        let dummy_g = crate::detail::primitives::range::indices_mindex($policy, $a.len)?;
        let dummy_e = DeviceColumnView::from_column(&dummy_e);
        let dummy_f = DeviceColumnView::from_column(&dummy_f);
        let dummy_g = DeviceColumnView::from_column(&dummy_g);
        wide_predicate_rank!(
            @launch $policy,
            crate::detail::api::Tuple4AsTuple7PredicateOp<KernelOp<R, $pred>>,
            ($a, $b, $c, $d, &dummy_e, &dummy_f, &dummy_g),
            ($ty0, $ty1, $ty2, $ty3, u32, u32, u32)
        )
    }};
    ($policy:ident; $pred:ty; ($ty0:ty, $ty1:ty, $ty2:ty, $ty3:ty, $ty4:ty); $a:expr, $b:expr, $c:expr, $d:expr, $e:expr) => {{
        let dummy_f = crate::detail::primitives::range::indices_mindex($policy, $a.len)?;
        let dummy_g = crate::detail::primitives::range::indices_mindex($policy, $a.len)?;
        let dummy_f = DeviceColumnView::from_column(&dummy_f);
        let dummy_g = DeviceColumnView::from_column(&dummy_g);
        wide_predicate_rank!(
            @launch $policy,
            crate::detail::api::Tuple5AsTuple7PredicateOp<KernelOp<R, $pred>>,
            ($a, $b, $c, $d, $e, &dummy_f, &dummy_g),
            ($ty0, $ty1, $ty2, $ty3, $ty4, u32, u32)
        )
    }};
    ($policy:ident; $pred:ty; ($ty0:ty, $ty1:ty, $ty2:ty, $ty3:ty, $ty4:ty, $ty5:ty); $a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr) => {{
        let dummy_g = crate::detail::primitives::range::indices_mindex($policy, $a.len)?;
        let dummy_g = DeviceColumnView::from_column(&dummy_g);
        wide_predicate_rank!(
            @launch $policy,
            crate::detail::api::Tuple6AsTuple7PredicateOp<KernelOp<R, $pred>>,
            ($a, $b, $c, $d, $e, $f, &dummy_g),
            ($ty0, $ty1, $ty2, $ty3, $ty4, $ty5, u32)
        )
    }};
    ($policy:ident; $pred:ty; ($ty0:ty, $ty1:ty, $ty2:ty, $ty3:ty, $ty4:ty, $ty5:ty, $ty6:ty); $a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr) => {{
        wide_predicate_rank!(
            @launch $policy,
            KernelOp<R, $pred>,
            ($a, $b, $c, $d, $e, $f, $g),
            ($ty0, $ty1, $ty2, $ty3, $ty4, $ty5, $ty6)
        )
    }};
    (@launch $policy:ident, $pred:ty, ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr), ($ty0:ty, $ty1:ty, $ty2:ty, $ty3:ty, $ty4:ty, $ty5:ty, $ty6:ty)) => {{
        let len = $a.len;
        let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
        if len == 0 {
            Ok(crate::detail::primitives::select::SelectedRankControl::empty(
                $policy.client(),
            ))
        } else {
            let client = $policy.client();
            let flag = client.empty(len * std::mem::size_of::<u32>());
            let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
            let invert_handle = client.create_from_slice(u32::as_bytes(&[0_u32]));
            let offsets = [
                u32::try_from($a.offset).map_err(|_| Error::LengthTooLarge { len: $a.offset })?,
                u32::try_from($b.offset).map_err(|_| Error::LengthTooLarge { len: $b.offset })?,
                u32::try_from($c.offset).map_err(|_| Error::LengthTooLarge { len: $c.offset })?,
                u32::try_from($d.offset).map_err(|_| Error::LengthTooLarge { len: $d.offset })?,
                u32::try_from($e.offset).map_err(|_| Error::LengthTooLarge { len: $e.offset })?,
                u32::try_from($f.offset).map_err(|_| Error::LengthTooLarge { len: $f.offset })?,
                u32::try_from($g.offset).map_err(|_| Error::LengthTooLarge { len: $g.offset })?,
            ];
            let offsets_handle = client.create_from_slice(u32::as_bytes(&offsets));
            let block_size = 256_u32;
            let block_count = len.div_ceil(block_size as usize);
            let block_count_u32 =
                u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;
            unsafe {
                crate::kernels::tuple7_view_predicate_flags_kernel::launch_unchecked::<
                    $ty0,
                    $ty1,
                    $ty2,
                    $ty3,
                    $ty4,
                    $ty5,
                    $ty6,
                    $pred,
                    R,
                >(
                    client,
                    CubeCount::Static(block_count_u32, 1, 1),
                    CubeDim::new_1d(block_size),
                    BufferArg::from_raw_parts($a.source.handle.clone(), $a.source.len()),
                    BufferArg::from_raw_parts($b.source.handle.clone(), $b.source.len()),
                    BufferArg::from_raw_parts($c.source.handle.clone(), $c.source.len()),
                    BufferArg::from_raw_parts($d.source.handle.clone(), $d.source.len()),
                    BufferArg::from_raw_parts($e.source.handle.clone(), $e.source.len()),
                    BufferArg::from_raw_parts($f.source.handle.clone(), $f.source.len()),
                    BufferArg::from_raw_parts($g.source.handle.clone(), $g.source.len()),
                    BufferArg::from_raw_parts(offsets_handle.clone(), 7),
                    BufferArg::from_raw_parts(len_handle.clone(), 1),
                    BufferArg::from_raw_parts(invert_handle.clone(), 1),
                    BufferArg::from_raw_parts(flag.clone(), len),
                );
            }
            crate::detail::primitives::select::selected_rank_from_flags($policy, len, len_u32, flag)
        }
    }};
}

macro_rules! impl_wide_zip_predicate_query {
    ($name:ident; $( $ty:ident : $field:ident ),+) => {
        impl<R, $( $ty, )+ Pred> KernelReadPredicateQuery<R, Pred>
            for $name<$( ColumnRead<R, $ty> ),+>
        where
            R: Runtime,
            $( $ty: MStorageElement + 'static, )+
            Pred: op::PredicateOp<R, ($( $ty, )+)>,
        {
            fn count_if(
                self,
                policy: &CubePolicy<R>,
                pred: Pred,
            ) -> Result<crate::MIndex, Error> {
                let _ = pred;
                let selected_rank = wide_predicate_rank!(
                    policy; Pred; ($( $ty ),+); $( &self.$field.column ),+
                )?;
                mindex_from_usize(crate::detail::primitives::select::selected_count(
                    policy,
                    &selected_rank,
                )?)
            }

            fn all_of(
                self,
                policy: &CubePolicy<R>,
                pred: Pred,
            ) -> Result<bool, Error> {
                let _ = pred;
                let len = self.len();
                let selected_rank = wide_predicate_rank!(
                    policy; Pred; ($( $ty ),+); $( &self.$field.column ),+
                )?;
                Ok(mindex_from_usize(crate::detail::primitives::select::selected_count(
                    policy,
                    &selected_rank,
                )?)? == mindex_from_usize(len)?)
            }

            fn any_of(
                self,
                policy: &CubePolicy<R>,
                pred: Pred,
            ) -> Result<bool, Error> {
                let _ = pred;
                let selected_rank = wide_predicate_rank!(
                    policy; Pred; ($( $ty ),+); $( &self.$field.column ),+
                )?;
                Ok(crate::detail::primitives::select::selected_count(policy, &selected_rank)? != 0)
            }

            fn none_of(
                self,
                policy: &CubePolicy<R>,
                pred: Pred,
            ) -> Result<bool, Error> {
                let _ = pred;
                let selected_rank = wide_predicate_rank!(
                    policy; Pred; ($( $ty ),+); $( &self.$field.column ),+
                )?;
                Ok(crate::detail::primitives::select::selected_count(policy, &selected_rank)? == 0)
            }

            fn find_if(
                self,
                policy: &CubePolicy<R>,
                pred: Pred,
            ) -> Result<Option<crate::MIndex>, Error> {
                let _ = pred;
                let selected_rank = wide_predicate_rank!(
                    policy; Pred; ($( $ty ),+); $( &self.$field.column ),+
                )?;
                let search = crate::detail::control::SearchControl::from_flags(
                    selected_rank.flag.clone(),
                    selected_rank.len,
                    selected_rank.len,
                );
                crate::detail::apply::QueryApply::first_flag(policy, search)
            }

            fn is_partitioned(
                self,
                policy: &CubePolicy<R>,
                pred: Pred,
            ) -> Result<bool, Error> {
                let _ = pred;
                let selected_rank = wide_predicate_rank!(
                    policy; Pred; ($( $ty ),+); $( &self.$field.column ),+
                )?;
                let first_rejected = crate::detail::primitives::search::first_unset_flag(
                    policy,
                    selected_rank.flag.clone(),
                    selected_rank.len,
                    selected_rank.len,
                )?
                .unwrap_or(mindex_from_usize(selected_rank.len)?);
                let selected_count =
                    crate::detail::primitives::select::selected_count(policy, &selected_rank)?;
                Ok(mindex_from_usize(selected_count)? == first_rejected)
            }
        }
    };
}

impl_wide_zip_predicate_query!(ZipRead4; A: a, B: b, C: c, D: d);
impl_wide_zip_predicate_query!(ZipRead5; A: a, B: b, C: c, D: d, E: e);
impl_wide_zip_predicate_query!(ZipRead6; A: a, B: b, C: c, D: d, E: e, F: f);
impl_wide_zip_predicate_query!(ZipRead7; A: a, B: b, C: c, D: d, E: e, F: f, G: g);

macro_rules! impl_flat_zip_predicate_query {
    ($name:ident, $view:ident, { $( $ty:ident : $field:ident => $view_field:ident ),+ }) => {
        impl<R, $( $ty, )+ Pred> KernelReadPredicateQuery<R, Pred>
            for $name<$( ColumnRead<R, $ty> ),+>
        where
            R: Runtime,
            $( $ty: MStorageElement + 'static, )+
            Pred: op::PredicateOp<R, ($( $ty, )+)>,
        {
            fn count_if(
                self,
                policy: &CubePolicy<R>,
                _pred: Pred,
            ) -> Result<crate::MIndex, Error> {
                crate::detail::count_if(
                    policy,
                    $view { $( $view_field: self.$field.column, )+ },
                    KernelOp::<R, Pred>::new(),
                )
            }

            fn all_of(
                self,
                policy: &CubePolicy<R>,
                _pred: Pred,
            ) -> Result<bool, Error> {
                crate::detail::all_of(
                    policy,
                    $view { $( $view_field: self.$field.column, )+ },
                    KernelOp::<R, Pred>::new(),
                )
            }

            fn any_of(
                self,
                policy: &CubePolicy<R>,
                _pred: Pred,
            ) -> Result<bool, Error> {
                crate::detail::any_of(
                    policy,
                    $view { $( $view_field: self.$field.column, )+ },
                    KernelOp::<R, Pred>::new(),
                )
            }

            fn none_of(
                self,
                policy: &CubePolicy<R>,
                _pred: Pred,
            ) -> Result<bool, Error> {
                crate::detail::none_of(
                    policy,
                    $view { $( $view_field: self.$field.column, )+ },
                    KernelOp::<R, Pred>::new(),
                )
            }

            fn find_if(
                self,
                policy: &CubePolicy<R>,
                _pred: Pred,
            ) -> Result<Option<crate::MIndex>, Error> {
                crate::detail::find_if(
                    policy,
                    $view { $( $view_field: self.$field.column, )+ },
                    KernelOp::<R, Pred>::new(),
                )
            }

            fn is_partitioned(
                self,
                policy: &CubePolicy<R>,
                _pred: Pred,
            ) -> Result<bool, Error> {
                crate::detail::is_partitioned(
                    policy,
                    $view { $( $view_field: self.$field.column, )+ },
                    KernelOp::<R, Pred>::new(),
                )
            }
        }
    };
}

impl_flat_zip_predicate_query!(ZipRead2, ZipView2, { A: a => left, B: b => right });
impl_flat_zip_predicate_query!(ZipRead3, ZipView3, { A: a => first, B: b => second, C: c => third });

impl<R, A, Output, Pred> KernelReadPartition<R, Output, Pred> for ZipRead1<ColumnRead<R, A>>
where
    R: Runtime,
    A: MStorageElement + 'static,
    Output: MIterMut<R, Item = (A,)>,
    Pred: op::PredicateOp<R, (A,)>,
{
    fn partition_into(
        self,
        policy: &CubePolicy<R>,
        _pred: Pred,
        output: Output,
    ) -> Result<crate::MIndex, Error> {
        let (matching, failing) =
            crate::detail::partition(policy, (self.a.column,), KernelOp::<R, Pred>::new())?;
        let split = mindex_from_usize(matching.0.len())?;
        output.write_split_from_inner(policy, matching, failing)?;
        Ok(split)
    }
}

macro_rules! impl_flat_zip_partition_small {
    ($name:ident, $view:ident, { $( $ty:ident : $field:ident => $view_field:ident ),+ }) => {
        impl<R, $( $ty, )+ Output, Pred> KernelReadPartition<R, Output, Pred>
            for $name<$( ColumnRead<R, $ty> ),+>
        where
            R: Runtime,
            $( $ty: MStorageElement + 'static, )+
            Output: MIterMut<R, Item = ($( $ty, )+)>,
            Pred: op::PredicateOp<R, ($( $ty, )+)>,
        {
            fn partition_into(
                self,
                policy: &CubePolicy<R>,
                _pred: Pred,
                output: Output,
            ) -> Result<crate::MIndex, Error> {
                let (matching, failing) = crate::detail::partition(
                    policy,
                    $view { $( $view_field: self.$field.column, )+ },
                    KernelOp::<R, Pred>::new(),
                )?;
                let split = mindex_from_usize(matching.0.len())?;
                output.write_split_from_inner(policy, matching, failing)?;
                Ok(split)
            }
        }
    };
}

impl_flat_zip_partition_small!(ZipRead2, ZipView2, { A: a => left, B: b => right });
impl_flat_zip_partition_small!(ZipRead3, ZipView3, { A: a => first, B: b => second, C: c => third });

macro_rules! impl_flat_zip_partition_wide {
    ($name:ident, $apply:ident; $( $ty:ident : $field:ident ),+) => {
        impl<R, $( $ty, )+ Output, Pred> KernelReadPartition<R, Output, Pred>
            for $name<$( ColumnRead<R, $ty> ),+>
        where
            R: Runtime,
            $( $ty: MStorageElement + 'static, )+
            Output: MIterMut<R, Item = ($( $ty, )+)>,
            Pred: op::PredicateOp<R, ($( $ty, )+)>,
        {
            fn partition_into(
                self,
                policy: &CubePolicy<R>,
                _pred: Pred,
                output: Output,
            ) -> Result<crate::MIndex, Error> {
                let selected_rank = wide_predicate_rank!(
                    policy; Pred; ($( $ty ),+); $( &self.$field.column ),+
                )?;
                let (split_rank, matching_count, failing_count) =
                    crate::detail::primitives::select::split_rank_from_selected(
                        policy,
                        selected_rank,
                    )?;
                let payload_apply =
                    crate::detail::apply::SplitPayloadApply::new(&split_rank, matching_count, failing_count);
                let (matching, failing) = payload_apply.$apply(policy, $( &self.$field.column, )+)?;
                let split = mindex_from_usize(matching_count)?;
                output.write_split_from_inner(policy, matching, failing)?;
                Ok(split)
            }
        }
    };
}

impl_flat_zip_partition_wide!(ZipRead4, apply_expr4; A: a, B: b, C: c, D: d);
impl_flat_zip_partition_wide!(ZipRead5, apply_expr5; A: a, B: b, C: c, D: d, E: e);
impl_flat_zip_partition_wide!(ZipRead6, apply_expr6; A: a, B: b, C: c, D: d, E: e, F: f);
impl_flat_zip_partition_wide!(ZipRead7, apply_expr7; A: a, B: b, C: c, D: d, E: e, F: f, G: g);

impl<R, A, Pred> KernelReadPredicateQuery<R, Pred> for ZipRead1<ColumnRead<R, A>>
where
    R: Runtime,
    A: MStorageElement + 'static,
    Pred: op::PredicateOp<R, (A,)>,
{
    fn count_if(self, policy: &CubePolicy<R>, _pred: Pred) -> Result<crate::MIndex, Error> {
        crate::detail::count_if(policy, (self.a.column,), KernelOp::<R, Pred>::new())
    }

    fn all_of(self, policy: &CubePolicy<R>, _pred: Pred) -> Result<bool, Error> {
        crate::detail::all_of(policy, (self.a.column,), KernelOp::<R, Pred>::new())
    }

    fn any_of(self, policy: &CubePolicy<R>, _pred: Pred) -> Result<bool, Error> {
        crate::detail::any_of(policy, (self.a.column,), KernelOp::<R, Pred>::new())
    }

    fn none_of(self, policy: &CubePolicy<R>, _pred: Pred) -> Result<bool, Error> {
        crate::detail::none_of(policy, (self.a.column,), KernelOp::<R, Pred>::new())
    }

    fn find_if(self, policy: &CubePolicy<R>, _pred: Pred) -> Result<Option<crate::MIndex>, Error> {
        crate::detail::find_if(policy, (self.a.column,), KernelOp::<R, Pred>::new())
    }

    fn is_partitioned(self, policy: &CubePolicy<R>, _pred: Pred) -> Result<bool, Error> {
        crate::detail::is_partitioned(policy, (self.a.column,), KernelOp::<R, Pred>::new())
    }
}
