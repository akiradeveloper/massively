use super::*;

pub trait MItemDispatch<R: Runtime>: Sized {
    fn transform_unary<Input, Op>(
        policy: &crate::detail::CubePolicy<R>,
        input: crate::detail::device::DeviceColumnView<R, Input>,
        op: Op,
    ) -> Result<<Self as MItem<R>>::Inner, Error>
    where
        Self: MItem<R>,
        Input: Scalar,
        Op: op::UnaryOp<R, (Input,), Output = Self>,
    {
        let _ = (policy, input, op);
        Err(Error::Launch {
            message: "transform is not supported for this output item shape".to_string(),
        })
    }

    fn transform_binary<Left, Right, Op>(
        policy: &crate::detail::CubePolicy<R>,
        left: crate::detail::device::DeviceColumnView<R, Left>,
        right: crate::detail::device::DeviceColumnView<R, Right>,
        op: Op,
    ) -> Result<<Self as MItem<R>>::Inner, Error>
    where
        Self: MItem<R>,
        Left: Scalar,
        Right: Scalar,
        Op: op::UnaryOp<R, (Left, Right), Output = Self>,
    {
        let _ = (policy, left, right, op);
        Err(Error::Launch {
            message: "transform is not supported for this output item shape".to_string(),
        })
    }

    fn transform_ternary<First, Second, Third, Op>(
        policy: &crate::detail::CubePolicy<R>,
        first: crate::detail::device::DeviceColumnView<R, First>,
        second: crate::detail::device::DeviceColumnView<R, Second>,
        third: crate::detail::device::DeviceColumnView<R, Third>,
        op: Op,
    ) -> Result<<Self as MItem<R>>::Inner, Error>
    where
        Self: MItem<R>,
        First: Scalar,
        Second: Scalar,
        Third: Scalar,
        Op: op::UnaryOp<R, (First, Second, Third), Output = Self>,
    {
        let _ = (policy, first, second, third, op);
        Err(Error::Launch {
            message: "transform is not supported for this output item shape".to_string(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn transform_septenary<First, Second, Third, Fourth, Fifth, Sixth, Seventh, Op>(
        policy: &crate::detail::CubePolicy<R>,
        first: crate::detail::device::DeviceColumnView<R, First>,
        second: crate::detail::device::DeviceColumnView<R, Second>,
        third: crate::detail::device::DeviceColumnView<R, Third>,
        fourth: crate::detail::device::DeviceColumnView<R, Fourth>,
        fifth: crate::detail::device::DeviceColumnView<R, Fifth>,
        sixth: crate::detail::device::DeviceColumnView<R, Sixth>,
        seventh: crate::detail::device::DeviceColumnView<R, Seventh>,
        op: Op,
    ) -> Result<<Self as MItem<R>>::Inner, Error>
    where
        Self: MItem<R>,
        First: Scalar,
        Second: Scalar,
        Third: Scalar,
        Fourth: Scalar,
        Fifth: Scalar,
        Sixth: Scalar,
        Seventh: Scalar,
        Op: op::UnaryOp<R, (First, Second, Third, Fourth, Fifth, Sixth, Seventh), Output = Self>,
    {
        let _ = (
            policy, first, second, third, fourth, fifth, sixth, seventh, op,
        );
        Err(Error::Launch {
            message: "transform is not supported for this output item shape".to_string(),
        })
    }

    fn reduce_inner<Op>(
        policy: &crate::detail::CubePolicy<R>,
        input: <Self as MItem<R>>::Inner,
        init: Self,
        op: Op,
    ) -> Result<Self, Error>
    where
        Self: MItem<R>,
        Op: op::ReductionOp<R, Self>,
    {
        let _ = (policy, input, init, op);
        Err(Error::Launch {
            message: "reduce is not supported for this item shape".to_string(),
        })
    }

    fn inner_product_with_right_item<LeftIter, RightIter, TransformOp, ReduceOp, Output>(
        policy: &crate::detail::CubePolicy<R>,
        left: LeftIter,
        right: RightIter,
        transform_op: TransformOp,
        init: Output,
        reduce_op: ReduceOp,
    ) -> Result<Output, Error>
    where
        Self: MItem<R>,
        LeftIter: MIter<R, Item = Self, Inner = <Self as MItem<R>>::View>,
        RightIter: MIter<R, Inner = <<RightIter as MIter<R>>::Item as MItem<R>>::View>,
        TransformOp: op::BinaryOp<R, Self, <RightIter as MIter<R>>::Item, Output = Output>,
        Output: MItem<R>,
        ReduceOp: op::ReductionOp<R, Output>,
    {
        let _ = (policy, left, right, transform_op, init, reduce_op);
        Err(Error::Launch {
            message: "inner_product is not supported for this iterator shape".to_string(),
        })
    }

    fn inner_product_with_left_scalar<
        LeftIter,
        RightIter,
        LeftScalar,
        TransformOp,
        ReduceOp,
        Output,
    >(
        policy: &crate::detail::CubePolicy<R>,
        left: LeftIter,
        right: RightIter,
        transform_op: TransformOp,
        init: Output,
        reduce_op: ReduceOp,
    ) -> Result<Output, Error>
    where
        Self: MItem<R>,
        LeftScalar: Scalar + 'static,
        (LeftScalar,): MItem<R, View = (crate::detail::device::DeviceColumnView<R, LeftScalar>,)>,
        LeftIter: MIter<R, Item = (LeftScalar,), Inner = <(LeftScalar,) as MItem<R>>::View>,
        RightIter: MIter<R, Item = Self, Inner = <Self as MItem<R>>::View>,
        TransformOp: op::BinaryOp<R, (LeftScalar,), Self, Output = Output>,
        Output: MItem<R>,
        ReduceOp: op::ReductionOp<R, Output>,
    {
        let _ = (policy, left, right, transform_op, init, reduce_op);
        Err(Error::Launch {
            message: "inner_product is not supported for this iterator shape".to_string(),
        })
    }
}
