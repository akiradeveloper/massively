use super::*;

pub trait MItemDispatch<B: Runtime>: Sized {
    fn transform_unary<Input, Op>(
        policy: &crate::detail::CubePolicy<B>,
        input: crate::detail::device::DeviceColumnView<B, Input>,
        op: Op,
    ) -> Result<<Self as MItem<B>>::Inner, Error>
    where
        Self: MItem<B>,
        Input: Scalar,
        Op: op::UnaryOp<B, (Input,), Output = Self>,
    {
        let _ = (policy, input, op);
        Err(Error::Launch {
            message: "transform is not supported for this output item shape".to_string(),
        })
    }

    fn transform_binary<Left, Right, Op>(
        policy: &crate::detail::CubePolicy<B>,
        left: crate::detail::device::DeviceColumnView<B, Left>,
        right: crate::detail::device::DeviceColumnView<B, Right>,
        op: Op,
    ) -> Result<<Self as MItem<B>>::Inner, Error>
    where
        Self: MItem<B>,
        Left: Scalar,
        Right: Scalar,
        Op: op::UnaryOp<B, (Left, Right), Output = Self>,
    {
        let _ = (policy, left, right, op);
        Err(Error::Launch {
            message: "transform is not supported for this output item shape".to_string(),
        })
    }

    fn transform_ternary<First, Second, Third, Op>(
        policy: &crate::detail::CubePolicy<B>,
        first: crate::detail::device::DeviceColumnView<B, First>,
        second: crate::detail::device::DeviceColumnView<B, Second>,
        third: crate::detail::device::DeviceColumnView<B, Third>,
        op: Op,
    ) -> Result<<Self as MItem<B>>::Inner, Error>
    where
        Self: MItem<B>,
        First: Scalar,
        Second: Scalar,
        Third: Scalar,
        Op: op::UnaryOp<B, (First, Second, Third), Output = Self>,
    {
        let _ = (policy, first, second, third, op);
        Err(Error::Launch {
            message: "transform is not supported for this output item shape".to_string(),
        })
    }

    fn reduce_inner<Op>(
        policy: &crate::detail::CubePolicy<B>,
        input: <Self as MItem<B>>::Inner,
        init: Self,
        op: Op,
    ) -> Result<Self, Error>
    where
        Self: MItem<B>,
        Op: op::ReductionOp<B, Self>,
    {
        let _ = (policy, input, init, op);
        Err(Error::Launch {
            message: "reduce is not supported for this item shape".to_string(),
        })
    }

    fn inner_product_with_right_item<LeftIter, RightIter, TransformOp, ReduceOp, Output>(
        policy: &crate::detail::CubePolicy<B>,
        left: LeftIter,
        right: RightIter,
        transform_op: TransformOp,
        init: Output,
        reduce_op: ReduceOp,
    ) -> Result<Output, Error>
    where
        Self: MItem<B>,
        LeftIter: MIter<B, Item = Self>,
        RightIter: MIter<B>,
        TransformOp: op::BinaryOp<B, Self, <RightIter as MIter<B>>::Item, Output = Output>,
        Output: MItem<B>,
        ReduceOp: op::ReductionOp<B, Output>,
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
        policy: &crate::detail::CubePolicy<B>,
        left: LeftIter,
        right: RightIter,
        transform_op: TransformOp,
        init: Output,
        reduce_op: ReduceOp,
    ) -> Result<Output, Error>
    where
        Self: MItem<B>,
        LeftScalar: Scalar + 'static,
        LeftIter: MIter<B, Item = (LeftScalar,)>,
        RightIter: MIter<B, Item = Self>,
        TransformOp: op::BinaryOp<B, (LeftScalar,), Self, Output = Output>,
        Output: MItem<B>,
        ReduceOp: op::ReductionOp<B, Output>,
    {
        let _ = (policy, left, right, transform_op, init, reduce_op);
        Err(Error::Launch {
            message: "inner_product is not supported for this iterator shape".to_string(),
        })
    }
}
