use super::*;

pub trait MItemDispatch<R: Runtime>: Sized {
    fn transform_scalar_input<Input, Op>(
        policy: &crate::detail::CubePolicy<R>,
        input: crate::detail::device::DeviceColumnView<R, Input>,
        op: Op,
    ) -> Result<<Self as MAlloc<R>>::Inner, Error>
    where
        Self: MAlloc<R>,
        Input: MStorageElement + crate::value::MItem<R>,
        Op: op::UnaryOp<R, Input, Output = Self>,
    {
        let _ = (policy, input, op);
        Err(Error::Launch {
            message: "transform is not supported for this input/output item shape".to_string(),
        })
    }

    fn transform_unary<Input, Op>(
        policy: &crate::detail::CubePolicy<R>,
        input: crate::detail::device::DeviceColumnView<R, Input>,
        op: Op,
    ) -> Result<<Self as MAlloc<R>>::Inner, Error>
    where
        Self: MAlloc<R>,
        Input: MStorageElement,
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
    ) -> Result<<Self as MAlloc<R>>::Inner, Error>
    where
        Self: MAlloc<R>,
        Left: MStorageElement,
        Right: MStorageElement,
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
    ) -> Result<<Self as MAlloc<R>>::Inner, Error>
    where
        Self: MAlloc<R>,
        First: MStorageElement,
        Second: MStorageElement,
        Third: MStorageElement,
        Op: op::UnaryOp<R, (First, Second, Third), Output = Self>,
    {
        let _ = (policy, first, second, third, op);
        Err(Error::Launch {
            message: "transform is not supported for this output item shape".to_string(),
        })
    }

    fn transform_logical3<Input, LeafA, LeafB, LeafC, Expr, Op>(
        policy: &crate::detail::CubePolicy<R>,
        bindings: crate::detail::device::KernelColumnBindings,
        len: usize,
        op: Op,
    ) -> Result<<Self as MAlloc<R>>::Inner, Error>
    where
        Self: MAlloc<R>,
        Input: crate::value::MItem<R> + Send + Sync,
        LeafA: MStorageElement,
        LeafB: MStorageElement,
        LeafC: MStorageElement,
        Expr: crate::expr::LogicalDeviceExpr3<Input, LeafA, LeafB, LeafC>,
        Op: op::UnaryOp<R, Input, Output = Self>,
    {
        let _ = (policy, bindings, len, op);
        Err(Error::Launch {
            message: "transform is not supported for this logical input shape".to_string(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn transform_logical7<Input, Leaf0, Leaf1, Leaf2, Leaf3, Leaf4, Leaf5, Leaf6, Leaf7, Expr, Op>(
        policy: &crate::detail::CubePolicy<R>,
        bindings: crate::detail::device::KernelColumnBindings,
        len: usize,
        op: Op,
    ) -> Result<<Self as MAlloc<R>>::Inner, Error>
    where
        Self: MAlloc<R>,
        Input: crate::value::MItem<R> + Send + Sync,
        Leaf0: MStorageElement,
        Leaf1: MStorageElement,
        Leaf2: MStorageElement,
        Leaf3: MStorageElement,
        Leaf4: MStorageElement,
        Leaf5: MStorageElement,
        Leaf6: MStorageElement,
        Leaf7: MStorageElement,
        Expr: crate::expr::LogicalDeviceExpr7<
                Input,
                Leaf0,
                Leaf1,
                Leaf2,
                Leaf3,
                Leaf4,
                Leaf5,
                Leaf6,
                Leaf7,
            >,
        Op: op::UnaryOp<R, Input, Output = Self>,
    {
        let _ = (policy, bindings, len, op);
        Err(Error::Launch {
            message: "transform is not supported for this logical input shape".to_string(),
        })
    }

    fn transform_quaternary<First, Second, Third, Fourth, Op>(
        policy: &crate::detail::CubePolicy<R>,
        first: crate::detail::device::DeviceColumnView<R, First>,
        second: crate::detail::device::DeviceColumnView<R, Second>,
        third: crate::detail::device::DeviceColumnView<R, Third>,
        fourth: crate::detail::device::DeviceColumnView<R, Fourth>,
        op: Op,
    ) -> Result<<Self as MAlloc<R>>::Inner, Error>
    where
        Self: MAlloc<R>,
        First: MStorageElement,
        Second: MStorageElement,
        Third: MStorageElement,
        Fourth: MStorageElement,
        Op: op::UnaryOp<R, (First, Second, Third, Fourth), Output = Self>,
    {
        let _ = (policy, first, second, third, fourth, op);
        Err(Error::Launch {
            message: "transform is not supported for this output item shape".to_string(),
        })
    }

    fn transform_quinary<First, Second, Third, Fourth, Fifth, Op>(
        policy: &crate::detail::CubePolicy<R>,
        first: crate::detail::device::DeviceColumnView<R, First>,
        second: crate::detail::device::DeviceColumnView<R, Second>,
        third: crate::detail::device::DeviceColumnView<R, Third>,
        fourth: crate::detail::device::DeviceColumnView<R, Fourth>,
        fifth: crate::detail::device::DeviceColumnView<R, Fifth>,
        op: Op,
    ) -> Result<<Self as MAlloc<R>>::Inner, Error>
    where
        Self: MAlloc<R>,
        First: MStorageElement,
        Second: MStorageElement,
        Third: MStorageElement,
        Fourth: MStorageElement,
        Fifth: MStorageElement,
        Op: op::UnaryOp<R, (First, Second, Third, Fourth, Fifth), Output = Self>,
    {
        let _ = (policy, first, second, third, fourth, fifth, op);
        Err(Error::Launch {
            message: "transform is not supported for this output item shape".to_string(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn transform_senary<First, Second, Third, Fourth, Fifth, Sixth, Op>(
        policy: &crate::detail::CubePolicy<R>,
        first: crate::detail::device::DeviceColumnView<R, First>,
        second: crate::detail::device::DeviceColumnView<R, Second>,
        third: crate::detail::device::DeviceColumnView<R, Third>,
        fourth: crate::detail::device::DeviceColumnView<R, Fourth>,
        fifth: crate::detail::device::DeviceColumnView<R, Fifth>,
        sixth: crate::detail::device::DeviceColumnView<R, Sixth>,
        op: Op,
    ) -> Result<<Self as MAlloc<R>>::Inner, Error>
    where
        Self: MAlloc<R>,
        First: MStorageElement,
        Second: MStorageElement,
        Third: MStorageElement,
        Fourth: MStorageElement,
        Fifth: MStorageElement,
        Sixth: MStorageElement,
        Op: op::UnaryOp<R, (First, Second, Third, Fourth, Fifth, Sixth), Output = Self>,
    {
        let _ = (policy, first, second, third, fourth, fifth, sixth, op);
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
    ) -> Result<<Self as MAlloc<R>>::Inner, Error>
    where
        Self: MAlloc<R>,
        First: MStorageElement,
        Second: MStorageElement,
        Third: MStorageElement,
        Fourth: MStorageElement,
        Fifth: MStorageElement,
        Sixth: MStorageElement,
        Seventh: MStorageElement,
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
        input: <Self as MAlloc<R>>::Inner,
        init: Self,
        op: Op,
    ) -> Result<Self, Error>
    where
        Self: MAlloc<R>,
        Op: op::ReductionOp<R, Self>,
    {
        let _ = (policy, input, init, op);
        Err(Error::Launch {
            message: "reduce is not supported for this item shape".to_string(),
        })
    }
}
