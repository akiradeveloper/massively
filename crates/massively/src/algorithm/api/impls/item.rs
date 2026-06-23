use super::*;

macro_rules! inner_product_left_item_body {
    ($B:ident; ($left_ty:ident); $policy:ident, $left:ident, $right:ident, $transform_op:ident, $init:ident, $reduce_op:ident) => {{
        <<RightIter as MIter<$B>>::Item as sealed::MItemDispatch<$B>>::inner_product_with_left_scalar::<
            LeftIter,
            RightIter,
            $left_ty,
            TransformOp,
            ReduceOp,
            Output,
        >($policy, $left, $right, $transform_op, $init, $reduce_op)
    }};
    ($B:ident; ($first:ident, $( $rest:ident ),+); $policy:ident, $left:ident, $right:ident, $transform_op:ident, $init:ident, $reduce_op:ident) => {{
        let _ = ($policy, $left, $right, $transform_op, $init, $reduce_op);
        Err(Error::Launch {
            message: "inner_product is not supported for this iterator shape".to_string(),
        })
    }};
}

macro_rules! inner_product_right_item_body {
    ($B:ident; ($right_ty:ident); $policy:ident, $left:ident, $right:ident, $transform_op:ident, $init:ident, $reduce_op:ident) => {{
        let left = column_view_at::<$B, LeftIter, LeftScalar>(&$left, 0, "inner_product")?;
        let right = column_view_at::<$B, RightIter, $right_ty>(&$right, 0, "inner_product")?;
        let transformed = <Output as sealed::MItemDispatch<$B>>::transform_binary(
            $policy,
            left,
            right,
            KernelTuple1InnerProductOp::<$B, TransformOp, Output>::new(),
        )?;
        let _ = $transform_op;
        <Output as sealed::MItemDispatch<$B>>::reduce_inner($policy, transformed, $init, $reduce_op)
    }};
    ($B:ident; ($first:ident, $( $rest:ident ),+); $policy:ident, $left:ident, $right:ident, $transform_op:ident, $init:ident, $reduce_op:ident) => {{
        let _ = ($policy, $left, $right, $transform_op, $init, $reduce_op);
        Err(Error::Launch {
            message: "inner_product is not supported for this iterator shape".to_string(),
        })
    }};
}

macro_rules! impl_mitem_tuple {
    ($( $ty:ident : $var:ident ),+) => {
        impl<B, $( $ty ),+> MItem<B> for ($( $ty, )+)
        where
            B: Backend,
            $( $ty: Scalar, )+
        {
            type Inner = ($( crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, $ty>, )+);
        }

        impl<B, $( $ty ),+> MVec<B> for ($( DeviceVec<B, $ty>, )+)
        where
            B: Backend,
            $( $ty: Scalar, )+
        {
            type Item = ($( $ty, )+);

            fn from_inner(inner: <Self::Item as MItem<B>>::Inner) -> Self {
                let ($( $var, )+) = inner;
                ($( DeviceVec::from_inner($var), )+)
            }

            fn len(&self) -> usize {
                self.0.len()
            }
        }

        impl<B, $( $ty ),+> sealed::MItemDispatch<B> for ($( $ty, )+)
        where
            B: Backend,
            $( $ty: Scalar, )+
        {
            fn transform_unary<Input, Op>(
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                input: crate::detail::device::DeviceColumnView<
                    <B as sealed::Backend>::Runtime,
                    Input,
                >,
                op: Op,
            ) -> Result<<Self as MItem<B>>::Inner, Error>
            where
                Input: Scalar,
                Op: op::UnaryOp<B, (Input,), Output = Self>,
                Self: crate::detail::TransformUnaryOutput<
                    <B as sealed::Backend>::Runtime,
                    Input,
                    KernelOp<B, Op>,
                >,
                <Self as crate::detail::MItemStorage<
                    <B as sealed::Backend>::Runtime,
                >>::Storage: crate::detail::MaterializeOutput<
                    Runtime = <B as sealed::Backend>::Runtime,
                    Output = ($(
                        crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage =
                    <Self as crate::detail::TransformUnaryOutput<
                        <B as sealed::Backend>::Runtime,
                        Input,
                        KernelOp<B, Op>,
                    >>::run(policy, input)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            fn transform_binary<Left, Right, Op>(
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                left: crate::detail::device::DeviceColumnView<
                    <B as sealed::Backend>::Runtime,
                    Left,
                >,
                right: crate::detail::device::DeviceColumnView<
                    <B as sealed::Backend>::Runtime,
                    Right,
                >,
                op: Op,
            ) -> Result<<Self as MItem<B>>::Inner, Error>
            where
                Left: Scalar,
                Right: Scalar,
                Op: op::UnaryOp<B, (Left, Right), Output = Self>,
                Self: crate::detail::TransformSoA2Output<
                    <B as sealed::Backend>::Runtime,
                    Left,
                    Right,
                    KernelOp<B, Op>,
                >,
                <Self as crate::detail::MItemStorage<
                    <B as sealed::Backend>::Runtime,
                >>::Storage: crate::detail::MaterializeOutput<
                    Runtime = <B as sealed::Backend>::Runtime,
                    Output = ($(
                        crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage =
                    <Self as crate::detail::TransformSoA2Output<
                        <B as sealed::Backend>::Runtime,
                        Left,
                        Right,
                        KernelOp<B, Op>,
                    >>::run(policy, left, right)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            fn transform_ternary<First, Second, Third, Op>(
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                first: crate::detail::device::DeviceColumnView<
                    <B as sealed::Backend>::Runtime,
                    First,
                >,
                second: crate::detail::device::DeviceColumnView<
                    <B as sealed::Backend>::Runtime,
                    Second,
                >,
                third: crate::detail::device::DeviceColumnView<
                    <B as sealed::Backend>::Runtime,
                    Third,
                >,
                op: Op,
            ) -> Result<<Self as MItem<B>>::Inner, Error>
            where
                First: Scalar,
                Second: Scalar,
                Third: Scalar,
                Op: op::UnaryOp<B, (First, Second, Third), Output = Self>,
                Self: crate::detail::TransformSoA3Output<
                    <B as sealed::Backend>::Runtime,
                    First,
                    Second,
                    Third,
                    KernelOp<B, Op>,
                >,
                <Self as crate::detail::MItemStorage<
                    <B as sealed::Backend>::Runtime,
                >>::Storage: crate::detail::MaterializeOutput<
                    Runtime = <B as sealed::Backend>::Runtime,
                    Output = ($(
                        crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage =
                    <Self as crate::detail::TransformSoA3Output<
                        <B as sealed::Backend>::Runtime,
                        First,
                        Second,
                        Third,
                        KernelOp<B, Op>,
                    >>::run(
                        policy,
                        first,
                        second,
                        third,
                )?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            fn reduce_inner<Op>(
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                input: <Self as MItem<B>>::Inner,
                init: Self,
                op: Op,
            ) -> Result<Self, Error>
            where
                Op: op::ReductionOp<B, Self>,
            {
                let _ = op;
                crate::detail::reduce(policy, input, init, KernelOp::<B, Op>::new())
            }

            fn inner_product_with_right_item<
                LeftIter,
                RightIter,
                TransformOp,
                ReduceOp,
                Output,
            >(
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                left: LeftIter,
                right: RightIter,
                transform_op: TransformOp,
                init: Output,
                reduce_op: ReduceOp,
            ) -> Result<Output, Error>
            where
                LeftIter: MIter<B, Item = Self>,
                RightIter: MIter<B>,
                TransformOp:
                    op::BinaryOp<B, Self, <RightIter as MIter<B>>::Item, Output = Output>,
                Output: MItem<B>,
                ReduceOp: op::ReductionOp<B, Output>,
            {
                inner_product_left_item_body!(
                    B;
                    ($( $ty ),+);
                    policy,
                    left,
                    right,
                    transform_op,
                    init,
                    reduce_op
                )
            }

            fn inner_product_with_left_scalar<
                LeftIter,
                RightIter,
                LeftScalar,
                TransformOp,
                ReduceOp,
                Output,
            >(
                policy: &crate::detail::CubePolicy<<B as sealed::Backend>::Runtime>,
                left: LeftIter,
                right: RightIter,
                transform_op: TransformOp,
                init: Output,
                reduce_op: ReduceOp,
            ) -> Result<Output, Error>
            where
                LeftScalar: Scalar + 'static,
                LeftIter: MIter<B, Item = (LeftScalar,)>,
                RightIter: MIter<B, Item = Self>,
                TransformOp: op::BinaryOp<B, (LeftScalar,), Self, Output = Output>,
                Output: MItem<B>,
                ReduceOp: op::ReductionOp<B, Output>,
            {
                inner_product_right_item_body!(
                    B;
                    ($( $ty ),+);
                    policy,
                    left,
                    right,
                    transform_op,
                    init,
                    reduce_op
                )
            }
        }
    };
}

impl_mitem_tuple!(A: a);
impl_mitem_tuple!(A: a, B0: b);
impl_mitem_tuple!(A: a, B0: b, C: c);

macro_rules! impl_wide_mitem_tuple {
    ($( $ty:ident : $var:ident ),+) => {
        impl<B, $( $ty ),+> MItem<B> for ($( $ty, )+)
        where
            B: Backend,
            $( $ty: Scalar, )+
        {
            type Inner = ($( crate::detail::DeviceVec<<B as sealed::Backend>::Runtime, $ty>, )+);
        }

        impl<B, $( $ty ),+> sealed::MItemDispatch<B> for ($( $ty, )+)
        where
            B: Backend,
            $( $ty: Scalar, )+
        {
        }

        impl<B, $( $ty ),+> MVec<B> for ($( DeviceVec<B, $ty>, )+)
        where
            B: Backend,
            $( $ty: Scalar, )+
        {
            type Item = ($( $ty, )+);

            fn from_inner(inner: <Self::Item as MItem<B>>::Inner) -> Self {
                let ($( $var, )+) = inner;
                ($( DeviceVec::from_inner($var), )+)
            }

            fn len(&self) -> usize {
                self.0.len()
            }
        }
    };
}

impl_wide_mitem_tuple!(A: a, B0: b, C: c, D: d);
impl_wide_mitem_tuple!(A: a, B0: b, C: c, D: d, E: e);
impl_wide_mitem_tuple!(A: a, B0: b, C: c, D: d, E: e, F: f);
impl_wide_mitem_tuple!(A: a, B0: b, C: c, D: d, E: e, F: f, G: g);
impl_wide_mitem_tuple!(A: a, B0: b, C: c, D: d, E: e, F: f, G: g, H: h);
impl_wide_mitem_tuple!(A: a, B0: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i);
impl_wide_mitem_tuple!(A: a, B0: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i, J: j);
impl_wide_mitem_tuple!(A: a, B0: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i, J: j, K: k);
impl_wide_mitem_tuple!(
    A: a,
    B0: b,
    C: c,
    D: d,
    E: e,
    F: f,
    G: g,
    H: h,
    I: i,
    J: j,
    K: k,
    L: l
);

impl<B, T> MVec<B> for SoA1<DeviceVec<B, T>>
where
    B: Backend,
    T: Scalar,
{
    type Item = (T,);

    fn from_inner(inner: <Self::Item as MItem<B>>::Inner) -> Self {
        Self(DeviceVec::from_inner(inner.0))
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<B, A, C> MVec<B> for SoA2<DeviceVec<B, A>, DeviceVec<B, C>>
where
    B: Backend,
    A: Scalar,
    C: Scalar,
{
    type Item = (A, C);

    fn from_inner(inner: <Self::Item as MItem<B>>::Inner) -> Self {
        Self(
            DeviceVec::from_inner(inner.0),
            DeviceVec::from_inner(inner.1),
        )
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<B, A, C, D> MVec<B> for SoA3<DeviceVec<B, A>, DeviceVec<B, C>, DeviceVec<B, D>>
where
    B: Backend,
    A: Scalar,
    C: Scalar,
    D: Scalar,
{
    type Item = (A, C, D);

    fn from_inner(inner: <Self::Item as MItem<B>>::Inner) -> Self {
        Self(
            DeviceVec::from_inner(inner.0),
            DeviceVec::from_inner(inner.1),
            DeviceVec::from_inner(inner.2),
        )
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}
