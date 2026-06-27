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
            B: Runtime,
            $( $ty: Scalar, )+
        {
            type Inner = ($( crate::detail::DeviceVec<B, $ty>, )+);
        }

        impl<B, $( $ty ),+> MVec<B> for ($( DeviceVec<B, $ty>, )+)
        where
            B: Runtime,
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
            B: Runtime,
            $( $ty: Scalar, )+
        {
            fn transform_unary<Input, Op>(
                policy: &crate::detail::CubePolicy<B>,
                input: crate::detail::device::DeviceColumnView<
                    B,
                    Input,
                >,
                op: Op,
            ) -> Result<<Self as MItem<B>>::Inner, Error>
            where
                Input: Scalar,
                Op: op::UnaryOp<B, (Input,), Output = Self>,
                Self: crate::detail::TransformUnaryOutput<
                    B,
                    Input,
                    KernelOp<B, Op>,
                >,
                <Self as crate::detail::MItemStorage<
                    B,
                >>::Storage: crate::detail::MaterializeOutput<
                    Runtime = B,
                    Output = ($(
                        crate::detail::DeviceVec<B, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage =
                    <Self as crate::detail::TransformUnaryOutput<
                        B,
                        Input,
                        KernelOp<B, Op>,
                    >>::run(policy, input)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            fn transform_binary<Left, Right, Op>(
                policy: &crate::detail::CubePolicy<B>,
                left: crate::detail::device::DeviceColumnView<
                    B,
                    Left,
                >,
                right: crate::detail::device::DeviceColumnView<
                    B,
                    Right,
                >,
                op: Op,
            ) -> Result<<Self as MItem<B>>::Inner, Error>
            where
                Left: Scalar,
                Right: Scalar,
                Op: op::UnaryOp<B, (Left, Right), Output = Self>,
                Self: crate::detail::TransformSoA2Output<
                    B,
                    Left,
                    Right,
                    KernelOp<B, Op>,
                >,
                <Self as crate::detail::MItemStorage<
                    B,
                >>::Storage: crate::detail::MaterializeOutput<
                    Runtime = B,
                    Output = ($(
                        crate::detail::DeviceVec<B, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage =
                    <Self as crate::detail::TransformSoA2Output<
                        B,
                        Left,
                        Right,
                        KernelOp<B, Op>,
                    >>::run(policy, left, right)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            fn transform_ternary<First, Second, Third, Op>(
                policy: &crate::detail::CubePolicy<B>,
                first: crate::detail::device::DeviceColumnView<
                    B,
                    First,
                >,
                second: crate::detail::device::DeviceColumnView<
                    B,
                    Second,
                >,
                third: crate::detail::device::DeviceColumnView<
                    B,
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
                    B,
                    First,
                    Second,
                    Third,
                    KernelOp<B, Op>,
                >,
                <Self as crate::detail::MItemStorage<
                    B,
                >>::Storage: crate::detail::MaterializeOutput<
                    Runtime = B,
                    Output = ($(
                        crate::detail::DeviceVec<B, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage =
                    <Self as crate::detail::TransformSoA3Output<
                        B,
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
                policy: &crate::detail::CubePolicy<B>,
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
                policy: &crate::detail::CubePolicy<B>,
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
                policy: &crate::detail::CubePolicy<B>,
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
            B: Runtime,
            $( $ty: Scalar, )+
        {
            type Inner = ($( crate::detail::DeviceVec<B, $ty>, )+);
        }

        impl<B, $( $ty ),+> sealed::MItemDispatch<B> for ($( $ty, )+)
        where
            B: Runtime,
            $( $ty: Scalar, )+
        {
        }

        impl<B, $( $ty ),+> MVec<B> for ($( DeviceVec<B, $ty>, )+)
        where
            B: Runtime,
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

macro_rules! impl_miter_mut_tuple {
    ($( $ty:ident : $var:ident : $idx:tt ),+) => {
        impl<'a, B, $( $ty ),+> MIterMut<B> for ($( DeviceSliceMut<'a, B, $ty>, )+)
        where
            B: Runtime,
            $( $ty: Scalar + 'static, )+
            ($( $ty, )+): MItem<B, Inner = ($( crate::detail::DeviceVec<B, $ty>, )+)>,
        {
            type Item = ($( $ty, )+);
            type Inner = ($( crate::detail::device::DeviceColumnMutView<B, $ty>, )+);

            fn into_inner(self) -> Self::Inner {
                ($(
                    crate::detail::device::DeviceColumnMutView::from_slice(
                        &self.$idx.source.inner,
                        self.$idx.offset,
                        self.$idx.len,
                    ),
                )+)
            }

            fn write_from_inner(
                self,
                policy: &crate::detail::CubePolicy<B>,
                inner: <Self::Item as MItem<B>>::Inner,
            ) -> Result<(), Error> {
                let output = self.into_inner();
                $(
                    {
                        let input =
                            crate::detail::device::DeviceColumnView::from_column(&inner.$idx);
                        crate::detail::api::device_expr_collect_into_with_policy(
                            policy,
                            &input,
                            &output.$idx,
                        )?;
                    }
                )+
                Ok(())
            }

            fn write_where_from_inner(
                self,
                policy: &crate::detail::CubePolicy<B>,
                inner: <Self::Item as MItem<B>>::Inner,
                stencil: crate::detail::api::PrecomputedSelection<B>,
            ) -> Result<(), Error>
            {
                let output = self.into_inner();
                $(
                    {
                        let input =
                            crate::detail::device::DeviceColumnView::from_column(&inner.$idx);
                        crate::detail::api::device_expr_copy_where_into_with_policy(
                            policy,
                            &input,
                            &stencil,
                            &output.$idx,
                            KernelOp::<B, StencilFlag>::new(),
                        )?;
                    }
                )+
                Ok(())
            }

            fn replace_where_inner(
                self,
                policy: &crate::detail::CubePolicy<B>,
                replacement: Self::Item,
                stencil: crate::detail::api::PrecomputedSelection<B>,
            ) -> Result<(), Error>
            {
                let output = self.into_inner();
                $(
                    crate::detail::api::replace_where_into_with_policy(
                        policy,
                        replacement.$idx,
                        &stencil,
                        &output.$idx,
                        KernelOp::<B, StencilFlag>::new(),
                    )?;
                )+
                Ok(())
            }

            fn len(&self) -> usize {
                self.0.len()
            }
        }

        impl<'a, B, $( $ty ),+> sealed::MIterMutDispatch<B> for ($( DeviceSliceMut<'a, B, $ty>, )+)
        where
            B: Runtime,
            $( $ty: Scalar + 'static, )+
            ($( $ty, )+): MItem<B, Inner = ($( crate::detail::DeviceVec<B, $ty>, )+)>,
        {
            fn validate_executor(&self, exec: &Executor<B>) -> Result<(), Error> {
                $(
                    exec.ensure_policy_id(self.$idx.source.inner.policy_id())?;
                    ensure_same_len(self.$idx.len(), self.0.len())?;
                )+
                Ok(())
            }

            fn column_mut_view_by_index_inner<U: 'static>(
                &self,
                index: usize,
            ) -> Result<Option<crate::detail::device::DeviceColumnMutView<B, U>>, Error>
            where
                U: Scalar,
            {
                $(
                    if index == $idx {
                        let source = &*self.$idx.source as &dyn Any;
                        let source = match source.downcast_ref::<DeviceVec<B, U>>() {
                            Some(source) => source,
                            None => return Ok(None),
                        };
                        return Ok(Some(crate::detail::device::DeviceColumnMutView::from_slice(
                            &source.inner,
                            self.$idx.offset,
                            self.$idx.len,
                        )));
                    }
                )+
                Ok(None)
            }
        }
    };
}

impl_miter_mut_tuple!(A: a: 0);
impl_miter_mut_tuple!(A: a: 0, B0: b: 1);
impl_miter_mut_tuple!(A: a: 0, B0: b: 1, C: c: 2);
impl_miter_mut_tuple!(A: a: 0, B0: b: 1, C: c: 2, D: d: 3);
impl_miter_mut_tuple!(A: a: 0, B0: b: 1, C: c: 2, D: d: 3, E: e: 4);
impl_miter_mut_tuple!(A: a: 0, B0: b: 1, C: c: 2, D: d: 3, E: e: 4, F: f: 5);
impl_miter_mut_tuple!(A: a: 0, B0: b: 1, C: c: 2, D: d: 3, E: e: 4, F: f: 5, G: g: 6);
impl_miter_mut_tuple!(A: a: 0, B0: b: 1, C: c: 2, D: d: 3, E: e: 4, F: f: 5, G: g: 6, H: h: 7);
impl_miter_mut_tuple!(A: a: 0, B0: b: 1, C: c: 2, D: d: 3, E: e: 4, F: f: 5, G: g: 6, H: h: 7, I: i: 8);
impl_miter_mut_tuple!(A: a: 0, B0: b: 1, C: c: 2, D: d: 3, E: e: 4, F: f: 5, G: g: 6, H: h: 7, I: i: 8, J: j: 9);
impl_miter_mut_tuple!(A: a: 0, B0: b: 1, C: c: 2, D: d: 3, E: e: 4, F: f: 5, G: g: 6, H: h: 7, I: i: 8, J: j: 9, K: k: 10);
impl_miter_mut_tuple!(A: a: 0, B0: b: 1, C: c: 2, D: d: 3, E: e: 4, F: f: 5, G: g: 6, H: h: 7, I: i: 8, J: j: 9, K: k: 10, L: l: 11);

impl<B, T> MVec<B> for SoA1<DeviceVec<B, T>>
where
    B: Runtime,
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
    B: Runtime,
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
    B: Runtime,
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
