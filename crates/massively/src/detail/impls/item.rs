use super::*;

macro_rules! inner_product_left_item_body {
    ($R:ident; ($left_ty:ident); $policy:ident, $left:ident, $right:ident, $transform_op:ident, $init:ident, $reduce_op:ident) => {{
        <<RightIter as MIter<$R>>::Item as sealed::MItemDispatch<$R>>::inner_product_with_left_scalar::<
            LeftIter,
            RightIter,
            $left_ty,
            TransformOp,
            ReduceOp,
            Output,
        >($policy, $left, $right, $transform_op, $init, $reduce_op)
    }};
    ($R:ident; ($first:ident, $( $rest:ident ),+); $policy:ident, $left:ident, $right:ident, $transform_op:ident, $init:ident, $reduce_op:ident) => {{
        let _ = ($policy, $left, $right, $transform_op, $init, $reduce_op);
        Err(Error::Launch {
            message: "inner_product is not supported for this iterator shape".to_string(),
        })
    }};
}

macro_rules! inner_product_right_item_body {
    ($R:ident; ($right_ty:ident); $policy:ident, $left:ident, $right:ident, $transform_op:ident, $init:ident, $reduce_op:ident) => {{
        let (left,) = $left.into_inner_with_policy($policy)?;
        let (right,) = $right.into_inner_with_policy($policy)?;
        let transformed = <Output as sealed::MItemDispatch<$R>>::transform_binary(
            $policy,
            left,
            right,
            KernelTuple1InnerProductOp::<$R, TransformOp, Output>::new(),
        )?;
        let _ = $transform_op;
        <Output as sealed::MItemDispatch<$R>>::reduce_inner($policy, transformed, $init, $reduce_op)
    }};
    ($R:ident; ($first:ident, $( $rest:ident ),+); $policy:ident, $left:ident, $right:ident, $transform_op:ident, $init:ident, $reduce_op:ident) => {{
        let _ = ($policy, $left, $right, $transform_op, $init, $reduce_op);
        Err(Error::Launch {
            message: "inner_product is not supported for this iterator shape".to_string(),
        })
    }};
}

macro_rules! impl_mitem_tuple {
    ($( $ty:ident : $var:ident ),+) => {
        impl<R, $( $ty ),+> MItem<R> for ($( $ty, )+)
        where
            R: Runtime,
            $( $ty: Scalar, )+
        {
            type Inner = ($( crate::detail::DeviceVec<R, $ty>, )+);
            type View = ($( crate::detail::device::DeviceColumnView<R, $ty>, )+);
            type Vec = ($( DeviceVec<R, $ty>, )+);

            fn vec_from_inner(inner: Self::Inner) -> Self::Vec {
                let ($( $var, )+) = inner;
                ($( DeviceVec::from_inner($var), )+)
            }
        }

        impl<R, $( $ty ),+> MVec<R> for ($( DeviceVec<R, $ty>, )+)
        where
            R: Runtime,
            $( $ty: Scalar, )+
        {
            type Item = ($( $ty, )+);

            fn from_inner(inner: <Self::Item as MItem<R>>::Inner) -> Self {
                <Self::Item as MItem<R>>::vec_from_inner(inner)
            }
        }

        impl<R, $( $ty ),+> sealed::MItemDispatch<R> for ($( $ty, )+)
        where
            R: Runtime,
            $( $ty: Scalar, )+
        {
            fn transform_unary<Input, Op>(
                policy: &crate::detail::CubePolicy<R>,
                input: crate::detail::device::DeviceColumnView<
                    R,
                    Input,
                >,
                op: Op,
            ) -> Result<<Self as MItem<R>>::Inner, Error>
            where
                Input: Scalar,
                Op: op::UnaryOp<R, (Input,), Output = Self>,
                Self: crate::detail::TransformUnaryOutput<
                    R,
                    Input,
                    KernelOp<R, Op>,
                >,
                <Self as crate::detail::MItemStorage<
                    R,
                >>::Storage: crate::detail::MaterializeOutput<
                    Runtime = R,
                    Output = ($(
                        crate::detail::DeviceVec<R, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage =
                    <Self as crate::detail::TransformUnaryOutput<
                        R,
                        Input,
                        KernelOp<R, Op>,
                    >>::run(policy, input)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            fn transform_binary<Left, Right, Op>(
                policy: &crate::detail::CubePolicy<R>,
                left: crate::detail::device::DeviceColumnView<
                    R,
                    Left,
                >,
                right: crate::detail::device::DeviceColumnView<
                    R,
                    Right,
                >,
                op: Op,
            ) -> Result<<Self as MItem<R>>::Inner, Error>
            where
                Left: Scalar,
                Right: Scalar,
                Op: op::UnaryOp<R, (Left, Right), Output = Self>,
                Self: crate::detail::TransformSoA2Output<
                    R,
                    Left,
                    Right,
                    KernelOp<R, Op>,
                >,
                <Self as crate::detail::MItemStorage<
                    R,
                >>::Storage: crate::detail::MaterializeOutput<
                    Runtime = R,
                    Output = ($(
                        crate::detail::DeviceVec<R, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage =
                    <Self as crate::detail::TransformSoA2Output<
                        R,
                        Left,
                        Right,
                        KernelOp<R, Op>,
                    >>::run(policy, left, right)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            fn transform_ternary<First, Second, Third, Op>(
                policy: &crate::detail::CubePolicy<R>,
                first: crate::detail::device::DeviceColumnView<
                    R,
                    First,
                >,
                second: crate::detail::device::DeviceColumnView<
                    R,
                    Second,
                >,
                third: crate::detail::device::DeviceColumnView<
                    R,
                    Third,
                >,
                op: Op,
            ) -> Result<<Self as MItem<R>>::Inner, Error>
            where
                First: Scalar,
                Second: Scalar,
                Third: Scalar,
                Op: op::UnaryOp<R, (First, Second, Third), Output = Self>,
                Self: crate::detail::TransformSoA3Output<
                    R,
                    First,
                    Second,
                    Third,
                    KernelOp<R, Op>,
                >,
                <Self as crate::detail::MItemStorage<
                    R,
                >>::Storage: crate::detail::MaterializeOutput<
                    Runtime = R,
                    Output = ($(
                        crate::detail::DeviceVec<R, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage =
                    <Self as crate::detail::TransformSoA3Output<
                        R,
                        First,
                        Second,
                        Third,
                        KernelOp<R, Op>,
                    >>::run(
                        policy,
                        first,
                        second,
                        third,
                )?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            fn reduce_inner<Op>(
                policy: &crate::detail::CubePolicy<R>,
                input: <Self as MItem<R>>::Inner,
                init: Self,
                op: Op,
            ) -> Result<Self, Error>
            where
                Op: op::ReductionOp<R, Self>,
            {
                let _ = op;
                crate::detail::reduce(policy, input, init, KernelOp::<R, Op>::new())
            }

            fn inner_product_with_right_item<
                LeftIter,
                RightIter,
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
                LeftIter: MIter<R, Item = Self, Inner = <Self as MItem<R>>::View>,
                RightIter: MIter<
                    R,
                    Inner = <<RightIter as MIter<R>>::Item as MItem<R>>::View,
                >,
                TransformOp:
                    op::BinaryOp<R, Self, <RightIter as MIter<R>>::Item, Output = Output>,
                Output: MItem<R>,
                ReduceOp: op::ReductionOp<R, Output>,
            {
                inner_product_left_item_body!(
                    R;
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
                policy: &crate::detail::CubePolicy<R>,
                left: LeftIter,
                right: RightIter,
                transform_op: TransformOp,
                init: Output,
                reduce_op: ReduceOp,
            ) -> Result<Output, Error>
            where
                LeftScalar: Scalar + 'static,
                (LeftScalar,):
                    MItem<R, View = (crate::detail::device::DeviceColumnView<R, LeftScalar>,)>,
                LeftIter:
                    MIter<R, Item = (LeftScalar,), Inner = <(LeftScalar,) as MItem<R>>::View>,
                RightIter: MIter<R, Item = Self, Inner = <Self as MItem<R>>::View>,
                TransformOp: op::BinaryOp<R, (LeftScalar,), Self, Output = Output>,
                Output: MItem<R>,
                ReduceOp: op::ReductionOp<R, Output>,
            {
                inner_product_right_item_body!(
                    R;
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
        impl<R, $( $ty ),+> MItem<R> for ($( $ty, )+)
        where
            R: Runtime,
            $( $ty: Scalar, )+
        {
            type Inner = ($( crate::detail::DeviceVec<R, $ty>, )+);
            type View = ($( crate::detail::device::DeviceColumnView<R, $ty>, )+);
            type Vec = ($( DeviceVec<R, $ty>, )+);

            fn vec_from_inner(inner: Self::Inner) -> Self::Vec {
                let ($( $var, )+) = inner;
                ($( DeviceVec::from_inner($var), )+)
            }
        }

        impl<R, $( $ty ),+> MVec<R> for ($( DeviceVec<R, $ty>, )+)
        where
            R: Runtime,
            $( $ty: Scalar, )+
        {
            type Item = ($( $ty, )+);

            fn from_inner(inner: <Self::Item as MItem<R>>::Inner) -> Self {
                <Self::Item as MItem<R>>::vec_from_inner(inner)
            }
        }

        impl<R, $( $ty ),+> sealed::MItemDispatch<R> for ($( $ty, )+)
        where
            R: Runtime,
            $( $ty: Scalar, )+
        {
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
        impl<'a, R, $( $ty ),+> MIterMut<R> for ($( DeviceSliceMut<'a, R, $ty>, )+)
        where
            R: Runtime,
            $( $ty: Scalar + 'static, )+
            ($( $ty, )+): MItem<R, Inner = ($( crate::detail::DeviceVec<R, $ty>, )+)>,
        {
            type Item = ($( $ty, )+);
            type Inner = ($( crate::detail::device::DeviceColumnMutView<R, $ty>, )+);

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
                policy: &crate::detail::CubePolicy<R>,
                inner: <Self::Item as MItem<R>>::Inner,
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
                policy: &crate::detail::CubePolicy<R>,
                inner: <Self::Item as MItem<R>>::Inner,
                stencil: crate::detail::api::PrecomputedSelection<R>,
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
                            KernelOp::<R, StencilFlag>::new(),
                        )?;
                    }
                )+
                Ok(())
            }

            fn replace_where_inner(
                self,
                policy: &crate::detail::CubePolicy<R>,
                replacement: Self::Item,
                stencil: crate::detail::api::PrecomputedSelection<R>,
            ) -> Result<(), Error>
            {
                let output = self.into_inner();
                $(
                    crate::detail::api::replace_where_into_with_control(
                        policy,
                        replacement.$idx,
                        stencil.control(),
                        &output.$idx,
                    )?;
                )+
                Ok(())
            }

            fn len(&self) -> usize {
                self.0.len()
            }
        }

        impl<'a, R, $( $ty ),+> sealed::MIterMutDispatch<R> for ($( DeviceSliceMut<'a, R, $ty>, )+)
        where
            R: Runtime,
            $( $ty: Scalar + 'static, )+
            ($( $ty, )+): MItem<R, Inner = ($( crate::detail::DeviceVec<R, $ty>, )+)>,
        {
            fn validate_executor(&self, exec: &Executor<R>) -> Result<(), Error> {
                $(
                    exec.ensure_policy_id(self.$idx.source.inner.policy_id())?;
                    ensure_same_len(self.$idx.len(), self.0.len())?;
                )+
                Ok(())
            }

            fn column_mut_view_by_index_inner<U: 'static>(
                &self,
                index: usize,
            ) -> Result<Option<crate::detail::device::DeviceColumnMutView<R, U>>, Error>
            where
                U: Scalar,
            {
                $(
                    if index == $idx {
                        let source = &*self.$idx.source as &dyn Any;
                        let source = match source.downcast_ref::<DeviceVec<R, U>>() {
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
