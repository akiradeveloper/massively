use super::*;

pub(crate) trait KernelReverseInput: Sized {
    type Runtime: Runtime;
    type Output;

    fn reverse_read(self, policy: &CubePolicy<Self::Runtime>) -> Result<Self::Output, Error>;
}

pub(crate) trait KernelSortInput<Less>: Sized {
    type Runtime: Runtime;
    type Output;

    fn sort_read(self, policy: &CubePolicy<Self::Runtime>) -> Result<Self::Output, Error>;
}

#[allow(dead_code)]

pub(crate) trait KernelPairOrderingInput<Other, Less>: Sized {
    type Runtime: Runtime;
    type Output;

    fn merge_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        other: Other,
        less: GpuOp<Less>,
    ) -> Result<Self::Output, Error>;

    fn set_union_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        other: Other,
        less: GpuOp<Less>,
    ) -> Result<Self::Output, Error>;

    fn set_intersection_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        other: Other,
        less: GpuOp<Less>,
    ) -> Result<Self::Output, Error>;

    fn set_difference_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        other: Other,
        less: GpuOp<Less>,
    ) -> Result<Self::Output, Error>;
}

impl<S> KernelReverseInput for S
where
    S: KernelColumn + KernelColumnAt<S0>,
    S::Item: Scalar + 'static,
    S::Expr: DeviceGpuExpr<S::Item>,
{
    type Runtime = S::Runtime;
    type Output = DeviceSoA1<DeviceVec<S::Runtime, S::Item>>;

    fn reverse_read(self, policy: &CubePolicy<Self::Runtime>) -> Result<Self::Output, Error> {
        Ok(DeviceSoA1 {
            source: crate::detail::api::device_expr_reverse_collect(policy, &self)?,
        })
    }
}

macro_rules! impl_kernel_reverse_tuple1 {
    ($target:ty, $field:tt) => {
        impl<S> KernelReverseInput for $target
        where
            S: KernelColumn + KernelColumnAt<S0>,
            S::Item: Scalar + 'static,
            S::Expr: DeviceGpuExpr<S::Item>,
        {
            type Runtime = S::Runtime;
            type Output = DeviceSoA1<DeviceVec<S::Runtime, S::Item>>;

            fn reverse_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
            ) -> Result<Self::Output, Error> {
                Ok(DeviceSoA1 {
                    source: crate::detail::api::device_expr_reverse_collect(policy, &self.$field)?,
                })
            }
        }
    };
}

impl_kernel_reverse_tuple1!((S,), 0);
impl_kernel_reverse_tuple1!(SoAView1<S>, source);
impl_kernel_reverse_tuple1!(DeviceSoA1<S>, source);

macro_rules! impl_kernel_reverse_tuple2 {
    ($target:ty, $left:tt, $right:tt) => {
        impl<A, C> KernelReverseInput for $target
        where
            A: KernelColumn + KernelColumnAt<S0>,
            C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
            A::Item: Scalar + 'static,
            C::Item: Scalar + 'static,
            A::Expr: DeviceGpuExpr<A::Item>,
            C::Expr: DeviceGpuExpr<C::Item>,
        {
            type Runtime = A::Runtime;
            type Output =
                DeviceSoA2<DeviceVec<A::Runtime, A::Item>, DeviceVec<A::Runtime, C::Item>>;

            fn reverse_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
            ) -> Result<Self::Output, Error> {
                validate_columns2(&self.$left, &self.$right)?;
                Ok(DeviceSoA2 {
                    left: crate::detail::api::device_expr_reverse_collect(policy, &self.$left)?,
                    right: crate::detail::api::device_expr_reverse_collect(policy, &self.$right)?,
                })
            }
        }
    };
}

impl_kernel_reverse_tuple2!(SoAView2<A, C>, left, right);
impl_kernel_reverse_tuple2!(DeviceSoA2<A, C>, left, right);

impl<Left, Right> KernelReverseInput for (Left, Right)
where
    SoAView2<Left, Right>: KernelReverseInput,
{
    type Runtime = <SoAView2<Left, Right> as KernelReverseInput>::Runtime;
    type Output = <SoAView2<Left, Right> as KernelReverseInput>::Output;

    fn reverse_read(self, policy: &CubePolicy<Self::Runtime>) -> Result<Self::Output, Error> {
        <SoAView2<Left, Right> as KernelReverseInput>::reverse_read(
            SoAView2 {
                left: self.0,
                right: self.1,
            },
            policy,
        )
    }
}

macro_rules! impl_kernel_reverse_tuple3 {
    ($target:ty, $first:tt, $second:tt, $third:tt) => {
        impl<A, C, D> KernelReverseInput for $target
        where
            A: KernelColumn + KernelColumnAt<S0>,
            C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
            D: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
            A::Item: Scalar + 'static,
            C::Item: Scalar + 'static,
            D::Item: Scalar + 'static,
            A::Expr: DeviceGpuExpr<A::Item>,
            C::Expr: DeviceGpuExpr<C::Item>,
            D::Expr: DeviceGpuExpr<D::Item>,
        {
            type Runtime = A::Runtime;
            type Output = DeviceSoA3<
                DeviceVec<A::Runtime, A::Item>,
                DeviceVec<A::Runtime, C::Item>,
                DeviceVec<A::Runtime, D::Item>,
            >;

            fn reverse_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
            ) -> Result<Self::Output, Error> {
                validate_columns3(&self.$first, &self.$second, &self.$third)?;
                Ok(DeviceSoA3 {
                    first: crate::detail::api::device_expr_reverse_collect(policy, &self.$first)?,
                    second: crate::detail::api::device_expr_reverse_collect(policy, &self.$second)?,
                    third: crate::detail::api::device_expr_reverse_collect(policy, &self.$third)?,
                })
            }
        }
    };
}

impl_kernel_reverse_tuple3!(SoAView3<A, C, D>, first, second, third);
impl_kernel_reverse_tuple3!(DeviceSoA3<A, C, D>, first, second, third);

impl<First, Second, Third> KernelReverseInput for (First, Second, Third)
where
    SoAView3<First, Second, Third>: KernelReverseInput,
{
    type Runtime = <SoAView3<First, Second, Third> as KernelReverseInput>::Runtime;
    type Output = <SoAView3<First, Second, Third> as KernelReverseInput>::Output;

    fn reverse_read(self, policy: &CubePolicy<Self::Runtime>) -> Result<Self::Output, Error> {
        <SoAView3<First, Second, Third> as KernelReverseInput>::reverse_read(
            SoAView3 {
                first: self.0,
                second: self.1,
                third: self.2,
            },
            policy,
        )
    }
}

impl<Source, Less> KernelSortInput<Less> for Source
where
    Source: ReadOnlyKernelColumn + KernelColumnAt<S0>,
    Source::Item: Scalar + 'static,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Less: BinaryPredicateOp<Source::Item>,
{
    type Runtime = Source::Runtime;
    type Output = DeviceSoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn sort_read(self, policy: &CubePolicy<Self::Runtime>) -> Result<Self::Output, Error> {
        Ok(DeviceSoA1 {
            source: primitive_ordering::sort_input_with_policy(
                policy,
                &self,
                crate::op::GpuOp::<Less>::new(),
            )?,
        })
    }
}

macro_rules! impl_kernel_sort_tuple1 {
    ($target:ty, $field:tt) => {
        impl<Source, Less> KernelSortInput<Less> for $target
        where
            Source: ReadOnlyKernelColumn + KernelColumnAt<S0>,
            Source::Item: Scalar + 'static,
            Source::Expr: DeviceGpuExpr<Source::Item>,
            Less: BinaryPredicateOp<Source::Item>,
        {
            type Runtime = Source::Runtime;
            type Output = DeviceSoA1<DeviceVec<Source::Runtime, Source::Item>>;

            fn sort_read(self, policy: &CubePolicy<Self::Runtime>) -> Result<Self::Output, Error> {
                Ok(DeviceSoA1 {
                    source: primitive_ordering::sort_input_with_policy(
                        policy,
                        &self.$field,
                        crate::op::GpuOp::<Less>::new(),
                    )?,
                })
            }
        }
    };
}

impl_kernel_sort_tuple1!(SoAView1<Source>, source);
impl_kernel_sort_tuple1!(DeviceSoA1<Source>, source);

impl<Source, Less> KernelSortInput<Less> for (Source,)
where
    Source: ReadOnlyKernelColumn + KernelColumnAt<S0>,
    Source::Item: Scalar + 'static,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Less: BinaryPredicateOp<(Source::Item,)>,
{
    type Runtime = Source::Runtime;
    type Output = DeviceSoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn sort_read(self, policy: &CubePolicy<Self::Runtime>) -> Result<Self::Output, Error> {
        <DeviceSoA1<Source> as KernelSortInput<crate::detail::api::Tuple1Less<Less>>>::sort_read(
            DeviceSoA1 { source: self.0 },
            policy,
        )
    }
}

macro_rules! impl_kernel_sort_tuple2 {
    ($target:ty, $left:tt, $right:tt) => {
        impl<Left, Right, Less> KernelSortInput<Less> for $target
        where
            Left: KernelColumn + KernelColumnAt<S0>,
            Right: KernelColumn<Runtime = Left::Runtime> + KernelColumnAt<S0>,
            Left::Item: Scalar + 'static,
            Right::Item: Scalar + 'static,
            Left::Expr: DeviceGpuExpr<Left::Item>,
            Right::Expr: DeviceGpuExpr<Right::Item>,
            Less: BinaryPredicateOp<(Left::Item, Right::Item)>,
        {
            type Runtime = Left::Runtime;
            type Output = DeviceSoA2<
                DeviceVec<Left::Runtime, Left::Item>,
                DeviceVec<Left::Runtime, Right::Item>,
            >;

            fn sort_read(self, policy: &CubePolicy<Self::Runtime>) -> Result<Self::Output, Error> {
                let (left, right) = primitive_ordering::sort_tuple2_input(
                    policy,
                    &self.$left,
                    &self.$right,
                    crate::op::GpuOp::<Less>::new(),
                )?;
                Ok(DeviceSoA2 { left, right })
            }
        }
    };
}

impl_kernel_sort_tuple2!(SoAView2<Left, Right>, left, right);
impl_kernel_sort_tuple2!(DeviceSoA2<Left, Right>, left, right);

impl<Left, Right, Less> KernelSortInput<Less> for (Left, Right)
where
    SoAView2<Left, Right>: KernelSortInput<Less>,
{
    type Runtime = <SoAView2<Left, Right> as KernelSortInput<Less>>::Runtime;
    type Output = <SoAView2<Left, Right> as KernelSortInput<Less>>::Output;

    fn sort_read(self, policy: &CubePolicy<Self::Runtime>) -> Result<Self::Output, Error> {
        <SoAView2<Left, Right> as KernelSortInput<Less>>::sort_read(
            SoAView2 {
                left: self.0,
                right: self.1,
            },
            policy,
        )
    }
}

macro_rules! impl_kernel_sort_tuple3 {
    ($target:ty, $first:tt, $second:tt, $third:tt) => {
        impl<First, Second, Third, Less> KernelSortInput<Less> for $target
        where
            First: KernelColumn + KernelColumnAt<S0>,
            Second: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
            Third: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
            First::Item: Scalar + 'static,
            Second::Item: Scalar + 'static,
            Third::Item: Scalar + 'static,
            First::Expr: DeviceGpuExpr<First::Item>,
            Second::Expr: DeviceGpuExpr<Second::Item>,
            Third::Expr: DeviceGpuExpr<Third::Item>,
            Less: BinaryPredicateOp<(First::Item, Second::Item, Third::Item)>,
        {
            type Runtime = First::Runtime;
            type Output = DeviceSoA3<
                DeviceVec<First::Runtime, First::Item>,
                DeviceVec<First::Runtime, Second::Item>,
                DeviceVec<First::Runtime, Third::Item>,
            >;

            fn sort_read(self, policy: &CubePolicy<Self::Runtime>) -> Result<Self::Output, Error> {
                let (first, second, third) = primitive_ordering::sort_tuple3_input(
                    policy,
                    &self.$first,
                    &self.$second,
                    &self.$third,
                    crate::op::GpuOp::<Less>::new(),
                )?;
                Ok(DeviceSoA3 {
                    first,
                    second,
                    third,
                })
            }
        }
    };
}

impl_kernel_sort_tuple3!(SoAView3<First, Second, Third>, first, second, third);
impl_kernel_sort_tuple3!(DeviceSoA3<First, Second, Third>, first, second, third);

impl<First, Second, Third, Less> KernelSortInput<Less> for (First, Second, Third)
where
    SoAView3<First, Second, Third>: KernelSortInput<Less>,
{
    type Runtime = <SoAView3<First, Second, Third> as KernelSortInput<Less>>::Runtime;
    type Output = <SoAView3<First, Second, Third> as KernelSortInput<Less>>::Output;

    fn sort_read(self, policy: &CubePolicy<Self::Runtime>) -> Result<Self::Output, Error> {
        <SoAView3<First, Second, Third> as KernelSortInput<Less>>::sort_read(
            SoAView3 {
                first: self.0,
                second: self.1,
                third: self.2,
            },
            policy,
        )
    }
}
