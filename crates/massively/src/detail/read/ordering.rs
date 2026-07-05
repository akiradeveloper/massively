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
    S::Item: MStorageElement + 'static,
    S::Expr: DeviceGpuExpr<S::Item>,
{
    type Runtime = S::Runtime;
    type Output = DeviceZip1<DeviceVec<S::Runtime, S::Item>>;

    fn reverse_read(self, policy: &CubePolicy<Self::Runtime>) -> Result<Self::Output, Error> {
        let control = crate::detail::control::RangeControl::reverse(self.len())?;
        let apply = crate::detail::apply::RangePayloadApply::new(&control);
        Ok(DeviceZip1 {
            source: apply.apply_expr(policy, &self)?,
        })
    }
}

macro_rules! impl_kernel_reverse_tuple1 {
    ($target:ty, $field:tt) => {
        impl<S> KernelReverseInput for $target
        where
            S: KernelColumn + KernelColumnAt<S0>,
            S::Item: MStorageElement + 'static,
            S::Expr: DeviceGpuExpr<S::Item>,
        {
            type Runtime = S::Runtime;
            type Output = DeviceZip1<DeviceVec<S::Runtime, S::Item>>;

            fn reverse_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
            ) -> Result<Self::Output, Error> {
                let control = crate::detail::control::RangeControl::reverse(self.$field.len())?;
                let apply = crate::detail::apply::RangePayloadApply::new(&control);
                Ok(DeviceZip1 {
                    source: apply.apply_expr(policy, &self.$field)?,
                })
            }
        }
    };
}

impl_kernel_reverse_tuple1!((S,), 0);
impl_kernel_reverse_tuple1!(ZipView1<S>, source);
impl_kernel_reverse_tuple1!(DeviceZip1<S>, source);

macro_rules! impl_kernel_reverse_tuple2 {
    ($target:ty, $left:tt, $right:tt) => {
        impl<A, C> KernelReverseInput for $target
        where
            A: KernelColumn + KernelColumnAt<S0>,
            C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
            A::Item: MStorageElement + 'static,
            C::Item: MStorageElement + 'static,
            A::Expr: DeviceGpuExpr<A::Item>,
            C::Expr: DeviceGpuExpr<C::Item>,
        {
            type Runtime = A::Runtime;
            type Output =
                DeviceZip2<DeviceVec<A::Runtime, A::Item>, DeviceVec<A::Runtime, C::Item>>;

            fn reverse_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
            ) -> Result<Self::Output, Error> {
                validate_columns2(&self.$left, &self.$right)?;
                let control = crate::detail::control::RangeControl::reverse(self.$left.len())?;
                let apply = crate::detail::apply::RangePayloadApply::new(&control);
                let (left, right) = apply.apply_expr2(policy, &self.$left, &self.$right)?;
                Ok(DeviceZip2 { left, right })
            }
        }
    };
}

impl_kernel_reverse_tuple2!(ZipView2<A, C>, left, right);
impl_kernel_reverse_tuple2!(DeviceZip2<A, C>, left, right);

impl<Left, Right> KernelReverseInput for (Left, Right)
where
    ZipView2<Left, Right>: KernelReverseInput,
{
    type Runtime = <ZipView2<Left, Right> as KernelReverseInput>::Runtime;
    type Output = <ZipView2<Left, Right> as KernelReverseInput>::Output;

    fn reverse_read(self, policy: &CubePolicy<Self::Runtime>) -> Result<Self::Output, Error> {
        <ZipView2<Left, Right> as KernelReverseInput>::reverse_read(
            ZipView2 {
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
            A::Item: MStorageElement + 'static,
            C::Item: MStorageElement + 'static,
            D::Item: MStorageElement + 'static,
            A::Expr: DeviceGpuExpr<A::Item>,
            C::Expr: DeviceGpuExpr<C::Item>,
            D::Expr: DeviceGpuExpr<D::Item>,
        {
            type Runtime = A::Runtime;
            type Output = DeviceZip3<
                DeviceVec<A::Runtime, A::Item>,
                DeviceVec<A::Runtime, C::Item>,
                DeviceVec<A::Runtime, D::Item>,
            >;

            fn reverse_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
            ) -> Result<Self::Output, Error> {
                validate_columns3(&self.$first, &self.$second, &self.$third)?;
                let control = crate::detail::control::RangeControl::reverse(self.$first.len())?;
                let apply = crate::detail::apply::RangePayloadApply::new(&control);
                let (first, second, third) =
                    apply.apply_expr3(policy, &self.$first, &self.$second, &self.$third)?;
                Ok(DeviceZip3 {
                    first,
                    second,
                    third,
                })
            }
        }
    };
}

impl_kernel_reverse_tuple3!(ZipView3<A, C, D>, first, second, third);
impl_kernel_reverse_tuple3!(DeviceZip3<A, C, D>, first, second, third);

impl<First, Second, Third> KernelReverseInput for (First, Second, Third)
where
    ZipView3<First, Second, Third>: KernelReverseInput,
{
    type Runtime = <ZipView3<First, Second, Third> as KernelReverseInput>::Runtime;
    type Output = <ZipView3<First, Second, Third> as KernelReverseInput>::Output;

    fn reverse_read(self, policy: &CubePolicy<Self::Runtime>) -> Result<Self::Output, Error> {
        <ZipView3<First, Second, Third> as KernelReverseInput>::reverse_read(
            ZipView3 {
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
    Source::Item: MStorageElement + 'static,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Less: BinaryPredicateOp<Source::Item>,
{
    type Runtime = Source::Runtime;
    type Output = DeviceZip1<DeviceVec<Source::Runtime, Source::Item>>;

    fn sort_read(self, policy: &CubePolicy<Self::Runtime>) -> Result<Self::Output, Error> {
        Ok(DeviceZip1 {
            source: crate::detail::apply::SortApply::apply_expr::<Source, Less>(policy, &self)?,
        })
    }
}

macro_rules! impl_kernel_sort_tuple1 {
    ($target:ty, $field:tt) => {
        impl<Source, Less> KernelSortInput<Less> for $target
        where
            Source: ReadOnlyKernelColumn + KernelColumnAt<S0>,
            Source::Item: MStorageElement + 'static,
            Source::Expr: DeviceGpuExpr<Source::Item>,
            Less: BinaryPredicateOp<Source::Item>,
        {
            type Runtime = Source::Runtime;
            type Output = DeviceZip1<DeviceVec<Source::Runtime, Source::Item>>;

            fn sort_read(self, policy: &CubePolicy<Self::Runtime>) -> Result<Self::Output, Error> {
                Ok(DeviceZip1 {
                    source: crate::detail::apply::SortApply::apply_expr::<Source, Less>(
                        policy,
                        &self.$field,
                    )?,
                })
            }
        }
    };
}

impl_kernel_sort_tuple1!(ZipView1<Source>, source);
impl_kernel_sort_tuple1!(DeviceZip1<Source>, source);

impl<Source, Less> KernelSortInput<Less> for (Source,)
where
    Source: ReadOnlyKernelColumn + KernelColumnAt<S0>,
    Source::Item: MStorageElement + 'static,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Less: BinaryPredicateOp<(Source::Item,)>,
{
    type Runtime = Source::Runtime;
    type Output = DeviceZip1<DeviceVec<Source::Runtime, Source::Item>>;

    fn sort_read(self, policy: &CubePolicy<Self::Runtime>) -> Result<Self::Output, Error> {
        <DeviceZip1<Source> as KernelSortInput<crate::detail::api::Tuple1Less<Less>>>::sort_read(
            DeviceZip1 { source: self.0 },
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
            Left::Item: MStorageElement + 'static,
            Right::Item: MStorageElement + 'static,
            Left::Expr: DeviceGpuExpr<Left::Item>,
            Right::Expr: DeviceGpuExpr<Right::Item>,
            Less: BinaryPredicateOp<(Left::Item, Right::Item)>,
        {
            type Runtime = Left::Runtime;
            type Output = DeviceZip2<
                DeviceVec<Left::Runtime, Left::Item>,
                DeviceVec<Left::Runtime, Right::Item>,
            >;

            fn sort_read(self, policy: &CubePolicy<Self::Runtime>) -> Result<Self::Output, Error> {
                let (left, right) = crate::detail::apply::SortApply::apply_expr2::<
                    Left,
                    Right,
                    Less,
                >(policy, &self.$left, &self.$right)?;
                Ok(DeviceZip2 { left, right })
            }
        }
    };
}

impl_kernel_sort_tuple2!(ZipView2<Left, Right>, left, right);
impl_kernel_sort_tuple2!(DeviceZip2<Left, Right>, left, right);

impl<Left, Right, Less> KernelSortInput<Less> for (Left, Right)
where
    ZipView2<Left, Right>: KernelSortInput<Less>,
{
    type Runtime = <ZipView2<Left, Right> as KernelSortInput<Less>>::Runtime;
    type Output = <ZipView2<Left, Right> as KernelSortInput<Less>>::Output;

    fn sort_read(self, policy: &CubePolicy<Self::Runtime>) -> Result<Self::Output, Error> {
        <ZipView2<Left, Right> as KernelSortInput<Less>>::sort_read(
            ZipView2 {
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
            First::Item: MStorageElement + 'static,
            Second::Item: MStorageElement + 'static,
            Third::Item: MStorageElement + 'static,
            First::Expr: DeviceGpuExpr<First::Item>,
            Second::Expr: DeviceGpuExpr<Second::Item>,
            Third::Expr: DeviceGpuExpr<Third::Item>,
            Less: BinaryPredicateOp<(First::Item, Second::Item, Third::Item)>,
        {
            type Runtime = First::Runtime;
            type Output = DeviceZip3<
                DeviceVec<First::Runtime, First::Item>,
                DeviceVec<First::Runtime, Second::Item>,
                DeviceVec<First::Runtime, Third::Item>,
            >;

            fn sort_read(self, policy: &CubePolicy<Self::Runtime>) -> Result<Self::Output, Error> {
                let (first, second, third) =
                    crate::detail::apply::SortApply::apply_expr3::<First, Second, Third, Less>(
                        policy,
                        &self.$first,
                        &self.$second,
                        &self.$third,
                    )?;
                Ok(DeviceZip3 {
                    first,
                    second,
                    third,
                })
            }
        }
    };
}

impl_kernel_sort_tuple3!(ZipView3<First, Second, Third>, first, second, third);
impl_kernel_sort_tuple3!(DeviceZip3<First, Second, Third>, first, second, third);

impl<First, Second, Third, Less> KernelSortInput<Less> for (First, Second, Third)
where
    ZipView3<First, Second, Third>: KernelSortInput<Less>,
{
    type Runtime = <ZipView3<First, Second, Third> as KernelSortInput<Less>>::Runtime;
    type Output = <ZipView3<First, Second, Third> as KernelSortInput<Less>>::Output;

    fn sort_read(self, policy: &CubePolicy<Self::Runtime>) -> Result<Self::Output, Error> {
        <ZipView3<First, Second, Third> as KernelSortInput<Less>>::sort_read(
            ZipView3 {
                first: self.0,
                second: self.1,
                third: self.2,
            },
            policy,
        )
    }
}
