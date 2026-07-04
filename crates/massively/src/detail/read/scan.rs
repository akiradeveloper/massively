use super::*;

pub(crate) trait KernelInclusiveScanInput<Op>: Sized {
    type Runtime: Runtime;
    type Output;

    fn inclusive_scan_read(self, policy: &CubePolicy<Self::Runtime>)
    -> Result<Self::Output, Error>;
}

pub(crate) trait KernelExclusiveScanInput<Op>: Sized {
    type Runtime: Runtime;
    type Init;
    type Output;

    fn exclusive_scan_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        init: Self::Init,
    ) -> Result<Self::Output, Error>;
}

pub(crate) trait KernelAdjacentDifferenceInput<Op>: Sized {
    type Runtime: Runtime;
    type Output;

    fn adjacent_difference_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
    ) -> Result<Self::Output, Error>;
}

struct LinearScanApply;

impl LinearScanApply {
    fn inclusive_expr1<R, T, Expr, Op>(
        policy: &CubePolicy<R>,
        input: &KernelColumnBindings,
        len: usize,
    ) -> Result<DeviceSoA1<DeviceVec<R, T>>, Error>
    where
        R: Runtime,
        T: Scalar + 'static,
        Expr: DeviceGpuExpr<T>,
        (T,): MItem<R>,
        Op: BinaryOp<(T,)>,
    {
        primitive_scan::inclusive_scan_tuple1_device_expr::<R, T, Expr, Op>(policy, input, len)
    }

    fn exclusive_expr1<R, T, Expr, Op>(
        policy: &CubePolicy<R>,
        input: &KernelColumnBindings,
        len: usize,
        init: (T,),
    ) -> Result<DeviceSoA1<DeviceVec<R, T>>, Error>
    where
        R: Runtime,
        T: Scalar + 'static,
        Expr: DeviceGpuExpr<T>,
        (T,): MItem<R>,
        Op: BinaryOp<(T,)>,
    {
        primitive_scan::exclusive_scan_tuple1_device_expr::<R, T, Expr, Op>(
            policy, input, len, init,
        )
    }

    fn adjacent_expr1<Source, Op>(
        policy: &CubePolicy<Source::Runtime>,
        source: &Source,
    ) -> Result<DeviceSoA1<DeviceVec<Source::Runtime, Source::Item>>, Error>
    where
        Source: KernelColumn + KernelColumnAt<S0>,
        Source::Item: Scalar + 'static,
        Source::Expr: DeviceGpuExpr<Source::Item>,
        Op: BinaryOp<Source::Item>,
    {
        let source = crate::detail::api::device_expr_adjacent_difference_with_policy::<Source, Op>(
            policy, source,
        )?;
        Ok(DeviceSoA1 { source })
    }

    fn inclusive_expr2<R, A, C, AExpr, CExpr, Op>(
        policy: &CubePolicy<R>,
        left: &KernelColumnBindings,
        right: &KernelColumnBindings,
        len: usize,
    ) -> Result<DeviceSoA2<DeviceVec<R, A>, DeviceVec<R, C>>, Error>
    where
        R: Runtime,
        A: Scalar + 'static,
        C: Scalar + 'static,
        AExpr: DeviceGpuExpr<A>,
        CExpr: DeviceGpuExpr<C>,
        (A, C): MItem<R>,
        Op: BinaryOp<(A, C)>,
    {
        primitive_scan::inclusive_scan_tuple2_device_expr::<R, A, C, AExpr, CExpr, Op>(
            policy, left, right, len,
        )
    }

    fn exclusive_expr2<R, A, C, AExpr, CExpr, Op>(
        policy: &CubePolicy<R>,
        left: &KernelColumnBindings,
        right: &KernelColumnBindings,
        len: usize,
        init: (A, C),
    ) -> Result<DeviceSoA2<DeviceVec<R, A>, DeviceVec<R, C>>, Error>
    where
        R: Runtime,
        A: Scalar + 'static,
        C: Scalar + 'static,
        AExpr: DeviceGpuExpr<A>,
        CExpr: DeviceGpuExpr<C>,
        (A, C): MItem<R>,
        Op: BinaryOp<(A, C)>,
    {
        primitive_scan::exclusive_scan_tuple2_device_expr::<R, A, C, AExpr, CExpr, Op>(
            policy, left, right, len, init,
        )
    }

    fn adjacent_expr2<R, A, C, AExpr, CExpr, Op>(
        policy: &CubePolicy<R>,
        left: &KernelColumnBindings,
        right: &KernelColumnBindings,
        len: usize,
    ) -> Result<DeviceSoA2<DeviceVec<R, A>, DeviceVec<R, C>>, Error>
    where
        R: Runtime,
        A: Scalar + 'static,
        C: Scalar + 'static,
        AExpr: DeviceGpuExpr<A>,
        CExpr: DeviceGpuExpr<C>,
        (A, C): MItem<R>,
        Op: BinaryOp<(A, C)>,
    {
        primitive_scan::adjacent_difference_tuple2_device_expr::<R, A, C, AExpr, CExpr, Op>(
            policy, left, right, len,
        )
    }

    fn inclusive_expr3<R, A, C, D, AExpr, CExpr, DExpr, Op>(
        policy: &CubePolicy<R>,
        first: &KernelColumnBindings,
        second: &KernelColumnBindings,
        third: &KernelColumnBindings,
        len: usize,
    ) -> Result<DeviceSoA3<DeviceVec<R, A>, DeviceVec<R, C>, DeviceVec<R, D>>, Error>
    where
        R: Runtime,
        A: Scalar + 'static,
        C: Scalar + 'static,
        D: Scalar + 'static,
        AExpr: DeviceGpuExpr<A>,
        CExpr: DeviceGpuExpr<C>,
        DExpr: DeviceGpuExpr<D>,
        (A, C, D): MItem<R>,
        Op: BinaryOp<(A, C, D)>,
    {
        primitive_scan::inclusive_scan_tuple3_device_expr::<R, A, C, D, AExpr, CExpr, DExpr, Op>(
            policy, first, second, third, len,
        )
    }

    fn exclusive_expr3<R, A, C, D, AExpr, CExpr, DExpr, Op>(
        policy: &CubePolicy<R>,
        first: &KernelColumnBindings,
        second: &KernelColumnBindings,
        third: &KernelColumnBindings,
        len: usize,
        init: (A, C, D),
    ) -> Result<DeviceSoA3<DeviceVec<R, A>, DeviceVec<R, C>, DeviceVec<R, D>>, Error>
    where
        R: Runtime,
        A: Scalar + 'static,
        C: Scalar + 'static,
        D: Scalar + 'static,
        AExpr: DeviceGpuExpr<A>,
        CExpr: DeviceGpuExpr<C>,
        DExpr: DeviceGpuExpr<D>,
        (A, C, D): MItem<R>,
        Op: BinaryOp<(A, C, D)>,
    {
        primitive_scan::exclusive_scan_tuple3_device_expr::<R, A, C, D, AExpr, CExpr, DExpr, Op>(
            policy, first, second, third, len, init,
        )
    }

    fn adjacent_expr3<R, A, C, D, AExpr, CExpr, DExpr, Op>(
        policy: &CubePolicy<R>,
        first: &KernelColumnBindings,
        second: &KernelColumnBindings,
        third: &KernelColumnBindings,
        len: usize,
    ) -> Result<DeviceSoA3<DeviceVec<R, A>, DeviceVec<R, C>, DeviceVec<R, D>>, Error>
    where
        R: Runtime,
        A: Scalar + 'static,
        C: Scalar + 'static,
        D: Scalar + 'static,
        AExpr: DeviceGpuExpr<A>,
        CExpr: DeviceGpuExpr<C>,
        DExpr: DeviceGpuExpr<D>,
        (A, C, D): MItem<R>,
        Op: BinaryOp<(A, C, D)>,
    {
        primitive_scan::adjacent_difference_tuple3_device_expr::<R, A, C, D, AExpr, CExpr, DExpr, Op>(
            policy, first, second, third, len,
        )
    }
}

macro_rules! impl_kernel_inclusive_scan_tuple1 {
    ($target:ty, $field:tt) => {
        impl<S, Op> KernelInclusiveScanInput<Op> for $target
        where
            S: KernelColumn + KernelColumnAt<S0>,
            S::Item: Scalar + 'static,
            S::Expr: DeviceGpuExpr<S::Item>,
            (S::Item,): MItem<S::Runtime>,
            Op: BinaryOp<(S::Item,)>,
        {
            type Runtime = S::Runtime;
            type Output = DeviceSoA1<DeviceVec<S::Runtime, S::Item>>;

            fn inclusive_scan_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
            ) -> Result<Self::Output, Error> {
                let len = <S as KernelColumn>::len(&self.$field);
                let bindings = <S as KernelColumn>::stage(&self.$field, policy)?;
                LinearScanApply::inclusive_expr1::<S::Runtime, S::Item, S::Expr, Op>(
                    policy, &bindings, len,
                )
            }
        }
    };
}

impl_kernel_inclusive_scan_tuple1!((S,), 0);
impl_kernel_inclusive_scan_tuple1!(SoAView1<S>, source);
impl_kernel_inclusive_scan_tuple1!(DeviceSoA1<S>, source);

macro_rules! impl_kernel_exclusive_scan_tuple1 {
    ($target:ty, $field:tt) => {
        impl<S, Op> KernelExclusiveScanInput<Op> for $target
        where
            S: KernelColumn + KernelColumnAt<S0>,
            S::Item: Scalar + 'static,
            S::Expr: DeviceGpuExpr<S::Item>,
            (S::Item,): MItem<S::Runtime>,
            Op: BinaryOp<(S::Item,)>,
        {
            type Runtime = S::Runtime;
            type Init = (S::Item,);
            type Output = DeviceSoA1<DeviceVec<S::Runtime, S::Item>>;

            fn exclusive_scan_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
                init: Self::Init,
            ) -> Result<Self::Output, Error> {
                let len = <S as KernelColumn>::len(&self.$field);
                let bindings = <S as KernelColumn>::stage(&self.$field, policy)?;
                LinearScanApply::exclusive_expr1::<S::Runtime, S::Item, S::Expr, Op>(
                    policy, &bindings, len, init,
                )
            }
        }
    };
}

impl_kernel_exclusive_scan_tuple1!((S,), 0);
impl_kernel_exclusive_scan_tuple1!(SoAView1<S>, source);
impl_kernel_exclusive_scan_tuple1!(DeviceSoA1<S>, source);

impl<S, Op> KernelAdjacentDifferenceInput<Op> for S
where
    S: KernelColumn + KernelColumnAt<S0>,
    S::Item: Scalar + 'static,
    S::Expr: DeviceGpuExpr<S::Item>,
    Op: BinaryOp<S::Item>,
{
    type Runtime = S::Runtime;
    type Output = DeviceSoA1<DeviceVec<S::Runtime, S::Item>>;

    fn adjacent_difference_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
    ) -> Result<Self::Output, Error> {
        LinearScanApply::adjacent_expr1::<S, Op>(policy, &self)
    }
}

macro_rules! impl_kernel_adjacent_difference_tuple1 {
    ($target:ty, $field:tt) => {
        impl<S, Op> KernelAdjacentDifferenceInput<Op> for $target
        where
            S: KernelColumn + KernelColumnAt<S0>,
            S::Item: Scalar + 'static,
            S::Expr: DeviceGpuExpr<S::Item>,
            (S::Item,): MItem<S::Runtime>,
            Op: BinaryOp<(S::Item,)>,
        {
            type Runtime = S::Runtime;
            type Output = DeviceSoA1<DeviceVec<S::Runtime, S::Item>>;

            fn adjacent_difference_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
            ) -> Result<Self::Output, Error> {
                LinearScanApply::adjacent_expr1::<S, crate::detail::api::Tuple1BinaryOp<Op>>(
                    policy,
                    &self.$field,
                )
            }
        }
    };
}

impl_kernel_adjacent_difference_tuple1!((S,), 0);
impl_kernel_adjacent_difference_tuple1!(SoAView1<S>, source);
impl_kernel_adjacent_difference_tuple1!(DeviceSoA1<S>, source);

macro_rules! impl_kernel_scan_tuple2 {
    ($target:ty, $left:tt, $right:tt) => {
        impl<A, C, Op> KernelInclusiveScanInput<Op> for $target
        where
            A: KernelColumn + KernelColumnAt<S0>,
            C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
            A::Item: Scalar + 'static,
            C::Item: Scalar + 'static,
            A::Expr: DeviceGpuExpr<A::Item>,
            C::Expr: DeviceGpuExpr<C::Item>,
            (A::Item, C::Item): MItem<A::Runtime>,
            Op: BinaryOp<(A::Item, C::Item)>,
        {
            type Runtime = A::Runtime;
            type Output =
                DeviceSoA2<DeviceVec<A::Runtime, A::Item>, DeviceVec<A::Runtime, C::Item>>;

            fn inclusive_scan_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
            ) -> Result<Self::Output, Error> {
                validate_columns2(&self.$left, &self.$right)?;
                let left = <A as KernelColumn>::stage(&self.$left, policy)?;
                let right = <C as KernelColumn>::stage(&self.$right, policy)?;
                LinearScanApply::inclusive_expr2::<
                    A::Runtime,
                    A::Item,
                    C::Item,
                    A::Expr,
                    C::Expr,
                    Op,
                >(policy, &left, &right, <A as KernelColumn>::len(&self.$left))
            }
        }

        impl<A, C, Op> KernelExclusiveScanInput<Op> for $target
        where
            A: KernelColumn + KernelColumnAt<S0>,
            C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
            A::Item: Scalar + 'static,
            C::Item: Scalar + 'static,
            A::Expr: DeviceGpuExpr<A::Item>,
            C::Expr: DeviceGpuExpr<C::Item>,
            (A::Item, C::Item): MItem<A::Runtime>,
            Op: BinaryOp<(A::Item, C::Item)>,
        {
            type Runtime = A::Runtime;
            type Init = (A::Item, C::Item);
            type Output =
                DeviceSoA2<DeviceVec<A::Runtime, A::Item>, DeviceVec<A::Runtime, C::Item>>;

            fn exclusive_scan_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
                init: Self::Init,
            ) -> Result<Self::Output, Error> {
                validate_columns2(&self.$left, &self.$right)?;
                let left = <A as KernelColumn>::stage(&self.$left, policy)?;
                let right = <C as KernelColumn>::stage(&self.$right, policy)?;
                LinearScanApply::exclusive_expr2::<
                    A::Runtime,
                    A::Item,
                    C::Item,
                    A::Expr,
                    C::Expr,
                    Op,
                >(
                    policy,
                    &left,
                    &right,
                    <A as KernelColumn>::len(&self.$left),
                    init,
                )
            }
        }

        impl<A, C, Op> KernelAdjacentDifferenceInput<Op> for $target
        where
            A: KernelColumn + KernelColumnAt<S0>,
            C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
            A::Item: Scalar + 'static,
            C::Item: Scalar + 'static,
            A::Expr: DeviceGpuExpr<A::Item>,
            C::Expr: DeviceGpuExpr<C::Item>,
            (A::Item, C::Item): MItem<A::Runtime>,
            Op: BinaryOp<(A::Item, C::Item)>,
        {
            type Runtime = A::Runtime;
            type Output =
                DeviceSoA2<DeviceVec<A::Runtime, A::Item>, DeviceVec<A::Runtime, C::Item>>;

            fn adjacent_difference_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
            ) -> Result<Self::Output, Error> {
                validate_columns2(&self.$left, &self.$right)?;
                let left = <A as KernelColumn>::stage(&self.$left, policy)?;
                let right = <C as KernelColumn>::stage(&self.$right, policy)?;
                LinearScanApply::adjacent_expr2::<
                    A::Runtime,
                    A::Item,
                    C::Item,
                    A::Expr,
                    C::Expr,
                    Op,
                >(policy, &left, &right, <A as KernelColumn>::len(&self.$left))
            }
        }
    };
}

impl_kernel_scan_tuple2!(SoAView2<A, C>, left, right);
impl_kernel_scan_tuple2!(DeviceSoA2<A, C>, left, right);

macro_rules! impl_kernel_scan_tuple3 {
    ($target:ty, $first:tt, $second:tt, $third:tt) => {
        impl<A, C, D, Op> KernelInclusiveScanInput<Op> for $target
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
            (A::Item, C::Item, D::Item): MItem<A::Runtime>,
            Op: BinaryOp<(A::Item, C::Item, D::Item)>,
        {
            type Runtime = A::Runtime;
            type Output = DeviceSoA3<
                DeviceVec<A::Runtime, A::Item>,
                DeviceVec<A::Runtime, C::Item>,
                DeviceVec<A::Runtime, D::Item>,
            >;

            fn inclusive_scan_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
            ) -> Result<Self::Output, Error> {
                validate_columns3(&self.$first, &self.$second, &self.$third)?;
                let first = <A as KernelColumn>::stage(&self.$first, policy)?;
                let second = <C as KernelColumn>::stage(&self.$second, policy)?;
                let third = <D as KernelColumn>::stage(&self.$third, policy)?;
                LinearScanApply::inclusive_expr3::<
                    A::Runtime,
                    A::Item,
                    C::Item,
                    D::Item,
                    A::Expr,
                    C::Expr,
                    D::Expr,
                    Op,
                >(
                    policy,
                    &first,
                    &second,
                    &third,
                    <A as KernelColumn>::len(&self.$first),
                )
            }
        }

        impl<A, C, D, Op> KernelExclusiveScanInput<Op> for $target
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
            (A::Item, C::Item, D::Item): MItem<A::Runtime>,
            Op: BinaryOp<(A::Item, C::Item, D::Item)>,
        {
            type Runtime = A::Runtime;
            type Init = (A::Item, C::Item, D::Item);
            type Output = DeviceSoA3<
                DeviceVec<A::Runtime, A::Item>,
                DeviceVec<A::Runtime, C::Item>,
                DeviceVec<A::Runtime, D::Item>,
            >;

            fn exclusive_scan_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
                init: Self::Init,
            ) -> Result<Self::Output, Error> {
                validate_columns3(&self.$first, &self.$second, &self.$third)?;
                let first = <A as KernelColumn>::stage(&self.$first, policy)?;
                let second = <C as KernelColumn>::stage(&self.$second, policy)?;
                let third = <D as KernelColumn>::stage(&self.$third, policy)?;
                LinearScanApply::exclusive_expr3::<
                    A::Runtime,
                    A::Item,
                    C::Item,
                    D::Item,
                    A::Expr,
                    C::Expr,
                    D::Expr,
                    Op,
                >(
                    policy,
                    &first,
                    &second,
                    &third,
                    <A as KernelColumn>::len(&self.$first),
                    init,
                )
            }
        }

        impl<A, C, D, Op> KernelAdjacentDifferenceInput<Op> for $target
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
            (A::Item, C::Item, D::Item): MItem<A::Runtime>,
            Op: BinaryOp<(A::Item, C::Item, D::Item)>,
        {
            type Runtime = A::Runtime;
            type Output = DeviceSoA3<
                DeviceVec<A::Runtime, A::Item>,
                DeviceVec<A::Runtime, C::Item>,
                DeviceVec<A::Runtime, D::Item>,
            >;

            fn adjacent_difference_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
            ) -> Result<Self::Output, Error> {
                validate_columns3(&self.$first, &self.$second, &self.$third)?;
                let first = <A as KernelColumn>::stage(&self.$first, policy)?;
                let second = <C as KernelColumn>::stage(&self.$second, policy)?;
                let third = <D as KernelColumn>::stage(&self.$third, policy)?;
                LinearScanApply::adjacent_expr3::<
                    A::Runtime,
                    A::Item,
                    C::Item,
                    D::Item,
                    A::Expr,
                    C::Expr,
                    D::Expr,
                    Op,
                >(
                    policy,
                    &first,
                    &second,
                    &third,
                    <A as KernelColumn>::len(&self.$first),
                )
            }
        }
    };
}

impl_kernel_scan_tuple3!(SoAView3<A, C, D>, first, second, third);
impl_kernel_scan_tuple3!(DeviceSoA3<A, C, D>, first, second, third);

impl<Left, Right, Op> KernelInclusiveScanInput<Op> for (Left, Right)
where
    SoAView2<Left, Right>: KernelInclusiveScanInput<Op>,
{
    type Runtime = <SoAView2<Left, Right> as KernelInclusiveScanInput<Op>>::Runtime;
    type Output = <SoAView2<Left, Right> as KernelInclusiveScanInput<Op>>::Output;

    fn inclusive_scan_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
    ) -> Result<Self::Output, Error> {
        <SoAView2<Left, Right> as KernelInclusiveScanInput<Op>>::inclusive_scan_read(
            SoAView2 {
                left: self.0,
                right: self.1,
            },
            policy,
        )
    }
}

impl<Left, Right, Op> KernelExclusiveScanInput<Op> for (Left, Right)
where
    SoAView2<Left, Right>: KernelExclusiveScanInput<Op>,
{
    type Runtime = <SoAView2<Left, Right> as KernelExclusiveScanInput<Op>>::Runtime;
    type Init = <SoAView2<Left, Right> as KernelExclusiveScanInput<Op>>::Init;
    type Output = <SoAView2<Left, Right> as KernelExclusiveScanInput<Op>>::Output;

    fn exclusive_scan_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        init: Self::Init,
    ) -> Result<Self::Output, Error> {
        <SoAView2<Left, Right> as KernelExclusiveScanInput<Op>>::exclusive_scan_read(
            SoAView2 {
                left: self.0,
                right: self.1,
            },
            policy,
            init,
        )
    }
}

impl<Left, Right, Op> KernelAdjacentDifferenceInput<Op> for (Left, Right)
where
    SoAView2<Left, Right>: KernelAdjacentDifferenceInput<Op>,
{
    type Runtime = <SoAView2<Left, Right> as KernelAdjacentDifferenceInput<Op>>::Runtime;
    type Output = <SoAView2<Left, Right> as KernelAdjacentDifferenceInput<Op>>::Output;

    fn adjacent_difference_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
    ) -> Result<Self::Output, Error> {
        <SoAView2<Left, Right> as KernelAdjacentDifferenceInput<Op>>::adjacent_difference_read(
            SoAView2 {
                left: self.0,
                right: self.1,
            },
            policy,
        )
    }
}

impl<First, Second, Third, Op> KernelInclusiveScanInput<Op> for (First, Second, Third)
where
    SoAView3<First, Second, Third>: KernelInclusiveScanInput<Op>,
{
    type Runtime = <SoAView3<First, Second, Third> as KernelInclusiveScanInput<Op>>::Runtime;
    type Output = <SoAView3<First, Second, Third> as KernelInclusiveScanInput<Op>>::Output;

    fn inclusive_scan_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
    ) -> Result<Self::Output, Error> {
        <SoAView3<First, Second, Third> as KernelInclusiveScanInput<Op>>::inclusive_scan_read(
            SoAView3 {
                first: self.0,
                second: self.1,
                third: self.2,
            },
            policy,
        )
    }
}

impl<First, Second, Third, Op> KernelExclusiveScanInput<Op> for (First, Second, Third)
where
    SoAView3<First, Second, Third>: KernelExclusiveScanInput<Op>,
{
    type Runtime = <SoAView3<First, Second, Third> as KernelExclusiveScanInput<Op>>::Runtime;
    type Init = <SoAView3<First, Second, Third> as KernelExclusiveScanInput<Op>>::Init;
    type Output = <SoAView3<First, Second, Third> as KernelExclusiveScanInput<Op>>::Output;

    fn exclusive_scan_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        init: Self::Init,
    ) -> Result<Self::Output, Error> {
        <SoAView3<First, Second, Third> as KernelExclusiveScanInput<Op>>::exclusive_scan_read(
            SoAView3 {
                first: self.0,
                second: self.1,
                third: self.2,
            },
            policy,
            init,
        )
    }
}

impl<First, Second, Third, Op> KernelAdjacentDifferenceInput<Op> for (First, Second, Third)
where
    SoAView3<First, Second, Third>: KernelAdjacentDifferenceInput<Op>,
{
    type Runtime = <SoAView3<First, Second, Third> as KernelAdjacentDifferenceInput<Op>>::Runtime;
    type Output = <SoAView3<First, Second, Third> as KernelAdjacentDifferenceInput<Op>>::Output;

    fn adjacent_difference_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
    ) -> Result<Self::Output, Error> {
        <SoAView3<First, Second, Third> as KernelAdjacentDifferenceInput<Op>>::adjacent_difference_read(
            SoAView3 {
                first: self.0,
                second: self.1,
                third: self.2,
            },
            policy,
        )
    }
}
