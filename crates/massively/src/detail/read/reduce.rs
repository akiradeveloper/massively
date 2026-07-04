use super::*;

pub(crate) trait KernelReduceInput<Op>: Sized {
    type Runtime: Runtime;
    type Item;

    fn reduce_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        init: Self::Item,
    ) -> Result<Self::Item, Error>;
}

struct LinearReduceApply;

impl LinearReduceApply {
    fn apply_expr1<R, T, Expr, Op>(
        policy: &CubePolicy<R>,
        input: &KernelColumnBindings,
        len: usize,
        init: (T,),
    ) -> Result<(T,), Error>
    where
        R: Runtime,
        T: Scalar + 'static,
        Expr: DeviceGpuExpr<T>,
        (T,): MItem<R>,
        Op: BinaryOp<(T,)>,
    {
        primitive_reduce::reduce_tuple1_device_expr::<R, T, Expr, Op>(policy, input, len, init)
    }

    fn apply_expr2<R, A, C, AExpr, CExpr, Op>(
        policy: &CubePolicy<R>,
        left: &KernelColumnBindings,
        right: &KernelColumnBindings,
        len: usize,
        init: (A, C),
    ) -> Result<(A, C), Error>
    where
        R: Runtime,
        A: Scalar + 'static,
        C: Scalar + 'static,
        AExpr: DeviceGpuExpr<A>,
        CExpr: DeviceGpuExpr<C>,
        (A, C): MItem<R>,
        Op: BinaryOp<(A, C)>,
    {
        primitive_reduce::reduce_tuple2_device_expr::<R, A, C, AExpr, CExpr, Op>(
            policy, left, right, len, init,
        )
    }

    fn apply_expr3<R, A, C, D, AExpr, CExpr, DExpr, Op>(
        policy: &CubePolicy<R>,
        first: &KernelColumnBindings,
        second: &KernelColumnBindings,
        third: &KernelColumnBindings,
        len: usize,
        init: (A, C, D),
    ) -> Result<(A, C, D), Error>
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
        primitive_reduce::reduce_tuple3_device_expr::<R, A, C, D, AExpr, CExpr, DExpr, Op>(
            policy, first, second, third, len, init,
        )
    }
}

macro_rules! impl_kernel_reduce_tuple1 {
    ($target:ty, $field:tt) => {
        impl<S, Op> KernelReduceInput<Op> for $target
        where
            S: KernelColumn + KernelColumnAt<S0>,
            S::Item: Scalar + 'static,
            S::Expr: DeviceGpuExpr<S::Item>,
            (S::Item,): MItem<S::Runtime>,
            Op: BinaryOp<(S::Item,)>,
        {
            type Runtime = S::Runtime;
            type Item = (S::Item,);

            fn reduce_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
                init: Self::Item,
            ) -> Result<Self::Item, Error> {
                let len = <S as KernelColumn>::len(&self.$field);
                let bindings = <S as KernelColumn>::stage(&self.$field, policy)?;
                LinearReduceApply::apply_expr1::<S::Runtime, S::Item, S::Expr, Op>(
                    policy, &bindings, len, init,
                )
            }
        }
    };
}

impl_kernel_reduce_tuple1!((S,), 0);
impl_kernel_reduce_tuple1!(SoAView1<S>, source);
impl_kernel_reduce_tuple1!(DeviceSoA1<S>, source);

macro_rules! impl_kernel_reduce_tuple2 {
    ($target:ty, $left:tt, $right:tt) => {
        impl<A, C, Op> KernelReduceInput<Op> for $target
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
            type Item = (A::Item, C::Item);

            fn reduce_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
                init: Self::Item,
            ) -> Result<Self::Item, Error> {
                <A as KernelColumn>::validate(&self.$left)?;
                <C as KernelColumn>::validate(&self.$right)?;
                ensure_same_len(
                    <C as KernelColumn>::len(&self.$right),
                    <A as KernelColumn>::len(&self.$left),
                )?;
                let left = <A as KernelColumn>::stage(&self.$left, policy)?;
                let right = <C as KernelColumn>::stage(&self.$right, policy)?;
                LinearReduceApply::apply_expr2::<
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
    };
}

impl_kernel_reduce_tuple2!(SoAView2<A, C>, left, right);
impl_kernel_reduce_tuple2!(DeviceSoA2<A, C>, left, right);

impl<Left, Right, Op> KernelReduceInput<Op> for (Left, Right)
where
    SoAView2<Left, Right>: KernelReduceInput<Op>,
{
    type Runtime = <SoAView2<Left, Right> as KernelReduceInput<Op>>::Runtime;
    type Item = <SoAView2<Left, Right> as KernelReduceInput<Op>>::Item;

    fn reduce_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        init: Self::Item,
    ) -> Result<Self::Item, Error> {
        <SoAView2<Left, Right> as KernelReduceInput<Op>>::reduce_read(
            SoAView2 {
                left: self.0,
                right: self.1,
            },
            policy,
            init,
        )
    }
}

macro_rules! impl_kernel_reduce_tuple3 {
    ($target:ty, $first:tt, $second:tt, $third:tt) => {
        impl<A, C, D, Op> KernelReduceInput<Op> for $target
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
            type Item = (A::Item, C::Item, D::Item);

            fn reduce_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
                init: Self::Item,
            ) -> Result<Self::Item, Error> {
                <A as KernelColumn>::validate(&self.$first)?;
                <C as KernelColumn>::validate(&self.$second)?;
                <D as KernelColumn>::validate(&self.$third)?;
                ensure_same_len(
                    <C as KernelColumn>::len(&self.$second),
                    <A as KernelColumn>::len(&self.$first),
                )?;
                ensure_same_len(
                    <D as KernelColumn>::len(&self.$third),
                    <A as KernelColumn>::len(&self.$first),
                )?;
                let first = <A as KernelColumn>::stage(&self.$first, policy)?;
                let second = <C as KernelColumn>::stage(&self.$second, policy)?;
                let third = <D as KernelColumn>::stage(&self.$third, policy)?;
                LinearReduceApply::apply_expr3::<
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
    };
}

impl_kernel_reduce_tuple3!(SoAView3<A, C, D>, first, second, third);
impl_kernel_reduce_tuple3!(DeviceSoA3<A, C, D>, first, second, third);

impl<First, Second, Third, Op> KernelReduceInput<Op> for (First, Second, Third)
where
    SoAView3<First, Second, Third>: KernelReduceInput<Op>,
{
    type Runtime = <SoAView3<First, Second, Third> as KernelReduceInput<Op>>::Runtime;
    type Item = <SoAView3<First, Second, Third> as KernelReduceInput<Op>>::Item;

    fn reduce_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        init: Self::Item,
    ) -> Result<Self::Item, Error> {
        <SoAView3<First, Second, Third> as KernelReduceInput<Op>>::reduce_read(
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
