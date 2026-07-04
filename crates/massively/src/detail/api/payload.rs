use super::*;

pub(in crate::detail) struct SelectedPayloadApply<'a> {
    control: &'a select::SelectedRankControl,
    count: usize,
}

impl<'a> SelectedPayloadApply<'a> {
    pub(in crate::detail) fn new(control: &'a select::SelectedRankControl, count: usize) -> Self {
        Self { control, count }
    }

    pub(in crate::detail) fn apply_expr<ExprSource>(
        &self,
        policy: &crate::policy::CubePolicy<ExprSource::Runtime>,
        expr: &ExprSource,
    ) -> Result<DeviceVec<ExprSource::Runtime, ExprSource::Item>, Error>
    where
        ExprSource: KernelColumn + KernelColumnAt<S0>,
        ExprSource::Runtime: Runtime,
        ExprSource::Item: CubePrimitive + CubeElement,
        ExprSource::Expr: DeviceGpuExpr<ExprSource::Item>,
    {
        expr::selection::device_expr_compact_with_selection_with_policy(
            policy,
            expr,
            self.control,
            self.count,
        )
    }

    pub(in crate::detail) fn apply_expr2<Left, Right>(
        &self,
        policy: &crate::policy::CubePolicy<Left::Runtime>,
        left: &Left,
        right: &Right,
    ) -> Result<
        (
            DeviceVec<Left::Runtime, Left::Item>,
            DeviceVec<Left::Runtime, Right::Item>,
        ),
        Error,
    >
    where
        Left: KernelColumn + KernelColumnAt<S0>,
        Right: KernelColumn<Runtime = Left::Runtime> + KernelColumnAt<S0>,
        Left::Runtime: Runtime,
        Left::Item: CubePrimitive + CubeElement,
        Right::Item: CubePrimitive + CubeElement,
        Left::Expr: DeviceGpuExpr<Left::Item>,
        Right::Expr: DeviceGpuExpr<Right::Item>,
    {
        expr::selection::device_expr_apply_selected2_with_policy(
            policy,
            left,
            right,
            self.control,
            self.count,
        )
    }

    pub(in crate::detail) fn apply_expr3<First, Second, Third>(
        &self,
        policy: &crate::policy::CubePolicy<First::Runtime>,
        first: &First,
        second: &Second,
        third: &Third,
    ) -> Result<
        (
            DeviceVec<First::Runtime, First::Item>,
            DeviceVec<First::Runtime, Second::Item>,
            DeviceVec<First::Runtime, Third::Item>,
        ),
        Error,
    >
    where
        First: KernelColumn + KernelColumnAt<S0>,
        Second: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
        Third: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
        First::Runtime: Runtime,
        First::Item: CubePrimitive + CubeElement,
        Second::Item: CubePrimitive + CubeElement,
        Third::Item: CubePrimitive + CubeElement,
        First::Expr: DeviceGpuExpr<First::Item>,
        Second::Expr: DeviceGpuExpr<Second::Item>,
        Third::Expr: DeviceGpuExpr<Third::Item>,
    {
        expr::selection::device_expr_apply_selected3_with_policy(
            policy,
            first,
            second,
            third,
            self.control,
            self.count,
        )
    }

    pub(in crate::detail) fn apply_expr4<A, B, C, D>(
        &self,
        policy: &crate::policy::CubePolicy<A::Runtime>,
        a: &A,
        b: &B,
        c: &C,
        d: &D,
    ) -> Result<
        (
            DeviceVec<A::Runtime, A::Item>,
            DeviceVec<A::Runtime, B::Item>,
            DeviceVec<A::Runtime, C::Item>,
            DeviceVec<A::Runtime, D::Item>,
        ),
        Error,
    >
    where
        A: KernelColumn + KernelColumnAt<S0>,
        B: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        D: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        A::Runtime: Runtime,
        A::Item: CubePrimitive + CubeElement,
        B::Item: CubePrimitive + CubeElement,
        C::Item: CubePrimitive + CubeElement,
        D::Item: CubePrimitive + CubeElement,
        A::Expr: DeviceGpuExpr<A::Item>,
        B::Expr: DeviceGpuExpr<B::Item>,
        C::Expr: DeviceGpuExpr<C::Item>,
        D::Expr: DeviceGpuExpr<D::Item>,
    {
        expr::selection::device_expr_apply_selected4_with_policy(
            policy,
            a,
            b,
            c,
            d,
            self.control,
            self.count,
        )
    }

    pub(in crate::detail) fn apply_expr5<A, B, C, D, E>(
        &self,
        policy: &crate::policy::CubePolicy<A::Runtime>,
        a: &A,
        b: &B,
        c: &C,
        d: &D,
        e: &E,
    ) -> Result<
        (
            DeviceVec<A::Runtime, A::Item>,
            DeviceVec<A::Runtime, B::Item>,
            DeviceVec<A::Runtime, C::Item>,
            DeviceVec<A::Runtime, D::Item>,
            DeviceVec<A::Runtime, E::Item>,
        ),
        Error,
    >
    where
        A: KernelColumn + KernelColumnAt<S0>,
        B: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        D: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        E: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        A::Runtime: Runtime,
        A::Item: CubePrimitive + CubeElement,
        B::Item: CubePrimitive + CubeElement,
        C::Item: CubePrimitive + CubeElement,
        D::Item: CubePrimitive + CubeElement,
        E::Item: CubePrimitive + CubeElement,
        A::Expr: DeviceGpuExpr<A::Item>,
        B::Expr: DeviceGpuExpr<B::Item>,
        C::Expr: DeviceGpuExpr<C::Item>,
        D::Expr: DeviceGpuExpr<D::Item>,
        E::Expr: DeviceGpuExpr<E::Item>,
    {
        expr::selection::device_expr_apply_selected5_with_policy(
            policy,
            a,
            b,
            c,
            d,
            e,
            self.control,
            self.count,
        )
    }

    pub(in crate::detail) fn apply_expr6<A, B, C, D, E, F>(
        &self,
        policy: &crate::policy::CubePolicy<A::Runtime>,
        a: &A,
        b: &B,
        c: &C,
        d: &D,
        e: &E,
        f: &F,
    ) -> Result<
        (
            DeviceVec<A::Runtime, A::Item>,
            DeviceVec<A::Runtime, B::Item>,
            DeviceVec<A::Runtime, C::Item>,
            DeviceVec<A::Runtime, D::Item>,
            DeviceVec<A::Runtime, E::Item>,
            DeviceVec<A::Runtime, F::Item>,
        ),
        Error,
    >
    where
        A: KernelColumn + KernelColumnAt<S0>,
        B: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        D: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        E: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        F: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        A::Runtime: Runtime,
        A::Item: CubePrimitive + CubeElement,
        B::Item: CubePrimitive + CubeElement,
        C::Item: CubePrimitive + CubeElement,
        D::Item: CubePrimitive + CubeElement,
        E::Item: CubePrimitive + CubeElement,
        F::Item: CubePrimitive + CubeElement,
        A::Expr: DeviceGpuExpr<A::Item>,
        B::Expr: DeviceGpuExpr<B::Item>,
        C::Expr: DeviceGpuExpr<C::Item>,
        D::Expr: DeviceGpuExpr<D::Item>,
        E::Expr: DeviceGpuExpr<E::Item>,
        F::Expr: DeviceGpuExpr<F::Item>,
    {
        expr::selection::device_expr_apply_selected6_with_policy(
            policy,
            a,
            b,
            c,
            d,
            e,
            f,
            self.control,
            self.count,
        )
    }

    pub(in crate::detail) fn apply_expr7<A, B, C, D, E, F, G>(
        &self,
        policy: &crate::policy::CubePolicy<A::Runtime>,
        a: &A,
        b: &B,
        c: &C,
        d: &D,
        e: &E,
        f: &F,
        g: &G,
    ) -> Result<
        (
            DeviceVec<A::Runtime, A::Item>,
            DeviceVec<A::Runtime, B::Item>,
            DeviceVec<A::Runtime, C::Item>,
            DeviceVec<A::Runtime, D::Item>,
            DeviceVec<A::Runtime, E::Item>,
            DeviceVec<A::Runtime, F::Item>,
            DeviceVec<A::Runtime, G::Item>,
        ),
        Error,
    >
    where
        A: KernelColumn + KernelColumnAt<S0>,
        B: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        D: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        E: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        F: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        G: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        A::Runtime: Runtime,
        A::Item: CubePrimitive + CubeElement,
        B::Item: CubePrimitive + CubeElement,
        C::Item: CubePrimitive + CubeElement,
        D::Item: CubePrimitive + CubeElement,
        E::Item: CubePrimitive + CubeElement,
        F::Item: CubePrimitive + CubeElement,
        G::Item: CubePrimitive + CubeElement,
        A::Expr: DeviceGpuExpr<A::Item>,
        B::Expr: DeviceGpuExpr<B::Item>,
        C::Expr: DeviceGpuExpr<C::Item>,
        D::Expr: DeviceGpuExpr<D::Item>,
        E::Expr: DeviceGpuExpr<E::Item>,
        F::Expr: DeviceGpuExpr<F::Item>,
        G::Expr: DeviceGpuExpr<G::Item>,
    {
        expr::selection::device_expr_apply_selected7_with_policy(
            policy,
            a,
            b,
            c,
            d,
            e,
            f,
            g,
            self.control,
            self.count,
        )
    }

    pub(in crate::detail) fn apply_value<R, T>(
        &self,
        policy: &crate::detail::CubePolicy<R>,
        value_handle: cubecl::server::Handle,
    ) -> Result<DeviceVec<R, T>, Error>
    where
        R: Runtime,
        T: CubePrimitive + CubeElement,
    {
        select::compact_value_with_count(policy, self.control, value_handle, self.count)
    }

    #[allow(dead_code)]
    pub(in crate::detail) fn apply_value2<R, Left, Right>(
        &self,
        policy: &crate::detail::CubePolicy<R>,
        left_handle: cubecl::server::Handle,
        right_handle: cubecl::server::Handle,
    ) -> Result<(DeviceVec<R, Left>, DeviceVec<R, Right>), Error>
    where
        R: Runtime,
        Left: CubePrimitive + CubeElement,
        Right: CubePrimitive + CubeElement,
    {
        Ok((
            self.apply_value(policy, left_handle)?,
            self.apply_value(policy, right_handle)?,
        ))
    }

    #[allow(dead_code)]
    pub(in crate::detail) fn apply_value3<R, First, Second, Third>(
        &self,
        policy: &crate::detail::CubePolicy<R>,
        first_handle: cubecl::server::Handle,
        second_handle: cubecl::server::Handle,
        third_handle: cubecl::server::Handle,
    ) -> Result<
        (
            DeviceVec<R, First>,
            DeviceVec<R, Second>,
            DeviceVec<R, Third>,
        ),
        Error,
    >
    where
        R: Runtime,
        First: CubePrimitive + CubeElement,
        Second: CubePrimitive + CubeElement,
        Third: CubePrimitive + CubeElement,
    {
        Ok((
            self.apply_value(policy, first_handle)?,
            self.apply_value(policy, second_handle)?,
            self.apply_value(policy, third_handle)?,
        ))
    }
}

pub(in crate::detail) struct SplitPayloadApply<'a> {
    control: &'a select::SplitRankControl,
    selected_count: usize,
    rejected_count: usize,
}

impl<'a> SplitPayloadApply<'a> {
    pub(in crate::detail) fn new(
        control: &'a select::SplitRankControl,
        selected_count: usize,
        rejected_count: usize,
    ) -> Self {
        Self {
            control,
            selected_count,
            rejected_count,
        }
    }

    pub(in crate::detail) fn apply_expr<ExprSource>(
        &self,
        policy: &crate::policy::CubePolicy<ExprSource::Runtime>,
        expr: &ExprSource,
    ) -> Result<
        (
            DeviceVec<ExprSource::Runtime, ExprSource::Item>,
            DeviceVec<ExprSource::Runtime, ExprSource::Item>,
        ),
        Error,
    >
    where
        ExprSource: KernelColumn + KernelColumnAt<S0>,
        ExprSource::Runtime: Runtime,
        ExprSource::Item: CubePrimitive + CubeElement,
        ExprSource::Expr: DeviceGpuExpr<ExprSource::Item>,
    {
        expr::selection::device_expr_compact_split_with_split_with_policy(
            policy,
            expr,
            self.control,
            self.selected_count,
            self.rejected_count,
        )
    }

    pub(in crate::detail) fn apply_expr2<Left, Right>(
        &self,
        policy: &crate::policy::CubePolicy<Left::Runtime>,
        left: &Left,
        right: &Right,
    ) -> Result<
        (
            (
                DeviceVec<Left::Runtime, Left::Item>,
                DeviceVec<Left::Runtime, Right::Item>,
            ),
            (
                DeviceVec<Left::Runtime, Left::Item>,
                DeviceVec<Left::Runtime, Right::Item>,
            ),
        ),
        Error,
    >
    where
        Left: KernelColumn + KernelColumnAt<S0>,
        Right: KernelColumn<Runtime = Left::Runtime> + KernelColumnAt<S0>,
        Left::Runtime: Runtime,
        Left::Item: CubePrimitive + CubeElement,
        Right::Item: CubePrimitive + CubeElement,
        Left::Expr: DeviceGpuExpr<Left::Item>,
        Right::Expr: DeviceGpuExpr<Right::Item>,
    {
        expr::selection::device_expr_apply_split2_with_policy(
            policy,
            left,
            right,
            self.control,
            self.selected_count,
            self.rejected_count,
        )
    }

    pub(in crate::detail) fn apply_expr3<First, Second, Third>(
        &self,
        policy: &crate::policy::CubePolicy<First::Runtime>,
        first: &First,
        second: &Second,
        third: &Third,
    ) -> Result<
        (
            (
                DeviceVec<First::Runtime, First::Item>,
                DeviceVec<First::Runtime, Second::Item>,
                DeviceVec<First::Runtime, Third::Item>,
            ),
            (
                DeviceVec<First::Runtime, First::Item>,
                DeviceVec<First::Runtime, Second::Item>,
                DeviceVec<First::Runtime, Third::Item>,
            ),
        ),
        Error,
    >
    where
        First: KernelColumn + KernelColumnAt<S0>,
        Second: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
        Third: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
        First::Runtime: Runtime,
        First::Item: CubePrimitive + CubeElement,
        Second::Item: CubePrimitive + CubeElement,
        Third::Item: CubePrimitive + CubeElement,
        First::Expr: DeviceGpuExpr<First::Item>,
        Second::Expr: DeviceGpuExpr<Second::Item>,
        Third::Expr: DeviceGpuExpr<Third::Item>,
    {
        expr::selection::device_expr_apply_split3_with_policy(
            policy,
            first,
            second,
            third,
            self.control,
            self.selected_count,
            self.rejected_count,
        )
    }

    pub(in crate::detail) fn apply_expr4<A, B, C, D>(
        &self,
        policy: &crate::policy::CubePolicy<A::Runtime>,
        a: &A,
        b: &B,
        c: &C,
        d: &D,
    ) -> Result<
        (
            (
                DeviceVec<A::Runtime, A::Item>,
                DeviceVec<A::Runtime, B::Item>,
                DeviceVec<A::Runtime, C::Item>,
                DeviceVec<A::Runtime, D::Item>,
            ),
            (
                DeviceVec<A::Runtime, A::Item>,
                DeviceVec<A::Runtime, B::Item>,
                DeviceVec<A::Runtime, C::Item>,
                DeviceVec<A::Runtime, D::Item>,
            ),
        ),
        Error,
    >
    where
        A: KernelColumn + KernelColumnAt<S0>,
        B: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        D: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        A::Runtime: Runtime,
        A::Item: CubePrimitive + CubeElement,
        B::Item: CubePrimitive + CubeElement,
        C::Item: CubePrimitive + CubeElement,
        D::Item: CubePrimitive + CubeElement,
        A::Expr: DeviceGpuExpr<A::Item>,
        B::Expr: DeviceGpuExpr<B::Item>,
        C::Expr: DeviceGpuExpr<C::Item>,
        D::Expr: DeviceGpuExpr<D::Item>,
    {
        expr::selection::device_expr_apply_split4_with_policy(
            policy,
            a,
            b,
            c,
            d,
            self.control,
            self.selected_count,
            self.rejected_count,
        )
    }

    pub(in crate::detail) fn apply_expr5<A, B, C, D, E>(
        &self,
        policy: &crate::policy::CubePolicy<A::Runtime>,
        a: &A,
        b: &B,
        c: &C,
        d: &D,
        e: &E,
    ) -> Result<
        (
            (
                DeviceVec<A::Runtime, A::Item>,
                DeviceVec<A::Runtime, B::Item>,
                DeviceVec<A::Runtime, C::Item>,
                DeviceVec<A::Runtime, D::Item>,
                DeviceVec<A::Runtime, E::Item>,
            ),
            (
                DeviceVec<A::Runtime, A::Item>,
                DeviceVec<A::Runtime, B::Item>,
                DeviceVec<A::Runtime, C::Item>,
                DeviceVec<A::Runtime, D::Item>,
                DeviceVec<A::Runtime, E::Item>,
            ),
        ),
        Error,
    >
    where
        A: KernelColumn + KernelColumnAt<S0>,
        B: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        D: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        E: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        A::Runtime: Runtime,
        A::Item: CubePrimitive + CubeElement,
        B::Item: CubePrimitive + CubeElement,
        C::Item: CubePrimitive + CubeElement,
        D::Item: CubePrimitive + CubeElement,
        E::Item: CubePrimitive + CubeElement,
        A::Expr: DeviceGpuExpr<A::Item>,
        B::Expr: DeviceGpuExpr<B::Item>,
        C::Expr: DeviceGpuExpr<C::Item>,
        D::Expr: DeviceGpuExpr<D::Item>,
        E::Expr: DeviceGpuExpr<E::Item>,
    {
        expr::selection::device_expr_apply_split5_with_policy(
            policy,
            a,
            b,
            c,
            d,
            e,
            self.control,
            self.selected_count,
            self.rejected_count,
        )
    }

    pub(in crate::detail) fn apply_expr6<A, B, C, D, E, F>(
        &self,
        policy: &crate::policy::CubePolicy<A::Runtime>,
        a: &A,
        b: &B,
        c: &C,
        d: &D,
        e: &E,
        f: &F,
    ) -> Result<
        (
            (
                DeviceVec<A::Runtime, A::Item>,
                DeviceVec<A::Runtime, B::Item>,
                DeviceVec<A::Runtime, C::Item>,
                DeviceVec<A::Runtime, D::Item>,
                DeviceVec<A::Runtime, E::Item>,
                DeviceVec<A::Runtime, F::Item>,
            ),
            (
                DeviceVec<A::Runtime, A::Item>,
                DeviceVec<A::Runtime, B::Item>,
                DeviceVec<A::Runtime, C::Item>,
                DeviceVec<A::Runtime, D::Item>,
                DeviceVec<A::Runtime, E::Item>,
                DeviceVec<A::Runtime, F::Item>,
            ),
        ),
        Error,
    >
    where
        A: KernelColumn + KernelColumnAt<S0>,
        B: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        D: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        E: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        F: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        A::Runtime: Runtime,
        A::Item: CubePrimitive + CubeElement,
        B::Item: CubePrimitive + CubeElement,
        C::Item: CubePrimitive + CubeElement,
        D::Item: CubePrimitive + CubeElement,
        E::Item: CubePrimitive + CubeElement,
        F::Item: CubePrimitive + CubeElement,
        A::Expr: DeviceGpuExpr<A::Item>,
        B::Expr: DeviceGpuExpr<B::Item>,
        C::Expr: DeviceGpuExpr<C::Item>,
        D::Expr: DeviceGpuExpr<D::Item>,
        E::Expr: DeviceGpuExpr<E::Item>,
        F::Expr: DeviceGpuExpr<F::Item>,
    {
        expr::selection::device_expr_apply_split6_with_policy(
            policy,
            a,
            b,
            c,
            d,
            e,
            f,
            self.control,
            self.selected_count,
            self.rejected_count,
        )
    }

    pub(in crate::detail) fn apply_expr7<A, B, C, D, E, F, G>(
        &self,
        policy: &crate::policy::CubePolicy<A::Runtime>,
        a: &A,
        b: &B,
        c: &C,
        d: &D,
        e: &E,
        f: &F,
        g: &G,
    ) -> Result<
        (
            (
                DeviceVec<A::Runtime, A::Item>,
                DeviceVec<A::Runtime, B::Item>,
                DeviceVec<A::Runtime, C::Item>,
                DeviceVec<A::Runtime, D::Item>,
                DeviceVec<A::Runtime, E::Item>,
                DeviceVec<A::Runtime, F::Item>,
                DeviceVec<A::Runtime, G::Item>,
            ),
            (
                DeviceVec<A::Runtime, A::Item>,
                DeviceVec<A::Runtime, B::Item>,
                DeviceVec<A::Runtime, C::Item>,
                DeviceVec<A::Runtime, D::Item>,
                DeviceVec<A::Runtime, E::Item>,
                DeviceVec<A::Runtime, F::Item>,
                DeviceVec<A::Runtime, G::Item>,
            ),
        ),
        Error,
    >
    where
        A: KernelColumn + KernelColumnAt<S0>,
        B: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        D: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        E: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        F: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        G: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
        A::Runtime: Runtime,
        A::Item: CubePrimitive + CubeElement,
        B::Item: CubePrimitive + CubeElement,
        C::Item: CubePrimitive + CubeElement,
        D::Item: CubePrimitive + CubeElement,
        E::Item: CubePrimitive + CubeElement,
        F::Item: CubePrimitive + CubeElement,
        G::Item: CubePrimitive + CubeElement,
        A::Expr: DeviceGpuExpr<A::Item>,
        B::Expr: DeviceGpuExpr<B::Item>,
        C::Expr: DeviceGpuExpr<C::Item>,
        D::Expr: DeviceGpuExpr<D::Item>,
        E::Expr: DeviceGpuExpr<E::Item>,
        F::Expr: DeviceGpuExpr<F::Item>,
        G::Expr: DeviceGpuExpr<G::Item>,
    {
        expr::selection::device_expr_apply_split7_with_policy(
            policy,
            a,
            b,
            c,
            d,
            e,
            f,
            g,
            self.control,
            self.selected_count,
            self.rejected_count,
        )
    }
}

pub(in crate::detail) fn device_expr_apply_selected_with_policy<ExprSource>(
    policy: &crate::policy::CubePolicy<ExprSource::Runtime>,
    expr: &ExprSource,
    control: &select::SelectedRankControl,
    count: usize,
) -> Result<DeviceVec<ExprSource::Runtime, ExprSource::Item>, Error>
where
    ExprSource: KernelColumn + KernelColumnAt<S0>,
    ExprSource::Runtime: Runtime,
    ExprSource::Item: CubePrimitive + CubeElement,
    ExprSource::Expr: DeviceGpuExpr<ExprSource::Item>,
{
    SelectedPayloadApply::new(control, count).apply_expr(policy, expr)
}

#[allow(dead_code)]
pub(in crate::detail) fn device_expr_apply_split_with_policy<ExprSource>(
    policy: &crate::policy::CubePolicy<ExprSource::Runtime>,
    expr: &ExprSource,
    control: &select::SplitRankControl,
    selected_count: usize,
    rejected_count: usize,
) -> Result<
    (
        DeviceVec<ExprSource::Runtime, ExprSource::Item>,
        DeviceVec<ExprSource::Runtime, ExprSource::Item>,
    ),
    Error,
>
where
    ExprSource: KernelColumn + KernelColumnAt<S0>,
    ExprSource::Runtime: Runtime,
    ExprSource::Item: CubePrimitive + CubeElement,
    ExprSource::Expr: DeviceGpuExpr<ExprSource::Item>,
{
    SplitPayloadApply::new(control, selected_count, rejected_count).apply_expr(policy, expr)
}

pub(in crate::detail) fn device_value_apply_selected_with_policy<R, T>(
    policy: &crate::detail::CubePolicy<R>,
    control: &select::SelectedRankControl,
    value_handle: cubecl::server::Handle,
    count: usize,
) -> Result<DeviceVec<R, T>, Error>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    SelectedPayloadApply::new(control, count).apply_value(policy, value_handle)
}
