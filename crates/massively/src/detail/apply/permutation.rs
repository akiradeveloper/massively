use super::*;

pub(in crate::detail) struct PermutationPayloadApply<'a, R: Runtime> {
    control: &'a crate::detail::control::PermutationControl<R>,
}

impl<'a, R: Runtime> PermutationPayloadApply<'a, R> {
    pub(in crate::detail) fn new(
        control: &'a crate::detail::control::PermutationControl<R>,
    ) -> Self {
        Self { control }
    }

    pub(in crate::detail) fn apply_expr<ExprSource>(
        &self,
        policy: &crate::policy::CubePolicy<R>,
        expr: &ExprSource,
    ) -> Result<DeviceVec<R, ExprSource::Item>, Error>
    where
        ExprSource: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        ExprSource::Item: CubePrimitive + CubeElement,
        ExprSource::Expr: GpuExpr<ExprSource::Item>,
    {
        ensure_same_len(expr.len(), self.control.len)?;
        let indices = self.control.indices(policy);
        expr::device_expr_gather_with_policy(policy, expr, &indices)
    }

    pub(in crate::detail) fn apply_expr_into<ExprSource>(
        &self,
        policy: &crate::policy::CubePolicy<R>,
        expr: &ExprSource,
        output: &DeviceColumnMutView<R, ExprSource::Item>,
    ) -> Result<(), Error>
    where
        ExprSource: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        ExprSource::Item: CubePrimitive + CubeElement,
        ExprSource::Expr: GpuExpr<ExprSource::Item>,
    {
        ensure_same_len(expr.len(), self.control.len)?;
        ensure_same_len(output.len, self.control.len)?;
        let indices = self.control.indices(policy);
        expr::device_expr_gather_into_with_policy(policy, expr, &indices, output)
    }

    pub(in crate::detail) fn apply_expr2<Left, Right>(
        &self,
        policy: &crate::policy::CubePolicy<R>,
        left: &Left,
        right: &Right,
    ) -> Result<(DeviceVec<R, Left::Item>, DeviceVec<R, Right::Item>), Error>
    where
        Left: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        Right: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        Left::Item: CubePrimitive + CubeElement,
        Right::Item: CubePrimitive + CubeElement,
        Left::Expr: GpuExpr<Left::Item>,
        Right::Expr: GpuExpr<Right::Item>,
    {
        Ok((
            self.apply_expr(policy, left)?,
            self.apply_expr(policy, right)?,
        ))
    }

    pub(in crate::detail) fn apply_expr3<First, Second, Third>(
        &self,
        policy: &crate::policy::CubePolicy<R>,
        first: &First,
        second: &Second,
        third: &Third,
    ) -> Result<
        (
            DeviceVec<R, First::Item>,
            DeviceVec<R, Second::Item>,
            DeviceVec<R, Third::Item>,
        ),
        Error,
    >
    where
        First: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        Second: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        Third: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        First::Item: CubePrimitive + CubeElement,
        Second::Item: CubePrimitive + CubeElement,
        Third::Item: CubePrimitive + CubeElement,
        First::Expr: GpuExpr<First::Item>,
        Second::Expr: GpuExpr<Second::Item>,
        Third::Expr: GpuExpr<Third::Item>,
    {
        Ok((
            self.apply_expr(policy, first)?,
            self.apply_expr(policy, second)?,
            self.apply_expr(policy, third)?,
        ))
    }

    pub(in crate::detail) fn apply_expr4<A, B, C, D>(
        &self,
        policy: &crate::policy::CubePolicy<R>,
        a: &A,
        b: &B,
        c: &C,
        d: &D,
    ) -> Result<
        (
            DeviceVec<R, A::Item>,
            DeviceVec<R, B::Item>,
            DeviceVec<R, C::Item>,
            DeviceVec<R, D::Item>,
        ),
        Error,
    >
    where
        A: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        B: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        C: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        D: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        A::Item: CubePrimitive + CubeElement,
        B::Item: CubePrimitive + CubeElement,
        C::Item: CubePrimitive + CubeElement,
        D::Item: CubePrimitive + CubeElement,
        A::Expr: GpuExpr<A::Item>,
        B::Expr: GpuExpr<B::Item>,
        C::Expr: GpuExpr<C::Item>,
        D::Expr: GpuExpr<D::Item>,
    {
        Ok((
            self.apply_expr(policy, a)?,
            self.apply_expr(policy, b)?,
            self.apply_expr(policy, c)?,
            self.apply_expr(policy, d)?,
        ))
    }

    pub(in crate::detail) fn apply_expr5<A, B, C, D, E>(
        &self,
        policy: &crate::policy::CubePolicy<R>,
        a: &A,
        b: &B,
        c: &C,
        d: &D,
        e: &E,
    ) -> Result<
        (
            DeviceVec<R, A::Item>,
            DeviceVec<R, B::Item>,
            DeviceVec<R, C::Item>,
            DeviceVec<R, D::Item>,
            DeviceVec<R, E::Item>,
        ),
        Error,
    >
    where
        A: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        B: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        C: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        D: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        E: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        A::Item: CubePrimitive + CubeElement,
        B::Item: CubePrimitive + CubeElement,
        C::Item: CubePrimitive + CubeElement,
        D::Item: CubePrimitive + CubeElement,
        E::Item: CubePrimitive + CubeElement,
        A::Expr: GpuExpr<A::Item>,
        B::Expr: GpuExpr<B::Item>,
        C::Expr: GpuExpr<C::Item>,
        D::Expr: GpuExpr<D::Item>,
        E::Expr: GpuExpr<E::Item>,
    {
        Ok((
            self.apply_expr(policy, a)?,
            self.apply_expr(policy, b)?,
            self.apply_expr(policy, c)?,
            self.apply_expr(policy, d)?,
            self.apply_expr(policy, e)?,
        ))
    }

    pub(in crate::detail) fn apply_expr6<A, B, C, D, E, F>(
        &self,
        policy: &crate::policy::CubePolicy<R>,
        a: &A,
        b: &B,
        c: &C,
        d: &D,
        e: &E,
        f: &F,
    ) -> Result<
        (
            DeviceVec<R, A::Item>,
            DeviceVec<R, B::Item>,
            DeviceVec<R, C::Item>,
            DeviceVec<R, D::Item>,
            DeviceVec<R, E::Item>,
            DeviceVec<R, F::Item>,
        ),
        Error,
    >
    where
        A: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        B: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        C: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        D: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        E: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        F: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        A::Item: CubePrimitive + CubeElement,
        B::Item: CubePrimitive + CubeElement,
        C::Item: CubePrimitive + CubeElement,
        D::Item: CubePrimitive + CubeElement,
        E::Item: CubePrimitive + CubeElement,
        F::Item: CubePrimitive + CubeElement,
        A::Expr: GpuExpr<A::Item>,
        B::Expr: GpuExpr<B::Item>,
        C::Expr: GpuExpr<C::Item>,
        D::Expr: GpuExpr<D::Item>,
        E::Expr: GpuExpr<E::Item>,
        F::Expr: GpuExpr<F::Item>,
    {
        Ok((
            self.apply_expr(policy, a)?,
            self.apply_expr(policy, b)?,
            self.apply_expr(policy, c)?,
            self.apply_expr(policy, d)?,
            self.apply_expr(policy, e)?,
            self.apply_expr(policy, f)?,
        ))
    }

    pub(in crate::detail) fn apply_expr7<A, B, C, D, E, F, G>(
        &self,
        policy: &crate::policy::CubePolicy<R>,
        a: &A,
        b: &B,
        c: &C,
        d: &D,
        e: &E,
        f: &F,
        g: &G,
    ) -> Result<
        (
            DeviceVec<R, A::Item>,
            DeviceVec<R, B::Item>,
            DeviceVec<R, C::Item>,
            DeviceVec<R, D::Item>,
            DeviceVec<R, E::Item>,
            DeviceVec<R, F::Item>,
            DeviceVec<R, G::Item>,
        ),
        Error,
    >
    where
        A: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        B: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        C: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        D: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        E: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        F: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        G: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        A::Item: CubePrimitive + CubeElement,
        B::Item: CubePrimitive + CubeElement,
        C::Item: CubePrimitive + CubeElement,
        D::Item: CubePrimitive + CubeElement,
        E::Item: CubePrimitive + CubeElement,
        F::Item: CubePrimitive + CubeElement,
        G::Item: CubePrimitive + CubeElement,
        A::Expr: GpuExpr<A::Item>,
        B::Expr: GpuExpr<B::Item>,
        C::Expr: GpuExpr<C::Item>,
        D::Expr: GpuExpr<D::Item>,
        E::Expr: GpuExpr<E::Item>,
        F::Expr: GpuExpr<F::Item>,
        G::Expr: GpuExpr<G::Item>,
    {
        Ok((
            self.apply_expr(policy, a)?,
            self.apply_expr(policy, b)?,
            self.apply_expr(policy, c)?,
            self.apply_expr(policy, d)?,
            self.apply_expr(policy, e)?,
            self.apply_expr(policy, f)?,
            self.apply_expr(policy, g)?,
        ))
    }
}

pub(in crate::detail) struct IndexedWriteApply<'a, R: Runtime> {
    control: &'a crate::detail::control::PermutationControl<R>,
}

impl<'a, R: Runtime> IndexedWriteApply<'a, R> {
    pub(in crate::detail) fn new(
        control: &'a crate::detail::control::PermutationControl<R>,
    ) -> Self {
        Self { control }
    }

    pub(in crate::detail) fn scatter_expr_into<ValueSource>(
        &self,
        policy: &crate::policy::CubePolicy<R>,
        values: &ValueSource,
        output: &DeviceColumnMutView<R, ValueSource::Item>,
    ) -> Result<(), Error>
    where
        ValueSource: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        ValueSource::Item: CubePrimitive + CubeElement,
        ValueSource::Expr: GpuExpr<ValueSource::Item>,
    {
        ensure_same_len(values.len(), self.control.len)?;
        let indices = self.control.indices(policy);
        expr::device_expr_scatter_into_with_policy(policy, values, &indices, output)
    }

    pub(in crate::detail) fn scatter_expr_where_into<ValueSource>(
        &self,
        policy: &crate::policy::CubePolicy<R>,
        values: &ValueSource,
        mask: &select::MaskControl,
        output: &DeviceColumnMutView<R, ValueSource::Item>,
    ) -> Result<(), Error>
    where
        ValueSource: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
        ValueSource::Item: CubePrimitive + CubeElement,
        ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
    {
        ensure_same_len(values.len(), self.control.len)?;
        ensure_same_len(mask.len, self.control.len)?;
        let indices = self.control.indices(policy);
        expr::device_expr_scatter_where_into_with_control(policy, values, &indices, mask, output)
    }
}

pub(in crate::detail) struct IndexedExprApply;

impl IndexedExprApply {
    pub(in crate::detail) fn gather_expr<InputSource, IndexSource>(
        policy: &crate::policy::CubePolicy<InputSource::Runtime>,
        input: &InputSource,
        indices: &IndexSource,
    ) -> Result<DeviceVec<InputSource::Runtime, InputSource::Item>, Error>
    where
        InputSource: KernelColumn + KernelColumnAt<S0>,
        InputSource::Runtime: Runtime,
        IndexSource: KernelColumn<Runtime = InputSource::Runtime, Item = u32> + KernelColumnAt<S0>,
        InputSource::Item: CubePrimitive + CubeElement,
        InputSource::Expr: GpuExpr<InputSource::Item>,
        IndexSource::Expr: GpuExpr<u32>,
    {
        expr::device_expr_gather_with_policy(policy, input, indices)
    }

    pub(in crate::detail) fn gather_expr_into<InputSource, IndexSource>(
        policy: &crate::policy::CubePolicy<InputSource::Runtime>,
        input: &InputSource,
        indices: &IndexSource,
        output: &DeviceColumnMutView<InputSource::Runtime, InputSource::Item>,
    ) -> Result<(), Error>
    where
        InputSource: KernelColumn + KernelColumnAt<S0>,
        InputSource::Runtime: Runtime,
        IndexSource: KernelColumn<Runtime = InputSource::Runtime, Item = u32> + KernelColumnAt<S0>,
        InputSource::Item: CubePrimitive + CubeElement,
        InputSource::Expr: GpuExpr<InputSource::Item>,
        IndexSource::Expr: GpuExpr<u32>,
    {
        expr::device_expr_gather_into_with_policy(policy, input, indices, output)
    }

    pub(in crate::detail) fn scatter_expr_into<ValueSource, IndexSource>(
        policy: &crate::policy::CubePolicy<ValueSource::Runtime>,
        values: &ValueSource,
        indices: &IndexSource,
        output: &DeviceColumnMutView<ValueSource::Runtime, ValueSource::Item>,
    ) -> Result<(), Error>
    where
        ValueSource: KernelColumn + KernelColumnAt<S0>,
        ValueSource::Runtime: Runtime,
        IndexSource: KernelColumn<Runtime = ValueSource::Runtime, Item = u32> + KernelColumnAt<S0>,
        ValueSource::Item: CubePrimitive + CubeElement,
        ValueSource::Expr: GpuExpr<ValueSource::Item>,
        IndexSource::Expr: GpuExpr<u32>,
    {
        expr::device_expr_scatter_into_with_policy(policy, values, indices, output)
    }
}
