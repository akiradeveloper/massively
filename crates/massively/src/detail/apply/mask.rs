use super::*;

pub(in crate::detail) struct MaskWriteApply<'a, R: Runtime, T: CubePrimitive + CubeElement> {
    mask: &'a select::MaskControl,
    output: &'a DeviceColumnMutView<R, T>,
}

impl<'a, R, T> MaskWriteApply<'a, R, T>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    pub(in crate::detail) fn new(
        mask: &'a select::MaskControl,
        output: &'a DeviceColumnMutView<R, T>,
    ) -> Self {
        Self { mask, output }
    }

    pub(in crate::detail) fn replace_value(
        &self,
        policy: &crate::policy::CubePolicy<R>,
        replacement: T,
    ) -> Result<(), Error> {
        expr::replace_where_into_with_control(policy, replacement, self.mask, self.output)
    }
}

pub(in crate::detail) struct MaskedIndexedExprApply;

impl MaskedIndexedExprApply {
    pub(in crate::detail) fn gather_where_expr_into<InputSource, IndexSource>(
        policy: &crate::policy::CubePolicy<InputSource::Runtime>,
        input: &InputSource,
        indices: &IndexSource,
        mask: &select::MaskControl,
        output: &DeviceColumnMutView<InputSource::Runtime, InputSource::Item>,
    ) -> Result<(), Error>
    where
        InputSource: KernelColumn + KernelColumnAt<S0>,
        InputSource::Runtime: Runtime,
        IndexSource: KernelColumn<Runtime = InputSource::Runtime, Item = u32> + KernelColumnAt<S0>,
        InputSource::Item: CubePrimitive + CubeElement,
        InputSource::Expr: GpuExpr<InputSource::Item>,
        IndexSource::Expr: DeviceGpuExpr<u32>,
    {
        expr::device_expr_gather_where_into_with_control(policy, input, indices, mask, output)
    }

    pub(in crate::detail) fn scatter_where_expr_into<ValueSource, IndexSource>(
        policy: &crate::policy::CubePolicy<ValueSource::Runtime>,
        values: &ValueSource,
        indices: &IndexSource,
        mask: &select::MaskControl,
        output: &DeviceColumnMutView<ValueSource::Runtime, ValueSource::Item>,
    ) -> Result<(), Error>
    where
        ValueSource: KernelColumn + KernelColumnAt<S0>,
        ValueSource::Runtime: Runtime,
        IndexSource: KernelColumn<Runtime = ValueSource::Runtime, Item = u32> + KernelColumnAt<S0>,
        ValueSource::Item: CubePrimitive + CubeElement,
        ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
        IndexSource::Expr: DeviceGpuExpr<u32>,
    {
        expr::device_expr_scatter_where_into_with_control(policy, values, indices, mask, output)
    }
}
