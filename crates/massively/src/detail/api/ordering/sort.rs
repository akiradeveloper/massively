use super::*;
use crate::detail::api::Tuple1Less;

/// Input accepted by [`sort`].
#[doc(hidden)]
pub trait SortInput<Less> {
    /// Runtime used by this input.
    type Runtime: Runtime;
    /// Output produced by sorting this input.
    type Output;

    /// Sorts this input.
    fn sort_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        less: GpuOp<Less>,
    ) -> Result<Self::Output, Error>;
}

impl<Source, Less> SortInput<Less> for SoA1<Source>
where
    Self: SoA<Item = (Source::Item,), Scalar = Source::Item>,
    Source: ReadOnlyKernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Less: BinaryPredicateOp<Source::Item>,
{
    type Runtime = Source::Runtime;
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn sort_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        SoA::validate(&self)?;
        Ok(SoA1 {
            source: ordering::sort_input_with_policy(policy, &self.source, GpuOp::<Less>::new())?,
        })
    }
}

impl<Source, Less> SortInput<Less> for Source
where
    Source: ReadOnlyKernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Less: BinaryPredicateOp<Source::Item>,
{
    type Runtime = Source::Runtime;
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn sort_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        <SoA1<Source> as SortInput<Less>>::sort_input(SoA1 { source: self }, policy, less)
    }
}

impl<Source, Less> SortInput<Less> for (Source,)
where
    Source: ReadOnlyKernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Less: BinaryPredicateOp<(Source::Item,)>,
{
    type Runtime = Source::Runtime;
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn sort_input(
        self,
        policy: &CubePolicy<Source::Runtime>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        <SoA1<Source> as SortInput<Tuple1Less<Less>>>::sort_input(
            SoA1 { source: self.0 },
            policy,
            GpuOp::<Tuple1Less<Less>>::new(),
        )
    }
}

impl<Left, Right, Less> SortInput<Less> for (Left, Right)
where
    SoAView2<Left, Right>: SortInput<Less>,
    Left: KernelColumnAt<S0>,
    Right: KernelColumnAt<<Left as KernelColumnAt<S0>>::Next>,
{
    type Runtime = <SoAView2<Left, Right> as SortInput<Less>>::Runtime;
    type Output = <SoAView2<Left, Right> as SortInput<Less>>::Output;

    fn sort_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        <SoAView2<Left, Right> as SortInput<Less>>::sort_input(
            SoAView2 {
                left: self.0,
                right: self.1,
            },
            policy,
            less,
        )
    }
}

impl<First, Second, Third, Less> SortInput<Less> for (First, Second, Third)
where
    SoAView3<First, Second, Third>: SortInput<Less>,
    First: KernelColumnAt<S0>,
    Second: KernelColumnAt<<First as KernelColumnAt<S0>>::Next>,
    Third: KernelColumnAt<<Second as KernelColumnAt<<First as KernelColumnAt<S0>>::Next>>::Next>,
{
    type Runtime = <SoAView3<First, Second, Third> as SortInput<Less>>::Runtime;
    type Output = <SoAView3<First, Second, Third> as SortInput<Less>>::Output;

    fn sort_input(
        self,
        policy: &CubePolicy<Self::Runtime>,
        less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        <SoAView3<First, Second, Third> as SortInput<Less>>::sort_input(
            SoAView3 {
                first: self.0,
                second: self.1,
                third: self.2,
            },
            policy,
            less,
        )
    }
}

impl<Left, Right, Less> SortInput<Less> for SoA2<Left, Right>
where
    Self: ReadOnlySoA<Item = (Left::Item, Right::Item), Scalar = Left::Item>,
    Left: ReadOnlyKernelColumn + KernelColumnAt<S0>,
    Right: ReadOnlyKernelColumn<Runtime = Left::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<Left as KernelColumnAt<S0>>::Next>,
    Left::Item: CubePrimitive + CubeElement,
    Right::Item: CubePrimitive + CubeElement,
    Left::Expr: DeviceGpuExpr<Left::Item>,
    Right::Expr: DeviceGpuExpr<Right::Item>,
    Less: BinaryPredicateOp<(Left::Item, Right::Item)>,
{
    type Runtime = Left::Runtime;
    type Output = SoA2<DeviceVec<Left::Runtime, Left::Item>, DeviceVec<Left::Runtime, Right::Item>>;

    fn sort_input(
        self,
        policy: &CubePolicy<Left::Runtime>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        let (first, second) =
            ordering::sort_tuple2_input(policy, &self.left, &self.right, GpuOp::<Less>::new())?;
        Ok(SoA2 {
            left: first,
            right: second,
        })
    }
}

impl<First, Second, Third, Less> SortInput<Less> for SoA3<First, Second, Third>
where
    Self: ReadOnlySoA<Item = (First::Item, Second::Item, Third::Item), Scalar = First::Item>,
    First: ReadOnlyKernelColumn + KernelColumnAt<S0>,
    Second: ReadOnlyKernelColumn<Runtime = First::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<First as KernelColumnAt<S0>>::Next>,
    Third: ReadOnlyKernelColumn<Runtime = First::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<Second as KernelColumnAt<<First as KernelColumnAt<S0>>::Next>>::Next>,
    First::Item: CubePrimitive + CubeElement,
    Second::Item: CubePrimitive + CubeElement,
    Third::Item: CubePrimitive + CubeElement,
    First::Expr: DeviceGpuExpr<First::Item>,
    Second::Expr: DeviceGpuExpr<Second::Item>,
    Third::Expr: DeviceGpuExpr<Third::Item>,
    Less: BinaryPredicateOp<(First::Item, Second::Item, Third::Item)>,
{
    type Runtime = First::Runtime;
    type Output = SoA3<
        DeviceVec<First::Runtime, First::Item>,
        DeviceVec<First::Runtime, Second::Item>,
        DeviceVec<First::Runtime, Third::Item>,
    >;

    fn sort_input(
        self,
        policy: &CubePolicy<First::Runtime>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        let (first, second, third) = ordering::sort_tuple3_input(
            policy,
            &self.first,
            &self.second,
            &self.third,
            GpuOp::<Less>::new(),
        )?;
        Ok(SoA3 {
            first,
            second,
            third,
        })
    }
}

impl<Left, Right, Less> SortInput<Less> for SoAView2<Left, Right>
where
    Self: ReadOnlySoA<Item = (Left::Item, Right::Item), Scalar = Left::Item>,
    Left: ReadOnlyKernelColumn + KernelColumnAt<S0>,
    Right: ReadOnlyKernelColumn<Runtime = Left::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<Left as KernelColumnAt<S0>>::Next>,
    Left::Item: CubePrimitive + CubeElement,
    Right::Item: CubePrimitive + CubeElement,
    Left::Expr: DeviceGpuExpr<Left::Item>,
    Right::Expr: DeviceGpuExpr<Right::Item>,
    Less: BinaryPredicateOp<(Left::Item, Right::Item)>,
{
    type Runtime = Left::Runtime;
    type Output = SoA2<DeviceVec<Left::Runtime, Left::Item>, DeviceVec<Left::Runtime, Right::Item>>;

    fn sort_input(
        self,
        policy: &CubePolicy<Left::Runtime>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        let (left, right) =
            ordering::sort_tuple2_input(policy, &self.left, &self.right, GpuOp::<Less>::new())?;
        Ok(SoA2 { left, right })
    }
}

impl<First, Second, Third, Less> SortInput<Less> for SoAView3<First, Second, Third>
where
    Self: ReadOnlySoA<Item = (First::Item, Second::Item, Third::Item), Scalar = First::Item>,
    First: ReadOnlyKernelColumn + KernelColumnAt<S0>,
    Second: ReadOnlyKernelColumn<Runtime = First::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<First as KernelColumnAt<S0>>::Next>,
    Third: ReadOnlyKernelColumn<Runtime = First::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<Second as KernelColumnAt<<First as KernelColumnAt<S0>>::Next>>::Next>,
    First::Item: CubePrimitive + CubeElement,
    Second::Item: CubePrimitive + CubeElement,
    Third::Item: CubePrimitive + CubeElement,
    First::Expr: DeviceGpuExpr<First::Item>,
    Second::Expr: DeviceGpuExpr<Second::Item>,
    Third::Expr: DeviceGpuExpr<Third::Item>,
    Less: BinaryPredicateOp<(First::Item, Second::Item, Third::Item)>,
{
    type Runtime = First::Runtime;
    type Output = SoA3<
        DeviceVec<First::Runtime, First::Item>,
        DeviceVec<First::Runtime, Second::Item>,
        DeviceVec<First::Runtime, Third::Item>,
    >;

    fn sort_input(
        self,
        policy: &CubePolicy<First::Runtime>,
        _less: GpuOp<Less>,
    ) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        let (first, second, third) = ordering::sort_tuple3_input(
            policy,
            &self.first,
            &self.second,
            &self.third,
            GpuOp::<Less>::new(),
        )?;
        Ok(SoA3 {
            first,
            second,
            third,
        })
    }
}

/// Sorts read-only SoA input and returns owned device storage.
pub fn sort<R, Input, Less>(
    policy: &CubePolicy<R>,
    input: Input,
    _less: Less,
) -> Result<<<Input as SortInput<Less>>::Output as MaterializeOutput>::Output, Error>
where
    R: Runtime,
    Input: SortInput<Less, Runtime = R>,
    <Input as SortInput<Less>>::Output: MaterializeOutput<Runtime = R>,
{
    materialize(policy, input.sort_input(policy, GpuOp::<Less>::new())?)
}
