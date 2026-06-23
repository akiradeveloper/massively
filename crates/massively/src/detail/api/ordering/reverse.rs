use super::*;

pub trait ReverseInput {
    /// Runtime used by this input.
    type Runtime: Runtime;

    /// Output produced by reversing this input.
    type Output;

    /// Reverses this input.
    fn reverse_input(self, policy: &CubePolicy<Self::Runtime>) -> Result<Self::Output, Error>;
}

impl<Source> ReverseInput for SoA1<Source>
where
    Self: SoA<Item = (Source::Item,), Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
{
    type Runtime = Source::Runtime;
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn reverse_input(self, policy: &CubePolicy<Source::Runtime>) -> Result<Self::Output, Error> {
        SoA::validate(&self)?;
        Ok(SoA1 {
            source: super::super::device_expr_reverse_collect(policy, &self.source)?,
        })
    }
}

impl<Source> ReverseInput for Source
where
    Source: ReadOnlyKernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
{
    type Runtime = Source::Runtime;
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn reverse_input(self, policy: &CubePolicy<Source::Runtime>) -> Result<Self::Output, Error> {
        <SoA1<Source> as ReverseInput>::reverse_input(SoA1 { source: self }, policy)
    }
}

impl<Source> ReverseInput for (Source,)
where
    Source: ReadOnlyKernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
{
    type Runtime = Source::Runtime;
    type Output = SoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn reverse_input(self, policy: &CubePolicy<Source::Runtime>) -> Result<Self::Output, Error> {
        <SoA1<Source> as ReverseInput>::reverse_input(SoA1 { source: self.0 }, policy)
    }
}

impl<Left, Right> ReverseInput for (Left, Right)
where
    SoAView2<Left, Right>: ReverseInput,
    Left: KernelColumnAt<S0>,
    Right: KernelColumnAt<<Left as KernelColumnAt<S0>>::Next>,
{
    type Runtime = <SoAView2<Left, Right> as ReverseInput>::Runtime;
    type Output = <SoAView2<Left, Right> as ReverseInput>::Output;

    fn reverse_input(self, policy: &CubePolicy<Self::Runtime>) -> Result<Self::Output, Error> {
        <SoAView2<Left, Right> as ReverseInput>::reverse_input(
            SoAView2 {
                left: self.0,
                right: self.1,
            },
            policy,
        )
    }
}

impl<First, Second, Third> ReverseInput for (First, Second, Third)
where
    SoAView3<First, Second, Third>: ReverseInput,
    First: KernelColumnAt<S0>,
    Second: KernelColumnAt<<First as KernelColumnAt<S0>>::Next>,
    Third: KernelColumnAt<<Second as KernelColumnAt<<First as KernelColumnAt<S0>>::Next>>::Next>,
{
    type Runtime = <SoAView3<First, Second, Third> as ReverseInput>::Runtime;
    type Output = <SoAView3<First, Second, Third> as ReverseInput>::Output;

    fn reverse_input(self, policy: &CubePolicy<Self::Runtime>) -> Result<Self::Output, Error> {
        <SoAView3<First, Second, Third> as ReverseInput>::reverse_input(
            SoAView3 {
                first: self.0,
                second: self.1,
                third: self.2,
            },
            policy,
        )
    }
}

macro_rules! impl_reverse_input {
    ($name:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ }) => {
        impl<$first, $( $rest ),+> ReverseInput for $name<$first, $( $rest ),+>
        where
            Self: ReadOnlySoA<Scalar = <$first as KernelColumn>::Item>,
            $first: ReadOnlyKernelColumn + KernelColumnAt<S0>,
            $(
                $rest: ReadOnlyKernelColumn<Runtime = <$first as KernelColumn>::Runtime>
                    + KernelColumnAt<S0>,
            )+
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
            )+
            $(
                <$rest as KernelColumn>::Expr: DeviceGpuExpr<<$rest as KernelColumn>::Item>,
            )+
        {
            type Runtime = <$first as KernelColumn>::Runtime;

            type Output = $name<
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$first as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            >;

            fn reverse_input(
                self,
                policy: &CubePolicy<<$first as KernelColumn>::Runtime>,
            ) -> Result<Self::Output, Error> {
                ReadOnlySoA::validate(&self)?;
                let $first_field = super::super::device_expr_reverse_collect(policy, &self.$first_field)?;
                $(
                    let $field = super::super::device_expr_reverse_collect(policy, &self.$field)?;
                )+

                Ok($name { $first_field, $( $field ),+ })
            }
        }

    };
}

impl_reverse_input!(SoA2<A, B> { left, right });
impl_reverse_input!(SoA3<A, B, C> { first, second, third });

impl<Left, Right> ReverseInput for SoAView2<Left, Right>
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
{
    type Runtime = Left::Runtime;

    type Output = SoA2<DeviceVec<Left::Runtime, Left::Item>, DeviceVec<Left::Runtime, Right::Item>>;

    fn reverse_input(self, policy: &CubePolicy<Left::Runtime>) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        Ok(SoA2 {
            left: super::super::device_expr_reverse_collect(policy, &self.left)?,
            right: super::super::device_expr_reverse_collect(policy, &self.right)?,
        })
    }
}

impl<First, Second, Third> ReverseInput for SoAView3<First, Second, Third>
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
{
    type Runtime = First::Runtime;

    type Output = SoA3<
        DeviceVec<First::Runtime, First::Item>,
        DeviceVec<First::Runtime, Second::Item>,
        DeviceVec<First::Runtime, Third::Item>,
    >;

    fn reverse_input(self, policy: &CubePolicy<First::Runtime>) -> Result<Self::Output, Error> {
        ReadOnlySoA::validate(&self)?;
        Ok(SoA3 {
            first: super::super::device_expr_reverse_collect(policy, &self.first)?,
            second: super::super::device_expr_reverse_collect(policy, &self.second)?,
            third: super::super::device_expr_reverse_collect(policy, &self.third)?,
        })
    }
}

/// Reverses read-only SoA input and returns new device storage.
pub fn reverse<Input>(
    policy: &CubePolicy<<Input as ReverseInput>::Runtime>,
    input: Input,
) -> Result<<<Input as ReverseInput>::Output as MaterializeOutput>::Output, Error>
where
    Input: ReverseInput,
    <Input as ReverseInput>::Output: MaterializeOutput<Runtime = <Input as ReverseInput>::Runtime>,
{
    materialize(policy, input.reverse_input(policy)?)
}
