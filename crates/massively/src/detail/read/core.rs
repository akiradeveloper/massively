use super::*;

pub(crate) trait KernelScalarRead<B: Runtime>: Sized {
    type Item: Scalar + 'static;
    type Args;
    type Expr;

    fn len(&self) -> usize;

    fn stage(&self, policy: &CubePolicy<B>) -> Result<Self::Args, Error>;
}

/// Host-side lowering for one logical algorithm item.
#[allow(dead_code)]
pub(crate) trait KernelItemRead<B: Runtime>: Sized {
    type Item: MItem<B>;
    type Args;
    type Expr;

    fn len(&self) -> usize;

    fn stage(&self, policy: &CubePolicy<B>) -> Result<Self::Args, Error>;
}

#[allow(dead_code)]
pub(crate) trait KernelIndexRead: Sized {
    type Runtime: Runtime;
    type Source: KernelColumn<Runtime = Self::Runtime, Item = u32> + KernelColumnAt<S0>;

    fn index_source(self) -> Result<Self::Source, Error>;
}

impl<B, S> KernelScalarRead<B> for S
where
    B: Runtime,
    S: KernelColumn<Runtime = B> + KernelColumnAt<S0>,
    S::Item: Scalar + 'static,
{
    type Item = S::Item;
    type Args = KernelColumnBindings;
    type Expr = S::Expr;

    fn len(&self) -> usize {
        <S as KernelColumn>::len(self)
    }

    fn stage(&self, policy: &CubePolicy<B>) -> Result<Self::Args, Error> {
        <S as KernelColumn>::stage(self, policy)
    }
}

impl<B, S> KernelItemRead<B> for (S,)
where
    B: Runtime,
    S: KernelScalarRead<B>,
    (S::Item,): MItem<B>,
{
    type Item = (S::Item,);
    type Args = S::Args;
    type Expr = (S::Expr,);

    fn len(&self) -> usize {
        <S as KernelScalarRead<B>>::len(&self.0)
    }

    fn stage(&self, policy: &CubePolicy<B>) -> Result<Self::Args, Error> {
        <S as KernelScalarRead<B>>::stage(&self.0, policy)
    }
}

impl<B, S> KernelItemRead<B> for SoAView1<S>
where
    B: Runtime,
    S: KernelScalarRead<B>,
    (S::Item,): MItem<B>,
{
    type Item = (S::Item,);
    type Args = S::Args;
    type Expr = (S::Expr,);

    fn len(&self) -> usize {
        <S as KernelScalarRead<B>>::len(&self.source)
    }

    fn stage(&self, policy: &CubePolicy<B>) -> Result<Self::Args, Error> {
        <S as KernelScalarRead<B>>::stage(&self.source, policy)
    }
}

impl<B, S> KernelItemRead<B> for DeviceSoA1<S>
where
    B: Runtime,
    S: KernelScalarRead<B>,
    (S::Item,): MItem<B>,
{
    type Item = (S::Item,);
    type Args = S::Args;
    type Expr = (S::Expr,);

    fn len(&self) -> usize {
        <S as KernelScalarRead<B>>::len(&self.source)
    }

    fn stage(&self, policy: &CubePolicy<B>) -> Result<Self::Args, Error> {
        <S as KernelScalarRead<B>>::stage(&self.source, policy)
    }
}

macro_rules! impl_kernel_item_read {
    (
        $target:ty,
        $first_field:tt : $first_source:ident
        $(, $field:tt : $source:ident )+
    ) => {
        impl<B, $first_source, $( $source ),+> KernelItemRead<B> for $target
        where
            B: Runtime,
            $first_source: KernelScalarRead<B>,
            $( $source: KernelScalarRead<B>, )+
            ($first_source::Item, $( $source::Item, )+): MItem<B>,
        {
            type Item = ($first_source::Item, $( $source::Item, )+);
            type Args = ($first_source::Args, $( $source::Args, )+);
            type Expr = ($first_source::Expr, $( $source::Expr, )+);

            fn len(&self) -> usize {
                <_ as KernelScalarRead<B>>::len(&self.$first_field)
            }

            fn stage(&self, policy: &CubePolicy<B>) -> Result<Self::Args, Error> {
                Ok((
                    <_ as KernelScalarRead<B>>::stage(&self.$first_field, policy)?,
                    $( <_ as KernelScalarRead<B>>::stage(&self.$field, policy)?, )+
                ))
            }
        }
    };
}

impl_kernel_item_read!(SoAView2<A, C>, left: A, right: C);
impl_kernel_item_read!(DeviceSoA2<A, C>, left: A, right: C);
impl_kernel_item_read!(SoAView3<A, C, D>, first: A, second: C, third: D);
impl_kernel_item_read!(DeviceSoA3<A, C, D>, first: A, second: C, third: D);

impl<Source> KernelIndexRead for SoAView1<Source>
where
    Self: ReadOnlySoA<Item = (u32,), Scalar = u32>,
    Source: KernelColumn<Item = u32> + KernelColumnAt<S0>,
    Source::Expr: DeviceGpuExpr<u32>,
{
    type Runtime = Source::Runtime;
    type Source = Source;

    fn index_source(self) -> Result<Self::Source, Error> {
        ReadOnlySoA::validate(&self)?;
        Ok(self.source)
    }
}

impl<Source> KernelIndexRead for Source
where
    Source: KernelColumn<Item = u32> + KernelColumnAt<S0>,
    Source::Expr: DeviceGpuExpr<u32>,
{
    type Runtime = Source::Runtime;
    type Source = Source;

    fn index_source(self) -> Result<Self::Source, Error> {
        <SoAView1<Source> as KernelIndexRead>::index_source(SoAView1 { source: self })
    }
}

impl<Source> KernelIndexRead for (Source,)
where
    Source: KernelColumn<Item = u32> + KernelColumnAt<S0>,
    Source::Expr: DeviceGpuExpr<u32>,
{
    type Runtime = Source::Runtime;
    type Source = Source;

    fn index_source(self) -> Result<Self::Source, Error> {
        <Source as KernelIndexRead>::index_source(self.0)
    }
}
