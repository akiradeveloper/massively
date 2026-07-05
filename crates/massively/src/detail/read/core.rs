use super::*;

pub(crate) trait KernelScalarRead<R: Runtime>: Sized {
    type Item: MStorageElement + 'static;
    type Args;
    type Expr;

    fn len(&self) -> usize;

    fn stage(&self, policy: &CubePolicy<R>) -> Result<Self::Args, Error>;
}

/// Host-side lowering for one logical algorithm item.
#[allow(dead_code)]
pub(crate) trait KernelItemRead<R: Runtime>: Sized {
    type Item: MItem<R>;
    type Args;
    type Expr;

    fn len(&self) -> usize;

    fn stage(&self, policy: &CubePolicy<R>) -> Result<Self::Args, Error>;
}

#[allow(dead_code)]
pub(crate) trait KernelIndexRead: Sized {
    type Runtime: Runtime;
    type Source: KernelColumn<Runtime = Self::Runtime, Item = u32> + KernelColumnAt<S0>;

    fn index_source(self) -> Result<Self::Source, Error>;
}

impl<R, S> KernelScalarRead<R> for S
where
    R: Runtime,
    S: KernelColumn<Runtime = R> + KernelColumnAt<S0>,
    S::Item: MStorageElement + 'static,
{
    type Item = S::Item;
    type Args = KernelColumnBindings;
    type Expr = S::Expr;

    fn len(&self) -> usize {
        <S as KernelColumn>::len(self)
    }

    fn stage(&self, policy: &CubePolicy<R>) -> Result<Self::Args, Error> {
        <S as KernelColumn>::stage(self, policy)
    }
}

impl<R, S> KernelItemRead<R> for (S,)
where
    R: Runtime,
    S: KernelScalarRead<R>,
    (S::Item,): MItem<R>,
{
    type Item = (S::Item,);
    type Args = S::Args;
    type Expr = (S::Expr,);

    fn len(&self) -> usize {
        <S as KernelScalarRead<R>>::len(&self.0)
    }

    fn stage(&self, policy: &CubePolicy<R>) -> Result<Self::Args, Error> {
        <S as KernelScalarRead<R>>::stage(&self.0, policy)
    }
}

impl<R, S> KernelItemRead<R> for ZipView1<S>
where
    R: Runtime,
    S: KernelScalarRead<R>,
    (S::Item,): MItem<R>,
{
    type Item = (S::Item,);
    type Args = S::Args;
    type Expr = (S::Expr,);

    fn len(&self) -> usize {
        <S as KernelScalarRead<R>>::len(&self.source)
    }

    fn stage(&self, policy: &CubePolicy<R>) -> Result<Self::Args, Error> {
        <S as KernelScalarRead<R>>::stage(&self.source, policy)
    }
}

impl<R, S> KernelItemRead<R> for DeviceZip1<S>
where
    R: Runtime,
    S: KernelScalarRead<R>,
    (S::Item,): MItem<R>,
{
    type Item = (S::Item,);
    type Args = S::Args;
    type Expr = (S::Expr,);

    fn len(&self) -> usize {
        <S as KernelScalarRead<R>>::len(&self.source)
    }

    fn stage(&self, policy: &CubePolicy<R>) -> Result<Self::Args, Error> {
        <S as KernelScalarRead<R>>::stage(&self.source, policy)
    }
}

macro_rules! impl_kernel_item_read {
    (
        $target:ty,
        $first_field:tt : $first_source:ident
        $(, $field:tt : $source:ident )+
    ) => {
        impl<R, $first_source, $( $source ),+> KernelItemRead<R> for $target
        where
            R: Runtime,
            $first_source: KernelScalarRead<R>,
            $( $source: KernelScalarRead<R>, )+
            ($first_source::Item, $( $source::Item, )+): MItem<R>,
        {
            type Item = ($first_source::Item, $( $source::Item, )+);
            type Args = ($first_source::Args, $( $source::Args, )+);
            type Expr = ($first_source::Expr, $( $source::Expr, )+);

            fn len(&self) -> usize {
                <_ as KernelScalarRead<R>>::len(&self.$first_field)
            }

            fn stage(&self, policy: &CubePolicy<R>) -> Result<Self::Args, Error> {
                Ok((
                    <_ as KernelScalarRead<R>>::stage(&self.$first_field, policy)?,
                    $( <_ as KernelScalarRead<R>>::stage(&self.$field, policy)?, )+
                ))
            }
        }
    };
}

impl_kernel_item_read!(ZipView2<A, C>, left: A, right: C);
impl_kernel_item_read!(DeviceZip2<A, C>, left: A, right: C);
impl_kernel_item_read!(ZipView3<A, C, D>, first: A, second: C, third: D);
impl_kernel_item_read!(DeviceZip3<A, C, D>, first: A, second: C, third: D);

impl<Source> KernelIndexRead for ZipView1<Source>
where
    Self: ReadOnlyZip<Item = (u32,), Scalar = u32>,
    Source: KernelColumn<Item = u32> + KernelColumnAt<S0>,
    Source::Expr: DeviceGpuExpr<u32>,
{
    type Runtime = Source::Runtime;
    type Source = Source;

    fn index_source(self) -> Result<Self::Source, Error> {
        ReadOnlyZip::validate(&self)?;
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
        <ZipView1<Source> as KernelIndexRead>::index_source(ZipView1 { source: self })
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
