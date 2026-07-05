use super::*;

#[allow(dead_code)]
pub(crate) trait KernelScatterInput<IndexSource>: Sized
where
    IndexSource: KernelColumn<Item = MIndex> + KernelColumnAt<S0>,
{
    type Runtime: Runtime;
    type Default;
    type Output;

    fn scatter_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: &IndexSource,
        len: usize,
        default: Self::Default,
    ) -> Result<Self::Output, Error>;
}

#[allow(dead_code)]
pub(crate) trait KernelScatterWhereInput<IndexSource, Stencil, Pred>: Sized
where
    IndexSource: KernelColumn<Item = MIndex> + KernelColumnAt<S0>,
{
    type Runtime: Runtime;
    type Default;
    type Output;

    fn scatter_where_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: &IndexSource,
        stencil: Stencil,
        len: usize,
        default: Self::Default,
    ) -> Result<Self::Output, Error>;
}

fn scatter_one_read<ValueSource, IndexSource>(
    policy: &CubePolicy<ValueSource::Runtime>,
    values: &ValueSource,
    indices: &IndexSource,
    len: usize,
    default: ValueSource::Item,
) -> Result<DeviceVec<ValueSource::Runtime, ValueSource::Item>, Error>
where
    ValueSource: KernelColumn + KernelColumnAt<S0>,
    ValueSource::Runtime: Runtime,
    IndexSource: KernelColumn<Runtime = ValueSource::Runtime, Item = MIndex> + KernelColumnAt<S0>,
    ValueSource::Item: MStorageElement + 'static,
    ValueSource::Expr: GpuExpr<ValueSource::Item>,
    IndexSource::Expr: DeviceGpuExpr<MIndex>,
{
    let index_values =
        crate::detail::apply::MaterializePayloadApply::collect_expr(policy, indices)?;
    let control = crate::detail::control::PermutationControl::from_indices(&index_values)?;
    let initial = primitive_range::filled(policy, len, default)?;
    let output = DeviceColumnMutView::from_slice(&initial, 0, len);
    let apply = crate::detail::apply::IndexedWriteApply::new(&control);
    apply.scatter_expr_into(policy, values, &output)?;
    Ok(initial)
}

fn scatter_where_one_read<ValueSource, IndexSource, Stencil, Pred>(
    policy: &CubePolicy<ValueSource::Runtime>,
    values: &ValueSource,
    indices: &IndexSource,
    stencil: &Stencil,
    len: usize,
    default: ValueSource::Item,
) -> Result<DeviceVec<ValueSource::Runtime, ValueSource::Item>, Error>
where
    ValueSource: KernelColumn + KernelColumnAt<S0>,
    ValueSource::Runtime: Runtime,
    IndexSource: KernelColumn<Runtime = ValueSource::Runtime, Item = MIndex> + KernelColumnAt<S0>,
    Stencil: crate::detail::api::SelectionStencil<Pred, Runtime = ValueSource::Runtime>,
    ValueSource::Item: MStorageElement + 'static,
    ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
    IndexSource::Expr: DeviceGpuExpr<MIndex>,
{
    let initial = primitive_range::filled(policy, len, default)?;
    <ValueSource as KernelColumn>::validate(values)?;
    <IndexSource as KernelColumn>::validate(indices)?;
    ensure_same_len(
        <ValueSource as KernelColumn>::len(values),
        <IndexSource as KernelColumn>::len(indices),
    )?;
    ensure_same_len(<ValueSource as KernelColumn>::len(values), stencil.len())?;
    let index_values =
        crate::detail::apply::MaterializePayloadApply::collect_expr(policy, indices)?;
    let write_control = crate::detail::control::PermutationControl::from_indices(&index_values)?;
    let mask = stencil.selection_flags_with_policy(policy, false)?;
    let output = DeviceColumnMutView::from_slice(&initial, 0, len);
    let apply = crate::detail::apply::IndexedWriteApply::new(&write_control);
    apply.scatter_expr_where_into(policy, values, &mask, &output)?;
    Ok(initial)
}

impl<ValueSource, IndexSource> KernelScatterInput<IndexSource> for ValueSource
where
    ValueSource: KernelColumn + KernelColumnAt<S0>,
    IndexSource: KernelColumn<Runtime = ValueSource::Runtime, Item = MIndex> + KernelColumnAt<S0>,
    ValueSource::Item: MStorageElement + 'static,
    ValueSource::Expr: GpuExpr<ValueSource::Item>,
    IndexSource::Expr: DeviceGpuExpr<MIndex>,
{
    type Runtime = ValueSource::Runtime;
    type Default = ValueSource::Item;
    type Output = DeviceZip1<DeviceVec<ValueSource::Runtime, ValueSource::Item>>;

    fn scatter_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: &IndexSource,
        len: usize,
        default: Self::Default,
    ) -> Result<Self::Output, Error> {
        Ok(DeviceZip1 {
            source: scatter_one_read::<ValueSource, IndexSource>(
                policy, &self, indices, len, default,
            )?,
        })
    }
}

macro_rules! impl_kernel_scatter_tuple1 {
    ($target:ty, $field:tt) => {
        impl<ValueSource, IndexSource> KernelScatterInput<IndexSource> for $target
        where
            ValueSource: KernelColumn + KernelColumnAt<S0>,
            IndexSource:
                KernelColumn<Runtime = ValueSource::Runtime, Item = MIndex> + KernelColumnAt<S0>,
            ValueSource::Item: MStorageElement + 'static,
            ValueSource::Expr: GpuExpr<ValueSource::Item>,
            IndexSource::Expr: DeviceGpuExpr<MIndex>,
        {
            type Runtime = ValueSource::Runtime;
            type Default = (ValueSource::Item,);
            type Output = DeviceZip1<DeviceVec<ValueSource::Runtime, ValueSource::Item>>;

            fn scatter_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
                indices: &IndexSource,
                len: usize,
                default: Self::Default,
            ) -> Result<Self::Output, Error> {
                Ok(DeviceZip1 {
                    source: scatter_one_read::<ValueSource, IndexSource>(
                        policy,
                        &self.$field,
                        indices,
                        len,
                        default.0,
                    )?,
                })
            }
        }
    };
}

impl_kernel_scatter_tuple1!(ZipView1<ValueSource>, source);
impl_kernel_scatter_tuple1!(DeviceZip1<ValueSource>, source);

impl<ValueSource, IndexSource> KernelScatterInput<IndexSource> for (ValueSource,)
where
    ValueSource: KernelColumn + KernelColumnAt<S0>,
    IndexSource: KernelColumn<Runtime = ValueSource::Runtime, Item = MIndex> + KernelColumnAt<S0>,
    ZipView1<ValueSource>: KernelScatterInput<IndexSource, Runtime = ValueSource::Runtime>,
{
    type Runtime = ValueSource::Runtime;
    type Default = <ZipView1<ValueSource> as KernelScatterInput<IndexSource>>::Default;
    type Output = <ZipView1<ValueSource> as KernelScatterInput<IndexSource>>::Output;

    fn scatter_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: &IndexSource,
        len: usize,
        default: Self::Default,
    ) -> Result<Self::Output, Error> {
        <ZipView1<ValueSource> as KernelScatterInput<IndexSource>>::scatter_read(
            ZipView1 { source: self.0 },
            policy,
            indices,
            len,
            default,
        )
    }
}

macro_rules! impl_kernel_scatter_tuple2 {
    ($target:ty, $out:ident, $left:tt, $right:tt) => {
        impl<Left, Right, IndexSource> KernelScatterInput<IndexSource> for $target
        where
            Left: KernelColumn + KernelColumnAt<S0>,
            Right: KernelColumn<Runtime = Left::Runtime> + KernelColumnAt<S0>,
            IndexSource: KernelColumn<Runtime = Left::Runtime, Item = MIndex> + KernelColumnAt<S0>,
            Left::Item: MStorageElement + 'static,
            Right::Item: MStorageElement + 'static,
            Left::Expr: GpuExpr<Left::Item>,
            Right::Expr: GpuExpr<Right::Item>,
            IndexSource::Expr: DeviceGpuExpr<MIndex>,
        {
            type Runtime = Left::Runtime;
            type Default = (Left::Item, Right::Item);
            type Output =
                $out<DeviceVec<Left::Runtime, Left::Item>, DeviceVec<Left::Runtime, Right::Item>>;

            fn scatter_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
                indices: &IndexSource,
                len: usize,
                default: Self::Default,
            ) -> Result<Self::Output, Error> {
                validate_columns2(&self.$left, &self.$right)?;
                let left = scatter_one_read::<Left, IndexSource>(
                    policy,
                    &self.$left,
                    indices,
                    len,
                    default.0,
                )?;
                let right = scatter_one_read::<Right, IndexSource>(
                    policy,
                    &self.$right,
                    indices,
                    len,
                    default.1,
                )?;
                Ok($out { left, right })
            }
        }
    };
}

impl_kernel_scatter_tuple2!(ZipView2<Left, Right>, DeviceZip2, left, right);
impl_kernel_scatter_tuple2!(DeviceZip2<Left, Right>, DeviceZip2, left, right);

impl<Left, Right, IndexSource> KernelScatterInput<IndexSource> for (Left, Right)
where
    IndexSource: KernelColumn<Item = MIndex> + KernelColumnAt<S0>,
    ZipView2<Left, Right>: KernelScatterInput<IndexSource>,
{
    type Runtime = <ZipView2<Left, Right> as KernelScatterInput<IndexSource>>::Runtime;
    type Default = <ZipView2<Left, Right> as KernelScatterInput<IndexSource>>::Default;
    type Output = <ZipView2<Left, Right> as KernelScatterInput<IndexSource>>::Output;

    fn scatter_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: &IndexSource,
        len: usize,
        default: Self::Default,
    ) -> Result<Self::Output, Error> {
        <ZipView2<Left, Right> as KernelScatterInput<IndexSource>>::scatter_read(
            ZipView2 {
                left: self.0,
                right: self.1,
            },
            policy,
            indices,
            len,
            default,
        )
    }
}

macro_rules! impl_kernel_scatter_tuple3 {
    ($target:ty, $out:ident, $first:tt, $second:tt, $third:tt) => {
        impl<First, Second, Third, IndexSource> KernelScatterInput<IndexSource> for $target
        where
            First: KernelColumn + KernelColumnAt<S0>,
            Second: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
            Third: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
            IndexSource: KernelColumn<Runtime = First::Runtime, Item = MIndex> + KernelColumnAt<S0>,
            First::Item: MStorageElement + 'static,
            Second::Item: MStorageElement + 'static,
            Third::Item: MStorageElement + 'static,
            First::Expr: GpuExpr<First::Item>,
            Second::Expr: GpuExpr<Second::Item>,
            Third::Expr: GpuExpr<Third::Item>,
            IndexSource::Expr: DeviceGpuExpr<MIndex>,
        {
            type Runtime = First::Runtime;
            type Default = (First::Item, Second::Item, Third::Item);
            type Output = $out<
                DeviceVec<First::Runtime, First::Item>,
                DeviceVec<First::Runtime, Second::Item>,
                DeviceVec<First::Runtime, Third::Item>,
            >;

            fn scatter_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
                indices: &IndexSource,
                len: usize,
                default: Self::Default,
            ) -> Result<Self::Output, Error> {
                validate_columns3(&self.$first, &self.$second, &self.$third)?;
                let first = scatter_one_read::<First, IndexSource>(
                    policy,
                    &self.$first,
                    indices,
                    len,
                    default.0,
                )?;
                let second = scatter_one_read::<Second, IndexSource>(
                    policy,
                    &self.$second,
                    indices,
                    len,
                    default.1,
                )?;
                let third = scatter_one_read::<Third, IndexSource>(
                    policy,
                    &self.$third,
                    indices,
                    len,
                    default.2,
                )?;
                Ok($out {
                    first,
                    second,
                    third,
                })
            }
        }
    };
}

impl_kernel_scatter_tuple3!(ZipView3<First, Second, Third>, DeviceZip3, first, second, third);
impl_kernel_scatter_tuple3!(DeviceZip3<First, Second, Third>, DeviceZip3, first, second, third);

impl<First, Second, Third, IndexSource> KernelScatterInput<IndexSource> for (First, Second, Third)
where
    IndexSource: KernelColumn<Item = MIndex> + KernelColumnAt<S0>,
    ZipView3<First, Second, Third>: KernelScatterInput<IndexSource>,
{
    type Runtime = <ZipView3<First, Second, Third> as KernelScatterInput<IndexSource>>::Runtime;
    type Default = <ZipView3<First, Second, Third> as KernelScatterInput<IndexSource>>::Default;
    type Output = <ZipView3<First, Second, Third> as KernelScatterInput<IndexSource>>::Output;

    fn scatter_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: &IndexSource,
        len: usize,
        default: Self::Default,
    ) -> Result<Self::Output, Error> {
        <ZipView3<First, Second, Third> as KernelScatterInput<IndexSource>>::scatter_read(
            ZipView3 {
                first: self.0,
                second: self.1,
                third: self.2,
            },
            policy,
            indices,
            len,
            default,
        )
    }
}

impl<ValueSource, IndexSource, Stencil, Pred> KernelScatterWhereInput<IndexSource, Stencil, Pred>
    for ValueSource
where
    ValueSource: KernelColumn + KernelColumnAt<S0>,
    IndexSource: KernelColumn<Runtime = ValueSource::Runtime, Item = MIndex> + KernelColumnAt<S0>,
    Stencil: crate::detail::api::SelectionStencil<Pred, Runtime = ValueSource::Runtime>,
    ValueSource::Item: MStorageElement + 'static,
    ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
    IndexSource::Expr: DeviceGpuExpr<MIndex>,
{
    type Runtime = ValueSource::Runtime;
    type Default = ValueSource::Item;
    type Output = DeviceZip1<DeviceVec<ValueSource::Runtime, ValueSource::Item>>;

    fn scatter_where_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: &IndexSource,
        stencil: Stencil,
        len: usize,
        default: Self::Default,
    ) -> Result<Self::Output, Error> {
        Ok(DeviceZip1 {
            source: scatter_where_one_read::<ValueSource, IndexSource, Stencil, Pred>(
                policy, &self, indices, &stencil, len, default,
            )?,
        })
    }
}

macro_rules! impl_kernel_scatter_where_tuple1 {
    ($target:ty, $field:tt) => {
        impl<ValueSource, IndexSource, Stencil, Pred>
            KernelScatterWhereInput<IndexSource, Stencil, Pred> for $target
        where
            ValueSource: KernelColumn + KernelColumnAt<S0>,
            IndexSource:
                KernelColumn<Runtime = ValueSource::Runtime, Item = MIndex> + KernelColumnAt<S0>,
            Stencil: crate::detail::api::SelectionStencil<Pred, Runtime = ValueSource::Runtime>,
            ValueSource::Item: MStorageElement + 'static,
            ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
            IndexSource::Expr: DeviceGpuExpr<MIndex>,
        {
            type Runtime = ValueSource::Runtime;
            type Default = (ValueSource::Item,);
            type Output = DeviceZip1<DeviceVec<ValueSource::Runtime, ValueSource::Item>>;

            fn scatter_where_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
                indices: &IndexSource,
                stencil: Stencil,
                len: usize,
                default: Self::Default,
            ) -> Result<Self::Output, Error> {
                Ok(DeviceZip1 {
                    source: scatter_where_one_read::<ValueSource, IndexSource, Stencil, Pred>(
                        policy,
                        &self.$field,
                        indices,
                        &stencil,
                        len,
                        default.0,
                    )?,
                })
            }
        }
    };
}

impl_kernel_scatter_where_tuple1!(ZipView1<ValueSource>, source);
impl_kernel_scatter_where_tuple1!(DeviceZip1<ValueSource>, source);

impl<ValueSource, IndexSource, Stencil, Pred> KernelScatterWhereInput<IndexSource, Stencil, Pred>
    for (ValueSource,)
where
    ValueSource: KernelColumn + KernelColumnAt<S0>,
    IndexSource: KernelColumn<Runtime = ValueSource::Runtime, Item = MIndex> + KernelColumnAt<S0>,
    ZipView1<ValueSource>:
        KernelScatterWhereInput<IndexSource, Stencil, Pred, Runtime = ValueSource::Runtime>,
{
    type Runtime = ValueSource::Runtime;
    type Default =
        <ZipView1<ValueSource> as KernelScatterWhereInput<IndexSource, Stencil, Pred>>::Default;
    type Output =
        <ZipView1<ValueSource> as KernelScatterWhereInput<IndexSource, Stencil, Pred>>::Output;

    fn scatter_where_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: &IndexSource,
        stencil: Stencil,
        len: usize,
        default: Self::Default,
    ) -> Result<Self::Output, Error> {
        <ZipView1<ValueSource> as KernelScatterWhereInput<IndexSource, Stencil, Pred>>::scatter_where_read(
            ZipView1 { source: self.0 },
            policy,
            indices,
            stencil,
            len,
            default,
        )
    }
}

macro_rules! impl_kernel_scatter_where_tuple2 {
    ($target:ty, $out:ident, $left:tt, $right:tt) => {
        impl<Left, Right, IndexSource, Stencil, Pred>
            KernelScatterWhereInput<IndexSource, Stencil, Pred> for $target
        where
            Left: KernelColumn + KernelColumnAt<S0>,
            Right: KernelColumn<Runtime = Left::Runtime> + KernelColumnAt<S0>,
            IndexSource: KernelColumn<Runtime = Left::Runtime, Item = MIndex> + KernelColumnAt<S0>,
            Stencil: crate::detail::api::SelectionStencil<Pred, Runtime = Left::Runtime>,
            Left::Item: MStorageElement + 'static,
            Right::Item: MStorageElement + 'static,
            Left::Expr: DeviceGpuExpr<Left::Item>,
            Right::Expr: DeviceGpuExpr<Right::Item>,
            IndexSource::Expr: DeviceGpuExpr<MIndex>,
        {
            type Runtime = Left::Runtime;
            type Default = (Left::Item, Right::Item);
            type Output =
                $out<DeviceVec<Left::Runtime, Left::Item>, DeviceVec<Left::Runtime, Right::Item>>;

            fn scatter_where_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
                indices: &IndexSource,
                stencil: Stencil,
                len: usize,
                default: Self::Default,
            ) -> Result<Self::Output, Error> {
                validate_columns2(&self.$left, &self.$right)?;
                let left = scatter_where_one_read::<Left, IndexSource, Stencil, Pred>(
                    policy,
                    &self.$left,
                    indices,
                    &stencil,
                    len,
                    default.0,
                )?;
                let right = scatter_where_one_read::<Right, IndexSource, Stencil, Pred>(
                    policy,
                    &self.$right,
                    indices,
                    &stencil,
                    len,
                    default.1,
                )?;
                Ok($out { left, right })
            }
        }
    };
}

impl_kernel_scatter_where_tuple2!(ZipView2<Left, Right>, DeviceZip2, left, right);
impl_kernel_scatter_where_tuple2!(DeviceZip2<Left, Right>, DeviceZip2, left, right);

impl<Left, Right, IndexSource, Stencil, Pred> KernelScatterWhereInput<IndexSource, Stencil, Pred>
    for (Left, Right)
where
    IndexSource: KernelColumn<Item = MIndex> + KernelColumnAt<S0>,
    ZipView2<Left, Right>: KernelScatterWhereInput<IndexSource, Stencil, Pred>,
{
    type Runtime =
        <ZipView2<Left, Right> as KernelScatterWhereInput<IndexSource, Stencil, Pred>>::Runtime;
    type Default =
        <ZipView2<Left, Right> as KernelScatterWhereInput<IndexSource, Stencil, Pred>>::Default;
    type Output =
        <ZipView2<Left, Right> as KernelScatterWhereInput<IndexSource, Stencil, Pred>>::Output;

    fn scatter_where_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: &IndexSource,
        stencil: Stencil,
        len: usize,
        default: Self::Default,
    ) -> Result<Self::Output, Error> {
        <ZipView2<Left, Right> as KernelScatterWhereInput<IndexSource, Stencil, Pred>>::scatter_where_read(
            ZipView2 {
                left: self.0,
                right: self.1,
            },
            policy,
            indices,
            stencil,
            len,
            default,
        )
    }
}

macro_rules! impl_kernel_scatter_where_tuple3 {
    ($target:ty, $out:ident, $first:tt, $second:tt, $third:tt) => {
        impl<First, Second, Third, IndexSource, Stencil, Pred>
            KernelScatterWhereInput<IndexSource, Stencil, Pred> for $target
        where
            First: KernelColumn + KernelColumnAt<S0>,
            Second: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
            Third: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
            IndexSource: KernelColumn<Runtime = First::Runtime, Item = MIndex> + KernelColumnAt<S0>,
            Stencil: crate::detail::api::SelectionStencil<Pred, Runtime = First::Runtime>,
            First::Item: MStorageElement + 'static,
            Second::Item: MStorageElement + 'static,
            Third::Item: MStorageElement + 'static,
            First::Expr: DeviceGpuExpr<First::Item>,
            Second::Expr: DeviceGpuExpr<Second::Item>,
            Third::Expr: DeviceGpuExpr<Third::Item>,
            IndexSource::Expr: DeviceGpuExpr<MIndex>,
        {
            type Runtime = First::Runtime;
            type Default = (First::Item, Second::Item, Third::Item);
            type Output = $out<
                DeviceVec<First::Runtime, First::Item>,
                DeviceVec<First::Runtime, Second::Item>,
                DeviceVec<First::Runtime, Third::Item>,
            >;

            fn scatter_where_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
                indices: &IndexSource,
                stencil: Stencil,
                len: usize,
                default: Self::Default,
            ) -> Result<Self::Output, Error> {
                validate_columns3(&self.$first, &self.$second, &self.$third)?;
                let first = scatter_where_one_read::<First, IndexSource, Stencil, Pred>(
                    policy,
                    &self.$first,
                    indices,
                    &stencil,
                    len,
                    default.0,
                )?;
                let second = scatter_where_one_read::<Second, IndexSource, Stencil, Pred>(
                    policy,
                    &self.$second,
                    indices,
                    &stencil,
                    len,
                    default.1,
                )?;
                let third = scatter_where_one_read::<Third, IndexSource, Stencil, Pred>(
                    policy,
                    &self.$third,
                    indices,
                    &stencil,
                    len,
                    default.2,
                )?;
                Ok($out {
                    first,
                    second,
                    third,
                })
            }
        }
    };
}

impl_kernel_scatter_where_tuple3!(
    ZipView3<First, Second, Third>,
    DeviceZip3,
    first,
    second,
    third
);
impl_kernel_scatter_where_tuple3!(
    DeviceZip3<First, Second, Third>,
    DeviceZip3,
    first,
    second,
    third
);

impl<First, Second, Third, IndexSource, Stencil, Pred>
    KernelScatterWhereInput<IndexSource, Stencil, Pred> for (First, Second, Third)
where
    IndexSource: KernelColumn<Item = MIndex> + KernelColumnAt<S0>,
    ZipView3<First, Second, Third>: KernelScatterWhereInput<IndexSource, Stencil, Pred>,
{
    type Runtime = <ZipView3<First, Second, Third> as KernelScatterWhereInput<
        IndexSource,
        Stencil,
        Pred,
    >>::Runtime;
    type Default = <ZipView3<First, Second, Third> as KernelScatterWhereInput<
        IndexSource,
        Stencil,
        Pred,
    >>::Default;
    type Output = <ZipView3<First, Second, Third> as KernelScatterWhereInput<
        IndexSource,
        Stencil,
        Pred,
    >>::Output;

    fn scatter_where_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: &IndexSource,
        stencil: Stencil,
        len: usize,
        default: Self::Default,
    ) -> Result<Self::Output, Error> {
        <ZipView3<First, Second, Third> as KernelScatterWhereInput<IndexSource, Stencil, Pred>>::scatter_where_read(
            ZipView3 {
                first: self.0,
                second: self.1,
                third: self.2,
            },
            policy,
            indices,
            stencil,
            len,
            default,
        )
    }
}
