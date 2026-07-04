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
    ValueSource::Item: Scalar + 'static,
    ValueSource::Expr: GpuExpr<ValueSource::Item>,
    IndexSource::Expr: GpuExpr<MIndex>,
{
    let initial = primitive_range::filled(policy, len, default)?;
    let output = DeviceColumnMutView::from_slice(&initial, 0, len);
    crate::detail::api::device_expr_scatter_into_with_policy(policy, values, indices, &output)?;
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
    ValueSource::Item: Scalar + 'static,
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
    let flags = stencil.selection_flags_with_policy(policy, false)?;
    let input_len = <ValueSource as KernelColumn>::len(values);
    let block_count = input_len.div_ceil(BLOCK_SCATTER_WHERE_SIZE as usize);
    let block_count_u32 =
        u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;
    let value_bindings = <ValueSource as KernelColumn>::stage(values, policy)?;
    let index_bindings = <IndexSource as KernelColumn>::stage(indices, policy)?;
    let value_slot0 = value_bindings.slot_or_first(0);
    let value_slot1 = value_bindings.slot_or_first(1);
    let value_slot2 = value_bindings.slot_or_first(2);
    let value_slot3 = value_bindings.slot_or_first(3);
    let index_slot0 = index_bindings.slot_or_first(0);
    let index_slot1 = index_bindings.slot_or_first(1);
    let index_slot2 = index_bindings.slot_or_first(2);
    let index_slot3 = index_bindings.slot_or_first(3);
    let value_slot_offsets = value_bindings.slot_offsets_handle(policy.client())?;
    let index_slot_offsets = index_bindings.slot_offsets_handle(policy.client())?;
    if input_len != 0 {
        unsafe {
            scatter_if_flags_kernel::launch_unchecked::<
                ValueSource::Item,
                ValueSource::Expr,
                IndexSource::Expr,
                ValueSource::Runtime,
            >(
                policy.client(),
                CubeCount::Static(block_count_u32, 1, 1),
                CubeDim::new_1d(BLOCK_SCATTER_WHERE_SIZE),
                BufferArg::from_raw_parts(value_slot0.0.clone(), value_slot0.1),
                BufferArg::from_raw_parts(value_slot1.0.clone(), value_slot1.1),
                BufferArg::from_raw_parts(value_slot2.0.clone(), value_slot2.1),
                BufferArg::from_raw_parts(value_slot3.0.clone(), value_slot3.1),
                BufferArg::from_raw_parts(value_slot_offsets.clone(), 4),
                BufferArg::from_raw_parts(index_slot0.0.clone(), index_slot0.1),
                BufferArg::from_raw_parts(index_slot1.0.clone(), index_slot1.1),
                BufferArg::from_raw_parts(index_slot2.0.clone(), index_slot2.1),
                BufferArg::from_raw_parts(index_slot3.0.clone(), index_slot3.1),
                BufferArg::from_raw_parts(index_slot_offsets.clone(), 4),
                BufferArg::from_raw_parts(flags.flag.clone(), flags.len),
                BufferArg::from_raw_parts(initial.handle.clone(), initial.len()),
            );
        }
    }
    Ok(initial)
}

impl<ValueSource, IndexSource> KernelScatterInput<IndexSource> for ValueSource
where
    ValueSource: KernelColumn + KernelColumnAt<S0>,
    IndexSource: KernelColumn<Runtime = ValueSource::Runtime, Item = MIndex> + KernelColumnAt<S0>,
    ValueSource::Item: Scalar + 'static,
    ValueSource::Expr: GpuExpr<ValueSource::Item>,
    IndexSource::Expr: GpuExpr<MIndex>,
{
    type Runtime = ValueSource::Runtime;
    type Default = ValueSource::Item;
    type Output = DeviceSoA1<DeviceVec<ValueSource::Runtime, ValueSource::Item>>;

    fn scatter_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: &IndexSource,
        len: usize,
        default: Self::Default,
    ) -> Result<Self::Output, Error> {
        Ok(DeviceSoA1 {
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
            ValueSource::Item: Scalar + 'static,
            ValueSource::Expr: GpuExpr<ValueSource::Item>,
            IndexSource::Expr: GpuExpr<MIndex>,
        {
            type Runtime = ValueSource::Runtime;
            type Default = (ValueSource::Item,);
            type Output = DeviceSoA1<DeviceVec<ValueSource::Runtime, ValueSource::Item>>;

            fn scatter_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
                indices: &IndexSource,
                len: usize,
                default: Self::Default,
            ) -> Result<Self::Output, Error> {
                Ok(DeviceSoA1 {
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

impl_kernel_scatter_tuple1!(SoAView1<ValueSource>, source);
impl_kernel_scatter_tuple1!(DeviceSoA1<ValueSource>, source);

impl<ValueSource, IndexSource> KernelScatterInput<IndexSource> for (ValueSource,)
where
    ValueSource: KernelColumn + KernelColumnAt<S0>,
    IndexSource: KernelColumn<Runtime = ValueSource::Runtime, Item = MIndex> + KernelColumnAt<S0>,
    SoAView1<ValueSource>: KernelScatterInput<IndexSource, Runtime = ValueSource::Runtime>,
{
    type Runtime = ValueSource::Runtime;
    type Default = <SoAView1<ValueSource> as KernelScatterInput<IndexSource>>::Default;
    type Output = <SoAView1<ValueSource> as KernelScatterInput<IndexSource>>::Output;

    fn scatter_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: &IndexSource,
        len: usize,
        default: Self::Default,
    ) -> Result<Self::Output, Error> {
        <SoAView1<ValueSource> as KernelScatterInput<IndexSource>>::scatter_read(
            SoAView1 { source: self.0 },
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
            Left::Item: Scalar + 'static,
            Right::Item: Scalar + 'static,
            Left::Expr: GpuExpr<Left::Item>,
            Right::Expr: GpuExpr<Right::Item>,
            IndexSource::Expr: GpuExpr<MIndex>,
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

impl_kernel_scatter_tuple2!(SoAView2<Left, Right>, DeviceSoA2, left, right);
impl_kernel_scatter_tuple2!(DeviceSoA2<Left, Right>, DeviceSoA2, left, right);

impl<Left, Right, IndexSource> KernelScatterInput<IndexSource> for (Left, Right)
where
    IndexSource: KernelColumn<Item = MIndex> + KernelColumnAt<S0>,
    SoAView2<Left, Right>: KernelScatterInput<IndexSource>,
{
    type Runtime = <SoAView2<Left, Right> as KernelScatterInput<IndexSource>>::Runtime;
    type Default = <SoAView2<Left, Right> as KernelScatterInput<IndexSource>>::Default;
    type Output = <SoAView2<Left, Right> as KernelScatterInput<IndexSource>>::Output;

    fn scatter_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: &IndexSource,
        len: usize,
        default: Self::Default,
    ) -> Result<Self::Output, Error> {
        <SoAView2<Left, Right> as KernelScatterInput<IndexSource>>::scatter_read(
            SoAView2 {
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
            First::Item: Scalar + 'static,
            Second::Item: Scalar + 'static,
            Third::Item: Scalar + 'static,
            First::Expr: GpuExpr<First::Item>,
            Second::Expr: GpuExpr<Second::Item>,
            Third::Expr: GpuExpr<Third::Item>,
            IndexSource::Expr: GpuExpr<MIndex>,
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

impl_kernel_scatter_tuple3!(SoAView3<First, Second, Third>, DeviceSoA3, first, second, third);
impl_kernel_scatter_tuple3!(DeviceSoA3<First, Second, Third>, DeviceSoA3, first, second, third);

impl<First, Second, Third, IndexSource> KernelScatterInput<IndexSource> for (First, Second, Third)
where
    IndexSource: KernelColumn<Item = MIndex> + KernelColumnAt<S0>,
    SoAView3<First, Second, Third>: KernelScatterInput<IndexSource>,
{
    type Runtime = <SoAView3<First, Second, Third> as KernelScatterInput<IndexSource>>::Runtime;
    type Default = <SoAView3<First, Second, Third> as KernelScatterInput<IndexSource>>::Default;
    type Output = <SoAView3<First, Second, Third> as KernelScatterInput<IndexSource>>::Output;

    fn scatter_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: &IndexSource,
        len: usize,
        default: Self::Default,
    ) -> Result<Self::Output, Error> {
        <SoAView3<First, Second, Third> as KernelScatterInput<IndexSource>>::scatter_read(
            SoAView3 {
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
    ValueSource::Item: Scalar + 'static,
    ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
    IndexSource::Expr: DeviceGpuExpr<MIndex>,
{
    type Runtime = ValueSource::Runtime;
    type Default = ValueSource::Item;
    type Output = DeviceSoA1<DeviceVec<ValueSource::Runtime, ValueSource::Item>>;

    fn scatter_where_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: &IndexSource,
        stencil: Stencil,
        len: usize,
        default: Self::Default,
    ) -> Result<Self::Output, Error> {
        Ok(DeviceSoA1 {
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
            ValueSource::Item: Scalar + 'static,
            ValueSource::Expr: DeviceGpuExpr<ValueSource::Item>,
            IndexSource::Expr: DeviceGpuExpr<MIndex>,
        {
            type Runtime = ValueSource::Runtime;
            type Default = (ValueSource::Item,);
            type Output = DeviceSoA1<DeviceVec<ValueSource::Runtime, ValueSource::Item>>;

            fn scatter_where_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
                indices: &IndexSource,
                stencil: Stencil,
                len: usize,
                default: Self::Default,
            ) -> Result<Self::Output, Error> {
                Ok(DeviceSoA1 {
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

impl_kernel_scatter_where_tuple1!(SoAView1<ValueSource>, source);
impl_kernel_scatter_where_tuple1!(DeviceSoA1<ValueSource>, source);

impl<ValueSource, IndexSource, Stencil, Pred> KernelScatterWhereInput<IndexSource, Stencil, Pred>
    for (ValueSource,)
where
    ValueSource: KernelColumn + KernelColumnAt<S0>,
    IndexSource: KernelColumn<Runtime = ValueSource::Runtime, Item = MIndex> + KernelColumnAt<S0>,
    SoAView1<ValueSource>:
        KernelScatterWhereInput<IndexSource, Stencil, Pred, Runtime = ValueSource::Runtime>,
{
    type Runtime = ValueSource::Runtime;
    type Default =
        <SoAView1<ValueSource> as KernelScatterWhereInput<IndexSource, Stencil, Pred>>::Default;
    type Output =
        <SoAView1<ValueSource> as KernelScatterWhereInput<IndexSource, Stencil, Pred>>::Output;

    fn scatter_where_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: &IndexSource,
        stencil: Stencil,
        len: usize,
        default: Self::Default,
    ) -> Result<Self::Output, Error> {
        <SoAView1<ValueSource> as KernelScatterWhereInput<IndexSource, Stencil, Pred>>::scatter_where_read(
            SoAView1 { source: self.0 },
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
            Left::Item: Scalar + 'static,
            Right::Item: Scalar + 'static,
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

impl_kernel_scatter_where_tuple2!(SoAView2<Left, Right>, DeviceSoA2, left, right);
impl_kernel_scatter_where_tuple2!(DeviceSoA2<Left, Right>, DeviceSoA2, left, right);

impl<Left, Right, IndexSource, Stencil, Pred> KernelScatterWhereInput<IndexSource, Stencil, Pred>
    for (Left, Right)
where
    IndexSource: KernelColumn<Item = MIndex> + KernelColumnAt<S0>,
    SoAView2<Left, Right>: KernelScatterWhereInput<IndexSource, Stencil, Pred>,
{
    type Runtime =
        <SoAView2<Left, Right> as KernelScatterWhereInput<IndexSource, Stencil, Pred>>::Runtime;
    type Default =
        <SoAView2<Left, Right> as KernelScatterWhereInput<IndexSource, Stencil, Pred>>::Default;
    type Output =
        <SoAView2<Left, Right> as KernelScatterWhereInput<IndexSource, Stencil, Pred>>::Output;

    fn scatter_where_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: &IndexSource,
        stencil: Stencil,
        len: usize,
        default: Self::Default,
    ) -> Result<Self::Output, Error> {
        <SoAView2<Left, Right> as KernelScatterWhereInput<IndexSource, Stencil, Pred>>::scatter_where_read(
            SoAView2 {
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
            First::Item: Scalar + 'static,
            Second::Item: Scalar + 'static,
            Third::Item: Scalar + 'static,
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
    SoAView3<First, Second, Third>,
    DeviceSoA3,
    first,
    second,
    third
);
impl_kernel_scatter_where_tuple3!(
    DeviceSoA3<First, Second, Third>,
    DeviceSoA3,
    first,
    second,
    third
);

impl<First, Second, Third, IndexSource, Stencil, Pred>
    KernelScatterWhereInput<IndexSource, Stencil, Pred> for (First, Second, Third)
where
    IndexSource: KernelColumn<Item = MIndex> + KernelColumnAt<S0>,
    SoAView3<First, Second, Third>: KernelScatterWhereInput<IndexSource, Stencil, Pred>,
{
    type Runtime = <SoAView3<First, Second, Third> as KernelScatterWhereInput<
        IndexSource,
        Stencil,
        Pred,
    >>::Runtime;
    type Default = <SoAView3<First, Second, Third> as KernelScatterWhereInput<
        IndexSource,
        Stencil,
        Pred,
    >>::Default;
    type Output = <SoAView3<First, Second, Third> as KernelScatterWhereInput<
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
        <SoAView3<First, Second, Third> as KernelScatterWhereInput<IndexSource, Stencil, Pred>>::scatter_where_read(
            SoAView3 {
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
