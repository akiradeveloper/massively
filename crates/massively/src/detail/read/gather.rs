use super::*;

#[allow(dead_code)]
pub(crate) trait KernelGatherInput<IndexSource>: Sized
where
    IndexSource: KernelColumn<Item = MIndex> + KernelColumnAt<S0>,
{
    type Runtime: Runtime;
    type Output;

    fn gather_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: &IndexSource,
    ) -> Result<Self::Output, Error>;
}

#[allow(dead_code)]
pub(crate) trait KernelGatherWhereInput<IndexSource, Stencil, Pred>: Sized
where
    IndexSource: KernelColumn<Item = MIndex> + KernelColumnAt<S0>,
{
    type Runtime: Runtime;
    type Output;
    type Default;

    fn gather_where_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: &IndexSource,
        stencil: Stencil,
        default: Self::Default,
    ) -> Result<Self::Output, Error>;
}

impl<InputSource, IndexSource> KernelGatherInput<IndexSource> for InputSource
where
    InputSource: KernelColumn + KernelColumnAt<S0>,
    IndexSource: KernelColumn<Runtime = InputSource::Runtime, Item = MIndex> + KernelColumnAt<S0>,
    InputSource::Item: Scalar + 'static,
    InputSource::Expr: GpuExpr<InputSource::Item>,
    IndexSource::Expr: DeviceGpuExpr<MIndex>,
{
    type Runtime = InputSource::Runtime;
    type Output = DeviceSoA1<DeviceVec<InputSource::Runtime, InputSource::Item>>;

    fn gather_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: &IndexSource,
    ) -> Result<Self::Output, Error> {
        let index_values =
            crate::detail::api::MaterializePayloadApply::collect_expr(policy, indices)?;
        let control = crate::detail::control::PermutationControl::from_indices(&index_values)?;
        let apply = crate::detail::api::PermutationPayloadApply::new(&control);
        Ok(DeviceSoA1 {
            source: apply.apply_expr(policy, &self)?,
        })
    }
}

macro_rules! impl_kernel_gather_tuple1 {
    ($target:ty, $field:tt) => {
        impl<InputSource, IndexSource> KernelGatherInput<IndexSource> for $target
        where
            InputSource: KernelColumn + KernelColumnAt<S0>,
            IndexSource:
                KernelColumn<Runtime = InputSource::Runtime, Item = MIndex> + KernelColumnAt<S0>,
            InputSource::Item: Scalar + 'static,
            InputSource::Expr: GpuExpr<InputSource::Item>,
            IndexSource::Expr: DeviceGpuExpr<MIndex>,
        {
            type Runtime = InputSource::Runtime;
            type Output = DeviceSoA1<DeviceVec<InputSource::Runtime, InputSource::Item>>;

            fn gather_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
                indices: &IndexSource,
            ) -> Result<Self::Output, Error> {
                <InputSource as KernelGatherInput<IndexSource>>::gather_read(
                    self.$field,
                    policy,
                    indices,
                )
            }
        }
    };
}

impl_kernel_gather_tuple1!(SoAView1<InputSource>, source);
impl_kernel_gather_tuple1!(DeviceSoA1<InputSource>, source);

impl<InputSource, IndexSource> KernelGatherInput<IndexSource> for (InputSource,)
where
    InputSource: KernelGatherInput<IndexSource>,
    IndexSource: KernelColumn<Item = MIndex> + KernelColumnAt<S0>,
{
    type Runtime = <InputSource as KernelGatherInput<IndexSource>>::Runtime;
    type Output = <InputSource as KernelGatherInput<IndexSource>>::Output;

    fn gather_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: &IndexSource,
    ) -> Result<Self::Output, Error> {
        self.0.gather_read(policy, indices)
    }
}

macro_rules! impl_kernel_gather_tuple2 {
    ($target:ty, $out:ident, $left:tt, $right:tt) => {
        impl<Left, Right, IndexSource> KernelGatherInput<IndexSource> for $target
        where
            Left: KernelColumn + KernelColumnAt<S0>,
            Right: KernelColumn<Runtime = Left::Runtime> + KernelColumnAt<S0>,
            IndexSource: KernelColumn<Runtime = Left::Runtime, Item = MIndex> + KernelColumnAt<S0>,
            Left::Item: Scalar + 'static,
            Right::Item: Scalar + 'static,
            Left::Expr: GpuExpr<Left::Item>,
            Right::Expr: GpuExpr<Right::Item>,
            IndexSource::Expr: DeviceGpuExpr<MIndex>,
        {
            type Runtime = Left::Runtime;
            type Output =
                $out<DeviceVec<Left::Runtime, Left::Item>, DeviceVec<Left::Runtime, Right::Item>>;

            fn gather_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
                indices: &IndexSource,
            ) -> Result<Self::Output, Error> {
                validate_columns2(&self.$left, &self.$right)?;
                let index_values =
                    crate::detail::api::MaterializePayloadApply::collect_expr(policy, indices)?;
                let control =
                    crate::detail::control::PermutationControl::from_indices(&index_values)?;
                let apply = crate::detail::api::PermutationPayloadApply::new(&control);
                let (left, right) = apply.apply_expr2(policy, &self.$left, &self.$right)?;
                Ok($out { left, right })
            }
        }
    };
}

impl_kernel_gather_tuple2!(SoAView2<Left, Right>, DeviceSoA2, left, right);
impl_kernel_gather_tuple2!(DeviceSoA2<Left, Right>, DeviceSoA2, left, right);

impl<Left, Right, IndexSource> KernelGatherInput<IndexSource> for (Left, Right)
where
    IndexSource: KernelColumn<Item = MIndex> + KernelColumnAt<S0>,
    SoAView2<Left, Right>: KernelGatherInput<IndexSource>,
{
    type Runtime = <SoAView2<Left, Right> as KernelGatherInput<IndexSource>>::Runtime;
    type Output = <SoAView2<Left, Right> as KernelGatherInput<IndexSource>>::Output;

    fn gather_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: &IndexSource,
    ) -> Result<Self::Output, Error> {
        <SoAView2<Left, Right> as KernelGatherInput<IndexSource>>::gather_read(
            SoAView2 {
                left: self.0,
                right: self.1,
            },
            policy,
            indices,
        )
    }
}

macro_rules! impl_kernel_gather_tuple3 {
    ($target:ty, $out:ident, $first:tt, $second:tt, $third:tt) => {
        impl<First, Second, Third, IndexSource> KernelGatherInput<IndexSource> for $target
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
            IndexSource::Expr: DeviceGpuExpr<MIndex>,
        {
            type Runtime = First::Runtime;
            type Output = $out<
                DeviceVec<First::Runtime, First::Item>,
                DeviceVec<First::Runtime, Second::Item>,
                DeviceVec<First::Runtime, Third::Item>,
            >;

            fn gather_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
                indices: &IndexSource,
            ) -> Result<Self::Output, Error> {
                validate_columns3(&self.$first, &self.$second, &self.$third)?;
                let index_values =
                    crate::detail::api::MaterializePayloadApply::collect_expr(policy, indices)?;
                let control =
                    crate::detail::control::PermutationControl::from_indices(&index_values)?;
                let apply = crate::detail::api::PermutationPayloadApply::new(&control);
                let (first, second, third) =
                    apply.apply_expr3(policy, &self.$first, &self.$second, &self.$third)?;
                Ok($out {
                    first,
                    second,
                    third,
                })
            }
        }
    };
}

impl_kernel_gather_tuple3!(SoAView3<First, Second, Third>, DeviceSoA3, first, second, third);
impl_kernel_gather_tuple3!(DeviceSoA3<First, Second, Third>, DeviceSoA3, first, second, third);

impl<First, Second, Third, IndexSource> KernelGatherInput<IndexSource> for (First, Second, Third)
where
    IndexSource: KernelColumn<Item = MIndex> + KernelColumnAt<S0>,
    SoAView3<First, Second, Third>: KernelGatherInput<IndexSource>,
{
    type Runtime = <SoAView3<First, Second, Third> as KernelGatherInput<IndexSource>>::Runtime;
    type Output = <SoAView3<First, Second, Third> as KernelGatherInput<IndexSource>>::Output;

    fn gather_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: &IndexSource,
    ) -> Result<Self::Output, Error> {
        <SoAView3<First, Second, Third> as KernelGatherInput<IndexSource>>::gather_read(
            SoAView3 {
                first: self.0,
                second: self.1,
                third: self.2,
            },
            policy,
            indices,
        )
    }
}

#[allow(dead_code)]
fn gather_where_one_read<InputSource, IndexSource, Stencil, Pred>(
    policy: &CubePolicy<InputSource::Runtime>,
    input: &InputSource,
    indices: &IndexSource,
    stencil: &Stencil,
    default: InputSource::Item,
) -> Result<DeviceVec<InputSource::Runtime, InputSource::Item>, Error>
where
    InputSource: KernelColumn + KernelColumnAt<S0>,
    IndexSource: KernelColumn<Runtime = InputSource::Runtime, Item = MIndex> + KernelColumnAt<S0>,
    Stencil: crate::detail::api::SelectionStencil<Pred, Runtime = InputSource::Runtime>,
    InputSource::Item: Scalar + 'static,
    InputSource::Expr: DeviceGpuExpr<InputSource::Item>,
    IndexSource::Expr: DeviceGpuExpr<MIndex>,
{
    <InputSource as KernelColumn>::validate(input)?;
    <IndexSource as KernelColumn>::validate(indices)?;
    ensure_same_len(<IndexSource as KernelColumn>::len(indices), stencil.len())?;
    let flags = stencil.selection_flags_with_policy(policy, false)?;

    let len = <IndexSource as KernelColumn>::len(indices);
    let output = primitive_range::filled(policy, len, default)?;
    let num_blocks = len.div_ceil(BLOCK_GATHER_WHERE_SIZE as usize);
    let num_blocks_u32 =
        u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
    let client = policy.client();
    let input_bindings = <InputSource as KernelColumn>::stage(input, policy)?;
    let index_bindings = <IndexSource as KernelColumn>::stage(indices, policy)?;
    let input_slot0 = input_bindings.slot_or_first(0);
    let input_slot1 = input_bindings.slot_or_first(1);
    let input_slot2 = input_bindings.slot_or_first(2);
    let input_slot3 = input_bindings.slot_or_first(3);
    let index_slot0 = index_bindings.slot_or_first(0);
    let index_slot1 = index_bindings.slot_or_first(1);
    let index_slot2 = index_bindings.slot_or_first(2);
    let index_slot3 = index_bindings.slot_or_first(3);
    let input_slot_offsets = input_bindings.slot_offsets_handle(client)?;
    let index_slot_offsets = index_bindings.slot_offsets_handle(client)?;

    if len != 0 {
        unsafe {
            gather_if_flags_kernel::launch_unchecked::<
                InputSource::Item,
                InputSource::Expr,
                IndexSource::Expr,
                InputSource::Runtime,
            >(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(BLOCK_GATHER_WHERE_SIZE),
                BufferArg::from_raw_parts(input_slot0.0.clone(), input_slot0.1),
                BufferArg::from_raw_parts(input_slot1.0.clone(), input_slot1.1),
                BufferArg::from_raw_parts(input_slot2.0.clone(), input_slot2.1),
                BufferArg::from_raw_parts(input_slot3.0.clone(), input_slot3.1),
                BufferArg::from_raw_parts(input_slot_offsets.clone(), 4),
                BufferArg::from_raw_parts(index_slot0.0.clone(), index_slot0.1),
                BufferArg::from_raw_parts(index_slot1.0.clone(), index_slot1.1),
                BufferArg::from_raw_parts(index_slot2.0.clone(), index_slot2.1),
                BufferArg::from_raw_parts(index_slot3.0.clone(), index_slot3.1),
                BufferArg::from_raw_parts(index_slot_offsets.clone(), 4),
                BufferArg::from_raw_parts(flags.flag.clone(), flags.len),
                BufferArg::from_raw_parts(output.handle.clone(), output.len()),
            );
        }
    }

    Ok(output)
}

impl<InputSource, IndexSource, Stencil, Pred> KernelGatherWhereInput<IndexSource, Stencil, Pred>
    for InputSource
where
    InputSource: KernelColumn + KernelColumnAt<S0>,
    IndexSource: KernelColumn<Runtime = InputSource::Runtime, Item = MIndex> + KernelColumnAt<S0>,
    Stencil: crate::detail::api::SelectionStencil<Pred, Runtime = InputSource::Runtime>,
    InputSource::Item: Scalar + 'static,
    InputSource::Expr: DeviceGpuExpr<InputSource::Item>,
    IndexSource::Expr: DeviceGpuExpr<MIndex>,
{
    type Runtime = InputSource::Runtime;
    type Output = DeviceSoA1<DeviceVec<InputSource::Runtime, InputSource::Item>>;
    type Default = InputSource::Item;

    fn gather_where_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: &IndexSource,
        stencil: Stencil,
        default: Self::Default,
    ) -> Result<Self::Output, Error> {
        Ok(DeviceSoA1 {
            source: gather_where_one_read::<InputSource, IndexSource, Stencil, Pred>(
                policy, &self, indices, &stencil, default,
            )?,
        })
    }
}

macro_rules! impl_kernel_gather_where_tuple1 {
    ($target:ty, $field:tt) => {
        impl<InputSource, IndexSource, Stencil, Pred>
            KernelGatherWhereInput<IndexSource, Stencil, Pred> for $target
        where
            InputSource: KernelColumn + KernelColumnAt<S0>,
            IndexSource:
                KernelColumn<Runtime = InputSource::Runtime, Item = MIndex> + KernelColumnAt<S0>,
            Stencil: crate::detail::api::SelectionStencil<Pred, Runtime = InputSource::Runtime>,
            InputSource::Item: Scalar + 'static,
            InputSource::Expr: DeviceGpuExpr<InputSource::Item>,
            IndexSource::Expr: DeviceGpuExpr<MIndex>,
        {
            type Runtime = InputSource::Runtime;
            type Output = DeviceSoA1<DeviceVec<InputSource::Runtime, InputSource::Item>>;
            type Default = (InputSource::Item,);

            fn gather_where_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
                indices: &IndexSource,
                stencil: Stencil,
                default: Self::Default,
            ) -> Result<Self::Output, Error> {
                Ok(DeviceSoA1 {
                    source: gather_where_one_read::<InputSource, IndexSource, Stencil, Pred>(
                        policy,
                        &self.$field,
                        indices,
                        &stencil,
                        default.0,
                    )?,
                })
            }
        }
    };
}

impl_kernel_gather_where_tuple1!(SoAView1<InputSource>, source);
impl_kernel_gather_where_tuple1!(DeviceSoA1<InputSource>, source);

impl<InputSource, IndexSource, Stencil, Pred> KernelGatherWhereInput<IndexSource, Stencil, Pred>
    for (InputSource,)
where
    InputSource: KernelColumn + KernelColumnAt<S0>,
    IndexSource: KernelColumn<Runtime = InputSource::Runtime, Item = MIndex> + KernelColumnAt<S0>,
    SoAView1<InputSource>:
        KernelGatherWhereInput<IndexSource, Stencil, Pred, Runtime = InputSource::Runtime>,
{
    type Runtime = InputSource::Runtime;
    type Output =
        <SoAView1<InputSource> as KernelGatherWhereInput<IndexSource, Stencil, Pred>>::Output;
    type Default =
        <SoAView1<InputSource> as KernelGatherWhereInput<IndexSource, Stencil, Pred>>::Default;

    fn gather_where_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: &IndexSource,
        stencil: Stencil,
        default: Self::Default,
    ) -> Result<Self::Output, Error> {
        <SoAView1<InputSource> as KernelGatherWhereInput<IndexSource, Stencil, Pred>>::gather_where_read(
            SoAView1 { source: self.0 },
            policy,
            indices,
            stencil,
            default,
        )
    }
}

macro_rules! impl_kernel_gather_where_tuple2 {
    ($target:ty, $out:ident, $left:tt, $right:tt) => {
        impl<Left, Right, IndexSource, Stencil, Pred>
            KernelGatherWhereInput<IndexSource, Stencil, Pred> for $target
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
            type Output =
                $out<DeviceVec<Left::Runtime, Left::Item>, DeviceVec<Left::Runtime, Right::Item>>;
            type Default = (Left::Item, Right::Item);

            fn gather_where_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
                indices: &IndexSource,
                stencil: Stencil,
                default: Self::Default,
            ) -> Result<Self::Output, Error> {
                validate_columns2(&self.$left, &self.$right)?;
                let left = gather_where_one_read::<Left, IndexSource, Stencil, Pred>(
                    policy,
                    &self.$left,
                    indices,
                    &stencil,
                    default.0,
                )?;
                let right = gather_where_one_read::<Right, IndexSource, Stencil, Pred>(
                    policy,
                    &self.$right,
                    indices,
                    &stencil,
                    default.1,
                )?;
                Ok($out { left, right })
            }
        }
    };
}

impl_kernel_gather_where_tuple2!(SoAView2<Left, Right>, DeviceSoA2, left, right);
impl_kernel_gather_where_tuple2!(DeviceSoA2<Left, Right>, DeviceSoA2, left, right);

impl<Left, Right, IndexSource, Stencil, Pred> KernelGatherWhereInput<IndexSource, Stencil, Pred>
    for (Left, Right)
where
    IndexSource: KernelColumn<Item = MIndex> + KernelColumnAt<S0>,
    SoAView2<Left, Right>: KernelGatherWhereInput<IndexSource, Stencil, Pred>,
{
    type Runtime =
        <SoAView2<Left, Right> as KernelGatherWhereInput<IndexSource, Stencil, Pred>>::Runtime;
    type Output =
        <SoAView2<Left, Right> as KernelGatherWhereInput<IndexSource, Stencil, Pred>>::Output;
    type Default =
        <SoAView2<Left, Right> as KernelGatherWhereInput<IndexSource, Stencil, Pred>>::Default;

    fn gather_where_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: &IndexSource,
        stencil: Stencil,
        default: Self::Default,
    ) -> Result<Self::Output, Error> {
        <SoAView2<Left, Right> as KernelGatherWhereInput<IndexSource, Stencil, Pred>>::gather_where_read(
            SoAView2 {
                left: self.0,
                right: self.1,
            },
            policy,
            indices,
            stencil,
            default,
        )
    }
}

macro_rules! impl_kernel_gather_where_tuple3 {
    ($target:ty, $out:ident, $first:tt, $second:tt, $third:tt) => {
        impl<First, Second, Third, IndexSource, Stencil, Pred>
            KernelGatherWhereInput<IndexSource, Stencil, Pred> for $target
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
            type Output = $out<
                DeviceVec<First::Runtime, First::Item>,
                DeviceVec<First::Runtime, Second::Item>,
                DeviceVec<First::Runtime, Third::Item>,
            >;
            type Default = (First::Item, Second::Item, Third::Item);

            fn gather_where_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
                indices: &IndexSource,
                stencil: Stencil,
                default: Self::Default,
            ) -> Result<Self::Output, Error> {
                validate_columns3(&self.$first, &self.$second, &self.$third)?;
                let first = gather_where_one_read::<First, IndexSource, Stencil, Pred>(
                    policy,
                    &self.$first,
                    indices,
                    &stencil,
                    default.0,
                )?;
                let second = gather_where_one_read::<Second, IndexSource, Stencil, Pred>(
                    policy,
                    &self.$second,
                    indices,
                    &stencil,
                    default.1,
                )?;
                let third = gather_where_one_read::<Third, IndexSource, Stencil, Pred>(
                    policy,
                    &self.$third,
                    indices,
                    &stencil,
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

impl_kernel_gather_where_tuple3!(SoAView3<First, Second, Third>, DeviceSoA3, first, second, third);
impl_kernel_gather_where_tuple3!(DeviceSoA3<First, Second, Third>, DeviceSoA3, first, second, third);

impl<First, Second, Third, IndexSource, Stencil, Pred>
    KernelGatherWhereInput<IndexSource, Stencil, Pred> for (First, Second, Third)
where
    IndexSource: KernelColumn<Item = MIndex> + KernelColumnAt<S0>,
    SoAView3<First, Second, Third>: KernelGatherWhereInput<IndexSource, Stencil, Pred>,
{
    type Runtime = <SoAView3<First, Second, Third> as KernelGatherWhereInput<
        IndexSource,
        Stencil,
        Pred,
    >>::Runtime;
    type Output = <SoAView3<First, Second, Third> as KernelGatherWhereInput<
        IndexSource,
        Stencil,
        Pred,
    >>::Output;
    type Default = <SoAView3<First, Second, Third> as KernelGatherWhereInput<
        IndexSource,
        Stencil,
        Pred,
    >>::Default;

    fn gather_where_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        indices: &IndexSource,
        stencil: Stencil,
        default: Self::Default,
    ) -> Result<Self::Output, Error> {
        <SoAView3<First, Second, Third> as KernelGatherWhereInput<IndexSource, Stencil, Pred>>::gather_where_read(
            SoAView3 {
                first: self.0,
                second: self.1,
                third: self.2,
            },
            policy,
            indices,
            stencil,
            default,
        )
    }
}
