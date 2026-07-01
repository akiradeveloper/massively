use super::*;

pub(crate) trait KernelCopyWhereInput<Stencil, Pred>: Sized {
    type Runtime: Runtime;
    type Output;

    fn copy_where_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        stencil: Stencil,
    ) -> Result<Self::Output, Error>;
}

#[allow(dead_code)]
pub(crate) trait KernelReplaceWhereInput<Stencil, Pred>: Sized {
    type Runtime: Runtime;
    type Item;
    type Output;

    fn replace_where_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        replacement: Self::Item,
        stencil: Stencil,
    ) -> Result<Self::Output, Error>;
}

#[allow(dead_code)]
pub(crate) trait KernelSelectInput<Pred>: Sized {
    type Runtime: Runtime;
    type Output;
    type Env: cubecl::prelude::LaunchArg + Copy;

    fn select_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        invert: bool,
        env: <Self::Env as cubecl::prelude::LaunchArg>::RuntimeArg<Self::Runtime>,
    ) -> Result<Self::Output, Error>;
}

#[allow(dead_code)]
pub(crate) trait KernelPredicateQueryInput<Pred>: Sized {
    type Runtime: Runtime;
    type Env: cubecl::prelude::LaunchArg + Copy;

    fn count_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        invert: bool,
        env: <Self::Env as cubecl::prelude::LaunchArg>::RuntimeArg<Self::Runtime>,
    ) -> Result<usize, Error>;

    fn find_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        invert: bool,
        env: <Self::Env as cubecl::prelude::LaunchArg>::RuntimeArg<Self::Runtime>,
    ) -> Result<Option<usize>, Error>;
}

#[allow(dead_code)]
pub(crate) trait KernelPartitionInput<Pred>: Sized {
    type Runtime: Runtime;
    type Output;
    type SplitOutput;
    type Env: cubecl::prelude::LaunchArg + Copy;

    fn is_partitioned_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        env: <Self::Env as cubecl::prelude::LaunchArg>::RuntimeArg<Self::Runtime>,
    ) -> Result<bool, Error>;

    fn partition_copy_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        env: <Self::Env as cubecl::prelude::LaunchArg>::RuntimeArg<Self::Runtime>,
    ) -> Result<Self::SplitOutput, Error>;
}

#[allow(dead_code)]
pub(crate) trait KernelUniqueInput<Pred>: Sized {
    type Runtime: Runtime;
    type Output;

    fn unique_read(self, policy: &CubePolicy<Self::Runtime>) -> Result<Self::Output, Error>;
}

impl<Source, Stencil, Pred> KernelCopyWhereInput<Stencil, Pred> for Source
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Stencil: crate::detail::api::SelectionStencil<Pred, Runtime = Source::Runtime>,
    Source::Item: Scalar + 'static,
    Source::Expr: DeviceGpuExpr<Source::Item>,
{
    type Runtime = Source::Runtime;
    type Output = DeviceSoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn copy_where_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        stencil: Stencil,
    ) -> Result<Self::Output, Error> {
        <Source as KernelColumn>::validate(&self)?;
        ensure_same_len(<Source as KernelColumn>::len(&self), stencil.len())?;
        let handles = stencil.selection_handles_with_policy(policy, false)?;
        let count = select::selected_count(policy, &handles)?;
        Ok(DeviceSoA1 {
            source: crate::detail::api::device_expr_compact_with_selection_with_policy(
                policy, &self, &handles, count,
            )?,
        })
    }
}

macro_rules! impl_kernel_copy_where_tuple1 {
    ($target:ty, $field:tt) => {
        impl<Source, Stencil, Pred> KernelCopyWhereInput<Stencil, Pred> for $target
        where
            Source: KernelColumn + KernelColumnAt<S0>,
            Stencil: crate::detail::api::SelectionStencil<Pred, Runtime = Source::Runtime>,
            Source::Item: Scalar + 'static,
            Source::Expr: DeviceGpuExpr<Source::Item>,
        {
            type Runtime = Source::Runtime;
            type Output = DeviceSoA1<DeviceVec<Source::Runtime, Source::Item>>;

            fn copy_where_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
                stencil: Stencil,
            ) -> Result<Self::Output, Error> {
                <Source as KernelColumn>::validate(&self.$field)?;
                ensure_same_len(<Source as KernelColumn>::len(&self.$field), stencil.len())?;
                let handles = stencil.selection_handles_with_policy(policy, false)?;
                let count = select::selected_count(policy, &handles)?;
                Ok(DeviceSoA1 {
                    source: crate::detail::api::device_expr_compact_with_selection_with_policy(
                        policy,
                        &self.$field,
                        &handles,
                        count,
                    )?,
                })
            }
        }
    };
}

impl_kernel_copy_where_tuple1!(SoAView1<Source>, source);
impl_kernel_copy_where_tuple1!(DeviceSoA1<Source>, source);

impl<Source, Stencil, Pred> KernelCopyWhereInput<Stencil, Pred> for (Source,)
where
    Source: KernelColumn + KernelColumnAt<S0>,
    SoAView1<Source>: KernelCopyWhereInput<Stencil, Pred, Runtime = Source::Runtime>,
{
    type Runtime = Source::Runtime;
    type Output = <SoAView1<Source> as KernelCopyWhereInput<Stencil, Pred>>::Output;

    fn copy_where_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        stencil: Stencil,
    ) -> Result<Self::Output, Error> {
        <SoAView1<Source> as KernelCopyWhereInput<Stencil, Pred>>::copy_where_read(
            SoAView1 { source: self.0 },
            policy,
            stencil,
        )
    }
}

macro_rules! impl_kernel_copy_where_tuple2 {
    ($target:ty, $out:ident, $left:tt, $right:tt) => {
        impl<Left, Right, Stencil, Pred> KernelCopyWhereInput<Stencil, Pred> for $target
        where
            Left: KernelColumn + KernelColumnAt<S0>,
            Right: KernelColumn<Runtime = Left::Runtime> + KernelColumnAt<S0>,
            Stencil: crate::detail::api::SelectionStencil<Pred, Runtime = Left::Runtime>,
            Left::Item: Scalar + 'static,
            Right::Item: Scalar + 'static,
            Left::Expr: DeviceGpuExpr<Left::Item>,
            Right::Expr: DeviceGpuExpr<Right::Item>,
        {
            type Runtime = Left::Runtime;
            type Output =
                $out<DeviceVec<Left::Runtime, Left::Item>, DeviceVec<Left::Runtime, Right::Item>>;

            fn copy_where_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
                stencil: Stencil,
            ) -> Result<Self::Output, Error> {
                validate_columns2(&self.$left, &self.$right)?;
                ensure_same_len(<Left as KernelColumn>::len(&self.$left), stencil.len())?;
                let handles = stencil.selection_handles_with_policy(policy, false)?;
                let count = select::selected_count(policy, &handles)?;
                Ok($out {
                    left: crate::detail::api::device_expr_compact_with_selection_with_policy(
                        policy,
                        &self.$left,
                        &handles,
                        count,
                    )?,
                    right: crate::detail::api::device_expr_compact_with_selection_with_policy(
                        policy,
                        &self.$right,
                        &handles,
                        count,
                    )?,
                })
            }
        }
    };
}

impl_kernel_copy_where_tuple2!(SoAView2<Left, Right>, DeviceSoA2, left, right);
impl_kernel_copy_where_tuple2!(DeviceSoA2<Left, Right>, DeviceSoA2, left, right);

impl<Left, Right, Stencil, Pred> KernelCopyWhereInput<Stencil, Pred> for (Left, Right)
where
    SoAView2<Left, Right>: KernelCopyWhereInput<Stencil, Pred>,
{
    type Runtime = <SoAView2<Left, Right> as KernelCopyWhereInput<Stencil, Pred>>::Runtime;
    type Output = <SoAView2<Left, Right> as KernelCopyWhereInput<Stencil, Pred>>::Output;

    fn copy_where_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        stencil: Stencil,
    ) -> Result<Self::Output, Error> {
        <SoAView2<Left, Right> as KernelCopyWhereInput<Stencil, Pred>>::copy_where_read(
            SoAView2 {
                left: self.0,
                right: self.1,
            },
            policy,
            stencil,
        )
    }
}

macro_rules! impl_kernel_copy_where_tuple3 {
    ($target:ty, $out:ident, $first:tt, $second:tt, $third:tt) => {
        impl<First, Second, Third, Stencil, Pred> KernelCopyWhereInput<Stencil, Pred> for $target
        where
            First: KernelColumn + KernelColumnAt<S0>,
            Second: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
            Third: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
            Stencil: crate::detail::api::SelectionStencil<Pred, Runtime = First::Runtime>,
            First::Item: Scalar + 'static,
            Second::Item: Scalar + 'static,
            Third::Item: Scalar + 'static,
            First::Expr: DeviceGpuExpr<First::Item>,
            Second::Expr: DeviceGpuExpr<Second::Item>,
            Third::Expr: DeviceGpuExpr<Third::Item>,
        {
            type Runtime = First::Runtime;
            type Output = $out<
                DeviceVec<First::Runtime, First::Item>,
                DeviceVec<First::Runtime, Second::Item>,
                DeviceVec<First::Runtime, Third::Item>,
            >;

            fn copy_where_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
                stencil: Stencil,
            ) -> Result<Self::Output, Error> {
                validate_columns3(&self.$first, &self.$second, &self.$third)?;
                ensure_same_len(<First as KernelColumn>::len(&self.$first), stencil.len())?;
                let handles = stencil.selection_handles_with_policy(policy, false)?;
                let count = select::selected_count(policy, &handles)?;
                Ok($out {
                    first: crate::detail::api::device_expr_compact_with_selection_with_policy(
                        policy,
                        &self.$first,
                        &handles,
                        count,
                    )?,
                    second: crate::detail::api::device_expr_compact_with_selection_with_policy(
                        policy,
                        &self.$second,
                        &handles,
                        count,
                    )?,
                    third: crate::detail::api::device_expr_compact_with_selection_with_policy(
                        policy,
                        &self.$third,
                        &handles,
                        count,
                    )?,
                })
            }
        }
    };
}

impl_kernel_copy_where_tuple3!(
    SoAView3<First, Second, Third>,
    DeviceSoA3,
    first,
    second,
    third
);
impl_kernel_copy_where_tuple3!(
    DeviceSoA3<First, Second, Third>,
    DeviceSoA3,
    first,
    second,
    third
);

impl<First, Second, Third, Stencil, Pred> KernelCopyWhereInput<Stencil, Pred>
    for (First, Second, Third)
where
    SoAView3<First, Second, Third>: KernelCopyWhereInput<Stencil, Pred>,
{
    type Runtime = <SoAView3<First, Second, Third> as KernelCopyWhereInput<Stencil, Pred>>::Runtime;
    type Output = <SoAView3<First, Second, Third> as KernelCopyWhereInput<Stencil, Pred>>::Output;

    fn copy_where_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        stencil: Stencil,
    ) -> Result<Self::Output, Error> {
        <SoAView3<First, Second, Third> as KernelCopyWhereInput<Stencil, Pred>>::copy_where_read(
            SoAView3 {
                first: self.0,
                second: self.1,
                third: self.2,
            },
            policy,
            stencil,
        )
    }
}

fn replace_one_with_flags_read<Source>(
    policy: &CubePolicy<Source::Runtime>,
    input: &Source,
    replacement: Source::Item,
    flag: &cubecl::server::Handle,
) -> Result<DeviceVec<Source::Runtime, Source::Item>, Error>
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: Scalar + 'static,
    Source::Expr: DeviceGpuExpr<Source::Item>,
{
    <Source as KernelColumn>::validate(input)?;
    let len = <Source as KernelColumn>::len(input);
    u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    if len == 0 {
        return Ok(policy.empty_device_vec());
    }

    let client = policy.client();
    let output_handle = client.empty(len * std::mem::size_of::<Source::Item>());
    let block_count = len.div_ceil(BLOCK_REPLACE_WHERE_SIZE as usize);
    let block_count_u32 =
        u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;
    let replacement_values = [replacement];
    let replacement_handle = client.create_from_slice(Source::Item::as_bytes(&replacement_values));
    let bindings = <Source as KernelColumn>::stage(input, policy)?;
    let slot_offsets = bindings.slot_offsets_handle(client)?;
    let slot0 = bindings.slots.first().unwrap();
    let slot1 = bindings.slots.get(1).unwrap_or(slot0);
    let slot2 = bindings.slots.get(2).unwrap_or(slot0);
    let slot3 = bindings.slots.get(3).unwrap_or(slot0);

    unsafe {
        replace_device_expr_with_flags_kernel::launch_unchecked::<
            Source::Item,
            Source::Expr,
            Source::Runtime,
        >(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_REPLACE_WHERE_SIZE),
            BufferArg::from_raw_parts(slot0.0.clone(), slot0.1),
            BufferArg::from_raw_parts(slot1.0.clone(), slot1.1),
            BufferArg::from_raw_parts(slot2.0.clone(), slot2.1),
            BufferArg::from_raw_parts(slot3.0.clone(), slot3.1),
            BufferArg::from_raw_parts(slot_offsets.clone(), 4),
            BufferArg::from_raw_parts(replacement_handle.clone(), 1),
            BufferArg::from_raw_parts(flag.clone(), len),
            BufferArg::from_raw_parts(output_handle.clone(), len),
        );
    }

    Ok(DeviceVec::from_handle(policy.id(), output_handle, len))
}

impl<Source, Stencil, Pred> KernelReplaceWhereInput<Stencil, Pred> for Source
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Stencil: crate::detail::api::SelectionStencil<Pred, Runtime = Source::Runtime>,
    Source::Item: Scalar + 'static,
    Source::Expr: DeviceGpuExpr<Source::Item>,
{
    type Runtime = Source::Runtime;
    type Item = Source::Item;
    type Output = DeviceSoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn replace_where_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        replacement: Self::Item,
        stencil: Stencil,
    ) -> Result<Self::Output, Error> {
        <Source as KernelColumn>::validate(&self)?;
        ensure_same_len(<Source as KernelColumn>::len(&self), stencil.len())?;
        let flags = stencil.selection_handles_with_policy(policy, false)?;
        Ok(DeviceSoA1 {
            source: replace_one_with_flags_read(policy, &self, replacement, &flags.flag)?,
        })
    }
}

macro_rules! impl_kernel_replace_where_tuple1 {
    ($target:ty, $field:tt) => {
        impl<Source, Stencil, Pred> KernelReplaceWhereInput<Stencil, Pred> for $target
        where
            Source: KernelColumn + KernelColumnAt<S0>,
            Stencil: crate::detail::api::SelectionStencil<Pred, Runtime = Source::Runtime>,
            Source::Item: Scalar + 'static,
            Source::Expr: DeviceGpuExpr<Source::Item>,
        {
            type Runtime = Source::Runtime;
            type Item = (Source::Item,);
            type Output = DeviceSoA1<DeviceVec<Source::Runtime, Source::Item>>;

            fn replace_where_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
                replacement: Self::Item,
                stencil: Stencil,
            ) -> Result<Self::Output, Error> {
                <Source as KernelColumn>::validate(&self.$field)?;
                ensure_same_len(<Source as KernelColumn>::len(&self.$field), stencil.len())?;
                let flags = stencil.selection_handles_with_policy(policy, false)?;
                Ok(DeviceSoA1 {
                    source: replace_one_with_flags_read(
                        policy,
                        &self.$field,
                        replacement.0,
                        &flags.flag,
                    )?,
                })
            }
        }
    };
}

impl_kernel_replace_where_tuple1!(SoAView1<Source>, source);
impl_kernel_replace_where_tuple1!(DeviceSoA1<Source>, source);

impl<Source, Stencil, Pred> KernelReplaceWhereInput<Stencil, Pred> for (Source,)
where
    Source: KernelColumn + KernelColumnAt<S0>,
    SoAView1<Source>: KernelReplaceWhereInput<Stencil, Pred, Runtime = Source::Runtime>,
{
    type Runtime = Source::Runtime;
    type Item = <SoAView1<Source> as KernelReplaceWhereInput<Stencil, Pred>>::Item;
    type Output = <SoAView1<Source> as KernelReplaceWhereInput<Stencil, Pred>>::Output;

    fn replace_where_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        replacement: Self::Item,
        stencil: Stencil,
    ) -> Result<Self::Output, Error> {
        <SoAView1<Source> as KernelReplaceWhereInput<Stencil, Pred>>::replace_where_read(
            SoAView1 { source: self.0 },
            policy,
            replacement,
            stencil,
        )
    }
}

macro_rules! impl_kernel_replace_where_tuple2 {
    ($target:ty, $out:ident, $left:tt, $right:tt) => {
        impl<Left, Right, Stencil, Pred> KernelReplaceWhereInput<Stencil, Pred> for $target
        where
            Left: KernelColumn + KernelColumnAt<S0>,
            Right: KernelColumn<Runtime = Left::Runtime> + KernelColumnAt<S0>,
            Stencil: crate::detail::api::SelectionStencil<Pred, Runtime = Left::Runtime>,
            Left::Item: Scalar + 'static,
            Right::Item: Scalar + 'static,
            Left::Expr: DeviceGpuExpr<Left::Item>,
            Right::Expr: DeviceGpuExpr<Right::Item>,
        {
            type Runtime = Left::Runtime;
            type Item = (Left::Item, Right::Item);
            type Output =
                $out<DeviceVec<Left::Runtime, Left::Item>, DeviceVec<Left::Runtime, Right::Item>>;

            fn replace_where_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
                replacement: Self::Item,
                stencil: Stencil,
            ) -> Result<Self::Output, Error> {
                validate_columns2(&self.$left, &self.$right)?;
                ensure_same_len(<Left as KernelColumn>::len(&self.$left), stencil.len())?;
                let flags = stencil.selection_handles_with_policy(policy, false)?;
                Ok($out {
                    left: replace_one_with_flags_read(
                        policy,
                        &self.$left,
                        replacement.0,
                        &flags.flag,
                    )?,
                    right: replace_one_with_flags_read(
                        policy,
                        &self.$right,
                        replacement.1,
                        &flags.flag,
                    )?,
                })
            }
        }
    };
}

impl_kernel_replace_where_tuple2!(SoAView2<Left, Right>, DeviceSoA2, left, right);
impl_kernel_replace_where_tuple2!(DeviceSoA2<Left, Right>, DeviceSoA2, left, right);

impl<Left, Right, Stencil, Pred> KernelReplaceWhereInput<Stencil, Pred> for (Left, Right)
where
    SoAView2<Left, Right>: KernelReplaceWhereInput<Stencil, Pred>,
{
    type Runtime = <SoAView2<Left, Right> as KernelReplaceWhereInput<Stencil, Pred>>::Runtime;
    type Item = <SoAView2<Left, Right> as KernelReplaceWhereInput<Stencil, Pred>>::Item;
    type Output = <SoAView2<Left, Right> as KernelReplaceWhereInput<Stencil, Pred>>::Output;

    fn replace_where_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        replacement: Self::Item,
        stencil: Stencil,
    ) -> Result<Self::Output, Error> {
        <SoAView2<Left, Right> as KernelReplaceWhereInput<Stencil, Pred>>::replace_where_read(
            SoAView2 {
                left: self.0,
                right: self.1,
            },
            policy,
            replacement,
            stencil,
        )
    }
}

macro_rules! impl_kernel_replace_where_tuple3 {
    ($target:ty, $out:ident, $first:tt, $second:tt, $third:tt) => {
        impl<First, Second, Third, Stencil, Pred> KernelReplaceWhereInput<Stencil, Pred> for $target
        where
            First: KernelColumn + KernelColumnAt<S0>,
            Second: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
            Third: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
            Stencil: crate::detail::api::SelectionStencil<Pred, Runtime = First::Runtime>,
            First::Item: Scalar + 'static,
            Second::Item: Scalar + 'static,
            Third::Item: Scalar + 'static,
            First::Expr: DeviceGpuExpr<First::Item>,
            Second::Expr: DeviceGpuExpr<Second::Item>,
            Third::Expr: DeviceGpuExpr<Third::Item>,
        {
            type Runtime = First::Runtime;
            type Item = (First::Item, Second::Item, Third::Item);
            type Output = $out<
                DeviceVec<First::Runtime, First::Item>,
                DeviceVec<First::Runtime, Second::Item>,
                DeviceVec<First::Runtime, Third::Item>,
            >;

            fn replace_where_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
                replacement: Self::Item,
                stencil: Stencil,
            ) -> Result<Self::Output, Error> {
                validate_columns3(&self.$first, &self.$second, &self.$third)?;
                ensure_same_len(<First as KernelColumn>::len(&self.$first), stencil.len())?;
                let flags = stencil.selection_handles_with_policy(policy, false)?;
                Ok($out {
                    first: replace_one_with_flags_read(
                        policy,
                        &self.$first,
                        replacement.0,
                        &flags.flag,
                    )?,
                    second: replace_one_with_flags_read(
                        policy,
                        &self.$second,
                        replacement.1,
                        &flags.flag,
                    )?,
                    third: replace_one_with_flags_read(
                        policy,
                        &self.$third,
                        replacement.2,
                        &flags.flag,
                    )?,
                })
            }
        }
    };
}

impl_kernel_replace_where_tuple3!(
    SoAView3<First, Second, Third>,
    DeviceSoA3,
    first,
    second,
    third
);
impl_kernel_replace_where_tuple3!(
    DeviceSoA3<First, Second, Third>,
    DeviceSoA3,
    first,
    second,
    third
);

impl<First, Second, Third, Stencil, Pred> KernelReplaceWhereInput<Stencil, Pred>
    for (First, Second, Third)
where
    SoAView3<First, Second, Third>: KernelReplaceWhereInput<Stencil, Pred>,
{
    type Runtime =
        <SoAView3<First, Second, Third> as KernelReplaceWhereInput<Stencil, Pred>>::Runtime;
    type Item = <SoAView3<First, Second, Third> as KernelReplaceWhereInput<Stencil, Pred>>::Item;
    type Output =
        <SoAView3<First, Second, Third> as KernelReplaceWhereInput<Stencil, Pred>>::Output;

    fn replace_where_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        replacement: Self::Item,
        stencil: Stencil,
    ) -> Result<Self::Output, Error> {
        <SoAView3<First, Second, Third> as KernelReplaceWhereInput<Stencil, Pred>>::replace_where_read(
            SoAView3 {
                first: self.0,
                second: self.1,
                third: self.2,
            },
            policy,
            replacement,
            stencil,
        )
    }
}

fn selection_block_count_read(len: usize) -> Result<u32, Error> {
    let block_count = len.div_ceil(256);
    u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })
}

fn tuple2_selection_handles_read<Left, Right, Pred>(
    policy: &CubePolicy<Left::Runtime>,
    left: &Left,
    right: &Right,
    invert: bool,
    env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<Left::Runtime>,
) -> Result<select::SelectionControl, Error>
where
    Left: KernelColumn + KernelColumnAt<S0>,
    Right: KernelColumn<Runtime = Left::Runtime> + KernelColumnAt<S0>,
    Left::Item: Scalar + 'static,
    Right::Item: Scalar + 'static,
    Left::Expr: DeviceGpuExpr<Left::Item>,
    Right::Expr: DeviceGpuExpr<Right::Item>,
    Pred: PredicateOp<(Left::Item, Right::Item)>,
{
    validate_columns2(left, right)?;
    let len = <Left as KernelColumn>::len(left);
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let client = policy.client();
    let flag = client.empty(len * std::mem::size_of::<u32>());
    if len != 0 {
        let block_count_u32 = selection_block_count_read(len)?;
        let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
        let invert_handle = client.create_from_slice(u32::as_bytes(&[if invert { 1 } else { 0 }]));
        let left_bindings = <Left as KernelColumn>::stage(left, policy)?;
        let right_bindings = <Right as KernelColumn>::stage(right, policy)?;
        let left_slot0 = left_bindings.slot_or_first(0);
        let left_slot1 = left_bindings.slot_or_first(1);
        let left_slot2 = left_bindings.slot_or_first(2);
        let left_slot3 = left_bindings.slot_or_first(3);
        let right_slot0 = right_bindings.slot_or_first(0);
        let right_slot1 = right_bindings.slot_or_first(1);
        let right_slot2 = right_bindings.slot_or_first(2);
        let right_slot3 = right_bindings.slot_or_first(3);
        let left_slot_offsets = left_bindings.slot_offsets_handle(client)?;
        let right_slot_offsets = right_bindings.slot_offsets_handle(client)?;
        unsafe {
            tuple2_predicate_device_expr_flags_kernel::launch_unchecked::<
                Left::Item,
                Right::Item,
                Left::Expr,
                Right::Expr,
                Pred,
                Left::Runtime,
            >(
                client,
                CubeCount::Static(block_count_u32, 1, 1),
                CubeDim::new_1d(256),
                env,
                BufferArg::from_raw_parts(left_slot0.0.clone(), left_slot0.1),
                BufferArg::from_raw_parts(left_slot1.0.clone(), left_slot1.1),
                BufferArg::from_raw_parts(left_slot2.0.clone(), left_slot2.1),
                BufferArg::from_raw_parts(left_slot3.0.clone(), left_slot3.1),
                BufferArg::from_raw_parts(left_slot_offsets.clone(), 4),
                BufferArg::from_raw_parts(right_slot0.0.clone(), right_slot0.1),
                BufferArg::from_raw_parts(right_slot1.0.clone(), right_slot1.1),
                BufferArg::from_raw_parts(right_slot2.0.clone(), right_slot2.1),
                BufferArg::from_raw_parts(right_slot3.0.clone(), right_slot3.1),
                BufferArg::from_raw_parts(right_slot_offsets.clone(), 4),
                BufferArg::from_raw_parts(len_handle.clone(), 1),
                BufferArg::from_raw_parts(invert_handle.clone(), 1),
                BufferArg::from_raw_parts(flag.clone(), len),
            );
        }
    }
    select::handles_from_flags(policy, len, len_u32, flag, policy.empty_handle())
}

fn tuple3_selection_handles_read<First, Second, Third, Pred>(
    policy: &CubePolicy<First::Runtime>,
    first: &First,
    second: &Second,
    third: &Third,
    invert: bool,
    env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<First::Runtime>,
) -> Result<select::SelectionControl, Error>
where
    First: KernelColumn + KernelColumnAt<S0>,
    Second: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
    Third: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
    First::Item: Scalar + 'static,
    Second::Item: Scalar + 'static,
    Third::Item: Scalar + 'static,
    First::Expr: DeviceGpuExpr<First::Item>,
    Second::Expr: DeviceGpuExpr<Second::Item>,
    Third::Expr: DeviceGpuExpr<Third::Item>,
    Pred: PredicateOp<(First::Item, Second::Item, Third::Item)>,
{
    validate_columns3(first, second, third)?;
    let len = <First as KernelColumn>::len(first);
    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let client = policy.client();
    let flag = client.empty(len * std::mem::size_of::<u32>());
    if len != 0 {
        let block_count_u32 = selection_block_count_read(len)?;
        let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
        let invert_handle = client.create_from_slice(u32::as_bytes(&[if invert { 1 } else { 0 }]));
        let first_bindings = <First as KernelColumn>::stage(first, policy)?;
        let second_bindings = <Second as KernelColumn>::stage(second, policy)?;
        let third_bindings = <Third as KernelColumn>::stage(third, policy)?;
        let first_slot0 = first_bindings.slot_or_first(0);
        let first_slot1 = first_bindings.slot_or_first(1);
        let first_slot2 = first_bindings.slot_or_first(2);
        let first_slot3 = first_bindings.slot_or_first(3);
        let second_slot0 = second_bindings.slot_or_first(0);
        let second_slot1 = second_bindings.slot_or_first(1);
        let second_slot2 = second_bindings.slot_or_first(2);
        let second_slot3 = second_bindings.slot_or_first(3);
        let third_slot0 = third_bindings.slot_or_first(0);
        let third_slot1 = third_bindings.slot_or_first(1);
        let third_slot2 = third_bindings.slot_or_first(2);
        let third_slot3 = third_bindings.slot_or_first(3);
        let first_slot_offsets = first_bindings.slot_offsets_handle(client)?;
        let second_slot_offsets = second_bindings.slot_offsets_handle(client)?;
        let third_slot_offsets = third_bindings.slot_offsets_handle(client)?;
        unsafe {
            tuple3_predicate_device_expr_flags_kernel::launch_unchecked::<
                First::Item,
                Second::Item,
                Third::Item,
                First::Expr,
                Second::Expr,
                Third::Expr,
                Pred,
                First::Runtime,
            >(
                client,
                CubeCount::Static(block_count_u32, 1, 1),
                CubeDim::new_1d(256),
                env,
                BufferArg::from_raw_parts(first_slot0.0.clone(), first_slot0.1),
                BufferArg::from_raw_parts(first_slot1.0.clone(), first_slot1.1),
                BufferArg::from_raw_parts(first_slot2.0.clone(), first_slot2.1),
                BufferArg::from_raw_parts(first_slot3.0.clone(), first_slot3.1),
                BufferArg::from_raw_parts(first_slot_offsets.clone(), 4),
                BufferArg::from_raw_parts(second_slot0.0.clone(), second_slot0.1),
                BufferArg::from_raw_parts(second_slot1.0.clone(), second_slot1.1),
                BufferArg::from_raw_parts(second_slot2.0.clone(), second_slot2.1),
                BufferArg::from_raw_parts(second_slot3.0.clone(), second_slot3.1),
                BufferArg::from_raw_parts(second_slot_offsets.clone(), 4),
                BufferArg::from_raw_parts(third_slot0.0.clone(), third_slot0.1),
                BufferArg::from_raw_parts(third_slot1.0.clone(), third_slot1.1),
                BufferArg::from_raw_parts(third_slot2.0.clone(), third_slot2.1),
                BufferArg::from_raw_parts(third_slot3.0.clone(), third_slot3.1),
                BufferArg::from_raw_parts(third_slot_offsets.clone(), 4),
                BufferArg::from_raw_parts(len_handle.clone(), 1),
                BufferArg::from_raw_parts(invert_handle.clone(), 1),
                BufferArg::from_raw_parts(flag.clone(), len),
            );
        }
    }
    select::handles_from_flags(policy, len, len_u32, flag, policy.empty_handle())
}

impl<Source, Pred> KernelSelectInput<Pred> for Source
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: Scalar + 'static,
    Source::Expr: DeviceGpuExpr<Source::Item> + GpuExpr<Source::Item>,
    Pred: PredicateOp<Source::Item>,
{
    type Runtime = Source::Runtime;
    type Output = DeviceSoA1<DeviceVec<Source::Runtime, Source::Item>>;
    type Env = Pred::Env;

    fn select_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        invert: bool,
        env: <Self::Env as cubecl::prelude::LaunchArg>::RuntimeArg<Self::Runtime>,
    ) -> Result<Self::Output, Error> {
        <Source as KernelColumn>::validate(&self)?;
        Ok(DeviceSoA1 {
            source: crate::detail::api::device_expr_copy_where_with_policy::<Source, Pred>(
                policy, &self, invert, env,
            )?,
        })
    }
}

macro_rules! impl_kernel_select_tuple1 {
    ($target:ty, $field:tt) => {
        impl<Source, Pred> KernelSelectInput<Pred> for $target
        where
            Source: KernelColumn + KernelColumnAt<S0>,
            Source::Item: Scalar + 'static,
            Source::Expr: DeviceGpuExpr<Source::Item> + GpuExpr<Source::Item>,
            Pred: PredicateOp<Source::Item>,
        {
            type Runtime = Source::Runtime;
            type Output = DeviceSoA1<DeviceVec<Source::Runtime, Source::Item>>;
            type Env = Pred::Env;

            fn select_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
                invert: bool,
                env: <Self::Env as cubecl::prelude::LaunchArg>::RuntimeArg<Self::Runtime>,
            ) -> Result<Self::Output, Error> {
                <Source as KernelColumn>::validate(&self.$field)?;
                Ok(DeviceSoA1 {
                    source: crate::detail::api::device_expr_copy_where_with_policy::<Source, Pred>(
                        policy,
                        &self.$field,
                        invert,
                        env,
                    )?,
                })
            }
        }
    };
}

impl_kernel_select_tuple1!(SoAView1<Source>, source);
impl_kernel_select_tuple1!(DeviceSoA1<Source>, source);

impl<Source, Pred> KernelSelectInput<Pred> for (Source,)
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source: KernelSelectInput<crate::detail::api::Tuple1PredicateOp<Pred>>,
{
    type Runtime =
        <Source as KernelSelectInput<crate::detail::api::Tuple1PredicateOp<Pred>>>::Runtime;
    type Output =
        <Source as KernelSelectInput<crate::detail::api::Tuple1PredicateOp<Pred>>>::Output;
    type Env = <Source as KernelSelectInput<crate::detail::api::Tuple1PredicateOp<Pred>>>::Env;

    fn select_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        invert: bool,
        env: <Self::Env as cubecl::prelude::LaunchArg>::RuntimeArg<Self::Runtime>,
    ) -> Result<Self::Output, Error> {
        <Source as KernelSelectInput<crate::detail::api::Tuple1PredicateOp<Pred>>>::select_read(
            self.0, policy, invert, env,
        )
    }
}

macro_rules! impl_kernel_select_tuple2 {
    ($target:ty, $out:ident, $left:tt, $right:tt) => {
        impl<Left, Right, Pred> KernelSelectInput<Pred> for $target
        where
            Left: KernelColumn + KernelColumnAt<S0>,
            Right: KernelColumn<Runtime = Left::Runtime> + KernelColumnAt<S0>,
            Left::Item: Scalar + 'static,
            Right::Item: Scalar + 'static,
            Left::Expr: DeviceGpuExpr<Left::Item>,
            Right::Expr: DeviceGpuExpr<Right::Item>,
            Pred: PredicateOp<(Left::Item, Right::Item)>,
        {
            type Runtime = Left::Runtime;
            type Output =
                $out<DeviceVec<Left::Runtime, Left::Item>, DeviceVec<Left::Runtime, Right::Item>>;
            type Env = Pred::Env;

            fn select_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
                invert: bool,
                env: <Self::Env as cubecl::prelude::LaunchArg>::RuntimeArg<Self::Runtime>,
            ) -> Result<Self::Output, Error> {
                let handles = tuple2_selection_handles_read::<Left, Right, Pred>(
                    policy,
                    &self.$left,
                    &self.$right,
                    invert,
                    env,
                )?;
                let count = select::selected_count(policy, &handles)?;
                Ok($out {
                    left: crate::detail::api::device_expr_compact_with_selection_with_policy(
                        policy,
                        &self.$left,
                        &handles,
                        count,
                    )?,
                    right: crate::detail::api::device_expr_compact_with_selection_with_policy(
                        policy,
                        &self.$right,
                        &handles,
                        count,
                    )?,
                })
            }
        }
    };
}

impl_kernel_select_tuple2!(SoAView2<Left, Right>, DeviceSoA2, left, right);
impl_kernel_select_tuple2!(DeviceSoA2<Left, Right>, DeviceSoA2, left, right);

impl<Left, Right, Pred> KernelSelectInput<Pred> for (Left, Right)
where
    SoAView2<Left, Right>: KernelSelectInput<Pred>,
{
    type Runtime = <SoAView2<Left, Right> as KernelSelectInput<Pred>>::Runtime;
    type Output = <SoAView2<Left, Right> as KernelSelectInput<Pred>>::Output;
    type Env = <SoAView2<Left, Right> as KernelSelectInput<Pred>>::Env;

    fn select_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        invert: bool,
        env: <Self::Env as cubecl::prelude::LaunchArg>::RuntimeArg<Self::Runtime>,
    ) -> Result<Self::Output, Error> {
        <SoAView2<Left, Right> as KernelSelectInput<Pred>>::select_read(
            SoAView2 {
                left: self.0,
                right: self.1,
            },
            policy,
            invert,
            env,
        )
    }
}

macro_rules! impl_kernel_select_tuple3 {
    ($target:ty, $out:ident, $first:tt, $second:tt, $third:tt) => {
        impl<First, Second, Third, Pred> KernelSelectInput<Pred> for $target
        where
            First: KernelColumn + KernelColumnAt<S0>,
            Second: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
            Third: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
            First::Item: Scalar + 'static,
            Second::Item: Scalar + 'static,
            Third::Item: Scalar + 'static,
            First::Expr: DeviceGpuExpr<First::Item>,
            Second::Expr: DeviceGpuExpr<Second::Item>,
            Third::Expr: DeviceGpuExpr<Third::Item>,
            Pred: PredicateOp<(First::Item, Second::Item, Third::Item)>,
        {
            type Runtime = First::Runtime;
            type Output = $out<
                DeviceVec<First::Runtime, First::Item>,
                DeviceVec<First::Runtime, Second::Item>,
                DeviceVec<First::Runtime, Third::Item>,
            >;
            type Env = Pred::Env;

            fn select_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
                invert: bool,
                env: <Self::Env as cubecl::prelude::LaunchArg>::RuntimeArg<Self::Runtime>,
            ) -> Result<Self::Output, Error> {
                let handles = tuple3_selection_handles_read::<First, Second, Third, Pred>(
                    policy,
                    &self.$first,
                    &self.$second,
                    &self.$third,
                    invert,
                    env,
                )?;
                let count = select::selected_count(policy, &handles)?;
                Ok($out {
                    first: crate::detail::api::device_expr_compact_with_selection_with_policy(
                        policy,
                        &self.$first,
                        &handles,
                        count,
                    )?,
                    second: crate::detail::api::device_expr_compact_with_selection_with_policy(
                        policy,
                        &self.$second,
                        &handles,
                        count,
                    )?,
                    third: crate::detail::api::device_expr_compact_with_selection_with_policy(
                        policy,
                        &self.$third,
                        &handles,
                        count,
                    )?,
                })
            }
        }
    };
}

impl_kernel_select_tuple3!(
    SoAView3<First, Second, Third>,
    DeviceSoA3,
    first,
    second,
    third
);
impl_kernel_select_tuple3!(
    DeviceSoA3<First, Second, Third>,
    DeviceSoA3,
    first,
    second,
    third
);

impl<First, Second, Third, Pred> KernelSelectInput<Pred> for (First, Second, Third)
where
    SoAView3<First, Second, Third>: KernelSelectInput<Pred>,
{
    type Runtime = <SoAView3<First, Second, Third> as KernelSelectInput<Pred>>::Runtime;
    type Output = <SoAView3<First, Second, Third> as KernelSelectInput<Pred>>::Output;
    type Env = <SoAView3<First, Second, Third> as KernelSelectInput<Pred>>::Env;

    fn select_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        invert: bool,
        env: <Self::Env as cubecl::prelude::LaunchArg>::RuntimeArg<Self::Runtime>,
    ) -> Result<Self::Output, Error> {
        <SoAView3<First, Second, Third> as KernelSelectInput<Pred>>::select_read(
            SoAView3 {
                first: self.0,
                second: self.1,
                third: self.2,
            },
            policy,
            invert,
            env,
        )
    }
}

impl<Source, Pred> KernelPredicateQueryInput<Pred> for Source
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: Scalar + 'static,
    Source::Expr: GpuExpr<Source::Item>,
    Pred: PredicateOp<Source::Item>,
{
    type Runtime = Source::Runtime;
    type Env = Pred::Env;

    fn count_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        invert: bool,
        env: <Self::Env as cubecl::prelude::LaunchArg>::RuntimeArg<Self::Runtime>,
    ) -> Result<usize, Error> {
        <Source as KernelColumn>::validate(&self)?;
        crate::detail::api::device_expr_count_if_with_policy::<Source, Pred>(
            policy, &self, invert, env,
        )
    }

    fn find_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        invert: bool,
        env: <Self::Env as cubecl::prelude::LaunchArg>::RuntimeArg<Self::Runtime>,
    ) -> Result<Option<usize>, Error> {
        <Source as KernelColumn>::validate(&self)?;
        crate::detail::api::device_expr_find_if_with_policy::<Source, Pred>(
            policy, &self, invert, env,
        )
    }
}

macro_rules! impl_kernel_predicate_query_tuple1 {
    ($target:ty, $field:tt) => {
        impl<Source, Pred> KernelPredicateQueryInput<Pred> for $target
        where
            Source: KernelColumn + KernelColumnAt<S0>,
            Source::Item: Scalar + 'static,
            Source::Expr: GpuExpr<Source::Item>,
            Pred: PredicateOp<Source::Item>,
        {
            type Runtime = Source::Runtime;
            type Env = Pred::Env;

            fn count_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
                invert: bool,
                env: <Self::Env as cubecl::prelude::LaunchArg>::RuntimeArg<Self::Runtime>,
            ) -> Result<usize, Error> {
                <Source as KernelColumn>::validate(&self.$field)?;
                crate::detail::api::device_expr_count_if_with_policy::<Source, Pred>(
                    policy,
                    &self.$field,
                    invert,
                    env,
                )
            }

            fn find_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
                invert: bool,
                env: <Self::Env as cubecl::prelude::LaunchArg>::RuntimeArg<Self::Runtime>,
            ) -> Result<Option<usize>, Error> {
                <Source as KernelColumn>::validate(&self.$field)?;
                crate::detail::api::device_expr_find_if_with_policy::<Source, Pred>(
                    policy,
                    &self.$field,
                    invert,
                    env,
                )
            }
        }
    };
}

impl_kernel_predicate_query_tuple1!(SoAView1<Source>, source);
impl_kernel_predicate_query_tuple1!(DeviceSoA1<Source>, source);

impl<Source, Pred> KernelPredicateQueryInput<Pred> for (Source,)
where
    Source: KernelPredicateQueryInput<crate::detail::api::Tuple1PredicateOp<Pred>>,
{
    type Runtime =
        <Source as KernelPredicateQueryInput<crate::detail::api::Tuple1PredicateOp<Pred>>>::Runtime;
    type Env =
        <Source as KernelPredicateQueryInput<crate::detail::api::Tuple1PredicateOp<Pred>>>::Env;

    fn count_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        invert: bool,
        env: <Self::Env as cubecl::prelude::LaunchArg>::RuntimeArg<Self::Runtime>,
    ) -> Result<usize, Error> {
        <Source as KernelPredicateQueryInput<crate::detail::api::Tuple1PredicateOp<Pred>>>::count_read(
            self.0, policy, invert, env,
        )
    }

    fn find_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        invert: bool,
        env: <Self::Env as cubecl::prelude::LaunchArg>::RuntimeArg<Self::Runtime>,
    ) -> Result<Option<usize>, Error> {
        <Source as KernelPredicateQueryInput<crate::detail::api::Tuple1PredicateOp<Pred>>>::find_read(
            self.0, policy, invert, env,
        )
    }
}

macro_rules! impl_kernel_predicate_query_tuple2 {
    ($target:ty, $left:tt, $right:tt) => {
        impl<Left, Right, Pred> KernelPredicateQueryInput<Pred> for $target
        where
            Left: KernelColumn + KernelColumnAt<S0>,
            Right: KernelColumn<Runtime = Left::Runtime> + KernelColumnAt<S0>,
            Left::Item: Scalar + 'static,
            Right::Item: Scalar + 'static,
            Left::Expr: DeviceGpuExpr<Left::Item>,
            Right::Expr: DeviceGpuExpr<Right::Item>,
            Pred: PredicateOp<(Left::Item, Right::Item)>,
        {
            type Runtime = Left::Runtime;
            type Env = Pred::Env;

            fn count_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
                invert: bool,
                env: <Self::Env as cubecl::prelude::LaunchArg>::RuntimeArg<Self::Runtime>,
            ) -> Result<usize, Error> {
                let handles = tuple2_selection_handles_read::<Left, Right, Pred>(
                    policy,
                    &self.$left,
                    &self.$right,
                    invert,
                    env,
                )?;
                select::selected_count(policy, &handles)
            }

            fn find_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
                invert: bool,
                env: <Self::Env as cubecl::prelude::LaunchArg>::RuntimeArg<Self::Runtime>,
            ) -> Result<Option<usize>, Error> {
                let handles = tuple2_selection_handles_read::<Left, Right, Pred>(
                    policy,
                    &self.$left,
                    &self.$right,
                    invert,
                    env,
                )?;
                primitive_search::first_flag(policy, handles.flag, handles.len, handles.len)
            }
        }
    };
}

impl_kernel_predicate_query_tuple2!(SoAView2<Left, Right>, left, right);
impl_kernel_predicate_query_tuple2!(DeviceSoA2<Left, Right>, left, right);

impl<Left, Right, Pred> KernelPredicateQueryInput<Pred> for (Left, Right)
where
    SoAView2<Left, Right>: KernelPredicateQueryInput<Pred>,
{
    type Runtime = <SoAView2<Left, Right> as KernelPredicateQueryInput<Pred>>::Runtime;
    type Env = <SoAView2<Left, Right> as KernelPredicateQueryInput<Pred>>::Env;

    fn count_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        invert: bool,
        env: <Self::Env as cubecl::prelude::LaunchArg>::RuntimeArg<Self::Runtime>,
    ) -> Result<usize, Error> {
        <SoAView2<Left, Right> as KernelPredicateQueryInput<Pred>>::count_read(
            SoAView2 {
                left: self.0,
                right: self.1,
            },
            policy,
            invert,
            env,
        )
    }

    fn find_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        invert: bool,
        env: <Self::Env as cubecl::prelude::LaunchArg>::RuntimeArg<Self::Runtime>,
    ) -> Result<Option<usize>, Error> {
        <SoAView2<Left, Right> as KernelPredicateQueryInput<Pred>>::find_read(
            SoAView2 {
                left: self.0,
                right: self.1,
            },
            policy,
            invert,
            env,
        )
    }
}

macro_rules! impl_kernel_predicate_query_tuple3 {
    ($target:ty, $first:tt, $second:tt, $third:tt) => {
        impl<First, Second, Third, Pred> KernelPredicateQueryInput<Pred> for $target
        where
            First: KernelColumn + KernelColumnAt<S0>,
            Second: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
            Third: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
            First::Item: Scalar + 'static,
            Second::Item: Scalar + 'static,
            Third::Item: Scalar + 'static,
            First::Expr: DeviceGpuExpr<First::Item>,
            Second::Expr: DeviceGpuExpr<Second::Item>,
            Third::Expr: DeviceGpuExpr<Third::Item>,
            Pred: PredicateOp<(First::Item, Second::Item, Third::Item)>,
        {
            type Runtime = First::Runtime;
            type Env = Pred::Env;

            fn count_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
                invert: bool,
                env: <Self::Env as cubecl::prelude::LaunchArg>::RuntimeArg<Self::Runtime>,
            ) -> Result<usize, Error> {
                let handles = tuple3_selection_handles_read::<First, Second, Third, Pred>(
                    policy,
                    &self.$first,
                    &self.$second,
                    &self.$third,
                    invert,
                    env,
                )?;
                select::selected_count(policy, &handles)
            }

            fn find_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
                invert: bool,
                env: <Self::Env as cubecl::prelude::LaunchArg>::RuntimeArg<Self::Runtime>,
            ) -> Result<Option<usize>, Error> {
                let handles = tuple3_selection_handles_read::<First, Second, Third, Pred>(
                    policy,
                    &self.$first,
                    &self.$second,
                    &self.$third,
                    invert,
                    env,
                )?;
                primitive_search::first_flag(policy, handles.flag, handles.len, handles.len)
            }
        }
    };
}

impl_kernel_predicate_query_tuple3!(SoAView3<First, Second, Third>, first, second, third);
impl_kernel_predicate_query_tuple3!(DeviceSoA3<First, Second, Third>, first, second, third);

impl<First, Second, Third, Pred> KernelPredicateQueryInput<Pred> for (First, Second, Third)
where
    SoAView3<First, Second, Third>: KernelPredicateQueryInput<Pred>,
{
    type Runtime = <SoAView3<First, Second, Third> as KernelPredicateQueryInput<Pred>>::Runtime;
    type Env = <SoAView3<First, Second, Third> as KernelPredicateQueryInput<Pred>>::Env;

    fn count_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        invert: bool,
        env: <Self::Env as cubecl::prelude::LaunchArg>::RuntimeArg<Self::Runtime>,
    ) -> Result<usize, Error> {
        <SoAView3<First, Second, Third> as KernelPredicateQueryInput<Pred>>::count_read(
            SoAView3 {
                first: self.0,
                second: self.1,
                third: self.2,
            },
            policy,
            invert,
            env,
        )
    }

    fn find_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        invert: bool,
        env: <Self::Env as cubecl::prelude::LaunchArg>::RuntimeArg<Self::Runtime>,
    ) -> Result<Option<usize>, Error> {
        <SoAView3<First, Second, Third> as KernelPredicateQueryInput<Pred>>::find_read(
            SoAView3 {
                first: self.0,
                second: self.1,
                third: self.2,
            },
            policy,
            invert,
            env,
        )
    }
}

fn is_partitioned_from_flags_read<R: Runtime>(
    policy: &CubePolicy<R>,
    handles: &select::SelectionControl,
) -> Result<bool, Error> {
    let first_rejected =
        primitive_search::first_unset_flag(policy, handles.flag.clone(), handles.len, handles.len)?
            .unwrap_or(handles.len);
    let selected_count = select::selected_count(policy, handles)?;
    Ok(selected_count == first_rejected)
}

fn is_partitioned_single_read<R: Runtime>(
    policy: &CubePolicy<R>,
    handles: select::SelectionControl,
) -> Result<bool, Error> {
    let Some(point) =
        primitive_search::first_unset_flag(policy, handles.flag.clone(), handles.len, handles.len)?
    else {
        return Ok(true);
    };
    if point + 1 >= handles.len {
        return Ok(true);
    }

    let client = policy.client();
    let point_u32 = u32::try_from(point).map_err(|_| Error::LengthTooLarge { len: point })?;
    let point_handle = client.create_from_slice(u32::as_bytes(&[point_u32]));
    let tail_flags = client.empty(handles.len * std::mem::size_of::<u32>());
    let block_count_u32 = selection_block_count_read(handles.len)?;
    unsafe {
        partition_tail_selected_flags_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(256),
            BufferArg::from_raw_parts(handles.flag.clone(), handles.len),
            BufferArg::from_raw_parts(point_handle.clone(), 1),
            BufferArg::from_raw_parts(tail_flags.clone(), handles.len),
        );
    }

    Ok(primitive_search::first_flag(policy, tail_flags, handles.len, handles.len)?.is_none())
}

impl<Source, Pred> KernelPartitionInput<Pred> for Source
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: Scalar + 'static,
    Source::Expr: DeviceGpuExpr<Source::Item> + GpuExpr<Source::Item>,
    Pred: PredicateOp<Source::Item>,
{
    type Runtime = Source::Runtime;
    type Output = DeviceSoA1<DeviceVec<Source::Runtime, Source::Item>>;
    type SplitOutput = (Self::Output, Self::Output);
    type Env = Pred::Env;

    fn is_partitioned_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        env: <Self::Env as cubecl::prelude::LaunchArg>::RuntimeArg<Self::Runtime>,
    ) -> Result<bool, Error> {
        <Source as KernelColumn>::validate(&self)?;
        let handles = crate::detail::api::device_expr_selection_handles_with_policy::<Source, Pred>(
            policy, &self, false, env,
        )?;
        is_partitioned_single_read(policy, handles)
    }

    fn partition_copy_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        env: <Self::Env as cubecl::prelude::LaunchArg>::RuntimeArg<Self::Runtime>,
    ) -> Result<Self::SplitOutput, Error> {
        <Source as KernelColumn>::validate(&self)?;
        let handles = crate::detail::api::device_expr_selection_handles_with_policy::<Source, Pred>(
            policy, &self, false, env,
        )?;
        let matching_count = select::selected_count(policy, &handles)?;
        let failing_count = handles.len - matching_count;
        let matching = crate::detail::api::device_expr_compact_with_selection_with_policy(
            policy,
            &self,
            &handles,
            matching_count,
        )?;
        let failing = crate::detail::api::device_expr_compact_rejected_with_selection_with_policy(
            policy,
            &self,
            &handles,
            failing_count,
        )?;
        Ok((
            DeviceSoA1 { source: matching },
            DeviceSoA1 { source: failing },
        ))
    }
}

macro_rules! impl_kernel_partition_tuple1 {
    ($target:ty, $field:tt) => {
        impl<Source, Pred> KernelPartitionInput<Pred> for $target
        where
            Source: KernelColumn + KernelColumnAt<S0>,
            Source::Item: Scalar + 'static,
            Source::Expr: DeviceGpuExpr<Source::Item> + GpuExpr<Source::Item>,
            Pred: PredicateOp<Source::Item>,
        {
            type Runtime = Source::Runtime;
            type Output = DeviceSoA1<DeviceVec<Source::Runtime, Source::Item>>;
            type SplitOutput = (Self::Output, Self::Output);
            type Env = Pred::Env;

            fn is_partitioned_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
                env: <Self::Env as cubecl::prelude::LaunchArg>::RuntimeArg<Self::Runtime>,
            ) -> Result<bool, Error> {
                <Source as KernelColumn>::validate(&self.$field)?;
                let handles = crate::detail::api::device_expr_selection_handles_with_policy::<
                    Source,
                    Pred,
                >(policy, &self.$field, false, env)?;
                is_partitioned_single_read(policy, handles)
            }

            fn partition_copy_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
                env: <Self::Env as cubecl::prelude::LaunchArg>::RuntimeArg<Self::Runtime>,
            ) -> Result<Self::SplitOutput, Error> {
                <Source as KernelColumn>::validate(&self.$field)?;
                let handles = crate::detail::api::device_expr_selection_handles_with_policy::<
                    Source,
                    Pred,
                >(policy, &self.$field, false, env)?;
                let matching_count = select::selected_count(policy, &handles)?;
                let failing_count = handles.len - matching_count;
                let matching = crate::detail::api::device_expr_compact_with_selection_with_policy(
                    policy,
                    &self.$field,
                    &handles,
                    matching_count,
                )?;
                let failing =
                    crate::detail::api::device_expr_compact_rejected_with_selection_with_policy(
                        policy,
                        &self.$field,
                        &handles,
                        failing_count,
                    )?;
                Ok((
                    DeviceSoA1 { source: matching },
                    DeviceSoA1 { source: failing },
                ))
            }
        }
    };
}

impl_kernel_partition_tuple1!(SoAView1<Source>, source);
impl_kernel_partition_tuple1!(DeviceSoA1<Source>, source);

impl<Source, Pred> KernelPartitionInput<Pred> for (Source,)
where
    Source: KernelPartitionInput<crate::detail::api::Tuple1PredicateOp<Pred>>,
{
    type Runtime =
        <Source as KernelPartitionInput<crate::detail::api::Tuple1PredicateOp<Pred>>>::Runtime;
    type Output =
        <Source as KernelPartitionInput<crate::detail::api::Tuple1PredicateOp<Pred>>>::Output;
    type SplitOutput =
        <Source as KernelPartitionInput<crate::detail::api::Tuple1PredicateOp<Pred>>>::SplitOutput;
    type Env = <Source as KernelPartitionInput<crate::detail::api::Tuple1PredicateOp<Pred>>>::Env;

    fn is_partitioned_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        env: <Self::Env as cubecl::prelude::LaunchArg>::RuntimeArg<Self::Runtime>,
    ) -> Result<bool, Error> {
        <Source as KernelPartitionInput<crate::detail::api::Tuple1PredicateOp<Pred>>>::is_partitioned_read(
            self.0, policy, env,
        )
    }

    fn partition_copy_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        env: <Self::Env as cubecl::prelude::LaunchArg>::RuntimeArg<Self::Runtime>,
    ) -> Result<Self::SplitOutput, Error> {
        <Source as KernelPartitionInput<crate::detail::api::Tuple1PredicateOp<Pred>>>::partition_copy_read(
            self.0, policy, env,
        )
    }
}

macro_rules! impl_kernel_partition_tuple2 {
    ($target:ty, $out:ident, $left:tt, $right:tt) => {
        impl<Left, Right, Pred> KernelPartitionInput<Pred> for $target
        where
            Left: KernelColumn + KernelColumnAt<S0>,
            Right: KernelColumn<Runtime = Left::Runtime> + KernelColumnAt<S0>,
            Left::Item: Scalar + 'static,
            Right::Item: Scalar + 'static,
            Left::Expr: DeviceGpuExpr<Left::Item>,
            Right::Expr: DeviceGpuExpr<Right::Item>,
            Pred: PredicateOp<(Left::Item, Right::Item)>,
        {
            type Runtime = Left::Runtime;
            type Output =
                $out<DeviceVec<Left::Runtime, Left::Item>, DeviceVec<Left::Runtime, Right::Item>>;
            type SplitOutput = (Self::Output, Self::Output);
            type Env = Pred::Env;

            fn is_partitioned_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
                env: <Self::Env as cubecl::prelude::LaunchArg>::RuntimeArg<Self::Runtime>,
            ) -> Result<bool, Error> {
                let handles = tuple2_selection_handles_read::<Left, Right, Pred>(
                    policy,
                    &self.$left,
                    &self.$right,
                    false,
                    env,
                )?;
                is_partitioned_from_flags_read(policy, &handles)
            }

            fn partition_copy_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
                env: <Self::Env as cubecl::prelude::LaunchArg>::RuntimeArg<Self::Runtime>,
            ) -> Result<Self::SplitOutput, Error> {
                let handles = tuple2_selection_handles_read::<Left, Right, Pred>(
                    policy,
                    &self.$left,
                    &self.$right,
                    false,
                    env,
                )?;
                let selected_count = select::selected_count(policy, &handles)?;
                let rejected_count = handles.len - selected_count;
                Ok((
                    $out {
                        left: crate::detail::api::device_expr_compact_with_selection_with_policy(
                            policy,
                            &self.$left,
                            &handles,
                            selected_count,
                        )?,
                        right: crate::detail::api::device_expr_compact_with_selection_with_policy(
                            policy,
                            &self.$right,
                            &handles,
                            selected_count,
                        )?,
                    },
                    $out {
                        left:
                            crate::detail::api::device_expr_compact_rejected_with_selection_with_policy(
                                policy,
                                &self.$left,
                                &handles,
                                rejected_count,
                            )?,
                        right:
                            crate::detail::api::device_expr_compact_rejected_with_selection_with_policy(
                                policy,
                                &self.$right,
                                &handles,
                                rejected_count,
                            )?,
                    },
                ))
            }
        }
    };
}

impl_kernel_partition_tuple2!(SoAView2<Left, Right>, DeviceSoA2, left, right);
impl_kernel_partition_tuple2!(DeviceSoA2<Left, Right>, DeviceSoA2, left, right);

impl<Left, Right, Pred> KernelPartitionInput<Pred> for (Left, Right)
where
    SoAView2<Left, Right>: KernelPartitionInput<Pred>,
{
    type Runtime = <SoAView2<Left, Right> as KernelPartitionInput<Pred>>::Runtime;
    type Output = <SoAView2<Left, Right> as KernelPartitionInput<Pred>>::Output;
    type SplitOutput = <SoAView2<Left, Right> as KernelPartitionInput<Pred>>::SplitOutput;
    type Env = <SoAView2<Left, Right> as KernelPartitionInput<Pred>>::Env;

    fn is_partitioned_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        env: <Self::Env as cubecl::prelude::LaunchArg>::RuntimeArg<Self::Runtime>,
    ) -> Result<bool, Error> {
        <SoAView2<Left, Right> as KernelPartitionInput<Pred>>::is_partitioned_read(
            SoAView2 {
                left: self.0,
                right: self.1,
            },
            policy,
            env,
        )
    }

    fn partition_copy_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        env: <Self::Env as cubecl::prelude::LaunchArg>::RuntimeArg<Self::Runtime>,
    ) -> Result<Self::SplitOutput, Error> {
        <SoAView2<Left, Right> as KernelPartitionInput<Pred>>::partition_copy_read(
            SoAView2 {
                left: self.0,
                right: self.1,
            },
            policy,
            env,
        )
    }
}

macro_rules! impl_kernel_partition_tuple3 {
    ($target:ty, $out:ident, $first:tt, $second:tt, $third:tt) => {
        impl<First, Second, Third, Pred> KernelPartitionInput<Pred> for $target
        where
            First: KernelColumn + KernelColumnAt<S0>,
            Second: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
            Third: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
            First::Item: Scalar + 'static,
            Second::Item: Scalar + 'static,
            Third::Item: Scalar + 'static,
            First::Expr: DeviceGpuExpr<First::Item>,
            Second::Expr: DeviceGpuExpr<Second::Item>,
            Third::Expr: DeviceGpuExpr<Third::Item>,
            Pred: PredicateOp<(First::Item, Second::Item, Third::Item)>,
        {
            type Runtime = First::Runtime;
            type Output = $out<
                DeviceVec<First::Runtime, First::Item>,
                DeviceVec<First::Runtime, Second::Item>,
                DeviceVec<First::Runtime, Third::Item>,
            >;
            type SplitOutput = (Self::Output, Self::Output);
            type Env = Pred::Env;

            fn is_partitioned_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
                env: <Self::Env as cubecl::prelude::LaunchArg>::RuntimeArg<Self::Runtime>,
            ) -> Result<bool, Error> {
                let handles = tuple3_selection_handles_read::<First, Second, Third, Pred>(
                    policy,
                    &self.$first,
                    &self.$second,
                    &self.$third,
                    false,
                    env,
                )?;
                is_partitioned_from_flags_read(policy, &handles)
            }

            fn partition_copy_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
                env: <Self::Env as cubecl::prelude::LaunchArg>::RuntimeArg<Self::Runtime>,
            ) -> Result<Self::SplitOutput, Error> {
                let handles = tuple3_selection_handles_read::<First, Second, Third, Pred>(
                    policy,
                    &self.$first,
                    &self.$second,
                    &self.$third,
                    false,
                    env,
                )?;
                let selected_count = select::selected_count(policy, &handles)?;
                let rejected_count = handles.len - selected_count;
                Ok((
                    $out {
                        first: crate::detail::api::device_expr_compact_with_selection_with_policy(
                            policy,
                            &self.$first,
                            &handles,
                            selected_count,
                        )?,
                        second: crate::detail::api::device_expr_compact_with_selection_with_policy(
                            policy,
                            &self.$second,
                            &handles,
                            selected_count,
                        )?,
                        third: crate::detail::api::device_expr_compact_with_selection_with_policy(
                            policy,
                            &self.$third,
                            &handles,
                            selected_count,
                        )?,
                    },
                    $out {
                        first:
                            crate::detail::api::device_expr_compact_rejected_with_selection_with_policy(
                                policy,
                                &self.$first,
                                &handles,
                                rejected_count,
                            )?,
                        second:
                            crate::detail::api::device_expr_compact_rejected_with_selection_with_policy(
                                policy,
                                &self.$second,
                                &handles,
                                rejected_count,
                            )?,
                        third:
                            crate::detail::api::device_expr_compact_rejected_with_selection_with_policy(
                                policy,
                                &self.$third,
                                &handles,
                                rejected_count,
                            )?,
                    },
                ))
            }
        }
    };
}

impl_kernel_partition_tuple3!(
    SoAView3<First, Second, Third>,
    DeviceSoA3,
    first,
    second,
    third
);
impl_kernel_partition_tuple3!(
    DeviceSoA3<First, Second, Third>,
    DeviceSoA3,
    first,
    second,
    third
);

impl<First, Second, Third, Pred> KernelPartitionInput<Pred> for (First, Second, Third)
where
    SoAView3<First, Second, Third>: KernelPartitionInput<Pred>,
{
    type Runtime = <SoAView3<First, Second, Third> as KernelPartitionInput<Pred>>::Runtime;
    type Output = <SoAView3<First, Second, Third> as KernelPartitionInput<Pred>>::Output;
    type SplitOutput = <SoAView3<First, Second, Third> as KernelPartitionInput<Pred>>::SplitOutput;
    type Env = <SoAView3<First, Second, Third> as KernelPartitionInput<Pred>>::Env;

    fn is_partitioned_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        env: <Self::Env as cubecl::prelude::LaunchArg>::RuntimeArg<Self::Runtime>,
    ) -> Result<bool, Error> {
        <SoAView3<First, Second, Third> as KernelPartitionInput<Pred>>::is_partitioned_read(
            SoAView3 {
                first: self.0,
                second: self.1,
                third: self.2,
            },
            policy,
            env,
        )
    }

    fn partition_copy_read(
        self,
        policy: &CubePolicy<Self::Runtime>,
        env: <Self::Env as cubecl::prelude::LaunchArg>::RuntimeArg<Self::Runtime>,
    ) -> Result<Self::SplitOutput, Error> {
        <SoAView3<First, Second, Third> as KernelPartitionInput<Pred>>::partition_copy_read(
            SoAView3 {
                first: self.0,
                second: self.1,
                third: self.2,
            },
            policy,
            env,
        )
    }
}

struct StagedUniqueColumn {
    slot0: (cubecl::server::Handle, usize),
    slot1: (cubecl::server::Handle, usize),
    slot2: (cubecl::server::Handle, usize),
    slot3: (cubecl::server::Handle, usize),
    slot_offsets: cubecl::server::Handle,
}

fn stage_unique_column<Source>(
    policy: &CubePolicy<Source::Runtime>,
    source: &Source,
) -> Result<StagedUniqueColumn, Error>
where
    Source: KernelColumn + KernelColumnAt<S0>,
{
    let bindings = <Source as KernelColumn>::stage(source, policy)?;
    let slot_offsets = bindings.slot_offsets_handle(policy.client())?;
    let slot0 = bindings.slots.first().unwrap();
    let slot1 = bindings.slots.get(1).unwrap_or(slot0);
    let slot2 = bindings.slots.get(2).unwrap_or(slot0);
    let slot3 = bindings.slots.get(3).unwrap_or(slot0);
    Ok(StagedUniqueColumn {
        slot0: (slot0.0.clone(), slot0.1),
        slot1: (slot1.0.clone(), slot1.1),
        slot2: (slot2.0.clone(), slot2.1),
        slot3: (slot3.0.clone(), slot3.1),
        slot_offsets,
    })
}

fn unique_block_count_read(len: usize) -> Result<u32, Error> {
    let block_count = len.div_ceil(BLOCK_UNIQUE_SIZE as usize);
    u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })
}

pub(in crate::detail::read) fn unique_one_flags_read<Source, Pred>(
    policy: &CubePolicy<Source::Runtime>,
    source: &Source,
) -> Result<cubecl::server::Handle, Error>
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: Scalar + 'static,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Pred: BinaryPredicateOp<Source::Item>,
{
    <Source as KernelColumn>::validate(source)?;
    let len = <Source as KernelColumn>::len(source);
    let client = policy.client();
    if len == 0 {
        return Ok(policy.empty_handle());
    }

    let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
    let flag_handle = client.empty(len * std::mem::size_of::<u32>());
    let bindings = <Source as KernelColumn>::stage(source, policy)?;
    let slot_offsets = bindings.slot_offsets_handle(client)?;
    let slot0 = bindings.slots.first().unwrap();
    let slot1 = bindings.slots.get(1).unwrap_or(slot0);
    let slot2 = bindings.slots.get(2).unwrap_or(slot0);
    let slot3 = bindings.slots.get(3).unwrap_or(slot0);
    let block_count_u32 = unique_block_count_read(len)?;

    unsafe {
        unique_by_key_device_expr_flags_kernel::launch_unchecked::<
            Source::Item,
            Source::Expr,
            Pred,
            Source::Runtime,
        >(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_UNIQUE_SIZE),
            BufferArg::from_raw_parts(slot0.0.clone(), slot0.1),
            BufferArg::from_raw_parts(slot1.0.clone(), slot1.1),
            BufferArg::from_raw_parts(slot2.0.clone(), slot2.1),
            BufferArg::from_raw_parts(slot3.0.clone(), slot3.1),
            BufferArg::from_raw_parts(slot_offsets.clone(), 4),
            BufferArg::from_raw_parts(len_handle.clone(), 1),
            BufferArg::from_raw_parts(flag_handle.clone(), len),
        );
    }

    Ok(flag_handle)
}

pub(crate) fn unique_tuple2_flags_read<Left, Right, Pred>(
    policy: &CubePolicy<Left::Runtime>,
    left: &Left,
    right: &Right,
) -> Result<cubecl::server::Handle, Error>
where
    Left: KernelColumn + KernelColumnAt<S0>,
    Right: KernelColumn<Runtime = Left::Runtime> + KernelColumnAt<S0>,
    Left::Item: Scalar + 'static,
    Right::Item: Scalar + 'static,
    Left::Expr: DeviceGpuExpr<Left::Item>,
    Right::Expr: DeviceGpuExpr<Right::Item>,
    Pred: BinaryPredicateOp<(Left::Item, Right::Item)>,
{
    validate_columns2(left, right)?;
    let len = <Left as KernelColumn>::len(left);
    let client = policy.client();
    if len == 0 {
        return Ok(policy.empty_handle());
    }

    let staged_left = stage_unique_column(policy, left)?;
    let staged_right = stage_unique_column(policy, right)?;
    let flag = client.empty(len * std::mem::size_of::<u32>());
    let block_count_u32 = unique_block_count_read(len)?;
    unsafe {
        tuple2_unique_device_expr_flags_kernel::launch_unchecked::<
            Left::Item,
            Right::Item,
            Left::Expr,
            Right::Expr,
            Pred,
            Left::Runtime,
        >(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_UNIQUE_SIZE),
            BufferArg::from_raw_parts(staged_left.slot0.0.clone(), staged_left.slot0.1),
            BufferArg::from_raw_parts(staged_left.slot1.0.clone(), staged_left.slot1.1),
            BufferArg::from_raw_parts(staged_left.slot2.0.clone(), staged_left.slot2.1),
            BufferArg::from_raw_parts(staged_left.slot3.0.clone(), staged_left.slot3.1),
            BufferArg::from_raw_parts(staged_left.slot_offsets.clone(), 4),
            BufferArg::from_raw_parts(staged_right.slot0.0.clone(), staged_right.slot0.1),
            BufferArg::from_raw_parts(staged_right.slot1.0.clone(), staged_right.slot1.1),
            BufferArg::from_raw_parts(staged_right.slot2.0.clone(), staged_right.slot2.1),
            BufferArg::from_raw_parts(staged_right.slot3.0.clone(), staged_right.slot3.1),
            BufferArg::from_raw_parts(staged_right.slot_offsets.clone(), 4),
            BufferArg::from_raw_parts(flag.clone(), len),
        );
    }
    Ok(flag)
}

pub(crate) fn unique_tuple3_flags_read<First, Second, Third, Pred>(
    policy: &CubePolicy<First::Runtime>,
    first: &First,
    second: &Second,
    third: &Third,
) -> Result<cubecl::server::Handle, Error>
where
    First: KernelColumn + KernelColumnAt<S0>,
    Second: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
    Third: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
    First::Item: Scalar + 'static,
    Second::Item: Scalar + 'static,
    Third::Item: Scalar + 'static,
    First::Expr: DeviceGpuExpr<First::Item>,
    Second::Expr: DeviceGpuExpr<Second::Item>,
    Third::Expr: DeviceGpuExpr<Third::Item>,
    Pred: BinaryPredicateOp<(First::Item, Second::Item, Third::Item)>,
{
    validate_columns3(first, second, third)?;
    let len = <First as KernelColumn>::len(first);
    let client = policy.client();
    if len == 0 {
        return Ok(policy.empty_handle());
    }

    let staged_first = stage_unique_column(policy, first)?;
    let staged_second = stage_unique_column(policy, second)?;
    let staged_third = stage_unique_column(policy, third)?;
    let flag = client.empty(len * std::mem::size_of::<u32>());
    let block_count_u32 = unique_block_count_read(len)?;
    unsafe {
        tuple3_unique_device_expr_flags_kernel::launch_unchecked::<
            First::Item,
            Second::Item,
            Third::Item,
            First::Expr,
            Second::Expr,
            Third::Expr,
            Pred,
            First::Runtime,
        >(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_UNIQUE_SIZE),
            BufferArg::from_raw_parts(staged_first.slot0.0.clone(), staged_first.slot0.1),
            BufferArg::from_raw_parts(staged_first.slot1.0.clone(), staged_first.slot1.1),
            BufferArg::from_raw_parts(staged_first.slot2.0.clone(), staged_first.slot2.1),
            BufferArg::from_raw_parts(staged_first.slot3.0.clone(), staged_first.slot3.1),
            BufferArg::from_raw_parts(staged_first.slot_offsets.clone(), 4),
            BufferArg::from_raw_parts(staged_second.slot0.0.clone(), staged_second.slot0.1),
            BufferArg::from_raw_parts(staged_second.slot1.0.clone(), staged_second.slot1.1),
            BufferArg::from_raw_parts(staged_second.slot2.0.clone(), staged_second.slot2.1),
            BufferArg::from_raw_parts(staged_second.slot3.0.clone(), staged_second.slot3.1),
            BufferArg::from_raw_parts(staged_second.slot_offsets.clone(), 4),
            BufferArg::from_raw_parts(staged_third.slot0.0.clone(), staged_third.slot0.1),
            BufferArg::from_raw_parts(staged_third.slot1.0.clone(), staged_third.slot1.1),
            BufferArg::from_raw_parts(staged_third.slot2.0.clone(), staged_third.slot2.1),
            BufferArg::from_raw_parts(staged_third.slot3.0.clone(), staged_third.slot3.1),
            BufferArg::from_raw_parts(staged_third.slot_offsets.clone(), 4),
            BufferArg::from_raw_parts(flag.clone(), len),
        );
    }
    Ok(flag)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn unique_tuple7_flags_read<A, B, C, D, E, F, G, Pred>(
    policy: &CubePolicy<A::Runtime>,
    a: &A,
    b: &B,
    c: &C,
    d: &D,
    e: &E,
    f: &F,
    g: &G,
) -> Result<cubecl::server::Handle, Error>
where
    A: KernelColumn + KernelColumnAt<S0>,
    B: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    D: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    E: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    F: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    G: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    A::Item: Scalar + 'static,
    B::Item: Scalar + 'static,
    C::Item: Scalar + 'static,
    D::Item: Scalar + 'static,
    E::Item: Scalar + 'static,
    F::Item: Scalar + 'static,
    G::Item: Scalar + 'static,
    A::Expr: DeviceGpuExpr<A::Item>,
    B::Expr: DeviceGpuExpr<B::Item>,
    C::Expr: DeviceGpuExpr<C::Item>,
    D::Expr: DeviceGpuExpr<D::Item>,
    E::Expr: DeviceGpuExpr<E::Item>,
    F::Expr: DeviceGpuExpr<F::Item>,
    G::Expr: DeviceGpuExpr<G::Item>,
    Pred: BinaryPredicateOp<(
        A::Item,
        B::Item,
        C::Item,
        D::Item,
        E::Item,
        F::Item,
        G::Item,
    )>,
{
    A::validate(a)?;
    B::validate(b)?;
    C::validate(c)?;
    D::validate(d)?;
    E::validate(e)?;
    F::validate(f)?;
    G::validate(g)?;
    let len = A::len(a);
    ensure_same_len(B::len(b), len)?;
    ensure_same_len(C::len(c), len)?;
    ensure_same_len(D::len(d), len)?;
    ensure_same_len(E::len(e), len)?;
    ensure_same_len(F::len(f), len)?;
    ensure_same_len(G::len(g), len)?;
    let client = policy.client();
    if len == 0 {
        return Ok(policy.empty_handle());
    }

    let staged_a = stage_unique_column(policy, a)?;
    let staged_b = stage_unique_column(policy, b)?;
    let staged_c = stage_unique_column(policy, c)?;
    let staged_d = stage_unique_column(policy, d)?;
    let staged_e = stage_unique_column(policy, e)?;
    let staged_f = stage_unique_column(policy, f)?;
    let staged_g = stage_unique_column(policy, g)?;
    let flag = client.empty(len * std::mem::size_of::<u32>());
    let block_count_u32 = unique_block_count_read(len)?;
    unsafe {
        tuple7_unique_device_expr_flags_kernel::launch_unchecked::<
            A::Item,
            B::Item,
            C::Item,
            D::Item,
            E::Item,
            F::Item,
            G::Item,
            A::Expr,
            B::Expr,
            C::Expr,
            D::Expr,
            E::Expr,
            F::Expr,
            G::Expr,
            Pred,
            A::Runtime,
        >(
            client,
            CubeCount::Static(block_count_u32, 1, 1),
            CubeDim::new_1d(BLOCK_UNIQUE_SIZE),
            BufferArg::from_raw_parts(staged_a.slot0.0.clone(), staged_a.slot0.1),
            BufferArg::from_raw_parts(staged_a.slot1.0.clone(), staged_a.slot1.1),
            BufferArg::from_raw_parts(staged_a.slot2.0.clone(), staged_a.slot2.1),
            BufferArg::from_raw_parts(staged_a.slot3.0.clone(), staged_a.slot3.1),
            BufferArg::from_raw_parts(staged_a.slot_offsets.clone(), 4),
            BufferArg::from_raw_parts(staged_b.slot0.0.clone(), staged_b.slot0.1),
            BufferArg::from_raw_parts(staged_b.slot1.0.clone(), staged_b.slot1.1),
            BufferArg::from_raw_parts(staged_b.slot2.0.clone(), staged_b.slot2.1),
            BufferArg::from_raw_parts(staged_b.slot3.0.clone(), staged_b.slot3.1),
            BufferArg::from_raw_parts(staged_b.slot_offsets.clone(), 4),
            BufferArg::from_raw_parts(staged_c.slot0.0.clone(), staged_c.slot0.1),
            BufferArg::from_raw_parts(staged_c.slot1.0.clone(), staged_c.slot1.1),
            BufferArg::from_raw_parts(staged_c.slot2.0.clone(), staged_c.slot2.1),
            BufferArg::from_raw_parts(staged_c.slot3.0.clone(), staged_c.slot3.1),
            BufferArg::from_raw_parts(staged_c.slot_offsets.clone(), 4),
            BufferArg::from_raw_parts(staged_d.slot0.0.clone(), staged_d.slot0.1),
            BufferArg::from_raw_parts(staged_d.slot1.0.clone(), staged_d.slot1.1),
            BufferArg::from_raw_parts(staged_d.slot2.0.clone(), staged_d.slot2.1),
            BufferArg::from_raw_parts(staged_d.slot3.0.clone(), staged_d.slot3.1),
            BufferArg::from_raw_parts(staged_d.slot_offsets.clone(), 4),
            BufferArg::from_raw_parts(staged_e.slot0.0.clone(), staged_e.slot0.1),
            BufferArg::from_raw_parts(staged_e.slot1.0.clone(), staged_e.slot1.1),
            BufferArg::from_raw_parts(staged_e.slot2.0.clone(), staged_e.slot2.1),
            BufferArg::from_raw_parts(staged_e.slot3.0.clone(), staged_e.slot3.1),
            BufferArg::from_raw_parts(staged_e.slot_offsets.clone(), 4),
            BufferArg::from_raw_parts(staged_f.slot0.0.clone(), staged_f.slot0.1),
            BufferArg::from_raw_parts(staged_f.slot1.0.clone(), staged_f.slot1.1),
            BufferArg::from_raw_parts(staged_f.slot2.0.clone(), staged_f.slot2.1),
            BufferArg::from_raw_parts(staged_f.slot3.0.clone(), staged_f.slot3.1),
            BufferArg::from_raw_parts(staged_f.slot_offsets.clone(), 4),
            BufferArg::from_raw_parts(staged_g.slot0.0.clone(), staged_g.slot0.1),
            BufferArg::from_raw_parts(staged_g.slot1.0.clone(), staged_g.slot1.1),
            BufferArg::from_raw_parts(staged_g.slot2.0.clone(), staged_g.slot2.1),
            BufferArg::from_raw_parts(staged_g.slot3.0.clone(), staged_g.slot3.1),
            BufferArg::from_raw_parts(staged_g.slot_offsets.clone(), 4),
            BufferArg::from_raw_parts(flag.clone(), len),
        );
    }
    Ok(flag)
}

impl<Source, Pred> KernelUniqueInput<Pred> for Source
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Item: Scalar + 'static,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Pred: BinaryPredicateOp<Source::Item>,
{
    type Runtime = Source::Runtime;
    type Output = DeviceSoA1<DeviceVec<Source::Runtime, Source::Item>>;

    fn unique_read(self, policy: &CubePolicy<Self::Runtime>) -> Result<Self::Output, Error> {
        let flags = unique_one_flags_read::<Source, Pred>(policy, &self)?;
        Ok(DeviceSoA1 {
            source: crate::detail::api::device_expr_compact_with_flags_with_policy(
                policy, &self, flags,
            )?,
        })
    }
}

macro_rules! impl_kernel_unique_tuple1 {
    ($target:ty, $field:tt) => {
        impl<Source, Pred> KernelUniqueInput<Pred> for $target
        where
            Source: KernelColumn + KernelColumnAt<S0>,
            Source::Item: Scalar + 'static,
            Source::Expr: DeviceGpuExpr<Source::Item>,
            Pred: BinaryPredicateOp<Source::Item>,
        {
            type Runtime = Source::Runtime;
            type Output = DeviceSoA1<DeviceVec<Source::Runtime, Source::Item>>;

            fn unique_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
            ) -> Result<Self::Output, Error> {
                let flags = unique_one_flags_read::<Source, Pred>(policy, &self.$field)?;
                Ok(DeviceSoA1 {
                    source: crate::detail::api::device_expr_compact_with_flags_with_policy(
                        policy,
                        &self.$field,
                        flags,
                    )?,
                })
            }
        }
    };
}

impl_kernel_unique_tuple1!(SoAView1<Source>, source);
impl_kernel_unique_tuple1!(DeviceSoA1<Source>, source);

impl<Source, Pred> KernelUniqueInput<Pred> for (Source,)
where
    Source: KernelUniqueInput<crate::detail::api::Tuple1Less<Pred>>,
{
    type Runtime = <Source as KernelUniqueInput<crate::detail::api::Tuple1Less<Pred>>>::Runtime;
    type Output = <Source as KernelUniqueInput<crate::detail::api::Tuple1Less<Pred>>>::Output;

    fn unique_read(self, policy: &CubePolicy<Self::Runtime>) -> Result<Self::Output, Error> {
        <Source as KernelUniqueInput<crate::detail::api::Tuple1Less<Pred>>>::unique_read(
            self.0, policy,
        )
    }
}

macro_rules! impl_kernel_unique_tuple2 {
    ($target:ty, $out:ident, $left:tt, $right:tt) => {
        impl<Left, Right, Pred> KernelUniqueInput<Pred> for $target
        where
            Left: KernelColumn + KernelColumnAt<S0>,
            Right: KernelColumn<Runtime = Left::Runtime> + KernelColumnAt<S0>,
            Left::Item: Scalar + 'static,
            Right::Item: Scalar + 'static,
            Left::Expr: DeviceGpuExpr<Left::Item>,
            Right::Expr: DeviceGpuExpr<Right::Item>,
            Pred: BinaryPredicateOp<(Left::Item, Right::Item)>,
        {
            type Runtime = Left::Runtime;
            type Output =
                $out<DeviceVec<Left::Runtime, Left::Item>, DeviceVec<Left::Runtime, Right::Item>>;

            fn unique_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
            ) -> Result<Self::Output, Error> {
                validate_columns2(&self.$left, &self.$right)?;
                let len = <Left as KernelColumn>::len(&self.$left);
                let left = stage_unique_column(policy, &self.$left)?;
                let right = stage_unique_column(policy, &self.$right)?;
                let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
                let client = policy.client();
                let flag = client.empty(len * std::mem::size_of::<u32>());
                if len != 0 {
                    let block_count_u32 = unique_block_count_read(len)?;
                    unsafe {
                        tuple2_unique_device_expr_flags_kernel::launch_unchecked::<
                            Left::Item,
                            Right::Item,
                            Left::Expr,
                            Right::Expr,
                            Pred,
                            Left::Runtime,
                        >(
                            client,
                            CubeCount::Static(block_count_u32, 1, 1),
                            CubeDim::new_1d(BLOCK_UNIQUE_SIZE),
                            BufferArg::from_raw_parts(left.slot0.0.clone(), left.slot0.1),
                            BufferArg::from_raw_parts(left.slot1.0.clone(), left.slot1.1),
                            BufferArg::from_raw_parts(left.slot2.0.clone(), left.slot2.1),
                            BufferArg::from_raw_parts(left.slot3.0.clone(), left.slot3.1),
                            BufferArg::from_raw_parts(left.slot_offsets.clone(), 4),
                            BufferArg::from_raw_parts(right.slot0.0.clone(), right.slot0.1),
                            BufferArg::from_raw_parts(right.slot1.0.clone(), right.slot1.1),
                            BufferArg::from_raw_parts(right.slot2.0.clone(), right.slot2.1),
                            BufferArg::from_raw_parts(right.slot3.0.clone(), right.slot3.1),
                            BufferArg::from_raw_parts(right.slot_offsets.clone(), 4),
                            BufferArg::from_raw_parts(flag.clone(), len),
                        );
                    }
                }
                let handles =
                    select::handles_from_flags(policy, len, len_u32, flag, policy.empty_handle())?;
                let count = select::selected_count(policy, &handles)?;
                Ok($out {
                    left: crate::detail::api::device_expr_compact_with_selection_with_policy(
                        policy,
                        &self.$left,
                        &handles,
                        count,
                    )?,
                    right: crate::detail::api::device_expr_compact_with_selection_with_policy(
                        policy,
                        &self.$right,
                        &handles,
                        count,
                    )?,
                })
            }
        }
    };
}

impl_kernel_unique_tuple2!(SoAView2<Left, Right>, DeviceSoA2, left, right);
impl_kernel_unique_tuple2!(DeviceSoA2<Left, Right>, DeviceSoA2, left, right);

impl<Left, Right, Pred> KernelUniqueInput<Pred> for (Left, Right)
where
    SoAView2<Left, Right>: KernelUniqueInput<Pred>,
{
    type Runtime = <SoAView2<Left, Right> as KernelUniqueInput<Pred>>::Runtime;
    type Output = <SoAView2<Left, Right> as KernelUniqueInput<Pred>>::Output;

    fn unique_read(self, policy: &CubePolicy<Self::Runtime>) -> Result<Self::Output, Error> {
        <SoAView2<Left, Right> as KernelUniqueInput<Pred>>::unique_read(
            SoAView2 {
                left: self.0,
                right: self.1,
            },
            policy,
        )
    }
}

macro_rules! impl_kernel_unique_tuple3 {
    ($target:ty, $out:ident, $first:tt, $second:tt, $third:tt) => {
        impl<First, Second, Third, Pred> KernelUniqueInput<Pred> for $target
        where
            First: KernelColumn + KernelColumnAt<S0>,
            Second: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
            Third: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
            First::Item: Scalar + 'static,
            Second::Item: Scalar + 'static,
            Third::Item: Scalar + 'static,
            First::Expr: DeviceGpuExpr<First::Item>,
            Second::Expr: DeviceGpuExpr<Second::Item>,
            Third::Expr: DeviceGpuExpr<Third::Item>,
            Pred: BinaryPredicateOp<(First::Item, Second::Item, Third::Item)>,
        {
            type Runtime = First::Runtime;
            type Output = $out<
                DeviceVec<First::Runtime, First::Item>,
                DeviceVec<First::Runtime, Second::Item>,
                DeviceVec<First::Runtime, Third::Item>,
            >;

            fn unique_read(
                self,
                policy: &CubePolicy<Self::Runtime>,
            ) -> Result<Self::Output, Error> {
                validate_columns3(&self.$first, &self.$second, &self.$third)?;
                let len = <First as KernelColumn>::len(&self.$first);
                let first = stage_unique_column(policy, &self.$first)?;
                let second = stage_unique_column(policy, &self.$second)?;
                let third = stage_unique_column(policy, &self.$third)?;
                let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
                let client = policy.client();
                let flag = client.empty(len * std::mem::size_of::<u32>());
                if len != 0 {
                    let block_count_u32 = unique_block_count_read(len)?;
                    unsafe {
                        tuple3_unique_device_expr_flags_kernel::launch_unchecked::<
                            First::Item,
                            Second::Item,
                            Third::Item,
                            First::Expr,
                            Second::Expr,
                            Third::Expr,
                            Pred,
                            First::Runtime,
                        >(
                            client,
                            CubeCount::Static(block_count_u32, 1, 1),
                            CubeDim::new_1d(BLOCK_UNIQUE_SIZE),
                            BufferArg::from_raw_parts(first.slot0.0.clone(), first.slot0.1),
                            BufferArg::from_raw_parts(first.slot1.0.clone(), first.slot1.1),
                            BufferArg::from_raw_parts(first.slot2.0.clone(), first.slot2.1),
                            BufferArg::from_raw_parts(first.slot3.0.clone(), first.slot3.1),
                            BufferArg::from_raw_parts(first.slot_offsets.clone(), 4),
                            BufferArg::from_raw_parts(second.slot0.0.clone(), second.slot0.1),
                            BufferArg::from_raw_parts(second.slot1.0.clone(), second.slot1.1),
                            BufferArg::from_raw_parts(second.slot2.0.clone(), second.slot2.1),
                            BufferArg::from_raw_parts(second.slot3.0.clone(), second.slot3.1),
                            BufferArg::from_raw_parts(second.slot_offsets.clone(), 4),
                            BufferArg::from_raw_parts(third.slot0.0.clone(), third.slot0.1),
                            BufferArg::from_raw_parts(third.slot1.0.clone(), third.slot1.1),
                            BufferArg::from_raw_parts(third.slot2.0.clone(), third.slot2.1),
                            BufferArg::from_raw_parts(third.slot3.0.clone(), third.slot3.1),
                            BufferArg::from_raw_parts(third.slot_offsets.clone(), 4),
                            BufferArg::from_raw_parts(flag.clone(), len),
                        );
                    }
                }
                let handles =
                    select::handles_from_flags(policy, len, len_u32, flag, policy.empty_handle())?;
                let count = select::selected_count(policy, &handles)?;
                Ok($out {
                    first: crate::detail::api::device_expr_compact_with_selection_with_policy(
                        policy,
                        &self.$first,
                        &handles,
                        count,
                    )?,
                    second: crate::detail::api::device_expr_compact_with_selection_with_policy(
                        policy,
                        &self.$second,
                        &handles,
                        count,
                    )?,
                    third: crate::detail::api::device_expr_compact_with_selection_with_policy(
                        policy,
                        &self.$third,
                        &handles,
                        count,
                    )?,
                })
            }
        }
    };
}

impl_kernel_unique_tuple3!(SoAView3<First, Second, Third>, DeviceSoA3, first, second, third);
impl_kernel_unique_tuple3!(DeviceSoA3<First, Second, Third>, DeviceSoA3, first, second, third);

impl<First, Second, Third, Pred> KernelUniqueInput<Pred> for (First, Second, Third)
where
    SoAView3<First, Second, Third>: KernelUniqueInput<Pred>,
{
    type Runtime = <SoAView3<First, Second, Third> as KernelUniqueInput<Pred>>::Runtime;
    type Output = <SoAView3<First, Second, Third> as KernelUniqueInput<Pred>>::Output;

    fn unique_read(self, policy: &CubePolicy<Self::Runtime>) -> Result<Self::Output, Error> {
        <SoAView3<First, Second, Third> as KernelUniqueInput<Pred>>::unique_read(
            SoAView3 {
                first: self.0,
                second: self.1,
                third: self.2,
            },
            policy,
        )
    }
}
