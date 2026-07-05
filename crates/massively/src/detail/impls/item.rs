use super::*;

macro_rules! soa_type {
    ($a:ty) => {
        SoA1<$a>
    };
    ($a:ty, $b:ty) => {
        SoA2<$a, $b>
    };
    ($a:ty, $b:ty, $c:ty) => {
        SoA3<$a, $b, $c>
    };
    ($a:ty, $b:ty, $c:ty, $d:ty) => {
        SoA4<$a, $b, $c, $d>
    };
    ($a:ty, $b:ty, $c:ty, $d:ty, $e:ty) => {
        SoA5<$a, $b, $c, $d, $e>
    };
    ($a:ty, $b:ty, $c:ty, $d:ty, $e:ty, $f:ty) => {
        SoA6<$a, $b, $c, $d, $e, $f>
    };
    ($a:ty, $b:ty, $c:ty, $d:ty, $e:ty, $f:ty, $g:ty) => {
        SoA7<$a, $b, $c, $d, $e, $f, $g>
    };
}

macro_rules! soa_value {
    ($a:expr) => {
        SoA1($a)
    };
    ($a:expr, $b:expr) => {
        SoA2($a, $b)
    };
    ($a:expr, $b:expr, $c:expr) => {
        SoA3($a, $b, $c)
    };
    ($a:expr, $b:expr, $c:expr, $d:expr) => {
        SoA4($a, $b, $c, $d)
    };
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr) => {
        SoA5($a, $b, $c, $d, $e)
    };
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr) => {
        SoA6($a, $b, $c, $d, $e, $f)
    };
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr) => {
        SoA7($a, $b, $c, $d, $e, $f, $g)
    };
}

macro_rules! soa_into_inner {
    ($value:expr; $a:ident) => {{
        let SoA1($a) = $value;
        ($a.inner,)
    }};
    ($value:expr; $a:ident, $b:ident) => {{
        let SoA2($a, $b) = $value;
        ($a.inner, $b.inner)
    }};
    ($value:expr; $a:ident, $b:ident, $c:ident) => {{
        let SoA3($a, $b, $c) = $value;
        ($a.inner, $b.inner, $c.inner)
    }};
    ($value:expr; $a:ident, $b:ident, $c:ident, $d:ident) => {{
        let SoA4($a, $b, $c, $d) = $value;
        ($a.inner, $b.inner, $c.inner, $d.inner)
    }};
    ($value:expr; $a:ident, $b:ident, $c:ident, $d:ident, $e:ident) => {{
        let SoA5($a, $b, $c, $d, $e) = $value;
        ($a.inner, $b.inner, $c.inner, $d.inner, $e.inner)
    }};
    ($value:expr; $a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident) => {{
        let SoA6($a, $b, $c, $d, $e, $f) = $value;
        ($a.inner, $b.inner, $c.inner, $d.inner, $e.inner, $f.inner)
    }};
    ($value:expr; $a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident, $g:ident) => {{
        let SoA7($a, $b, $c, $d, $e, $f, $g) = $value;
        (
            $a.inner, $b.inner, $c.inner, $d.inner, $e.inner, $f.inner, $g.inner,
        )
    }};
}

macro_rules! alloc_inner {
    ($exec:expr, $len:expr; $( $ty:ty ),+) => {{
        let len = $len;
        let policy = $exec.policy();
        if len == 0 {
            Ok(($( policy.empty_device_vec::<$ty>(), )+))
        } else {
            let client = policy.client();
            let len_usize = usize_from_mindex(len);
            Ok(($(
                crate::detail::DeviceVec::from_handle(
                    policy.id(),
                    client.empty(len_usize * std::mem::size_of::<$ty>()),
                    len,
                ),
            )+))
        }
    }};
}

macro_rules! impl_scalar_mitem {
    ($( $ty:ty ),+ $(,)?) => {
        $(
            impl<R> MItem<R> for $ty
            where
                R: Runtime,
                $ty: MStorageElement + 'static,
            {
            }

            impl<R> sealed::MItemDispatch<R> for $ty
            where
                R: Runtime,
                $ty: MStorageElement + 'static,
            {
            }
        )+
    };
}

impl_scalar_mitem!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

macro_rules! impl_mitem_tuple {
    ($( $ty:ident : $var:ident ),+) => {
        impl<R, $( $ty ),+> MItem<R> for ($( $ty, )+)
        where
            R: Runtime,
            $( $ty: MStorageElement, )+
        {
        }

        impl<R, $( $ty ),+> MAlloc<R> for ($( $ty, )+)
        where
            R: Runtime,
            $( $ty: MStorageElement, )+
        {
            type Inner = ($( crate::detail::DeviceVec<R, $ty>, )+);
            type View = ($( crate::detail::device::DeviceColumnView<R, $ty>, )+);
            type Storage = soa_type!($( DeviceVec<R, $ty> ),+);

            fn storage_from_inner(inner: Self::Inner) -> Self::Storage {
                let ($( $var, )+) = inner;
                soa_value!($( DeviceVec::from_inner($var) ),+)
            }

            fn alloc_storage(exec: &Executor<R>, len: MIndex) -> Result<Self::Storage, Error> {
                Ok(Self::storage_from_inner(alloc_inner!(exec, len; $( $ty ),+)?))
            }
        }

        impl<R, $( $ty ),+> StorageFromInner<R> for soa_type!($( DeviceVec<R, $ty> ),+)
        where
            R: Runtime,
            $( $ty: MStorageElement + 'static, )+
        {
            type Item = ($( $ty, )+);

            fn from_inner(inner: <Self::Item as MAlloc<R>>::Inner) -> Self {
                <Self::Item as MAlloc<R>>::storage_from_inner(inner)
            }

            fn into_inner(self) -> <Self::Item as MAlloc<R>>::Inner {
                soa_into_inner!(self; $( $var ),+)
            }

            fn len(&self) -> MIndex {
                self.0.len()
            }
        }

        impl<R, $( $ty ),+> sealed::MItemDispatch<R> for ($( $ty, )+)
        where
            R: Runtime,
            $( $ty: MStorageElement, )+
        {
            fn transform_scalar_input<Input, Op>(
                policy: &crate::detail::CubePolicy<R>,
                input: crate::detail::device::DeviceColumnView<R, Input>,
                op: Op,
                env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<<Self as MAlloc<R>>::Inner, Error>
            where
                Input: MStorageElement + MItem<R>,
                Op: op::UnaryOp<R, Input, Output = Self>,
                Self: crate::detail::TransformUnaryOutput<
                    R,
                    Input,
                    KernelScalarInputOp<R, Op>,
                >,
                <Self as crate::detail::MItemStorage<
                    R,
                >>::Storage: crate::detail::MaterializeOutput<
                    Runtime = R,
                    Output = ($(
                        crate::detail::DeviceVec<R, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage = crate::detail::apply::TransformPayloadApply::unary::<Self, R, Input, KernelScalarInputOp<R, Op>>(policy, input, env)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            fn transform_unary<Input, Op>(
                policy: &crate::detail::CubePolicy<R>,
                input: crate::detail::device::DeviceColumnView<
                    R,
                    Input,
                >,
                op: Op,
                env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<<Self as MAlloc<R>>::Inner, Error>
            where
                Input: MStorageElement,
                Op: op::UnaryOp<R, (Input,), Output = Self>,
                Self: crate::detail::TransformUnaryOutput<
                    R,
                    Input,
                    KernelOp<R, Op>,
                >,
                <Self as crate::detail::MItemStorage<
                    R,
                >>::Storage: crate::detail::MaterializeOutput<
                    Runtime = R,
                    Output = ($(
                        crate::detail::DeviceVec<R, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage = crate::detail::apply::TransformPayloadApply::unary::<Self, R, Input, KernelOp<R, Op>>(policy, input, env)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            fn transform_binary<Left, Right, Op>(
                policy: &crate::detail::CubePolicy<R>,
                left: crate::detail::device::DeviceColumnView<
                    R,
                    Left,
                >,
                right: crate::detail::device::DeviceColumnView<
                    R,
                    Right,
                >,
                op: Op,
                env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<<Self as MAlloc<R>>::Inner, Error>
            where
                Left: MStorageElement,
                Right: MStorageElement,
                Op: op::UnaryOp<R, (Left, Right), Output = Self>,
                Self: crate::detail::TransformSoA2Output<
                    R,
                    Left,
                    Right,
                    KernelOp<R, Op>,
                >,
                <Self as crate::detail::MItemStorage<
                    R,
                >>::Storage: crate::detail::MaterializeOutput<
                    Runtime = R,
                    Output = ($(
                        crate::detail::DeviceVec<R, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage = crate::detail::apply::TransformPayloadApply::soa2::<Self, R, Left, Right, KernelOp<R, Op>>(policy, left, right, env)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            fn transform_ternary<First, Second, Third, Op>(
                policy: &crate::detail::CubePolicy<R>,
                first: crate::detail::device::DeviceColumnView<
                    R,
                    First,
                >,
                second: crate::detail::device::DeviceColumnView<
                    R,
                    Second,
                >,
                third: crate::detail::device::DeviceColumnView<
                    R,
                    Third,
                >,
                op: Op,
                env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<<Self as MAlloc<R>>::Inner, Error>
            where
                First: MStorageElement,
                Second: MStorageElement,
                Third: MStorageElement,
                Op: op::UnaryOp<R, (First, Second, Third), Output = Self>,
                Self: crate::detail::TransformSoA3Output<
                    R,
                    First,
                    Second,
                    Third,
                    KernelOp<R, Op>,
                >,
                <Self as crate::detail::MItemStorage<
                    R,
                >>::Storage: crate::detail::MaterializeOutput<
                    Runtime = R,
                    Output = ($(
                        crate::detail::DeviceVec<R, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage = crate::detail::apply::TransformPayloadApply::soa3::<Self, R, First, Second, Third, KernelOp<R, Op>>(policy, first, second, third, env)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            fn transform_quaternary<First, Second, Third, Fourth, Op>(
                policy: &crate::detail::CubePolicy<R>,
                first: crate::detail::device::DeviceColumnView<R, First>,
                second: crate::detail::device::DeviceColumnView<R, Second>,
                third: crate::detail::device::DeviceColumnView<R, Third>,
                fourth: crate::detail::device::DeviceColumnView<R, Fourth>,
                op: Op,
                env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<<Self as MAlloc<R>>::Inner, Error>
            where
                First: MStorageElement,
                Second: MStorageElement,
                Third: MStorageElement,
                Fourth: MStorageElement,
                Op: op::UnaryOp<R, (First, Second, Third, Fourth), Output = Self>,
                Self: crate::detail::TransformSoA4Output<
                    R,
                    First,
                    Second,
                    Third,
                    Fourth,
                    KernelOp<R, Op>,
                >,
                <Self as crate::detail::MItemStorage<
                    R,
                >>::Storage: crate::detail::MaterializeOutput<
                    Runtime = R,
                    Output = ($(
                        crate::detail::DeviceVec<R, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage = crate::detail::apply::TransformPayloadApply::soa4::<Self, R, First, Second, Third, Fourth, KernelOp<R, Op>>(policy, first, second, third, fourth, env)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            fn transform_quinary<First, Second, Third, Fourth, Fifth, Op>(
                policy: &crate::detail::CubePolicy<R>,
                first: crate::detail::device::DeviceColumnView<R, First>,
                second: crate::detail::device::DeviceColumnView<R, Second>,
                third: crate::detail::device::DeviceColumnView<R, Third>,
                fourth: crate::detail::device::DeviceColumnView<R, Fourth>,
                fifth: crate::detail::device::DeviceColumnView<R, Fifth>,
                op: Op,
                env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<<Self as MAlloc<R>>::Inner, Error>
            where
                First: MStorageElement,
                Second: MStorageElement,
                Third: MStorageElement,
                Fourth: MStorageElement,
                Fifth: MStorageElement,
                Op: op::UnaryOp<R, (First, Second, Third, Fourth, Fifth), Output = Self>,
                Self: crate::detail::TransformSoA5Output<
                    R,
                    First,
                    Second,
                    Third,
                    Fourth,
                    Fifth,
                    KernelOp<R, Op>,
                >,
                <Self as crate::detail::MItemStorage<
                    R,
                >>::Storage: crate::detail::MaterializeOutput<
                    Runtime = R,
                    Output = ($(
                        crate::detail::DeviceVec<R, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage = crate::detail::apply::TransformPayloadApply::soa5::<Self, R, First, Second, Third, Fourth, Fifth, KernelOp<R, Op>>(policy, first, second, third, fourth, fifth, env)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            #[allow(clippy::too_many_arguments)]
            fn transform_senary<First, Second, Third, Fourth, Fifth, Sixth, Op>(
                policy: &crate::detail::CubePolicy<R>,
                first: crate::detail::device::DeviceColumnView<R, First>,
                second: crate::detail::device::DeviceColumnView<R, Second>,
                third: crate::detail::device::DeviceColumnView<R, Third>,
                fourth: crate::detail::device::DeviceColumnView<R, Fourth>,
                fifth: crate::detail::device::DeviceColumnView<R, Fifth>,
                sixth: crate::detail::device::DeviceColumnView<R, Sixth>,
                op: Op,
                env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<<Self as MAlloc<R>>::Inner, Error>
            where
                First: MStorageElement,
                Second: MStorageElement,
                Third: MStorageElement,
                Fourth: MStorageElement,
                Fifth: MStorageElement,
                Sixth: MStorageElement,
                Op: op::UnaryOp<R, (First, Second, Third, Fourth, Fifth, Sixth), Output = Self>,
                Self: crate::detail::TransformSoA6Output<
                    R,
                    First,
                    Second,
                    Third,
                    Fourth,
                    Fifth,
                    Sixth,
                    KernelOp<R, Op>,
                >,
                <Self as crate::detail::MItemStorage<
                    R,
                >>::Storage: crate::detail::MaterializeOutput<
                    Runtime = R,
                    Output = ($(
                        crate::detail::DeviceVec<R, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage = crate::detail::apply::TransformPayloadApply::soa6::<Self, R, First, Second, Third, Fourth, Fifth, Sixth, KernelOp<R, Op>>(policy, first, second, third, fourth, fifth, sixth, env)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            #[allow(clippy::too_many_arguments)]
            fn transform_septenary<First, Second, Third, Fourth, Fifth, Sixth, Seventh, Op>(
                policy: &crate::detail::CubePolicy<R>,
                first: crate::detail::device::DeviceColumnView<R, First>,
                second: crate::detail::device::DeviceColumnView<R, Second>,
                third: crate::detail::device::DeviceColumnView<R, Third>,
                fourth: crate::detail::device::DeviceColumnView<R, Fourth>,
                fifth: crate::detail::device::DeviceColumnView<R, Fifth>,
                sixth: crate::detail::device::DeviceColumnView<R, Sixth>,
                seventh: crate::detail::device::DeviceColumnView<R, Seventh>,
                op: Op,
                env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<<Self as MAlloc<R>>::Inner, Error>
            where
                First: MStorageElement,
                Second: MStorageElement,
                Third: MStorageElement,
                Fourth: MStorageElement,
                Fifth: MStorageElement,
                Sixth: MStorageElement,
                Seventh: MStorageElement,
                Op: op::UnaryOp<
                    R,
                    (First, Second, Third, Fourth, Fifth, Sixth, Seventh),
                    Output = Self,
                >,
                Self: crate::detail::TransformSoA7Output<
                    R,
                    First,
                    Second,
                    Third,
                    Fourth,
                    Fifth,
                    Sixth,
                    Seventh,
                    KernelOp<R, Op>,
                >,
                <Self as crate::detail::MItemStorage<
                    R,
                >>::Storage: crate::detail::MaterializeOutput<
                    Runtime = R,
                    Output = ($(
                        crate::detail::DeviceVec<R, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage = crate::detail::apply::TransformPayloadApply::soa7::<Self, R, First, Second, Third, Fourth, Fifth, Sixth, Seventh, KernelOp<R, Op>>(policy, first, second, third, fourth, fifth, sixth, seventh, env)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            fn reduce_inner<Op>(
                policy: &crate::detail::CubePolicy<R>,
                input: <Self as MAlloc<R>>::Inner,
                init: Self,
                op: Op,
            ) -> Result<Self, Error>
            where
                Op: op::ReductionOp<R, Self>,
            {
                let _ = op;
                crate::detail::reduce(policy, input, init, KernelOp::<R, Op>::new())
            }

        }
    };
}

impl_mitem_tuple!(A: a);
impl_mitem_tuple!(A: a, B0: b);
impl_mitem_tuple!(A: a, B0: b, C: c);

macro_rules! impl_wide_mitem_tuple {
    ($( $ty:ident : $var:ident ),+) => {
        impl<R, $( $ty ),+> MItem<R> for ($( $ty, )+)
        where
            R: Runtime,
            $( $ty: MStorageElement, )+
        {
        }

        impl<R, $( $ty ),+> MAlloc<R> for ($( $ty, )+)
        where
            R: Runtime,
            $( $ty: MStorageElement, )+
        {
            type Inner = ($( crate::detail::DeviceVec<R, $ty>, )+);
            type View = ($( crate::detail::device::DeviceColumnView<R, $ty>, )+);
            type Storage = soa_type!($( DeviceVec<R, $ty> ),+);

            fn storage_from_inner(inner: Self::Inner) -> Self::Storage {
                let ($( $var, )+) = inner;
                soa_value!($( DeviceVec::from_inner($var) ),+)
            }

            fn alloc_storage(exec: &Executor<R>, len: MIndex) -> Result<Self::Storage, Error> {
                Ok(Self::storage_from_inner(alloc_inner!(exec, len; $( $ty ),+)?))
            }
        }

        impl<R, $( $ty ),+> StorageFromInner<R> for soa_type!($( DeviceVec<R, $ty> ),+)
        where
            R: Runtime,
            $( $ty: MStorageElement + 'static, )+
        {
            type Item = ($( $ty, )+);

            fn from_inner(inner: <Self::Item as MAlloc<R>>::Inner) -> Self {
                <Self::Item as MAlloc<R>>::storage_from_inner(inner)
            }

            fn into_inner(self) -> <Self::Item as MAlloc<R>>::Inner {
                soa_into_inner!(self; $( $var ),+)
            }

            fn len(&self) -> MIndex {
                self.0.len()
            }
        }

        impl<R, $( $ty ),+> sealed::MItemDispatch<R> for ($( $ty, )+)
        where
            R: Runtime,
            $( $ty: MStorageElement, )+
        {
            fn transform_scalar_input<Input, Op>(
                policy: &crate::detail::CubePolicy<R>,
                input: crate::detail::device::DeviceColumnView<R, Input>,
                op: Op,
                env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<<Self as MAlloc<R>>::Inner, Error>
            where
                Input: MStorageElement + MItem<R>,
                Op: op::UnaryOp<R, Input, Output = Self>,
                Self: crate::detail::TransformUnaryOutput<
                    R,
                    Input,
                    KernelScalarInputOp<R, Op>,
                >,
                <Self as crate::detail::MItemStorage<
                    R,
                >>::Storage: crate::detail::MaterializeOutput<
                    Runtime = R,
                    Output = ($(
                        crate::detail::DeviceVec<R, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage = crate::detail::apply::TransformPayloadApply::unary::<Self, R, Input, KernelScalarInputOp<R, Op>>(policy, input, env)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            fn transform_unary<Input, Op>(
                policy: &crate::detail::CubePolicy<R>,
                input: crate::detail::device::DeviceColumnView<
                    R,
                    Input,
                >,
                op: Op,
                env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<<Self as MAlloc<R>>::Inner, Error>
            where
                Input: MStorageElement,
                Op: op::UnaryOp<R, (Input,), Output = Self>,
                Self: crate::detail::TransformUnaryOutput<
                    R,
                    Input,
                    KernelOp<R, Op>,
                >,
                <Self as crate::detail::MItemStorage<
                    R,
                >>::Storage: crate::detail::MaterializeOutput<
                    Runtime = R,
                    Output = ($(
                        crate::detail::DeviceVec<R, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage = crate::detail::apply::TransformPayloadApply::unary::<Self, R, Input, KernelOp<R, Op>>(policy, input, env)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            fn transform_binary<Left, Right, Op>(
                policy: &crate::detail::CubePolicy<R>,
                left: crate::detail::device::DeviceColumnView<
                    R,
                    Left,
                >,
                right: crate::detail::device::DeviceColumnView<
                    R,
                    Right,
                >,
                op: Op,
                env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<<Self as MAlloc<R>>::Inner, Error>
            where
                Left: MStorageElement,
                Right: MStorageElement,
                Op: op::UnaryOp<R, (Left, Right), Output = Self>,
                Self: crate::detail::TransformSoA2Output<
                    R,
                    Left,
                    Right,
                    KernelOp<R, Op>,
                >,
                <Self as crate::detail::MItemStorage<
                    R,
                >>::Storage: crate::detail::MaterializeOutput<
                    Runtime = R,
                    Output = ($(
                        crate::detail::DeviceVec<R, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage = crate::detail::apply::TransformPayloadApply::soa2::<Self, R, Left, Right, KernelOp<R, Op>>(policy, left, right, env)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            fn transform_ternary<First, Second, Third, Op>(
                policy: &crate::detail::CubePolicy<R>,
                first: crate::detail::device::DeviceColumnView<
                    R,
                    First,
                >,
                second: crate::detail::device::DeviceColumnView<
                    R,
                    Second,
                >,
                third: crate::detail::device::DeviceColumnView<
                    R,
                    Third,
                >,
                op: Op,
                env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<<Self as MAlloc<R>>::Inner, Error>
            where
                First: MStorageElement,
                Second: MStorageElement,
                Third: MStorageElement,
                Op: op::UnaryOp<R, (First, Second, Third), Output = Self>,
                Self: crate::detail::TransformSoA3Output<
                    R,
                    First,
                    Second,
                    Third,
                    KernelOp<R, Op>,
                >,
                <Self as crate::detail::MItemStorage<
                    R,
                >>::Storage: crate::detail::MaterializeOutput<
                    Runtime = R,
                    Output = ($(
                        crate::detail::DeviceVec<R, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage = crate::detail::apply::TransformPayloadApply::soa3::<Self, R, First, Second, Third, KernelOp<R, Op>>(policy, first, second, third, env)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            fn transform_quaternary<First, Second, Third, Fourth, Op>(
                policy: &crate::detail::CubePolicy<R>,
                first: crate::detail::device::DeviceColumnView<R, First>,
                second: crate::detail::device::DeviceColumnView<R, Second>,
                third: crate::detail::device::DeviceColumnView<R, Third>,
                fourth: crate::detail::device::DeviceColumnView<R, Fourth>,
                op: Op,
                env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<<Self as MAlloc<R>>::Inner, Error>
            where
                First: MStorageElement,
                Second: MStorageElement,
                Third: MStorageElement,
                Fourth: MStorageElement,
                Op: op::UnaryOp<R, (First, Second, Third, Fourth), Output = Self>,
                Self: crate::detail::TransformSoA4Output<
                    R,
                    First,
                    Second,
                    Third,
                    Fourth,
                    KernelOp<R, Op>,
                >,
                <Self as crate::detail::MItemStorage<
                    R,
                >>::Storage: crate::detail::MaterializeOutput<
                    Runtime = R,
                    Output = ($(
                        crate::detail::DeviceVec<R, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage = crate::detail::apply::TransformPayloadApply::soa4::<Self, R, First, Second, Third, Fourth, KernelOp<R, Op>>(policy, first, second, third, fourth, env)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            fn transform_quinary<First, Second, Third, Fourth, Fifth, Op>(
                policy: &crate::detail::CubePolicy<R>,
                first: crate::detail::device::DeviceColumnView<R, First>,
                second: crate::detail::device::DeviceColumnView<R, Second>,
                third: crate::detail::device::DeviceColumnView<R, Third>,
                fourth: crate::detail::device::DeviceColumnView<R, Fourth>,
                fifth: crate::detail::device::DeviceColumnView<R, Fifth>,
                op: Op,
                env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<<Self as MAlloc<R>>::Inner, Error>
            where
                First: MStorageElement,
                Second: MStorageElement,
                Third: MStorageElement,
                Fourth: MStorageElement,
                Fifth: MStorageElement,
                Op: op::UnaryOp<R, (First, Second, Third, Fourth, Fifth), Output = Self>,
                Self: crate::detail::TransformSoA5Output<
                    R,
                    First,
                    Second,
                    Third,
                    Fourth,
                    Fifth,
                    KernelOp<R, Op>,
                >,
                <Self as crate::detail::MItemStorage<
                    R,
                >>::Storage: crate::detail::MaterializeOutput<
                    Runtime = R,
                    Output = ($(
                        crate::detail::DeviceVec<R, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage = crate::detail::apply::TransformPayloadApply::soa5::<Self, R, First, Second, Third, Fourth, Fifth, KernelOp<R, Op>>(policy, first, second, third, fourth, fifth, env)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            #[allow(clippy::too_many_arguments)]
            fn transform_senary<First, Second, Third, Fourth, Fifth, Sixth, Op>(
                policy: &crate::detail::CubePolicy<R>,
                first: crate::detail::device::DeviceColumnView<R, First>,
                second: crate::detail::device::DeviceColumnView<R, Second>,
                third: crate::detail::device::DeviceColumnView<R, Third>,
                fourth: crate::detail::device::DeviceColumnView<R, Fourth>,
                fifth: crate::detail::device::DeviceColumnView<R, Fifth>,
                sixth: crate::detail::device::DeviceColumnView<R, Sixth>,
                op: Op,
                env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<<Self as MAlloc<R>>::Inner, Error>
            where
                First: MStorageElement,
                Second: MStorageElement,
                Third: MStorageElement,
                Fourth: MStorageElement,
                Fifth: MStorageElement,
                Sixth: MStorageElement,
                Op: op::UnaryOp<R, (First, Second, Third, Fourth, Fifth, Sixth), Output = Self>,
                Self: crate::detail::TransformSoA6Output<
                    R,
                    First,
                    Second,
                    Third,
                    Fourth,
                    Fifth,
                    Sixth,
                    KernelOp<R, Op>,
                >,
                <Self as crate::detail::MItemStorage<
                    R,
                >>::Storage: crate::detail::MaterializeOutput<
                    Runtime = R,
                    Output = ($(
                        crate::detail::DeviceVec<R, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage = crate::detail::apply::TransformPayloadApply::soa6::<Self, R, First, Second, Third, Fourth, Fifth, Sixth, KernelOp<R, Op>>(policy, first, second, third, fourth, fifth, sixth, env)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            #[allow(clippy::too_many_arguments)]
            fn transform_septenary<First, Second, Third, Fourth, Fifth, Sixth, Seventh, Op>(
                policy: &crate::detail::CubePolicy<R>,
                first: crate::detail::device::DeviceColumnView<R, First>,
                second: crate::detail::device::DeviceColumnView<R, Second>,
                third: crate::detail::device::DeviceColumnView<R, Third>,
                fourth: crate::detail::device::DeviceColumnView<R, Fourth>,
                fifth: crate::detail::device::DeviceColumnView<R, Fifth>,
                sixth: crate::detail::device::DeviceColumnView<R, Sixth>,
                seventh: crate::detail::device::DeviceColumnView<R, Seventh>,
                op: Op,
                env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<<Self as MAlloc<R>>::Inner, Error>
            where
                First: MStorageElement,
                Second: MStorageElement,
                Third: MStorageElement,
                Fourth: MStorageElement,
                Fifth: MStorageElement,
                Sixth: MStorageElement,
                Seventh: MStorageElement,
                Op: op::UnaryOp<
                    R,
                    (First, Second, Third, Fourth, Fifth, Sixth, Seventh),
                    Output = Self,
                >,
                Self: crate::detail::TransformSoA7Output<
                    R,
                    First,
                    Second,
                    Third,
                    Fourth,
                    Fifth,
                    Sixth,
                    Seventh,
                    KernelOp<R, Op>,
                >,
                <Self as crate::detail::MItemStorage<
                    R,
                >>::Storage: crate::detail::MaterializeOutput<
                    Runtime = R,
                    Output = ($(
                        crate::detail::DeviceVec<R, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage = crate::detail::apply::TransformPayloadApply::soa7::<Self, R, First, Second, Third, Fourth, Fifth, Sixth, Seventh, KernelOp<R, Op>>(policy, first, second, third, fourth, fifth, sixth, seventh, env)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }
        }
    };
}

impl_wide_mitem_tuple!(A: a, B0: b, C: c, D: d);
impl_wide_mitem_tuple!(A: a, B0: b, C: c, D: d, E: e);
impl_wide_mitem_tuple!(A: a, B0: b, C: c, D: d, E: e, F: f);
impl_wide_mitem_tuple!(A: a, B0: b, C: c, D: d, E: e, F: f, G: g);
