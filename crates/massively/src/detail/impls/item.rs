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

macro_rules! alloc_inner {
    ($exec:expr, $len:expr; $( $ty:ty ),+) => {{
        let len = $len;
        u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
        let policy = $exec.policy();
        if len == 0 {
            Ok(($( policy.empty_device_vec::<$ty>(), )+))
        } else {
            let client = policy.client();
            Ok(($(
                crate::detail::DeviceVec::from_handle(
                    policy.id(),
                    client.empty(len * std::mem::size_of::<$ty>()),
                    len,
                ),
            )+))
        }
    }};
}

macro_rules! impl_mitem_tuple {
    ($( $ty:ident : $var:ident ),+) => {
        impl<R, $( $ty ),+> MItem<R> for ($( $ty, )+)
        where
            R: Runtime,
            $( $ty: Scalar, )+
        {
            type Inner = ($( crate::detail::DeviceVec<R, $ty>, )+);
            type View = ($( crate::detail::device::DeviceColumnView<R, $ty>, )+);
            type Vec = soa_type!($( DeviceVec<R, $ty> ),+);

            fn vec_from_inner(inner: Self::Inner) -> Self::Vec {
                let ($( $var, )+) = inner;
                soa_value!($( DeviceVec::from_inner($var) ),+)
            }

            fn alloc_vec(exec: &Executor<R>, len: usize) -> Result<Self::Vec, Error> {
                Ok(Self::vec_from_inner(alloc_inner!(exec, len; $( $ty ),+)?))
            }
        }

        impl<R, $( $ty ),+> MVec<R> for soa_type!($( DeviceVec<R, $ty> ),+)
        where
            R: Runtime,
            $( $ty: Scalar + 'static, )+
            for<'a> soa_type!($( DeviceSlice<'a, R, $ty> ),+): MIter<R, Item = ($( $ty, )+)>,
            for<'a> soa_type!($( DeviceSliceMut<'a, R, $ty> ),+): MIterMut<R, Item = ($( $ty, )+)>,
        {
            type Item = ($( $ty, )+);
            type Slice<'a>
                = soa_type!($( DeviceSlice<'a, R, $ty> ),+)
            where
                Self: 'a;
            type SliceMut<'a>
                = soa_type!($( DeviceSliceMut<'a, R, $ty> ),+)
            where
                Self: 'a;

            fn from_inner(inner: <Self::Item as MItem<R>>::Inner) -> Self {
                <Self::Item as MItem<R>>::vec_from_inner(inner)
            }

            fn len(&self) -> usize {
                self.0.len()
            }

            fn slice<Bounds>(&self, range: Bounds) -> Self::Slice<'_>
            where
                Bounds: std::ops::RangeBounds<usize>,
            {
                <soa_type!($( DeviceVec<R, $ty> ),+)>::slice(self, range)
            }

            fn slice_mut<Bounds>(&self, range: Bounds) -> Self::SliceMut<'_>
            where
                Bounds: std::ops::RangeBounds<usize>,
            {
                <soa_type!($( DeviceVec<R, $ty> ),+)>::slice_mut(self, range)
            }
        }

        impl<R, $( $ty ),+> sealed::MItemDispatch<R> for ($( $ty, )+)
        where
            R: Runtime,
            $( $ty: Scalar, )+
        {
            fn transform_unary<Input, Op>(
                policy: &crate::detail::CubePolicy<R>,
                input: crate::detail::device::DeviceColumnView<
                    R,
                    Input,
                >,
                op: Op,
            ) -> Result<<Self as MItem<R>>::Inner, Error>
            where
                Input: Scalar,
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
                let storage =
                    <Self as crate::detail::TransformUnaryOutput<
                        R,
                        Input,
                        KernelOp<R, Op>,
                    >>::run(policy, input)?;
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
            ) -> Result<<Self as MItem<R>>::Inner, Error>
            where
                Left: Scalar,
                Right: Scalar,
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
                let storage =
                    <Self as crate::detail::TransformSoA2Output<
                        R,
                        Left,
                        Right,
                        KernelOp<R, Op>,
                    >>::run(policy, left, right)?;
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
            ) -> Result<<Self as MItem<R>>::Inner, Error>
            where
                First: Scalar,
                Second: Scalar,
                Third: Scalar,
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
                let storage =
                    <Self as crate::detail::TransformSoA3Output<
                        R,
                        First,
                        Second,
                        Third,
                        KernelOp<R, Op>,
                    >>::run(
                        policy,
                        first,
                        second,
                        third,
                )?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            fn transform_quaternary<First, Second, Third, Fourth, Op>(
                policy: &crate::detail::CubePolicy<R>,
                first: crate::detail::device::DeviceColumnView<R, First>,
                second: crate::detail::device::DeviceColumnView<R, Second>,
                third: crate::detail::device::DeviceColumnView<R, Third>,
                fourth: crate::detail::device::DeviceColumnView<R, Fourth>,
                op: Op,
            ) -> Result<<Self as MItem<R>>::Inner, Error>
            where
                First: Scalar,
                Second: Scalar,
                Third: Scalar,
                Fourth: Scalar,
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
                let storage =
                    <Self as crate::detail::TransformSoA4Output<
                        R,
                        First,
                        Second,
                        Third,
                        Fourth,
                        KernelOp<R, Op>,
                    >>::run(policy, first, second, third, fourth)?;
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
            ) -> Result<<Self as MItem<R>>::Inner, Error>
            where
                First: Scalar,
                Second: Scalar,
                Third: Scalar,
                Fourth: Scalar,
                Fifth: Scalar,
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
                let storage =
                    <Self as crate::detail::TransformSoA5Output<
                        R,
                        First,
                        Second,
                        Third,
                        Fourth,
                        Fifth,
                        KernelOp<R, Op>,
                    >>::run(policy, first, second, third, fourth, fifth)?;
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
            ) -> Result<<Self as MItem<R>>::Inner, Error>
            where
                First: Scalar,
                Second: Scalar,
                Third: Scalar,
                Fourth: Scalar,
                Fifth: Scalar,
                Sixth: Scalar,
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
                let storage =
                    <Self as crate::detail::TransformSoA6Output<
                        R,
                        First,
                        Second,
                        Third,
                        Fourth,
                        Fifth,
                        Sixth,
                        KernelOp<R, Op>,
                    >>::run(policy, first, second, third, fourth, fifth, sixth)?;
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
            ) -> Result<<Self as MItem<R>>::Inner, Error>
            where
                First: Scalar,
                Second: Scalar,
                Third: Scalar,
                Fourth: Scalar,
                Fifth: Scalar,
                Sixth: Scalar,
                Seventh: Scalar,
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
                let storage =
                    <Self as crate::detail::TransformSoA7Output<
                        R,
                        First,
                        Second,
                        Third,
                        Fourth,
                        Fifth,
                        Sixth,
                        Seventh,
                        KernelOp<R, Op>,
                    >>::run(
                        policy,
                        first,
                        second,
                        third,
                        fourth,
                        fifth,
                        sixth,
                        seventh,
                )?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            fn reduce_inner<Op>(
                policy: &crate::detail::CubePolicy<R>,
                input: <Self as MItem<R>>::Inner,
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
            $( $ty: Scalar, )+
        {
            type Inner = ($( crate::detail::DeviceVec<R, $ty>, )+);
            type View = ($( crate::detail::device::DeviceColumnView<R, $ty>, )+);
            type Vec = soa_type!($( DeviceVec<R, $ty> ),+);

            fn vec_from_inner(inner: Self::Inner) -> Self::Vec {
                let ($( $var, )+) = inner;
                soa_value!($( DeviceVec::from_inner($var) ),+)
            }

            fn alloc_vec(exec: &Executor<R>, len: usize) -> Result<Self::Vec, Error> {
                Ok(Self::vec_from_inner(alloc_inner!(exec, len; $( $ty ),+)?))
            }
        }

        impl<R, $( $ty ),+> MVec<R> for soa_type!($( DeviceVec<R, $ty> ),+)
        where
            R: Runtime,
            $( $ty: Scalar + 'static, )+
            for<'a> soa_type!($( DeviceSlice<'a, R, $ty> ),+): MIter<R, Item = ($( $ty, )+)>,
            for<'a> soa_type!($( DeviceSliceMut<'a, R, $ty> ),+): MIterMut<R, Item = ($( $ty, )+)>,
        {
            type Item = ($( $ty, )+);
            type Slice<'a>
                = soa_type!($( DeviceSlice<'a, R, $ty> ),+)
            where
                Self: 'a;
            type SliceMut<'a>
                = soa_type!($( DeviceSliceMut<'a, R, $ty> ),+)
            where
                Self: 'a;

            fn from_inner(inner: <Self::Item as MItem<R>>::Inner) -> Self {
                <Self::Item as MItem<R>>::vec_from_inner(inner)
            }

            fn len(&self) -> usize {
                self.0.len()
            }

            fn slice<Bounds>(&self, range: Bounds) -> Self::Slice<'_>
            where
                Bounds: std::ops::RangeBounds<usize>,
            {
                <soa_type!($( DeviceVec<R, $ty> ),+)>::slice(self, range)
            }

            fn slice_mut<Bounds>(&self, range: Bounds) -> Self::SliceMut<'_>
            where
                Bounds: std::ops::RangeBounds<usize>,
            {
                <soa_type!($( DeviceVec<R, $ty> ),+)>::slice_mut(self, range)
            }
        }

        impl<R, $( $ty ),+> sealed::MItemDispatch<R> for ($( $ty, )+)
        where
            R: Runtime,
            $( $ty: Scalar, )+
        {
            fn transform_unary<Input, Op>(
                policy: &crate::detail::CubePolicy<R>,
                input: crate::detail::device::DeviceColumnView<
                    R,
                    Input,
                >,
                op: Op,
            ) -> Result<<Self as MItem<R>>::Inner, Error>
            where
                Input: Scalar,
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
                let storage =
                    <Self as crate::detail::TransformUnaryOutput<
                        R,
                        Input,
                        KernelOp<R, Op>,
                    >>::run(policy, input)?;
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
            ) -> Result<<Self as MItem<R>>::Inner, Error>
            where
                First: Scalar,
                Second: Scalar,
                Third: Scalar,
                Fourth: Scalar,
                Fifth: Scalar,
                Sixth: Scalar,
                Seventh: Scalar,
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
                let storage =
                    <Self as crate::detail::TransformSoA7Output<
                        R,
                        First,
                        Second,
                        Third,
                        Fourth,
                        Fifth,
                        Sixth,
                        Seventh,
                        KernelOp<R, Op>,
                    >>::run(
                        policy,
                        first,
                        second,
                        third,
                        fourth,
                        fifth,
                        sixth,
                        seventh,
                )?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }
        }
    };
}

impl_wide_mitem_tuple!(A: a, B0: b, C: c, D: d);
impl_wide_mitem_tuple!(A: a, B0: b, C: c, D: d, E: e);
impl_wide_mitem_tuple!(A: a, B0: b, C: c, D: d, E: e, F: f);
impl_wide_mitem_tuple!(A: a, B0: b, C: c, D: d, E: e, F: f, G: g);
