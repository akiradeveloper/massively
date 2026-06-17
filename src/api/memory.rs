use crate::{
    device::{
        DeviceVec, KernelColumn, KernelColumnAt, S0, SoA, SoA1, SoA2, SoA3, SoA4, SoA5, SoA6, SoA7,
        SoA8, SoA9, SoA10, SoA11, SoA12, SoVA, SoVA1, SoVA2, SoVA3, SoVA4, SoVA5, SoVA6, SoVA7,
        SoVA8, SoVA9, SoVA10, SoVA11, SoVA12, StorageKernelColumn,
    },
    error::Error,
    expr::DeviceGpuExpr,
    kernels::*,
    op::{GpuOp, UnaryOp},
};
use cubecl::prelude::*;

/// Borrowed input accepted by `zip`.
#[doc(hidden)]
pub trait BorrowedZipInput {
    /// Borrowed SoA returned for this tuple shape.
    type Output;

    /// Builds the borrowed SoA.
    fn borrowed_zip(self) -> Self::Output;
}

pub(crate) trait BorrowedZipSource: KernelColumn {}

impl<R, T> BorrowedZipSource for &DeviceVec<R, T>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
}

impl<Left, Right> BorrowedZipInput for (Left, Right)
where
    Left: BorrowedZipSource + KernelColumnAt<S0>,
    Right: BorrowedZipSource
        + KernelColumn<Runtime = <Left as KernelColumn>::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<Left as KernelColumnAt<S0>>::Next>,
    <Left as KernelColumn>::Item: CubePrimitive + CubeElement,
    <Right as KernelColumn>::Item: CubePrimitive + CubeElement,
{
    type Output = SoA2<Left, Right>;

    fn borrowed_zip(self) -> Self::Output {
        SoA2 {
            left: self.0,
            right: self.1,
        }
    }
}

impl<Left, Right> BorrowedZipInput for (SoVA1<Left>, SoVA1<Right>)
where
    Left: BorrowedZipSource + KernelColumnAt<S0>,
    Right: BorrowedZipSource
        + KernelColumn<Runtime = <Left as KernelColumn>::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<Left as KernelColumnAt<S0>>::Next>,
    <Left as KernelColumn>::Item: CubePrimitive + CubeElement,
    <Right as KernelColumn>::Item: CubePrimitive + CubeElement,
{
    type Output = SoA2<Left, Right>;

    fn borrowed_zip(self) -> Self::Output {
        SoA2 {
            left: self.0.source,
            right: self.1.source,
        }
    }
}

impl<First, Second, Third> BorrowedZipInput for (First, Second, Third)
where
    First: BorrowedZipSource + KernelColumnAt<S0>,
    Second: BorrowedZipSource
        + KernelColumn<Runtime = <First as KernelColumn>::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<First as KernelColumnAt<S0>>::Next>,
    Third: BorrowedZipSource
        + KernelColumn<Runtime = <First as KernelColumn>::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<Second as KernelColumnAt<<First as KernelColumnAt<S0>>::Next>>::Next>,
    <First as KernelColumn>::Item: CubePrimitive + CubeElement,
    <Second as KernelColumn>::Item: CubePrimitive + CubeElement,
    <Third as KernelColumn>::Item: CubePrimitive + CubeElement,
{
    type Output = SoA3<First, Second, Third>;

    fn borrowed_zip(self) -> Self::Output {
        SoA3 {
            first: self.0,
            second: self.1,
            third: self.2,
        }
    }
}

impl<First, Second, Third> BorrowedZipInput for (SoVA1<First>, SoVA1<Second>, SoVA1<Third>)
where
    First: BorrowedZipSource + KernelColumnAt<S0>,
    Second: BorrowedZipSource
        + KernelColumn<Runtime = <First as KernelColumn>::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<First as KernelColumnAt<S0>>::Next>,
    Third: BorrowedZipSource
        + KernelColumn<Runtime = <First as KernelColumn>::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<Second as KernelColumnAt<<First as KernelColumnAt<S0>>::Next>>::Next>,
    <First as KernelColumn>::Item: CubePrimitive + CubeElement,
    <Second as KernelColumn>::Item: CubePrimitive + CubeElement,
    <Third as KernelColumn>::Item: CubePrimitive + CubeElement,
{
    type Output = SoA3<First, Second, Third>;

    fn borrowed_zip(self) -> Self::Output {
        SoA3 {
            first: self.0.source,
            second: self.1.source,
            third: self.2.source,
        }
    }
}

macro_rules! impl_borrowed_zip_input {
    ($name:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ }) => {
        impl<$first, $( $rest ),+> BorrowedZipInput for ($first, $( $rest ),+)
        where
            $first: BorrowedZipSource + KernelColumnAt<S0>,
            $(
                $rest: BorrowedZipSource
                    + KernelColumn<Runtime = <$first as KernelColumn>::Runtime>
                    + KernelColumnAt<S0>,
            )+
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
            )+
        {
            type Output = $name<$first, $( $rest ),+>;

            fn borrowed_zip(self) -> Self::Output {
                let ($first_field, $( $field ),+) = self;
                $name { $first_field, $( $field ),+ }
            }
        }
    };
}

impl_borrowed_zip_input!(SoA4<A, B, C, D> { a, b, c, d });
impl_borrowed_zip_input!(SoA5<A, B, C, D, E> { a, b, c, d, e });
impl_borrowed_zip_input!(SoA6<A, B, C, D, E, F> { a, b, c, d, e, f });
impl_borrowed_zip_input!(SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g });
impl_borrowed_zip_input!(SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h });
impl_borrowed_zip_input!(SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i });
impl_borrowed_zip_input!(SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j });
impl_borrowed_zip_input!(SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k });
impl_borrowed_zip_input!(SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l });

macro_rules! impl_borrowed_zip_soa1_input {
    ($name:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ }) => {
        impl<$first, $( $rest ),+> BorrowedZipInput for (SoVA1<$first>, $( SoVA1<$rest> ),+)
        where
            $first: BorrowedZipSource + KernelColumnAt<S0>,
            $(
                $rest: BorrowedZipSource
                    + KernelColumn<Runtime = <$first as KernelColumn>::Runtime>
                    + KernelColumnAt<S0>,
            )+
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            $(
                <$rest as KernelColumn>::Item: CubePrimitive + CubeElement,
            )+
        {
            type Output = $name<$first, $( $rest ),+>;

            fn borrowed_zip(self) -> Self::Output {
                let ($first_field, $( $field ),+) = self;
                $name {
                    $first_field: $first_field.source,
                    $( $field: $field.source, )+
                }
            }
        }
    };
}

impl_borrowed_zip_soa1_input!(SoA4<A, B, C, D> { a, b, c, d });
impl_borrowed_zip_soa1_input!(SoA5<A, B, C, D, E> { a, b, c, d, e });
impl_borrowed_zip_soa1_input!(SoA6<A, B, C, D, E, F> { a, b, c, d, e, f });
impl_borrowed_zip_soa1_input!(SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g });
impl_borrowed_zip_soa1_input!(SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h });
impl_borrowed_zip_soa1_input!(SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i });
impl_borrowed_zip_soa1_input!(SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j });
impl_borrowed_zip_soa1_input!(SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k });
impl_borrowed_zip_soa1_input!(SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l });

/// Combines two borrowed columns into a wider read-only SoA input.
///
/// `zip` is for borrowing algorithms such as [`transform`], [`reduce`](crate::reduce), and
/// [`gather`](crate::gather). It does not allocate device storage; it creates a
/// typed read-only view over existing device columns.
///
/// ```no_run
/// use massively::{CubeWgpu, transform, zip};
///
/// # struct Add;
/// # #[cubecl::cube]
/// # impl massively::op::UnaryOp<(f32, f32)> for Add {
/// #     type Output = f32;
/// #     fn apply(input: (f32, f32)) -> f32 { input.0 + input.1 }
/// # }
/// # fn main() -> Result<(), massively::Error> {
/// let policy = CubeWgpu::new();
/// let x = policy.to_device(&[1.0_f32, 2.0])?;
/// let y = policy.to_device(&[10.0_f32, 20.0])?;
///
/// let output = transform(zip(&x, &y), Add)?;
/// assert_eq!(output.to_vec()?, vec![11.0, 22.0]);
/// # Ok(())
/// # }
/// ```
pub fn zip<Left, Right>(left: Left, right: Right) -> <(Left, Right) as BorrowedZipInput>::Output
where
    (Left, Right): BorrowedZipInput,
{
    (left, right).borrowed_zip()
}

/// Convenience wrapper over binary [`zip`] for three borrowed columns.
pub fn zip3<A, B, C>(a: A, b: B, c: C) -> <(A, B, C) as BorrowedZipInput>::Output
where
    (A, B, C): BorrowedZipInput,
{
    (a, b, c).borrowed_zip()
}

macro_rules! define_borrowed_zip_n {
    ($func:ident < $( $ty:ident : $var:ident ),+ >) => {
        /// Convenience wrapper over binary [`zip`] for borrowed columns.
        pub fn $func<$( $ty ),+>($( $var: $ty ),+) -> <($( $ty ),+) as BorrowedZipInput>::Output
        where
            ($( $ty ),+): BorrowedZipInput,
        {
            ($( $var ),+).borrowed_zip()
        }
    };
}

define_borrowed_zip_n!(zip4<A: a, B: b, C: c, D: d>);
define_borrowed_zip_n!(zip5<A: a, B: b, C: c, D: d, E: e>);
define_borrowed_zip_n!(zip6<A: a, B: b, C: c, D: d, E: e, F: f>);
define_borrowed_zip_n!(zip7<A: a, B: b, C: c, D: d, E: e, F: f, G: g>);
define_borrowed_zip_n!(zip8<A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h>);
define_borrowed_zip_n!(zip9<A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i>);
define_borrowed_zip_n!(zip10<A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i, J: j>);
define_borrowed_zip_n!(zip11<A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i, J: j, K: k>);
define_borrowed_zip_n!(zip12<A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i, J: j, K: k, L: l>);

/// Storage shape used for a transformed device value.
#[doc(hidden)]
pub trait StorageOutput<R: Runtime>: CubeType {
    type Storage;
}

macro_rules! impl_scalar_storage_output {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl<R> StorageOutput<R> for $ty
            where
                R: Runtime,
            {
                type Storage = SoA1<DeviceVec<R, $ty>>;
            }
        )+
    };
}

impl_scalar_storage_output!(f32, f64, u8, u16, u32, u64, i8, i16, i32, i64, bool);

impl<R, A, B> StorageOutput<R> for (A, B)
where
    R: Runtime,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
{
    type Storage = SoA2<DeviceVec<R, A>, DeviceVec<R, B>>;
}

impl<R, A, B, C> StorageOutput<R> for (A, B, C)
where
    R: Runtime,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    C: CubePrimitive + CubeElement,
{
    type Storage = SoA3<DeviceVec<R, A>, DeviceVec<R, B>, DeviceVec<R, C>>;
}

macro_rules! impl_tuple_storage_output {
    ($($soa:ident < $( $ty:ident ),+ >),+ $(,)?) => {
        $(
            impl<R, $( $ty ),+> StorageOutput<R> for ($( $ty, )+)
            where
                R: Runtime,
                $( $ty: CubePrimitive + CubeElement, )+
            {
                type Storage = $soa<$( DeviceVec<R, $ty> ),+>;
            }
        )+
    };
}

impl_tuple_storage_output!(
    SoA4<A, B, C, D>,
    SoA5<A, B, C, D, E>,
    SoA6<A, B, C, D, E, F>,
    SoA7<A, B, C, D, E, F, G>,
    SoA8<A, B, C, D, E, F, G, H>,
    SoA9<A, B, C, D, E, F, G, H, I>,
    SoA10<A, B, C, D, E, F, G, H, I, J>,
    SoA11<A, B, C, D, E, F, G, H, I, J, K>,
    SoA12<A, B, C, D, E, F, G, H, I, J, K, L>,
);

trait TransformUnaryOutput<R, T, Op>: StorageOutput<R>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    Op: UnaryOp<T, Output = Self>,
{
    fn run(input: &DeviceVec<R, T>) -> Result<<Self as StorageOutput<R>>::Storage, Error>;
}

macro_rules! impl_scalar_transform_unary_output {
    ($($out:ty),+ $(,)?) => {
        $(
            impl<R, T, Op> TransformUnaryOutput<R, T, Op> for $out
            where
                R: Runtime,
                T: CubePrimitive + CubeElement,
                Op: UnaryOp<T, Output = $out>,
            {
                fn run(input: &DeviceVec<R, T>) -> Result<<Self as StorageOutput<R>>::Storage, Error> {
                    let len = input.len();
                    let client = input.policy().client();
                    let output_handle = client.empty(len * std::mem::size_of::<$out>());
                    if len != 0 {
                        let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
                        let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
                        let block_size = 256_u32;
                        let block_count = len.div_ceil(block_size as usize);
                        let block_count_u32 = u32::try_from(block_count)
                            .map_err(|_| Error::LengthTooLarge { len: block_count })?;
                        unsafe {
                            transform_unary_kernel::launch_unchecked::<T, $out, Op, R>(
                                client,
                                CubeCount::Static(block_count_u32, 1, 1),
                                CubeDim::new_1d(block_size),
                                ArrayArg::from_raw_parts::<T>(&input.handle, len, 1),
                                ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
                                ArrayArg::from_raw_parts::<$out>(&output_handle, len, 1),
                            )
                            .map_err(|err| Error::Launch {
                                message: format!("{err:?}"),
                            })?;
                        }
                    }
                    Ok(SoA1 {
                        source: DeviceVec::from_handle(input.policy().clone(), output_handle, len),
                    })
                }
            }
        )+
    };
}

impl_scalar_transform_unary_output!(f32, f64, u8, u16, u32, u64, i8, i16, i32, i64, bool);

impl<R, T, A, B, Op> TransformUnaryOutput<R, T, Op> for (A, B)
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    Op: UnaryOp<T, Output = (A, B)>,
{
    fn run(input: &DeviceVec<R, T>) -> Result<<Self as StorageOutput<R>>::Storage, Error> {
        let len = input.len();
        let client = input.policy().client();
        let output_a = client.empty(len * std::mem::size_of::<A>());
        let output_b = client.empty(len * std::mem::size_of::<B>());
        if len != 0 {
            let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
            let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
            let block_size = 256_u32;
            let block_count = len.div_ceil(block_size as usize);
            let block_count_u32 = u32::try_from(block_count)
                .map_err(|_| Error::LengthTooLarge { len: block_count })?;
            unsafe {
                transform_unary_tuple2_kernel::launch_unchecked::<T, A, B, Op, R>(
                    client,
                    CubeCount::Static(block_count_u32, 1, 1),
                    CubeDim::new_1d(block_size),
                    ArrayArg::from_raw_parts::<T>(&input.handle, len, 1),
                    ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
                    ArrayArg::from_raw_parts::<A>(&output_a, len, 1),
                    ArrayArg::from_raw_parts::<B>(&output_b, len, 1),
                )
                .map_err(|err| Error::Launch {
                    message: format!("{err:?}"),
                })?;
            }
        }
        Ok(SoA2 {
            left: DeviceVec::from_handle(input.policy().clone(), output_a, len),
            right: DeviceVec::from_handle(input.policy().clone(), output_b, len),
        })
    }
}

impl<R, T, A, B, C, Op> TransformUnaryOutput<R, T, Op> for (A, B, C)
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    C: CubePrimitive + CubeElement,
    Op: UnaryOp<T, Output = (A, B, C)>,
{
    fn run(input: &DeviceVec<R, T>) -> Result<<Self as StorageOutput<R>>::Storage, Error> {
        let len = input.len();
        let client = input.policy().client();
        let output_a = client.empty(len * std::mem::size_of::<A>());
        let output_b = client.empty(len * std::mem::size_of::<B>());
        let output_c = client.empty(len * std::mem::size_of::<C>());
        if len != 0 {
            let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
            let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
            let block_size = 256_u32;
            let block_count = len.div_ceil(block_size as usize);
            let block_count_u32 = u32::try_from(block_count)
                .map_err(|_| Error::LengthTooLarge { len: block_count })?;
            unsafe {
                transform_unary_tuple3_kernel::launch_unchecked::<T, A, B, C, Op, R>(
                    client,
                    CubeCount::Static(block_count_u32, 1, 1),
                    CubeDim::new_1d(block_size),
                    ArrayArg::from_raw_parts::<T>(&input.handle, len, 1),
                    ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
                    ArrayArg::from_raw_parts::<A>(&output_a, len, 1),
                    ArrayArg::from_raw_parts::<B>(&output_b, len, 1),
                    ArrayArg::from_raw_parts::<C>(&output_c, len, 1),
                )
                .map_err(|err| Error::Launch {
                    message: format!("{err:?}"),
                })?;
            }
        }
        Ok(SoA3 {
            first: DeviceVec::from_handle(input.policy().clone(), output_a, len),
            second: DeviceVec::from_handle(input.policy().clone(), output_b, len),
            third: DeviceVec::from_handle(input.policy().clone(), output_c, len),
        })
    }
}

macro_rules! impl_transform_unary_tuple_output {
    (
        $kernel:ident,
        $soa:ident,
        ($( $out_ty:ident : $out_handle:ident : $field:ident ),+)
    ) => {
        impl<R, T, $( $out_ty, )+ Op> TransformUnaryOutput<R, T, Op> for ($( $out_ty, )+)
        where
            R: Runtime,
            T: CubePrimitive + CubeElement,
            $( $out_ty: CubePrimitive + CubeElement, )+
            Op: UnaryOp<T, Output = ($( $out_ty, )+)>,
        {
            fn run(input: &DeviceVec<R, T>) -> Result<<Self as StorageOutput<R>>::Storage, Error> {
                let len = input.len();
                let client = input.policy().client();
                $(
                    let $out_handle = client.empty(len * std::mem::size_of::<$out_ty>());
                )+
                if len != 0 {
                    let len_u32 = u32::try_from(len)
                        .map_err(|_| Error::LengthTooLarge { len })?;
                    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
                    let block_size = 256_u32;
                    let block_count = len.div_ceil(block_size as usize);
                    let block_count_u32 = u32::try_from(block_count)
                        .map_err(|_| Error::LengthTooLarge { len: block_count })?;
                    unsafe {
                        $kernel::launch_unchecked::<T, $( $out_ty, )+ Op, R>(
                            client,
                            CubeCount::Static(block_count_u32, 1, 1),
                            CubeDim::new_1d(block_size),
                            ArrayArg::from_raw_parts::<T>(&input.handle, len, 1),
                            ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
                            $(
                                ArrayArg::from_raw_parts::<$out_ty>(&$out_handle, len, 1),
                            )+
                        )
                        .map_err(|err| Error::Launch {
                            message: format!("{err:?}"),
                        })?;
                    }
                }
                Ok($soa {
                    $(
                        $field: DeviceVec::from_handle(input.policy().clone(), $out_handle, len),
                    )+
                })
            }
        }
    };
}

impl_transform_unary_tuple_output!(
    transform_unary_tuple4_kernel,
    SoA4,
    (A: output_a: a, B: output_b: b, C: output_c: c, D: output_d: d)
);
impl_transform_unary_tuple_output!(
    transform_unary_tuple5_kernel,
    SoA5,
    (A: output_a: a, B: output_b: b, C: output_c: c, D: output_d: d, E: output_e: e)
);
impl_transform_unary_tuple_output!(
    transform_unary_tuple6_kernel,
    SoA6,
    (A: output_a: a, B: output_b: b, C: output_c: c, D: output_d: d, E: output_e: e, F: output_f: f)
);
impl_transform_unary_tuple_output!(
    transform_unary_tuple7_kernel,
    SoA7,
    (A: output_a: a, B: output_b: b, C: output_c: c, D: output_d: d, E: output_e: e, F: output_f: f, G: output_g: g)
);
impl_transform_unary_tuple_output!(
    transform_unary_tuple8_kernel,
    SoA8,
    (A: output_a: a, B: output_b: b, C: output_c: c, D: output_d: d, E: output_e: e, F: output_f: f, G: output_g: g, H: output_h: h)
);
impl_transform_unary_tuple_output!(
    transform_unary_tuple9_kernel,
    SoA9,
    (A: output_a: a, B: output_b: b, C: output_c: c, D: output_d: d, E: output_e: e, F: output_f: f, G: output_g: g, H: output_h: h, I: output_i: i)
);
impl_transform_unary_tuple_output!(
    transform_unary_tuple10_kernel,
    SoA10,
    (A: output_a: a, B: output_b: b, C: output_c: c, D: output_d: d, E: output_e: e, F: output_f: f, G: output_g: g, H: output_h: h, I: output_i: i, J: output_j: j)
);
impl_transform_unary_tuple_output!(
    transform_unary_tuple11_kernel,
    SoA11,
    (A: output_a: a, B: output_b: b, C: output_c: c, D: output_d: d, E: output_e: e, F: output_f: f, G: output_g: g, H: output_h: h, I: output_i: i, J: output_j: j, K: output_k: k)
);
impl_transform_unary_tuple_output!(
    transform_unary_tuple12_kernel,
    SoA12,
    (A: output_a: a, B: output_b: b, C: output_c: c, D: output_d: d, E: output_e: e, F: output_f: f, G: output_g: g, H: output_h: h, I: output_i: i, J: output_j: j, K: output_k: k, L: output_l: l)
);

/// Input accepted by returning [`transform`].
#[doc(hidden)]
pub trait TransformInput<Op> {
    type Output;

    fn transform_input(self, op: GpuOp<Op>) -> Result<Self::Output, Error>;
}

impl<Source, Op> TransformInput<Op> for SoVA1<Source>
where
    Self: SoVA<Item = Source::Item, Scalar = Source::Item>,
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Runtime: Runtime,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Op: UnaryOp<Source::Item>,
    Op::Output: TransformUnaryOutput<Source::Runtime, Source::Item, Op>,
{
    type Output = <Op::Output as StorageOutput<Source::Runtime>>::Storage;

    fn transform_input(self, _op: GpuOp<Op>) -> Result<Self::Output, Error> {
        SoVA::validate(&self)?;
        let input = super::device_expr_collect(&self.source)?;
        <Op::Output as TransformUnaryOutput<Source::Runtime, Source::Item, Op>>::run(&input)
    }
}

impl<Source, Op> TransformInput<Op> for Source
where
    Source: KernelColumn + KernelColumnAt<S0>,
    Source::Runtime: Runtime,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
    Op: UnaryOp<Source::Item>,
    Op::Output: TransformUnaryOutput<Source::Runtime, Source::Item, Op>,
{
    type Output = <Op::Output as StorageOutput<Source::Runtime>>::Storage;

    fn transform_input(self, op: GpuOp<Op>) -> Result<Self::Output, Error> {
        <SoVA1<Source> as TransformInput<Op>>::transform_input(SoVA1 { source: self }, op)
    }
}

macro_rules! impl_transform_tuple_output {
    (
        ($trait_name:ident < $first_in:ident : $first_arg:ident, $( $in_ty:ident : $arg:ident ),+ >),
        $kernel:ident,
        $soa:ident,
        ($( $out_ty:ident : $out_handle:ident : $out_field:ident ),+)
    ) => {
        impl<R, $first_in, $( $in_ty, )+ $( $out_ty, )+ Op>
            $trait_name<R, $first_in, $( $in_ty, )+ Op> for ($( $out_ty, )+)
        where
            R: Runtime,
            $first_in: CubePrimitive + CubeElement,
            $( $in_ty: CubePrimitive + CubeElement, )+
            $( $out_ty: CubePrimitive + CubeElement, )+
            Op: UnaryOp<($first_in, $( $in_ty, )+), Output = ($( $out_ty, )+)>,
        {
            fn run(
                policy: &crate::policy::CubePolicy<R>,
                $first_arg: &DeviceVec<R, $first_in>,
                $( $arg: &DeviceVec<R, $in_ty>, )+
            ) -> Result<<Self as StorageOutput<R>>::Storage, Error> {
                let len = $first_arg.len();
                let client = policy.client();
                $(
                    let $out_handle = client.empty(len * std::mem::size_of::<$out_ty>());
                )+
                if len != 0 {
                    let len_u32 = u32::try_from(len)
                        .map_err(|_| Error::LengthTooLarge { len })?;
                    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
                    let block_size = 256_u32;
                    let block_count = len.div_ceil(block_size as usize);
                    let block_count_u32 = u32::try_from(block_count)
                        .map_err(|_| Error::LengthTooLarge { len: block_count })?;
                    unsafe {
                        $kernel::launch_unchecked::<
                            $first_in, $( $in_ty, )+ $( $out_ty, )+ Op, R,
                        >(
                            client,
                            CubeCount::Static(block_count_u32, 1, 1),
                            CubeDim::new_1d(block_size),
                            ArrayArg::from_raw_parts::<$first_in>(&$first_arg.handle, len, 1),
                            $(
                                ArrayArg::from_raw_parts::<$in_ty>(&$arg.handle, len, 1),
                            )+
                            ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
                            $(
                                ArrayArg::from_raw_parts::<$out_ty>(&$out_handle, len, 1),
                            )+
                        )
                        .map_err(|err| Error::Launch {
                            message: format!("{err:?}"),
                        })?;
                    }
                }
                Ok($soa {
                    $(
                        $out_field: DeviceVec::from_handle(policy.clone(), $out_handle, len),
                    )+
                })
            }
        }
    };
}

macro_rules! impl_transform_tuple_output_arity {
    ($input:tt, 2, $kernel:ident) => {
        impl_transform_tuple_output!(
            $input,
            $kernel,
            SoA2,
            (OutA: out_a: left, OutB: out_b: right)
        );
    };
    ($input:tt, 3, $kernel:ident) => {
        impl_transform_tuple_output!(
            $input,
            $kernel,
            SoA3,
            (OutA: out_a: first, OutB: out_b: second, OutC: out_c: third)
        );
    };
    ($input:tt, 4, $kernel:ident) => {
        impl_transform_tuple_output!(
            $input,
            $kernel,
            SoA4,
            (OutA: out_a: a, OutB: out_b: b, OutC: out_c: c, OutD: out_d: d)
        );
    };
    ($input:tt, 5, $kernel:ident) => {
        impl_transform_tuple_output!(
            $input,
            $kernel,
            SoA5,
            (OutA: out_a: a, OutB: out_b: b, OutC: out_c: c, OutD: out_d: d, OutE: out_e: e)
        );
    };
    ($input:tt, 6, $kernel:ident) => {
        impl_transform_tuple_output!(
            $input,
            $kernel,
            SoA6,
            (OutA: out_a: a, OutB: out_b: b, OutC: out_c: c, OutD: out_d: d, OutE: out_e: e, OutF: out_f: f)
        );
    };
    ($input:tt, 7, $kernel:ident) => {
        impl_transform_tuple_output!(
            $input,
            $kernel,
            SoA7,
            (OutA: out_a: a, OutB: out_b: b, OutC: out_c: c, OutD: out_d: d, OutE: out_e: e, OutF: out_f: f, OutG: out_g: g)
        );
    };
    ($input:tt, 8, $kernel:ident) => {
        impl_transform_tuple_output!(
            $input,
            $kernel,
            SoA8,
            (OutA: out_a: a, OutB: out_b: b, OutC: out_c: c, OutD: out_d: d, OutE: out_e: e, OutF: out_f: f, OutG: out_g: g, OutH: out_h: h)
        );
    };
    ($input:tt, 9, $kernel:ident) => {
        impl_transform_tuple_output!(
            $input,
            $kernel,
            SoA9,
            (OutA: out_a: a, OutB: out_b: b, OutC: out_c: c, OutD: out_d: d, OutE: out_e: e, OutF: out_f: f, OutG: out_g: g, OutH: out_h: h, OutI: out_i: i)
        );
    };
    ($input:tt, 10, $kernel:ident) => {
        impl_transform_tuple_output!(
            $input,
            $kernel,
            SoA10,
            (OutA: out_a: a, OutB: out_b: b, OutC: out_c: c, OutD: out_d: d, OutE: out_e: e, OutF: out_f: f, OutG: out_g: g, OutH: out_h: h, OutI: out_i: i, OutJ: out_j: j)
        );
    };
    ($input:tt, 11, $kernel:ident) => {
        impl_transform_tuple_output!(
            $input,
            $kernel,
            SoA11,
            (OutA: out_a: a, OutB: out_b: b, OutC: out_c: c, OutD: out_d: d, OutE: out_e: e, OutF: out_f: f, OutG: out_g: g, OutH: out_h: h, OutI: out_i: i, OutJ: out_j: j, OutK: out_k: k)
        );
    };
    ($input:tt, 12, $kernel:ident) => {
        impl_transform_tuple_output!(
            $input,
            $kernel,
            SoA12,
            (OutA: out_a: a, OutB: out_b: b, OutC: out_c: c, OutD: out_d: d, OutE: out_e: e, OutF: out_f: f, OutG: out_g: g, OutH: out_h: h, OutI: out_i: i, OutJ: out_j: j, OutK: out_k: k, OutL: out_l: l)
        );
    };
}

macro_rules! impl_transform_tuple_outputs {
    (
        $trait_name:ident < $first_in:ident : $first_arg:ident, $( $in_ty:ident : $arg:ident ),+ >,
        $( $arity:tt => $kernel:ident ),+ $(,)?
    ) => {
        impl_transform_tuple_outputs!(
            @inner
            ($trait_name < $first_in : $first_arg, $( $in_ty : $arg ),+ >),
            $( $arity => $kernel ),+
        );
    };
    (
        @inner
        $input:tt,
        $( $arity:tt => $kernel:ident ),+ $(,)?
    ) => {
        $(
            impl_transform_tuple_output_arity!(
                $input,
                $arity,
                $kernel
            );
        )+
    };
}

#[doc(hidden)]
pub trait TransformSoA2Output<R, InA, InB, Op>: CubeType + StorageOutput<R>
where
    R: Runtime,
    InA: CubePrimitive + CubeElement,
    InB: CubePrimitive + CubeElement,
    Op: UnaryOp<(InA, InB), Output = Self>,
{
    fn run(
        policy: &crate::policy::CubePolicy<R>,
        left: &DeviceVec<R, InA>,
        right: &DeviceVec<R, InB>,
    ) -> Result<<Self as StorageOutput<R>>::Storage, Error>;
}

macro_rules! impl_scalar_transform_soa2_output {
    ($($out:ty),+ $(,)?) => {
        $(
            impl<R, InA, InB, Op> TransformSoA2Output<R, InA, InB, Op> for $out
            where
                R: Runtime,
                InA: CubePrimitive + CubeElement,
                InB: CubePrimitive + CubeElement,
                Op: UnaryOp<(InA, InB), Output = $out>,
            {
                fn run(
                    policy: &crate::policy::CubePolicy<R>,
                    left: &DeviceVec<R, InA>,
                    right: &DeviceVec<R, InB>,
                ) -> Result<<Self as StorageOutput<R>>::Storage, Error> {
                    let len = left.len();
                    let client = policy.client();
                    let output_handle = client.empty(len * std::mem::size_of::<$out>());
                    if len != 0 {
                        let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
                        let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
                        let block_size = 256_u32;
                        let block_count = len.div_ceil(block_size as usize);
                        let block_count_u32 = u32::try_from(block_count)
                            .map_err(|_| Error::LengthTooLarge { len: block_count })?;
                        unsafe {
                            transform_tuple2_kernel::launch_unchecked::<InA, InB, $out, Op, R>(
                                client,
                                CubeCount::Static(block_count_u32, 1, 1),
                                CubeDim::new_1d(block_size),
                                ArrayArg::from_raw_parts::<InA>(&left.handle, len, 1),
                                ArrayArg::from_raw_parts::<InB>(&right.handle, len, 1),
                                ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
                                ArrayArg::from_raw_parts::<$out>(&output_handle, len, 1),
                            )
                            .map_err(|err| Error::Launch {
                                message: format!("{err:?}"),
                            })?;
                        }
                    }
                    Ok(SoA1 {
                        source: DeviceVec::from_handle(policy.clone(), output_handle, len),
                    })
                }
            }
        )+
    };
}

impl_scalar_transform_soa2_output!(f32, f64, u8, u16, u32, u64, i8, i16, i32, i64, bool);

impl<R, InA, InB, OutA, OutB, Op> TransformSoA2Output<R, InA, InB, Op> for (OutA, OutB)
where
    R: Runtime,
    InA: CubePrimitive + CubeElement,
    InB: CubePrimitive + CubeElement,
    OutA: CubePrimitive + CubeElement,
    OutB: CubePrimitive + CubeElement,
    Op: UnaryOp<(InA, InB), Output = (OutA, OutB)>,
{
    fn run(
        policy: &crate::policy::CubePolicy<R>,
        left: &DeviceVec<R, InA>,
        right: &DeviceVec<R, InB>,
    ) -> Result<<Self as StorageOutput<R>>::Storage, Error> {
        let len = left.len();
        let client = policy.client();
        let output_a = client.empty(len * std::mem::size_of::<OutA>());
        let output_b = client.empty(len * std::mem::size_of::<OutB>());
        if len != 0 {
            let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
            let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
            let block_size = 256_u32;
            let block_count = len.div_ceil(block_size as usize);
            let block_count_u32 = u32::try_from(block_count)
                .map_err(|_| Error::LengthTooLarge { len: block_count })?;
            unsafe {
                transform_tuple2_to_tuple2_kernel::launch_unchecked::<InA, InB, OutA, OutB, Op, R>(
                    client,
                    CubeCount::Static(block_count_u32, 1, 1),
                    CubeDim::new_1d(block_size),
                    ArrayArg::from_raw_parts::<InA>(&left.handle, len, 1),
                    ArrayArg::from_raw_parts::<InB>(&right.handle, len, 1),
                    ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
                    ArrayArg::from_raw_parts::<OutA>(&output_a, len, 1),
                    ArrayArg::from_raw_parts::<OutB>(&output_b, len, 1),
                )
                .map_err(|err| Error::Launch {
                    message: format!("{err:?}"),
                })?;
            }
        }
        Ok(SoA2 {
            left: DeviceVec::from_handle(policy.clone(), output_a, len),
            right: DeviceVec::from_handle(policy.clone(), output_b, len),
        })
    }
}

impl<Left, Right, Op> TransformInput<Op> for SoVA2<Left, Right>
where
    Self: SoVA<Runtime = Left::Runtime, Item = (Left::Item, Right::Item), Scalar = Left::Item>,
    Left: KernelColumn + KernelColumnAt<S0>,
    Right: KernelColumn<Runtime = Left::Runtime> + KernelColumnAt<S0>,
    Left::Runtime: Runtime,
    Left::Item: CubePrimitive + CubeElement,
    Right::Item: CubePrimitive + CubeElement,
    Left::Expr: DeviceGpuExpr<Left::Item>,
    Right::Expr: DeviceGpuExpr<Right::Item>,
    Op: UnaryOp<(Left::Item, Right::Item)>,
    Op::Output: TransformSoA2Output<Left::Runtime, Left::Item, Right::Item, Op>,
{
    type Output = <Op::Output as StorageOutput<Left::Runtime>>::Storage;

    fn transform_input(self, _op: GpuOp<Op>) -> Result<Self::Output, Error> {
        SoVA::validate(&self)?;
        let policy = self.policy().clone();
        let left = super::device_expr_collect(&self.left)?;
        let right = super::device_expr_collect(&self.right)?;
        <Op::Output as TransformSoA2Output<Left::Runtime, Left::Item, Right::Item, Op>>::run(
            &policy, &left, &right,
        )
    }
}

impl<Left, Right, Op> TransformInput<Op> for SoA2<Left, Right>
where
    Self: SoVA<Runtime = Left::Runtime, Item = (Left::Item, Right::Item), Scalar = Left::Item>,
    Left: KernelColumn + KernelColumnAt<S0>,
    Right: KernelColumn<Runtime = Left::Runtime> + KernelColumnAt<S0>,
    Left::Runtime: Runtime,
    Left::Item: CubePrimitive + CubeElement,
    Right::Item: CubePrimitive + CubeElement,
    Left::Expr: DeviceGpuExpr<Left::Item>,
    Right::Expr: DeviceGpuExpr<Right::Item>,
    Op: UnaryOp<(Left::Item, Right::Item)>,
    Op::Output: TransformSoA2Output<Left::Runtime, Left::Item, Right::Item, Op>,
{
    type Output = <Op::Output as StorageOutput<Left::Runtime>>::Storage;

    fn transform_input(self, _op: GpuOp<Op>) -> Result<Self::Output, Error> {
        SoVA::validate(&self)?;
        let policy = self.policy().clone();
        let left = super::device_expr_collect(&self.left)?;
        let right = super::device_expr_collect(&self.right)?;
        <Op::Output as TransformSoA2Output<Left::Runtime, Left::Item, Right::Item, Op>>::run(
            &policy, &left, &right,
        )
    }
}

#[doc(hidden)]
pub trait TransformSoA3Output<R, InA, InB, InC, Op>: CubeType + StorageOutput<R>
where
    R: Runtime,
    InA: CubePrimitive + CubeElement,
    InB: CubePrimitive + CubeElement,
    InC: CubePrimitive + CubeElement,
    Op: UnaryOp<(InA, InB, InC), Output = Self>,
{
    fn run(
        policy: &crate::policy::CubePolicy<R>,
        first: &DeviceVec<R, InA>,
        second: &DeviceVec<R, InB>,
        third: &DeviceVec<R, InC>,
    ) -> Result<<Self as StorageOutput<R>>::Storage, Error>;
}

macro_rules! impl_scalar_transform_soa3_output {
    ($($out:ty),+ $(,)?) => {
        $(
            impl<R, InA, InB, InC, Op> TransformSoA3Output<R, InA, InB, InC, Op> for $out
            where
                R: Runtime,
                InA: CubePrimitive + CubeElement,
                InB: CubePrimitive + CubeElement,
                InC: CubePrimitive + CubeElement,
                Op: UnaryOp<(InA, InB, InC), Output = $out>,
            {
                fn run(
                    policy: &crate::policy::CubePolicy<R>,
                    first: &DeviceVec<R, InA>,
                    second: &DeviceVec<R, InB>,
                    third: &DeviceVec<R, InC>,
                ) -> Result<<Self as StorageOutput<R>>::Storage, Error> {
                    let len = first.len();
                    let client = policy.client();
                    let output_handle = client.empty(len * std::mem::size_of::<$out>());
                    if len != 0 {
                        let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
                        let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
                        let block_size = 256_u32;
                        let block_count = len.div_ceil(block_size as usize);
                        let block_count_u32 = u32::try_from(block_count)
                            .map_err(|_| Error::LengthTooLarge { len: block_count })?;
                        unsafe {
                            transform_tuple3_kernel::launch_unchecked::<
                                InA,
                                InB,
                                InC,
                                $out,
                                Op,
                                R,
                            >(
                                client,
                                CubeCount::Static(block_count_u32, 1, 1),
                                CubeDim::new_1d(block_size),
                                ArrayArg::from_raw_parts::<InA>(&first.handle, len, 1),
                                ArrayArg::from_raw_parts::<InB>(&second.handle, len, 1),
                                ArrayArg::from_raw_parts::<InC>(&third.handle, len, 1),
                                ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
                                ArrayArg::from_raw_parts::<$out>(&output_handle, len, 1),
                            )
                            .map_err(|err| Error::Launch {
                                message: format!("{err:?}"),
                            })?;
                        }
                    }
                    Ok(SoA1 {
                        source: DeviceVec::from_handle(policy.clone(), output_handle, len),
                    })
                }
            }
        )+
    };
}

impl_scalar_transform_soa3_output!(f32, f64, u8, u16, u32, u64, i8, i16, i32, i64, bool);

impl<R, InA, InB, InC, OutA, OutB, OutC, Op> TransformSoA3Output<R, InA, InB, InC, Op>
    for (OutA, OutB, OutC)
where
    R: Runtime,
    InA: CubePrimitive + CubeElement,
    InB: CubePrimitive + CubeElement,
    InC: CubePrimitive + CubeElement,
    OutA: CubePrimitive + CubeElement,
    OutB: CubePrimitive + CubeElement,
    OutC: CubePrimitive + CubeElement,
    Op: UnaryOp<(InA, InB, InC), Output = (OutA, OutB, OutC)>,
{
    fn run(
        policy: &crate::policy::CubePolicy<R>,
        first: &DeviceVec<R, InA>,
        second: &DeviceVec<R, InB>,
        third: &DeviceVec<R, InC>,
    ) -> Result<<Self as StorageOutput<R>>::Storage, Error> {
        let len = first.len();
        let client = policy.client();
        let output_a = client.empty(len * std::mem::size_of::<OutA>());
        let output_b = client.empty(len * std::mem::size_of::<OutB>());
        let output_c = client.empty(len * std::mem::size_of::<OutC>());
        if len != 0 {
            let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
            let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
            let block_size = 256_u32;
            let block_count = len.div_ceil(block_size as usize);
            let block_count_u32 = u32::try_from(block_count)
                .map_err(|_| Error::LengthTooLarge { len: block_count })?;
            unsafe {
                transform_tuple3_to_tuple3_kernel::launch_unchecked::<
                    InA,
                    InB,
                    InC,
                    OutA,
                    OutB,
                    OutC,
                    Op,
                    R,
                >(
                    client,
                    CubeCount::Static(block_count_u32, 1, 1),
                    CubeDim::new_1d(block_size),
                    ArrayArg::from_raw_parts::<InA>(&first.handle, len, 1),
                    ArrayArg::from_raw_parts::<InB>(&second.handle, len, 1),
                    ArrayArg::from_raw_parts::<InC>(&third.handle, len, 1),
                    ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
                    ArrayArg::from_raw_parts::<OutA>(&output_a, len, 1),
                    ArrayArg::from_raw_parts::<OutB>(&output_b, len, 1),
                    ArrayArg::from_raw_parts::<OutC>(&output_c, len, 1),
                )
                .map_err(|err| Error::Launch {
                    message: format!("{err:?}"),
                })?;
            }
        }
        Ok(SoA3 {
            first: DeviceVec::from_handle(policy.clone(), output_a, len),
            second: DeviceVec::from_handle(policy.clone(), output_b, len),
            third: DeviceVec::from_handle(policy.clone(), output_c, len),
        })
    }
}

impl<First, Second, Third, Op> TransformInput<Op> for SoVA3<First, Second, Third>
where
    Self: SoVA<
            Runtime = First::Runtime,
            Item = (First::Item, Second::Item, Third::Item),
            Scalar = First::Item,
        >,
    First: KernelColumn + KernelColumnAt<S0>,
    Second: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
    Third: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
    First::Runtime: Runtime,
    First::Item: CubePrimitive + CubeElement,
    Second::Item: CubePrimitive + CubeElement,
    Third::Item: CubePrimitive + CubeElement,
    First::Expr: DeviceGpuExpr<First::Item>,
    Second::Expr: DeviceGpuExpr<Second::Item>,
    Third::Expr: DeviceGpuExpr<Third::Item>,
    Op: UnaryOp<(First::Item, Second::Item, Third::Item)>,
    Op::Output: TransformSoA3Output<First::Runtime, First::Item, Second::Item, Third::Item, Op>,
{
    type Output = <Op::Output as StorageOutput<First::Runtime>>::Storage;

    fn transform_input(self, _op: GpuOp<Op>) -> Result<Self::Output, Error> {
        SoVA::validate(&self)?;
        let policy = self.policy().clone();
        let first = super::device_expr_collect(&self.first)?;
        let second = super::device_expr_collect(&self.second)?;
        let third = super::device_expr_collect(&self.third)?;
        <Op::Output as TransformSoA3Output<
            First::Runtime,
            First::Item,
            Second::Item,
            Third::Item,
            Op,
        >>::run(&policy, &first, &second, &third)
    }
}

impl<First, Second, Third, Op> TransformInput<Op> for SoA3<First, Second, Third>
where
    Self: SoVA<
            Runtime = First::Runtime,
            Item = (First::Item, Second::Item, Third::Item),
            Scalar = First::Item,
        >,
    First: KernelColumn + KernelColumnAt<S0>,
    Second: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
    Third: KernelColumn<Runtime = First::Runtime> + KernelColumnAt<S0>,
    First::Runtime: Runtime,
    First::Item: CubePrimitive + CubeElement,
    Second::Item: CubePrimitive + CubeElement,
    Third::Item: CubePrimitive + CubeElement,
    First::Expr: DeviceGpuExpr<First::Item>,
    Second::Expr: DeviceGpuExpr<Second::Item>,
    Third::Expr: DeviceGpuExpr<Third::Item>,
    Op: UnaryOp<(First::Item, Second::Item, Third::Item)>,
    Op::Output: TransformSoA3Output<First::Runtime, First::Item, Second::Item, Third::Item, Op>,
{
    type Output = <Op::Output as StorageOutput<First::Runtime>>::Storage;

    fn transform_input(self, _op: GpuOp<Op>) -> Result<Self::Output, Error> {
        SoVA::validate(&self)?;
        let policy = self.policy().clone();
        let first = super::device_expr_collect(&self.first)?;
        let second = super::device_expr_collect(&self.second)?;
        let third = super::device_expr_collect(&self.third)?;
        <Op::Output as TransformSoA3Output<
            First::Runtime,
            First::Item,
            Second::Item,
            Third::Item,
            Op,
        >>::run(&policy, &first, &second, &third)
    }
}

#[doc(hidden)]
pub trait TransformSoA4Output<R, InA, InB, InC, InD, Op>: CubeType + StorageOutput<R>
where
    R: Runtime,
    InA: CubePrimitive + CubeElement,
    InB: CubePrimitive + CubeElement,
    InC: CubePrimitive + CubeElement,
    InD: CubePrimitive + CubeElement,
    Op: UnaryOp<(InA, InB, InC, InD), Output = Self>,
{
    fn run(
        policy: &crate::policy::CubePolicy<R>,
        a: &DeviceVec<R, InA>,
        b: &DeviceVec<R, InB>,
        c: &DeviceVec<R, InC>,
        d: &DeviceVec<R, InD>,
    ) -> Result<<Self as StorageOutput<R>>::Storage, Error>;
}

macro_rules! impl_scalar_transform_soa4_output {
    ($($out:ty),+ $(,)?) => {
        $(
            impl<R, InA, InB, InC, InD, Op> TransformSoA4Output<R, InA, InB, InC, InD, Op> for $out
            where
                R: Runtime,
                InA: CubePrimitive + CubeElement,
                InB: CubePrimitive + CubeElement,
                InC: CubePrimitive + CubeElement,
                InD: CubePrimitive + CubeElement,
                Op: UnaryOp<(InA, InB, InC, InD), Output = $out>,
            {
                fn run(
                    policy: &crate::policy::CubePolicy<R>,
                    a: &DeviceVec<R, InA>,
                    b: &DeviceVec<R, InB>,
                    c: &DeviceVec<R, InC>,
                    d: &DeviceVec<R, InD>,
                ) -> Result<<Self as StorageOutput<R>>::Storage, Error> {
                    let len = a.len();
                    let client = policy.client();
                    let output_handle = client.empty(len * std::mem::size_of::<$out>());
                    if len != 0 {
                        let len_u32 = u32::try_from(len)
                            .map_err(|_| Error::LengthTooLarge { len })?;
                        let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
                        let block_size = 256_u32;
                        let block_count = len.div_ceil(block_size as usize);
                        let block_count_u32 = u32::try_from(block_count)
                            .map_err(|_| Error::LengthTooLarge { len: block_count })?;
                        unsafe {
                            transform_tuple4_kernel::launch_unchecked::<
                                InA, InB, InC, InD, $out, Op, R,
                            >(
                                client,
                                CubeCount::Static(block_count_u32, 1, 1),
                                CubeDim::new_1d(block_size),
                                ArrayArg::from_raw_parts::<InA>(&a.handle, len, 1),
                                ArrayArg::from_raw_parts::<InB>(&b.handle, len, 1),
                                ArrayArg::from_raw_parts::<InC>(&c.handle, len, 1),
                                ArrayArg::from_raw_parts::<InD>(&d.handle, len, 1),
                                ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
                                ArrayArg::from_raw_parts::<$out>(&output_handle, len, 1),
                            )
                            .map_err(|err| Error::Launch {
                                message: format!("{err:?}"),
                            })?;
                        }
                    }
                    Ok(SoA1 {
                        source: DeviceVec::from_handle(policy.clone(), output_handle, len),
                    })
                }
            }
        )+
    };
}

impl_scalar_transform_soa4_output!(f32, f64, u8, u16, u32, u64, i8, i16, i32, i64, bool);

impl<R, InA, InB, InC, InD, OutA, OutB, OutC, OutD, Op>
    TransformSoA4Output<R, InA, InB, InC, InD, Op> for (OutA, OutB, OutC, OutD)
where
    R: Runtime,
    InA: CubePrimitive + CubeElement,
    InB: CubePrimitive + CubeElement,
    InC: CubePrimitive + CubeElement,
    InD: CubePrimitive + CubeElement,
    OutA: CubePrimitive + CubeElement,
    OutB: CubePrimitive + CubeElement,
    OutC: CubePrimitive + CubeElement,
    OutD: CubePrimitive + CubeElement,
    Op: UnaryOp<(InA, InB, InC, InD), Output = (OutA, OutB, OutC, OutD)>,
{
    fn run(
        policy: &crate::policy::CubePolicy<R>,
        a: &DeviceVec<R, InA>,
        b: &DeviceVec<R, InB>,
        c: &DeviceVec<R, InC>,
        d: &DeviceVec<R, InD>,
    ) -> Result<<Self as StorageOutput<R>>::Storage, Error> {
        let len = a.len();
        let client = policy.client();
        let output_a = client.empty(len * std::mem::size_of::<OutA>());
        let output_b = client.empty(len * std::mem::size_of::<OutB>());
        let output_c = client.empty(len * std::mem::size_of::<OutC>());
        let output_d = client.empty(len * std::mem::size_of::<OutD>());
        if len != 0 {
            let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
            let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
            let block_size = 256_u32;
            let block_count = len.div_ceil(block_size as usize);
            let block_count_u32 = u32::try_from(block_count)
                .map_err(|_| Error::LengthTooLarge { len: block_count })?;
            unsafe {
                transform_tuple4_to_tuple4_kernel::launch_unchecked::<
                    InA,
                    InB,
                    InC,
                    InD,
                    OutA,
                    OutB,
                    OutC,
                    OutD,
                    Op,
                    R,
                >(
                    client,
                    CubeCount::Static(block_count_u32, 1, 1),
                    CubeDim::new_1d(block_size),
                    ArrayArg::from_raw_parts::<InA>(&a.handle, len, 1),
                    ArrayArg::from_raw_parts::<InB>(&b.handle, len, 1),
                    ArrayArg::from_raw_parts::<InC>(&c.handle, len, 1),
                    ArrayArg::from_raw_parts::<InD>(&d.handle, len, 1),
                    ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
                    ArrayArg::from_raw_parts::<OutA>(&output_a, len, 1),
                    ArrayArg::from_raw_parts::<OutB>(&output_b, len, 1),
                    ArrayArg::from_raw_parts::<OutC>(&output_c, len, 1),
                    ArrayArg::from_raw_parts::<OutD>(&output_d, len, 1),
                )
                .map_err(|err| Error::Launch {
                    message: format!("{err:?}"),
                })?;
            }
        }
        Ok(SoA4 {
            a: DeviceVec::from_handle(policy.clone(), output_a, len),
            b: DeviceVec::from_handle(policy.clone(), output_b, len),
            c: DeviceVec::from_handle(policy.clone(), output_c, len),
            d: DeviceVec::from_handle(policy.clone(), output_d, len),
        })
    }
}

impl<A, B, C, D, Op> TransformInput<Op> for SoVA4<A, B, C, D>
where
    Self: SoVA<Runtime = A::Runtime, Item = (A::Item, B::Item, C::Item, D::Item), Scalar = A::Item>,
    A: KernelColumn + KernelColumnAt<S0>,
    B: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    D: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    A::Runtime: Runtime,
    A::Item: CubePrimitive + CubeElement,
    B::Item: CubePrimitive + CubeElement,
    C::Item: CubePrimitive + CubeElement,
    D::Item: CubePrimitive + CubeElement,
    A::Expr: DeviceGpuExpr<A::Item>,
    B::Expr: DeviceGpuExpr<B::Item>,
    C::Expr: DeviceGpuExpr<C::Item>,
    D::Expr: DeviceGpuExpr<D::Item>,
    Op: UnaryOp<(A::Item, B::Item, C::Item, D::Item)>,
    Op::Output: TransformSoA4Output<A::Runtime, A::Item, B::Item, C::Item, D::Item, Op>,
{
    type Output = <Op::Output as StorageOutput<A::Runtime>>::Storage;

    fn transform_input(self, _op: GpuOp<Op>) -> Result<Self::Output, Error> {
        SoVA::validate(&self)?;
        let policy = self.policy().clone();
        let a = super::device_expr_collect(&self.a)?;
        let b = super::device_expr_collect(&self.b)?;
        let c = super::device_expr_collect(&self.c)?;
        let d = super::device_expr_collect(&self.d)?;
        <Op::Output as TransformSoA4Output<A::Runtime, A::Item, B::Item, C::Item, D::Item, Op>>::run(
            &policy, &a, &b, &c, &d,
        )
    }
}

impl<A, B, C, D, Op> TransformInput<Op> for SoA4<A, B, C, D>
where
    Self: SoVA<Runtime = A::Runtime, Item = (A::Item, B::Item, C::Item, D::Item), Scalar = A::Item>,
    A: KernelColumn + KernelColumnAt<S0>,
    B: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    D: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    A::Runtime: Runtime,
    A::Item: CubePrimitive + CubeElement,
    B::Item: CubePrimitive + CubeElement,
    C::Item: CubePrimitive + CubeElement,
    D::Item: CubePrimitive + CubeElement,
    A::Expr: DeviceGpuExpr<A::Item>,
    B::Expr: DeviceGpuExpr<B::Item>,
    C::Expr: DeviceGpuExpr<C::Item>,
    D::Expr: DeviceGpuExpr<D::Item>,
    Op: UnaryOp<(A::Item, B::Item, C::Item, D::Item)>,
    Op::Output: TransformSoA4Output<A::Runtime, A::Item, B::Item, C::Item, D::Item, Op>,
{
    type Output = <Op::Output as StorageOutput<A::Runtime>>::Storage;

    fn transform_input(self, _op: GpuOp<Op>) -> Result<Self::Output, Error> {
        SoVA::validate(&self)?;
        let policy = self.policy().clone();
        let a = super::device_expr_collect(&self.a)?;
        let b = super::device_expr_collect(&self.b)?;
        let c = super::device_expr_collect(&self.c)?;
        let d = super::device_expr_collect(&self.d)?;
        <Op::Output as TransformSoA4Output<A::Runtime, A::Item, B::Item, C::Item, D::Item, Op>>::run(
            &policy, &a, &b, &c, &d,
        )
    }
}

macro_rules! define_transform_soa_output {
    (
        $trait_name:ident,
        $sova_name:ident < $first:ident : $first_field:ident, $( $col:ident : $field:ident ),+ >,
        $soa_name:ident { $first_soa_field:ident, $( $soa_field:ident ),+ },
        $scalar_kernel:ident,
        $tuple_kernel:ident,
        ($first_out:ident : $first_out_handle:ident, $( $out:ident : $out_handle:ident ),+)
    ) => {
        #[doc(hidden)]
        pub trait $trait_name<R, $first, $( $col ),+, Op>: CubeType + StorageOutput<R>
        where
            R: Runtime,
            $first: CubePrimitive + CubeElement,
            $( $col: CubePrimitive + CubeElement, )+
            Op: UnaryOp<($first, $( $col, )+), Output = Self>,
        {
            fn run(
                policy: &crate::policy::CubePolicy<R>,
                $first_field: &DeviceVec<R, $first>,
                $( $field: &DeviceVec<R, $col>, )+
            ) -> Result<<Self as StorageOutput<R>>::Storage, Error>;
        }

        macro_rules! impl_scalar_output {
            ($scalar:ty) => {
                impl<R, $first, $( $col ),+, Op> $trait_name<R, $first, $( $col ),+, Op>
                    for $scalar
                where
                    R: Runtime,
                    $first: CubePrimitive + CubeElement,
                    $( $col: CubePrimitive + CubeElement, )+
                    Op: UnaryOp<($first, $( $col, )+), Output = $scalar>,
                {
                    fn run(
                        policy: &crate::policy::CubePolicy<R>,
                        $first_field: &DeviceVec<R, $first>,
                        $( $field: &DeviceVec<R, $col>, )+
                    ) -> Result<<Self as StorageOutput<R>>::Storage, Error> {
                        let len = $first_field.len();
                        let client = policy.client();
                        let output_handle = client.empty(len * std::mem::size_of::<$scalar>());
                        if len != 0 {
                            let len_u32 = u32::try_from(len)
                                .map_err(|_| Error::LengthTooLarge { len })?;
                            let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
                            let block_size = 256_u32;
                            let block_count = len.div_ceil(block_size as usize);
                            let block_count_u32 = u32::try_from(block_count)
                                .map_err(|_| Error::LengthTooLarge { len: block_count })?;
                            unsafe {
                                $scalar_kernel::launch_unchecked::<
                                    $first, $( $col, )+ $scalar, Op, R,
                                >(
                                    client,
                                    CubeCount::Static(block_count_u32, 1, 1),
                                    CubeDim::new_1d(block_size),
                                    ArrayArg::from_raw_parts::<$first>(
                                        &$first_field.handle,
                                        len,
                                        1,
                                    ),
                                    $(
                                        ArrayArg::from_raw_parts::<$col>(
                                            &$field.handle,
                                            len,
                                            1,
                                        ),
                                    )+
                                    ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
                                    ArrayArg::from_raw_parts::<$scalar>(&output_handle, len, 1),
                                )
                                .map_err(|err| Error::Launch {
                                    message: format!("{err:?}"),
                                })?;
                            }
                        }
                        Ok(SoA1 {
                            source: DeviceVec::from_handle(policy.clone(), output_handle, len),
                        })
                    }
                }
            };
        }

        impl_scalar_output!(f32);
        impl_scalar_output!(f64);
        impl_scalar_output!(u8);
        impl_scalar_output!(u16);
        impl_scalar_output!(u32);
        impl_scalar_output!(u64);
        impl_scalar_output!(i8);
        impl_scalar_output!(i16);
        impl_scalar_output!(i32);
        impl_scalar_output!(i64);
        impl_scalar_output!(bool);

        impl<R, $first, $( $col, )+ $first_out, $( $out, )+ Op>
            $trait_name<R, $first, $( $col ),+, Op>
            for ($first_out, $( $out, )+)
        where
            R: Runtime,
            $first: CubePrimitive + CubeElement,
            $( $col: CubePrimitive + CubeElement, )+
            $first_out: CubePrimitive + CubeElement,
            $( $out: CubePrimitive + CubeElement, )+
            Op: UnaryOp<($first, $( $col, )+), Output = ($first_out, $( $out, )+)>,
        {
            fn run(
                policy: &crate::policy::CubePolicy<R>,
                $first_field: &DeviceVec<R, $first>,
                $( $field: &DeviceVec<R, $col>, )+
            ) -> Result<<Self as StorageOutput<R>>::Storage, Error> {
                let len = $first_field.len();
                let client = policy.client();
                let $first_out_handle = client.empty(len * std::mem::size_of::<$first_out>());
                $(
                    let $out_handle = client.empty(len * std::mem::size_of::<$out>());
                )+
                if len != 0 {
                    let len_u32 = u32::try_from(len)
                        .map_err(|_| Error::LengthTooLarge { len })?;
                    let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
                    let block_size = 256_u32;
                    let block_count = len.div_ceil(block_size as usize);
                    let block_count_u32 = u32::try_from(block_count)
                        .map_err(|_| Error::LengthTooLarge { len: block_count })?;
                    unsafe {
                        $tuple_kernel::launch_unchecked::<
                            $first, $( $col, )+ $first_out, $( $out, )+ Op, R,
                        >(
                            client,
                            CubeCount::Static(block_count_u32, 1, 1),
                            CubeDim::new_1d(block_size),
                            ArrayArg::from_raw_parts::<$first>(
                                &$first_field.handle,
                                len,
                                1,
                            ),
                            $(
                                ArrayArg::from_raw_parts::<$col>(
                                    &$field.handle,
                                    len,
                                    1,
                                ),
                            )+
                            ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
                            ArrayArg::from_raw_parts::<$first_out>(
                                &$first_out_handle,
                                len,
                                1,
                            ),
                            $(
                                ArrayArg::from_raw_parts::<$out>(
                                    &$out_handle,
                                    len,
                                    1,
                                ),
                            )+
                        )
                        .map_err(|err| Error::Launch {
                            message: format!("{err:?}"),
                        })?;
                    }
                }
                Ok($soa_name {
                    $first_soa_field: DeviceVec::from_handle(
                        policy.clone(),
                        $first_out_handle,
                        len,
                    ),
                    $(
                        $soa_field: DeviceVec::from_handle(policy.clone(), $out_handle, len),
                    )+
                })
            }
        }

        impl<$first, $( $col ),+, Op> TransformInput<Op>
            for $sova_name<$first, $( $col ),+>
        where
            Self: SoVA<
                Runtime = <$first as KernelColumn>::Runtime,
                Item = (
                    <$first as KernelColumn>::Item,
                    $( <$col as KernelColumn>::Item, )+
                ),
                Scalar = <$first as KernelColumn>::Item,
            >,
            $first: KernelColumn + KernelColumnAt<S0>,
            $( $col: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>, )+
            <$first as KernelColumn>::Runtime: Runtime,
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            $( <$col as KernelColumn>::Item: CubePrimitive + CubeElement, )+
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $( <$col as KernelColumn>::Expr: DeviceGpuExpr<<$col as KernelColumn>::Item>, )+
            Op: UnaryOp<(
                <$first as KernelColumn>::Item,
                $( <$col as KernelColumn>::Item, )+
            )>,
            Op::Output: $trait_name<
                <$first as KernelColumn>::Runtime,
                <$first as KernelColumn>::Item,
                $( <$col as KernelColumn>::Item, )+
                Op,
            >,
        {
            type Output =
                <Op::Output as StorageOutput<<$first as KernelColumn>::Runtime>>::Storage;

            fn transform_input(self, _op: GpuOp<Op>) -> Result<Self::Output, Error> {
                SoVA::validate(&self)?;
                let policy = self.policy().clone();
                let $first_field = super::device_expr_collect(&self.$first_field)?;
                $(
                    let $field = super::device_expr_collect(&self.$field)?;
                )+
                <Op::Output as $trait_name<
                    <$first as KernelColumn>::Runtime,
                    <$first as KernelColumn>::Item,
                    $( <$col as KernelColumn>::Item, )+
                    Op,
                >>::run(&policy, &$first_field, $( &$field, )+)
            }
        }

        impl<$first, $( $col ),+, Op> TransformInput<Op>
            for $soa_name<$first, $( $col ),+>
        where
            Self: SoVA<
                Runtime = <$first as KernelColumn>::Runtime,
                Item = (
                    <$first as KernelColumn>::Item,
                    $( <$col as KernelColumn>::Item, )+
                ),
                Scalar = <$first as KernelColumn>::Item,
            >,
            $first: KernelColumn + KernelColumnAt<S0>,
            $( $col: KernelColumn<Runtime = <$first as KernelColumn>::Runtime> + KernelColumnAt<S0>, )+
            <$first as KernelColumn>::Runtime: Runtime,
            <$first as KernelColumn>::Item: CubePrimitive + CubeElement,
            $( <$col as KernelColumn>::Item: CubePrimitive + CubeElement, )+
            <$first as KernelColumn>::Expr: DeviceGpuExpr<<$first as KernelColumn>::Item>,
            $( <$col as KernelColumn>::Expr: DeviceGpuExpr<<$col as KernelColumn>::Item>, )+
            Op: UnaryOp<(
                <$first as KernelColumn>::Item,
                $( <$col as KernelColumn>::Item, )+
            )>,
            Op::Output: $trait_name<
                <$first as KernelColumn>::Runtime,
                <$first as KernelColumn>::Item,
                $( <$col as KernelColumn>::Item, )+
                Op,
            >,
        {
            type Output =
                <Op::Output as StorageOutput<<$first as KernelColumn>::Runtime>>::Storage;

            fn transform_input(self, _op: GpuOp<Op>) -> Result<Self::Output, Error> {
                SoVA::validate(&self)?;
                let policy = self.policy().clone();
                let $first_field = super::device_expr_collect(&self.$first_field)?;
                $(
                    let $field = super::device_expr_collect(&self.$field)?;
                )+
                <Op::Output as $trait_name<
                    <$first as KernelColumn>::Runtime,
                    <$first as KernelColumn>::Item,
                    $( <$col as KernelColumn>::Item, )+
                    Op,
                >>::run(&policy, &$first_field, $( &$field, )+)
            }
        }
    };
}

define_transform_soa_output!(
    TransformSoA5Output,
    SoVA5<A: a, B: b, C: c, D: d, E: e>,
    SoA5 { a, b, c, d, e },
    transform_tuple5_kernel,
    transform_tuple5_to_tuple5_kernel,
    (OutA: out_a, OutB: out_b, OutC: out_c, OutD: out_d, OutE: out_e)
);
define_transform_soa_output!(
    TransformSoA6Output,
    SoVA6<A: a, B: b, C: c, D: d, E: e, F: f>,
    SoA6 { a, b, c, d, e, f },
    transform_tuple6_kernel,
    transform_tuple6_to_tuple6_kernel,
    (OutA: out_a, OutB: out_b, OutC: out_c, OutD: out_d, OutE: out_e, OutF: out_f)
);
define_transform_soa_output!(
    TransformSoA7Output,
    SoVA7<A: a, B: b, C: c, D: d, E: e, F: f, G: g>,
    SoA7 { a, b, c, d, e, f, g },
    transform_tuple7_kernel,
    transform_tuple7_to_tuple7_kernel,
    (OutA: out_a, OutB: out_b, OutC: out_c, OutD: out_d, OutE: out_e, OutF: out_f, OutG: out_g)
);
define_transform_soa_output!(
    TransformSoA8Output,
    SoVA8<A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h>,
    SoA8 { a, b, c, d, e, f, g, h },
    transform_tuple8_kernel,
    transform_tuple8_to_tuple8_kernel,
    (OutA: out_a, OutB: out_b, OutC: out_c, OutD: out_d, OutE: out_e, OutF: out_f, OutG: out_g, OutH: out_h)
);
define_transform_soa_output!(
    TransformSoA9Output,
    SoVA9<A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i>,
    SoA9 { a, b, c, d, e, f, g, h, i },
    transform_tuple9_kernel,
    transform_tuple9_to_tuple9_kernel,
    (OutA: out_a, OutB: out_b, OutC: out_c, OutD: out_d, OutE: out_e, OutF: out_f, OutG: out_g, OutH: out_h, OutI: out_i)
);
define_transform_soa_output!(
    TransformSoA10Output,
    SoVA10<A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i, J: j>,
    SoA10 { a, b, c, d, e, f, g, h, i, j },
    transform_tuple10_kernel,
    transform_tuple10_to_tuple10_kernel,
    (OutA: out_a, OutB: out_b, OutC: out_c, OutD: out_d, OutE: out_e, OutF: out_f, OutG: out_g, OutH: out_h, OutI: out_i, OutJ: out_j)
);
define_transform_soa_output!(
    TransformSoA11Output,
    SoVA11<A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i, J: j, K: k>,
    SoA11 { a, b, c, d, e, f, g, h, i, j, k },
    transform_tuple11_kernel,
    transform_tuple11_to_tuple11_kernel,
    (OutA: out_a, OutB: out_b, OutC: out_c, OutD: out_d, OutE: out_e, OutF: out_f, OutG: out_g, OutH: out_h, OutI: out_i, OutJ: out_j, OutK: out_k)
);

#[doc(hidden)]
pub trait TransformSoA12Output<R, A, B, C, D, E, F, G, H, I, J, K, L, Op>:
    CubeType + StorageOutput<R>
where
    R: Runtime,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    C: CubePrimitive + CubeElement,
    D: CubePrimitive + CubeElement,
    E: CubePrimitive + CubeElement,
    F: CubePrimitive + CubeElement,
    G: CubePrimitive + CubeElement,
    H: CubePrimitive + CubeElement,
    I: CubePrimitive + CubeElement,
    J: CubePrimitive + CubeElement,
    K: CubePrimitive + CubeElement,
    L: CubePrimitive + CubeElement,
    Op: UnaryOp<(A, B, C, D, E, F, G, H, I, J, K, L), Output = Self>,
{
    // The output value chooses its storage shape through StorageOutput; this
    // trait only selects the kernel needed for this input arity.
    fn run(
        policy: &crate::policy::CubePolicy<R>,
        a: &DeviceVec<R, A>,
        b: &DeviceVec<R, B>,
        c: &DeviceVec<R, C>,
        d: &DeviceVec<R, D>,
        e: &DeviceVec<R, E>,
        f: &DeviceVec<R, F>,
        g: &DeviceVec<R, G>,
        h: &DeviceVec<R, H>,
        i: &DeviceVec<R, I>,
        j: &DeviceVec<R, J>,
        k: &DeviceVec<R, K>,
        l: &DeviceVec<R, L>,
    ) -> Result<<Self as StorageOutput<R>>::Storage, Error>;
}

macro_rules! impl_scalar_transform_soa12_output {
    ($($out:ty),+ $(,)?) => {
        $(
            impl<R, A, B, C, D, E, F, G, H, I, J, K, L, Op>
                TransformSoA12Output<R, A, B, C, D, E, F, G, H, I, J, K, L, Op> for $out
            where
                R: Runtime,
                A: CubePrimitive + CubeElement,
                B: CubePrimitive + CubeElement,
                C: CubePrimitive + CubeElement,
                D: CubePrimitive + CubeElement,
                E: CubePrimitive + CubeElement,
                F: CubePrimitive + CubeElement,
                G: CubePrimitive + CubeElement,
                H: CubePrimitive + CubeElement,
                I: CubePrimitive + CubeElement,
                J: CubePrimitive + CubeElement,
                K: CubePrimitive + CubeElement,
                L: CubePrimitive + CubeElement,
                Op: UnaryOp<(A, B, C, D, E, F, G, H, I, J, K, L), Output = $out>,
            {
                fn run(
                    policy: &crate::policy::CubePolicy<R>,
                    a: &DeviceVec<R, A>,
                    b: &DeviceVec<R, B>,
                    c: &DeviceVec<R, C>,
                    d: &DeviceVec<R, D>,
                    e: &DeviceVec<R, E>,
                    f: &DeviceVec<R, F>,
                    g: &DeviceVec<R, G>,
                    h: &DeviceVec<R, H>,
                    i: &DeviceVec<R, I>,
                    j: &DeviceVec<R, J>,
                    k: &DeviceVec<R, K>,
                    l: &DeviceVec<R, L>,
                ) -> Result<<Self as StorageOutput<R>>::Storage, Error> {
                    let len = a.len();
                    let client = policy.client();
                    let output_handle = client.empty(len * std::mem::size_of::<$out>());
                    if len != 0 {
                        let len_u32 = u32::try_from(len)
                            .map_err(|_| Error::LengthTooLarge { len })?;
                        let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
                        let block_size = 256_u32;
                        let block_count = len.div_ceil(block_size as usize);
                        let block_count_u32 = u32::try_from(block_count)
                            .map_err(|_| Error::LengthTooLarge { len: block_count })?;
                        unsafe {
                            transform_tuple12_kernel::launch_unchecked::<
                                A, B, C, D, E, F, G, H, I, J, K, L, $out, Op, R,
                            >(
                                client,
                                CubeCount::Static(block_count_u32, 1, 1),
                                CubeDim::new_1d(block_size),
                                ArrayArg::from_raw_parts::<A>(&a.handle, len, 1),
                                ArrayArg::from_raw_parts::<B>(&b.handle, len, 1),
                                ArrayArg::from_raw_parts::<C>(&c.handle, len, 1),
                                ArrayArg::from_raw_parts::<D>(&d.handle, len, 1),
                                ArrayArg::from_raw_parts::<E>(&e.handle, len, 1),
                                ArrayArg::from_raw_parts::<F>(&f.handle, len, 1),
                                ArrayArg::from_raw_parts::<G>(&g.handle, len, 1),
                                ArrayArg::from_raw_parts::<H>(&h.handle, len, 1),
                                ArrayArg::from_raw_parts::<I>(&i.handle, len, 1),
                                ArrayArg::from_raw_parts::<J>(&j.handle, len, 1),
                                ArrayArg::from_raw_parts::<K>(&k.handle, len, 1),
                                ArrayArg::from_raw_parts::<L>(&l.handle, len, 1),
                                ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
                                ArrayArg::from_raw_parts::<$out>(&output_handle, len, 1),
                            )
                            .map_err(|err| Error::Launch {
                                message: format!("{err:?}"),
                            })?;
                        }
                    }
                    Ok(SoA1 {
                        source: DeviceVec::from_handle(policy.clone(), output_handle, len),
                    })
                }
            }
        )+
    };
}

impl_scalar_transform_soa12_output!(f32, f64, u8, u16, u32, u64, i8, i16, i32, i64, bool);

impl_transform_tuple_outputs!(
    TransformSoA2Output<A: a, B: b>,
    3 => transform_tuple2_to_tuple3_kernel,
    4 => transform_tuple2_to_tuple4_kernel,
    5 => transform_tuple2_to_tuple5_kernel,
    6 => transform_tuple2_to_tuple6_kernel,
    7 => transform_tuple2_to_tuple7_kernel,
    8 => transform_tuple2_to_tuple8_kernel,
    9 => transform_tuple2_to_tuple9_kernel,
    10 => transform_tuple2_to_tuple10_kernel,
    11 => transform_tuple2_to_tuple11_kernel,
    12 => transform_tuple2_to_tuple12_kernel,
);
impl_transform_tuple_outputs!(
    TransformSoA3Output<A: a, B: b, C: c>,
    2 => transform_tuple3_to_tuple2_kernel,
    4 => transform_tuple3_to_tuple4_kernel,
    5 => transform_tuple3_to_tuple5_kernel,
    6 => transform_tuple3_to_tuple6_kernel,
    7 => transform_tuple3_to_tuple7_kernel,
    8 => transform_tuple3_to_tuple8_kernel,
    9 => transform_tuple3_to_tuple9_kernel,
    10 => transform_tuple3_to_tuple10_kernel,
    11 => transform_tuple3_to_tuple11_kernel,
    12 => transform_tuple3_to_tuple12_kernel,
);
impl_transform_tuple_outputs!(
    TransformSoA4Output<A: a, B: b, C: c, D: d>,
    2 => transform_tuple4_to_tuple2_kernel,
    3 => transform_tuple4_to_tuple3_kernel,
    5 => transform_tuple4_to_tuple5_kernel,
    6 => transform_tuple4_to_tuple6_kernel,
    7 => transform_tuple4_to_tuple7_kernel,
    8 => transform_tuple4_to_tuple8_kernel,
    9 => transform_tuple4_to_tuple9_kernel,
    10 => transform_tuple4_to_tuple10_kernel,
    11 => transform_tuple4_to_tuple11_kernel,
    12 => transform_tuple4_to_tuple12_kernel,
);
impl_transform_tuple_outputs!(
    TransformSoA5Output<A: a, B: b, C: c, D: d, E: e>,
    2 => transform_tuple5_to_tuple2_kernel,
    3 => transform_tuple5_to_tuple3_kernel,
    4 => transform_tuple5_to_tuple4_kernel,
    6 => transform_tuple5_to_tuple6_kernel,
    7 => transform_tuple5_to_tuple7_kernel,
    8 => transform_tuple5_to_tuple8_kernel,
    9 => transform_tuple5_to_tuple9_kernel,
    10 => transform_tuple5_to_tuple10_kernel,
    11 => transform_tuple5_to_tuple11_kernel,
    12 => transform_tuple5_to_tuple12_kernel,
);
impl_transform_tuple_outputs!(
    TransformSoA6Output<A: a, B: b, C: c, D: d, E: e, F: f>,
    2 => transform_tuple6_to_tuple2_kernel,
    3 => transform_tuple6_to_tuple3_kernel,
    4 => transform_tuple6_to_tuple4_kernel,
    5 => transform_tuple6_to_tuple5_kernel,
    7 => transform_tuple6_to_tuple7_kernel,
    8 => transform_tuple6_to_tuple8_kernel,
    9 => transform_tuple6_to_tuple9_kernel,
    10 => transform_tuple6_to_tuple10_kernel,
    11 => transform_tuple6_to_tuple11_kernel,
    12 => transform_tuple6_to_tuple12_kernel,
);
impl_transform_tuple_outputs!(
    TransformSoA7Output<A: a, B: b, C: c, D: d, E: e, F: f, G: g>,
    2 => transform_tuple7_to_tuple2_kernel,
    3 => transform_tuple7_to_tuple3_kernel,
    4 => transform_tuple7_to_tuple4_kernel,
    5 => transform_tuple7_to_tuple5_kernel,
    6 => transform_tuple7_to_tuple6_kernel,
    8 => transform_tuple7_to_tuple8_kernel,
    9 => transform_tuple7_to_tuple9_kernel,
    10 => transform_tuple7_to_tuple10_kernel,
    11 => transform_tuple7_to_tuple11_kernel,
    12 => transform_tuple7_to_tuple12_kernel,
);
impl_transform_tuple_outputs!(
    TransformSoA8Output<A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h>,
    2 => transform_tuple8_to_tuple2_kernel,
    3 => transform_tuple8_to_tuple3_kernel,
    4 => transform_tuple8_to_tuple4_kernel,
    5 => transform_tuple8_to_tuple5_kernel,
    6 => transform_tuple8_to_tuple6_kernel,
    7 => transform_tuple8_to_tuple7_kernel,
    9 => transform_tuple8_to_tuple9_kernel,
    10 => transform_tuple8_to_tuple10_kernel,
    11 => transform_tuple8_to_tuple11_kernel,
    12 => transform_tuple8_to_tuple12_kernel,
);
impl_transform_tuple_outputs!(
    TransformSoA9Output<A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i>,
    2 => transform_tuple9_to_tuple2_kernel,
    3 => transform_tuple9_to_tuple3_kernel,
    4 => transform_tuple9_to_tuple4_kernel,
    5 => transform_tuple9_to_tuple5_kernel,
    6 => transform_tuple9_to_tuple6_kernel,
    7 => transform_tuple9_to_tuple7_kernel,
    8 => transform_tuple9_to_tuple8_kernel,
    10 => transform_tuple9_to_tuple10_kernel,
    11 => transform_tuple9_to_tuple11_kernel,
    12 => transform_tuple9_to_tuple12_kernel,
);
impl_transform_tuple_outputs!(
    TransformSoA10Output<A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i, J: j>,
    2 => transform_tuple10_to_tuple2_kernel,
    3 => transform_tuple10_to_tuple3_kernel,
    4 => transform_tuple10_to_tuple4_kernel,
    5 => transform_tuple10_to_tuple5_kernel,
    6 => transform_tuple10_to_tuple6_kernel,
    7 => transform_tuple10_to_tuple7_kernel,
    8 => transform_tuple10_to_tuple8_kernel,
    9 => transform_tuple10_to_tuple9_kernel,
    11 => transform_tuple10_to_tuple11_kernel,
    12 => transform_tuple10_to_tuple12_kernel,
);
impl_transform_tuple_outputs!(
    TransformSoA11Output<A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i, J: j, K: k>,
    2 => transform_tuple11_to_tuple2_kernel,
    3 => transform_tuple11_to_tuple3_kernel,
    4 => transform_tuple11_to_tuple4_kernel,
    5 => transform_tuple11_to_tuple5_kernel,
    6 => transform_tuple11_to_tuple6_kernel,
    7 => transform_tuple11_to_tuple7_kernel,
    8 => transform_tuple11_to_tuple8_kernel,
    9 => transform_tuple11_to_tuple9_kernel,
    10 => transform_tuple11_to_tuple10_kernel,
    12 => transform_tuple11_to_tuple12_kernel,
);
impl_transform_tuple_outputs!(
    TransformSoA12Output<A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h, I: i, J: j, K: k, L: l>,
    2 => transform_tuple12_to_tuple2_kernel,
    3 => transform_tuple12_to_tuple3_kernel,
    4 => transform_tuple12_to_tuple4_kernel,
    5 => transform_tuple12_to_tuple5_kernel,
    6 => transform_tuple12_to_tuple6_kernel,
    7 => transform_tuple12_to_tuple7_kernel,
    8 => transform_tuple12_to_tuple8_kernel,
    9 => transform_tuple12_to_tuple9_kernel,
    10 => transform_tuple12_to_tuple10_kernel,
    11 => transform_tuple12_to_tuple11_kernel,
);

impl<
    R,
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
    I,
    J,
    K,
    L,
    OutA,
    OutB,
    OutC,
    OutD,
    OutE,
    OutF,
    OutG,
    OutH,
    OutI,
    OutJ,
    OutK,
    OutL,
    Op,
> TransformSoA12Output<R, A, B, C, D, E, F, G, H, I, J, K, L, Op>
    for (
        OutA,
        OutB,
        OutC,
        OutD,
        OutE,
        OutF,
        OutG,
        OutH,
        OutI,
        OutJ,
        OutK,
        OutL,
    )
where
    R: Runtime,
    A: CubePrimitive + CubeElement,
    B: CubePrimitive + CubeElement,
    C: CubePrimitive + CubeElement,
    D: CubePrimitive + CubeElement,
    E: CubePrimitive + CubeElement,
    F: CubePrimitive + CubeElement,
    G: CubePrimitive + CubeElement,
    H: CubePrimitive + CubeElement,
    I: CubePrimitive + CubeElement,
    J: CubePrimitive + CubeElement,
    K: CubePrimitive + CubeElement,
    L: CubePrimitive + CubeElement,
    OutA: CubePrimitive + CubeElement,
    OutB: CubePrimitive + CubeElement,
    OutC: CubePrimitive + CubeElement,
    OutD: CubePrimitive + CubeElement,
    OutE: CubePrimitive + CubeElement,
    OutF: CubePrimitive + CubeElement,
    OutG: CubePrimitive + CubeElement,
    OutH: CubePrimitive + CubeElement,
    OutI: CubePrimitive + CubeElement,
    OutJ: CubePrimitive + CubeElement,
    OutK: CubePrimitive + CubeElement,
    OutL: CubePrimitive + CubeElement,
    Op: UnaryOp<
            (A, B, C, D, E, F, G, H, I, J, K, L),
            Output = (
                OutA,
                OutB,
                OutC,
                OutD,
                OutE,
                OutF,
                OutG,
                OutH,
                OutI,
                OutJ,
                OutK,
                OutL,
            ),
        >,
{
    fn run(
        policy: &crate::policy::CubePolicy<R>,
        a: &DeviceVec<R, A>,
        b: &DeviceVec<R, B>,
        c: &DeviceVec<R, C>,
        d: &DeviceVec<R, D>,
        e: &DeviceVec<R, E>,
        f: &DeviceVec<R, F>,
        g: &DeviceVec<R, G>,
        h: &DeviceVec<R, H>,
        i: &DeviceVec<R, I>,
        j: &DeviceVec<R, J>,
        k: &DeviceVec<R, K>,
        l: &DeviceVec<R, L>,
    ) -> Result<<Self as StorageOutput<R>>::Storage, Error> {
        let len = a.len();
        let client = policy.client();
        let out_a = client.empty(len * std::mem::size_of::<OutA>());
        let out_b = client.empty(len * std::mem::size_of::<OutB>());
        let out_c = client.empty(len * std::mem::size_of::<OutC>());
        let out_d = client.empty(len * std::mem::size_of::<OutD>());
        let out_e = client.empty(len * std::mem::size_of::<OutE>());
        let out_f = client.empty(len * std::mem::size_of::<OutF>());
        let out_g = client.empty(len * std::mem::size_of::<OutG>());
        let out_h = client.empty(len * std::mem::size_of::<OutH>());
        let out_i = client.empty(len * std::mem::size_of::<OutI>());
        let out_j = client.empty(len * std::mem::size_of::<OutJ>());
        let out_k = client.empty(len * std::mem::size_of::<OutK>());
        let out_l = client.empty(len * std::mem::size_of::<OutL>());
        if len != 0 {
            let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
            let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
            let block_size = 256_u32;
            let block_count = len.div_ceil(block_size as usize);
            let block_count_u32 = u32::try_from(block_count)
                .map_err(|_| Error::LengthTooLarge { len: block_count })?;
            unsafe {
                transform_tuple12_to_tuple12_kernel::launch_unchecked::<
                    A,
                    B,
                    C,
                    D,
                    E,
                    F,
                    G,
                    H,
                    I,
                    J,
                    K,
                    L,
                    OutA,
                    OutB,
                    OutC,
                    OutD,
                    OutE,
                    OutF,
                    OutG,
                    OutH,
                    OutI,
                    OutJ,
                    OutK,
                    OutL,
                    Op,
                    R,
                >(
                    client,
                    CubeCount::Static(block_count_u32, 1, 1),
                    CubeDim::new_1d(block_size),
                    ArrayArg::from_raw_parts::<A>(&a.handle, len, 1),
                    ArrayArg::from_raw_parts::<B>(&b.handle, len, 1),
                    ArrayArg::from_raw_parts::<C>(&c.handle, len, 1),
                    ArrayArg::from_raw_parts::<D>(&d.handle, len, 1),
                    ArrayArg::from_raw_parts::<E>(&e.handle, len, 1),
                    ArrayArg::from_raw_parts::<F>(&f.handle, len, 1),
                    ArrayArg::from_raw_parts::<G>(&g.handle, len, 1),
                    ArrayArg::from_raw_parts::<H>(&h.handle, len, 1),
                    ArrayArg::from_raw_parts::<I>(&i.handle, len, 1),
                    ArrayArg::from_raw_parts::<J>(&j.handle, len, 1),
                    ArrayArg::from_raw_parts::<K>(&k.handle, len, 1),
                    ArrayArg::from_raw_parts::<L>(&l.handle, len, 1),
                    ArrayArg::from_raw_parts::<u32>(&len_handle, 1, 1),
                    ArrayArg::from_raw_parts::<OutA>(&out_a, len, 1),
                    ArrayArg::from_raw_parts::<OutB>(&out_b, len, 1),
                    ArrayArg::from_raw_parts::<OutC>(&out_c, len, 1),
                    ArrayArg::from_raw_parts::<OutD>(&out_d, len, 1),
                    ArrayArg::from_raw_parts::<OutE>(&out_e, len, 1),
                    ArrayArg::from_raw_parts::<OutF>(&out_f, len, 1),
                    ArrayArg::from_raw_parts::<OutG>(&out_g, len, 1),
                    ArrayArg::from_raw_parts::<OutH>(&out_h, len, 1),
                    ArrayArg::from_raw_parts::<OutI>(&out_i, len, 1),
                    ArrayArg::from_raw_parts::<OutJ>(&out_j, len, 1),
                    ArrayArg::from_raw_parts::<OutK>(&out_k, len, 1),
                    ArrayArg::from_raw_parts::<OutL>(&out_l, len, 1),
                )
                .map_err(|err| Error::Launch {
                    message: format!("{err:?}"),
                })?;
            }
        }
        Ok(SoA12 {
            a: DeviceVec::from_handle(policy.clone(), out_a, len),
            b: DeviceVec::from_handle(policy.clone(), out_b, len),
            c: DeviceVec::from_handle(policy.clone(), out_c, len),
            d: DeviceVec::from_handle(policy.clone(), out_d, len),
            e: DeviceVec::from_handle(policy.clone(), out_e, len),
            f: DeviceVec::from_handle(policy.clone(), out_f, len),
            g: DeviceVec::from_handle(policy.clone(), out_g, len),
            h: DeviceVec::from_handle(policy.clone(), out_h, len),
            i: DeviceVec::from_handle(policy.clone(), out_i, len),
            j: DeviceVec::from_handle(policy.clone(), out_j, len),
            k: DeviceVec::from_handle(policy.clone(), out_k, len),
            l: DeviceVec::from_handle(policy.clone(), out_l, len),
        })
    }
}

impl<A, B, C, D, E, F, G, H, I, J, K, L, Op> TransformInput<Op>
    for SoVA12<A, B, C, D, E, F, G, H, I, J, K, L>
where
    Self: SoVA<
            Runtime = A::Runtime,
            Item = (
                A::Item,
                B::Item,
                C::Item,
                D::Item,
                E::Item,
                F::Item,
                G::Item,
                H::Item,
                I::Item,
                J::Item,
                K::Item,
                L::Item,
            ),
            Scalar = A::Item,
        >,
    A: KernelColumn + KernelColumnAt<S0>,
    B: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    D: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    E: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    F: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    G: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    H: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    I: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    J: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    K: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    L: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    A::Item: CubePrimitive + CubeElement,
    B::Item: CubePrimitive + CubeElement,
    C::Item: CubePrimitive + CubeElement,
    D::Item: CubePrimitive + CubeElement,
    E::Item: CubePrimitive + CubeElement,
    F::Item: CubePrimitive + CubeElement,
    G::Item: CubePrimitive + CubeElement,
    H::Item: CubePrimitive + CubeElement,
    I::Item: CubePrimitive + CubeElement,
    J::Item: CubePrimitive + CubeElement,
    K::Item: CubePrimitive + CubeElement,
    L::Item: CubePrimitive + CubeElement,
    A::Expr: DeviceGpuExpr<A::Item>,
    B::Expr: DeviceGpuExpr<B::Item>,
    C::Expr: DeviceGpuExpr<C::Item>,
    D::Expr: DeviceGpuExpr<D::Item>,
    E::Expr: DeviceGpuExpr<E::Item>,
    F::Expr: DeviceGpuExpr<F::Item>,
    G::Expr: DeviceGpuExpr<G::Item>,
    H::Expr: DeviceGpuExpr<H::Item>,
    I::Expr: DeviceGpuExpr<I::Item>,
    J::Expr: DeviceGpuExpr<J::Item>,
    K::Expr: DeviceGpuExpr<K::Item>,
    L::Expr: DeviceGpuExpr<L::Item>,
    Op: UnaryOp<(
        A::Item,
        B::Item,
        C::Item,
        D::Item,
        E::Item,
        F::Item,
        G::Item,
        H::Item,
        I::Item,
        J::Item,
        K::Item,
        L::Item,
    )>,
    Op::Output: TransformSoA12Output<
            A::Runtime,
            A::Item,
            B::Item,
            C::Item,
            D::Item,
            E::Item,
            F::Item,
            G::Item,
            H::Item,
            I::Item,
            J::Item,
            K::Item,
            L::Item,
            Op,
        >,
{
    type Output = <Op::Output as StorageOutput<A::Runtime>>::Storage;

    fn transform_input(self, _op: GpuOp<Op>) -> Result<Self::Output, Error> {
        SoVA::validate(&self)?;
        let policy = self.policy().clone();
        let a = super::device_expr_collect(&self.a)?;
        let b = super::device_expr_collect(&self.b)?;
        let c = super::device_expr_collect(&self.c)?;
        let d = super::device_expr_collect(&self.d)?;
        let e = super::device_expr_collect(&self.e)?;
        let f = super::device_expr_collect(&self.f)?;
        let g = super::device_expr_collect(&self.g)?;
        let h = super::device_expr_collect(&self.h)?;
        let i = super::device_expr_collect(&self.i)?;
        let j = super::device_expr_collect(&self.j)?;
        let k = super::device_expr_collect(&self.k)?;
        let l = super::device_expr_collect(&self.l)?;
        <Op::Output as TransformSoA12Output<
            A::Runtime,
            A::Item,
            B::Item,
            C::Item,
            D::Item,
            E::Item,
            F::Item,
            G::Item,
            H::Item,
            I::Item,
            J::Item,
            K::Item,
            L::Item,
            Op,
        >>::run(&policy, &a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l)
    }
}

impl<A, B, C, D, E, F, G, H, I, J, K, L, Op> TransformInput<Op>
    for SoA12<A, B, C, D, E, F, G, H, I, J, K, L>
where
    Self: SoVA<
            Runtime = A::Runtime,
            Item = (
                A::Item,
                B::Item,
                C::Item,
                D::Item,
                E::Item,
                F::Item,
                G::Item,
                H::Item,
                I::Item,
                J::Item,
                K::Item,
                L::Item,
            ),
            Scalar = A::Item,
        >,
    A: KernelColumn + KernelColumnAt<S0>,
    B: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    C: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    D: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    E: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    F: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    G: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    H: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    I: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    J: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    K: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    L: KernelColumn<Runtime = A::Runtime> + KernelColumnAt<S0>,
    A::Item: CubePrimitive + CubeElement,
    B::Item: CubePrimitive + CubeElement,
    C::Item: CubePrimitive + CubeElement,
    D::Item: CubePrimitive + CubeElement,
    E::Item: CubePrimitive + CubeElement,
    F::Item: CubePrimitive + CubeElement,
    G::Item: CubePrimitive + CubeElement,
    H::Item: CubePrimitive + CubeElement,
    I::Item: CubePrimitive + CubeElement,
    J::Item: CubePrimitive + CubeElement,
    K::Item: CubePrimitive + CubeElement,
    L::Item: CubePrimitive + CubeElement,
    A::Expr: DeviceGpuExpr<A::Item>,
    B::Expr: DeviceGpuExpr<B::Item>,
    C::Expr: DeviceGpuExpr<C::Item>,
    D::Expr: DeviceGpuExpr<D::Item>,
    E::Expr: DeviceGpuExpr<E::Item>,
    F::Expr: DeviceGpuExpr<F::Item>,
    G::Expr: DeviceGpuExpr<G::Item>,
    H::Expr: DeviceGpuExpr<H::Item>,
    I::Expr: DeviceGpuExpr<I::Item>,
    J::Expr: DeviceGpuExpr<J::Item>,
    K::Expr: DeviceGpuExpr<K::Item>,
    L::Expr: DeviceGpuExpr<L::Item>,
    Op: UnaryOp<(
        A::Item,
        B::Item,
        C::Item,
        D::Item,
        E::Item,
        F::Item,
        G::Item,
        H::Item,
        I::Item,
        J::Item,
        K::Item,
        L::Item,
    )>,
    Op::Output: TransformSoA12Output<
            A::Runtime,
            A::Item,
            B::Item,
            C::Item,
            D::Item,
            E::Item,
            F::Item,
            G::Item,
            H::Item,
            I::Item,
            J::Item,
            K::Item,
            L::Item,
            Op,
        >,
{
    type Output = <Op::Output as StorageOutput<A::Runtime>>::Storage;

    fn transform_input(self, _op: GpuOp<Op>) -> Result<Self::Output, Error> {
        SoVA::validate(&self)?;
        let policy = self.policy().clone();
        let a = super::device_expr_collect(&self.a)?;
        let b = super::device_expr_collect(&self.b)?;
        let c = super::device_expr_collect(&self.c)?;
        let d = super::device_expr_collect(&self.d)?;
        let e = super::device_expr_collect(&self.e)?;
        let f = super::device_expr_collect(&self.f)?;
        let g = super::device_expr_collect(&self.g)?;
        let h = super::device_expr_collect(&self.h)?;
        let i = super::device_expr_collect(&self.i)?;
        let j = super::device_expr_collect(&self.j)?;
        let k = super::device_expr_collect(&self.k)?;
        let l = super::device_expr_collect(&self.l)?;
        <Op::Output as TransformSoA12Output<
            A::Runtime,
            A::Item,
            B::Item,
            C::Item,
            D::Item,
            E::Item,
            F::Item,
            G::Item,
            H::Item,
            I::Item,
            J::Item,
            K::Item,
            L::Item,
            Op,
        >>::run(&policy, &a, &b, &c, &d, &e, &f, &g, &h, &i, &j, &k, &l)
    }
}

/// Internal output that can be materialized into public owned device values.
#[doc(hidden)]
pub trait MaterializeOutput {
    /// Public output produced by materializing this internal output.
    type Output;

    /// Materializes this internal output.
    fn materialize_output(self) -> Result<Self::Output, Error>;
}

impl<Left, Right> MaterializeOutput for SoA2<Left, Right>
where
    Self: SoA<Item = (Left::Item, Right::Item), Scalar = Left::Item>,
    Left: StorageKernelColumn + KernelColumnAt<S0>,
    Right: StorageKernelColumn<Runtime = Left::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<Left as KernelColumnAt<S0>>::Next>,
    Left::Item: CubePrimitive + CubeElement,
    Right::Item: CubePrimitive + CubeElement,
    Left::Expr: DeviceGpuExpr<Left::Item>,
    Right::Expr: DeviceGpuExpr<Right::Item>,
{
    type Output = (
        DeviceVec<Left::Runtime, Left::Item>,
        DeviceVec<Left::Runtime, Right::Item>,
    );

    fn materialize_output(self) -> Result<Self::Output, Error> {
        SoA::validate(&self)?;
        let left = super::device_expr_collect(&self.left)?;
        let right = super::device_expr_collect(&self.right)?;
        Ok((left, right))
    }
}

impl<Source> MaterializeOutput for SoA1<Source>
where
    Self: SoA<Item = Source::Item, Scalar = Source::Item>,
    Source: StorageKernelColumn + KernelColumnAt<S0>,
    Source::Item: CubePrimitive + CubeElement,
    Source::Expr: DeviceGpuExpr<Source::Item>,
{
    type Output = DeviceVec<Source::Runtime, Source::Item>;

    fn materialize_output(self) -> Result<Self::Output, Error> {
        SoA::validate(&self)?;
        let source = super::device_expr_collect(&self.source)?;
        Ok(source)
    }
}

impl<R, T> MaterializeOutput for DeviceVec<R, T>
where
    R: Runtime,
    T: CubePrimitive + CubeElement,
{
    type Output = Self;

    fn materialize_output(self) -> Result<Self::Output, Error> {
        Ok(self)
    }
}

impl<First, Second, Third> MaterializeOutput for SoA3<First, Second, Third>
where
    Self: SoA<Item = (First::Item, Second::Item, Third::Item), Scalar = First::Item>,
    First: StorageKernelColumn + KernelColumnAt<S0>,
    Second: StorageKernelColumn<Runtime = First::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<First as KernelColumnAt<S0>>::Next>,
    Third: StorageKernelColumn<Runtime = First::Runtime>
        + KernelColumnAt<S0>
        + KernelColumnAt<<Second as KernelColumnAt<<First as KernelColumnAt<S0>>::Next>>::Next>,
    First::Item: CubePrimitive + CubeElement,
    Second::Item: CubePrimitive + CubeElement,
    Third::Item: CubePrimitive + CubeElement,
    First::Expr: DeviceGpuExpr<First::Item>,
    Second::Expr: DeviceGpuExpr<Second::Item>,
    Third::Expr: DeviceGpuExpr<Third::Item>,
{
    type Output = (
        DeviceVec<First::Runtime, First::Item>,
        DeviceVec<First::Runtime, Second::Item>,
        DeviceVec<First::Runtime, Third::Item>,
    );

    fn materialize_output(self) -> Result<Self::Output, Error> {
        SoA::validate(&self)?;
        let first = super::device_expr_collect(&self.first)?;
        let second = super::device_expr_collect(&self.second)?;
        let third = super::device_expr_collect(&self.third)?;
        Ok((first, second, third))
    }
}

macro_rules! impl_materialize_output {
    ($name:ident < $first:ident, $( $rest:ident ),+ > { $first_field:ident, $( $field:ident ),+ }) => {
        impl<$first, $( $rest ),+> MaterializeOutput for $name<$first, $( $rest ),+>
        where
            Self: SoA,
            $first: StorageKernelColumn + KernelColumnAt<S0>,
            $(
                $rest: StorageKernelColumn<Runtime = <$first as KernelColumn>::Runtime>
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
            type Output = (
                DeviceVec<<$first as KernelColumn>::Runtime, <$first as KernelColumn>::Item>,
                $( DeviceVec<<$rest as KernelColumn>::Runtime, <$rest as KernelColumn>::Item> ),+
            );

            fn materialize_output(self) -> Result<Self::Output, Error> {
                SoA::validate(&self)?;
                let $first_field = super::device_expr_collect(&self.$first_field)?;
                $(
                    let $field = super::device_expr_collect(&self.$field)?;
                )+
                Ok(($first_field, $( $field ),+))
            }
        }
    };
}

impl_materialize_output!(SoA4<A, B, C, D> { a, b, c, d });
impl_materialize_output!(SoA5<A, B, C, D, E> { a, b, c, d, e });
impl_materialize_output!(SoA6<A, B, C, D, E, F> { a, b, c, d, e, f });
impl_materialize_output!(SoA7<A, B, C, D, E, F, G> { a, b, c, d, e, f, g });
impl_materialize_output!(SoA8<A, B, C, D, E, F, G, H> { a, b, c, d, e, f, g, h });
impl_materialize_output!(SoA9<A, B, C, D, E, F, G, H, I> { a, b, c, d, e, f, g, h, i });
impl_materialize_output!(SoA10<A, B, C, D, E, F, G, H, I, J> { a, b, c, d, e, f, g, h, i, j });
impl_materialize_output!(SoA11<A, B, C, D, E, F, G, H, I, J, K> { a, b, c, d, e, f, g, h, i, j, k });
impl_materialize_output!(SoA12<A, B, C, D, E, F, G, H, I, J, K, L> { a, b, c, d, e, f, g, h, i, j, k, l });

impl<Left, Right> MaterializeOutput for (Left, Right)
where
    Left: MaterializeOutput,
    Right: MaterializeOutput,
{
    type Output = (Left::Output, Right::Output);

    fn materialize_output(self) -> Result<Self::Output, Error> {
        Ok((self.0.materialize_output()?, self.1.materialize_output()?))
    }
}

pub(crate) fn materialize<Source>(
    source: Source,
) -> Result<<Source as MaterializeOutput>::Output, Error>
where
    Source: MaterializeOutput,
{
    source.materialize_output()
}

/// Applies a read-only transform and returns owned device storage.
///
/// The input may be a borrowed [`DeviceVec`](crate::DeviceVec) or an SoA built
/// with [`zip`]. The returned value is owned device storage: `DeviceVec` for one
/// column or a tuple of `DeviceVec`s for multiple columns.
pub fn transform<Source, Op>(
    source: Source,
    _op: Op,
) -> Result<<<Source as TransformInput<Op>>::Output as MaterializeOutput>::Output, Error>
where
    Source: TransformInput<Op>,
    <Source as TransformInput<Op>>::Output: MaterializeOutput,
{
    materialize(source.transform_input(GpuOp::<Op>::new())?)
}
