use core::marker::PhantomData;
use cubecl::prelude::*;

/// Compile-time unary operator used by transform-style algorithms.
///
/// Implement this trait on a unit-like marker type and pass the marker value
/// to [`transform`](crate::transform).
///
/// ```no_run
/// use cubecl::prelude::*;
///
/// struct AddOne;
///
/// #[cubecl::cube]
/// impl<R> massively::op::UnaryOp<R, (f32,)> for AddOne
/// where
///     R: cubecl::prelude::Runtime,
/// {
///     type Output = (f32,);
///
///     fn apply(input: (f32,)) -> (f32,) {
///         (input.0 + 1.0,)
///     }
/// }
/// ```
#[cube]
pub trait UnaryOp<R, Input>: 'static + Send + Sync
where
    R: cubecl::prelude::Runtime,
    Input: crate::MItem<R>,
{
    /// Output value produced for one logical input element.
    type Output: crate::MItem<R>;

    /// Maps one logical input element.
    fn apply(input: Input) -> Self::Output;
}

/// Composition of two unary operators.
///
/// `Compose<First, Second>` applies `First` and then feeds the result into
/// `Second`.
///
/// ```no_run
/// # use cubecl::prelude::*;
/// # struct AddOffset;
/// # struct Square;
/// # #[cubecl::cube]
/// # impl<R: Runtime> massively::op::UnaryOp<R, (u32,)> for AddOffset {
/// #     type Output = (u32,);
/// #     fn apply(input: (u32,)) -> (u32,) { (input.0 + 3,) }
/// # }
/// # #[cubecl::cube]
/// # impl<R: Runtime> massively::op::UnaryOp<R, (u32,)> for Square {
/// #     type Output = (u32,);
/// #     fn apply(input: (u32,)) -> (u32,) { (input.0 * input.0,) }
/// # }
/// let op = massively::op::compose(AddOffset, Square);
/// # let _ = op;
/// // transform(&exec, input, op, out)
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct Compose<First, Second> {
    _ops: PhantomData<fn() -> (First, Second)>,
}

impl<First, Second> Compose<First, Second> {
    /// Creates a composed operator marker.
    pub const fn new() -> Self {
        Self { _ops: PhantomData }
    }
}

/// Creates a marker for the composition `second(first(input))`.
pub fn compose<First, Second>(_first: First, _second: Second) -> Compose<First, Second> {
    Compose::new()
}

#[cube]
impl<R, Input, First, Second> UnaryOp<R, Input> for Compose<First, Second>
where
    R: cubecl::prelude::Runtime,
    Input: crate::MItem<R>,
    First: UnaryOp<R, Input>,
    Second: UnaryOp<R, First::Output>,
{
    type Output = Second::Output;

    fn apply(input: Input) -> Self::Output {
        Second::apply(First::apply(input))
    }
}

/// Compile-time binary transform.
#[cube]
pub trait BinaryOp<R, X, Y>: 'static + Send + Sync
where
    R: cubecl::prelude::Runtime,
    X: crate::MItem<R>,
    Y: crate::MItem<R>,
{
    type Output: crate::MItem<R>;

    /// Combines two values.
    fn apply(lhs: X, rhs: Y) -> Self::Output;
}

/// Compile-time same-type binary operator used by reductions and scans.
///
/// ```no_run
/// use cubecl::prelude::*;
///
/// struct Sum;
///
/// #[cubecl::cube]
/// impl<R> massively::op::ReductionOp<R, (f32,)> for Sum
/// where
///     R: cubecl::prelude::Runtime,
/// {
///     fn apply(lhs: (f32,), rhs: (f32,)) -> (f32,) {
///         (lhs.0 + rhs.0,)
///     }
/// }
/// ```
#[cube]
pub trait ReductionOp<R, X>: 'static + Send + Sync
where
    R: cubecl::prelude::Runtime,
    X: crate::MItem<R>,
{
    /// Combines two values.
    fn apply(lhs: X, rhs: X) -> X;
}

/// Compile-time predicate used by conditional algorithms such as
/// [`count_if`](crate::count_if) and [`find_if`](crate::find_if).
///
/// Stencil algorithms such as [`copy_where`](crate::copy_where) and
/// [`remove_where`](crate::remove_where) use a `u32` flag column in the public
/// API instead of taking a predicate marker.
///
/// ```no_run
/// use cubecl::prelude::*;
///
/// struct Positive;
///
/// #[cubecl::cube]
/// impl<R> massively::op::PredicateOp<R, (f32,)> for Positive
/// where
///     R: cubecl::prelude::Runtime,
/// {
///     fn apply(input: (f32,)) -> bool {
///         input.0 > 0.0
///     }
/// }
/// ```
#[cube]
pub trait PredicateOp<R, T>: 'static + Send + Sync
where
    R: cubecl::prelude::Runtime,
    T: crate::MItem<R>,
{
    /// Returns whether the element should be processed.
    fn apply(input: T) -> bool;
}

/// Compile-time binary predicate used by search and ordering algorithms.
///
/// ```no_run
/// use cubecl::prelude::*;
///
/// struct Less;
///
/// #[cubecl::cube]
/// impl<R> massively::op::BinaryPredicateOp<R, (f32,)> for Less
/// where
///     R: cubecl::prelude::Runtime,
/// {
///     fn apply(lhs: (f32,), rhs: (f32,)) -> bool {
///         lhs.0 < rhs.0
///     }
/// }
/// ```
#[cube]
pub trait BinaryPredicateOp<R, T>: 'static + Send + Sync
where
    R: cubecl::prelude::Runtime,
    T: crate::MItem<R>,
{
    /// Returns whether the pair matches.
    fn apply(lhs: T, rhs: T) -> bool;
}

/// Built-in equality predicate for algorithms whose Rust API does not take an
/// explicit key comparator.
pub struct Equal;

macro_rules! impl_scalar_binary_predicates {
    ($( $ty:ty ),+ $(,)?) => {
        $(
            #[cube]
            impl<R> BinaryPredicateOp<R, $ty> for Equal
            where
                R: cubecl::prelude::Runtime,
            {
                fn apply(lhs: $ty, rhs: $ty) -> bool {
                    lhs == rhs
                }
            }

            #[cube]
            impl<R> BinaryPredicateOp<R, $ty> for Less
            where
                R: cubecl::prelude::Runtime,
            {
                fn apply(lhs: $ty, rhs: $ty) -> bool {
                    lhs < rhs
                }
            }
        )+
    };
}

#[cube]
impl<R, T> BinaryPredicateOp<R, (T,)> for Equal
where
    R: cubecl::prelude::Runtime,
    T: crate::MItem<R>,
    (T,): crate::MItem<R> + CubeType<ExpandType = (<T as CubeType>::ExpandType,)>,
    Self: BinaryPredicateOp<R, T>,
{
    fn apply(lhs: (T,), rhs: (T,)) -> bool {
        <Self as BinaryPredicateOp<R, T>>::apply(lhs.0, rhs.0)
    }
}

#[cube]
impl<R, A, B> BinaryPredicateOp<R, (A, B)> for Equal
where
    R: cubecl::prelude::Runtime,
    A: crate::MItem<R>,
    B: crate::MItem<R>,
    (A, B): crate::MItem<R>
        + CubeType<ExpandType = (<A as CubeType>::ExpandType, <B as CubeType>::ExpandType)>,
    Self: BinaryPredicateOp<R, A> + BinaryPredicateOp<R, B>,
{
    fn apply(lhs: (A, B), rhs: (A, B)) -> bool {
        <Self as BinaryPredicateOp<R, A>>::apply(lhs.0, rhs.0)
            && <Self as BinaryPredicateOp<R, B>>::apply(lhs.1, rhs.1)
    }
}

#[cube]
impl<R, A, B, C> BinaryPredicateOp<R, (A, B, C)> for Equal
where
    R: cubecl::prelude::Runtime,
    A: crate::MItem<R>,
    B: crate::MItem<R>,
    C: crate::MItem<R>,
    (A, B, C): crate::MItem<R>
        + CubeType<
            ExpandType = (
                <A as CubeType>::ExpandType,
                <B as CubeType>::ExpandType,
                <C as CubeType>::ExpandType,
            ),
        >,
    Self: BinaryPredicateOp<R, A> + BinaryPredicateOp<R, B> + BinaryPredicateOp<R, C>,
{
    fn apply(lhs: (A, B, C), rhs: (A, B, C)) -> bool {
        <Self as BinaryPredicateOp<R, A>>::apply(lhs.0, rhs.0)
            && <Self as BinaryPredicateOp<R, B>>::apply(lhs.1, rhs.1)
            && <Self as BinaryPredicateOp<R, C>>::apply(lhs.2, rhs.2)
    }
}

/// Built-in lexicographical less-than predicate.
pub struct Less;

#[cube]
impl<R, T> BinaryPredicateOp<R, (T,)> for Less
where
    R: cubecl::prelude::Runtime,
    T: crate::MItem<R>,
    (T,): crate::MItem<R> + CubeType<ExpandType = (<T as CubeType>::ExpandType,)>,
    Self: BinaryPredicateOp<R, T>,
{
    fn apply(lhs: (T,), rhs: (T,)) -> bool {
        <Self as BinaryPredicateOp<R, T>>::apply(lhs.0, rhs.0)
    }
}

impl_scalar_binary_predicates!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

#[cube]
impl<R, A, B> BinaryPredicateOp<R, (A, B)> for Less
where
    R: cubecl::prelude::Runtime,
    A: crate::MItem<R>,
    B: crate::MItem<R>,
    A::ExpandType: Clone,
    (A, B): crate::MItem<R>
        + CubeType<ExpandType = (<A as CubeType>::ExpandType, <B as CubeType>::ExpandType)>,
    Self: BinaryPredicateOp<R, A> + BinaryPredicateOp<R, B>,
    Equal: BinaryPredicateOp<R, A>,
{
    fn apply(lhs: (A, B), rhs: (A, B)) -> bool {
        <Self as BinaryPredicateOp<R, A>>::apply(lhs.0.clone(), rhs.0.clone())
            || (<Equal as BinaryPredicateOp<R, A>>::apply(lhs.0, rhs.0)
                && <Self as BinaryPredicateOp<R, B>>::apply(lhs.1, rhs.1))
    }
}

#[cube]
impl<R, A, B, C> BinaryPredicateOp<R, (A, B, C)> for Less
where
    R: cubecl::prelude::Runtime,
    A: crate::MItem<R>,
    B: crate::MItem<R>,
    C: crate::MItem<R>,
    A::ExpandType: Clone,
    B::ExpandType: Clone,
    (A, B, C): crate::MItem<R>
        + CubeType<
            ExpandType = (
                <A as CubeType>::ExpandType,
                <B as CubeType>::ExpandType,
                <C as CubeType>::ExpandType,
            ),
        >,
    Self: BinaryPredicateOp<R, A> + BinaryPredicateOp<R, B> + BinaryPredicateOp<R, C>,
    Equal: BinaryPredicateOp<R, A> + BinaryPredicateOp<R, B>,
{
    fn apply(lhs: (A, B, C), rhs: (A, B, C)) -> bool {
        <Self as BinaryPredicateOp<R, A>>::apply(lhs.0.clone(), rhs.0.clone())
            || (<Equal as BinaryPredicateOp<R, A>>::apply(lhs.0.clone(), rhs.0.clone())
                && <Self as BinaryPredicateOp<R, B>>::apply(lhs.1.clone(), rhs.1.clone()))
            || (<Equal as BinaryPredicateOp<R, A>>::apply(lhs.0, rhs.0)
                && <Equal as BinaryPredicateOp<R, B>>::apply(lhs.1, rhs.1)
                && <Self as BinaryPredicateOp<R, C>>::apply(lhs.2, rhs.2))
    }
}

/// Runtime-local operation traits used by generated CubeCL kernels.
///
/// These are intentionally crate-private. They allow the detail layer to keep
/// scalar kernels scalar while the public API exposes only `MItem` operators.
pub(crate) mod kernel {
    use cubecl::prelude::*;

    #[cube]
    pub trait UnaryOp<Input: CubeType>: 'static + Send + Sync {
        type Output: CubeType;

        fn apply(input: Input) -> Self::Output;
    }

    #[cube]
    pub trait BinaryOp<T: CubeType>: 'static + Send + Sync {
        fn apply(lhs: T, rhs: T) -> T;
    }

    #[cube]
    pub trait PredicateOp<T: CubeType>: 'static + Send + Sync {
        fn apply(input: T) -> bool;
    }

    #[cube]
    pub trait BinaryPredicateOp<T: CubeType>: 'static + Send + Sync {
        fn apply(lhs: T, rhs: T) -> bool;
    }
}

/// Internal value-level handle for a GPU operation marker.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct GpuOp<Op> {
    _op: core::marker::PhantomData<fn() -> Op>,
}

impl<Op> GpuOp<Op> {
    /// Creates a new operation marker.
    pub(crate) const fn new() -> Self {
        Self {
            _op: core::marker::PhantomData,
        }
    }
}
