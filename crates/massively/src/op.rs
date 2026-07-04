use core::marker::PhantomData;
use cubecl::prelude::*;

/// Compile-time unary operator used by transform-style algorithms.
///
/// Implement this trait on a unit-like marker type and pass both the marker
/// value and its captured environment to [`transform`](crate::transform).
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
///     type Env = ();
///     type Output = (f32,);
///
///     fn apply(_env: (), input: (f32,)) -> (f32,) {
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
    /// Captured environment passed to the operation.
    type Env: LaunchArg + Copy + CubeType<ExpandType: Clone>;

    /// Output value produced for one logical input element.
    type Output: crate::MItem<R>;

    /// Maps one logical input element.
    fn apply(env: Self::Env, input: Input) -> Self::Output;
}

/// Composition of two unary operators.
///
/// `Compose<First, Second>` applies `First` and then feeds the result into
/// `Second`. The composed environment is the pair `(First::Env, Second::Env)`.
///
/// ```no_run
/// # use cubecl::prelude::*;
/// # struct AddOffset;
/// # struct Square;
/// # #[cubecl::cube]
/// # impl<R: Runtime> massively::op::UnaryOp<R, (u32,)> for AddOffset {
/// #     type Env = u32;
/// #     type Output = (u32,);
/// #     fn apply(offset: u32, input: (u32,)) -> (u32,) { (input.0 + offset,) }
/// # }
/// # #[cubecl::cube]
/// # impl<R: Runtime> massively::op::UnaryOp<R, (u32,)> for Square {
/// #     type Env = ();
/// #     type Output = (u32,);
/// #     fn apply(_env: (), input: (u32,)) -> (u32,) { (input.0 * input.0,) }
/// # }
/// let op = massively::op::compose(AddOffset, Square);
/// # let _ = op;
/// // transform(&exec, input, op, (3_u32, ()), out)
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

/// Unary operator that ignores its input and returns the captured environment.
///
/// This is useful with [`transform`](crate::transform) when an algorithm needs
/// a constant item stream without defining a custom operation marker.
///
/// ```no_run
/// # use cubecl::prelude::*;
/// # use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// # let exec = massively::Executor::<WgpuRuntime>::new(WgpuDevice::Cpu);
/// # let input = exec.to_device(&[1_u32, 2, 3]).unwrap();
/// let output = exec.to_device(&[0_u32; 3]).unwrap();
/// massively::transform(
///     &exec,
///     massively::SoA1(input.slice(..)),
///     massively::op::Constant::<(u32,)>::new(),
///     (42_u32,),
///     massively::SoA1(output.slice_mut(..)),
/// )
/// .unwrap();
/// # let _ = output;
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct Constant<Out> {
    _out: PhantomData<fn() -> Out>,
}

impl<Out> Constant<Out> {
    /// Creates a constant operator marker.
    pub const fn new() -> Self {
        Self { _out: PhantomData }
    }
}

/// Creates a marker for a unary operator that always returns its environment.
pub fn constant<Out>() -> Constant<Out> {
    Constant::new()
}

#[cube]
impl<R, Input, First, Second> UnaryOp<R, Input> for Compose<First, Second>
where
    R: cubecl::prelude::Runtime,
    Input: crate::MItem<R>,
    First: UnaryOp<R, Input>,
    Second: UnaryOp<R, First::Output>,
{
    type Env = (First::Env, Second::Env);
    type Output = Second::Output;

    fn apply(env: Self::Env, input: Input) -> Self::Output {
        Second::apply(env.1, First::apply(env.0, input))
    }
}

#[cube]
impl<R, Input, Out> UnaryOp<R, Input> for Constant<Out>
where
    R: cubecl::prelude::Runtime,
    Input: crate::MItem<R>,
    Out: crate::MItem<R> + LaunchArg + CubeType<ExpandType: Clone>,
{
    type Env = Out;
    type Output = Out;

    fn apply(env: Self::Env, _input: Input) -> Self::Output {
        env
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
///     type Env = ();
///
///     fn apply(_env: (), input: (f32,)) -> bool {
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
    /// Captured environment passed to the predicate.
    type Env: LaunchArg + Copy + CubeType<ExpandType: Clone>;

    /// Returns whether the element should be processed.
    fn apply(env: Self::Env, input: T) -> bool;
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

#[cube]
impl<R, T> BinaryPredicateOp<R, (T,)> for Equal
where
    R: cubecl::prelude::Runtime,
    T: CubePrimitive + CubeElement + PartialEq,
{
    fn apply(lhs: (T,), rhs: (T,)) -> bool {
        lhs.0 == rhs.0
    }
}

#[cube]
impl<R, A, C> BinaryPredicateOp<R, (A, C)> for Equal
where
    R: cubecl::prelude::Runtime,
    A: CubePrimitive + CubeElement + PartialEq,
    C: CubePrimitive + CubeElement + PartialEq,
{
    fn apply(lhs: (A, C), rhs: (A, C)) -> bool {
        lhs.0 == rhs.0 && lhs.1 == rhs.1
    }
}

#[cube]
impl<R, A, C, D> BinaryPredicateOp<R, (A, C, D)> for Equal
where
    R: cubecl::prelude::Runtime,
    A: CubePrimitive + CubeElement + PartialEq,
    C: CubePrimitive + CubeElement + PartialEq,
    D: CubePrimitive + CubeElement + PartialEq,
{
    fn apply(lhs: (A, C, D), rhs: (A, C, D)) -> bool {
        lhs.0 == rhs.0 && lhs.1 == rhs.1 && lhs.2 == rhs.2
    }
}

/// Built-in lexicographical less-than predicate.
pub struct Less;

#[cube]
impl<R, T> BinaryPredicateOp<R, (T,)> for Less
where
    R: cubecl::prelude::Runtime,
    T: CubePrimitive + CubeElement + PartialOrd,
{
    fn apply(lhs: (T,), rhs: (T,)) -> bool {
        lhs.0 < rhs.0
    }
}

#[cube]
impl<R, A, C> BinaryPredicateOp<R, (A, C)> for Less
where
    R: cubecl::prelude::Runtime,
    A: CubePrimitive + CubeElement + PartialOrd,
    C: CubePrimitive + CubeElement + PartialOrd,
{
    fn apply(lhs: (A, C), rhs: (A, C)) -> bool {
        lhs.0 < rhs.0 || (lhs.0 == rhs.0 && lhs.1 < rhs.1)
    }
}

#[cube]
impl<R, A, C, D> BinaryPredicateOp<R, (A, C, D)> for Less
where
    R: cubecl::prelude::Runtime,
    A: CubePrimitive + CubeElement + PartialOrd,
    C: CubePrimitive + CubeElement + PartialOrd,
    D: CubePrimitive + CubeElement + PartialOrd,
{
    fn apply(lhs: (A, C, D), rhs: (A, C, D)) -> bool {
        lhs.0 < rhs.0
            || (lhs.0 == rhs.0 && lhs.1 < rhs.1)
            || (lhs.0 == rhs.0 && lhs.1 == rhs.1 && lhs.2 < rhs.2)
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
        type Env: LaunchArg + Copy + CubeType<ExpandType: Clone>;
        type Output: CubeType;

        fn apply(env: Self::Env, input: Input) -> Self::Output;
    }

    #[cube]
    pub trait BinaryOp<T: CubeType>: 'static + Send + Sync {
        fn apply(lhs: T, rhs: T) -> T;
    }

    #[cube]
    pub trait PredicateOp<T: CubeType>: 'static + Send + Sync {
        type Env: LaunchArg + Copy + CubeType<ExpandType: Clone>;

        fn apply(env: Self::Env, input: T) -> bool;
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
