use cubecl::prelude::*;

/// Compile-time unary operator used by transform-style algorithms.
///
/// Implement this trait on a unit-like marker type and pass the marker value to
/// [`transform`](crate::transform).
///
/// ```no_run
/// use cubecl::prelude::*;
///
/// struct AddOne;
///
/// #[cubecl::cube]
/// impl<B> massively::op::UnaryOp<B, (f32,)> for AddOne
/// where
///     B: cubecl::prelude::Runtime,
/// {
///     type Output = (f32,);
///
///     fn apply(input: (f32,)) -> (f32,) {
///         (input.0 + 1.0,)
///     }
/// }
/// ```
#[cube]
pub trait UnaryOp<B, Input>: 'static + Send + Sync
where
    B: cubecl::prelude::Runtime,
    Input: crate::MItem<B>,
{
    /// Output value produced for one logical input element.
    type Output: crate::MItem<B>;

    /// Maps one logical input element.
    fn apply(input: Input) -> Self::Output;
}

/// Compile-time binary transform used by algorithms such as
/// [`inner_product`](crate::inner_product).
#[cube]
pub trait BinaryOp<B, X, Y>: 'static + Send + Sync
where
    B: cubecl::prelude::Runtime,
    X: crate::MItem<B>,
    Y: crate::MItem<B>,
{
    type Output: crate::MItem<B>;

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
/// impl<B> massively::op::ReductionOp<B, (f32,)> for Sum
/// where
///     B: cubecl::prelude::Runtime,
/// {
///     fn apply(lhs: (f32,), rhs: (f32,)) -> (f32,) {
///         (lhs.0 + rhs.0,)
///     }
/// }
/// ```
#[cube]
pub trait ReductionOp<B, X>: 'static + Send + Sync
where
    B: cubecl::prelude::Runtime,
    X: crate::MItem<B>,
{
    /// Combines two values.
    fn apply(lhs: X, rhs: X) -> X;
}

/// Compile-time predicate used by conditional algorithms such as
/// [`remove_if`](crate::remove_if), [`count_if`](crate::count_if), and
/// [`find_if`](crate::find_if).
///
/// Stencil algorithms such as [`copy_if`](crate::copy_if) use a `u32` flag
/// column in the public API instead of taking a predicate marker.
///
/// ```no_run
/// use cubecl::prelude::*;
///
/// struct Positive;
///
/// #[cubecl::cube]
/// impl<B> massively::op::PredicateOp<B, (f32,)> for Positive
/// where
///     B: cubecl::prelude::Runtime,
/// {
///     fn apply(input: (f32,)) -> bool {
///         input.0 > 0.0
///     }
/// }
/// ```
#[cube]
pub trait PredicateOp<B, T>: 'static + Send + Sync
where
    B: cubecl::prelude::Runtime,
    T: crate::MItem<B>,
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
/// impl<B> massively::op::BinaryPredicateOp<B, (f32,)> for Less
/// where
///     B: cubecl::prelude::Runtime,
/// {
///     fn apply(lhs: (f32,), rhs: (f32,)) -> bool {
///         lhs.0 < rhs.0
///     }
/// }
/// ```
#[cube]
pub trait BinaryPredicateOp<B, T>: 'static + Send + Sync
where
    B: cubecl::prelude::Runtime,
    T: crate::MItem<B>,
{
    /// Returns whether the pair matches.
    fn apply(lhs: T, rhs: T) -> bool;
}

/// Built-in equality predicate for algorithms whose Rust API does not take an
/// explicit key comparator.
pub struct Equal;

#[cube]
impl<B, T> BinaryPredicateOp<B, (T,)> for Equal
where
    B: cubecl::prelude::Runtime,
    T: CubePrimitive + CubeElement + PartialEq,
{
    fn apply(lhs: (T,), rhs: (T,)) -> bool {
        lhs.0 == rhs.0
    }
}

#[cube]
impl<B, A, C> BinaryPredicateOp<B, (A, C)> for Equal
where
    B: cubecl::prelude::Runtime,
    A: CubePrimitive + CubeElement + PartialEq,
    C: CubePrimitive + CubeElement + PartialEq,
{
    fn apply(lhs: (A, C), rhs: (A, C)) -> bool {
        lhs.0 == rhs.0 && lhs.1 == rhs.1
    }
}

#[cube]
impl<B, A, C, D> BinaryPredicateOp<B, (A, C, D)> for Equal
where
    B: cubecl::prelude::Runtime,
    A: CubePrimitive + CubeElement + PartialEq,
    C: CubePrimitive + CubeElement + PartialEq,
    D: CubePrimitive + CubeElement + PartialEq,
{
    fn apply(lhs: (A, C, D), rhs: (A, C, D)) -> bool {
        lhs.0 == rhs.0 && lhs.1 == rhs.1 && lhs.2 == rhs.2
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
