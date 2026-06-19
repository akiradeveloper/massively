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
/// impl massively::op::UnaryOp<(f32,)> for AddOne {
///     type Output = (f32,);
///
///     fn apply(input: (f32,)) -> (f32,) {
///         (input.0 + 1.0,)
///     }
/// }
/// ```
#[cube]
pub trait UnaryOp<Input: CubeType>: 'static + Send + Sync {
    /// Output value produced for one logical input element.
    type Output: CubeType;

    /// Maps one logical input element.
    fn apply(input: Input) -> Self::Output;
}

/// Compile-time binary operator used by reductions and scans.
///
/// ```no_run
/// use cubecl::prelude::*;
///
/// struct Sum;
///
/// #[cubecl::cube]
/// impl massively::op::BinaryOp<f32> for Sum {
///     fn apply(lhs: f32, rhs: f32) -> f32 {
///         lhs + rhs
///     }
/// }
/// ```
#[cube]
pub trait BinaryOp<T: CubeType>: 'static + Send + Sync {
    /// Combines two values.
    fn apply(lhs: T, rhs: T) -> T;
}

/// Compile-time predicate used by conditional algorithms such as
/// [`copy_if`](crate::copy_if), [`remove_if`](crate::remove_if), and
/// [`count_if`](crate::count_if).
///
/// ```no_run
/// use cubecl::prelude::*;
///
/// struct Positive;
///
/// #[cubecl::cube]
/// impl massively::op::PredicateOp<f32> for Positive {
///     fn apply(input: f32) -> bool {
///         input > 0.0
///     }
/// }
/// ```
#[cube]
pub trait PredicateOp<T: CubeType>: 'static + Send + Sync {
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
/// impl massively::op::BinaryPredicateOp<f32> for Less {
///     fn apply(lhs: f32, rhs: f32) -> bool {
///         lhs < rhs
///     }
/// }
/// ```
#[cube]
pub trait BinaryPredicateOp<T: CubeType>: 'static + Send + Sync {
    /// Returns whether the pair matches.
    fn apply(lhs: T, rhs: T) -> bool;
}

/// Built-in equality predicate for algorithms whose Rust API does not take an
/// explicit key comparator.
pub struct Equal;

#[cube]
impl<T> BinaryPredicateOp<T> for Equal
where
    T: CubePrimitive + PartialEq,
{
    fn apply(lhs: T, rhs: T) -> bool {
        lhs == rhs
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
