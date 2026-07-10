//! Canonical tuple values used by `zipN` iterators.
//!
//! The nested binary representation is an implementation detail for most users. Use the `tupleN`
//! constructors and `flattenN` helpers to work with a flat sequence of leaves.

/// Public two-element tuple value.
pub type Tuple2<A, B> = (A, B);

/// Public three-element tuple value.
pub type Tuple3<A, B, C> = (Tuple2<A, B>, C);

/// Public four-element tuple value.
pub type Tuple4<A, B, C, D> = (Tuple3<A, B, C>, D);

/// Public five-element tuple value.
pub type Tuple5<A, B, C, D, E> = (Tuple4<A, B, C, D>, E);

/// Public six-element tuple value.
pub type Tuple6<A, B, C, D, E, F> = (Tuple5<A, B, C, D, E>, F);

/// Public seven-element tuple value.
pub type Tuple7<A, B, C, D, E, F, G> = (Tuple6<A, B, C, D, E, F>, G);

/// Constructs a two-element tuple value.
///
/// # Examples
///
/// ```
/// use massively::tuple2;
///
/// let (first, second) = tuple2(1, 2);
/// assert_eq!((first, second), (1, 2));
/// ```
pub fn tuple2<A, B>(a: A, b: B) -> Tuple2<A, B> {
    (a, b)
}

// CubeCL rewrites calls inside `#[cube]` code to `tupleN::expand`. Defining
// these expanders in terms of `NativeExpand<T>` keeps `tupleN(a, b, ...)`
// inferable; a generated generic cube function would require a turbofish at
// every call site. They must be public for expansion in downstream crates, but
// are not part of the documented massively API.
#[doc(hidden)]
pub mod tuple2 {
    use cubecl::{frontend::NativeExpand, prelude::*};

    pub fn expand<A: CubePrimitive, B: CubePrimitive>(
        _scope: &Scope,
        a: NativeExpand<A>,
        b: NativeExpand<B>,
    ) -> (NativeExpand<A>, NativeExpand<B>) {
        (a, b)
    }
}

/// Constructs a three-element tuple value.
///
/// Use [`flatten3`] to destructure the canonical nested representation.
///
/// # Examples
///
/// ```
/// use massively::{flatten3, tuple3};
///
/// assert_eq!(flatten3(tuple3(1, 2, 3)), (1, 2, 3));
/// ```
pub fn tuple3<A, B, C>(a: A, b: B, c: C) -> Tuple3<A, B, C> {
    ((a, b), c)
}

#[doc(hidden)]
pub mod tuple3 {
    use cubecl::{frontend::NativeExpand, prelude::*};

    pub fn expand<A: CubePrimitive, B: CubePrimitive, C: CubePrimitive>(
        _scope: &Scope,
        a: NativeExpand<A>,
        b: NativeExpand<B>,
        c: NativeExpand<C>,
    ) -> ((NativeExpand<A>, NativeExpand<B>), NativeExpand<C>) {
        ((a, b), c)
    }
}

/// Constructs a four-element tuple value. See [`flatten4`] for destructuring.
pub fn tuple4<A, B, C, D>(a: A, b: B, c: C, d: D) -> Tuple4<A, B, C, D> {
    (((a, b), c), d)
}

#[doc(hidden)]
pub mod tuple4 {
    use cubecl::{frontend::NativeExpand, prelude::*};

    pub fn expand<A: CubePrimitive, B: CubePrimitive, C: CubePrimitive, D: CubePrimitive>(
        _scope: &Scope,
        a: NativeExpand<A>,
        b: NativeExpand<B>,
        c: NativeExpand<C>,
        d: NativeExpand<D>,
    ) -> (
        ((NativeExpand<A>, NativeExpand<B>), NativeExpand<C>),
        NativeExpand<D>,
    ) {
        (((a, b), c), d)
    }
}

/// Constructs a five-element tuple value. See [`flatten5`] for destructuring.
pub fn tuple5<A, B, C, D, E>(a: A, b: B, c: C, d: D, e: E) -> Tuple5<A, B, C, D, E> {
    ((((a, b), c), d), e)
}

#[doc(hidden)]
pub mod tuple5 {
    use cubecl::{frontend::NativeExpand, prelude::*};

    pub fn expand<
        A: CubePrimitive,
        B: CubePrimitive,
        C: CubePrimitive,
        D: CubePrimitive,
        E: CubePrimitive,
    >(
        _scope: &Scope,
        a: NativeExpand<A>,
        b: NativeExpand<B>,
        c: NativeExpand<C>,
        d: NativeExpand<D>,
        e: NativeExpand<E>,
    ) -> (
        (
            ((NativeExpand<A>, NativeExpand<B>), NativeExpand<C>),
            NativeExpand<D>,
        ),
        NativeExpand<E>,
    ) {
        ((((a, b), c), d), e)
    }
}

/// Constructs a six-element tuple value. See [`flatten6`] for destructuring.
pub fn tuple6<A, B, C, D, E, F>(a: A, b: B, c: C, d: D, e: E, f: F) -> Tuple6<A, B, C, D, E, F> {
    (((((a, b), c), d), e), f)
}

#[doc(hidden)]
pub mod tuple6 {
    use cubecl::{frontend::NativeExpand, prelude::*};

    pub fn expand<
        A: CubePrimitive,
        B: CubePrimitive,
        C: CubePrimitive,
        D: CubePrimitive,
        E: CubePrimitive,
        F: CubePrimitive,
    >(
        _scope: &Scope,
        a: NativeExpand<A>,
        b: NativeExpand<B>,
        c: NativeExpand<C>,
        d: NativeExpand<D>,
        e: NativeExpand<E>,
        f: NativeExpand<F>,
    ) -> (
        (
            (
                ((NativeExpand<A>, NativeExpand<B>), NativeExpand<C>),
                NativeExpand<D>,
            ),
            NativeExpand<E>,
        ),
        NativeExpand<F>,
    ) {
        (((((a, b), c), d), e), f)
    }
}

/// Constructs a seven-element tuple value. See [`flatten7`] for destructuring.
#[allow(clippy::too_many_arguments)]
pub fn tuple7<A, B, C, D, E, F, G>(
    a: A,
    b: B,
    c: C,
    d: D,
    e: E,
    f: F,
    g: G,
) -> Tuple7<A, B, C, D, E, F, G> {
    ((((((a, b), c), d), e), f), g)
}

#[doc(hidden)]
pub mod tuple7 {
    use cubecl::{frontend::NativeExpand, prelude::*};

    #[allow(clippy::too_many_arguments)]
    pub fn expand<
        A: CubePrimitive,
        B: CubePrimitive,
        C: CubePrimitive,
        D: CubePrimitive,
        E: CubePrimitive,
        F: CubePrimitive,
        G: CubePrimitive,
    >(
        _scope: &Scope,
        a: NativeExpand<A>,
        b: NativeExpand<B>,
        c: NativeExpand<C>,
        d: NativeExpand<D>,
        e: NativeExpand<E>,
        f: NativeExpand<F>,
        g: NativeExpand<G>,
    ) -> (
        (
            (
                (
                    ((NativeExpand<A>, NativeExpand<B>), NativeExpand<C>),
                    NativeExpand<D>,
                ),
                NativeExpand<E>,
            ),
            NativeExpand<F>,
        ),
        NativeExpand<G>,
    ) {
        ((((((a, b), c), d), e), f), g)
    }
}

/// Flattens a public three-element tuple value into a Rust three-tuple.
///
/// # Examples
///
/// ```
/// use massively::{flatten3, tuple3};
///
/// let (a, b, c) = flatten3(tuple3("a", "b", "c"));
/// assert_eq!((a, b, c), ("a", "b", "c"));
/// ```
pub fn flatten3<A, B, C>(value: Tuple3<A, B, C>) -> (A, B, C) {
    let ((a, b), c) = value;
    (a, b, c)
}

#[doc(hidden)]
pub mod flatten3 {
    use cubecl::{frontend::NativeExpand, prelude::*};

    pub fn expand<A: CubePrimitive, B: CubePrimitive, C: CubePrimitive>(
        _scope: &Scope,
        value: ((NativeExpand<A>, NativeExpand<B>), NativeExpand<C>),
    ) -> (NativeExpand<A>, NativeExpand<B>, NativeExpand<C>) {
        let ((a, b), c) = value;
        (a, b, c)
    }
}

/// Flattens a public four-element tuple value into a Rust four-tuple.
///
/// # Examples
///
/// ```
/// use massively::{flatten4, tuple4};
///
/// assert_eq!(flatten4(tuple4(1, 2, 3, 4)), (1, 2, 3, 4));
/// ```
pub fn flatten4<A, B, C, D>(value: Tuple4<A, B, C, D>) -> (A, B, C, D) {
    let (((a, b), c), d) = value;
    (a, b, c, d)
}

#[doc(hidden)]
pub mod flatten4 {
    use cubecl::{frontend::NativeExpand, prelude::*};

    pub fn expand<A: CubePrimitive, B: CubePrimitive, C: CubePrimitive, D: CubePrimitive>(
        _scope: &Scope,
        value: (
            ((NativeExpand<A>, NativeExpand<B>), NativeExpand<C>),
            NativeExpand<D>,
        ),
    ) -> (
        NativeExpand<A>,
        NativeExpand<B>,
        NativeExpand<C>,
        NativeExpand<D>,
    ) {
        let (((a, b), c), d) = value;
        (a, b, c, d)
    }
}

/// Flattens a public five-element tuple value into a Rust five-tuple.
///
/// # Examples
///
/// ```
/// use massively::{flatten5, tuple5};
///
/// assert_eq!(flatten5(tuple5(1, 2, 3, 4, 5)), (1, 2, 3, 4, 5));
/// ```
pub fn flatten5<A, B, C, D, E>(value: Tuple5<A, B, C, D, E>) -> (A, B, C, D, E) {
    let ((((a, b), c), d), e) = value;
    (a, b, c, d, e)
}

#[doc(hidden)]
pub mod flatten5 {
    use cubecl::{frontend::NativeExpand, prelude::*};

    pub fn expand<
        A: CubePrimitive,
        B: CubePrimitive,
        C: CubePrimitive,
        D: CubePrimitive,
        E: CubePrimitive,
    >(
        _scope: &Scope,
        value: (
            (
                ((NativeExpand<A>, NativeExpand<B>), NativeExpand<C>),
                NativeExpand<D>,
            ),
            NativeExpand<E>,
        ),
    ) -> (
        NativeExpand<A>,
        NativeExpand<B>,
        NativeExpand<C>,
        NativeExpand<D>,
        NativeExpand<E>,
    ) {
        let ((((a, b), c), d), e) = value;
        (a, b, c, d, e)
    }
}

/// Flattens a public six-element tuple value into a Rust six-tuple.
///
/// # Examples
///
/// ```
/// use massively::{flatten6, tuple6};
///
/// assert_eq!(flatten6(tuple6(1, 2, 3, 4, 5, 6)), (1, 2, 3, 4, 5, 6));
/// ```
pub fn flatten6<A, B, C, D, E, F>(value: Tuple6<A, B, C, D, E, F>) -> (A, B, C, D, E, F) {
    let (((((a, b), c), d), e), f) = value;
    (a, b, c, d, e, f)
}

#[doc(hidden)]
pub mod flatten6 {
    use cubecl::{frontend::NativeExpand, prelude::*};

    pub fn expand<
        A: CubePrimitive,
        B: CubePrimitive,
        C: CubePrimitive,
        D: CubePrimitive,
        E: CubePrimitive,
        F: CubePrimitive,
    >(
        _scope: &Scope,
        value: (
            (
                (
                    ((NativeExpand<A>, NativeExpand<B>), NativeExpand<C>),
                    NativeExpand<D>,
                ),
                NativeExpand<E>,
            ),
            NativeExpand<F>,
        ),
    ) -> (
        NativeExpand<A>,
        NativeExpand<B>,
        NativeExpand<C>,
        NativeExpand<D>,
        NativeExpand<E>,
        NativeExpand<F>,
    ) {
        let (((((a, b), c), d), e), f) = value;
        (a, b, c, d, e, f)
    }
}

/// Flattens a public seven-element tuple value into a Rust seven-tuple.
///
/// # Examples
///
/// ```
/// use massively::{flatten7, tuple7};
///
/// assert_eq!(
///     flatten7(tuple7(1, 2, 3, 4, 5, 6, 7)),
///     (1, 2, 3, 4, 5, 6, 7),
/// );
/// ```
pub fn flatten7<A, B, C, D, E, F, G>(value: Tuple7<A, B, C, D, E, F, G>) -> (A, B, C, D, E, F, G) {
    let ((((((a, b), c), d), e), f), g) = value;
    (a, b, c, d, e, f, g)
}

#[doc(hidden)]
pub mod flatten7 {
    use cubecl::{frontend::NativeExpand, prelude::*};

    pub fn expand<
        A: CubePrimitive,
        B: CubePrimitive,
        C: CubePrimitive,
        D: CubePrimitive,
        E: CubePrimitive,
        F: CubePrimitive,
        G: CubePrimitive,
    >(
        _scope: &Scope,
        value: (
            (
                (
                    (
                        ((NativeExpand<A>, NativeExpand<B>), NativeExpand<C>),
                        NativeExpand<D>,
                    ),
                    NativeExpand<E>,
                ),
                NativeExpand<F>,
            ),
            NativeExpand<G>,
        ),
    ) -> (
        NativeExpand<A>,
        NativeExpand<B>,
        NativeExpand<C>,
        NativeExpand<D>,
        NativeExpand<E>,
        NativeExpand<F>,
        NativeExpand<G>,
    ) {
        let ((((((a, b), c), d), e), f), g) = value;
        (a, b, c, d, e, f, g)
    }
}
