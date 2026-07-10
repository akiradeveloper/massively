//! Storage arity and write-boundary normalization.

use cubecl::prelude::*;

/// A physical storage arity supported by public storage values.
pub trait StorageArity: private::Sealed + 'static {}

macro_rules! define_arities {
    ($($arity:ident),+ $(,)?) => {
        $(
            #[doc = concat!("Storage arity marker `", stringify!($arity), "`.")]
            #[derive(Clone, Copy, Debug, Eq, PartialEq)]
            pub struct $arity;

            impl StorageArity for $arity {}
            impl private::Sealed for $arity {}
        )+
    };
}

define_arities!(S1, S2, S3, S4, S5, S6, S7);

/// Type-level addition for storage arities whose sum is at most seven.
#[doc(hidden)]
pub trait AddStorageArity<Rhs: StorageArity>: StorageArity {
    type Output: StorageArity;
}

macro_rules! impl_add_arity {
    ($lhs:ty, $rhs:ty => $output:ty) => {
        impl AddStorageArity<$rhs> for $lhs {
            type Output = $output;
        }
    };
}

impl_add_arity!(S1, S1 => S2);
impl_add_arity!(S1, S2 => S3);
impl_add_arity!(S1, S3 => S4);
impl_add_arity!(S1, S4 => S5);
impl_add_arity!(S1, S5 => S6);
impl_add_arity!(S1, S6 => S7);
impl_add_arity!(S2, S1 => S3);
impl_add_arity!(S2, S2 => S4);
impl_add_arity!(S2, S3 => S5);
impl_add_arity!(S2, S4 => S6);
impl_add_arity!(S2, S5 => S7);
impl_add_arity!(S3, S1 => S4);
impl_add_arity!(S3, S2 => S5);
impl_add_arity!(S3, S3 => S6);
impl_add_arity!(S3, S4 => S7);
impl_add_arity!(S4, S1 => S5);
impl_add_arity!(S4, S2 => S6);
impl_add_arity!(S4, S3 => S7);
impl_add_arity!(S5, S1 => S6);
impl_add_arity!(S5, S2 => S7);
impl_add_arity!(S6, S1 => S7);

/// The final leaf in a non-empty, ordered storage-leaf list.
#[doc(hidden)]
#[derive(CubeType, Clone, Copy, Debug, Eq, PartialEq)]
pub struct Last<T: CubeType> {
    pub(crate) value: T,
}

#[cubecl::cube]
impl<T: CubeType> Last<T> {
    #[allow(dead_code)]
    fn new(value: T) -> Self {
        Last::<T> { value }
    }
}

/// A non-final leaf in a non-empty, ordered storage-leaf list.
#[doc(hidden)]
#[derive(CubeType, Clone, Copy, Debug, Eq, PartialEq)]
pub struct More<Head: CubeType, Tail: CubeType> {
    pub(crate) head: Head,
    pub(crate) tail: Tail,
}

#[cubecl::cube]
impl<Head: CubeType, Tail: CubeType> More<Head, Tail> {
    #[allow(dead_code)]
    fn new(head: Head, tail: Tail) -> Self {
        More::<Head, Tail> { head, tail }
    }
}

/// Concatenates two non-empty leaf lists and can split the result again.
#[doc(hidden)]
#[cubecl::cube]
pub trait Concat<Rhs: CubeType>: CubeType + Sized {
    type Output: CubeType;

    fn concat(self, rhs: Rhs) -> Self::Output;
    fn split(output: Self::Output) -> (Self, Rhs);
}

#[cubecl::cube]
impl<Head: CubeType, Rhs: CubeType> Concat<Rhs> for Last<Head> {
    type Output = More<Head, Rhs>;

    fn concat(self, rhs: Rhs) -> Self::Output {
        More::<Head, Rhs> {
            head: self.value,
            tail: rhs,
        }
    }

    fn split(output: Self::Output) -> (Self, Rhs) {
        (Last::<Head> { value: output.head }, output.tail)
    }
}

#[cubecl::cube]
impl<Head: CubeType, Tail: CubeType, Rhs: CubeType> Concat<Rhs> for More<Head, Tail>
where
    Tail: Concat<Rhs>,
{
    type Output = More<Head, Tail::Output>;

    fn concat(self, rhs: Rhs) -> Self::Output {
        More::<Head, Tail::Output> {
            head: self.head,
            tail: self.tail.concat(rhs),
        }
    }

    fn split(output: Self::Output) -> (Self, Rhs) {
        let (tail, rhs) = Tail::split(output.tail);
        (
            More::<Head, Tail> {
                head: output.head,
                tail,
            },
            rhs,
        )
    }
}

/// Describes the ordered physical leaves of a semantic item shape.
///
/// Tuple nesting is erased only by [`into_storage_leaves`](Self::into_storage_leaves)
/// and restored by [`from_storage_leaves`](Self::from_storage_leaves).  The
/// associated leaf-list type records both leaf order and leaf types.
pub trait StorageLayout: CubeType + Sized + Send + Sync + 'static {
    type StorageArity: StorageArity;
    type StorageLeaves: CubeType + SelectLeaves;
    type DeviceLayout: Decompose<Self, Leaves = Self::StorageLeaves>
        + Recompose<Self, Leaves = Self::StorageLeaves>;

    fn into_storage_leaves(self) -> Self::StorageLeaves;
    fn from_storage_leaves(leaves: Self::StorageLeaves) -> Self;
}

macro_rules! impl_scalar_layout {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl StorageLayout for $ty {
                type StorageArity = S1;
                type StorageLeaves = Last<Self>;
                type DeviceLayout = ScalarLayout<Self>;

                fn into_storage_leaves(self) -> Self::StorageLeaves {
                    Last::<$ty> { value: self }
                }

                fn from_storage_leaves(leaves: Self::StorageLeaves) -> Self {
                    leaves.value
                }
            }
        )+
    };
}

impl_scalar_layout!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

impl<Left, Right> StorageLayout for (Left, Right)
where
    Left: StorageLayout + 'static,
    Right: StorageLayout + 'static,
    Left::StorageArity: AddStorageArity<Right::StorageArity>,
    Left::StorageLeaves: Concat<Right::StorageLeaves>,
    <Left::StorageLeaves as Concat<Right::StorageLeaves>>::Output: SelectLeaves,
{
    type StorageArity = <Left::StorageArity as AddStorageArity<Right::StorageArity>>::Output;
    type StorageLeaves = <Left::StorageLeaves as Concat<Right::StorageLeaves>>::Output;
    type DeviceLayout = PairLayout<Left::DeviceLayout, Right::DeviceLayout>;

    fn into_storage_leaves(self) -> Self::StorageLeaves {
        self.0
            .into_storage_leaves()
            .concat(self.1.into_storage_leaves())
    }

    fn from_storage_leaves(leaves: Self::StorageLeaves) -> Self {
        let (left, right) = <Left::StorageLeaves as Concat<Right::StorageLeaves>>::split(leaves);
        (
            Left::from_storage_leaves(left),
            Right::from_storage_leaves(right),
        )
    }
}

/// Converts a semantic value when it crosses an output write boundary.
///
/// This trait belongs to the output item.  An implementation exists only when
/// source and output have the same storage arity and exactly the same ordered
/// storage-leaf types.  Their tuple association may differ.
///
/// Algorithms apply this conversion automatically. In this example the input item is
/// `(u32, (u32, u32))`, while [`crate::zip3`] exposes the output item in canonical
/// left-associated form `((u32, u32), u32)`:
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, op::Identity, transform, zip2, zip3};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let a = exec.to_device(&[1_u32, 2]);
/// let b = exec.to_device(&[10_u32, 20]);
/// let c = exec.to_device(&[100_u32, 200]);
/// let out_a = exec.alloc::<u32>(2);
/// let out_b = exec.alloc::<u32>(2);
/// let out_c = exec.alloc::<u32>(2);
///
/// let input = zip2(a.slice(..), zip2(b.slice(..), c.slice(..)));
/// let output = zip3(
///     out_a.slice_mut(..),
///     out_b.slice_mut(..),
///     out_c.slice_mut(..),
/// );
/// transform(&exec, input, Identity, output).unwrap();
///
/// assert_eq!(exec.to_host(&out_a).unwrap(), vec![1, 2]);
/// assert_eq!(exec.to_host(&out_b).unwrap(), vec![10, 20]);
/// assert_eq!(exec.to_host(&out_c).unwrap(), vec![100, 200]);
/// ```
#[cubecl::cube]
pub trait WriteFrom<Source: CubeType>: CubeType {
    fn write_from(source: Source) -> Self;
}

#[cubecl::cube]
#[doc(hidden)]
impl<Source, Output> WriteFrom<Source> for Output
where
    Source: StorageLayout,
    Output:
        StorageLayout<StorageArity = Source::StorageArity, StorageLeaves = Source::StorageLeaves>,
    Source::DeviceLayout: Decompose<Source, Leaves = Source::StorageLeaves>,
    Output::DeviceLayout: Recompose<Output, Leaves = Source::StorageLeaves>,
{
    fn write_from(source: Source) -> Output {
        let leaves = Source::DeviceLayout::decompose(source);
        Output::DeviceLayout::recompose(leaves)
    }
}

/// Device-side layout marker for one physical leaf.
#[doc(hidden)]
pub struct ScalarLayout<T> {
    _marker: core::marker::PhantomData<fn() -> T>,
}

/// Device-side layout marker for a semantic pair.
#[doc(hidden)]
pub struct PairLayout<Left, Right> {
    _marker: core::marker::PhantomData<fn() -> (Left, Right)>,
}

/// Decomposes a semantic value into its ordered physical leaves in device code.
#[doc(hidden)]
#[cubecl::cube]
pub trait Decompose<Item: CubeType>: 'static + Send + Sync {
    type Leaves: CubeType;
    fn decompose(item: Item) -> Self::Leaves;
}

#[doc(hidden)]
#[cubecl::cube]
pub trait Recompose<Item: CubeType>: 'static + Send + Sync {
    type Leaves: CubeType;
    fn recompose(leaves: Self::Leaves) -> Item;
}

/// Selects one of two physical leaf lists without assigning a composite
/// semantic value in a device branch.
#[doc(hidden)]
#[cubecl::cube]
pub trait SelectLeaves: CubeType + Sized {
    fn select(condition: bool, if_true: Self, if_false: Self) -> Self;
}

#[cubecl::cube]
impl<T: CubePrimitive> SelectLeaves for Last<T> {
    fn select(condition: bool, if_true: Self, if_false: Self) -> Self {
        Last::<T> {
            value: if condition {
                if_true.value
            } else {
                if_false.value
            },
        }
    }
}

#[cubecl::cube]
impl<Head, Tail> SelectLeaves for More<Head, Tail>
where
    Head: CubePrimitive,
    Tail: SelectLeaves,
{
    fn select(condition: bool, if_true: Self, if_false: Self) -> Self {
        More::<Head, Tail> {
            head: if condition {
                if_true.head
            } else {
                if_false.head
            },
            tail: Tail::select(condition, if_true.tail, if_false.tail),
        }
    }
}

#[cubecl::cube]
impl<T: CubeType + 'static> Decompose<T> for ScalarLayout<T> {
    type Leaves = Last<T>;

    fn decompose(item: T) -> Self::Leaves {
        Last::<T> { value: item }
    }
}

#[cubecl::cube]
impl<T: CubeType + 'static> Recompose<T> for ScalarLayout<T> {
    type Leaves = Last<T>;
    fn recompose(leaves: Self::Leaves) -> T {
        leaves.value
    }
}

#[cubecl::cube]
impl<LeftItem, RightItem, Left, Right> Decompose<(LeftItem, RightItem)> for PairLayout<Left, Right>
where
    LeftItem: CubeType + 'static,
    RightItem: CubeType + 'static,
    Left: Decompose<LeftItem>,
    Right: Decompose<RightItem>,
    Left::Leaves: Concat<Right::Leaves>,
{
    type Leaves = <Left::Leaves as Concat<Right::Leaves>>::Output;

    fn decompose(item: (LeftItem, RightItem)) -> Self::Leaves {
        Left::decompose(item.0).concat(Right::decompose(item.1))
    }
}

#[cubecl::cube]
impl<LeftItem, RightItem, Left, Right> Recompose<(LeftItem, RightItem)> for PairLayout<Left, Right>
where
    LeftItem: CubeType + 'static,
    RightItem: CubeType + 'static,
    Left: Recompose<LeftItem>,
    Right: Recompose<RightItem>,
    Left::Leaves: Concat<Right::Leaves>,
{
    type Leaves = <Left::Leaves as Concat<Right::Leaves>>::Output;

    fn recompose(leaves: Self::Leaves) -> (LeftItem, RightItem) {
        let (left, right) = Left::Leaves::split(leaves);
        (Left::recompose(left), Right::recompose(right))
    }
}

macro_rules! define_store_leaves {
    (
        $trait_name:ident;
        $leaves:ty;
        $( $leaf:ident : $out:ident ),+;
        $this:ident, $offsets:ident, $index:ident;
        $body:block
    ) => {
        #[doc(hidden)]
        #[cubecl::cube]
        #[allow(unused_mut)]
        pub trait $trait_name<$( $leaf: CubePrimitive ),+>: CubeType {
            fn store(
                self,
                $( $out: &mut [$leaf], )+
                offsets: &[u32],
                index: usize,
            );
        }

        #[cubecl::cube]
        #[allow(unused_mut)]
        impl<$( $leaf: CubePrimitive ),+> $trait_name<$( $leaf ),+> for $leaves {
            fn store(
                self,
                $( $out: &mut [$leaf], )+
                $offsets: &[u32],
                $index: usize,
            ) {
                let $this = self;
                $body
            }
        }
    };
}

define_store_leaves!(StoreLeaves1; Last<L0>; L0: out0; this, offsets, index; {
    out0[offsets[0] as usize + index] = this.value;
});
define_store_leaves!(StoreLeaves2; More<L0, Last<L1>>; L0: out0, L1: out1; this, offsets, index; {
    out0[offsets[0] as usize + index] = this.head;
    out1[offsets[1] as usize + index] = this.tail.value;
});
define_store_leaves!(StoreLeaves3; More<L0, More<L1, Last<L2>>>; L0: out0, L1: out1, L2: out2; this, offsets, index; {
    out0[offsets[0] as usize + index] = this.head;
    out1[offsets[1] as usize + index] = this.tail.head;
    out2[offsets[2] as usize + index] = this.tail.tail.value;
});
define_store_leaves!(StoreLeaves4; More<L0, More<L1, More<L2, Last<L3>>>>; L0: out0, L1: out1, L2: out2, L3: out3; this, offsets, index; {
    out0[offsets[0] as usize + index] = this.head;
    out1[offsets[1] as usize + index] = this.tail.head;
    out2[offsets[2] as usize + index] = this.tail.tail.head;
    out3[offsets[3] as usize + index] = this.tail.tail.tail.value;
});
define_store_leaves!(StoreLeaves5; More<L0, More<L1, More<L2, More<L3, Last<L4>>>>>; L0: out0, L1: out1, L2: out2, L3: out3, L4: out4; this, offsets, index; {
    out0[offsets[0] as usize + index] = this.head;
    out1[offsets[1] as usize + index] = this.tail.head;
    out2[offsets[2] as usize + index] = this.tail.tail.head;
    out3[offsets[3] as usize + index] = this.tail.tail.tail.head;
    out4[offsets[4] as usize + index] = this.tail.tail.tail.tail.value;
});
define_store_leaves!(StoreLeaves6; More<L0, More<L1, More<L2, More<L3, More<L4, Last<L5>>>>>>; L0: out0, L1: out1, L2: out2, L3: out3, L4: out4, L5: out5; this, offsets, index; {
    out0[offsets[0] as usize + index] = this.head;
    out1[offsets[1] as usize + index] = this.tail.head;
    out2[offsets[2] as usize + index] = this.tail.tail.head;
    out3[offsets[3] as usize + index] = this.tail.tail.tail.head;
    out4[offsets[4] as usize + index] = this.tail.tail.tail.tail.head;
    out5[offsets[5] as usize + index] = this.tail.tail.tail.tail.tail.value;
});
define_store_leaves!(StoreLeaves7; More<L0, More<L1, More<L2, More<L3, More<L4, More<L5, Last<L6>>>>>>>; L0: out0, L1: out1, L2: out2, L3: out3, L4: out4, L5: out5, L6: out6; this, offsets, index; {
    out0[offsets[0] as usize + index] = this.head;
    out1[offsets[1] as usize + index] = this.tail.head;
    out2[offsets[2] as usize + index] = this.tail.tail.head;
    out3[offsets[3] as usize + index] = this.tail.tail.tail.head;
    out4[offsets[4] as usize + index] = this.tail.tail.tail.tail.head;
    out5[offsets[5] as usize + index] = this.tail.tail.tail.tail.tail.head;
    out6[offsets[6] as usize + index] = this.tail.tail.tail.tail.tail.tail.value;
});

macro_rules! define_select_store_leaves {
    (
        $trait_name:ident;
        $leaves:ty;
        $( $leaf:ident : $out:ident : $offset_index:literal : $unselected:ident : $selected:expr ),+;
        $this:ident, $select:ident, $offsets:ident, $index:ident
    ) => {
        /// Stores a conditional composite value one primitive leaf at a time.
        /// CubeCL cannot conditionally assign an arbitrary composite
        /// `CubeType`, but each physical leaf is a `CubePrimitive`.
        #[doc(hidden)]
        #[cubecl::cube]
        #[allow(unused_mut)]
        pub trait $trait_name<$( $leaf: CubePrimitive ),+>: CubeType {
            fn select_store(
                self,
                select: u32,
                $( $unselected: $leaf, )+
                $( $out: &mut [$leaf], )+
                offsets: &[u32],
                index: usize,
            );
        }

        #[cubecl::cube]
        #[allow(unused_mut)]
        impl<$( $leaf: CubePrimitive ),+> $trait_name<$( $leaf ),+> for $leaves {
            fn select_store(
                self,
                $select: u32,
                $( $unselected: $leaf, )+
                $( $out: &mut [$leaf], )+
                $offsets: &[u32],
                $index: usize,
            ) {
                let $this = self;
                $(
                    $out[$offsets[$offset_index] as usize + $index] =
                        if $select != 0u32 { $selected } else { $unselected };
                )+
            }
        }
    };
}

define_select_store_leaves!(SelectStoreLeaves2;
    More<L0, Last<L1>>;
    L0:out0:0:unselected0:selected.head,
    L1:out1:1:unselected1:selected.tail.value;
    selected, select, offsets, index
);
define_select_store_leaves!(SelectStoreLeaves3;
    More<L0, More<L1, Last<L2>>>;
    L0:out0:0:unselected0:selected.head,
    L1:out1:1:unselected1:selected.tail.head,
    L2:out2:2:unselected2:selected.tail.tail.value;
    selected, select, offsets, index
);
define_select_store_leaves!(SelectStoreLeaves4;
    More<L0, More<L1, More<L2, Last<L3>>>>;
    L0:out0:0:unselected0:selected.head,
    L1:out1:1:unselected1:selected.tail.head,
    L2:out2:2:unselected2:selected.tail.tail.head,
    L3:out3:3:unselected3:selected.tail.tail.tail.value;
    selected, select, offsets, index
);
define_select_store_leaves!(SelectStoreLeaves5;
    More<L0, More<L1, More<L2, More<L3, Last<L4>>>>>;
    L0:out0:0:unselected0:selected.head,
    L1:out1:1:unselected1:selected.tail.head,
    L2:out2:2:unselected2:selected.tail.tail.head,
    L3:out3:3:unselected3:selected.tail.tail.tail.head,
    L4:out4:4:unselected4:selected.tail.tail.tail.tail.value;
    selected, select, offsets, index
);
define_select_store_leaves!(SelectStoreLeaves6;
    More<L0, More<L1, More<L2, More<L3, More<L4, Last<L5>>>>>>;
    L0:out0:0:unselected0:selected.head,
    L1:out1:1:unselected1:selected.tail.head,
    L2:out2:2:unselected2:selected.tail.tail.head,
    L3:out3:3:unselected3:selected.tail.tail.tail.head,
    L4:out4:4:unselected4:selected.tail.tail.tail.tail.head,
    L5:out5:5:unselected5:selected.tail.tail.tail.tail.tail.value;
    selected, select, offsets, index
);
define_select_store_leaves!(SelectStoreLeaves7;
    More<L0, More<L1, More<L2, More<L3, More<L4, More<L5, Last<L6>>>>>>>;
    L0:out0:0:unselected0:selected.head,
    L1:out1:1:unselected1:selected.tail.head,
    L2:out2:2:unselected2:selected.tail.tail.head,
    L3:out3:3:unselected3:selected.tail.tail.tail.head,
    L4:out4:4:unselected4:selected.tail.tail.tail.tail.head,
    L5:out5:5:unselected5:selected.tail.tail.tail.tail.tail.head,
    L6:out6:6:unselected6:selected.tail.tail.tail.tail.tail.tail.value;
    selected, select, offsets, index
);

macro_rules! define_load_leaves {
    ($trait_name:ident; $leaves:ty; $( $leaf:ident:$input:ident ),+; $offsets:ident,$index:ident; $body:expr) => {
        #[doc(hidden)]
        #[cubecl::cube]
        pub trait $trait_name<$( $leaf: CubePrimitive ),+>: CubeType {
            fn load($( $input: &[$leaf], )+ offsets: &[u32], index: usize) -> Self;
        }

        #[cubecl::cube]
        impl<$( $leaf: CubePrimitive ),+> $trait_name<$( $leaf ),+> for $leaves {
            fn load($( $input: &[$leaf], )+ $offsets: &[u32], $index: usize) -> Self {
                $body
            }
        }
    };
}

define_load_leaves!(LoadLeaves2; More<L0,Last<L1>>; L0:in0,L1:in1; offsets,index;
    More::new(in0[offsets[0] as usize + index], Last::new(in1[offsets[1] as usize + index]))
);
define_load_leaves!(LoadLeaves3; More<L0,More<L1,Last<L2>>>; L0:in0,L1:in1,L2:in2; offsets,index;
    More::new(in0[offsets[0] as usize + index], More::new(in1[offsets[1] as usize + index], Last::new(in2[offsets[2] as usize + index])))
);
define_load_leaves!(LoadLeaves4; More<L0,More<L1,More<L2,Last<L3>>>>; L0:in0,L1:in1,L2:in2,L3:in3; offsets,index;
    More::new(in0[offsets[0] as usize + index], More::new(in1[offsets[1] as usize + index], More::new(in2[offsets[2] as usize + index], Last::new(in3[offsets[3] as usize + index]))))
);
define_load_leaves!(LoadLeaves5; More<L0,More<L1,More<L2,More<L3,Last<L4>>>>>; L0:in0,L1:in1,L2:in2,L3:in3,L4:in4; offsets,index;
    More::new(in0[offsets[0] as usize + index], More::new(in1[offsets[1] as usize + index], More::new(in2[offsets[2] as usize + index], More::new(in3[offsets[3] as usize + index], Last::new(in4[offsets[4] as usize + index])))))
);
define_load_leaves!(LoadLeaves6; More<L0,More<L1,More<L2,More<L3,More<L4,Last<L5>>>>>>; L0:in0,L1:in1,L2:in2,L3:in3,L4:in4,L5:in5; offsets,index;
    More::new(in0[offsets[0] as usize + index], More::new(in1[offsets[1] as usize + index], More::new(in2[offsets[2] as usize + index], More::new(in3[offsets[3] as usize + index], More::new(in4[offsets[4] as usize + index], Last::new(in5[offsets[5] as usize + index]))))))
);
define_load_leaves!(LoadLeaves7; More<L0,More<L1,More<L2,More<L3,More<L4,More<L5,Last<L6>>>>>>>; L0:in0,L1:in1,L2:in2,L3:in3,L4:in4,L5:in5,L6:in6; offsets,index;
    More::new(in0[offsets[0] as usize + index], More::new(in1[offsets[1] as usize + index], More::new(in2[offsets[2] as usize + index], More::new(in3[offsets[3] as usize + index], More::new(in4[offsets[4] as usize + index], More::new(in5[offsets[5] as usize + index], Last::new(in6[offsets[6] as usize + index])))))))
);

macro_rules! define_load_mut_leaves {
    ($trait_name:ident; $leaves:ty; $( $leaf:ident:$input:ident ),+; $offsets:ident,$index:ident; $body:expr) => {
        #[doc(hidden)]
        #[cubecl::cube]
        pub trait $trait_name<$( $leaf: CubePrimitive ),+>: CubeType {
            fn load_mut($( $input: &mut [$leaf], )+ offsets: &[u32], index: usize) -> Self;
        }

        #[cubecl::cube]
        impl<$( $leaf: CubePrimitive ),+> $trait_name<$( $leaf ),+> for $leaves {
            fn load_mut($( $input: &mut [$leaf], )+ $offsets: &[u32], $index: usize) -> Self {
                $body
            }
        }
    };
}

define_load_mut_leaves!(LoadMutLeaves2; More<L0,Last<L1>>; L0:in0,L1:in1; offsets,index;
    More::new(in0[offsets[0] as usize + index], Last::new(in1[offsets[1] as usize + index]))
);
define_load_mut_leaves!(LoadMutLeaves3; More<L0,More<L1,Last<L2>>>; L0:in0,L1:in1,L2:in2; offsets,index;
    More::new(in0[offsets[0] as usize + index], More::new(in1[offsets[1] as usize + index], Last::new(in2[offsets[2] as usize + index])))
);
define_load_mut_leaves!(LoadMutLeaves4; More<L0,More<L1,More<L2,Last<L3>>>>; L0:in0,L1:in1,L2:in2,L3:in3; offsets,index;
    More::new(in0[offsets[0] as usize + index], More::new(in1[offsets[1] as usize + index], More::new(in2[offsets[2] as usize + index], Last::new(in3[offsets[3] as usize + index]))))
);
define_load_mut_leaves!(LoadMutLeaves5; More<L0,More<L1,More<L2,More<L3,Last<L4>>>>>; L0:in0,L1:in1,L2:in2,L3:in3,L4:in4; offsets,index;
    More::new(in0[offsets[0] as usize + index], More::new(in1[offsets[1] as usize + index], More::new(in2[offsets[2] as usize + index], More::new(in3[offsets[3] as usize + index], Last::new(in4[offsets[4] as usize + index])))))
);
define_load_mut_leaves!(LoadMutLeaves6; More<L0,More<L1,More<L2,More<L3,More<L4,Last<L5>>>>>>; L0:in0,L1:in1,L2:in2,L3:in3,L4:in4,L5:in5; offsets,index;
    More::new(in0[offsets[0] as usize + index], More::new(in1[offsets[1] as usize + index], More::new(in2[offsets[2] as usize + index], More::new(in3[offsets[3] as usize + index], More::new(in4[offsets[4] as usize + index], Last::new(in5[offsets[5] as usize + index]))))))
);
define_load_mut_leaves!(LoadMutLeaves7; More<L0,More<L1,More<L2,More<L3,More<L4,More<L5,Last<L6>>>>>>>; L0:in0,L1:in1,L2:in2,L3:in3,L4:in4,L5:in5,L6:in6; offsets,index;
    More::new(in0[offsets[0] as usize + index], More::new(in1[offsets[1] as usize + index], More::new(in2[offsets[2] as usize + index], More::new(in3[offsets[3] as usize + index], More::new(in4[offsets[4] as usize + index], More::new(in5[offsets[5] as usize + index], Last::new(in6[offsets[6] as usize + index])))))))
);

mod private {
    pub trait Sealed {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use static_assertions::{assert_impl_all, assert_not_impl_any};

    type RightAssociated3 = (u32, (f32, i16));
    type LeftAssociated3 = ((u32, f32), i16);
    type WrongOrder3 = ((u32, i16), f32);
    type WrongArity2 = (u32, f32);
    type Unsupported8 = (u8, (u8, (u8, (u8, (u8, (u8, (u8, u8)))))));

    #[cubecl::cube]
    #[allow(dead_code)]
    fn cubecl_accepts_nested_semantic_items(source: (u32, (f32, i16))) -> ((u32, f32), i16) {
        ((source.0, source.1.0), source.1.1)
    }

    #[cubecl::cube]
    #[allow(dead_code)]
    fn cubecl_accepts_recursive_leaf_concat(
        a: u32,
        b: f32,
        c: i16,
    ) -> More<u32, More<f32, Last<i16>>> {
        let left = Last::new(a);
        let right = More::new(b, Last::new(c));
        left.concat(right)
    }

    #[test]
    fn write_from_reassociates_without_changing_semantic_leaf_values() {
        let source: RightAssociated3 = (7, (2.5, -3));
        let output = LeftAssociated3::write_from(source);
        assert_eq!(output, ((7, 2.5), -3));
    }

    #[test]
    fn deeply_nested_layout_supports_seven_leaves() {
        type Source = (u8, (u16, (u32, (u64, (i8, (i16, i32))))));
        type Output = ((((((u8, u16), u32), u64), i8), i16), i32);

        let source: Source = (1, (2, (3, (4, (5, (6, 7))))));
        let output = Output::write_from(source);
        assert_eq!(output, ((((((1, 2), 3), 4), 5), 6), 7));
    }

    #[test]
    fn layout_arity_is_derived_from_the_binary_tree() {
        fn assert_arity<T, A>()
        where
            T: StorageLayout<StorageArity = A>,
            A: StorageArity,
        {
        }

        assert_arity::<u32, S1>();
        assert_arity::<RightAssociated3, S3>();
        assert_arity::<((u8, u16), (u32, f32)), S4>();
    }

    assert_impl_all!(LeftAssociated3: WriteFrom<RightAssociated3>);
    assert_not_impl_any!(WrongOrder3: WriteFrom<RightAssociated3>);
    assert_not_impl_any!(WrongArity2: WriteFrom<RightAssociated3>);
    assert_not_impl_any!(Unsupported8: StorageLayout);
}
