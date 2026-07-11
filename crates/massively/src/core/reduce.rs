//! Read-arity and storage-arity indexed reductions.

use core::marker::PhantomData;
use cubecl::prelude::*;

use crate::{
    A1, A2, A3, A4, A5, A6, A7, A8, Column, Constant, Counting, Error, Executor, MStorageElement,
    Permute, ReadExpression, ReverseCounting, S1, S2, S3, S4, S5, S6, S7, StorageLayout, Taken,
    Transform, Zip,
    eval::{Eval1, Eval2, Eval3, Eval4, Eval5, Eval6, Eval7, Eval8},
    launch::cube_count_1d,
    op::UnaryOp,
    read::{
        BindSlots, Env0, Env1, Env2, Env3, Env4, Env5, Env6, Env7, LowerReadExpression, TakenSource,
    },
    storage::{
        Decompose, Last, LoadLeaves2, LoadLeaves3, LoadLeaves4, LoadLeaves5, LoadLeaves6,
        LoadLeaves7, More, MutableLeaves, MutableLeavesExpand, PlaneShuffleLeaves, Recompose,
        ScalarLayout, StoreLeaves2, StoreLeaves2Expand, StoreLeaves3, StoreLeaves3Expand,
        StoreLeaves4, StoreLeaves4Expand, StoreLeaves5, StoreLeaves5Expand, StoreLeaves6,
        StoreLeaves6Expand, StoreLeaves7, StoreLeaves7Expand,
    },
};

const BLOCK_SIZE: u32 = 256;
const ITEMS_PER_UNIT: usize = 256;
const TILE_SIZE: usize = BLOCK_SIZE as usize * ITEMS_PER_UNIT;

/// Associative binary operation used by scans and reductions.
///
/// Scans preserve operand order. Reductions may regroup and reorder operands
/// across GPU units, so operations passed to `reduce` must also be commutative.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, vector::reduce};
/// use massively::op::ReductionOp;
///
/// struct Add;
///
/// #[cubecl::cube]
/// impl ReductionOp<u32> for Add {
///     fn apply(lhs: u32, rhs: u32) -> u32 {
///         lhs + rhs
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[1_u32, 2, 3]);
///
/// assert_eq!(reduce(&exec, input.slice(..), 0, Add).unwrap(), 6);
/// ```
#[cubecl::cube]
pub trait ReductionOp<Item: CubeType>: 'static + Send + Sync {
    fn apply(lhs: Item, rhs: Item) -> Item;
}

/// Semantic items supported by the first single-leaf reduction slice.
pub trait ReduceStorage1:
    MStorageElement + StorageLayout<StorageArity = S1> + Send + Sync + 'static
{
}

impl<T> ReduceStorage1 for T where
    T: MStorageElement + StorageLayout<StorageArity = S1> + Send + Sync + 'static
{
}

/// Dispatch key combining read arity and reduction storage arity.
#[derive(Clone, Copy, Debug, Default)]
pub struct Dispatch<Read, Storage>(PhantomData<fn() -> (Read, Storage)>);

#[cubecl::cube]
fn accumulate_register<Item, Leaves, Layout, Op>(cells: &Leaves::Cells, value: Item)
where
    Item: CubeType,
    Leaves: MutableLeaves,
    Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
    Op: ReductionOp<Item>,
{
    let accumulated = Op::apply(Layout::recompose(Leaves::read(cells)), value);
    Leaves::store(cells, Layout::decompose(accumulated));
}

#[cubecl::cube]
fn reduce_plane_full<Item, Leaves, Layout, Op>(value: Item) -> Leaves
where
    Item: CubeType,
    Leaves: MutableLeaves + PlaneShuffleLeaves,
    Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
    Op: ReductionOp<Item>,
{
    let cells = Layout::decompose(value).into_cells();
    let offset = RuntimeCell::<u32>::new(PLANE_DIM / 2u32);
    while offset.read() > 0u32 {
        let right = Leaves::shuffle_leaves_down(Leaves::read(&cells), offset.read());
        if UNIT_POS_PLANE < offset.read() {
            let combined = Op::apply(
                Layout::recompose(Leaves::read(&cells)),
                Layout::recompose(right),
            );
            Leaves::store(&cells, Layout::decompose(combined));
        }
        offset.store(offset.read() / 2u32);
    }
    Leaves::read(&cells)
}

#[cubecl::cube]
fn reduce_plane_valid<Item, Leaves, Layout, Op>(value: Item, valid: u32) -> (Leaves, u32)
where
    Item: CubeType,
    Leaves: MutableLeaves + PlaneShuffleLeaves,
    Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
    Op: ReductionOp<Item>,
{
    let cells = Layout::decompose(value).into_cells();
    let cell_valid = RuntimeCell::<u32>::new(valid);
    let offset = RuntimeCell::<u32>::new(PLANE_DIM / 2u32);
    while offset.read() > 0u32 {
        let right = Leaves::shuffle_leaves_down(Leaves::read(&cells), offset.read());
        let right_cells = right.into_cells();
        let right_valid = plane_shuffle_down(cell_valid.read(), offset.read());
        if UNIT_POS_PLANE < offset.read() && right_valid != 0u32 {
            if cell_valid.read() != 0u32 {
                let combined = Op::apply(
                    Layout::recompose(Leaves::read(&cells)),
                    Layout::recompose(Leaves::read(&right_cells)),
                );
                Leaves::store(&cells, Layout::decompose(combined));
            } else {
                Leaves::store(&cells, Leaves::read(&right_cells));
                cell_valid.store(1u32);
            }
        }
        offset.store(offset.read() / 2u32);
    }
    (Leaves::read(&cells), cell_valid.read())
}

#[cubecl::cube]
fn finish_storage1_workgroup<Item, Op>(
    value: Item,
    valid: u32,
    plane_values: &mut Shared<[Item]>,
    plane_valid: &mut Shared<[u32]>,
    partials: &mut [Item],
) where
    Item: CubePrimitive,
    Op: ReductionOp<Item>,
{
    if UNIT_POS_PLANE == 0u32 {
        plane_values[PLANE_POS as usize] = value;
        plane_valid[PLANE_POS as usize] = valid;
    }
    sync_cube();

    if PLANE_POS == 0u32 {
        let plane_count = (CUBE_DIM + PLANE_DIM - 1u32) / PLANE_DIM;
        let lane = UNIT_POS_PLANE;
        let source = if lane < plane_count {
            UNIT_POS_PLANE as usize
        } else {
            0usize
        };
        let source_valid = if lane < plane_count {
            plane_valid[source]
        } else {
            0u32
        };
        let accumulator = RuntimeCell::<Item>::new(plane_values[source]);
        let accumulator_valid = RuntimeCell::<u32>::new(source_valid);
        let cursor = RuntimeCell::<u32>::new(lane + PLANE_DIM);
        while cursor.read() < plane_count {
            let index = cursor.read() as usize;
            if plane_valid[index] != 0u32 {
                if accumulator_valid.read() != 0u32 {
                    accumulator.store(Op::apply(accumulator.read(), plane_values[index]));
                } else {
                    accumulator.store(plane_values[index]);
                    accumulator_valid.store(1u32);
                }
            }
            cursor.store(cursor.read() + PLANE_DIM);
        }
        let result = reduce_plane_valid::<Item, Last<Item>, ScalarLayout<Item>, Op>(
            accumulator.read(),
            accumulator_valid.read(),
        );
        if UNIT_POS_PLANE == 0u32 && result.1 != 0u32 {
            partials[CUBE_POS as usize] = result.0.value;
        }
    }
}

#[doc(hidden)]
pub struct StagedBindings {
    pub(crate) slots: Vec<(cubecl::server::Handle, usize)>,
    pub(crate) offsets: Vec<u32>,
}

impl StagedBindings {
    pub(crate) fn new() -> Self {
        Self {
            slots: Vec::new(),
            offsets: Vec::new(),
        }
    }

    fn push(&mut self, handle: cubecl::server::Handle, len: usize, offset: u32) {
        self.slots.push((handle, len));
        self.offsets.push(offset);
    }
}

/// Host-side staging following the same left-first recursion as [`BindSlots`].
#[doc(hidden)]
pub trait StageRead<R: Runtime, Env>: BindSlots<Env> {
    fn logical_len(&self) -> Result<usize, Error>;
    fn stage_at(
        &self,
        client: &ComputeClient<R>,
        owner: u64,
        bindings: &mut StagedBindings,
    ) -> Result<(), Error>;
}

macro_rules! impl_leaf_staging {
    (impl <$( $env_ty:ident ),*> $env:ty) => {
        impl<R, T, $( $env_ty ),*> StageRead<R, $env> for Column<T>
        where
            R: Runtime,
            T: MStorageElement,
            Column<T>: BindSlots<$env>,
        {
            fn logical_len(&self) -> Result<usize, Error> {
                Ok(self.len)
            }

            fn stage_at(
                &self,
                _client: &ComputeClient<R>,
                owner: u64,
                bindings: &mut StagedBindings,
            ) -> Result<(), Error> {
                if self.owner != Some(owner) {
                    return Err(Error::ForeignExecutor);
                }
                let handle = self.handle.clone().ok_or(Error::UnboundColumn)?;
                bindings.push(handle, self.buffer_len, self.offset);
                Ok(())
            }
        }

        impl<R, T, $( $env_ty ),*> StageRead<R, $env> for Constant<T>
        where
            R: Runtime,
            T: MStorageElement,
            Constant<T>: BindSlots<$env>,
        {
            fn logical_len(&self) -> Result<usize, Error> {
                Ok(self.len)
            }

            fn stage_at(
                &self,
                client: &ComputeClient<R>,
                _owner: u64,
                bindings: &mut StagedBindings,
            ) -> Result<(), Error> {
                let handle = client.create_from_slice(T::as_bytes(&[self.value]));
                bindings.push(handle, 1, 0);
                Ok(())
            }
        }

        impl<R, $( $env_ty ),*> StageRead<R, $env> for Counting
        where
            R: Runtime,
            Counting: BindSlots<$env>,
        {
            fn logical_len(&self) -> Result<usize, Error> {
                Ok(self.len)
            }

            fn stage_at(
                &self,
                client: &ComputeClient<R>,
                _owner: u64,
                bindings: &mut StagedBindings,
            ) -> Result<(), Error> {
                let handle = client.create_from_slice(u32::as_bytes(&[self.start]));
                bindings.push(handle, 1, 0);
                Ok(())
            }
        }

        impl<R, $( $env_ty ),*> StageRead<R, $env> for ReverseCounting
        where
            R: Runtime,
            ReverseCounting: BindSlots<$env>,
        {
            fn logical_len(&self) -> Result<usize, Error> {
                Ok(self.len)
            }

            fn stage_at(
                &self,
                client: &ComputeClient<R>,
                _owner: u64,
                bindings: &mut StagedBindings,
            ) -> Result<(), Error> {
                let start = u32::try_from(self.start).map_err(|_| Error::LengthTooLarge {
                    len: self.start.saturating_add(1),
                })?;
                let handle = client.create_from_slice(u32::as_bytes(&[start]));
                bindings.push(handle, 1, 0);
                Ok(())
            }
        }
    };
}

impl_leaf_staging!(impl <> Env0);
impl_leaf_staging!(impl <L0> Env1<L0>);
impl_leaf_staging!(impl <L0, L1> Env2<L0, L1>);
impl_leaf_staging!(impl <L0, L1, L2> Env3<L0, L1, L2>);
impl_leaf_staging!(impl <L0, L1, L2, L3> Env4<L0, L1, L2, L3>);
impl_leaf_staging!(impl <L0, L1, L2, L3, L4> Env5<L0, L1, L2, L3, L4>);
impl_leaf_staging!(impl <L0, L1, L2, L3, L4, L5> Env6<L0, L1, L2, L3, L4, L5>);
impl_leaf_staging!(impl <L0, L1, L2, L3, L4, L5, L6> Env7<L0, L1, L2, L3, L4, L5, L6>);

impl<R, Source, Env> StageRead<R, Env> for Taken<Source>
where
    R: Runtime,
    Source: TakenSource,
    Source::Read: StageRead<R, Env>,
    Taken<Source>: BindSlots<Env>,
{
    fn logical_len(&self) -> Result<usize, Error> {
        Ok(self.len as usize)
    }

    fn stage_at(
        &self,
        client: &ComputeClient<R>,
        owner: u64,
        bindings: &mut StagedBindings,
    ) -> Result<(), Error> {
        self.lower().stage_at(client, owner, bindings)
    }
}

impl<R, Left, Right, Env> StageRead<R, Env> for Zip<Left, Right>
where
    R: Runtime,
    Left: StageRead<R, Env>,
    Right: StageRead<R, Left::NextEnv>,
    Zip<Left, Right>: BindSlots<Env>,
{
    fn logical_len(&self) -> Result<usize, Error> {
        let left = self.0.logical_len()?;
        let right = self.1.logical_len()?;
        if left != right {
            return Err(Error::LengthMismatch { left, right });
        }
        Ok(left)
    }

    fn stage_at(
        &self,
        client: &ComputeClient<R>,
        owner: u64,
        bindings: &mut StagedBindings,
    ) -> Result<(), Error> {
        self.0.stage_at(client, owner, bindings)?;
        self.1.stage_at(client, owner, bindings)
    }
}

impl<R, Input, Op, Env> StageRead<R, Env> for Transform<Input, Op>
where
    R: Runtime,
    Input: ReadExpression + StageRead<R, Env>,
    Op: UnaryOp<Input::Item>,
    Transform<Input, Op>: BindSlots<Env>,
{
    fn logical_len(&self) -> Result<usize, Error> {
        self.input.logical_len()
    }

    fn stage_at(
        &self,
        client: &ComputeClient<R>,
        owner: u64,
        bindings: &mut StagedBindings,
    ) -> Result<(), Error> {
        self.input.stage_at(client, owner, bindings)
    }
}

impl<R, Input, Op, Env> StageRead<R, Env> for crate::read::IndexedTransform<Input, Op>
where
    R: Runtime,
    Input: ReadExpression + StageRead<R, Env>,
    Op: crate::op::IndexedUnaryOp<Input::Item>,
    crate::read::IndexedTransform<Input, Op>: BindSlots<Env>,
{
    fn logical_len(&self) -> Result<usize, Error> {
        self.input.logical_len()
    }

    fn stage_at(
        &self,
        client: &ComputeClient<R>,
        owner: u64,
        bindings: &mut StagedBindings,
    ) -> Result<(), Error> {
        self.input.stage_at(client, owner, bindings)
    }
}

impl<R, Input, Op, Env> StageRead<R, Env> for crate::read::AdjacentIndexedTransform<Input, Op>
where
    R: Runtime,
    Input: ReadExpression + StageRead<R, Env>,
    Op: crate::op::IndexedBinaryOp<Input::Item>,
    crate::read::AdjacentIndexedTransform<Input, Op>: BindSlots<Env>,
{
    fn logical_len(&self) -> Result<usize, Error> {
        self.input.logical_len()
    }

    fn stage_at(
        &self,
        client: &ComputeClient<R>,
        owner: u64,
        bindings: &mut StagedBindings,
    ) -> Result<(), Error> {
        self.input.stage_at(client, owner, bindings)
    }
}

impl<R, Input, Op, Env> StageRead<R, Env> for crate::read::Adjacent<Input, Op>
where
    R: Runtime,
    Input: ReadExpression + StageRead<R, Env>,
    Op: ReductionOp<Input::Item>,
    crate::read::Adjacent<Input, Op>: BindSlots<Env>,
{
    fn logical_len(&self) -> Result<usize, Error> {
        self.input.logical_len()
    }

    fn stage_at(
        &self,
        client: &ComputeClient<R>,
        owner: u64,
        bindings: &mut StagedBindings,
    ) -> Result<(), Error> {
        self.input.stage_at(client, owner, bindings)
    }
}

impl<R, Input, Output, Env> StageRead<R, Env> for crate::read::Reassociate<Input, Output>
where
    R: Runtime,
    Input: ReadExpression + StageRead<R, Env>,
    Input::Item: StorageLayout,
    Output: StorageLayout + crate::WriteFrom<Input::Item> + 'static,
    crate::read::Reassociate<Input, Output>: BindSlots<Env>,
{
    fn logical_len(&self) -> Result<usize, Error> {
        self.input.logical_len()
    }

    fn stage_at(
        &self,
        client: &ComputeClient<R>,
        owner: u64,
        bindings: &mut StagedBindings,
    ) -> Result<(), Error> {
        self.input.stage_at(client, owner, bindings)
    }
}

impl<R, Input, Env> StageRead<R, Env> for crate::read::Slice<R, Input>
where
    R: Runtime,
    Input: StageRead<R, Env>,
    crate::read::Slice<R, Input>: BindSlots<Env>,
{
    fn logical_len(&self) -> Result<usize, Error> {
        self.input.logical_len()
    }

    fn stage_at(
        &self,
        client: &ComputeClient<R>,
        owner: u64,
        bindings: &mut StagedBindings,
    ) -> Result<(), Error> {
        self.input.stage_at(client, owner, bindings)
    }
}

impl<R, Values, Indices, Env> StageRead<R, Env> for Permute<Values, Indices>
where
    R: Runtime,
    Values: StageRead<R, Env>,
    Indices: StageRead<R, Values::NextEnv>,
    Permute<Values, Indices>: BindSlots<Env>,
{
    fn logical_len(&self) -> Result<usize, Error> {
        self.indices.logical_len()
    }

    fn stage_at(
        &self,
        client: &ComputeClient<R>,
        owner: u64,
        bindings: &mut StagedBindings,
    ) -> Result<(), Error> {
        self.values.stage_at(client, owner, bindings)?;
        self.indices.stage_at(client, owner, bindings)
    }
}

impl<R, Values, Env> StageRead<R, Env> for crate::read::Reverse<Values>
where
    R: Runtime,
    Values: StageRead<R, Env>,
    ReverseCounting: StageRead<R, Values::NextEnv>,
    crate::read::Reverse<Values>: BindSlots<Env>,
{
    fn logical_len(&self) -> Result<usize, Error> {
        match self.len {
            Some(len) => Ok(len),
            None => self.values.logical_len(),
        }
    }

    fn stage_at(
        &self,
        client: &ComputeClient<R>,
        owner: u64,
        bindings: &mut StagedBindings,
    ) -> Result<(), Error> {
        let input_len = self.values.logical_len()?;
        self.values.stage_at(client, owner, bindings)?;
        self.indices(input_len).stage_at(client, owner, bindings)
    }
}

macro_rules! define_reduce_eval_storage1_kernel {
    ($name:ident, $eval:ident, $method:ident; $( $leaf:ident : $slot:ident ),+ $(,)?) => {
        #[cubecl::cube(launch_unchecked, explicit_define)]
        fn $name<
            Item: CubePrimitive,
            $( $leaf: CubePrimitive, )+
            Expr: $eval<Item, $( $leaf ),+>,
            Op: ReductionOp<Item>,
        >(
            $( $slot: &[$leaf], )+
            offsets: &[u32],
            len: &[u32],
            partials: &mut [Item],
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = BLOCK_SIZE as usize;
            let logical_len = len[0] as usize;
            let mut plane_values = Shared::<[Item]>::new_slice(cube_dim);
            let mut plane_valid = Shared::<[u32]>::new_slice(cube_dim);
            let tile_start = (CUBE_POS as usize) * TILE_SIZE;
            let first_index = tile_start + unit;
            let safe_index = if first_index < logical_len {
                first_index
            } else {
                logical_len - 1usize
            };
            let accumulator = RuntimeCell::<Item>::new(
                Expr::$method($( $slot, )+ offsets, safe_index),
            );

            if tile_start + TILE_SIZE <= logical_len {
                for item in 1usize..ITEMS_PER_UNIT {
                    let value = Expr::$method(
                        $( $slot, )+
                        offsets,
                        first_index + item * cube_dim,
                    );
                    accumulator.store(Op::apply(accumulator.read(), value));
                }
                let result = reduce_plane_full::<Item, Last<Item>, ScalarLayout<Item>, Op>(
                    accumulator.read(),
                );
                finish_storage1_workgroup::<Item, Op>(
                    result.value,
                    1u32,
                    &mut plane_values,
                    &mut plane_valid,
                    partials,
                );
            } else {
                for item in 1usize..ITEMS_PER_UNIT {
                    let index = first_index + item * cube_dim;
                    if index < logical_len {
                        let value = Expr::$method($( $slot, )+ offsets, index);
                        accumulator.store(Op::apply(accumulator.read(), value));
                    }
                }
                let result = reduce_plane_valid::<Item, Last<Item>, ScalarLayout<Item>, Op>(
                    accumulator.read(),
                    if first_index < logical_len { 1u32 } else { 0u32 },
                );
                finish_storage1_workgroup::<Item, Op>(
                    result.0.value,
                    result.1,
                    &mut plane_values,
                    &mut plane_valid,
                    partials,
                );
            }
        }
    };
}

define_reduce_eval_storage1_kernel!(
    reduce_eval1_storage1_partials_kernel,
    Eval1,
    eval1;
    L0: slot0
);
define_reduce_eval_storage1_kernel!(
    reduce_eval2_storage1_partials_kernel,
    Eval2,
    eval2;
    L0: slot0,
    L1: slot1
);
define_reduce_eval_storage1_kernel!(
    reduce_eval3_storage1_partials_kernel,
    Eval3,
    eval3;
    L0: slot0,
    L1: slot1,
    L2: slot2
);
define_reduce_eval_storage1_kernel!(
    reduce_eval4_storage1_partials_kernel,
    Eval4,
    eval4;
    L0: slot0,
    L1: slot1,
    L2: slot2,
    L3: slot3
);
define_reduce_eval_storage1_kernel!(
    reduce_eval5_storage1_partials_kernel,
    Eval5,
    eval5;
    L0: slot0,
    L1: slot1,
    L2: slot2,
    L3: slot3,
    L4: slot4
);
define_reduce_eval_storage1_kernel!(
    reduce_eval6_storage1_partials_kernel,
    Eval6,
    eval6;
    L0: slot0,
    L1: slot1,
    L2: slot2,
    L3: slot3,
    L4: slot4,
    L5: slot5
);
define_reduce_eval_storage1_kernel!(
    reduce_eval7_storage1_partials_kernel,
    Eval7,
    eval7;
    L0: slot0,
    L1: slot1,
    L2: slot2,
    L3: slot3,
    L4: slot4,
    L5: slot5,
    L6: slot6
);
define_reduce_eval_storage1_kernel!(
    reduce_eval8_storage1_partials_kernel,
    Eval8,
    eval8;
    L0: slot0,
    L1: slot1,
    L2: slot2,
    L3: slot3,
    L4: slot4,
    L5: slot5,
    L6: slot6,
    L7: slot7
);

macro_rules! define_multi_reduce_eval_kernel {
    (
        $name:ident,$eval:ident,$method:ident,$load_trait:ident,$store_trait:ident;
        [$( $leaf:ident:$slot:ident ),+];
        [$first_out:ident:$first_partial:ident:$first_shared:ident $(, $out_ty:ident:$partial:ident:$shared:ident )*]
    ) => {
        #[cubecl::cube(launch_unchecked, explicit_define)]
        fn $name<
            Item: CubeType + Send + Sync + 'static,
            $( $leaf: CubePrimitive, )+
            $first_out: CubePrimitive,
            $( $out_ty: CubePrimitive, )*
            Leaves: CubeType + Send + Sync + 'static
                + $load_trait<$first_out, $( $out_ty ),*>
                + $store_trait<$first_out, $( $out_ty ),*>
                + MutableLeaves
                + PlaneShuffleLeaves,
            Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
            Expr: $eval<Item, $( $leaf ),+>,
            Op: ReductionOp<Item>,
        >(
            $( $slot: &[$leaf], )+
            read_offsets: &[u32],
            len: &[u32],
            zero_offsets: &[u32],
            $first_partial: &mut [$first_out],
            $( $partial: &mut [$out_ty], )*
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = BLOCK_SIZE as usize;
            let logical_len = len[0] as usize;
            let mut $first_shared = Shared::<[$first_out]>::new_slice(cube_dim);
            $( let mut $shared = Shared::<[$out_ty]>::new_slice(cube_dim); )*
            let mut plane_valid = Shared::<[u32]>::new_slice(cube_dim);
            let tile_start = (CUBE_POS as usize) * TILE_SIZE;
            let first_index = tile_start + unit;
            let safe_index = if first_index < logical_len {
                first_index
            } else {
                logical_len - 1usize
            };
            let accumulator = Layout::decompose(
                Expr::$method($( $slot, )+ read_offsets, safe_index),
            ).into_cells();

            if tile_start + TILE_SIZE <= logical_len {
                for item in 1usize..ITEMS_PER_UNIT {
                    let value = Expr::$method(
                        $( $slot, )+
                        read_offsets,
                        first_index + item * cube_dim,
                    );
                    accumulate_register::<Item, Leaves, Layout, Op>(&accumulator, value);
                }
                let result = reduce_plane_full::<Item, Leaves, Layout, Op>(
                    Layout::recompose(Leaves::read(&accumulator)),
                );
                if UNIT_POS_PLANE == 0u32 {
                    result.store(
                        &mut $first_shared,
                        $( &mut $shared, )*
                        zero_offsets,
                        PLANE_POS as usize,
                    );
                    plane_valid[PLANE_POS as usize] = 1u32;
                }
                sync_cube();

                if PLANE_POS == 0u32 {
                    let plane_count = (CUBE_DIM + PLANE_DIM - 1u32) / PLANE_DIM;
                    let source = if UNIT_POS_PLANE < plane_count {
                        UNIT_POS_PLANE as usize
                    } else {
                        0usize
                    };
                    let source_valid = if UNIT_POS_PLANE < plane_count {
                        plane_valid[source]
                    } else {
                        0u32
                    };
                    let source_value = Layout::recompose(Leaves::load(
                        &$first_shared,
                        $( &$shared, )*
                        zero_offsets,
                        source,
                    ));
                    let block_accumulator = Layout::decompose(source_value).into_cells();
                    let block_valid = RuntimeCell::<u32>::new(source_valid);
                    let cursor = RuntimeCell::<u32>::new(UNIT_POS_PLANE + PLANE_DIM);
                    while cursor.read() < plane_count {
                        let index = cursor.read() as usize;
                        if block_valid.read() != 0u32 && plane_valid[index] != 0u32 {
                            let value = Layout::recompose(Leaves::load(
                                &$first_shared,
                                $( &$shared, )*
                                zero_offsets,
                                index,
                            ));
                            accumulate_register::<Item, Leaves, Layout, Op>(
                                &block_accumulator,
                                value,
                            );
                        }
                        cursor.store(cursor.read() + PLANE_DIM);
                    }
                    let block_result = reduce_plane_valid::<Item, Leaves, Layout, Op>(
                        Layout::recompose(Leaves::read(&block_accumulator)),
                        block_valid.read(),
                    );
                    if UNIT_POS_PLANE == 0u32 && block_result.1 != 0u32 {
                        block_result.0.store(
                            $first_partial,
                            $( $partial, )*
                            zero_offsets,
                            CUBE_POS as usize,
                        );
                    }
                }
            } else {
                for item in 1usize..ITEMS_PER_UNIT {
                    let index = first_index + item * cube_dim;
                    if index < logical_len {
                        let value = Expr::$method(
                            $( $slot, )+
                            read_offsets,
                            index,
                        );
                        accumulate_register::<Item, Leaves, Layout, Op>(&accumulator, value);
                    }
                }
                let result = reduce_plane_valid::<Item, Leaves, Layout, Op>(
                    Layout::recompose(Leaves::read(&accumulator)),
                    if first_index < logical_len { 1u32 } else { 0u32 },
                );
                if UNIT_POS_PLANE == 0u32 {
                    result.0.store(
                        &mut $first_shared,
                        $( &mut $shared, )*
                        zero_offsets,
                        PLANE_POS as usize,
                    );
                    plane_valid[PLANE_POS as usize] = result.1;
                }
                sync_cube();

                if PLANE_POS == 0u32 {
                    let plane_count = (CUBE_DIM + PLANE_DIM - 1u32) / PLANE_DIM;
                    let source = if UNIT_POS_PLANE < plane_count {
                        UNIT_POS_PLANE as usize
                    } else {
                        0usize
                    };
                    let source_valid = if UNIT_POS_PLANE < plane_count {
                        plane_valid[source]
                    } else {
                        0u32
                    };
                    let source_value = Layout::recompose(Leaves::load(
                        &$first_shared,
                        $( &$shared, )*
                        zero_offsets,
                        source,
                    ));
                    let block_accumulator = Layout::decompose(source_value).into_cells();
                    let block_valid = RuntimeCell::<u32>::new(source_valid);
                    let cursor = RuntimeCell::<u32>::new(UNIT_POS_PLANE + PLANE_DIM);
                    while cursor.read() < plane_count {
                        let index = cursor.read() as usize;
                        if block_valid.read() != 0u32 && plane_valid[index] != 0u32 {
                            let value = Layout::recompose(Leaves::load(
                                &$first_shared,
                                $( &$shared, )*
                                zero_offsets,
                                index,
                            ));
                            accumulate_register::<Item, Leaves, Layout, Op>(
                                &block_accumulator,
                                value,
                            );
                        }
                        cursor.store(cursor.read() + PLANE_DIM);
                    }
                    let block_result = reduce_plane_valid::<Item, Leaves, Layout, Op>(
                        Layout::recompose(Leaves::read(&block_accumulator)),
                        block_valid.read(),
                    );
                    if UNIT_POS_PLANE == 0u32 && block_result.1 != 0u32 {
                        block_result.0.store(
                            $first_partial,
                            $( $partial, )*
                            zero_offsets,
                            CUBE_POS as usize,
                        );
                    }
                }
            }
        }
    };
}

macro_rules! define_multi_reduce_eval_kernels {
    ($eval:ident,$method:ident; [$( $leaf:ident:$slot:ident ),+]; [$k2:ident,$k3:ident,$k4:ident,$k5:ident,$k6:ident,$k7:ident]) => {
        define_multi_reduce_eval_kernel!($k2,$eval,$method,LoadLeaves2,StoreLeaves2; [$($leaf:$slot),+]; [O0:partial0:shared0,O1:partial1:shared1]);
        define_multi_reduce_eval_kernel!($k3,$eval,$method,LoadLeaves3,StoreLeaves3; [$($leaf:$slot),+]; [O0:partial0:shared0,O1:partial1:shared1,O2:partial2:shared2]);
        define_multi_reduce_eval_kernel!($k4,$eval,$method,LoadLeaves4,StoreLeaves4; [$($leaf:$slot),+]; [O0:partial0:shared0,O1:partial1:shared1,O2:partial2:shared2,O3:partial3:shared3]);
        define_multi_reduce_eval_kernel!($k5,$eval,$method,LoadLeaves5,StoreLeaves5; [$($leaf:$slot),+]; [O0:partial0:shared0,O1:partial1:shared1,O2:partial2:shared2,O3:partial3:shared3,O4:partial4:shared4]);
        define_multi_reduce_eval_kernel!($k6,$eval,$method,LoadLeaves6,StoreLeaves6; [$($leaf:$slot),+]; [O0:partial0:shared0,O1:partial1:shared1,O2:partial2:shared2,O3:partial3:shared3,O4:partial4:shared4,O5:partial5:shared5]);
        define_multi_reduce_eval_kernel!($k7,$eval,$method,LoadLeaves7,StoreLeaves7; [$($leaf:$slot),+]; [O0:partial0:shared0,O1:partial1:shared1,O2:partial2:shared2,O3:partial3:shared3,O4:partial4:shared4,O5:partial5:shared5,O6:partial6:shared6]);
    };
}

define_multi_reduce_eval_kernels!(Eval1,eval1; [L0:slot0]; [reduce_eval1_s2,reduce_eval1_s3,reduce_eval1_s4,reduce_eval1_s5,reduce_eval1_s6,reduce_eval1_s7]);
define_multi_reduce_eval_kernels!(Eval2,eval2; [L0:slot0,L1:slot1]; [reduce_eval2_s2,reduce_eval2_s3,reduce_eval2_s4,reduce_eval2_s5,reduce_eval2_s6,reduce_eval2_s7]);
define_multi_reduce_eval_kernels!(Eval3,eval3; [L0:slot0,L1:slot1,L2:slot2]; [reduce_eval3_s2,reduce_eval3_s3,reduce_eval3_s4,reduce_eval3_s5,reduce_eval3_s6,reduce_eval3_s7]);
define_multi_reduce_eval_kernels!(Eval4,eval4; [L0:slot0,L1:slot1,L2:slot2,L3:slot3]; [reduce_eval4_s2,reduce_eval4_s3,reduce_eval4_s4,reduce_eval4_s5,reduce_eval4_s6,reduce_eval4_s7]);
define_multi_reduce_eval_kernels!(Eval5,eval5; [L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4]; [reduce_eval5_s2,reduce_eval5_s3,reduce_eval5_s4,reduce_eval5_s5,reduce_eval5_s6,reduce_eval5_s7]);
define_multi_reduce_eval_kernels!(Eval6,eval6; [L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5]; [reduce_eval6_s2,reduce_eval6_s3,reduce_eval6_s4,reduce_eval6_s5,reduce_eval6_s6,reduce_eval6_s7]);
define_multi_reduce_eval_kernels!(Eval7,eval7; [L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5,L6:slot6]; [reduce_eval7_s2,reduce_eval7_s3,reduce_eval7_s4,reduce_eval7_s5,reduce_eval7_s6,reduce_eval7_s7]);
define_multi_reduce_eval_kernels!(Eval8,eval8; [L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5,L6:slot6,L7:slot7]; [reduce_eval8_s2,reduce_eval8_s3,reduce_eval8_s4,reduce_eval8_s5,reduce_eval8_s6,reduce_eval8_s7]);

macro_rules! define_storage_reduce_kernel {
    ($name:ident,$load_trait:ident,$store_trait:ident; [$first_out:ident:$first_input:ident:$first_partial:ident:$first_shared:ident $(, $out_ty:ident:$input:ident:$partial:ident:$shared:ident )*]) => {
        #[cubecl::cube(launch_unchecked, explicit_define)]
        fn $name<
            Item: CubeType + Send + Sync + 'static,
            $first_out: CubePrimitive,
            $( $out_ty: CubePrimitive, )*
            Leaves: CubeType + Send + Sync + 'static
                + $load_trait<$first_out, $( $out_ty ),*>
                + $store_trait<$first_out, $( $out_ty ),*>
                + MutableLeaves
                + PlaneShuffleLeaves,
            Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
            Op: ReductionOp<Item>,
        >(
            $first_input: &[$first_out],
            $( $input: &[$out_ty], )*
            len: &[u32],
            zero_offsets: &[u32],
            $first_partial: &mut [$first_out],
            $( $partial: &mut [$out_ty], )*
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = BLOCK_SIZE as usize;
            let logical_len = len[0] as usize;
            let mut $first_shared = Shared::<[$first_out]>::new_slice(cube_dim);
            $( let mut $shared = Shared::<[$out_ty]>::new_slice(cube_dim); )*
            let mut plane_valid = Shared::<[u32]>::new_slice(cube_dim);
            let tile_start = (CUBE_POS as usize) * TILE_SIZE;
            let first_index = tile_start + unit;
            let safe_index = if first_index < logical_len {
                first_index
            } else {
                logical_len - 1usize
            };
            let accumulator = Leaves::load(
                $first_input,
                $( $input, )*
                zero_offsets,
                safe_index,
            ).into_cells();

            if tile_start + TILE_SIZE <= logical_len {
                for item in 1usize..ITEMS_PER_UNIT {
                    let value = Layout::recompose(Leaves::load(
                        $first_input,
                        $( $input, )*
                        zero_offsets,
                        first_index + item * cube_dim,
                    ));
                    accumulate_register::<Item, Leaves, Layout, Op>(&accumulator, value);
                }
                let result = reduce_plane_full::<Item, Leaves, Layout, Op>(
                    Layout::recompose(Leaves::read(&accumulator)),
                );
                if UNIT_POS_PLANE == 0u32 {
                    result.store(
                        &mut $first_shared,
                        $( &mut $shared, )*
                        zero_offsets,
                        PLANE_POS as usize,
                    );
                    plane_valid[PLANE_POS as usize] = 1u32;
                }
                sync_cube();

                if PLANE_POS == 0u32 {
                    let plane_count = (CUBE_DIM + PLANE_DIM - 1u32) / PLANE_DIM;
                    let source = if UNIT_POS_PLANE < plane_count {
                        UNIT_POS_PLANE as usize
                    } else {
                        0usize
                    };
                    let source_valid = if UNIT_POS_PLANE < plane_count {
                        plane_valid[source]
                    } else {
                        0u32
                    };
                    let source_value = Layout::recompose(Leaves::load(
                        &$first_shared,
                        $( &$shared, )*
                        zero_offsets,
                        source,
                    ));
                    let block_accumulator = Layout::decompose(source_value).into_cells();
                    let block_valid = RuntimeCell::<u32>::new(source_valid);
                    let cursor = RuntimeCell::<u32>::new(UNIT_POS_PLANE + PLANE_DIM);
                    while cursor.read() < plane_count {
                        let index = cursor.read() as usize;
                        if block_valid.read() != 0u32 && plane_valid[index] != 0u32 {
                            let value = Layout::recompose(Leaves::load(
                                &$first_shared,
                                $( &$shared, )*
                                zero_offsets,
                                index,
                            ));
                            accumulate_register::<Item, Leaves, Layout, Op>(
                                &block_accumulator,
                                value,
                            );
                        }
                        cursor.store(cursor.read() + PLANE_DIM);
                    }
                    let block_result = reduce_plane_valid::<Item, Leaves, Layout, Op>(
                        Layout::recompose(Leaves::read(&block_accumulator)),
                        block_valid.read(),
                    );
                    if UNIT_POS_PLANE == 0u32 && block_result.1 != 0u32 {
                        block_result.0.store(
                            $first_partial,
                            $( $partial, )*
                            zero_offsets,
                            CUBE_POS as usize,
                        );
                    }
                }
            } else {
                for item in 1usize..ITEMS_PER_UNIT {
                    let index = first_index + item * cube_dim;
                    if index < logical_len {
                        let value = Layout::recompose(Leaves::load(
                            $first_input,
                            $( $input, )*
                            zero_offsets,
                            index,
                        ));
                        accumulate_register::<Item, Leaves, Layout, Op>(&accumulator, value);
                    }
                }
                let result = reduce_plane_valid::<Item, Leaves, Layout, Op>(
                    Layout::recompose(Leaves::read(&accumulator)),
                    if first_index < logical_len { 1u32 } else { 0u32 },
                );
                if UNIT_POS_PLANE == 0u32 {
                    result.0.store(
                        &mut $first_shared,
                        $( &mut $shared, )*
                        zero_offsets,
                        PLANE_POS as usize,
                    );
                    plane_valid[PLANE_POS as usize] = result.1;
                }
                sync_cube();

                if PLANE_POS == 0u32 {
                    let plane_count = (CUBE_DIM + PLANE_DIM - 1u32) / PLANE_DIM;
                    let source = if UNIT_POS_PLANE < plane_count {
                        UNIT_POS_PLANE as usize
                    } else {
                        0usize
                    };
                    let source_valid = if UNIT_POS_PLANE < plane_count {
                        plane_valid[source]
                    } else {
                        0u32
                    };
                    let source_value = Layout::recompose(Leaves::load(
                        &$first_shared,
                        $( &$shared, )*
                        zero_offsets,
                        source,
                    ));
                    let block_accumulator = Layout::decompose(source_value).into_cells();
                    let block_valid = RuntimeCell::<u32>::new(source_valid);
                    let cursor = RuntimeCell::<u32>::new(UNIT_POS_PLANE + PLANE_DIM);
                    while cursor.read() < plane_count {
                        let index = cursor.read() as usize;
                        if block_valid.read() != 0u32 && plane_valid[index] != 0u32 {
                            let value = Layout::recompose(Leaves::load(
                                &$first_shared,
                                $( &$shared, )*
                                zero_offsets,
                                index,
                            ));
                            accumulate_register::<Item, Leaves, Layout, Op>(
                                &block_accumulator,
                                value,
                            );
                        }
                        cursor.store(cursor.read() + PLANE_DIM);
                    }
                    let block_result = reduce_plane_valid::<Item, Leaves, Layout, Op>(
                        Layout::recompose(Leaves::read(&block_accumulator)),
                        block_valid.read(),
                    );
                    if UNIT_POS_PLANE == 0u32 && block_result.1 != 0u32 {
                        block_result.0.store(
                            $first_partial,
                            $( $partial, )*
                            zero_offsets,
                            CUBE_POS as usize,
                        );
                    }
                }
            }
        }
    };
}

define_storage_reduce_kernel!(reduce_storage_s2,LoadLeaves2,StoreLeaves2; [O0:input0:partial0:shared0,O1:input1:partial1:shared1]);
define_storage_reduce_kernel!(reduce_storage_s3,LoadLeaves3,StoreLeaves3; [O0:input0:partial0:shared0,O1:input1:partial1:shared1,O2:input2:partial2:shared2]);
define_storage_reduce_kernel!(reduce_storage_s4,LoadLeaves4,StoreLeaves4; [O0:input0:partial0:shared0,O1:input1:partial1:shared1,O2:input2:partial2:shared2,O3:input3:partial3:shared3]);
define_storage_reduce_kernel!(reduce_storage_s5,LoadLeaves5,StoreLeaves5; [O0:input0:partial0:shared0,O1:input1:partial1:shared1,O2:input2:partial2:shared2,O3:input3:partial3:shared3,O4:input4:partial4:shared4]);
define_storage_reduce_kernel!(reduce_storage_s6,LoadLeaves6,StoreLeaves6; [O0:input0:partial0:shared0,O1:input1:partial1:shared1,O2:input2:partial2:shared2,O3:input3:partial3:shared3,O4:input4:partial4:shared4,O5:input5:partial5:shared5]);
define_storage_reduce_kernel!(reduce_storage_s7,LoadLeaves7,StoreLeaves7; [O0:input0:partial0:shared0,O1:input1:partial1:shared1,O2:input2:partial2:shared2,O3:input3:partial3:shared3,O4:input4:partial4:shared4,O5:input5:partial5:shared5,O6:input6:partial6:shared6]);

macro_rules! define_multi_reduce_finalize_kernel {
    ($name:ident,$load_trait:ident,$store_trait:ident; [$( $out_ty:ident:$partial:ident:$init:ident:$output:ident ),+]) => {
        #[cubecl::cube(launch_unchecked, explicit_define)]
        fn $name<
            Item: CubeType + Send + Sync + 'static,
            $( $out_ty: CubePrimitive, )+
            Leaves: CubeType + Send + Sync + 'static
                + $load_trait<$( $out_ty ),+>
                + $store_trait<$( $out_ty ),+>,
            Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
            Op: ReductionOp<Item>,
        >(
            $( $partial: &[$out_ty], )+
            $( $init: &[$out_ty], )+
            zero_offsets: &[u32],
            $( $output: &mut [$out_ty], )+
        ) {
            if ABSOLUTE_POS == 0 {
                let initial = Layout::recompose(Leaves::load(
                    $( $init, )+ zero_offsets, 0,
                ));
                let reduced = Layout::recompose(Leaves::load(
                    $( $partial, )+ zero_offsets, 0,
                ));
                Layout::decompose(Op::apply(initial, reduced)).store(
                    $( $output, )+ zero_offsets, 0,
                );
            }
        }
    };
}

define_multi_reduce_finalize_kernel!(reduce_finalize_s2,LoadLeaves2,StoreLeaves2; [O0:partial0:init0:out0,O1:partial1:init1:out1]);
define_multi_reduce_finalize_kernel!(reduce_finalize_s3,LoadLeaves3,StoreLeaves3; [O0:partial0:init0:out0,O1:partial1:init1:out1,O2:partial2:init2:out2]);
define_multi_reduce_finalize_kernel!(reduce_finalize_s4,LoadLeaves4,StoreLeaves4; [O0:partial0:init0:out0,O1:partial1:init1:out1,O2:partial2:init2:out2,O3:partial3:init3:out3]);
define_multi_reduce_finalize_kernel!(reduce_finalize_s5,LoadLeaves5,StoreLeaves5; [O0:partial0:init0:out0,O1:partial1:init1:out1,O2:partial2:init2:out2,O3:partial3:init3:out3,O4:partial4:init4:out4]);
define_multi_reduce_finalize_kernel!(reduce_finalize_s6,LoadLeaves6,StoreLeaves6; [O0:partial0:init0:out0,O1:partial1:init1:out1,O2:partial2:init2:out2,O3:partial3:init3:out3,O4:partial4:init4:out4,O5:partial5:init5:out5]);
define_multi_reduce_finalize_kernel!(reduce_finalize_s7,LoadLeaves7,StoreLeaves7; [O0:partial0:init0:out0,O1:partial1:init1:out1,O2:partial2:init2:out2,O3:partial3:init3:out3,O4:partial4:init4:out4,O5:partial5:init5:out5,O6:partial6:init6:out6]);

#[cubecl::cube(launch_unchecked, explicit_define)]
fn reduce_storage1_partials_kernel<Item: CubePrimitive, Op: ReductionOp<Item>>(
    input: &[Item],
    len: &[u32],
    partials: &mut [Item],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = BLOCK_SIZE as usize;
    let logical_len = len[0] as usize;
    let mut plane_values = Shared::<[Item]>::new_slice(cube_dim);
    let mut plane_valid = Shared::<[u32]>::new_slice(cube_dim);
    let tile_start = (CUBE_POS as usize) * TILE_SIZE;
    let first_index = tile_start + unit;
    let safe_index = if first_index < logical_len {
        first_index
    } else {
        logical_len - 1usize
    };
    let accumulator = RuntimeCell::<Item>::new(input[safe_index]);

    if tile_start + TILE_SIZE <= logical_len {
        for item in 1usize..ITEMS_PER_UNIT {
            accumulator.store(Op::apply(
                accumulator.read(),
                input[first_index + item * cube_dim],
            ));
        }
        let result =
            reduce_plane_full::<Item, Last<Item>, ScalarLayout<Item>, Op>(accumulator.read());
        finish_storage1_workgroup::<Item, Op>(
            result.value,
            1u32,
            &mut plane_values,
            &mut plane_valid,
            partials,
        );
    } else {
        for item in 1usize..ITEMS_PER_UNIT {
            let index = first_index + item * cube_dim;
            if index < logical_len {
                accumulator.store(Op::apply(accumulator.read(), input[index]));
            }
        }
        let result = reduce_plane_valid::<Item, Last<Item>, ScalarLayout<Item>, Op>(
            accumulator.read(),
            if first_index < logical_len {
                1u32
            } else {
                0u32
            },
        );
        finish_storage1_workgroup::<Item, Op>(
            result.0.value,
            result.1,
            &mut plane_values,
            &mut plane_valid,
            partials,
        );
    }
}

#[cubecl::cube(launch_unchecked, explicit_define)]
fn reduce_storage1_finalize_kernel<Item: CubePrimitive, Op: ReductionOp<Item>>(
    partial: &[Item],
    init: &[Item],
    output: &mut [Item],
) {
    if ABSOLUTE_POS == 0 {
        output[0] = Op::apply(init[0], partial[0]);
    }
}

fn pass_block_count(len: usize) -> usize {
    len.div_ceil(TILE_SIZE).max(1)
}

fn checked_u32(len: usize) -> Result<u32, Error> {
    u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })
}

fn finish_storage1_reduce<R, Item, Op>(
    exec: &Executor<R>,
    mut current: cubecl::server::Handle,
    mut current_len: usize,
    init: Item,
) -> Result<Item, Error>
where
    R: Runtime,
    Item: ReduceStorage1,
    Op: ReductionOp<Item>,
{
    let client = exec.client();
    while current_len > 1 {
        let next_len = pass_block_count(current_len);
        let len = client.create_from_slice(u32::as_bytes(&[checked_u32(current_len)?]));
        let next = client.empty(next_len * size_of::<Item>());
        unsafe {
            reduce_storage1_partials_kernel::launch_unchecked::<Item, Op, R>(
                client,
                cube_count_1d(next_len)?,
                CubeDim::new_1d(BLOCK_SIZE),
                BufferArg::from_raw_parts(current, current_len),
                BufferArg::from_raw_parts(len, 1),
                BufferArg::from_raw_parts(next.clone(), next_len),
            );
        }
        current = next;
        current_len = next_len;
    }

    let init_handle = client.create_from_slice(Item::as_bytes(&[init]));
    let output = client.empty(size_of::<Item>());
    unsafe {
        reduce_storage1_finalize_kernel::launch_unchecked::<Item, Op, R>(
            client,
            CubeCount::new_single(),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(current, 1),
            BufferArg::from_raw_parts(init_handle, 1),
            BufferArg::from_raw_parts(output.clone(), 1),
        );
    }
    let bytes = client.read_one(output).map_err(|err| Error::Launch {
        message: format!("{err:?}"),
    })?;
    Ok(Item::from_bytes(&bytes)[0])
}

/// Consumer-specific reduction dispatch.
#[doc(hidden)]
pub trait ReduceDispatch<R: Runtime, Input, Item, Op, Slots> {
    fn execute(exec: &Executor<R>, input: &Input, init: Item) -> Result<Item, Error>;
}

impl<R, Input, Item, Op, L0> ReduceDispatch<R, Input, Item, Op, Env1<L0>> for Dispatch<A1, S1>
where
    R: Runtime,
    Item: ReduceStorage1,
    L0: MStorageElement,
    Op: ReductionOp<Item>,
    Input: ReadExpression<Item = Item, ReadArity = A1>
        + BindSlots<Env0, NextEnv = Env1<L0>>
        + LowerReadExpression<Slots = Env1<L0>>
        + StageRead<R, Env0>,
    Input::DeviceExpr: Eval1<Item, L0>,
{
    fn execute(exec: &Executor<R>, input: &Input, init: Item) -> Result<Item, Error> {
        let len = input.logical_len()?;
        if len == 0 {
            return Ok(init);
        }
        let mut bindings = StagedBindings::new();
        input.stage_at(exec.client(), exec.id(), &mut bindings)?;
        debug_assert_eq!(bindings.slots.len(), 1);
        let blocks = pass_block_count(len);
        let offsets = exec
            .client()
            .create_from_slice(u32::as_bytes(&bindings.offsets));
        let len_handle = exec
            .client()
            .create_from_slice(u32::as_bytes(&[checked_u32(len)?]));
        let partials = exec.client().empty(blocks * size_of::<Item>());
        unsafe {
            reduce_eval1_storage1_partials_kernel::launch_unchecked::<
                Item,
                L0,
                Input::DeviceExpr,
                Op,
                R,
            >(
                exec.client(),
                cube_count_1d(blocks)?,
                CubeDim::new_1d(BLOCK_SIZE),
                BufferArg::from_raw_parts(bindings.slots[0].0.clone(), bindings.slots[0].1),
                BufferArg::from_raw_parts(offsets, 1),
                BufferArg::from_raw_parts(len_handle, 1),
                BufferArg::from_raw_parts(partials.clone(), blocks),
            );
        }
        finish_storage1_reduce::<R, Item, Op>(exec, partials, blocks, init)
    }
}

impl<R, Input, Item, Op, L0, L1> ReduceDispatch<R, Input, Item, Op, Env2<L0, L1>>
    for Dispatch<A2, S1>
where
    R: Runtime,
    Item: ReduceStorage1,
    L0: MStorageElement,
    L1: MStorageElement,
    Op: ReductionOp<Item>,
    Input: ReadExpression<Item = Item, ReadArity = A2>
        + BindSlots<Env0, NextEnv = Env2<L0, L1>>
        + LowerReadExpression<Slots = Env2<L0, L1>>
        + StageRead<R, Env0>,
    Input::DeviceExpr: Eval2<Item, L0, L1>,
{
    fn execute(exec: &Executor<R>, input: &Input, init: Item) -> Result<Item, Error> {
        let len = input.logical_len()?;
        if len == 0 {
            return Ok(init);
        }
        let mut bindings = StagedBindings::new();
        input.stage_at(exec.client(), exec.id(), &mut bindings)?;
        debug_assert_eq!(bindings.slots.len(), 2);
        let blocks = pass_block_count(len);
        let offsets = exec
            .client()
            .create_from_slice(u32::as_bytes(&bindings.offsets));
        let len_handle = exec
            .client()
            .create_from_slice(u32::as_bytes(&[checked_u32(len)?]));
        let partials = exec.client().empty(blocks * size_of::<Item>());
        unsafe {
            reduce_eval2_storage1_partials_kernel::launch_unchecked::<
                Item,
                L0,
                L1,
                Input::DeviceExpr,
                Op,
                R,
            >(
                exec.client(),
                cube_count_1d(blocks)?,
                CubeDim::new_1d(BLOCK_SIZE),
                BufferArg::from_raw_parts(bindings.slots[0].0.clone(), bindings.slots[0].1),
                BufferArg::from_raw_parts(bindings.slots[1].0.clone(), bindings.slots[1].1),
                BufferArg::from_raw_parts(offsets, 2),
                BufferArg::from_raw_parts(len_handle, 1),
                BufferArg::from_raw_parts(partials.clone(), blocks),
            );
        }
        finish_storage1_reduce::<R, Item, Op>(exec, partials, blocks, init)
    }
}

impl<R, Input, Item, Op, L0, L1, L2> ReduceDispatch<R, Input, Item, Op, Env3<L0, L1, L2>>
    for Dispatch<A3, S1>
where
    R: Runtime,
    Item: ReduceStorage1,
    L0: MStorageElement,
    L1: MStorageElement,
    L2: MStorageElement,
    Op: ReductionOp<Item>,
    Input: ReadExpression<Item = Item, ReadArity = A3>
        + BindSlots<Env0, NextEnv = Env3<L0, L1, L2>>
        + LowerReadExpression<Slots = Env3<L0, L1, L2>>
        + StageRead<R, Env0>,
    Input::DeviceExpr: Eval3<Item, L0, L1, L2>,
{
    fn execute(exec: &Executor<R>, input: &Input, init: Item) -> Result<Item, Error> {
        let len = input.logical_len()?;
        if len == 0 {
            return Ok(init);
        }
        let mut bindings = StagedBindings::new();
        input.stage_at(exec.client(), exec.id(), &mut bindings)?;
        debug_assert_eq!(bindings.slots.len(), 3);
        let blocks = pass_block_count(len);
        let offsets = exec
            .client()
            .create_from_slice(u32::as_bytes(&bindings.offsets));
        let len_handle = exec
            .client()
            .create_from_slice(u32::as_bytes(&[checked_u32(len)?]));
        let partials = exec.client().empty(blocks * size_of::<Item>());
        unsafe {
            reduce_eval3_storage1_partials_kernel::launch_unchecked::<
                Item,
                L0,
                L1,
                L2,
                Input::DeviceExpr,
                Op,
                R,
            >(
                exec.client(),
                cube_count_1d(blocks)?,
                CubeDim::new_1d(BLOCK_SIZE),
                BufferArg::from_raw_parts(bindings.slots[0].0.clone(), bindings.slots[0].1),
                BufferArg::from_raw_parts(bindings.slots[1].0.clone(), bindings.slots[1].1),
                BufferArg::from_raw_parts(bindings.slots[2].0.clone(), bindings.slots[2].1),
                BufferArg::from_raw_parts(offsets, 3),
                BufferArg::from_raw_parts(len_handle, 1),
                BufferArg::from_raw_parts(partials.clone(), blocks),
            );
        }
        finish_storage1_reduce::<R, Item, Op>(exec, partials, blocks, init)
    }
}

macro_rules! impl_reduce_storage1_dispatch {
    (
        $arity:ty,$eval:ident,$kernel:ident,$env:ty;
        [$( $leaf:ident:$index:literal ),+ $(,)?]
    ) => {
        impl<R, Input, Item, Op, $( $leaf ),+>
            ReduceDispatch<R, Input, Item, Op, $env> for Dispatch<$arity, S1>
        where
            R: Runtime,
            Item: ReduceStorage1,
            $( $leaf: MStorageElement, )+
            Op: ReductionOp<Item>,
            Input: ReadExpression<Item = Item, ReadArity = $arity>
                + BindSlots<Env0, NextEnv = $env>
                + LowerReadExpression<Slots = $env>
                + StageRead<R, Env0>,
            Input::DeviceExpr: $eval<Item, $( $leaf ),+>,
        {
            fn execute(exec: &Executor<R>, input: &Input, init: Item) -> Result<Item, Error> {
                let len = input.logical_len()?;
                if len == 0 {
                    return Ok(init);
                }
                let mut bindings = StagedBindings::new();
                input.stage_at(exec.client(), exec.id(), &mut bindings)?;
                debug_assert_eq!(bindings.slots.len(), impl_reduce_storage1_dispatch!(@count $( $leaf ),+));
                let blocks = pass_block_count(len);
                let offsets = exec.client().create_from_slice(u32::as_bytes(&bindings.offsets));
                let len_handle = exec.client().create_from_slice(u32::as_bytes(&[checked_u32(len)?]));
                let partials = exec.client().empty(blocks * size_of::<Item>());
                unsafe {
                    $kernel::launch_unchecked::<Item, $( $leaf, )+ Input::DeviceExpr, Op, R>(
                        exec.client(),
                        cube_count_1d(blocks)?,
                        CubeDim::new_1d(BLOCK_SIZE),
                        $( BufferArg::from_raw_parts(bindings.slots[$index].0.clone(), bindings.slots[$index].1), )+
                        BufferArg::from_raw_parts(offsets, bindings.offsets.len()),
                        BufferArg::from_raw_parts(len_handle, 1),
                        BufferArg::from_raw_parts(partials.clone(), blocks),
                    );
                }
                finish_storage1_reduce::<R, Item, Op>(exec, partials, blocks, init)
            }
        }
    };
    (@count $first:ident $(, $rest:ident)*) => { 1usize $( + { let _ = stringify!($rest); 1usize } )* };
}

impl_reduce_storage1_dispatch!(A4,Eval4,reduce_eval4_storage1_partials_kernel,Env4<L0,L1,L2,L3>; [L0:0,L1:1,L2:2,L3:3]);
impl_reduce_storage1_dispatch!(A5,Eval5,reduce_eval5_storage1_partials_kernel,Env5<L0,L1,L2,L3,L4>; [L0:0,L1:1,L2:2,L3:3,L4:4]);
impl_reduce_storage1_dispatch!(A6,Eval6,reduce_eval6_storage1_partials_kernel,Env6<L0,L1,L2,L3,L4,L5>; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5]);
impl_reduce_storage1_dispatch!(A7,Eval7,reduce_eval7_storage1_partials_kernel,Env7<L0,L1,L2,L3,L4,L5,L6>; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5,L6:6]);
impl_reduce_storage1_dispatch!(A8,Eval8,reduce_eval8_storage1_partials_kernel,crate::read::Env8<L0,L1,L2,L3,L4,L5,L6,L7>; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5,L6:6,L7:7]);

macro_rules! leaf_value {
    ($root:ident; $first:ident $( . $rest:ident )*) => {
        $root.$first $( .$rest )*
    };
}

macro_rules! build_leaf_list {
    ($v0:ident,$v1:ident) => {
        More {
            head: $v0,
            tail: Last { value: $v1 },
        }
    };
    ($v0:ident,$v1:ident,$v2:ident) => {
        More {
            head: $v0,
            tail: More {
                head: $v1,
                tail: Last { value: $v2 },
            },
        }
    };
    ($v0:ident,$v1:ident,$v2:ident,$v3:ident) => {
        More {
            head: $v0,
            tail: More {
                head: $v1,
                tail: More {
                    head: $v2,
                    tail: Last { value: $v3 },
                },
            },
        }
    };
    ($v0:ident,$v1:ident,$v2:ident,$v3:ident,$v4:ident) => {
        More {
            head: $v0,
            tail: More {
                head: $v1,
                tail: More {
                    head: $v2,
                    tail: More {
                        head: $v3,
                        tail: Last { value: $v4 },
                    },
                },
            },
        }
    };
    ($v0:ident,$v1:ident,$v2:ident,$v3:ident,$v4:ident,$v5:ident) => {
        More {
            head: $v0,
            tail: More {
                head: $v1,
                tail: More {
                    head: $v2,
                    tail: More {
                        head: $v3,
                        tail: More {
                            head: $v4,
                            tail: Last { value: $v5 },
                        },
                    },
                },
            },
        }
    };
    ($v0:ident,$v1:ident,$v2:ident,$v3:ident,$v4:ident,$v5:ident,$v6:ident) => {
        More {
            head: $v0,
            tail: More {
                head: $v1,
                tail: More {
                    head: $v2,
                    tail: More {
                        head: $v3,
                        tail: More {
                            head: $v4,
                            tail: More {
                                head: $v5,
                                tail: Last { value: $v6 },
                            },
                        },
                    },
                },
            },
        }
    };
}

macro_rules! impl_multi_reduce_dispatch {
    (
        $read_arity:ty,$storage_arity:ty,$eval:ident,$eval_kernel:ident,
        $storage_kernel:ident,$finalize_kernel:ident,$read_env:ty,$leaves:ty,
        $load_trait:ident,$store_trait:ident;
        [$( $leaf:ident:$read_index:literal ),+];
        [$( $out_ty:ident:$current:ident:$next:ident:$init_handle:ident:$output_handle:ident:$value:ident:$first_field:ident $(.$rest_field:ident)* ),+]
    ) => {
        impl<R, Input, Item, Op, $( $leaf, )+ $( $out_ty ),+>
            ReduceDispatch<R, Input, Item, Op, $read_env>
            for Dispatch<$read_arity, $storage_arity>
        where
            R: Runtime,
            Item: StorageLayout<StorageArity = $storage_arity, StorageLeaves = $leaves>
                + Send
                + Sync
                + 'static,
            Item::DeviceLayout: Decompose<Item, Leaves = $leaves>
                + Recompose<Item, Leaves = $leaves>,
            $( $leaf: MStorageElement, )+
            $( $out_ty: MStorageElement, )+
            $leaves: $load_trait<$( $out_ty ),+>
                + $store_trait<$( $out_ty ),+>
                + Send
                + Sync
                + 'static,
            Op: ReductionOp<Item>,
            Input: ReadExpression<Item = Item, ReadArity = $read_arity>
                + BindSlots<Env0, NextEnv = $read_env>
                + LowerReadExpression<Slots = $read_env>
                + StageRead<R, Env0>,
            Input::DeviceExpr: $eval<Item, $( $leaf ),+>,
        {
            fn execute(exec: &Executor<R>, input: &Input, init: Item) -> Result<Item, Error> {
                let len = input.logical_len()?;
                if len == 0 {
                    return Ok(init);
                }
                let client = exec.client();
                let mut bindings = StagedBindings::new();
                input.stage_at(client, exec.id(), &mut bindings)?;
                let blocks = pass_block_count(len);
                let offsets = client.create_from_slice(u32::as_bytes(&bindings.offsets));
                let len_handle = client.create_from_slice(u32::as_bytes(&[checked_u32(len)?]));
                let zero_offsets = vec![$({ let _ = stringify!($out_ty); 0u32 }),+];
                let zero_handle = client.create_from_slice(u32::as_bytes(&zero_offsets));
                $( let mut $current = client.empty(blocks * size_of::<$out_ty>()); )+
                unsafe {
                    $eval_kernel::launch_unchecked::<
                        Item,
                        $( $leaf, )+
                        $( $out_ty, )+
                        $leaves,
                        Item::DeviceLayout,
                        Input::DeviceExpr,
                        Op,
                        R,
                    >(
                        client,
                        cube_count_1d(blocks)?,
                        CubeDim::new_1d(BLOCK_SIZE),
                        $( BufferArg::from_raw_parts(bindings.slots[$read_index].0.clone(), bindings.slots[$read_index].1), )+
                        BufferArg::from_raw_parts(offsets, bindings.offsets.len()),
                        BufferArg::from_raw_parts(len_handle, 1),
                        BufferArg::from_raw_parts(zero_handle.clone(), zero_offsets.len()),
                        $( BufferArg::from_raw_parts($current.clone(), blocks), )+
                    );
                }

                let mut current_len = blocks;
                while current_len > 1 {
                    let next_len = pass_block_count(current_len);
                    let current_len_handle = client.create_from_slice(u32::as_bytes(&[checked_u32(current_len)?]));
                    $( let $next = client.empty(next_len * size_of::<$out_ty>()); )+
                    unsafe {
                        $storage_kernel::launch_unchecked::<
                            Item,
                            $( $out_ty, )+
                            $leaves,
                            Item::DeviceLayout,
                            Op,
                            R,
                        >(
                            client,
                            cube_count_1d(next_len)?,
                            CubeDim::new_1d(BLOCK_SIZE),
                            $( BufferArg::from_raw_parts($current.clone(), current_len), )+
                            BufferArg::from_raw_parts(current_len_handle, 1),
                            BufferArg::from_raw_parts(zero_handle.clone(), zero_offsets.len()),
                            $( BufferArg::from_raw_parts($next.clone(), next_len), )+
                        );
                    }
                    $( $current = $next; )+
                    current_len = next_len;
                }

                let init_leaves = init.into_storage_leaves();
                $(
                    let $init_handle = client.create_from_slice($out_ty::as_bytes(&[
                        leaf_value!(init_leaves; $first_field $(.$rest_field)*)
                    ]));
                    let $output_handle = client.empty(size_of::<$out_ty>());
                )+
                unsafe {
                    $finalize_kernel::launch_unchecked::<
                        Item,
                        $( $out_ty, )+
                        $leaves,
                        Item::DeviceLayout,
                        Op,
                        R,
                    >(
                        client,
                        CubeCount::new_single(),
                        CubeDim::new_1d(1),
                        $( BufferArg::from_raw_parts($current, 1), )+
                        $( BufferArg::from_raw_parts($init_handle, 1), )+
                        BufferArg::from_raw_parts(zero_handle, zero_offsets.len()),
                        $( BufferArg::from_raw_parts($output_handle.clone(), 1), )+
                    );
                }
                $(
                    let bytes = client.read_one($output_handle).map_err(|err| Error::Launch {
                        message: format!("{err:?}"),
                    })?;
                    let $value = $out_ty::from_bytes(&bytes)[0];
                )+
                Ok(Item::from_storage_leaves(build_leaf_list!($( $value ),+)))
            }
        }
    };
}

macro_rules! impl_multi_reduce_dispatches_for_eval {
    ($arity:ty,$eval:ident,$env:ty; [$( $leaf:ident:$index:literal ),+]; [$k2:ident,$k3:ident,$k4:ident,$k5:ident,$k6:ident,$k7:ident]) => {
        impl_multi_reduce_dispatch!($arity,S2,$eval,$k2,reduce_storage_s2,reduce_finalize_s2,$env,More<O0,Last<O1>>,LoadLeaves2,StoreLeaves2; [$($leaf:$index),+]; [O0:current0:next0:init0:output0:value0:head,O1:current1:next1:init1:output1:value1:tail.value]);
        impl_multi_reduce_dispatch!($arity,S3,$eval,$k3,reduce_storage_s3,reduce_finalize_s3,$env,More<O0,More<O1,Last<O2>>>,LoadLeaves3,StoreLeaves3; [$($leaf:$index),+]; [O0:current0:next0:init0:output0:value0:head,O1:current1:next1:init1:output1:value1:tail.head,O2:current2:next2:init2:output2:value2:tail.tail.value]);
        impl_multi_reduce_dispatch!($arity,S4,$eval,$k4,reduce_storage_s4,reduce_finalize_s4,$env,More<O0,More<O1,More<O2,Last<O3>>>>,LoadLeaves4,StoreLeaves4; [$($leaf:$index),+]; [O0:current0:next0:init0:output0:value0:head,O1:current1:next1:init1:output1:value1:tail.head,O2:current2:next2:init2:output2:value2:tail.tail.head,O3:current3:next3:init3:output3:value3:tail.tail.tail.value]);
        impl_multi_reduce_dispatch!($arity,S5,$eval,$k5,reduce_storage_s5,reduce_finalize_s5,$env,More<O0,More<O1,More<O2,More<O3,Last<O4>>>>>,LoadLeaves5,StoreLeaves5; [$($leaf:$index),+]; [O0:current0:next0:init0:output0:value0:head,O1:current1:next1:init1:output1:value1:tail.head,O2:current2:next2:init2:output2:value2:tail.tail.head,O3:current3:next3:init3:output3:value3:tail.tail.tail.head,O4:current4:next4:init4:output4:value4:tail.tail.tail.tail.value]);
        impl_multi_reduce_dispatch!($arity,S6,$eval,$k6,reduce_storage_s6,reduce_finalize_s6,$env,More<O0,More<O1,More<O2,More<O3,More<O4,Last<O5>>>>>>,LoadLeaves6,StoreLeaves6; [$($leaf:$index),+]; [O0:current0:next0:init0:output0:value0:head,O1:current1:next1:init1:output1:value1:tail.head,O2:current2:next2:init2:output2:value2:tail.tail.head,O3:current3:next3:init3:output3:value3:tail.tail.tail.head,O4:current4:next4:init4:output4:value4:tail.tail.tail.tail.head,O5:current5:next5:init5:output5:value5:tail.tail.tail.tail.tail.value]);
        impl_multi_reduce_dispatch!($arity,S7,$eval,$k7,reduce_storage_s7,reduce_finalize_s7,$env,More<O0,More<O1,More<O2,More<O3,More<O4,More<O5,Last<O6>>>>>>>,LoadLeaves7,StoreLeaves7; [$($leaf:$index),+]; [O0:current0:next0:init0:output0:value0:head,O1:current1:next1:init1:output1:value1:tail.head,O2:current2:next2:init2:output2:value2:tail.tail.head,O3:current3:next3:init3:output3:value3:tail.tail.tail.head,O4:current4:next4:init4:output4:value4:tail.tail.tail.tail.head,O5:current5:next5:init5:output5:value5:tail.tail.tail.tail.tail.head,O6:current6:next6:init6:output6:value6:tail.tail.tail.tail.tail.tail.value]);
    };
}

impl_multi_reduce_dispatches_for_eval!(A1,Eval1,Env1<L0>; [L0:0]; [reduce_eval1_s2,reduce_eval1_s3,reduce_eval1_s4,reduce_eval1_s5,reduce_eval1_s6,reduce_eval1_s7]);
impl_multi_reduce_dispatches_for_eval!(A2,Eval2,Env2<L0,L1>; [L0:0,L1:1]; [reduce_eval2_s2,reduce_eval2_s3,reduce_eval2_s4,reduce_eval2_s5,reduce_eval2_s6,reduce_eval2_s7]);
impl_multi_reduce_dispatches_for_eval!(A3,Eval3,Env3<L0,L1,L2>; [L0:0,L1:1,L2:2]; [reduce_eval3_s2,reduce_eval3_s3,reduce_eval3_s4,reduce_eval3_s5,reduce_eval3_s6,reduce_eval3_s7]);
impl_multi_reduce_dispatches_for_eval!(A4,Eval4,Env4<L0,L1,L2,L3>; [L0:0,L1:1,L2:2,L3:3]; [reduce_eval4_s2,reduce_eval4_s3,reduce_eval4_s4,reduce_eval4_s5,reduce_eval4_s6,reduce_eval4_s7]);
impl_multi_reduce_dispatches_for_eval!(A5,Eval5,Env5<L0,L1,L2,L3,L4>; [L0:0,L1:1,L2:2,L3:3,L4:4]; [reduce_eval5_s2,reduce_eval5_s3,reduce_eval5_s4,reduce_eval5_s5,reduce_eval5_s6,reduce_eval5_s7]);
impl_multi_reduce_dispatches_for_eval!(A6,Eval6,Env6<L0,L1,L2,L3,L4,L5>; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5]; [reduce_eval6_s2,reduce_eval6_s3,reduce_eval6_s4,reduce_eval6_s5,reduce_eval6_s6,reduce_eval6_s7]);
impl_multi_reduce_dispatches_for_eval!(A7,Eval7,Env7<L0,L1,L2,L3,L4,L5,L6>; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5,L6:6]; [reduce_eval7_s2,reduce_eval7_s3,reduce_eval7_s4,reduce_eval7_s5,reduce_eval7_s6,reduce_eval7_s7]);
impl_multi_reduce_dispatches_for_eval!(A8,Eval8,crate::read::Env8<L0,L1,L2,L3,L4,L5,L6,L7>; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5,L6:6,L7:7]; [reduce_eval8_s2,reduce_eval8_s3,reduce_eval8_s4,reduce_eval8_s5,reduce_eval8_s6,reduce_eval8_s7]);

/// Reduces all input items, starting from `init`.
pub(crate) fn reduce<R, Input, Op>(
    exec: &Executor<R>,
    input: Input,
    init: Input::Item,
    _op: Op,
) -> Result<Input::Item, Error>
where
    R: Runtime,
    Input: ReadExpression + LowerReadExpression + StageRead<R, Env0>,
    Input::Item: StorageLayout,
    Op: ReductionOp<Input::Item>,
    Dispatch<Input::ReadArity, <Input::Item as StorageLayout>::StorageArity>:
        ReduceDispatch<R, Input, Input::Item, Op, Input::Slots>,
{
    <Dispatch<Input::ReadArity, <Input::Item as StorageLayout>::StorageArity> as ReduceDispatch<
        R,
        Input,
        Input::Item,
        Op,
        Input::Slots,
    >>::execute(exec, &input, init)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::Identity;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    struct Sum;

    #[cubecl::cube]
    impl ReductionOp<u32> for Sum {
        fn apply(lhs: u32, rhs: u32) -> u32 {
            lhs + rhs
        }
    }

    struct Double;

    #[cubecl::cube]
    impl UnaryOp<u32> for Double {
        type Output = u32;

        fn apply(input: u32) -> u32 {
            input * 2
        }
    }

    struct AddPair;

    #[cubecl::cube]
    impl UnaryOp<(u32, u32)> for AddPair {
        type Output = u32;

        fn apply(input: (u32, u32)) -> u32 {
            input.0 + input.1
        }
    }

    struct AddNestedTriple;

    #[cubecl::cube]
    impl UnaryOp<((u32, u32), u32)> for AddNestedTriple {
        type Output = u32;

        fn apply(input: ((u32, u32), u32)) -> u32 {
            input.0.0 + input.0.1 + input.1
        }
    }

    struct AddNestedFour;

    #[cubecl::cube]
    impl UnaryOp<(((u32, u32), u32), u32)> for AddNestedFour {
        type Output = u32;

        fn apply(input: (((u32, u32), u32), u32)) -> u32 {
            input.0.0.0 + input.0.0.1 + input.0.1 + input.1
        }
    }

    type Seven = (u32, (u32, (u32, (u32, (u32, (u32, u32))))));
    struct AddSeven;

    #[cubecl::cube]
    impl ReductionOp<Seven> for AddSeven {
        fn apply(lhs: Seven, rhs: Seven) -> Seven {
            (
                lhs.0 + rhs.0,
                (
                    lhs.1.0 + rhs.1.0,
                    (
                        lhs.1.1.0 + rhs.1.1.0,
                        (
                            lhs.1.1.1.0 + rhs.1.1.1.0,
                            (
                                lhs.1.1.1.1.0 + rhs.1.1.1.1.0,
                                (
                                    lhs.1.1.1.1.1.0 + rhs.1.1.1.1.1.0,
                                    lhs.1.1.1.1.1.1 + rhs.1.1.1.1.1.1,
                                ),
                            ),
                        ),
                    ),
                ),
            )
        }
    }

    fn executor() -> Executor<WgpuRuntime> {
        Executor::new(WgpuDevice::DefaultDevice)
    }

    #[test]
    fn dispatch_a1_storage1_fuses_column_transform_reduce() {
        let exec = executor();
        let len = 4097;
        let values = exec.to_device(&vec![1_u32; len]);
        let input = Transform::new(values.column(), Double);

        let output = reduce(&exec, input, 7, Sum).unwrap();
        assert_eq!(output, 7 + 2 * len as u32);
    }

    #[test]
    fn dispatch_a2_storage1_fuses_binary_zip_transform_reduce() {
        let exec = executor();
        let len = 4097;
        let left = exec.to_device(&vec![1_u32; len]);
        let right = exec.to_device(&vec![2_u32; len]);
        let input = Transform::new(Zip::new(left.column(), right.column()), AddPair);

        let output = reduce(&exec, input, 0, Sum).unwrap();
        assert_eq!(output, 3 * len as u32);
    }

    #[test]
    fn dispatch_a3_storage1_preserves_nested_zip_semantics() {
        let exec = executor();
        let len = 4097;
        let first = exec.to_device(&vec![1_u32; len]);
        let second = exec.to_device(&vec![2_u32; len]);
        let third = exec.to_device(&vec![3_u32; len]);
        let input = Transform::new(
            Zip::new(Zip::new(first.column(), second.column()), third.column()),
            AddNestedTriple,
        );

        let output = reduce(&exec, input, 0, Sum).unwrap();
        assert_eq!(output, 6 * len as u32);
    }

    #[test]
    fn empty_reduce_returns_init_without_launching_or_staging() {
        let exec = executor();
        let input = Transform::new(Column::<u32>::new(), Identity);
        assert_eq!(reduce(&exec, input, 42, Sum).unwrap(), 42);
    }

    #[test]
    fn zip_length_mismatch_is_rejected_before_launch() {
        let exec = executor();
        let left = exec.to_device(&[1_u32, 2]);
        let right = exec.to_device(&[3_u32]);
        let input = Transform::new(Zip::new(left.column(), right.column()), AddPair);
        assert_eq!(
            reduce(&exec, input, 0, Sum),
            Err(Error::LengthMismatch { left: 2, right: 1 })
        );
    }

    type FourColumns = Zip<Zip<Zip<Column<u32>, Column<u32>>, Column<u32>>, Column<u32>>;
    type FourInput = Transform<FourColumns, AddNestedFour>;
    type FourExpr = <FourInput as LowerReadExpression>::DeviceExpr;

    #[test]
    fn dispatch_a4_storage1_is_a_regular_evaluator_path() {
        let exec = executor();
        let columns: Vec<_> = (1_u32..=4)
            .map(|value| exec.to_device(&vec![value; 513]))
            .collect();
        let input = Transform::new(
            Zip::new(
                Zip::new(
                    Zip::new(columns[0].column(), columns[1].column()),
                    columns[2].column(),
                ),
                columns[3].column(),
            ),
            AddNestedFour,
        );
        assert_eq!(reduce(&exec, input, 5, Sum).unwrap(), 5 + 10 * 513);
    }

    #[test]
    fn dispatch_a8_storage7_reduces_semantic_item_with_physical_leaf_partials() {
        let exec = executor();
        let len = TILE_SIZE * 2 + 17;
        let columns: Vec<_> = (1_u32..=7)
            .map(|value| exec.to_device(&vec![value; len]))
            .collect();
        let seven = Zip::new(
            columns[0].column(),
            Zip::new(
                columns[1].column(),
                Zip::new(
                    columns[2].column(),
                    Zip::new(
                        columns[3].column(),
                        Zip::new(
                            columns[4].column(),
                            Zip::new(columns[5].column(), columns[6].column()),
                        ),
                    ),
                ),
            ),
        );
        let input = Permute::new(seven, Counting::new(0, len));
        let init: Seven = (10, (20, (30, (40, (50, (60, 70))))));

        let output = reduce(&exec, input, init, AddSeven).unwrap();
        assert_eq!(
            output,
            (
                10 + len as u32,
                (
                    20 + 2 * len as u32,
                    (
                        30 + 3 * len as u32,
                        (
                            40 + 4 * len as u32,
                            (
                                50 + 5 * len as u32,
                                (60 + 6 * len as u32, 70 + 7 * len as u32),
                            ),
                        ),
                    ),
                ),
            )
        );
    }

    #[allow(dead_code)]
    fn a4_expression_still_has_a_valid_evaluator()
    where
        FourExpr: crate::eval::Eval4<u32, u32, u32, u32, u32>,
    {
    }
}
