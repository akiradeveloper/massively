use cubecl::prelude::Runtime;

use crate::{Error, Executor, Zip};

pub enum PairResult {
    Bool(bool),
    Index(Option<u32>),
}

pub struct Write<Output, Slots> {
    output: Output,
    _slots: core::marker::PhantomData<fn() -> Slots>,
}

pub struct Read<Input, Slots> {
    input: Input,
    _slots: core::marker::PhantomData<fn() -> Slots>,
}

impl<Input, Slots> Read<Input, Slots> {
    pub fn new(input: Input) -> Self {
        Self {
            input,
            _slots: core::marker::PhantomData,
        }
    }
}

pub trait AnyRead<R: Runtime>: Sized {
    type Item: crate::api::iter::MItem<R>;

    fn normalize(
        self,
        exec: &Executor<R>,
    ) -> Result<<Self::Item as crate::CanonicalAlloc<R>>::CanonicalStorage, Error>;
}

impl<R, Input, Slots> AnyRead<R> for Read<Input, Slots>
where
    R: Runtime,
    Input: crate::read::ReadExpression + crate::read::LowerReadExpression<Slots = Slots>,
    Input::Item: crate::api::iter::MItem<R>,
    Slots: KernelRead<R, Input>,
{
    type Item = Input::Item;

    fn normalize(
        self,
        exec: &Executor<R>,
    ) -> Result<<Self::Item as crate::CanonicalAlloc<R>>::CanonicalStorage, Error> {
        <Slots as KernelRead<R, Input>>::normalize(self.input, exec)
    }
}

pub fn run_pair<R, Item, Left, Right, Op>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    op: Op,
    mode: u8,
) -> Result<PairResult, Error>
where
    R: Runtime,
    Item: crate::api::iter::MItem<R>,
    Item::StorageLeaves: KernelItem<R, Item>,
    Left: AnyRead<R, Item = Item>,
    Right: AnyRead<R, Item = Item>,
    Op: crate::BinaryPredicateOp<Item>,
{
    let left = left.normalize(exec)?;
    let right = right.normalize(exec)?;
    <Item::StorageLeaves as KernelItem<R, Item>>::pair(exec, &left, &right, op, mode)
}

impl<Output, Slots> Write<Output, Slots> {
    pub fn new(output: Output) -> Self {
        Self {
            output,
            _slots: core::marker::PhantomData,
        }
    }
}

macro_rules! scan_method_decl {
        ($name:ident, $arity:ty, $env:ty, $eval:ident, [$($leaf:ident),+]) => {
            fn $name<Input, Op, $($leaf),+>(
                self,
                exec: &Executor<R>,
                input: Input,
                op: Op,
            ) -> Result<(), Error>
            where
                $($leaf: crate::MStorageElement,)+
                Input: crate::read::ReadExpression<Item = Self::Item, ReadArity = $arity>
                    + crate::read::BindSlots<crate::read::Env0, NextEnv = $env>
                    + crate::read::LowerReadExpression<Slots = $env>
                    + crate::reduce::StageRead<R, crate::read::Env0>,
                Input::DeviceExpr: crate::eval::$eval<Self::Item, $($leaf),+>,
                Op: crate::ReductionOp<Self::Item>;
        };
    }

macro_rules! sort_method_decl {
        ($name:ident, $arity:ty, $env:ty, $eval:ident, [$($leaf:ident),+]) => {
            fn $name<Input, Less, $($leaf),+>(
                self,
                exec: &Executor<R>,
                input: Input,
                less: Less,
            ) -> Result<(), Error>
            where
                $($leaf: crate::MStorageElement,)+
                Input: crate::read::ReadExpression<Item = Self::Item, ReadArity = $arity>
                    + crate::read::BindSlots<crate::read::Env0, NextEnv = $env>
                    + crate::read::LowerReadExpression<Slots = $env>
                    + crate::reduce::StageRead<R, crate::read::Env0>,
                Input::DeviceExpr: crate::eval::$eval<Self::Item, $($leaf),+>,
                Less: crate::BinaryPredicateOp<Self::Item>;
        };
    }

macro_rules! unique_method_decl {
        ($name:ident, $arity:ty, $env:ty, $eval:ident, [$($leaf:ident),+]) => {
            fn $name<Input, Equal, $($leaf),+>(
                self,
                exec: &Executor<R>,
                input: Input,
                equal: Equal,
            ) -> Result<u32, Error>
            where
                $($leaf: crate::MStorageElement,)+
                Input: Clone
                    + crate::read::ReadExpression<Item = Self::Item, ReadArity = $arity>
                    + crate::read::BindSlots<crate::read::Env0, NextEnv = $env>
                    + crate::read::LowerReadExpression<
                        DeviceExpr = <Input as crate::read::BindSlots<crate::read::Env0>>::Expr,
                        Slots = $env,
                    >
                    + crate::reduce::StageRead<R, crate::read::Env0>,
                Input::DeviceExpr: crate::eval::$eval<Self::Item, $($leaf),+>,
                Equal: crate::BinaryPredicateOp<Self::Item>;
        };
    }

macro_rules! select_method_decl {
        ($name:ident, $arity:ty, $env:ty, $eval:ident, [$($leaf:ident),+]) => {
            fn $name<Input, $($leaf),+>(
                self,
                exec: &Executor<R>,
                input: Input,
                flags: crate::Column<u32>,
                invert: bool,
            ) -> Result<u32, Error>
            where
                $($leaf: crate::MStorageElement,)+
                Input: Clone
                    + crate::read::ReadExpression<Item = Self::Item, ReadArity = $arity>
                    + crate::read::BindSlots<crate::read::Env0, NextEnv = $env>
                    + crate::read::LowerReadExpression<Slots = $env>
                    + crate::reduce::StageRead<R, crate::read::Env0>,
                Input::DeviceExpr: crate::eval::$eval<Self::Item, $($leaf),+>;
        };
    }

macro_rules! partition_method_decl {
        ($name:ident, $arity:ty, $env:ty, $eval:ident, [$($leaf:ident),+]) => {
            fn $name<Input, Pred, $($leaf),+>(
                self,
                exec: &Executor<R>,
                input: Input,
                pred: Pred,
            ) -> Result<u32, Error>
            where
                $($leaf: crate::MStorageElement,)+
                Input: Clone
                    + crate::read::ReadExpression<Item = Self::Item, ReadArity = $arity>
                    + crate::read::BindSlots<crate::read::Env0, NextEnv = $env>
                    + crate::read::LowerReadExpression<
                        DeviceExpr = <Input as crate::read::BindSlots<crate::read::Env0>>::Expr,
                        Slots = $env,
                    >
                    + crate::reduce::StageRead<R, crate::read::Env0>,
                Input::DeviceExpr: crate::eval::$eval<Self::Item, $($leaf),+>,
                Pred: crate::PredicateOp<Self::Item>;
        };
    }

macro_rules! indexed_method_decl {
        ($name:ident, $arity:ty, $env:ty, $eval:ident, [$($leaf:ident),+]) => {
            fn $name<Input, $($leaf),+>(
                self,
                exec: &Executor<R>,
                input: Input,
                indices: crate::Column<u32>,
                flags: Option<crate::Column<u32>>,
                scatter: bool,
            ) -> Result<(), Error>
            where
                $($leaf: crate::MStorageElement,)+
                Input: Clone
                    + crate::read::ReadExpression<Item = Self::Item, ReadArity = $arity>
                    + crate::read::BindSlots<crate::read::Env0, NextEnv = $env>
                    + crate::read::LowerReadExpression<
                        DeviceExpr = <Input as crate::read::BindSlots<crate::read::Env0>>::Expr,
                        Slots = $env,
                    >
                    + crate::reduce::StageRead<R, crate::read::Env0>,
                Input::DeviceExpr: crate::eval::$eval<Self::Item, $($leaf),+>;
        };
    }

macro_rules! transform_where_method_decl {
        ($name:ident, $arity:ty, $env:ty, $eval:ident, [$($leaf:ident),+]) => {
            fn $name<Input, Op, $($leaf),+>(
                self,
                exec: &Executor<R>,
                input: Input,
                op: Op,
                flags: crate::Column<u32>,
            ) -> Result<(), Error>
            where
                $($leaf: crate::MStorageElement,)+
                Input: Clone
                    + crate::read::ReadExpression<ReadArity = $arity>
                    + crate::read::BindSlots<crate::read::Env0, NextEnv = $env>
                    + crate::read::LowerReadExpression<
                        DeviceExpr = <Input as crate::read::BindSlots<crate::read::Env0>>::Expr,
                        Slots = $env,
                    >
                    + crate::reduce::StageRead<R, crate::read::Env0>,
                Input::DeviceExpr: crate::eval::$eval<Input::Item, $($leaf),+>,
                Op: crate::UnaryOp<Input::Item>,
                Self::Item: crate::WriteFrom<Op::Output>;
        };
    }

macro_rules! exclusive_method_decl {
        ($name:ident, $arity:ty, $env:ty, $eval:ident, [$($leaf:ident),+]) => {
            fn $name<Input, Op, $($leaf),+>(
                self,
                exec: &Executor<R>,
                input: Input,
                init: Self::Item,
                op: Op,
            ) -> Result<(), Error>
            where
                $($leaf: crate::MStorageElement,)+
                Input: Clone
                    + crate::read::ReadExpression<Item = Self::Item, ReadArity = $arity>
                    + crate::read::BindSlots<crate::read::Env0, NextEnv = $env>
                    + crate::read::LowerReadExpression<
                        DeviceExpr = <Input as crate::read::BindSlots<crate::read::Env0>>::Expr,
                        Slots = $env,
                    >
                    + crate::reduce::StageRead<R, crate::read::Env0>,
                Input::DeviceExpr: crate::eval::$eval<Self::Item, $($leaf),+>,
                Op: crate::ReductionOp<Self::Item>;
        };
    }

pub trait KernelWrite<R: Runtime>: Sized {
    type Item: crate::api::iter::MItem<R>;
    type Output;

    fn fill(self, exec: &Executor<R>, value: Self::Item) -> Result<(), Error>;

    fn replace(
        self,
        exec: &Executor<R>,
        value: Self::Item,
        flags: crate::Column<u32>,
    ) -> Result<(), Error>;

    fn materialize_storage(
        self,
        exec: &Executor<R>,
        input: &<Self::Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
    ) -> Result<(), Error>;

    fn gather_storage(
        self,
        exec: &Executor<R>,
        input: &<Self::Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
        indices: crate::Column<u32>,
    ) -> Result<(), Error>;

    fn select_storage(
        self,
        exec: &Executor<R>,
        input: &<Self::Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
        flags: crate::DeviceVec<R, u32>,
    ) -> Result<u32, Error>;

    fn select_storage_control(
        self,
        exec: &Executor<R>,
        input: &<Self::Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
        control: &crate::selection::SelectionControl<R>,
    ) -> Result<u32, Error>;

    fn concat_storage(
        self,
        exec: &Executor<R>,
        left: &<Self::Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
        right: &<Self::Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
        control: &crate::merge::MergeControl<R>,
    ) -> Result<(), Error>;

    fn set_storage<Less>(
        self,
        exec: &Executor<R>,
        left: &<Self::Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
        right: &<Self::Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
        less: Less,
        mode: u8,
    ) -> Result<u32, Error>
    where
        Less: crate::BinaryPredicateOp<Self::Item>;

    fn materialize_a1<Input, L0>(self, exec: &Executor<R>, input: Input) -> Result<(), Error>
    where
        L0: crate::MStorageElement,
        Input: crate::read::ReadExpression<ReadArity = crate::A1>
            + crate::read::LowerReadExpression<Slots = crate::read::Env1<L0>>
            + crate::reduce::StageRead<R, crate::read::Env0>,
        Input::Item: crate::StorageLayout,
        Self::Item: crate::WriteFrom<Input::Item>,
        Input::DeviceExpr: crate::eval::Eval1<Input::Item, L0>;

    fn materialize_a2<Input, L0, L1>(self, exec: &Executor<R>, input: Input) -> Result<(), Error>
    where
        L0: crate::MStorageElement,
        L1: crate::MStorageElement,
        Input: crate::read::ReadExpression<ReadArity = crate::A2>
            + crate::read::LowerReadExpression<Slots = crate::read::Env2<L0, L1>>
            + crate::reduce::StageRead<R, crate::read::Env0>,
        Input::Item: crate::StorageLayout,
        Self::Item: crate::WriteFrom<Input::Item>,
        Input::DeviceExpr: crate::eval::Eval2<Input::Item, L0, L1>;

    fn materialize_a3<Input, L0, L1, L2>(
        self,
        exec: &Executor<R>,
        input: Input,
    ) -> Result<(), Error>
    where
        L0: crate::MStorageElement,
        L1: crate::MStorageElement,
        L2: crate::MStorageElement,
        Input: crate::read::ReadExpression<ReadArity = crate::A3>
            + crate::read::LowerReadExpression<Slots = crate::read::Env3<L0, L1, L2>>
            + crate::reduce::StageRead<R, crate::read::Env0>,
        Input::Item: crate::StorageLayout,
        Self::Item: crate::WriteFrom<Input::Item>,
        Input::DeviceExpr: crate::eval::Eval3<Input::Item, L0, L1, L2>;

    fn materialize_a4<Input, L0, L1, L2, L3>(
        self,
        exec: &Executor<R>,
        input: Input,
    ) -> Result<(), Error>
    where
        L0: crate::MStorageElement,
        L1: crate::MStorageElement,
        L2: crate::MStorageElement,
        L3: crate::MStorageElement,
        Input: crate::read::ReadExpression<ReadArity = crate::A4>
            + crate::read::LowerReadExpression<Slots = crate::read::Env4<L0, L1, L2, L3>>
            + crate::reduce::StageRead<R, crate::read::Env0>,
        Input::Item: crate::StorageLayout,
        Self::Item: crate::WriteFrom<Input::Item>,
        Input::DeviceExpr: crate::eval::Eval4<Input::Item, L0, L1, L2, L3>;

    fn materialize_a5<Input, L0, L1, L2, L3, L4>(
        self,
        exec: &Executor<R>,
        input: Input,
    ) -> Result<(), Error>
    where
        L0: crate::MStorageElement,
        L1: crate::MStorageElement,
        L2: crate::MStorageElement,
        L3: crate::MStorageElement,
        L4: crate::MStorageElement,
        Input: crate::read::ReadExpression<ReadArity = crate::A5>
            + crate::read::LowerReadExpression<Slots = crate::read::Env5<L0, L1, L2, L3, L4>>
            + crate::reduce::StageRead<R, crate::read::Env0>,
        Input::Item: crate::StorageLayout,
        Self::Item: crate::WriteFrom<Input::Item>,
        Input::DeviceExpr: crate::eval::Eval5<Input::Item, L0, L1, L2, L3, L4>;

    fn materialize_a6<Input, L0, L1, L2, L3, L4, L5>(
        self,
        exec: &Executor<R>,
        input: Input,
    ) -> Result<(), Error>
    where
        L0: crate::MStorageElement,
        L1: crate::MStorageElement,
        L2: crate::MStorageElement,
        L3: crate::MStorageElement,
        L4: crate::MStorageElement,
        L5: crate::MStorageElement,
        Input: crate::read::ReadExpression<ReadArity = crate::A6>
            + crate::read::LowerReadExpression<Slots = crate::read::Env6<L0, L1, L2, L3, L4, L5>>
            + crate::reduce::StageRead<R, crate::read::Env0>,
        Input::Item: crate::StorageLayout,
        Self::Item: crate::WriteFrom<Input::Item>,
        Input::DeviceExpr: crate::eval::Eval6<Input::Item, L0, L1, L2, L3, L4, L5>;

    fn materialize_a7<Input, L0, L1, L2, L3, L4, L5, L6>(
        self,
        exec: &Executor<R>,
        input: Input,
    ) -> Result<(), Error>
    where
        L0: crate::MStorageElement,
        L1: crate::MStorageElement,
        L2: crate::MStorageElement,
        L3: crate::MStorageElement,
        L4: crate::MStorageElement,
        L5: crate::MStorageElement,
        L6: crate::MStorageElement,
        Input: crate::read::ReadExpression<ReadArity = crate::A7>
            + crate::read::LowerReadExpression<Slots = crate::read::Env7<L0, L1, L2, L3, L4, L5, L6>>
            + crate::reduce::StageRead<R, crate::read::Env0>,
        Input::Item: crate::StorageLayout,
        Self::Item: crate::WriteFrom<Input::Item>,
        Input::DeviceExpr: crate::eval::Eval7<Input::Item, L0, L1, L2, L3, L4, L5, L6>;

    fn materialize_a8<Input, L0, L1, L2, L3, L4, L5, L6, L7>(
        self,
        exec: &Executor<R>,
        input: Input,
    ) -> Result<(), Error>
    where
        L0: crate::MStorageElement,
        L1: crate::MStorageElement,
        L2: crate::MStorageElement,
        L3: crate::MStorageElement,
        L4: crate::MStorageElement,
        L5: crate::MStorageElement,
        L6: crate::MStorageElement,
        L7: crate::MStorageElement,
        Input: crate::read::ReadExpression<ReadArity = crate::A8>
            + crate::read::LowerReadExpression<
                Slots = crate::read::Env8<L0, L1, L2, L3, L4, L5, L6, L7>,
            > + crate::reduce::StageRead<R, crate::read::Env0>,
        Input::Item: crate::StorageLayout,
        Self::Item: crate::WriteFrom<Input::Item>,
        Input::DeviceExpr: crate::eval::Eval8<Input::Item, L0, L1, L2, L3, L4, L5, L6, L7>;

    scan_method_decl!(
        inclusive_scan_a1,
        crate::A1,
        crate::read::Env1<L0>,
        Eval1,
        [L0]
    );
    scan_method_decl!(inclusive_scan_a2, crate::A2, crate::read::Env2<L0, L1>, Eval2, [L0, L1]);
    scan_method_decl!(inclusive_scan_a3, crate::A3, crate::read::Env3<L0, L1, L2>, Eval3, [L0, L1, L2]);
    scan_method_decl!(inclusive_scan_a4, crate::A4, crate::read::Env4<L0, L1, L2, L3>, Eval4, [L0, L1, L2, L3]);
    scan_method_decl!(inclusive_scan_a5, crate::A5, crate::read::Env5<L0, L1, L2, L3, L4>, Eval5, [L0, L1, L2, L3, L4]);
    scan_method_decl!(inclusive_scan_a6, crate::A6, crate::read::Env6<L0, L1, L2, L3, L4, L5>, Eval6, [L0, L1, L2, L3, L4, L5]);
    scan_method_decl!(inclusive_scan_a7, crate::A7, crate::read::Env7<L0, L1, L2, L3, L4, L5, L6>, Eval7, [L0, L1, L2, L3, L4, L5, L6]);
    scan_method_decl!(inclusive_scan_a8, crate::A8, crate::read::Env8<L0, L1, L2, L3, L4, L5, L6, L7>, Eval8, [L0, L1, L2, L3, L4, L5, L6, L7]);

    sort_method_decl!(sort_a1, crate::A1, crate::read::Env1<L0>, Eval1, [L0]);
    sort_method_decl!(sort_a2, crate::A2, crate::read::Env2<L0, L1>, Eval2, [L0, L1]);
    sort_method_decl!(sort_a3, crate::A3, crate::read::Env3<L0, L1, L2>, Eval3, [L0, L1, L2]);
    sort_method_decl!(sort_a4, crate::A4, crate::read::Env4<L0, L1, L2, L3>, Eval4, [L0, L1, L2, L3]);
    sort_method_decl!(sort_a5, crate::A5, crate::read::Env5<L0, L1, L2, L3, L4>, Eval5, [L0, L1, L2, L3, L4]);
    sort_method_decl!(sort_a6, crate::A6, crate::read::Env6<L0, L1, L2, L3, L4, L5>, Eval6, [L0, L1, L2, L3, L4, L5]);
    sort_method_decl!(sort_a7, crate::A7, crate::read::Env7<L0, L1, L2, L3, L4, L5, L6>, Eval7, [L0, L1, L2, L3, L4, L5, L6]);
    sort_method_decl!(sort_a8, crate::A8, crate::read::Env8<L0, L1, L2, L3, L4, L5, L6, L7>, Eval8, [L0, L1, L2, L3, L4, L5, L6, L7]);

    unique_method_decl!(unique_a1, crate::A1, crate::read::Env1<L0>, Eval1, [L0]);
    unique_method_decl!(unique_a2, crate::A2, crate::read::Env2<L0, L1>, Eval2, [L0, L1]);
    unique_method_decl!(unique_a3, crate::A3, crate::read::Env3<L0, L1, L2>, Eval3, [L0, L1, L2]);
    unique_method_decl!(unique_a4, crate::A4, crate::read::Env4<L0, L1, L2, L3>, Eval4, [L0, L1, L2, L3]);
    unique_method_decl!(unique_a5, crate::A5, crate::read::Env5<L0, L1, L2, L3, L4>, Eval5, [L0, L1, L2, L3, L4]);
    unique_method_decl!(unique_a6, crate::A6, crate::read::Env6<L0, L1, L2, L3, L4, L5>, Eval6, [L0, L1, L2, L3, L4, L5]);
    unique_method_decl!(unique_a7, crate::A7, crate::read::Env7<L0, L1, L2, L3, L4, L5, L6>, Eval7, [L0, L1, L2, L3, L4, L5, L6]);
    unique_method_decl!(unique_a8, crate::A8, crate::read::Env8<L0, L1, L2, L3, L4, L5, L6, L7>, Eval8, [L0, L1, L2, L3, L4, L5, L6, L7]);

    select_method_decl!(select_a1, crate::A1, crate::read::Env1<L0>, Eval1, [L0]);
    select_method_decl!(select_a2, crate::A2, crate::read::Env2<L0, L1>, Eval2, [L0, L1]);
    select_method_decl!(select_a3, crate::A3, crate::read::Env3<L0, L1, L2>, Eval3, [L0, L1, L2]);
    select_method_decl!(select_a4, crate::A4, crate::read::Env4<L0, L1, L2, L3>, Eval4, [L0, L1, L2, L3]);
    select_method_decl!(select_a5, crate::A5, crate::read::Env5<L0, L1, L2, L3, L4>, Eval5, [L0, L1, L2, L3, L4]);
    select_method_decl!(select_a6, crate::A6, crate::read::Env6<L0, L1, L2, L3, L4, L5>, Eval6, [L0, L1, L2, L3, L4, L5]);
    select_method_decl!(select_a7, crate::A7, crate::read::Env7<L0, L1, L2, L3, L4, L5, L6>, Eval7, [L0, L1, L2, L3, L4, L5, L6]);
    select_method_decl!(select_a8, crate::A8, crate::read::Env8<L0, L1, L2, L3, L4, L5, L6, L7>, Eval8, [L0, L1, L2, L3, L4, L5, L6, L7]);

    partition_method_decl!(partition_a1, crate::A1, crate::read::Env1<L0>, Eval1, [L0]);
    partition_method_decl!(partition_a2, crate::A2, crate::read::Env2<L0, L1>, Eval2, [L0, L1]);
    partition_method_decl!(partition_a3, crate::A3, crate::read::Env3<L0, L1, L2>, Eval3, [L0, L1, L2]);
    partition_method_decl!(partition_a4, crate::A4, crate::read::Env4<L0, L1, L2, L3>, Eval4, [L0, L1, L2, L3]);
    partition_method_decl!(partition_a5, crate::A5, crate::read::Env5<L0, L1, L2, L3, L4>, Eval5, [L0, L1, L2, L3, L4]);
    partition_method_decl!(partition_a6, crate::A6, crate::read::Env6<L0, L1, L2, L3, L4, L5>, Eval6, [L0, L1, L2, L3, L4, L5]);
    partition_method_decl!(partition_a7, crate::A7, crate::read::Env7<L0, L1, L2, L3, L4, L5, L6>, Eval7, [L0, L1, L2, L3, L4, L5, L6]);
    partition_method_decl!(partition_a8, crate::A8, crate::read::Env8<L0, L1, L2, L3, L4, L5, L6, L7>, Eval8, [L0, L1, L2, L3, L4, L5, L6, L7]);

    indexed_method_decl!(indexed_a1, crate::A1, crate::read::Env1<L0>, Eval1, [L0]);
    indexed_method_decl!(indexed_a2, crate::A2, crate::read::Env2<L0, L1>, Eval2, [L0, L1]);
    indexed_method_decl!(indexed_a3, crate::A3, crate::read::Env3<L0, L1, L2>, Eval3, [L0, L1, L2]);
    indexed_method_decl!(indexed_a4, crate::A4, crate::read::Env4<L0, L1, L2, L3>, Eval4, [L0, L1, L2, L3]);
    indexed_method_decl!(indexed_a5, crate::A5, crate::read::Env5<L0, L1, L2, L3, L4>, Eval5, [L0, L1, L2, L3, L4]);
    indexed_method_decl!(indexed_a6, crate::A6, crate::read::Env6<L0, L1, L2, L3, L4, L5>, Eval6, [L0, L1, L2, L3, L4, L5]);
    indexed_method_decl!(indexed_a7, crate::A7, crate::read::Env7<L0, L1, L2, L3, L4, L5, L6>, Eval7, [L0, L1, L2, L3, L4, L5, L6]);
    indexed_method_decl!(indexed_a8, crate::A8, crate::read::Env8<L0, L1, L2, L3, L4, L5, L6, L7>, Eval8, [L0, L1, L2, L3, L4, L5, L6, L7]);

    transform_where_method_decl!(
        transform_where_a1,
        crate::A1,
        crate::read::Env1<L0>,
        Eval1,
        [L0]
    );
    transform_where_method_decl!(transform_where_a2, crate::A2, crate::read::Env2<L0, L1>, Eval2, [L0, L1]);
    transform_where_method_decl!(transform_where_a3, crate::A3, crate::read::Env3<L0, L1, L2>, Eval3, [L0, L1, L2]);
    transform_where_method_decl!(transform_where_a4, crate::A4, crate::read::Env4<L0, L1, L2, L3>, Eval4, [L0, L1, L2, L3]);
    transform_where_method_decl!(transform_where_a5, crate::A5, crate::read::Env5<L0, L1, L2, L3, L4>, Eval5, [L0, L1, L2, L3, L4]);
    transform_where_method_decl!(transform_where_a6, crate::A6, crate::read::Env6<L0, L1, L2, L3, L4, L5>, Eval6, [L0, L1, L2, L3, L4, L5]);
    transform_where_method_decl!(transform_where_a7, crate::A7, crate::read::Env7<L0, L1, L2, L3, L4, L5, L6>, Eval7, [L0, L1, L2, L3, L4, L5, L6]);
    transform_where_method_decl!(transform_where_a8, crate::A8, crate::read::Env8<L0, L1, L2, L3, L4, L5, L6, L7>, Eval8, [L0, L1, L2, L3, L4, L5, L6, L7]);

    exclusive_method_decl!(exclusive_a1, crate::A1, crate::read::Env1<L0>, Eval1, [L0]);
    exclusive_method_decl!(exclusive_a2, crate::A2, crate::read::Env2<L0, L1>, Eval2, [L0, L1]);
    exclusive_method_decl!(exclusive_a3, crate::A3, crate::read::Env3<L0, L1, L2>, Eval3, [L0, L1, L2]);
    exclusive_method_decl!(exclusive_a4, crate::A4, crate::read::Env4<L0, L1, L2, L3>, Eval4, [L0, L1, L2, L3]);
    exclusive_method_decl!(exclusive_a5, crate::A5, crate::read::Env5<L0, L1, L2, L3, L4>, Eval5, [L0, L1, L2, L3, L4]);
    exclusive_method_decl!(exclusive_a6, crate::A6, crate::read::Env6<L0, L1, L2, L3, L4, L5>, Eval6, [L0, L1, L2, L3, L4, L5]);
    exclusive_method_decl!(exclusive_a7, crate::A7, crate::read::Env7<L0, L1, L2, L3, L4, L5, L6>, Eval7, [L0, L1, L2, L3, L4, L5, L6]);
    exclusive_method_decl!(exclusive_a8, crate::A8, crate::read::Env8<L0, L1, L2, L3, L4, L5, L6, L7>, Eval8, [L0, L1, L2, L3, L4, L5, L6, L7]);
}

macro_rules! materialize_method {
        ($name:ident, $arity:ty, $env:ty, $eval:ident, [$($leaf:ident),+]; $write_arity:ty, $write_slots:ty) => {
            fn $name<Input, $($leaf),+>(
                self,
                exec: &Executor<R>,
                input: Input,
            ) -> Result<(), Error>
            where
                $($leaf: crate::MStorageElement,)+
                Input: crate::read::ReadExpression<ReadArity = $arity>
                    + crate::read::LowerReadExpression<Slots = $env>
                    + crate::reduce::StageRead<R, crate::read::Env0>,
                Input::Item: crate::StorageLayout,
            Self::Item: crate::WriteFrom<Input::Item>,
                Input::DeviceExpr: crate::eval::$eval<Input::Item, $($leaf),+>,
            {
                <crate::Dispatch<$arity, $write_arity> as crate::transform::MaterializeDispatch<
                    R,
                    Input,
                    Output,
                    $env,
                    $write_slots,
                >>::run(exec, &input, &self.output)
            }
        };
    }

macro_rules! scan_method_impl {
        ($name:ident, $read_arity:ty, $read_env:ty, $eval:ident, [$($leaf:ident),+]; $write_arity:ty, $write_env:ty) => {
            fn $name<Input, Op, $($leaf),+>(
                self,
                exec: &Executor<R>,
                input: Input,
                op: Op,
            ) -> Result<(), Error>
            where
                $($leaf: crate::MStorageElement,)+
                Input: crate::read::ReadExpression<Item = Self::Item, ReadArity = $read_arity>
                    + crate::read::BindSlots<crate::read::Env0, NextEnv = $read_env>
                    + crate::read::LowerReadExpression<Slots = $read_env>
                    + crate::reduce::StageRead<R, crate::read::Env0>,
                Input::DeviceExpr: crate::eval::$eval<Self::Item, $($leaf),+>,
                Op: crate::ReductionOp<Self::Item>,
            {
                <crate::Dispatch<$read_arity, $write_arity> as crate::scan::InclusiveScanDispatch<
                    R,
                    Input,
                    Output,
                    Self::Item,
                    $read_env,
                    $write_env,
                    Op,
                >>::run(exec, &input, op, &self.output)
            }
        };
    }

macro_rules! sort_method_impl {
        ($name:ident, $read_arity:ty, $read_env:ty, $eval:ident, [$($leaf:ident),+]) => {
            fn $name<Input, Less, $($leaf),+>(
                self,
                exec: &Executor<R>,
                input: Input,
                less: Less,
            ) -> Result<(), Error>
            where
                $($leaf: crate::MStorageElement,)+
                Input: crate::read::ReadExpression<Item = Self::Item, ReadArity = $read_arity>
                    + crate::read::BindSlots<crate::read::Env0, NextEnv = $read_env>
                    + crate::read::LowerReadExpression<Slots = $read_env>
                    + crate::reduce::StageRead<R, crate::read::Env0>,
                Input::DeviceExpr: crate::eval::$eval<Self::Item, $($leaf),+>,
                Less: crate::BinaryPredicateOp<Self::Item>,
            {
                crate::ordering::sort(exec, input, less, self.output)
            }
        };
    }

macro_rules! unique_method_impl {
        ($name:ident, $read_arity:ty, $read_env:ty, $eval:ident, [$($leaf:ident),+]) => {
            fn $name<Input, Equal, $($leaf),+>(
                self,
                exec: &Executor<R>,
                input: Input,
                equal: Equal,
            ) -> Result<u32, Error>
            where
                $($leaf: crate::MStorageElement,)+
                Input: Clone
                    + crate::read::ReadExpression<Item = Self::Item, ReadArity = $read_arity>
                    + crate::read::BindSlots<crate::read::Env0, NextEnv = $read_env>
                    + crate::read::LowerReadExpression<
                        DeviceExpr = <Input as crate::read::BindSlots<crate::read::Env0>>::Expr,
                        Slots = $read_env,
                    >
                    + crate::reduce::StageRead<R, crate::read::Env0>,
                Input::DeviceExpr: crate::eval::$eval<Self::Item, $($leaf),+>,
                Equal: crate::BinaryPredicateOp<Self::Item>,
            {
                crate::ordering::unique(exec, input, equal, self.output)
            }
        };
    }

macro_rules! select_method_impl {
        ($name:ident, $read_arity:ty, $read_env:ty, $eval:ident, [$($leaf:ident),+]) => {
            fn $name<Input, $($leaf),+>(
                self,
                exec: &Executor<R>,
                input: Input,
                flags: crate::Column<u32>,
                invert: bool,
            ) -> Result<u32, Error>
            where
                $($leaf: crate::MStorageElement,)+
                Input: Clone
                    + crate::read::ReadExpression<Item = Self::Item, ReadArity = $read_arity>
                    + crate::read::BindSlots<crate::read::Env0, NextEnv = $read_env>
                    + crate::read::LowerReadExpression<Slots = $read_env>
                    + crate::reduce::StageRead<R, crate::read::Env0>,
                Input::DeviceExpr: crate::eval::$eval<Self::Item, $($leaf),+>,
            {
                if invert {
                    crate::selection::remove_where(exec, input, flags, self.output)
                } else {
                    crate::selection::copy_where(exec, input, flags, self.output)
                }
            }
        };
    }

macro_rules! partition_method_impl {
        ($name:ident, $read_arity:ty, $read_env:ty, $eval:ident, [$($leaf:ident),+]) => {
            fn $name<Input, Pred, $($leaf),+>(
                self,
                exec: &Executor<R>,
                input: Input,
                pred: Pred,
            ) -> Result<u32, Error>
            where
                $($leaf: crate::MStorageElement,)+
                Input: Clone
                    + crate::read::ReadExpression<Item = Self::Item, ReadArity = $read_arity>
                    + crate::read::BindSlots<crate::read::Env0, NextEnv = $read_env>
                    + crate::read::LowerReadExpression<
                        DeviceExpr = <Input as crate::read::BindSlots<crate::read::Env0>>::Expr,
                        Slots = $read_env,
                    >
                    + crate::reduce::StageRead<R, crate::read::Env0>,
                Input::DeviceExpr: crate::eval::$eval<Self::Item, $($leaf),+>,
                Pred: crate::PredicateOp<Self::Item>,
            {
                crate::selection::partition(exec, input, pred, self.output)
            }
        };
    }

macro_rules! indexed_method_impl {
        ($name:ident, $read_arity:ty, $read_env:ty, $eval:ident, [$($leaf:ident),+]) => {
            fn $name<Input, $($leaf),+>(
                self,
                exec: &Executor<R>,
                input: Input,
                indices: crate::Column<u32>,
                flags: Option<crate::Column<u32>>,
                scatter: bool,
            ) -> Result<(), Error>
            where
                $($leaf: crate::MStorageElement,)+
                Input: Clone
                    + crate::read::ReadExpression<Item = Self::Item, ReadArity = $read_arity>
                    + crate::read::BindSlots<crate::read::Env0, NextEnv = $read_env>
                    + crate::read::LowerReadExpression<
                        DeviceExpr = <Input as crate::read::BindSlots<crate::read::Env0>>::Expr,
                        Slots = $read_env,
                    >
                    + crate::reduce::StageRead<R, crate::read::Env0>,
                Input::DeviceExpr: crate::eval::$eval<Self::Item, $($leaf),+>,
            {
                match (scatter, flags) {
                    (false, None) => {
                        crate::indexed::apply_permutation(exec, input, indices, self.output)
                    }
                    (false, Some(flags)) => {
                        crate::indexed::gather_where(exec, input, indices, flags, self.output)
                    }
                    (true, None) => crate::core::scatter::scatter(exec, input, indices, self.output),
                    (true, Some(flags)) => {
                        crate::core::scatter::scatter_where(exec, input, indices, flags, self.output)
                    }
                }
            }
        };
    }

macro_rules! transform_where_method_impl {
        ($name:ident, $read_arity:ty, $read_env:ty, $eval:ident, [$($leaf:ident),+]; $write_slots:ty) => {
            fn $name<Input, Op, $($leaf),+>(
                self,
                exec: &Executor<R>,
                input: Input,
                op: Op,
                flags: crate::Column<u32>,
            ) -> Result<(), Error>
            where
                $($leaf: crate::MStorageElement,)+
                Input: Clone
                    + crate::read::ReadExpression<ReadArity = $read_arity>
                    + crate::read::BindSlots<crate::read::Env0, NextEnv = $read_env>
                    + crate::read::LowerReadExpression<
                        DeviceExpr = <Input as crate::read::BindSlots<crate::read::Env0>>::Expr,
                        Slots = $read_env,
                    >
                    + crate::reduce::StageRead<R, crate::read::Env0>,
                Input::DeviceExpr: crate::eval::$eval<Input::Item, $($leaf),+>,
                Op: crate::UnaryOp<Input::Item>,
                Item: crate::WriteFrom<Op::Output>,
            {
                let input_len = input.logical_len()?;
                let output_len = self.output.logical_len()?;
                if input_len != flags.len() {
                    return Err(Error::LengthMismatch { left: input_len, right: flags.len() });
                }
                if input_len != output_len {
                    return Err(Error::LengthMismatch { left: input_len, right: output_len });
                }
                let temporary = exec.alloc_canonical::<Item>(input_len);
                let temporary_output = crate::output::ReassociatedOutput::<
                    _,
                    Item,
                    $write_slots,
                >::new(crate::CanonicalStorage::write(&temporary));
                crate::transform::transform(exec, input, op, temporary_output)?;
                let flags = crate::selection::FlagInput::materialize_flags(flags, exec)?;
                crate::masked::MaskedCopyInput::masked_copy(
                    crate::CanonicalStorage::read(&temporary),
                    exec,
                    &flags,
                    self.output,
                )
            }
        };
    }

macro_rules! exclusive_method_impl {
        ($name:ident, $read_arity:ty, $read_env:ty, $eval:ident, [$($leaf:ident),+]) => {
            fn $name<Input, Op, $($leaf),+>(
                self,
                exec: &Executor<R>,
                input: Input,
                init: Self::Item,
                op: Op,
            ) -> Result<(), Error>
            where
                $($leaf: crate::MStorageElement,)+
                Input: Clone
                    + crate::read::ReadExpression<Item = Self::Item, ReadArity = $read_arity>
                    + crate::read::BindSlots<crate::read::Env0, NextEnv = $read_env>
                    + crate::read::LowerReadExpression<
                        DeviceExpr = <Input as crate::read::BindSlots<crate::read::Env0>>::Expr,
                        Slots = $read_env,
                    >
                    + crate::reduce::StageRead<R, crate::read::Env0>,
                Input::DeviceExpr: crate::eval::$eval<Self::Item, $($leaf),+>,
                Op: crate::ReductionOp<Self::Item>,
            {
                crate::scan::exclusive_scan(exec, input, init, op, self.output)
            }
        };
    }

macro_rules! storage_write_methods {
    () => {
        fn materialize_storage(
            self,
            exec: &Executor<R>,
            input: &<Self::Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
        ) -> Result<(), Error> {
            let input = crate::read::Reassociate::<_, Self::Item>::new(
                crate::CanonicalStorage::read(input),
            );
            crate::materialize(exec, input, self.output)
        }

        fn gather_storage(
            self,
            exec: &Executor<R>,
            input: &<Self::Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
            indices: crate::Column<u32>,
        ) -> Result<(), Error> {
            let input = crate::read::Reassociate::<_, Self::Item>::new(
                crate::CanonicalStorage::read(input),
            );
            crate::indexed::gather_direct(exec, input, indices, self.output)
        }

        fn select_storage(
            self,
            exec: &Executor<R>,
            input: &<Self::Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
            flags: crate::DeviceVec<R, u32>,
        ) -> Result<u32, Error> {
            let input = crate::read::Reassociate::<_, Self::Item>::new(
                crate::CanonicalStorage::read(input),
            );
            let control = crate::selection::SelectionControl::from_flags(exec, flags)?;
            crate::selection::CopySelected::copy_selected(input, exec, &control, self.output)
        }

        fn select_storage_control(
            self,
            exec: &Executor<R>,
            input: &<Self::Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
            control: &crate::selection::SelectionControl<R>,
        ) -> Result<u32, Error> {
            let input = crate::read::Reassociate::<_, Self::Item>::new(
                crate::CanonicalStorage::read(input),
            );
            crate::selection::CopySelected::copy_selected(input, exec, control, self.output)
        }

        fn concat_storage(
            self,
            exec: &Executor<R>,
            left: &<Self::Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
            right: &<Self::Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
            control: &crate::merge::MergeControl<R>,
        ) -> Result<(), Error> {
            let left =
                crate::read::Reassociate::<_, Self::Item>::new(crate::CanonicalStorage::read(left));
            let right = crate::read::Reassociate::<_, Self::Item>::new(
                crate::CanonicalStorage::read(right),
            );
            crate::merge::ConcatApply::concat_apply(left, exec, right, control, self.output)
        }

        fn set_storage<Less>(
            self,
            exec: &Executor<R>,
            left: &<Self::Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
            right: &<Self::Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
            less: Less,
            mode: u8,
        ) -> Result<u32, Error>
        where
            Less: crate::BinaryPredicateOp<Self::Item>,
        {
            let left =
                crate::read::Reassociate::<_, Self::Item>::new(crate::CanonicalStorage::read(left));
            let right = crate::read::Reassociate::<_, Self::Item>::new(
                crate::CanonicalStorage::read(right),
            );
            match mode {
                0 => crate::core::set::set_union(exec, left, right, less, self.output),
                1 => crate::core::set::set_intersection(exec, left, right, less, self.output),
                2 => crate::core::set::set_difference(exec, left, right, less, self.output),
                _ => unreachable!("invalid set operation"),
            }
        }
    };
}

macro_rules! impl_kernel_write {
        ($slots:ty, $arity:ty, $item:ident, $leaves:ty, [$($output_leaf:ident),+]; $($extra:tt)*) => {
            impl<R, Output, $item, $($output_leaf),+> KernelWrite<R> for Write<Output, $slots>
            where
                R: Runtime,
                $($output_leaf: crate::MStorageElement,)+
                Output: crate::output::OutputExpression<Item = $item, StorageArity = $arity>
                    + crate::output::LowerOutputExpression<Slots = $slots>
                    + crate::output::StageOutput<R, crate::read::Env0>
                    + crate::selection::FillOutput<R>
                    + crate::selection::ReplaceOutput<R>
                    + crate::output::SliceOutput,
                $item: crate::StorageLayout<
                    StorageArity = $arity,
                    StorageLeaves = $leaves,
                >,
                $($extra)*
            {
                type Item = $item;
                type Output = Output;

                fn fill(self, exec: &Executor<R>, value: Self::Item) -> Result<(), Error> {
                    crate::selection::fill(exec, value, self.output)
                }


                fn replace(
                    self,
                    exec: &Executor<R>,
                    value: Self::Item,
                    flags: crate::Column<u32>,
                ) -> Result<(), Error> {
                    crate::selection::replace_where(exec, value, flags, self.output)
                }

                storage_write_methods!();

                materialize_method!(materialize_a1, crate::A1, crate::read::Env1<L0>, Eval1, [L0]; $arity, $slots);
                materialize_method!(materialize_a2, crate::A2, crate::read::Env2<L0, L1>, Eval2, [L0, L1]; $arity, $slots);
                materialize_method!(materialize_a3, crate::A3, crate::read::Env3<L0, L1, L2>, Eval3, [L0, L1, L2]; $arity, $slots);
                materialize_method!(materialize_a4, crate::A4, crate::read::Env4<L0, L1, L2, L3>, Eval4, [L0, L1, L2, L3]; $arity, $slots);
                materialize_method!(materialize_a5, crate::A5, crate::read::Env5<L0, L1, L2, L3, L4>, Eval5, [L0, L1, L2, L3, L4]; $arity, $slots);
                materialize_method!(materialize_a6, crate::A6, crate::read::Env6<L0, L1, L2, L3, L4, L5>, Eval6, [L0, L1, L2, L3, L4, L5]; $arity, $slots);
                materialize_method!(materialize_a7, crate::A7, crate::read::Env7<L0, L1, L2, L3, L4, L5, L6>, Eval7, [L0, L1, L2, L3, L4, L5, L6]; $arity, $slots);
                materialize_method!(materialize_a8, crate::A8, crate::read::Env8<L0, L1, L2, L3, L4, L5, L6, L7>, Eval8, [L0, L1, L2, L3, L4, L5, L6, L7]; $arity, $slots);

                scan_method_impl!(inclusive_scan_a1, crate::A1, crate::read::Env1<L0>, Eval1, [L0]; $arity, $slots);
                scan_method_impl!(inclusive_scan_a2, crate::A2, crate::read::Env2<L0, L1>, Eval2, [L0, L1]; $arity, $slots);
                scan_method_impl!(inclusive_scan_a3, crate::A3, crate::read::Env3<L0, L1, L2>, Eval3, [L0, L1, L2]; $arity, $slots);
                scan_method_impl!(inclusive_scan_a4, crate::A4, crate::read::Env4<L0, L1, L2, L3>, Eval4, [L0, L1, L2, L3]; $arity, $slots);
                scan_method_impl!(inclusive_scan_a5, crate::A5, crate::read::Env5<L0, L1, L2, L3, L4>, Eval5, [L0, L1, L2, L3, L4]; $arity, $slots);
                scan_method_impl!(inclusive_scan_a6, crate::A6, crate::read::Env6<L0, L1, L2, L3, L4, L5>, Eval6, [L0, L1, L2, L3, L4, L5]; $arity, $slots);
                scan_method_impl!(inclusive_scan_a7, crate::A7, crate::read::Env7<L0, L1, L2, L3, L4, L5, L6>, Eval7, [L0, L1, L2, L3, L4, L5, L6]; $arity, $slots);
                scan_method_impl!(inclusive_scan_a8, crate::A8, crate::read::Env8<L0, L1, L2, L3, L4, L5, L6, L7>, Eval8, [L0, L1, L2, L3, L4, L5, L6, L7]; $arity, $slots);

                sort_method_impl!(sort_a1, crate::A1, crate::read::Env1<L0>, Eval1, [L0]);
                sort_method_impl!(sort_a2, crate::A2, crate::read::Env2<L0, L1>, Eval2, [L0, L1]);
                sort_method_impl!(sort_a3, crate::A3, crate::read::Env3<L0, L1, L2>, Eval3, [L0, L1, L2]);
                sort_method_impl!(sort_a4, crate::A4, crate::read::Env4<L0, L1, L2, L3>, Eval4, [L0, L1, L2, L3]);
                sort_method_impl!(sort_a5, crate::A5, crate::read::Env5<L0, L1, L2, L3, L4>, Eval5, [L0, L1, L2, L3, L4]);
                sort_method_impl!(sort_a6, crate::A6, crate::read::Env6<L0, L1, L2, L3, L4, L5>, Eval6, [L0, L1, L2, L3, L4, L5]);
                sort_method_impl!(sort_a7, crate::A7, crate::read::Env7<L0, L1, L2, L3, L4, L5, L6>, Eval7, [L0, L1, L2, L3, L4, L5, L6]);
                sort_method_impl!(sort_a8, crate::A8, crate::read::Env8<L0, L1, L2, L3, L4, L5, L6, L7>, Eval8, [L0, L1, L2, L3, L4, L5, L6, L7]);

                unique_method_impl!(unique_a1, crate::A1, crate::read::Env1<L0>, Eval1, [L0]);
                unique_method_impl!(unique_a2, crate::A2, crate::read::Env2<L0, L1>, Eval2, [L0, L1]);
                unique_method_impl!(unique_a3, crate::A3, crate::read::Env3<L0, L1, L2>, Eval3, [L0, L1, L2]);
                unique_method_impl!(unique_a4, crate::A4, crate::read::Env4<L0, L1, L2, L3>, Eval4, [L0, L1, L2, L3]);
                unique_method_impl!(unique_a5, crate::A5, crate::read::Env5<L0, L1, L2, L3, L4>, Eval5, [L0, L1, L2, L3, L4]);
                unique_method_impl!(unique_a6, crate::A6, crate::read::Env6<L0, L1, L2, L3, L4, L5>, Eval6, [L0, L1, L2, L3, L4, L5]);
                unique_method_impl!(unique_a7, crate::A7, crate::read::Env7<L0, L1, L2, L3, L4, L5, L6>, Eval7, [L0, L1, L2, L3, L4, L5, L6]);
                unique_method_impl!(unique_a8, crate::A8, crate::read::Env8<L0, L1, L2, L3, L4, L5, L6, L7>, Eval8, [L0, L1, L2, L3, L4, L5, L6, L7]);

                select_method_impl!(select_a1, crate::A1, crate::read::Env1<L0>, Eval1, [L0]);
                select_method_impl!(select_a2, crate::A2, crate::read::Env2<L0, L1>, Eval2, [L0, L1]);
                select_method_impl!(select_a3, crate::A3, crate::read::Env3<L0, L1, L2>, Eval3, [L0, L1, L2]);
                select_method_impl!(select_a4, crate::A4, crate::read::Env4<L0, L1, L2, L3>, Eval4, [L0, L1, L2, L3]);
                select_method_impl!(select_a5, crate::A5, crate::read::Env5<L0, L1, L2, L3, L4>, Eval5, [L0, L1, L2, L3, L4]);
                select_method_impl!(select_a6, crate::A6, crate::read::Env6<L0, L1, L2, L3, L4, L5>, Eval6, [L0, L1, L2, L3, L4, L5]);
                select_method_impl!(select_a7, crate::A7, crate::read::Env7<L0, L1, L2, L3, L4, L5, L6>, Eval7, [L0, L1, L2, L3, L4, L5, L6]);
                select_method_impl!(select_a8, crate::A8, crate::read::Env8<L0, L1, L2, L3, L4, L5, L6, L7>, Eval8, [L0, L1, L2, L3, L4, L5, L6, L7]);

                partition_method_impl!(partition_a1, crate::A1, crate::read::Env1<L0>, Eval1, [L0]);
                partition_method_impl!(partition_a2, crate::A2, crate::read::Env2<L0, L1>, Eval2, [L0, L1]);
                partition_method_impl!(partition_a3, crate::A3, crate::read::Env3<L0, L1, L2>, Eval3, [L0, L1, L2]);
                partition_method_impl!(partition_a4, crate::A4, crate::read::Env4<L0, L1, L2, L3>, Eval4, [L0, L1, L2, L3]);
                partition_method_impl!(partition_a5, crate::A5, crate::read::Env5<L0, L1, L2, L3, L4>, Eval5, [L0, L1, L2, L3, L4]);
                partition_method_impl!(partition_a6, crate::A6, crate::read::Env6<L0, L1, L2, L3, L4, L5>, Eval6, [L0, L1, L2, L3, L4, L5]);
                partition_method_impl!(partition_a7, crate::A7, crate::read::Env7<L0, L1, L2, L3, L4, L5, L6>, Eval7, [L0, L1, L2, L3, L4, L5, L6]);
                partition_method_impl!(partition_a8, crate::A8, crate::read::Env8<L0, L1, L2, L3, L4, L5, L6, L7>, Eval8, [L0, L1, L2, L3, L4, L5, L6, L7]);

                indexed_method_impl!(indexed_a1, crate::A1, crate::read::Env1<L0>, Eval1, [L0]);
                indexed_method_impl!(indexed_a2, crate::A2, crate::read::Env2<L0, L1>, Eval2, [L0, L1]);
                indexed_method_impl!(indexed_a3, crate::A3, crate::read::Env3<L0, L1, L2>, Eval3, [L0, L1, L2]);
                indexed_method_impl!(indexed_a4, crate::A4, crate::read::Env4<L0, L1, L2, L3>, Eval4, [L0, L1, L2, L3]);
                indexed_method_impl!(indexed_a5, crate::A5, crate::read::Env5<L0, L1, L2, L3, L4>, Eval5, [L0, L1, L2, L3, L4]);
                indexed_method_impl!(indexed_a6, crate::A6, crate::read::Env6<L0, L1, L2, L3, L4, L5>, Eval6, [L0, L1, L2, L3, L4, L5]);
                indexed_method_impl!(indexed_a7, crate::A7, crate::read::Env7<L0, L1, L2, L3, L4, L5, L6>, Eval7, [L0, L1, L2, L3, L4, L5, L6]);
                indexed_method_impl!(indexed_a8, crate::A8, crate::read::Env8<L0, L1, L2, L3, L4, L5, L6, L7>, Eval8, [L0, L1, L2, L3, L4, L5, L6, L7]);

                transform_where_method_impl!(transform_where_a1, crate::A1, crate::read::Env1<L0>, Eval1, [L0]; $slots);
                transform_where_method_impl!(transform_where_a2, crate::A2, crate::read::Env2<L0, L1>, Eval2, [L0, L1]; $slots);
                transform_where_method_impl!(transform_where_a3, crate::A3, crate::read::Env3<L0, L1, L2>, Eval3, [L0, L1, L2]; $slots);
                transform_where_method_impl!(transform_where_a4, crate::A4, crate::read::Env4<L0, L1, L2, L3>, Eval4, [L0, L1, L2, L3]; $slots);
                transform_where_method_impl!(transform_where_a5, crate::A5, crate::read::Env5<L0, L1, L2, L3, L4>, Eval5, [L0, L1, L2, L3, L4]; $slots);
                transform_where_method_impl!(transform_where_a6, crate::A6, crate::read::Env6<L0, L1, L2, L3, L4, L5>, Eval6, [L0, L1, L2, L3, L4, L5]; $slots);
                transform_where_method_impl!(transform_where_a7, crate::A7, crate::read::Env7<L0, L1, L2, L3, L4, L5, L6>, Eval7, [L0, L1, L2, L3, L4, L5, L6]; $slots);
                transform_where_method_impl!(transform_where_a8, crate::A8, crate::read::Env8<L0, L1, L2, L3, L4, L5, L6, L7>, Eval8, [L0, L1, L2, L3, L4, L5, L6, L7]; $slots);

                exclusive_method_impl!(exclusive_a1, crate::A1, crate::read::Env1<L0>, Eval1, [L0]);
                exclusive_method_impl!(exclusive_a2, crate::A2, crate::read::Env2<L0, L1>, Eval2, [L0, L1]);
                exclusive_method_impl!(exclusive_a3, crate::A3, crate::read::Env3<L0, L1, L2>, Eval3, [L0, L1, L2]);
                exclusive_method_impl!(exclusive_a4, crate::A4, crate::read::Env4<L0, L1, L2, L3>, Eval4, [L0, L1, L2, L3]);
                exclusive_method_impl!(exclusive_a5, crate::A5, crate::read::Env5<L0, L1, L2, L3, L4>, Eval5, [L0, L1, L2, L3, L4]);
                exclusive_method_impl!(exclusive_a6, crate::A6, crate::read::Env6<L0, L1, L2, L3, L4, L5>, Eval6, [L0, L1, L2, L3, L4, L5]);
                exclusive_method_impl!(exclusive_a7, crate::A7, crate::read::Env7<L0, L1, L2, L3, L4, L5, L6>, Eval7, [L0, L1, L2, L3, L4, L5, L6]);
                exclusive_method_impl!(exclusive_a8, crate::A8, crate::read::Env8<L0, L1, L2, L3, L4, L5, L6, L7>, Eval8, [L0, L1, L2, L3, L4, L5, L6, L7]);
            }
        };
    }

impl<R, Output, Item> KernelWrite<R> for Write<Output, crate::read::Env1<Item>>
where
    R: Runtime,
    Item: crate::MStorageElement,
    Output: crate::output::OutputExpression<Item = Item, StorageArity = crate::S1>
        + crate::output::LowerOutputExpression<Slots = crate::read::Env1<Item>>
        + crate::output::StageOutput<R, crate::read::Env0>
        + crate::selection::FillOutput<R>
        + crate::selection::ReplaceOutput<R>
        + crate::output::SliceOutput,
{
    type Item = Item;
    type Output = Output;

    fn fill(self, exec: &Executor<R>, value: Self::Item) -> Result<(), Error> {
        crate::selection::fill(exec, value, self.output)
    }

    fn replace(
        self,
        exec: &Executor<R>,
        value: Self::Item,
        flags: crate::Column<u32>,
    ) -> Result<(), Error> {
        crate::selection::replace_where(exec, value, flags, self.output)
    }

    storage_write_methods!();

    materialize_method!(materialize_a1, crate::A1, crate::read::Env1<L0>, Eval1, [L0]; crate::S1, crate::read::Env1<Item>);
    materialize_method!(materialize_a2, crate::A2, crate::read::Env2<L0, L1>, Eval2, [L0, L1]; crate::S1, crate::read::Env1<Item>);
    materialize_method!(materialize_a3, crate::A3, crate::read::Env3<L0, L1, L2>, Eval3, [L0, L1, L2]; crate::S1, crate::read::Env1<Item>);
    materialize_method!(materialize_a4, crate::A4, crate::read::Env4<L0, L1, L2, L3>, Eval4, [L0, L1, L2, L3]; crate::S1, crate::read::Env1<Item>);
    materialize_method!(materialize_a5, crate::A5, crate::read::Env5<L0, L1, L2, L3, L4>, Eval5, [L0, L1, L2, L3, L4]; crate::S1, crate::read::Env1<Item>);
    materialize_method!(materialize_a6, crate::A6, crate::read::Env6<L0, L1, L2, L3, L4, L5>, Eval6, [L0, L1, L2, L3, L4, L5]; crate::S1, crate::read::Env1<Item>);
    materialize_method!(materialize_a7, crate::A7, crate::read::Env7<L0, L1, L2, L3, L4, L5, L6>, Eval7, [L0, L1, L2, L3, L4, L5, L6]; crate::S1, crate::read::Env1<Item>);
    materialize_method!(materialize_a8, crate::A8, crate::read::Env8<L0, L1, L2, L3, L4, L5, L6, L7>, Eval8, [L0, L1, L2, L3, L4, L5, L6, L7]; crate::S1, crate::read::Env1<Item>);

    scan_method_impl!(inclusive_scan_a1, crate::A1, crate::read::Env1<L0>, Eval1, [L0]; crate::S1, crate::read::Env1<Item>);
    scan_method_impl!(inclusive_scan_a2, crate::A2, crate::read::Env2<L0, L1>, Eval2, [L0, L1]; crate::S1, crate::read::Env1<Item>);
    scan_method_impl!(inclusive_scan_a3, crate::A3, crate::read::Env3<L0, L1, L2>, Eval3, [L0, L1, L2]; crate::S1, crate::read::Env1<Item>);
    scan_method_impl!(inclusive_scan_a4, crate::A4, crate::read::Env4<L0, L1, L2, L3>, Eval4, [L0, L1, L2, L3]; crate::S1, crate::read::Env1<Item>);
    scan_method_impl!(inclusive_scan_a5, crate::A5, crate::read::Env5<L0, L1, L2, L3, L4>, Eval5, [L0, L1, L2, L3, L4]; crate::S1, crate::read::Env1<Item>);
    scan_method_impl!(inclusive_scan_a6, crate::A6, crate::read::Env6<L0, L1, L2, L3, L4, L5>, Eval6, [L0, L1, L2, L3, L4, L5]; crate::S1, crate::read::Env1<Item>);
    scan_method_impl!(inclusive_scan_a7, crate::A7, crate::read::Env7<L0, L1, L2, L3, L4, L5, L6>, Eval7, [L0, L1, L2, L3, L4, L5, L6]; crate::S1, crate::read::Env1<Item>);
    scan_method_impl!(inclusive_scan_a8, crate::A8, crate::read::Env8<L0, L1, L2, L3, L4, L5, L6, L7>, Eval8, [L0, L1, L2, L3, L4, L5, L6, L7]; crate::S1, crate::read::Env1<Item>);

    sort_method_impl!(sort_a1, crate::A1, crate::read::Env1<L0>, Eval1, [L0]);
    sort_method_impl!(sort_a2, crate::A2, crate::read::Env2<L0, L1>, Eval2, [L0, L1]);
    sort_method_impl!(sort_a3, crate::A3, crate::read::Env3<L0, L1, L2>, Eval3, [L0, L1, L2]);
    sort_method_impl!(sort_a4, crate::A4, crate::read::Env4<L0, L1, L2, L3>, Eval4, [L0, L1, L2, L3]);
    sort_method_impl!(sort_a5, crate::A5, crate::read::Env5<L0, L1, L2, L3, L4>, Eval5, [L0, L1, L2, L3, L4]);
    sort_method_impl!(sort_a6, crate::A6, crate::read::Env6<L0, L1, L2, L3, L4, L5>, Eval6, [L0, L1, L2, L3, L4, L5]);
    sort_method_impl!(sort_a7, crate::A7, crate::read::Env7<L0, L1, L2, L3, L4, L5, L6>, Eval7, [L0, L1, L2, L3, L4, L5, L6]);
    sort_method_impl!(sort_a8, crate::A8, crate::read::Env8<L0, L1, L2, L3, L4, L5, L6, L7>, Eval8, [L0, L1, L2, L3, L4, L5, L6, L7]);

    unique_method_impl!(unique_a1, crate::A1, crate::read::Env1<L0>, Eval1, [L0]);
    unique_method_impl!(unique_a2, crate::A2, crate::read::Env2<L0, L1>, Eval2, [L0, L1]);
    unique_method_impl!(unique_a3, crate::A3, crate::read::Env3<L0, L1, L2>, Eval3, [L0, L1, L2]);
    unique_method_impl!(unique_a4, crate::A4, crate::read::Env4<L0, L1, L2, L3>, Eval4, [L0, L1, L2, L3]);
    unique_method_impl!(unique_a5, crate::A5, crate::read::Env5<L0, L1, L2, L3, L4>, Eval5, [L0, L1, L2, L3, L4]);
    unique_method_impl!(unique_a6, crate::A6, crate::read::Env6<L0, L1, L2, L3, L4, L5>, Eval6, [L0, L1, L2, L3, L4, L5]);
    unique_method_impl!(unique_a7, crate::A7, crate::read::Env7<L0, L1, L2, L3, L4, L5, L6>, Eval7, [L0, L1, L2, L3, L4, L5, L6]);
    unique_method_impl!(unique_a8, crate::A8, crate::read::Env8<L0, L1, L2, L3, L4, L5, L6, L7>, Eval8, [L0, L1, L2, L3, L4, L5, L6, L7]);

    select_method_impl!(select_a1, crate::A1, crate::read::Env1<L0>, Eval1, [L0]);
    select_method_impl!(select_a2, crate::A2, crate::read::Env2<L0, L1>, Eval2, [L0, L1]);
    select_method_impl!(select_a3, crate::A3, crate::read::Env3<L0, L1, L2>, Eval3, [L0, L1, L2]);
    select_method_impl!(select_a4, crate::A4, crate::read::Env4<L0, L1, L2, L3>, Eval4, [L0, L1, L2, L3]);
    select_method_impl!(select_a5, crate::A5, crate::read::Env5<L0, L1, L2, L3, L4>, Eval5, [L0, L1, L2, L3, L4]);
    select_method_impl!(select_a6, crate::A6, crate::read::Env6<L0, L1, L2, L3, L4, L5>, Eval6, [L0, L1, L2, L3, L4, L5]);
    select_method_impl!(select_a7, crate::A7, crate::read::Env7<L0, L1, L2, L3, L4, L5, L6>, Eval7, [L0, L1, L2, L3, L4, L5, L6]);
    select_method_impl!(select_a8, crate::A8, crate::read::Env8<L0, L1, L2, L3, L4, L5, L6, L7>, Eval8, [L0, L1, L2, L3, L4, L5, L6, L7]);

    partition_method_impl!(partition_a1, crate::A1, crate::read::Env1<L0>, Eval1, [L0]);
    partition_method_impl!(partition_a2, crate::A2, crate::read::Env2<L0, L1>, Eval2, [L0, L1]);
    partition_method_impl!(partition_a3, crate::A3, crate::read::Env3<L0, L1, L2>, Eval3, [L0, L1, L2]);
    partition_method_impl!(partition_a4, crate::A4, crate::read::Env4<L0, L1, L2, L3>, Eval4, [L0, L1, L2, L3]);
    partition_method_impl!(partition_a5, crate::A5, crate::read::Env5<L0, L1, L2, L3, L4>, Eval5, [L0, L1, L2, L3, L4]);
    partition_method_impl!(partition_a6, crate::A6, crate::read::Env6<L0, L1, L2, L3, L4, L5>, Eval6, [L0, L1, L2, L3, L4, L5]);
    partition_method_impl!(partition_a7, crate::A7, crate::read::Env7<L0, L1, L2, L3, L4, L5, L6>, Eval7, [L0, L1, L2, L3, L4, L5, L6]);
    partition_method_impl!(partition_a8, crate::A8, crate::read::Env8<L0, L1, L2, L3, L4, L5, L6, L7>, Eval8, [L0, L1, L2, L3, L4, L5, L6, L7]);

    indexed_method_impl!(indexed_a1, crate::A1, crate::read::Env1<L0>, Eval1, [L0]);
    indexed_method_impl!(indexed_a2, crate::A2, crate::read::Env2<L0, L1>, Eval2, [L0, L1]);
    indexed_method_impl!(indexed_a3, crate::A3, crate::read::Env3<L0, L1, L2>, Eval3, [L0, L1, L2]);
    indexed_method_impl!(indexed_a4, crate::A4, crate::read::Env4<L0, L1, L2, L3>, Eval4, [L0, L1, L2, L3]);
    indexed_method_impl!(indexed_a5, crate::A5, crate::read::Env5<L0, L1, L2, L3, L4>, Eval5, [L0, L1, L2, L3, L4]);
    indexed_method_impl!(indexed_a6, crate::A6, crate::read::Env6<L0, L1, L2, L3, L4, L5>, Eval6, [L0, L1, L2, L3, L4, L5]);
    indexed_method_impl!(indexed_a7, crate::A7, crate::read::Env7<L0, L1, L2, L3, L4, L5, L6>, Eval7, [L0, L1, L2, L3, L4, L5, L6]);
    indexed_method_impl!(indexed_a8, crate::A8, crate::read::Env8<L0, L1, L2, L3, L4, L5, L6, L7>, Eval8, [L0, L1, L2, L3, L4, L5, L6, L7]);

    transform_where_method_impl!(
        transform_where_a1,
        crate::A1,
        crate::read::Env1<L0>,
        Eval1,
        [L0];
        crate::read::Env1<Item>
    );
    transform_where_method_impl!(transform_where_a2, crate::A2, crate::read::Env2<L0, L1>, Eval2, [L0, L1]; crate::read::Env1<Item>);
    transform_where_method_impl!(transform_where_a3, crate::A3, crate::read::Env3<L0, L1, L2>, Eval3, [L0, L1, L2]; crate::read::Env1<Item>);
    transform_where_method_impl!(transform_where_a4, crate::A4, crate::read::Env4<L0, L1, L2, L3>, Eval4, [L0, L1, L2, L3]; crate::read::Env1<Item>);
    transform_where_method_impl!(transform_where_a5, crate::A5, crate::read::Env5<L0, L1, L2, L3, L4>, Eval5, [L0, L1, L2, L3, L4]; crate::read::Env1<Item>);
    transform_where_method_impl!(transform_where_a6, crate::A6, crate::read::Env6<L0, L1, L2, L3, L4, L5>, Eval6, [L0, L1, L2, L3, L4, L5]; crate::read::Env1<Item>);
    transform_where_method_impl!(transform_where_a7, crate::A7, crate::read::Env7<L0, L1, L2, L3, L4, L5, L6>, Eval7, [L0, L1, L2, L3, L4, L5, L6]; crate::read::Env1<Item>);
    transform_where_method_impl!(transform_where_a8, crate::A8, crate::read::Env8<L0, L1, L2, L3, L4, L5, L6, L7>, Eval8, [L0, L1, L2, L3, L4, L5, L6, L7]; crate::read::Env1<Item>);

    exclusive_method_impl!(exclusive_a1, crate::A1, crate::read::Env1<L0>, Eval1, [L0]);
    exclusive_method_impl!(exclusive_a2, crate::A2, crate::read::Env2<L0, L1>, Eval2, [L0, L1]);
    exclusive_method_impl!(exclusive_a3, crate::A3, crate::read::Env3<L0, L1, L2>, Eval3, [L0, L1, L2]);
    exclusive_method_impl!(exclusive_a4, crate::A4, crate::read::Env4<L0, L1, L2, L3>, Eval4, [L0, L1, L2, L3]);
    exclusive_method_impl!(exclusive_a5, crate::A5, crate::read::Env5<L0, L1, L2, L3, L4>, Eval5, [L0, L1, L2, L3, L4]);
    exclusive_method_impl!(exclusive_a6, crate::A6, crate::read::Env6<L0, L1, L2, L3, L4, L5>, Eval6, [L0, L1, L2, L3, L4, L5]);
    exclusive_method_impl!(exclusive_a7, crate::A7, crate::read::Env7<L0, L1, L2, L3, L4, L5, L6>, Eval7, [L0, L1, L2, L3, L4, L5, L6]);
    exclusive_method_impl!(exclusive_a8, crate::A8, crate::read::Env8<L0, L1, L2, L3, L4, L5, L6, L7>, Eval8, [L0, L1, L2, L3, L4, L5, L6, L7]);
}

impl_kernel_write!(crate::read::Env2<O0, O1>, crate::S2, Item, crate::storage::More<O0, crate::storage::Last<O1>>, [O0, O1];);
impl_kernel_write!(crate::read::Env3<O0, O1, O2>, crate::S3, Item, crate::storage::More<O0, crate::storage::More<O1, crate::storage::Last<O2>>>, [O0, O1, O2];);
impl_kernel_write!(crate::read::Env4<O0, O1, O2, O3>, crate::S4, Item, crate::storage::More<O0, crate::storage::More<O1, crate::storage::More<O2, crate::storage::Last<O3>>>>, [O0, O1, O2, O3];);
impl_kernel_write!(crate::read::Env5<O0, O1, O2, O3, O4>, crate::S5, Item, crate::storage::More<O0, crate::storage::More<O1, crate::storage::More<O2, crate::storage::More<O3, crate::storage::Last<O4>>>>>, [O0, O1, O2, O3, O4];);
impl_kernel_write!(crate::read::Env6<O0, O1, O2, O3, O4, O5>, crate::S6, Item, crate::storage::More<O0, crate::storage::More<O1, crate::storage::More<O2, crate::storage::More<O3, crate::storage::More<O4, crate::storage::Last<O5>>>>>>, [O0, O1, O2, O3, O4, O5];);
impl_kernel_write!(crate::read::Env7<O0, O1, O2, O3, O4, O5, O6>, crate::S7, Item, crate::storage::More<O0, crate::storage::More<O1, crate::storage::More<O2, crate::storage::More<O3, crate::storage::More<O4, crate::storage::More<O5, crate::storage::Last<O6>>>>>>>, [O0, O1, O2, O3, O4, O5, O6];);

macro_rules! reduce_method_decl {
        ($name:ident, $arity:ty, $env:ty, $eval:ident, [$($leaf:ident),+]) => {
            fn $name<Input, Op, $($leaf),+>(
                input: Input,
                exec: &Executor<R>,
                init: Item,
                op: Op,
            ) -> Result<Item, Error>
            where
                $($leaf: crate::MStorageElement,)+
                Input: crate::read::ReadExpression<Item = Item, ReadArity = $arity>
                    + crate::read::BindSlots<crate::read::Env0, NextEnv = $env>
                    + crate::read::LowerReadExpression<Slots = $env>
                    + crate::reduce::StageRead<R, crate::read::Env0>,
                Input::DeviceExpr: crate::eval::$eval<Item, $($leaf),+>,
                Op: crate::ReductionOp<Item>;
        };
    }

macro_rules! normalize_method_decl {
        ($name:ident, $arity:ty, $env:ty, $eval:ident, [$($leaf:ident),+]) => {
            fn $name<Input, $($leaf),+>(
                input: Input,
                exec: &Executor<R>,
            ) -> Result<<Item as crate::CanonicalAlloc<R>>::CanonicalStorage, Error>
            where
                $($leaf: crate::MStorageElement,)+
                Input: Clone
                    + crate::read::ReadExpression<Item = Item, ReadArity = $arity>
                    + crate::read::BindSlots<crate::read::Env0, NextEnv = $env>
                    + crate::read::LowerReadExpression<
                        DeviceExpr = <Input as crate::read::BindSlots<crate::read::Env0>>::Expr,
                        Slots = $env,
                    >
                    + crate::reduce::StageRead<R, crate::read::Env0>,
                Input::DeviceExpr: crate::eval::$eval<Item, $($leaf),+>;
        };
    }

pub trait KernelItem<R: Runtime, Item: crate::StorageLayout + crate::CanonicalAlloc<R>>:
    Sized
{
    type ReboundWrite<Output>: KernelWrite<R, Item = Item>
    where
        Output: crate::output::OutputExpression
            + crate::output::StageOutput<R, crate::read::Env0>
            + crate::selection::FillOutput<R>
            + crate::selection::ReplaceOutput<R>
            + crate::output::SliceOutput,
        Output::Item: crate::WriteFrom<Item>;

    fn rebind_write<Output>(output: Output) -> Self::ReboundWrite<Output>
    where
        Output: crate::output::OutputExpression
            + crate::output::StageOutput<R, crate::read::Env0>
            + crate::selection::FillOutput<R>
            + crate::selection::ReplaceOutput<R>
            + crate::output::SliceOutput,
        Output::Item: crate::WriteFrom<Item>;

    fn pair<Op>(
        exec: &Executor<R>,
        left: &<Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
        right: &<Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
        op: Op,
        mode: u8,
    ) -> Result<PairResult, Error>
    where
        Op: crate::BinaryPredicateOp<Item>;

    fn bounds<Less>(
        exec: &Executor<R>,
        source: &<Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
        values: &<Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
        less: Less,
        upper: bool,
    ) -> Result<crate::DeviceVec<R, u32>, Error>
    where
        Less: crate::BinaryPredicateOp<Item>;

    fn merge_control<Less>(
        exec: &Executor<R>,
        left: &<Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
        right: &<Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
        less: Less,
    ) -> Result<crate::merge::MergeControl<R>, Error>
    where
        Less: crate::BinaryPredicateOp<Item>;

    fn sort_control<Less>(
        exec: &Executor<R>,
        input: &<Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
        less: Less,
    ) -> Result<crate::DeviceVec<R, u32>, Error>
    where
        Less: crate::BinaryPredicateOp<Item>;

    fn sort_ordering<Less>(
        exec: &Executor<R>,
        input: <Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
        _less: Less,
    ) -> Result<
        crate::ordering::sort::OrderingResult<
            R,
            <Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
        >,
        Error,
    >
    where
        Less: crate::BinaryPredicateOp<Item>;

    fn segment_heads<Equal>(
        exec: &Executor<R>,
        input: &<Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
        equal: Equal,
    ) -> Result<crate::DeviceVec<R, u32>, Error>
    where
        Equal: crate::BinaryPredicateOp<Item>;

    fn segmented<Op>(
        exec: &Executor<R>,
        input: &<Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
        heads: &crate::DeviceVec<R, u32>,
        init: Option<Item>,
        op: Op,
        mode: u8,
    ) -> Result<<Item as crate::CanonicalAlloc<R>>::CanonicalStorage, Error>
    where
        Op: crate::ReductionOp<Item>;

    reduce_method_decl!(reduce_a1, crate::A1, crate::read::Env1<L0>, Eval1, [L0]);
    reduce_method_decl!(reduce_a2, crate::A2, crate::read::Env2<L0, L1>, Eval2, [L0, L1]);
    reduce_method_decl!(reduce_a3, crate::A3, crate::read::Env3<L0, L1, L2>, Eval3, [L0, L1, L2]);
    reduce_method_decl!(reduce_a4, crate::A4, crate::read::Env4<L0, L1, L2, L3>, Eval4, [L0, L1, L2, L3]);
    reduce_method_decl!(reduce_a5, crate::A5, crate::read::Env5<L0, L1, L2, L3, L4>, Eval5, [L0, L1, L2, L3, L4]);
    reduce_method_decl!(reduce_a6, crate::A6, crate::read::Env6<L0, L1, L2, L3, L4, L5>, Eval6, [L0, L1, L2, L3, L4, L5]);
    reduce_method_decl!(reduce_a7, crate::A7, crate::read::Env7<L0, L1, L2, L3, L4, L5, L6>, Eval7, [L0, L1, L2, L3, L4, L5, L6]);
    reduce_method_decl!(reduce_a8, crate::A8, crate::read::Env8<L0, L1, L2, L3, L4, L5, L6, L7>, Eval8, [L0, L1, L2, L3, L4, L5, L6, L7]);

    normalize_method_decl!(normalize_a1, crate::A1, crate::read::Env1<L0>, Eval1, [L0]);
    normalize_method_decl!(normalize_a2, crate::A2, crate::read::Env2<L0, L1>, Eval2, [L0, L1]);
    normalize_method_decl!(normalize_a3, crate::A3, crate::read::Env3<L0, L1, L2>, Eval3, [L0, L1, L2]);
    normalize_method_decl!(normalize_a4, crate::A4, crate::read::Env4<L0, L1, L2, L3>, Eval4, [L0, L1, L2, L3]);
    normalize_method_decl!(normalize_a5, crate::A5, crate::read::Env5<L0, L1, L2, L3, L4>, Eval5, [L0, L1, L2, L3, L4]);
    normalize_method_decl!(normalize_a6, crate::A6, crate::read::Env6<L0, L1, L2, L3, L4, L5>, Eval6, [L0, L1, L2, L3, L4, L5]);
    normalize_method_decl!(normalize_a7, crate::A7, crate::read::Env7<L0, L1, L2, L3, L4, L5, L6>, Eval7, [L0, L1, L2, L3, L4, L5, L6]);
    normalize_method_decl!(normalize_a8, crate::A8, crate::read::Env8<L0, L1, L2, L3, L4, L5, L6, L7>, Eval8, [L0, L1, L2, L3, L4, L5, L6, L7]);
}

macro_rules! reduce_method_impl {
        ($name:ident, $arity:ty, $env:ty, $eval:ident, [$($leaf:ident),+]; $storage:ty) => {
            fn $name<Input, Op, $($leaf),+>(
                input: Input,
                exec: &Executor<R>,
                init: Item,
                _op: Op,
            ) -> Result<Item, Error>
            where
                $($leaf: crate::MStorageElement,)+
                Input: crate::read::ReadExpression<Item = Item, ReadArity = $arity>
                    + crate::read::BindSlots<crate::read::Env0, NextEnv = $env>
                    + crate::read::LowerReadExpression<Slots = $env>
                    + crate::reduce::StageRead<R, crate::read::Env0>,
                Input::DeviceExpr: crate::eval::$eval<Item, $($leaf),+>,
                Op: crate::ReductionOp<Item>,
            {
                <crate::Dispatch<$arity, $storage> as crate::reduce::ReduceDispatch<
                    R,
                    Input,
                    Item,
                    Op,
                    $env,
                >>::execute(exec, &input, init)
            }
        };
    }

macro_rules! normalize_method_impl {
        ($name:ident, $arity:ty, $env:ty, $eval:ident, [$($leaf:ident),+]) => {
            fn $name<Input, $($leaf),+>(
                input: Input,
                exec: &Executor<R>,
            ) -> Result<<Item as crate::CanonicalAlloc<R>>::CanonicalStorage, Error>
            where
                $($leaf: crate::MStorageElement,)+
                Input: Clone
                    + crate::read::ReadExpression<Item = Item, ReadArity = $arity>
                    + crate::read::BindSlots<crate::read::Env0, NextEnv = $env>
                    + crate::read::LowerReadExpression<
                        DeviceExpr = <Input as crate::read::BindSlots<crate::read::Env0>>::Expr,
                        Slots = $env,
                    >
                    + crate::reduce::StageRead<R, crate::read::Env0>,
                Input::DeviceExpr: crate::eval::$eval<Item, $($leaf),+>,
            {
                crate::allocation::NormalizeInput::normalize(input, exec)
            }
        };
    }

macro_rules! impl_kernel_item {
        ($self_ty:ty, $storage:ty, $write_slots:ty, [$($output_leaf:ident),+]; $($extra:tt)*) => {
            impl<R, Item, $($output_leaf),+> KernelItem<R, Item> for $self_ty
            where
                R: Runtime,
                $($output_leaf: crate::MStorageElement,)+
                Item: crate::StorageLayout<StorageArity = $storage, StorageLeaves = $self_ty>,
                $($extra)*
            {
                type ReboundWrite<Output> = Write<
                    crate::output::ReassociatedOutput<Output, Item, $write_slots>,
                    $write_slots,
                >
                where
                    Output: crate::output::OutputExpression
                        + crate::output::StageOutput<R, crate::read::Env0>
                        + crate::selection::FillOutput<R>
                        + crate::selection::ReplaceOutput<R>
                        + crate::output::SliceOutput,
                    Output::Item: crate::WriteFrom<Item>;

                fn rebind_write<Output>(output: Output) -> Self::ReboundWrite<Output>
                where
                    Output: crate::output::OutputExpression
                        + crate::output::StageOutput<R, crate::read::Env0>
                        + crate::selection::FillOutput<R>
                        + crate::selection::ReplaceOutput<R>
                        + crate::output::SliceOutput,
                    Output::Item: crate::WriteFrom<Item>,
                {
                    Write::new(crate::output::ReassociatedOutput::new(output))
                }

                fn pair<Op>(
                    exec: &Executor<R>,
                    left: &<Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
                    right: &<Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
                    op: Op,
                    mode: u8,
                ) -> Result<PairResult, Error>
                where
                    Op: crate::BinaryPredicateOp<Item>,
                {
                    let left = crate::read::Reassociate::<_, Item>::new(
                        crate::CanonicalStorage::read(left),
                    );
                    let right = crate::read::Reassociate::<_, Item>::new(
                        crate::CanonicalStorage::read(right),
                    );
                    match mode {
                        0 => Ok(PairResult::Bool(crate::search::equal(exec, left, right, op)?)),
                        1 => Ok(PairResult::Index(crate::search::mismatch(exec, left, right, op)?)),
                        2 => Ok(PairResult::Bool(crate::search::lexicographical_compare(
                            exec, left, right, op,
                        )?)),
                        3 => Ok(PairResult::Index(crate::search::find_first_of(
                            exec, left, right, op,
                        )?)),
                        _ => unreachable!("invalid pair operation"),
                    }
                }

                fn bounds<Less>(
                    exec: &Executor<R>,
                    source: &<Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
                    values: &<Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
                    less: Less,
                    upper: bool,
                ) -> Result<crate::DeviceVec<R, u32>, Error>
                where
                    Less: crate::BinaryPredicateOp<Item>,
                {
                    let source = crate::read::Reassociate::<_, Item>::new(
                        crate::CanonicalStorage::read(source),
                    );
                    let values = crate::read::Reassociate::<_, Item>::new(
                        crate::CanonicalStorage::read(values),
                    );
                    if upper {
                        crate::search::upper_bounds_storage(exec, source, values, less)
                    } else {
                        crate::search::lower_bounds_storage(exec, source, values, less)
                    }
                }

                fn merge_control<Less>(
                    exec: &Executor<R>,
                    left: &<Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
                    right: &<Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
                    less: Less,
                ) -> Result<crate::merge::MergeControl<R>, Error>
                where
                    Less: crate::BinaryPredicateOp<Item>,
                {
                    let left = crate::read::Reassociate::<_, Item>::new(crate::CanonicalStorage::read(left));
                    let right = crate::read::Reassociate::<_, Item>::new(crate::CanonicalStorage::read(right));
                    crate::merge::merge_control_with(exec, left, right, less)
                }

                fn sort_control<Less>(
                    exec: &Executor<R>,
                    input: &<Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
                    less: Less,
                ) -> Result<crate::DeviceVec<R, u32>, Error>
                where
                    Less: crate::BinaryPredicateOp<Item>,
                {
                    let input = crate::read::Reassociate::<_, Item>::new(crate::CanonicalStorage::read(input));
                    crate::ordering::sort_control_with(exec, input, less)
                }

                fn sort_ordering<Less>(
                    exec: &Executor<R>,
                    input: <Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
                    _less: Less,
                ) -> Result<
                    crate::ordering::sort::OrderingResult<
                        R,
                        <Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
                    >,
                    Error,
                >
                where
                    Less: crate::BinaryPredicateOp<Item>,
                {
                    <Item as crate::ordering::sort::SortStorageItem<R, Less>>::sort_storage(
                        exec, input, true,
                    )
                }

                fn segment_heads<Equal>(
                    exec: &Executor<R>,
                    input: &<Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
                    equal: Equal,
                ) -> Result<crate::DeviceVec<R, u32>, Error>
                where
                    Equal: crate::BinaryPredicateOp<Item>,
                {
                    let input = crate::read::Reassociate::<_, Item>::new(crate::CanonicalStorage::read(input));
                    crate::core::by_key::segment_heads_with(exec, input, equal)
                }

                fn segmented<Op>(
                    exec: &Executor<R>,
                    input: &<Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
                    heads: &crate::DeviceVec<R, u32>,
                    init: Option<Item>,
                    op: Op,
                    mode: u8,
                ) -> Result<<Item as crate::CanonicalAlloc<R>>::CanonicalStorage, Error>
                where
                    Op: crate::ReductionOp<Item>,
                {
                    let input = crate::read::Reassociate::<_, Item>::new(crate::CanonicalStorage::read(input));
                    match mode {
                        0 => crate::core::by_key::segmented_inclusive_with(exec, input, heads, op),
                        1 => crate::core::by_key::segmented_exclusive_with(
                            exec,
                            input,
                            heads,
                            init.expect("exclusive segmented scan requires init"),
                            op,
                        ),
                        2 => crate::core::by_key::segmented_reduced_with(
                            exec,
                            input,
                            heads,
                            init.expect("segmented reduction requires init"),
                            op,
                        ),
                        _ => unreachable!("invalid segmented operation"),
                    }
                }

                reduce_method_impl!(reduce_a1, crate::A1, crate::read::Env1<L0>, Eval1, [L0]; $storage);
                reduce_method_impl!(reduce_a2, crate::A2, crate::read::Env2<L0, L1>, Eval2, [L0, L1]; $storage);
                reduce_method_impl!(reduce_a3, crate::A3, crate::read::Env3<L0, L1, L2>, Eval3, [L0, L1, L2]; $storage);
                reduce_method_impl!(reduce_a4, crate::A4, crate::read::Env4<L0, L1, L2, L3>, Eval4, [L0, L1, L2, L3]; $storage);
                reduce_method_impl!(reduce_a5, crate::A5, crate::read::Env5<L0, L1, L2, L3, L4>, Eval5, [L0, L1, L2, L3, L4]; $storage);
                reduce_method_impl!(reduce_a6, crate::A6, crate::read::Env6<L0, L1, L2, L3, L4, L5>, Eval6, [L0, L1, L2, L3, L4, L5]; $storage);
                reduce_method_impl!(reduce_a7, crate::A7, crate::read::Env7<L0, L1, L2, L3, L4, L5, L6>, Eval7, [L0, L1, L2, L3, L4, L5, L6]; $storage);
                reduce_method_impl!(reduce_a8, crate::A8, crate::read::Env8<L0, L1, L2, L3, L4, L5, L6, L7>, Eval8, [L0, L1, L2, L3, L4, L5, L6, L7]; $storage);

                normalize_method_impl!(normalize_a1, crate::A1, crate::read::Env1<L0>, Eval1, [L0]);
                normalize_method_impl!(normalize_a2, crate::A2, crate::read::Env2<L0, L1>, Eval2, [L0, L1]);
                normalize_method_impl!(normalize_a3, crate::A3, crate::read::Env3<L0, L1, L2>, Eval3, [L0, L1, L2]);
                normalize_method_impl!(normalize_a4, crate::A4, crate::read::Env4<L0, L1, L2, L3>, Eval4, [L0, L1, L2, L3]);
                normalize_method_impl!(normalize_a5, crate::A5, crate::read::Env5<L0, L1, L2, L3, L4>, Eval5, [L0, L1, L2, L3, L4]);
                normalize_method_impl!(normalize_a6, crate::A6, crate::read::Env6<L0, L1, L2, L3, L4, L5>, Eval6, [L0, L1, L2, L3, L4, L5]);
                normalize_method_impl!(normalize_a7, crate::A7, crate::read::Env7<L0, L1, L2, L3, L4, L5, L6>, Eval7, [L0, L1, L2, L3, L4, L5, L6]);
                normalize_method_impl!(normalize_a8, crate::A8, crate::read::Env8<L0, L1, L2, L3, L4, L5, L6, L7>, Eval8, [L0, L1, L2, L3, L4, L5, L6, L7]);
            }
        };
    }

impl<R, Item> KernelItem<R, Item> for crate::storage::Last<Item>
where
    R: Runtime,
    Item: crate::MStorageElement,
{
    type ReboundWrite<Output>
        = Write<
        crate::output::ReassociatedOutput<Output, Item, crate::read::Env1<Item>>,
        crate::read::Env1<Item>,
    >
    where
        Output: crate::output::OutputExpression
            + crate::output::StageOutput<R, crate::read::Env0>
            + crate::selection::FillOutput<R>
            + crate::selection::ReplaceOutput<R>
            + crate::output::SliceOutput,
        Output::Item: crate::WriteFrom<Item>;

    fn rebind_write<Output>(output: Output) -> Self::ReboundWrite<Output>
    where
        Output: crate::output::OutputExpression
            + crate::output::StageOutput<R, crate::read::Env0>
            + crate::selection::FillOutput<R>
            + crate::selection::ReplaceOutput<R>
            + crate::output::SliceOutput,
        Output::Item: crate::WriteFrom<Item>,
    {
        Write::new(crate::output::ReassociatedOutput::new(output))
    }

    fn pair<Op>(
        exec: &Executor<R>,
        left: &<Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
        right: &<Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
        op: Op,
        mode: u8,
    ) -> Result<PairResult, Error>
    where
        Op: crate::BinaryPredicateOp<Item>,
    {
        let left = crate::read::Reassociate::<_, Item>::new(crate::CanonicalStorage::read(left));
        let right = crate::read::Reassociate::<_, Item>::new(crate::CanonicalStorage::read(right));
        match mode {
            0 => Ok(PairResult::Bool(crate::search::equal(
                exec, left, right, op,
            )?)),
            1 => Ok(PairResult::Index(crate::search::mismatch(
                exec, left, right, op,
            )?)),
            2 => Ok(PairResult::Bool(crate::search::lexicographical_compare(
                exec, left, right, op,
            )?)),
            3 => Ok(PairResult::Index(crate::search::find_first_of(
                exec, left, right, op,
            )?)),
            _ => unreachable!("invalid pair operation"),
        }
    }

    fn bounds<Less>(
        exec: &Executor<R>,
        source: &<Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
        values: &<Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
        less: Less,
        upper: bool,
    ) -> Result<crate::DeviceVec<R, u32>, Error>
    where
        Less: crate::BinaryPredicateOp<Item>,
    {
        let source =
            crate::read::Reassociate::<_, Item>::new(crate::CanonicalStorage::read(source));
        let values =
            crate::read::Reassociate::<_, Item>::new(crate::CanonicalStorage::read(values));
        if upper {
            crate::search::upper_bounds_storage(exec, source, values, less)
        } else {
            crate::search::lower_bounds_storage(exec, source, values, less)
        }
    }

    fn merge_control<Less>(
        exec: &Executor<R>,
        left: &<Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
        right: &<Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
        less: Less,
    ) -> Result<crate::merge::MergeControl<R>, Error>
    where
        Less: crate::BinaryPredicateOp<Item>,
    {
        let left = crate::read::Reassociate::<_, Item>::new(crate::CanonicalStorage::read(left));
        let right = crate::read::Reassociate::<_, Item>::new(crate::CanonicalStorage::read(right));
        crate::merge::merge_control_with(exec, left, right, less)
    }

    fn sort_control<Less>(
        exec: &Executor<R>,
        input: &<Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
        less: Less,
    ) -> Result<crate::DeviceVec<R, u32>, Error>
    where
        Less: crate::BinaryPredicateOp<Item>,
    {
        let input = crate::read::Reassociate::<_, Item>::new(crate::CanonicalStorage::read(input));
        crate::ordering::sort_control_with(exec, input, less)
    }

    fn sort_ordering<Less>(
        exec: &Executor<R>,
        input: <Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
        _less: Less,
    ) -> Result<
        crate::ordering::sort::OrderingResult<
            R,
            <Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
        >,
        Error,
    >
    where
        Less: crate::BinaryPredicateOp<Item>,
    {
        <Item as crate::ordering::sort::SortStorageItem<R, Less>>::sort_storage(exec, input, true)
    }

    fn segment_heads<Equal>(
        exec: &Executor<R>,
        input: &<Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
        equal: Equal,
    ) -> Result<crate::DeviceVec<R, u32>, Error>
    where
        Equal: crate::BinaryPredicateOp<Item>,
    {
        let input = crate::read::Reassociate::<_, Item>::new(crate::CanonicalStorage::read(input));
        crate::core::by_key::segment_heads_with(exec, input, equal)
    }

    fn segmented<Op>(
        exec: &Executor<R>,
        input: &<Item as crate::CanonicalAlloc<R>>::CanonicalStorage,
        heads: &crate::DeviceVec<R, u32>,
        init: Option<Item>,
        op: Op,
        mode: u8,
    ) -> Result<<Item as crate::CanonicalAlloc<R>>::CanonicalStorage, Error>
    where
        Op: crate::ReductionOp<Item>,
    {
        let input = crate::read::Reassociate::<_, Item>::new(crate::CanonicalStorage::read(input));
        match mode {
            0 => crate::core::by_key::segmented_inclusive_with(exec, input, heads, op),
            1 => crate::core::by_key::segmented_exclusive_with(
                exec,
                input,
                heads,
                init.expect("exclusive segmented scan requires init"),
                op,
            ),
            2 => crate::core::by_key::segmented_reduced_with(
                exec,
                input,
                heads,
                init.expect("segmented reduction requires init"),
                op,
            ),
            _ => unreachable!("invalid segmented operation"),
        }
    }

    reduce_method_impl!(reduce_a1, crate::A1, crate::read::Env1<L0>, Eval1, [L0]; crate::S1);
    reduce_method_impl!(reduce_a2, crate::A2, crate::read::Env2<L0, L1>, Eval2, [L0, L1]; crate::S1);
    reduce_method_impl!(reduce_a3, crate::A3, crate::read::Env3<L0, L1, L2>, Eval3, [L0, L1, L2]; crate::S1);
    reduce_method_impl!(reduce_a4, crate::A4, crate::read::Env4<L0, L1, L2, L3>, Eval4, [L0, L1, L2, L3]; crate::S1);
    reduce_method_impl!(reduce_a5, crate::A5, crate::read::Env5<L0, L1, L2, L3, L4>, Eval5, [L0, L1, L2, L3, L4]; crate::S1);
    reduce_method_impl!(reduce_a6, crate::A6, crate::read::Env6<L0, L1, L2, L3, L4, L5>, Eval6, [L0, L1, L2, L3, L4, L5]; crate::S1);
    reduce_method_impl!(reduce_a7, crate::A7, crate::read::Env7<L0, L1, L2, L3, L4, L5, L6>, Eval7, [L0, L1, L2, L3, L4, L5, L6]; crate::S1);
    reduce_method_impl!(reduce_a8, crate::A8, crate::read::Env8<L0, L1, L2, L3, L4, L5, L6, L7>, Eval8, [L0, L1, L2, L3, L4, L5, L6, L7]; crate::S1);

    normalize_method_impl!(normalize_a1, crate::A1, crate::read::Env1<L0>, Eval1, [L0]);
    normalize_method_impl!(normalize_a2, crate::A2, crate::read::Env2<L0, L1>, Eval2, [L0, L1]);
    normalize_method_impl!(normalize_a3, crate::A3, crate::read::Env3<L0, L1, L2>, Eval3, [L0, L1, L2]);
    normalize_method_impl!(normalize_a4, crate::A4, crate::read::Env4<L0, L1, L2, L3>, Eval4, [L0, L1, L2, L3]);
    normalize_method_impl!(normalize_a5, crate::A5, crate::read::Env5<L0, L1, L2, L3, L4>, Eval5, [L0, L1, L2, L3, L4]);
    normalize_method_impl!(normalize_a6, crate::A6, crate::read::Env6<L0, L1, L2, L3, L4, L5>, Eval6, [L0, L1, L2, L3, L4, L5]);
    normalize_method_impl!(normalize_a7, crate::A7, crate::read::Env7<L0, L1, L2, L3, L4, L5, L6>, Eval7, [L0, L1, L2, L3, L4, L5, L6]);
    normalize_method_impl!(normalize_a8, crate::A8, crate::read::Env8<L0, L1, L2, L3, L4, L5, L6, L7>, Eval8, [L0, L1, L2, L3, L4, L5, L6, L7]);
}
impl_kernel_item!(crate::storage::More<O0, crate::storage::Last<O1>>, crate::S2, crate::read::Env2<O0, O1>, [O0, O1];);
impl_kernel_item!(crate::storage::More<O0, crate::storage::More<O1, crate::storage::Last<O2>>>, crate::S3, crate::read::Env3<O0, O1, O2>, [O0, O1, O2];);
impl_kernel_item!(crate::storage::More<O0, crate::storage::More<O1, crate::storage::More<O2, crate::storage::Last<O3>>>>, crate::S4, crate::read::Env4<O0, O1, O2, O3>, [O0, O1, O2, O3];);
impl_kernel_item!(crate::storage::More<O0, crate::storage::More<O1, crate::storage::More<O2, crate::storage::More<O3, crate::storage::Last<O4>>>>>, crate::S5, crate::read::Env5<O0, O1, O2, O3, O4>, [O0, O1, O2, O3, O4];);
impl_kernel_item!(crate::storage::More<O0, crate::storage::More<O1, crate::storage::More<O2, crate::storage::More<O3, crate::storage::More<O4, crate::storage::Last<O5>>>>>>, crate::S6, crate::read::Env6<O0, O1, O2, O3, O4, O5>, [O0, O1, O2, O3, O4, O5];);
impl_kernel_item!(crate::storage::More<O0, crate::storage::More<O1, crate::storage::More<O2, crate::storage::More<O3, crate::storage::More<O4, crate::storage::More<O5, crate::storage::Last<O6>>>>>>>, crate::S7, crate::read::Env7<O0, O1, O2, O3, O4, O5, O6>, [O0, O1, O2, O3, O4, O5, O6];);

pub trait KernelRead<R: Runtime, Input>: Sized {
    fn normalize(
        input: Input,
        exec: &Executor<R>,
    ) -> Result<<Input::Item as crate::CanonicalAlloc<R>>::CanonicalStorage, Error>
    where
        Input: crate::read::ReadExpression,
        Input::Item: crate::api::iter::MItem<R>;

    fn transform<Output, Op>(
        input: Input,
        exec: &Executor<R>,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Output: KernelWrite<R>,
        Op: crate::UnaryOp<<Input as crate::read::ReadExpression>::Item>,
        Output::Item: crate::WriteFrom<
                <Op as crate::UnaryOp<<Input as crate::read::ReadExpression>::Item>>::Output,
            >,
        Input: crate::read::ReadExpression;

    fn count_if<Pred>(input: Input, exec: &Executor<R>, pred: Pred) -> Result<u32, Error>
    where
        Input: crate::read::ReadExpression,
        Pred: crate::PredicateOp<Input::Item>;

    fn all_of<Pred>(input: Input, exec: &Executor<R>, pred: Pred) -> Result<bool, Error>
    where
        Input: crate::read::ReadExpression,
        Pred: crate::PredicateOp<Input::Item>;

    fn any_of<Pred>(input: Input, exec: &Executor<R>, pred: Pred) -> Result<bool, Error>
    where
        Input: crate::read::ReadExpression,
        Pred: crate::PredicateOp<Input::Item>;

    fn none_of<Pred>(input: Input, exec: &Executor<R>, pred: Pred) -> Result<bool, Error>
    where
        Input: crate::read::ReadExpression,
        Pred: crate::PredicateOp<Input::Item>;

    fn find_if<Pred>(input: Input, exec: &Executor<R>, pred: Pred) -> Result<Option<u32>, Error>
    where
        Input: crate::read::ReadExpression,
        Pred: crate::PredicateOp<Input::Item>;

    fn is_partitioned<Pred>(input: Input, exec: &Executor<R>, pred: Pred) -> Result<bool, Error>
    where
        Input: crate::read::ReadExpression,
        Pred: crate::PredicateOp<Input::Item>;

    fn reduce<Op>(
        input: Input,
        exec: &Executor<R>,
        init: <Input as crate::read::ReadExpression>::Item,
        op: Op,
    ) -> Result<<Input as crate::read::ReadExpression>::Item, Error>
    where
        Input: crate::read::ReadExpression,
        Op: crate::ReductionOp<Input::Item>;

    fn adjacent_find<Equal>(
        input: Input,
        exec: &Executor<R>,
        equal: Equal,
    ) -> Result<Option<u32>, Error>
    where
        Input: crate::read::ReadExpression,
        Equal: crate::BinaryPredicateOp<Input::Item>;

    fn is_sorted_until<Less>(input: Input, exec: &Executor<R>, less: Less) -> Result<u32, Error>
    where
        Input: crate::read::ReadExpression,
        Less: crate::BinaryPredicateOp<Input::Item>;

    fn is_sorted<Less>(input: Input, exec: &Executor<R>, less: Less) -> Result<bool, Error>
    where
        Input: crate::read::ReadExpression,
        Less: crate::BinaryPredicateOp<Input::Item>;

    fn min_element<Less>(
        input: Input,
        exec: &Executor<R>,
        less: Less,
    ) -> Result<Option<u32>, Error>
    where
        Input: crate::read::ReadExpression,
        Less: crate::BinaryPredicateOp<Input::Item>;

    fn max_element<Less>(
        input: Input,
        exec: &Executor<R>,
        less: Less,
    ) -> Result<Option<u32>, Error>
    where
        Input: crate::read::ReadExpression,
        Less: crate::BinaryPredicateOp<Input::Item>;

    fn minmax_element<Less>(
        input: Input,
        exec: &Executor<R>,
        less: Less,
    ) -> Result<Option<(u32, u32)>, Error>
    where
        Input: crate::read::ReadExpression,
        Less: crate::BinaryPredicateOp<Input::Item>;

    fn inclusive_scan<Output, Op>(
        input: Input,
        exec: &Executor<R>,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Input: crate::read::ReadExpression,
        Output: KernelWrite<R, Item = Input::Item>,
        Op: crate::ReductionOp<Input::Item>;

    fn adjacent_difference<Output, Op>(
        input: Input,
        exec: &Executor<R>,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Input: crate::read::ReadExpression,
        Output: KernelWrite<R, Item = Input::Item>,
        Op: crate::ReductionOp<Input::Item>;

    fn sort<Output, Less>(
        input: Input,
        exec: &Executor<R>,
        less: Less,
        output: Output,
    ) -> Result<(), Error>
    where
        Input: crate::read::ReadExpression,
        Output: KernelWrite<R, Item = Input::Item>,
        Less: crate::BinaryPredicateOp<Input::Item>;

    fn unique<Output, Equal>(
        input: Input,
        exec: &Executor<R>,
        equal: Equal,
        output: Output,
    ) -> Result<u32, Error>
    where
        Input: crate::read::ReadExpression,
        Output: KernelWrite<R, Item = Input::Item>,
        Equal: crate::BinaryPredicateOp<Input::Item>;

    fn select<Output>(
        input: Input,
        exec: &Executor<R>,
        flags: crate::Column<u32>,
        invert: bool,
        output: Output,
    ) -> Result<u32, Error>
    where
        Input: crate::read::ReadExpression,
        Output: KernelWrite<R, Item = Input::Item>;

    fn partition<Output, Pred>(
        input: Input,
        exec: &Executor<R>,
        pred: Pred,
        output: Output,
    ) -> Result<u32, Error>
    where
        Input: crate::read::ReadExpression,
        Output: KernelWrite<R, Item = Input::Item>,
        Pred: crate::PredicateOp<Input::Item>;

    fn indexed<Output>(
        input: Input,
        exec: &Executor<R>,
        indices: crate::Column<u32>,
        flags: Option<crate::Column<u32>>,
        scatter: bool,
        output: Output,
    ) -> Result<(), Error>
    where
        Input: crate::read::ReadExpression,
        Output: KernelWrite<R, Item = Input::Item>;

    fn transform_where<Output, Op>(
        input: Input,
        exec: &Executor<R>,
        op: Op,
        flags: crate::Column<u32>,
        output: Output,
    ) -> Result<(), Error>
    where
        Input: crate::read::ReadExpression,
        Output: KernelWrite<R>,
        Op: crate::UnaryOp<Input::Item>,
        Output::Item: crate::WriteFrom<Op::Output>;

    fn exclusive_scan<Output, Op>(
        input: Input,
        exec: &Executor<R>,
        init: Input::Item,
        op: Op,
        output: Output,
    ) -> Result<(), Error>
    where
        Input: crate::read::ReadExpression,
        Output: KernelWrite<R, Item = Input::Item>,
        Op: crate::ReductionOp<Input::Item>;
}

macro_rules! impl_kernel_read {
        ($env:ty, $arity:ty, $eval:ident, $method:ident, $reduce_method:ident, $normalize_method:ident, $scan_method:ident, $sort_method:ident, $unique_method:ident, $select_method:ident, $partition_method:ident, $indexed_method:ident, $transform_where_method:ident, $exclusive_method:ident, [$($leaf:ident),+]) => {
            impl<R, Input, $($leaf),+> KernelRead<R, Input> for $env
            where
                R: Runtime,
                $($leaf: crate::MStorageElement,)+
                Input: Clone + crate::read::ReadExpression<ReadArity = $arity>
                    + crate::read::BindSlots<crate::read::Env0, NextEnv = $env>
                    + crate::read::LowerReadExpression<
                        DeviceExpr = <Input as crate::read::BindSlots<crate::read::Env0>>::Expr,
                        Slots = $env,
                    >
                    + crate::reduce::StageRead<R, crate::read::Env0>,
                <Input as crate::read::BindSlots<crate::read::Env0>>::Expr:
                    crate::eval::$eval<Input::Item, $($leaf),+>,
                Input::Item: crate::api::iter::MItem<R>,
                <Input::Item as crate::StorageLayout>::StorageLeaves:
                    KernelItem<R, Input::Item>,
            {
                fn normalize(
                    input: Input,
                    exec: &Executor<R>,
                ) -> Result<<Input::Item as crate::CanonicalAlloc<R>>::CanonicalStorage, Error>
                where
                    Input::Item: crate::api::iter::MItem<R>,
                {
                    <<Input::Item as crate::StorageLayout>::StorageLeaves as KernelItem<
                        R,
                        Input::Item,
                    >>::$normalize_method(input, exec)
                }

                fn transform<Output, Op>(
                    input: Input,
                    exec: &Executor<R>,
                    op: Op,
                    output: Output,
                ) -> Result<(), Error>
                where
                    Output: KernelWrite<R>,
                    Op: crate::UnaryOp<Input::Item>,
                    Output::Item:
                        crate::WriteFrom<<Op as crate::UnaryOp<Input::Item>>::Output>,
                {
                    output.$method(exec, crate::read::Transform::new(input, op))
                }

                fn count_if<Pred>(
                    input: Input,
                    exec: &Executor<R>,
                    pred: Pred,
                ) -> Result<u32, Error>
                where
                    Pred: crate::PredicateOp<Input::Item>,
                {
                    crate::predicate::count_if(exec, input, pred)
                }

                fn all_of<Pred>(
                    input: Input,
                    exec: &Executor<R>,
                    pred: Pred,
                ) -> Result<bool, Error>
                where
                    Pred: crate::PredicateOp<Input::Item>,
                {
                    crate::predicate::all_of(exec, input, pred)
                }

                fn any_of<Pred>(
                    input: Input,
                    exec: &Executor<R>,
                    pred: Pred,
                ) -> Result<bool, Error>
                where
                    Pred: crate::PredicateOp<Input::Item>,
                {
                    crate::predicate::any_of(exec, input, pred)
                }

                fn none_of<Pred>(
                    input: Input,
                    exec: &Executor<R>,
                    pred: Pred,
                ) -> Result<bool, Error>
                where
                    Pred: crate::PredicateOp<Input::Item>,
                {
                    crate::predicate::none_of(exec, input, pred)
                }

                fn find_if<Pred>(
                    input: Input,
                    exec: &Executor<R>,
                    pred: Pred,
                ) -> Result<Option<u32>, Error>
                where
                    Pred: crate::PredicateOp<Input::Item>,
                {
                    crate::predicate::find_if(exec, input, pred)
                }

                fn is_partitioned<Pred>(
                    input: Input,
                    exec: &Executor<R>,
                    pred: Pred,
                ) -> Result<bool, Error>
                where
                    Pred: crate::PredicateOp<Input::Item>,
                {
                    crate::predicate::is_partitioned(exec, input, pred)
                }

                fn reduce<Op>(
                    input: Input,
                    exec: &Executor<R>,
                    init: Input::Item,
                    op: Op,
                ) -> Result<Input::Item, Error>
                where
                    Op: crate::ReductionOp<Input::Item>,
                {
                    <<Input::Item as crate::StorageLayout>::StorageLeaves as KernelItem<
                        R,
                        Input::Item,
                    >>::$reduce_method(input, exec, init, op)
                }

                fn adjacent_find<Equal>(
                    input: Input,
                    exec: &Executor<R>,
                    equal: Equal,
                ) -> Result<Option<u32>, Error>
                where
                    Equal: crate::BinaryPredicateOp<Input::Item>,
                {
                    crate::ordering::adjacent_find(exec, input, equal)
                }

                fn is_sorted_until<Less>(
                    input: Input,
                    exec: &Executor<R>,
                    less: Less,
                ) -> Result<u32, Error>
                where
                    Less: crate::BinaryPredicateOp<Input::Item>,
                {
                    crate::ordering::is_sorted_until(exec, input, less)
                }

                fn is_sorted<Less>(
                    input: Input,
                    exec: &Executor<R>,
                    less: Less,
                ) -> Result<bool, Error>
                where
                    Less: crate::BinaryPredicateOp<Input::Item>,
                {
                    crate::ordering::is_sorted(exec, input, less)
                }

                fn min_element<Less>(
                    input: Input,
                    exec: &Executor<R>,
                    less: Less,
                ) -> Result<Option<u32>, Error>
                where
                    Less: crate::BinaryPredicateOp<Input::Item>,
                {
                    crate::ordering::min_element(exec, input, less)
                }

                fn max_element<Less>(
                    input: Input,
                    exec: &Executor<R>,
                    less: Less,
                ) -> Result<Option<u32>, Error>
                where
                    Less: crate::BinaryPredicateOp<Input::Item>,
                {
                    crate::ordering::max_element(exec, input, less)
                }

                fn minmax_element<Less>(
                    input: Input,
                    exec: &Executor<R>,
                    less: Less,
                ) -> Result<Option<(u32, u32)>, Error>
                where
                    Less: crate::BinaryPredicateOp<Input::Item>,
                {
                    crate::ordering::minmax_element(exec, input, less)
                }

                fn inclusive_scan<Output, Op>(
                    input: Input,
                    exec: &Executor<R>,
                    op: Op,
                    output: Output,
                ) -> Result<(), Error>
                where
                    Output: KernelWrite<R, Item = Input::Item>,
                    Op: crate::ReductionOp<Input::Item>,
                {
                    output.$scan_method(exec, input, op)
                }

                fn adjacent_difference<Output, Op>(
                    input: Input,
                    exec: &Executor<R>,
                    op: Op,
                    output: Output,
                ) -> Result<(), Error>
                where
                    Output: KernelWrite<R, Item = Input::Item>,
                    Op: crate::ReductionOp<Input::Item>,
                {
                    output.$method(exec, crate::read::Adjacent::new(input, op))
                }

                fn sort<Output, Less>(
                    input: Input,
                    exec: &Executor<R>,
                    less: Less,
                    output: Output,
                ) -> Result<(), Error>
                where
                    Output: KernelWrite<R, Item = Input::Item>,
                    Less: crate::BinaryPredicateOp<Input::Item>,
                {
                    output.$sort_method(exec, input, less)
                }

                fn unique<Output, Equal>(
                    input: Input,
                    exec: &Executor<R>,
                    equal: Equal,
                    output: Output,
                ) -> Result<u32, Error>
                where
                    Output: KernelWrite<R, Item = Input::Item>,
                    Equal: crate::BinaryPredicateOp<Input::Item>,
                {
                    output.$unique_method(exec, input, equal)
                }

                fn select<Output>(
                    input: Input,
                    exec: &Executor<R>,
                    flags: crate::Column<u32>,
                    invert: bool,
                    output: Output,
                ) -> Result<u32, Error>
                where
                    Output: KernelWrite<R, Item = Input::Item>,
                {
                    output.$select_method(exec, input, flags, invert)
                }

                fn partition<Output, Pred>(
                    input: Input,
                    exec: &Executor<R>,
                    pred: Pred,
                    output: Output,
                ) -> Result<u32, Error>
                where
                    Output: KernelWrite<R, Item = Input::Item>,
                    Pred: crate::PredicateOp<Input::Item>,
                {
                    output.$partition_method(exec, input, pred)
                }

                fn indexed<Output>(
                    input: Input,
                    exec: &Executor<R>,
                    indices: crate::Column<u32>,
                    flags: Option<crate::Column<u32>>,
                    scatter: bool,
                    output: Output,
                ) -> Result<(), Error>
                where
                    Output: KernelWrite<R, Item = Input::Item>,
                {
                    output.$indexed_method(exec, input, indices, flags, scatter)
                }

                fn transform_where<Output, Op>(
                    input: Input,
                    exec: &Executor<R>,
                    op: Op,
                    flags: crate::Column<u32>,
                    output: Output,
                ) -> Result<(), Error>
                where
                    Output: KernelWrite<R>,
                    Op: crate::UnaryOp<Input::Item>,
                    Output::Item: crate::WriteFrom<Op::Output>,
                {
                    output.$transform_where_method(exec, input, op, flags)
                }

                fn exclusive_scan<Output, Op>(
                    input: Input,
                    exec: &Executor<R>,
                    init: Input::Item,
                    op: Op,
                    output: Output,
                ) -> Result<(), Error>
                where
                    Output: KernelWrite<R, Item = Input::Item>,
                    Op: crate::ReductionOp<Input::Item>,
                {
                    output.$exclusive_method(exec, input, init, op)
                }

            }
        };
    }

impl_kernel_read!(
    crate::read::Env1<L0>,
    crate::A1,
    Eval1,
    materialize_a1,
    reduce_a1,
    normalize_a1,
    inclusive_scan_a1,
    sort_a1,
    unique_a1,
    select_a1,
    partition_a1,
    indexed_a1,
    transform_where_a1,
    exclusive_a1,
    [L0]
);
impl_kernel_read!(crate::read::Env2<L0, L1>, crate::A2, Eval2, materialize_a2, reduce_a2, normalize_a2, inclusive_scan_a2, sort_a2, unique_a2, select_a2, partition_a2, indexed_a2, transform_where_a2, exclusive_a2, [L0, L1]);
impl_kernel_read!(crate::read::Env3<L0, L1, L2>, crate::A3, Eval3, materialize_a3, reduce_a3, normalize_a3, inclusive_scan_a3, sort_a3, unique_a3, select_a3, partition_a3, indexed_a3, transform_where_a3, exclusive_a3, [L0, L1, L2]);
impl_kernel_read!(crate::read::Env4<L0, L1, L2, L3>, crate::A4, Eval4, materialize_a4, reduce_a4, normalize_a4, inclusive_scan_a4, sort_a4, unique_a4, select_a4, partition_a4, indexed_a4, transform_where_a4, exclusive_a4, [L0, L1, L2, L3]);
impl_kernel_read!(crate::read::Env5<L0, L1, L2, L3, L4>, crate::A5, Eval5, materialize_a5, reduce_a5, normalize_a5, inclusive_scan_a5, sort_a5, unique_a5, select_a5, partition_a5, indexed_a5, transform_where_a5, exclusive_a5, [L0, L1, L2, L3, L4]);
impl_kernel_read!(crate::read::Env6<L0, L1, L2, L3, L4, L5>, crate::A6, Eval6, materialize_a6, reduce_a6, normalize_a6, inclusive_scan_a6, sort_a6, unique_a6, select_a6, partition_a6, indexed_a6, transform_where_a6, exclusive_a6, [L0, L1, L2, L3, L4, L5]);
impl_kernel_read!(crate::read::Env7<L0, L1, L2, L3, L4, L5, L6>, crate::A7, Eval7, materialize_a7, reduce_a7, normalize_a7, inclusive_scan_a7, sort_a7, unique_a7, select_a7, partition_a7, indexed_a7, transform_where_a7, exclusive_a7, [L0, L1, L2, L3, L4, L5, L6]);
impl_kernel_read!(crate::read::Env8<L0, L1, L2, L3, L4, L5, L6, L7>, crate::A8, Eval8, materialize_a8, reduce_a8, normalize_a8, inclusive_scan_a8, sort_a8, unique_a8, select_a8, partition_a8, indexed_a8, transform_where_a8, exclusive_a8, [L0, L1, L2, L3, L4, L5, L6, L7]);

pub trait IterLength {
    fn logical_len(&self) -> Result<usize, Error>;
}

impl<T> IterLength for crate::read::Column<T>
where
    T: crate::MStorageElement,
{
    fn logical_len(&self) -> Result<usize, Error> {
        Ok(self.len())
    }
}

impl<T> IterLength for crate::read::Constant<T> {
    fn logical_len(&self) -> Result<usize, Error> {
        Ok(self.len)
    }
}

impl IterLength for crate::read::Counting {
    fn logical_len(&self) -> Result<usize, Error> {
        Ok(self.len)
    }
}

impl IterLength for crate::read::ReverseCounting {
    fn logical_len(&self) -> Result<usize, Error> {
        Ok(self.len)
    }
}

impl<Source> IterLength for crate::read::Taken<Source> {
    fn logical_len(&self) -> Result<usize, Error> {
        Ok(self.len as usize)
    }
}

impl<Left, Right> IterLength for Zip<Left, Right>
where
    Left: IterLength,
    Right: IterLength,
{
    fn logical_len(&self) -> Result<usize, Error> {
        let left = self.0.logical_len()?;
        let right = self.1.logical_len()?;
        if left == right {
            Ok(left)
        } else {
            Err(Error::LengthMismatch { left, right })
        }
    }
}

impl<Input, Op> IterLength for crate::read::Transform<Input, Op>
where
    Input: IterLength,
{
    fn logical_len(&self) -> Result<usize, Error> {
        self.input.logical_len()
    }
}

impl<Input, Op> IterLength for crate::read::Adjacent<Input, Op>
where
    Input: IterLength,
{
    fn logical_len(&self) -> Result<usize, Error> {
        self.input.logical_len()
    }
}

impl<Values, Indices> IterLength for crate::read::Permute<Values, Indices>
where
    Indices: IterLength,
{
    fn logical_len(&self) -> Result<usize, Error> {
        self.indices.logical_len()
    }
}

impl<Input, Output> IterLength for crate::read::Reassociate<Input, Output>
where
    Input: IterLength,
{
    fn logical_len(&self) -> Result<usize, Error> {
        self.input.logical_len()
    }
}

impl<Runtime, Input> IterLength for crate::read::Slice<Runtime, Input>
where
    Input: IterLength,
{
    fn logical_len(&self) -> Result<usize, Error> {
        self.input.logical_len()
    }
}
