use crate::{
    detail::{
        api::{
            MItemStorage, TransformUnaryOutput, TransformZip2Output, TransformZip3Output,
            TransformZip4Output, TransformZip5Output, TransformZip6Output, TransformZip7Output,
        },
        device::DeviceColumnView,
        op::kernel::UnaryOp,
        policy::CubePolicy,
    },
    error::Error,
};
use cubecl::prelude::*;

pub(in crate::detail) struct TransformPayloadApply;

impl TransformPayloadApply {
    pub(in crate::detail) fn unary<Output, R, Input, Op>(
        policy: &CubePolicy<R>,
        input: DeviceColumnView<R, Input>,
        env: <Op::Env as LaunchArg>::RuntimeArg<R>,
    ) -> Result<<Output as MItemStorage<R>>::Storage, Error>
    where
        R: Runtime,
        Input: CubePrimitive + CubeElement,
        Output: TransformUnaryOutput<R, Input, Op>,
        Op: UnaryOp<(Input,), Output = Output>,
    {
        Output::run(policy, input, env)
    }

    pub(in crate::detail) fn zip2<Output, R, A, B, Op>(
        policy: &CubePolicy<R>,
        a: DeviceColumnView<R, A>,
        b: DeviceColumnView<R, B>,
        env: <Op::Env as LaunchArg>::RuntimeArg<R>,
    ) -> Result<<Output as MItemStorage<R>>::Storage, Error>
    where
        R: Runtime,
        A: CubePrimitive + CubeElement,
        B: CubePrimitive + CubeElement,
        Output: TransformZip2Output<R, A, B, Op>,
        Op: UnaryOp<(A, B), Output = Output>,
    {
        Output::run(policy, a, b, env)
    }

    pub(in crate::detail) fn zip3<Output, R, A, B, C, Op>(
        policy: &CubePolicy<R>,
        a: DeviceColumnView<R, A>,
        b: DeviceColumnView<R, B>,
        c: DeviceColumnView<R, C>,
        env: <Op::Env as LaunchArg>::RuntimeArg<R>,
    ) -> Result<<Output as MItemStorage<R>>::Storage, Error>
    where
        R: Runtime,
        A: CubePrimitive + CubeElement,
        B: CubePrimitive + CubeElement,
        C: CubePrimitive + CubeElement,
        Output: TransformZip3Output<R, A, B, C, Op>,
        Op: UnaryOp<(A, B, C), Output = Output>,
    {
        Output::run(policy, a, b, c, env)
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::detail) fn zip4<Output, R, A, B, C, D, Op>(
        policy: &CubePolicy<R>,
        a: DeviceColumnView<R, A>,
        b: DeviceColumnView<R, B>,
        c: DeviceColumnView<R, C>,
        d: DeviceColumnView<R, D>,
        env: <Op::Env as LaunchArg>::RuntimeArg<R>,
    ) -> Result<<Output as MItemStorage<R>>::Storage, Error>
    where
        R: Runtime,
        A: CubePrimitive + CubeElement,
        B: CubePrimitive + CubeElement,
        C: CubePrimitive + CubeElement,
        D: CubePrimitive + CubeElement,
        Output: TransformZip4Output<R, A, B, C, D, Op>,
        Op: UnaryOp<(A, B, C, D), Output = Output>,
    {
        Output::run(policy, a, b, c, d, env)
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::detail) fn zip5<Output, R, A, B, C, D, E, Op>(
        policy: &CubePolicy<R>,
        a: DeviceColumnView<R, A>,
        b: DeviceColumnView<R, B>,
        c: DeviceColumnView<R, C>,
        d: DeviceColumnView<R, D>,
        e: DeviceColumnView<R, E>,
        env: <Op::Env as LaunchArg>::RuntimeArg<R>,
    ) -> Result<<Output as MItemStorage<R>>::Storage, Error>
    where
        R: Runtime,
        A: CubePrimitive + CubeElement,
        B: CubePrimitive + CubeElement,
        C: CubePrimitive + CubeElement,
        D: CubePrimitive + CubeElement,
        E: CubePrimitive + CubeElement,
        Output: TransformZip5Output<R, A, B, C, D, E, Op>,
        Op: UnaryOp<(A, B, C, D, E), Output = Output>,
    {
        Output::run(policy, a, b, c, d, e, env)
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::detail) fn zip6<Output, R, A, B, C, D, E, F, Op>(
        policy: &CubePolicy<R>,
        a: DeviceColumnView<R, A>,
        b: DeviceColumnView<R, B>,
        c: DeviceColumnView<R, C>,
        d: DeviceColumnView<R, D>,
        e: DeviceColumnView<R, E>,
        f: DeviceColumnView<R, F>,
        env: <Op::Env as LaunchArg>::RuntimeArg<R>,
    ) -> Result<<Output as MItemStorage<R>>::Storage, Error>
    where
        R: Runtime,
        A: CubePrimitive + CubeElement,
        B: CubePrimitive + CubeElement,
        C: CubePrimitive + CubeElement,
        D: CubePrimitive + CubeElement,
        E: CubePrimitive + CubeElement,
        F: CubePrimitive + CubeElement,
        Output: TransformZip6Output<R, A, B, C, D, E, F, Op>,
        Op: UnaryOp<(A, B, C, D, E, F), Output = Output>,
    {
        Output::run(policy, a, b, c, d, e, f, env)
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::detail) fn zip7<Output, R, A, B, C, D, E, F, G, Op>(
        policy: &CubePolicy<R>,
        a: DeviceColumnView<R, A>,
        b: DeviceColumnView<R, B>,
        c: DeviceColumnView<R, C>,
        d: DeviceColumnView<R, D>,
        e: DeviceColumnView<R, E>,
        f: DeviceColumnView<R, F>,
        g: DeviceColumnView<R, G>,
        env: <Op::Env as LaunchArg>::RuntimeArg<R>,
    ) -> Result<<Output as MItemStorage<R>>::Storage, Error>
    where
        R: Runtime,
        A: CubePrimitive + CubeElement,
        B: CubePrimitive + CubeElement,
        C: CubePrimitive + CubeElement,
        D: CubePrimitive + CubeElement,
        E: CubePrimitive + CubeElement,
        F: CubePrimitive + CubeElement,
        G: CubePrimitive + CubeElement,
        Output: TransformZip7Output<R, A, B, C, D, E, F, G, Op>,
        Op: UnaryOp<(A, B, C, D, E, F, G), Output = Output>,
    {
        Output::run(policy, a, b, c, d, e, f, g, env)
    }
}
