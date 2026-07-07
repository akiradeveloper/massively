#![allow(dead_code)]

use super::*;

pub(in crate::detail) struct MergePayloadApply<'a> {
    control: &'a crate::detail::control::MergeByKeyControl,
}

impl<'a> MergePayloadApply<'a> {
    pub(in crate::detail) fn new(control: &'a crate::detail::control::MergeByKeyControl) -> Self {
        Self { control }
    }

    pub(in crate::detail) fn apply_expr<LeftValue, RightValue>(
        &self,
        policy: &crate::policy::CubePolicy<LeftValue::Runtime>,
        left: &LeftValue,
        right: &RightValue,
    ) -> Result<DeviceVec<LeftValue::Runtime, LeftValue::Item>, Error>
    where
        LeftValue: KernelColumn + KernelColumnAt<S0>,
        RightValue:
            KernelColumn<Runtime = LeftValue::Runtime, Item = LeftValue::Item> + KernelColumnAt<S0>,
        LeftValue::Item: CubePrimitive + CubeElement,
        LeftValue::Expr: DeviceGpuExpr<LeftValue::Item>,
        RightValue::Expr: DeviceGpuExpr<RightValue::Item>,
    {
        device_expr_merge_by_key_values_with_control_with_policy(policy, left, right, self.control)
    }

    pub(in crate::detail) fn apply_expr_into<LeftValue, RightValue>(
        &self,
        policy: &crate::policy::CubePolicy<LeftValue::Runtime>,
        left: &LeftValue,
        right: &RightValue,
        output: &DeviceColumnMutView<LeftValue::Runtime, LeftValue::Item>,
    ) -> Result<(), Error>
    where
        LeftValue: KernelColumn + KernelColumnAt<S0>,
        RightValue:
            KernelColumn<Runtime = LeftValue::Runtime, Item = LeftValue::Item> + KernelColumnAt<S0>,
        LeftValue::Item: CubePrimitive + CubeElement,
        LeftValue::Expr: DeviceGpuExpr<LeftValue::Item>,
        RightValue::Expr: DeviceGpuExpr<RightValue::Item>,
    {
        device_expr_merge_by_key_values_into_with_control_with_policy(
            policy,
            left,
            right,
            self.control,
            output,
        )
    }

    pub(in crate::detail) fn apply_expr2<LeftA, LeftB, RightA, RightB>(
        &self,
        policy: &crate::policy::CubePolicy<LeftA::Runtime>,
        left_a: &LeftA,
        left_b: &LeftB,
        right_a: &RightA,
        right_b: &RightB,
    ) -> Result<
        (
            DeviceVec<LeftA::Runtime, LeftA::Item>,
            DeviceVec<LeftA::Runtime, LeftB::Item>,
        ),
        Error,
    >
    where
        LeftA: KernelColumn + KernelColumnAt<S0>,
        LeftB: KernelColumn<Runtime = LeftA::Runtime> + KernelColumnAt<S0>,
        RightA: KernelColumn<Runtime = LeftA::Runtime, Item = LeftA::Item> + KernelColumnAt<S0>,
        RightB: KernelColumn<Runtime = LeftA::Runtime, Item = LeftB::Item> + KernelColumnAt<S0>,
        LeftA::Item: CubePrimitive + CubeElement,
        LeftB::Item: CubePrimitive + CubeElement,
        LeftA::Expr: DeviceGpuExpr<LeftA::Item>,
        LeftB::Expr: DeviceGpuExpr<LeftB::Item>,
        RightA::Expr: DeviceGpuExpr<RightA::Item>,
        RightB::Expr: DeviceGpuExpr<RightB::Item>,
    {
        Ok((
            self.apply_expr(policy, left_a, right_a)?,
            self.apply_expr(policy, left_b, right_b)?,
        ))
    }

    pub(in crate::detail) fn apply_expr3<LeftA, LeftB, LeftC, RightA, RightB, RightC>(
        &self,
        policy: &crate::policy::CubePolicy<LeftA::Runtime>,
        left_a: &LeftA,
        left_b: &LeftB,
        left_c: &LeftC,
        right_a: &RightA,
        right_b: &RightB,
        right_c: &RightC,
    ) -> Result<
        (
            DeviceVec<LeftA::Runtime, LeftA::Item>,
            DeviceVec<LeftA::Runtime, LeftB::Item>,
            DeviceVec<LeftA::Runtime, LeftC::Item>,
        ),
        Error,
    >
    where
        LeftA: KernelColumn + KernelColumnAt<S0>,
        LeftB: KernelColumn<Runtime = LeftA::Runtime> + KernelColumnAt<S0>,
        LeftC: KernelColumn<Runtime = LeftA::Runtime> + KernelColumnAt<S0>,
        RightA: KernelColumn<Runtime = LeftA::Runtime, Item = LeftA::Item> + KernelColumnAt<S0>,
        RightB: KernelColumn<Runtime = LeftA::Runtime, Item = LeftB::Item> + KernelColumnAt<S0>,
        RightC: KernelColumn<Runtime = LeftA::Runtime, Item = LeftC::Item> + KernelColumnAt<S0>,
        LeftA::Item: CubePrimitive + CubeElement,
        LeftB::Item: CubePrimitive + CubeElement,
        LeftC::Item: CubePrimitive + CubeElement,
        LeftA::Expr: DeviceGpuExpr<LeftA::Item>,
        LeftB::Expr: DeviceGpuExpr<LeftB::Item>,
        LeftC::Expr: DeviceGpuExpr<LeftC::Item>,
        RightA::Expr: DeviceGpuExpr<RightA::Item>,
        RightB::Expr: DeviceGpuExpr<RightB::Item>,
        RightC::Expr: DeviceGpuExpr<RightC::Item>,
    {
        Ok((
            self.apply_expr(policy, left_a, right_a)?,
            self.apply_expr(policy, left_b, right_b)?,
            self.apply_expr(policy, left_c, right_c)?,
        ))
    }

    pub(in crate::detail) fn apply_expr4<LA, LB, LC, LD, RA, RB, RC, RD>(
        &self,
        policy: &crate::policy::CubePolicy<LA::Runtime>,
        left: (&LA, &LB, &LC, &LD),
        right: (&RA, &RB, &RC, &RD),
    ) -> Result<
        (
            DeviceVec<LA::Runtime, LA::Item>,
            DeviceVec<LA::Runtime, LB::Item>,
            DeviceVec<LA::Runtime, LC::Item>,
            DeviceVec<LA::Runtime, LD::Item>,
        ),
        Error,
    >
    where
        LA: KernelColumn + KernelColumnAt<S0>,
        LB: KernelColumn<Runtime = LA::Runtime> + KernelColumnAt<S0>,
        LC: KernelColumn<Runtime = LA::Runtime> + KernelColumnAt<S0>,
        LD: KernelColumn<Runtime = LA::Runtime> + KernelColumnAt<S0>,
        RA: KernelColumn<Runtime = LA::Runtime, Item = LA::Item> + KernelColumnAt<S0>,
        RB: KernelColumn<Runtime = LA::Runtime, Item = LB::Item> + KernelColumnAt<S0>,
        RC: KernelColumn<Runtime = LA::Runtime, Item = LC::Item> + KernelColumnAt<S0>,
        RD: KernelColumn<Runtime = LA::Runtime, Item = LD::Item> + KernelColumnAt<S0>,
        LA::Item: CubePrimitive + CubeElement,
        LB::Item: CubePrimitive + CubeElement,
        LC::Item: CubePrimitive + CubeElement,
        LD::Item: CubePrimitive + CubeElement,
        LA::Expr: DeviceGpuExpr<LA::Item>,
        LB::Expr: DeviceGpuExpr<LB::Item>,
        LC::Expr: DeviceGpuExpr<LC::Item>,
        LD::Expr: DeviceGpuExpr<LD::Item>,
        RA::Expr: DeviceGpuExpr<RA::Item>,
        RB::Expr: DeviceGpuExpr<RB::Item>,
        RC::Expr: DeviceGpuExpr<RC::Item>,
        RD::Expr: DeviceGpuExpr<RD::Item>,
    {
        Ok((
            self.apply_expr(policy, left.0, right.0)?,
            self.apply_expr(policy, left.1, right.1)?,
            self.apply_expr(policy, left.2, right.2)?,
            self.apply_expr(policy, left.3, right.3)?,
        ))
    }

    pub(in crate::detail) fn apply_expr5<LA, LB, LC, LD, LE, RA, RB, RC, RD, RE>(
        &self,
        policy: &crate::policy::CubePolicy<LA::Runtime>,
        left: (&LA, &LB, &LC, &LD, &LE),
        right: (&RA, &RB, &RC, &RD, &RE),
    ) -> Result<
        (
            DeviceVec<LA::Runtime, LA::Item>,
            DeviceVec<LA::Runtime, LB::Item>,
            DeviceVec<LA::Runtime, LC::Item>,
            DeviceVec<LA::Runtime, LD::Item>,
            DeviceVec<LA::Runtime, LE::Item>,
        ),
        Error,
    >
    where
        LA: KernelColumn + KernelColumnAt<S0>,
        LB: KernelColumn<Runtime = LA::Runtime> + KernelColumnAt<S0>,
        LC: KernelColumn<Runtime = LA::Runtime> + KernelColumnAt<S0>,
        LD: KernelColumn<Runtime = LA::Runtime> + KernelColumnAt<S0>,
        LE: KernelColumn<Runtime = LA::Runtime> + KernelColumnAt<S0>,
        RA: KernelColumn<Runtime = LA::Runtime, Item = LA::Item> + KernelColumnAt<S0>,
        RB: KernelColumn<Runtime = LA::Runtime, Item = LB::Item> + KernelColumnAt<S0>,
        RC: KernelColumn<Runtime = LA::Runtime, Item = LC::Item> + KernelColumnAt<S0>,
        RD: KernelColumn<Runtime = LA::Runtime, Item = LD::Item> + KernelColumnAt<S0>,
        RE: KernelColumn<Runtime = LA::Runtime, Item = LE::Item> + KernelColumnAt<S0>,
        LA::Item: CubePrimitive + CubeElement,
        LB::Item: CubePrimitive + CubeElement,
        LC::Item: CubePrimitive + CubeElement,
        LD::Item: CubePrimitive + CubeElement,
        LE::Item: CubePrimitive + CubeElement,
        LA::Expr: DeviceGpuExpr<LA::Item>,
        LB::Expr: DeviceGpuExpr<LB::Item>,
        LC::Expr: DeviceGpuExpr<LC::Item>,
        LD::Expr: DeviceGpuExpr<LD::Item>,
        LE::Expr: DeviceGpuExpr<LE::Item>,
        RA::Expr: DeviceGpuExpr<RA::Item>,
        RB::Expr: DeviceGpuExpr<RB::Item>,
        RC::Expr: DeviceGpuExpr<RC::Item>,
        RD::Expr: DeviceGpuExpr<RD::Item>,
        RE::Expr: DeviceGpuExpr<RE::Item>,
    {
        let (a, b, c, d) = self.apply_expr4(
            policy,
            (left.0, left.1, left.2, left.3),
            (right.0, right.1, right.2, right.3),
        )?;
        Ok((a, b, c, d, self.apply_expr(policy, left.4, right.4)?))
    }

    pub(in crate::detail) fn apply_expr6<LA, LB, LC, LD, LE, LF, RA, RB, RC, RD, RE, RF>(
        &self,
        policy: &crate::policy::CubePolicy<LA::Runtime>,
        left: (&LA, &LB, &LC, &LD, &LE, &LF),
        right: (&RA, &RB, &RC, &RD, &RE, &RF),
    ) -> Result<
        (
            DeviceVec<LA::Runtime, LA::Item>,
            DeviceVec<LA::Runtime, LB::Item>,
            DeviceVec<LA::Runtime, LC::Item>,
            DeviceVec<LA::Runtime, LD::Item>,
            DeviceVec<LA::Runtime, LE::Item>,
            DeviceVec<LA::Runtime, LF::Item>,
        ),
        Error,
    >
    where
        LA: KernelColumn + KernelColumnAt<S0>,
        LB: KernelColumn<Runtime = LA::Runtime> + KernelColumnAt<S0>,
        LC: KernelColumn<Runtime = LA::Runtime> + KernelColumnAt<S0>,
        LD: KernelColumn<Runtime = LA::Runtime> + KernelColumnAt<S0>,
        LE: KernelColumn<Runtime = LA::Runtime> + KernelColumnAt<S0>,
        LF: KernelColumn<Runtime = LA::Runtime> + KernelColumnAt<S0>,
        RA: KernelColumn<Runtime = LA::Runtime, Item = LA::Item> + KernelColumnAt<S0>,
        RB: KernelColumn<Runtime = LA::Runtime, Item = LB::Item> + KernelColumnAt<S0>,
        RC: KernelColumn<Runtime = LA::Runtime, Item = LC::Item> + KernelColumnAt<S0>,
        RD: KernelColumn<Runtime = LA::Runtime, Item = LD::Item> + KernelColumnAt<S0>,
        RE: KernelColumn<Runtime = LA::Runtime, Item = LE::Item> + KernelColumnAt<S0>,
        RF: KernelColumn<Runtime = LA::Runtime, Item = LF::Item> + KernelColumnAt<S0>,
        LA::Item: CubePrimitive + CubeElement,
        LB::Item: CubePrimitive + CubeElement,
        LC::Item: CubePrimitive + CubeElement,
        LD::Item: CubePrimitive + CubeElement,
        LE::Item: CubePrimitive + CubeElement,
        LF::Item: CubePrimitive + CubeElement,
        LA::Expr: DeviceGpuExpr<LA::Item>,
        LB::Expr: DeviceGpuExpr<LB::Item>,
        LC::Expr: DeviceGpuExpr<LC::Item>,
        LD::Expr: DeviceGpuExpr<LD::Item>,
        LE::Expr: DeviceGpuExpr<LE::Item>,
        LF::Expr: DeviceGpuExpr<LF::Item>,
        RA::Expr: DeviceGpuExpr<RA::Item>,
        RB::Expr: DeviceGpuExpr<RB::Item>,
        RC::Expr: DeviceGpuExpr<RC::Item>,
        RD::Expr: DeviceGpuExpr<RD::Item>,
        RE::Expr: DeviceGpuExpr<RE::Item>,
        RF::Expr: DeviceGpuExpr<RF::Item>,
    {
        let (a, b, c, d, e) = self.apply_expr5(
            policy,
            (left.0, left.1, left.2, left.3, left.4),
            (right.0, right.1, right.2, right.3, right.4),
        )?;
        Ok((a, b, c, d, e, self.apply_expr(policy, left.5, right.5)?))
    }

    pub(in crate::detail) fn apply_expr7<LA, LB, LC, LD, LE, LF, LG, RA, RB, RC, RD, RE, RF, RG>(
        &self,
        policy: &crate::policy::CubePolicy<LA::Runtime>,
        left: (&LA, &LB, &LC, &LD, &LE, &LF, &LG),
        right: (&RA, &RB, &RC, &RD, &RE, &RF, &RG),
    ) -> Result<
        (
            DeviceVec<LA::Runtime, LA::Item>,
            DeviceVec<LA::Runtime, LB::Item>,
            DeviceVec<LA::Runtime, LC::Item>,
            DeviceVec<LA::Runtime, LD::Item>,
            DeviceVec<LA::Runtime, LE::Item>,
            DeviceVec<LA::Runtime, LF::Item>,
            DeviceVec<LA::Runtime, LG::Item>,
        ),
        Error,
    >
    where
        LA: KernelColumn + KernelColumnAt<S0>,
        LB: KernelColumn<Runtime = LA::Runtime> + KernelColumnAt<S0>,
        LC: KernelColumn<Runtime = LA::Runtime> + KernelColumnAt<S0>,
        LD: KernelColumn<Runtime = LA::Runtime> + KernelColumnAt<S0>,
        LE: KernelColumn<Runtime = LA::Runtime> + KernelColumnAt<S0>,
        LF: KernelColumn<Runtime = LA::Runtime> + KernelColumnAt<S0>,
        LG: KernelColumn<Runtime = LA::Runtime> + KernelColumnAt<S0>,
        RA: KernelColumn<Runtime = LA::Runtime, Item = LA::Item> + KernelColumnAt<S0>,
        RB: KernelColumn<Runtime = LA::Runtime, Item = LB::Item> + KernelColumnAt<S0>,
        RC: KernelColumn<Runtime = LA::Runtime, Item = LC::Item> + KernelColumnAt<S0>,
        RD: KernelColumn<Runtime = LA::Runtime, Item = LD::Item> + KernelColumnAt<S0>,
        RE: KernelColumn<Runtime = LA::Runtime, Item = LE::Item> + KernelColumnAt<S0>,
        RF: KernelColumn<Runtime = LA::Runtime, Item = LF::Item> + KernelColumnAt<S0>,
        RG: KernelColumn<Runtime = LA::Runtime, Item = LG::Item> + KernelColumnAt<S0>,
        LA::Item: CubePrimitive + CubeElement,
        LB::Item: CubePrimitive + CubeElement,
        LC::Item: CubePrimitive + CubeElement,
        LD::Item: CubePrimitive + CubeElement,
        LE::Item: CubePrimitive + CubeElement,
        LF::Item: CubePrimitive + CubeElement,
        LG::Item: CubePrimitive + CubeElement,
        LA::Expr: DeviceGpuExpr<LA::Item>,
        LB::Expr: DeviceGpuExpr<LB::Item>,
        LC::Expr: DeviceGpuExpr<LC::Item>,
        LD::Expr: DeviceGpuExpr<LD::Item>,
        LE::Expr: DeviceGpuExpr<LE::Item>,
        LF::Expr: DeviceGpuExpr<LF::Item>,
        LG::Expr: DeviceGpuExpr<LG::Item>,
        RA::Expr: DeviceGpuExpr<RA::Item>,
        RB::Expr: DeviceGpuExpr<RB::Item>,
        RC::Expr: DeviceGpuExpr<RC::Item>,
        RD::Expr: DeviceGpuExpr<RD::Item>,
        RE::Expr: DeviceGpuExpr<RE::Item>,
        RF::Expr: DeviceGpuExpr<RF::Item>,
        RG::Expr: DeviceGpuExpr<RG::Item>,
    {
        let (a, b, c, d, e, f) = self.apply_expr6(
            policy,
            (left.0, left.1, left.2, left.3, left.4, left.5),
            (right.0, right.1, right.2, right.3, right.4, right.5),
        )?;
        Ok((a, b, c, d, e, f, self.apply_expr(policy, left.6, right.6)?))
    }
}
