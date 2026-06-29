use super::*;

macro_rules! impl_miter_view {
    ($input:ident; 0, 1) => {
        crate::detail::device::SoAView2 {
            left: $input.0,
            right: $input.1,
        }
    };

    ($input:ident; 0, 1, 2) => {
        crate::detail::device::SoAView3 {
            first: $input.0,
            second: $input.1,
            third: $input.2,
        }
    };
}

macro_rules! impl_unique_by_key_dispatch_body {
    ($self:ident, $policy:ident, $values:ident, $eq:ident, $input:ident; 0, 1, 2) => {{
        <Values as sealed::MIterDispatch<R>>::unique_by_three_key_dispatch(
            $values, $policy, $input.0, $input.1, $input.2, $eq,
        )
    }};
    ($self:ident, $policy:ident, $values:ident, $eq:ident, $input:ident; $( $idx:tt ),+) => {{
        let _ = ($policy, $values, $eq, $input);
        Err(Error::Launch {
            message: "unique_by_key is not supported for this key iterator shape".to_string(),
        })
    }};
}

macro_rules! impl_sort_by_key_dispatch_body {
    ($policy:ident, $values:ident, $less:ident, $input:ident; 0, 1, 2) => {{
        <Values as sealed::MIterDispatch<R>>::sort_by_three_key_dispatch(
            $values, $policy, $input.0, $input.1, $input.2, $less,
        )
    }};
    ($policy:ident, $values:ident, $less:ident, $input:ident; $( $idx:tt ),+) => {{
        let _ = ($policy, $values, $less, $input);
        Err(Error::Launch {
            message: "sort_by_key is not supported for this key iterator shape".to_string(),
        })
    }};
}

macro_rules! impl_inclusive_scan_by_key_dispatch_body {
    ($policy:ident, $values:ident, $key_eq:ident, $op:ident, $input:ident; 0, 1, 2) => {{
        <Values as sealed::MIterDispatch<R>>::inclusive_scan_by_three_key_dispatch(
            $values, $policy, $input.0, $input.1, $input.2, $key_eq, $op,
        )
    }};
    ($policy:ident, $values:ident, $key_eq:ident, $op:ident, $input:ident; $( $idx:tt ),+) => {{
        let _ = ($policy, $values, $key_eq, $op, $input);
        Err(Error::Launch {
            message: "inclusive_scan_by_key is not supported for this key iterator shape"
                .to_string(),
        })
    }};
}

macro_rules! impl_exclusive_scan_by_key_dispatch_body {
    ($policy:ident, $values:ident, $key_eq:ident, $init:ident, $op:ident, $input:ident; 0, 1, 2) => {{
        <Values as sealed::MIterDispatch<R>>::exclusive_scan_by_three_key_dispatch(
            $values, $policy, $input.0, $input.1, $input.2, $key_eq, $init, $op,
        )
    }};
    ($policy:ident, $values:ident, $key_eq:ident, $init:ident, $op:ident, $input:ident; $( $idx:tt ),+) => {{
        let _ = ($policy, $values, $key_eq, $init, $op, $input);
        Err(Error::Launch {
            message: "exclusive_scan_by_key is not supported for this key iterator shape"
                .to_string(),
        })
    }};
}

macro_rules! impl_reduce_by_key_dispatch_body {
    ($policy:ident, $values:ident, $key_eq:ident, $init:ident, $op:ident, $input:ident; 0, 1, 2) => {{
        <Values as sealed::MIterDispatch<R>>::reduce_by_three_key_dispatch(
            $values, $policy, $input.0, $input.1, $input.2, $key_eq, $init, $op,
        )
    }};
    ($policy:ident, $values:ident, $key_eq:ident, $init:ident, $op:ident, $input:ident; $( $idx:tt ),+) => {{
        let _ = ($policy, $values, $key_eq, $init, $op, $input);
        Err(Error::Launch {
            message: "reduce_by_key is not supported for this key iterator shape".to_string(),
        })
    }};
}

macro_rules! impl_merge_by_key_dispatch_body {
    ($policy:ident, $right_keys:ident, $left_values:ident, $right_values:ident, $less:ident, $left_input:ident; 0, 1, 2) => {{
        let right_input = $right_keys.into_inner_with_policy($policy)?;
        <LeftValues as sealed::MIterDispatch<R>>::merge_by_three_key_same_dispatch(
            $left_values,
            $policy,
            $left_input.0,
            $left_input.1,
            $left_input.2,
            right_input.0,
            right_input.1,
            right_input.2,
            $right_values,
            $less,
        )
    }};
    ($policy:ident, $right_keys:ident, $left_values:ident, $right_values:ident, $less:ident, $left_input:ident; $( $idx:tt ),+) => {{
        let _ = (
            $policy,
            $right_keys,
            $left_values,
            $right_values,
            $less,
            $left_input,
        );
        Err(Error::Launch {
            message: "merge_by_key is not supported for this key iterator shape".to_string(),
        })
    }};
}

macro_rules! impl_wide_transform_dispatch_body {
    ($policy:ident, $input:ident, $op:ident; 0, 1, 2, 3, 4, 5, 6) => {{
        <Output::Item as sealed::MItemDispatch<R>>::transform_septenary(
            $policy, $input.0, $input.1, $input.2, $input.3, $input.4, $input.5, $input.6, $op,
        )
    }};
    ($policy:ident, $input:ident, $op:ident; $( $idx:tt ),+) => {{
        let _ = ($policy, $input, $op);
        Err(Error::Launch {
            message: "transform is not supported for this iterator shape".to_string(),
        })
    }};
}

macro_rules! impl_wide_sort_by_single_key_dispatch_body {
    ($policy:ident, $keys:ident, $less:ident, $input:ident; 0, 1, 2, 3, 4, 5, 6) => {{
        crate::detail::sort_by_key(
            $policy,
            ($keys,),
            (
                $input.0, $input.1, $input.2, $input.3, $input.4, $input.5, $input.6,
            ),
            KernelOp::<R, Less>::new(),
        )
    }};
    ($policy:ident, $keys:ident, $less:ident, $input:ident; $( $idx:tt ),+) => {{
        let _ = ($policy, $keys, $less, $input);
        Err(Error::Launch {
            message: "sort_by_key is not supported for this value iterator shape".to_string(),
        })
    }};
}

macro_rules! impl_wide_sort_dispatch_body {
    ($policy:ident, $input:ident; 0, 1, 2, 3, 4, 5, 6) => {{
        let indices = crate::detail::primitives::ordering::sort_tuple7_indices_input::<
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            KernelOp<R, Less>,
        >(
            $policy,
            &$input.0,
            &$input.1,
            &$input.2,
            &$input.3,
            &$input.4,
            &$input.5,
            &$input.6,
            crate::op::GpuOp::<KernelOp<R, Less>>::new(),
        )?;
        Ok((
            crate::detail::api::device_expr_gather_with_policy($policy, &$input.0, &indices)?,
            crate::detail::api::device_expr_gather_with_policy($policy, &$input.1, &indices)?,
            crate::detail::api::device_expr_gather_with_policy($policy, &$input.2, &indices)?,
            crate::detail::api::device_expr_gather_with_policy($policy, &$input.3, &indices)?,
            crate::detail::api::device_expr_gather_with_policy($policy, &$input.4, &indices)?,
            crate::detail::api::device_expr_gather_with_policy($policy, &$input.5, &indices)?,
            crate::detail::api::device_expr_gather_with_policy($policy, &$input.6, &indices)?,
        ))
    }};
    ($policy:ident, $input:ident; $( $idx:tt ),+) => {{
        let _ = ($policy, $input);
        Err(Error::Launch {
            message: "sort is not supported for this iterator shape".to_string(),
        })
    }};
}

macro_rules! impl_wide_sort_by_three_key_dispatch_body {
    ($policy:ident, $first_key:ident, $second_key:ident, $third_key:ident, $less:ident, $input:ident; 0, 1, 2, 3, 4, 5, 6) => {{
        crate::detail::sort_by_key(
            $policy,
            ($first_key, $second_key, $third_key),
            (
                $input.0, $input.1, $input.2, $input.3, $input.4, $input.5, $input.6,
            ),
            KernelOp::<R, Less>::new(),
        )
    }};
    ($policy:ident, $first_key:ident, $second_key:ident, $third_key:ident, $less:ident, $input:ident; $( $idx:tt ),+) => {{
        let _ = ($policy, $first_key, $second_key, $third_key, $less, $input);
        Err(Error::Launch {
            message: "sort_by_key is not supported for this value iterator shape".to_string(),
        })
    }};
}

macro_rules! impl_wide_merge_by_three_key_dispatch_body {
    ($policy:ident, $left_values:ident, $right_values:ident, $left_first_key:ident, $left_second_key:ident, $left_third_key:ident, $right_first_key:ident, $right_second_key:ident, $right_third_key:ident, $less:ident; 0, 1, 2, 3, 4, 5, 6) => {{
        crate::detail::merge_by_key(
            $policy,
            ($left_first_key, $left_second_key, $left_third_key),
            (
                $left_values.0,
                $left_values.1,
                $left_values.2,
                $left_values.3,
                $left_values.4,
                $left_values.5,
                $left_values.6,
            ),
            ($right_first_key, $right_second_key, $right_third_key),
            (
                $right_values.0,
                $right_values.1,
                $right_values.2,
                $right_values.3,
                $right_values.4,
                $right_values.5,
                $right_values.6,
            ),
            KernelOp::<R, Less>::new(),
        )
    }};
    ($policy:ident, $left_values:ident, $right_values:ident, $left_first_key:ident, $left_second_key:ident, $left_third_key:ident, $right_first_key:ident, $right_second_key:ident, $right_third_key:ident, $less:ident; $( $idx:tt ),+) => {{
        let _ = (
            $policy,
            $left_values,
            $right_values,
            $left_first_key,
            $left_second_key,
            $left_third_key,
            $right_first_key,
            $right_second_key,
            $right_third_key,
            $less,
        );
        Err(Error::Launch {
            message: "merge_by_key is not supported for this value iterator shape".to_string(),
        })
    }};
}

macro_rules! impl_wide_unique_dispatch_body {
    ($policy:ident, $input:ident; 0, 1, 2, 3, 4, 5, 6) => {{
        crate::detail::read::unique_tuple7_flags_read::<_, _, _, _, _, _, _, KernelOp<R, Pred>>(
            $policy, &$input.0, &$input.1, &$input.2, &$input.3, &$input.4, &$input.5, &$input.6,
        )
    }};
    ($policy:ident, $input:ident; $( $idx:tt ),+) => {{
        let _ = $policy;
        let _ = &$input;
        Err(Error::Launch {
            message: "unique is not supported for this iterator shape".to_string(),
        })
    }};
}

macro_rules! impl_wide_reduce_dispatch_body {
    ($policy:ident, $input:ident, $init:ident; $ty0:ident, $ty1:ident, $ty2:ident, $ty3:ident, $ty4:ident, $ty5:ident, $ty6:ident; 0, 1, 2, 3, 4, 5, 6) => {{
        use crate::detail::device::KernelColumn;
        let a = KernelColumn::stage(&$input.0, $policy)?;
        let b = KernelColumn::stage(&$input.1, $policy)?;
        let c = KernelColumn::stage(&$input.2, $policy)?;
        let d = KernelColumn::stage(&$input.3, $policy)?;
        let e = KernelColumn::stage(&$input.4, $policy)?;
        let f = KernelColumn::stage(&$input.5, $policy)?;
        let g = KernelColumn::stage(&$input.6, $policy)?;
        crate::detail::primitives::reduce::reduce_tuple7_device_expr::<
            R,
            $ty0,
            $ty1,
            $ty2,
            $ty3,
            $ty4,
            $ty5,
            $ty6,
            <crate::detail::device::DeviceColumnView<R, $ty0> as KernelColumn>::Expr,
            <crate::detail::device::DeviceColumnView<R, $ty1> as KernelColumn>::Expr,
            <crate::detail::device::DeviceColumnView<R, $ty2> as KernelColumn>::Expr,
            <crate::detail::device::DeviceColumnView<R, $ty3> as KernelColumn>::Expr,
            <crate::detail::device::DeviceColumnView<R, $ty4> as KernelColumn>::Expr,
            <crate::detail::device::DeviceColumnView<R, $ty5> as KernelColumn>::Expr,
            <crate::detail::device::DeviceColumnView<R, $ty6> as KernelColumn>::Expr,
            KernelOp<R, Op>,
        >($policy, &a, &b, &c, &d, &e, &f, &g, $input.0.len, $init)
    }};
    ($policy:ident, $input:ident, $init:ident; $( $ty:ident ),+; $( $idx:tt ),+) => {{
        let _ = ($policy, $init);
        let _ = &$input;
        Err(Error::Launch {
            message: "reduce is not supported for this iterator shape".to_string(),
        })
    }};
}

macro_rules! impl_wide_inclusive_scan_dispatch_body {
    ($policy:ident, $input:ident; $ty0:ident, $ty1:ident, $ty2:ident, $ty3:ident, $ty4:ident, $ty5:ident, $ty6:ident; 0, 1, 2, 3, 4, 5, 6) => {{
        crate::detail::primitives::scan::inclusive_scan_tuple7_device_views::<
            R,
            $ty0,
            $ty1,
            $ty2,
            $ty3,
            $ty4,
            $ty5,
            $ty6,
            KernelOp<R, Op>,
        >(
            $policy, &$input.0, &$input.1, &$input.2, &$input.3, &$input.4, &$input.5, &$input.6,
        )
    }};
    ($policy:ident, $input:ident; $( $ty:ident ),+; $( $idx:tt ),+) => {{
        let _ = $policy;
        let _ = &$input;
        Err(Error::Launch {
            message: "inclusive_scan is not supported for this iterator shape".to_string(),
        })
    }};
}

macro_rules! impl_wide_exclusive_scan_dispatch_body {
    ($policy:ident, $input:ident, $init:ident; $ty0:ident, $ty1:ident, $ty2:ident, $ty3:ident, $ty4:ident, $ty5:ident, $ty6:ident; 0, 1, 2, 3, 4, 5, 6) => {{
        crate::detail::primitives::scan::exclusive_scan_tuple7_device_views::<
            R,
            $ty0,
            $ty1,
            $ty2,
            $ty3,
            $ty4,
            $ty5,
            $ty6,
            KernelOp<R, Op>,
        >(
            $policy, &$input.0, &$input.1, &$input.2, &$input.3, &$input.4, &$input.5, &$input.6,
            $init,
        )
    }};
    ($policy:ident, $input:ident, $init:ident; $( $ty:ident ),+; $( $idx:tt ),+) => {{
        let _ = ($policy, $init);
        let _ = &$input;
        Err(Error::Launch {
            message: "exclusive_scan is not supported for this iterator shape".to_string(),
        })
    }};
}

macro_rules! impl_tuple_inclusive_scan_by_three_key_values_body {
    ($policy:ident, $input:ident, $control:ident; 0, 1) => {{
        crate::detail::read::inclusive_scan_by_flags_two::<
            _,
            _,
            (K1, K2, K3),
            (),
            KernelOp<R, KeyEq>,
            KernelOp<R, Op>,
        >($policy, &$input.0, &$input.1, &$control)
        .map(|inner| (inner.left, inner.right))
    }};
    ($policy:ident, $input:ident, $control:ident; 0, 1, 2) => {{
        crate::detail::read::inclusive_scan_by_flags_three::<
            _,
            _,
            _,
            (K1, K2, K3),
            (),
            KernelOp<R, KeyEq>,
            KernelOp<R, Op>,
        >($policy, &$input.0, &$input.1, &$input.2, &$control)
        .map(|inner| (inner.first, inner.second, inner.third))
    }};
}

macro_rules! impl_tuple_exclusive_scan_by_three_key_values_body {
    ($policy:ident, $input:ident, $control:ident, $init:ident; 0, 1) => {{
        crate::detail::read::exclusive_scan_by_flags_two::<
            _,
            _,
            (K1, K2, K3),
            (),
            KernelOp<R, KeyEq>,
            KernelOp<R, Op>,
        >($policy, &$input.0, &$input.1, &$control, $init)
        .map(|inner| (inner.left, inner.right))
    }};
    ($policy:ident, $input:ident, $control:ident, $init:ident; 0, 1, 2) => {{
        crate::detail::read::exclusive_scan_by_flags_three::<
            _,
            _,
            _,
            (K1, K2, K3),
            (),
            KernelOp<R, KeyEq>,
            KernelOp<R, Op>,
        >($policy, &$input.0, &$input.1, &$input.2, &$control, $init)
        .map(|inner| (inner.first, inner.second, inner.third))
    }};
}

macro_rules! impl_tuple_reduce_by_three_key_values_body {
    ($policy:ident, $input:ident, $init:ident, $first_key:ident, $second_key:ident, $third_key:ident, $head_flags:ident, $end_flags:ident, $len_u32:ident, $control:ident; $ty0:ident, $ty1:ident; 0, 1) => {{
        let inclusive = crate::detail::read::inclusive_scan_by_flags_two::<
            _,
            _,
            (K1, K2, K3),
            (),
            KernelOp<R, KeyEq>,
            KernelOp<R, Op>,
        >($policy, &$input.0, &$input.1, &$control)?;
        let client = $policy.client();
        let len_handle = client.create_from_slice(u32::as_bytes(&[$len_u32]));
        let init_a = client.create_from_slice($ty0::as_bytes(&[$init.0]));
        let init_b = client.create_from_slice($ty1::as_bytes(&[$init.1]));
        let reduced_a_handle = client.empty($first_key.len * std::mem::size_of::<$ty0>());
        let reduced_b_handle = client.empty($first_key.len * std::mem::size_of::<$ty1>());
        let num_blocks = $first_key
            .len
            .div_ceil(crate::detail::primitives::scan::BLOCK_SCAN_SIZE as usize);
        let num_blocks_u32 =
            u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
        unsafe {
            crate::kernels::reduce_by_key_tuple2_apply_init_kernel::launch_unchecked::<
                $ty0,
                $ty1,
                KernelOp<R, Op>,
                R,
            >(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(crate::detail::primitives::scan::BLOCK_SCAN_SIZE),
                BufferArg::from_raw_parts(inclusive.left.handle.clone(), $first_key.len),
                BufferArg::from_raw_parts(inclusive.right.handle.clone(), $first_key.len),
                BufferArg::from_raw_parts(init_a.clone(), 1),
                BufferArg::from_raw_parts(init_b.clone(), 1),
                BufferArg::from_raw_parts(len_handle.clone(), 1),
                BufferArg::from_raw_parts(reduced_a_handle.clone(), $first_key.len),
                BufferArg::from_raw_parts(reduced_b_handle.clone(), $first_key.len),
            );
        }
        let key_inner = (
            crate::detail::api::device_expr_compact_with_flags_with_policy(
                $policy,
                &$first_key,
                $end_flags.clone(),
            )?,
            crate::detail::api::device_expr_compact_with_flags_with_policy(
                $policy,
                &$second_key,
                $end_flags.clone(),
            )?,
            crate::detail::api::device_expr_compact_with_flags_with_policy(
                $policy,
                &$third_key,
                $end_flags.clone(),
            )?,
        );
        let value_a_handles = crate::detail::primitives::select::handles_from_flags(
            $policy,
            $first_key.len,
            $len_u32,
            $end_flags.clone(),
            reduced_a_handle,
        )?;
        let value_b_handles = crate::detail::primitives::select::handles_from_flags(
            $policy,
            $first_key.len,
            $len_u32,
            $end_flags,
            reduced_b_handle,
        )?;
        Ok((
            key_inner,
            (
                crate::detail::primitives::select::compact::<R, $ty0>($policy, value_a_handles)?,
                crate::detail::primitives::select::compact::<R, $ty1>($policy, value_b_handles)?,
            ),
        ))
    }};
    ($policy:ident, $input:ident, $init:ident, $first_key:ident, $second_key:ident, $third_key:ident, $head_flags:ident, $end_flags:ident, $len_u32:ident, $control:ident; $ty0:ident, $ty1:ident, $ty2:ident; 0, 1, 2) => {{
        let inclusive = crate::detail::read::inclusive_scan_by_flags_three::<
            _,
            _,
            _,
            (K1, K2, K3),
            (),
            KernelOp<R, KeyEq>,
            KernelOp<R, Op>,
        >($policy, &$input.0, &$input.1, &$input.2, &$control)?;
        let client = $policy.client();
        let len_handle = client.create_from_slice(u32::as_bytes(&[$len_u32]));
        let init_a = client.create_from_slice($ty0::as_bytes(&[$init.0]));
        let init_b = client.create_from_slice($ty1::as_bytes(&[$init.1]));
        let init_c = client.create_from_slice($ty2::as_bytes(&[$init.2]));
        let reduced_a_handle = client.empty($first_key.len * std::mem::size_of::<$ty0>());
        let reduced_b_handle = client.empty($first_key.len * std::mem::size_of::<$ty1>());
        let reduced_c_handle = client.empty($first_key.len * std::mem::size_of::<$ty2>());
        let num_blocks = $first_key
            .len
            .div_ceil(crate::detail::primitives::scan::BLOCK_SCAN_SIZE as usize);
        let num_blocks_u32 =
            u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
        unsafe {
            crate::kernels::reduce_by_key_tuple3_apply_init_kernel::launch_unchecked::<
                $ty0,
                $ty1,
                $ty2,
                KernelOp<R, Op>,
                R,
            >(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(crate::detail::primitives::scan::BLOCK_SCAN_SIZE),
                BufferArg::from_raw_parts(inclusive.first.handle.clone(), $first_key.len),
                BufferArg::from_raw_parts(inclusive.second.handle.clone(), $first_key.len),
                BufferArg::from_raw_parts(inclusive.third.handle.clone(), $first_key.len),
                BufferArg::from_raw_parts(init_a.clone(), 1),
                BufferArg::from_raw_parts(init_b.clone(), 1),
                BufferArg::from_raw_parts(init_c.clone(), 1),
                BufferArg::from_raw_parts(len_handle.clone(), 1),
                BufferArg::from_raw_parts(reduced_a_handle.clone(), $first_key.len),
                BufferArg::from_raw_parts(reduced_b_handle.clone(), $first_key.len),
                BufferArg::from_raw_parts(reduced_c_handle.clone(), $first_key.len),
            );
        }
        let key_inner = (
            crate::detail::api::device_expr_compact_with_flags_with_policy(
                $policy,
                &$first_key,
                $end_flags.clone(),
            )?,
            crate::detail::api::device_expr_compact_with_flags_with_policy(
                $policy,
                &$second_key,
                $end_flags.clone(),
            )?,
            crate::detail::api::device_expr_compact_with_flags_with_policy(
                $policy,
                &$third_key,
                $end_flags.clone(),
            )?,
        );
        let value_a_handles = crate::detail::primitives::select::handles_from_flags(
            $policy,
            $first_key.len,
            $len_u32,
            $end_flags.clone(),
            reduced_a_handle,
        )?;
        let value_b_handles = crate::detail::primitives::select::handles_from_flags(
            $policy,
            $first_key.len,
            $len_u32,
            $end_flags.clone(),
            reduced_b_handle,
        )?;
        let value_c_handles = crate::detail::primitives::select::handles_from_flags(
            $policy,
            $first_key.len,
            $len_u32,
            $end_flags,
            reduced_c_handle,
        )?;
        Ok((
            key_inner,
            (
                crate::detail::primitives::select::compact::<R, $ty0>($policy, value_a_handles)?,
                crate::detail::primitives::select::compact::<R, $ty1>($policy, value_b_handles)?,
                crate::detail::primitives::select::compact::<R, $ty2>($policy, value_c_handles)?,
            ),
        ))
    }};
}

macro_rules! impl_wide_inclusive_scan_by_three_key_values_body {
    ($policy:ident, $input:ident, $control:ident; $ty0:ident, $ty1:ident, $ty2:ident, $ty3:ident, $ty4:ident, $ty5:ident, $ty6:ident; 0, 1, 2, 3, 4, 5, 6) => {{
        crate::detail::read::inclusive_scan_by_flags_seven_views::<
            R,
            $ty0,
            $ty1,
            $ty2,
            $ty3,
            $ty4,
            $ty5,
            $ty6,
            (K1, K2, K3),
            (),
            KernelOp<R, KeyEq>,
            KernelOp<R, Op>,
        >(
            $policy, &$input.0, &$input.1, &$input.2, &$input.3, &$input.4, &$input.5, &$input.6,
            &$control,
        )
    }};
    ($policy:ident, $input:ident, $control:ident; $( $ty:ident ),+; $( $idx:tt ),+) => {{
        let _ = ($policy, $control);
        let _ = &$input;
        Err(Error::Launch {
            message: "inclusive_scan_by_key is not supported for this value iterator shape"
                .to_string(),
        })
    }};
}

macro_rules! impl_wide_exclusive_scan_by_three_key_values_body {
    ($policy:ident, $input:ident, $control:ident, $init:ident; $ty0:ident, $ty1:ident, $ty2:ident, $ty3:ident, $ty4:ident, $ty5:ident, $ty6:ident; 0, 1, 2, 3, 4, 5, 6) => {{
        crate::detail::read::exclusive_scan_by_flags_seven_views::<
            R,
            $ty0,
            $ty1,
            $ty2,
            $ty3,
            $ty4,
            $ty5,
            $ty6,
            (K1, K2, K3),
            (),
            KernelOp<R, KeyEq>,
            KernelOp<R, Op>,
        >(
            $policy, &$input.0, &$input.1, &$input.2, &$input.3, &$input.4, &$input.5, &$input.6,
            &$control, $init,
        )
    }};
    ($policy:ident, $input:ident, $control:ident, $init:ident; $( $ty:ident ),+; $( $idx:tt ),+) => {{
        let _ = ($policy, $control, $init);
        let _ = &$input;
        Err(Error::Launch {
            message: "exclusive_scan_by_key is not supported for this value iterator shape"
                .to_string(),
        })
    }};
}

macro_rules! impl_wide_reduce_by_three_key_values_body {
    ($policy:ident, $input:ident, $init:ident, $first_key:ident, $second_key:ident, $third_key:ident, $end_flags:ident, $len_u32:ident, $control:ident; $ty0:ident, $ty1:ident, $ty2:ident, $ty3:ident, $ty4:ident, $ty5:ident, $ty6:ident; 0, 1, 2, 3, 4, 5, 6) => {{
        let inclusive = crate::detail::read::inclusive_scan_by_flags_seven_views::<
            R,
            $ty0,
            $ty1,
            $ty2,
            $ty3,
            $ty4,
            $ty5,
            $ty6,
            (K1, K2, K3),
            (),
            KernelOp<R, KeyEq>,
            KernelOp<R, Op>,
        >(
            $policy, &$input.0, &$input.1, &$input.2, &$input.3, &$input.4, &$input.5, &$input.6,
            &$control,
        )?;
        let client = $policy.client();
        let len_handle = client.create_from_slice(u32::as_bytes(&[$len_u32]));
        let init_a = client.create_from_slice($ty0::as_bytes(&[$init.0]));
        let init_b = client.create_from_slice($ty1::as_bytes(&[$init.1]));
        let init_c = client.create_from_slice($ty2::as_bytes(&[$init.2]));
        let init_d = client.create_from_slice($ty3::as_bytes(&[$init.3]));
        let init_e = client.create_from_slice($ty4::as_bytes(&[$init.4]));
        let init_f = client.create_from_slice($ty5::as_bytes(&[$init.5]));
        let init_g = client.create_from_slice($ty6::as_bytes(&[$init.6]));
        let reduced_a_handle = client.empty($first_key.len * std::mem::size_of::<$ty0>());
        let reduced_b_handle = client.empty($first_key.len * std::mem::size_of::<$ty1>());
        let reduced_c_handle = client.empty($first_key.len * std::mem::size_of::<$ty2>());
        let reduced_d_handle = client.empty($first_key.len * std::mem::size_of::<$ty3>());
        let reduced_e_handle = client.empty($first_key.len * std::mem::size_of::<$ty4>());
        let reduced_f_handle = client.empty($first_key.len * std::mem::size_of::<$ty5>());
        let reduced_g_handle = client.empty($first_key.len * std::mem::size_of::<$ty6>());
        let num_blocks = $first_key
            .len
            .div_ceil(crate::detail::primitives::scan::BLOCK_SCAN_SIZE as usize);
        let num_blocks_u32 =
            u32::try_from(num_blocks).map_err(|_| Error::LengthTooLarge { len: num_blocks })?;
        unsafe {
            crate::kernels::reduce_by_key_tuple7_apply_init_kernel::launch_unchecked::<
                $ty0,
                $ty1,
                $ty2,
                $ty3,
                $ty4,
                $ty5,
                $ty6,
                KernelOp<R, Op>,
                R,
            >(
                client,
                CubeCount::Static(num_blocks_u32, 1, 1),
                CubeDim::new_1d(crate::detail::primitives::scan::BLOCK_SCAN_SIZE),
                BufferArg::from_raw_parts(inclusive.0.handle.clone(), $first_key.len),
                BufferArg::from_raw_parts(inclusive.1.handle.clone(), $first_key.len),
                BufferArg::from_raw_parts(inclusive.2.handle.clone(), $first_key.len),
                BufferArg::from_raw_parts(inclusive.3.handle.clone(), $first_key.len),
                BufferArg::from_raw_parts(inclusive.4.handle.clone(), $first_key.len),
                BufferArg::from_raw_parts(inclusive.5.handle.clone(), $first_key.len),
                BufferArg::from_raw_parts(inclusive.6.handle.clone(), $first_key.len),
                BufferArg::from_raw_parts(init_a.clone(), 1),
                BufferArg::from_raw_parts(init_b.clone(), 1),
                BufferArg::from_raw_parts(init_c.clone(), 1),
                BufferArg::from_raw_parts(init_d.clone(), 1),
                BufferArg::from_raw_parts(init_e.clone(), 1),
                BufferArg::from_raw_parts(init_f.clone(), 1),
                BufferArg::from_raw_parts(init_g.clone(), 1),
                BufferArg::from_raw_parts(len_handle.clone(), 1),
                BufferArg::from_raw_parts(reduced_a_handle.clone(), $first_key.len),
                BufferArg::from_raw_parts(reduced_b_handle.clone(), $first_key.len),
                BufferArg::from_raw_parts(reduced_c_handle.clone(), $first_key.len),
                BufferArg::from_raw_parts(reduced_d_handle.clone(), $first_key.len),
                BufferArg::from_raw_parts(reduced_e_handle.clone(), $first_key.len),
                BufferArg::from_raw_parts(reduced_f_handle.clone(), $first_key.len),
                BufferArg::from_raw_parts(reduced_g_handle.clone(), $first_key.len),
            );
        }
        let key_inner = (
            crate::detail::api::device_expr_compact_with_flags_with_policy(
                $policy,
                &$first_key,
                $end_flags.clone(),
            )?,
            crate::detail::api::device_expr_compact_with_flags_with_policy(
                $policy,
                &$second_key,
                $end_flags.clone(),
            )?,
            crate::detail::api::device_expr_compact_with_flags_with_policy(
                $policy,
                &$third_key,
                $end_flags.clone(),
            )?,
        );
        let value_a_handles = crate::detail::primitives::select::handles_from_flags(
            $policy,
            $first_key.len,
            $len_u32,
            $end_flags.clone(),
            reduced_a_handle,
        )?;
        let value_b_handles = crate::detail::primitives::select::handles_from_flags(
            $policy,
            $first_key.len,
            $len_u32,
            $end_flags.clone(),
            reduced_b_handle,
        )?;
        let value_c_handles = crate::detail::primitives::select::handles_from_flags(
            $policy,
            $first_key.len,
            $len_u32,
            $end_flags.clone(),
            reduced_c_handle,
        )?;
        let value_d_handles = crate::detail::primitives::select::handles_from_flags(
            $policy,
            $first_key.len,
            $len_u32,
            $end_flags.clone(),
            reduced_d_handle,
        )?;
        let value_e_handles = crate::detail::primitives::select::handles_from_flags(
            $policy,
            $first_key.len,
            $len_u32,
            $end_flags.clone(),
            reduced_e_handle,
        )?;
        let value_f_handles = crate::detail::primitives::select::handles_from_flags(
            $policy,
            $first_key.len,
            $len_u32,
            $end_flags.clone(),
            reduced_f_handle,
        )?;
        let value_g_handles = crate::detail::primitives::select::handles_from_flags(
            $policy,
            $first_key.len,
            $len_u32,
            $end_flags,
            reduced_g_handle,
        )?;
        Ok((
            key_inner,
            (
                crate::detail::primitives::select::compact::<R, $ty0>($policy, value_a_handles)?,
                crate::detail::primitives::select::compact::<R, $ty1>($policy, value_b_handles)?,
                crate::detail::primitives::select::compact::<R, $ty2>($policy, value_c_handles)?,
                crate::detail::primitives::select::compact::<R, $ty3>($policy, value_d_handles)?,
                crate::detail::primitives::select::compact::<R, $ty4>($policy, value_e_handles)?,
                crate::detail::primitives::select::compact::<R, $ty5>($policy, value_f_handles)?,
                crate::detail::primitives::select::compact::<R, $ty6>($policy, value_g_handles)?,
            ),
        ))
    }};
    ($policy:ident, $input:ident, $init:ident, $first_key:ident, $second_key:ident, $third_key:ident, $end_flags:ident, $len_u32:ident, $control:ident; $( $ty:ident ),+; $( $idx:tt ),+) => {{
        let _ = (
            $policy,
            $input,
            $init,
            $first_key,
            $second_key,
            $third_key,
            $end_flags,
            $len_u32,
            $control,
        );
        Err(Error::Launch {
            message: "reduce_by_key is not supported for this value iterator shape".to_string(),
        })
    }};
}

macro_rules! impl_miter_soa {
    ($name:ident; $( $ty:ident : $idx:tt : $tmp:ident ),+ => $transform:ident) => {
        impl<'a, R, $( $ty ),+> MIter<R> for $name<$( crate::runtime::DeviceSlice<'a, R, $ty> ),+>
        where
            R: Runtime,
            $( $ty: Scalar + 'static, )+
            ($( $ty, )+): MItem<
                R,
                Inner = ($( crate::detail::DeviceVec<R, $ty>, )+),
            >,
        {
            type Item = ($( $ty, )+);
            type Inner = ($( crate::detail::device::DeviceColumnView<R, $ty>, )+);

            fn len(&self) -> usize {
                self.0.len()
            }

            fn into_inner(self) -> Self::Inner {
                unreachable!("read-only MIter lowering requires a CubePolicy")
            }

            fn into_inner_with_policy(
                self,
                policy: &crate::detail::CubePolicy<R>,
            ) -> Result<Self::Inner, Error> {
                let _ = policy;
                Ok(($( self.$idx.column_view(), )+))
            }
        }

        impl<'a, R, $( $ty ),+> sealed::MIterDispatch<R> for $name<$( crate::runtime::DeviceSlice<'a, R, $ty> ),+>
        where
            R: Runtime,
            $( $ty: Scalar + 'static, )+
            ($( $ty, )+): MItem<
                R,
                Inner = ($( crate::detail::DeviceVec<R, $ty>, )+),
            >,
        {
            fn validate_executor(&self, exec: &Executor<R>) -> Result<(), Error> {
                $(
                    exec.ensure_policy_id(self.$idx.policy_id())?;
                    ensure_same_len(self.$idx.len(), self.0.len())?;
                )+
                Ok(())
            }

            fn transform_dispatch<Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                Output: MIterMut<R>,
                Op: op::UnaryOp<R, <Self as MIter<R>>::Item, Output = Output::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = <Output::Item as sealed::MItemDispatch<R>>::$transform(
                    policy,
                    $( input.$idx, )+
                    op,
                )?;
                output.write_from_inner(policy, inner)
            }

            fn map_dispatch<Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                op: Op,
            ) -> Result<Output, Error>
            where
                Output: MVec<R>,
                Op: op::UnaryOp<R, <Self as MIter<R>>::Item, Output = Output::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = <Output::Item as sealed::MItemDispatch<R>>::$transform(
                    policy,
                    $( input.$idx, )+
                    op,
                )?;
                Ok(array_from_inner::<R, Output::Item, Output>(inner))
            }

            fn transform_where_dispatch<Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                op: Op,
                stencil: crate::detail::api::PrecomputedSelection<R>,
                output: Output,
            ) -> Result<(), Error>
            where
                Output: MIterMut<R>,
                Op: op::UnaryOp<R, <Self as MIter<R>>::Item, Output = Output::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = <Output::Item as sealed::MItemDispatch<R>>::$transform(
                    policy,
                    $( input.$idx, )+
                    op,
                )?;
                output.write_where_from_inner(policy, inner, stencil)
            }

            fn reverse_dispatch<Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
            ) -> Result<Output, Error>
            where
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = crate::detail::reverse(policy, impl_miter_view!(input; $( $idx ),+))?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn sort_dispatch<Less, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _less: Less,
            ) -> Result<Output, Error>
            where
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = crate::detail::sort(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<R, Less>::new())?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn sort_by_single_key_dispatch<K, Less, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                keys: crate::detail::device::DeviceColumnView<R, K>,
                _less: Less,
            ) -> Result<(KeyOutput, ValueOutput), Error>
            where
                K: Scalar + 'static,
                Less: op::BinaryPredicateOp<R, (K,)>,
                KeyOutput: MVec<R, Item = (K,)>,
                ValueOutput: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let values = self.into_inner_with_policy(policy)?;
                let values = impl_miter_view!(values; $( $idx ),+);
                let (key_inner, value_inner) =
                    crate::detail::sort_by_key(policy, (keys,), (values,), KernelOp::<R, Less>::new())?;
                Ok((
                    array_from_inner::<R, (K,), KeyOutput>(key_inner),
                    array_from_inner::<R, <Self as MIter<R>>::Item, ValueOutput>(value_inner),
                ))
            }

            fn unique_by_single_key_dispatch<K, Eq, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                keys: crate::detail::device::DeviceColumnView<R, K>,
                _eq: Eq,
            ) -> Result<(KeyOutput, ValueOutput), Error>
            where
                K: Scalar + 'static,
                Eq: op::BinaryPredicateOp<R, (K,)>,
                KeyOutput: MVec<R, Item = (K,)>,
                ValueOutput: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let values = self.into_inner_with_policy(policy)?;
                let values = impl_miter_view!(values; $( $idx ),+);
                let (key_inner, value_inner) =
                    crate::detail::unique_by_key(policy, (keys,), (values,), KernelOp::<R, Eq>::new())?;
                Ok((
                    array_from_inner::<R, (K,), KeyOutput>(key_inner),
                    array_from_inner::<R, <Self as MIter<R>>::Item, ValueOutput>(value_inner),
                ))
            }

            fn sort_by_three_key_dispatch<K1, K2, K3, Less, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                first_key: crate::detail::device::DeviceColumnView<R, K1>,
                second_key: crate::detail::device::DeviceColumnView<R, K2>,
                third_key: crate::detail::device::DeviceColumnView<R, K3>,
                _less: Less,
            ) -> Result<(KeyOutput, ValueOutput), Error>
            where
                K1: Scalar + 'static,
                K2: Scalar + 'static,
                K3: Scalar + 'static,
                Less: op::BinaryPredicateOp<R, (K1, K2, K3)>,
                KeyOutput: MVec<R, Item = (K1, K2, K3)>,
                ValueOutput: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let values = self.into_inner_with_policy(policy)?;
                let values = impl_miter_view!(values; $( $idx ),+);
                let (key_inner, value_inner) = crate::detail::sort_by_key(
                    policy,
                    (first_key, second_key, third_key),
                    values,
                    KernelOp::<R, Less>::new(),
                )?;
                Ok((
                    array_from_inner::<R, (K1, K2, K3), KeyOutput>(key_inner),
                    array_from_inner::<R, <Self as MIter<R>>::Item, ValueOutput>(value_inner),
                ))
            }

            fn sort_by_key_dispatch<Values, Less, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                values: Values,
                less: Less,
            ) -> Result<(KeyOutput, ValueOutput), Error>
            where
                Values: MIter<R>,
                <Self as MIter<R>>::Item: cubecl::prelude::CubeType,
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
                KeyOutput: MVec<R, Item = <Self as MIter<R>>::Item>,
                ValueOutput: MVec<R, Item = <Values as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                impl_sort_by_key_dispatch_body!(policy, values, less, input; $( $idx ),+)
            }

            fn unique_by_key_dispatch<Values, Eq, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                values: Values,
                eq: Eq,
            ) -> Result<(KeyOutput, ValueOutput), Error>
            where
                Values: MIter<R>,
                <Self as MIter<R>>::Item: cubecl::prelude::CubeType,
                Eq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
                KeyOutput: MVec<R, Item = <Self as MIter<R>>::Item>,
                ValueOutput: MVec<R, Item = <Values as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                impl_unique_by_key_dispatch_body!(self, policy, values, eq, input; $( $idx ),+)
            }

            fn inclusive_scan_by_key_dispatch<Values, KeyEq, Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                values: Values,
                key_eq: KeyEq,
                op: Op,
            ) -> Result<Output, Error>
            where
                Values: MIter<R>,
                <Self as MIter<R>>::Item: cubecl::prelude::CubeType,
                KeyEq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
                Op: op::ReductionOp<R, <Values as MIter<R>>::Item>,
                Output: MVec<R, Item = <Values as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                impl_inclusive_scan_by_key_dispatch_body!(
                    policy, values, key_eq, op, input; $( $idx ),+
                )
            }

            fn exclusive_scan_by_key_dispatch<Values, KeyEq, Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                values: Values,
                key_eq: KeyEq,
                init: <Values as MIter<R>>::Item,
                op: Op,
            ) -> Result<Output, Error>
            where
                Values: MIter<R>,
                <Self as MIter<R>>::Item: cubecl::prelude::CubeType,
                KeyEq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
                Op: op::ReductionOp<R, <Values as MIter<R>>::Item>,
                Output: MVec<R, Item = <Values as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                impl_exclusive_scan_by_key_dispatch_body!(
                    policy, values, key_eq, init, op, input; $( $idx ),+
                )
            }

            fn inclusive_scan_by_three_key_dispatch<K1, K2, K3, KeyEq, Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                first_key: crate::detail::device::DeviceColumnView<R, K1>,
                second_key: crate::detail::device::DeviceColumnView<R, K2>,
                third_key: crate::detail::device::DeviceColumnView<R, K3>,
                _key_eq: KeyEq,
                _op: Op,
            ) -> Result<Output, Error>
            where
                K1: Scalar + 'static,
                K2: Scalar + 'static,
                K3: Scalar + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K1, K2, K3)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                ensure_same_len(input.0.len, first_key.len)?;
                let head_flags = crate::detail::read::unique_tuple3_flags_read::<
                    crate::detail::device::DeviceColumnView<R, K1>,
                    crate::detail::device::DeviceColumnView<R, K2>,
                    crate::detail::device::DeviceColumnView<R, K3>,
                    KernelOp<R, KeyEq>,
                >(
                    policy,
                    &first_key,
                    &second_key,
                    &third_key,
                )?;
                let len_u32 = u32::try_from(first_key.len)
                    .map_err(|_| Error::LengthTooLarge { len: first_key.len })?;
                let control: crate::detail::control::ScanByKeyControl<
                    R,
                    (K1, K2, K3),
                    (),
                    KernelOp<R, KeyEq>,
                > = crate::detail::control::ScanByKeyControl {
                    key_bindings: crate::detail::device::KernelColumnBindings::empty(policy.client()),
                    head_flags,
                    len: first_key.len,
                    len_u32,
                    _marker: std::marker::PhantomData,
                };
                let inner = impl_tuple_inclusive_scan_by_three_key_values_body!(
                    policy, input, control; $( $idx ),+
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn exclusive_scan_by_three_key_dispatch<K1, K2, K3, KeyEq, Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                first_key: crate::detail::device::DeviceColumnView<R, K1>,
                second_key: crate::detail::device::DeviceColumnView<R, K2>,
                third_key: crate::detail::device::DeviceColumnView<R, K3>,
                _key_eq: KeyEq,
                init: <Self as MIter<R>>::Item,
                _op: Op,
            ) -> Result<Output, Error>
            where
                K1: Scalar + 'static,
                K2: Scalar + 'static,
                K3: Scalar + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K1, K2, K3)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                ensure_same_len(input.0.len, first_key.len)?;
                let head_flags = crate::detail::read::unique_tuple3_flags_read::<
                    crate::detail::device::DeviceColumnView<R, K1>,
                    crate::detail::device::DeviceColumnView<R, K2>,
                    crate::detail::device::DeviceColumnView<R, K3>,
                    KernelOp<R, KeyEq>,
                >(
                    policy,
                    &first_key,
                    &second_key,
                    &third_key,
                )?;
                let len_u32 = u32::try_from(first_key.len)
                    .map_err(|_| Error::LengthTooLarge { len: first_key.len })?;
                let control: crate::detail::control::ScanByKeyControl<
                    R,
                    (K1, K2, K3),
                    (),
                    KernelOp<R, KeyEq>,
                > = crate::detail::control::ScanByKeyControl {
                    key_bindings: crate::detail::device::KernelColumnBindings::empty(policy.client()),
                    head_flags,
                    len: first_key.len,
                    len_u32,
                    _marker: std::marker::PhantomData,
                };
                let inner = impl_tuple_exclusive_scan_by_three_key_values_body!(
                    policy, input, control, init; $( $idx ),+
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn inclusive_scan_by_single_key_dispatch<K, KeyEq, Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                keys: crate::detail::device::DeviceColumnView<R, K>,
                _key_eq: KeyEq,
                _op: Op,
            ) -> Result<Output, Error>
            where
                K: Scalar + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K,)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let values = self.into_inner_with_policy(policy)?;
                let values = impl_miter_view!(values; $( $idx ),+);
                let inner = crate::detail::inclusive_scan_by_key(
                    policy,
                    keys,
                    values,
                    KernelTuple1Op::<R, KeyEq>::new(),
                    KernelOp::<R, Op>::new(),
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn exclusive_scan_by_single_key_dispatch<K, KeyEq, Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                keys: crate::detail::device::DeviceColumnView<R, K>,
                _key_eq: KeyEq,
                init: <Self as MIter<R>>::Item,
                _op: Op,
            ) -> Result<Output, Error>
            where
                K: Scalar + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K,)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let values = self.into_inner_with_policy(policy)?;
                let values = impl_miter_view!(values; $( $idx ),+);
                let inner = crate::detail::exclusive_scan_by_key(
                    policy,
                    keys,
                    values,
                    KernelTuple1Op::<R, KeyEq>::new(),
                    init,
                    KernelOp::<R, Op>::new(),
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn reduce_by_single_key_dispatch<K, KeyEq, Op, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                keys: crate::detail::device::DeviceColumnView<R, K>,
                _key_eq: KeyEq,
                init: <Self as MIter<R>>::Item,
                _op: Op,
            ) -> Result<(KeyOutput, ValueOutput), Error>
            where
                K: Scalar + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K,)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                KeyOutput: MVec<R, Item = (K,)>,
                ValueOutput: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let values = self.into_inner_with_policy(policy)?;
                let (key_inner, value_inner) = crate::detail::reduce_by_key(
                    policy,
                    (keys,),
                    values,
                    KernelOp::<R, KeyEq>::new(),
                    init,
                    KernelOp::<R, Op>::new(),
                )?;
                Ok((
                    array_from_inner::<R, (K,), KeyOutput>(key_inner),
                    array_from_inner::<R, <Self as MIter<R>>::Item, ValueOutput>(value_inner),
                ))
            }

            fn reduce_by_three_key_dispatch<K1, K2, K3, KeyEq, Op, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                first_key: crate::detail::device::DeviceColumnView<R, K1>,
                second_key: crate::detail::device::DeviceColumnView<R, K2>,
                third_key: crate::detail::device::DeviceColumnView<R, K3>,
                _key_eq: KeyEq,
                init: <Self as MIter<R>>::Item,
                _op: Op,
            ) -> Result<(KeyOutput, ValueOutput), Error>
            where
                K1: Scalar + 'static,
                K2: Scalar + 'static,
                K3: Scalar + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K1, K2, K3)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                KeyOutput: MVec<R, Item = (K1, K2, K3)>,
                ValueOutput: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                ensure_same_len(input.0.len, first_key.len)?;
                if first_key.len == 0 {
                    let key_inner = (
                        policy.empty_device_vec(),
                        policy.empty_device_vec(),
                        policy.empty_device_vec(),
                    );
                    let value_inner = ($( {
                        let _ = stringify!($ty);
                        policy.empty_device_vec()
                    }, )+);
                    return Ok((
                        array_from_inner::<R, (K1, K2, K3), KeyOutput>(key_inner),
                        array_from_inner::<R, <Self as MIter<R>>::Item, ValueOutput>(value_inner),
                    ));
                }
                let head_flags = crate::detail::read::unique_tuple3_flags_read::<
                    crate::detail::device::DeviceColumnView<R, K1>,
                    crate::detail::device::DeviceColumnView<R, K2>,
                    crate::detail::device::DeviceColumnView<R, K3>,
                    KernelOp<R, KeyEq>,
                >(
                    policy,
                    &first_key,
                    &second_key,
                    &third_key,
                )?;
                let end_flags =
                    crate::detail::impls::end_flags_from_head_flags(policy, head_flags.clone(), first_key.len)?;
                let len_u32 = u32::try_from(first_key.len)
                    .map_err(|_| Error::LengthTooLarge { len: first_key.len })?;
                let control: crate::detail::control::ScanByKeyControl<
                    R,
                    (K1, K2, K3),
                    (),
                    KernelOp<R, KeyEq>,
                > = crate::detail::control::ScanByKeyControl {
                    key_bindings: crate::detail::device::KernelColumnBindings::empty(policy.client()),
                    head_flags,
                    len: first_key.len,
                    len_u32,
                    _marker: std::marker::PhantomData,
                };
                let (key_inner, value_inner) = impl_tuple_reduce_by_three_key_values_body!(
                    policy, input, init, first_key, second_key, third_key, head_flags, end_flags, len_u32, control; $( $ty ),+; $( $idx ),+
                )?;
                Ok((
                    array_from_inner::<R, (K1, K2, K3), KeyOutput>(key_inner),
                    array_from_inner::<R, <Self as MIter<R>>::Item, ValueOutput>(value_inner),
                ))
            }

            fn reduce_by_key_dispatch<Values, KeyEq, Op, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                values: Values,
                key_eq: KeyEq,
                init: <Values as MIter<R>>::Item,
                op: Op,
            ) -> Result<(KeyOutput, ValueOutput), Error>
            where
                Values: MIter<R>,
                <Self as MIter<R>>::Item: cubecl::prelude::CubeType,
                KeyEq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
                Op: op::ReductionOp<R, <Values as MIter<R>>::Item>,
                KeyOutput: MVec<R, Item = <Self as MIter<R>>::Item>,
                ValueOutput: MVec<R, Item = <Values as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                impl_reduce_by_key_dispatch_body!(
                    policy, values, key_eq, init, op, input; $( $idx ),+
                )
            }

            fn merge_by_single_key_same_dispatch<K, RightValues, Less, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                left_keys: crate::detail::device::DeviceColumnView<R, K>,
                right_keys: crate::detail::device::DeviceColumnView<R, K>,
                right_values: RightValues,
                _less: Less,
            ) -> Result<(KeyOutput, ValueOutput), Error>
            where
                RightValues: MIter<
                    R,
                    Item = <Self as MIter<R>>::Item,
                    Inner = <Self as MIter<R>>::Inner,
                >,
                K: Scalar + 'static,
                Less: op::BinaryPredicateOp<R, (K,)>,
                KeyOutput: MVec<R, Item = (K,)>,
                ValueOutput: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let left_values = self.into_inner_with_policy(policy)?;
                let right_values = right_values.into_inner_with_policy(policy)?;
                let (key_inner, value_inner) = crate::detail::merge_by_key(
                    policy,
                    crate::detail::device::SoAView1 { source: left_keys },
                    impl_miter_view!(left_values; $( $idx ),+),
                    crate::detail::device::SoAView1 { source: right_keys },
                    impl_miter_view!(right_values; $( $idx ),+),
                    KernelTuple1Op::<R, Less>::new(),
                )?;
                Ok((
                    array_from_inner::<R, (K,), KeyOutput>(key_inner),
                    array_from_inner::<R, <Self as MIter<R>>::Item, ValueOutput>(value_inner),
                ))
            }

            fn merge_by_three_key_same_dispatch<K1, K2, K3, RightValues, Less, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                left_first_key: crate::detail::device::DeviceColumnView<R, K1>,
                left_second_key: crate::detail::device::DeviceColumnView<R, K2>,
                left_third_key: crate::detail::device::DeviceColumnView<R, K3>,
                right_first_key: crate::detail::device::DeviceColumnView<R, K1>,
                right_second_key: crate::detail::device::DeviceColumnView<R, K2>,
                right_third_key: crate::detail::device::DeviceColumnView<R, K3>,
                right_values: RightValues,
                _less: Less,
            ) -> Result<(KeyOutput, ValueOutput), Error>
            where
                RightValues: MIter<
                    R,
                    Item = <Self as MIter<R>>::Item,
                    Inner = <Self as MIter<R>>::Inner,
                >,
                K1: Scalar + 'static,
                K2: Scalar + 'static,
                K3: Scalar + 'static,
                Less: op::BinaryPredicateOp<R, (K1, K2, K3)>,
                KeyOutput: MVec<R, Item = (K1, K2, K3)>,
                ValueOutput: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let left_values = self.into_inner_with_policy(policy)?;
                let right_values = right_values.into_inner_with_policy(policy)?;
                let (key_inner, value_inner) = crate::detail::merge_by_key(
                    policy,
                    (left_first_key, left_second_key, left_third_key),
                    impl_miter_view!(left_values; $( $idx ),+),
                    (right_first_key, right_second_key, right_third_key),
                    impl_miter_view!(right_values; $( $idx ),+),
                    KernelOp::<R, Less>::new(),
                )?;
                Ok((
                    array_from_inner::<R, (K1, K2, K3), KeyOutput>(key_inner),
                    array_from_inner::<R, <Self as MIter<R>>::Item, ValueOutput>(value_inner),
                ))
            }

            fn merge_by_key_dispatch<RightKeys, LeftValues, RightValues, Less, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right_keys: RightKeys,
                left_values: LeftValues,
                right_values: RightValues,
                less: Less,
            ) -> Result<(KeyOutput, ValueOutput), Error>
            where
                RightKeys: MIter<R, Item = <Self as MIter<R>>::Item, Inner = <Self as MIter<R>>::Inner>,
                LeftValues: MIter<R>,
                RightValues: MIter<R, Item = <LeftValues as MIter<R>>::Item, Inner = <LeftValues as MIter<R>>::Inner>,
                <Self as MIter<R>>::Item: cubecl::prelude::CubeType,
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
                KeyOutput: MVec<R, Item = <Self as MIter<R>>::Item>,
                ValueOutput: MVec<R, Item = <LeftValues as MIter<R>>::Item>,
            {
                let left_input = self.into_inner_with_policy(policy)?;
                impl_merge_by_key_dispatch_body!(
                    policy, right_keys, left_values, right_values, less, left_input; $( $idx ),+
                )
            }

            fn gather_dispatch<Indices, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                indices: Indices,
                output: Output,
            ) -> Result<(), Error>
            where
                Indices: crate::detail::device::KernelColumn<Runtime = R, Item = u32>
                    + crate::detail::device::KernelColumnAt<crate::detail::device::S0>,
                <Indices as crate::detail::device::KernelColumn>::Expr: crate::expr::GpuExpr<u32>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                $(
                    let $tmp = <Output as sealed::MIterMutDispatch<R>>::column_mut_view_by_index_inner::<$ty>(
                        &output,
                        $idx,
                    )?
                    .ok_or_else(|| Error::Launch {
                        message: "gather output must match input shape".to_string(),
                    })?;
                    crate::detail::api::device_expr_gather_into_with_policy(
                        policy,
                        &input.$idx,
                        &indices,
                        &$tmp,
                    )?;
                )+
                Ok(())
            }

            fn permute_dispatch<Indices, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                indices: Indices,
            ) -> Result<Output, Error>
            where
                Indices: crate::detail::device::KernelColumn<Runtime = R, Item = u32>
                    + crate::detail::device::KernelColumnAt<crate::detail::device::S0>,
                <Indices as crate::detail::device::KernelColumn>::Expr:
                    crate::expr::GpuExpr<u32> + crate::expr::DeviceGpuExpr<u32>,
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                $(
                    let $tmp = crate::detail::api::device_expr_gather_with_policy(
                        policy,
                        &input.$idx,
                        &indices,
                    )?;
                )+
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(($($tmp,)+)))
            }

            fn reduce_dispatch<Op>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                init: <Self as MIter<R>>::Item,
                _op: Op,
            ) -> Result<<Self as MIter<R>>::Item, Error>
            where
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                crate::detail::reduce(policy, impl_miter_view!(input; $( $idx ),+), init, KernelOp::<R, Op>::new())
            }

            fn inclusive_scan_dispatch<Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _op: Op,
            ) -> Result<Output, Error>
            where
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = crate::detail::inclusive_scan(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    KernelOp::<R, Op>::new(),
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn exclusive_scan_dispatch<Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                init: <Self as MIter<R>>::Item,
                _op: Op,
            ) -> Result<Output, Error>
            where
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = crate::detail::exclusive_scan(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    init,
                    KernelOp::<R, Op>::new(),
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn adjacent_difference_dispatch<Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _op: Op,
            ) -> Result<Output, Error>
            where
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = crate::detail::adjacent_difference(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    KernelOp::<R, Op>::new(),
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn copy_where_dispatch<Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                stencil: crate::detail::api::PrecomputedSelection<R>,
            ) -> Result<Output, Error>
            where
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = crate::detail::copy_where(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    stencil,
                    KernelOp::<R, StencilFlag>::new(),
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn remove_if_dispatch<Pred, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _pred: Pred,
            ) -> Result<Output, Error>
            where
                Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = crate::detail::remove_if(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    KernelOp::<R, Pred>::new(),
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn remove_where_dispatch<Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                stencil: crate::detail::api::PrecomputedSelection<R>,
            ) -> Result<Output, Error>
            where
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = crate::detail::copy_where(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    stencil,
                    KernelOp::<R, StencilFlag>::new(),
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn count_if_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _pred: Pred,
            ) -> Result<usize, Error>
            where
                Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                crate::detail::count_if(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<R, Pred>::new())
            }

            fn all_of_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _pred: Pred,
            ) -> Result<bool, Error>
            where
                Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                crate::detail::all_of(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<R, Pred>::new())
            }

            fn any_of_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _pred: Pred,
            ) -> Result<bool, Error>
            where
                Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                crate::detail::any_of(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<R, Pred>::new())
            }

            fn none_of_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _pred: Pred,
            ) -> Result<bool, Error>
            where
                Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                crate::detail::none_of(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<R, Pred>::new())
            }

            fn find_if_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _pred: Pred,
            ) -> Result<Option<usize>, Error>
            where
                Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                crate::detail::find_if(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<R, Pred>::new())
            }

            fn partition_dispatch<Pred, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _pred: Pred,
            ) -> Result<(Output, Output), Error>
            where
                Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let (matching, failing) = crate::detail::partition(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    KernelOp::<R, Pred>::new(),
                )?;
                Ok((
                    array_from_inner::<R, <Self as MIter<R>>::Item, Output>(matching),
                    array_from_inner::<R, <Self as MIter<R>>::Item, Output>(failing),
                ))
            }

            fn is_partitioned_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _pred: Pred,
            ) -> Result<bool, Error>
            where
                Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                crate::detail::is_partitioned(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<R, Pred>::new())
            }

            fn replace_where_dispatch<Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                replacement: <Self as MIter<R>>::Item,
                stencil: crate::detail::api::PrecomputedSelection<R>,
            ) -> Result<Output, Error>
            where
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = crate::detail::replace_where(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    replacement,
                    stencil,
                    KernelOp::<R, StencilFlag>::new(),
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn unique_dispatch<Pred, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _pred: Pred,
            ) -> Result<Output, Error>
            where
                Pred: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = crate::detail::unique(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    KernelOp::<R, Pred>::new(),
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn min_element_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _less: Less,
            ) -> Result<Option<usize>, Error>
            where
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                crate::detail::min_element(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<R, Less>::new())
            }

            fn max_element_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _less: Less,
            ) -> Result<Option<usize>, Error>
            where
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                crate::detail::max_element(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<R, Less>::new())
            }

            fn minmax_element_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _less: Less,
            ) -> Result<Option<(usize, usize)>, Error>
            where
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                crate::detail::minmax_element(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<R, Less>::new())
            }

            fn adjacent_find_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _pred: Pred,
            ) -> Result<Option<usize>, Error>
            where
                Pred: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                crate::detail::adjacent_find(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<R, Pred>::new())
            }

            fn lower_bound_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                value: <Self as MIter<R>>::Item,
                _less: Less,
            ) -> Result<usize, Error>
            where
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                crate::detail::lower_bound(policy, impl_miter_view!(input; $( $idx ),+), value, KernelOp::<R, Less>::new())
            }

            fn upper_bound_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                value: <Self as MIter<R>>::Item,
                _less: Less,
            ) -> Result<usize, Error>
            where
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                crate::detail::upper_bound(policy, impl_miter_view!(input; $( $idx ),+), value, KernelOp::<R, Less>::new())
            }

            fn equal_range_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                value: <Self as MIter<R>>::Item,
                _less: Less,
            ) -> Result<(usize, usize), Error>
            where
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                crate::detail::equal_range(policy, impl_miter_view!(input; $( $idx ),+), value, KernelOp::<R, Less>::new())
            }

            fn is_sorted_until_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _less: Less,
            ) -> Result<usize, Error>
            where
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                crate::detail::is_sorted_until(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<R, Less>::new())
            }

            fn is_sorted_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _less: Less,
            ) -> Result<bool, Error>
            where
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                crate::detail::is_sorted(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<R, Less>::new())
            }

            fn gather_where_dispatch<Indices, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                indices: Indices,
                stencil: crate::detail::api::PrecomputedSelection<R>,
                output: Output,
            ) -> Result<(), Error>
            where
                Indices: crate::detail::device::KernelColumn<Runtime = R, Item = u32>
                    + crate::detail::device::KernelColumnAt<crate::detail::device::S0>,
                <Indices as crate::detail::device::KernelColumn>::Expr:
                    crate::expr::GpuExpr<u32> + crate::expr::DeviceGpuExpr<u32>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                $(
                    let $tmp = <Output as sealed::MIterMutDispatch<R>>::column_mut_view_by_index_inner::<$ty>(
                        &output,
                        $idx,
                    )?
                    .ok_or_else(|| Error::Launch {
                        message: "gather_where output must match input shape".to_string(),
                    })?;
                    crate::detail::api::device_expr_gather_where_into_with_control(
                        policy,
                        &input.$idx,
                        &indices,
                        stencil.control(),
                        &$tmp,
                    )?;
                )+
                Ok(())
            }

            fn scatter_dispatch<Indices, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                indices: Indices,
                output: Output,
            ) -> Result<(), Error>
            where
                Indices: crate::detail::device::KernelColumn<Runtime = R, Item = u32>
                    + crate::detail::device::KernelColumnAt<crate::detail::device::S0>,
                <Indices as crate::detail::device::KernelColumn>::Expr: crate::expr::GpuExpr<u32>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                $(
                    let $tmp = <Output as sealed::MIterMutDispatch<R>>::column_mut_view_by_index_inner::<$ty>(
                        &output,
                        $idx,
                    )?
                    .ok_or_else(|| Error::Launch {
                        message: "scatter output must match input shape".to_string(),
                    })?;
                    crate::detail::api::device_expr_scatter_into_with_policy(
                        policy,
                        &input.$idx,
                        &indices,
                        &$tmp,
                    )?;
                )+
                Ok(())
            }

            fn scatter_where_dispatch<Indices, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                indices: Indices,
                stencil: crate::detail::api::PrecomputedSelection<R>,
                output: Output,
            ) -> Result<(), Error>
            where
                Indices: crate::detail::device::KernelColumn<Runtime = R, Item = u32>
                    + crate::detail::device::KernelColumnAt<crate::detail::device::S0>,
                <Indices as crate::detail::device::KernelColumn>::Expr:
                    crate::expr::GpuExpr<u32> + crate::expr::DeviceGpuExpr<u32>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                $(
                    let $tmp = <Output as sealed::MIterMutDispatch<R>>::column_mut_view_by_index_inner::<$ty>(
                        &output,
                        $idx,
                    )?
                    .ok_or_else(|| Error::Launch {
                        message: "scatter_where output must match input shape".to_string(),
                    })?;
                    crate::detail::api::device_expr_scatter_where_into_with_control(
                        policy,
                        &input.$idx,
                        &indices,
                        stencil.control(),
                        &$tmp,
                    )?;
                )+
                Ok(())
            }

            fn equal_dispatch<Right, Eq>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right: Right,
                _eq: Eq,
            ) -> Result<bool, Error>
            where
                Right: MIter<
                    R,
                    Item = <Self as MIter<R>>::Item,
                    Inner = <Self as MIter<R>>::Inner,
                >,
                Eq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_inner_with_policy(policy)?;
                let right = right.into_inner_with_policy(policy)?;
                crate::detail::equal(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<R, Eq>::new(),
                )
            }

            fn mismatch_dispatch<Right, Eq>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right: Right,
                _eq: Eq,
            ) -> Result<Option<usize>, Error>
            where
                Right: MIter<
                    R,
                    Item = <Self as MIter<R>>::Item,
                    Inner = <Self as MIter<R>>::Inner,
                >,
                Eq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_inner_with_policy(policy)?;
                let right = right.into_inner_with_policy(policy)?;
                crate::detail::mismatch(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<R, Eq>::new(),
                )
            }

            fn find_first_of_dispatch<Needles, Eq>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                needles: Needles,
                _eq: Eq,
            ) -> Result<Option<usize>, Error>
            where
                Needles: MIter<
                    R,
                    Item = <Self as MIter<R>>::Item,
                    Inner = <Self as MIter<R>>::Inner,
                >,
                Eq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let needles = needles.into_inner_with_policy(policy)?;
                crate::detail::find_first_of(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    impl_miter_view!(needles; $( $idx ),+),
                    KernelOp::<R, Eq>::new(),
                )
            }

            fn lexicographical_compare_dispatch<Right, Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right: Right,
                _less: Less,
            ) -> Result<bool, Error>
            where
                Right: MIter<
                    R,
                    Item = <Self as MIter<R>>::Item,
                    Inner = <Self as MIter<R>>::Inner,
                >,
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_inner_with_policy(policy)?;
                let right = right.into_inner_with_policy(policy)?;
                crate::detail::lexicographical_compare(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<R, Less>::new(),
                )
            }

            fn merge_dispatch<Right, Output, Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right: Right,
                _less: Less,
            ) -> Result<Output, Error>
            where
                Right: MIter<
                    R,
                    Item = <Self as MIter<R>>::Item,
                    Inner = <Self as MIter<R>>::Inner,
                >,
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_inner_with_policy(policy)?;
                let right = right.into_inner_with_policy(policy)?;
                let inner = crate::detail::merge(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<R, Less>::new(),
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn set_union_dispatch<Right, Output, Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right: Right,
                _less: Less,
            ) -> Result<Output, Error>
            where
                Right: MIter<
                    R,
                    Item = <Self as MIter<R>>::Item,
                    Inner = <Self as MIter<R>>::Inner,
                >,
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_inner_with_policy(policy)?;
                let right = right.into_inner_with_policy(policy)?;
                let inner = crate::detail::set_union(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<R, Less>::new(),
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn set_intersection_dispatch<Right, Output, Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right: Right,
                _less: Less,
            ) -> Result<Output, Error>
            where
                Right: MIter<
                    R,
                    Item = <Self as MIter<R>>::Item,
                    Inner = <Self as MIter<R>>::Inner,
                >,
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_inner_with_policy(policy)?;
                let right = right.into_inner_with_policy(policy)?;
                let inner = crate::detail::set_intersection(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<R, Less>::new(),
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn set_difference_dispatch<Right, Output, Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right: Right,
                _less: Less,
            ) -> Result<Output, Error>
            where
                Right: MIter<
                    R,
                    Item = <Self as MIter<R>>::Item,
                    Inner = <Self as MIter<R>>::Inner,
                >,
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_inner_with_policy(policy)?;
                let right = right.into_inner_with_policy(policy)?;
                let inner = crate::detail::set_difference(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<R, Less>::new(),
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn equal_same_dispatch<Eq>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right: Self,
                _eq: Eq,
            ) -> Result<bool, Error>
            where
                Eq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_inner_with_policy(policy)?;
                let right = right.into_inner_with_policy(policy)?;
                crate::detail::equal(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<R, Eq>::new(),
                )
            }

            fn mismatch_same_dispatch<Eq>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right: Self,
                _eq: Eq,
            ) -> Result<Option<usize>, Error>
            where
                Eq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_inner_with_policy(policy)?;
                let right = right.into_inner_with_policy(policy)?;
                crate::detail::mismatch(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<R, Eq>::new(),
                )
            }

            fn find_first_of_same_dispatch<Eq>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                needles: Self,
                _eq: Eq,
            ) -> Result<Option<usize>, Error>
            where
                Eq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let needles = needles.into_inner_with_policy(policy)?;
                crate::detail::find_first_of(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    impl_miter_view!(needles; $( $idx ),+),
                    KernelOp::<R, Eq>::new(),
                )
            }

            fn lexicographical_compare_same_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right: Self,
                _less: Less,
            ) -> Result<bool, Error>
            where
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_inner_with_policy(policy)?;
                let right = right.into_inner_with_policy(policy)?;
                crate::detail::lexicographical_compare(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<R, Less>::new(),
                )
            }

            fn merge_same_dispatch<Output, Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right: Self,
                _less: Less,
            ) -> Result<Output, Error>
            where
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_inner_with_policy(policy)?;
                let right = right.into_inner_with_policy(policy)?;
                let inner = crate::detail::merge(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<R, Less>::new(),
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn set_union_same_dispatch<Output, Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right: Self,
                _less: Less,
            ) -> Result<Output, Error>
            where
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_inner_with_policy(policy)?;
                let right = right.into_inner_with_policy(policy)?;
                let inner = crate::detail::set_union(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<R, Less>::new(),
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn set_intersection_same_dispatch<Output, Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right: Self,
                _less: Less,
            ) -> Result<Output, Error>
            where
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_inner_with_policy(policy)?;
                let right = right.into_inner_with_policy(policy)?;
                let inner = crate::detail::set_intersection(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<R, Less>::new(),
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn set_difference_same_dispatch<Output, Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right: Self,
                _less: Less,
            ) -> Result<Output, Error>
            where
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_inner_with_policy(policy)?;
                let right = right.into_inner_with_policy(policy)?;
                let inner = crate::detail::set_difference(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<R, Less>::new(),
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

        }
    };
}

macro_rules! impl_miter_mut_soa {
    ($name:ident; $( $ty:ident : $idx:tt ),+) => {
        impl<'a, R, $( $ty ),+> MIterMut<R> for $name<$( DeviceSliceMut<'a, R, $ty> ),+>
        where
            R: Runtime,
            $( $ty: Scalar + 'static, )+
            ($( $ty, )+): MItem<
                R,
                Inner = ($( crate::detail::DeviceVec<R, $ty>, )+),
            >,
        {
            type Item = ($( $ty, )+);
            type Inner = ($( crate::detail::device::DeviceColumnMutView<R, $ty>, )+);

            fn len(&self) -> usize {
                self.0.len()
            }

            fn into_inner(self) -> Self::Inner {
                ($(
                    crate::detail::device::DeviceColumnMutView::from_slice(
                        &self.$idx.source.inner,
                        self.$idx.offset,
                        self.$idx.len,
                    ),
                )+)
            }

            fn write_from_inner(
                self,
                policy: &crate::detail::CubePolicy<R>,
                inner: <Self::Item as MItem<R>>::Inner,
            ) -> Result<(), Error> {
                let output = self.into_inner();
                $(
                    {
                        let input =
                            crate::detail::device::DeviceColumnView::from_column(&inner.$idx);
                        crate::detail::api::device_expr_collect_into_with_policy(
                            policy,
                            &input,
                            &output.$idx,
                        )?;
                    }
                )+
                Ok(())
            }

            fn write_where_from_inner(
                self,
                policy: &crate::detail::CubePolicy<R>,
                inner: <Self::Item as MItem<R>>::Inner,
                stencil: crate::detail::api::PrecomputedSelection<R>,
            ) -> Result<(), Error>
            {
                let output = self.into_inner();
                $(
                    {
                        let input =
                            crate::detail::device::DeviceColumnView::from_column(&inner.$idx);
                        crate::detail::api::device_expr_copy_where_into_with_policy(
                            policy,
                            &input,
                            &stencil,
                            &output.$idx,
                            KernelOp::<R, StencilFlag>::new(),
                        )?;
                    }
                )+
                Ok(())
            }

            fn replace_where_inner(
                self,
                policy: &crate::detail::CubePolicy<R>,
                replacement: Self::Item,
                stencil: crate::detail::api::PrecomputedSelection<R>,
            ) -> Result<(), Error>
            {
                let output = self.into_inner();
                $(
                    crate::detail::api::replace_where_into_with_control(
                        policy,
                        replacement.$idx,
                        stencil.control(),
                        &output.$idx,
                    )?;
                )+
                Ok(())
            }
        }

        impl<'a, R, $( $ty ),+> sealed::MIterMutDispatch<R> for $name<$( DeviceSliceMut<'a, R, $ty> ),+>
        where
            R: Runtime,
            $( $ty: Scalar + 'static, )+
            ($( $ty, )+): MItem<
                R,
                Inner = ($( crate::detail::DeviceVec<R, $ty>, )+),
            >,
        {
            fn validate_executor(&self, exec: &Executor<R>) -> Result<(), Error> {
                $(
                    exec.ensure_policy_id(self.$idx.source.inner.policy_id())?;
                )+
                $(
                    ensure_same_len(self.$idx.len(), self.0.len())?;
                )+
                Ok(())
            }

            fn column_mut_view_by_index_inner<U: 'static>(
                &self,
                index: usize,
            ) -> Result<
                Option<crate::detail::device::DeviceColumnMutView<R, U>>,
                Error,
            >
            where
                U: Scalar,
            {
                $(
                    if index == $idx {
                        let source = &*self.$idx.source as &dyn Any;
                        let source = match source.downcast_ref::<DeviceVec<R, U>>() {
                            Some(source) => source,
                            None => return Ok(None),
                        };
                        return Ok(Some(crate::detail::device::DeviceColumnMutView::from_slice(
                            &source.inner,
                            self.$idx.offset,
                            self.$idx.len,
                        )));
                    }
                )+
                Ok(None)
            }

        }
    };
}

macro_rules! impl_wide_miter_soa {
    ($name:ident; $( $ty:ident : $idx:tt : $tmp:ident ),+) => {
        impl<'a, R, $( $ty ),+> MIter<R> for $name<$( crate::runtime::DeviceSlice<'a, R, $ty> ),+>
        where
            R: Runtime,
            $( $ty: Scalar + 'static, )+
            ($( $ty, )+): MItem<
                R,
                Inner = ($( crate::detail::DeviceVec<R, $ty>, )+),
            >,
        {
            type Item = ($( $ty, )+);
            type Inner = ($( crate::detail::device::DeviceColumnView<R, $ty>, )+);

            fn len(&self) -> usize {
                self.0.len()
            }

            fn into_inner(self) -> Self::Inner {
                unreachable!("read-only MIter lowering requires a CubePolicy")
            }

            fn into_inner_with_policy(
                self,
                policy: &crate::detail::CubePolicy<R>,
            ) -> Result<Self::Inner, Error> {
                let _ = policy;
                Ok(($( self.$idx.column_view(), )+))
            }
        }

        impl<'a, R, $( $ty ),+> sealed::MIterDispatch<R> for $name<$( crate::runtime::DeviceSlice<'a, R, $ty> ),+>
        where
            R: Runtime,
            $( $ty: Scalar + 'static, )+
            ($( $ty, )+): MItem<
                R,
                Inner = ($( crate::detail::DeviceVec<R, $ty>, )+),
            >,
        {
            fn validate_executor(&self, exec: &Executor<R>) -> Result<(), Error> {
                $(
                    exec.ensure_policy_id(self.$idx.policy_id())?;
                    ensure_same_len(self.$idx.len(), self.0.len())?;
                )+
                Ok(())
            }

            fn transform_dispatch<Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                Output: MIterMut<R>,
                Op: op::UnaryOp<R, <Self as MIter<R>>::Item, Output = Output::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = impl_wide_transform_dispatch_body!(policy, input, op; $( $idx ),+)?;
                output.write_from_inner(policy, inner)
            }

            fn map_dispatch<Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                op: Op,
            ) -> Result<Output, Error>
            where
                Output: MVec<R>,
                Op: op::UnaryOp<R, <Self as MIter<R>>::Item, Output = Output::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = impl_wide_transform_dispatch_body!(
                    policy,
                    input,
                    op;
                    $( $idx ),+
                )?;
                Ok(array_from_inner::<R, Output::Item, Output>(inner))
            }

            fn transform_where_dispatch<Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                op: Op,
                stencil: crate::detail::api::PrecomputedSelection<R>,
                output: Output,
            ) -> Result<(), Error>
            where
                Output: MIterMut<R>,
                Op: op::UnaryOp<R, <Self as MIter<R>>::Item, Output = Output::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = impl_wide_transform_dispatch_body!(policy, input, op; $( $idx ),+)?;
                output.write_where_from_inner(policy, inner, stencil)
            }

            fn reverse_dispatch<Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
            ) -> Result<Output, Error>
            where
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                $(
                    let ($tmp,) = crate::detail::reverse(policy, (input.$idx,))?;
                )+
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(($($tmp,)+)))
            }

            fn sort_dispatch<Less, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _less: Less,
            ) -> Result<Output, Error>
            where
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = impl_wide_sort_dispatch_body!(policy, input; $( $idx ),+)?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn sort_by_single_key_dispatch<K, Less, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                keys: crate::detail::device::DeviceColumnView<R, K>,
                _less: Less,
            ) -> Result<(KeyOutput, ValueOutput), Error>
            where
                K: Scalar + 'static,
                Less: op::BinaryPredicateOp<R, (K,)>,
                KeyOutput: MVec<R, Item = (K,)>,
                ValueOutput: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let (key_inner, value_inner) = impl_wide_sort_by_single_key_dispatch_body!(
                    policy, keys, _less, input; $( $idx ),+
                )?;
                Ok((
                    array_from_inner::<R, (K,), KeyOutput>(key_inner),
                    array_from_inner::<R, <Self as MIter<R>>::Item, ValueOutput>(value_inner),
                ))
            }

            fn sort_by_three_key_dispatch<K1, K2, K3, Less, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                first_key: crate::detail::device::DeviceColumnView<R, K1>,
                second_key: crate::detail::device::DeviceColumnView<R, K2>,
                third_key: crate::detail::device::DeviceColumnView<R, K3>,
                _less: Less,
            ) -> Result<(KeyOutput, ValueOutput), Error>
            where
                K1: Scalar + 'static,
                K2: Scalar + 'static,
                K3: Scalar + 'static,
                Less: op::BinaryPredicateOp<R, (K1, K2, K3)>,
                KeyOutput: MVec<R, Item = (K1, K2, K3)>,
                ValueOutput: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let (key_inner, value_inner) = impl_wide_sort_by_three_key_dispatch_body!(
                    policy, first_key, second_key, third_key, _less, input; $( $idx ),+
                )?;
                Ok((
                    array_from_inner::<R, (K1, K2, K3), KeyOutput>(key_inner),
                    array_from_inner::<R, <Self as MIter<R>>::Item, ValueOutput>(value_inner),
                ))
            }

            fn merge_by_three_key_same_dispatch<K1, K2, K3, RightValues, Less, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                left_first_key: crate::detail::device::DeviceColumnView<R, K1>,
                left_second_key: crate::detail::device::DeviceColumnView<R, K2>,
                left_third_key: crate::detail::device::DeviceColumnView<R, K3>,
                right_first_key: crate::detail::device::DeviceColumnView<R, K1>,
                right_second_key: crate::detail::device::DeviceColumnView<R, K2>,
                right_third_key: crate::detail::device::DeviceColumnView<R, K3>,
                right_values: RightValues,
                _less: Less,
            ) -> Result<(KeyOutput, ValueOutput), Error>
            where
                RightValues: MIter<
                    R,
                    Item = <Self as MIter<R>>::Item,
                    Inner = <Self as MIter<R>>::Inner,
                >,
                K1: Scalar + 'static,
                K2: Scalar + 'static,
                K3: Scalar + 'static,
                Less: op::BinaryPredicateOp<R, (K1, K2, K3)>,
                KeyOutput: MVec<R, Item = (K1, K2, K3)>,
                ValueOutput: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let left_values = self.into_inner_with_policy(policy)?;
                let right_values = right_values.into_inner_with_policy(policy)?;
                let (key_inner, value_inner) = impl_wide_merge_by_three_key_dispatch_body!(
                    policy,
                    left_values,
                    right_values,
                    left_first_key,
                    left_second_key,
                    left_third_key,
                    right_first_key,
                    right_second_key,
                    right_third_key,
                    _less;
                    $( $idx ),+
                )?;
                Ok((
                    array_from_inner::<R, (K1, K2, K3), KeyOutput>(key_inner),
                    array_from_inner::<R, <Self as MIter<R>>::Item, ValueOutput>(value_inner),
                ))
            }

            fn copy_where_dispatch<Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                stencil: crate::detail::api::PrecomputedSelection<R>,
            ) -> Result<Output, Error>
            where
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                ensure_same_len(input.0.len, stencil.control().len)?;
                $(
                    let $tmp = crate::detail::api::device_expr_compact_with_flags_with_policy(
                        policy,
                        &input.$idx,
                        stencil.control().flag.clone(),
                    )?;
                )+
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(($($tmp,)+)))
            }

            fn remove_where_dispatch<Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                stencil: crate::detail::api::PrecomputedSelection<R>,
            ) -> Result<Output, Error>
            where
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                ensure_same_len(input.0.len, stencil.control().len)?;
                $(
                    let $tmp = crate::detail::api::device_expr_compact_with_flags_with_policy(
                        policy,
                        &input.$idx,
                        stencil.control().flag.clone(),
                    )?;
                )+
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(($($tmp,)+)))
            }

            fn unique_dispatch<Pred, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _pred: Pred,
            ) -> Result<Output, Error>
            where
                Pred: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let flags: cubecl::server::Handle =
                    impl_wide_unique_dispatch_body!(policy, input; $( $idx ),+)?;
                $(
                    let $tmp = crate::detail::api::device_expr_compact_with_flags_with_policy(
                        policy,
                        &input.$idx,
                        flags.clone(),
                    )?;
                )+
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(($($tmp,)+)))
            }

            fn reduce_dispatch<Op>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                init: <Self as MIter<R>>::Item,
                _op: Op,
            ) -> Result<<Self as MIter<R>>::Item, Error>
            where
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                impl_wide_reduce_dispatch_body!(policy, input, init; $( $ty ),+; $( $idx ),+)
            }

            fn inclusive_scan_dispatch<Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _op: Op,
            ) -> Result<Output, Error>
            where
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = impl_wide_inclusive_scan_dispatch_body!(policy, input; $( $ty ),+; $( $idx ),+)?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn exclusive_scan_dispatch<Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                init: <Self as MIter<R>>::Item,
                _op: Op,
            ) -> Result<Output, Error>
            where
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = impl_wide_exclusive_scan_dispatch_body!(policy, input, init; $( $ty ),+; $( $idx ),+)?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn gather_dispatch<Indices, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                indices: Indices,
                output: Output,
            ) -> Result<(), Error>
            where
                Indices: crate::detail::device::KernelColumn<Runtime = R, Item = u32>
                    + crate::detail::device::KernelColumnAt<crate::detail::device::S0>,
                <Indices as crate::detail::device::KernelColumn>::Expr:
                    crate::expr::GpuExpr<u32> + crate::expr::DeviceGpuExpr<u32>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                $(
                    let $tmp = <Output as sealed::MIterMutDispatch<R>>::column_mut_view_by_index_inner::<$ty>(
                        &output,
                        $idx,
                    )?
                    .ok_or_else(|| Error::Launch {
                        message: "gather output must match input shape".to_string(),
                    })?;
                    crate::detail::api::device_expr_gather_into_with_policy(
                        policy,
                        &input.$idx,
                        &indices,
                        &$tmp,
                    )?;
                )+
                Ok(())
            }

            fn permute_dispatch<Indices, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                indices: Indices,
            ) -> Result<Output, Error>
            where
                Indices: crate::detail::device::KernelColumn<Runtime = R, Item = u32>
                    + crate::detail::device::KernelColumnAt<crate::detail::device::S0>,
                <Indices as crate::detail::device::KernelColumn>::Expr:
                    crate::expr::GpuExpr<u32> + crate::expr::DeviceGpuExpr<u32>,
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                $(
                    let $tmp = crate::detail::api::device_expr_gather_with_policy(
                        policy,
                        &input.$idx,
                        &indices,
                    )?;
                )+
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(($($tmp,)+)))
            }

            fn gather_where_dispatch<Indices, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                indices: Indices,
                stencil: crate::detail::api::PrecomputedSelection<R>,
                output: Output,
            ) -> Result<(), Error>
            where
                Indices: crate::detail::device::KernelColumn<Runtime = R, Item = u32>
                    + crate::detail::device::KernelColumnAt<crate::detail::device::S0>,
                <Indices as crate::detail::device::KernelColumn>::Expr:
                    crate::expr::GpuExpr<u32> + crate::expr::DeviceGpuExpr<u32>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                $(
                    let $tmp = <Output as sealed::MIterMutDispatch<R>>::column_mut_view_by_index_inner::<$ty>(
                        &output,
                        $idx,
                    )?
                    .ok_or_else(|| Error::Launch {
                        message: "gather_where output must match input shape".to_string(),
                    })?;
                    crate::detail::api::device_expr_gather_where_into_with_control(
                        policy,
                        &input.$idx,
                        &indices,
                        stencil.control(),
                        &$tmp,
                    )?;
                )+
                Ok(())
            }

            fn scatter_dispatch<Indices, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                indices: Indices,
                output: Output,
            ) -> Result<(), Error>
            where
                Indices: crate::detail::device::KernelColumn<Runtime = R, Item = u32>
                    + crate::detail::device::KernelColumnAt<crate::detail::device::S0>,
                <Indices as crate::detail::device::KernelColumn>::Expr: crate::expr::GpuExpr<u32>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                $(
                    let $tmp = <Output as sealed::MIterMutDispatch<R>>::column_mut_view_by_index_inner::<$ty>(
                        &output,
                        $idx,
                    )?
                    .ok_or_else(|| Error::Launch {
                        message: "scatter output must match input shape".to_string(),
                    })?;
                    crate::detail::api::device_expr_scatter_into_with_policy(
                        policy,
                        &input.$idx,
                        &indices,
                        &$tmp,
                    )?;
                )+
                Ok(())
            }

            fn scatter_where_dispatch<Indices, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                indices: Indices,
                stencil: crate::detail::api::PrecomputedSelection<R>,
                output: Output,
            ) -> Result<(), Error>
            where
                Indices: crate::detail::device::KernelColumn<Runtime = R, Item = u32>
                    + crate::detail::device::KernelColumnAt<crate::detail::device::S0>,
                <Indices as crate::detail::device::KernelColumn>::Expr:
                    crate::expr::GpuExpr<u32> + crate::expr::DeviceGpuExpr<u32>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                $(
                    let $tmp = <Output as sealed::MIterMutDispatch<R>>::column_mut_view_by_index_inner::<$ty>(
                        &output,
                        $idx,
                    )?
                    .ok_or_else(|| Error::Launch {
                        message: "scatter_where output must match input shape".to_string(),
                    })?;
                    crate::detail::api::device_expr_scatter_where_into_with_control(
                        policy,
                        &input.$idx,
                        &indices,
                        stencil.control(),
                        &$tmp,
                    )?;
                )+
                Ok(())
            }

            fn inclusive_scan_by_three_key_dispatch<K1, K2, K3, KeyEq, Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                first_key: crate::detail::device::DeviceColumnView<R, K1>,
                second_key: crate::detail::device::DeviceColumnView<R, K2>,
                third_key: crate::detail::device::DeviceColumnView<R, K3>,
                _key_eq: KeyEq,
                _op: Op,
            ) -> Result<Output, Error>
            where
                K1: Scalar + 'static,
                K2: Scalar + 'static,
                K3: Scalar + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K1, K2, K3)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                ensure_same_len(input.0.len, first_key.len)?;
                let head_flags = crate::detail::read::unique_tuple3_flags_read::<
                    crate::detail::device::DeviceColumnView<R, K1>,
                    crate::detail::device::DeviceColumnView<R, K2>,
                    crate::detail::device::DeviceColumnView<R, K3>,
                    KernelOp<R, KeyEq>,
                >(
                    policy,
                    &first_key,
                    &second_key,
                    &third_key,
                )?;
                let len_u32 = u32::try_from(first_key.len)
                    .map_err(|_| Error::LengthTooLarge { len: first_key.len })?;
                let control: crate::detail::control::ScanByKeyControl<
                    R,
                    (K1, K2, K3),
                    (),
                    KernelOp<R, KeyEq>,
                > = crate::detail::control::ScanByKeyControl {
                    key_bindings: crate::detail::device::KernelColumnBindings::empty(policy.client()),
                    head_flags,
                    len: first_key.len,
                    len_u32,
                    _marker: std::marker::PhantomData,
                };
                let inner = impl_wide_inclusive_scan_by_three_key_values_body!(
                    policy, input, control; $( $ty ),+; $( $idx ),+
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn exclusive_scan_by_three_key_dispatch<K1, K2, K3, KeyEq, Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                first_key: crate::detail::device::DeviceColumnView<R, K1>,
                second_key: crate::detail::device::DeviceColumnView<R, K2>,
                third_key: crate::detail::device::DeviceColumnView<R, K3>,
                _key_eq: KeyEq,
                init: <Self as MIter<R>>::Item,
                _op: Op,
            ) -> Result<Output, Error>
            where
                K1: Scalar + 'static,
                K2: Scalar + 'static,
                K3: Scalar + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K1, K2, K3)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                ensure_same_len(input.0.len, first_key.len)?;
                let head_flags = crate::detail::read::unique_tuple3_flags_read::<
                    crate::detail::device::DeviceColumnView<R, K1>,
                    crate::detail::device::DeviceColumnView<R, K2>,
                    crate::detail::device::DeviceColumnView<R, K3>,
                    KernelOp<R, KeyEq>,
                >(
                    policy,
                    &first_key,
                    &second_key,
                    &third_key,
                )?;
                let len_u32 = u32::try_from(first_key.len)
                    .map_err(|_| Error::LengthTooLarge { len: first_key.len })?;
                let control: crate::detail::control::ScanByKeyControl<
                    R,
                    (K1, K2, K3),
                    (),
                    KernelOp<R, KeyEq>,
                > = crate::detail::control::ScanByKeyControl {
                    key_bindings: crate::detail::device::KernelColumnBindings::empty(policy.client()),
                    head_flags,
                    len: first_key.len,
                    len_u32,
                    _marker: std::marker::PhantomData,
                };
                let inner = impl_wide_exclusive_scan_by_three_key_values_body!(
                    policy, input, control, init; $( $ty ),+; $( $idx ),+
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn reduce_by_three_key_dispatch<K1, K2, K3, KeyEq, Op, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                first_key: crate::detail::device::DeviceColumnView<R, K1>,
                second_key: crate::detail::device::DeviceColumnView<R, K2>,
                third_key: crate::detail::device::DeviceColumnView<R, K3>,
                _key_eq: KeyEq,
                init: <Self as MIter<R>>::Item,
                _op: Op,
            ) -> Result<(KeyOutput, ValueOutput), Error>
            where
                K1: Scalar + 'static,
                K2: Scalar + 'static,
                K3: Scalar + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K1, K2, K3)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                KeyOutput: MVec<R, Item = (K1, K2, K3)>,
                ValueOutput: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                ensure_same_len(input.0.len, first_key.len)?;
                if first_key.len == 0 {
                    let key_inner = (
                        policy.empty_device_vec(),
                        policy.empty_device_vec(),
                        policy.empty_device_vec(),
                    );
                    let value_inner = ($( {
                        let _ = stringify!($ty);
                        policy.empty_device_vec()
                    }, )+);
                    return Ok((
                        array_from_inner::<R, (K1, K2, K3), KeyOutput>(key_inner),
                        array_from_inner::<R, <Self as MIter<R>>::Item, ValueOutput>(value_inner),
                    ));
                }
                let head_flags = crate::detail::read::unique_tuple3_flags_read::<
                    crate::detail::device::DeviceColumnView<R, K1>,
                    crate::detail::device::DeviceColumnView<R, K2>,
                    crate::detail::device::DeviceColumnView<R, K3>,
                    KernelOp<R, KeyEq>,
                >(
                    policy,
                    &first_key,
                    &second_key,
                    &third_key,
                )?;
                let end_flags =
                    crate::detail::impls::end_flags_from_head_flags(policy, head_flags.clone(), first_key.len)?;
                let len_u32 = u32::try_from(first_key.len)
                    .map_err(|_| Error::LengthTooLarge { len: first_key.len })?;
                let control: crate::detail::control::ScanByKeyControl<
                    R,
                    (K1, K2, K3),
                    (),
                    KernelOp<R, KeyEq>,
                > = crate::detail::control::ScanByKeyControl {
                    key_bindings: crate::detail::device::KernelColumnBindings::empty(policy.client()),
                    head_flags,
                    len: first_key.len,
                    len_u32,
                    _marker: std::marker::PhantomData,
                };
                let (key_inner, value_inner) = impl_wide_reduce_by_three_key_values_body!(
                    policy, input, init, first_key, second_key, third_key, end_flags, len_u32, control; $( $ty ),+; $( $idx ),+
                )?;
                Ok((
                    array_from_inner::<R, (K1, K2, K3), KeyOutput>(key_inner),
                    array_from_inner::<R, <Self as MIter<R>>::Item, ValueOutput>(value_inner),
                ))
            }

            fn unique_by_three_key_dispatch<K1, K2, K3, Eq, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                first_key: crate::detail::device::DeviceColumnView<R, K1>,
                second_key: crate::detail::device::DeviceColumnView<R, K2>,
                third_key: crate::detail::device::DeviceColumnView<R, K3>,
                _eq: Eq,
            ) -> Result<(KeyOutput, ValueOutput), Error>
            where
                K1: Scalar + 'static,
                K2: Scalar + 'static,
                K3: Scalar + 'static,
                Eq: op::BinaryPredicateOp<R, (K1, K2, K3)>,
                KeyOutput: MVec<R, Item = (K1, K2, K3)>,
                ValueOutput: MVec<R, Item = <Self as MIter<R>>::Item>,
            {
                let values = self.into_inner_with_policy(policy)?;
                ensure_same_len(values.0.len, first_key.len)?;
                let flags = crate::detail::read::unique_tuple3_flags_read::<
                    _,
                    _,
                    _,
                    KernelOp<R, Eq>,
                >(
                    policy,
                    &first_key,
                    &second_key,
                    &third_key,
                )?;
                let key_inner = (
                    crate::detail::api::device_expr_compact_with_flags_with_policy(
                        policy,
                        &first_key,
                        flags.clone(),
                    )?,
                    crate::detail::api::device_expr_compact_with_flags_with_policy(
                        policy,
                        &second_key,
                        flags.clone(),
                    )?,
                    crate::detail::api::device_expr_compact_with_flags_with_policy(
                        policy,
                        &third_key,
                        flags.clone(),
                    )?,
                );
                $(
                    let $tmp = crate::detail::api::device_expr_compact_with_flags_with_policy(
                        policy,
                        &values.$idx,
                        flags.clone(),
                    )?;
                )+
                Ok((
                    array_from_inner::<R, (K1, K2, K3), KeyOutput>(key_inner),
                    array_from_inner::<R, <Self as MIter<R>>::Item, ValueOutput>(($($tmp,)+)),
                ))
            }
        }
    };
}

impl_miter_soa!(SoA2; A: 0: a, C: 1: c => transform_binary);
impl_miter_soa!(SoA3; A: 0: a, C: 1: c, D: 2: d => transform_ternary);
impl_wide_miter_soa!(SoA4; A: 0: a, C: 1: c, D: 2: d, E: 3: e);
impl_wide_miter_soa!(SoA5; A: 0: a, C: 1: c, D: 2: d, E: 3: e, F: 4: f);
impl_wide_miter_soa!(SoA6; A: 0: a, C: 1: c, D: 2: d, E: 3: e, F: 4: f, G: 5: g);
impl_wide_miter_soa!(SoA7; A: 0: a, C: 1: c, D: 2: d, E: 3: e, F: 4: f, G: 5: g, H: 6: h);
impl_miter_mut_soa!(SoA2; A: 0, C: 1);
impl_miter_mut_soa!(SoA3; A: 0, C: 1, D: 2);
impl_miter_mut_soa!(SoA4; A: 0, C: 1, D: 2, E: 3);
impl_miter_mut_soa!(SoA5; A: 0, C: 1, D: 2, E: 3, F: 4);
impl_miter_mut_soa!(SoA6; A: 0, C: 1, D: 2, E: 3, F: 4, G: 5);
impl_miter_mut_soa!(SoA7; A: 0, C: 1, D: 2, E: 3, F: 4, G: 5, H: 6);
