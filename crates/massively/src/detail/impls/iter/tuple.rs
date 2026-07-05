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

    ($input:ident; 0, 1, 2, 3) => {
        crate::detail::device::SoAView4 {
            a: $input.0,
            b: $input.1,
            c: $input.2,
            d: $input.3,
        }
    };

    ($input:ident; 0, 1, 2, 3, 4) => {
        crate::detail::device::SoAView5 {
            a: $input.0,
            b: $input.1,
            c: $input.2,
            d: $input.3,
            e: $input.4,
        }
    };

    ($input:ident; 0, 1, 2, 3, 4, 5) => {
        crate::detail::device::SoAView6 {
            a: $input.0,
            b: $input.1,
            c: $input.2,
            d: $input.3,
            e: $input.4,
            f: $input.5,
        }
    };

    ($input:ident; 0, 1, 2, 3, 4, 5, 6) => {
        crate::detail::device::SoAView7 {
            a: $input.0,
            b: $input.1,
            c: $input.2,
            d: $input.3,
            e: $input.4,
            f: $input.5,
            g: $input.6,
        }
    };
}

macro_rules! impl_unique_by_key_dispatch_body {
    ($self:ident, $policy:ident, $values:ident, $eq:ident, $input:ident; 0, 1) => {{
        <Values as sealed::MIterDispatch<R>>::unique_by_two_key_dispatch(
            $values, $policy, $input.0, $input.1, $eq,
        )
    }};
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

macro_rules! impl_unique_by_key_into_dispatch_body {
    ($policy:ident, $values:ident, $eq:ident, $input:ident, $key_output:ident, $value_output:ident; 0, 1) => {{
        <Values as sealed::MIterDispatch<R>>::unique_by_two_key_into_dispatch(
            $values,
            $policy,
            $input.0,
            $input.1,
            $eq,
            $key_output,
            $value_output,
        )
    }};
    ($policy:ident, $values:ident, $eq:ident, $input:ident, $key_output:ident, $value_output:ident; 0, 1, 2) => {{
        <Values as sealed::MIterDispatch<R>>::unique_by_three_key_into_dispatch(
            $values,
            $policy,
            $input.0,
            $input.1,
            $input.2,
            $eq,
            $key_output,
            $value_output,
        )
    }};
    ($policy:ident, $values:ident, $eq:ident, $input:ident, $key_output:ident, $value_output:ident; $( $idx:tt ),+) => {{
        let _ = ($policy, $values, $eq, $input, $key_output, $value_output);
        Err(Error::Launch {
            message: "unique_by_key is not supported for this key iterator shape".to_string(),
        })
    }};
}

macro_rules! impl_sort_by_key_dispatch_body {
    ($policy:ident, $values:ident, $less:ident, $input:ident; 0, 1) => {{
        <Values as sealed::MIterDispatch<R>>::sort_by_two_key_dispatch(
            $values, $policy, $input.0, $input.1, $less,
        )
    }};
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

macro_rules! impl_sort_by_key_into_dispatch_body {
    ($policy:ident, $values:ident, $less:ident, $input:ident, $key_output:ident, $value_output:ident; 0, 1) => {{
        <Values as sealed::MIterDispatch<R>>::sort_by_two_key_into_dispatch(
            $values,
            $policy,
            $input.0,
            $input.1,
            $less,
            $key_output,
            $value_output,
        )
    }};
    ($policy:ident, $values:ident, $less:ident, $input:ident, $key_output:ident, $value_output:ident; 0, 1, 2) => {{
        <Values as sealed::MIterDispatch<R>>::sort_by_three_key_into_dispatch(
            $values,
            $policy,
            $input.0,
            $input.1,
            $input.2,
            $less,
            $key_output,
            $value_output,
        )
    }};
    ($policy:ident, $values:ident, $less:ident, $input:ident, $key_output:ident, $value_output:ident; $( $idx:tt ),+) => {{
        let _ = ($policy, $values, $less, $input, $key_output, $value_output);
        Err(Error::Launch {
            message: "sort_by_key is not supported for this key iterator shape".to_string(),
        })
    }};
}

macro_rules! impl_inclusive_scan_by_key_dispatch_body {
    ($policy:ident, $values:ident, $key_eq:ident, $op:ident, $input:ident; 0, 1) => {{
        <Values as sealed::MIterDispatch<R>>::inclusive_scan_by_two_key_dispatch(
            $values, $policy, $input.0, $input.1, $key_eq, $op,
        )
    }};
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

macro_rules! impl_inclusive_scan_by_key_into_dispatch_body {
    ($policy:ident, $values:ident, $key_eq:ident, $op:ident, $input:ident, $output:ident; 0, 1) => {{
        <Values as sealed::MIterDispatch<R>>::inclusive_scan_by_two_key_into_dispatch(
            $values, $policy, $input.0, $input.1, $key_eq, $op, $output,
        )
    }};
    ($policy:ident, $values:ident, $key_eq:ident, $op:ident, $input:ident, $output:ident; 0, 1, 2) => {{
        <Values as sealed::MIterDispatch<R>>::inclusive_scan_by_three_key_into_dispatch(
            $values, $policy, $input.0, $input.1, $input.2, $key_eq, $op, $output,
        )
    }};
    ($policy:ident, $values:ident, $key_eq:ident, $op:ident, $input:ident, $output:ident; $( $idx:tt ),+) => {{
        let _ = ($policy, $values, $key_eq, $op, $input, $output);
        Err(Error::Launch {
            message: "inclusive_scan_by_key is not supported for this key iterator shape"
                .to_string(),
        })
    }};
}

macro_rules! impl_exclusive_scan_by_key_dispatch_body {
    ($policy:ident, $values:ident, $key_eq:ident, $init:ident, $op:ident, $input:ident; 0, 1) => {{
        <Values as sealed::MIterDispatch<R>>::exclusive_scan_by_two_key_dispatch(
            $values, $policy, $input.0, $input.1, $key_eq, $init, $op,
        )
    }};
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

macro_rules! impl_exclusive_scan_by_key_into_dispatch_body {
    ($policy:ident, $values:ident, $key_eq:ident, $init:ident, $op:ident, $input:ident, $output:ident; 0, 1) => {{
        <Values as sealed::MIterDispatch<R>>::exclusive_scan_by_two_key_into_dispatch(
            $values, $policy, $input.0, $input.1, $key_eq, $init, $op, $output,
        )
    }};
    ($policy:ident, $values:ident, $key_eq:ident, $init:ident, $op:ident, $input:ident, $output:ident; 0, 1, 2) => {{
        <Values as sealed::MIterDispatch<R>>::exclusive_scan_by_three_key_into_dispatch(
            $values, $policy, $input.0, $input.1, $input.2, $key_eq, $init, $op, $output,
        )
    }};
    ($policy:ident, $values:ident, $key_eq:ident, $init:ident, $op:ident, $input:ident, $output:ident; $( $idx:tt ),+) => {{
        let _ = ($policy, $values, $key_eq, $init, $op, $input, $output);
        Err(Error::Launch {
            message: "exclusive_scan_by_key is not supported for this key iterator shape"
                .to_string(),
        })
    }};
}

macro_rules! impl_reduce_by_key_dispatch_body {
    ($policy:ident, $values:ident, $key_eq:ident, $init:ident, $op:ident, $input:ident; 0, 1) => {{
        <Values as sealed::MIterDispatch<R>>::reduce_by_two_key_dispatch(
            $values, $policy, $input.0, $input.1, $key_eq, $init, $op,
        )
    }};
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

macro_rules! impl_reduce_by_key_into_dispatch_body {
    ($policy:ident, $values:ident, $key_eq:ident, $init:ident, $op:ident, $input:ident, $key_output:ident, $value_output:ident; 0, 1) => {{
        <Values as sealed::MIterDispatch<R>>::reduce_by_two_key_into_dispatch(
            $values,
            $policy,
            $input.0,
            $input.1,
            $key_eq,
            $init,
            $op,
            $key_output,
            $value_output,
        )
    }};
    ($policy:ident, $values:ident, $key_eq:ident, $init:ident, $op:ident, $input:ident, $key_output:ident, $value_output:ident; 0, 1, 2) => {{
        <Values as sealed::MIterDispatch<R>>::reduce_by_three_key_into_dispatch(
            $values,
            $policy,
            $input.0,
            $input.1,
            $input.2,
            $key_eq,
            $init,
            $op,
            $key_output,
            $value_output,
        )
    }};
    ($policy:ident, $values:ident, $key_eq:ident, $init:ident, $op:ident, $input:ident, $key_output:ident, $value_output:ident; $( $idx:tt ),+) => {{
        let _ = (
            $policy,
            $values,
            $key_eq,
            $init,
            $op,
            $input,
            $key_output,
            $value_output,
        );
        Err(Error::Launch {
            message: "reduce_by_key is not supported for this key iterator shape".to_string(),
        })
    }};
}

macro_rules! impl_merge_by_key_dispatch_body {
    ($policy:ident, $right_keys:ident, $left_values:ident, $right_values:ident, $less:ident, $left_input:ident; 0, 1) => {{
        let right_input = $right_keys.into_view_with_policy($policy)?;
        <LeftValues as sealed::MIterDispatch<R>>::merge_by_two_key_same_dispatch(
            $left_values,
            $policy,
            $left_input.0,
            $left_input.1,
            right_input.0,
            right_input.1,
            $right_values,
            $less,
        )
    }};
    ($policy:ident, $right_keys:ident, $left_values:ident, $right_values:ident, $less:ident, $left_input:ident; 0, 1, 2) => {{
        let right_input = $right_keys.into_view_with_policy($policy)?;
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
    ($policy:ident, $input:ident, $op:ident, $env:ident; 0, 1, 2, 3) => {{
        <Output::Item as sealed::MItemDispatch<R>>::transform_quaternary(
            $policy, $input.0, $input.1, $input.2, $input.3, $op, $env,
        )
    }};
    ($policy:ident, $input:ident, $op:ident, $env:ident; 0, 1, 2, 3, 4) => {{
        <Output::Item as sealed::MItemDispatch<R>>::transform_quinary(
            $policy, $input.0, $input.1, $input.2, $input.3, $input.4, $op, $env,
        )
    }};
    ($policy:ident, $input:ident, $op:ident, $env:ident; 0, 1, 2, 3, 4, 5) => {{
        <Output::Item as sealed::MItemDispatch<R>>::transform_senary(
            $policy, $input.0, $input.1, $input.2, $input.3, $input.4, $input.5, $op, $env,
        )
    }};
    ($policy:ident, $input:ident, $op:ident, $env:ident; 0, 1, 2, 3, 4, 5, 6) => {{
        <Output::Item as sealed::MItemDispatch<R>>::transform_septenary(
            $policy, $input.0, $input.1, $input.2, $input.3, $input.4, $input.5, $input.6, $op,
            $env,
        )
    }};
    ($policy:ident, $input:ident, $op:ident, $env:ident; $( $idx:tt ),+) => {{
        let _ = ($policy, $input, $op, $env);
        Err(Error::Launch {
            message: "transform is not supported for this iterator shape".to_string(),
        })
    }};
}

macro_rules! impl_merge_by_key_into_dispatch_body {
    ($policy:ident, $right_keys:ident, $left_values:ident, $right_values:ident, $less:ident, $left_input:ident, $key_output:ident, $value_output:ident; 0, 1) => {{
        let right_input = $right_keys.into_view_with_policy($policy)?;
        <LeftValues as sealed::MIterDispatch<R>>::merge_by_two_key_same_into_dispatch(
            $left_values,
            $policy,
            $left_input.0,
            $left_input.1,
            right_input.0,
            right_input.1,
            $right_values,
            $less,
            $key_output,
            $value_output,
        )
    }};
    ($policy:ident, $right_keys:ident, $left_values:ident, $right_values:ident, $less:ident, $left_input:ident, $key_output:ident, $value_output:ident; 0, 1, 2) => {{
        let right_input = $right_keys.into_view_with_policy($policy)?;
        <LeftValues as sealed::MIterDispatch<R>>::merge_by_three_key_same_into_dispatch(
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
            $key_output,
            $value_output,
        )
    }};
    ($policy:ident, $right_keys:ident, $left_values:ident, $right_values:ident, $less:ident, $left_input:ident, $key_output:ident, $value_output:ident; $( $idx:tt ),+) => {{
        let _ = (
            $policy,
            $right_keys,
            $left_values,
            $right_values,
            $less,
            $left_input,
            $key_output,
            $value_output,
        );
        Err(Error::Launch {
            message: "merge_by_key is not supported for this key iterator shape".to_string(),
        })
    }};
}

macro_rules! impl_wide_sort_by_single_key_dispatch_body {
    ($policy:ident, $keys:ident, $less:ident, $input:ident; 0, 1, 2, 3) => {{
        crate::detail::sort_by_key(
            $policy,
            ($keys,),
            ($input.0, $input.1, $input.2, $input.3),
            KernelOp::<R, Less>::new(),
        )
    }};
    ($policy:ident, $keys:ident, $less:ident, $input:ident; 0, 1, 2, 3, 4) => {{
        crate::detail::sort_by_key(
            $policy,
            ($keys,),
            ($input.0, $input.1, $input.2, $input.3, $input.4),
            KernelOp::<R, Less>::new(),
        )
    }};
    ($policy:ident, $keys:ident, $less:ident, $input:ident; 0, 1, 2, 3, 4, 5) => {{
        crate::detail::sort_by_key(
            $policy,
            ($keys,),
            ($input.0, $input.1, $input.2, $input.3, $input.4, $input.5),
            KernelOp::<R, Less>::new(),
        )
    }};
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
    ($policy:ident, $input:ident; 0, 1, 2, 3) => {{
        let dummy4 = crate::detail::primitives::range::indices_mindex($policy, $input.0.len)?;
        let dummy5 = crate::detail::primitives::range::indices_mindex($policy, $input.0.len)?;
        let dummy6 = crate::detail::primitives::range::indices_mindex($policy, $input.0.len)?;
        let dummy4 = crate::detail::device::DeviceColumnView::from_column(&dummy4);
        let dummy5 = crate::detail::device::DeviceColumnView::from_column(&dummy5);
        let dummy6 = crate::detail::device::DeviceColumnView::from_column(&dummy6);
        let indices = crate::detail::primitives::ordering::sort_tuple7_indices_input::<
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            crate::detail::api::Tuple4AsTuple7BinaryPredicateOp<KernelOp<R, Less>>,
        >(
            $policy,
            &$input.0,
            &$input.1,
            &$input.2,
            &$input.3,
            &dummy4,
            &dummy5,
            &dummy6,
            crate::op::GpuOp::<
                crate::detail::api::Tuple4AsTuple7BinaryPredicateOp<KernelOp<R, Less>>,
            >::new(),
        )?;
        let control = crate::detail::control::OrderingControl::from_sorted_indices(&indices)?;
        let apply = crate::detail::apply::PermutationPayloadApply::new(control.permutation());
        apply.apply_expr4($policy, &$input.0, &$input.1, &$input.2, &$input.3)
    }};
    ($policy:ident, $input:ident; 0, 1, 2, 3, 4) => {{
        let dummy5 = crate::detail::primitives::range::indices_mindex($policy, $input.0.len)?;
        let dummy6 = crate::detail::primitives::range::indices_mindex($policy, $input.0.len)?;
        let dummy5 = crate::detail::device::DeviceColumnView::from_column(&dummy5);
        let dummy6 = crate::detail::device::DeviceColumnView::from_column(&dummy6);
        let indices = crate::detail::primitives::ordering::sort_tuple7_indices_input::<
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            crate::detail::api::Tuple5AsTuple7BinaryPredicateOp<KernelOp<R, Less>>,
        >(
            $policy,
            &$input.0,
            &$input.1,
            &$input.2,
            &$input.3,
            &$input.4,
            &dummy5,
            &dummy6,
            crate::op::GpuOp::<
                crate::detail::api::Tuple5AsTuple7BinaryPredicateOp<KernelOp<R, Less>>,
            >::new(),
        )?;
        let control = crate::detail::control::OrderingControl::from_sorted_indices(&indices)?;
        let apply = crate::detail::apply::PermutationPayloadApply::new(control.permutation());
        apply.apply_expr5(
            $policy, &$input.0, &$input.1, &$input.2, &$input.3, &$input.4,
        )
    }};
    ($policy:ident, $input:ident; 0, 1, 2, 3, 4, 5) => {{
        let dummy6 = crate::detail::primitives::range::indices_mindex($policy, $input.0.len)?;
        let dummy6 = crate::detail::device::DeviceColumnView::from_column(&dummy6);
        let indices = crate::detail::primitives::ordering::sort_tuple7_indices_input::<
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            crate::detail::api::Tuple6AsTuple7BinaryPredicateOp<KernelOp<R, Less>>,
        >(
            $policy,
            &$input.0,
            &$input.1,
            &$input.2,
            &$input.3,
            &$input.4,
            &$input.5,
            &dummy6,
            crate::op::GpuOp::<
                crate::detail::api::Tuple6AsTuple7BinaryPredicateOp<KernelOp<R, Less>>,
            >::new(),
        )?;
        let control = crate::detail::control::OrderingControl::from_sorted_indices(&indices)?;
        let apply = crate::detail::apply::PermutationPayloadApply::new(control.permutation());
        apply.apply_expr6(
            $policy, &$input.0, &$input.1, &$input.2, &$input.3, &$input.4, &$input.5,
        )
    }};
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
        let control = crate::detail::control::OrderingControl::from_sorted_indices(&indices)?;
        let apply = crate::detail::apply::PermutationPayloadApply::new(control.permutation());
        apply.apply_expr7(
            $policy, &$input.0, &$input.1, &$input.2, &$input.3, &$input.4, &$input.5, &$input.6,
        )
    }};
    ($policy:ident, $input:ident; $( $idx:tt ),+) => {{
        let _ = ($policy, $input);
        Err(Error::Launch {
            message: "sort is not supported for this iterator shape".to_string(),
        })
    }};
}

macro_rules! impl_wide_sort_by_three_key_dispatch_body {
    ($policy:ident, $first_key:ident, $second_key:ident, $third_key:ident, $less:ident, $input:ident; 0, 1, 2, 3) => {{
        crate::detail::sort_by_key(
            $policy,
            ($first_key, $second_key, $third_key),
            ($input.0, $input.1, $input.2, $input.3),
            KernelOp::<R, Less>::new(),
        )
    }};
    ($policy:ident, $first_key:ident, $second_key:ident, $third_key:ident, $less:ident, $input:ident; 0, 1, 2, 3, 4) => {{
        crate::detail::sort_by_key(
            $policy,
            ($first_key, $second_key, $third_key),
            ($input.0, $input.1, $input.2, $input.3, $input.4),
            KernelOp::<R, Less>::new(),
        )
    }};
    ($policy:ident, $first_key:ident, $second_key:ident, $third_key:ident, $less:ident, $input:ident; 0, 1, 2, 3, 4, 5) => {{
        crate::detail::sort_by_key(
            $policy,
            ($first_key, $second_key, $third_key),
            ($input.0, $input.1, $input.2, $input.3, $input.4, $input.5),
            KernelOp::<R, Less>::new(),
        )
    }};
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

macro_rules! impl_wide_sort_by_two_key_dispatch_body {
    ($policy:ident, $first_key:ident, $second_key:ident, $less:ident, $input:ident; 0, 1, 2, 3) => {{
        crate::detail::sort_by_key(
            $policy,
            ($first_key, $second_key),
            ($input.0, $input.1, $input.2, $input.3),
            KernelOp::<R, Less>::new(),
        )
    }};
    ($policy:ident, $first_key:ident, $second_key:ident, $less:ident, $input:ident; 0, 1, 2, 3, 4) => {{
        crate::detail::sort_by_key(
            $policy,
            ($first_key, $second_key),
            ($input.0, $input.1, $input.2, $input.3, $input.4),
            KernelOp::<R, Less>::new(),
        )
    }};
    ($policy:ident, $first_key:ident, $second_key:ident, $less:ident, $input:ident; 0, 1, 2, 3, 4, 5) => {{
        crate::detail::sort_by_key(
            $policy,
            ($first_key, $second_key),
            ($input.0, $input.1, $input.2, $input.3, $input.4, $input.5),
            KernelOp::<R, Less>::new(),
        )
    }};
    ($policy:ident, $first_key:ident, $second_key:ident, $less:ident, $input:ident; 0, 1, 2, 3, 4, 5, 6) => {{
        crate::detail::sort_by_key(
            $policy,
            ($first_key, $second_key),
            (
                $input.0, $input.1, $input.2, $input.3, $input.4, $input.5, $input.6,
            ),
            KernelOp::<R, Less>::new(),
        )
    }};
    ($policy:ident, $first_key:ident, $second_key:ident, $less:ident, $input:ident; $( $idx:tt ),+) => {{
        let _ = ($policy, $first_key, $second_key, $less, $input);
        Err(Error::Launch {
            message: "sort_by_key is not supported for this value iterator shape".to_string(),
        })
    }};
}

macro_rules! impl_wide_merge_by_single_key_dispatch_body {
    ($policy:ident, $left_values:ident, $right_values:ident, $left_key:ident, $right_key:ident, $less:ident; 0, 1, 2, 3) => {{
        let left_dummy4 =
            crate::detail::primitives::range::indices_mindex($policy, $left_values.0.len)?;
        let left_dummy5 =
            crate::detail::primitives::range::indices_mindex($policy, $left_values.0.len)?;
        let left_dummy6 =
            crate::detail::primitives::range::indices_mindex($policy, $left_values.0.len)?;
        let right_dummy4 =
            crate::detail::primitives::range::indices_mindex($policy, $right_values.0.len)?;
        let right_dummy5 =
            crate::detail::primitives::range::indices_mindex($policy, $right_values.0.len)?;
        let right_dummy6 =
            crate::detail::primitives::range::indices_mindex($policy, $right_values.0.len)?;
        let left_dummy4 = crate::detail::device::DeviceColumnView::from_column(&left_dummy4);
        let left_dummy5 = crate::detail::device::DeviceColumnView::from_column(&left_dummy5);
        let left_dummy6 = crate::detail::device::DeviceColumnView::from_column(&left_dummy6);
        let right_dummy4 = crate::detail::device::DeviceColumnView::from_column(&right_dummy4);
        let right_dummy5 = crate::detail::device::DeviceColumnView::from_column(&right_dummy5);
        let right_dummy6 = crate::detail::device::DeviceColumnView::from_column(&right_dummy6);
        let (key_inner, (a, b, c, d, _, _, _)) = crate::detail::merge_by_key(
            $policy,
            ($left_key,),
            (
                $left_values.0,
                $left_values.1,
                $left_values.2,
                $left_values.3,
                left_dummy4,
                left_dummy5,
                left_dummy6,
            ),
            ($right_key,),
            (
                $right_values.0,
                $right_values.1,
                $right_values.2,
                $right_values.3,
                right_dummy4,
                right_dummy5,
                right_dummy6,
            ),
            KernelOp::<R, Less>::new(),
        )?;
        Ok((key_inner, (a, b, c, d)))
    }};
    ($policy:ident, $left_values:ident, $right_values:ident, $left_key:ident, $right_key:ident, $less:ident; 0, 1, 2, 3, 4) => {{
        let left_dummy5 =
            crate::detail::primitives::range::indices_mindex($policy, $left_values.0.len)?;
        let left_dummy6 =
            crate::detail::primitives::range::indices_mindex($policy, $left_values.0.len)?;
        let right_dummy5 =
            crate::detail::primitives::range::indices_mindex($policy, $right_values.0.len)?;
        let right_dummy6 =
            crate::detail::primitives::range::indices_mindex($policy, $right_values.0.len)?;
        let left_dummy5 = crate::detail::device::DeviceColumnView::from_column(&left_dummy5);
        let left_dummy6 = crate::detail::device::DeviceColumnView::from_column(&left_dummy6);
        let right_dummy5 = crate::detail::device::DeviceColumnView::from_column(&right_dummy5);
        let right_dummy6 = crate::detail::device::DeviceColumnView::from_column(&right_dummy6);
        let (key_inner, (a, b, c, d, e, _, _)) = crate::detail::merge_by_key(
            $policy,
            ($left_key,),
            (
                $left_values.0,
                $left_values.1,
                $left_values.2,
                $left_values.3,
                $left_values.4,
                left_dummy5,
                left_dummy6,
            ),
            ($right_key,),
            (
                $right_values.0,
                $right_values.1,
                $right_values.2,
                $right_values.3,
                $right_values.4,
                right_dummy5,
                right_dummy6,
            ),
            KernelOp::<R, Less>::new(),
        )?;
        Ok((key_inner, (a, b, c, d, e)))
    }};
    ($policy:ident, $left_values:ident, $right_values:ident, $left_key:ident, $right_key:ident, $less:ident; 0, 1, 2, 3, 4, 5) => {{
        let left_dummy6 =
            crate::detail::primitives::range::indices_mindex($policy, $left_values.0.len)?;
        let right_dummy6 =
            crate::detail::primitives::range::indices_mindex($policy, $right_values.0.len)?;
        let left_dummy6 = crate::detail::device::DeviceColumnView::from_column(&left_dummy6);
        let right_dummy6 = crate::detail::device::DeviceColumnView::from_column(&right_dummy6);
        let (key_inner, (a, b, c, d, e, f, _)) = crate::detail::merge_by_key(
            $policy,
            ($left_key,),
            (
                $left_values.0,
                $left_values.1,
                $left_values.2,
                $left_values.3,
                $left_values.4,
                $left_values.5,
                left_dummy6,
            ),
            ($right_key,),
            (
                $right_values.0,
                $right_values.1,
                $right_values.2,
                $right_values.3,
                $right_values.4,
                $right_values.5,
                right_dummy6,
            ),
            KernelOp::<R, Less>::new(),
        )?;
        Ok((key_inner, (a, b, c, d, e, f)))
    }};
    ($policy:ident, $left_values:ident, $right_values:ident, $left_key:ident, $right_key:ident, $less:ident; 0, 1, 2, 3, 4, 5, 6) => {{
        crate::detail::merge_by_key(
            $policy,
            ($left_key,),
            (
                $left_values.0,
                $left_values.1,
                $left_values.2,
                $left_values.3,
                $left_values.4,
                $left_values.5,
                $left_values.6,
            ),
            ($right_key,),
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
    ($policy:ident, $left_values:ident, $right_values:ident, $left_key:ident, $right_key:ident, $less:ident; $( $idx:tt ),+) => {{
        let _ = (
            $policy,
            $left_values,
            $right_values,
            $left_key,
            $right_key,
            $less,
        );
        Err(Error::Launch {
            message: "merge_by_key is not supported for this value iterator shape".to_string(),
        })
    }};
}

macro_rules! impl_wide_unique_dispatch_body {
    ($policy:ident, $input:ident; 0, 1, 2, 3) => {{
        let dummy4 = crate::detail::primitives::range::indices_mindex($policy, $input.0.len)?;
        let dummy5 = crate::detail::primitives::range::indices_mindex($policy, $input.0.len)?;
        let dummy6 = crate::detail::primitives::range::indices_mindex($policy, $input.0.len)?;
        let dummy4 = crate::detail::device::DeviceColumnView::from_column(&dummy4);
        let dummy5 = crate::detail::device::DeviceColumnView::from_column(&dummy5);
        let dummy6 = crate::detail::device::DeviceColumnView::from_column(&dummy6);
        crate::detail::read::unique_tuple7_flags_read::<
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            crate::detail::api::Tuple4AsTuple7BinaryPredicateOp<KernelOp<R, Pred>>,
        >(
            $policy, &$input.0, &$input.1, &$input.2, &$input.3, &dummy4, &dummy5, &dummy6,
        )
    }};
    ($policy:ident, $input:ident; 0, 1, 2, 3, 4) => {{
        let dummy5 = crate::detail::primitives::range::indices_mindex($policy, $input.0.len)?;
        let dummy6 = crate::detail::primitives::range::indices_mindex($policy, $input.0.len)?;
        let dummy5 = crate::detail::device::DeviceColumnView::from_column(&dummy5);
        let dummy6 = crate::detail::device::DeviceColumnView::from_column(&dummy6);
        crate::detail::read::unique_tuple7_flags_read::<
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            crate::detail::api::Tuple5AsTuple7BinaryPredicateOp<KernelOp<R, Pred>>,
        >(
            $policy, &$input.0, &$input.1, &$input.2, &$input.3, &$input.4, &dummy5, &dummy6,
        )
    }};
    ($policy:ident, $input:ident; 0, 1, 2, 3, 4, 5) => {{
        let dummy6 = crate::detail::primitives::range::indices_mindex($policy, $input.0.len)?;
        let dummy6 = crate::detail::device::DeviceColumnView::from_column(&dummy6);
        crate::detail::read::unique_tuple7_flags_read::<
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            crate::detail::api::Tuple6AsTuple7BinaryPredicateOp<KernelOp<R, Pred>>,
        >(
            $policy, &$input.0, &$input.1, &$input.2, &$input.3, &$input.4, &$input.5, &dummy6,
        )
    }};
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
    ($policy:ident, $input:ident, $init:ident; $ty0:ident, $ty1:ident, $ty2:ident, $ty3:ident; 0, 1, 2, 3) => {{
        crate::detail::apply::LinearReduceApply::apply_views4::<
            R,
            $ty0,
            $ty1,
            $ty2,
            $ty3,
            KernelOp<R, Op>,
        >($policy, &$input.0, &$input.1, &$input.2, &$input.3, $init)
    }};
    ($policy:ident, $input:ident, $init:ident; $ty0:ident, $ty1:ident, $ty2:ident, $ty3:ident, $ty4:ident; 0, 1, 2, 3, 4) => {{
        crate::detail::apply::LinearReduceApply::apply_views5::<
            R,
            $ty0,
            $ty1,
            $ty2,
            $ty3,
            $ty4,
            KernelOp<R, Op>,
        >(
            $policy, &$input.0, &$input.1, &$input.2, &$input.3, &$input.4, $init,
        )
    }};
    ($policy:ident, $input:ident, $init:ident; $ty0:ident, $ty1:ident, $ty2:ident, $ty3:ident, $ty4:ident, $ty5:ident; 0, 1, 2, 3, 4, 5) => {{
        crate::detail::apply::LinearReduceApply::apply_views6::<
            R,
            $ty0,
            $ty1,
            $ty2,
            $ty3,
            $ty4,
            $ty5,
            KernelOp<R, Op>,
        >(
            $policy, &$input.0, &$input.1, &$input.2, &$input.3, &$input.4, &$input.5, $init,
        )
    }};
    ($policy:ident, $input:ident, $init:ident; $ty0:ident, $ty1:ident, $ty2:ident, $ty3:ident, $ty4:ident, $ty5:ident, $ty6:ident; 0, 1, 2, 3, 4, 5, 6) => {{
        crate::detail::apply::LinearReduceApply::apply_views7::<
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
            message: "reduce is not supported for this iterator shape".to_string(),
        })
    }};
}

macro_rules! impl_wide_inclusive_scan_dispatch_body {
    ($policy:ident, $input:ident; $ty0:ident, $ty1:ident, $ty2:ident, $ty3:ident; 0, 1, 2, 3) => {{
        crate::detail::apply::LinearScanApply::inclusive_views4::<
            R,
            $ty0,
            $ty1,
            $ty2,
            $ty3,
            KernelOp<R, Op>,
        >($policy, &$input.0, &$input.1, &$input.2, &$input.3)
    }};
    ($policy:ident, $input:ident; $ty0:ident, $ty1:ident, $ty2:ident, $ty3:ident, $ty4:ident; 0, 1, 2, 3, 4) => {{
        crate::detail::apply::LinearScanApply::inclusive_views5::<
            R,
            $ty0,
            $ty1,
            $ty2,
            $ty3,
            $ty4,
            KernelOp<R, Op>,
        >(
            $policy, &$input.0, &$input.1, &$input.2, &$input.3, &$input.4,
        )
    }};
    ($policy:ident, $input:ident; $ty0:ident, $ty1:ident, $ty2:ident, $ty3:ident, $ty4:ident, $ty5:ident; 0, 1, 2, 3, 4, 5) => {{
        crate::detail::apply::LinearScanApply::inclusive_views6::<
            R,
            $ty0,
            $ty1,
            $ty2,
            $ty3,
            $ty4,
            $ty5,
            KernelOp<R, Op>,
        >(
            $policy, &$input.0, &$input.1, &$input.2, &$input.3, &$input.4, &$input.5,
        )
    }};
    ($policy:ident, $input:ident; $ty0:ident, $ty1:ident, $ty2:ident, $ty3:ident, $ty4:ident, $ty5:ident, $ty6:ident; 0, 1, 2, 3, 4, 5, 6) => {{
        crate::detail::apply::LinearScanApply::inclusive_views7::<
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
    ($policy:ident, $input:ident, $init:ident; $ty0:ident, $ty1:ident, $ty2:ident, $ty3:ident; 0, 1, 2, 3) => {{
        crate::detail::apply::LinearScanApply::exclusive_views4::<
            R,
            $ty0,
            $ty1,
            $ty2,
            $ty3,
            KernelOp<R, Op>,
        >($policy, &$input.0, &$input.1, &$input.2, &$input.3, $init)
    }};
    ($policy:ident, $input:ident, $init:ident; $ty0:ident, $ty1:ident, $ty2:ident, $ty3:ident, $ty4:ident; 0, 1, 2, 3, 4) => {{
        crate::detail::apply::LinearScanApply::exclusive_views5::<
            R,
            $ty0,
            $ty1,
            $ty2,
            $ty3,
            $ty4,
            KernelOp<R, Op>,
        >(
            $policy, &$input.0, &$input.1, &$input.2, &$input.3, &$input.4, $init,
        )
    }};
    ($policy:ident, $input:ident, $init:ident; $ty0:ident, $ty1:ident, $ty2:ident, $ty3:ident, $ty4:ident, $ty5:ident; 0, 1, 2, 3, 4, 5) => {{
        crate::detail::apply::LinearScanApply::exclusive_views6::<
            R,
            $ty0,
            $ty1,
            $ty2,
            $ty3,
            $ty4,
            $ty5,
            KernelOp<R, Op>,
        >(
            $policy, &$input.0, &$input.1, &$input.2, &$input.3, &$input.4, &$input.5, $init,
        )
    }};
    ($policy:ident, $input:ident, $init:ident; $ty0:ident, $ty1:ident, $ty2:ident, $ty3:ident, $ty4:ident, $ty5:ident, $ty6:ident; 0, 1, 2, 3, 4, 5, 6) => {{
        crate::detail::apply::LinearScanApply::exclusive_views7::<
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

macro_rules! impl_wide_adjacent_difference_dispatch_body {
    ($policy:ident, $input:ident; $ty0:ident, $ty1:ident, $ty2:ident, $ty3:ident; 0, 1, 2, 3) => {{
        crate::detail::apply::LinearScanApply::adjacent_views4::<
            R,
            $ty0,
            $ty1,
            $ty2,
            $ty3,
            KernelOp<R, Op>,
        >($policy, &$input.0, &$input.1, &$input.2, &$input.3)
    }};
    ($policy:ident, $input:ident; $ty0:ident, $ty1:ident, $ty2:ident, $ty3:ident, $ty4:ident; 0, 1, 2, 3, 4) => {{
        crate::detail::apply::LinearScanApply::adjacent_views5::<
            R,
            $ty0,
            $ty1,
            $ty2,
            $ty3,
            $ty4,
            KernelOp<R, Op>,
        >(
            $policy, &$input.0, &$input.1, &$input.2, &$input.3, &$input.4,
        )
    }};
    ($policy:ident, $input:ident; $ty0:ident, $ty1:ident, $ty2:ident, $ty3:ident, $ty4:ident, $ty5:ident; 0, 1, 2, 3, 4, 5) => {{
        crate::detail::apply::LinearScanApply::adjacent_views6::<
            R,
            $ty0,
            $ty1,
            $ty2,
            $ty3,
            $ty4,
            $ty5,
            KernelOp<R, Op>,
        >(
            $policy, &$input.0, &$input.1, &$input.2, &$input.3, &$input.4, &$input.5,
        )
    }};
    ($policy:ident, $input:ident; $ty0:ident, $ty1:ident, $ty2:ident, $ty3:ident, $ty4:ident, $ty5:ident, $ty6:ident; 0, 1, 2, 3, 4, 5, 6) => {{
        crate::detail::apply::LinearScanApply::adjacent_views7::<
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
            message: "adjacent_difference is not supported for this iterator shape".to_string(),
        })
    }};
}

macro_rules! impl_wide_predicate_selection_body {
    ($policy:ident, $input:ident, $env:ident, $invert:expr; $ty0:ident, $ty1:ident, $ty2:ident, $ty3:ident; 0, 1, 2, 3) => {{
        let dummy4 = crate::detail::primitives::range::indices_mindex($policy, $input.0.len)?;
        let dummy5 = crate::detail::primitives::range::indices_mindex($policy, $input.0.len)?;
        let dummy6 = crate::detail::primitives::range::indices_mindex($policy, $input.0.len)?;
        let dummy4 = crate::detail::device::DeviceColumnView::from_column(&dummy4);
        let dummy5 = crate::detail::device::DeviceColumnView::from_column(&dummy5);
        let dummy6 = crate::detail::device::DeviceColumnView::from_column(&dummy6);
        impl_wide_predicate_selection_body!(
            @launch
            $policy,
            $env,
            $invert,
            crate::detail::api::Tuple4AsTuple7PredicateOp<KernelOp<R, Pred>>,
            ($ty0, $ty1, $ty2, $ty3, u32, u32, u32),
            (&$input.0, &$input.1, &$input.2, &$input.3, &dummy4, &dummy5, &dummy6)
        )
    }};
    ($policy:ident, $input:ident, $env:ident, $invert:expr; $ty0:ident, $ty1:ident, $ty2:ident, $ty3:ident, $ty4:ident; 0, 1, 2, 3, 4) => {{
        let dummy5 = crate::detail::primitives::range::indices_mindex($policy, $input.0.len)?;
        let dummy6 = crate::detail::primitives::range::indices_mindex($policy, $input.0.len)?;
        let dummy5 = crate::detail::device::DeviceColumnView::from_column(&dummy5);
        let dummy6 = crate::detail::device::DeviceColumnView::from_column(&dummy6);
        impl_wide_predicate_selection_body!(
            @launch
            $policy,
            $env,
            $invert,
            crate::detail::api::Tuple5AsTuple7PredicateOp<KernelOp<R, Pred>>,
            ($ty0, $ty1, $ty2, $ty3, $ty4, u32, u32),
            (&$input.0, &$input.1, &$input.2, &$input.3, &$input.4, &dummy5, &dummy6)
        )
    }};
    ($policy:ident, $input:ident, $env:ident, $invert:expr; $ty0:ident, $ty1:ident, $ty2:ident, $ty3:ident, $ty4:ident, $ty5:ident; 0, 1, 2, 3, 4, 5) => {{
        let dummy6 = crate::detail::primitives::range::indices_mindex($policy, $input.0.len)?;
        let dummy6 = crate::detail::device::DeviceColumnView::from_column(&dummy6);
        impl_wide_predicate_selection_body!(
            @launch
            $policy,
            $env,
            $invert,
            crate::detail::api::Tuple6AsTuple7PredicateOp<KernelOp<R, Pred>>,
            ($ty0, $ty1, $ty2, $ty3, $ty4, $ty5, u32),
            (&$input.0, &$input.1, &$input.2, &$input.3, &$input.4, &$input.5, &dummy6)
        )
    }};
    ($policy:ident, $input:ident, $env:ident, $invert:expr; $ty0:ident, $ty1:ident, $ty2:ident, $ty3:ident, $ty4:ident, $ty5:ident, $ty6:ident; 0, 1, 2, 3, 4, 5, 6) => {{
        impl_wide_predicate_selection_body!(
            @launch
            $policy,
            $env,
            $invert,
            KernelOp<R, Pred>,
            ($ty0, $ty1, $ty2, $ty3, $ty4, $ty5, $ty6),
            (&$input.0, &$input.1, &$input.2, &$input.3, &$input.4, &$input.5, &$input.6)
        )
    }};
    (@launch $policy:ident, $env:ident, $invert:expr, $pred:ty, ($ty0:ty, $ty1:ty, $ty2:ty, $ty3:ty, $ty4:ty, $ty5:ty, $ty6:ty), ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr)) => {{
        let len = $a.len;
        let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
        if len == 0 {
            Ok(crate::detail::primitives::select::SelectedRankControl::empty(
                $policy.client(),
            ))
        } else {
            let client = $policy.client();
            let flag = client.empty(len * std::mem::size_of::<u32>());
            let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
            let invert_handle =
                client.create_from_slice(u32::as_bytes(&[if $invert { 1_u32 } else { 0_u32 }]));
            let offsets = [
                u32::try_from($a.offset).map_err(|_| Error::LengthTooLarge { len: $a.offset })?,
                u32::try_from($b.offset).map_err(|_| Error::LengthTooLarge { len: $b.offset })?,
                u32::try_from($c.offset).map_err(|_| Error::LengthTooLarge { len: $c.offset })?,
                u32::try_from($d.offset).map_err(|_| Error::LengthTooLarge { len: $d.offset })?,
                u32::try_from($e.offset).map_err(|_| Error::LengthTooLarge { len: $e.offset })?,
                u32::try_from($f.offset).map_err(|_| Error::LengthTooLarge { len: $f.offset })?,
                u32::try_from($g.offset).map_err(|_| Error::LengthTooLarge { len: $g.offset })?,
            ];
            let offsets_handle = client.create_from_slice(u32::as_bytes(&offsets));
            let block_size = 256_u32;
            let block_count = len.div_ceil(block_size as usize);
            let block_count_u32 =
                u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;
            unsafe {
                crate::kernels::tuple7_view_predicate_flags_kernel::launch_unchecked::<
                    $ty0,
                    $ty1,
                    $ty2,
                    $ty3,
                    $ty4,
                    $ty5,
                    $ty6,
                    $pred,
                    R,
                >(
                    client,
                    CubeCount::Static(block_count_u32, 1, 1),
                    CubeDim::new_1d(block_size),
                    $env,
                    BufferArg::from_raw_parts($a.source.handle.clone(), $a.source.len()),
                    BufferArg::from_raw_parts($b.source.handle.clone(), $b.source.len()),
                    BufferArg::from_raw_parts($c.source.handle.clone(), $c.source.len()),
                    BufferArg::from_raw_parts($d.source.handle.clone(), $d.source.len()),
                    BufferArg::from_raw_parts($e.source.handle.clone(), $e.source.len()),
                    BufferArg::from_raw_parts($f.source.handle.clone(), $f.source.len()),
                    BufferArg::from_raw_parts($g.source.handle.clone(), $g.source.len()),
                    BufferArg::from_raw_parts(offsets_handle.clone(), 7),
                    BufferArg::from_raw_parts(len_handle.clone(), 1),
                    BufferArg::from_raw_parts(invert_handle.clone(), 1),
                    BufferArg::from_raw_parts(flag.clone(), len),
                );
            }
            crate::detail::primitives::select::selected_rank_from_flags($policy, len, len_u32, flag)
        }
    }};
}

macro_rules! impl_wide_binary_predicate_views {
    ($policy:ident, $left:ident, $right:ident, $op:ident, $body:ident; $ty0:ident, $ty1:ident, $ty2:ident, $ty3:ident; 0, 1, 2, 3) => {{
        let left_dummy4 = crate::detail::primitives::range::indices_mindex($policy, $left.0.len)?;
        let left_dummy5 = crate::detail::primitives::range::indices_mindex($policy, $left.0.len)?;
        let left_dummy6 = crate::detail::primitives::range::indices_mindex($policy, $left.0.len)?;
        let right_dummy4 = crate::detail::primitives::range::indices_mindex($policy, $right.0.len)?;
        let right_dummy5 = crate::detail::primitives::range::indices_mindex($policy, $right.0.len)?;
        let right_dummy6 = crate::detail::primitives::range::indices_mindex($policy, $right.0.len)?;
        let left_dummy4 = crate::detail::device::DeviceColumnView::from_column(&left_dummy4);
        let left_dummy5 = crate::detail::device::DeviceColumnView::from_column(&left_dummy5);
        let left_dummy6 = crate::detail::device::DeviceColumnView::from_column(&left_dummy6);
        let right_dummy4 = crate::detail::device::DeviceColumnView::from_column(&right_dummy4);
        let right_dummy5 = crate::detail::device::DeviceColumnView::from_column(&right_dummy5);
        let right_dummy6 = crate::detail::device::DeviceColumnView::from_column(&right_dummy6);
        $body!(
            $policy,
            crate::detail::api::Tuple4AsTuple7BinaryPredicateOp<KernelOp<R, $op>>,
            ($ty0, $ty1, $ty2, $ty3, u32, u32, u32),
            (&$left.0, &$left.1, &$left.2, &$left.3, &left_dummy4, &left_dummy5, &left_dummy6),
            (&$right.0, &$right.1, &$right.2, &$right.3, &right_dummy4, &right_dummy5, &right_dummy6)
        )
    }};
    ($policy:ident, $left:ident, $right:ident, $op:ident, $body:ident; $ty0:ident, $ty1:ident, $ty2:ident, $ty3:ident, $ty4:ident; 0, 1, 2, 3, 4) => {{
        let left_dummy5 = crate::detail::primitives::range::indices_mindex($policy, $left.0.len)?;
        let left_dummy6 = crate::detail::primitives::range::indices_mindex($policy, $left.0.len)?;
        let right_dummy5 = crate::detail::primitives::range::indices_mindex($policy, $right.0.len)?;
        let right_dummy6 = crate::detail::primitives::range::indices_mindex($policy, $right.0.len)?;
        let left_dummy5 = crate::detail::device::DeviceColumnView::from_column(&left_dummy5);
        let left_dummy6 = crate::detail::device::DeviceColumnView::from_column(&left_dummy6);
        let right_dummy5 = crate::detail::device::DeviceColumnView::from_column(&right_dummy5);
        let right_dummy6 = crate::detail::device::DeviceColumnView::from_column(&right_dummy6);
        $body!(
            $policy,
            crate::detail::api::Tuple5AsTuple7BinaryPredicateOp<KernelOp<R, $op>>,
            ($ty0, $ty1, $ty2, $ty3, $ty4, u32, u32),
            (&$left.0, &$left.1, &$left.2, &$left.3, &$left.4, &left_dummy5, &left_dummy6),
            (&$right.0, &$right.1, &$right.2, &$right.3, &$right.4, &right_dummy5, &right_dummy6)
        )
    }};
    ($policy:ident, $left:ident, $right:ident, $op:ident, $body:ident; $ty0:ident, $ty1:ident, $ty2:ident, $ty3:ident, $ty4:ident, $ty5:ident; 0, 1, 2, 3, 4, 5) => {{
        let left_dummy6 = crate::detail::primitives::range::indices_mindex($policy, $left.0.len)?;
        let right_dummy6 = crate::detail::primitives::range::indices_mindex($policy, $right.0.len)?;
        let left_dummy6 = crate::detail::device::DeviceColumnView::from_column(&left_dummy6);
        let right_dummy6 = crate::detail::device::DeviceColumnView::from_column(&right_dummy6);
        $body!(
            $policy,
            crate::detail::api::Tuple6AsTuple7BinaryPredicateOp<KernelOp<R, $op>>,
            ($ty0, $ty1, $ty2, $ty3, $ty4, $ty5, u32),
            (&$left.0, &$left.1, &$left.2, &$left.3, &$left.4, &$left.5, &left_dummy6),
            (&$right.0, &$right.1, &$right.2, &$right.3, &$right.4, &$right.5, &right_dummy6)
        )
    }};
    ($policy:ident, $left:ident, $right:ident, $op:ident, $body:ident; $ty0:ident, $ty1:ident, $ty2:ident, $ty3:ident, $ty4:ident, $ty5:ident, $ty6:ident; 0, 1, 2, 3, 4, 5, 6) => {{
        $body!(
            $policy,
            KernelOp<R, $op>,
            ($ty0, $ty1, $ty2, $ty3, $ty4, $ty5, $ty6),
            (&$left.0, &$left.1, &$left.2, &$left.3, &$left.4, &$left.5, &$left.6),
            (&$right.0, &$right.1, &$right.2, &$right.3, &$right.4, &$right.5, &$right.6)
        )
    }};
}

macro_rules! impl_wide_mismatch_from_views {
    ($policy:ident, $eq:ty, ($ty0:ty, $ty1:ty, $ty2:ty, $ty3:ty, $ty4:ty, $ty5:ty, $ty6:ty), ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr), ($ra:expr, $rb:expr, $rc:expr, $rd:expr, $re:expr, $rf:expr, $rg:expr)) => {{
        let min_len = $a.len.min($ra.len);
        if min_len == 0 {
            if $a.len == $ra.len {
                Ok(None)
            } else {
                Ok(Some(0))
            }
        } else {
            let client = $policy.client();
            let flag = client.empty(min_len * std::mem::size_of::<u32>());
            let len_handle = client.create_from_slice(u32::as_bytes(&[
                u32::try_from(min_len).map_err(|_| Error::LengthTooLarge { len: min_len })?,
            ]));
            let left_offsets = [
                u32::try_from($a.offset).map_err(|_| Error::LengthTooLarge { len: $a.offset })?,
                u32::try_from($b.offset).map_err(|_| Error::LengthTooLarge { len: $b.offset })?,
                u32::try_from($c.offset).map_err(|_| Error::LengthTooLarge { len: $c.offset })?,
                u32::try_from($d.offset).map_err(|_| Error::LengthTooLarge { len: $d.offset })?,
                u32::try_from($e.offset).map_err(|_| Error::LengthTooLarge { len: $e.offset })?,
                u32::try_from($f.offset).map_err(|_| Error::LengthTooLarge { len: $f.offset })?,
                u32::try_from($g.offset).map_err(|_| Error::LengthTooLarge { len: $g.offset })?,
            ];
            let right_offsets = [
                u32::try_from($ra.offset).map_err(|_| Error::LengthTooLarge { len: $ra.offset })?,
                u32::try_from($rb.offset).map_err(|_| Error::LengthTooLarge { len: $rb.offset })?,
                u32::try_from($rc.offset).map_err(|_| Error::LengthTooLarge { len: $rc.offset })?,
                u32::try_from($rd.offset).map_err(|_| Error::LengthTooLarge { len: $rd.offset })?,
                u32::try_from($re.offset).map_err(|_| Error::LengthTooLarge { len: $re.offset })?,
                u32::try_from($rf.offset).map_err(|_| Error::LengthTooLarge { len: $rf.offset })?,
                u32::try_from($rg.offset).map_err(|_| Error::LengthTooLarge { len: $rg.offset })?,
            ];
            let left_offsets = client.create_from_slice(u32::as_bytes(&left_offsets));
            let right_offsets = client.create_from_slice(u32::as_bytes(&right_offsets));
            let block_size = 256_u32;
            let block_count = min_len.div_ceil(block_size as usize);
            let block_count_u32 =
                u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;
            unsafe {
                crate::kernels::tuple7_view_mismatch_flags_kernel::launch_unchecked::<
                    $ty0,
                    $ty1,
                    $ty2,
                    $ty3,
                    $ty4,
                    $ty5,
                    $ty6,
                    $eq,
                    R,
                >(
                    client,
                    CubeCount::Static(block_count_u32, 1, 1),
                    CubeDim::new_1d(block_size),
                    BufferArg::from_raw_parts($a.source.handle.clone(), $a.source.len()),
                    BufferArg::from_raw_parts($b.source.handle.clone(), $b.source.len()),
                    BufferArg::from_raw_parts($c.source.handle.clone(), $c.source.len()),
                    BufferArg::from_raw_parts($d.source.handle.clone(), $d.source.len()),
                    BufferArg::from_raw_parts($e.source.handle.clone(), $e.source.len()),
                    BufferArg::from_raw_parts($f.source.handle.clone(), $f.source.len()),
                    BufferArg::from_raw_parts($g.source.handle.clone(), $g.source.len()),
                    BufferArg::from_raw_parts(left_offsets.clone(), 7),
                    BufferArg::from_raw_parts($ra.source.handle.clone(), $ra.source.len()),
                    BufferArg::from_raw_parts($rb.source.handle.clone(), $rb.source.len()),
                    BufferArg::from_raw_parts($rc.source.handle.clone(), $rc.source.len()),
                    BufferArg::from_raw_parts($rd.source.handle.clone(), $rd.source.len()),
                    BufferArg::from_raw_parts($re.source.handle.clone(), $re.source.len()),
                    BufferArg::from_raw_parts($rf.source.handle.clone(), $rf.source.len()),
                    BufferArg::from_raw_parts($rg.source.handle.clone(), $rg.source.len()),
                    BufferArg::from_raw_parts(right_offsets.clone(), 7),
                    BufferArg::from_raw_parts(len_handle.clone(), 1),
                    BufferArg::from_raw_parts(flag.clone(), min_len),
                );
            }
            let search =
                crate::detail::control::SearchControl::from_flags(flag, min_len, min_len);
            if let Some(index) = crate::detail::apply::QueryApply::first_flag($policy, search)? {
                Ok(Some(index))
            } else if $a.len == $ra.len {
                Ok(None)
            } else {
                Ok(Some(mindex_from_usize(min_len)?))
            }
        }
    }};
}

macro_rules! impl_wide_adjacent_find_from_views {
    ($policy:ident, $eq:ty, ($ty0:ty, $ty1:ty, $ty2:ty, $ty3:ty, $ty4:ty, $ty5:ty, $ty6:ty), ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr), ($ra:expr, $rb:expr, $rc:expr, $rd:expr, $re:expr, $rf:expr, $rg:expr)) => {{
        let _ = ($ra, $rb, $rc, $rd, $re, $rf, $rg);
        let len = $a.len;
        if len < 2 {
            Ok(None)
        } else {
            let client = $policy.client();
            let flag = client.empty(len * std::mem::size_of::<u32>());
            let len_handle = client.create_from_slice(u32::as_bytes(&[
                u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?,
            ]));
            let offsets = [
                u32::try_from($a.offset).map_err(|_| Error::LengthTooLarge { len: $a.offset })?,
                u32::try_from($b.offset).map_err(|_| Error::LengthTooLarge { len: $b.offset })?,
                u32::try_from($c.offset).map_err(|_| Error::LengthTooLarge { len: $c.offset })?,
                u32::try_from($d.offset).map_err(|_| Error::LengthTooLarge { len: $d.offset })?,
                u32::try_from($e.offset).map_err(|_| Error::LengthTooLarge { len: $e.offset })?,
                u32::try_from($f.offset).map_err(|_| Error::LengthTooLarge { len: $f.offset })?,
                u32::try_from($g.offset).map_err(|_| Error::LengthTooLarge { len: $g.offset })?,
            ];
            let offsets = client.create_from_slice(u32::as_bytes(&offsets));
            let block_size = 256_u32;
            let block_count = len.div_ceil(block_size as usize);
            let block_count_u32 =
                u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;
            unsafe {
                crate::kernels::tuple7_view_adjacent_find_flags_kernel::launch_unchecked::<
                    $ty0,
                    $ty1,
                    $ty2,
                    $ty3,
                    $ty4,
                    $ty5,
                    $ty6,
                    $eq,
                    R,
                >(
                    client,
                    CubeCount::Static(block_count_u32, 1, 1),
                    CubeDim::new_1d(block_size),
                    BufferArg::from_raw_parts($a.source.handle.clone(), $a.source.len()),
                    BufferArg::from_raw_parts($b.source.handle.clone(), $b.source.len()),
                    BufferArg::from_raw_parts($c.source.handle.clone(), $c.source.len()),
                    BufferArg::from_raw_parts($d.source.handle.clone(), $d.source.len()),
                    BufferArg::from_raw_parts($e.source.handle.clone(), $e.source.len()),
                    BufferArg::from_raw_parts($f.source.handle.clone(), $f.source.len()),
                    BufferArg::from_raw_parts($g.source.handle.clone(), $g.source.len()),
                    BufferArg::from_raw_parts(offsets.clone(), 7),
                    BufferArg::from_raw_parts(len_handle.clone(), 1),
                    BufferArg::from_raw_parts(flag.clone(), len),
                );
            }
            let search = crate::detail::control::SearchControl::from_flags(flag, len, len - 1);
            crate::detail::apply::QueryApply::first_flag($policy, search)
        }
    }};
}

macro_rules! impl_wide_find_first_of_from_views {
    ($policy:ident, $eq:ty, ($ty0:ty, $ty1:ty, $ty2:ty, $ty3:ty, $ty4:ty, $ty5:ty, $ty6:ty), ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr), ($ra:expr, $rb:expr, $rc:expr, $rd:expr, $re:expr, $rf:expr, $rg:expr)) => {{
        let len = $a.len;
        let needle_len = $ra.len;
        if len == 0 || needle_len == 0 {
            Ok(None)
        } else {
            let client = $policy.client();
            let flag = client.empty(len * std::mem::size_of::<u32>());
            let needle_len_handle = client.create_from_slice(u32::as_bytes(&[
                u32::try_from(needle_len).map_err(|_| Error::LengthTooLarge { len: needle_len })?,
            ]));
            let input_offsets = [
                u32::try_from($a.offset).map_err(|_| Error::LengthTooLarge { len: $a.offset })?,
                u32::try_from($b.offset).map_err(|_| Error::LengthTooLarge { len: $b.offset })?,
                u32::try_from($c.offset).map_err(|_| Error::LengthTooLarge { len: $c.offset })?,
                u32::try_from($d.offset).map_err(|_| Error::LengthTooLarge { len: $d.offset })?,
                u32::try_from($e.offset).map_err(|_| Error::LengthTooLarge { len: $e.offset })?,
                u32::try_from($f.offset).map_err(|_| Error::LengthTooLarge { len: $f.offset })?,
                u32::try_from($g.offset).map_err(|_| Error::LengthTooLarge { len: $g.offset })?,
            ];
            let needle_offsets = [
                u32::try_from($ra.offset).map_err(|_| Error::LengthTooLarge { len: $ra.offset })?,
                u32::try_from($rb.offset).map_err(|_| Error::LengthTooLarge { len: $rb.offset })?,
                u32::try_from($rc.offset).map_err(|_| Error::LengthTooLarge { len: $rc.offset })?,
                u32::try_from($rd.offset).map_err(|_| Error::LengthTooLarge { len: $rd.offset })?,
                u32::try_from($re.offset).map_err(|_| Error::LengthTooLarge { len: $re.offset })?,
                u32::try_from($rf.offset).map_err(|_| Error::LengthTooLarge { len: $rf.offset })?,
                u32::try_from($rg.offset).map_err(|_| Error::LengthTooLarge { len: $rg.offset })?,
            ];
            let input_offsets = client.create_from_slice(u32::as_bytes(&input_offsets));
            let needle_offsets = client.create_from_slice(u32::as_bytes(&needle_offsets));
            let block_size = 256_u32;
            let block_count = len.div_ceil(block_size as usize);
            let block_count_u32 =
                u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;
            unsafe {
                crate::kernels::tuple7_view_find_first_of_flags_kernel::launch_unchecked::<
                    $ty0,
                    $ty1,
                    $ty2,
                    $ty3,
                    $ty4,
                    $ty5,
                    $ty6,
                    $eq,
                    R,
                >(
                    client,
                    CubeCount::Static(block_count_u32, 1, 1),
                    CubeDim::new_1d(block_size),
                    BufferArg::from_raw_parts($a.source.handle.clone(), $a.source.len()),
                    BufferArg::from_raw_parts($b.source.handle.clone(), $b.source.len()),
                    BufferArg::from_raw_parts($c.source.handle.clone(), $c.source.len()),
                    BufferArg::from_raw_parts($d.source.handle.clone(), $d.source.len()),
                    BufferArg::from_raw_parts($e.source.handle.clone(), $e.source.len()),
                    BufferArg::from_raw_parts($f.source.handle.clone(), $f.source.len()),
                    BufferArg::from_raw_parts($g.source.handle.clone(), $g.source.len()),
                    BufferArg::from_raw_parts(input_offsets.clone(), 7),
                    BufferArg::from_raw_parts($ra.source.handle.clone(), $ra.source.len()),
                    BufferArg::from_raw_parts($rb.source.handle.clone(), $rb.source.len()),
                    BufferArg::from_raw_parts($rc.source.handle.clone(), $rc.source.len()),
                    BufferArg::from_raw_parts($rd.source.handle.clone(), $rd.source.len()),
                    BufferArg::from_raw_parts($re.source.handle.clone(), $re.source.len()),
                    BufferArg::from_raw_parts($rf.source.handle.clone(), $rf.source.len()),
                    BufferArg::from_raw_parts($rg.source.handle.clone(), $rg.source.len()),
                    BufferArg::from_raw_parts(needle_offsets.clone(), 7),
                    BufferArg::from_raw_parts(needle_len_handle.clone(), 1),
                    BufferArg::from_raw_parts(flag.clone(), len),
                );
            }
            let search = crate::detail::control::SearchControl::from_flags(flag, len, len);
            crate::detail::apply::QueryApply::first_flag($policy, search)
        }
    }};
}

macro_rules! impl_wide_bound_many_from_views {
    ($policy:ident, $less:ty, $kernel:ident, ($ty0:ty, $ty1:ty, $ty2:ty, $ty3:ty, $ty4:ty, $ty5:ty, $ty6:ty), ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr), ($ra:expr, $rb:expr, $rc:expr, $rd:expr, $re:expr, $rf:expr, $rg:expr)) => {{
        let source_len = $a.len;
        let value_len = $ra.len;
        if value_len == 0 {
            Ok($policy.empty_device_vec())
        } else if source_len == 0 {
            $policy.device_filled(value_len, 0u32)
        } else {
            let client = $policy.client();
            let source_offsets = [
                u32::try_from($a.offset).map_err(|_| Error::LengthTooLarge { len: $a.offset })?,
                u32::try_from($b.offset).map_err(|_| Error::LengthTooLarge { len: $b.offset })?,
                u32::try_from($c.offset).map_err(|_| Error::LengthTooLarge { len: $c.offset })?,
                u32::try_from($d.offset).map_err(|_| Error::LengthTooLarge { len: $d.offset })?,
                u32::try_from($e.offset).map_err(|_| Error::LengthTooLarge { len: $e.offset })?,
                u32::try_from($f.offset).map_err(|_| Error::LengthTooLarge { len: $f.offset })?,
                u32::try_from($g.offset).map_err(|_| Error::LengthTooLarge { len: $g.offset })?,
            ];
            let value_offsets = [
                u32::try_from($ra.offset).map_err(|_| Error::LengthTooLarge { len: $ra.offset })?,
                u32::try_from($rb.offset).map_err(|_| Error::LengthTooLarge { len: $rb.offset })?,
                u32::try_from($rc.offset).map_err(|_| Error::LengthTooLarge { len: $rc.offset })?,
                u32::try_from($rd.offset).map_err(|_| Error::LengthTooLarge { len: $rd.offset })?,
                u32::try_from($re.offset).map_err(|_| Error::LengthTooLarge { len: $re.offset })?,
                u32::try_from($rf.offset).map_err(|_| Error::LengthTooLarge { len: $rf.offset })?,
                u32::try_from($rg.offset).map_err(|_| Error::LengthTooLarge { len: $rg.offset })?,
            ];
            let source_len_u32 =
                u32::try_from(source_len).map_err(|_| Error::LengthTooLarge { len: source_len })?;
            let value_len_u32 =
                u32::try_from(value_len).map_err(|_| Error::LengthTooLarge { len: value_len })?;
            let source_offsets = client.create_from_slice(u32::as_bytes(&source_offsets));
            let value_offsets = client.create_from_slice(u32::as_bytes(&value_offsets));
            let source_len_handle = client.create_from_slice(u32::as_bytes(&[source_len_u32]));
            let value_len_handle = client.create_from_slice(u32::as_bytes(&[value_len_u32]));
            let output_handle = client.empty(value_len * std::mem::size_of::<u32>());
            let block_size = 256_u32;
            let block_count = value_len.div_ceil(block_size as usize);
            let block_count_u32 =
                u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;
            unsafe {
                crate::kernels::$kernel::launch_unchecked::<
                    $ty0,
                    $ty1,
                    $ty2,
                    $ty3,
                    $ty4,
                    $ty5,
                    $ty6,
                    $less,
                    R,
                >(
                    client,
                    CubeCount::Static(block_count_u32, 1, 1),
                    CubeDim::new_1d(block_size),
                    BufferArg::from_raw_parts($a.source.handle.clone(), $a.source.len()),
                    BufferArg::from_raw_parts($b.source.handle.clone(), $b.source.len()),
                    BufferArg::from_raw_parts($c.source.handle.clone(), $c.source.len()),
                    BufferArg::from_raw_parts($d.source.handle.clone(), $d.source.len()),
                    BufferArg::from_raw_parts($e.source.handle.clone(), $e.source.len()),
                    BufferArg::from_raw_parts($f.source.handle.clone(), $f.source.len()),
                    BufferArg::from_raw_parts($g.source.handle.clone(), $g.source.len()),
                    BufferArg::from_raw_parts(source_offsets.clone(), 7),
                    BufferArg::from_raw_parts($ra.source.handle.clone(), $ra.source.len()),
                    BufferArg::from_raw_parts($rb.source.handle.clone(), $rb.source.len()),
                    BufferArg::from_raw_parts($rc.source.handle.clone(), $rc.source.len()),
                    BufferArg::from_raw_parts($rd.source.handle.clone(), $rd.source.len()),
                    BufferArg::from_raw_parts($re.source.handle.clone(), $re.source.len()),
                    BufferArg::from_raw_parts($rf.source.handle.clone(), $rf.source.len()),
                    BufferArg::from_raw_parts($rg.source.handle.clone(), $rg.source.len()),
                    BufferArg::from_raw_parts(value_offsets.clone(), 7),
                    BufferArg::from_raw_parts(source_len_handle.clone(), 1),
                    BufferArg::from_raw_parts(value_len_handle.clone(), 1),
                    BufferArg::from_raw_parts(output_handle.clone(), value_len),
                );
            }
            Ok(crate::detail::DeviceVec::from_handle($policy.id(), output_handle, value_len))
        }
    }};
}

macro_rules! impl_wide_lower_bound_many_from_views {
    ($policy:ident, $less:ty, ($ty0:ty, $ty1:ty, $ty2:ty, $ty3:ty, $ty4:ty, $ty5:ty, $ty6:ty), ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr), ($ra:expr, $rb:expr, $rc:expr, $rd:expr, $re:expr, $rf:expr, $rg:expr)) => {
        impl_wide_bound_many_from_views!(
            $policy,
            $less,
            tuple7_view_lower_bound_many_kernel,
            ($ty0, $ty1, $ty2, $ty3, $ty4, $ty5, $ty6),
            ($a, $b, $c, $d, $e, $f, $g),
            ($ra, $rb, $rc, $rd, $re, $rf, $rg)
        )
    };
}

macro_rules! impl_wide_upper_bound_many_from_views {
    ($policy:ident, $less:ty, ($ty0:ty, $ty1:ty, $ty2:ty, $ty3:ty, $ty4:ty, $ty5:ty, $ty6:ty), ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr), ($ra:expr, $rb:expr, $rc:expr, $rd:expr, $re:expr, $rf:expr, $rg:expr)) => {
        impl_wide_bound_many_from_views!(
            $policy,
            $less,
            tuple7_view_upper_bound_many_kernel,
            ($ty0, $ty1, $ty2, $ty3, $ty4, $ty5, $ty6),
            ($a, $b, $c, $d, $e, $f, $g),
            ($ra, $rb, $rc, $rd, $re, $rf, $rg)
        )
    };
}

macro_rules! impl_wide_lexicographical_compare_from_views {
    ($policy:ident, $less:ty, ($ty0:ty, $ty1:ty, $ty2:ty, $ty3:ty, $ty4:ty, $ty5:ty, $ty6:ty), ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr), ($ra:expr, $rb:expr, $rc:expr, $rd:expr, $re:expr, $rf:expr, $rg:expr)) => {{
        let left_len = $a.len;
        let right_len = $ra.len;
        let min_len = left_len.min(right_len);
        if min_len == 0 {
            Ok(left_len < right_len)
        } else {
            let client = $policy.client();
            let left_offsets = [
                u32::try_from($a.offset).map_err(|_| Error::LengthTooLarge { len: $a.offset })?,
                u32::try_from($b.offset).map_err(|_| Error::LengthTooLarge { len: $b.offset })?,
                u32::try_from($c.offset).map_err(|_| Error::LengthTooLarge { len: $c.offset })?,
                u32::try_from($d.offset).map_err(|_| Error::LengthTooLarge { len: $d.offset })?,
                u32::try_from($e.offset).map_err(|_| Error::LengthTooLarge { len: $e.offset })?,
                u32::try_from($f.offset).map_err(|_| Error::LengthTooLarge { len: $f.offset })?,
                u32::try_from($g.offset).map_err(|_| Error::LengthTooLarge { len: $g.offset })?,
            ];
            let right_offsets = [
                u32::try_from($ra.offset).map_err(|_| Error::LengthTooLarge { len: $ra.offset })?,
                u32::try_from($rb.offset).map_err(|_| Error::LengthTooLarge { len: $rb.offset })?,
                u32::try_from($rc.offset).map_err(|_| Error::LengthTooLarge { len: $rc.offset })?,
                u32::try_from($rd.offset).map_err(|_| Error::LengthTooLarge { len: $rd.offset })?,
                u32::try_from($re.offset).map_err(|_| Error::LengthTooLarge { len: $re.offset })?,
                u32::try_from($rf.offset).map_err(|_| Error::LengthTooLarge { len: $rf.offset })?,
                u32::try_from($rg.offset).map_err(|_| Error::LengthTooLarge { len: $rg.offset })?,
            ];
            let left_offsets = client.create_from_slice(u32::as_bytes(&left_offsets));
            let right_offsets = client.create_from_slice(u32::as_bytes(&right_offsets));
            let len_handle = client.create_from_slice(u32::as_bytes(&[
                u32::try_from(min_len).map_err(|_| Error::LengthTooLarge { len: min_len })?,
            ]));
            let flag = client.empty(min_len * std::mem::size_of::<u32>());
            let block_size = 256_u32;
            let block_count = min_len.div_ceil(block_size as usize);
            let block_count_u32 =
                u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;
            unsafe {
                crate::kernels::tuple7_view_lexicographical_diff_flags_kernel::launch_unchecked::<
                    $ty0,
                    $ty1,
                    $ty2,
                    $ty3,
                    $ty4,
                    $ty5,
                    $ty6,
                    $less,
                    R,
                >(
                    client,
                    CubeCount::Static(block_count_u32, 1, 1),
                    CubeDim::new_1d(block_size),
                    BufferArg::from_raw_parts($a.source.handle.clone(), $a.source.len()),
                    BufferArg::from_raw_parts($b.source.handle.clone(), $b.source.len()),
                    BufferArg::from_raw_parts($c.source.handle.clone(), $c.source.len()),
                    BufferArg::from_raw_parts($d.source.handle.clone(), $d.source.len()),
                    BufferArg::from_raw_parts($e.source.handle.clone(), $e.source.len()),
                    BufferArg::from_raw_parts($f.source.handle.clone(), $f.source.len()),
                    BufferArg::from_raw_parts($g.source.handle.clone(), $g.source.len()),
                    BufferArg::from_raw_parts(left_offsets.clone(), 7),
                    BufferArg::from_raw_parts($ra.source.handle.clone(), $ra.source.len()),
                    BufferArg::from_raw_parts($rb.source.handle.clone(), $rb.source.len()),
                    BufferArg::from_raw_parts($rc.source.handle.clone(), $rc.source.len()),
                    BufferArg::from_raw_parts($rd.source.handle.clone(), $rd.source.len()),
                    BufferArg::from_raw_parts($re.source.handle.clone(), $re.source.len()),
                    BufferArg::from_raw_parts($rf.source.handle.clone(), $rf.source.len()),
                    BufferArg::from_raw_parts($rg.source.handle.clone(), $rg.source.len()),
                    BufferArg::from_raw_parts(right_offsets.clone(), 7),
                    BufferArg::from_raw_parts(len_handle.clone(), 1),
                    BufferArg::from_raw_parts(flag.clone(), min_len),
                );
            }
            let search =
                crate::detail::control::SearchControl::from_flags(flag, min_len, min_len);
            let Some(index) = crate::detail::apply::QueryApply::first_flag($policy, search)? else {
                return Ok(left_len < right_len);
            };
            let index_handle = client.create_from_slice(u32::as_bytes(&[index as u32]));
            let output_handle = client.empty(std::mem::size_of::<u32>());
            unsafe {
                crate::kernels::tuple7_view_lexicographical_compare_at_kernel::launch_unchecked::<
                    $ty0,
                    $ty1,
                    $ty2,
                    $ty3,
                    $ty4,
                    $ty5,
                    $ty6,
                    $less,
                    R,
                >(
                    client,
                    CubeCount::new_single(),
                    CubeDim::new_1d(1),
                    BufferArg::from_raw_parts($a.source.handle.clone(), $a.source.len()),
                    BufferArg::from_raw_parts($b.source.handle.clone(), $b.source.len()),
                    BufferArg::from_raw_parts($c.source.handle.clone(), $c.source.len()),
                    BufferArg::from_raw_parts($d.source.handle.clone(), $d.source.len()),
                    BufferArg::from_raw_parts($e.source.handle.clone(), $e.source.len()),
                    BufferArg::from_raw_parts($f.source.handle.clone(), $f.source.len()),
                    BufferArg::from_raw_parts($g.source.handle.clone(), $g.source.len()),
                    BufferArg::from_raw_parts(left_offsets.clone(), 7),
                    BufferArg::from_raw_parts($ra.source.handle.clone(), $ra.source.len()),
                    BufferArg::from_raw_parts($rb.source.handle.clone(), $rb.source.len()),
                    BufferArg::from_raw_parts($rc.source.handle.clone(), $rc.source.len()),
                    BufferArg::from_raw_parts($rd.source.handle.clone(), $rd.source.len()),
                    BufferArg::from_raw_parts($re.source.handle.clone(), $re.source.len()),
                    BufferArg::from_raw_parts($rf.source.handle.clone(), $rf.source.len()),
                    BufferArg::from_raw_parts($rg.source.handle.clone(), $rg.source.len()),
                    BufferArg::from_raw_parts(right_offsets.clone(), 7),
                    BufferArg::from_raw_parts(index_handle.clone(), 1),
                    BufferArg::from_raw_parts(output_handle.clone(), 1),
                );
            }
            Ok(crate::detail::primitives::scan::read_u32_scalar::<R>(client, output_handle)? != 0)
        }
    }};
}

macro_rules! impl_wide_set_membership_flags_from_views {
    ($policy:ident, $less:ty, $keep_intersection:expr, ($ty0:ty, $ty1:ty, $ty2:ty, $ty3:ty, $ty4:ty, $ty5:ty, $ty6:ty), ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr), ($ra:expr, $rb:expr, $rc:expr, $rd:expr, $re:expr, $rf:expr, $rg:expr)) => {{
        let len = $a.len;
        if len == 0 {
            Ok($policy.empty_handle())
        } else {
            let client = $policy.client();
            let candidate_offsets = [
                u32::try_from($a.offset).map_err(|_| Error::LengthTooLarge { len: $a.offset })?,
                u32::try_from($b.offset).map_err(|_| Error::LengthTooLarge { len: $b.offset })?,
                u32::try_from($c.offset).map_err(|_| Error::LengthTooLarge { len: $c.offset })?,
                u32::try_from($d.offset).map_err(|_| Error::LengthTooLarge { len: $d.offset })?,
                u32::try_from($e.offset).map_err(|_| Error::LengthTooLarge { len: $e.offset })?,
                u32::try_from($f.offset).map_err(|_| Error::LengthTooLarge { len: $f.offset })?,
                u32::try_from($g.offset).map_err(|_| Error::LengthTooLarge { len: $g.offset })?,
            ];
            let source_offsets = [
                u32::try_from($ra.offset).map_err(|_| Error::LengthTooLarge { len: $ra.offset })?,
                u32::try_from($rb.offset).map_err(|_| Error::LengthTooLarge { len: $rb.offset })?,
                u32::try_from($rc.offset).map_err(|_| Error::LengthTooLarge { len: $rc.offset })?,
                u32::try_from($rd.offset).map_err(|_| Error::LengthTooLarge { len: $rd.offset })?,
                u32::try_from($re.offset).map_err(|_| Error::LengthTooLarge { len: $re.offset })?,
                u32::try_from($rf.offset).map_err(|_| Error::LengthTooLarge { len: $rf.offset })?,
                u32::try_from($rg.offset).map_err(|_| Error::LengthTooLarge { len: $rg.offset })?,
            ];
            let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
            let source_len = $ra.len;
            let source_len_u32 =
                u32::try_from(source_len).map_err(|_| Error::LengthTooLarge { len: source_len })?;
            let candidate_offsets = client.create_from_slice(u32::as_bytes(&candidate_offsets));
            let source_offsets = client.create_from_slice(u32::as_bytes(&source_offsets));
            let len_handle = client.create_from_slice(u32::as_bytes(&[len_u32]));
            let source_len_handle = client.create_from_slice(u32::as_bytes(&[source_len_u32]));
            let keep_intersection_handle = client.create_from_slice(u32::as_bytes(&[
                if $keep_intersection { 1_u32 } else { 0_u32 },
            ]));
            let flag = client.empty(len * std::mem::size_of::<u32>());
            let block_size = 256_u32;
            let block_count = len.div_ceil(block_size as usize);
            let block_count_u32 =
                u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;
            unsafe {
                crate::kernels::tuple7_view_set_membership_flags_kernel::launch_unchecked::<
                    $ty0,
                    $ty1,
                    $ty2,
                    $ty3,
                    $ty4,
                    $ty5,
                    $ty6,
                    $less,
                    R,
                >(
                    client,
                    CubeCount::Static(block_count_u32, 1, 1),
                    CubeDim::new_1d(block_size),
                    BufferArg::from_raw_parts($a.source.handle.clone(), $a.source.len()),
                    BufferArg::from_raw_parts($b.source.handle.clone(), $b.source.len()),
                    BufferArg::from_raw_parts($c.source.handle.clone(), $c.source.len()),
                    BufferArg::from_raw_parts($d.source.handle.clone(), $d.source.len()),
                    BufferArg::from_raw_parts($e.source.handle.clone(), $e.source.len()),
                    BufferArg::from_raw_parts($f.source.handle.clone(), $f.source.len()),
                    BufferArg::from_raw_parts($g.source.handle.clone(), $g.source.len()),
                    BufferArg::from_raw_parts(candidate_offsets.clone(), 7),
                    BufferArg::from_raw_parts($ra.source.handle.clone(), $ra.source.len()),
                    BufferArg::from_raw_parts($rb.source.handle.clone(), $rb.source.len()),
                    BufferArg::from_raw_parts($rc.source.handle.clone(), $rc.source.len()),
                    BufferArg::from_raw_parts($rd.source.handle.clone(), $rd.source.len()),
                    BufferArg::from_raw_parts($re.source.handle.clone(), $re.source.len()),
                    BufferArg::from_raw_parts($rf.source.handle.clone(), $rf.source.len()),
                    BufferArg::from_raw_parts($rg.source.handle.clone(), $rg.source.len()),
                    BufferArg::from_raw_parts(source_offsets.clone(), 7),
                    BufferArg::from_raw_parts(len_handle.clone(), 1),
                    BufferArg::from_raw_parts(source_len_handle.clone(), 1),
                    BufferArg::from_raw_parts(keep_intersection_handle.clone(), 1),
                    BufferArg::from_raw_parts(flag.clone(), len),
                );
            }
            Ok(flag)
        }
    }};
}

macro_rules! impl_wide_set_difference_flags_from_views {
    ($policy:ident, $less:ty, ($($ty:ty),+), ($($left:expr),+), ($($right:expr),+)) => {
        impl_wide_set_membership_flags_from_views!(
            $policy,
            $less,
            false,
            ($($ty),+),
            ($($left),+),
            ($($right),+)
        )
    };
}

macro_rules! impl_wide_set_intersection_flags_from_views {
    ($policy:ident, $less:ty, ($($ty:ty),+), ($($left:expr),+), ($($right:expr),+)) => {
        impl_wide_set_membership_flags_from_views!(
            $policy,
            $less,
            true,
            ($($ty),+),
            ($($left),+),
            ($($right),+)
        )
    };
}

macro_rules! impl_tuple_inclusive_scan_by_three_key_values_body {
    ($policy:ident, $input:ident, $control:ident; 0, 1) => {{
        crate::detail::read::inclusive_scan_by_flags_two::<_, _, KernelOp<R, Op>>(
            $policy, &$input.0, &$input.1, &$control,
        )
        .map(|inner| (inner.left, inner.right))
    }};
    ($policy:ident, $input:ident, $control:ident; 0, 1, 2) => {{
        crate::detail::read::inclusive_scan_by_flags_three::<_, _, _, KernelOp<R, Op>>(
            $policy, &$input.0, &$input.1, &$input.2, &$control,
        )
        .map(|inner| (inner.first, inner.second, inner.third))
    }};
}

macro_rules! impl_tuple_inclusive_scan_by_two_key_values_body {
    ($policy:ident, $input:ident, $control:ident; 0, 1) => {{
        crate::detail::read::inclusive_scan_by_flags_two::<_, _, KernelOp<R, Op>>(
            $policy, &$input.0, &$input.1, &$control,
        )
        .map(|inner| (inner.left, inner.right))
    }};
    ($policy:ident, $input:ident, $control:ident; 0, 1, 2) => {{
        crate::detail::read::inclusive_scan_by_flags_three::<_, _, _, KernelOp<R, Op>>(
            $policy, &$input.0, &$input.1, &$input.2, &$control,
        )
        .map(|inner| (inner.first, inner.second, inner.third))
    }};
}

macro_rules! impl_tuple_exclusive_scan_by_three_key_values_body {
    ($policy:ident, $input:ident, $control:ident, $init:ident; 0, 1) => {{
        crate::detail::read::exclusive_scan_by_flags_two::<_, _, KernelOp<R, Op>>(
            $policy, &$input.0, &$input.1, &$control, $init,
        )
        .map(|inner| (inner.left, inner.right))
    }};
    ($policy:ident, $input:ident, $control:ident, $init:ident; 0, 1, 2) => {{
        crate::detail::read::exclusive_scan_by_flags_three::<_, _, _, KernelOp<R, Op>>(
            $policy, &$input.0, &$input.1, &$input.2, &$control, $init,
        )
        .map(|inner| (inner.first, inner.second, inner.third))
    }};
}

macro_rules! impl_tuple_exclusive_scan_by_two_key_values_body {
    ($policy:ident, $input:ident, $control:ident, $init:ident; 0, 1) => {{
        crate::detail::read::exclusive_scan_by_flags_two::<_, _, KernelOp<R, Op>>(
            $policy, &$input.0, &$input.1, &$control, $init,
        )
        .map(|inner| (inner.left, inner.right))
    }};
    ($policy:ident, $input:ident, $control:ident, $init:ident; 0, 1, 2) => {{
        crate::detail::read::exclusive_scan_by_flags_three::<_, _, _, KernelOp<R, Op>>(
            $policy, &$input.0, &$input.1, &$input.2, &$control, $init,
        )
        .map(|inner| (inner.first, inner.second, inner.third))
    }};
}

macro_rules! impl_tuple_reduce_by_three_key_values_body {
    ($policy:ident, $input:ident, $init:ident, $first_key:ident, $second_key:ident, $third_key:ident, $head_flags:ident, $end_flags:ident, $len_u32:ident, $control:ident; $ty0:ident, $ty1:ident; 0, 1) => {{
        let value_selected_rank = crate::detail::primitives::select::selected_rank_from_flags(
            $policy,
            $first_key.len,
            $len_u32,
            $end_flags.clone(),
        )?;
        let value_count =
            crate::detail::primitives::select::selected_count($policy, &value_selected_rank)?;
        let segment = crate::detail::control::SegmentControl::from_head_end_flags(
            $control.head_flags.clone(),
            $end_flags,
            $first_key.len,
            $len_u32,
        );
        let reduce_control = crate::detail::control::ReduceByKeyControl::from_segment(
            segment,
            value_selected_rank,
            value_count,
        );
        let payload_apply = crate::detail::apply::SelectedPayloadApply::new(
            &reduce_control.output_selection,
            reduce_control.output_count,
        );
        let key_inner = (
            payload_apply.apply_expr($policy, &$first_key)?,
            payload_apply.apply_expr($policy, &$second_key)?,
            payload_apply.apply_expr($policy, &$third_key)?,
        );
        let reduce_apply = crate::detail::apply::SegmentedReduceApply::new(&reduce_control);
        let values = reduce_apply
            .apply_expr2::<_, _, KernelOp<R, Op>>($policy, &$input.0, &$input.1, $init)?;
        Ok((key_inner, (values.left, values.right)))
    }};
    ($policy:ident, $input:ident, $init:ident, $first_key:ident, $second_key:ident, $third_key:ident, $head_flags:ident, $end_flags:ident, $len_u32:ident, $control:ident; $ty0:ident, $ty1:ident, $ty2:ident; 0, 1, 2) => {{
        let value_selected_rank = crate::detail::primitives::select::selected_rank_from_flags(
            $policy,
            $first_key.len,
            $len_u32,
            $end_flags.clone(),
        )?;
        let value_count =
            crate::detail::primitives::select::selected_count($policy, &value_selected_rank)?;
        let segment = crate::detail::control::SegmentControl::from_head_end_flags(
            $control.head_flags.clone(),
            $end_flags,
            $first_key.len,
            $len_u32,
        );
        let reduce_control = crate::detail::control::ReduceByKeyControl::from_segment(
            segment,
            value_selected_rank,
            value_count,
        );
        let payload_apply = crate::detail::apply::SelectedPayloadApply::new(
            &reduce_control.output_selection,
            reduce_control.output_count,
        );
        let key_inner = (
            payload_apply.apply_expr($policy, &$first_key)?,
            payload_apply.apply_expr($policy, &$second_key)?,
            payload_apply.apply_expr($policy, &$third_key)?,
        );
        let reduce_apply = crate::detail::apply::SegmentedReduceApply::new(&reduce_control);
        let values = reduce_apply.apply_expr3::<_, _, _, KernelOp<R, Op>>(
            $policy, &$input.0, &$input.1, &$input.2, $init,
        )?;
        Ok((key_inner, (values.first, values.second, values.third)))
    }};
}

macro_rules! impl_tuple_reduce_by_two_key_values_body {
    ($policy:ident, $input:ident, $init:ident, $first_key:ident, $second_key:ident, $end_flags:ident, $len_u32:ident, $control:ident; $ty0:ident, $ty1:ident; 0, 1) => {{
        let value_selected_rank = crate::detail::primitives::select::selected_rank_from_flags(
            $policy,
            $first_key.len,
            $len_u32,
            $end_flags.clone(),
        )?;
        let value_count =
            crate::detail::primitives::select::selected_count($policy, &value_selected_rank)?;
        let segment = crate::detail::control::SegmentControl::from_head_end_flags(
            $control.head_flags.clone(),
            $end_flags,
            $first_key.len,
            $len_u32,
        );
        let reduce_control = crate::detail::control::ReduceByKeyControl::from_segment(
            segment,
            value_selected_rank,
            value_count,
        );
        let payload_apply = crate::detail::apply::SelectedPayloadApply::new(
            &reduce_control.output_selection,
            reduce_control.output_count,
        );
        let key_inner = (
            payload_apply.apply_expr($policy, &$first_key)?,
            payload_apply.apply_expr($policy, &$second_key)?,
        );
        let reduce_apply = crate::detail::apply::SegmentedReduceApply::new(&reduce_control);
        let values = reduce_apply
            .apply_expr2::<_, _, KernelOp<R, Op>>($policy, &$input.0, &$input.1, $init)?;
        Ok((key_inner, (values.left, values.right)))
    }};
    ($policy:ident, $input:ident, $init:ident, $first_key:ident, $second_key:ident, $end_flags:ident, $len_u32:ident, $control:ident; $ty0:ident, $ty1:ident, $ty2:ident; 0, 1, 2) => {{
        let value_selected_rank = crate::detail::primitives::select::selected_rank_from_flags(
            $policy,
            $first_key.len,
            $len_u32,
            $end_flags.clone(),
        )?;
        let value_count =
            crate::detail::primitives::select::selected_count($policy, &value_selected_rank)?;
        let segment = crate::detail::control::SegmentControl::from_head_end_flags(
            $control.head_flags.clone(),
            $end_flags,
            $first_key.len,
            $len_u32,
        );
        let reduce_control = crate::detail::control::ReduceByKeyControl::from_segment(
            segment,
            value_selected_rank,
            value_count,
        );
        let payload_apply = crate::detail::apply::SelectedPayloadApply::new(
            &reduce_control.output_selection,
            reduce_control.output_count,
        );
        let key_inner = (
            payload_apply.apply_expr($policy, &$first_key)?,
            payload_apply.apply_expr($policy, &$second_key)?,
        );
        let reduce_apply = crate::detail::apply::SegmentedReduceApply::new(&reduce_control);
        let values = reduce_apply.apply_expr3::<_, _, _, KernelOp<R, Op>>(
            $policy, &$input.0, &$input.1, &$input.2, $init,
        )?;
        Ok((key_inner, (values.first, values.second, values.third)))
    }};
}

macro_rules! impl_wide_inclusive_scan_by_single_key_values_body {
    ($policy:ident, $keys:ident, $input:ident; $ty0:ident, $ty1:ident, $ty2:ident, $ty3:ident; 0, 1, 2, 3) => {{
        let dummy4 = crate::detail::primitives::range::indices_mindex($policy, $input.0.len)?;
        let dummy5 = crate::detail::primitives::range::indices_mindex($policy, $input.0.len)?;
        let dummy6 = crate::detail::primitives::range::indices_mindex($policy, $input.0.len)?;
        let dummy4 = crate::detail::device::DeviceColumnView::from_column(&dummy4);
        let dummy5 = crate::detail::device::DeviceColumnView::from_column(&dummy5);
        let dummy6 = crate::detail::device::DeviceColumnView::from_column(&dummy6);
        let control =
            <(crate::detail::device::DeviceColumnView<R, K>,) as crate::detail::read::KernelScanByKeyKeys<
                KernelOp<R, KeyEq>,
            >>::scan_by_key_control(($keys,), $policy)?;
        let (a, b, c, d, _, _, _) =
            crate::detail::read::inclusive_scan_by_flags_seven_views::<
                R,
                $ty0,
                $ty1,
                $ty2,
                $ty3,
                u32,
                u32,
                u32,
                crate::detail::api::Tuple4AsTuple7BinaryOp<KernelOp<R, Op>>,
            >(
                $policy, &$input.0, &$input.1, &$input.2, &$input.3, &dummy4, &dummy5, &dummy6,
                &control,
            )?;
        Ok((a, b, c, d))
    }};
    ($policy:ident, $keys:ident, $input:ident; $ty0:ident, $ty1:ident, $ty2:ident, $ty3:ident, $ty4:ident; 0, 1, 2, 3, 4) => {{
        let dummy5 = crate::detail::primitives::range::indices_mindex($policy, $input.0.len)?;
        let dummy6 = crate::detail::primitives::range::indices_mindex($policy, $input.0.len)?;
        let dummy5 = crate::detail::device::DeviceColumnView::from_column(&dummy5);
        let dummy6 = crate::detail::device::DeviceColumnView::from_column(&dummy6);
        let control =
            <(crate::detail::device::DeviceColumnView<R, K>,) as crate::detail::read::KernelScanByKeyKeys<
                KernelOp<R, KeyEq>,
            >>::scan_by_key_control(($keys,), $policy)?;
        let (a, b, c, d, e, _, _) =
            crate::detail::read::inclusive_scan_by_flags_seven_views::<
                R,
                $ty0,
                $ty1,
                $ty2,
                $ty3,
                $ty4,
                u32,
                u32,
                crate::detail::api::Tuple5AsTuple7BinaryOp<KernelOp<R, Op>>,
            >(
                $policy, &$input.0, &$input.1, &$input.2, &$input.3, &$input.4, &dummy5, &dummy6,
                &control,
            )?;
        Ok((a, b, c, d, e))
    }};
    ($policy:ident, $keys:ident, $input:ident; $ty0:ident, $ty1:ident, $ty2:ident, $ty3:ident, $ty4:ident, $ty5:ident; 0, 1, 2, 3, 4, 5) => {{
        let dummy6 = crate::detail::primitives::range::indices_mindex($policy, $input.0.len)?;
        let dummy6 = crate::detail::device::DeviceColumnView::from_column(&dummy6);
        let control =
            <(crate::detail::device::DeviceColumnView<R, K>,) as crate::detail::read::KernelScanByKeyKeys<
                KernelOp<R, KeyEq>,
            >>::scan_by_key_control(($keys,), $policy)?;
        let (a, b, c, d, e, f, _) =
            crate::detail::read::inclusive_scan_by_flags_seven_views::<
                R,
                $ty0,
                $ty1,
                $ty2,
                $ty3,
                $ty4,
                $ty5,
                u32,
                crate::detail::api::Tuple6AsTuple7BinaryOp<KernelOp<R, Op>>,
            >(
                $policy, &$input.0, &$input.1, &$input.2, &$input.3, &$input.4, &$input.5, &dummy6,
                &control,
            )?;
        Ok((a, b, c, d, e, f))
    }};
    ($policy:ident, $keys:ident, $input:ident; $ty0:ident, $ty1:ident, $ty2:ident, $ty3:ident, $ty4:ident, $ty5:ident, $ty6:ident; 0, 1, 2, 3, 4, 5, 6) => {{
        let control =
            <(crate::detail::device::DeviceColumnView<R, K>,) as crate::detail::read::KernelScanByKeyKeys<
                KernelOp<R, KeyEq>,
            >>::scan_by_key_control(($keys,), $policy)?;
        crate::detail::read::inclusive_scan_by_flags_seven_views::<
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
            &control,
        )
    }};
}

macro_rules! impl_wide_exclusive_scan_by_single_key_values_body {
    ($policy:ident, $keys:ident, $input:ident, $init:ident; $ty0:ident, $ty1:ident, $ty2:ident, $ty3:ident; 0, 1, 2, 3) => {{
        let dummy4 = crate::detail::primitives::range::indices_mindex($policy, $input.0.len)?;
        let dummy5 = crate::detail::primitives::range::indices_mindex($policy, $input.0.len)?;
        let dummy6 = crate::detail::primitives::range::indices_mindex($policy, $input.0.len)?;
        let dummy4 = crate::detail::device::DeviceColumnView::from_column(&dummy4);
        let dummy5 = crate::detail::device::DeviceColumnView::from_column(&dummy5);
        let dummy6 = crate::detail::device::DeviceColumnView::from_column(&dummy6);
        let control =
            <(crate::detail::device::DeviceColumnView<R, K>,) as crate::detail::read::KernelScanByKeyKeys<
                KernelOp<R, KeyEq>,
            >>::scan_by_key_control(($keys,), $policy)?;
        let (a, b, c, d, _, _, _) =
            crate::detail::read::exclusive_scan_by_flags_seven_views::<
                R, $ty0, $ty1, $ty2, $ty3, u32, u32, u32,
                crate::detail::api::Tuple4AsTuple7BinaryOp<KernelOp<R, Op>>,
            >(
                $policy, &$input.0, &$input.1, &$input.2, &$input.3, &dummy4, &dummy5, &dummy6,
                &control, ($init.0, $init.1, $init.2, $init.3, 0, 0, 0),
            )?;
        Ok((a, b, c, d))
    }};
    ($policy:ident, $keys:ident, $input:ident, $init:ident; $ty0:ident, $ty1:ident, $ty2:ident, $ty3:ident, $ty4:ident; 0, 1, 2, 3, 4) => {{
        let dummy5 = crate::detail::primitives::range::indices_mindex($policy, $input.0.len)?;
        let dummy6 = crate::detail::primitives::range::indices_mindex($policy, $input.0.len)?;
        let dummy5 = crate::detail::device::DeviceColumnView::from_column(&dummy5);
        let dummy6 = crate::detail::device::DeviceColumnView::from_column(&dummy6);
        let control =
            <(crate::detail::device::DeviceColumnView<R, K>,) as crate::detail::read::KernelScanByKeyKeys<
                KernelOp<R, KeyEq>,
            >>::scan_by_key_control(($keys,), $policy)?;
        let (a, b, c, d, e, _, _) =
            crate::detail::read::exclusive_scan_by_flags_seven_views::<
                R, $ty0, $ty1, $ty2, $ty3, $ty4, u32, u32,
                crate::detail::api::Tuple5AsTuple7BinaryOp<KernelOp<R, Op>>,
            >(
                $policy, &$input.0, &$input.1, &$input.2, &$input.3, &$input.4, &dummy5, &dummy6,
                &control, ($init.0, $init.1, $init.2, $init.3, $init.4, 0, 0),
            )?;
        Ok((a, b, c, d, e))
    }};
    ($policy:ident, $keys:ident, $input:ident, $init:ident; $ty0:ident, $ty1:ident, $ty2:ident, $ty3:ident, $ty4:ident, $ty5:ident; 0, 1, 2, 3, 4, 5) => {{
        let dummy6 = crate::detail::primitives::range::indices_mindex($policy, $input.0.len)?;
        let dummy6 = crate::detail::device::DeviceColumnView::from_column(&dummy6);
        let control =
            <(crate::detail::device::DeviceColumnView<R, K>,) as crate::detail::read::KernelScanByKeyKeys<
                KernelOp<R, KeyEq>,
            >>::scan_by_key_control(($keys,), $policy)?;
        let (a, b, c, d, e, f, _) =
            crate::detail::read::exclusive_scan_by_flags_seven_views::<
                R, $ty0, $ty1, $ty2, $ty3, $ty4, $ty5, u32,
                crate::detail::api::Tuple6AsTuple7BinaryOp<KernelOp<R, Op>>,
            >(
                $policy, &$input.0, &$input.1, &$input.2, &$input.3, &$input.4, &$input.5, &dummy6,
                &control, ($init.0, $init.1, $init.2, $init.3, $init.4, $init.5, 0),
            )?;
        Ok((a, b, c, d, e, f))
    }};
    ($policy:ident, $keys:ident, $input:ident, $init:ident; $ty0:ident, $ty1:ident, $ty2:ident, $ty3:ident, $ty4:ident, $ty5:ident, $ty6:ident; 0, 1, 2, 3, 4, 5, 6) => {{
        let control =
            <(crate::detail::device::DeviceColumnView<R, K>,) as crate::detail::read::KernelScanByKeyKeys<
                KernelOp<R, KeyEq>,
            >>::scan_by_key_control(($keys,), $policy)?;
        crate::detail::read::exclusive_scan_by_flags_seven_views::<
            R, $ty0, $ty1, $ty2, $ty3, $ty4, $ty5, $ty6,
            KernelOp<R, Op>,
        >(
            $policy, &$input.0, &$input.1, &$input.2, &$input.3, &$input.4, &$input.5, &$input.6,
            &control, $init,
        )
    }};
}

macro_rules! impl_wide_reduce_by_single_key_values_body {
    ($policy:ident, $keys:ident, $input:ident, $init:ident; $ty0:ident, $ty1:ident, $ty2:ident, $ty3:ident; 0, 1, 2, 3) => {{
        let dummy4 = crate::detail::primitives::range::indices_mindex($policy, $input.0.len)?;
        let dummy5 = crate::detail::primitives::range::indices_mindex($policy, $input.0.len)?;
        let dummy6 = crate::detail::primitives::range::indices_mindex($policy, $input.0.len)?;
        let dummy4 = crate::detail::device::DeviceColumnView::from_column(&dummy4);
        let dummy5 = crate::detail::device::DeviceColumnView::from_column(&dummy5);
        let dummy6 = crate::detail::device::DeviceColumnView::from_column(&dummy6);
        let (key_inner, control) =
            <(crate::detail::device::DeviceColumnView<R, K>,) as crate::detail::read::KernelReduceByKeyKeys<
                KernelOp<R, KeyEq>,
            >>::reduce_by_key_control(($keys,), $policy)?;
        let (a, b, c, d, _, _, _) = impl_wide_reduce_by_single_key_tuple7_values_body!(
            $policy,
            $input.0,
            $input.1,
            $input.2,
            $input.3,
            dummy4,
            dummy5,
            dummy6,
            ($init.0, $init.1, $init.2, $init.3, 0, 0, 0),
            control;
            $ty0, $ty1, $ty2, $ty3, u32, u32, u32;
            crate::detail::api::Tuple4AsTuple7BinaryOp<KernelOp<R, Op>>
        )?;
        Ok(((key_inner.source,), (a, b, c, d)))
    }};
    ($policy:ident, $keys:ident, $input:ident, $init:ident; $ty0:ident, $ty1:ident, $ty2:ident, $ty3:ident, $ty4:ident; 0, 1, 2, 3, 4) => {{
        let dummy5 = crate::detail::primitives::range::indices_mindex($policy, $input.0.len)?;
        let dummy6 = crate::detail::primitives::range::indices_mindex($policy, $input.0.len)?;
        let dummy5 = crate::detail::device::DeviceColumnView::from_column(&dummy5);
        let dummy6 = crate::detail::device::DeviceColumnView::from_column(&dummy6);
        let (key_inner, control) =
            <(crate::detail::device::DeviceColumnView<R, K>,) as crate::detail::read::KernelReduceByKeyKeys<
                KernelOp<R, KeyEq>,
            >>::reduce_by_key_control(($keys,), $policy)?;
        let (a, b, c, d, e, _, _) = impl_wide_reduce_by_single_key_tuple7_values_body!(
            $policy,
            $input.0,
            $input.1,
            $input.2,
            $input.3,
            $input.4,
            dummy5,
            dummy6,
            ($init.0, $init.1, $init.2, $init.3, $init.4, 0, 0),
            control;
            $ty0, $ty1, $ty2, $ty3, $ty4, u32, u32;
            crate::detail::api::Tuple5AsTuple7BinaryOp<KernelOp<R, Op>>
        )?;
        Ok(((key_inner.source,), (a, b, c, d, e)))
    }};
    ($policy:ident, $keys:ident, $input:ident, $init:ident; $ty0:ident, $ty1:ident, $ty2:ident, $ty3:ident, $ty4:ident, $ty5:ident; 0, 1, 2, 3, 4, 5) => {{
        let dummy6 = crate::detail::primitives::range::indices_mindex($policy, $input.0.len)?;
        let dummy6 = crate::detail::device::DeviceColumnView::from_column(&dummy6);
        let (key_inner, control) =
            <(crate::detail::device::DeviceColumnView<R, K>,) as crate::detail::read::KernelReduceByKeyKeys<
                KernelOp<R, KeyEq>,
            >>::reduce_by_key_control(($keys,), $policy)?;
        let (a, b, c, d, e, f, _) = impl_wide_reduce_by_single_key_tuple7_values_body!(
            $policy,
            $input.0,
            $input.1,
            $input.2,
            $input.3,
            $input.4,
            $input.5,
            dummy6,
            ($init.0, $init.1, $init.2, $init.3, $init.4, $init.5, 0),
            control;
            $ty0, $ty1, $ty2, $ty3, $ty4, $ty5, u32;
            crate::detail::api::Tuple6AsTuple7BinaryOp<KernelOp<R, Op>>
        )?;
        Ok(((key_inner.source,), (a, b, c, d, e, f)))
    }};
    ($policy:ident, $keys:ident, $input:ident, $init:ident; $ty0:ident, $ty1:ident, $ty2:ident, $ty3:ident, $ty4:ident, $ty5:ident, $ty6:ident; 0, 1, 2, 3, 4, 5, 6) => {{
        let (key_inner, control) =
            <(crate::detail::device::DeviceColumnView<R, K>,) as crate::detail::read::KernelReduceByKeyKeys<
                KernelOp<R, KeyEq>,
            >>::reduce_by_key_control(($keys,), $policy)?;
        let values = impl_wide_reduce_by_single_key_tuple7_values_body!(
            $policy,
            $input.0,
            $input.1,
            $input.2,
            $input.3,
            $input.4,
            $input.5,
            $input.6,
            $init,
            control;
            $ty0, $ty1, $ty2, $ty3, $ty4, $ty5, $ty6;
            KernelOp<R, Op>
        )?;
        Ok(((key_inner.source,), values))
    }};
}

macro_rules! impl_wide_reduce_by_single_key_tuple7_values_body {
    ($policy:ident, $a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr, $init:expr, $control:ident; $ty0:ident, $ty1:ident, $ty2:ident, $ty3:ident, $ty4:ident, $ty5:ident, $ty6:ident; $op:ty) => {{
        let reduce_apply = crate::detail::apply::SegmentedReduceApply::new(&$control);
        reduce_apply.apply_views7::<$ty0, $ty1, $ty2, $ty3, $ty4, $ty5, $ty6, $op>(
            $policy, &$a, &$b, &$c, &$d, &$e, &$f, &$g, $init,
        )
    }};
}

macro_rules! impl_wide_materialize_inner {
    ($policy:ident, $input:ident; $( $idx:tt : $tmp:ident ),+) => {{
        $(
            let ($tmp,) = crate::detail::reverse($policy, ($input.$idx,))?;
        )+
        ($($tmp,)+)
    }};
}

macro_rules! impl_wide_sort_or_materialize_body {
    ($policy:ident, $input:ident; 0: $a:ident, 1: $b:ident, 2: $c:ident, 3: $d:ident) => {
        impl_wide_sort_dispatch_body!($policy, $input; 0, 1, 2, 3)
    };
    ($policy:ident, $input:ident; 0: $a:ident, 1: $b:ident, 2: $c:ident, 3: $d:ident, 4: $e:ident) => {
        impl_wide_sort_dispatch_body!($policy, $input; 0, 1, 2, 3, 4)
    };
    ($policy:ident, $input:ident; 0: $a:ident, 1: $b:ident, 2: $c:ident, 3: $d:ident, 4: $e:ident, 5: $f:ident) => {
        impl_wide_sort_dispatch_body!($policy, $input; 0, 1, 2, 3, 4, 5)
    };
    ($policy:ident, $input:ident; 0: $a:ident, 1: $b:ident, 2: $c:ident, 3: $d:ident, 4: $e:ident, 5: $f:ident, 6: $g:ident) => {
        impl_wide_sort_dispatch_body!($policy, $input; 0, 1, 2, 3, 4, 5, 6)
    };
    ($policy:ident, $input:ident; $( $idx:tt : $tmp:ident ),+) => {
        Ok(impl_wide_materialize_inner!($policy, $input; $( $idx: $tmp ),+))
    };
}

macro_rules! impl_wide_scan_or_materialize_body {
    ($policy:ident, $input:ident; $($ty:ident),+; 0: $a:ident, 1: $b:ident, 2: $c:ident, 3: $d:ident) => {
        impl_wide_inclusive_scan_dispatch_body!($policy, $input; $($ty),+; 0, 1, 2, 3)
    };
    ($policy:ident, $input:ident; $($ty:ident),+; 0: $a:ident, 1: $b:ident, 2: $c:ident, 3: $d:ident, 4: $e:ident) => {
        impl_wide_inclusive_scan_dispatch_body!($policy, $input; $($ty),+; 0, 1, 2, 3, 4)
    };
    ($policy:ident, $input:ident; $($ty:ident),+; 0: $a:ident, 1: $b:ident, 2: $c:ident, 3: $d:ident, 4: $e:ident, 5: $f:ident) => {
        impl_wide_inclusive_scan_dispatch_body!($policy, $input; $($ty),+; 0, 1, 2, 3, 4, 5)
    };
    ($policy:ident, $input:ident; $($ty:ident),+; 0: $a:ident, 1: $b:ident, 2: $c:ident, 3: $d:ident, 4: $e:ident, 5: $f:ident, 6: $g:ident) => {
        impl_wide_inclusive_scan_dispatch_body!($policy, $input; $($ty),+; 0, 1, 2, 3, 4, 5, 6)
    };
    ($policy:ident, $input:ident; $($ty:ident),+; $( $idx:tt : $tmp:ident ),+) => {
        Ok(impl_wide_materialize_inner!($policy, $input; $( $idx: $tmp ),+))
    };
}

macro_rules! impl_wide_exclusive_scan_or_materialize_body {
    ($policy:ident, $input:ident, $init:ident; $($ty:ident),+; 0: $a:ident, 1: $b:ident, 2: $c:ident, 3: $d:ident) => {
        impl_wide_exclusive_scan_dispatch_body!($policy, $input, $init; $($ty),+; 0, 1, 2, 3)
    };
    ($policy:ident, $input:ident, $init:ident; $($ty:ident),+; 0: $a:ident, 1: $b:ident, 2: $c:ident, 3: $d:ident, 4: $e:ident) => {
        impl_wide_exclusive_scan_dispatch_body!($policy, $input, $init; $($ty),+; 0, 1, 2, 3, 4)
    };
    ($policy:ident, $input:ident, $init:ident; $($ty:ident),+; 0: $a:ident, 1: $b:ident, 2: $c:ident, 3: $d:ident, 4: $e:ident, 5: $f:ident) => {
        impl_wide_exclusive_scan_dispatch_body!($policy, $input, $init; $($ty),+; 0, 1, 2, 3, 4, 5)
    };
    ($policy:ident, $input:ident, $init:ident; $($ty:ident),+; 0: $a:ident, 1: $b:ident, 2: $c:ident, 3: $d:ident, 4: $e:ident, 5: $f:ident, 6: $g:ident) => {
        impl_wide_exclusive_scan_dispatch_body!($policy, $input, $init; $($ty),+; 0, 1, 2, 3, 4, 5, 6)
    };
    ($policy:ident, $input:ident, $init:ident; $($ty:ident),+; $( $idx:tt : $tmp:ident ),+) => {{
        let _ = $init;
        Ok(impl_wide_materialize_inner!($policy, $input; $( $idx: $tmp ),+))
    }};
}

macro_rules! impl_wide_reduce_or_init_body {
    ($policy:ident, $input:ident, $init:ident; $($ty:ident),+; 0: $a:ident, 1: $b:ident, 2: $c:ident, 3: $d:ident) => {
        impl_wide_reduce_dispatch_body!($policy, $input, $init; $($ty),+; 0, 1, 2, 3)
    };
    ($policy:ident, $input:ident, $init:ident; $($ty:ident),+; 0: $a:ident, 1: $b:ident, 2: $c:ident, 3: $d:ident, 4: $e:ident) => {
        impl_wide_reduce_dispatch_body!($policy, $input, $init; $($ty),+; 0, 1, 2, 3, 4)
    };
    ($policy:ident, $input:ident, $init:ident; $($ty:ident),+; 0: $a:ident, 1: $b:ident, 2: $c:ident, 3: $d:ident, 4: $e:ident, 5: $f:ident) => {
        impl_wide_reduce_dispatch_body!($policy, $input, $init; $($ty),+; 0, 1, 2, 3, 4, 5)
    };
    ($policy:ident, $input:ident, $init:ident; $($ty:ident),+; 0: $a:ident, 1: $b:ident, 2: $c:ident, 3: $d:ident, 4: $e:ident, 5: $f:ident, 6: $g:ident) => {
        impl_wide_reduce_dispatch_body!($policy, $input, $init; $($ty),+; 0, 1, 2, 3, 4, 5, 6)
    };
    ($policy:ident, $input:ident, $init:ident; $($ty:ident),+; $( $idx:tt : $tmp:ident ),+) => {{
        let _ = ($policy, $input);
        Ok($init)
    }};
}

macro_rules! impl_wide_sort_by_single_key_or_materialize_body {
    ($policy:ident, $keys:ident, $less:ident, $input:ident; 0: $a:ident, 1: $b:ident, 2: $c:ident, 3: $d:ident) => {
        impl_wide_sort_by_single_key_dispatch_body!($policy, $keys, $less, $input; 0, 1, 2, 3)
    };
    ($policy:ident, $keys:ident, $less:ident, $input:ident; 0: $a:ident, 1: $b:ident, 2: $c:ident, 3: $d:ident, 4: $e:ident) => {
        impl_wide_sort_by_single_key_dispatch_body!($policy, $keys, $less, $input; 0, 1, 2, 3, 4)
    };
    ($policy:ident, $keys:ident, $less:ident, $input:ident; 0: $a:ident, 1: $b:ident, 2: $c:ident, 3: $d:ident, 4: $e:ident, 5: $f:ident) => {
        impl_wide_sort_by_single_key_dispatch_body!($policy, $keys, $less, $input; 0, 1, 2, 3, 4, 5)
    };
    ($policy:ident, $keys:ident, $less:ident, $input:ident; 0: $a:ident, 1: $b:ident, 2: $c:ident, 3: $d:ident, 4: $e:ident, 5: $f:ident, 6: $g:ident) => {
        impl_wide_sort_by_single_key_dispatch_body!($policy, $keys, $less, $input; 0, 1, 2, 3, 4, 5, 6)
    };
    ($policy:ident, $keys:ident, $less:ident, $input:ident; $( $idx:tt : $tmp:ident ),+) => {{
        let _ = $less;
        let (key_inner,) = crate::detail::reverse($policy, ($keys,))?;
        let value_inner = impl_wide_materialize_inner!($policy, $input; $( $idx: $tmp ),+);
        Ok(((key_inner,), value_inner))
    }};
}

macro_rules! impl_wide_unique_inner_or_materialize_body {
    ($policy:ident, $input:ident; 0: $a:ident, 1: $b:ident, 2: $c:ident, 3: $d:ident) => {{
        let flags: cubecl::server::Handle =
            impl_wide_unique_dispatch_body!($policy, $input; 0, 1, 2, 3)?;
        let selected_rank = crate::detail::primitives::select::selected_rank_from_flags(
            $policy,
            $input.0.len,
            u32::try_from($input.0.len).map_err(|_| Error::LengthTooLarge { len: $input.0.len })?,
            flags,
        )?;
        let count = crate::detail::primitives::select::selected_count($policy, &selected_rank)?;
        let payload_apply = crate::detail::apply::SelectedPayloadApply::new(&selected_rank, count);
        Ok((
            payload_apply.apply_expr($policy, &$input.0)?,
            payload_apply.apply_expr($policy, &$input.1)?,
            payload_apply.apply_expr($policy, &$input.2)?,
            payload_apply.apply_expr($policy, &$input.3)?,
        ))
    }};
    ($policy:ident, $input:ident; 0: $a:ident, 1: $b:ident, 2: $c:ident, 3: $d:ident, 4: $e:ident) => {{
        let flags: cubecl::server::Handle =
            impl_wide_unique_dispatch_body!($policy, $input; 0, 1, 2, 3, 4)?;
        let selected_rank = crate::detail::primitives::select::selected_rank_from_flags(
            $policy,
            $input.0.len,
            u32::try_from($input.0.len).map_err(|_| Error::LengthTooLarge { len: $input.0.len })?,
            flags,
        )?;
        let count = crate::detail::primitives::select::selected_count($policy, &selected_rank)?;
        let payload_apply = crate::detail::apply::SelectedPayloadApply::new(&selected_rank, count);
        Ok((
            payload_apply.apply_expr($policy, &$input.0)?,
            payload_apply.apply_expr($policy, &$input.1)?,
            payload_apply.apply_expr($policy, &$input.2)?,
            payload_apply.apply_expr($policy, &$input.3)?,
            payload_apply.apply_expr($policy, &$input.4)?,
        ))
    }};
    ($policy:ident, $input:ident; 0: $a:ident, 1: $b:ident, 2: $c:ident, 3: $d:ident, 4: $e:ident, 5: $f:ident) => {{
        let flags: cubecl::server::Handle =
            impl_wide_unique_dispatch_body!($policy, $input; 0, 1, 2, 3, 4, 5)?;
        let selected_rank = crate::detail::primitives::select::selected_rank_from_flags(
            $policy,
            $input.0.len,
            u32::try_from($input.0.len).map_err(|_| Error::LengthTooLarge { len: $input.0.len })?,
            flags,
        )?;
        let count = crate::detail::primitives::select::selected_count($policy, &selected_rank)?;
        let payload_apply = crate::detail::apply::SelectedPayloadApply::new(&selected_rank, count);
        Ok((
            payload_apply.apply_expr($policy, &$input.0)?,
            payload_apply.apply_expr($policy, &$input.1)?,
            payload_apply.apply_expr($policy, &$input.2)?,
            payload_apply.apply_expr($policy, &$input.3)?,
            payload_apply.apply_expr($policy, &$input.4)?,
            payload_apply.apply_expr($policy, &$input.5)?,
        ))
    }};
    ($policy:ident, $input:ident; 0: $a:ident, 1: $b:ident, 2: $c:ident, 3: $d:ident, 4: $e:ident, 5: $f:ident, 6: $g:ident) => {{
        let flags: cubecl::server::Handle =
            impl_wide_unique_dispatch_body!($policy, $input; 0, 1, 2, 3, 4, 5, 6)?;
        let selected_rank = crate::detail::primitives::select::selected_rank_from_flags(
            $policy,
            $input.0.len,
            u32::try_from($input.0.len).map_err(|_| Error::LengthTooLarge { len: $input.0.len })?,
            flags,
        )?;
        let count = crate::detail::primitives::select::selected_count($policy, &selected_rank)?;
        let payload_apply = crate::detail::apply::SelectedPayloadApply::new(&selected_rank, count);
        Ok((
            payload_apply.apply_expr($policy, &$input.0)?,
            payload_apply.apply_expr($policy, &$input.1)?,
            payload_apply.apply_expr($policy, &$input.2)?,
            payload_apply.apply_expr($policy, &$input.3)?,
            payload_apply.apply_expr($policy, &$input.4)?,
            payload_apply.apply_expr($policy, &$input.5)?,
            payload_apply.apply_expr($policy, &$input.6)?,
        ))
    }};
    ($policy:ident, $input:ident; $( $idx:tt : $tmp:ident ),+) => {
        Ok(impl_wide_materialize_inner!($policy, $input; $( $idx: $tmp ),+))
    };
}

macro_rules! impl_miter_soa {
    ($name:ident; $( $ty:ident : $idx:tt : $tmp:ident ),+ => $transform:ident) => {
        impl<'a, R, $( $ty ),+> MIter<R> for $name<$( crate::runtime::DeviceSlice<'a, R, $ty> ),+>
        where
            R: Runtime,
            $( $ty: MStorageElement + 'static, )+
            ($( $ty, )+): MAlloc<R, Inner = ($( crate::detail::DeviceVec<R, $ty>, )+),
                View = ($( crate::detail::device::DeviceColumnView<R, $ty>, )+),
            >,
        {
            type Item = ($( $ty, )+);
            type Inner = ($( crate::detail::device::DeviceColumnView<R, $ty>, )+);

            fn len(&self) -> MIndex {
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

            fn into_view_with_policy(
                self,
                policy: &crate::detail::CubePolicy<R>,
            ) -> Result<<Self::Item as MAlloc<R>>::View, Error> {
                let _ = policy;
                Ok(($( self.$idx.column_view(), )+))
            }
        }

        impl<'a, R, $( $ty ),+> sealed::MIterDispatch<R> for $name<$( crate::runtime::DeviceSlice<'a, R, $ty> ),+>
        where
            R: Runtime,
            $( $ty: MStorageElement + 'static, )+
            ($( $ty, )+): MAlloc<R, Inner = ($( crate::detail::DeviceVec<R, $ty>, )+),
                View = ($( crate::detail::device::DeviceColumnView<R, $ty>, )+),
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
                env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
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
                    env,
                )?;
                output.write_from_inner(policy, inner)
            }

            fn map_dispatch<Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                op: Op,
                env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<Output, Error>
            where
                Output: StorageFromInner<R>,
                Op: op::UnaryOp<R, <Self as MIter<R>>::Item, Output = Output::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = <Output::Item as sealed::MItemDispatch<R>>::$transform(
                    policy,
                    $( input.$idx, )+
                    op,
                    env,
                )?;
                Ok(array_from_inner::<R, Output::Item, Output>(inner))
            }

            fn transform_where_dispatch<Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                op: Op,
                env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
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
                    env,
                )?;
                output.write_where_from_inner(policy, inner, stencil)
            }

            fn reverse_dispatch<Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
            ) -> Result<Output, Error>
            where
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = crate::detail::sort(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<R, Less>::new())?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn reverse_into_dispatch<Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                output: Output,
            ) -> Result<(), Error>
            where
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = crate::detail::reverse(policy, impl_miter_view!(input; $( $idx ),+))?;
                output.write_from_inner(policy, inner)
            }

            fn sort_into_dispatch<Less, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _less: Less,
                output: Output,
            ) -> Result<(), Error>
            where
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = crate::detail::sort(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    KernelOp::<R, Less>::new(),
                )?;
                output.write_from_inner(policy, inner)
            }

            fn sort_by_single_key_dispatch<K, Less, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                keys: crate::detail::device::DeviceColumnView<R, K>,
                _less: Less,
            ) -> Result<(KeyOutput, ValueOutput), Error>
            where
                K: MStorageElement + 'static,
                Less: op::BinaryPredicateOp<R, (K,)>,
                KeyOutput: StorageFromInner<R, Item = (K,)>,
                ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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

            fn sort_by_single_key_into_dispatch<K, Less, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                keys: crate::detail::device::DeviceColumnView<R, K>,
                _less: Less,
                key_output: KeyOutput,
                value_output: ValueOutput,
            ) -> Result<(), Error>
            where
                K: MStorageElement + 'static,
                Less: op::BinaryPredicateOp<R, (K,)>,
                KeyOutput: MIterMut<R, Item = (K,)>,
                ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let values = self.into_inner_with_policy(policy)?;
                let values = impl_miter_view!(values; $( $idx ),+);
                let (key_inner, value_inner) =
                    crate::detail::sort_by_key(policy, (keys,), (values,), KernelOp::<R, Less>::new())?;
                key_output.write_from_inner(policy, key_inner)?;
                value_output.write_from_inner(policy, value_inner)
            }

            fn unique_by_single_key_dispatch<K, Eq, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                keys: crate::detail::device::DeviceColumnView<R, K>,
                _eq: Eq,
            ) -> Result<(KeyOutput, ValueOutput), Error>
            where
                K: MStorageElement + 'static,
                Eq: op::BinaryPredicateOp<R, (K,)>,
                KeyOutput: StorageFromInner<R, Item = (K,)>,
                ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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

            fn unique_by_single_key_into_dispatch<K, Eq, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                keys: crate::detail::device::DeviceColumnView<R, K>,
                _eq: Eq,
                key_output: KeyOutput,
                value_output: ValueOutput,
            ) -> Result<MIndex, Error>
            where
                K: MStorageElement + 'static,
                Eq: op::BinaryPredicateOp<R, (K,)>,
                KeyOutput: MIterMut<R, Item = (K,)>,
                ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let values = self.into_inner_with_policy(policy)?;
                let values = impl_miter_view!(values; $( $idx ),+);
                let (key_inner, value_inner) =
                    crate::detail::unique_by_key(policy, (keys,), (values,), KernelOp::<R, Eq>::new())?;
                let len = mindex_from_usize(key_inner.0.len())?;
                key_output.write_prefix_from_inner(policy, key_inner)?;
                value_output.write_prefix_from_inner(policy, value_inner)?;
                Ok(len)
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
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                K3: MStorageElement + 'static,
                Less: op::BinaryPredicateOp<R, (K1, K2, K3)>,
                KeyOutput: StorageFromInner<R, Item = (K1, K2, K3)>,
                ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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

            fn sort_by_three_key_into_dispatch<K1, K2, K3, Less, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                first_key: crate::detail::device::DeviceColumnView<R, K1>,
                second_key: crate::detail::device::DeviceColumnView<R, K2>,
                third_key: crate::detail::device::DeviceColumnView<R, K3>,
                _less: Less,
                key_output: KeyOutput,
                value_output: ValueOutput,
            ) -> Result<(), Error>
            where
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                K3: MStorageElement + 'static,
                Less: op::BinaryPredicateOp<R, (K1, K2, K3)>,
                KeyOutput: MIterMut<R, Item = (K1, K2, K3)>,
                ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let values = self.into_inner_with_policy(policy)?;
                let values = impl_miter_view!(values; $( $idx ),+);
                let (key_inner, value_inner) = crate::detail::sort_by_key(
                    policy,
                    (first_key, second_key, third_key),
                    values,
                    KernelOp::<R, Less>::new(),
                )?;
                key_output.write_from_inner(policy, key_inner)?;
                value_output.write_from_inner(policy, value_inner)
            }

            fn sort_by_two_key_dispatch<K1, K2, Less, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                first_key: crate::detail::device::DeviceColumnView<R, K1>,
                second_key: crate::detail::device::DeviceColumnView<R, K2>,
                _less: Less,
            ) -> Result<(KeyOutput, ValueOutput), Error>
            where
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                Less: op::BinaryPredicateOp<R, (K1, K2)>,
                KeyOutput: StorageFromInner<R, Item = (K1, K2)>,
                ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
            {
                let values = self.into_inner_with_policy(policy)?;
                let values = impl_miter_view!(values; $( $idx ),+);
                let (key_inner, value_inner) = crate::detail::sort_by_key(
                    policy,
                    (first_key, second_key),
                    values,
                    KernelOp::<R, Less>::new(),
                )?;
                Ok((
                    array_from_inner::<R, (K1, K2), KeyOutput>(key_inner),
                    array_from_inner::<R, <Self as MIter<R>>::Item, ValueOutput>(value_inner),
                ))
            }

            fn sort_by_two_key_into_dispatch<K1, K2, Less, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                first_key: crate::detail::device::DeviceColumnView<R, K1>,
                second_key: crate::detail::device::DeviceColumnView<R, K2>,
                _less: Less,
                key_output: KeyOutput,
                value_output: ValueOutput,
            ) -> Result<(), Error>
            where
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                Less: op::BinaryPredicateOp<R, (K1, K2)>,
                KeyOutput: MIterMut<R, Item = (K1, K2)>,
                ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let values = self.into_inner_with_policy(policy)?;
                let values = impl_miter_view!(values; $( $idx ),+);
                let (key_inner, value_inner) = crate::detail::sort_by_key(
                    policy,
                    (first_key, second_key),
                    values,
                    KernelOp::<R, Less>::new(),
                )?;
                key_output.write_from_inner(policy, key_inner)?;
                value_output.write_from_inner(policy, value_inner)
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
                KeyOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
                ValueOutput: StorageFromInner<R, Item = <Values as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                impl_sort_by_key_dispatch_body!(policy, values, less, input; $( $idx ),+)
            }

            fn sort_by_key_into_dispatch<Values, Less, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                values: Values,
                less: Less,
                key_output: KeyOutput,
                value_output: ValueOutput,
            ) -> Result<(), Error>
            where
                Values: MIter<R>,
                <Self as MIter<R>>::Item: cubecl::prelude::CubeType,
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
                KeyOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
                ValueOutput: MIterMut<R, Item = <Values as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                impl_sort_by_key_into_dispatch_body!(
                    policy, values, less, input, key_output, value_output; $( $idx ),+
                )
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
                KeyOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
                ValueOutput: StorageFromInner<R, Item = <Values as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                impl_unique_by_key_dispatch_body!(self, policy, values, eq, input; $( $idx ),+)
            }

            fn unique_by_key_into_dispatch<Values, Eq, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                values: Values,
                eq: Eq,
                key_output: KeyOutput,
                value_output: ValueOutput,
            ) -> Result<MIndex, Error>
            where
                Values: MIter<R>,
                <Self as MIter<R>>::Item: cubecl::prelude::CubeType,
                Eq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
                KeyOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
                ValueOutput: MIterMut<R, Item = <Values as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                impl_unique_by_key_into_dispatch_body!(
                    policy, values, eq, input, key_output, value_output; $( $idx ),+
                )
            }

            fn unique_by_two_key_dispatch<K1, K2, Eq, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                first_key: crate::detail::device::DeviceColumnView<R, K1>,
                second_key: crate::detail::device::DeviceColumnView<R, K2>,
                _eq: Eq,
            ) -> Result<(KeyOutput, ValueOutput), Error>
            where
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                Eq: op::BinaryPredicateOp<R, (K1, K2)>,
                KeyOutput: StorageFromInner<R, Item = (K1, K2)>,
                ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
            {
                let values = self.into_inner_with_policy(policy)?;
                ensure_same_len(values.0.len, first_key.len)?;
                let flags = crate::detail::read::unique_tuple2_flags_read::<
                    crate::detail::device::DeviceColumnView<R, K1>,
                    crate::detail::device::DeviceColumnView<R, K2>,
                    KernelOp<R, Eq>,
                >(policy, &first_key, &second_key)?;
                let selected_rank = crate::detail::primitives::select::selected_rank_from_flags(
                    policy,
                    first_key.len,
                    u32::try_from(first_key.len)
                        .map_err(|_| Error::LengthTooLarge { len: first_key.len })?,
                    flags,
                )?;
                let count =
                    crate::detail::primitives::select::selected_count(policy, &selected_rank)?;
                let payload_apply =
                    crate::detail::apply::SelectedPayloadApply::new(&selected_rank, count);
                let key_inner = (
                    payload_apply.apply_expr(policy, &first_key)?,
                    payload_apply.apply_expr(policy, &second_key)?,
                );
                $(
                    let $tmp = payload_apply.apply_expr(policy, &values.$idx)?;
                )+
                Ok((
                    array_from_inner::<R, (K1, K2), KeyOutput>(key_inner),
                    array_from_inner::<R, <Self as MIter<R>>::Item, ValueOutput>(($($tmp,)+)),
                ))
            }

            fn unique_by_two_key_into_dispatch<K1, K2, Eq, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                first_key: crate::detail::device::DeviceColumnView<R, K1>,
                second_key: crate::detail::device::DeviceColumnView<R, K2>,
                _eq: Eq,
                key_output: KeyOutput,
                value_output: ValueOutput,
            ) -> Result<MIndex, Error>
            where
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                Eq: op::BinaryPredicateOp<R, (K1, K2)>,
                KeyOutput: MIterMut<R, Item = (K1, K2)>,
                ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let values = self.into_inner_with_policy(policy)?;
                ensure_same_len(values.0.len, first_key.len)?;
                let flags = crate::detail::read::unique_tuple2_flags_read::<
                    crate::detail::device::DeviceColumnView<R, K1>,
                    crate::detail::device::DeviceColumnView<R, K2>,
                    KernelOp<R, Eq>,
                >(policy, &first_key, &second_key)?;
                let selected_rank = crate::detail::primitives::select::selected_rank_from_flags(
                    policy,
                    first_key.len,
                    u32::try_from(first_key.len)
                        .map_err(|_| Error::LengthTooLarge { len: first_key.len })?,
                    flags,
                )?;
                let count =
                    crate::detail::primitives::select::selected_count(policy, &selected_rank)?;
                let payload_apply =
                    crate::detail::apply::SelectedPayloadApply::new(&selected_rank, count);
                let key_inner = (
                    payload_apply.apply_expr(policy, &first_key)?,
                    payload_apply.apply_expr(policy, &second_key)?,
                );
                $(
                    let $tmp = payload_apply.apply_expr(policy, &values.$idx)?;
                )+
                let len = mindex_from_usize(key_inner.0.len())?;
                key_output.write_prefix_from_inner(policy, key_inner)?;
                value_output.write_prefix_from_inner(policy, ($($tmp,)+))?;
                Ok(len)
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
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                K3: MStorageElement + 'static,
                Eq: op::BinaryPredicateOp<R, (K1, K2, K3)>,
                KeyOutput: StorageFromInner<R, Item = (K1, K2, K3)>,
                ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
            {
                let values = self.into_inner_with_policy(policy)?;
                ensure_same_len(values.0.len, first_key.len)?;
                let flags = crate::detail::read::unique_tuple3_flags_read::<
                    crate::detail::device::DeviceColumnView<R, K1>,
                    crate::detail::device::DeviceColumnView<R, K2>,
                    crate::detail::device::DeviceColumnView<R, K3>,
                    KernelOp<R, Eq>,
                >(
                    policy,
                    &first_key,
                    &second_key,
                    &third_key,
                )?;
                let selected_rank = crate::detail::primitives::select::selected_rank_from_flags(
                    policy,
                    first_key.len,
                    u32::try_from(first_key.len)
                        .map_err(|_| Error::LengthTooLarge { len: first_key.len })?,
                    flags,
                )?;
                let count =
                    crate::detail::primitives::select::selected_count(policy, &selected_rank)?;
                let payload_apply =
                    crate::detail::apply::SelectedPayloadApply::new(&selected_rank, count);
                let key_inner = (
                    payload_apply.apply_expr(policy, &first_key)?,
                    payload_apply.apply_expr(policy, &second_key)?,
                    payload_apply.apply_expr(policy, &third_key)?,
                );
                $(
                    let $tmp = payload_apply.apply_expr(policy, &values.$idx)?;
                )+
                Ok((
                    array_from_inner::<R, (K1, K2, K3), KeyOutput>(key_inner),
                    array_from_inner::<R, <Self as MIter<R>>::Item, ValueOutput>(($($tmp,)+)),
                ))
            }

            fn unique_by_three_key_into_dispatch<K1, K2, K3, Eq, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                first_key: crate::detail::device::DeviceColumnView<R, K1>,
                second_key: crate::detail::device::DeviceColumnView<R, K2>,
                third_key: crate::detail::device::DeviceColumnView<R, K3>,
                _eq: Eq,
                key_output: KeyOutput,
                value_output: ValueOutput,
            ) -> Result<MIndex, Error>
            where
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                K3: MStorageElement + 'static,
                Eq: op::BinaryPredicateOp<R, (K1, K2, K3)>,
                KeyOutput: MIterMut<R, Item = (K1, K2, K3)>,
                ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let values = self.into_inner_with_policy(policy)?;
                ensure_same_len(values.0.len, first_key.len)?;
                let flags = crate::detail::read::unique_tuple3_flags_read::<
                    crate::detail::device::DeviceColumnView<R, K1>,
                    crate::detail::device::DeviceColumnView<R, K2>,
                    crate::detail::device::DeviceColumnView<R, K3>,
                    KernelOp<R, Eq>,
                >(
                    policy,
                    &first_key,
                    &second_key,
                    &third_key,
                )?;
                let selected_rank = crate::detail::primitives::select::selected_rank_from_flags(
                    policy,
                    first_key.len,
                    u32::try_from(first_key.len)
                        .map_err(|_| Error::LengthTooLarge { len: first_key.len })?,
                    flags,
                )?;
                let count =
                    crate::detail::primitives::select::selected_count(policy, &selected_rank)?;
                let payload_apply =
                    crate::detail::apply::SelectedPayloadApply::new(&selected_rank, count);
                let key_inner = (
                    payload_apply.apply_expr(policy, &first_key)?,
                    payload_apply.apply_expr(policy, &second_key)?,
                    payload_apply.apply_expr(policy, &third_key)?,
                );
                $(
                    let $tmp = payload_apply.apply_expr(policy, &values.$idx)?;
                )+
                let len = mindex_from_usize(key_inner.0.len())?;
                key_output.write_prefix_from_inner(policy, key_inner)?;
                value_output.write_prefix_from_inner(policy, ($($tmp,)+))?;
                Ok(len)
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
                Output: StorageFromInner<R, Item = <Values as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                impl_inclusive_scan_by_key_dispatch_body!(
                    policy, values, key_eq, op, input; $( $idx ),+
                )
            }

            fn inclusive_scan_by_key_into_dispatch<Values, KeyEq, Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                values: Values,
                key_eq: KeyEq,
                op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                Values: MIter<R>,
                KeyEq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
                Op: op::ReductionOp<R, <Values as MIter<R>>::Item>,
                Output: MIterMut<R, Item = <Values as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                impl_inclusive_scan_by_key_into_dispatch_body!(
                    policy, values, key_eq, op, input, output; $( $idx ),+
                )
            }

            fn inclusive_scan_by_two_key_dispatch<K1, K2, KeyEq, Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                first_key: crate::detail::device::DeviceColumnView<R, K1>,
                second_key: crate::detail::device::DeviceColumnView<R, K2>,
                _key_eq: KeyEq,
                _op: Op,
            ) -> Result<Output, Error>
            where
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K1, K2)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                ensure_same_len(input.0.len, first_key.len)?;
                let head_flags = crate::detail::read::unique_tuple2_flags_read::<
                    crate::detail::device::DeviceColumnView<R, K1>,
                    crate::detail::device::DeviceColumnView<R, K2>,
                    KernelOp<R, KeyEq>,
                >(policy, &first_key, &second_key)?;
                let len_u32 = u32::try_from(first_key.len)
                    .map_err(|_| Error::LengthTooLarge { len: first_key.len })?;
                let control: crate::detail::control::ScanByKeyControl<R> = crate::detail::control::ScanByKeyControl {
                    head_flags,
                    len: first_key.len,
                    len_u32,
                    _runtime: std::marker::PhantomData,
                };
                let inner = impl_tuple_inclusive_scan_by_two_key_values_body!(
                    policy, input, control; $( $idx ),+
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn inclusive_scan_by_two_key_into_dispatch<K1, K2, KeyEq, Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                first_key: crate::detail::device::DeviceColumnView<R, K1>,
                second_key: crate::detail::device::DeviceColumnView<R, K2>,
                _key_eq: KeyEq,
                _op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K1, K2)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                ensure_same_len(input.0.len, first_key.len)?;
                let head_flags = crate::detail::read::unique_tuple2_flags_read::<
                    crate::detail::device::DeviceColumnView<R, K1>,
                    crate::detail::device::DeviceColumnView<R, K2>,
                    KernelOp<R, KeyEq>,
                >(policy, &first_key, &second_key)?;
                let len_u32 = u32::try_from(first_key.len)
                    .map_err(|_| Error::LengthTooLarge { len: first_key.len })?;
                let control: crate::detail::control::ScanByKeyControl<R> = crate::detail::control::ScanByKeyControl {
                    head_flags,
                    len: first_key.len,
                    len_u32,
                    _runtime: std::marker::PhantomData,
                };
                let inner = impl_tuple_inclusive_scan_by_two_key_values_body!(
                    policy, input, control; $( $idx ),+
                )?;
                output.write_from_inner(policy, inner)
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
                Output: StorageFromInner<R, Item = <Values as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                impl_exclusive_scan_by_key_dispatch_body!(
                    policy, values, key_eq, init, op, input; $( $idx ),+
                )
            }

            fn exclusive_scan_by_key_into_dispatch<Values, KeyEq, Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                values: Values,
                key_eq: KeyEq,
                init: <Values as MIter<R>>::Item,
                op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                Values: MIter<R>,
                KeyEq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
                Op: op::ReductionOp<R, <Values as MIter<R>>::Item>,
                Output: MIterMut<R, Item = <Values as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                impl_exclusive_scan_by_key_into_dispatch_body!(
                    policy, values, key_eq, init, op, input, output; $( $idx ),+
                )
            }

            fn exclusive_scan_by_two_key_dispatch<K1, K2, KeyEq, Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                first_key: crate::detail::device::DeviceColumnView<R, K1>,
                second_key: crate::detail::device::DeviceColumnView<R, K2>,
                _key_eq: KeyEq,
                init: <Self as MIter<R>>::Item,
                _op: Op,
            ) -> Result<Output, Error>
            where
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K1, K2)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                ensure_same_len(input.0.len, first_key.len)?;
                let head_flags = crate::detail::read::unique_tuple2_flags_read::<
                    crate::detail::device::DeviceColumnView<R, K1>,
                    crate::detail::device::DeviceColumnView<R, K2>,
                    KernelOp<R, KeyEq>,
                >(policy, &first_key, &second_key)?;
                let len_u32 = u32::try_from(first_key.len)
                    .map_err(|_| Error::LengthTooLarge { len: first_key.len })?;
                let control: crate::detail::control::ScanByKeyControl<R> = crate::detail::control::ScanByKeyControl {
                    head_flags,
                    len: first_key.len,
                    len_u32,
                    _runtime: std::marker::PhantomData,
                };
                let inner = impl_tuple_exclusive_scan_by_two_key_values_body!(
                    policy, input, control, init; $( $idx ),+
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn exclusive_scan_by_two_key_into_dispatch<K1, K2, KeyEq, Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                first_key: crate::detail::device::DeviceColumnView<R, K1>,
                second_key: crate::detail::device::DeviceColumnView<R, K2>,
                _key_eq: KeyEq,
                init: <Self as MIter<R>>::Item,
                _op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K1, K2)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                ensure_same_len(input.0.len, first_key.len)?;
                let head_flags = crate::detail::read::unique_tuple2_flags_read::<
                    crate::detail::device::DeviceColumnView<R, K1>,
                    crate::detail::device::DeviceColumnView<R, K2>,
                    KernelOp<R, KeyEq>,
                >(policy, &first_key, &second_key)?;
                let len_u32 = u32::try_from(first_key.len)
                    .map_err(|_| Error::LengthTooLarge { len: first_key.len })?;
                let control: crate::detail::control::ScanByKeyControl<R> = crate::detail::control::ScanByKeyControl {
                    head_flags,
                    len: first_key.len,
                    len_u32,
                    _runtime: std::marker::PhantomData,
                };
                let inner = impl_tuple_exclusive_scan_by_two_key_values_body!(
                    policy, input, control, init; $( $idx ),+
                )?;
                output.write_from_inner(policy, inner)
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
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                K3: MStorageElement + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K1, K2, K3)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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
                let control: crate::detail::control::ScanByKeyControl<R> = crate::detail::control::ScanByKeyControl {
                    head_flags,
                    len: first_key.len,
                    len_u32,
                    _runtime: std::marker::PhantomData,
                };
                let inner = impl_tuple_inclusive_scan_by_three_key_values_body!(
                    policy, input, control; $( $idx ),+
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn inclusive_scan_by_three_key_into_dispatch<K1, K2, K3, KeyEq, Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                first_key: crate::detail::device::DeviceColumnView<R, K1>,
                second_key: crate::detail::device::DeviceColumnView<R, K2>,
                third_key: crate::detail::device::DeviceColumnView<R, K3>,
                _key_eq: KeyEq,
                _op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                K3: MStorageElement + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K1, K2, K3)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
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
                let control: crate::detail::control::ScanByKeyControl<R> = crate::detail::control::ScanByKeyControl {
                    head_flags,
                    len: first_key.len,
                    len_u32,
                    _runtime: std::marker::PhantomData,
                };
                let inner = impl_tuple_inclusive_scan_by_three_key_values_body!(
                    policy, input, control; $( $idx ),+
                )?;
                output.write_from_inner(policy, inner)
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
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                K3: MStorageElement + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K1, K2, K3)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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
                let control: crate::detail::control::ScanByKeyControl<R> = crate::detail::control::ScanByKeyControl {
                    head_flags,
                    len: first_key.len,
                    len_u32,
                    _runtime: std::marker::PhantomData,
                };
                let inner = impl_tuple_exclusive_scan_by_three_key_values_body!(
                    policy, input, control, init; $( $idx ),+
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn exclusive_scan_by_three_key_into_dispatch<K1, K2, K3, KeyEq, Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                first_key: crate::detail::device::DeviceColumnView<R, K1>,
                second_key: crate::detail::device::DeviceColumnView<R, K2>,
                third_key: crate::detail::device::DeviceColumnView<R, K3>,
                _key_eq: KeyEq,
                init: <Self as MIter<R>>::Item,
                _op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                K3: MStorageElement + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K1, K2, K3)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
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
                let control: crate::detail::control::ScanByKeyControl<R> = crate::detail::control::ScanByKeyControl {
                    head_flags,
                    len: first_key.len,
                    len_u32,
                    _runtime: std::marker::PhantomData,
                };
                let inner = impl_tuple_exclusive_scan_by_three_key_values_body!(
                    policy, input, control, init; $( $idx ),+
                )?;
                output.write_from_inner(policy, inner)
            }

            fn inclusive_scan_by_single_key_dispatch<K, KeyEq, Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                keys: crate::detail::device::DeviceColumnView<R, K>,
                _key_eq: KeyEq,
                _op: Op,
            ) -> Result<Output, Error>
            where
                K: MStorageElement + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K,)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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

            fn inclusive_scan_by_single_key_into_dispatch<K, KeyEq, Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                keys: crate::detail::device::DeviceColumnView<R, K>,
                _key_eq: KeyEq,
                _op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                K: MStorageElement + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K,)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
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
                output.write_from_inner(policy, inner)
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
                K: MStorageElement + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K,)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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

            fn exclusive_scan_by_single_key_into_dispatch<K, KeyEq, Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                keys: crate::detail::device::DeviceColumnView<R, K>,
                _key_eq: KeyEq,
                init: <Self as MIter<R>>::Item,
                _op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                K: MStorageElement + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K,)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
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
                output.write_from_inner(policy, inner)
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
                K: MStorageElement + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K,)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                KeyOutput: StorageFromInner<R, Item = (K,)>,
                ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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

            fn reduce_by_single_key_into_dispatch<K, KeyEq, Op, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                keys: crate::detail::device::DeviceColumnView<R, K>,
                _key_eq: KeyEq,
                init: <Self as MIter<R>>::Item,
                _op: Op,
                key_output: KeyOutput,
                value_output: ValueOutput,
            ) -> Result<MIndex, Error>
            where
                K: MStorageElement + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K,)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                KeyOutput: MIterMut<R, Item = (K,)>,
                ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
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
                let len = mindex_from_usize(key_inner.0.len())?;
                key_output.write_prefix_from_inner(policy, key_inner)?;
                value_output.write_prefix_from_inner(policy, value_inner)?;
                Ok(len)
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
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                K3: MStorageElement + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K1, K2, K3)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                KeyOutput: StorageFromInner<R, Item = (K1, K2, K3)>,
                ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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
                let control: crate::detail::control::ScanByKeyControl<R> = crate::detail::control::ScanByKeyControl {
                    head_flags,
                    len: first_key.len,
                    len_u32,
                    _runtime: std::marker::PhantomData,
                };
                let (key_inner, value_inner) = impl_tuple_reduce_by_three_key_values_body!(
                    policy, input, init, first_key, second_key, third_key, head_flags, end_flags, len_u32, control; $( $ty ),+; $( $idx ),+
                )?;
                Ok((
                    array_from_inner::<R, (K1, K2, K3), KeyOutput>(key_inner),
                    array_from_inner::<R, <Self as MIter<R>>::Item, ValueOutput>(value_inner),
                ))
            }

            fn reduce_by_three_key_into_dispatch<K1, K2, K3, KeyEq, Op, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                first_key: crate::detail::device::DeviceColumnView<R, K1>,
                second_key: crate::detail::device::DeviceColumnView<R, K2>,
                third_key: crate::detail::device::DeviceColumnView<R, K3>,
                _key_eq: KeyEq,
                init: <Self as MIter<R>>::Item,
                _op: Op,
                key_output: KeyOutput,
                value_output: ValueOutput,
            ) -> Result<MIndex, Error>
            where
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                K3: MStorageElement + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K1, K2, K3)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                KeyOutput: MIterMut<R, Item = (K1, K2, K3)>,
                ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
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
                    key_output.write_prefix_from_inner(policy, key_inner)?;
                    value_output.write_prefix_from_inner(policy, value_inner)?;
                    return Ok(0);
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
                let control: crate::detail::control::ScanByKeyControl<R> = crate::detail::control::ScanByKeyControl {
                    head_flags,
                    len: first_key.len,
                    len_u32,
                    _runtime: std::marker::PhantomData,
                };
                let (key_inner, value_inner) = impl_tuple_reduce_by_three_key_values_body!(
                    policy, input, init, first_key, second_key, third_key, head_flags, end_flags, len_u32, control; $( $ty ),+; $( $idx ),+
                )?;
                let len = mindex_from_usize(key_inner.0.len())?;
                key_output.write_prefix_from_inner(policy, key_inner)?;
                value_output.write_prefix_from_inner(policy, value_inner)?;
                Ok(len)
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
                KeyOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
                ValueOutput: StorageFromInner<R, Item = <Values as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                impl_reduce_by_key_dispatch_body!(
                    policy, values, key_eq, init, op, input; $( $idx ),+
                )
            }

            fn reduce_by_key_into_dispatch<Values, KeyEq, Op, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                values: Values,
                key_eq: KeyEq,
                init: <Values as MIter<R>>::Item,
                op: Op,
                key_output: KeyOutput,
                value_output: ValueOutput,
            ) -> Result<MIndex, Error>
            where
                Values: MIter<R>,
                <Self as MIter<R>>::Item: cubecl::prelude::CubeType,
                KeyEq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
                Op: op::ReductionOp<R, <Values as MIter<R>>::Item>,
                KeyOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
                ValueOutput: MIterMut<R, Item = <Values as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                impl_reduce_by_key_into_dispatch_body!(
                    policy, values, key_eq, init, op, input, key_output, value_output; $( $idx ),+
                )
            }

            fn reduce_by_two_key_dispatch<K1, K2, KeyEq, Op, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                first_key: crate::detail::device::DeviceColumnView<R, K1>,
                second_key: crate::detail::device::DeviceColumnView<R, K2>,
                _key_eq: KeyEq,
                init: <Self as MIter<R>>::Item,
                _op: Op,
            ) -> Result<(KeyOutput, ValueOutput), Error>
            where
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K1, K2)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                KeyOutput: StorageFromInner<R, Item = (K1, K2)>,
                ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                ensure_same_len(input.0.len, first_key.len)?;
                if first_key.len == 0 {
                    let key_inner = (policy.empty_device_vec(), policy.empty_device_vec());
                    let value_inner = ($( {
                        let _ = stringify!($ty);
                        policy.empty_device_vec()
                    }, )+);
                    return Ok((
                        array_from_inner::<R, (K1, K2), KeyOutput>(key_inner),
                        array_from_inner::<R, <Self as MIter<R>>::Item, ValueOutput>(value_inner),
                    ));
                }
                let head_flags = crate::detail::read::unique_tuple2_flags_read::<
                    crate::detail::device::DeviceColumnView<R, K1>,
                    crate::detail::device::DeviceColumnView<R, K2>,
                    KernelOp<R, KeyEq>,
                >(policy, &first_key, &second_key)?;
                let end_flags =
                    crate::detail::impls::end_flags_from_head_flags(policy, head_flags.clone(), first_key.len)?;
                let len_u32 = u32::try_from(first_key.len)
                    .map_err(|_| Error::LengthTooLarge { len: first_key.len })?;
                let control: crate::detail::control::ScanByKeyControl<R> = crate::detail::control::ScanByKeyControl {
                    head_flags,
                    len: first_key.len,
                    len_u32,
                    _runtime: std::marker::PhantomData,
                };
                let (key_inner, value_inner) = impl_tuple_reduce_by_two_key_values_body!(
                    policy, input, init, first_key, second_key, end_flags, len_u32, control; $( $ty ),+; $( $idx ),+
                )?;
                Ok((
                    array_from_inner::<R, (K1, K2), KeyOutput>(key_inner),
                    array_from_inner::<R, <Self as MIter<R>>::Item, ValueOutput>(value_inner),
                ))
            }

            fn reduce_by_two_key_into_dispatch<K1, K2, KeyEq, Op, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                first_key: crate::detail::device::DeviceColumnView<R, K1>,
                second_key: crate::detail::device::DeviceColumnView<R, K2>,
                _key_eq: KeyEq,
                init: <Self as MIter<R>>::Item,
                _op: Op,
                key_output: KeyOutput,
                value_output: ValueOutput,
            ) -> Result<MIndex, Error>
            where
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K1, K2)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                KeyOutput: MIterMut<R, Item = (K1, K2)>,
                ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                ensure_same_len(input.0.len, first_key.len)?;
                if first_key.len == 0 {
                    let key_inner = (policy.empty_device_vec(), policy.empty_device_vec());
                    let value_inner = ($( {
                        let _ = stringify!($ty);
                        policy.empty_device_vec()
                    }, )+);
                    key_output.write_prefix_from_inner(policy, key_inner)?;
                    value_output.write_prefix_from_inner(policy, value_inner)?;
                    return Ok(0);
                }
                let head_flags = crate::detail::read::unique_tuple2_flags_read::<
                    crate::detail::device::DeviceColumnView<R, K1>,
                    crate::detail::device::DeviceColumnView<R, K2>,
                    KernelOp<R, KeyEq>,
                >(policy, &first_key, &second_key)?;
                let end_flags =
                    crate::detail::impls::end_flags_from_head_flags(policy, head_flags.clone(), first_key.len)?;
                let len_u32 = u32::try_from(first_key.len)
                    .map_err(|_| Error::LengthTooLarge { len: first_key.len })?;
                let control: crate::detail::control::ScanByKeyControl<R> = crate::detail::control::ScanByKeyControl {
                    head_flags,
                    len: first_key.len,
                    len_u32,
                    _runtime: std::marker::PhantomData,
                };
                let (key_inner, value_inner) = impl_tuple_reduce_by_two_key_values_body!(
                    policy, input, init, first_key, second_key, end_flags, len_u32, control; $( $ty ),+; $( $idx ),+
                )?;
                let len = mindex_from_usize(key_inner.0.len())?;
                key_output.write_prefix_from_inner(policy, key_inner)?;
                value_output.write_prefix_from_inner(policy, value_inner)?;
                Ok(len)
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
                RightValues: MIter<R, Item = <Self as MIter<R>>::Item>,
                K: MStorageElement + 'static,
                Less: op::BinaryPredicateOp<R, (K,)>,
                KeyOutput: StorageFromInner<R, Item = (K,)>,
                ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
            {
                let left_values = self.into_view_with_policy(policy)?;
                let right_values = right_values.into_view_with_policy(policy)?;
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

            fn merge_by_single_key_same_into_dispatch<K, RightValues, Less, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                left_keys: crate::detail::device::DeviceColumnView<R, K>,
                right_keys: crate::detail::device::DeviceColumnView<R, K>,
                right_values: RightValues,
                _less: Less,
                key_output: KeyOutput,
                value_output: ValueOutput,
            ) -> Result<(), Error>
            where
                RightValues: MIter<R, Item = <Self as MIter<R>>::Item>,
                K: MStorageElement + 'static,
                Less: op::BinaryPredicateOp<R, (K,)>,
                KeyOutput: MIterMut<R, Item = (K,)>,
                ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let left_values = self.into_view_with_policy(policy)?;
                let right_values = right_values.into_view_with_policy(policy)?;
                let (key_inner, value_inner) = crate::detail::merge_by_key(
                    policy,
                    crate::detail::device::SoAView1 { source: left_keys },
                    impl_miter_view!(left_values; $( $idx ),+),
                    crate::detail::device::SoAView1 { source: right_keys },
                    impl_miter_view!(right_values; $( $idx ),+),
                    KernelTuple1Op::<R, Less>::new(),
                )?;
                key_output.write_from_inner(policy, key_inner)?;
                value_output.write_from_inner(policy, value_inner)
            }

            fn merge_by_two_key_same_dispatch<K1, K2, RightValues, Less, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                left_first_key: crate::detail::device::DeviceColumnView<R, K1>,
                left_second_key: crate::detail::device::DeviceColumnView<R, K2>,
                right_first_key: crate::detail::device::DeviceColumnView<R, K1>,
                right_second_key: crate::detail::device::DeviceColumnView<R, K2>,
                right_values: RightValues,
                _less: Less,
            ) -> Result<(KeyOutput, ValueOutput), Error>
            where
                RightValues: MIter<R, Item = <Self as MIter<R>>::Item>,
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                Less: op::BinaryPredicateOp<R, (K1, K2)>,
                KeyOutput: StorageFromInner<R, Item = (K1, K2)>,
                ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
            {
                let left_values = self.into_view_with_policy(policy)?;
                let right_values = right_values.into_view_with_policy(policy)?;
                let (key_inner, value_inner) = crate::detail::merge_by_key(
                    policy,
                    (left_first_key, left_second_key),
                    left_values,
                    (right_first_key, right_second_key),
                    right_values,
                    KernelOp::<R, Less>::new(),
                )?;
                Ok((
                    array_from_inner::<R, (K1, K2), KeyOutput>(key_inner),
                    array_from_inner::<R, <Self as MIter<R>>::Item, ValueOutput>(value_inner),
                ))
            }

            fn merge_by_two_key_same_into_dispatch<K1, K2, RightValues, Less, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                left_first_key: crate::detail::device::DeviceColumnView<R, K1>,
                left_second_key: crate::detail::device::DeviceColumnView<R, K2>,
                right_first_key: crate::detail::device::DeviceColumnView<R, K1>,
                right_second_key: crate::detail::device::DeviceColumnView<R, K2>,
                right_values: RightValues,
                _less: Less,
                key_output: KeyOutput,
                value_output: ValueOutput,
            ) -> Result<(), Error>
            where
                RightValues: MIter<R, Item = <Self as MIter<R>>::Item>,
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                Less: op::BinaryPredicateOp<R, (K1, K2)>,
                KeyOutput: MIterMut<R, Item = (K1, K2)>,
                ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let left_values = self.into_view_with_policy(policy)?;
                let right_values = right_values.into_view_with_policy(policy)?;
                let (key_inner, value_inner) = crate::detail::merge_by_key(
                    policy,
                    (left_first_key, left_second_key),
                    left_values,
                    (right_first_key, right_second_key),
                    right_values,
                    KernelOp::<R, Less>::new(),
                )?;
                key_output.write_from_inner(policy, key_inner)?;
                value_output.write_from_inner(policy, value_inner)
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
                RightValues: MIter<R, Item = <Self as MIter<R>>::Item>,
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                K3: MStorageElement + 'static,
                Less: op::BinaryPredicateOp<R, (K1, K2, K3)>,
                KeyOutput: StorageFromInner<R, Item = (K1, K2, K3)>,
                ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
            {
                let left_values = self.into_view_with_policy(policy)?;
                let right_values = right_values.into_view_with_policy(policy)?;
                let (key_inner, value_inner) = crate::detail::merge_by_key(
                    policy,
                    (left_first_key, left_second_key, left_third_key),
                    left_values,
                    (right_first_key, right_second_key, right_third_key),
                    right_values,
                    KernelOp::<R, Less>::new(),
                )?;
                Ok((
                    array_from_inner::<R, (K1, K2, K3), KeyOutput>(key_inner),
                    array_from_inner::<R, <Self as MIter<R>>::Item, ValueOutput>(value_inner),
                ))
            }

            fn merge_by_three_key_same_into_dispatch<
                K1,
                K2,
                K3,
                RightValues,
                Less,
                KeyOutput,
                ValueOutput,
            >(
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
                key_output: KeyOutput,
                value_output: ValueOutput,
            ) -> Result<(), Error>
            where
                RightValues: MIter<R, Item = <Self as MIter<R>>::Item>,
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                K3: MStorageElement + 'static,
                Less: op::BinaryPredicateOp<R, (K1, K2, K3)>,
                KeyOutput: MIterMut<R, Item = (K1, K2, K3)>,
                ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let left_values = self.into_view_with_policy(policy)?;
                let right_values = right_values.into_view_with_policy(policy)?;
                let (key_inner, value_inner) = crate::detail::merge_by_key(
                    policy,
                    (left_first_key, left_second_key, left_third_key),
                    left_values,
                    (right_first_key, right_second_key, right_third_key),
                    right_values,
                    KernelOp::<R, Less>::new(),
                )?;
                key_output.write_from_inner(policy, key_inner)?;
                value_output.write_from_inner(policy, value_inner)
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
                RightKeys: MIter<R, Item = <Self as MIter<R>>::Item>,
                LeftValues: MIter<R>,
                RightValues: MIter<R, Item = <LeftValues as MIter<R>>::Item>,
                <Self as MIter<R>>::Item: cubecl::prelude::CubeType,
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
                KeyOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
                ValueOutput: StorageFromInner<R, Item = <LeftValues as MIter<R>>::Item>,
            {
                let left_input = self.into_view_with_policy(policy)?;
                impl_merge_by_key_dispatch_body!(
                    policy, right_keys, left_values, right_values, less, left_input; $( $idx ),+
                )
            }

            fn merge_by_key_into_dispatch<
                RightKeys,
                LeftValues,
                RightValues,
                Less,
                KeyOutput,
                ValueOutput,
            >(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right_keys: RightKeys,
                left_values: LeftValues,
                right_values: RightValues,
                less: Less,
                key_output: KeyOutput,
                value_output: ValueOutput,
            ) -> Result<(), Error>
            where
                RightKeys: MIter<R, Item = <Self as MIter<R>>::Item>,
                LeftValues: MIter<R>,
                RightValues: MIter<R, Item = <LeftValues as MIter<R>>::Item>,
                <Self as MIter<R>>::Item: cubecl::prelude::CubeType,
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
                KeyOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
                ValueOutput: MIterMut<R, Item = <LeftValues as MIter<R>>::Item>,
            {
                let left_input = self.into_view_with_policy(policy)?;
                impl_merge_by_key_into_dispatch_body!(
                    policy,
                    right_keys,
                    left_values,
                    right_values,
                    less,
                    left_input,
                    key_output,
                    value_output;
                    $( $idx ),+
                )
            }

            fn gather_dispatch<Indices, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                indices: Indices,
                output: Output,
            ) -> Result<(), Error>
            where
                Indices: MIter<R, Item = MIndex>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let indices =
                    <Indices as sealed::MIterDispatch<R>>::index_column_dispatch(indices, policy)?;
                let input = self.into_inner_with_policy(policy)?;
                $(
                    let $tmp = <Output as sealed::MIterMutDispatch<R>>::column_mut_view_by_index_inner::<$ty>(
                        &output,
                        $idx,
                    )?
                    .ok_or_else(|| Error::Launch {
                        message: "gather output must match input shape".to_string(),
                    })?;
                    crate::detail::apply::IndexedExprApply::gather_expr_into(
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
                Indices: MIter<R, Item = MIndex>,
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
            {
                let indices =
                    <Indices as sealed::MIterDispatch<R>>::index_column_dispatch(indices, policy)?;
                let input = self.into_inner_with_policy(policy)?;
                $(
                    let $tmp = crate::detail::apply::IndexedExprApply::gather_expr(
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
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = crate::detail::adjacent_difference(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    KernelOp::<R, Op>::new(),
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn inclusive_scan_into_dispatch<Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = crate::detail::inclusive_scan(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    KernelOp::<R, Op>::new(),
                )?;
                output.write_from_inner(policy, inner)
            }

            fn exclusive_scan_into_dispatch<Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                init: <Self as MIter<R>>::Item,
                _op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = crate::detail::exclusive_scan(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    init,
                    KernelOp::<R, Op>::new(),
                )?;
                output.write_from_inner(policy, inner)
            }

            fn adjacent_difference_into_dispatch<Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = crate::detail::adjacent_difference(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    KernelOp::<R, Op>::new(),
                )?;
                output.write_from_inner(policy, inner)
            }

            fn copy_where_dispatch<Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                stencil: crate::detail::api::PrecomputedSelection<R>,
            ) -> Result<Output, Error>
            where
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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

            fn copy_where_into_dispatch<Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                stencil: crate::detail::api::PrecomputedSelection<R>,
                output: Output,
            ) -> Result<MIndex, Error>
            where
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = crate::detail::copy_where(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    stencil,
                    KernelOp::<R, StencilFlag>::new(),
                )?;
                let len = mindex_from_usize(inner.0.len())?;
                output.write_prefix_from_inner(policy, inner)?;
                Ok(len)
            }

            fn remove_if_dispatch<Pred, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _pred: Pred,
                env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<Output, Error>
            where
                Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = crate::detail::remove_if(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    KernelOp::<R, Pred>::new(),
                    env,
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn remove_where_dispatch<Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                stencil: crate::detail::api::PrecomputedSelection<R>,
            ) -> Result<Output, Error>
            where
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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

            fn remove_where_into_dispatch<Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                stencil: crate::detail::api::PrecomputedSelection<R>,
                output: Output,
            ) -> Result<MIndex, Error>
            where
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = crate::detail::copy_where(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    stencil,
                    KernelOp::<R, StencilFlag>::new(),
                )?;
                let len = mindex_from_usize(inner.0.len())?;
                output.write_prefix_from_inner(policy, inner)?;
                Ok(len)
            }

            fn count_if_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _pred: Pred,
                env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<MIndex, Error>
            where
                Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                crate::detail::count_if(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<R, Pred>::new(), env)
            }

            fn all_of_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _pred: Pred,
                env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<bool, Error>
            where
                Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                crate::detail::all_of(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<R, Pred>::new(), env)
            }

            fn any_of_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _pred: Pred,
                env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<bool, Error>
            where
                Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                crate::detail::any_of(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<R, Pred>::new(), env)
            }

            fn none_of_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _pred: Pred,
                env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<bool, Error>
            where
                Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                crate::detail::none_of(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<R, Pred>::new(), env)
            }

            fn find_if_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _pred: Pred,
                env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<Option<MIndex>, Error>
            where
                Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                crate::detail::find_if(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<R, Pred>::new(), env)
            }

            fn partition_dispatch<Pred, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _pred: Pred,
                env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<(Output, Output), Error>
            where
                Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let (matching, failing) = crate::detail::partition(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    KernelOp::<R, Pred>::new(),
                    env,
                )?;
                Ok((
                    array_from_inner::<R, <Self as MIter<R>>::Item, Output>(matching),
                    array_from_inner::<R, <Self as MIter<R>>::Item, Output>(failing),
                ))
            }

            fn partition_into_dispatch<Pred, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _pred: Pred,
                env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
                output: Output,
            ) -> Result<MIndex, Error>
            where
                Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let (matching, failing) = crate::detail::partition(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    KernelOp::<R, Pred>::new(),
                    env,
                )?;
                let split = mindex_from_usize(matching.0.len())?;
                output.write_split_from_inner(policy, matching, failing)?;
                Ok(split)
            }

            fn is_partitioned_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _pred: Pred,
                env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<bool, Error>
            where
                Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                crate::detail::is_partitioned(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<R, Pred>::new(), env)
            }

            fn replace_where_dispatch<Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                replacement: <Self as MIter<R>>::Item,
                stencil: crate::detail::api::PrecomputedSelection<R>,
            ) -> Result<Output, Error>
            where
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = crate::detail::unique(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    KernelOp::<R, Pred>::new(),
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn unique_into_dispatch<Pred, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _pred: Pred,
                output: Output,
            ) -> Result<MIndex, Error>
            where
                Pred: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = crate::detail::unique(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    KernelOp::<R, Pred>::new(),
                )?;
                let len = mindex_from_usize(inner.0.len())?;
                output.write_prefix_from_inner(policy, inner)?;
                Ok(len)
            }

            fn min_element_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _less: Less,
            ) -> Result<Option<MIndex>, Error>
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
            ) -> Result<Option<MIndex>, Error>
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
            ) -> Result<Option<(MIndex, MIndex)>, Error>
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
            ) -> Result<Option<MIndex>, Error>
            where
                Pred: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                crate::detail::adjacent_find(policy, impl_miter_view!(input; $( $idx ),+), KernelOp::<R, Pred>::new())
            }

            fn lower_bound_dispatch<Values, Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                values: Values,
                _less: Less,
            ) -> Result<crate::runtime::DeviceVec<R, MIndex>, Error>
            where
                Values: MIter<R, Item = <Self as MIter<R>>::Item>,
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_view_with_policy(policy)?;
                let values = values.into_view_with_policy(policy)?;
                let inner = crate::detail::lower_bound_many(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    impl_miter_view!(values; $( $idx ),+),
                    KernelOp::<R, Less>::new(),
                )?;
                Ok(crate::runtime::DeviceVec::from_inner(inner))
            }

            fn upper_bound_dispatch<Values, Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                values: Values,
                _less: Less,
            ) -> Result<crate::runtime::DeviceVec<R, MIndex>, Error>
            where
                Values: MIter<R, Item = <Self as MIter<R>>::Item>,
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_view_with_policy(policy)?;
                let values = values.into_view_with_policy(policy)?;
                let inner = crate::detail::upper_bound_many(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    impl_miter_view!(values; $( $idx ),+),
                    KernelOp::<R, Less>::new(),
                )?;
                Ok(crate::runtime::DeviceVec::from_inner(inner))
            }

            fn is_sorted_until_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _less: Less,
            ) -> Result<MIndex, Error>
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
                Indices: MIter<R, Item = MIndex>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let indices =
                    <Indices as sealed::MIterDispatch<R>>::index_column_dispatch(indices, policy)?;
                let input = self.into_inner_with_policy(policy)?;
                let mask = stencil.mask();
                $(
                    let $tmp = <Output as sealed::MIterMutDispatch<R>>::column_mut_view_by_index_inner::<$ty>(
                        &output,
                        $idx,
                    )?
                    .ok_or_else(|| Error::Launch {
                        message: "gather_where output must match input shape".to_string(),
                    })?;
                    crate::detail::apply::MaskedIndexedExprApply::gather_where_expr_into(
                        policy,
                        &input.$idx,
                        &indices,
                        &mask,
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
                Indices: MIter<R, Item = MIndex>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let indices =
                    <Indices as sealed::MIterDispatch<R>>::index_column_dispatch(indices, policy)?;
                let input = self.into_inner_with_policy(policy)?;
                $(
                    let $tmp = <Output as sealed::MIterMutDispatch<R>>::column_mut_view_by_index_inner::<$ty>(
                        &output,
                        $idx,
                    )?
                    .ok_or_else(|| Error::Launch {
                        message: "scatter output must match input shape".to_string(),
                    })?;
                    crate::detail::apply::IndexedExprApply::scatter_expr_into(
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
                Indices: MIter<R, Item = MIndex>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let indices =
                    <Indices as sealed::MIterDispatch<R>>::index_column_dispatch(indices, policy)?;
                let input = self.into_inner_with_policy(policy)?;
                let mask = stencil.mask();
                $(
                    let $tmp = <Output as sealed::MIterMutDispatch<R>>::column_mut_view_by_index_inner::<$ty>(
                        &output,
                        $idx,
                    )?
                    .ok_or_else(|| Error::Launch {
                        message: "scatter_where output must match input shape".to_string(),
                    })?;
                    crate::detail::apply::MaskedIndexedExprApply::scatter_where_expr_into(
                        policy,
                        &input.$idx,
                        &indices,
                        &mask,
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
                Right: MIter<R, Item = <Self as MIter<R>>::Item>,
                Eq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_view_with_policy(policy)?;
                let right = right.into_view_with_policy(policy)?;
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
            ) -> Result<Option<MIndex>, Error>
            where
                Right: MIter<R, Item = <Self as MIter<R>>::Item>,
                Eq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_view_with_policy(policy)?;
                let right = right.into_view_with_policy(policy)?;
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
            ) -> Result<Option<MIndex>, Error>
            where
                Needles: MIter<R, Item = <Self as MIter<R>>::Item>,
                Eq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_view_with_policy(policy)?;
                let needles = needles.into_view_with_policy(policy)?;
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
                Right: MIter<R, Item = <Self as MIter<R>>::Item>,
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_view_with_policy(policy)?;
                let right = right.into_view_with_policy(policy)?;
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
                Right: MIter<R, Item = <Self as MIter<R>>::Item>,
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_view_with_policy(policy)?;
                let right = right.into_view_with_policy(policy)?;
                let inner = crate::detail::merge(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<R, Less>::new(),
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn merge_into_dispatch<Right, Output, Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right: Right,
                _less: Less,
                output: Output,
            ) -> Result<(), Error>
            where
                Right: MIter<R, Item = <Self as MIter<R>>::Item>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_view_with_policy(policy)?;
                let right = right.into_view_with_policy(policy)?;
                let inner = crate::detail::merge(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<R, Less>::new(),
                )?;
                output.write_from_inner(policy, inner)
            }

            fn set_union_dispatch<Right, Output, Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right: Right,
                _less: Less,
            ) -> Result<Output, Error>
            where
                Right: MIter<R, Item = <Self as MIter<R>>::Item>,
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_view_with_policy(policy)?;
                let right = right.into_view_with_policy(policy)?;
                let inner = crate::detail::set_union(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<R, Less>::new(),
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn set_union_into_dispatch<Right, Output, Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right: Right,
                _less: Less,
                output: Output,
            ) -> Result<MIndex, Error>
            where
                Right: MIter<R, Item = <Self as MIter<R>>::Item>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_view_with_policy(policy)?;
                let right = right.into_view_with_policy(policy)?;
                let inner = crate::detail::set_union(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<R, Less>::new(),
                )?;
                let len = mindex_from_usize(inner.0.len())?;
                output.write_prefix_from_inner(policy, inner)?;
                Ok(len)
            }

            fn set_intersection_dispatch<Right, Output, Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right: Right,
                _less: Less,
            ) -> Result<Output, Error>
            where
                Right: MIter<R, Item = <Self as MIter<R>>::Item>,
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_view_with_policy(policy)?;
                let right = right.into_view_with_policy(policy)?;
                let inner = crate::detail::set_intersection(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<R, Less>::new(),
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn set_intersection_into_dispatch<Right, Output, Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right: Right,
                _less: Less,
                output: Output,
            ) -> Result<MIndex, Error>
            where
                Right: MIter<R, Item = <Self as MIter<R>>::Item>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_view_with_policy(policy)?;
                let right = right.into_view_with_policy(policy)?;
                let inner = crate::detail::set_intersection(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<R, Less>::new(),
                )?;
                let len = mindex_from_usize(inner.0.len())?;
                output.write_prefix_from_inner(policy, inner)?;
                Ok(len)
            }

            fn set_difference_dispatch<Right, Output, Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right: Right,
                _less: Less,
            ) -> Result<Output, Error>
            where
                Right: MIter<R, Item = <Self as MIter<R>>::Item>,
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_view_with_policy(policy)?;
                let right = right.into_view_with_policy(policy)?;
                let inner = crate::detail::set_difference(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<R, Less>::new(),
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn set_difference_into_dispatch<Right, Output, Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right: Right,
                _less: Less,
                output: Output,
            ) -> Result<MIndex, Error>
            where
                Right: MIter<R, Item = <Self as MIter<R>>::Item>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_view_with_policy(policy)?;
                let right = right.into_view_with_policy(policy)?;
                let inner = crate::detail::set_difference(
                    policy,
                    impl_miter_view!(left; $( $idx ),+),
                    impl_miter_view!(right; $( $idx ),+),
                    KernelOp::<R, Less>::new(),
                )?;
                let len = mindex_from_usize(inner.0.len())?;
                output.write_prefix_from_inner(policy, inner)?;
                Ok(len)
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
            ) -> Result<Option<MIndex>, Error>
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
            ) -> Result<Option<MIndex>, Error>
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
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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
            $( $ty: MStorageElement + 'static, )+
            ($( $ty, )+): MAlloc<R, Inner = ($( crate::detail::DeviceVec<R, $ty>, )+),
                View = ($( crate::detail::device::DeviceColumnView<R, $ty>, )+),
            >,
        {
            type Item = ($( $ty, )+);
            type Inner = ($( crate::detail::device::DeviceColumnMutView<R, $ty>, )+);

            fn len(&self) -> MIndex {
                self.0.len()
            }

            fn into_inner(self) -> Self::Inner {
                ($(
                    crate::detail::device::DeviceColumnMutView::from_slice(
                        &self.$idx.source.inner,
                        usize_from_mindex(self.$idx.offset),
                        usize_from_mindex(self.$idx.len),
                    ),
                )+)
            }

            fn write_from_inner(
                self,
                policy: &crate::detail::CubePolicy<R>,
                inner: <Self::Item as MAlloc<R>>::Inner,
            ) -> Result<(), Error> {
                let output = self.into_inner();
                $(
                    {
                        let input =
                            crate::detail::device::DeviceColumnView::from_column(&inner.$idx);
                        crate::detail::apply::MaterializeWriteApply::new(&output.$idx)
                            .collect_expr(policy, &input)?;
                    }
                )+
                Ok(())
            }

            fn write_prefix_from_inner(
                self,
                policy: &crate::detail::CubePolicy<R>,
                inner: <Self::Item as MAlloc<R>>::Inner,
            ) -> Result<(), Error> {
                let mut output = self.into_inner();
                $(
                    {
                        let input =
                            crate::detail::device::DeviceColumnView::from_column(&inner.$idx);
                        if input.len > output.$idx.len {
                            return Err(Error::LengthMismatch {
                                input: input.len,
                                output: output.$idx.len,
                            });
                        }
                        output.$idx.len = input.len;
                        crate::detail::apply::MaterializeWriteApply::new(&output.$idx)
                            .collect_expr(policy, &input)?;
                    }
                )+
                Ok(())
            }

            fn write_split_from_inner(
                self,
                policy: &crate::detail::CubePolicy<R>,
                selected: <Self::Item as MAlloc<R>>::Inner,
                rejected: <Self::Item as MAlloc<R>>::Inner,
            ) -> Result<(), Error> {
                let output = self.into_inner();
                $(
                    {
                        let selected_input =
                            crate::detail::device::DeviceColumnView::from_column(&selected.$idx);
                        let rejected_input =
                            crate::detail::device::DeviceColumnView::from_column(&rejected.$idx);
                        let input_len = selected_input.len + rejected_input.len;
                        if input_len > output.$idx.len {
                            return Err(Error::LengthMismatch {
                                input: input_len,
                                output: output.$idx.len,
                            });
                        }
                        let mut selected_output = output.$idx.clone();
                        selected_output.len = selected_input.len;
                        crate::detail::apply::MaterializeWriteApply::new(&selected_output)
                            .collect_expr(policy, &selected_input)?;

                        let mut rejected_output = output.$idx.clone();
                        rejected_output.offset += selected_input.len;
                        rejected_output.len = rejected_input.len;
                        crate::detail::apply::MaterializeWriteApply::new(&rejected_output)
                            .collect_expr(policy, &rejected_input)?;
                    }
                )+
                Ok(())
            }

            fn write_where_from_inner(
                self,
                policy: &crate::detail::CubePolicy<R>,
                inner: <Self::Item as MAlloc<R>>::Inner,
                stencil: crate::detail::api::PrecomputedSelection<R>,
            ) -> Result<(), Error>
            {
                let output = self.into_inner();
                $(
                    {
                        let input =
                            crate::detail::device::DeviceColumnView::from_column(&inner.$idx);
                        crate::detail::apply::MaterializeWriteApply::new(&output.$idx)
                            .copy_where_expr(
                                policy,
                                &input,
                                &stencil,
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
                let mask = stencil.mask();
                $(
                    crate::detail::apply::MaskWriteApply::new(&mask, &output.$idx)
                        .replace_value(policy, replacement.$idx)?;
                )+
                Ok(())
            }

            fn fill_inner(
                self,
                policy: &crate::detail::CubePolicy<R>,
                value: Self::Item,
            ) -> Result<(), Error>
            {
                let output = self.into_inner();
                $(
                    crate::detail::apply::FillWriteApply::new(&output.$idx)
                        .fill_value(policy, value.$idx)?;
                )+
                Ok(())
            }
        }

        impl<'a, R, $( $ty ),+> sealed::MIterMutDispatch<R> for $name<$( DeviceSliceMut<'a, R, $ty> ),+>
        where
            R: Runtime,
            $( $ty: MStorageElement + 'static, )+
            ($( $ty, )+): MAlloc<R, Inner = ($( crate::detail::DeviceVec<R, $ty>, )+),
                View = ($( crate::detail::device::DeviceColumnView<R, $ty>, )+),
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
                U: MStorageElement,
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
                            usize_from_mindex(self.$idx.offset),
                            usize_from_mindex(self.$idx.len),
                        )));
                    }
                )+
                Ok(None)
            }

        }
    };
}

macro_rules! impl_wide_miter_soa {
    ($name:ident; $selected_apply:ident; $( $ty:ident : $idx:tt : $tmp:ident ),+) => {
        impl<'a, R, $( $ty ),+> MIter<R> for $name<$( crate::runtime::DeviceSlice<'a, R, $ty> ),+>
        where
            R: Runtime,
            $( $ty: MStorageElement + 'static, )+
            ($( $ty, )+): MAlloc<R, Inner = ($( crate::detail::DeviceVec<R, $ty>, )+),
                View = ($( crate::detail::device::DeviceColumnView<R, $ty>, )+),
            >,
        {
            type Item = ($( $ty, )+);
            type Inner = ($( crate::detail::device::DeviceColumnView<R, $ty>, )+);

            fn len(&self) -> MIndex {
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

            fn into_view_with_policy(
                self,
                policy: &crate::detail::CubePolicy<R>,
            ) -> Result<<Self::Item as MAlloc<R>>::View, Error> {
                let _ = policy;
                Ok(($( self.$idx.column_view(), )+))
            }
        }

        impl<'a, R, $( $ty ),+> sealed::MIterDispatch<R> for $name<$( crate::runtime::DeviceSlice<'a, R, $ty> ),+>
        where
            R: Runtime,
            $( $ty: MStorageElement + 'static, )+
            ($( $ty, )+): MAlloc<R, Inner = ($( crate::detail::DeviceVec<R, $ty>, )+),
                View = ($( crate::detail::device::DeviceColumnView<R, $ty>, )+),
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
                env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
                output: Output,
            ) -> Result<(), Error>
            where
                Output: MIterMut<R>,
                Op: op::UnaryOp<R, <Self as MIter<R>>::Item, Output = Output::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner =
                    impl_wide_transform_dispatch_body!(policy, input, op, env; $( $idx ),+)?;
                output.write_from_inner(policy, inner)
            }

            fn map_dispatch<Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                op: Op,
                env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<Output, Error>
            where
                Output: StorageFromInner<R>,
                Op: op::UnaryOp<R, <Self as MIter<R>>::Item, Output = Output::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = impl_wide_transform_dispatch_body!(
                    policy,
                    input,
                    op,
                    env;
                    $( $idx ),+
                )?;
                Ok(array_from_inner::<R, Output::Item, Output>(inner))
            }

            fn transform_where_dispatch<Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                op: Op,
                env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
                stencil: crate::detail::api::PrecomputedSelection<R>,
                output: Output,
            ) -> Result<(), Error>
            where
                Output: MIterMut<R>,
                Op: op::UnaryOp<R, <Self as MIter<R>>::Item, Output = Output::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner =
                    impl_wide_transform_dispatch_body!(policy, input, op, env; $( $idx ),+)?;
                output.write_where_from_inner(policy, inner, stencil)
            }

            fn reverse_dispatch<Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
            ) -> Result<Output, Error>
            where
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = impl_wide_sort_or_materialize_body!(policy, input; $( $idx: $tmp ),+)?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn reverse_into_dispatch<Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                output: Output,
            ) -> Result<(), Error>
            where
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                $(
                    let ($tmp,) = crate::detail::reverse(policy, (input.$idx,))?;
                )+
                output.write_from_inner(policy, ($($tmp,)+))
            }

            fn sort_into_dispatch<Less, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _less: Less,
                output: Output,
            ) -> Result<(), Error>
            where
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = impl_wide_sort_or_materialize_body!(policy, input; $( $idx: $tmp ),+)?;
                output.write_from_inner(policy, inner)
            }

            fn sort_by_single_key_dispatch<K, Less, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                keys: crate::detail::device::DeviceColumnView<R, K>,
                _less: Less,
            ) -> Result<(KeyOutput, ValueOutput), Error>
            where
                K: MStorageElement + 'static,
                Less: op::BinaryPredicateOp<R, (K,)>,
                KeyOutput: StorageFromInner<R, Item = (K,)>,
                ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let (key_inner, value_inner) =
                    impl_wide_sort_by_single_key_or_materialize_body!(
                        policy, keys, _less, input; $( $idx: $tmp ),+
                    )?;
                Ok((
                    array_from_inner::<R, (K,), KeyOutput>(key_inner),
                    array_from_inner::<R, <Self as MIter<R>>::Item, ValueOutput>(value_inner),
                ))
            }

            fn sort_by_single_key_into_dispatch<K, Less, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                keys: crate::detail::device::DeviceColumnView<R, K>,
                _less: Less,
                key_output: KeyOutput,
                value_output: ValueOutput,
            ) -> Result<(), Error>
            where
                K: MStorageElement + 'static,
                Less: op::BinaryPredicateOp<R, (K,)>,
                KeyOutput: MIterMut<R, Item = (K,)>,
                ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let (key_inner, value_inner) =
                    impl_wide_sort_by_single_key_or_materialize_body!(
                        policy, keys, _less, input; $( $idx: $tmp ),+
                    )?;
                key_output.write_from_inner(policy, key_inner)?;
                value_output.write_from_inner(policy, value_inner)
            }

            fn unique_by_single_key_dispatch<K, Eq, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                keys: crate::detail::device::DeviceColumnView<R, K>,
                _eq: Eq,
            ) -> Result<(KeyOutput, ValueOutput), Error>
            where
                K: MStorageElement + 'static,
                Eq: op::BinaryPredicateOp<R, (K,)>,
                KeyOutput: StorageFromInner<R, Item = (K,)>,
                ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let (key_inner, control) =
                    <(crate::detail::device::DeviceColumnView<R, K>,) as crate::detail::read::KernelUniqueByKeyKeys<
                        KernelOp<R, Eq>,
                    >>::unique_by_key_control((keys,), policy)?;
                let payload_apply =
                    crate::detail::apply::SelectedPayloadApply::new(&control.selection, control.count);
                let value_inner = payload_apply.$selected_apply(policy, $( &input.$idx, )+)?;
                Ok((
                    array_from_inner::<R, (K,), KeyOutput>((key_inner.source,)),
                    array_from_inner::<R, <Self as MIter<R>>::Item, ValueOutput>(value_inner),
                ))
            }

            fn unique_by_single_key_into_dispatch<K, Eq, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                keys: crate::detail::device::DeviceColumnView<R, K>,
                _eq: Eq,
                key_output: KeyOutput,
                value_output: ValueOutput,
            ) -> Result<MIndex, Error>
            where
                K: MStorageElement + 'static,
                Eq: op::BinaryPredicateOp<R, (K,)>,
                KeyOutput: MIterMut<R, Item = (K,)>,
                ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let (key_inner, control) =
                    <(crate::detail::device::DeviceColumnView<R, K>,) as crate::detail::read::KernelUniqueByKeyKeys<
                        KernelOp<R, Eq>,
                    >>::unique_by_key_control((keys,), policy)?;
                let payload_apply =
                    crate::detail::apply::SelectedPayloadApply::new(&control.selection, control.count);
                let value_inner = payload_apply.$selected_apply(policy, $( &input.$idx, )+)?;
                let key_inner = (key_inner.source,);
                let len = mindex_from_usize(key_inner.0.len())?;
                key_output.write_prefix_from_inner(policy, key_inner)?;
                value_output.write_prefix_from_inner(policy, value_inner)?;
                Ok(len)
            }

            fn sort_by_two_key_dispatch<K1, K2, Less, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                first_key: crate::detail::device::DeviceColumnView<R, K1>,
                second_key: crate::detail::device::DeviceColumnView<R, K2>,
                _less: Less,
            ) -> Result<(KeyOutput, ValueOutput), Error>
            where
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                Less: op::BinaryPredicateOp<R, (K1, K2)>,
                KeyOutput: StorageFromInner<R, Item = (K1, K2)>,
                ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let (key_inner, value_inner) = impl_wide_sort_by_two_key_dispatch_body!(
                    policy, first_key, second_key, _less, input; $( $idx ),+
                )?;
                Ok((
                    array_from_inner::<R, (K1, K2), KeyOutput>(key_inner),
                    array_from_inner::<R, <Self as MIter<R>>::Item, ValueOutput>(value_inner),
                ))
            }

            fn sort_by_two_key_into_dispatch<K1, K2, Less, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                first_key: crate::detail::device::DeviceColumnView<R, K1>,
                second_key: crate::detail::device::DeviceColumnView<R, K2>,
                _less: Less,
                key_output: KeyOutput,
                value_output: ValueOutput,
            ) -> Result<(), Error>
            where
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                Less: op::BinaryPredicateOp<R, (K1, K2)>,
                KeyOutput: MIterMut<R, Item = (K1, K2)>,
                ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let (key_inner, value_inner) = impl_wide_sort_by_two_key_dispatch_body!(
                    policy, first_key, second_key, _less, input; $( $idx ),+
                )?;
                key_output.write_from_inner(policy, key_inner)?;
                value_output.write_from_inner(policy, value_inner)
            }

            fn unique_by_two_key_dispatch<K1, K2, Eq, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                first_key: crate::detail::device::DeviceColumnView<R, K1>,
                second_key: crate::detail::device::DeviceColumnView<R, K2>,
                _eq: Eq,
            ) -> Result<(KeyOutput, ValueOutput), Error>
            where
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                Eq: op::BinaryPredicateOp<R, (K1, K2)>,
                KeyOutput: StorageFromInner<R, Item = (K1, K2)>,
                ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                ensure_same_len(input.0.len, first_key.len)?;
                let flags = crate::detail::read::unique_tuple2_flags_read::<
                    crate::detail::device::DeviceColumnView<R, K1>,
                    crate::detail::device::DeviceColumnView<R, K2>,
                    KernelOp<R, Eq>,
                >(policy, &first_key, &second_key)?;
                let selected_rank = crate::detail::primitives::select::selected_rank_from_flags(
                    policy,
                    first_key.len,
                    u32::try_from(first_key.len)
                        .map_err(|_| Error::LengthTooLarge { len: first_key.len })?,
                    flags,
                )?;
                let count =
                    crate::detail::primitives::select::selected_count(policy, &selected_rank)?;
                let payload_apply =
                    crate::detail::apply::SelectedPayloadApply::new(&selected_rank, count);
                let key_inner = (
                    payload_apply.apply_expr(policy, &first_key)?,
                    payload_apply.apply_expr(policy, &second_key)?,
                );
                let ($($tmp,)+) = payload_apply.$selected_apply(policy, $( &input.$idx, )+)?;
                Ok((
                    array_from_inner::<R, (K1, K2), KeyOutput>(key_inner),
                    array_from_inner::<R, <Self as MIter<R>>::Item, ValueOutput>(($($tmp,)+)),
                ))
            }

            fn unique_by_two_key_into_dispatch<K1, K2, Eq, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                first_key: crate::detail::device::DeviceColumnView<R, K1>,
                second_key: crate::detail::device::DeviceColumnView<R, K2>,
                _eq: Eq,
                key_output: KeyOutput,
                value_output: ValueOutput,
            ) -> Result<MIndex, Error>
            where
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                Eq: op::BinaryPredicateOp<R, (K1, K2)>,
                KeyOutput: MIterMut<R, Item = (K1, K2)>,
                ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                ensure_same_len(input.0.len, first_key.len)?;
                let flags = crate::detail::read::unique_tuple2_flags_read::<
                    crate::detail::device::DeviceColumnView<R, K1>,
                    crate::detail::device::DeviceColumnView<R, K2>,
                    KernelOp<R, Eq>,
                >(policy, &first_key, &second_key)?;
                let selected_rank = crate::detail::primitives::select::selected_rank_from_flags(
                    policy,
                    first_key.len,
                    u32::try_from(first_key.len)
                        .map_err(|_| Error::LengthTooLarge { len: first_key.len })?,
                    flags,
                )?;
                let count =
                    crate::detail::primitives::select::selected_count(policy, &selected_rank)?;
                let payload_apply =
                    crate::detail::apply::SelectedPayloadApply::new(&selected_rank, count);
                let key_inner = (
                    payload_apply.apply_expr(policy, &first_key)?,
                    payload_apply.apply_expr(policy, &second_key)?,
                );
                let value_inner = payload_apply.$selected_apply(policy, $( &input.$idx, )+)?;
                let len = mindex_from_usize(count)?;
                key_output.write_prefix_from_inner(policy, key_inner)?;
                value_output.write_prefix_from_inner(policy, value_inner)?;
                Ok(len)
            }

            fn inclusive_scan_by_single_key_dispatch<K, KeyEq, Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                keys: crate::detail::device::DeviceColumnView<R, K>,
                _key_eq: KeyEq,
                _op: Op,
            ) -> Result<Output, Error>
            where
                K: MStorageElement + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K,)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = impl_wide_inclusive_scan_by_single_key_values_body!(
                    policy, keys, input; $( $ty ),+; $( $idx ),+
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn inclusive_scan_by_single_key_into_dispatch<K, KeyEq, Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                keys: crate::detail::device::DeviceColumnView<R, K>,
                _key_eq: KeyEq,
                _op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                K: MStorageElement + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K,)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = impl_wide_inclusive_scan_by_single_key_values_body!(
                    policy, keys, input; $( $ty ),+; $( $idx ),+
                )?;
                output.write_from_inner(policy, inner)
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
                K: MStorageElement + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K,)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = impl_wide_exclusive_scan_by_single_key_values_body!(
                    policy, keys, input, init; $( $ty ),+; $( $idx ),+
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn exclusive_scan_by_single_key_into_dispatch<K, KeyEq, Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                keys: crate::detail::device::DeviceColumnView<R, K>,
                _key_eq: KeyEq,
                init: <Self as MIter<R>>::Item,
                _op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                K: MStorageElement + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K,)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = impl_wide_exclusive_scan_by_single_key_values_body!(
                    policy, keys, input, init; $( $ty ),+; $( $idx ),+
                )?;
                output.write_from_inner(policy, inner)
            }

            fn reduce_by_single_key_dispatch<K, KeyEq, Op, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                keys: crate::detail::device::DeviceColumnView<R, K>,
                _key_eq: KeyEq,
                _init: <Self as MIter<R>>::Item,
                _op: Op,
            ) -> Result<(KeyOutput, ValueOutput), Error>
            where
                K: MStorageElement + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K,)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                KeyOutput: StorageFromInner<R, Item = (K,)>,
                ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let (key_inner, value_inner) = impl_wide_reduce_by_single_key_values_body!(
                    policy, keys, input, _init; $( $ty ),+; $( $idx ),+
                )?;
                Ok((
                    array_from_inner::<R, (K,), KeyOutput>(key_inner),
                    array_from_inner::<R, <Self as MIter<R>>::Item, ValueOutput>(value_inner),
                ))
            }

            fn reduce_by_single_key_into_dispatch<K, KeyEq, Op, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                keys: crate::detail::device::DeviceColumnView<R, K>,
                _key_eq: KeyEq,
                _init: <Self as MIter<R>>::Item,
                _op: Op,
                key_output: KeyOutput,
                value_output: ValueOutput,
            ) -> Result<MIndex, Error>
            where
                K: MStorageElement + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K,)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                KeyOutput: MIterMut<R, Item = (K,)>,
                ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let (key_inner, value_inner) = impl_wide_reduce_by_single_key_values_body!(
                    policy, keys, input, _init; $( $ty ),+; $( $idx ),+
                )?;
                let len = mindex_from_usize(key_inner.0.len())?;
                key_output.write_prefix_from_inner(policy, key_inner)?;
                value_output.write_prefix_from_inner(policy, value_inner)?;
                Ok(len)
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
                RightValues: MIter<R, Item = <Self as MIter<R>>::Item>,
                K: MStorageElement + 'static,
                Less: op::BinaryPredicateOp<R, (K,)>,
                KeyOutput: StorageFromInner<R, Item = (K,)>,
                ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
            {
                let left_values = self.into_view_with_policy(policy)?;
                let right_values = right_values.into_view_with_policy(policy)?;
                let (key_inner, value_inner) = impl_wide_merge_by_single_key_dispatch_body!(
                    policy, left_values, right_values, left_keys, right_keys, Less; $( $idx ),+
                )?;
                Ok((
                    array_from_inner::<R, (K,), KeyOutput>(key_inner),
                    array_from_inner::<R, <Self as MIter<R>>::Item, ValueOutput>(value_inner),
                ))
            }

            fn merge_by_single_key_same_into_dispatch<K, RightValues, Less, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                left_keys: crate::detail::device::DeviceColumnView<R, K>,
                right_keys: crate::detail::device::DeviceColumnView<R, K>,
                right_values: RightValues,
                _less: Less,
                key_output: KeyOutput,
                value_output: ValueOutput,
            ) -> Result<(), Error>
            where
                RightValues: MIter<R, Item = <Self as MIter<R>>::Item>,
                K: MStorageElement + 'static,
                Less: op::BinaryPredicateOp<R, (K,)>,
                KeyOutput: MIterMut<R, Item = (K,)>,
                ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let left_values = self.into_view_with_policy(policy)?;
                let right_values = right_values.into_view_with_policy(policy)?;
                let (key_inner, value_inner) = impl_wide_merge_by_single_key_dispatch_body!(
                    policy, left_values, right_values, left_keys, right_keys, Less; $( $idx ),+
                )?;
                key_output.write_from_inner(policy, key_inner)?;
                value_output.write_from_inner(policy, value_inner)
            }

            fn inclusive_scan_by_two_key_dispatch<K1, K2, KeyEq, Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                first_key: crate::detail::device::DeviceColumnView<R, K1>,
                second_key: crate::detail::device::DeviceColumnView<R, K2>,
                _key_eq: KeyEq,
                _op: Op,
            ) -> Result<Output, Error>
            where
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K1, K2)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = crate::detail::inclusive_scan_by_key(
                    policy,
                    (first_key, second_key),
                    input,
                    KernelOp::<R, KeyEq>::new(),
                    KernelOp::<R, Op>::new(),
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn inclusive_scan_by_two_key_into_dispatch<K1, K2, KeyEq, Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                first_key: crate::detail::device::DeviceColumnView<R, K1>,
                second_key: crate::detail::device::DeviceColumnView<R, K2>,
                _key_eq: KeyEq,
                _op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K1, K2)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = crate::detail::inclusive_scan_by_key(
                    policy,
                    (first_key, second_key),
                    input,
                    KernelOp::<R, KeyEq>::new(),
                    KernelOp::<R, Op>::new(),
                )?;
                output.write_from_inner(policy, inner)
            }

            fn exclusive_scan_by_two_key_dispatch<K1, K2, KeyEq, Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                first_key: crate::detail::device::DeviceColumnView<R, K1>,
                second_key: crate::detail::device::DeviceColumnView<R, K2>,
                _key_eq: KeyEq,
                init: <Self as MIter<R>>::Item,
                _op: Op,
            ) -> Result<Output, Error>
            where
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K1, K2)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = crate::detail::exclusive_scan_by_key(
                    policy,
                    (first_key, second_key),
                    input,
                    KernelOp::<R, KeyEq>::new(),
                    init,
                    KernelOp::<R, Op>::new(),
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn exclusive_scan_by_two_key_into_dispatch<K1, K2, KeyEq, Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                first_key: crate::detail::device::DeviceColumnView<R, K1>,
                second_key: crate::detail::device::DeviceColumnView<R, K2>,
                _key_eq: KeyEq,
                init: <Self as MIter<R>>::Item,
                _op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K1, K2)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = crate::detail::exclusive_scan_by_key(
                    policy,
                    (first_key, second_key),
                    input,
                    KernelOp::<R, KeyEq>::new(),
                    init,
                    KernelOp::<R, Op>::new(),
                )?;
                output.write_from_inner(policy, inner)
            }

            fn reduce_by_two_key_dispatch<K1, K2, KeyEq, Op, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                first_key: crate::detail::device::DeviceColumnView<R, K1>,
                second_key: crate::detail::device::DeviceColumnView<R, K2>,
                _key_eq: KeyEq,
                init: <Self as MIter<R>>::Item,
                _op: Op,
            ) -> Result<(KeyOutput, ValueOutput), Error>
            where
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K1, K2)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                KeyOutput: StorageFromInner<R, Item = (K1, K2)>,
                ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let (key_inner, value_inner) = crate::detail::reduce_by_key(
                    policy,
                    (first_key, second_key),
                    input,
                    KernelOp::<R, KeyEq>::new(),
                    init,
                    KernelOp::<R, Op>::new(),
                )?;
                Ok((
                    array_from_inner::<R, (K1, K2), KeyOutput>(key_inner),
                    array_from_inner::<R, <Self as MIter<R>>::Item, ValueOutput>(value_inner),
                ))
            }

            fn reduce_by_two_key_into_dispatch<K1, K2, KeyEq, Op, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                first_key: crate::detail::device::DeviceColumnView<R, K1>,
                second_key: crate::detail::device::DeviceColumnView<R, K2>,
                _key_eq: KeyEq,
                init: <Self as MIter<R>>::Item,
                _op: Op,
                key_output: KeyOutput,
                value_output: ValueOutput,
            ) -> Result<MIndex, Error>
            where
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K1, K2)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                KeyOutput: MIterMut<R, Item = (K1, K2)>,
                ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let (key_inner, value_inner) = crate::detail::reduce_by_key(
                    policy,
                    (first_key, second_key),
                    input,
                    KernelOp::<R, KeyEq>::new(),
                    init,
                    KernelOp::<R, Op>::new(),
                )?;
                let len = mindex_from_usize(key_inner.0.len())?;
                key_output.write_prefix_from_inner(policy, key_inner)?;
                value_output.write_prefix_from_inner(policy, value_inner)?;
                Ok(len)
            }

            fn merge_by_two_key_same_dispatch<K1, K2, RightValues, Less, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                left_first_key: crate::detail::device::DeviceColumnView<R, K1>,
                left_second_key: crate::detail::device::DeviceColumnView<R, K2>,
                right_first_key: crate::detail::device::DeviceColumnView<R, K1>,
                right_second_key: crate::detail::device::DeviceColumnView<R, K2>,
                right_values: RightValues,
                _less: Less,
            ) -> Result<(KeyOutput, ValueOutput), Error>
            where
                RightValues: MIter<R, Item = <Self as MIter<R>>::Item>,
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                Less: op::BinaryPredicateOp<R, (K1, K2)>,
                KeyOutput: StorageFromInner<R, Item = (K1, K2)>,
                ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
            {
                let left_values = self.into_view_with_policy(policy)?;
                let right_values = right_values.into_view_with_policy(policy)?;
                let (key_inner, value_inner) = crate::detail::merge_by_key(
                    policy,
                    (left_first_key, left_second_key),
                    left_values,
                    (right_first_key, right_second_key),
                    right_values,
                    KernelOp::<R, Less>::new(),
                )?;
                Ok((
                    array_from_inner::<R, (K1, K2), KeyOutput>(key_inner),
                    array_from_inner::<R, <Self as MIter<R>>::Item, ValueOutput>(value_inner),
                ))
            }

            fn merge_by_two_key_same_into_dispatch<K1, K2, RightValues, Less, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                left_first_key: crate::detail::device::DeviceColumnView<R, K1>,
                left_second_key: crate::detail::device::DeviceColumnView<R, K2>,
                right_first_key: crate::detail::device::DeviceColumnView<R, K1>,
                right_second_key: crate::detail::device::DeviceColumnView<R, K2>,
                right_values: RightValues,
                _less: Less,
                key_output: KeyOutput,
                value_output: ValueOutput,
            ) -> Result<(), Error>
            where
                RightValues: MIter<R, Item = <Self as MIter<R>>::Item>,
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                Less: op::BinaryPredicateOp<R, (K1, K2)>,
                KeyOutput: MIterMut<R, Item = (K1, K2)>,
                ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let left_values = self.into_view_with_policy(policy)?;
                let right_values = right_values.into_view_with_policy(policy)?;
                let (key_inner, value_inner) = crate::detail::merge_by_key(
                    policy,
                    (left_first_key, left_second_key),
                    left_values,
                    (right_first_key, right_second_key),
                    right_values,
                    KernelOp::<R, Less>::new(),
                )?;
                key_output.write_from_inner(policy, key_inner)?;
                value_output.write_from_inner(policy, value_inner)
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
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                K3: MStorageElement + 'static,
                Less: op::BinaryPredicateOp<R, (K1, K2, K3)>,
                KeyOutput: StorageFromInner<R, Item = (K1, K2, K3)>,
                ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let (key_inner, value_inner) =
                    impl_wide_sort_by_three_key_dispatch_body!(
                        policy, first_key, second_key, third_key, _less, input; $( $idx ),+
                    )?;
                Ok((
                    array_from_inner::<R, (K1, K2, K3), KeyOutput>(key_inner),
                    array_from_inner::<R, <Self as MIter<R>>::Item, ValueOutput>(value_inner),
                ))
            }

            fn sort_by_three_key_into_dispatch<K1, K2, K3, Less, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                first_key: crate::detail::device::DeviceColumnView<R, K1>,
                second_key: crate::detail::device::DeviceColumnView<R, K2>,
                third_key: crate::detail::device::DeviceColumnView<R, K3>,
                _less: Less,
                key_output: KeyOutput,
                value_output: ValueOutput,
            ) -> Result<(), Error>
            where
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                K3: MStorageElement + 'static,
                Less: op::BinaryPredicateOp<R, (K1, K2, K3)>,
                KeyOutput: MIterMut<R, Item = (K1, K2, K3)>,
                ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let (key_inner, value_inner) =
                    impl_wide_sort_by_three_key_dispatch_body!(
                        policy, first_key, second_key, third_key, _less, input; $( $idx ),+
                    )?;
                key_output.write_from_inner(policy, key_inner)?;
                value_output.write_from_inner(policy, value_inner)
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
                RightValues: MIter<R, Item = <Self as MIter<R>>::Item>,
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                K3: MStorageElement + 'static,
                Less: op::BinaryPredicateOp<R, (K1, K2, K3)>,
                KeyOutput: StorageFromInner<R, Item = (K1, K2, K3)>,
                ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
            {
                let left_values = self.into_view_with_policy(policy)?;
                let right_values = right_values.into_view_with_policy(policy)?;
                let (key_inner, value_inner) = crate::detail::merge_by_key(
                    policy,
                    (left_first_key, left_second_key, left_third_key),
                    left_values,
                    (right_first_key, right_second_key, right_third_key),
                    right_values,
                    KernelOp::<R, Less>::new(),
                )?;
                Ok((
                    array_from_inner::<R, (K1, K2, K3), KeyOutput>(key_inner),
                    array_from_inner::<R, <Self as MIter<R>>::Item, ValueOutput>(value_inner),
                ))
            }

            fn merge_by_three_key_same_into_dispatch<K1, K2, K3, RightValues, Less, KeyOutput, ValueOutput>(
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
                key_output: KeyOutput,
                value_output: ValueOutput,
            ) -> Result<(), Error>
            where
                RightValues: MIter<R, Item = <Self as MIter<R>>::Item>,
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                K3: MStorageElement + 'static,
                Less: op::BinaryPredicateOp<R, (K1, K2, K3)>,
                KeyOutput: MIterMut<R, Item = (K1, K2, K3)>,
                ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let left_values = self.into_view_with_policy(policy)?;
                let right_values = right_values.into_view_with_policy(policy)?;
                let (key_inner, value_inner) = crate::detail::merge_by_key(
                    policy,
                    (left_first_key, left_second_key, left_third_key),
                    left_values,
                    (right_first_key, right_second_key, right_third_key),
                    right_values,
                    KernelOp::<R, Less>::new(),
                )?;
                key_output.write_from_inner(policy, key_inner)?;
                value_output.write_from_inner(policy, value_inner)
            }

            fn copy_where_dispatch<Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                stencil: crate::detail::api::PrecomputedSelection<R>,
            ) -> Result<Output, Error>
            where
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let selected_rank = stencil.selected_rank();
                ensure_same_len(input.0.len, selected_rank.len)?;
                let count = crate::detail::primitives::select::selected_count(
                    policy,
                    selected_rank,
                )?;
                let payload_apply =
                    crate::detail::apply::SelectedPayloadApply::new(selected_rank, count);
                let ($($tmp,)+) = payload_apply.$selected_apply(policy, $( &input.$idx, )+)?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(($($tmp,)+)))
            }

            fn remove_where_dispatch<Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                stencil: crate::detail::api::PrecomputedSelection<R>,
            ) -> Result<Output, Error>
            where
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let selected_rank = stencil.selected_rank();
                ensure_same_len(input.0.len, selected_rank.len)?;
                let count = crate::detail::primitives::select::selected_count(
                    policy,
                    selected_rank,
                )?;
                let payload_apply =
                    crate::detail::apply::SelectedPayloadApply::new(selected_rank, count);
                let ($($tmp,)+) = payload_apply.$selected_apply(policy, $( &input.$idx, )+)?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(($($tmp,)+)))
            }

            fn copy_where_into_dispatch<Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                stencil: crate::detail::api::PrecomputedSelection<R>,
                output: Output,
            ) -> Result<MIndex, Error>
            where
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let selected_rank = stencil.selected_rank();
                ensure_same_len(input.0.len, selected_rank.len)?;
                let count = crate::detail::primitives::select::selected_count(
                    policy,
                    selected_rank,
                )?;
                let payload_apply =
                    crate::detail::apply::SelectedPayloadApply::new(selected_rank, count);
                let inner = payload_apply.$selected_apply(policy, $( &input.$idx, )+)?;
                let len = mindex_from_usize(count)?;
                output.write_prefix_from_inner(policy, inner)?;
                Ok(len)
            }

            fn remove_where_into_dispatch<Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                stencil: crate::detail::api::PrecomputedSelection<R>,
                output: Output,
            ) -> Result<MIndex, Error>
            where
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let selected_rank = stencil.selected_rank();
                ensure_same_len(input.0.len, selected_rank.len)?;
                let count = crate::detail::primitives::select::selected_count(
                    policy,
                    selected_rank,
                )?;
                let payload_apply =
                    crate::detail::apply::SelectedPayloadApply::new(selected_rank, count);
                let inner = payload_apply.$selected_apply(policy, $( &input.$idx, )+)?;
                let len = mindex_from_usize(count)?;
                output.write_prefix_from_inner(policy, inner)?;
                Ok(len)
            }

            fn unique_dispatch<Pred, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _pred: Pred,
            ) -> Result<Output, Error>
            where
                Pred: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = impl_wide_unique_inner_or_materialize_body!(policy, input; $( $idx: $tmp ),+)?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn unique_into_dispatch<Pred, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _pred: Pred,
                output: Output,
            ) -> Result<MIndex, Error>
            where
                Pred: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = impl_wide_unique_inner_or_materialize_body!(policy, input; $( $idx: $tmp ),+)?;
                let len = mindex_from_usize(inner.0.len())?;
                output.write_prefix_from_inner(policy, inner)?;
                Ok(len)
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
                impl_wide_reduce_or_init_body!(policy, input, init; $( $ty ),+; $( $idx: $tmp ),+)
            }

            fn inclusive_scan_dispatch<Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _op: Op,
            ) -> Result<Output, Error>
            where
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = impl_wide_scan_or_materialize_body!(
                    policy, input; $( $ty ),+; $( $idx: $tmp ),+
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
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = impl_wide_exclusive_scan_or_materialize_body!(
                    policy, input, init; $( $ty ),+; $( $idx: $tmp ),+
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn inclusive_scan_into_dispatch<Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = impl_wide_scan_or_materialize_body!(
                    policy, input; $( $ty ),+; $( $idx: $tmp ),+
                )?;
                output.write_from_inner(policy, inner)
            }

            fn exclusive_scan_into_dispatch<Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                init: <Self as MIter<R>>::Item,
                _op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = impl_wide_exclusive_scan_or_materialize_body!(
                    policy, input, init; $( $ty ),+; $( $idx: $tmp ),+
                )?;
                output.write_from_inner(policy, inner)
            }

            fn gather_dispatch<Indices, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                indices: Indices,
                output: Output,
            ) -> Result<(), Error>
            where
                Indices: MIter<R, Item = MIndex>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let indices =
                    <Indices as sealed::MIterDispatch<R>>::index_column_dispatch(indices, policy)?;
                let input = self.into_inner_with_policy(policy)?;
                $(
                    let $tmp = <Output as sealed::MIterMutDispatch<R>>::column_mut_view_by_index_inner::<$ty>(
                        &output,
                        $idx,
                    )?
                    .ok_or_else(|| Error::Launch {
                        message: "gather output must match input shape".to_string(),
                    })?;
                    crate::detail::apply::IndexedExprApply::gather_expr_into(
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
                Indices: MIter<R, Item = MIndex>,
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
            {
                let indices =
                    <Indices as sealed::MIterDispatch<R>>::index_column_dispatch(indices, policy)?;
                let input = self.into_inner_with_policy(policy)?;
                $(
                    let $tmp = crate::detail::apply::IndexedExprApply::gather_expr(
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
                Indices: MIter<R, Item = MIndex>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let indices =
                    <Indices as sealed::MIterDispatch<R>>::index_column_dispatch(indices, policy)?;
                let input = self.into_inner_with_policy(policy)?;
                let mask = stencil.mask();
                $(
                    let $tmp = <Output as sealed::MIterMutDispatch<R>>::column_mut_view_by_index_inner::<$ty>(
                        &output,
                        $idx,
                    )?
                    .ok_or_else(|| Error::Launch {
                        message: "gather_where output must match input shape".to_string(),
                    })?;
                    crate::detail::apply::MaskedIndexedExprApply::gather_where_expr_into(
                        policy,
                        &input.$idx,
                        &indices,
                        &mask,
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
                Indices: MIter<R, Item = MIndex>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let indices =
                    <Indices as sealed::MIterDispatch<R>>::index_column_dispatch(indices, policy)?;
                let input = self.into_inner_with_policy(policy)?;
                $(
                    let $tmp = <Output as sealed::MIterMutDispatch<R>>::column_mut_view_by_index_inner::<$ty>(
                        &output,
                        $idx,
                    )?
                    .ok_or_else(|| Error::Launch {
                        message: "scatter output must match input shape".to_string(),
                    })?;
                    crate::detail::apply::IndexedExprApply::scatter_expr_into(
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
                Indices: MIter<R, Item = MIndex>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let indices =
                    <Indices as sealed::MIterDispatch<R>>::index_column_dispatch(indices, policy)?;
                let input = self.into_inner_with_policy(policy)?;
                let mask = stencil.mask();
                $(
                    let $tmp = <Output as sealed::MIterMutDispatch<R>>::column_mut_view_by_index_inner::<$ty>(
                        &output,
                        $idx,
                    )?
                    .ok_or_else(|| Error::Launch {
                        message: "scatter_where output must match input shape".to_string(),
                    })?;
                    crate::detail::apply::MaskedIndexedExprApply::scatter_where_expr_into(
                        policy,
                        &input.$idx,
                        &indices,
                        &mask,
                        &$tmp,
                    )?;
                )+
                Ok(())
            }

            fn adjacent_difference_dispatch<Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _op: Op,
            ) -> Result<Output, Error>
            where
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = impl_wide_adjacent_difference_dispatch_body!(
                    policy, input; $( $ty ),+; $( $idx ),+
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn adjacent_difference_into_dispatch<Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = impl_wide_adjacent_difference_dispatch_body!(
                    policy, input; $( $ty ),+; $( $idx ),+
                )?;
                output.write_from_inner(policy, inner)
            }

            fn remove_if_dispatch<Pred, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _pred: Pred,
                _env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<Output, Error>
            where
                Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = impl_wide_materialize_inner!(policy, input; $( $idx: $tmp ),+);
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn count_if_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _pred: Pred,
                env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<MIndex, Error>
            where
                Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let selected_rank = impl_wide_predicate_selection_body!(
                    policy, input, env, false; $( $ty ),+; $( $idx ),+
                )?;
                mindex_from_usize(crate::detail::primitives::select::selected_count(
                    policy, &selected_rank,
                )?)
            }

            fn all_of_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _pred: Pred,
                env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<bool, Error>
            where
                Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let len = self.len();
                let input = self.into_inner_with_policy(policy)?;
                let selected_rank = impl_wide_predicate_selection_body!(
                    policy, input, env, false; $( $ty ),+; $( $idx ),+
                )?;
                Ok(mindex_from_usize(crate::detail::primitives::select::selected_count(
                    policy, &selected_rank,
                )?)? == len)
            }

            fn any_of_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _pred: Pred,
                env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<bool, Error>
            where
                Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let selected_rank = impl_wide_predicate_selection_body!(
                    policy, input, env, false; $( $ty ),+; $( $idx ),+
                )?;
                Ok(crate::detail::primitives::select::selected_count(policy, &selected_rank)? != 0)
            }

            fn none_of_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _pred: Pred,
                env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<bool, Error>
            where
                Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let selected_rank = impl_wide_predicate_selection_body!(
                    policy, input, env, false; $( $ty ),+; $( $idx ),+
                )?;
                Ok(crate::detail::primitives::select::selected_count(policy, &selected_rank)? == 0)
            }

            fn find_if_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _pred: Pred,
                env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<Option<MIndex>, Error>
            where
                Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let selected_rank = impl_wide_predicate_selection_body!(
                    policy, input, env, false; $( $ty ),+; $( $idx ),+
                )?;
                let search = crate::detail::control::SearchControl::from_flags(
                    selected_rank.flag.clone(),
                    selected_rank.len,
                    selected_rank.len,
                );
                crate::detail::apply::QueryApply::first_flag(policy, search)
            }

            fn partition_dispatch<Pred, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _pred: Pred,
                _env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<(Output, Output), Error>
            where
                Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let selected_rank = impl_wide_predicate_selection_body!(
                    policy, input, _env, false; $( $ty ),+; $( $idx ),+
                )?;
                let (split_rank, matching_count, failing_count) =
                    crate::detail::primitives::select::split_rank_from_selected(
                        policy,
                        selected_rank,
                )?;
                let payload_apply = crate::detail::apply::SplitPayloadApply::new(
                    &split_rank,
                    matching_count,
                    failing_count,
                );
                let (matching, failing) = payload_apply.$selected_apply(policy, $( &input.$idx, )+)?;
                Ok((
                    array_from_inner::<R, <Self as MIter<R>>::Item, Output>(matching),
                    array_from_inner::<R, <Self as MIter<R>>::Item, Output>(failing),
                ))
            }

            fn partition_into_dispatch<Pred, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _pred: Pred,
                _env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
                output: Output,
            ) -> Result<MIndex, Error>
            where
                Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let selected_rank = impl_wide_predicate_selection_body!(
                    policy, input, _env, false; $( $ty ),+; $( $idx ),+
                )?;
                let (split_rank, matching_count, failing_count) =
                    crate::detail::primitives::select::split_rank_from_selected(
                        policy,
                        selected_rank,
                    )?;
                let payload_apply = crate::detail::apply::SplitPayloadApply::new(
                    &split_rank,
                    matching_count,
                    failing_count,
                );
                let (matching, failing) = payload_apply.$selected_apply(policy, $( &input.$idx, )+)?;
                let split = mindex_from_usize(matching_count)?;
                output.write_split_from_inner(policy, matching, failing)?;
                Ok(split)
            }

            fn is_partitioned_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _pred: Pred,
                env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<bool, Error>
            where
                Pred: op::PredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let selected_rank = impl_wide_predicate_selection_body!(
                    policy, input, env, false; $( $ty ),+; $( $idx ),+
                )?;
                let first_rejected = crate::detail::primitives::search::first_unset_flag(
                    policy,
                    selected_rank.flag.clone(),
                    selected_rank.len,
                    selected_rank.len,
                )?
                .unwrap_or(mindex_from_usize(selected_rank.len)?);
                let selected_count =
                    crate::detail::primitives::select::selected_count(policy, &selected_rank)?;
                Ok(mindex_from_usize(selected_count)? == first_rejected)
            }

            fn min_element_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _less: Less,
            ) -> Result<Option<MIndex>, Error>
            where
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_view_with_policy(policy)?;
                crate::detail::min_element(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    KernelOp::<R, Less>::new(),
                )
            }

            fn max_element_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _less: Less,
            ) -> Result<Option<MIndex>, Error>
            where
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_view_with_policy(policy)?;
                crate::detail::max_element(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    KernelOp::<R, Less>::new(),
                )
            }

            fn minmax_element_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _less: Less,
            ) -> Result<Option<(MIndex, MIndex)>, Error>
            where
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_view_with_policy(policy)?;
                crate::detail::minmax_element(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    KernelOp::<R, Less>::new(),
                )
            }

            fn adjacent_find_dispatch<Pred>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _pred: Pred,
            ) -> Result<Option<MIndex>, Error>
            where
                Pred: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_view_with_policy(policy)?;
                impl_wide_binary_predicate_views!(
                    policy,
                    input,
                    input,
                    Pred,
                    impl_wide_adjacent_find_from_views;
                    $( $ty ),+;
                    $( $idx ),+
                )
            }

            fn lower_bound_dispatch<Values, Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                values: Values,
                _less: Less,
            ) -> Result<crate::runtime::DeviceVec<R, MIndex>, Error>
            where
                Values: MIter<R, Item = <Self as MIter<R>>::Item>,
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_view_with_policy(policy)?;
                let values = values.into_view_with_policy(policy)?;
                Ok(crate::runtime::DeviceVec::from_inner(
                    impl_wide_binary_predicate_views!(
                        policy,
                        input,
                        values,
                        Less,
                        impl_wide_lower_bound_many_from_views;
                        $( $ty ),+;
                        $( $idx ),+
                    )?,
                ))
            }

            fn upper_bound_dispatch<Values, Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                values: Values,
                _less: Less,
            ) -> Result<crate::runtime::DeviceVec<R, MIndex>, Error>
            where
                Values: MIter<R, Item = <Self as MIter<R>>::Item>,
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_view_with_policy(policy)?;
                let values = values.into_view_with_policy(policy)?;
                Ok(crate::runtime::DeviceVec::from_inner(
                    impl_wide_binary_predicate_views!(
                        policy,
                        input,
                        values,
                        Less,
                        impl_wide_upper_bound_many_from_views;
                        $( $ty ),+;
                        $( $idx ),+
                    )?,
                ))
            }

            fn is_sorted_until_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _less: Less,
            ) -> Result<MIndex, Error>
            where
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_view_with_policy(policy)?;
                crate::detail::is_sorted_until(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    KernelOp::<R, Less>::new(),
                )
            }

            fn is_sorted_dispatch<Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                _less: Less,
            ) -> Result<bool, Error>
            where
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_view_with_policy(policy)?;
                crate::detail::is_sorted(
                    policy,
                    impl_miter_view!(input; $( $idx ),+),
                    KernelOp::<R, Less>::new(),
                )
            }

            fn equal_dispatch<Right, Eq>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right: Right,
                _eq: Eq,
            ) -> Result<bool, Error>
            where
                Right: MIter<R, Item = <Self as MIter<R>>::Item>,
                Eq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_view_with_policy(policy)?;
                let right = right.into_view_with_policy(policy)?;
                Ok(impl_wide_binary_predicate_views!(
                    policy,
                    left,
                    right,
                    Eq,
                    impl_wide_mismatch_from_views;
                    $( $ty ),+;
                    $( $idx ),+
                )?
                .is_none())
            }

            fn mismatch_dispatch<Right, Eq>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right: Right,
                _eq: Eq,
            ) -> Result<Option<MIndex>, Error>
            where
                Right: MIter<R, Item = <Self as MIter<R>>::Item>,
                Eq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_view_with_policy(policy)?;
                let right = right.into_view_with_policy(policy)?;
                impl_wide_binary_predicate_views!(
                    policy,
                    left,
                    right,
                    Eq,
                    impl_wide_mismatch_from_views;
                    $( $ty ),+;
                    $( $idx ),+
                )
            }

            fn find_first_of_dispatch<Needles, Eq>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                needles: Needles,
                _eq: Eq,
            ) -> Result<Option<MIndex>, Error>
            where
                Needles: MIter<R, Item = <Self as MIter<R>>::Item>,
                Eq: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let input = self.into_view_with_policy(policy)?;
                let needles = needles.into_view_with_policy(policy)?;
                impl_wide_binary_predicate_views!(
                    policy,
                    input,
                    needles,
                    Eq,
                    impl_wide_find_first_of_from_views;
                    $( $ty ),+;
                    $( $idx ),+
                )
            }

            fn lexicographical_compare_dispatch<Right, Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right: Right,
                _less: Less,
            ) -> Result<bool, Error>
            where
                Right: MIter<R, Item = <Self as MIter<R>>::Item>,
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_view_with_policy(policy)?;
                let right = right.into_view_with_policy(policy)?;
                impl_wide_binary_predicate_views!(
                    policy,
                    left,
                    right,
                    Less,
                    impl_wide_lexicographical_compare_from_views;
                    $( $ty ),+;
                    $( $idx ),+
                )
            }

            fn merge_dispatch<Right, Output, Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right: Right,
                _less: Less,
            ) -> Result<Output, Error>
            where
                Right: MIter<R, Item = <Self as MIter<R>>::Item>,
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_view_with_policy(policy)?;
                let right = right.into_view_with_policy(policy)?;
                $(
                    let $tmp = {
                        let left = crate::detail::apply::MaterializePayloadApply::collect_expr(
                            policy,
                            &left.$idx,
                        )?;
                        let right = crate::detail::apply::MaterializePayloadApply::collect_expr(
                            policy,
                            &right.$idx,
                        )?;
                        crate::detail::apply::ConcatPayloadApply::apply_values(
                            policy,
                            &left,
                            &right,
                        )?
                    };
                )+
                let input = ($($tmp,)+);
                let inner = impl_wide_sort_or_materialize_body!(policy, input; $( $idx: $tmp ),+)?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn merge_into_dispatch<Right, Output, Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right: Right,
                _less: Less,
                output: Output,
            ) -> Result<(), Error>
            where
                Right: MIter<R, Item = <Self as MIter<R>>::Item>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_view_with_policy(policy)?;
                let right = right.into_view_with_policy(policy)?;
                $(
                    let $tmp = {
                        let left = crate::detail::apply::MaterializePayloadApply::collect_expr(
                            policy,
                            &left.$idx,
                        )?;
                        let right = crate::detail::apply::MaterializePayloadApply::collect_expr(
                            policy,
                            &right.$idx,
                        )?;
                        crate::detail::apply::ConcatPayloadApply::apply_values(
                            policy,
                            &left,
                            &right,
                        )?
                    };
                )+
                let input = ($($tmp,)+);
                let inner = impl_wide_sort_or_materialize_body!(policy, input; $( $idx: $tmp ),+)?;
                output.write_from_inner(policy, inner)
            }

            fn set_union_dispatch<Right, Output, Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right: Right,
                _less: Less,
            ) -> Result<Output, Error>
            where
                Right: MIter<R, Item = <Self as MIter<R>>::Item>,
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_view_with_policy(policy)?;
                let right = right.into_view_with_policy(policy)?;
                let right_extra_flags: cubecl::server::Handle = impl_wide_binary_predicate_views!(
                    policy,
                    right,
                    left,
                    Less,
                    impl_wide_set_difference_flags_from_views;
                    $( $ty ),+;
                    $( $idx ),+
                )?;
                let right_extra_rank = crate::detail::primitives::select::selected_rank_from_flags(
                    policy,
                    right.0.len,
                    u32::try_from(right.0.len)
                        .map_err(|_| Error::LengthTooLarge { len: right.0.len })?,
                    right_extra_flags,
                )?;
                let right_extra_count =
                    crate::detail::primitives::select::selected_count(policy, &right_extra_rank)?;
                let right_extra_apply = crate::detail::apply::SelectedPayloadApply::new(
                    &right_extra_rank,
                    right_extra_count,
                );
                let ($($tmp,)+) = right_extra_apply.$selected_apply(policy, $( &right.$idx, )+)?;
                $(
                    let $tmp = {
                        let left = crate::detail::apply::MaterializePayloadApply::collect_expr(
                            policy,
                            &left.$idx,
                        )?;
                        crate::detail::apply::ConcatPayloadApply::apply_values(
                            policy,
                            &left,
                            &$tmp,
                        )?
                    };
                )+
                let input = ($($tmp,)+);
                let inner = impl_wide_sort_or_materialize_body!(policy, input; $( $idx: $tmp ),+)?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn set_union_into_dispatch<Right, Output, Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right: Right,
                _less: Less,
                output: Output,
            ) -> Result<MIndex, Error>
            where
                Right: MIter<R, Item = <Self as MIter<R>>::Item>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_view_with_policy(policy)?;
                let right = right.into_view_with_policy(policy)?;
                let right_extra_flags: cubecl::server::Handle = impl_wide_binary_predicate_views!(
                    policy,
                    right,
                    left,
                    Less,
                    impl_wide_set_difference_flags_from_views;
                    $( $ty ),+;
                    $( $idx ),+
                )?;
                let right_extra_rank = crate::detail::primitives::select::selected_rank_from_flags(
                    policy,
                    right.0.len,
                    u32::try_from(right.0.len)
                        .map_err(|_| Error::LengthTooLarge { len: right.0.len })?,
                    right_extra_flags,
                )?;
                let right_extra_count =
                    crate::detail::primitives::select::selected_count(policy, &right_extra_rank)?;
                let right_extra_apply = crate::detail::apply::SelectedPayloadApply::new(
                    &right_extra_rank,
                    right_extra_count,
                );
                let ($($tmp,)+) = right_extra_apply.$selected_apply(policy, $( &right.$idx, )+)?;
                $(
                    let $tmp = {
                        let left = crate::detail::apply::MaterializePayloadApply::collect_expr(
                            policy,
                            &left.$idx,
                        )?;
                        crate::detail::apply::ConcatPayloadApply::apply_values(
                            policy,
                            &left,
                            &$tmp,
                        )?
                    };
                )+
                let input = ($($tmp,)+);
                let inner = impl_wide_sort_or_materialize_body!(policy, input; $( $idx: $tmp ),+)?;
                let len = mindex_from_usize(inner.0.len())?;
                output.write_prefix_from_inner(policy, inner)?;
                Ok(len)
            }

            fn set_intersection_dispatch<Right, Output, Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right: Right,
                _less: Less,
            ) -> Result<Output, Error>
            where
                Right: MIter<R, Item = <Self as MIter<R>>::Item>,
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_view_with_policy(policy)?;
                let right = right.into_view_with_policy(policy)?;
                let flags: cubecl::server::Handle = impl_wide_binary_predicate_views!(
                    policy,
                    left,
                    right,
                    Less,
                    impl_wide_set_intersection_flags_from_views;
                    $( $ty ),+;
                    $( $idx ),+
                )?;
                let selected_rank = crate::detail::primitives::select::selected_rank_from_flags(
                    policy,
                    left.0.len,
                    u32::try_from(left.0.len)
                        .map_err(|_| Error::LengthTooLarge { len: left.0.len })?,
                    flags,
                )?;
                let count =
                    crate::detail::primitives::select::selected_count(policy, &selected_rank)?;
                let payload_apply =
                    crate::detail::apply::SelectedPayloadApply::new(&selected_rank, count);
                let ($($tmp,)+) = payload_apply.$selected_apply(policy, $( &left.$idx, )+)?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(($($tmp,)+)))
            }

            fn set_intersection_into_dispatch<Right, Output, Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right: Right,
                _less: Less,
                output: Output,
            ) -> Result<MIndex, Error>
            where
                Right: MIter<R, Item = <Self as MIter<R>>::Item>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_view_with_policy(policy)?;
                let right = right.into_view_with_policy(policy)?;
                let flags: cubecl::server::Handle = impl_wide_binary_predicate_views!(
                    policy,
                    left,
                    right,
                    Less,
                    impl_wide_set_intersection_flags_from_views;
                    $( $ty ),+;
                    $( $idx ),+
                )?;
                let selected_rank = crate::detail::primitives::select::selected_rank_from_flags(
                    policy,
                    left.0.len,
                    u32::try_from(left.0.len)
                        .map_err(|_| Error::LengthTooLarge { len: left.0.len })?,
                    flags,
                )?;
                let count =
                    crate::detail::primitives::select::selected_count(policy, &selected_rank)?;
                let payload_apply =
                    crate::detail::apply::SelectedPayloadApply::new(&selected_rank, count);
                let inner = payload_apply.$selected_apply(policy, $( &left.$idx, )+)?;
                let len = mindex_from_usize(count)?;
                output.write_prefix_from_inner(policy, inner)?;
                Ok(len)
            }

            fn set_difference_dispatch<Right, Output, Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right: Right,
                _less: Less,
            ) -> Result<Output, Error>
            where
                Right: MIter<R, Item = <Self as MIter<R>>::Item>,
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_view_with_policy(policy)?;
                let right = right.into_view_with_policy(policy)?;
                let flags: cubecl::server::Handle = impl_wide_binary_predicate_views!(
                    policy,
                    left,
                    right,
                    Less,
                    impl_wide_set_difference_flags_from_views;
                    $( $ty ),+;
                    $( $idx ),+
                )?;
                let selected_rank = crate::detail::primitives::select::selected_rank_from_flags(
                    policy,
                    left.0.len,
                    u32::try_from(left.0.len)
                        .map_err(|_| Error::LengthTooLarge { len: left.0.len })?,
                    flags,
                )?;
                let count =
                    crate::detail::primitives::select::selected_count(policy, &selected_rank)?;
                let payload_apply =
                    crate::detail::apply::SelectedPayloadApply::new(&selected_rank, count);
                let ($($tmp,)+) = payload_apply.$selected_apply(policy, $( &left.$idx, )+)?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(($($tmp,)+)))
            }

            fn set_difference_into_dispatch<Right, Output, Less>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                right: Right,
                _less: Less,
                output: Output,
            ) -> Result<MIndex, Error>
            where
                Right: MIter<R, Item = <Self as MIter<R>>::Item>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
                Less: op::BinaryPredicateOp<R, <Self as MIter<R>>::Item>,
            {
                let left = self.into_view_with_policy(policy)?;
                let right = right.into_view_with_policy(policy)?;
                let flags: cubecl::server::Handle = impl_wide_binary_predicate_views!(
                    policy,
                    left,
                    right,
                    Less,
                    impl_wide_set_difference_flags_from_views;
                    $( $ty ),+;
                    $( $idx ),+
                )?;
                let selected_rank = crate::detail::primitives::select::selected_rank_from_flags(
                    policy,
                    left.0.len,
                    u32::try_from(left.0.len)
                        .map_err(|_| Error::LengthTooLarge { len: left.0.len })?,
                    flags,
                )?;
                let count =
                    crate::detail::primitives::select::selected_count(policy, &selected_rank)?;
                let payload_apply =
                    crate::detail::apply::SelectedPayloadApply::new(&selected_rank, count);
                let inner = payload_apply.$selected_apply(policy, $( &left.$idx, )+)?;
                let len = mindex_from_usize(count)?;
                output.write_prefix_from_inner(policy, inner)?;
                Ok(len)
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
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                K3: MStorageElement + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K1, K2, K3)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = crate::detail::inclusive_scan_by_key(
                    policy,
                    (first_key, second_key, third_key),
                    input,
                    KernelOp::<R, KeyEq>::new(),
                    KernelOp::<R, Op>::new(),
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn inclusive_scan_by_three_key_into_dispatch<K1, K2, K3, KeyEq, Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                first_key: crate::detail::device::DeviceColumnView<R, K1>,
                second_key: crate::detail::device::DeviceColumnView<R, K2>,
                third_key: crate::detail::device::DeviceColumnView<R, K3>,
                _key_eq: KeyEq,
                _op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                K3: MStorageElement + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K1, K2, K3)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = crate::detail::inclusive_scan_by_key(
                    policy,
                    (first_key, second_key, third_key),
                    input,
                    KernelOp::<R, KeyEq>::new(),
                    KernelOp::<R, Op>::new(),
                )?;
                output.write_from_inner(policy, inner)
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
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                K3: MStorageElement + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K1, K2, K3)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = crate::detail::exclusive_scan_by_key(
                    policy,
                    (first_key, second_key, third_key),
                    input,
                    KernelOp::<R, KeyEq>::new(),
                    init,
                    KernelOp::<R, Op>::new(),
                )?;
                Ok(array_from_inner::<R, <Self as MIter<R>>::Item, Output>(inner))
            }

            fn exclusive_scan_by_three_key_into_dispatch<K1, K2, K3, KeyEq, Op, Output>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                first_key: crate::detail::device::DeviceColumnView<R, K1>,
                second_key: crate::detail::device::DeviceColumnView<R, K2>,
                third_key: crate::detail::device::DeviceColumnView<R, K3>,
                _key_eq: KeyEq,
                init: <Self as MIter<R>>::Item,
                _op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                K3: MStorageElement + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K1, K2, K3)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                Output: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let inner = crate::detail::exclusive_scan_by_key(
                    policy,
                    (first_key, second_key, third_key),
                    input,
                    KernelOp::<R, KeyEq>::new(),
                    init,
                    KernelOp::<R, Op>::new(),
                )?;
                output.write_from_inner(policy, inner)
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
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                K3: MStorageElement + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K1, K2, K3)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                KeyOutput: StorageFromInner<R, Item = (K1, K2, K3)>,
                ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let (key_inner, value_inner) = crate::detail::reduce_by_key(
                    policy,
                    (first_key, second_key, third_key),
                    input,
                    KernelOp::<R, KeyEq>::new(),
                    init,
                    KernelOp::<R, Op>::new(),
                )?;
                Ok((
                    array_from_inner::<R, (K1, K2, K3), KeyOutput>(key_inner),
                    array_from_inner::<R, <Self as MIter<R>>::Item, ValueOutput>(value_inner),
                ))
            }

            fn reduce_by_three_key_into_dispatch<K1, K2, K3, KeyEq, Op, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                first_key: crate::detail::device::DeviceColumnView<R, K1>,
                second_key: crate::detail::device::DeviceColumnView<R, K2>,
                third_key: crate::detail::device::DeviceColumnView<R, K3>,
                _key_eq: KeyEq,
                init: <Self as MIter<R>>::Item,
                _op: Op,
                key_output: KeyOutput,
                value_output: ValueOutput,
            ) -> Result<MIndex, Error>
            where
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                K3: MStorageElement + 'static,
                KeyEq: op::BinaryPredicateOp<R, (K1, K2, K3)>,
                Op: op::ReductionOp<R, <Self as MIter<R>>::Item>,
                KeyOutput: MIterMut<R, Item = (K1, K2, K3)>,
                ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
            {
                let input = self.into_inner_with_policy(policy)?;
                let (key_inner, value_inner) = crate::detail::reduce_by_key(
                    policy,
                    (first_key, second_key, third_key),
                    input,
                    KernelOp::<R, KeyEq>::new(),
                    init,
                    KernelOp::<R, Op>::new(),
                )?;
                let len = mindex_from_usize(key_inner.0.len())?;
                key_output.write_prefix_from_inner(policy, key_inner)?;
                value_output.write_prefix_from_inner(policy, value_inner)?;
                Ok(len)
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
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                K3: MStorageElement + 'static,
                Eq: op::BinaryPredicateOp<R, (K1, K2, K3)>,
                KeyOutput: StorageFromInner<R, Item = (K1, K2, K3)>,
                ValueOutput: StorageFromInner<R, Item = <Self as MIter<R>>::Item>,
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
                let selected_rank = crate::detail::primitives::select::selected_rank_from_flags(
                    policy,
                    first_key.len,
                    u32::try_from(first_key.len)
                        .map_err(|_| Error::LengthTooLarge { len: first_key.len })?,
                    flags,
                )?;
                let count =
                    crate::detail::primitives::select::selected_count(policy, &selected_rank)?;
                let payload_apply =
                    crate::detail::apply::SelectedPayloadApply::new(&selected_rank, count);
                let key_inner = (
                    payload_apply.apply_expr(policy, &first_key)?,
                    payload_apply.apply_expr(policy, &second_key)?,
                    payload_apply.apply_expr(policy, &third_key)?,
                );
                let ($($tmp,)+) = payload_apply.$selected_apply(policy, $( &values.$idx, )+)?;
                Ok((
                    array_from_inner::<R, (K1, K2, K3), KeyOutput>(key_inner),
                    array_from_inner::<R, <Self as MIter<R>>::Item, ValueOutput>(($($tmp,)+)),
                ))
            }

            fn unique_by_three_key_into_dispatch<K1, K2, K3, Eq, KeyOutput, ValueOutput>(
                self,
                policy: &crate::detail::CubePolicy<R>,
                first_key: crate::detail::device::DeviceColumnView<R, K1>,
                second_key: crate::detail::device::DeviceColumnView<R, K2>,
                third_key: crate::detail::device::DeviceColumnView<R, K3>,
                _eq: Eq,
                key_output: KeyOutput,
                value_output: ValueOutput,
            ) -> Result<MIndex, Error>
            where
                K1: MStorageElement + 'static,
                K2: MStorageElement + 'static,
                K3: MStorageElement + 'static,
                Eq: op::BinaryPredicateOp<R, (K1, K2, K3)>,
                KeyOutput: MIterMut<R, Item = (K1, K2, K3)>,
                ValueOutput: MIterMut<R, Item = <Self as MIter<R>>::Item>,
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
                let selected_rank = crate::detail::primitives::select::selected_rank_from_flags(
                    policy,
                    first_key.len,
                    u32::try_from(first_key.len)
                        .map_err(|_| Error::LengthTooLarge { len: first_key.len })?,
                    flags,
                )?;
                let count =
                    crate::detail::primitives::select::selected_count(policy, &selected_rank)?;
                let payload_apply =
                    crate::detail::apply::SelectedPayloadApply::new(&selected_rank, count);
                let key_inner = (
                    payload_apply.apply_expr(policy, &first_key)?,
                    payload_apply.apply_expr(policy, &second_key)?,
                    payload_apply.apply_expr(policy, &third_key)?,
                );
                let value_inner = payload_apply.$selected_apply(policy, $( &values.$idx, )+)?;
                let len = mindex_from_usize(count)?;
                key_output.write_prefix_from_inner(policy, key_inner)?;
                value_output.write_prefix_from_inner(policy, value_inner)?;
                Ok(len)
            }
        }
    };
}

impl_miter_soa!(SoA2; A: 0: a, C: 1: c => transform_binary);
impl_miter_soa!(SoA3; A: 0: a, C: 1: c, D: 2: d => transform_ternary);
impl_wide_miter_soa!(SoA4; apply_expr4; A: 0: a, C: 1: c, D: 2: d, E: 3: e);
impl_wide_miter_soa!(SoA5; apply_expr5; A: 0: a, C: 1: c, D: 2: d, E: 3: e, F: 4: f);
impl_wide_miter_soa!(SoA6; apply_expr6; A: 0: a, C: 1: c, D: 2: d, E: 3: e, F: 4: f, G: 5: g);
impl_wide_miter_soa!(SoA7; apply_expr7; A: 0: a, C: 1: c, D: 2: d, E: 3: e, F: 4: f, G: 5: g, H: 6: h);
impl_miter_mut_soa!(SoA2; A: 0, C: 1);
impl_miter_mut_soa!(SoA3; A: 0, C: 1, D: 2);
impl_miter_mut_soa!(SoA4; A: 0, C: 1, D: 2, E: 3);
impl_miter_mut_soa!(SoA5; A: 0, C: 1, D: 2, E: 3, F: 4);
impl_miter_mut_soa!(SoA6; A: 0, C: 1, D: 2, E: 3, F: 4, G: 5);
impl_miter_mut_soa!(SoA7; A: 0, C: 1, D: 2, E: 3, F: 4, G: 5, H: 6);
