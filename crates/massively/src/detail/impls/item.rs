use super::*;

macro_rules! zip_type {
    ($a:ty) => {
        Zip1<$a>
    };
    ($a:ty, $b:ty) => {
        Zip2<$a, $b>
    };
    ($a:ty, $b:ty, $c:ty) => {
        Zip3<$a, $b, $c>
    };
    ($a:ty, $b:ty, $c:ty, $d:ty) => {
        Zip4<$a, $b, $c, $d>
    };
    ($a:ty, $b:ty, $c:ty, $d:ty, $e:ty) => {
        Zip5<$a, $b, $c, $d, $e>
    };
    ($a:ty, $b:ty, $c:ty, $d:ty, $e:ty, $f:ty) => {
        Zip6<$a, $b, $c, $d, $e, $f>
    };
    ($a:ty, $b:ty, $c:ty, $d:ty, $e:ty, $f:ty, $g:ty) => {
        Zip7<$a, $b, $c, $d, $e, $f, $g>
    };
}

macro_rules! zip_value {
    ($a:expr) => {
        Zip1($a)
    };
    ($a:expr, $b:expr) => {
        Zip2($a, $b)
    };
    ($a:expr, $b:expr, $c:expr) => {
        Zip3($a, $b, $c)
    };
    ($a:expr, $b:expr, $c:expr, $d:expr) => {
        Zip4($a, $b, $c, $d)
    };
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr) => {
        Zip5($a, $b, $c, $d, $e)
    };
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr) => {
        Zip6($a, $b, $c, $d, $e, $f)
    };
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr) => {
        Zip7($a, $b, $c, $d, $e, $f, $g)
    };
}

macro_rules! zip_view_from_tuple {
    ($value:expr; $a:ident) => {{
        let ($a,) = $value;
        crate::detail::device::ZipView1 { source: $a }
    }};
    ($value:expr; $a:ident, $b:ident) => {{
        let ($a, $b) = $value;
        crate::detail::device::ZipView2 {
            left: $a,
            right: $b,
        }
    }};
    ($value:expr; $a:ident, $b:ident, $c:ident) => {{
        let ($a, $b, $c) = $value;
        crate::detail::device::ZipView3 {
            first: $a,
            second: $b,
            third: $c,
        }
    }};
}

macro_rules! partition_input_from_tuple {
    ($value:expr; $a:ident) => {{
        let ($a,) = $value;
        ($a,)
    }};
    ($value:expr; $a:ident, $b:ident) => {
        zip_view_from_tuple!($value; $a, $b)
    };
    ($value:expr; $a:ident, $b:ident, $c:ident) => {
        zip_view_from_tuple!($value; $a, $b, $c)
    };
}

macro_rules! scan_input_from_tuple {
    ($value:expr; $a:ident) => {{
        let ($a,) = $value;
        ($a,)
    }};
    ($value:expr; $a:ident, $b:ident) => {
        zip_view_from_tuple!($value; $a, $b)
    };
    ($value:expr; $a:ident, $b:ident, $c:ident) => {
        zip_view_from_tuple!($value; $a, $b, $c)
    };
}

macro_rules! indexed_gather_inner_from_tuple {
    ($policy:expr, $input:expr, $indices:expr; $( $var:ident ),+) => {{
        let ($( $var, )+) = $input;
        Ok(($(
            crate::detail::apply::IndexedExprApply::gather_expr($policy, &$var, &$indices)?,
        )+))
    }};
}

macro_rules! wide_view_from_inner_tuple {
    ($inner:expr; $( $var:ident ),+) => {{
        let ($( $var, )+) = $inner;
        ($(
            crate::detail::device::DeviceColumnView::from_column(&$var),
        )+)
    }};
}

macro_rules! wide_set_selected_inner_from_tuple {
    ($policy:expr, $left:expr, $right:expr, $keep:expr; $less:ty; $a:ident, $b:ident, $c:ident, $d:ident) => {{
        let ($a, $b, $c, $d) = $left;
        let (right_a, right_b, right_c, right_d) = $right;
        let len = $a.len;
        let dummy_e = crate::detail::primitives::range::indices_mindex($policy, len)?;
        let dummy_f = crate::detail::primitives::range::indices_mindex($policy, len)?;
        let dummy_g = crate::detail::primitives::range::indices_mindex($policy, len)?;
        let right_dummy_e = crate::detail::primitives::range::indices_mindex($policy, right_a.len)?;
        let right_dummy_f = crate::detail::primitives::range::indices_mindex($policy, right_a.len)?;
        let right_dummy_g = crate::detail::primitives::range::indices_mindex($policy, right_a.len)?;
        let dummy_e = crate::detail::device::DeviceColumnView::from_column(&dummy_e);
        let dummy_f = crate::detail::device::DeviceColumnView::from_column(&dummy_f);
        let dummy_g = crate::detail::device::DeviceColumnView::from_column(&dummy_g);
        let right_dummy_e = crate::detail::device::DeviceColumnView::from_column(&right_dummy_e);
        let right_dummy_f = crate::detail::device::DeviceColumnView::from_column(&right_dummy_f);
        let right_dummy_g = crate::detail::device::DeviceColumnView::from_column(&right_dummy_g);
        let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
        let flags = crate::detail::read::tuple7_view_set_membership_flags_read::<
            _,
            _,
            _,
            _,
            _,
            MIndex,
            MIndex,
            MIndex,
            crate::detail::api::Tuple4AsTuple7BinaryPredicateOp<$less>,
        >(
            $policy,
            &$a,
            &$b,
            &$c,
            &$d,
            &dummy_e,
            &dummy_f,
            &dummy_g,
            &right_a,
            &right_b,
            &right_c,
            &right_d,
            &right_dummy_e,
            &right_dummy_f,
            &right_dummy_g,
            $keep,
        )?;
        let selected_rank = crate::detail::primitives::select::selected_rank_from_flags(
            $policy, len, len_u32, flags,
        )?;
        let count = crate::detail::primitives::select::selected_count($policy, &selected_rank)?;
        let apply = crate::detail::apply::SelectedPayloadApply::new(&selected_rank, count);
        Ok((apply.apply_expr4($policy, &$a, &$b, &$c, &$d)?, count))
    }};
    ($policy:expr, $left:expr, $right:expr, $keep:expr; $less:ty; $a:ident, $b:ident, $c:ident, $d:ident, $e:ident) => {{
        let ($a, $b, $c, $d, $e) = $left;
        let (right_a, right_b, right_c, right_d, right_e) = $right;
        let len = $a.len;
        let dummy_f = crate::detail::primitives::range::indices_mindex($policy, len)?;
        let dummy_g = crate::detail::primitives::range::indices_mindex($policy, len)?;
        let right_dummy_f = crate::detail::primitives::range::indices_mindex($policy, right_a.len)?;
        let right_dummy_g = crate::detail::primitives::range::indices_mindex($policy, right_a.len)?;
        let dummy_f = crate::detail::device::DeviceColumnView::from_column(&dummy_f);
        let dummy_g = crate::detail::device::DeviceColumnView::from_column(&dummy_g);
        let right_dummy_f = crate::detail::device::DeviceColumnView::from_column(&right_dummy_f);
        let right_dummy_g = crate::detail::device::DeviceColumnView::from_column(&right_dummy_g);
        let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
        let flags = crate::detail::read::tuple7_view_set_membership_flags_read::<
            _,
            _,
            _,
            _,
            _,
            _,
            MIndex,
            MIndex,
            crate::detail::api::Tuple5AsTuple7BinaryPredicateOp<$less>,
        >(
            $policy,
            &$a,
            &$b,
            &$c,
            &$d,
            &$e,
            &dummy_f,
            &dummy_g,
            &right_a,
            &right_b,
            &right_c,
            &right_d,
            &right_e,
            &right_dummy_f,
            &right_dummy_g,
            $keep,
        )?;
        let selected_rank = crate::detail::primitives::select::selected_rank_from_flags(
            $policy, len, len_u32, flags,
        )?;
        let count = crate::detail::primitives::select::selected_count($policy, &selected_rank)?;
        let apply = crate::detail::apply::SelectedPayloadApply::new(&selected_rank, count);
        Ok((apply.apply_expr5($policy, &$a, &$b, &$c, &$d, &$e)?, count))
    }};
    ($policy:expr, $left:expr, $right:expr, $keep:expr; $less:ty; $a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident) => {{
        let ($a, $b, $c, $d, $e, $f) = $left;
        let (right_a, right_b, right_c, right_d, right_e, right_f) = $right;
        let len = $a.len;
        let dummy_g = crate::detail::primitives::range::indices_mindex($policy, len)?;
        let right_dummy_g = crate::detail::primitives::range::indices_mindex($policy, right_a.len)?;
        let dummy_g = crate::detail::device::DeviceColumnView::from_column(&dummy_g);
        let right_dummy_g = crate::detail::device::DeviceColumnView::from_column(&right_dummy_g);
        let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
        let flags = crate::detail::read::tuple7_view_set_membership_flags_read::<
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            MIndex,
            crate::detail::api::Tuple6AsTuple7BinaryPredicateOp<$less>,
        >(
            $policy,
            &$a,
            &$b,
            &$c,
            &$d,
            &$e,
            &$f,
            &dummy_g,
            &right_a,
            &right_b,
            &right_c,
            &right_d,
            &right_e,
            &right_f,
            &right_dummy_g,
            $keep,
        )?;
        let selected_rank = crate::detail::primitives::select::selected_rank_from_flags(
            $policy, len, len_u32, flags,
        )?;
        let count = crate::detail::primitives::select::selected_count($policy, &selected_rank)?;
        let apply = crate::detail::apply::SelectedPayloadApply::new(&selected_rank, count);
        Ok((
            apply.apply_expr6($policy, &$a, &$b, &$c, &$d, &$e, &$f)?,
            count,
        ))
    }};
    ($policy:expr, $left:expr, $right:expr, $keep:expr; $less:ty; $a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident, $g:ident) => {{
        let ($a, $b, $c, $d, $e, $f, $g) = $left;
        let (right_a, right_b, right_c, right_d, right_e, right_f, right_g) = $right;
        let len = $a.len;
        let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
        let flags = crate::detail::read::tuple7_view_set_membership_flags_read::<
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            $less,
        >(
            $policy, &$a, &$b, &$c, &$d, &$e, &$f, &$g, &right_a, &right_b, &right_c, &right_d,
            &right_e, &right_f, &right_g, $keep,
        )?;
        let selected_rank = crate::detail::primitives::select::selected_rank_from_flags(
            $policy, len, len_u32, flags,
        )?;
        let count = crate::detail::primitives::select::selected_count($policy, &selected_rank)?;
        let apply = crate::detail::apply::SelectedPayloadApply::new(&selected_rank, count);
        Ok((
            apply.apply_expr7($policy, &$a, &$b, &$c, &$d, &$e, &$f, &$g)?,
            count,
        ))
    }};
}

macro_rules! indexed_apply_into_from_tuple {
    (
        $method:path,
        $policy:expr,
        $input:expr,
        $indices:expr,
        $output:expr;
        $( $ty:ident : $var:ident => $index:expr ),+
    ) => {{
        let ($( $var, )+) = $input;
        $(
            let out = <Output as crate::iter::MIterMut<R>>::column_mut_view_by_index_inner::<$ty>(
                &$output,
                $index,
            )?
            .ok_or_else(|| Error::Launch {
                message: "indexed output must match input shape".to_string(),
            })?;
            $method($policy, &$var, &$indices, &out)?;
        )+
        Ok(())
    }};
    (
        $method:path,
        $policy:expr,
        $input:expr,
        $indices:expr,
        $output:expr,
        $mask:expr;
        $( $ty:ident : $var:ident => $index:expr ),+
    ) => {{
        let ($( $var, )+) = $input;
        $(
            let out = <Output as crate::iter::MIterMut<R>>::column_mut_view_by_index_inner::<$ty>(
                &$output,
                $index,
            )?
            .ok_or_else(|| Error::Launch {
                message: "indexed output must match input shape".to_string(),
            })?;
            $method($policy, &$var, &$indices, $mask, &out)?;
        )+
        Ok(())
    }};
}

macro_rules! indexed_apply_arity {
    ($method:path, $policy:expr, $input:expr, $indices:expr, $output:expr $(, $mask:expr)?; A: $a:ident) => {
        indexed_apply_into_from_tuple!(
            $method,
            $policy, $input, $indices, $output $(, $mask)?;
            A: $a => 0
        )
    };
    ($method:path, $policy:expr, $input:expr, $indices:expr, $output:expr $(, $mask:expr)?; A: $a:ident, B: $b:ident) => {
        indexed_apply_into_from_tuple!(
            $method,
            $policy, $input, $indices, $output $(, $mask)?;
            A: $a => 0, B: $b => 1
        )
    };
    ($method:path, $policy:expr, $input:expr, $indices:expr, $output:expr $(, $mask:expr)?; A: $a:ident, B: $b:ident, C: $c:ident) => {
        indexed_apply_into_from_tuple!(
            $method,
            $policy, $input, $indices, $output $(, $mask)?;
            A: $a => 0, B: $b => 1, C: $c => 2
        )
    };
    ($method:path, $policy:expr, $input:expr, $indices:expr, $output:expr $(, $mask:expr)?; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident) => {
        indexed_apply_into_from_tuple!(
            $method,
            $policy, $input, $indices, $output $(, $mask)?;
            A: $a => 0, B: $b => 1, C: $c => 2, D: $d => 3
        )
    };
    ($method:path, $policy:expr, $input:expr, $indices:expr, $output:expr $(, $mask:expr)?; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident, E: $e:ident) => {
        indexed_apply_into_from_tuple!(
            $method,
            $policy, $input, $indices, $output $(, $mask)?;
            A: $a => 0, B: $b => 1, C: $c => 2, D: $d => 3, E: $e => 4
        )
    };
    ($method:path, $policy:expr, $input:expr, $indices:expr, $output:expr $(, $mask:expr)?; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident, E: $e:ident, F: $f:ident) => {
        indexed_apply_into_from_tuple!(
            $method,
            $policy, $input, $indices, $output $(, $mask)?;
            A: $a => 0, B: $b => 1, C: $c => 2, D: $d => 3, E: $e => 4, F: $f => 5
        )
    };
    ($method:path, $policy:expr, $input:expr, $indices:expr, $output:expr $(, $mask:expr)?; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident, E: $e:ident, F: $f:ident, G: $g:ident) => {
        indexed_apply_into_from_tuple!(
            $method,
            $policy, $input, $indices, $output $(, $mask)?;
            A: $a => 0, B: $b => 1, C: $c => 2, D: $d => 3, E: $e => 4, F: $f => 5, G: $g => 6
        )
    };
}

macro_rules! tuple_set_less {
    ($less:ident; $a:ident) => {
        crate::detail::api::Tuple1Less::<KernelOp<R, $less>>::default()
    };
    ($less:ident; $a:ident, $( $rest:ident ),+) => {
        KernelOp::<R, $less>::new()
    };
}

macro_rules! wide_reverse_from_tuple {
    ($policy:expr, $input:expr; $( $var:ident ),+) => {{
        let ($( $var, )+) = $input;
        $(
            let ($var,) = crate::detail::reverse($policy, ($var,))?;
        )+
        Ok(($( $var, )+))
    }};
}

macro_rules! wide_sort_from_tuple {
    ($policy:expr, $input:expr; $a:ident, $b:ident, $c:ident, $d:ident) => {{
        let ($a, $b, $c, $d) = $input;
        let dummy4 = crate::detail::primitives::range::indices_mindex($policy, $a.len)?;
        let dummy5 = crate::detail::primitives::range::indices_mindex($policy, $a.len)?;
        let dummy6 = crate::detail::primitives::range::indices_mindex($policy, $a.len)?;
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
            &$a,
            &$b,
            &$c,
            &$d,
            &dummy4,
            &dummy5,
            &dummy6,
            crate::op::GpuOp::<
                crate::detail::api::Tuple4AsTuple7BinaryPredicateOp<KernelOp<R, Less>>,
            >::new(),
        )?;
        let control = crate::detail::control::OrderingControl::from_sorted_indices(&indices)?;
        let apply = crate::detail::apply::PermutationPayloadApply::new(control.permutation());
        apply.apply_expr4($policy, &$a, &$b, &$c, &$d)
    }};
    ($policy:expr, $input:expr; $a:ident, $b:ident, $c:ident, $d:ident, $e:ident) => {{
        let ($a, $b, $c, $d, $e) = $input;
        let dummy5 = crate::detail::primitives::range::indices_mindex($policy, $a.len)?;
        let dummy6 = crate::detail::primitives::range::indices_mindex($policy, $a.len)?;
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
            &$a,
            &$b,
            &$c,
            &$d,
            &$e,
            &dummy5,
            &dummy6,
            crate::op::GpuOp::<
                crate::detail::api::Tuple5AsTuple7BinaryPredicateOp<KernelOp<R, Less>>,
            >::new(),
        )?;
        let control = crate::detail::control::OrderingControl::from_sorted_indices(&indices)?;
        let apply = crate::detail::apply::PermutationPayloadApply::new(control.permutation());
        apply.apply_expr5($policy, &$a, &$b, &$c, &$d, &$e)
    }};
    ($policy:expr, $input:expr; $a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident) => {{
        let ($a, $b, $c, $d, $e, $f) = $input;
        let dummy6 = crate::detail::primitives::range::indices_mindex($policy, $a.len)?;
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
            &$a,
            &$b,
            &$c,
            &$d,
            &$e,
            &$f,
            &dummy6,
            crate::op::GpuOp::<
                crate::detail::api::Tuple6AsTuple7BinaryPredicateOp<KernelOp<R, Less>>,
            >::new(),
        )?;
        let control = crate::detail::control::OrderingControl::from_sorted_indices(&indices)?;
        let apply = crate::detail::apply::PermutationPayloadApply::new(control.permutation());
        apply.apply_expr6($policy, &$a, &$b, &$c, &$d, &$e, &$f)
    }};
    ($policy:expr, $input:expr; $a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident, $g:ident) => {{
        let ($a, $b, $c, $d, $e, $f, $g) = $input;
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
            &$a,
            &$b,
            &$c,
            &$d,
            &$e,
            &$f,
            &$g,
            crate::op::GpuOp::<KernelOp<R, Less>>::new(),
        )?;
        let control = crate::detail::control::OrderingControl::from_sorted_indices(&indices)?;
        let apply = crate::detail::apply::PermutationPayloadApply::new(control.permutation());
        apply.apply_expr7($policy, &$a, &$b, &$c, &$d, &$e, &$f, &$g)
    }};
}

macro_rules! wide_merge_from_tuple {
    ($policy:expr, $left:expr, $right:expr; $a:ident, $b:ident, $c:ident, $d:ident) => {{
        let ($a, $b, $c, $d) = $left;
        let (ra, rb, rc, rd) = $right;
        let $a = wide_merge_column!($policy, $a, ra)?;
        let $b = wide_merge_column!($policy, $b, rb)?;
        let $c = wide_merge_column!($policy, $c, rc)?;
        let $d = wide_merge_column!($policy, $d, rd)?;
        let input = ($a, $b, $c, $d);
        wide_sort_from_tuple!($policy, input; $a, $b, $c, $d)
    }};
    ($policy:expr, $left:expr, $right:expr; $a:ident, $b:ident, $c:ident, $d:ident, $e:ident) => {{
        let ($a, $b, $c, $d, $e) = $left;
        let (ra, rb, rc, rd, re) = $right;
        let $a = wide_merge_column!($policy, $a, ra)?;
        let $b = wide_merge_column!($policy, $b, rb)?;
        let $c = wide_merge_column!($policy, $c, rc)?;
        let $d = wide_merge_column!($policy, $d, rd)?;
        let $e = wide_merge_column!($policy, $e, re)?;
        let input = ($a, $b, $c, $d, $e);
        wide_sort_from_tuple!($policy, input; $a, $b, $c, $d, $e)
    }};
    ($policy:expr, $left:expr, $right:expr; $a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident) => {{
        let ($a, $b, $c, $d, $e, $f) = $left;
        let (ra, rb, rc, rd, re, rf) = $right;
        let $a = wide_merge_column!($policy, $a, ra)?;
        let $b = wide_merge_column!($policy, $b, rb)?;
        let $c = wide_merge_column!($policy, $c, rc)?;
        let $d = wide_merge_column!($policy, $d, rd)?;
        let $e = wide_merge_column!($policy, $e, re)?;
        let $f = wide_merge_column!($policy, $f, rf)?;
        let input = ($a, $b, $c, $d, $e, $f);
        wide_sort_from_tuple!($policy, input; $a, $b, $c, $d, $e, $f)
    }};
    ($policy:expr, $left:expr, $right:expr; $a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident, $g:ident) => {{
        let ($a, $b, $c, $d, $e, $f, $g) = $left;
        let (ra, rb, rc, rd, re, rf, rg) = $right;
        let $a = wide_merge_column!($policy, $a, ra)?;
        let $b = wide_merge_column!($policy, $b, rb)?;
        let $c = wide_merge_column!($policy, $c, rc)?;
        let $d = wide_merge_column!($policy, $d, rd)?;
        let $e = wide_merge_column!($policy, $e, re)?;
        let $f = wide_merge_column!($policy, $f, rf)?;
        let $g = wide_merge_column!($policy, $g, rg)?;
        let input = ($a, $b, $c, $d, $e, $f, $g);
        wide_sort_from_tuple!($policy, input; $a, $b, $c, $d, $e, $f, $g)
    }};
}

macro_rules! wide_merge_column {
    ($policy:expr, $left:expr, $right:expr) => {{
        let left = crate::detail::apply::MaterializePayloadApply::collect_expr($policy, &$left)?;
        let right = crate::detail::apply::MaterializePayloadApply::collect_expr($policy, &$right)?;
        crate::detail::apply::ConcatPayloadApply::apply_values($policy, &left, &right)
    }};
}

macro_rules! wide_unique_by_key_values_from_tuple {
    ($policy:expr, $values:expr, $control:expr; $( $var:ident ),+) => {{
        let ($( $var, )+) = $values;
        $(
            <_ as crate::detail::device::KernelColumn>::validate(&$var)?;
        )+
        let apply = crate::detail::apply::SelectedPayloadApply::new(
            &$control.selection,
            $control.count,
        );
        wide_unique_by_key_values_from_tuple!(@apply apply, $policy; $( $var ),+)
    }};
    (@apply $apply:ident, $policy:expr; $a:ident, $b:ident, $c:ident, $d:ident) => {
        $apply.apply_expr4($policy, &$a, &$b, &$c, &$d)
    };
    (@apply $apply:ident, $policy:expr; $a:ident, $b:ident, $c:ident, $d:ident, $e:ident) => {
        $apply.apply_expr5($policy, &$a, &$b, &$c, &$d, &$e)
    };
    (@apply $apply:ident, $policy:expr; $a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident) => {
        $apply.apply_expr6($policy, &$a, &$b, &$c, &$d, &$e, &$f)
    };
    (@apply $apply:ident, $policy:expr; $a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident, $g:ident) => {
        $apply.apply_expr7($policy, &$a, &$b, &$c, &$d, &$e, &$f, &$g)
    };
}

macro_rules! wide_unique_from_tuple {
    ($policy:expr, $input:expr; $a:ident, $b:ident, $c:ident, $d:ident) => {{
        let ($a, $b, $c, $d) = $input;
        let len = crate::detail::device::KernelColumn::len(&$a);
        let dummy_e = crate::detail::primitives::range::indices_mindex($policy, len)?;
        let dummy_f = crate::detail::primitives::range::indices_mindex($policy, len)?;
        let dummy_g = crate::detail::primitives::range::indices_mindex($policy, len)?;
        let dummy_e = crate::detail::device::DeviceColumnView::from_column(&dummy_e);
        let dummy_f = crate::detail::device::DeviceColumnView::from_column(&dummy_f);
        let dummy_g = crate::detail::device::DeviceColumnView::from_column(&dummy_g);
        let flags = crate::detail::read::unique_tuple7_flags_read::<
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            crate::detail::api::Tuple4AsTuple7BinaryPredicateOp<KernelOp<R, Pred>>,
        >(
            $policy, &$a, &$b, &$c, &$d, &dummy_e, &dummy_f, &dummy_g,
        )?;
        let len_u32 = mindex_from_usize(len)?;
        let selected_rank =
            crate::detail::primitives::select::selected_rank_from_flags($policy, len, len_u32, flags)?;
        let count = crate::detail::primitives::select::selected_count($policy, &selected_rank)?;
        let apply = crate::detail::apply::SelectedPayloadApply::new(&selected_rank, count);
        apply.apply_expr4($policy, &$a, &$b, &$c, &$d)
    }};
    ($policy:expr, $input:expr; $a:ident, $b:ident, $c:ident, $d:ident, $e:ident) => {{
        let ($a, $b, $c, $d, $e) = $input;
        let len = crate::detail::device::KernelColumn::len(&$a);
        let dummy_f = crate::detail::primitives::range::indices_mindex($policy, len)?;
        let dummy_g = crate::detail::primitives::range::indices_mindex($policy, len)?;
        let dummy_f = crate::detail::device::DeviceColumnView::from_column(&dummy_f);
        let dummy_g = crate::detail::device::DeviceColumnView::from_column(&dummy_g);
        let flags = crate::detail::read::unique_tuple7_flags_read::<
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            crate::detail::api::Tuple5AsTuple7BinaryPredicateOp<KernelOp<R, Pred>>,
        >(
            $policy, &$a, &$b, &$c, &$d, &$e, &dummy_f, &dummy_g,
        )?;
        let len_u32 = mindex_from_usize(len)?;
        let selected_rank =
            crate::detail::primitives::select::selected_rank_from_flags($policy, len, len_u32, flags)?;
        let count = crate::detail::primitives::select::selected_count($policy, &selected_rank)?;
        let apply = crate::detail::apply::SelectedPayloadApply::new(&selected_rank, count);
        apply.apply_expr5($policy, &$a, &$b, &$c, &$d, &$e)
    }};
    ($policy:expr, $input:expr; $a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident) => {{
        let ($a, $b, $c, $d, $e, $f) = $input;
        let len = crate::detail::device::KernelColumn::len(&$a);
        let dummy_g = crate::detail::primitives::range::indices_mindex($policy, len)?;
        let dummy_g = crate::detail::device::DeviceColumnView::from_column(&dummy_g);
        let flags = crate::detail::read::unique_tuple7_flags_read::<
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            crate::detail::api::Tuple6AsTuple7BinaryPredicateOp<KernelOp<R, Pred>>,
        >(
            $policy, &$a, &$b, &$c, &$d, &$e, &$f, &dummy_g,
        )?;
        let len_u32 = mindex_from_usize(len)?;
        let selected_rank =
            crate::detail::primitives::select::selected_rank_from_flags($policy, len, len_u32, flags)?;
        let count = crate::detail::primitives::select::selected_count($policy, &selected_rank)?;
        let apply = crate::detail::apply::SelectedPayloadApply::new(&selected_rank, count);
        apply.apply_expr6($policy, &$a, &$b, &$c, &$d, &$e, &$f)
    }};
    ($policy:expr, $input:expr; $a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident, $g:ident) => {{
        let ($a, $b, $c, $d, $e, $f, $g) = $input;
        let len = crate::detail::device::KernelColumn::len(&$a);
        let flags = crate::detail::read::unique_tuple7_flags_read::<
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            KernelOp<R, Pred>,
        >(
            $policy, &$a, &$b, &$c, &$d, &$e, &$f, &$g,
        )?;
        let len_u32 = mindex_from_usize(len)?;
        let selected_rank =
            crate::detail::primitives::select::selected_rank_from_flags($policy, len, len_u32, flags)?;
        let count = crate::detail::primitives::select::selected_count($policy, &selected_rank)?;
        let apply = crate::detail::apply::SelectedPayloadApply::new(&selected_rank, count);
        apply.apply_expr7($policy, &$a, &$b, &$c, &$d, &$e, &$f, &$g)
    }};
}

macro_rules! wide_reduce_from_tuple {
    ($policy:expr, $init:expr; $a:ident, $b:ident, $c:ident, $d:ident) => {
        crate::detail::apply::LinearReduceApply::apply_views4::<
            R,
            A,
            B,
            C,
            D,
            KernelOp<R, Op>,
        >($policy, &$a, &$b, &$c, &$d, $init)
    };
    ($policy:expr, $init:expr; $a:ident, $b:ident, $c:ident, $d:ident, $e:ident) => {
        crate::detail::apply::LinearReduceApply::apply_views5::<
            R,
            A,
            B,
            C,
            D,
            E,
            KernelOp<R, Op>,
        >($policy, &$a, &$b, &$c, &$d, &$e, $init)
    };
    ($policy:expr, $init:expr; $a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident) => {
        crate::detail::apply::LinearReduceApply::apply_views6::<
            R,
            A,
            B,
            C,
            D,
            E,
            F,
            KernelOp<R, Op>,
        >($policy, &$a, &$b, &$c, &$d, &$e, &$f, $init)
    };
    ($policy:expr, $init:expr; $a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident, $g:ident) => {
        crate::detail::apply::LinearReduceApply::apply_views7::<
            R,
            A,
            B,
            C,
            D,
            E,
            F,
            G,
            KernelOp<R, Op>,
        >($policy, &$a, &$b, &$c, &$d, &$e, &$f, &$g, $init)
    };
}

macro_rules! wide_adjacent_difference_from_tuple {
    ($policy:expr, $input:expr; $( $var:ident ),+) => {{
        let ($( $var, )+) = $input;
        wide_scan_apply_from_tuple!(@adjacent $policy; $( $var ),+)
    }};
}

macro_rules! wide_inclusive_scan_from_tuple {
    ($policy:expr, $input:expr; $( $var:ident ),+) => {{
        let ($( $var, )+) = $input;
        wide_scan_apply_from_tuple!(@inclusive $policy; $( $var ),+)
    }};
}

macro_rules! wide_exclusive_scan_from_tuple {
    ($policy:expr, $input:expr, $init:expr; $( $var:ident ),+) => {{
        let ($( $var, )+) = $input;
        wide_scan_apply_from_tuple!(@exclusive $policy, $init; $( $var ),+)
    }};
}

macro_rules! wide_scan_apply_from_tuple {
    (@adjacent $policy:expr; $a:ident, $b:ident, $c:ident, $d:ident) => {
        crate::detail::apply::LinearScanApply::adjacent_views4::<R, A, B, C, D, KernelOp<R, Op>>(
            $policy, &$a, &$b, &$c, &$d,
        )
    };
    (@adjacent $policy:expr; $a:ident, $b:ident, $c:ident, $d:ident, $e:ident) => {
        crate::detail::apply::LinearScanApply::adjacent_views5::<R, A, B, C, D, E, KernelOp<R, Op>>(
            $policy, &$a, &$b, &$c, &$d, &$e,
        )
    };
    (@adjacent $policy:expr; $a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident) => {
        crate::detail::apply::LinearScanApply::adjacent_views6::<R, A, B, C, D, E, F, KernelOp<R, Op>>(
            $policy, &$a, &$b, &$c, &$d, &$e, &$f,
        )
    };
    (@adjacent $policy:expr; $a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident, $g:ident) => {
        crate::detail::apply::LinearScanApply::adjacent_views7::<R, A, B, C, D, E, F, G, KernelOp<R, Op>>(
            $policy, &$a, &$b, &$c, &$d, &$e, &$f, &$g,
        )
    };
    (@inclusive $policy:expr; $a:ident, $b:ident, $c:ident, $d:ident) => {
        crate::detail::apply::LinearScanApply::inclusive_views4::<R, A, B, C, D, KernelOp<R, Op>>(
            $policy, &$a, &$b, &$c, &$d,
        )
    };
    (@inclusive $policy:expr; $a:ident, $b:ident, $c:ident, $d:ident, $e:ident) => {
        crate::detail::apply::LinearScanApply::inclusive_views5::<R, A, B, C, D, E, KernelOp<R, Op>>(
            $policy, &$a, &$b, &$c, &$d, &$e,
        )
    };
    (@inclusive $policy:expr; $a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident) => {
        crate::detail::apply::LinearScanApply::inclusive_views6::<R, A, B, C, D, E, F, KernelOp<R, Op>>(
            $policy, &$a, &$b, &$c, &$d, &$e, &$f,
        )
    };
    (@inclusive $policy:expr; $a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident, $g:ident) => {
        crate::detail::apply::LinearScanApply::inclusive_views7::<R, A, B, C, D, E, F, G, KernelOp<R, Op>>(
            $policy, &$a, &$b, &$c, &$d, &$e, &$f, &$g,
        )
    };
    (@exclusive $policy:expr, $init:expr; $a:ident, $b:ident, $c:ident, $d:ident) => {
        crate::detail::apply::LinearScanApply::exclusive_views4::<R, A, B, C, D, KernelOp<R, Op>>(
            $policy, &$a, &$b, &$c, &$d, $init,
        )
    };
    (@exclusive $policy:expr, $init:expr; $a:ident, $b:ident, $c:ident, $d:ident, $e:ident) => {
        crate::detail::apply::LinearScanApply::exclusive_views5::<R, A, B, C, D, E, KernelOp<R, Op>>(
            $policy, &$a, &$b, &$c, &$d, &$e, $init,
        )
    };
    (@exclusive $policy:expr, $init:expr; $a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident) => {
        crate::detail::apply::LinearScanApply::exclusive_views6::<R, A, B, C, D, E, F, KernelOp<R, Op>>(
            $policy, &$a, &$b, &$c, &$d, &$e, &$f, $init,
        )
    };
    (@exclusive $policy:expr, $init:expr; $a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident, $g:ident) => {
        crate::detail::apply::LinearScanApply::exclusive_views7::<R, A, B, C, D, E, F, G, KernelOp<R, Op>>(
            $policy, &$a, &$b, &$c, &$d, &$e, &$f, &$g, $init,
        )
    };
}

macro_rules! wide_predicate_rank_from_tuple {
    ($policy:expr, $env:expr; $pred:ty; $a:expr, $b:expr, $c:expr, $d:expr) => {{
        let dummy_e = crate::detail::primitives::range::indices_mindex($policy, $a.len)?;
        let dummy_f = crate::detail::primitives::range::indices_mindex($policy, $a.len)?;
        let dummy_g = crate::detail::primitives::range::indices_mindex($policy, $a.len)?;
        let dummy_e = crate::detail::device::DeviceColumnView::from_column(&dummy_e);
        let dummy_f = crate::detail::device::DeviceColumnView::from_column(&dummy_f);
        let dummy_g = crate::detail::device::DeviceColumnView::from_column(&dummy_g);
        wide_predicate_rank_from_tuple!(
            @launch $policy, $env,
            crate::detail::api::Tuple4AsTuple7PredicateOp<KernelOp<R, $pred>>,
            ($a, $b, $c, $d, &dummy_e, &dummy_f, &dummy_g),
            (A, B, C, D, u32, u32, u32)
        )
    }};
    ($policy:expr, $env:expr; $pred:ty; $a:expr, $b:expr, $c:expr, $d:expr, $e:expr) => {{
        let dummy_f = crate::detail::primitives::range::indices_mindex($policy, $a.len)?;
        let dummy_g = crate::detail::primitives::range::indices_mindex($policy, $a.len)?;
        let dummy_f = crate::detail::device::DeviceColumnView::from_column(&dummy_f);
        let dummy_g = crate::detail::device::DeviceColumnView::from_column(&dummy_g);
        wide_predicate_rank_from_tuple!(
            @launch $policy, $env,
            crate::detail::api::Tuple5AsTuple7PredicateOp<KernelOp<R, $pred>>,
            ($a, $b, $c, $d, $e, &dummy_f, &dummy_g),
            (A, B, C, D, E, u32, u32)
        )
    }};
    ($policy:expr, $env:expr; $pred:ty; $a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr) => {{
        let dummy_g = crate::detail::primitives::range::indices_mindex($policy, $a.len)?;
        let dummy_g = crate::detail::device::DeviceColumnView::from_column(&dummy_g);
        wide_predicate_rank_from_tuple!(
            @launch $policy, $env,
            crate::detail::api::Tuple6AsTuple7PredicateOp<KernelOp<R, $pred>>,
            ($a, $b, $c, $d, $e, $f, &dummy_g),
            (A, B, C, D, E, F, u32)
        )
    }};
    ($policy:expr, $env:expr; $pred:ty; $a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr) => {
        wide_predicate_rank_from_tuple!(
            @launch $policy, $env,
            KernelOp<R, $pred>,
            ($a, $b, $c, $d, $e, $f, $g),
            (A, B, C, D, E, F, G)
        )
    };
    (@launch $policy:expr, $env:expr, $pred:ty, ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr), ($ty0:ty, $ty1:ty, $ty2:ty, $ty3:ty, $ty4:ty, $ty5:ty, $ty6:ty)) => {{
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
            let invert_handle = client.create_from_slice(u32::as_bytes(&[0_u32]));
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

macro_rules! wide_partition_apply_from_tuple {
    ($apply:ident, $policy:expr; $a:ident, $b:ident, $c:ident, $d:ident) => {
        $apply.apply_expr4($policy, &$a, &$b, &$c, &$d)
    };
    ($apply:ident, $policy:expr; $a:ident, $b:ident, $c:ident, $d:ident, $e:ident) => {
        $apply.apply_expr5($policy, &$a, &$b, &$c, &$d, &$e)
    };
    ($apply:ident, $policy:expr; $a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident) => {
        $apply.apply_expr6($policy, &$a, &$b, &$c, &$d, &$e, &$f)
    };
    ($apply:ident, $policy:expr; $a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident, $g:ident) => {
        $apply.apply_expr7($policy, &$a, &$b, &$c, &$d, &$e, &$f, &$g)
    };
}

macro_rules! zip_into_inner {
    ($value:expr; $a:ident) => {{
        let Zip1($a) = $value;
        ($a.inner,)
    }};
    ($value:expr; $a:ident, $b:ident) => {{
        let Zip2($a, $b) = $value;
        ($a.inner, $b.inner)
    }};
    ($value:expr; $a:ident, $b:ident, $c:ident) => {{
        let Zip3($a, $b, $c) = $value;
        ($a.inner, $b.inner, $c.inner)
    }};
    ($value:expr; $a:ident, $b:ident, $c:ident, $d:ident) => {{
        let Zip4($a, $b, $c, $d) = $value;
        ($a.inner, $b.inner, $c.inner, $d.inner)
    }};
    ($value:expr; $a:ident, $b:ident, $c:ident, $d:ident, $e:ident) => {{
        let Zip5($a, $b, $c, $d, $e) = $value;
        ($a.inner, $b.inner, $c.inner, $d.inner, $e.inner)
    }};
    ($value:expr; $a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident) => {{
        let Zip6($a, $b, $c, $d, $e, $f) = $value;
        ($a.inner, $b.inner, $c.inner, $d.inner, $e.inner, $f.inner)
    }};
    ($value:expr; $a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident, $g:ident) => {{
        let Zip7($a, $b, $c, $d, $e, $f, $g) = $value;
        (
            $a.inner, $b.inner, $c.inner, $d.inner, $e.inner, $f.inner, $g.inner,
        )
    }};
}

macro_rules! transform_from_tuple_view {
    ($output:ty, $policy:expr, $op:expr, $env:expr; $a:ident) => {
        <$output as sealed::MItemDispatch<R>>::transform_unary($policy, $a, $op, $env)
    };
    ($output:ty, $policy:expr, $op:expr, $env:expr; $a:ident, $b:ident) => {
        <$output as sealed::MItemDispatch<R>>::transform_binary($policy, $a, $b, $op, $env)
    };
    ($output:ty, $policy:expr, $op:expr, $env:expr; $a:ident, $b:ident, $c:ident) => {
        <$output as sealed::MItemDispatch<R>>::transform_ternary($policy, $a, $b, $c, $op, $env)
    };
    ($output:ty, $policy:expr, $op:expr, $env:expr; $a:ident, $b:ident, $c:ident, $d:ident) => {
        <$output as sealed::MItemDispatch<R>>::transform_quaternary(
            $policy, $a, $b, $c, $d, $op, $env,
        )
    };
    ($output:ty, $policy:expr, $op:expr, $env:expr; $a:ident, $b:ident, $c:ident, $d:ident, $e:ident) => {
        <$output as sealed::MItemDispatch<R>>::transform_quinary(
            $policy, $a, $b, $c, $d, $e, $op, $env,
        )
    };
    ($output:ty, $policy:expr, $op:expr, $env:expr; $a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident) => {
        <$output as sealed::MItemDispatch<R>>::transform_senary(
            $policy, $a, $b, $c, $d, $e, $f, $op, $env,
        )
    };
    ($output:ty, $policy:expr, $op:expr, $env:expr; $a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident, $g:ident) => {
        <$output as sealed::MItemDispatch<R>>::transform_septenary(
            $policy, $a, $b, $c, $d, $e, $f, $g, $op, $env,
        )
    };
}

macro_rules! alloc_inner {
    ($exec:expr, $len:expr; $( $ty:ty ),+) => {{
        let len = $len;
        let policy = $exec.policy();
        if len == 0 {
            Ok(($( policy.empty_device_vec::<$ty>(), )+))
        } else {
            let client = policy.client();
            let len_usize = usize_from_mindex(len);
            Ok(($(
                crate::detail::DeviceVec::from_handle(
                    policy.id(),
                    client.empty(len_usize * std::mem::size_of::<$ty>()),
                    len,
                ),
            )+))
        }
    }};
}

macro_rules! impl_scalar_mitem {
    ($( $ty:ty ),+ $(,)?) => {
        $(
            impl<R> MAlloc<R> for $ty
            where
                R: Runtime,
                $ty: MStorageElement + 'static,
            {
                type Inner = crate::detail::DeviceVec<R, $ty>;
                type View = crate::detail::device::DeviceColumnView<R, $ty>;
                type Storage = DeviceVec<R, $ty>;

                fn storage_from_inner(inner: Self::Inner) -> Self::Storage {
                    DeviceVec::from_inner(inner)
                }

                fn alloc_storage(exec: &Executor<R>, len: MIndex) -> Result<Self::Storage, Error> {
                    let policy = exec.policy();
                    if len == 0 {
                        Ok(Self::storage_from_inner(policy.empty_device_vec::<$ty>()))
                    } else {
                        let client = policy.client();
                        let len_usize = usize_from_mindex(len);
                        Ok(Self::storage_from_inner(crate::detail::DeviceVec::from_handle(
                            policy.id(),
                            client.empty(len_usize * std::mem::size_of::<$ty>()),
                            len,
                        )))
                    }
                }

                fn inclusive_scan_by_key_values_from_inner<KeyEq, Op, Output>(
                    policy: &crate::detail::CubePolicy<R>,
                    values: Self::Inner,
                    control: &crate::detail::control::ScanByKeyControl<R>,
                    _op: Op,
                    output: Output,
                ) -> Result<(), Error>
                where
                    Op: op::ReductionOp<R, Self>,
                    Output: MIterMut<R, Item = Self>,
                {
                    let apply = crate::detail::apply::SegmentedScanApply::new(control);
                    let values = crate::detail::device::DeviceColumnView::from_column(&values);
                    let scanned = apply.inclusive_expr::<
                        crate::detail::device::DeviceColumnView<R, $ty>,
                        crate::detail::op_adapter::KernelScalarTuple1Op<R, Op>,
                    >(policy, &values)?;
                    output.write_from_inner(policy, scanned)
                }

                fn exclusive_scan_by_key_values_from_inner<KeyEq, Op, Output>(
                    policy: &crate::detail::CubePolicy<R>,
                    values: Self::Inner,
                    control: &crate::detail::control::ScanByKeyControl<R>,
                    init: Self,
                    _op: Op,
                    output: Output,
                ) -> Result<(), Error>
                where
                    Op: op::ReductionOp<R, Self>,
                    Output: MIterMut<R, Item = Self>,
                {
                    let apply = crate::detail::apply::SegmentedScanApply::new(control);
                    let values = crate::detail::device::DeviceColumnView::from_column(&values);
                    let scanned = apply.exclusive_expr::<
                        crate::detail::device::DeviceColumnView<R, $ty>,
                        crate::detail::op_adapter::KernelScalarTuple1Op<R, Op>,
                    >(policy, &values, init)?;
                    output.write_from_inner(policy, scanned)
                }

                fn transform_from_view<Output, Op>(
                    policy: &crate::detail::CubePolicy<R>,
                    input: Self::View,
                    op: Op,
                    env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
                    output: Output,
                ) -> Result<(), Error>
                where
                    Output: MIterMut<R>,
                    Output::Item: MAlloc<R> + sealed::MItemDispatch<R>,
                    Op: op::UnaryOp<R, Self, Output = Output::Item>,
                {
                    let inner = <Output::Item as sealed::MItemDispatch<R>>::transform_scalar_input(
                        policy,
                        input,
                        op,
                        env,
                    )?;
                    output.write_from_inner(policy, inner)
                }

                fn transform_where_from_view<Output, Op>(
                    policy: &crate::detail::CubePolicy<R>,
                    input: Self::View,
                    op: Op,
                    env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
                    stencil: crate::detail::api::PrecomputedSelection<R>,
                    output: Output,
                ) -> Result<(), Error>
                where
                    Output: MIterMut<R>,
                    Output::Item: MAlloc<R> + sealed::MItemDispatch<R>,
                    Op: op::UnaryOp<R, Self, Output = Output::Item>,
                {
                    let inner = <Output::Item as sealed::MItemDispatch<R>>::transform_scalar_input(
                        policy,
                        input,
                        op,
                        env,
                    )?;
                    output.write_where_from_inner(policy, inner, stencil)
                }
            }

            impl<R> StorageFromInner<R> for DeviceVec<R, $ty>
            where
                R: Runtime,
                $ty: MStorageElement + 'static,
            {
                type Item = $ty;

                fn from_inner(inner: <Self::Item as MAlloc<R>>::Inner) -> Self {
                    <Self::Item as MAlloc<R>>::storage_from_inner(inner)
                }

                fn into_inner(self) -> <Self::Item as MAlloc<R>>::Inner {
                    self.inner
                }

                fn len(&self) -> MIndex {
                    self.len()
                }
            }

            impl<R> sealed::MItemDispatch<R> for $ty
            where
                R: Runtime,
                $ty: MStorageElement + 'static,
            {
            }
        )+
    };
}

impl_scalar_mitem!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

macro_rules! impl_mitem_tuple {
    ($( $ty:ident : $var:ident ),+) => {
        impl<R, $( $ty ),+> MAlloc<R> for ($( $ty, )+)
        where
            R: Runtime,
            $( $ty: MStorageElement, )+
        {
            type Inner = ($( crate::detail::DeviceVec<R, $ty>, )+);
            type View = ($( crate::detail::device::DeviceColumnView<R, $ty>, )+);
            type Storage = zip_type!($( DeviceVec<R, $ty> ),+);

            fn storage_from_inner(inner: Self::Inner) -> Self::Storage {
                let ($( $var, )+) = inner;
                zip_value!($( DeviceVec::from_inner($var) ),+)
            }

            fn alloc_storage(exec: &Executor<R>, len: MIndex) -> Result<Self::Storage, Error> {
                Ok(Self::storage_from_inner(alloc_inner!(exec, len; $( $ty ),+)?))
            }

            fn reverse_from_view<Output>(
                policy: &crate::detail::CubePolicy<R>,
                input: Self::View,
                output: Output,
            ) -> Result<(), Error>
            where
                Output: MIterMut<R, Item = Self>,
            {
                let input = zip_view_from_tuple!(input; $( $var ),+);
                let inner = crate::detail::reverse(policy, input)?;
                output.write_from_inner(policy, inner)
            }

            fn sort_from_view<Less, Output>(
                policy: &crate::detail::CubePolicy<R>,
                input: Self::View,
                _less: Less,
                output: Output,
            ) -> Result<(), Error>
            where
                Less: op::BinaryPredicateOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                let input = zip_view_from_tuple!(input; $( $var ),+);
                let inner = crate::detail::sort(policy, input, tuple_set_less!(Less; $( $var ),+))?;
                output.write_from_inner(policy, inner)
            }

            fn sort_by_key_control_from_view<Less>(
                policy: &crate::detail::CubePolicy<R>,
                keys: Self::View,
                _less: Less,
            ) -> Result<(Self::Inner, crate::detail::DeviceVec<R, MIndex>), Error>
            where
                Less: op::BinaryPredicateOp<R, Self>,
            {
                let (keys, indices) =
                    <_ as crate::detail::read::KernelSortByKeyKeys<KernelOp<R, Less>>>::sort_by_key_control(
                        keys,
                        policy,
                    )?;
                let inner = crate::detail::api::MaterializeOutput::materialize_output(keys, policy)?;
                Ok((inner, indices))
            }

            fn sort_by_key_values_from_view<Output>(
                policy: &crate::detail::CubePolicy<R>,
                values: Self::View,
                control: &crate::detail::control::PermutationControl<R>,
                output: Output,
            ) -> Result<(), Error>
            where
                Output: MIterMut<R, Item = Self>,
            {
                let values = zip_view_from_tuple!(values; $( $var ),+);
                let values =
                    crate::detail::read::KernelSortByKeyValues::sort_by_key_values(
                        values,
                        policy,
                        control,
                    )?;
                let inner = crate::detail::api::MaterializeOutput::materialize_output(values, policy)?;
                output.write_from_inner(policy, inner)
            }

            fn merge_by_key_control_from_views<Less>(
                policy: &crate::detail::CubePolicy<R>,
                left_keys: Self::View,
                right_keys: Self::View,
                _less: Less,
            ) -> Result<(Self::Inner, crate::detail::control::MergeByKeyControl), Error>
            where
                Less: op::BinaryPredicateOp<R, Self>,
            {
                let (keys, control) =
                    <_ as crate::detail::read::KernelMergeByKeyKeys<_, KernelOp<R, Less>>>::merge_by_key_control(
                        left_keys,
                        policy,
                        right_keys,
                    )?;
                let inner = crate::detail::api::MaterializeOutput::materialize_output(keys, policy)?;
                Ok((inner, control))
            }

            fn merge_by_key_values_from_views<Output>(
                policy: &crate::detail::CubePolicy<R>,
                left_values: Self::View,
                right_values: Self::View,
                control: &crate::detail::control::MergeByKeyControl,
                output: Output,
            ) -> Result<(), Error>
            where
                Output: MIterMut<R, Item = Self>,
            {
                let left_values = zip_view_from_tuple!(left_values; $( $var ),+);
                let right_values = zip_view_from_tuple!(right_values; $( $var ),+);
                let values =
                    crate::detail::read::KernelMergeByKeyValues::merge_by_key_values(
                        left_values,
                        policy,
                        right_values,
                        control,
                    )?;
                let inner = crate::detail::api::MaterializeOutput::materialize_output(values, policy)?;
                output.write_from_inner(policy, inner)
            }

            fn unique_by_key_control_from_view<Eq>(
                policy: &crate::detail::CubePolicy<R>,
                keys: Self::View,
                _eq: Eq,
            ) -> Result<(Self::Inner, crate::detail::control::UniqueByKeyControl), Error>
            where
                Eq: op::BinaryPredicateOp<R, Self>,
            {
                let (keys, control) =
                    <_ as crate::detail::read::KernelUniqueByKeyKeys<KernelOp<R, Eq>>>::unique_by_key_control(
                        keys,
                        policy,
                    )?;
                let inner = crate::detail::api::MaterializeOutput::materialize_output(keys, policy)?;
                Ok((inner, control))
            }

            fn unique_by_key_values_from_view<Output>(
                policy: &crate::detail::CubePolicy<R>,
                values: Self::View,
                control: &crate::detail::control::UniqueByKeyControl,
                output: Output,
            ) -> Result<(), Error>
            where
                Output: MIterMut<R, Item = Self>,
            {
                let values = zip_view_from_tuple!(values; $( $var ),+);
                let values =
                    crate::detail::read::KernelUniqueByKeyValues::unique_by_key_values(
                        values,
                        policy,
                        control,
                    )?;
                let inner = crate::detail::api::MaterializeOutput::materialize_output(values, policy)?;
                output.write_prefix_from_inner(policy, inner)
            }

            fn reduce_by_key_control_from_view<KeyEq>(
                policy: &crate::detail::CubePolicy<R>,
                keys: Self::View,
                _key_eq: KeyEq,
            ) -> Result<(Self::Inner, crate::detail::control::ReduceByKeyControl<R>), Error>
            where
                KeyEq: op::BinaryPredicateOp<R, Self>,
            {
                let (keys, control) =
                    <_ as crate::detail::read::KernelReduceByKeyKeys<KernelOp<R, KeyEq>>>::reduce_by_key_control(
                        keys,
                        policy,
                    )?;
                let inner = crate::detail::api::MaterializeOutput::materialize_output(keys, policy)?;
                Ok((inner, control))
            }

            fn reduce_by_key_values_from_view<KeyEq, Op, Output>(
                policy: &crate::detail::CubePolicy<R>,
                values: Self::View,
                control: &crate::detail::control::ReduceByKeyControl<R>,
                init: Self,
                _op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                Op: op::ReductionOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                let values =
                    <_ as crate::detail::read::KernelReduceByKeyValues<
                        crate::detail::control::ReduceByKeyControl<R>,
                        KernelOp<R, KeyEq>,
                        KernelOp<R, Op>,
                    >>::reduce_by_key_values(values, policy, control, init)?;
                let inner = crate::detail::api::MaterializeOutput::materialize_output(values, policy)?;
                output.write_prefix_from_inner(policy, inner)
            }

            fn inclusive_scan_by_key_values_from_inner<KeyEq, Op, Output>(
                policy: &crate::detail::CubePolicy<R>,
                values: Self::Inner,
                control: &crate::detail::control::ScanByKeyControl<R>,
                op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                Op: op::ReductionOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                <Self as crate::detail::read::ScanByKeyValueItem<R>>::inclusive_scan_by_key_values_from_inner::<
                    Op,
                    Output,
                >(policy, values, control, op, output)
            }

            fn exclusive_scan_by_key_values_from_inner<KeyEq, Op, Output>(
                policy: &crate::detail::CubePolicy<R>,
                values: Self::Inner,
                control: &crate::detail::control::ScanByKeyControl<R>,
                init: Self,
                op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                Op: op::ReductionOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                <Self as crate::detail::read::ScanByKeyValueItem<R>>::exclusive_scan_by_key_values_from_inner::<
                    Op,
                    Output,
                >(policy, values, control, init, op, output)
            }

            fn copy_selected_from_view<Output>(
                policy: &crate::detail::CubePolicy<R>,
                values: Self::View,
                stencil: crate::detail::api::PrecomputedSelection<R>,
                output: Output,
            ) -> Result<MIndex, Error>
            where
                Output: MIterMut<R, Item = Self>,
            {
                let values = zip_view_from_tuple!(values; $( $var ),+);
                let inner = crate::detail::copy_where(
                    policy,
                    values,
                    stencil,
                    KernelOp::<R, crate::detail::op_adapter::StencilFlag>::new(),
                )?;
                let len = mindex_from_usize(inner.0.len())?;
                output.write_prefix_from_inner(policy, inner)?;
                Ok(len)
            }

            fn gather_from_view<Output>(
                policy: &crate::detail::CubePolicy<R>,
                values: Self::View,
                indices: crate::detail::device::DeviceColumnView<R, MIndex>,
                output: Output,
            ) -> Result<(), Error>
            where
                Output: MIterMut<R, Item = Self>,
            {
                let inner = indexed_gather_inner_from_tuple!(policy, values, indices; $( $var ),+)?;
                output.write_from_inner(policy, inner)
            }

            fn gather_where_from_view<Output>(
                policy: &crate::detail::CubePolicy<R>,
                values: Self::View,
                indices: crate::detail::device::DeviceColumnView<R, MIndex>,
                stencil: crate::detail::api::PrecomputedSelection<R>,
                output: Output,
            ) -> Result<(), Error>
            where
                Output: MIterMut<R, Item = Self>,
            {
                let mask = stencil.mask();
                indexed_apply_arity!(
                    crate::detail::apply::MaskedIndexedExprApply::gather_where_expr_into,
                    policy, values, indices, output, &mask;
                    $( $ty: $var ),+
                )
            }

            fn scatter_from_view<Output>(
                policy: &crate::detail::CubePolicy<R>,
                values: Self::View,
                indices: crate::detail::device::DeviceColumnView<R, MIndex>,
                output: Output,
            ) -> Result<(), Error>
            where
                Output: MIterMut<R, Item = Self>,
            {
                indexed_apply_arity!(
                    crate::detail::apply::IndexedExprApply::scatter_expr_into,
                    policy, values, indices, output;
                    $( $ty: $var ),+
                )
            }

            fn scatter_where_from_view<Output>(
                policy: &crate::detail::CubePolicy<R>,
                values: Self::View,
                indices: crate::detail::device::DeviceColumnView<R, MIndex>,
                stencil: crate::detail::api::PrecomputedSelection<R>,
                output: Output,
            ) -> Result<(), Error>
            where
                Output: MIterMut<R, Item = Self>,
            {
                let mask = stencil.mask();
                indexed_apply_arity!(
                    crate::detail::apply::MaskedIndexedExprApply::scatter_where_expr_into,
                    policy, values, indices, output, &mask;
                    $( $ty: $var ),+
                )
            }

            fn transform_from_view<Output, Op>(
                policy: &crate::detail::CubePolicy<R>,
                input: Self::View,
                op: Op,
                env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
                output: Output,
            ) -> Result<(), Error>
            where
                Output: MIterMut<R>,
                Output::Item: MAlloc<R> + sealed::MItemDispatch<R>,
                Op: op::UnaryOp<R, Self, Output = Output::Item>,
            {
                let ($( $var, )+) = input;
                let inner = transform_from_tuple_view!(
                    Output::Item,
                    policy,
                    op,
                    env;
                    $( $var ),+
                )?;
                output.write_from_inner(policy, inner)
            }

            fn transform_where_from_view<Output, Op>(
                policy: &crate::detail::CubePolicy<R>,
                input: Self::View,
                op: Op,
                env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
                stencil: crate::detail::api::PrecomputedSelection<R>,
                output: Output,
            ) -> Result<(), Error>
            where
                Output: MIterMut<R>,
                Output::Item: MAlloc<R> + sealed::MItemDispatch<R>,
                Op: op::UnaryOp<R, Self, Output = Output::Item>,
            {
                let ($( $var, )+) = input;
                let inner = transform_from_tuple_view!(
                    Output::Item,
                    policy,
                    op,
                    env;
                    $( $var ),+
                )?;
                output.write_where_from_inner(policy, inner, stencil)
            }

            fn unique_from_view<Pred, Output>(
                policy: &crate::detail::CubePolicy<R>,
                input: Self::View,
                _pred: Pred,
                output: Output,
            ) -> Result<MIndex, Error>
            where
                Pred: op::BinaryPredicateOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                let input = zip_view_from_tuple!(input; $( $var ),+);
                let inner = crate::detail::unique(policy, input, tuple_set_less!(Pred; $( $var ),+))?;
                let len = mindex_from_usize(inner.0.len())?;
                output.write_prefix_from_inner(policy, inner)?;
                Ok(len)
            }

            fn reduce_from_view<Op>(
                policy: &crate::detail::CubePolicy<R>,
                input: Self::View,
                init: Self,
                _op: Op,
            ) -> Result<Self, Error>
            where
                Op: op::ReductionOp<R, Self>,
            {
                crate::detail::reduce(policy, input, init, KernelOp::<R, Op>::new())
            }

            fn partition_from_view<Pred, Output>(
                policy: &crate::detail::CubePolicy<R>,
                input: Self::View,
                _pred: Pred,
                env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
                output: Output,
            ) -> Result<MIndex, Error>
            where
                Pred: op::PredicateOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                let input = partition_input_from_tuple!(input; $( $var ),+);
                let (matching, failing) =
                    crate::detail::partition(policy, input, KernelOp::<R, Pred>::new(), env)?;
                let split = mindex_from_usize(matching.0.len())?;
                output.write_split_from_inner(policy, matching, failing)?;
                Ok(split)
            }

            fn adjacent_difference_from_view<Op, Output>(
                policy: &crate::detail::CubePolicy<R>,
                input: Self::View,
                _op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                Op: op::ReductionOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                let input = scan_input_from_tuple!(input; $( $var ),+);
                let inner =
                    crate::detail::adjacent_difference(policy, input, KernelOp::<R, Op>::new())?;
                output.write_from_inner(policy, inner)
            }

            fn inclusive_scan_from_view<Op, Output>(
                policy: &crate::detail::CubePolicy<R>,
                input: Self::View,
                _op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                Op: op::ReductionOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                let input = scan_input_from_tuple!(input; $( $var ),+);
                let inner =
                    crate::detail::inclusive_scan(policy, input, KernelOp::<R, Op>::new())?;
                output.write_from_inner(policy, inner)
            }

            fn exclusive_scan_from_view<Op, Output>(
                policy: &crate::detail::CubePolicy<R>,
                input: Self::View,
                init: Self,
                _op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                Op: op::ReductionOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                let input = scan_input_from_tuple!(input; $( $var ),+);
                let inner =
                    crate::detail::exclusive_scan(policy, input, init, KernelOp::<R, Op>::new())?;
                output.write_from_inner(policy, inner)
            }

            fn merge_from_views<Less, Output>(
                policy: &crate::detail::CubePolicy<R>,
                left: Self::View,
                right: Self::View,
                _less: Less,
                output: Output,
            ) -> Result<(), Error>
            where
                Less: op::BinaryPredicateOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                let left = zip_view_from_tuple!(left; $( $var ),+);
                let right = zip_view_from_tuple!(right; $( $var ),+);
                let inner = crate::detail::merge(policy, left, right, tuple_set_less!(Less; $( $var ),+))?;
                output.write_from_inner(policy, inner)
            }

            fn set_union_from_views<Less, Output>(
                policy: &crate::detail::CubePolicy<R>,
                left: Self::View,
                right: Self::View,
                _less: Less,
                output: Output,
            ) -> Result<MIndex, Error>
            where
                Less: op::BinaryPredicateOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                let left = zip_view_from_tuple!(left; $( $var ),+);
                let right = zip_view_from_tuple!(right; $( $var ),+);
                let inner = crate::detail::set_union(policy, left, right, tuple_set_less!(Less; $( $var ),+))?;
                let len = mindex_from_usize(inner.0.len())?;
                output.write_prefix_from_inner(policy, inner)?;
                Ok(len)
            }

            fn set_intersection_from_views<Less, Output>(
                policy: &crate::detail::CubePolicy<R>,
                left: Self::View,
                right: Self::View,
                _less: Less,
                output: Output,
            ) -> Result<MIndex, Error>
            where
                Less: op::BinaryPredicateOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                let left = zip_view_from_tuple!(left; $( $var ),+);
                let right = zip_view_from_tuple!(right; $( $var ),+);
                let inner = crate::detail::set_intersection(policy, left, right, tuple_set_less!(Less; $( $var ),+))?;
                let len = mindex_from_usize(inner.0.len())?;
                output.write_prefix_from_inner(policy, inner)?;
                Ok(len)
            }

            fn set_difference_from_views<Less, Output>(
                policy: &crate::detail::CubePolicy<R>,
                left: Self::View,
                right: Self::View,
                _less: Less,
                output: Output,
            ) -> Result<MIndex, Error>
            where
                Less: op::BinaryPredicateOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                let left = zip_view_from_tuple!(left; $( $var ),+);
                let right = zip_view_from_tuple!(right; $( $var ),+);
                let inner = crate::detail::set_difference(policy, left, right, tuple_set_less!(Less; $( $var ),+))?;
                let len = mindex_from_usize(inner.0.len())?;
                output.write_prefix_from_inner(policy, inner)?;
                Ok(len)
            }
        }

        impl<R, $( $ty ),+> StorageFromInner<R> for zip_type!($( DeviceVec<R, $ty> ),+)
        where
            R: Runtime,
            $( $ty: MStorageElement + 'static, )+
        {
            type Item = ($( $ty, )+);

            fn from_inner(inner: <Self::Item as MAlloc<R>>::Inner) -> Self {
                <Self::Item as MAlloc<R>>::storage_from_inner(inner)
            }

            fn into_inner(self) -> <Self::Item as MAlloc<R>>::Inner {
                zip_into_inner!(self; $( $var ),+)
            }

            fn len(&self) -> MIndex {
                self.0.len()
            }
        }

        impl<R, $( $ty ),+> sealed::MItemDispatch<R> for ($( $ty, )+)
        where
            R: Runtime,
            $( $ty: MStorageElement, )+
        {
            fn transform_scalar_input<Input, Op>(
                policy: &crate::detail::CubePolicy<R>,
                input: crate::detail::device::DeviceColumnView<R, Input>,
                op: Op,
                env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<<Self as MAlloc<R>>::Inner, Error>
            where
                Input: MStorageElement + MItem<R>,
                Op: op::UnaryOp<R, Input, Output = Self>,
                Self: crate::detail::TransformUnaryOutput<
                    R,
                    Input,
                    KernelScalarInputOp<R, Op>,
                >,
                <Self as crate::detail::MItemStorage<
                    R,
                >>::Storage: crate::detail::MaterializeOutput<
                    Runtime = R,
                    Output = ($(
                        crate::detail::DeviceVec<R, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage = crate::detail::apply::TransformPayloadApply::unary::<Self, R, Input, KernelScalarInputOp<R, Op>>(policy, input, env)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            fn transform_unary<Input, Op>(
                policy: &crate::detail::CubePolicy<R>,
                input: crate::detail::device::DeviceColumnView<
                    R,
                    Input,
                >,
                op: Op,
                env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<<Self as MAlloc<R>>::Inner, Error>
            where
                Input: MStorageElement,
                Op: op::UnaryOp<R, (Input,), Output = Self>,
                Self: crate::detail::TransformUnaryOutput<
                    R,
                    Input,
                    KernelOp<R, Op>,
                >,
                <Self as crate::detail::MItemStorage<
                    R,
                >>::Storage: crate::detail::MaterializeOutput<
                    Runtime = R,
                    Output = ($(
                        crate::detail::DeviceVec<R, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage = crate::detail::apply::TransformPayloadApply::unary::<Self, R, Input, KernelOp<R, Op>>(policy, input, env)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            fn transform_binary<Left, Right, Op>(
                policy: &crate::detail::CubePolicy<R>,
                left: crate::detail::device::DeviceColumnView<
                    R,
                    Left,
                >,
                right: crate::detail::device::DeviceColumnView<
                    R,
                    Right,
                >,
                op: Op,
                env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<<Self as MAlloc<R>>::Inner, Error>
            where
                Left: MStorageElement,
                Right: MStorageElement,
                Op: op::UnaryOp<R, (Left, Right), Output = Self>,
                Self: crate::detail::TransformZip2Output<
                    R,
                    Left,
                    Right,
                    KernelOp<R, Op>,
                >,
                <Self as crate::detail::MItemStorage<
                    R,
                >>::Storage: crate::detail::MaterializeOutput<
                    Runtime = R,
                    Output = ($(
                        crate::detail::DeviceVec<R, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage = crate::detail::apply::TransformPayloadApply::zip2::<Self, R, Left, Right, KernelOp<R, Op>>(policy, left, right, env)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            fn transform_ternary<First, Second, Third, Op>(
                policy: &crate::detail::CubePolicy<R>,
                first: crate::detail::device::DeviceColumnView<
                    R,
                    First,
                >,
                second: crate::detail::device::DeviceColumnView<
                    R,
                    Second,
                >,
                third: crate::detail::device::DeviceColumnView<
                    R,
                    Third,
                >,
                op: Op,
                env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<<Self as MAlloc<R>>::Inner, Error>
            where
                First: MStorageElement,
                Second: MStorageElement,
                Third: MStorageElement,
                Op: op::UnaryOp<R, (First, Second, Third), Output = Self>,
                Self: crate::detail::TransformZip3Output<
                    R,
                    First,
                    Second,
                    Third,
                    KernelOp<R, Op>,
                >,
                <Self as crate::detail::MItemStorage<
                    R,
                >>::Storage: crate::detail::MaterializeOutput<
                    Runtime = R,
                    Output = ($(
                        crate::detail::DeviceVec<R, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage = crate::detail::apply::TransformPayloadApply::zip3::<Self, R, First, Second, Third, KernelOp<R, Op>>(policy, first, second, third, env)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            fn transform_logical3<Input, LeafA, LeafB, LeafC, Expr, Op>(
                policy: &crate::detail::CubePolicy<R>,
                bindings: crate::detail::device::KernelColumnBindings,
                len: usize,
                op: Op,
                env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<<Self as MAlloc<R>>::Inner, Error>
            where
                Input: MItem<R> + Send + Sync,
                LeafA: MStorageElement,
                LeafB: MStorageElement,
                LeafC: MStorageElement,
                Expr: crate::expr::LogicalDeviceExpr3<Input, LeafA, LeafB, LeafC>,
                Op: op::UnaryOp<R, Input, Output = Self>,
                Self: crate::detail::TransformLogical3Output<
                    R,
                    Input,
                    LeafA,
                    LeafB,
                    LeafC,
                    Expr,
                    KernelOp<R, Op>,
                >,
                <Self as crate::detail::MItemStorage<
                    R,
                >>::Storage: crate::detail::MaterializeOutput<
                    Runtime = R,
                    Output = ($(
                        crate::detail::DeviceVec<R, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage = <Self as crate::detail::TransformLogical3Output<
                    R,
                    Input,
                    LeafA,
                    LeafB,
                    LeafC,
                    Expr,
                    KernelOp<R, Op>,
                >>::run_logical3(policy, bindings, len, env)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            #[allow(clippy::too_many_arguments)]
            fn transform_logical7<
                Input,
                Leaf0,
                Leaf1,
                Leaf2,
                Leaf3,
                Leaf4,
                Leaf5,
                Leaf6,
                Expr,
                Op,
            >(
                policy: &crate::detail::CubePolicy<R>,
                bindings: crate::detail::device::KernelColumnBindings,
                len: usize,
                op: Op,
                env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<<Self as MAlloc<R>>::Inner, Error>
            where
                Input: MItem<R> + Send + Sync,
                Leaf0: MStorageElement,
                Leaf1: MStorageElement,
                Leaf2: MStorageElement,
                Leaf3: MStorageElement,
                Leaf4: MStorageElement,
                Leaf5: MStorageElement,
                Leaf6: MStorageElement,
                Expr: crate::expr::LogicalDeviceExpr7<
                    Input,
                    Leaf0,
                    Leaf1,
                    Leaf2,
                    Leaf3,
                    Leaf4,
                    Leaf5,
                    Leaf6,
                >,
                Op: op::UnaryOp<R, Input, Output = Self>,
                Self: crate::detail::TransformLogical7Output<
                    R,
                    Input,
                    Leaf0,
                    Leaf1,
                    Leaf2,
                    Leaf3,
                    Leaf4,
                    Leaf5,
                    Leaf6,
                    Expr,
                    KernelOp<R, Op>,
                >,
                <Self as crate::detail::MItemStorage<
                    R,
                >>::Storage: crate::detail::MaterializeOutput<
                    Runtime = R,
                    Output = ($(
                        crate::detail::DeviceVec<R, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage = <Self as crate::detail::TransformLogical7Output<
                    R,
                    Input,
                    Leaf0,
                    Leaf1,
                    Leaf2,
                    Leaf3,
                    Leaf4,
                    Leaf5,
                    Leaf6,
                    Expr,
                    KernelOp<R, Op>,
                >>::run_logical7(policy, bindings, len, env)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            fn transform_quaternary<First, Second, Third, Fourth, Op>(
                policy: &crate::detail::CubePolicy<R>,
                first: crate::detail::device::DeviceColumnView<R, First>,
                second: crate::detail::device::DeviceColumnView<R, Second>,
                third: crate::detail::device::DeviceColumnView<R, Third>,
                fourth: crate::detail::device::DeviceColumnView<R, Fourth>,
                op: Op,
                env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<<Self as MAlloc<R>>::Inner, Error>
            where
                First: MStorageElement,
                Second: MStorageElement,
                Third: MStorageElement,
                Fourth: MStorageElement,
                Op: op::UnaryOp<R, (First, Second, Third, Fourth), Output = Self>,
                Self: crate::detail::TransformZip4Output<
                    R,
                    First,
                    Second,
                    Third,
                    Fourth,
                    KernelOp<R, Op>,
                >,
                <Self as crate::detail::MItemStorage<
                    R,
                >>::Storage: crate::detail::MaterializeOutput<
                    Runtime = R,
                    Output = ($(
                        crate::detail::DeviceVec<R, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage = crate::detail::apply::TransformPayloadApply::zip4::<Self, R, First, Second, Third, Fourth, KernelOp<R, Op>>(policy, first, second, third, fourth, env)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            fn transform_quinary<First, Second, Third, Fourth, Fifth, Op>(
                policy: &crate::detail::CubePolicy<R>,
                first: crate::detail::device::DeviceColumnView<R, First>,
                second: crate::detail::device::DeviceColumnView<R, Second>,
                third: crate::detail::device::DeviceColumnView<R, Third>,
                fourth: crate::detail::device::DeviceColumnView<R, Fourth>,
                fifth: crate::detail::device::DeviceColumnView<R, Fifth>,
                op: Op,
                env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<<Self as MAlloc<R>>::Inner, Error>
            where
                First: MStorageElement,
                Second: MStorageElement,
                Third: MStorageElement,
                Fourth: MStorageElement,
                Fifth: MStorageElement,
                Op: op::UnaryOp<R, (First, Second, Third, Fourth, Fifth), Output = Self>,
                Self: crate::detail::TransformZip5Output<
                    R,
                    First,
                    Second,
                    Third,
                    Fourth,
                    Fifth,
                    KernelOp<R, Op>,
                >,
                <Self as crate::detail::MItemStorage<
                    R,
                >>::Storage: crate::detail::MaterializeOutput<
                    Runtime = R,
                    Output = ($(
                        crate::detail::DeviceVec<R, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage = crate::detail::apply::TransformPayloadApply::zip5::<Self, R, First, Second, Third, Fourth, Fifth, KernelOp<R, Op>>(policy, first, second, third, fourth, fifth, env)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            #[allow(clippy::too_many_arguments)]
            fn transform_senary<First, Second, Third, Fourth, Fifth, Sixth, Op>(
                policy: &crate::detail::CubePolicy<R>,
                first: crate::detail::device::DeviceColumnView<R, First>,
                second: crate::detail::device::DeviceColumnView<R, Second>,
                third: crate::detail::device::DeviceColumnView<R, Third>,
                fourth: crate::detail::device::DeviceColumnView<R, Fourth>,
                fifth: crate::detail::device::DeviceColumnView<R, Fifth>,
                sixth: crate::detail::device::DeviceColumnView<R, Sixth>,
                op: Op,
                env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<<Self as MAlloc<R>>::Inner, Error>
            where
                First: MStorageElement,
                Second: MStorageElement,
                Third: MStorageElement,
                Fourth: MStorageElement,
                Fifth: MStorageElement,
                Sixth: MStorageElement,
                Op: op::UnaryOp<R, (First, Second, Third, Fourth, Fifth, Sixth), Output = Self>,
                Self: crate::detail::TransformZip6Output<
                    R,
                    First,
                    Second,
                    Third,
                    Fourth,
                    Fifth,
                    Sixth,
                    KernelOp<R, Op>,
                >,
                <Self as crate::detail::MItemStorage<
                    R,
                >>::Storage: crate::detail::MaterializeOutput<
                    Runtime = R,
                    Output = ($(
                        crate::detail::DeviceVec<R, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage = crate::detail::apply::TransformPayloadApply::zip6::<Self, R, First, Second, Third, Fourth, Fifth, Sixth, KernelOp<R, Op>>(policy, first, second, third, fourth, fifth, sixth, env)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            #[allow(clippy::too_many_arguments)]
            fn transform_septenary<First, Second, Third, Fourth, Fifth, Sixth, Seventh, Op>(
                policy: &crate::detail::CubePolicy<R>,
                first: crate::detail::device::DeviceColumnView<R, First>,
                second: crate::detail::device::DeviceColumnView<R, Second>,
                third: crate::detail::device::DeviceColumnView<R, Third>,
                fourth: crate::detail::device::DeviceColumnView<R, Fourth>,
                fifth: crate::detail::device::DeviceColumnView<R, Fifth>,
                sixth: crate::detail::device::DeviceColumnView<R, Sixth>,
                seventh: crate::detail::device::DeviceColumnView<R, Seventh>,
                op: Op,
                env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<<Self as MAlloc<R>>::Inner, Error>
            where
                First: MStorageElement,
                Second: MStorageElement,
                Third: MStorageElement,
                Fourth: MStorageElement,
                Fifth: MStorageElement,
                Sixth: MStorageElement,
                Seventh: MStorageElement,
                Op: op::UnaryOp<
                    R,
                    (First, Second, Third, Fourth, Fifth, Sixth, Seventh),
                    Output = Self,
                >,
                Self: crate::detail::TransformZip7Output<
                    R,
                    First,
                    Second,
                    Third,
                    Fourth,
                    Fifth,
                    Sixth,
                    Seventh,
                    KernelOp<R, Op>,
                >,
                <Self as crate::detail::MItemStorage<
                    R,
                >>::Storage: crate::detail::MaterializeOutput<
                    Runtime = R,
                    Output = ($(
                        crate::detail::DeviceVec<R, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage = crate::detail::apply::TransformPayloadApply::zip7::<Self, R, First, Second, Third, Fourth, Fifth, Sixth, Seventh, KernelOp<R, Op>>(policy, first, second, third, fourth, fifth, sixth, seventh, env)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            fn reduce_inner<Op>(
                policy: &crate::detail::CubePolicy<R>,
                input: <Self as MAlloc<R>>::Inner,
                init: Self,
                op: Op,
            ) -> Result<Self, Error>
            where
                Op: op::ReductionOp<R, Self>,
            {
                let _ = op;
                crate::detail::reduce(policy, input, init, KernelOp::<R, Op>::new())
            }

        }
    };
}

impl_mitem_tuple!(A: a);
impl_mitem_tuple!(A: a, B: b);
impl_mitem_tuple!(A: a, B: b, C: c);

macro_rules! impl_wide_mitem_tuple {
    ($( $ty:ident : $var:ident ),+) => {
        impl<R, $( $ty ),+> MAlloc<R> for ($( $ty, )+)
        where
            R: Runtime,
            $( $ty: MStorageElement, )+
        {
            type Inner = ($( crate::detail::DeviceVec<R, $ty>, )+);
            type View = ($( crate::detail::device::DeviceColumnView<R, $ty>, )+);
            type Storage = zip_type!($( DeviceVec<R, $ty> ),+);

            fn storage_from_inner(inner: Self::Inner) -> Self::Storage {
                let ($( $var, )+) = inner;
                zip_value!($( DeviceVec::from_inner($var) ),+)
            }

            fn alloc_storage(exec: &Executor<R>, len: MIndex) -> Result<Self::Storage, Error> {
                Ok(Self::storage_from_inner(alloc_inner!(exec, len; $( $ty ),+)?))
            }

            fn reverse_from_view<Output>(
                policy: &crate::detail::CubePolicy<R>,
                input: Self::View,
                output: Output,
            ) -> Result<(), Error>
            where
                Output: MIterMut<R, Item = Self>,
            {
                let inner = wide_reverse_from_tuple!(policy, input; $( $var ),+)?;
                output.write_from_inner(policy, inner)
            }

            fn sort_from_view<Less, Output>(
                policy: &crate::detail::CubePolicy<R>,
                input: Self::View,
                _less: Less,
                output: Output,
            ) -> Result<(), Error>
            where
                Less: op::BinaryPredicateOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                let inner = wide_sort_from_tuple!(policy, input; $( $var ),+)?;
                output.write_from_inner(policy, inner)
            }

            fn sort_by_key_values_from_view<Output>(
                policy: &crate::detail::CubePolicy<R>,
                values: Self::View,
                control: &crate::detail::control::PermutationControl<R>,
                output: Output,
            ) -> Result<(), Error>
            where
                Output: MIterMut<R, Item = Self>,
            {
                let values =
                    crate::detail::read::KernelSortByKeyValues::sort_by_key_values(
                        values,
                        policy,
                        control,
                    )?;
                let inner = crate::detail::api::MaterializeOutput::materialize_output(values, policy)?;
                output.write_from_inner(policy, inner)
            }

            fn merge_by_key_values_from_views<Output>(
                policy: &crate::detail::CubePolicy<R>,
                left_values: Self::View,
                right_values: Self::View,
                control: &crate::detail::control::MergeByKeyControl,
                output: Output,
            ) -> Result<(), Error>
            where
                Output: MIterMut<R, Item = Self>,
            {
                let values =
                    crate::detail::read::KernelMergeByKeyValues::merge_by_key_values(
                        left_values,
                        policy,
                        right_values,
                        control,
                    )?;
                let inner = crate::detail::api::MaterializeOutput::materialize_output(values, policy)?;
                output.write_from_inner(policy, inner)
            }

            fn unique_by_key_values_from_view<Output>(
                policy: &crate::detail::CubePolicy<R>,
                values: Self::View,
                control: &crate::detail::control::UniqueByKeyControl,
                output: Output,
            ) -> Result<(), Error>
            where
                Output: MIterMut<R, Item = Self>,
            {
                let values = wide_unique_by_key_values_from_tuple!(policy, values, control; $( $var ),+)?;
                let inner = crate::detail::api::MaterializeOutput::materialize_output(values, policy)?;
                output.write_prefix_from_inner(policy, inner)
            }

            fn reduce_by_key_values_from_view<KeyEq, Op, Output>(
                policy: &crate::detail::CubePolicy<R>,
                values: Self::View,
                control: &crate::detail::control::ReduceByKeyControl<R>,
                init: Self,
                _op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                Op: op::ReductionOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                let values =
                    <_ as crate::detail::read::KernelReduceByKeyValues<
                        crate::detail::control::ReduceByKeyControl<R>,
                        KernelOp<R, KeyEq>,
                        KernelOp<R, Op>,
                    >>::reduce_by_key_values(values, policy, control, init)?;
                let inner = crate::detail::api::MaterializeOutput::materialize_output(values, policy)?;
                output.write_prefix_from_inner(policy, inner)
            }

            fn inclusive_scan_by_key_values_from_inner<KeyEq, Op, Output>(
                policy: &crate::detail::CubePolicy<R>,
                values: Self::Inner,
                control: &crate::detail::control::ScanByKeyControl<R>,
                op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                Op: op::ReductionOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                <Self as crate::detail::read::ScanByKeyValueItem<R>>::inclusive_scan_by_key_values_from_inner::<
                    Op,
                    Output,
                >(policy, values, control, op, output)
            }

            fn exclusive_scan_by_key_values_from_inner<KeyEq, Op, Output>(
                policy: &crate::detail::CubePolicy<R>,
                values: Self::Inner,
                control: &crate::detail::control::ScanByKeyControl<R>,
                init: Self,
                op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                Op: op::ReductionOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                <Self as crate::detail::read::ScanByKeyValueItem<R>>::exclusive_scan_by_key_values_from_inner::<
                    Op,
                    Output,
                >(policy, values, control, init, op, output)
            }

            fn copy_selected_from_view<Output>(
                policy: &crate::detail::CubePolicy<R>,
                values: Self::View,
                stencil: crate::detail::api::PrecomputedSelection<R>,
                output: Output,
            ) -> Result<MIndex, Error>
            where
                Output: MIterMut<R, Item = Self>,
            {
                let selected_rank = stencil.selected_rank();
                let count =
                    crate::detail::primitives::select::selected_count(policy, selected_rank)?;
                let control = crate::detail::control::UniqueByKeyControl {
                    selection: selected_rank.clone(),
                    count,
                };
                let values = wide_unique_by_key_values_from_tuple!(
                    policy,
                    values,
                    control;
                    $( $var ),+
                )?;
                let inner = crate::detail::api::MaterializeOutput::materialize_output(values, policy)?;
                let len = mindex_from_usize(count)?;
                output.write_prefix_from_inner(policy, inner)?;
                Ok(len)
            }

            fn gather_from_view<Output>(
                policy: &crate::detail::CubePolicy<R>,
                values: Self::View,
                indices: crate::detail::device::DeviceColumnView<R, MIndex>,
                output: Output,
            ) -> Result<(), Error>
            where
                Output: MIterMut<R, Item = Self>,
            {
                let inner = indexed_gather_inner_from_tuple!(policy, values, indices; $( $var ),+)?;
                output.write_from_inner(policy, inner)
            }

            fn gather_where_from_view<Output>(
                policy: &crate::detail::CubePolicy<R>,
                values: Self::View,
                indices: crate::detail::device::DeviceColumnView<R, MIndex>,
                stencil: crate::detail::api::PrecomputedSelection<R>,
                output: Output,
            ) -> Result<(), Error>
            where
                Output: MIterMut<R, Item = Self>,
            {
                let mask = stencil.mask();
                indexed_apply_arity!(
                    crate::detail::apply::MaskedIndexedExprApply::gather_where_expr_into,
                    policy, values, indices, output, &mask;
                    $( $ty: $var ),+
                )
            }

            fn scatter_from_view<Output>(
                policy: &crate::detail::CubePolicy<R>,
                values: Self::View,
                indices: crate::detail::device::DeviceColumnView<R, MIndex>,
                output: Output,
            ) -> Result<(), Error>
            where
                Output: MIterMut<R, Item = Self>,
            {
                indexed_apply_arity!(
                    crate::detail::apply::IndexedExprApply::scatter_expr_into,
                    policy, values, indices, output;
                    $( $ty: $var ),+
                )
            }

            fn scatter_where_from_view<Output>(
                policy: &crate::detail::CubePolicy<R>,
                values: Self::View,
                indices: crate::detail::device::DeviceColumnView<R, MIndex>,
                stencil: crate::detail::api::PrecomputedSelection<R>,
                output: Output,
            ) -> Result<(), Error>
            where
                Output: MIterMut<R, Item = Self>,
            {
                let mask = stencil.mask();
                indexed_apply_arity!(
                    crate::detail::apply::MaskedIndexedExprApply::scatter_where_expr_into,
                    policy, values, indices, output, &mask;
                    $( $ty: $var ),+
                )
            }

            fn unique_from_view<Pred, Output>(
                policy: &crate::detail::CubePolicy<R>,
                input: Self::View,
                _pred: Pred,
                output: Output,
            ) -> Result<MIndex, Error>
            where
                Pred: op::BinaryPredicateOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                let inner = wide_unique_from_tuple!(policy, input; $( $var ),+)?;
                let len = mindex_from_usize(inner.0.len())?;
                output.write_prefix_from_inner(policy, inner)?;
                Ok(len)
            }

            fn reduce_from_view<Op>(
                policy: &crate::detail::CubePolicy<R>,
                input: Self::View,
                init: Self,
                _op: Op,
            ) -> Result<Self, Error>
            where
                Op: op::ReductionOp<R, Self>,
            {
                let ($( $var, )+) = input;
                wide_reduce_from_tuple!(policy, init; $( $var ),+)
            }

            fn partition_from_view<Pred, Output>(
                policy: &crate::detail::CubePolicy<R>,
                input: Self::View,
                _pred: Pred,
                env: <Pred::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
                output: Output,
            ) -> Result<MIndex, Error>
            where
                Pred: op::PredicateOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                let ($( $var, )+) = input;
                let selected_rank = wide_predicate_rank_from_tuple!(
                    policy,
                    env;
                    Pred;
                    $( &$var ),+
                )?;
                let (split_rank, matching_count, failing_count) =
                    crate::detail::primitives::select::split_rank_from_selected(
                        policy,
                        selected_rank,
                    )?;
                let apply = crate::detail::apply::SplitPayloadApply::new(
                    &split_rank,
                    matching_count,
                    failing_count,
                );
                let (matching, failing) =
                    wide_partition_apply_from_tuple!(apply, policy; $( $var ),+)?;
                let split = mindex_from_usize(matching_count)?;
                output.write_split_from_inner(policy, matching, failing)?;
                Ok(split)
            }

            fn adjacent_difference_from_view<Op, Output>(
                policy: &crate::detail::CubePolicy<R>,
                input: Self::View,
                _op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                Op: op::ReductionOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                let inner = wide_adjacent_difference_from_tuple!(policy, input; $( $var ),+)?;
                output.write_from_inner(policy, inner)
            }

            fn inclusive_scan_from_view<Op, Output>(
                policy: &crate::detail::CubePolicy<R>,
                input: Self::View,
                _op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                Op: op::ReductionOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                let inner = wide_inclusive_scan_from_tuple!(policy, input; $( $var ),+)?;
                output.write_from_inner(policy, inner)
            }

            fn exclusive_scan_from_view<Op, Output>(
                policy: &crate::detail::CubePolicy<R>,
                input: Self::View,
                init: Self,
                _op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                Op: op::ReductionOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                let inner = wide_exclusive_scan_from_tuple!(policy, input, init; $( $var ),+)?;
                output.write_from_inner(policy, inner)
            }

            fn merge_from_views<Less, Output>(
                policy: &crate::detail::CubePolicy<R>,
                left: Self::View,
                right: Self::View,
                _less: Less,
                output: Output,
            ) -> Result<(), Error>
            where
                Less: op::BinaryPredicateOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                let inner = wide_merge_from_tuple!(policy, left, right; $( $var ),+)?;
                output.write_from_inner(policy, inner)
            }

            fn set_union_from_views<Less, Output>(
                policy: &crate::detail::CubePolicy<R>,
                left: Self::View,
                right: Self::View,
                _less: Less,
                output: Output,
            ) -> Result<MIndex, Error>
            where
                Less: op::BinaryPredicateOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                let (right_only, _) = wide_set_selected_inner_from_tuple!(
                    policy,
                    right,
                    left.clone(),
                    false;
                    KernelOp<R, Less>;
                    $( $var ),+
                )?;
                let right_only = wide_view_from_inner_tuple!(right_only; $( $var ),+);
                let inner = wide_merge_from_tuple!(policy, left, right_only; $( $var ),+)?;
                let len = inner.0.len();
                output.write_prefix_from_inner(policy, inner)?;
                mindex_from_usize(len)
            }

            fn set_intersection_from_views<Less, Output>(
                policy: &crate::detail::CubePolicy<R>,
                left: Self::View,
                right: Self::View,
                _less: Less,
                output: Output,
            ) -> Result<MIndex, Error>
            where
                Less: op::BinaryPredicateOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                let (inner, count) = wide_set_selected_inner_from_tuple!(
                    policy,
                    left,
                    right,
                    true;
                    KernelOp<R, Less>;
                    $( $var ),+
                )?;
                output.write_prefix_from_inner(policy, inner)?;
                mindex_from_usize(count)
            }

            fn set_difference_from_views<Less, Output>(
                policy: &crate::detail::CubePolicy<R>,
                left: Self::View,
                right: Self::View,
                _less: Less,
                output: Output,
            ) -> Result<MIndex, Error>
            where
                Less: op::BinaryPredicateOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                let (inner, count) = wide_set_selected_inner_from_tuple!(
                    policy,
                    left,
                    right,
                    false;
                    KernelOp<R, Less>;
                    $( $var ),+
                )?;
                output.write_prefix_from_inner(policy, inner)?;
                mindex_from_usize(count)
            }

        }

        impl<R, $( $ty ),+> StorageFromInner<R> for zip_type!($( DeviceVec<R, $ty> ),+)
        where
            R: Runtime,
            $( $ty: MStorageElement + 'static, )+
        {
            type Item = ($( $ty, )+);

            fn from_inner(inner: <Self::Item as MAlloc<R>>::Inner) -> Self {
                <Self::Item as MAlloc<R>>::storage_from_inner(inner)
            }

            fn into_inner(self) -> <Self::Item as MAlloc<R>>::Inner {
                zip_into_inner!(self; $( $var ),+)
            }

            fn len(&self) -> MIndex {
                self.0.len()
            }
        }

        impl<R, $( $ty ),+> sealed::MItemDispatch<R> for ($( $ty, )+)
        where
            R: Runtime,
            $( $ty: MStorageElement, )+
        {
            fn transform_scalar_input<Input, Op>(
                policy: &crate::detail::CubePolicy<R>,
                input: crate::detail::device::DeviceColumnView<R, Input>,
                op: Op,
                env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<<Self as MAlloc<R>>::Inner, Error>
            where
                Input: MStorageElement + MItem<R>,
                Op: op::UnaryOp<R, Input, Output = Self>,
                Self: crate::detail::TransformUnaryOutput<
                    R,
                    Input,
                    KernelScalarInputOp<R, Op>,
                >,
                <Self as crate::detail::MItemStorage<
                    R,
                >>::Storage: crate::detail::MaterializeOutput<
                    Runtime = R,
                    Output = ($(
                        crate::detail::DeviceVec<R, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage = crate::detail::apply::TransformPayloadApply::unary::<Self, R, Input, KernelScalarInputOp<R, Op>>(policy, input, env)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            fn transform_unary<Input, Op>(
                policy: &crate::detail::CubePolicy<R>,
                input: crate::detail::device::DeviceColumnView<
                    R,
                    Input,
                >,
                op: Op,
                env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<<Self as MAlloc<R>>::Inner, Error>
            where
                Input: MStorageElement,
                Op: op::UnaryOp<R, (Input,), Output = Self>,
                Self: crate::detail::TransformUnaryOutput<
                    R,
                    Input,
                    KernelOp<R, Op>,
                >,
                <Self as crate::detail::MItemStorage<
                    R,
                >>::Storage: crate::detail::MaterializeOutput<
                    Runtime = R,
                    Output = ($(
                        crate::detail::DeviceVec<R, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage = crate::detail::apply::TransformPayloadApply::unary::<Self, R, Input, KernelOp<R, Op>>(policy, input, env)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            fn transform_binary<Left, Right, Op>(
                policy: &crate::detail::CubePolicy<R>,
                left: crate::detail::device::DeviceColumnView<
                    R,
                    Left,
                >,
                right: crate::detail::device::DeviceColumnView<
                    R,
                    Right,
                >,
                op: Op,
                env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<<Self as MAlloc<R>>::Inner, Error>
            where
                Left: MStorageElement,
                Right: MStorageElement,
                Op: op::UnaryOp<R, (Left, Right), Output = Self>,
                Self: crate::detail::TransformZip2Output<
                    R,
                    Left,
                    Right,
                    KernelOp<R, Op>,
                >,
                <Self as crate::detail::MItemStorage<
                    R,
                >>::Storage: crate::detail::MaterializeOutput<
                    Runtime = R,
                    Output = ($(
                        crate::detail::DeviceVec<R, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage = crate::detail::apply::TransformPayloadApply::zip2::<Self, R, Left, Right, KernelOp<R, Op>>(policy, left, right, env)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            fn transform_ternary<First, Second, Third, Op>(
                policy: &crate::detail::CubePolicy<R>,
                first: crate::detail::device::DeviceColumnView<
                    R,
                    First,
                >,
                second: crate::detail::device::DeviceColumnView<
                    R,
                    Second,
                >,
                third: crate::detail::device::DeviceColumnView<
                    R,
                    Third,
                >,
                op: Op,
                env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<<Self as MAlloc<R>>::Inner, Error>
            where
                First: MStorageElement,
                Second: MStorageElement,
                Third: MStorageElement,
                Op: op::UnaryOp<R, (First, Second, Third), Output = Self>,
                Self: crate::detail::TransformZip3Output<
                    R,
                    First,
                    Second,
                    Third,
                    KernelOp<R, Op>,
                >,
                <Self as crate::detail::MItemStorage<
                    R,
                >>::Storage: crate::detail::MaterializeOutput<
                    Runtime = R,
                    Output = ($(
                        crate::detail::DeviceVec<R, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage = crate::detail::apply::TransformPayloadApply::zip3::<Self, R, First, Second, Third, KernelOp<R, Op>>(policy, first, second, third, env)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            fn transform_logical3<Input, LeafA, LeafB, LeafC, Expr, Op>(
                policy: &crate::detail::CubePolicy<R>,
                bindings: crate::detail::device::KernelColumnBindings,
                len: usize,
                op: Op,
                env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<<Self as MAlloc<R>>::Inner, Error>
            where
                Input: MItem<R> + Send + Sync,
                LeafA: MStorageElement,
                LeafB: MStorageElement,
                LeafC: MStorageElement,
                Expr: crate::expr::LogicalDeviceExpr3<Input, LeafA, LeafB, LeafC>,
                Op: op::UnaryOp<R, Input, Output = Self>,
                Self: crate::detail::TransformLogical3Output<
                    R,
                    Input,
                    LeafA,
                    LeafB,
                    LeafC,
                    Expr,
                    KernelOp<R, Op>,
                >,
                <Self as crate::detail::MItemStorage<
                    R,
                >>::Storage: crate::detail::MaterializeOutput<
                    Runtime = R,
                    Output = ($(
                        crate::detail::DeviceVec<R, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage = <Self as crate::detail::TransformLogical3Output<
                    R,
                    Input,
                    LeafA,
                    LeafB,
                    LeafC,
                    Expr,
                    KernelOp<R, Op>,
                >>::run_logical3(policy, bindings, len, env)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            #[allow(clippy::too_many_arguments)]
            fn transform_logical7<
                Input,
                Leaf0,
                Leaf1,
                Leaf2,
                Leaf3,
                Leaf4,
                Leaf5,
                Leaf6,
                Expr,
                Op,
            >(
                policy: &crate::detail::CubePolicy<R>,
                bindings: crate::detail::device::KernelColumnBindings,
                len: usize,
                op: Op,
                env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<<Self as MAlloc<R>>::Inner, Error>
            where
                Input: MItem<R> + Send + Sync,
                Leaf0: MStorageElement,
                Leaf1: MStorageElement,
                Leaf2: MStorageElement,
                Leaf3: MStorageElement,
                Leaf4: MStorageElement,
                Leaf5: MStorageElement,
                Leaf6: MStorageElement,
                Expr: crate::expr::LogicalDeviceExpr7<
                    Input,
                    Leaf0,
                    Leaf1,
                    Leaf2,
                    Leaf3,
                    Leaf4,
                    Leaf5,
                    Leaf6,
                >,
                Op: op::UnaryOp<R, Input, Output = Self>,
                Self: crate::detail::TransformLogical7Output<
                    R,
                    Input,
                    Leaf0,
                    Leaf1,
                    Leaf2,
                    Leaf3,
                    Leaf4,
                    Leaf5,
                    Leaf6,
                    Expr,
                    KernelOp<R, Op>,
                >,
                <Self as crate::detail::MItemStorage<
                    R,
                >>::Storage: crate::detail::MaterializeOutput<
                    Runtime = R,
                    Output = ($(
                        crate::detail::DeviceVec<R, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage = <Self as crate::detail::TransformLogical7Output<
                    R,
                    Input,
                    Leaf0,
                    Leaf1,
                    Leaf2,
                    Leaf3,
                    Leaf4,
                    Leaf5,
                    Leaf6,
                    Expr,
                    KernelOp<R, Op>,
                >>::run_logical7(policy, bindings, len, env)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            fn transform_quaternary<First, Second, Third, Fourth, Op>(
                policy: &crate::detail::CubePolicy<R>,
                first: crate::detail::device::DeviceColumnView<R, First>,
                second: crate::detail::device::DeviceColumnView<R, Second>,
                third: crate::detail::device::DeviceColumnView<R, Third>,
                fourth: crate::detail::device::DeviceColumnView<R, Fourth>,
                op: Op,
                env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<<Self as MAlloc<R>>::Inner, Error>
            where
                First: MStorageElement,
                Second: MStorageElement,
                Third: MStorageElement,
                Fourth: MStorageElement,
                Op: op::UnaryOp<R, (First, Second, Third, Fourth), Output = Self>,
                Self: crate::detail::TransformZip4Output<
                    R,
                    First,
                    Second,
                    Third,
                    Fourth,
                    KernelOp<R, Op>,
                >,
                <Self as crate::detail::MItemStorage<
                    R,
                >>::Storage: crate::detail::MaterializeOutput<
                    Runtime = R,
                    Output = ($(
                        crate::detail::DeviceVec<R, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage = crate::detail::apply::TransformPayloadApply::zip4::<Self, R, First, Second, Third, Fourth, KernelOp<R, Op>>(policy, first, second, third, fourth, env)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            fn transform_quinary<First, Second, Third, Fourth, Fifth, Op>(
                policy: &crate::detail::CubePolicy<R>,
                first: crate::detail::device::DeviceColumnView<R, First>,
                second: crate::detail::device::DeviceColumnView<R, Second>,
                third: crate::detail::device::DeviceColumnView<R, Third>,
                fourth: crate::detail::device::DeviceColumnView<R, Fourth>,
                fifth: crate::detail::device::DeviceColumnView<R, Fifth>,
                op: Op,
                env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<<Self as MAlloc<R>>::Inner, Error>
            where
                First: MStorageElement,
                Second: MStorageElement,
                Third: MStorageElement,
                Fourth: MStorageElement,
                Fifth: MStorageElement,
                Op: op::UnaryOp<R, (First, Second, Third, Fourth, Fifth), Output = Self>,
                Self: crate::detail::TransformZip5Output<
                    R,
                    First,
                    Second,
                    Third,
                    Fourth,
                    Fifth,
                    KernelOp<R, Op>,
                >,
                <Self as crate::detail::MItemStorage<
                    R,
                >>::Storage: crate::detail::MaterializeOutput<
                    Runtime = R,
                    Output = ($(
                        crate::detail::DeviceVec<R, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage = crate::detail::apply::TransformPayloadApply::zip5::<Self, R, First, Second, Third, Fourth, Fifth, KernelOp<R, Op>>(policy, first, second, third, fourth, fifth, env)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            #[allow(clippy::too_many_arguments)]
            fn transform_senary<First, Second, Third, Fourth, Fifth, Sixth, Op>(
                policy: &crate::detail::CubePolicy<R>,
                first: crate::detail::device::DeviceColumnView<R, First>,
                second: crate::detail::device::DeviceColumnView<R, Second>,
                third: crate::detail::device::DeviceColumnView<R, Third>,
                fourth: crate::detail::device::DeviceColumnView<R, Fourth>,
                fifth: crate::detail::device::DeviceColumnView<R, Fifth>,
                sixth: crate::detail::device::DeviceColumnView<R, Sixth>,
                op: Op,
                env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<<Self as MAlloc<R>>::Inner, Error>
            where
                First: MStorageElement,
                Second: MStorageElement,
                Third: MStorageElement,
                Fourth: MStorageElement,
                Fifth: MStorageElement,
                Sixth: MStorageElement,
                Op: op::UnaryOp<R, (First, Second, Third, Fourth, Fifth, Sixth), Output = Self>,
                Self: crate::detail::TransformZip6Output<
                    R,
                    First,
                    Second,
                    Third,
                    Fourth,
                    Fifth,
                    Sixth,
                    KernelOp<R, Op>,
                >,
                <Self as crate::detail::MItemStorage<
                    R,
                >>::Storage: crate::detail::MaterializeOutput<
                    Runtime = R,
                    Output = ($(
                        crate::detail::DeviceVec<R, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage = crate::detail::apply::TransformPayloadApply::zip6::<Self, R, First, Second, Third, Fourth, Fifth, Sixth, KernelOp<R, Op>>(policy, first, second, third, fourth, fifth, sixth, env)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            #[allow(clippy::too_many_arguments)]
            fn transform_septenary<First, Second, Third, Fourth, Fifth, Sixth, Seventh, Op>(
                policy: &crate::detail::CubePolicy<R>,
                first: crate::detail::device::DeviceColumnView<R, First>,
                second: crate::detail::device::DeviceColumnView<R, Second>,
                third: crate::detail::device::DeviceColumnView<R, Third>,
                fourth: crate::detail::device::DeviceColumnView<R, Fourth>,
                fifth: crate::detail::device::DeviceColumnView<R, Fifth>,
                sixth: crate::detail::device::DeviceColumnView<R, Sixth>,
                seventh: crate::detail::device::DeviceColumnView<R, Seventh>,
                op: Op,
                env: <Op::Env as cubecl::prelude::LaunchArg>::RuntimeArg<R>,
            ) -> Result<<Self as MAlloc<R>>::Inner, Error>
            where
                First: MStorageElement,
                Second: MStorageElement,
                Third: MStorageElement,
                Fourth: MStorageElement,
                Fifth: MStorageElement,
                Sixth: MStorageElement,
                Seventh: MStorageElement,
                Op: op::UnaryOp<
                    R,
                    (First, Second, Third, Fourth, Fifth, Sixth, Seventh),
                    Output = Self,
                >,
                Self: crate::detail::TransformZip7Output<
                    R,
                    First,
                    Second,
                    Third,
                    Fourth,
                    Fifth,
                    Sixth,
                    Seventh,
                    KernelOp<R, Op>,
                >,
                <Self as crate::detail::MItemStorage<
                    R,
                >>::Storage: crate::detail::MaterializeOutput<
                    Runtime = R,
                    Output = ($(
                        crate::detail::DeviceVec<R, $ty>,
                    )+),
                >,
            {
                let _ = op;
                let storage = crate::detail::apply::TransformPayloadApply::zip7::<Self, R, First, Second, Third, Fourth, Fifth, Sixth, Seventh, KernelOp<R, Op>>(policy, first, second, third, fourth, fifth, sixth, seventh, env)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }
        }
    };
}

impl_wide_mitem_tuple!(A: a, B: b, C: c, D: d);
impl_wide_mitem_tuple!(A: a, B: b, C: c, D: d, E: e);
impl_wide_mitem_tuple!(A: a, B: b, C: c, D: d, E: e, F: f);
impl_wide_mitem_tuple!(A: a, B: b, C: c, D: d, E: e, F: f, G: g);
