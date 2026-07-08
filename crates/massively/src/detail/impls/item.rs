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

macro_rules! adjacent_logical7_into_output {
    (
        $kernel:ident,
        $policy:expr,
        $input:expr,
        $output:expr;
        $( $ty:ident : $var:ident => $index:expr ),+
    ) => {{
        let input = $input;
        let len = crate::detail::read::KernelRead::len(&input);
        $(
            let $var = <Output as crate::iter::MIterMut<R>>::column_mut_view_by_index_inner::<$ty>(
                &$output,
                $index,
            )?
            .ok_or_else(|| Error::Launch {
                message: "adjacent_difference output must match input shape".to_string(),
            })?;
        )+
        if len != 0 {
            let client = $policy.client();
            let mut bindings = crate::detail::device::KernelColumnBindings::empty(client);
            <Read as crate::detail::read::KernelReadAtEnv<R, crate::detail::read::Env0>>::stage_at_env(
                &input,
                &mut bindings,
            )?;
            bindings.finish();
            let offsets = bindings.slot_offsets7_handle(client)?;
            let slot0 = bindings.slot_or_first(0);
            let slot1 = bindings.slot_or_first(1);
            let slot2 = bindings.slot_or_first(2);
            let slot3 = bindings.slot_or_first(3);
            let slot4 = bindings.slot_or_first(4);
            let slot5 = bindings.slot_or_first(5);
            let slot6 = bindings.slot_or_first(6);
            let metadata = [
                u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?,
                $(
                    u32::try_from($var.offset)
                        .map_err(|_| Error::LengthTooLarge { len: $var.offset })?,
                )+
            ];
            let metadata_handle = client.create_from_slice(u32::as_bytes(&metadata));
            let block_size = 256_u32;
            let block_count = len.div_ceil(block_size as usize);
            let block_count_u32 =
                u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;
            unsafe {
                crate::kernels::$kernel::launch_unchecked::<
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::Leaf0,
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::Leaf1,
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::Leaf2,
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::Leaf3,
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::Leaf4,
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::Leaf5,
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::Leaf6,
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::ExprAt,
                    KernelOp<R, Op>,
                    $( $ty, )+
                    R,
                >(
                    client,
                    CubeCount::Static(block_count_u32, 1, 1),
                    CubeDim::new_1d(block_size),
                    BufferArg::from_raw_parts(slot0.0.clone(), slot0.1),
                    BufferArg::from_raw_parts(slot1.0.clone(), slot1.1),
                    BufferArg::from_raw_parts(slot2.0.clone(), slot2.1),
                    BufferArg::from_raw_parts(slot3.0.clone(), slot3.1),
                    BufferArg::from_raw_parts(slot4.0.clone(), slot4.1),
                    BufferArg::from_raw_parts(slot5.0.clone(), slot5.1),
                    BufferArg::from_raw_parts(slot6.0.clone(), slot6.1),
                    BufferArg::from_raw_parts(offsets.clone(), 7),
                    BufferArg::from_raw_parts(metadata_handle.clone(), metadata.len()),
                    $(
                        BufferArg::from_raw_parts($var.source.handle.clone(), $var.source.len()),
                    )+
                );
            }
        }
        Ok(())
    }};
}

macro_rules! adjacent_logical7_arity {
    ($kernel:ident, $policy:expr, $input:expr, $output:expr; A: $a:ident) => {
        adjacent_logical7_into_output!($kernel, $policy, $input, $output; A: $a => 0)
    };
    ($kernel:ident, $policy:expr, $input:expr, $output:expr; A: $a:ident, B: $b:ident) => {
        adjacent_logical7_into_output!($kernel, $policy, $input, $output; A: $a => 0, B: $b => 1)
    };
    ($kernel:ident, $policy:expr, $input:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident) => {
        adjacent_logical7_into_output!($kernel, $policy, $input, $output; A: $a => 0, B: $b => 1, C: $c => 2)
    };
    ($kernel:ident, $policy:expr, $input:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident) => {
        adjacent_logical7_into_output!($kernel, $policy, $input, $output; A: $a => 0, B: $b => 1, C: $c => 2, D: $d => 3)
    };
    ($kernel:ident, $policy:expr, $input:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident, E: $e:ident) => {
        adjacent_logical7_into_output!($kernel, $policy, $input, $output; A: $a => 0, B: $b => 1, C: $c => 2, D: $d => 3, E: $e => 4)
    };
    ($kernel:ident, $policy:expr, $input:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident, E: $e:ident, F: $f:ident) => {
        adjacent_logical7_into_output!($kernel, $policy, $input, $output; A: $a => 0, B: $b => 1, C: $c => 2, D: $d => 3, E: $e => 4, F: $f => 5)
    };
    ($kernel:ident, $policy:expr, $input:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident, E: $e:ident, F: $f:ident, G: $g:ident) => {
        adjacent_logical7_into_output!($kernel, $policy, $input, $output; A: $a => 0, B: $b => 1, C: $c => 2, D: $d => 3, E: $e => 4, F: $f => 5, G: $g => 6)
    };
}

macro_rules! adjacent_logical7_auto_arity {
    ($policy:expr, $input:expr, $output:expr; A: $a:ident) => {
        adjacent_logical7_arity!(adjacent_logical7_to_tuple1_kernel, $policy, $input, $output; A: $a)
    };
    ($policy:expr, $input:expr, $output:expr; A: $a:ident, B: $b:ident) => {
        adjacent_logical7_arity!(adjacent_logical7_to_tuple2_kernel, $policy, $input, $output; A: $a, B: $b)
    };
    ($policy:expr, $input:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident) => {
        adjacent_logical7_arity!(adjacent_logical7_to_tuple3_kernel, $policy, $input, $output; A: $a, B: $b, C: $c)
    };
    ($policy:expr, $input:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident) => {
        adjacent_logical7_arity!(adjacent_logical7_to_tuple4_kernel, $policy, $input, $output; A: $a, B: $b, C: $c, D: $d)
    };
    ($policy:expr, $input:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident, E: $e:ident) => {
        adjacent_logical7_arity!(adjacent_logical7_to_tuple5_kernel, $policy, $input, $output; A: $a, B: $b, C: $c, D: $d, E: $e)
    };
    ($policy:expr, $input:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident, E: $e:ident, F: $f:ident) => {
        adjacent_logical7_arity!(adjacent_logical7_to_tuple6_kernel, $policy, $input, $output; A: $a, B: $b, C: $c, D: $d, E: $e, F: $f)
    };
    ($policy:expr, $input:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident, E: $e:ident, F: $f:ident, G: $g:ident) => {
        adjacent_logical7_arity!(adjacent_logical7_to_tuple7_kernel, $policy, $input, $output; A: $a, B: $b, C: $c, D: $d, E: $e, F: $f, G: $g)
    };
}

#[allow(unused_macros)]
macro_rules! inclusive_scan_logical7_into_output {
    (
        $kernel:ident,
        $policy:expr,
        $input:expr,
        $output:expr;
        $( $ty:ident : $var:ident => $index:expr ),+
    ) => {{
        let input = $input;
        let len = crate::detail::read::KernelRead::len(&input);
        $(
            let $var = <Output as crate::iter::MIterMut<R>>::column_mut_view_by_index_inner::<$ty>(
                &$output,
                $index,
            )?
            .ok_or_else(|| Error::Launch {
                message: "inclusive_scan output must match input shape".to_string(),
            })?;
        )+
        if len != 0 {
            let client = $policy.client();
            let mut bindings = crate::detail::device::KernelColumnBindings::empty(client);
            <Read as crate::detail::read::KernelReadAtEnv<R, crate::detail::read::Env0>>::stage_at_env(
                &input,
                &mut bindings,
            )?;
            bindings.finish();
            let offsets = bindings.slot_offsets7_handle(client)?;
            let slot0 = bindings.slot_or_first(0);
            let slot1 = bindings.slot_or_first(1);
            let slot2 = bindings.slot_or_first(2);
            let slot3 = bindings.slot_or_first(3);
            let slot4 = bindings.slot_or_first(4);
            let slot5 = bindings.slot_or_first(5);
            let slot6 = bindings.slot_or_first(6);
            let metadata = [
                u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?,
                $(
                    u32::try_from($var.offset)
                        .map_err(|_| Error::LengthTooLarge { len: $var.offset })?,
                )+
            ];
            let metadata_handle = client.create_from_slice(u32::as_bytes(&metadata));
            let block_size = 256_u32;
            let block_count = len.div_ceil(block_size as usize);
            let block_count_u32 =
                u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;
            unsafe {
                crate::kernels::$kernel::launch_unchecked::<
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::Leaf0,
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::Leaf1,
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::Leaf2,
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::Leaf3,
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::Leaf4,
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::Leaf5,
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::Leaf6,
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::ExprAt,
                    KernelOp<R, Op>,
                    $( $ty, )+
                    R,
                >(
                    client,
                    CubeCount::Static(block_count_u32, 1, 1),
                    CubeDim::new_1d(block_size),
                    BufferArg::from_raw_parts(slot0.0.clone(), slot0.1),
                    BufferArg::from_raw_parts(slot1.0.clone(), slot1.1),
                    BufferArg::from_raw_parts(slot2.0.clone(), slot2.1),
                    BufferArg::from_raw_parts(slot3.0.clone(), slot3.1),
                    BufferArg::from_raw_parts(slot4.0.clone(), slot4.1),
                    BufferArg::from_raw_parts(slot5.0.clone(), slot5.1),
                    BufferArg::from_raw_parts(slot6.0.clone(), slot6.1),
                    BufferArg::from_raw_parts(offsets.clone(), 7),
                    BufferArg::from_raw_parts(metadata_handle.clone(), metadata.len()),
                    $(
                        BufferArg::from_raw_parts($var.source.handle.clone(), $var.source.len()),
                    )+
                );
            }
        }
        Ok(())
    }};
}

#[allow(unused_macros)]
macro_rules! exclusive_scan_logical7_into_output {
    (
        $kernel:ident,
        $policy:expr,
        $input:expr,
        $init:expr,
        $output:expr;
        $( $ty:ident : $var:ident : $init_var:ident => $index:expr ),+
    ) => {{
        let input = $input;
        let ($( $init_var, )+) = $init;
        let len = crate::detail::read::KernelRead::len(&input);
        $(
            let $var = <Output as crate::iter::MIterMut<R>>::column_mut_view_by_index_inner::<$ty>(
                &$output,
                $index,
            )?
            .ok_or_else(|| Error::Launch {
                message: "exclusive_scan output must match input shape".to_string(),
            })?;
        )+
        if len != 0 {
            let client = $policy.client();
            let mut bindings = crate::detail::device::KernelColumnBindings::empty(client);
            <Read as crate::detail::read::KernelReadAtEnv<R, crate::detail::read::Env0>>::stage_at_env(
                &input,
                &mut bindings,
            )?;
            bindings.finish();
            let offsets = bindings.slot_offsets7_handle(client)?;
            let slot0 = bindings.slot_or_first(0);
            let slot1 = bindings.slot_or_first(1);
            let slot2 = bindings.slot_or_first(2);
            let slot3 = bindings.slot_or_first(3);
            let slot4 = bindings.slot_or_first(4);
            let slot5 = bindings.slot_or_first(5);
            let slot6 = bindings.slot_or_first(6);
            let metadata = [
                u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?,
                $(
                    u32::try_from($var.offset)
                        .map_err(|_| Error::LengthTooLarge { len: $var.offset })?,
                )+
            ];
            let metadata_handle = client.create_from_slice(u32::as_bytes(&metadata));
            $(
                let $init_var = client.create_from_slice(<$ty as CubeElement>::as_bytes(&[$init_var]));
            )+
            let block_size = 256_u32;
            let block_count = len.div_ceil(block_size as usize);
            let block_count_u32 =
                u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;
            unsafe {
                crate::kernels::$kernel::launch_unchecked::<
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::Leaf0,
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::Leaf1,
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::Leaf2,
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::Leaf3,
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::Leaf4,
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::Leaf5,
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::Leaf6,
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::ExprAt,
                    KernelOp<R, Op>,
                    $( $ty, )+
                    R,
                >(
                    client,
                    CubeCount::Static(block_count_u32, 1, 1),
                    CubeDim::new_1d(block_size),
                    BufferArg::from_raw_parts(slot0.0.clone(), slot0.1),
                    BufferArg::from_raw_parts(slot1.0.clone(), slot1.1),
                    BufferArg::from_raw_parts(slot2.0.clone(), slot2.1),
                    BufferArg::from_raw_parts(slot3.0.clone(), slot3.1),
                    BufferArg::from_raw_parts(slot4.0.clone(), slot4.1),
                    BufferArg::from_raw_parts(slot5.0.clone(), slot5.1),
                    BufferArg::from_raw_parts(slot6.0.clone(), slot6.1),
                    BufferArg::from_raw_parts(offsets.clone(), 7),
                    BufferArg::from_raw_parts(metadata_handle.clone(), metadata.len()),
                    $(
                        BufferArg::from_raw_parts($init_var.clone(), 1),
                    )+
                    $(
                        BufferArg::from_raw_parts($var.source.handle.clone(), $var.source.len()),
                    )+
                );
            }
        }
        Ok(())
    }};
}

#[allow(unused_macros)]
macro_rules! inclusive_scan_logical7_auto_arity {
    ($policy:expr, $input:expr, $output:expr; A: $a:ident) => {
        inclusive_scan_logical7_into_output!(inclusive_scan_logical7_to_tuple1_kernel, $policy, $input, $output; A: $a => 0)
    };
    ($policy:expr, $input:expr, $output:expr; A: $a:ident, B: $b:ident) => {
        inclusive_scan_logical7_into_output!(inclusive_scan_logical7_to_tuple2_kernel, $policy, $input, $output; A: $a => 0, B: $b => 1)
    };
    ($policy:expr, $input:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident) => {
        inclusive_scan_logical7_into_output!(inclusive_scan_logical7_to_tuple3_kernel, $policy, $input, $output; A: $a => 0, B: $b => 1, C: $c => 2)
    };
    ($policy:expr, $input:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident) => {
        inclusive_scan_logical7_into_output!(inclusive_scan_logical7_to_tuple4_kernel, $policy, $input, $output; A: $a => 0, B: $b => 1, C: $c => 2, D: $d => 3)
    };
    ($policy:expr, $input:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident, E: $e:ident) => {
        inclusive_scan_logical7_into_output!(inclusive_scan_logical7_to_tuple5_kernel, $policy, $input, $output; A: $a => 0, B: $b => 1, C: $c => 2, D: $d => 3, E: $e => 4)
    };
    ($policy:expr, $input:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident, E: $e:ident, F: $f:ident) => {
        inclusive_scan_logical7_into_output!(inclusive_scan_logical7_to_tuple6_kernel, $policy, $input, $output; A: $a => 0, B: $b => 1, C: $c => 2, D: $d => 3, E: $e => 4, F: $f => 5)
    };
    ($policy:expr, $input:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident, E: $e:ident, F: $f:ident, G: $g:ident) => {
        inclusive_scan_logical7_into_output!(inclusive_scan_logical7_to_tuple7_kernel, $policy, $input, $output; A: $a => 0, B: $b => 1, C: $c => 2, D: $d => 3, E: $e => 4, F: $f => 5, G: $g => 6)
    };
}

#[allow(unused_macros)]
macro_rules! exclusive_scan_logical7_auto_arity {
    ($policy:expr, $input:expr, $init:expr, $output:expr; A: $a:ident) => {
        exclusive_scan_logical7_into_output!(exclusive_scan_logical7_to_tuple1_kernel, $policy, $input, $init, $output; A: $a: init_a => 0)
    };
    ($policy:expr, $input:expr, $init:expr, $output:expr; A: $a:ident, B: $b:ident) => {
        exclusive_scan_logical7_into_output!(exclusive_scan_logical7_to_tuple2_kernel, $policy, $input, $init, $output; A: $a: init_a => 0, B: $b: init_b => 1)
    };
    ($policy:expr, $input:expr, $init:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident) => {
        exclusive_scan_logical7_into_output!(exclusive_scan_logical7_to_tuple3_kernel, $policy, $input, $init, $output; A: $a: init_a => 0, B: $b: init_b => 1, C: $c: init_c => 2)
    };
    ($policy:expr, $input:expr, $init:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident) => {
        exclusive_scan_logical7_into_output!(exclusive_scan_logical7_to_tuple4_kernel, $policy, $input, $init, $output; A: $a: init_a => 0, B: $b: init_b => 1, C: $c: init_c => 2, D: $d: init_d => 3)
    };
    ($policy:expr, $input:expr, $init:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident, E: $e:ident) => {
        exclusive_scan_logical7_into_output!(exclusive_scan_logical7_to_tuple5_kernel, $policy, $input, $init, $output; A: $a: init_a => 0, B: $b: init_b => 1, C: $c: init_c => 2, D: $d: init_d => 3, E: $e: init_e => 4)
    };
    ($policy:expr, $input:expr, $init:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident, E: $e:ident, F: $f:ident) => {
        exclusive_scan_logical7_into_output!(exclusive_scan_logical7_to_tuple6_kernel, $policy, $input, $init, $output; A: $a: init_a => 0, B: $b: init_b => 1, C: $c: init_c => 2, D: $d: init_d => 3, E: $e: init_e => 4, F: $f: init_f => 5)
    };
    ($policy:expr, $input:expr, $init:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident, E: $e:ident, F: $f:ident, G: $g:ident) => {
        exclusive_scan_logical7_into_output!(exclusive_scan_logical7_to_tuple7_kernel, $policy, $input, $init, $output; A: $a: init_a => 0, B: $b: init_b => 1, C: $c: init_c => 2, D: $d: init_d => 3, E: $e: init_e => 4, F: $f: init_f => 5, G: $g: init_g => 6)
    };
}

macro_rules! reduce_by_key_logical7_into_output {
    (
        $kernel:ident,
        $policy:expr,
        $input:expr,
        $selection:expr,
        $output_count:expr,
        $init:expr,
        $output:expr;
        $( $ty:ident : $var:ident : $init_var:ident => $index:expr ),+
    ) => {{
        let input = $input;
        let selection = $selection;
        let ($( $init_var, )+) = $init;
        let len = crate::detail::read::KernelRead::len(&input);
        ensure_same_len(len, selection.len)?;
        let _ = $output_count;
        $(
            let $var = <Output as crate::iter::MIterMut<R>>::column_mut_view_by_index_inner::<$ty>(
                &$output,
                $index,
            )?
            .ok_or_else(|| Error::Launch {
                message: "reduce_by_key output must match input shape".to_string(),
            })?;
        )+
        if len != 0 {
            let client = $policy.client();
            let mut bindings = crate::detail::device::KernelColumnBindings::empty(client);
            <Read as crate::detail::read::KernelReadAtEnv<R, crate::detail::read::Env0>>::stage_at_env(
                &input,
                &mut bindings,
            )?;
            bindings.finish();
            let offsets = bindings.slot_offsets7_handle(client)?;
            let slot0 = bindings.slot_or_first(0);
            let slot1 = bindings.slot_or_first(1);
            let slot2 = bindings.slot_or_first(2);
            let slot3 = bindings.slot_or_first(3);
            let slot4 = bindings.slot_or_first(4);
            let slot5 = bindings.slot_or_first(5);
            let slot6 = bindings.slot_or_first(6);
            let metadata = [
                u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?,
                $(
                    u32::try_from($var.offset)
                        .map_err(|_| Error::LengthTooLarge { len: $var.offset })?,
                )+
            ];
            let metadata_handle = client.create_from_slice(u32::as_bytes(&metadata));
            $(
                let $init_var = client.create_from_slice(<$ty as CubeElement>::as_bytes(&[$init_var]));
            )+
            let block_size = 256_u32;
            let block_count = len.div_ceil(block_size as usize);
            let block_count_u32 =
                u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;
            unsafe {
                crate::kernels::$kernel::launch_unchecked::<
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::Leaf0,
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::Leaf1,
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::Leaf2,
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::Leaf3,
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::Leaf4,
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::Leaf5,
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::Leaf6,
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::ExprAt,
                    KernelOp<R, Op>,
                    $( $ty, )+
                    R,
                >(
                    client,
                    CubeCount::Static(block_count_u32, 1, 1),
                    CubeDim::new_1d(block_size),
                    BufferArg::from_raw_parts(selection.flag.clone(), selection.len),
                    BufferArg::from_raw_parts(selection.position.clone(), selection.len),
                    BufferArg::from_raw_parts(slot0.0.clone(), slot0.1),
                    BufferArg::from_raw_parts(slot1.0.clone(), slot1.1),
                    BufferArg::from_raw_parts(slot2.0.clone(), slot2.1),
                    BufferArg::from_raw_parts(slot3.0.clone(), slot3.1),
                    BufferArg::from_raw_parts(slot4.0.clone(), slot4.1),
                    BufferArg::from_raw_parts(slot5.0.clone(), slot5.1),
                    BufferArg::from_raw_parts(slot6.0.clone(), slot6.1),
                    BufferArg::from_raw_parts(offsets.clone(), 7),
                    BufferArg::from_raw_parts(metadata_handle.clone(), metadata.len()),
                    $(
                        BufferArg::from_raw_parts($init_var.clone(), 1),
                    )+
                    $(
                        BufferArg::from_raw_parts($var.source.handle.clone(), $var.source.len()),
                    )+
                );
            }
        }
        Ok(())
    }};
}

macro_rules! reduce_by_key_logical7_auto_arity {
    ($policy:expr, $input:expr, $selection:expr, $output_count:expr, $init:expr, $output:expr; A: $a:ident) => {
        reduce_by_key_logical7_into_output!(reduce_by_key_logical7_to_tuple1_kernel, $policy, $input, $selection, $output_count, $init, $output; A: $a: init_a => 0)
    };
    ($policy:expr, $input:expr, $selection:expr, $output_count:expr, $init:expr, $output:expr; A: $a:ident, B: $b:ident) => {
        reduce_by_key_logical7_into_output!(reduce_by_key_logical7_to_tuple2_kernel, $policy, $input, $selection, $output_count, $init, $output; A: $a: init_a => 0, B: $b: init_b => 1)
    };
    ($policy:expr, $input:expr, $selection:expr, $output_count:expr, $init:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident) => {
        reduce_by_key_logical7_into_output!(reduce_by_key_logical7_to_tuple3_kernel, $policy, $input, $selection, $output_count, $init, $output; A: $a: init_a => 0, B: $b: init_b => 1, C: $c: init_c => 2)
    };
    ($policy:expr, $input:expr, $selection:expr, $output_count:expr, $init:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident) => {
        reduce_by_key_logical7_into_output!(reduce_by_key_logical7_to_tuple4_kernel, $policy, $input, $selection, $output_count, $init, $output; A: $a: init_a => 0, B: $b: init_b => 1, C: $c: init_c => 2, D: $d: init_d => 3)
    };
    ($policy:expr, $input:expr, $selection:expr, $output_count:expr, $init:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident, E: $e:ident) => {
        reduce_by_key_logical7_into_output!(reduce_by_key_logical7_to_tuple5_kernel, $policy, $input, $selection, $output_count, $init, $output; A: $a: init_a => 0, B: $b: init_b => 1, C: $c: init_c => 2, D: $d: init_d => 3, E: $e: init_e => 4)
    };
    ($policy:expr, $input:expr, $selection:expr, $output_count:expr, $init:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident, E: $e:ident, F: $f:ident) => {
        reduce_by_key_logical7_into_output!(reduce_by_key_logical7_to_tuple6_kernel, $policy, $input, $selection, $output_count, $init, $output; A: $a: init_a => 0, B: $b: init_b => 1, C: $c: init_c => 2, D: $d: init_d => 3, E: $e: init_e => 4, F: $f: init_f => 5)
    };
    ($policy:expr, $input:expr, $selection:expr, $output_count:expr, $init:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident, E: $e:ident, F: $f:ident, G: $g:ident) => {
        reduce_by_key_logical7_into_output!(reduce_by_key_logical7_to_tuple7_kernel, $policy, $input, $selection, $output_count, $init, $output; A: $a: init_a => 0, B: $b: init_b => 1, C: $c: init_c => 2, D: $d: init_d => 3, E: $e: init_e => 4, F: $f: init_f => 5, G: $g: init_g => 6)
    };
}

#[allow(unused_macros)]
macro_rules! merge_logical7_into_output {
    (
        $kernel:ident,
        $policy:expr,
        $left:expr,
        $right:expr,
        $output:expr;
        $( $ty:ident : $var:ident => $index:expr ),+
    ) => {{
        let left = $left;
        let right = $right;
        let left_len = crate::detail::read::KernelRead::len(&left);
        let right_len = crate::detail::read::KernelRead::len(&right);
        $(
            let $var = <Output as crate::iter::MIterMut<R>>::column_mut_view_by_index_inner::<$ty>(
                &$output,
                $index,
            )?
            .ok_or_else(|| Error::Launch {
                message: "merge output must match input shape".to_string(),
            })?;
        )+
        if left_len + right_len != 0 {
            let client = $policy.client();
            let mut left_bindings = crate::detail::device::KernelColumnBindings::empty(client);
            <Left as crate::detail::read::KernelReadAtEnv<R, crate::detail::read::Env0>>::stage_at_env(
                &left,
                &mut left_bindings,
            )?;
            left_bindings.finish();
            let mut right_bindings = crate::detail::device::KernelColumnBindings::empty(client);
            <Right as crate::detail::read::KernelReadAtEnv<R, crate::detail::read::Env0>>::stage_at_env(
                &right,
                &mut right_bindings,
            )?;
            right_bindings.finish();
            let left_offsets = left_bindings.slot_offsets7_handle(client)?;
            let right_offsets = right_bindings.slot_offsets7_handle(client)?;
            let left_slot0 = left_bindings.slot_or_first(0);
            let left_slot1 = left_bindings.slot_or_first(1);
            let left_slot2 = left_bindings.slot_or_first(2);
            let left_slot3 = left_bindings.slot_or_first(3);
            let left_slot4 = left_bindings.slot_or_first(4);
            let left_slot5 = left_bindings.slot_or_first(5);
            let left_slot6 = left_bindings.slot_or_first(6);
            let right_slot0 = right_bindings.slot_or_first(0);
            let right_slot1 = right_bindings.slot_or_first(1);
            let right_slot2 = right_bindings.slot_or_first(2);
            let right_slot3 = right_bindings.slot_or_first(3);
            let right_slot4 = right_bindings.slot_or_first(4);
            let right_slot5 = right_bindings.slot_or_first(5);
            let right_slot6 = right_bindings.slot_or_first(6);
            let metadata = [
                u32::try_from(left_len).map_err(|_| Error::LengthTooLarge { len: left_len })?,
                u32::try_from(right_len).map_err(|_| Error::LengthTooLarge { len: right_len })?,
                $(
                    u32::try_from($var.offset)
                        .map_err(|_| Error::LengthTooLarge { len: $var.offset })?,
                )+
            ];
            let metadata_handle = client.create_from_slice(u32::as_bytes(&metadata));
            let len = left_len + right_len;
            let block_size = 256_u32;
            let launch = crate::detail::launch::launch_1d(client, len, block_size)?;
            unsafe {
                crate::kernels::$kernel::launch_unchecked::<
                    <Left as crate::detail::read::KernelReadBoundMany<R>>::Leaf0,
                    <Left as crate::detail::read::KernelReadBoundMany<R>>::Leaf1,
                    <Left as crate::detail::read::KernelReadBoundMany<R>>::Leaf2,
                    <Left as crate::detail::read::KernelReadBoundMany<R>>::Leaf3,
                    <Left as crate::detail::read::KernelReadBoundMany<R>>::Leaf4,
                    <Left as crate::detail::read::KernelReadBoundMany<R>>::Leaf5,
                    <Left as crate::detail::read::KernelReadBoundMany<R>>::Leaf6,
                    <Right as crate::detail::read::KernelReadBoundMany<R>>::Leaf0,
                    <Right as crate::detail::read::KernelReadBoundMany<R>>::Leaf1,
                    <Right as crate::detail::read::KernelReadBoundMany<R>>::Leaf2,
                    <Right as crate::detail::read::KernelReadBoundMany<R>>::Leaf3,
                    <Right as crate::detail::read::KernelReadBoundMany<R>>::Leaf4,
                    <Right as crate::detail::read::KernelReadBoundMany<R>>::Leaf5,
                    <Right as crate::detail::read::KernelReadBoundMany<R>>::Leaf6,
                    <Left as crate::detail::read::KernelReadBoundMany<R>>::ExprAt,
                    <Right as crate::detail::read::KernelReadBoundMany<R>>::ExprAt,
                    KernelOp<R, Less>,
                    $( $ty, )+
                    R,
                >(
                    client,
                    launch.cube_count(),
                    CubeDim::new_1d(block_size),
                    BufferArg::from_raw_parts(left_slot0.0.clone(), left_slot0.1),
                    BufferArg::from_raw_parts(left_slot1.0.clone(), left_slot1.1),
                    BufferArg::from_raw_parts(left_slot2.0.clone(), left_slot2.1),
                    BufferArg::from_raw_parts(left_slot3.0.clone(), left_slot3.1),
                    BufferArg::from_raw_parts(left_slot4.0.clone(), left_slot4.1),
                    BufferArg::from_raw_parts(left_slot5.0.clone(), left_slot5.1),
                    BufferArg::from_raw_parts(left_slot6.0.clone(), left_slot6.1),
                    BufferArg::from_raw_parts(left_offsets.clone(), 7),
                    BufferArg::from_raw_parts(right_slot0.0.clone(), right_slot0.1),
                    BufferArg::from_raw_parts(right_slot1.0.clone(), right_slot1.1),
                    BufferArg::from_raw_parts(right_slot2.0.clone(), right_slot2.1),
                    BufferArg::from_raw_parts(right_slot3.0.clone(), right_slot3.1),
                    BufferArg::from_raw_parts(right_slot4.0.clone(), right_slot4.1),
                    BufferArg::from_raw_parts(right_slot5.0.clone(), right_slot5.1),
                    BufferArg::from_raw_parts(right_slot6.0.clone(), right_slot6.1),
                    BufferArg::from_raw_parts(right_offsets.clone(), 7),
                    BufferArg::from_raw_parts(metadata_handle.clone(), metadata.len()),
                    $(
                        BufferArg::from_raw_parts($var.source.handle.clone(), $var.source.len()),
                    )+
                );
            }
        }
        Ok(())
    }};
}

#[allow(unused_macros)]
macro_rules! merge_logical7_auto_arity {
    ($policy:expr, $left:expr, $right:expr, $output:expr; A: $a:ident) => {
        merge_logical7_into_output!(merge_logical7_to_tuple1_kernel, $policy, $left, $right, $output; A: $a => 0)
    };
    ($policy:expr, $left:expr, $right:expr, $output:expr; A: $a:ident, B: $b:ident) => {
        merge_logical7_into_output!(merge_logical7_to_tuple2_kernel, $policy, $left, $right, $output; A: $a => 0, B: $b => 1)
    };
    ($policy:expr, $left:expr, $right:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident) => {
        merge_logical7_into_output!(merge_logical7_to_tuple3_kernel, $policy, $left, $right, $output; A: $a => 0, B: $b => 1, C: $c => 2)
    };
    ($policy:expr, $left:expr, $right:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident) => {
        merge_logical7_into_output!(merge_logical7_to_tuple4_kernel, $policy, $left, $right, $output; A: $a => 0, B: $b => 1, C: $c => 2, D: $d => 3)
    };
    ($policy:expr, $left:expr, $right:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident, E: $e:ident) => {
        merge_logical7_into_output!(merge_logical7_to_tuple5_kernel, $policy, $left, $right, $output; A: $a => 0, B: $b => 1, C: $c => 2, D: $d => 3, E: $e => 4)
    };
    ($policy:expr, $left:expr, $right:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident, E: $e:ident, F: $f:ident) => {
        merge_logical7_into_output!(merge_logical7_to_tuple6_kernel, $policy, $left, $right, $output; A: $a => 0, B: $b => 1, C: $c => 2, D: $d => 3, E: $e => 4, F: $f => 5)
    };
    ($policy:expr, $left:expr, $right:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident, E: $e:ident, F: $f:ident, G: $g:ident) => {
        merge_logical7_into_output!(merge_logical7_to_tuple7_kernel, $policy, $left, $right, $output; A: $a => 0, B: $b => 1, C: $c => 2, D: $d => 3, E: $e => 4, F: $f => 5, G: $g => 6)
    };
}

macro_rules! set_union_logical7_into_output {
    (
        $kernel:ident,
        $policy:expr,
        $left:expr,
        $right:expr,
        $right_only:expr,
        $output:expr;
        $( $ty:ident : $var:ident => $index:expr ),+
    ) => {{
        let left = $left;
        let right = $right;
        let right_only = $right_only;
        let left_len = crate::detail::read::KernelRead::len(&left);
        let right_len = crate::detail::read::KernelRead::len(&right);
        ensure_same_len(right_len, right_only.len)?;
        let count = crate::detail::primitives::select::selected_count($policy, right_only)?;
        $(
            let $var = <Output as crate::iter::MIterMut<R>>::column_mut_view_by_index_inner::<$ty>(
                &$output,
                $index,
            )?
            .ok_or_else(|| Error::Launch {
                message: "set_union output must match input shape".to_string(),
            })?;
        )+
        if left_len + right_len != 0 {
            let client = $policy.client();
            let mut left_bindings = crate::detail::device::KernelColumnBindings::empty(client);
            <Left as crate::detail::read::KernelReadAtEnv<R, crate::detail::read::Env0>>::stage_at_env(
                &left,
                &mut left_bindings,
            )?;
            left_bindings.finish();
            let mut right_bindings = crate::detail::device::KernelColumnBindings::empty(client);
            <Right as crate::detail::read::KernelReadAtEnv<R, crate::detail::read::Env0>>::stage_at_env(
                &right,
                &mut right_bindings,
            )?;
            right_bindings.finish();
            let left_offsets = left_bindings.slot_offsets7_handle(client)?;
            let right_offsets = right_bindings.slot_offsets7_handle(client)?;
            let left_slot0 = left_bindings.slot_or_first(0);
            let left_slot1 = left_bindings.slot_or_first(1);
            let left_slot2 = left_bindings.slot_or_first(2);
            let left_slot3 = left_bindings.slot_or_first(3);
            let left_slot4 = left_bindings.slot_or_first(4);
            let left_slot5 = left_bindings.slot_or_first(5);
            let left_slot6 = left_bindings.slot_or_first(6);
            let right_slot0 = right_bindings.slot_or_first(0);
            let right_slot1 = right_bindings.slot_or_first(1);
            let right_slot2 = right_bindings.slot_or_first(2);
            let right_slot3 = right_bindings.slot_or_first(3);
            let right_slot4 = right_bindings.slot_or_first(4);
            let right_slot5 = right_bindings.slot_or_first(5);
            let right_slot6 = right_bindings.slot_or_first(6);
            let metadata = [
                u32::try_from(left_len).map_err(|_| Error::LengthTooLarge { len: left_len })?,
                u32::try_from(right_len).map_err(|_| Error::LengthTooLarge { len: right_len })?,
                $(
                    u32::try_from($var.offset)
                        .map_err(|_| Error::LengthTooLarge { len: $var.offset })?,
                )+
            ];
            let metadata_handle = client.create_from_slice(u32::as_bytes(&metadata));
            let len = left_len + right_len;
            let block_size = 256_u32;
            let launch = crate::detail::launch::launch_1d(client, len, block_size)?;
            unsafe {
                crate::kernels::$kernel::launch_unchecked::<
                    <Left as crate::detail::read::KernelReadBoundMany<R>>::Leaf0,
                    <Left as crate::detail::read::KernelReadBoundMany<R>>::Leaf1,
                    <Left as crate::detail::read::KernelReadBoundMany<R>>::Leaf2,
                    <Left as crate::detail::read::KernelReadBoundMany<R>>::Leaf3,
                    <Left as crate::detail::read::KernelReadBoundMany<R>>::Leaf4,
                    <Left as crate::detail::read::KernelReadBoundMany<R>>::Leaf5,
                    <Left as crate::detail::read::KernelReadBoundMany<R>>::Leaf6,
                    <Right as crate::detail::read::KernelReadBoundMany<R>>::Leaf0,
                    <Right as crate::detail::read::KernelReadBoundMany<R>>::Leaf1,
                    <Right as crate::detail::read::KernelReadBoundMany<R>>::Leaf2,
                    <Right as crate::detail::read::KernelReadBoundMany<R>>::Leaf3,
                    <Right as crate::detail::read::KernelReadBoundMany<R>>::Leaf4,
                    <Right as crate::detail::read::KernelReadBoundMany<R>>::Leaf5,
                    <Right as crate::detail::read::KernelReadBoundMany<R>>::Leaf6,
                    <Left as crate::detail::read::KernelReadBoundMany<R>>::ExprAt,
                    <Right as crate::detail::read::KernelReadBoundMany<R>>::ExprAt,
                    KernelOp<R, Less>,
                    $( $ty, )+
                    R,
                >(
                    client,
                    launch.cube_count(),
                    CubeDim::new_1d(block_size),
                    BufferArg::from_raw_parts(right_only.flag.clone(), right_only.len),
                    BufferArg::from_raw_parts(right_only.position.clone(), right_only.len),
                    BufferArg::from_raw_parts(left_slot0.0.clone(), left_slot0.1),
                    BufferArg::from_raw_parts(left_slot1.0.clone(), left_slot1.1),
                    BufferArg::from_raw_parts(left_slot2.0.clone(), left_slot2.1),
                    BufferArg::from_raw_parts(left_slot3.0.clone(), left_slot3.1),
                    BufferArg::from_raw_parts(left_slot4.0.clone(), left_slot4.1),
                    BufferArg::from_raw_parts(left_slot5.0.clone(), left_slot5.1),
                    BufferArg::from_raw_parts(left_slot6.0.clone(), left_slot6.1),
                    BufferArg::from_raw_parts(left_offsets.clone(), 7),
                    BufferArg::from_raw_parts(right_slot0.0.clone(), right_slot0.1),
                    BufferArg::from_raw_parts(right_slot1.0.clone(), right_slot1.1),
                    BufferArg::from_raw_parts(right_slot2.0.clone(), right_slot2.1),
                    BufferArg::from_raw_parts(right_slot3.0.clone(), right_slot3.1),
                    BufferArg::from_raw_parts(right_slot4.0.clone(), right_slot4.1),
                    BufferArg::from_raw_parts(right_slot5.0.clone(), right_slot5.1),
                    BufferArg::from_raw_parts(right_slot6.0.clone(), right_slot6.1),
                    BufferArg::from_raw_parts(right_offsets.clone(), 7),
                    BufferArg::from_raw_parts(metadata_handle.clone(), metadata.len()),
                    $(
                        BufferArg::from_raw_parts($var.source.handle.clone(), $var.source.len()),
                    )+
                );
            }
        }
        mindex_from_usize(left_len + count)
    }};
}

macro_rules! set_union_logical7_auto_arity {
    ($policy:expr, $left:expr, $right:expr, $right_only:expr, $output:expr; A: $a:ident) => {
        set_union_logical7_into_output!(set_union_logical7_to_tuple1_kernel, $policy, $left, $right, $right_only, $output; A: $a => 0)
    };
    ($policy:expr, $left:expr, $right:expr, $right_only:expr, $output:expr; A: $a:ident, B: $b:ident) => {
        set_union_logical7_into_output!(set_union_logical7_to_tuple2_kernel, $policy, $left, $right, $right_only, $output; A: $a => 0, B: $b => 1)
    };
    ($policy:expr, $left:expr, $right:expr, $right_only:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident) => {
        set_union_logical7_into_output!(set_union_logical7_to_tuple3_kernel, $policy, $left, $right, $right_only, $output; A: $a => 0, B: $b => 1, C: $c => 2)
    };
    ($policy:expr, $left:expr, $right:expr, $right_only:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident) => {
        set_union_logical7_into_output!(set_union_logical7_to_tuple4_kernel, $policy, $left, $right, $right_only, $output; A: $a => 0, B: $b => 1, C: $c => 2, D: $d => 3)
    };
    ($policy:expr, $left:expr, $right:expr, $right_only:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident, E: $e:ident) => {
        set_union_logical7_into_output!(set_union_logical7_to_tuple5_kernel, $policy, $left, $right, $right_only, $output; A: $a => 0, B: $b => 1, C: $c => 2, D: $d => 3, E: $e => 4)
    };
    ($policy:expr, $left:expr, $right:expr, $right_only:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident, E: $e:ident, F: $f:ident) => {
        set_union_logical7_into_output!(set_union_logical7_to_tuple6_kernel, $policy, $left, $right, $right_only, $output; A: $a => 0, B: $b => 1, C: $c => 2, D: $d => 3, E: $e => 4, F: $f => 5)
    };
    ($policy:expr, $left:expr, $right:expr, $right_only:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident, E: $e:ident, F: $f:ident, G: $g:ident) => {
        set_union_logical7_into_output!(set_union_logical7_to_tuple7_kernel, $policy, $left, $right, $right_only, $output; A: $a => 0, B: $b => 1, C: $c => 2, D: $d => 3, E: $e => 4, F: $f => 5, G: $g => 6)
    };
}

macro_rules! partition_logical7_into_output {
    (
        $kernel:ident,
        $policy:expr,
        $input:expr,
        $split_rank:expr,
        $matching_count:expr,
        $output:expr;
        $( $ty:ident : $var:ident => $index:expr ),+
    ) => {{
        let input = $input;
        let split_rank = $split_rank;
        let len = crate::detail::read::KernelRead::len(&input);
        ensure_same_len(len, split_rank.len)?;
        $(
            let $var = <Output as crate::iter::MIterMut<R>>::column_mut_view_by_index_inner::<$ty>(
                &$output,
                $index,
            )?
            .ok_or_else(|| Error::Launch {
                message: "partition output must match input shape".to_string(),
            })?;
        )+
        if len != 0 {
            let client = $policy.client();
            let mut bindings = crate::detail::device::KernelColumnBindings::empty(client);
            <Read as crate::detail::read::KernelReadAtEnv<R, crate::detail::read::Env0>>::stage_at_env(
                &input,
                &mut bindings,
            )?;
            bindings.finish();
            let offsets = bindings.slot_offsets7_handle(client)?;
            let slot0 = bindings.slot_or_first(0);
            let slot1 = bindings.slot_or_first(1);
            let slot2 = bindings.slot_or_first(2);
            let slot3 = bindings.slot_or_first(3);
            let slot4 = bindings.slot_or_first(4);
            let slot5 = bindings.slot_or_first(5);
            let slot6 = bindings.slot_or_first(6);
            let metadata = [
                u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?,
                u32::try_from($matching_count)
                    .map_err(|_| Error::LengthTooLarge { len: $matching_count })?,
                $(
                    u32::try_from($var.offset)
                        .map_err(|_| Error::LengthTooLarge { len: $var.offset })?,
                )+
            ];
            let metadata_handle = client.create_from_slice(u32::as_bytes(&metadata));
            let block_size = 256_u32;
            let block_count = len.div_ceil(block_size as usize);
            let block_count_u32 =
                u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;
            unsafe {
                crate::kernels::$kernel::launch_unchecked::<
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::Leaf0,
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::Leaf1,
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::Leaf2,
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::Leaf3,
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::Leaf4,
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::Leaf5,
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::Leaf6,
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::ExprAt,
                    $( $ty, )+
                    R,
                >(
                    client,
                    CubeCount::Static(block_count_u32, 1, 1),
                    CubeDim::new_1d(block_size),
                    BufferArg::from_raw_parts(split_rank.flag.clone(), split_rank.len),
                    BufferArg::from_raw_parts(split_rank.position.clone(), split_rank.len),
                    BufferArg::from_raw_parts(slot0.0.clone(), slot0.1),
                    BufferArg::from_raw_parts(slot1.0.clone(), slot1.1),
                    BufferArg::from_raw_parts(slot2.0.clone(), slot2.1),
                    BufferArg::from_raw_parts(slot3.0.clone(), slot3.1),
                    BufferArg::from_raw_parts(slot4.0.clone(), slot4.1),
                    BufferArg::from_raw_parts(slot5.0.clone(), slot5.1),
                    BufferArg::from_raw_parts(slot6.0.clone(), slot6.1),
                    BufferArg::from_raw_parts(offsets.clone(), 7),
                    BufferArg::from_raw_parts(metadata_handle.clone(), metadata.len()),
                    $(
                        BufferArg::from_raw_parts($var.source.handle.clone(), $var.source.len()),
                    )+
                );
            }
        }
        Ok(())
    }};
}

macro_rules! partition_logical7_auto_arity {
    ($policy:expr, $input:expr, $split_rank:expr, $matching_count:expr, $output:expr; A: $a:ident) => {
        partition_logical7_into_output!(partition_logical7_to_tuple1_kernel, $policy, $input, $split_rank, $matching_count, $output; A: $a => 0)
    };
    ($policy:expr, $input:expr, $split_rank:expr, $matching_count:expr, $output:expr; A: $a:ident, B: $b:ident) => {
        partition_logical7_into_output!(partition_logical7_to_tuple2_kernel, $policy, $input, $split_rank, $matching_count, $output; A: $a => 0, B: $b => 1)
    };
    ($policy:expr, $input:expr, $split_rank:expr, $matching_count:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident) => {
        partition_logical7_into_output!(partition_logical7_to_tuple3_kernel, $policy, $input, $split_rank, $matching_count, $output; A: $a => 0, B: $b => 1, C: $c => 2)
    };
    ($policy:expr, $input:expr, $split_rank:expr, $matching_count:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident) => {
        partition_logical7_into_output!(partition_logical7_to_tuple4_kernel, $policy, $input, $split_rank, $matching_count, $output; A: $a => 0, B: $b => 1, C: $c => 2, D: $d => 3)
    };
    ($policy:expr, $input:expr, $split_rank:expr, $matching_count:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident, E: $e:ident) => {
        partition_logical7_into_output!(partition_logical7_to_tuple5_kernel, $policy, $input, $split_rank, $matching_count, $output; A: $a => 0, B: $b => 1, C: $c => 2, D: $d => 3, E: $e => 4)
    };
    ($policy:expr, $input:expr, $split_rank:expr, $matching_count:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident, E: $e:ident, F: $f:ident) => {
        partition_logical7_into_output!(partition_logical7_to_tuple6_kernel, $policy, $input, $split_rank, $matching_count, $output; A: $a => 0, B: $b => 1, C: $c => 2, D: $d => 3, E: $e => 4, F: $f => 5)
    };
    ($policy:expr, $input:expr, $split_rank:expr, $matching_count:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident, E: $e:ident, F: $f:ident, G: $g:ident) => {
        partition_logical7_into_output!(partition_logical7_to_tuple7_kernel, $policy, $input, $split_rank, $matching_count, $output; A: $a => 0, B: $b => 1, C: $c => 2, D: $d => 3, E: $e => 4, F: $f => 5, G: $g => 6)
    };
}

macro_rules! scatter_logical7_into_output {
    (
        $kernel:ident,
        $policy:expr,
        $values:expr,
        $indices:expr,
        $output:expr
        $(, $mask:expr)?;
        $( $ty:ident : $var:ident => $index:expr ),+
    ) => {{
        let values = $values;
        let indices = $indices;
        let len = crate::detail::read::KernelRead::len(&values);
        ensure_same_len(len, crate::detail::read::KernelRead::len(&indices))?;
        $(
            let mask = $mask;
            ensure_same_len(len, mask.len)?;
        )?
        $(
            let $var = <Output as crate::iter::MIterMut<R>>::column_mut_view_by_index_inner::<$ty>(
                &$output,
                $index,
            )?
            .ok_or_else(|| Error::Launch {
                message: "scatter output must match input shape".to_string(),
            })?;
        )+
        if len != 0 {
            let client = $policy.client();
            let mut value_bindings = crate::detail::device::KernelColumnBindings::empty(client);
            <Read as crate::detail::read::KernelReadAtEnv<R, crate::detail::read::Env0>>::stage_at_env(
                &values,
                &mut value_bindings,
            )?;
            value_bindings.finish();
            let mut index_bindings = crate::detail::device::KernelColumnBindings::empty(client);
            <IndexSource as crate::detail::read::KernelReadAtEnv<R, crate::detail::read::Env0>>::stage_at_env(
                &indices,
                &mut index_bindings,
            )?;
            index_bindings.finish();
            let value_offsets = value_bindings.slot_offsets7_handle(client)?;
            let index_offsets = index_bindings.slot_offsets7_handle(client)?;
            let value_slot0 = value_bindings.slot_or_first(0);
            let value_slot1 = value_bindings.slot_or_first(1);
            let value_slot2 = value_bindings.slot_or_first(2);
            let value_slot3 = value_bindings.slot_or_first(3);
            let value_slot4 = value_bindings.slot_or_first(4);
            let value_slot5 = value_bindings.slot_or_first(5);
            let value_slot6 = value_bindings.slot_or_first(6);
            let index_slot0 = index_bindings.slot_or_first(0);
            let index_slot1 = index_bindings.slot_or_first(1);
            let index_slot2 = index_bindings.slot_or_first(2);
            let index_slot3 = index_bindings.slot_or_first(3);
            let index_slot4 = index_bindings.slot_or_first(4);
            let index_slot5 = index_bindings.slot_or_first(5);
            let index_slot6 = index_bindings.slot_or_first(6);
            let metadata = [
                u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?,
                $(
                    u32::try_from($var.offset)
                        .map_err(|_| Error::LengthTooLarge { len: $var.offset })?,
                )+
            ];
            let metadata_handle = client.create_from_slice(u32::as_bytes(&metadata));
            let block_size = 256_u32;
            let block_count = len.div_ceil(block_size as usize);
            let block_count_u32 =
                u32::try_from(block_count).map_err(|_| Error::LengthTooLarge { len: block_count })?;
            unsafe {
                crate::kernels::$kernel::launch_unchecked::<
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::Leaf0,
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::Leaf1,
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::Leaf2,
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::Leaf3,
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::Leaf4,
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::Leaf5,
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::Leaf6,
                    <IndexSource as crate::detail::read::KernelReadBoundMany<R>>::Leaf0,
                    <IndexSource as crate::detail::read::KernelReadBoundMany<R>>::Leaf1,
                    <IndexSource as crate::detail::read::KernelReadBoundMany<R>>::Leaf2,
                    <IndexSource as crate::detail::read::KernelReadBoundMany<R>>::Leaf3,
                    <IndexSource as crate::detail::read::KernelReadBoundMany<R>>::Leaf4,
                    <IndexSource as crate::detail::read::KernelReadBoundMany<R>>::Leaf5,
                    <IndexSource as crate::detail::read::KernelReadBoundMany<R>>::Leaf6,
                    <Read as crate::detail::read::KernelReadBoundMany<R>>::ExprAt,
                    <IndexSource as crate::detail::read::KernelReadBoundMany<R>>::ExprAt,
                    $( $ty, )+
                    R,
                >(
                    client,
                    CubeCount::Static(block_count_u32, 1, 1),
                    CubeDim::new_1d(block_size),
                    BufferArg::from_raw_parts(value_slot0.0.clone(), value_slot0.1),
                    BufferArg::from_raw_parts(value_slot1.0.clone(), value_slot1.1),
                    BufferArg::from_raw_parts(value_slot2.0.clone(), value_slot2.1),
                    BufferArg::from_raw_parts(value_slot3.0.clone(), value_slot3.1),
                    BufferArg::from_raw_parts(value_slot4.0.clone(), value_slot4.1),
                    BufferArg::from_raw_parts(value_slot5.0.clone(), value_slot5.1),
                    BufferArg::from_raw_parts(value_slot6.0.clone(), value_slot6.1),
                    BufferArg::from_raw_parts(value_offsets.clone(), 7),
                    BufferArg::from_raw_parts(index_slot0.0.clone(), index_slot0.1),
                    BufferArg::from_raw_parts(index_slot1.0.clone(), index_slot1.1),
                    BufferArg::from_raw_parts(index_slot2.0.clone(), index_slot2.1),
                    BufferArg::from_raw_parts(index_slot3.0.clone(), index_slot3.1),
                    BufferArg::from_raw_parts(index_slot4.0.clone(), index_slot4.1),
                    BufferArg::from_raw_parts(index_slot5.0.clone(), index_slot5.1),
                    BufferArg::from_raw_parts(index_slot6.0.clone(), index_slot6.1),
                    BufferArg::from_raw_parts(index_offsets.clone(), 7),
                    BufferArg::from_raw_parts(metadata_handle.clone(), metadata.len()),
                    $(
                        BufferArg::from_raw_parts(mask.flag.clone(), {
                            let _ = stringify!($mask);
                            mask.len
                        }),
                    )?
                    $(
                        BufferArg::from_raw_parts($var.source.handle.clone(), $var.source.len()),
                    )+
                );
            }
        }
        Ok(())
    }};
}

macro_rules! scatter_where_logical7_arity {
    ($kernel:ident, $policy:expr, $values:expr, $indices:expr, $mask:expr, $output:expr; A: $a:ident) => {
        scatter_logical7_into_output!($kernel, $policy, $values, $indices, $output, $mask; A: $a => 0)
    };
    ($kernel:ident, $policy:expr, $values:expr, $indices:expr, $mask:expr, $output:expr; A: $a:ident, B: $b:ident) => {
        scatter_logical7_into_output!($kernel, $policy, $values, $indices, $output, $mask; A: $a => 0, B: $b => 1)
    };
    ($kernel:ident, $policy:expr, $values:expr, $indices:expr, $mask:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident) => {
        scatter_logical7_into_output!($kernel, $policy, $values, $indices, $output, $mask; A: $a => 0, B: $b => 1, C: $c => 2)
    };
    ($kernel:ident, $policy:expr, $values:expr, $indices:expr, $mask:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident) => {
        scatter_logical7_into_output!($kernel, $policy, $values, $indices, $output, $mask; A: $a => 0, B: $b => 1, C: $c => 2, D: $d => 3)
    };
    ($kernel:ident, $policy:expr, $values:expr, $indices:expr, $mask:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident, E: $e:ident) => {
        scatter_logical7_into_output!($kernel, $policy, $values, $indices, $output, $mask; A: $a => 0, B: $b => 1, C: $c => 2, D: $d => 3, E: $e => 4)
    };
    ($kernel:ident, $policy:expr, $values:expr, $indices:expr, $mask:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident, E: $e:ident, F: $f:ident) => {
        scatter_logical7_into_output!($kernel, $policy, $values, $indices, $output, $mask; A: $a => 0, B: $b => 1, C: $c => 2, D: $d => 3, E: $e => 4, F: $f => 5)
    };
    ($kernel:ident, $policy:expr, $values:expr, $indices:expr, $mask:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident, E: $e:ident, F: $f:ident, G: $g:ident) => {
        scatter_logical7_into_output!($kernel, $policy, $values, $indices, $output, $mask; A: $a => 0, B: $b => 1, C: $c => 2, D: $d => 3, E: $e => 4, F: $f => 5, G: $g => 6)
    };
}

macro_rules! scatter_where_logical7_auto_arity {
    ($policy:expr, $values:expr, $indices:expr, $mask:expr, $output:expr; A: $a:ident) => {
        scatter_where_logical7_arity!(scatter_where_logical7_to_tuple1_kernel, $policy, $values, $indices, $mask, $output; A: $a)
    };
    ($policy:expr, $values:expr, $indices:expr, $mask:expr, $output:expr; A: $a:ident, B: $b:ident) => {
        scatter_where_logical7_arity!(scatter_where_logical7_to_tuple2_kernel, $policy, $values, $indices, $mask, $output; A: $a, B: $b)
    };
    ($policy:expr, $values:expr, $indices:expr, $mask:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident) => {
        scatter_where_logical7_arity!(scatter_where_logical7_to_tuple3_kernel, $policy, $values, $indices, $mask, $output; A: $a, B: $b, C: $c)
    };
    ($policy:expr, $values:expr, $indices:expr, $mask:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident) => {
        scatter_where_logical7_arity!(scatter_where_logical7_to_tuple4_kernel, $policy, $values, $indices, $mask, $output; A: $a, B: $b, C: $c, D: $d)
    };
    ($policy:expr, $values:expr, $indices:expr, $mask:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident, E: $e:ident) => {
        scatter_where_logical7_arity!(scatter_where_logical7_to_tuple5_kernel, $policy, $values, $indices, $mask, $output; A: $a, B: $b, C: $c, D: $d, E: $e)
    };
    ($policy:expr, $values:expr, $indices:expr, $mask:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident, E: $e:ident, F: $f:ident) => {
        scatter_where_logical7_arity!(scatter_where_logical7_to_tuple6_kernel, $policy, $values, $indices, $mask, $output; A: $a, B: $b, C: $c, D: $d, E: $e, F: $f)
    };
    ($policy:expr, $values:expr, $indices:expr, $mask:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident, E: $e:ident, F: $f:ident, G: $g:ident) => {
        scatter_where_logical7_arity!(scatter_where_logical7_to_tuple7_kernel, $policy, $values, $indices, $mask, $output; A: $a, B: $b, C: $c, D: $d, E: $e, F: $f, G: $g)
    };
}

macro_rules! scatter_logical7_arity {
    ($kernel:ident, $policy:expr, $values:expr, $indices:expr, $output:expr; A: $a:ident) => {
        scatter_logical7_into_output!($kernel, $policy, $values, $indices, $output; A: $a => 0)
    };
    ($kernel:ident, $policy:expr, $values:expr, $indices:expr, $output:expr; A: $a:ident, B: $b:ident) => {
        scatter_logical7_into_output!($kernel, $policy, $values, $indices, $output; A: $a => 0, B: $b => 1)
    };
    ($kernel:ident, $policy:expr, $values:expr, $indices:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident) => {
        scatter_logical7_into_output!($kernel, $policy, $values, $indices, $output; A: $a => 0, B: $b => 1, C: $c => 2)
    };
    ($kernel:ident, $policy:expr, $values:expr, $indices:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident) => {
        scatter_logical7_into_output!($kernel, $policy, $values, $indices, $output; A: $a => 0, B: $b => 1, C: $c => 2, D: $d => 3)
    };
    ($kernel:ident, $policy:expr, $values:expr, $indices:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident, E: $e:ident) => {
        scatter_logical7_into_output!($kernel, $policy, $values, $indices, $output; A: $a => 0, B: $b => 1, C: $c => 2, D: $d => 3, E: $e => 4)
    };
    ($kernel:ident, $policy:expr, $values:expr, $indices:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident, E: $e:ident, F: $f:ident) => {
        scatter_logical7_into_output!($kernel, $policy, $values, $indices, $output; A: $a => 0, B: $b => 1, C: $c => 2, D: $d => 3, E: $e => 4, F: $f => 5)
    };
    ($kernel:ident, $policy:expr, $values:expr, $indices:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident, E: $e:ident, F: $f:ident, G: $g:ident) => {
        scatter_logical7_into_output!($kernel, $policy, $values, $indices, $output; A: $a => 0, B: $b => 1, C: $c => 2, D: $d => 3, E: $e => 4, F: $f => 5, G: $g => 6)
    };
}

macro_rules! scatter_logical7_auto_arity {
    ($policy:expr, $values:expr, $indices:expr, $output:expr; A: $a:ident) => {
        scatter_logical7_arity!(scatter_logical7_to_tuple1_kernel, $policy, $values, $indices, $output; A: $a)
    };
    ($policy:expr, $values:expr, $indices:expr, $output:expr; A: $a:ident, B: $b:ident) => {
        scatter_logical7_arity!(scatter_logical7_to_tuple2_kernel, $policy, $values, $indices, $output; A: $a, B: $b)
    };
    ($policy:expr, $values:expr, $indices:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident) => {
        scatter_logical7_arity!(scatter_logical7_to_tuple3_kernel, $policy, $values, $indices, $output; A: $a, B: $b, C: $c)
    };
    ($policy:expr, $values:expr, $indices:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident) => {
        scatter_logical7_arity!(scatter_logical7_to_tuple4_kernel, $policy, $values, $indices, $output; A: $a, B: $b, C: $c, D: $d)
    };
    ($policy:expr, $values:expr, $indices:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident, E: $e:ident) => {
        scatter_logical7_arity!(scatter_logical7_to_tuple5_kernel, $policy, $values, $indices, $output; A: $a, B: $b, C: $c, D: $d, E: $e)
    };
    ($policy:expr, $values:expr, $indices:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident, E: $e:ident, F: $f:ident) => {
        scatter_logical7_arity!(scatter_logical7_to_tuple6_kernel, $policy, $values, $indices, $output; A: $a, B: $b, C: $c, D: $d, E: $e, F: $f)
    };
    ($policy:expr, $values:expr, $indices:expr, $output:expr; A: $a:ident, B: $b:ident, C: $c:ident, D: $d:ident, E: $e:ident, F: $f:ident, G: $g:ident) => {
        scatter_logical7_arity!(scatter_logical7_to_tuple7_kernel, $policy, $values, $indices, $output; A: $a, B: $b, C: $c, D: $d, E: $e, F: $f, G: $g)
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
    ($policy:expr; $pred:ty; $a:expr, $b:expr, $c:expr, $d:expr) => {{
        let dummy_e = crate::detail::primitives::range::indices_mindex($policy, $a.len)?;
        let dummy_f = crate::detail::primitives::range::indices_mindex($policy, $a.len)?;
        let dummy_g = crate::detail::primitives::range::indices_mindex($policy, $a.len)?;
        let dummy_e = crate::detail::device::DeviceColumnView::from_column(&dummy_e);
        let dummy_f = crate::detail::device::DeviceColumnView::from_column(&dummy_f);
        let dummy_g = crate::detail::device::DeviceColumnView::from_column(&dummy_g);
        wide_predicate_rank_from_tuple!(
            @launch $policy,
            crate::detail::api::Tuple4AsTuple7PredicateOp<KernelOp<R, $pred>>,
            ($a, $b, $c, $d, &dummy_e, &dummy_f, &dummy_g),
            (A, B, C, D, u32, u32, u32)
        )
    }};
    ($policy:expr; $pred:ty; $a:expr, $b:expr, $c:expr, $d:expr, $e:expr) => {{
        let dummy_f = crate::detail::primitives::range::indices_mindex($policy, $a.len)?;
        let dummy_g = crate::detail::primitives::range::indices_mindex($policy, $a.len)?;
        let dummy_f = crate::detail::device::DeviceColumnView::from_column(&dummy_f);
        let dummy_g = crate::detail::device::DeviceColumnView::from_column(&dummy_g);
        wide_predicate_rank_from_tuple!(
            @launch $policy,
            crate::detail::api::Tuple5AsTuple7PredicateOp<KernelOp<R, $pred>>,
            ($a, $b, $c, $d, $e, &dummy_f, &dummy_g),
            (A, B, C, D, E, u32, u32)
        )
    }};
    ($policy:expr; $pred:ty; $a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr) => {{
        let dummy_g = crate::detail::primitives::range::indices_mindex($policy, $a.len)?;
        let dummy_g = crate::detail::device::DeviceColumnView::from_column(&dummy_g);
        wide_predicate_rank_from_tuple!(
            @launch $policy,
            crate::detail::api::Tuple6AsTuple7PredicateOp<KernelOp<R, $pred>>,
            ($a, $b, $c, $d, $e, $f, &dummy_g),
            (A, B, C, D, E, F, u32)
        )
    }};
    ($policy:expr; $pred:ty; $a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr) => {
        wide_predicate_rank_from_tuple!(
            @launch $policy,
            KernelOp<R, $pred>,
            ($a, $b, $c, $d, $e, $f, $g),
            (A, B, C, D, E, F, G)
        )
    };
    (@launch $policy:expr, $pred:ty, ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr), ($ty0:ty, $ty1:ty, $ty2:ty, $ty3:ty, $ty4:ty, $ty5:ty, $ty6:ty)) => {{
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
    ($output:ty, $policy:expr, $op:expr; $a:ident) => {
        <$output as sealed::MItemDispatch<R>>::transform_unary($policy, $a, $op)
    };
    ($output:ty, $policy:expr, $op:expr; $a:ident, $b:ident) => {
        <$output as sealed::MItemDispatch<R>>::transform_binary($policy, $a, $b, $op)
    };
    ($output:ty, $policy:expr, $op:expr; $a:ident, $b:ident, $c:ident) => {
        <$output as sealed::MItemDispatch<R>>::transform_ternary($policy, $a, $b, $c, $op)
    };
    ($output:ty, $policy:expr, $op:expr; $a:ident, $b:ident, $c:ident, $d:ident) => {
        <$output as sealed::MItemDispatch<R>>::transform_quaternary($policy, $a, $b, $c, $d, $op)
    };
    ($output:ty, $policy:expr, $op:expr; $a:ident, $b:ident, $c:ident, $d:ident, $e:ident) => {
        <$output as sealed::MItemDispatch<R>>::transform_quinary($policy, $a, $b, $c, $d, $e, $op)
    };
    ($output:ty, $policy:expr, $op:expr; $a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident) => {
        <$output as sealed::MItemDispatch<R>>::transform_senary(
            $policy, $a, $b, $c, $d, $e, $f, $op,
        )
    };
    ($output:ty, $policy:expr, $op:expr; $a:ident, $b:ident, $c:ident, $d:ident, $e:ident, $f:ident, $g:ident) => {
        <$output as sealed::MItemDispatch<R>>::transform_septenary(
            $policy, $a, $b, $c, $d, $e, $f, $g, $op,
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

macro_rules! impl_scalar_mitem_dispatch {
    ($( $ty:ty ),+ $(,)?) => {
        $(
            impl<R> sealed::MItemDispatch<R> for $ty
            where
                R: Runtime,
                $ty: MStorageElement + 'static,
            {
            }
        )+
    };
}

impl_scalar_mitem_dispatch!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

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

            fn gather_from_view<IndexSource, Output>(
                policy: &crate::detail::CubePolicy<R>,
                values: Self::View,
                indices: IndexSource,
                output: Output,
            ) -> Result<(), Error>
            where
                IndexSource: crate::detail::read::KernelReadBoundMany<R, Item = MIndex>,
                Output: MIterMut<R, Item = Self>,
            {
                let inner = indexed_gather_inner_from_tuple!(policy, values, indices; $( $var ),+)?;
                output.write_from_inner(policy, inner)
            }

            fn gather_where_from_view<IndexSource, Output>(
                policy: &crate::detail::CubePolicy<R>,
                values: Self::View,
                indices: IndexSource,
                stencil: crate::detail::api::PrecomputedSelection<R>,
                output: Output,
            ) -> Result<(), Error>
            where
                IndexSource: crate::detail::read::KernelReadBoundMany<R, Item = MIndex>,
                Output: MIterMut<R, Item = Self>,
            {
                let mask = stencil.mask();
                indexed_apply_arity!(
                    crate::detail::apply::MaskedIndexedExprApply::gather_where_expr_into,
                    policy, values, indices, output, &mask;
                    $( $ty: $var ),+
                )
            }

            fn scatter_from_view<IndexSource, Output>(
                policy: &crate::detail::CubePolicy<R>,
                values: Self::View,
                indices: IndexSource,
                output: Output,
            ) -> Result<(), Error>
            where
                IndexSource: crate::detail::read::KernelReadBoundMany<R, Item = MIndex>,
                Output: MIterMut<R, Item = Self>,
            {
                indexed_apply_arity!(
                    crate::detail::apply::IndexedExprApply::scatter_expr_into,
                    policy, values, indices, output;
                    $( $ty: $var ),+
                )
            }

            fn scatter_where_from_view<IndexSource, Output>(
                policy: &crate::detail::CubePolicy<R>,
                values: Self::View,
                indices: IndexSource,
                stencil: crate::detail::api::PrecomputedSelection<R>,
                output: Output,
            ) -> Result<(), Error>
            where
                IndexSource: crate::detail::read::KernelReadBoundMany<R, Item = MIndex>,
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
                    op;
                    $( $var ),+
                )?;
                output.write_from_inner(policy, inner)
            }

            fn transform_where_from_view<Output, Op>(
                policy: &crate::detail::CubePolicy<R>,
                input: Self::View,
                op: Op,
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
                    op;
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
                output: Output,
            ) -> Result<MIndex, Error>
            where
                Pred: op::PredicateOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                let input = partition_input_from_tuple!(input; $( $var ),+);
                let (matching, failing) =
                    crate::detail::partition(policy, input, KernelOp::<R, Pred>::new())?;
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
                let storage = crate::detail::apply::TransformPayloadApply::unary::<Self, R, Input, KernelScalarInputOp<R, Op>>(policy, input)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            fn transform_unary<Input, Op>(
                policy: &crate::detail::CubePolicy<R>,
                input: crate::detail::device::DeviceColumnView<
                    R,
                    Input,
                >,
                op: Op,
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
                let storage = crate::detail::apply::TransformPayloadApply::unary::<Self, R, Input, KernelOp<R, Op>>(policy, input)?;
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
                let storage = crate::detail::apply::TransformPayloadApply::zip2::<Self, R, Left, Right, KernelOp<R, Op>>(policy, left, right)?;
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
                let storage = crate::detail::apply::TransformPayloadApply::zip3::<Self, R, First, Second, Third, KernelOp<R, Op>>(policy, first, second, third)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            fn transform_logical3<Input, LeafA, LeafB, LeafC, Expr, Op>(
                policy: &crate::detail::CubePolicy<R>,
                bindings: crate::detail::device::KernelColumnBindings,
                len: usize,
                op: Op,
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
                >>::run_logical3(policy, bindings, len)?;
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
                >>::run_logical7(policy, bindings, len)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            fn transform_quaternary<First, Second, Third, Fourth, Op>(
                policy: &crate::detail::CubePolicy<R>,
                first: crate::detail::device::DeviceColumnView<R, First>,
                second: crate::detail::device::DeviceColumnView<R, Second>,
                third: crate::detail::device::DeviceColumnView<R, Third>,
                fourth: crate::detail::device::DeviceColumnView<R, Fourth>,
                op: Op,
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
                let storage = crate::detail::apply::TransformPayloadApply::zip4::<Self, R, First, Second, Third, Fourth, KernelOp<R, Op>>(policy, first, second, third, fourth)?;
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
                let storage = crate::detail::apply::TransformPayloadApply::zip5::<Self, R, First, Second, Third, Fourth, Fifth, KernelOp<R, Op>>(policy, first, second, third, fourth, fifth)?;
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
                let storage = crate::detail::apply::TransformPayloadApply::zip6::<Self, R, First, Second, Third, Fourth, Fifth, Sixth, KernelOp<R, Op>>(policy, first, second, third, fourth, fifth, sixth)?;
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
                let storage = crate::detail::apply::TransformPayloadApply::zip7::<Self, R, First, Second, Third, Fourth, Fifth, Sixth, Seventh, KernelOp<R, Op>>(policy, first, second, third, fourth, fifth, sixth, seventh)?;
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

        impl<R, $( $ty ),+> crate::detail::write::MItemWriteDispatch<R> for ($( $ty, )+)
        where
            R: Runtime,
            $( $ty: MStorageElement, )+
        {

            fn reduce_by_key_values_from_read<KeyEq, Op, Read, Output>(
                policy: &crate::detail::CubePolicy<R>,
                values: Read,
                selection: &crate::detail::control::SelectedRankControl,
                output_count: usize,
                init: Self,
                _op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                Read: crate::detail::read::KernelReadBoundMany<R, Item = Self>,
                Op: op::ReductionOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                let _ = std::marker::PhantomData::<KeyEq>;
                reduce_by_key_logical7_auto_arity!(
                    policy, values, selection, output_count, init, output;
                    $( $ty: $var ),+
                )
            }

            fn copy_selected_from_read<Read, Output>(
                policy: &crate::detail::CubePolicy<R>,
                values: Read,
                stencil: crate::detail::api::PrecomputedSelection<R>,
                output: Output,
            ) -> Result<MIndex, Error>
            where
                Read: crate::detail::read::KernelReadBoundMany<R, Item = Self>,
                Output: MIterMut<R, Item = Self>,
            {
                crate::detail::read::copy_selected_logical7_read(
                    values,
                    policy,
                    stencil,
                    output,
                )
            }

            fn gather_from_read<Read, IndexSource, Output>(
                policy: &crate::detail::CubePolicy<R>,
                values: Read,
                indices: IndexSource,
                output: Output,
            ) -> Result<(), Error>
            where
                Read: crate::detail::read::KernelReadBoundMany<R, Item = Self>,
                IndexSource: crate::detail::read::KernelReadBoundMany<R, Item = MIndex>,
                Output: MIterMut<R, Item = Self>,
            {
                crate::detail::read::gather_logical7_read(values, policy, indices, output)
            }

            fn gather_where_from_read<Read, IndexSource, Output>(
                policy: &crate::detail::CubePolicy<R>,
                values: Read,
                indices: IndexSource,
                stencil: crate::detail::api::PrecomputedSelection<R>,
                output: Output,
            ) -> Result<(), Error>
            where
                Read: crate::detail::read::KernelReadBoundMany<R, Item = Self>,
                IndexSource: crate::detail::read::KernelReadBoundMany<R, Item = MIndex>,
                Output: MIterMut<R, Item = Self>,
            {
                crate::detail::read::gather_where_logical7_read(
                    values,
                    policy,
                    indices,
                    stencil,
                    output,
                )
            }

            fn scatter_from_read<Read, IndexSource, Output>(
                policy: &crate::detail::CubePolicy<R>,
                values: Read,
                indices: IndexSource,
                output: Output,
            ) -> Result<(), Error>
            where
                Read: crate::detail::read::KernelReadBoundMany<R, Item = Self>,
                IndexSource: crate::detail::read::KernelReadBoundMany<R, Item = MIndex>,
                Output: MIterMut<R, Item = Self>,
            {
                scatter_logical7_auto_arity!(
                    policy, values, indices, output;
                    $( $ty: $var ),+
                )
            }

            fn scatter_where_from_read<Read, IndexSource, Output>(
                policy: &crate::detail::CubePolicy<R>,
                values: Read,
                indices: IndexSource,
                stencil: crate::detail::api::PrecomputedSelection<R>,
                output: Output,
            ) -> Result<(), Error>
            where
                Read: crate::detail::read::KernelReadBoundMany<R, Item = Self>,
                IndexSource: crate::detail::read::KernelReadBoundMany<R, Item = MIndex>,
                Output: MIterMut<R, Item = Self>,
            {
                let mask = stencil.mask();
                scatter_where_logical7_auto_arity!(
                    policy, values, indices, &mask, output;
                    $( $ty: $var ),+
                )
            }

            fn unique_from_read<Read, Pred, Output>(
                policy: &crate::detail::CubePolicy<R>,
                input: Read,
                _pred: Pred,
                output: Output,
            ) -> Result<MIndex, Error>
            where
                Read: crate::detail::read::KernelReadBoundMany<R, Item = Self>,
                Pred: op::BinaryPredicateOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                let len = crate::detail::read::KernelRead::len(&input);
                let Some(flags) =
                    crate::detail::read::unique_logical7_flags_read::<R, _, Pred>(
                        &input, policy,
                    )?
                else {
                    return Ok(0);
                };
                let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
                let selected_rank =
                    crate::detail::primitives::select::selected_rank_from_flags(
                        policy, len, len_u32, flags,
                    )?;
                let selection =
                    crate::detail::api::PrecomputedSelection::from_selected_rank(selected_rank);
                crate::detail::read::copy_selected_logical7_read(input, policy, selection, output)
            }

            fn partition_from_read<Read, Pred, Output>(
                policy: &crate::detail::CubePolicy<R>,
                input: Read,
                _pred: Pred,
                output: Output,
            ) -> Result<MIndex, Error>
            where
                Read: crate::detail::read::KernelReadBoundMany<R, Item = Self>,
                Pred: op::PredicateOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                let len = crate::detail::read::KernelRead::len(&input);
                let Some(flags) =
                    crate::detail::read::logical7_predicate_flags_read::<R, _, Pred>(
                        &input, policy, false,
                    )?
                else {
                    return Ok(0);
                };
                let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
                let selected_rank =
                    crate::detail::primitives::select::selected_rank_from_flags(
                        policy, len, len_u32, flags,
                    )?;
                let (split_rank, matching_count, _failing_count) =
                    crate::detail::primitives::select::split_rank_from_selected(
                        policy,
                        selected_rank,
                    )?;
                partition_logical7_auto_arity!(
                    policy, input, &split_rank, matching_count, output;
                    $( $ty: $var ),+
                )?;
                mindex_from_usize(matching_count)
            }

            fn sort_from_read<Read, Less, Output>(
                policy: &crate::detail::CubePolicy<R>,
                input: Read,
                _less: Less,
                output: Output,
            ) -> Result<(), Error>
            where
                Read: crate::detail::read::KernelReadBoundMany<R, Item = Self>,
                Less: op::BinaryPredicateOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                let input = crate::detail::read::materialize_logical7_read(input, policy)?;
                let input = wide_view_from_inner_tuple!(input; $( $var ),+);
                let input = zip_view_from_tuple!(input; $( $var ),+);
                let inner = crate::detail::sort(policy, input, tuple_set_less!(Less; $( $var ),+))?;
                output.write_from_inner(policy, inner)
            }

            fn sort_by_key_keys_from_read<Read, Less, Output>(
                policy: &crate::detail::CubePolicy<R>,
                keys: Read,
                _less: Less,
                output: Output,
            ) -> Result<crate::detail::DeviceVec<R, MIndex>, Error>
            where
                Read: crate::detail::read::KernelReadBoundMany<R, Item = Self>,
                Less: op::BinaryPredicateOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                let keys = crate::detail::read::materialize_logical7_read(keys, policy)?;
                let keys = wide_view_from_inner_tuple!(keys; $( $var ),+);
                let (inner, indices) =
                    <Self as MAlloc<R>>::sort_by_key_control_from_view(policy, keys, _less)?;
                output.write_from_inner(policy, inner)?;
                Ok(indices)
            }

            fn merge_by_key_keys_from_read<Left, Right, Less, Output>(
                policy: &crate::detail::CubePolicy<R>,
                left: Left,
                right: Right,
                _less: Less,
                output: Output,
            ) -> Result<crate::detail::control::MergeByKeyControl, Error>
            where
                Left: crate::detail::read::KernelReadBoundMany<R, Item = Self>,
                Right: crate::detail::read::KernelReadBoundMany<R, Item = Self>,
                Less: op::BinaryPredicateOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                let left = crate::detail::read::materialize_logical7_read(left, policy)?;
                let right = crate::detail::read::materialize_logical7_read(right, policy)?;
                let left = wide_view_from_inner_tuple!(left; $( $var ),+);
                let right = wide_view_from_inner_tuple!(right; $( $var ),+);
                let (inner, control) =
                    <Self as MAlloc<R>>::merge_by_key_control_from_views(policy, left, right, _less)?;
                output.write_from_inner(policy, inner)?;
                Ok(control)
            }

            fn merge_by_key_values_from_read<Left, Right, Output>(
                policy: &crate::detail::CubePolicy<R>,
                left: Left,
                right: Right,
                control: &crate::detail::control::MergeByKeyControl,
                output: Output,
            ) -> Result<(), Error>
            where
                Left: crate::detail::read::KernelReadBoundMany<R, Item = Self>,
                Right: crate::detail::read::KernelReadBoundMany<R, Item = Self>,
                Output: MIterMut<R, Item = Self>,
            {
                let left = crate::detail::read::materialize_logical7_read(left, policy)?;
                let right = crate::detail::read::materialize_logical7_read(right, policy)?;
                let left = wide_view_from_inner_tuple!(left; $( $var ),+);
                let right = wide_view_from_inner_tuple!(right; $( $var ),+);
                <Self as MAlloc<R>>::merge_by_key_values_from_views(policy, left, right, control, output)
            }

            fn adjacent_difference_from_read<Read, Op, Output>(
                policy: &crate::detail::CubePolicy<R>,
                input: Read,
                _op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                Read: crate::detail::read::KernelReadBoundMany<R, Item = Self>,
                Op: op::ReductionOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                adjacent_logical7_auto_arity!(
                    policy, input, output;
                    $( $ty: $var ),+
                )
            }

            fn inclusive_scan_from_read<Read, Op, Output>(
                policy: &crate::detail::CubePolicy<R>,
                input: Read,
                _op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                Read: crate::detail::read::KernelReadBoundMany<R, Item = Self>,
                Op: op::ReductionOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                let input = crate::detail::read::materialize_logical7_read(input, policy)?;
                let input = scan_input_from_tuple!(input; $( $var ),+);
                let inner =
                    crate::detail::inclusive_scan(policy, input, KernelOp::<R, Op>::new())?;
                output.write_from_inner(policy, inner)
            }

            fn exclusive_scan_from_read<Read, Op, Output>(
                policy: &crate::detail::CubePolicy<R>,
                input: Read,
                init: Self,
                _op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                Read: crate::detail::read::KernelReadBoundMany<R, Item = Self>,
                Op: op::ReductionOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                let input = crate::detail::read::materialize_logical7_read(input, policy)?;
                let input = scan_input_from_tuple!(input; $( $var ),+);
                let inner =
                    crate::detail::exclusive_scan(policy, input, init, KernelOp::<R, Op>::new())?;
                output.write_from_inner(policy, inner)
            }

            fn merge_from_read<Left, Right, Less, Output>(
                policy: &crate::detail::CubePolicy<R>,
                left: Left,
                right: Right,
                _less: Less,
                output: Output,
            ) -> Result<(), Error>
            where
                Left: crate::detail::read::KernelReadBoundMany<R, Item = Self>,
                Right: crate::detail::read::KernelReadBoundMany<R, Item = Self>,
                Less: op::BinaryPredicateOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                let left_inner = crate::detail::read::materialize_logical7_read(left, policy)?;
                let right_inner = crate::detail::read::materialize_logical7_read(right, policy)?;
                let left = zip_view_from_tuple!(left_inner; $( $var ),+);
                let right = zip_view_from_tuple!(right_inner; $( $var ),+);
                let inner = crate::detail::merge(policy, left, right, tuple_set_less!(Less; $( $var ),+))?;
                output.write_from_inner(policy, inner)
            }

            fn set_union_from_read<Left, Right, Less, Output>(
                policy: &crate::detail::CubePolicy<R>,
                left: Left,
                right: Right,
                right_only: &crate::detail::control::SelectedRankControl,
                _less: Less,
                output: Output,
            ) -> Result<MIndex, Error>
            where
                Left: crate::detail::read::KernelReadBoundMany<R, Item = Self>,
                Right: crate::detail::read::KernelReadBoundMany<R, Item = Self>,
                Less: op::BinaryPredicateOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                let _ = right_only;
                let left_inner = crate::detail::read::materialize_logical7_read(left, policy)?;
                let right_inner = crate::detail::read::materialize_logical7_read(right, policy)?;
                let left = zip_view_from_tuple!(left_inner; $( $var ),+);
                let right = zip_view_from_tuple!(right_inner; $( $var ),+);
                let inner = crate::detail::set_union(policy, left, right, tuple_set_less!(Less; $( $var ),+))?;
                let len = mindex_from_usize(inner.0.len())?;
                output.write_prefix_from_inner(policy, inner)?;
                Ok(len)
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

            fn gather_from_view<IndexSource, Output>(
                policy: &crate::detail::CubePolicy<R>,
                values: Self::View,
                indices: IndexSource,
                output: Output,
            ) -> Result<(), Error>
            where
                IndexSource: crate::detail::read::KernelReadBoundMany<R, Item = MIndex>,
                Output: MIterMut<R, Item = Self>,
            {
                let inner = indexed_gather_inner_from_tuple!(policy, values, indices; $( $var ),+)?;
                output.write_from_inner(policy, inner)
            }

            fn gather_where_from_view<IndexSource, Output>(
                policy: &crate::detail::CubePolicy<R>,
                values: Self::View,
                indices: IndexSource,
                stencil: crate::detail::api::PrecomputedSelection<R>,
                output: Output,
            ) -> Result<(), Error>
            where
                IndexSource: crate::detail::read::KernelReadBoundMany<R, Item = MIndex>,
                Output: MIterMut<R, Item = Self>,
            {
                let mask = stencil.mask();
                indexed_apply_arity!(
                    crate::detail::apply::MaskedIndexedExprApply::gather_where_expr_into,
                    policy, values, indices, output, &mask;
                    $( $ty: $var ),+
                )
            }

            fn scatter_from_view<IndexSource, Output>(
                policy: &crate::detail::CubePolicy<R>,
                values: Self::View,
                indices: IndexSource,
                output: Output,
            ) -> Result<(), Error>
            where
                IndexSource: crate::detail::read::KernelReadBoundMany<R, Item = MIndex>,
                Output: MIterMut<R, Item = Self>,
            {
                indexed_apply_arity!(
                    crate::detail::apply::IndexedExprApply::scatter_expr_into,
                    policy, values, indices, output;
                    $( $ty: $var ),+
                )
            }

            fn scatter_where_from_view<IndexSource, Output>(
                policy: &crate::detail::CubePolicy<R>,
                values: Self::View,
                indices: IndexSource,
                stencil: crate::detail::api::PrecomputedSelection<R>,
                output: Output,
            ) -> Result<(), Error>
            where
                IndexSource: crate::detail::read::KernelReadBoundMany<R, Item = MIndex>,
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
                output: Output,
            ) -> Result<MIndex, Error>
            where
                Pred: op::PredicateOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                let ($( $var, )+) = input;
                let selected_rank = wide_predicate_rank_from_tuple!(
                    policy;
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
                let storage = crate::detail::apply::TransformPayloadApply::unary::<Self, R, Input, KernelScalarInputOp<R, Op>>(policy, input)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            fn transform_unary<Input, Op>(
                policy: &crate::detail::CubePolicy<R>,
                input: crate::detail::device::DeviceColumnView<
                    R,
                    Input,
                >,
                op: Op,
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
                let storage = crate::detail::apply::TransformPayloadApply::unary::<Self, R, Input, KernelOp<R, Op>>(policy, input)?;
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
                let storage = crate::detail::apply::TransformPayloadApply::zip2::<Self, R, Left, Right, KernelOp<R, Op>>(policy, left, right)?;
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
                let storage = crate::detail::apply::TransformPayloadApply::zip3::<Self, R, First, Second, Third, KernelOp<R, Op>>(policy, first, second, third)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            fn transform_logical3<Input, LeafA, LeafB, LeafC, Expr, Op>(
                policy: &crate::detail::CubePolicy<R>,
                bindings: crate::detail::device::KernelColumnBindings,
                len: usize,
                op: Op,
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
                >>::run_logical3(policy, bindings, len)?;
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
                >>::run_logical7(policy, bindings, len)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

            fn transform_quaternary<First, Second, Third, Fourth, Op>(
                policy: &crate::detail::CubePolicy<R>,
                first: crate::detail::device::DeviceColumnView<R, First>,
                second: crate::detail::device::DeviceColumnView<R, Second>,
                third: crate::detail::device::DeviceColumnView<R, Third>,
                fourth: crate::detail::device::DeviceColumnView<R, Fourth>,
                op: Op,
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
                let storage = crate::detail::apply::TransformPayloadApply::zip4::<Self, R, First, Second, Third, Fourth, KernelOp<R, Op>>(policy, first, second, third, fourth)?;
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
                let storage = crate::detail::apply::TransformPayloadApply::zip5::<Self, R, First, Second, Third, Fourth, Fifth, KernelOp<R, Op>>(policy, first, second, third, fourth, fifth)?;
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
                let storage = crate::detail::apply::TransformPayloadApply::zip6::<Self, R, First, Second, Third, Fourth, Fifth, Sixth, KernelOp<R, Op>>(policy, first, second, third, fourth, fifth, sixth)?;
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
                let storage = crate::detail::apply::TransformPayloadApply::zip7::<Self, R, First, Second, Third, Fourth, Fifth, Sixth, Seventh, KernelOp<R, Op>>(policy, first, second, third, fourth, fifth, sixth, seventh)?;
                crate::detail::MaterializeOutput::materialize_output(storage, policy)
            }

        }

        impl<R, $( $ty ),+> crate::detail::write::MItemWriteDispatch<R> for ($( $ty, )+)
        where
            R: Runtime,
            $( $ty: MStorageElement, )+
        {

            fn reduce_by_key_values_from_read<KeyEq, Op, Read, Output>(
                policy: &crate::detail::CubePolicy<R>,
                values: Read,
                selection: &crate::detail::control::SelectedRankControl,
                output_count: usize,
                init: Self,
                _op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                Read: crate::detail::read::KernelReadBoundMany<R, Item = Self>,
                Op: op::ReductionOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                let _ = std::marker::PhantomData::<KeyEq>;
                reduce_by_key_logical7_auto_arity!(
                    policy, values, selection, output_count, init, output;
                    $( $ty: $var ),+
                )
            }

            fn copy_selected_from_read<Read, Output>(
                policy: &crate::detail::CubePolicy<R>,
                values: Read,
                stencil: crate::detail::api::PrecomputedSelection<R>,
                output: Output,
            ) -> Result<MIndex, Error>
            where
                Read: crate::detail::read::KernelReadBoundMany<R, Item = Self>,
                Output: MIterMut<R, Item = Self>,
            {
                crate::detail::read::copy_selected_logical7_read(
                    values,
                    policy,
                    stencil,
                    output,
                )
            }

            fn gather_from_read<Read, IndexSource, Output>(
                policy: &crate::detail::CubePolicy<R>,
                values: Read,
                indices: IndexSource,
                output: Output,
            ) -> Result<(), Error>
            where
                Read: crate::detail::read::KernelReadBoundMany<R, Item = Self>,
                IndexSource: crate::detail::read::KernelReadBoundMany<R, Item = MIndex>,
                Output: MIterMut<R, Item = Self>,
            {
                crate::detail::read::gather_logical7_read(values, policy, indices, output)
            }

            fn gather_where_from_read<Read, IndexSource, Output>(
                policy: &crate::detail::CubePolicy<R>,
                values: Read,
                indices: IndexSource,
                stencil: crate::detail::api::PrecomputedSelection<R>,
                output: Output,
            ) -> Result<(), Error>
            where
                Read: crate::detail::read::KernelReadBoundMany<R, Item = Self>,
                IndexSource: crate::detail::read::KernelReadBoundMany<R, Item = MIndex>,
                Output: MIterMut<R, Item = Self>,
            {
                crate::detail::read::gather_where_logical7_read(
                    values,
                    policy,
                    indices,
                    stencil,
                    output,
                )
            }

            fn scatter_from_read<Read, IndexSource, Output>(
                policy: &crate::detail::CubePolicy<R>,
                values: Read,
                indices: IndexSource,
                output: Output,
            ) -> Result<(), Error>
            where
                Read: crate::detail::read::KernelReadBoundMany<R, Item = Self>,
                IndexSource: crate::detail::read::KernelReadBoundMany<R, Item = MIndex>,
                Output: MIterMut<R, Item = Self>,
            {
                scatter_logical7_auto_arity!(
                    policy, values, indices, output;
                    $( $ty: $var ),+
                )
            }

            fn scatter_where_from_read<Read, IndexSource, Output>(
                policy: &crate::detail::CubePolicy<R>,
                values: Read,
                indices: IndexSource,
                stencil: crate::detail::api::PrecomputedSelection<R>,
                output: Output,
            ) -> Result<(), Error>
            where
                Read: crate::detail::read::KernelReadBoundMany<R, Item = Self>,
                IndexSource: crate::detail::read::KernelReadBoundMany<R, Item = MIndex>,
                Output: MIterMut<R, Item = Self>,
            {
                let mask = stencil.mask();
                scatter_where_logical7_auto_arity!(
                    policy, values, indices, &mask, output;
                    $( $ty: $var ),+
                )
            }

            fn unique_from_read<Read, Pred, Output>(
                policy: &crate::detail::CubePolicy<R>,
                input: Read,
                _pred: Pred,
                output: Output,
            ) -> Result<MIndex, Error>
            where
                Read: crate::detail::read::KernelReadBoundMany<R, Item = Self>,
                Pred: op::BinaryPredicateOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                let len = crate::detail::read::KernelRead::len(&input);
                let Some(flags) =
                    crate::detail::read::unique_logical7_flags_read::<R, _, Pred>(
                        &input, policy,
                    )?
                else {
                    return Ok(0);
                };
                let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
                let selected_rank =
                    crate::detail::primitives::select::selected_rank_from_flags(
                        policy, len, len_u32, flags,
                    )?;
                let selection =
                    crate::detail::api::PrecomputedSelection::from_selected_rank(selected_rank);
                crate::detail::read::copy_selected_logical7_read(input, policy, selection, output)
            }

            fn partition_from_read<Read, Pred, Output>(
                policy: &crate::detail::CubePolicy<R>,
                input: Read,
                _pred: Pred,
                output: Output,
            ) -> Result<MIndex, Error>
            where
                Read: crate::detail::read::KernelReadBoundMany<R, Item = Self>,
                Pred: op::PredicateOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                let len = crate::detail::read::KernelRead::len(&input);
                let Some(flags) =
                    crate::detail::read::logical7_predicate_flags_read::<R, _, Pred>(
                        &input, policy, false,
                    )?
                else {
                    return Ok(0);
                };
                let len_u32 = u32::try_from(len).map_err(|_| Error::LengthTooLarge { len })?;
                let selected_rank =
                    crate::detail::primitives::select::selected_rank_from_flags(
                        policy, len, len_u32, flags,
                    )?;
                let (split_rank, matching_count, _failing_count) =
                    crate::detail::primitives::select::split_rank_from_selected(
                        policy,
                        selected_rank,
                    )?;
                partition_logical7_auto_arity!(
                    policy, input, &split_rank, matching_count, output;
                    $( $ty: $var ),+
                )?;
                mindex_from_usize(matching_count)
            }

            fn sort_from_read<Read, Less, Output>(
                policy: &crate::detail::CubePolicy<R>,
                input: Read,
                _less: Less,
                output: Output,
            ) -> Result<(), Error>
            where
                Read: crate::detail::read::KernelReadBoundMany<R, Item = Self>,
                Less: op::BinaryPredicateOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                let input = crate::detail::read::materialize_logical7_read(input, policy)?;
                let inner = wide_sort_from_tuple!(policy, input; $( $var ),+)?;
                output.write_from_inner(policy, inner)
            }

            fn merge_by_key_values_from_read<Left, Right, Output>(
                policy: &crate::detail::CubePolicy<R>,
                left: Left,
                right: Right,
                control: &crate::detail::control::MergeByKeyControl,
                output: Output,
            ) -> Result<(), Error>
            where
                Left: crate::detail::read::KernelReadBoundMany<R, Item = Self>,
                Right: crate::detail::read::KernelReadBoundMany<R, Item = Self>,
                Output: MIterMut<R, Item = Self>,
            {
                let left = crate::detail::read::materialize_logical7_read(left, policy)?;
                let right = crate::detail::read::materialize_logical7_read(right, policy)?;
                let left = wide_view_from_inner_tuple!(left; $( $var ),+);
                let right = wide_view_from_inner_tuple!(right; $( $var ),+);
                <Self as MAlloc<R>>::merge_by_key_values_from_views(policy, left, right, control, output)
            }

            fn adjacent_difference_from_read<Read, Op, Output>(
                policy: &crate::detail::CubePolicy<R>,
                input: Read,
                _op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                Read: crate::detail::read::KernelReadBoundMany<R, Item = Self>,
                Op: op::ReductionOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                adjacent_logical7_auto_arity!(
                    policy, input, output;
                    $( $ty: $var ),+
                )
            }

            fn inclusive_scan_from_read<Read, Op, Output>(
                policy: &crate::detail::CubePolicy<R>,
                input: Read,
                _op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                Read: crate::detail::read::KernelReadBoundMany<R, Item = Self>,
                Op: op::ReductionOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                let input = crate::detail::read::materialize_logical7_read(input, policy)?;
                let input = wide_view_from_inner_tuple!(input; $( $var ),+);
                let inner = wide_inclusive_scan_from_tuple!(policy, input; $( $var ),+)?;
                output.write_from_inner(policy, inner)
            }

            fn exclusive_scan_from_read<Read, Op, Output>(
                policy: &crate::detail::CubePolicy<R>,
                input: Read,
                init: Self,
                _op: Op,
                output: Output,
            ) -> Result<(), Error>
            where
                Read: crate::detail::read::KernelReadBoundMany<R, Item = Self>,
                Op: op::ReductionOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                let input = crate::detail::read::materialize_logical7_read(input, policy)?;
                let input = wide_view_from_inner_tuple!(input; $( $var ),+);
                let inner = wide_exclusive_scan_from_tuple!(policy, input, init; $( $var ),+)?;
                output.write_from_inner(policy, inner)
            }

            fn merge_from_read<Left, Right, Less, Output>(
                policy: &crate::detail::CubePolicy<R>,
                left: Left,
                right: Right,
                _less: Less,
                output: Output,
            ) -> Result<(), Error>
            where
                Left: crate::detail::read::KernelReadBoundMany<R, Item = Self>,
                Right: crate::detail::read::KernelReadBoundMany<R, Item = Self>,
                Less: op::BinaryPredicateOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                let left_inner = crate::detail::read::materialize_logical7_read(left, policy)?;
                let right_inner = crate::detail::read::materialize_logical7_read(right, policy)?;
                let inner = wide_merge_from_tuple!(policy, left_inner, right_inner; $( $var ),+)?;
                output.write_from_inner(policy, inner)
            }

            fn set_union_from_read<Left, Right, Less, Output>(
                policy: &crate::detail::CubePolicy<R>,
                left: Left,
                right: Right,
                right_only: &crate::detail::control::SelectedRankControl,
                _less: Less,
                output: Output,
            ) -> Result<MIndex, Error>
            where
                Left: crate::detail::read::KernelReadBoundMany<R, Item = Self>,
                Right: crate::detail::read::KernelReadBoundMany<R, Item = Self>,
                Less: op::BinaryPredicateOp<R, Self>,
                Output: MIterMut<R, Item = Self>,
            {
                set_union_logical7_auto_arity!(
                    policy, left, right, right_only, output;
                    $( $ty: $var ),+
                )
            }
        }
    };
}

impl_wide_mitem_tuple!(A: a, B: b, C: c, D: d);
impl_wide_mitem_tuple!(A: a, B: b, C: c, D: d, E: e);
impl_wide_mitem_tuple!(A: a, B: b, C: c, D: d, E: e, F: f);
impl_wide_mitem_tuple!(A: a, B: b, C: c, D: d, E: e, F: f, G: g);
