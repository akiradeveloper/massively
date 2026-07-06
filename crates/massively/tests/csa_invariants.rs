use std::{fs, path::Path};

fn crate_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn read(relative: &str) -> String {
    fs::read_to_string(crate_root().join(relative)).expect("source file should be readable")
}

fn rust_sources_under(relative: &str) -> Vec<(String, String)> {
    fn visit(path: &Path, out: &mut Vec<(String, String)>) {
        if path.is_dir() {
            for entry in fs::read_dir(path).expect("source dir should be readable") {
                let entry = entry.expect("source entry should be readable");
                visit(&entry.path(), out);
            }
            return;
        }

        if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
            let source = fs::read_to_string(path).expect("source file should be readable");
            out.push((path.display().to_string(), source));
        }
    }

    let mut sources = Vec::new();
    visit(&crate_root().join(relative), &mut sources);
    sources
}

#[test]
fn selection_control_objects_do_not_bind_payload_handles() {
    let source = read("src/detail/control/selection.rs");

    assert!(
        source.contains("struct MaskControl"),
        "CSA selection control should expose MaskControl"
    );
    assert!(
        source.contains("struct SelectedRankControl"),
        "CSA selection control should expose SelectedRankControl"
    );
    assert!(
        source.contains("struct SplitRankControl"),
        "CSA selection control should expose SplitRankControl"
    );
    assert!(
        !source.contains("SelectionHandles"),
        "payload bindings must not live in detail/control"
    );
    assert!(
        !source.contains("value:"),
        "selection control objects must not own payload value handles"
    );
}

#[test]
fn selection_family_has_no_legacy_handles_or_aliases() {
    let forbidden = [
        "SelectionHandles",
        "SelectionControl",
        "handles_from_flags",
        "handles_for_value",
        "compact_with_count",
        "selection_handles_with_policy",
        "device_expr_selection_handles_with_policy",
        "device_expr_copy_where_with_policy",
        "compact_rejected_with_selection",
        "device_expr_compact_with_flags_with_policy",
        "device_expr_compact_selected_with_split_with_policy",
        "device_expr_compact_rejected_with_split_with_policy",
        "compact_rejected_scatter_device_expr_kernel",
    ];

    for (path, source) in rust_sources_under("src") {
        for token in forbidden {
            assert!(
                !source.contains(token),
                "{} must not appear in {}",
                token,
                path
            );
        }
    }
}

#[test]
fn partition_payload_application_uses_split_rank_control() {
    let read_selection = read("src/detail/read/selection.rs");
    let read_kernel = read("src/detail/read/kernel.rs");
    let payload = read("src/detail/apply/selection.rs");

    assert!(
        read_selection.contains("split_rank_from_selected"),
        "read partition should build SplitRankControl before payload application"
    );
    assert!(
        read_kernel.contains("split_rank_from_selected"),
        "wide tuple partition should build SplitRankControl before payload application"
    );
    assert!(
        payload.contains("struct SplitPayloadApply")
            && payload.contains("pub(in crate::detail) fn apply_expr"),
        "partition payload application should be owned by SplitPayloadApply"
    );
    assert!(
        payload.contains("control: &'a select::SplitRankControl"),
        "partition payload helpers should take SplitRankControl explicitly"
    );
    assert!(
        read_selection.contains("SplitPayloadApply::new")
            && read_kernel.contains("SplitPayloadApply::new"),
        "partition call sites should use the CSA split payload-apply boundary"
    );
}

#[test]
fn payload_apply_boundary_is_explicit() {
    let detail_mod = read("src/detail/mod.rs");
    let api_mod = read("src/detail/api/mod.rs");
    let apply_mod = read("src/detail/apply/mod.rs");
    let selection_payload = read("src/detail/apply/selection.rs");

    assert!(
        detail_mod.contains("pub(crate) mod apply;"),
        "detail should expose apply as a first-class internal module"
    );
    assert!(
        !api_mod.contains("mod payload;"),
        "detail/api should not own the payload-apply module"
    );
    assert!(
        !api_mod.contains("crate::detail::apply::"),
        "detail/api should not re-export apply objects"
    );
    assert!(
        apply_mod.contains("mod selection;")
            && apply_mod.contains("mod permutation;")
            && apply_mod.contains("mod query;")
            && apply_mod.contains("mod materialize;")
            && apply_mod.contains("mod mask;")
            && apply_mod.contains("mod range;")
            && apply_mod.contains("mod merge;")
            && apply_mod.contains("mod ordering;")
            && apply_mod.contains("mod search;")
            && apply_mod.contains("mod scan;")
            && apply_mod.contains("mod reduce;")
            && apply_mod.contains("mod transform;")
            && apply_mod.contains("pub(in crate::detail) use selection::")
            && apply_mod.contains("pub(in crate::detail) use permutation::")
            && apply_mod.contains("pub(in crate::detail) use query::")
            && apply_mod.contains("pub(in crate::detail) use materialize::")
            && apply_mod.contains("pub(in crate::detail) use mask::")
            && apply_mod.contains("pub(in crate::detail) use range::")
            && apply_mod.contains("pub(in crate::detail) use merge::")
            && apply_mod.contains("pub(in crate::detail) use ordering::")
            && apply_mod.contains("pub(in crate::detail) use search::")
            && apply_mod.contains("pub(in crate::detail) use scan::")
            && apply_mod.contains("pub(in crate::detail) use reduce::")
            && apply_mod.contains("pub(in crate::detail) use transform::"),
        "detail/apply should split apply families into their own modules"
    );
    assert!(
        selection_payload.contains("struct SelectedPayloadApply"),
        "selected payload apply should have a typed CSA operation object"
    );
    assert!(
        selection_payload.contains("struct SplitPayloadApply"),
        "split payload apply should have a typed CSA operation object"
    );
    assert!(
        selection_payload.contains("SelectedRankControl")
            && selection_payload.contains("SplitRankControl"),
        "payload apply should be typed by CSA controls"
    );
    assert!(
        selection_payload.contains("fn apply_expr") && selection_payload.contains("fn apply_value"),
        "payload apply objects should own payload application methods"
    );
    assert!(
        selection_payload.contains("fn apply_expr2")
            && selection_payload.contains("fn apply_expr3")
            && selection_payload.contains("fn apply_expr4")
            && selection_payload.contains("fn apply_expr5")
            && selection_payload.contains("fn apply_expr6")
            && selection_payload.contains("fn apply_expr7")
            && selection_payload.contains("fn apply_value2")
            && selection_payload.contains("fn apply_value3"),
        "payload apply should expose multi-column insertion points"
    );
    assert!(
        selection_payload.contains("device_expr_apply_selected2_with_policy")
            && selection_payload.contains("device_expr_apply_selected3_with_policy")
            && selection_payload.contains("device_expr_apply_selected4_with_policy")
            && selection_payload.contains("device_expr_apply_selected5_with_policy")
            && selection_payload.contains("device_expr_apply_selected6_with_policy")
            && selection_payload.contains("device_expr_apply_selected7_with_policy"),
        "tuple selected payload apply should route through fused CSA apply helpers"
    );
    assert!(
        selection_payload.contains("device_expr_apply_split2_with_policy")
            && selection_payload.contains("device_expr_apply_split3_with_policy")
            && selection_payload.contains("device_expr_apply_split4_with_policy")
            && selection_payload.contains("device_expr_apply_split5_with_policy")
            && selection_payload.contains("device_expr_apply_split6_with_policy")
            && selection_payload.contains("device_expr_apply_split7_with_policy"),
        "tuple split payload apply should route through fused CSA apply helpers"
    );
    assert!(
        selection_payload.contains("device_expr_compact_with_selection_with_policy")
            && selection_payload.contains("device_expr_compact_split_with_split_with_policy"),
        "payload apply wrappers should own the compact implementation vocabulary"
    );
}

#[test]
fn transform_dispatch_uses_transform_payload_apply() {
    let apply = read("src/detail/apply/transform.rs");
    let item_impls = read("src/detail/impls/item.rs");

    assert!(
        apply.contains("struct TransformPayloadApply")
            && apply.contains("fn unary<")
            && apply.contains("fn zip2<")
            && apply.contains("fn zip3<")
            && apply.contains("fn zip4<")
            && apply.contains("fn zip5<")
            && apply.contains("fn zip6<")
            && apply.contains("fn zip7<")
            && apply.contains("Output::run(policy"),
        "TransformPayloadApply should own transform payload dispatch boundaries"
    );
    assert!(
        item_impls.matches("TransformPayloadApply::").count() >= 14
            && item_impls.contains("TransformPayloadApply::unary")
            && item_impls.contains("TransformPayloadApply::zip7")
            && !item_impls.contains(">>::run(policy"),
        "MItem transform dispatch should route through TransformPayloadApply"
    );
}

#[test]
fn public_algorithm_apis_hide_iterator_lowering_details() {
    for relative in [
        "src/algorithm/api/scan.rs",
        "src/algorithm/api/reduce.rs",
        "src/algorithm/api/ordering.rs",
        "src/algorithm/api/unique.rs",
        "src/algorithm/api/indexed.rs",
        "src/algorithm/api/selection.rs",
        "src/algorithm/api/set.rs",
        "src/algorithm/api/search.rs",
        "src/algorithm/api/predicate.rs",
        "src/algorithm/api/transform.rs",
    ] {
        let source = read(relative);
        for forbidden in [
            "MIterDispatch",
            "MIterMutDispatch",
            "MExecutableIter",
            "MAlloc",
            "KernelRead",
            "lower_read(",
            "into_alloc_view_with_policy",
            " as MAlloc",
            "as crate::value::MAlloc",
        ] {
            assert!(
                !source.contains(forbidden),
                "{relative} must not expose internal iterator lowering detail {forbidden}"
            );
        }
    }

    let ordering = read("src/algorithm/api/ordering.rs");
    let unique = read("src/algorithm/api/unique.rs");
    let scan = read("src/algorithm/api/scan.rs");
    let reduce = read("src/algorithm/api/reduce.rs");
    assert!(ordering.contains("sort_by_key_with_policy"));
    assert!(ordering.contains("merge_by_key_with_policy"));
    assert!(unique.contains("unique_by_key_with_policy"));
    assert!(scan.contains("inclusive_scan_by_key_with_policy"));
    assert!(scan.contains("exclusive_scan_by_key_with_policy"));
    assert!(reduce.contains("reduce_by_key_with_policy"));
}

#[test]
fn legacy_miter_dispatch_module_is_removed() {
    let dispatch_mod = read("src/detail/dispatch/mod.rs");
    let iter = read("src/iter/mod.rs");

    assert!(
        !crate_root().join("src/detail/dispatch/iter.rs").exists()
            && !crate_root()
                .join("src/detail/dispatch/iter_mut.rs")
                .exists()
            && !dispatch_mod.contains("mod iter;")
            && !dispatch_mod.contains("mod iter_mut;")
            && !dispatch_mod.contains("MIterDispatch")
            && !dispatch_mod.contains("MIterMutDispatch")
            && !iter.contains("MExecutableIter"),
        "legacy MIterDispatch/MIterMutDispatch/MExecutableIter surface should not exist"
    );
    assert!(
        iter.contains("pub trait MIter")
            && iter.contains("type Read: crate::detail::read::KernelRead")
            && iter.contains("fn lower_read(")
            && iter.contains("fn validate_executor("),
        "MIter should own the hidden lowering hooks needed by public read APIs"
    );
}

#[test]
fn scan_by_key_into_writes_through_segmented_scan_apply() {
    let scan_apply = read("src/detail/apply/scan.rs");
    let read_scan = read("src/detail/read/by_key/scan.rs");
    let api_scan = read("src/algorithm/api/scan.rs");
    let iter = read("src/iter/mod.rs");
    let scan_kernels = read("src/detail/kernels/scan.rs");

    assert!(
        scan_apply.contains("fn inclusive_expr_into")
            && scan_apply.contains("inclusive_scan_by_flags_one_into")
            && scan_apply.contains("fn exclusive_expr_into")
            && scan_apply.contains("exclusive_scan_by_flags_one_into"),
        "SegmentedScanApply should expose direct-output single-column scan-by-key helpers"
    );
    assert!(
        scan_apply.contains("fn inclusive_expr2_into")
            && scan_apply.contains("inclusive_scan_by_flags_two_into")
            && scan_apply.contains("fn exclusive_expr2_into")
            && scan_apply.contains("exclusive_scan_by_flags_two_into")
            && scan_apply.contains("fn inclusive_expr3_into")
            && scan_apply.contains("inclusive_scan_by_flags_three_into")
            && scan_apply.contains("fn exclusive_expr3_into")
            && scan_apply.contains("exclusive_scan_by_flags_three_into"),
        "SegmentedScanApply should expose direct-output tuple scan-by-key helpers"
    );
    assert!(
        scan_apply.contains("fn inclusive_views7_into")
            && scan_apply.contains("inclusive_scan_by_flags_seven_views_into")
            && scan_apply.contains("fn exclusive_views7_into")
            && scan_apply.contains("exclusive_scan_by_flags_seven_views_into"),
        "SegmentedScanApply should expose direct-output wide tuple scan-by-key helpers"
    );
    assert!(
        read_scan
            .matches("SegmentedScanApply::new(control)")
            .count()
            >= 7
            && read_scan.contains("apply.inclusive_expr::<")
            && read_scan.contains("apply.exclusive_expr::<")
            && read_scan.contains("apply.inclusive_expr2::<")
            && read_scan.contains("apply.exclusive_expr2::<")
            && read_scan.contains("apply.inclusive_expr3::<")
            && read_scan.contains("apply.exclusive_expr3::<")
            && read_scan.contains("apply.inclusive_views7::<")
            && read_scan.contains("apply.exclusive_views7::<"),
        "scan_by_key read paths should call SegmentedScanApply helpers for scalar, tuple, and wide values"
    );
    assert!(
        api_scan.contains("inclusive_scan_by_key_with_policy")
            && api_scan.contains("exclusive_scan_by_key_with_policy")
            && !api_scan.contains("lower_read(")
            && !api_scan.contains("KernelRead")
            && iter.contains("pub trait MIter")
            && iter.contains("fn inclusive_scan_by_key_with_policy")
            && iter.contains("fn exclusive_scan_by_key_with_policy")
            && iter.contains("inclusive_scan_by_key_read")
            && iter.contains("exclusive_scan_by_key_read"),
        "public scan_by_key APIs should delegate to hidden MIter lowering hooks before execution"
    );
    assert!(
        scan_kernels.contains("output_offset: &[u32]")
            && scan_kernels.contains("output[output_offset[0] as usize + global]"),
        "single-column scan-by-key kernels should support DeviceSliceMut output offsets"
    );
    assert!(
        scan_kernels.contains("output_offsets: &[u32]")
            && scan_kernels.contains("output_a[output_offsets[0] as usize + global]")
            && scan_kernels.contains("output_b[output_offsets[1] as usize + global]")
            && scan_kernels.contains("output_c[output_offsets[2] as usize + global]")
            && scan_kernels.contains("output_g[output_offsets[6] as usize + global]"),
        "tuple scan-by-key kernels should support per-column DeviceSliceMut output offsets"
    );
}

#[test]
fn apply_objects_live_under_detail_apply() {
    for (path, source) in rust_sources_under("src/detail") {
        if path.contains("/src/detail/apply/") {
            continue;
        }

        for line in source.lines() {
            let defines_apply_struct = line.contains("struct ") && line.contains("Apply");
            assert!(
                !defines_apply_struct,
                "apply operation structs should live under detail/apply, found `{}` in {}",
                line.trim(),
                path
            );
        }
    }
}

#[test]
fn selection_call_sites_use_payload_apply_vocabulary() {
    let read_kernel = read("src/detail/read/kernel.rs");
    let read_selection = read("src/detail/read/selection.rs");
    let by_key_reduce = read("src/detail/read/by_key/reduce.rs");
    let by_key_selection = read("src/detail/read/by_key/selection.rs");

    assert!(
        read_kernel.contains("SelectedPayloadApply::new"),
        "wide tuple selection paths should use typed selected payload apply"
    );
    assert!(
        read_kernel.contains("SplitPayloadApply::new"),
        "wide tuple partition should use typed split payload apply"
    );
    assert!(
        read_selection.contains("SelectedPayloadApply::new")
            && read_selection.contains("SplitPayloadApply::new"),
        "read selection copy/unique/partition paths should use typed payload apply operations"
    );
    assert!(
        read_selection.contains("apply_expr2") && read_selection.contains("apply_expr3"),
        "read selection tuple paths should route through multi-column payload apply hooks"
    );
    assert!(
        by_key_reduce.contains("SelectedPayloadApply::new")
            && by_key_selection.contains("SelectedPayloadApply::new"),
        "by-key selection and reduce should use typed selected payload apply"
    );
    assert!(
        !read_kernel.contains("let handles =") && !read_selection.contains("let handles ="),
        "SelectedRankControl values should not use legacy handles naming"
    );
    assert!(
        !read_kernel.contains("device_expr_compact_with_selection_with_policy")
            && !read_selection.contains("device_expr_compact_with_selection_with_policy")
            && !by_key_reduce.contains("compact_value_with_count"),
        "selection call sites should not name compact implementation helpers directly"
    );
    assert!(
        !read_kernel.contains("device_expr_apply_selected_with_policy")
            && !read_kernel.contains("device_expr_apply_split_with_policy")
            && !read_kernel.contains("device_value_apply_selected_with_policy")
            && !read_selection.contains("device_expr_apply_selected_with_policy")
            && !read_selection.contains("device_expr_apply_split_with_policy")
            && !read_selection.contains("device_value_apply_selected_with_policy")
            && !by_key_reduce.contains("device_value_apply_selected_with_policy")
            && !by_key_selection.contains("device_expr_apply_selected_with_policy"),
        "wide tuple and by-key call sites should use payload apply objects, not wrapper functions"
    );
}

#[test]
fn compact_implementation_vocabulary_is_payload_private() {
    let allowed = [
        "src/detail/apply/mod.rs",
        "src/detail/api/expr/mod.rs",
        "src/detail/api/expr/selection.rs",
        "src/detail/apply/selection.rs",
        "src/detail/primitives/select.rs",
    ];
    let forbidden = [
        "device_expr_compact_with_selection_with_policy",
        "device_expr_compact_split_with_split_with_policy",
        "compact_value_with_count",
    ];

    for (path, source) in rust_sources_under("src") {
        if allowed.iter().any(|allowed| path.ends_with(allowed)) {
            continue;
        }

        for token in forbidden {
            assert!(
                !source.contains(token),
                "{} must not appear outside payload/expr implementation boundary in {}",
                token,
                path
            );
        }
    }
}

#[test]
fn selected_payload_wrapper_functions_are_retired() {
    let forbidden = [
        "device_expr_apply_selected_with_policy",
        "device_expr_apply_split_with_policy",
        "device_value_apply_selected_with_policy",
    ];

    for (path, source) in rust_sources_under("src") {
        for token in forbidden {
            assert!(
                !source.contains(token),
                "{} should be retired in favor of typed payload apply objects in {}",
                token,
                path
            );
        }
    }
}

#[test]
fn fused_split_kernel_is_the_partition_apply_kernel() {
    let expr_selection = read("src/detail/api/expr/selection.rs");

    assert!(
        expr_selection.contains("device_expr_compact_split_with_split_with_policy"),
        "expr implementation should keep the fused split compact helper"
    );
    assert!(
        expr_selection.contains("compact_split_scatter_device_expr_kernel"),
        "expr implementation should launch the fused split scatter kernel"
    );
    assert!(
        expr_selection.contains("control: &select::SplitRankControl"),
        "fused split implementation should take SplitRankControl explicitly"
    );
}

#[test]
fn tuple_selected_payload_apply_has_fused_apply_kernels() {
    let payload = read("src/detail/apply/selection.rs");
    let expr_selection = read("src/detail/api/expr/selection.rs");
    let kernels = read("src/detail/kernels/expr.rs");

    assert!(
        payload.contains("device_expr_apply_selected2_with_policy")
            && payload.contains("device_expr_apply_selected3_with_policy")
            && payload.contains("device_expr_apply_selected4_with_policy")
            && payload.contains("device_expr_apply_selected5_with_policy")
            && payload.contains("device_expr_apply_selected6_with_policy")
            && payload.contains("device_expr_apply_selected7_with_policy"),
        "SelectedPayloadApply tuple hooks should call fused selected apply helpers"
    );
    assert!(
        expr_selection.contains("device_expr_apply_selected2_with_policy")
            && expr_selection.contains("selected_apply_tuple2_device_expr_kernel")
            && expr_selection.contains("device_expr_apply_selected3_with_policy")
            && expr_selection.contains("selected_apply_tuple3_device_expr_kernel")
            && expr_selection.contains("device_expr_apply_selected4_with_policy")
            && expr_selection.contains("selected_apply_tuple4_device_expr_kernel")
            && expr_selection.contains("device_expr_apply_selected5_with_policy")
            && expr_selection.contains("selected_apply_tuple5_device_expr_kernel")
            && expr_selection.contains("device_expr_apply_selected6_with_policy")
            && expr_selection.contains("selected_apply_tuple6_device_expr_kernel")
            && expr_selection.contains("device_expr_apply_selected7_with_policy")
            && expr_selection.contains("selected_apply_tuple7_device_expr_kernel"),
        "tuple selected apply helpers should launch fused tuple kernels"
    );
    assert!(
        kernels.contains("define_selected_apply_tuple_device_expr_kernel")
            && kernels.contains("selected_apply_tuple2_device_expr_kernel")
            && kernels.contains("selected_apply_tuple3_device_expr_kernel")
            && kernels.contains("selected_apply_tuple4_device_expr_kernel")
            && kernels.contains("selected_apply_tuple5_device_expr_kernel")
            && kernels.contains("selected_apply_tuple6_device_expr_kernel")
            && kernels.contains("selected_apply_tuple7_device_expr_kernel"),
        "tuple selected apply kernels should be generated through macro-shaped boundaries"
    );
}

#[test]
fn tuple_split_payload_apply_has_fused_apply_kernels() {
    let payload = read("src/detail/apply/selection.rs");
    let read_kernel = read("src/detail/read/kernel.rs");
    let expr_selection = read("src/detail/api/expr/selection.rs");
    let kernels = read("src/detail/kernels/expr.rs");

    assert!(
        payload.contains("device_expr_apply_split2_with_policy")
            && payload.contains("device_expr_apply_split3_with_policy")
            && payload.contains("device_expr_apply_split4_with_policy")
            && payload.contains("device_expr_apply_split5_with_policy")
            && payload.contains("device_expr_apply_split6_with_policy")
            && payload.contains("device_expr_apply_split7_with_policy"),
        "SplitPayloadApply tuple hooks should call split apply helpers"
    );
    assert!(
        read_kernel.contains("SplitPayloadApply::new(&split_rank")
            && read_kernel.contains("payload_apply.$apply"),
        "wide tuple partition should route through the arity-specific split apply hook"
    );
    assert!(
        expr_selection.contains("device_expr_apply_split2_with_policy")
            && expr_selection.contains("split_apply_tuple2_device_expr_kernel")
            && expr_selection.contains("device_expr_apply_split3_with_policy")
            && expr_selection.contains("split_apply_tuple3_device_expr_kernel")
            && expr_selection.contains("device_expr_apply_split4_with_policy")
            && expr_selection.contains("split_apply_tuple4_device_expr_kernel")
            && expr_selection.contains("device_expr_apply_split5_with_policy")
            && expr_selection.contains("split_apply_tuple5_device_expr_kernel")
            && expr_selection.contains("device_expr_apply_split6_with_policy")
            && expr_selection.contains("split_apply_tuple6_device_expr_kernel")
            && expr_selection.contains("device_expr_apply_split7_with_policy")
            && expr_selection.contains("device_expr_apply_split4_with_policy")
            && expr_selection.contains("device_expr_apply_split3_with_policy"),
        "tuple split apply helpers should launch fused tuple kernels up to arity 6 and stage arity 7"
    );
    assert!(
        kernels.contains("define_split_apply_tuple_device_expr_kernel")
            && kernels.contains("split_apply_tuple2_device_expr_kernel")
            && kernels.contains("split_apply_tuple3_device_expr_kernel")
            && kernels.contains("split_apply_tuple4_device_expr_kernel")
            && kernels.contains("split_apply_tuple5_device_expr_kernel")
            && kernels.contains("split_apply_tuple6_device_expr_kernel")
            && !kernels.contains("split_apply_tuple7_device_expr_kernel"),
        "tuple split apply kernels should be generated through macro-shaped boundaries up to the backend-safe arity"
    );
}

#[test]
fn flags_only_consumers_stop_at_mask_control() {
    let expr_selection = read("src/detail/api/expr/selection.rs");
    let indexed = read("src/detail/api/expr/indexed.rs");

    assert!(
        expr_selection.contains("replace_where_into_with_control")
            && expr_selection.contains("control: &select::MaskControl"),
        "replace_where should consume only MaskControl"
    );
    assert!(
        indexed.contains("device_expr_gather_where_into_with_control")
            && indexed.contains("control: &select::MaskControl"),
        "gather_where should consume only MaskControl"
    );
    assert!(
        indexed.contains("device_expr_scatter_where_into_with_control")
            && indexed.contains("control: &select::MaskControl"),
        "scatter_where should consume only MaskControl"
    );
}

#[test]
fn mask_consumers_use_mask_apply_boundaries() {
    let payload = read("src/detail/apply/mask.rs");
    let api_mod = read("src/detail/api/mod.rs");
    let single_impls = read("src/detail/impls/iter/single.rs");
    let tuple_impls = read("src/detail/impls/iter/tuple.rs");
    let read_kernel = read("src/detail/read/kernel.rs");

    assert!(
        payload.contains("struct MaskWriteApply")
            && payload.contains("mask: &'a select::MaskControl")
            && payload.contains("fn replace_value")
            && payload.contains("expr::replace_where_into_with_control"),
        "MaskWriteApply should own fixed-position replace write boundaries"
    );
    assert!(
        payload.contains("struct MaskedIndexedExprApply")
            && payload.contains("fn gather_where_expr_into")
            && payload.contains("fn scatter_where_expr_into")
            && payload.contains("expr::device_expr_gather_where_into_with_control")
            && payload.contains("expr::device_expr_scatter_where_into_with_control"),
        "MaskedIndexedExprApply should own masked indexed write boundaries"
    );
    assert!(
        !api_mod.contains("MaskWriteApply")
            && !api_mod.contains("MaskedIndexedExprApply")
            && !api_mod.contains("replace_where_into_with_control")
            && !api_mod.contains("device_expr_gather_where_into_with_control")
            && !api_mod.contains("device_expr_scatter_where_into_with_control"),
        "detail api should not expose mask apply objects or raw mask wrapper re-exports"
    );
    assert!(
        read_kernel.contains("MaskedIndexedExprApply::gather_where_expr_into")
            && read_kernel.contains("MaskedIndexedExprApply::scatter_where_expr_into")
            && single_impls.contains("MaskWriteApply::new(&mask, &output)")
            && !single_impls.contains("device_expr_gather_where_into_with_control")
            && !single_impls.contains("device_expr_scatter_where_into_with_control")
            && !single_impls.contains("replace_where_into_with_control"),
        "single-column mask consumers should use typed mask apply objects"
    );
    assert!(
        read_kernel
            .matches("MaskedIndexedExprApply::gather_where_expr_into")
            .count()
            >= 2
            && read_kernel
                .matches("MaskedIndexedExprApply::scatter_where_expr_into")
                .count()
                >= 2
            && tuple_impls.contains("MaskWriteApply::new(&mask, &output.$idx)")
            && !tuple_impls.contains("device_expr_gather_where_into_with_control")
            && !tuple_impls.contains("device_expr_scatter_where_into_with_control")
            && !tuple_impls.contains("replace_where_into_with_control"),
        "tuple mask consumers should use typed mask apply objects"
    );
}

#[test]
fn sort_by_key_values_use_permutation_payload_apply() {
    let api_by_key = read("src/detail/api/ordering/by_key.rs");
    let by_key_ordering = read("src/detail/read/by_key/ordering.rs");
    let control = read("src/detail/control/ordering.rs");
    let payload = read("src/detail/apply/permutation.rs");

    assert!(
        control.contains("struct OrderingControl")
            && control.contains("fn from_sorted_indices")
            && control.contains("fn permutation"),
        "OrderingControl should represent materialized sorted order"
    );
    assert!(
        api_by_key.contains("OrderingControl::from_sorted_indices(&indices)")
            && api_by_key.contains("values.sort_by_key_values(policy, control.permutation())"),
        "sort_by_key should keep sorted order as OrderingControl before applying values"
    );
    assert!(
        by_key_ordering.contains("control: &crate::detail::control::PermutationControl")
            && by_key_ordering.contains("PermutationPayloadApply::new(control)")
            && !by_key_ordering.contains("device_expr_gather_with_policy(policy, &self"),
        "sort_by_key values should use permutation payload apply instead of raw gather helpers"
    );
    assert!(
        payload.contains("struct PermutationPayloadApply")
            && payload.contains("control: &'a crate::detail::control::PermutationControl")
            && payload.contains("device_expr_gather_with_policy(policy, expr, &indices)")
            && payload.contains("fn apply_expr_into")
            && payload
                .contains("device_expr_gather_into_with_policy(policy, expr, &indices, output)"),
        "PermutationPayloadApply should own the gather implementation boundary"
    );
}

#[test]
fn sort_by_key_into_writes_through_permutation_payload_apply() {
    let read_ordering = read("src/detail/read/by_key/ordering.rs");
    let api_ordering = read("src/algorithm/api/ordering.rs");

    assert!(
        read_ordering
            .matches("PermutationPayloadApply::new(control)")
            .count()
            >= 4
            && read_ordering.contains("apply.apply_expr(policy")
            && read_ordering.contains("apply.apply_expr2(policy")
            && read_ordering.contains("apply.apply_expr3(policy")
            && read_ordering.contains("apply.apply_expr7("),
        "sort_by_key read paths should route value payloads through PermutationPayloadApply"
    );
    assert!(
        api_ordering.contains("keys.sort_by_key_with_policy")
            && !api_ordering.contains("sort_by_key_dispatch")
            && !api_ordering.contains("crate::detail::sort_by_key("),
        "public sort_by_key API should stay on the MIter hidden method surface"
    );
}

#[test]
fn merge_by_key_into_writes_through_merge_payload_apply() {
    let read_ordering = read("src/detail/read/by_key/ordering.rs");
    let api_ordering = read("src/algorithm/api/ordering.rs");
    let payload = read("src/detail/apply/merge.rs");

    assert!(
        payload.contains("fn apply_expr_into")
            && payload.contains("device_expr_merge_by_key_values_into_with_control_with_policy("),
        "MergePayloadApply should expose direct-output merge-by-key payload application"
    );
    assert!(
        read_ordering.contains("MergeByKeyControlApply::apply_keys1")
            && read_ordering.contains("MergeByKeyControlApply::apply_keys2")
            && read_ordering.contains("MergeByKeyControlApply::apply_keys3")
            && read_ordering
                .matches("MergePayloadApply::new(control)")
                .count()
                >= 4
            && read_ordering.contains("apply.apply_expr(policy")
            && read_ordering.contains("apply.apply_expr2(")
            && read_ordering.contains("apply.apply_expr3(")
            && read_ordering.contains("apply.apply_expr7("),
        "merge_by_key read paths should route keys through MergeByKeyControlApply and values through MergePayloadApply"
    );
    assert!(
        api_ordering.contains("left_keys.merge_by_key_with_policy")
            && !api_ordering.contains("merge_by_key_dispatch")
            && !api_ordering.contains("crate::detail::merge_by_key("),
        "public merge_by_key API should stay on the MIter hidden method surface"
    );
}

#[test]
fn unique_by_key_into_writes_through_selected_payload_apply() {
    let read_selection = read("src/detail/read/by_key/selection.rs");
    let api_unique = read("src/algorithm/api/unique.rs");
    let payload = read("src/detail/apply/selection.rs");

    assert!(
        payload.contains("fn apply_expr_into")
            && payload.contains("device_expr_compact_into_with_selection_with_policy("),
        "SelectedPayloadApply should expose direct-output compact application"
    );
    assert!(
        read_selection.contains("unique_one_flags_read")
            && read_selection.matches("SelectedPayloadApply::new(").count() >= 4
            && read_selection.contains("payload_apply.apply_expr(policy")
            && read_selection
                .matches("payload_apply.apply_expr(policy")
                .count()
                >= 7,
        "unique_by_key read paths should route selected keys/values through SelectedPayloadApply"
    );
    assert!(
        api_unique.contains("keys.unique_by_key_with_policy")
            && !api_unique.contains("unique_by_key_dispatch")
            && !api_unique.contains("crate::detail::unique_by_key("),
        "public unique_by_key API should stay on the MIter hidden method surface"
    );
}

#[test]
fn reduce_by_key_into_writes_through_segmented_reduce_apply() {
    let read_reduce = read("src/detail/read/by_key/reduce.rs");
    let api_reduce = read("src/algorithm/api/reduce.rs");
    let apply = read("src/detail/apply/reduce.rs");

    assert!(
        apply.contains("fn apply_expr_into")
            && apply.contains("fn apply_expr2_into")
            && apply.contains("fn apply_expr3_into")
            && apply.contains("fn apply_views7_into")
            && apply.contains("reduce_by_key_tuple7_scanned_values_into"),
        "SegmentedReduceApply should expose direct-output reduce-by-key value application"
    );
    assert!(
        read_reduce
            .matches("SegmentedReduceApply::new(control)")
            .count()
            >= 4
            && read_reduce.contains("apply.apply_expr::<ValueSource, Op>")
            && read_reduce.contains("apply.apply_expr2::<ValueA, ValueB, Op>")
            && read_reduce.contains("apply.apply_expr3::<ValueA, ValueB, ValueC, Op>")
            && read_reduce.contains("apply.apply_views7::<A, B, C, D, E, F, G, Op>"),
        "reduce_by_key read paths should route values through SegmentedReduceApply"
    );
    assert!(
        api_reduce.contains("keys.reduce_by_key_with_policy")
            && !api_reduce.contains("reduce_by_key_dispatch")
            && !api_reduce.contains("crate::detail::reduce_by_key("),
        "public reduce_by_key API should stay on the MIter hidden method surface"
    );
}

#[test]
fn sort_values_use_sort_apply() {
    let apply = read("src/detail/apply/ordering.rs");
    let call_sites = read("src/detail/read/ordering.rs");

    assert!(
        apply.contains("fn apply_expr")
            && apply.contains("fn apply_expr2")
            && apply.contains("fn apply_expr3")
            && apply.contains("primitive_ordering::sort_input_with_policy")
            && apply.contains("primitive_ordering::sort_tuple2_input")
            && apply.contains("primitive_ordering::sort_tuple3_input"),
        "SortApply should own arity1-3 sort implementation boundaries"
    );
    assert!(
        call_sites.matches("SortApply::apply_expr").count() >= 2
            && call_sites.contains("SortApply::apply_expr2")
            && call_sites.contains("SortApply::apply_expr3")
            && !call_sites.contains("primitive_ordering::sort_input_with_policy")
            && !call_sites.contains("primitive_ordering::sort_tuple2_input")
            && !call_sites.contains("primitive_ordering::sort_tuple3_input"),
        "sort read call sites should route through SortApply"
    );
}

#[test]
fn sort_by_key_keys_use_sort_by_key_apply() {
    let apply = read("src/detail/apply/ordering.rs");
    let call_sites = read("src/detail/read/by_key/ordering.rs");

    assert!(
        apply.contains("fn apply_keys1")
            && apply.contains("fn apply_keys2")
            && apply.contains("fn apply_keys3")
            && apply.contains("primitive_ordering::sort_by_key_input_with_policy")
            && apply.contains("primitive_ordering::sort_tuple3_by_key_input_with_policy"),
        "SortByKeyApply should own key ordering implementation boundaries"
    );
    assert!(
        call_sites.contains("SortByKeyApply::apply_keys1")
            && call_sites.contains("SortByKeyApply::apply_keys2")
            && call_sites.contains("SortByKeyApply::apply_keys3")
            && !call_sites.contains("primitive_ordering::sort_by_key_input_with_policy")
            && !call_sites.contains("primitive_ordering::sort_tuple3_by_key_input_with_policy"),
        "sort-by-key key call sites should route through SortByKeyApply"
    );
}

#[test]
fn merge_by_key_keys_use_merge_by_key_control_apply() {
    let apply = read("src/detail/apply/ordering.rs");
    let call_sites = read("src/detail/read/by_key/ordering.rs");

    assert!(
        apply.contains("struct MergeByKeyControlApply")
            && apply.contains("fn apply_keys1")
            && apply.contains("fn apply_keys2")
            && apply.contains("fn apply_keys3")
            && apply.contains("device_expr_merge_by_key_control_with_policy")
            && apply.contains("device_expr_merge_tuple2_by_key_control_with_policy")
            && apply.contains("device_expr_merge_tuple3_by_key_control_with_policy"),
        "MergeByKeyControlApply should own merge-by-key key/control implementation boundaries"
    );
    assert!(
        call_sites.contains("MergeByKeyControlApply::apply_keys1")
            && call_sites.contains("MergeByKeyControlApply::apply_keys2")
            && call_sites.contains("MergeByKeyControlApply::apply_keys3")
            && !call_sites.contains("device_expr_merge_by_key_control_with_policy")
            && !call_sites.contains("device_expr_merge_tuple2_by_key_control_with_policy")
            && !call_sites.contains("device_expr_merge_tuple3_by_key_control_with_policy"),
        "merge-by-key key call sites should route through MergeByKeyControlApply"
    );
}

#[test]
fn wide_sort_values_use_permutation_payload_apply() {
    let item_impls = read("src/detail/impls/item.rs");

    assert!(
        item_impls.contains("OrderingControl::from_sorted_indices(&indices)")
            && item_impls.contains("PermutationPayloadApply::new(control.permutation())")
            && item_impls.contains("apply.apply_expr4")
            && item_impls.contains("apply.apply_expr5")
            && item_impls.contains("apply.apply_expr6")
            && item_impls.contains("apply.apply_expr7"),
        "wide tuple sort read paths should lower sorted indices to OrderingControl and apply payload through PermutationPayloadApply"
    );
}

#[test]
fn gather_read_uses_permutation_payload_apply() {
    let gather = read("src/detail/read/gather.rs");

    assert!(
        gather.contains("PermutationControl::from_indices(&index_values)")
            && gather.contains("PermutationPayloadApply::new(&control)")
            && !gather.contains("device_expr_gather_with_policy(policy, &self"),
        "gather read paths should lower index payload to PermutationControl and apply payload through PermutationPayloadApply"
    );
}

#[test]
fn indexed_expr_dispatch_uses_indexed_expr_apply() {
    let payload = read("src/detail/apply/permutation.rs");
    let api_mod = read("src/detail/api/mod.rs");
    let read_kernel = read("src/detail/read/kernel.rs");

    assert!(
        payload.contains("struct IndexedExprApply")
            && payload.contains("fn gather_expr")
            && payload.contains("fn gather_expr_into")
            && payload.contains("fn scatter_expr_into")
            && payload.contains("device_expr_gather_with_policy(policy, input, indices)")
            && payload
                .contains("device_expr_gather_into_with_policy(policy, input, indices, output)")
            && payload
                .contains("device_expr_scatter_into_with_policy(policy, values, indices, output)"),
        "IndexedExprApply should own allocation-free indexed expr apply boundaries"
    );
    assert!(
        !api_mod.contains("IndexedExprApply")
            && !api_mod.contains("device_expr_gather_into_with_policy")
            && !api_mod.contains("device_expr_scatter_into_with_policy"),
        "detail api should not expose indexed expr apply objects or raw indexed wrapper re-exports"
    );
    assert!(
        read_kernel.contains("IndexedExprApply::gather_expr_into")
            && read_kernel.contains("IndexedExprApply::scatter_expr_into")
            && !read_kernel.contains("device_expr_gather_into_with_policy")
            && !read_kernel.contains("device_expr_gather_with_policy")
            && !read_kernel.contains("device_expr_scatter_into_with_policy"),
        "single-column indexed dispatch should use IndexedExprApply"
    );
    assert!(
        read_kernel
            .matches("IndexedExprApply::gather_expr_into")
            .count()
            >= 2
            && read_kernel
                .matches("IndexedExprApply::scatter_expr_into")
                .count()
                >= 2,
        "tuple indexed dispatch should use IndexedExprApply"
    );
}

#[test]
fn scatter_read_uses_indexed_write_apply() {
    let scatter = read("src/detail/read/scatter.rs");
    let payload = read("src/detail/apply/permutation.rs");

    assert!(
        scatter.contains("PermutationControl::from_indices(&index_values)")
            && scatter.contains("IndexedWriteApply::new(&control)")
            && !scatter.contains("device_expr_scatter_into_with_policy(policy, values, indices"),
        "scatter read paths should lower indices to PermutationControl and write payload through IndexedWriteApply"
    );
    assert!(
        payload.contains("struct IndexedWriteApply")
            && payload.contains("scatter_expr_into")
            && payload
                .contains("device_expr_scatter_into_with_policy(policy, values, &indices, output)"),
        "IndexedWriteApply should own the scatter implementation boundary"
    );
}

#[test]
fn materialize_write_paths_use_materialize_write_apply() {
    let payload = read("src/detail/apply/materialize.rs");
    let api_mod = read("src/detail/api/mod.rs");
    let single_impls = read("src/detail/impls/iter/single.rs");
    let tuple_impls = read("src/detail/impls/iter/tuple.rs");

    assert!(
        payload.contains("struct MaterializeWriteApply")
            && payload.contains("fn collect_expr")
            && payload.contains("fn copy_where_expr")
            && payload.contains("device_expr_collect_into_with_policy(policy, expr, self.output)")
            && payload.contains(
                "device_expr_copy_where_into_with_policy(policy, expr, stencil, self.output, pred)"
            ),
        "MaterializeWriteApply should own contiguous collect/copy-where write boundaries"
    );
    assert!(
        !api_mod.contains("MaterializeWriteApply")
            && !api_mod.contains("device_expr_collect_into_with_policy")
            && !api_mod.contains("device_expr_copy_where_into_with_policy"),
        "detail api should not expose typed write apply objects or collect/copy wrapper re-exports"
    );
    assert!(
        single_impls.contains("MaterializeWriteApply::new(&output).collect_expr")
            && single_impls.contains("MaterializeWriteApply::new(&output).copy_where_expr")
            && !single_impls.contains("device_expr_collect_into_with_policy")
            && !single_impls.contains("device_expr_copy_where_into_with_policy"),
        "single-column write paths should use MaterializeWriteApply"
    );
    assert!(
        tuple_impls.contains("MaterializeWriteApply::new(&output.$idx)")
            && !tuple_impls.contains("device_expr_collect_into_with_policy")
            && !tuple_impls.contains("device_expr_copy_where_into_with_policy"),
        "tuple write paths should use MaterializeWriteApply per column"
    );
}

#[test]
fn fill_and_concat_use_payload_apply_boundaries() {
    let materialize_payload = read("src/detail/apply/materialize.rs");
    let range_payload = read("src/detail/apply/range.rs");
    let api_mod = read("src/detail/api/mod.rs");
    let single_impls = read("src/detail/impls/iter/single.rs");
    let tuple_impls = read("src/detail/impls/iter/tuple.rs");
    let item_impls = read("src/detail/impls/item.rs");

    assert!(
        materialize_payload.contains("struct FillWriteApply")
            && materialize_payload.contains("fn fill_value")
            && materialize_payload
                .contains("primitives::fill_slice_with_policy(policy, value, self.output)"),
        "FillWriteApply should own fill-slice write boundaries"
    );
    assert!(
        range_payload.contains("struct ConcatPayloadApply")
            && range_payload.contains("fn apply_values")
            && range_payload
                .contains("primitives::range::concat_device_with_policy(policy, left, right)"),
        "ConcatPayloadApply should own concat materialization boundaries"
    );
    assert!(
        !api_mod.contains("FillWriteApply") && !api_mod.contains("ConcatPayloadApply"),
        "detail api should not expose fill/concat apply objects"
    );
    assert!(
        single_impls.contains("FillWriteApply::new(&output).fill_value")
            && !single_impls.contains("fill_slice_with_policy"),
        "single-column fill should use FillWriteApply"
    );
    assert!(
        tuple_impls.contains("FillWriteApply::new(&output.$idx)")
            && item_impls.contains("ConcatPayloadApply::apply_values")
            && !tuple_impls.contains("fill_slice_with_policy")
            && !tuple_impls.contains("concat_device_with_policy"),
        "tuple fill and wide concat should use typed apply objects"
    );
}

#[test]
fn materialize_payload_paths_use_materialize_payload_apply() {
    let payload = read("src/detail/apply/materialize.rs");
    let api_mod = read("src/detail/api/mod.rs");
    let memory = read("src/detail/api/memory.rs");
    let gather = read("src/detail/read/gather.rs");
    let scatter = read("src/detail/read/scatter.rs");
    let item_impls = read("src/detail/impls/item.rs");

    assert!(
        payload.contains("struct MaterializePayloadApply")
            && payload.contains("fn collect_expr")
            && payload.contains("expr::device_expr_collect_with_policy(policy, expr)"),
        "MaterializePayloadApply should own expression-to-owned-payload collect boundaries"
    );
    assert!(
        !api_mod.contains("MaterializePayloadApply")
            && !api_mod.contains("device_expr_collect_with_policy"),
        "detail api should not expose MaterializePayloadApply or the raw collect re-export"
    );
    assert!(
        memory.contains("MaterializePayloadApply::collect_expr")
            && gather.contains("MaterializePayloadApply::collect_expr")
            && scatter.contains("MaterializePayloadApply::collect_expr")
            && item_impls.contains("MaterializePayloadApply::collect_expr"),
        "owned materialize call sites should use MaterializePayloadApply"
    );

    let allowed = [
        "src/detail/apply/materialize.rs",
        "src/detail/api/expr/mod.rs",
        "src/detail/api/expr/collect.rs",
    ];
    for (path, source) in rust_sources_under("src") {
        if allowed.iter().any(|allowed| path.ends_with(allowed)) {
            continue;
        }
        assert!(
            !source.contains("device_expr_collect_with_policy"),
            "raw collect implementation should not be visible outside materialize apply boundary in {}",
            path
        );
    }
}

#[test]
fn scatter_where_read_combines_mask_control_and_indexed_write_apply() {
    let scatter = read("src/detail/read/scatter.rs");
    let payload = read("src/detail/apply/permutation.rs");

    assert!(
        scatter.contains("let mask = stencil.selection_flags_with_policy(policy, false)?;")
            && scatter.contains("IndexedWriteApply::new(&write_control)")
            && scatter.contains("apply.scatter_expr_where_into(policy, values, &mask, &output)?")
            && !scatter.contains("scatter_if_flags_kernel::launch_unchecked"),
        "scatter_where read paths should combine MaskControl with IndexedWriteApply instead of launching raw kernels"
    );
    assert!(
        payload.contains("scatter_expr_where_into")
            && payload.contains("mask: &select::MaskControl")
            && payload.contains("device_expr_scatter_where_into_with_control(policy, values, &indices, mask, output)"),
        "IndexedWriteApply should own the masked scatter implementation boundary"
    );
}

#[test]
fn reverse_read_uses_range_payload_apply() {
    let ordering = read("src/detail/read/ordering.rs");
    let control = read("src/detail/control/range.rs");
    let payload = read("src/detail/apply/range.rs");

    assert!(
        ordering.contains("RangeControl::reverse")
            && ordering.contains("RangePayloadApply::new(&control)")
            && !ordering.contains("device_expr_reverse_collect(policy, &self"),
        "reverse read paths should build RangeControl and apply payload through RangePayloadApply"
    );
    assert!(
        control.contains("struct RangeControl") && control.contains("RangeMapping::Reverse"),
        "RangeControl should represent reverse range mapping"
    );
    assert!(
        payload.contains("struct RangePayloadApply")
            && payload.contains("RangeMapping::Reverse")
            && payload.contains("device_expr_reverse_collect(policy, expr)"),
        "RangePayloadApply should own the reverse collect implementation boundary"
    );
}

#[test]
fn scan_by_key_values_use_segmented_scan_apply() {
    let apply = read("src/detail/apply/scan.rs");
    let call_sites = read("src/detail/read/by_key/scan.rs");

    assert!(
        apply.contains("struct SegmentedScanApply")
            && apply.contains("control: &'a ScanByKeyControl<R>")
            && apply.contains(
                "inclusive_scan_by_flags_one::<Source, Op>(policy, source, self.control)"
            )
            && apply.contains(
                "exclusive_scan_by_flags_one::<Source, Op>(policy, source, self.control, init)"
            )
            && apply.contains("fn inclusive_views4")
            && apply.contains("fn inclusive_views5")
            && apply.contains("fn inclusive_views6")
            && apply.contains("inclusive_scan_by_flags_seven_views::<R, A, B, C, D, E, F, G, Op>")
            && apply.contains("fn exclusive_views4")
            && apply.contains("fn exclusive_views5")
            && apply.contains("fn exclusive_views6")
            && apply.contains("exclusive_scan_by_flags_seven_views::<R, A, B, C, D, E, F, G, Op>"),
        "SegmentedScanApply should own the segmented scan helper boundary"
    );
    assert!(
        call_sites
            .matches("SegmentedScanApply::new(control)")
            .count()
            >= 14
            && call_sites.contains("apply.inclusive_views4::<A, B, C, D, Op>")
            && call_sites.contains("apply.inclusive_views5::<A, B, C, D, E, Op>")
            && call_sites.contains("apply.inclusive_views6::<A, B, C, D, E, F, Op>")
            && call_sites.contains("apply.inclusive_views7::<A, B, C, D, E, F, G, Op>")
            && call_sites.contains("apply.exclusive_views4::<A, B, C, D, Op>")
            && call_sites.contains("apply.exclusive_views5::<A, B, C, D, E, Op>")
            && call_sites.contains("apply.exclusive_views6::<A, B, C, D, E, F, Op>")
            && call_sites.contains("apply.exclusive_views7::<A, B, C, D, E, F, G, Op>")
            && !call_sites.contains("primitive_range::indices_mindex")
            && !call_sites.contains("Tuple4AsTuple7BinaryOp<Op>")
            && !call_sites.contains("Tuple5AsTuple7BinaryOp<Op>")
            && !call_sites.contains("Tuple6AsTuple7BinaryOp<Op>")
            && !call_sites.contains("struct SegmentedScanApply"),
        "scan-by-key value arities should apply payload through SegmentedScanApply"
    );
}

#[test]
fn linear_scan_values_use_linear_scan_apply() {
    let apply = read("src/detail/apply/scan.rs");
    let call_sites = read("src/detail/read/scan.rs");
    let wide_call_sites = read("src/detail/read/kernel.rs");

    assert!(
        apply.contains("fn inclusive_expr1")
            && apply.contains("fn exclusive_expr1")
            && apply.contains("fn adjacent_expr1")
            && apply.contains("fn inclusive_expr2")
            && apply.contains("fn exclusive_expr2")
            && apply.contains("fn adjacent_expr2")
            && apply.contains("fn inclusive_expr3")
            && apply.contains("fn exclusive_expr3")
            && apply.contains("fn adjacent_expr3")
            && apply.contains("fn inclusive_views4")
            && apply.contains("fn inclusive_views5")
            && apply.contains("fn inclusive_views6")
            && apply.contains("fn inclusive_views7")
            && apply.contains("fn exclusive_views4")
            && apply.contains("fn exclusive_views5")
            && apply.contains("fn exclusive_views6")
            && apply.contains("fn exclusive_views7")
            && apply.contains("fn adjacent_views4")
            && apply.contains("fn adjacent_views5")
            && apply.contains("fn adjacent_views6")
            && apply.contains("fn adjacent_views7"),
        "LinearScanApply should own linear scan and adjacent-difference apply boundaries"
    );
    assert!(
        call_sites.matches("LinearScanApply::").count() >= 9
            && !call_sites.contains("primitive_scan::inclusive_scan_tuple")
            && !call_sites.contains("primitive_scan::exclusive_scan_tuple")
            && !call_sites.contains("primitive_scan::adjacent_difference_tuple")
            && !call_sites.contains("device_expr_adjacent_difference_with_policy"),
        "linear scan read paths should route through LinearScanApply"
    );
    assert!(
        wide_call_sites
            .contains("impl_wide_zip_scan!(ZipRead4, inclusive_views4, exclusive_views4")
            && wide_call_sites
                .contains("impl_wide_zip_scan!(ZipRead7, inclusive_views7, exclusive_views7")
            && wide_call_sites
                .contains("impl_wide_zip_adjacent_difference!(ZipRead4, adjacent_views4")
            && wide_call_sites
                .contains("impl_wide_zip_adjacent_difference!(ZipRead7, adjacent_views7")
            && wide_call_sites.contains("LinearScanApply::$inclusive")
            && wide_call_sites.contains("LinearScanApply::$exclusive")
            && wide_call_sites.contains("LinearScanApply::$apply")
            && !wide_call_sites.contains("inclusive_scan_tuple7_device_views")
            && !wide_call_sites.contains("exclusive_scan_tuple7_device_views")
            && !wide_call_sites.contains("adjacent_difference_tuple7_device_views"),
        "linear wide tuple scan dispatch should route through LinearScanApply instead of primitives"
    );
}

#[test]
fn linear_reduce_values_use_linear_reduce_apply() {
    let apply = read("src/detail/apply/reduce.rs");
    let call_sites = read("src/detail/read/reduce.rs");
    let wide_call_sites = read("src/detail/read/kernel.rs");

    assert!(
        apply.contains("fn apply_expr1")
            && apply.contains("fn apply_expr2")
            && apply.contains("fn apply_expr3")
            && apply.contains("fn apply_views4")
            && apply.contains("fn apply_views5")
            && apply.contains("fn apply_views6")
            && apply.contains("fn apply_views7"),
        "LinearReduceApply should own linear reduce apply boundaries"
    );
    assert!(
        call_sites.matches("LinearReduceApply::").count() >= 3
            && !call_sites.contains("primitive_reduce::reduce_tuple"),
        "linear reduce read paths should route through LinearReduceApply"
    );
    assert!(
        wide_call_sites.contains("impl_flat_zip_reduce_wide!(ZipRead4, apply_views4")
            && wide_call_sites.contains("impl_flat_zip_reduce_wide!(ZipRead7, apply_views7")
            && wide_call_sites.contains("LinearReduceApply::$method")
            && !wide_call_sites.contains("reduce_tuple7_device_expr"),
        "linear wide tuple reduce dispatch should route through LinearReduceApply instead of primitives"
    );
}

#[test]
fn reduce_by_key_values_use_segmented_reduce_apply() {
    let apply = read("src/detail/apply/reduce.rs");
    let call_sites = read("src/detail/read/by_key/reduce.rs");

    assert!(
        apply.contains("struct SegmentedReduceApply")
            && apply.contains("control: &'a ReduceByKeyControl<R>")
            && apply.contains("SegmentedScanApply::new(&scan_control)")
            && apply.contains("fn apply_views4")
            && apply.contains("fn apply_views5")
            && apply.contains("fn apply_views6")
            && apply.contains("fn apply_views7")
            && apply.contains("reduce_by_key_tuple7_scanned_values!")
            && apply.contains("SelectedPayloadApply::new"),
        "SegmentedReduceApply should own segmented scan, init application, and selected output compaction"
    );
    assert!(
        call_sites
            .matches("SegmentedReduceApply::new(control)")
            .count()
            >= 7
            && call_sites.contains("apply.apply_expr::<ValueSource, Op>")
            && call_sites.contains("apply.apply_expr2::<ValueA, ValueB, Op>")
            && call_sites.contains("apply.apply_expr3::<ValueA, ValueB, ValueC, Op>")
            && call_sites.contains("apply.apply_views4::<A, B, C, D, Op>")
            && call_sites.contains("apply.apply_views5::<A, B, C, D, E, Op>")
            && call_sites.contains("apply.apply_views6::<A, B, C, D, E, F, Op>")
            && call_sites.contains("apply.apply_views7::<A, B, C, D, E, F, G, Op>")
            && !call_sites.contains("primitive_range::indices_mindex")
            && !call_sites.contains("Tuple4AsTuple7BinaryOp<Op>")
            && !call_sites.contains("Tuple5AsTuple7BinaryOp<Op>")
            && !call_sites.contains("Tuple6AsTuple7BinaryOp<Op>")
            && !call_sites.contains("struct SegmentedReduceApply"),
        "reduce-by-key value arities should apply payload through SegmentedReduceApply"
    );
    assert!(
        call_sites
            .matches("SegmentedReduceApply::new(control)")
            .count()
            >= 7
            && !call_sites.contains("reduce_by_key_apply_init_kernel")
            && !call_sites.contains("reduce_by_key_tuple2_apply_init_kernel")
            && !call_sites.contains("reduce_by_key_tuple3_apply_init_kernel")
            && !call_sites.contains("reduce_by_key_tuple7_apply_init_kernel"),
        "reduce-by-key read paths should route apply-init work through SegmentedReduceApply"
    );
}

#[test]
fn by_key_control_generation_uses_segment_control() {
    let control = read("src/detail/control/segment.rs");
    let scan = read("src/detail/read/by_key/scan.rs");
    let reduce = read("src/detail/read/by_key/reduce.rs");
    let selection = read("src/detail/read/by_key/selection.rs");

    assert!(
        control.contains("struct SegmentControl")
            && control.contains("fn from_head_flags")
            && control.contains("fn from_head_end_flags")
            && control.contains("fn from_segment"),
        "SegmentControl should be the by-key control family constructor boundary"
    );
    assert!(
        scan.matches("SegmentControl::from_head_flags").count() >= 3
            && scan.contains("ScanByKeyControl::from_segment(&segment)"),
        "scan-by-key control generation should build SegmentControl before deriving ScanByKeyControl"
    );
    assert!(
        reduce
            .matches("SegmentControl::from_head_end_flags")
            .count()
            >= 4
            && reduce.contains("ReduceByKeyControl::from_segment"),
        "reduce-by-key control generation should build SegmentControl before deriving ReduceByKeyControl"
    );
    assert!(
        selection.contains("SegmentControl::from_head_flags(flags, len, len_u32)")
            && selection.contains(
                "selected_rank_from_flags(policy, len, len_u32, segment.head_flags.clone())"
            ),
        "unique-by-key control generation should bridge SegmentControl into SelectedRankControl"
    );
}

#[test]
fn merge_by_key_values_use_merge_payload_apply() {
    let ordering = read("src/detail/read/by_key/ordering.rs");
    let payload = read("src/detail/apply/merge.rs");

    assert!(
        payload.contains("struct MergePayloadApply")
            && payload.contains("control: &'a crate::detail::control::MergeByKeyControl")
            && payload.contains("device_expr_merge_by_key_values_with_control_with_policy(policy, left, right, self.control)"),
        "MergePayloadApply should own the merge-by-key values implementation boundary"
    );
    assert!(
        ordering.matches("MergePayloadApply::new(control)").count() >= 5
            && ordering.contains("apply.apply_expr(policy, &self, &right_values)")
            && ordering.contains("apply.apply_expr2")
            && ordering.contains("apply.apply_expr3")
            && ordering.contains("apply.apply_expr7")
            && !ordering.contains("device_expr_merge_by_key_values_with_control_with_policy"),
        "merge-by-key value arities should apply payload through MergePayloadApply"
    );
}

#[test]
fn plain_merge_uses_merge_expr_apply() {
    let apply = read("src/detail/apply/ordering.rs");
    let control = read("src/detail/control/ordering.rs");
    let call_sites = read("src/detail/api/ordering/mod.rs");
    let kernels = read("src/detail/kernels/ordering.rs");

    assert!(
        control.contains("struct MergeControl")
            && control.contains("source_side")
            && control.contains("source_index")
            && control.contains("fn as_merge_by_key_control"),
        "plain merge should have a true MergeControl carrying source side/index"
    );
    assert!(
        apply.contains("struct MergeControlApply")
            && apply.contains("device_expr_merge_control_with_policy::<Left, Right, Less>")
            && apply.contains("MergePayloadApply::new(&payload_control)")
            && apply.contains(".apply_expr(policy, left, right)"),
        "MergeExprApply should compose MergeControlApply with MergePayloadApply"
    );
    assert!(
        kernels.contains("merge_path_control_device_expr_kernel")
            && call_sites.contains("device_expr_merge_control_with_policy")
            && call_sites.matches("MergeExprApply::apply_expr").count() >= 2
            && !call_sites.contains("device_expr_merge_with_policy::<")
            && !call_sites.contains("merge_path_device_expr_kernel::launch_unchecked"),
        "plain merge call sites should build MergeControl and avoid direct fused payload launch"
    );
}

#[test]
fn set_algorithms_use_membership_control_and_selected_payload_apply() {
    let apply = read("src/detail/apply/ordering.rs");
    let ordering = read("src/detail/api/ordering/mod.rs");
    let macro_start = ordering
        .find("macro_rules! impl_tuple_pair_ordering")
        .expect("tuple pair ordering macro should exist");
    let macro_end = ordering[macro_start..]
        .find("pub fn merge")
        .map(|offset| macro_start + offset)
        .expect("public merge function should delimit tuple pair ordering macro");
    let tuple_pair_ordering = &ordering[macro_start..macro_end];

    assert!(
        apply.contains("struct SetMembershipControlApply")
            && apply.contains("fn set_union_expr")
            && apply.contains("fn set_intersection_expr")
            && apply.contains("fn set_difference_expr")
            && apply.contains("fn tuple2_membership_expr_flags_with_policy")
            && apply.contains("fn tuple3_membership_expr_flags_with_policy"),
        "SetMembershipControlApply should own set membership control-generation boundaries"
    );
    assert!(
        ordering.contains("SetMembershipControlApply::set_union_expr")
            && ordering.contains("SetMembershipControlApply::set_intersection_expr")
            && ordering.contains("SetMembershipControlApply::set_difference_expr")
            && ordering.contains("SetMembershipControlApply::$membership_expr_fn")
            && !tuple_pair_ordering.contains("let flags = $membership_expr_fn::<"),
        "set algorithm call sites should route membership control generation through SetMembershipControlApply"
    );
    assert!(
        tuple_pair_ordering
            .matches("SelectedPayloadApply::new(&selection, count)")
            .count()
            >= 3
            && tuple_pair_ordering.contains("selected_apply.$selected_apply")
            && !tuple_pair_ordering.contains("device_expr_apply_selected_with_policy"),
        "tuple set_union/set_intersection/set_difference should apply selected payload through SelectedPayloadApply"
    );
}

#[test]
fn predicate_queries_use_query_apply() {
    let selection = read("src/detail/read/selection.rs");
    let payload = read("src/detail/apply/query.rs");

    assert!(
        payload.contains("struct QueryApply")
            && payload.contains("fn count_expr")
            && payload.contains("fn find_expr")
            && payload.contains("fn count_selected")
            && payload.contains("fn first_selected"),
        "QueryApply should own predicate query readback boundaries"
    );
    assert!(
        selection.contains("QueryApply::count_expr::<Source, Pred>")
            && selection.contains("QueryApply::find_expr::<Source, Pred>")
            && selection.contains("QueryApply::count_selected(policy, &selected_rank)")
            && selection.contains("QueryApply::first_selected(policy, selected_rank)"),
        "predicate query read paths should go through QueryApply"
    );
}

#[test]
fn search_queries_use_search_control_and_query_apply() {
    let control = read("src/detail/control/search.rs");
    let control_mod = read("src/detail/control/mod.rs");
    let payload = read("src/detail/apply/query.rs");
    let search_apply = read("src/detail/apply/search.rs");
    let search = read("src/detail/api/search.rs");
    let selection = read("src/detail/read/selection.rs");
    let read_kernel = read("src/detail/read/kernel.rs");

    assert!(
        control.contains("struct SearchControl")
            && control.contains("fn from_flags")
            && control_mod.contains("pub(crate) use search::SearchControl"),
        "SearchControl should represent flag-based scalar/index query controls"
    );
    assert!(
        payload.contains("fn first_flag")
            && payload.contains("fn first_flag_or")
            && payload.contains("fn first_unset_flag")
            && payload.contains("fn first_unset_flag_or")
            && payload.contains("fn minmax_expr"),
        "QueryApply should own flag readback helpers for search-style queries"
    );
    assert!(
        search_apply.contains("struct SearchControlApply")
            && search_apply.contains("fn adjacent_find_expr")
            && search_apply.contains("fn lower_bound_expr")
            && search_apply.contains("fn upper_bound_expr")
            && search_apply.contains("fn is_sorted_until_expr")
            && search_apply.contains("fn mismatch_expr")
            && search_apply.contains("fn find_first_of_expr")
            && search_apply.contains("fn lexicographical_compare_expr"),
        "SearchControlApply should own scalar search control/query operation boundaries"
    );
    assert!(
        search.matches("SearchControl::from_flags").count() >= 10
            && search.matches("QueryApply::first_flag").count() >= 5
            && search.matches("QueryApply::first_flag_or").count() >= 5
            && search.matches("QueryApply::minmax_expr").count() >= 3
            && search.contains("SearchControlApply::adjacent_find_expr")
            && search.contains("SearchControlApply::lower_bound_expr")
            && search.contains("SearchControlApply::upper_bound_expr")
            && search.contains("SearchControlApply::is_sorted_until_expr")
            && search.contains("SearchControlApply::mismatch_expr")
            && search.contains("SearchControlApply::find_first_of_expr")
            && search.contains("SearchControlApply::lexicographical_compare_expr")
            && !search.contains("search::first_flag")
            && !search.contains("primitives::search")
            && !search.contains("device_expr_minmax_element_with_policy"),
        "search algorithms should turn flags into SearchControl and read through QueryApply"
    );
    assert!(
        selection.contains("QueryApply::first_unset_flag")
            && selection.contains("QueryApply::first_flag(policy, search)")
            && !selection.contains("primitive_search::first_flag"),
        "partition query readback should use SearchControl through QueryApply"
    );
    assert!(
        read_kernel.contains("SearchControl::from_flags")
            && read_kernel.contains("QueryApply::first_flag"),
        "wide tuple search fast paths should use SearchControl and QueryApply for readback"
    );
}

#[test]
fn search_many_outputs_use_search_payload_apply() {
    let search = read("src/detail/api/search.rs");
    let apply = read("src/detail/apply/search.rs");
    let call_sites = &search;

    assert!(
        apply.contains("fn lower_bound_many_expr")
            && apply.contains("fn upper_bound_many_expr")
            && apply.contains("trait TupleSearchPayloadApply")
            && apply.contains("fn lower_bound_many_payload")
            && apply.contains("fn upper_bound_many_payload")
            && apply.contains("impl_tuple_search_payload_apply!")
            && apply.contains("fn empty_or_zero")
            && apply.contains("fn prepare")
            && apply.contains("fn finish")
            && apply.contains("lower_bound_device_expr_many_kernel::launch_unchecked")
            && apply.contains("upper_bound_device_expr_many_kernel::launch_unchecked")
            && apply.contains("tuple7_lower_bound_device_expr_many_kernel")
            && apply.contains("tuple7_upper_bound_device_expr_many_kernel"),
        "SearchPayloadApply should own single-column and tuple many-bound payload materialization"
    );
    assert!(
        call_sites.contains("SearchPayloadApply::lower_bound_many_expr")
            && call_sites.contains("SearchPayloadApply::upper_bound_many_expr")
            && !call_sites.contains("device_expr_lower_bound_many")
            && !call_sites.contains("device_expr_upper_bound_many"),
        "single-column many-bound call sites should route through SearchPayloadApply"
    );
    assert!(
        call_sites.contains("TupleSearchPayloadApply")
            && call_sites.contains("lower_bound_many_payload(self, policy, values)")
            && call_sites.contains("upper_bound_many_payload(self, policy, values)")
            && !call_sites.contains("lower_bound_device_expr_many_kernel::launch_unchecked")
            && !call_sites.contains("upper_bound_device_expr_many_kernel::launch_unchecked"),
        "tuple many-bound paths should route kernel launches through TupleSearchPayloadApply"
    );
}

#[test]
fn search_many_kernel_launches_stay_inside_payload_apply() {
    let many_bound_launches = [
        "lower_bound_device_expr_many_kernel::launch_unchecked",
        "upper_bound_device_expr_many_kernel::launch_unchecked",
        "tuple2_lower_bound_device_expr_many_kernel::launch_unchecked",
        "tuple2_upper_bound_device_expr_many_kernel::launch_unchecked",
        "tuple3_lower_bound_device_expr_many_kernel::launch_unchecked",
        "tuple3_upper_bound_device_expr_many_kernel::launch_unchecked",
        "tuple4_lower_bound_device_expr_many_kernel::launch_unchecked",
        "tuple4_upper_bound_device_expr_many_kernel::launch_unchecked",
        "tuple5_lower_bound_device_expr_many_kernel::launch_unchecked",
        "tuple5_upper_bound_device_expr_many_kernel::launch_unchecked",
        "tuple6_lower_bound_device_expr_many_kernel::launch_unchecked",
        "tuple6_upper_bound_device_expr_many_kernel::launch_unchecked",
        "tuple7_lower_bound_device_expr_many_kernel::launch_unchecked",
        "tuple7_upper_bound_device_expr_many_kernel::launch_unchecked",
    ];

    for (path, source) in rust_sources_under("src/detail") {
        for launch in many_bound_launches {
            if !source.contains(launch) {
                continue;
            }

            assert!(
                path.ends_with("src/detail/apply/search.rs"),
                "many-bound search launch should stay inside SearchPayloadApply/TupleSearchPayloadApply, found {} in {}",
                launch,
                path
            );
        }
    }
}

#[test]
fn csa_documentation_names_active_family_boundaries() {
    let csa = fs::read_to_string(crate_root().join("../../doc.ai/ALGORITHM_CSA.md"))
        .expect("CSA design doc should be readable");

    for token in [
        "SortApply",
        "SortByKeyApply",
        "OrderingControl",
        "PermutationPayloadApply",
        "SearchControl",
        "QueryApply",
        "SearchPayloadApply",
        "TupleSearchPayloadApply",
        "MergeControlApply",
        "MergeControl",
        "MergeByKeyControlApply",
        "MergePayloadApply",
        "arity multiplication",
        "多列対応は各 algorithm call site ではなく、主に apply 側の責務にする",
        "control generation",
    ] {
        assert!(
            csa.contains(token),
            "CSA design doc should name active boundary `{}`",
            token
        );
    }
}

#[test]
fn wide_tuple_selection_reuses_selected_rank_for_copy_and_remove_where() {
    let read_kernel = read("src/detail/read/kernel.rs");

    assert!(
        read_kernel.contains("let selected_rank = stencil.selected_rank();"),
        "wide tuple copy/remove where should reuse the precomputed SelectedRankControl"
    );
    assert!(
        read_kernel.contains("SelectedPayloadApply::new(selected_rank, count)"),
        "wide tuple copy/remove where should apply a shared SelectedRankControl through payload apply"
    );
    assert!(
        !read_kernel.contains("stencil.selected_rank().flag.clone()"),
        "wide tuple copy/remove where must not rebuild rank from cloned stencil flags per column"
    );
}

#[test]
fn detail_control_does_not_launch_kernels() {
    let control_dir = crate_root().join("src/detail/control");
    let forbidden = ["launch_unchecked", "CubeCount", "CubeDim", "BufferArg"];

    for entry in fs::read_dir(control_dir).expect("control dir should be readable") {
        let entry = entry.expect("control entry should be readable");
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
            continue;
        }

        let source = fs::read_to_string(&path).expect("control source should be readable");
        for token in forbidden {
            assert!(
                !source.contains(token),
                "{} must not appear in {}",
                token,
                path.display()
            );
        }
    }
}

#[test]
fn raw_kernel_launches_stay_in_csa_implementation_boundaries() {
    let allowed = [
        "src/detail/api/expr/collect.rs",
        "src/detail/api/expr/indexed.rs",
        "src/detail/api/expr/scan.rs",
        "src/detail/api/expr/search.rs",
        "src/detail/api/expr/selection.rs",
        "src/detail/api/memory.rs",
        "src/detail/api/ordering/mod.rs",
        "src/detail/api/search.rs",
        "src/detail/api/selection_control.rs",
        "src/detail/apply/reduce.rs",
        "src/detail/apply/search.rs",
        "src/detail/impls/item.rs",
        "src/detail/impls/iter/tuple.rs",
        "src/detail/impls/mod.rs",
        "src/detail/kernels/expr.rs",
        "src/detail/kernels/ordering.rs",
        "src/detail/kernels/range.rs",
        "src/detail/kernels/scan.rs",
        "src/detail/kernels/selection.rs",
        "src/detail/primitives/ordering/radix.rs",
        "src/detail/primitives/ordering/sort.rs",
        "src/detail/primitives/range.rs",
        "src/detail/primitives/reduce.rs",
        "src/detail/primitives/scan.rs",
        "src/detail/primitives/search.rs",
        "src/detail/primitives/select.rs",
        "src/detail/read/by_key/reduce.rs",
        "src/detail/read/by_key/scan.rs",
        "src/detail/read/gather.rs",
        "src/detail/read/kernel.rs",
        "src/detail/read/selection.rs",
    ];

    for (path, source) in rust_sources_under("src/detail") {
        if !source.contains("launch_unchecked") {
            continue;
        }

        assert!(
            allowed.iter().any(|allowed| path.ends_with(allowed)),
            "raw kernel launch should stay in CSA implementation boundaries, found in {}",
            path
        );
    }
}
