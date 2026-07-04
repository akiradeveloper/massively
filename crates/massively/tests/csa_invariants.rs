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
    let tuple_impls = read("src/detail/impls/iter/tuple.rs");
    let payload = read("src/detail/api/payload.rs");

    assert!(
        read_selection.contains("split_rank_from_selected"),
        "read partition should build SplitRankControl before payload application"
    );
    assert!(
        tuple_impls.contains("split_rank_from_selected"),
        "wide tuple partition should build SplitRankControl before payload application"
    );
    assert!(
        payload.contains("device_expr_apply_split_with_policy"),
        "partition payload application should fuse selected and rejected split application"
    );
    assert!(
        payload.contains("control: &select::SplitRankControl"),
        "partition payload helpers should take SplitRankControl explicitly"
    );
    assert!(
        read_selection.contains("SplitPayloadApply::new")
            && tuple_impls.contains("SplitPayloadApply::new"),
        "partition call sites should use the CSA split payload-apply boundary"
    );
}

#[test]
fn payload_apply_boundary_is_explicit() {
    let api_mod = read("src/detail/api/mod.rs");
    let payload = read("src/detail/api/payload.rs");

    assert!(
        api_mod.contains("mod payload;"),
        "detail/api should expose a payload-apply module"
    );
    assert!(
        payload.contains("device_expr_apply_selected_with_policy"),
        "selected expr payload apply should have a CSA entry point"
    );
    assert!(
        payload.contains("struct SelectedPayloadApply"),
        "selected payload apply should have a typed CSA operation object"
    );
    assert!(
        payload.contains("struct SplitPayloadApply"),
        "split payload apply should have a typed CSA operation object"
    );
    assert!(
        payload.contains("device_expr_apply_split_with_policy"),
        "split expr payload apply should have a CSA entry point"
    );
    assert!(
        payload.contains("device_value_apply_selected_with_policy"),
        "raw value-handle payload apply should have a CSA entry point"
    );
    assert!(
        payload.contains("SelectedRankControl") && payload.contains("SplitRankControl"),
        "payload apply should be typed by CSA controls"
    );
    assert!(
        payload.contains("fn apply_expr") && payload.contains("fn apply_value"),
        "payload apply objects should own payload application methods"
    );
    assert!(
        payload.contains("fn apply_expr2")
            && payload.contains("fn apply_expr3")
            && payload.contains("fn apply_expr4")
            && payload.contains("fn apply_expr5")
            && payload.contains("fn apply_expr6")
            && payload.contains("fn apply_expr7")
            && payload.contains("fn apply_value2")
            && payload.contains("fn apply_value3"),
        "payload apply should expose multi-column insertion points"
    );
    assert!(
        payload.contains("device_expr_apply_selected2_with_policy")
            && payload.contains("device_expr_apply_selected3_with_policy")
            && payload.contains("device_expr_apply_selected4_with_policy")
            && payload.contains("device_expr_apply_selected5_with_policy")
            && payload.contains("device_expr_apply_selected6_with_policy")
            && payload.contains("device_expr_apply_selected7_with_policy"),
        "tuple selected payload apply should route through fused CSA apply helpers"
    );
    assert!(
        payload.contains("device_expr_apply_split2_with_policy")
            && payload.contains("device_expr_apply_split3_with_policy")
            && payload.contains("device_expr_apply_split4_with_policy")
            && payload.contains("device_expr_apply_split5_with_policy")
            && payload.contains("device_expr_apply_split6_with_policy")
            && payload.contains("device_expr_apply_split7_with_policy"),
        "tuple split payload apply should route through fused CSA apply helpers"
    );
    assert!(
        payload.contains("device_expr_compact_with_selection_with_policy")
            && payload.contains("device_expr_compact_split_with_split_with_policy"),
        "payload apply wrappers should own the compact implementation vocabulary"
    );
}

#[test]
fn selection_call_sites_use_payload_apply_vocabulary() {
    let tuple_impls = read("src/detail/impls/iter/tuple.rs");
    let read_selection = read("src/detail/read/selection.rs");
    let by_key_reduce = read("src/detail/read/by_key/reduce.rs");
    let by_key_selection = read("src/detail/read/by_key/selection.rs");

    assert!(
        tuple_impls.contains("SelectedPayloadApply::new"),
        "wide tuple selection paths should use typed selected payload apply"
    );
    assert!(
        tuple_impls.contains("SplitPayloadApply::new"),
        "wide tuple partition should use typed split payload apply"
    );
    assert!(
        read_selection.contains("device_expr_apply_selected_with_policy"),
        "read selection paths should use selected payload apply"
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
        !tuple_impls.contains("let handles =") && !read_selection.contains("let handles ="),
        "SelectedRankControl values should not use legacy handles naming"
    );
    assert!(
        !tuple_impls.contains("device_expr_compact_with_selection_with_policy")
            && !read_selection.contains("device_expr_compact_with_selection_with_policy")
            && !by_key_reduce.contains("compact_value_with_count"),
        "selection call sites should not name compact implementation helpers directly"
    );
    assert!(
        !tuple_impls.contains("device_expr_apply_selected_with_policy")
            && !tuple_impls.contains("device_expr_apply_split_with_policy")
            && !tuple_impls.contains("device_value_apply_selected_with_policy")
            && !by_key_reduce.contains("device_value_apply_selected_with_policy")
            && !by_key_selection.contains("device_expr_apply_selected_with_policy"),
        "wide tuple and by-key call sites should use payload apply objects, not wrapper functions"
    );
}

#[test]
fn compact_implementation_vocabulary_is_payload_private() {
    let allowed = [
        "src/detail/api/payload.rs",
        "src/detail/api/expr/mod.rs",
        "src/detail/api/expr/selection.rs",
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
    let payload = read("src/detail/api/payload.rs");
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
    let payload = read("src/detail/api/payload.rs");
    let tuple_impls = read("src/detail/impls/iter/tuple.rs");
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
        tuple_impls.contains("SplitPayloadApply::new")
            && tuple_impls.contains("payload_apply.$selected_apply"),
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
fn wide_tuple_selection_reuses_selected_rank_for_copy_and_remove_where() {
    let tuple_impls = read("src/detail/impls/iter/tuple.rs");

    assert!(
        tuple_impls.contains("let selected_rank = stencil.selected_rank();"),
        "wide tuple copy/remove where should reuse the precomputed SelectedRankControl"
    );
    assert!(
        tuple_impls.contains("SelectedPayloadApply::new(selected_rank, count)"),
        "wide tuple copy/remove where should apply a shared SelectedRankControl through payload apply"
    );
    assert!(
        !tuple_impls.contains("stencil.selected_rank().flag.clone()"),
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
