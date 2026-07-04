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
        payload.contains("struct SelectedPayloadApply"),
        "selected payload apply should have a typed CSA operation object"
    );
    assert!(
        payload.contains("struct SplitPayloadApply"),
        "split payload apply should have a typed CSA operation object"
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
fn mask_consumers_use_mask_apply_boundaries() {
    let payload = read("src/detail/api/payload.rs");
    let api_mod = read("src/detail/api/mod.rs");
    let single_impls = read("src/detail/impls/iter/single.rs");
    let tuple_impls = read("src/detail/impls/iter/tuple.rs");

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
        api_mod.contains("MaskWriteApply")
            && api_mod.contains("MaskedIndexedExprApply")
            && !api_mod.contains("replace_where_into_with_control")
            && !api_mod.contains("device_expr_gather_where_into_with_control")
            && !api_mod.contains("device_expr_scatter_where_into_with_control"),
        "detail api should expose mask apply objects instead of raw mask wrapper re-exports"
    );
    assert!(
        single_impls.contains("MaskedIndexedExprApply::gather_where_expr_into")
            && single_impls.contains("MaskedIndexedExprApply::scatter_where_expr_into")
            && single_impls.contains("MaskWriteApply::new(&mask, &output)")
            && !single_impls.contains("device_expr_gather_where_into_with_control")
            && !single_impls.contains("device_expr_scatter_where_into_with_control")
            && !single_impls.contains("replace_where_into_with_control"),
        "single-column mask consumers should use typed mask apply objects"
    );
    assert!(
        tuple_impls.contains("MaskedIndexedExprApply::gather_where_expr_into")
            && tuple_impls.contains("MaskedIndexedExprApply::scatter_where_expr_into")
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
    let payload = read("src/detail/api/payload.rs");

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
            && payload.contains("device_expr_gather_with_policy(policy, expr, &indices)"),
        "PermutationPayloadApply should own the gather implementation boundary"
    );
}

#[test]
fn sort_values_use_sort_apply() {
    let ordering = read("src/detail/read/ordering.rs");
    let apply_start = ordering
        .find("struct SortApply")
        .expect("SortApply should exist");
    let call_start = ordering[apply_start..]
        .find("impl<Source, Less> KernelSortInput<Less> for Source")
        .map(|offset| apply_start + offset)
        .expect("sort apply should precede sort impls");
    let apply = &ordering[apply_start..call_start];
    let call_sites = &ordering[call_start..];

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
    let by_key_ordering = read("src/detail/read/by_key/ordering.rs");
    let apply_start = by_key_ordering
        .find("struct SortByKeyApply")
        .expect("SortByKeyApply should exist");
    let call_start = by_key_ordering[apply_start..]
        .find("pub(crate) trait KernelMergeByKeyKeys")
        .map(|offset| apply_start + offset)
        .expect("sort-by-key apply should precede merge-by-key traits");
    let apply = &by_key_ordering[apply_start..call_start];
    let call_sites = &by_key_ordering[call_start..];

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
fn wide_sort_values_use_permutation_payload_apply() {
    let tuple_impls = read("src/detail/impls/iter/tuple.rs");
    let macro_start = tuple_impls
        .find("macro_rules! impl_wide_sort_dispatch_body")
        .expect("wide sort dispatch macro should exist");
    let macro_end = tuple_impls[macro_start..]
        .find("macro_rules! impl_wide_sort_by_three_key_dispatch_body")
        .map(|offset| macro_start + offset)
        .expect("next wide sort-by-key macro should delimit sort dispatch");
    let wide_sort = &tuple_impls[macro_start..macro_end];

    assert!(
        wide_sort
            .matches("OrderingControl::from_sorted_indices(&indices)")
            .count()
            == 4
            && wide_sort
                .matches("PermutationPayloadApply::new(control.permutation())")
                .count()
                == 4
            && wide_sort.contains("apply.apply_expr4")
            && wide_sort.contains("apply.apply_expr5")
            && wide_sort.contains("apply.apply_expr6")
            && wide_sort.contains("apply.apply_expr7")
            && !wide_sort.contains("device_expr_gather_with_policy"),
        "wide tuple sort should lower sorted indices to OrderingControl and apply payload through PermutationPayloadApply"
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
    let payload = read("src/detail/api/payload.rs");
    let api_mod = read("src/detail/api/mod.rs");
    let single_impls = read("src/detail/impls/iter/single.rs");
    let tuple_impls = read("src/detail/impls/iter/tuple.rs");

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
        api_mod.contains("IndexedExprApply")
            && !api_mod.contains("device_expr_gather_into_with_policy")
            && !api_mod.contains("device_expr_scatter_into_with_policy"),
        "detail api should expose the indexed expr apply object instead of raw indexed wrapper re-exports"
    );
    assert!(
        single_impls.contains("IndexedExprApply::gather_expr_into")
            && single_impls.contains("IndexedExprApply::gather_expr")
            && single_impls.contains("IndexedExprApply::scatter_expr_into")
            && !single_impls.contains("device_expr_gather_into_with_policy")
            && !single_impls.contains("device_expr_gather_with_policy")
            && !single_impls.contains("device_expr_scatter_into_with_policy"),
        "single-column indexed dispatch should use IndexedExprApply"
    );
    assert!(
        tuple_impls.contains("IndexedExprApply::gather_expr_into")
            && tuple_impls.contains("IndexedExprApply::gather_expr")
            && tuple_impls.contains("IndexedExprApply::scatter_expr_into")
            && !tuple_impls.contains("device_expr_gather_into_with_policy")
            && !tuple_impls.contains("device_expr_gather_with_policy")
            && !tuple_impls.contains("device_expr_scatter_into_with_policy"),
        "tuple indexed dispatch should use IndexedExprApply"
    );
}

#[test]
fn scatter_read_uses_indexed_write_apply() {
    let scatter = read("src/detail/read/scatter.rs");
    let payload = read("src/detail/api/payload.rs");

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
    let payload = read("src/detail/api/payload.rs");
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
        api_mod.contains("MaterializeWriteApply")
            && !api_mod.contains("device_expr_collect_into_with_policy")
            && !api_mod.contains("device_expr_copy_where_into_with_policy"),
        "detail api should expose the typed write apply object instead of collect/copy wrapper re-exports"
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
    let payload = read("src/detail/api/payload.rs");
    let api_mod = read("src/detail/api/mod.rs");
    let single_impls = read("src/detail/impls/iter/single.rs");
    let tuple_impls = read("src/detail/impls/iter/tuple.rs");

    assert!(
        payload.contains("struct FillWriteApply")
            && payload.contains("fn fill_value")
            && payload.contains("primitives::fill_slice_with_policy(policy, value, self.output)"),
        "FillWriteApply should own fill-slice write boundaries"
    );
    assert!(
        payload.contains("struct ConcatPayloadApply")
            && payload.contains("fn apply_values")
            && payload
                .contains("primitives::range::concat_device_with_policy(policy, left, right)"),
        "ConcatPayloadApply should own concat materialization boundaries"
    );
    assert!(
        api_mod.contains("FillWriteApply") && api_mod.contains("ConcatPayloadApply"),
        "detail api should expose fill/concat apply objects"
    );
    assert!(
        single_impls.contains("FillWriteApply::new(&output).fill_value")
            && !single_impls.contains("fill_slice_with_policy"),
        "single-column fill should use FillWriteApply"
    );
    assert!(
        tuple_impls.contains("FillWriteApply::new(&output.$idx)")
            && tuple_impls.contains("ConcatPayloadApply::apply_values")
            && !tuple_impls.contains("fill_slice_with_policy")
            && !tuple_impls.contains("concat_device_with_policy"),
        "tuple fill and wide concat should use typed apply objects"
    );
}

#[test]
fn materialize_payload_paths_use_materialize_payload_apply() {
    let payload = read("src/detail/api/payload.rs");
    let api_mod = read("src/detail/api/mod.rs");
    let memory = read("src/detail/api/memory.rs");
    let gather = read("src/detail/read/gather.rs");
    let scatter = read("src/detail/read/scatter.rs");
    let tuple_impls = read("src/detail/impls/iter/tuple.rs");
    let ordering = read("src/detail/api/ordering/mod.rs");

    assert!(
        payload.contains("struct MaterializePayloadApply")
            && payload.contains("fn collect_expr")
            && payload.contains("expr::device_expr_collect_with_policy(policy, expr)"),
        "MaterializePayloadApply should own expression-to-owned-payload collect boundaries"
    );
    assert!(
        api_mod.contains("MaterializePayloadApply")
            && !api_mod.contains("device_expr_collect_with_policy"),
        "detail api should expose MaterializePayloadApply instead of the raw collect re-export"
    );
    assert!(
        memory.contains("MaterializePayloadApply::collect_expr")
            && gather.contains("MaterializePayloadApply::collect_expr")
            && scatter.contains("MaterializePayloadApply::collect_expr")
            && tuple_impls.contains("MaterializePayloadApply::collect_expr")
            && ordering.contains("MaterializePayloadApply::collect_expr"),
        "owned materialize call sites should use MaterializePayloadApply"
    );

    let allowed = [
        "src/detail/api/payload.rs",
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
    let payload = read("src/detail/api/payload.rs");

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
    let payload = read("src/detail/api/payload.rs");

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
    let scan = read("src/detail/read/by_key/scan.rs");

    assert!(
        scan.contains("struct SegmentedScanApply")
            && scan.contains("control: &'a ScanByKeyControl<R>")
            && scan.contains(
                "inclusive_scan_by_flags_one::<Source, Op>(policy, source, self.control)"
            )
            && scan.contains(
                "exclusive_scan_by_flags_one::<Source, Op>(policy, source, self.control, init)"
            )
            && scan.contains("inclusive_scan_by_flags_seven_views::<R, A, B, C, D, E, F, G, Op>")
            && scan.contains("exclusive_scan_by_flags_seven_views::<R, A, B, C, D, E, F, G, Op>"),
        "SegmentedScanApply should own the segmented scan helper boundary"
    );
    assert!(
        scan.matches("SegmentedScanApply::new(control)").count() >= 14,
        "scan-by-key value arities should apply payload through SegmentedScanApply"
    );
}

#[test]
fn linear_scan_values_use_linear_scan_apply() {
    let scan = read("src/detail/read/scan.rs");
    let apply_start = scan
        .find("struct LinearScanApply")
        .expect("linear scan apply should exist");
    let macro_start = scan[apply_start..]
        .find("macro_rules! impl_kernel_inclusive_scan_tuple1")
        .map(|offset| apply_start + offset)
        .expect("linear scan apply should precede scan impls");
    let apply = &scan[apply_start..macro_start];
    let call_sites = &scan[macro_start..];

    assert!(
        apply.contains("fn inclusive_expr1")
            && apply.contains("fn exclusive_expr1")
            && apply.contains("fn adjacent_expr1")
            && apply.contains("fn inclusive_expr2")
            && apply.contains("fn exclusive_expr2")
            && apply.contains("fn adjacent_expr2")
            && apply.contains("fn inclusive_expr3")
            && apply.contains("fn exclusive_expr3")
            && apply.contains("fn adjacent_expr3"),
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
}

#[test]
fn linear_reduce_values_use_linear_reduce_apply() {
    let reduce = read("src/detail/read/reduce.rs");
    let apply_start = reduce
        .find("struct LinearReduceApply")
        .expect("linear reduce apply should exist");
    let macro_start = reduce[apply_start..]
        .find("macro_rules! impl_kernel_reduce_tuple1")
        .map(|offset| apply_start + offset)
        .expect("linear reduce apply should precede reduce impls");
    let apply = &reduce[apply_start..macro_start];
    let call_sites = &reduce[macro_start..];

    assert!(
        apply.contains("fn apply_expr1")
            && apply.contains("fn apply_expr2")
            && apply.contains("fn apply_expr3"),
        "LinearReduceApply should own linear reduce apply boundaries"
    );
    assert!(
        call_sites.matches("LinearReduceApply::").count() >= 3
            && !call_sites.contains("primitive_reduce::reduce_tuple"),
        "linear reduce read paths should route through LinearReduceApply"
    );
}

#[test]
fn reduce_by_key_values_use_segmented_reduce_apply() {
    let reduce = read("src/detail/read/by_key/reduce.rs");

    assert!(
        reduce.contains("struct SegmentedReduceApply")
            && reduce.contains("control: &'a ReduceByKeyControl<R>")
            && reduce.contains("SegmentedScanApply::new(&scan_control)")
            && reduce.contains("reduce_by_key_tuple7_scanned_values!")
            && reduce.contains("SelectedPayloadApply::new"),
        "SegmentedReduceApply should own segmented scan, init application, and selected output compaction"
    );
    assert!(
        reduce.matches("SegmentedReduceApply::new(control)").count() >= 7
            && reduce.contains("apply.apply_expr::<ValueSource, Op>")
            && reduce.contains("apply.apply_expr2::<ValueA, ValueB, Op>")
            && reduce.contains("apply.apply_expr3::<ValueA, ValueB, ValueC, Op>")
            && reduce.contains("apply.apply_views7::<A, B, C, D, E, F, G, Op>"),
        "reduce-by-key value arities should apply payload through SegmentedReduceApply"
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
    let payload = read("src/detail/api/payload.rs");

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
    let ordering = read("src/detail/api/ordering/mod.rs");
    let apply_start = ordering
        .find("struct MergeExprApply")
        .expect("plain merge apply should exist");
    let set_union_start = ordering[apply_start..]
        .find("fn device_expr_membership_compact_with_policy")
        .map(|offset| apply_start + offset)
        .expect("merge apply should precede membership helpers");
    let apply = &ordering[apply_start..set_union_start];
    let call_sites = &ordering[set_union_start..];

    assert!(
        apply.contains("fn apply_expr")
            && apply.contains("device_expr_merge_with_policy::<Left, Right, Less>"),
        "MergeExprApply should own the plain merge implementation boundary"
    );
    assert!(
        call_sites.matches("MergeExprApply::apply_expr").count() >= 2
            && !call_sites.contains("device_expr_merge_with_policy::<"),
        "plain merge and set_union should route through MergeExprApply"
    );
}

#[test]
fn tuple_set_algorithms_use_selected_payload_apply() {
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
    let payload = read("src/detail/api/payload.rs");

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
    let payload = read("src/detail/api/payload.rs");
    let search = read("src/detail/api/search.rs");
    let selection = read("src/detail/read/selection.rs");
    let tuple_impls = read("src/detail/impls/iter/tuple.rs");

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
        search.matches("SearchControl::from_flags").count() >= 10
            && search.matches("QueryApply::first_flag").count() >= 5
            && search.matches("QueryApply::first_flag_or").count() >= 5
            && search.matches("QueryApply::minmax_expr").count() >= 3
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
        tuple_impls.matches("SearchControl::from_flags").count() >= 5
            && tuple_impls.contains("QueryApply::first_flag")
            && !tuple_impls.contains("primitives::search::first_flag"),
        "wide tuple search fast paths should use SearchControl and QueryApply for readback"
    );
}

#[test]
fn search_many_outputs_use_search_payload_apply() {
    let search = read("src/detail/api/search.rs");
    let apply_start = search
        .find("struct SearchPayloadApply")
        .expect("SearchPayloadApply should exist");
    let stage_start = search[apply_start..]
        .find("fn stage_search_column")
        .map(|offset| apply_start + offset)
        .expect("search payload apply should precede staging helpers");
    let apply = &search[apply_start..stage_start];
    let call_sites = &search[stage_start..];

    assert!(
        apply.contains("fn lower_bound_many_expr")
            && apply.contains("fn upper_bound_many_expr")
            && apply.contains("fn empty_or_zero")
            && apply.contains("fn prepare")
            && apply.contains("fn finish")
            && apply.contains("lower_bound_device_expr_many_kernel::launch_unchecked")
            && apply.contains("upper_bound_device_expr_many_kernel::launch_unchecked"),
        "SearchPayloadApply should own single-column many-bound payload materialization"
    );
    assert!(
        call_sites.contains("SearchPayloadApply::lower_bound_many_expr")
            && call_sites.contains("SearchPayloadApply::upper_bound_many_expr")
            && !call_sites.contains("device_expr_lower_bound_many")
            && !call_sites.contains("device_expr_upper_bound_many"),
        "single-column many-bound call sites should route through SearchPayloadApply"
    );
    assert!(
        call_sites
            .matches("SearchPayloadApply::empty_or_zero")
            .count()
            >= 2
            && call_sites.matches("SearchPayloadApply::prepare").count() >= 2
            && call_sites.matches("SearchPayloadApply::finish").count() >= 2,
        "tuple many-bound paths should share SearchPayloadApply output preparation and finish boundaries"
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
