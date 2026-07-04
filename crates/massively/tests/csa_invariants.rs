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
    let expr_selection = read("src/detail/api/expr/selection.rs");

    assert!(
        read_selection.contains("split_rank_from_selected"),
        "read partition should build SplitRankControl before payload application"
    );
    assert!(
        tuple_impls.contains("split_rank_from_selected"),
        "wide tuple partition should build SplitRankControl before payload application"
    );
    assert!(
        expr_selection.contains("device_expr_compact_split_with_split_with_policy"),
        "partition payload application should fuse selected and rejected split application"
    );
    assert!(
        expr_selection.contains("compact_split_scatter_device_expr_kernel"),
        "partition payload application should launch the fused split scatter kernel"
    );
    assert!(
        expr_selection.contains("control: &select::SplitRankControl"),
        "partition payload helpers should take SplitRankControl explicitly"
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
        tuple_impls.contains("device_expr_compact_with_selection_with_policy"),
        "wide tuple copy/remove where should apply a shared SelectedRankControl to payload columns"
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
