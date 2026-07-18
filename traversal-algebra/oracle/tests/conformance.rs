use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{
    Executor, MStorage,
    graph::{self as gpu_graph, Csr as GpuCsr},
    op::{Identity, ReductionOp, UnaryOp},
    zip2,
};
use proptest::{
    prelude::*,
    test_runner::{Config, TestCaseError, TestCaseResult, TestRunner},
};
use std::cell::RefCell;
use traversal_algebra_oracle::{
    CASES, CubeClCost, CubeClMachine, DestinationStrategy, LeanOracle, OracleCase,
    graph::{self as oracle_graph, Csr as OracleCsr, EdgeContext, Observation, op as oracle_op},
};

const DEFAULT_PROPERTY_CASES: u32 = 256;
const DEFAULT_SEMANTIC_CASES: u32 = 32;
const DEFAULT_SCALE_VERTICES: usize = 1_025;

#[derive(Clone, Debug)]
struct GraphCase {
    offsets: Vec<u32>,
    destinations: Vec<u32>,
    frontier: Vec<u32>,
    vertex_values: Vec<u32>,
    edge_values: Vec<u32>,
    generation: String,
}

struct One;

#[cubecl::cube]
impl UnaryOp<u32> for One {
    type Output = u32;

    fn apply(_input: u32) -> u32 {
        1u32
    }
}

struct AddPair;

#[cubecl::cube]
impl UnaryOp<(u32, u32)> for AddPair {
    type Output = u32;

    fn apply(input: (u32, u32)) -> u32 {
        input.0 + input.1
    }
}

struct Add;

#[cubecl::cube]
impl ReductionOp<u32> for Add {
    fn apply(left: u32, right: u32) -> u32 {
        left + right
    }
}

fn oracle_csr(case: &GraphCase) -> OracleCsr {
    OracleCsr::new(case.destinations.clone(), case.offsets.clone())
}

fn property_cases() -> u32 {
    environment_u32("TRAVERSAL_ALGEBRA_PROPTEST_CASES", DEFAULT_PROPERTY_CASES)
}

fn semantic_cases() -> u32 {
    environment_u32("TRAVERSAL_ALGEBRA_SEMANTIC_CASES", DEFAULT_SEMANTIC_CASES)
}

fn environment_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn active_edge_count(case: &GraphCase) -> u64 {
    case.frontier
        .iter()
        .map(|&source| {
            let source = source as usize;
            u64::from(case.offsets[source + 1] - case.offsets[source])
        })
        .sum()
}

fn reduction_depth(items: u64) -> u64 {
    if items == 0 {
        0
    } else {
        u64::from(u64::BITS - items.leading_zeros())
    }
}

fn certificate_machine(case: &GraphCase) -> CubeClMachine {
    let selector = case.offsets.len() + case.destinations.len() + case.frontier.len();
    let workgroup_size = [32, 64, 96, 128, 256][selector % 5];
    let subgroup_size = [8, 16, 32][selector % 3];
    CubeClMachine {
        workgroup_size,
        subgroup_size,
    }
}

fn assert_machine_geometry(cost: CubeClCost, machine: CubeClMachine) -> TestCaseResult {
    let workgroup_size = u64::from(machine.workgroup_size);
    let subgroup_size = u64::from(machine.subgroup_size);
    let subgroups_per_workgroup = workgroup_size.div_ceil(subgroup_size);
    prop_assert!(cost.scheduled_threads >= cost.logical_threads);
    prop_assert_eq!(cost.scheduled_threads % workgroup_size, 0);
    prop_assert_eq!(
        cost.scheduled_subgroups,
        (cost.scheduled_threads / workgroup_size) * subgroups_per_workgroup
    );
    Ok(())
}

fn assert_sequential_cost(
    total: CubeClCost,
    prefix: CubeClCost,
    suffix: CubeClCost,
) -> TestCaseResult {
    prop_assert_eq!(
        total.logical_threads,
        prefix.logical_threads + suffix.logical_threads
    );
    prop_assert_eq!(
        total.scheduled_threads,
        prefix.scheduled_threads + suffix.scheduled_threads
    );
    prop_assert_eq!(
        total.scheduled_subgroups,
        prefix.scheduled_subgroups + suffix.scheduled_subgroups
    );
    prop_assert_eq!(total.scalar_work, prefix.scalar_work + suffix.scalar_work);
    prop_assert_eq!(total.span, prefix.span + suffix.span);
    prop_assert_eq!(
        total.global_loads,
        prefix.global_loads + suffix.global_loads
    );
    prop_assert_eq!(
        total.global_stores,
        prefix.global_stores + suffix.global_stores
    );
    prop_assert_eq!(
        total.host_read_words,
        prefix.host_read_words + suffix.host_read_words
    );
    prop_assert_eq!(
        total.atomic_operations,
        prefix.atomic_operations + suffix.atomic_operations
    );
    prop_assert_eq!(total.barriers, prefix.barriers + suffix.barriers);
    prop_assert_eq!(total.launches, prefix.launches + suffix.launches);
    prop_assert_eq!(
        total.allocated_words,
        prefix.allocated_words + suffix.allocated_words
    );
    prop_assert_eq!(
        total.materializations,
        prefix.materializations + suffix.materializations
    );
    Ok(())
}

fn small_graph_case() -> impl Strategy<Value = GraphCase> {
    (0usize..=12).prop_flat_map(|vertices| {
        if vertices == 0 {
            return Just(case_from_rows(Vec::new(), Vec::new(), 0)).boxed();
        }
        (
            prop::collection::vec(
                prop::collection::vec(0u32..vertices as u32, 0..=12),
                vertices,
            ),
            prop::collection::vec(0u32..vertices as u32, 0..=48),
            any::<u64>(),
        )
            .prop_map(|(rows, frontier, seed)| case_from_rows(rows, frontier, seed))
            .boxed()
    })
}

fn structured_graph_case() -> impl Strategy<Value = GraphCase> {
    const SIZES: &[usize] = &[0, 1, 2, 7, 15, 31, 32, 33, 63, 64, 65, 127, 255, 256, 257];
    (prop::sample::select(SIZES), 0u8..5, 0u8..5, any::<u64>()).prop_map(
        |(vertices, family, frontier_kind, seed)| {
            structured_case(vertices, family, frontier_kind, seed)
        },
    )
}

fn graph_case() -> impl Strategy<Value = GraphCase> {
    prop_oneof![3 => small_graph_case(), 2 => structured_graph_case()]
}

fn case_from_rows(rows: Vec<Vec<u32>>, frontier: Vec<u32>, seed: u64) -> GraphCase {
    let mut offsets = Vec::with_capacity(rows.len() + 1);
    let mut destinations = Vec::new();
    offsets.push(0);
    for row in rows {
        destinations.extend(row);
        offsets.push(destinations.len() as u32);
    }
    let vertex_values = (0..offsets.len() - 1)
        .map(|vertex| bounded_value(seed, vertex as u64, 97))
        .collect();
    let edge_values = (0..destinations.len())
        .map(|edge| bounded_value(seed.rotate_left(17), edge as u64, 53))
        .collect();
    GraphCase {
        offsets,
        destinations,
        frontier,
        vertex_values,
        edge_values,
        generation: format!("rows(seed={seed:#018x})"),
    }
}

fn structured_case(vertices: usize, family: u8, frontier_kind: u8, seed: u64) -> GraphCase {
    if vertices == 0 {
        return case_from_rows(Vec::new(), Vec::new(), seed);
    }
    let mut rows = vec![Vec::new(); vertices];
    let mut random = SplitMix64(seed);
    match family {
        // Directed multigraph with variable row lengths and deliberately
        // unsorted rows.
        0 => {
            for (source, row) in rows.iter_mut().enumerate() {
                let degree = (random.next() as usize % 17).min(vertices.saturating_mul(2));
                for _ in 0..degree {
                    row.push((random.next() as usize % vertices) as u32);
                }
                if source % 11 == 0 {
                    row.reverse();
                }
            }
        }
        // A high-degree hub plus many isolates.
        1 => {
            for target in 0..vertices {
                rows[0].push(target as u32);
                if target % 3 == 0 {
                    rows[0].push(target as u32);
                }
            }
            for source in (1..vertices).step_by(7) {
                rows[source].push(0);
            }
        }
        // Ordered bipartite rows.
        2 => {
            let split = vertices / 2;
            for (source, row) in rows.iter_mut().enumerate().take(split) {
                for step in 0..8.min(vertices - split) {
                    row.push((split + (source * 7 + step * 3) % (vertices - split)) as u32);
                }
                row.sort_unstable();
            }
        }
        // Boundary-crossing regular rows, including self-loops.
        3 => {
            for (source, row) in rows.iter_mut().enumerate() {
                for step in 0..9.min(vertices) {
                    row.push((source + step * 31) as u32 % vertices as u32);
                }
            }
        }
        // Parallel edges and repeated self-loops with alternating empty rows.
        _ => {
            for (source, row) in rows.iter_mut().enumerate().step_by(2) {
                let target = (source * 13 + 1) % vertices;
                row.extend([source as u32, source as u32, target as u32, target as u32]);
            }
        }
    }

    let frontier = match frontier_kind {
        0 => Vec::new(),
        1 => (0..vertices as u32).collect(),
        2 => (0..vertices as u32).rev().collect(),
        3 => (0..vertices.min(128))
            .flat_map(|vertex| [vertex as u32, vertex as u32])
            .collect(),
        _ => {
            let length = (random.next() as usize % (vertices.saturating_mul(3) + 1)).min(1_024);
            (0..length)
                .map(|_| (random.next() as usize % vertices) as u32)
                .collect()
        }
    };
    let mut case = case_from_rows(rows, frontier, seed);
    case.generation = format!(
        "structured(vertices={vertices}, family={family}, frontier={frontier_kind}, seed={seed:#018x})"
    );
    case
}

fn scale_case(vertices: usize) -> GraphCase {
    assert!(vertices > 0);
    let mut rows = vec![Vec::with_capacity(6); vertices];
    for (source, row) in rows.iter_mut().enumerate() {
        row.extend([
            source as u32,
            ((source + 1) % vertices) as u32,
            ((source + vertices - 1) % vertices) as u32,
            ((source * 17 + 11) % vertices) as u32,
            ((source * 31 + 7) % vertices) as u32,
            ((source * 43 + 3) % vertices) as u32,
        ]);
    }
    let frontier = (0..vertices as u32).collect();
    let seed = 0x5CA1_E123_9ABC_DEF0;
    let mut case = case_from_rows(rows, frontier, seed);
    case.generation = format!("scale(vertices={vertices}, seed={seed:#018x})");
    case
}

fn bounded_value(seed: u64, index: u64, modulus: u32) -> u32 {
    let mut random = SplitMix64(seed ^ index.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    (random.next() % modulus as u64) as u32
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }
}

fn generated_case(case: &OracleCase) -> GraphCase {
    let rows = case
        .offsets
        .windows(2)
        .map(|range| case.destinations[range[0] as usize..range[1] as usize].to_vec())
        .collect();
    let mut generated = case_from_rows(rows, case.frontier.to_vec(), 0xC0DE_CAFE);
    generated.generation = format!("fixture({})", case.name);
    generated
}

fn oracle_failure(error: impl std::fmt::Display) -> TestCaseError {
    TestCaseError::fail(format!("Lean oracle failed: {error}"))
}

fn gpu_failure(context: &str, error: impl std::fmt::Debug) -> TestCaseError {
    TestCaseError::fail(format!("{context} failed: {error:?}"))
}

fn expected_contexts(
    oracle: &mut LeanOracle,
    case: &GraphCase,
) -> Result<Vec<EdgeContext>, TestCaseError> {
    let query = oracle_graph::traverse(oracle_csr(case), case.frontier.clone())
        .map_err(oracle_failure)?
        .map(
            oracle_graph::zip2(
                oracle_graph::zip2(oracle_graph::source_id(), oracle_graph::destination_id()),
                oracle_graph::edge_id(),
            ),
            oracle_op::Identity,
        )
        .emit();
    let values = match oracle.observe(query).map_err(oracle_failure)? {
        Observation::Emitted(values) => values,
        observation => {
            return Err(TestCaseError::fail(format!(
                "context query returned wrong observation shape: {observation:?}"
            )));
        }
    };
    Ok(values
        .into_iter()
        .map(|(source, destination, edge)| EdgeContext {
            source,
            destination,
            edge,
        })
        .collect())
}

fn compare_core(
    oracle: &mut LeanOracle,
    exec: &Executor<WgpuRuntime>,
    case: &GraphCase,
) -> TestCaseResult {
    let _generation = &case.generation;
    let expected_edges = expected_contexts(oracle, case)?;
    let expected_source_counts = oracle
        .evaluate(
            oracle_graph::traverse(oracle_csr(case), case.frontier.clone())
                .map_err(oracle_failure)?
                .map(oracle_graph::edge_id(), oracle_op::One)
                .reduce_by_source(0, oracle_op::Add)
                .map_err(oracle_failure)?,
        )
        .map_err(oracle_failure)?;
    let expected_destination_counts = oracle
        .evaluate(
            oracle_graph::traverse(oracle_csr(case), case.frontier.clone())
                .map_err(oracle_failure)?
                .map(oracle_graph::edge_id(), oracle_op::One)
                .reduce_by_destination(0, oracle_op::Add)
                .map_err(oracle_failure)?,
        )
        .map_err(oracle_failure)?;

    let offsets = exec.to_device(&case.offsets);
    let destinations = exec.to_device(&case.destinations);
    let frontier = exec.to_device(&case.frontier);
    let csr = || GpuCsr::new(destinations.slice(..), offsets.slice(..));

    let traversal = gpu_graph::traverse(exec, csr(), frontier.slice(..))
        .map_err(|error| gpu_failure("traverse", error))?;
    prop_assert_eq!(traversal.edge_count() as usize, expected_edges.len());
    let emitted = traversal
        .map(
            zip2(
                zip2(gpu_graph::source_id(), gpu_graph::destination_id()),
                gpu_graph::edge_id(),
            ),
            Identity,
        )
        .emit(exec)
        .map_err(|error| gpu_failure("context emit", error))?;
    let (sources, emitted_destinations, edges) = MStorage::into_columns(emitted);
    let actual_edges = exec
        .to_host(&sources)
        .map_err(|error| gpu_failure("source readback", error))?
        .into_iter()
        .zip(
            exec.to_host(&emitted_destinations)
                .map_err(|error| gpu_failure("destination readback", error))?,
        )
        .zip(
            exec.to_host(&edges)
                .map_err(|error| gpu_failure("edge readback", error))?,
        )
        .map(|((source, destination), edge)| EdgeContext {
            source,
            destination,
            edge,
        })
        .collect::<Vec<_>>();
    prop_assert_eq!(actual_edges, expected_edges);

    let source_counts = gpu_graph::traverse(exec, csr(), frontier.slice(..))
        .map_err(|error| gpu_failure("source traverse", error))?
        .map(gpu_graph::edge_id(), One)
        .reduce_by_source(exec, 0, Add)
        .map_err(|error| gpu_failure("source reduction", error))?;
    prop_assert_eq!(
        exec.to_host(&source_counts)
            .map_err(|error| gpu_failure("source-count readback", error))?,
        expected_source_counts
    );

    let destination_counts = gpu_graph::traverse(exec, csr(), frontier.slice(..))
        .map_err(|error| gpu_failure("destination traverse", error))?
        .map(gpu_graph::edge_id(), One)
        .reduce_by_destination(exec, 0, Add)
        .map_err(|error| gpu_failure("destination reduction", error))?;
    prop_assert_eq!(
        exec.to_host(&destination_counts)
            .map_err(|error| gpu_failure("destination-count readback", error))?,
        expected_destination_counts
    );
    Ok(())
}

fn compare_fine_semantics(
    oracle: &mut LeanOracle,
    exec: &Executor<WgpuRuntime>,
    case: &GraphCase,
) -> TestCaseResult {
    let _generation = &case.generation;
    let expected_source = oracle
        .evaluate(
            oracle_graph::traverse(oracle_csr(case), case.frontier.clone())
                .map_err(oracle_failure)?
                .map(
                    oracle_graph::source(case.vertex_values.clone()),
                    oracle_op::Identity,
                )
                .emit(),
        )
        .map_err(oracle_failure)?;
    let expected_destination = oracle
        .evaluate(
            oracle_graph::traverse(oracle_csr(case), case.frontier.clone())
                .map_err(oracle_failure)?
                .map(
                    oracle_graph::destination(case.vertex_values.clone()),
                    oracle_op::Identity,
                )
                .emit(),
        )
        .map_err(oracle_failure)?;
    let expected_edge = oracle
        .evaluate(
            oracle_graph::traverse(oracle_csr(case), case.frontier.clone())
                .map_err(oracle_failure)?
                .map(
                    oracle_graph::edge(case.edge_values.clone()),
                    oracle_op::Identity,
                )
                .emit(),
        )
        .map_err(oracle_failure)?;
    let expected_pair = oracle
        .evaluate(
            oracle_graph::traverse(oracle_csr(case), case.frontier.clone())
                .map_err(oracle_failure)?
                .map(
                    oracle_graph::zip2(
                        oracle_graph::source(case.vertex_values.clone()),
                        oracle_graph::edge(case.edge_values.clone()),
                    ),
                    oracle_op::Identity,
                )
                .emit(),
        )
        .map_err(oracle_failure)?;

    let oracle_mapped = || {
        oracle_graph::traverse(oracle_csr(case), case.frontier.clone())
            .map_err(oracle_failure)
            .map(|traversal| {
                traversal.map(
                    oracle_graph::zip2(
                        oracle_graph::source(case.vertex_values.clone()),
                        oracle_graph::edge(case.edge_values.clone()),
                    ),
                    oracle_op::AddPair,
                )
            })
    };
    let expected_mapped = oracle
        .evaluate(oracle_mapped()?.emit())
        .map_err(oracle_failure)?;
    let expected_source_reduced = oracle
        .evaluate(
            oracle_mapped()?
                .reduce_by_source(0, oracle_op::Add)
                .map_err(oracle_failure)?,
        )
        .map_err(oracle_failure)?;
    let expected_destination_reduced = oracle
        .evaluate(
            oracle_mapped()?
                .reduce_by_destination(0, oracle_op::Add)
                .map_err(oracle_failure)?,
        )
        .map_err(oracle_failure)?;

    let offsets = exec.to_device(&case.offsets);
    let destinations = exec.to_device(&case.destinations);
    let frontier = exec.to_device(&case.frontier);
    let vertex_values = exec.to_device(&case.vertex_values);
    let edge_values = exec.to_device(&case.edge_values);
    let csr = || GpuCsr::new(destinations.slice(..), offsets.slice(..));
    let traversal = || {
        gpu_graph::traverse(exec, csr(), frontier.slice(..))
            .map_err(|error| gpu_failure("semantic traversal", error))
    };

    let actual_source = traversal()?
        .map(gpu_graph::source(vertex_values.slice(..)), Identity)
        .emit(exec)
        .map_err(|error| gpu_failure("source-value emit", error))?;
    prop_assert_eq!(
        exec.to_host(&actual_source)
            .map_err(|error| gpu_failure("source-value readback", error))?,
        expected_source
    );

    let actual_destination = traversal()?
        .map(gpu_graph::destination(vertex_values.slice(..)), Identity)
        .emit(exec)
        .map_err(|error| gpu_failure("destination-value emit", error))?;
    prop_assert_eq!(
        exec.to_host(&actual_destination)
            .map_err(|error| gpu_failure("destination-value readback", error))?,
        expected_destination
    );

    let actual_edge = traversal()?
        .map(gpu_graph::edge(edge_values.slice(..)), Identity)
        .emit(exec)
        .map_err(|error| gpu_failure("edge-value emit", error))?;
    prop_assert_eq!(
        exec.to_host(&actual_edge)
            .map_err(|error| gpu_failure("edge-value readback", error))?,
        expected_edge
    );

    let actual_pair = traversal()?
        .map(
            zip2(
                gpu_graph::source(vertex_values.slice(..)),
                gpu_graph::edge(edge_values.slice(..)),
            ),
            Identity,
        )
        .emit(exec)
        .map_err(|error| gpu_failure("product emit", error))?;
    let (actual_pair_left, actual_pair_right) = MStorage::into_columns(actual_pair);
    let actual_pair = exec
        .to_host(&actual_pair_left)
        .map_err(|error| gpu_failure("product-left readback", error))?
        .into_iter()
        .zip(
            exec.to_host(&actual_pair_right)
                .map_err(|error| gpu_failure("product-right readback", error))?,
        )
        .collect::<Vec<_>>();
    prop_assert_eq!(actual_pair, expected_pair);

    let mapped = || {
        traversal().and_then(|traversal| {
            Ok(traversal.map(
                zip2(
                    gpu_graph::source(vertex_values.slice(..)),
                    gpu_graph::edge(edge_values.slice(..)),
                ),
                AddPair,
            ))
        })
    };
    let actual_mapped = mapped()?
        .emit(exec)
        .map_err(|error| gpu_failure("mapped emit", error))?;
    prop_assert_eq!(
        exec.to_host(&actual_mapped)
            .map_err(|error| gpu_failure("mapped readback", error))?,
        expected_mapped
    );

    let actual_source_reduced = mapped()?
        .reduce_by_source(exec, 0, Add)
        .map_err(|error| gpu_failure("mapped source reduction", error))?;
    prop_assert_eq!(
        exec.to_host(&actual_source_reduced)
            .map_err(|error| gpu_failure("mapped source reduction readback", error))?,
        expected_source_reduced
    );

    let actual_destination_reduced = mapped()?
        .reduce_by_destination(exec, 0, Add)
        .map_err(|error| gpu_failure("mapped destination reduction", error))?;
    prop_assert_eq!(
        exec.to_host(&actual_destination_reduced)
            .map_err(|error| gpu_failure("mapped destination reduction readback", error))?,
        expected_destination_reduced
    );
    Ok(())
}

#[test]
fn generated_regressions_equal_the_typed_lean_oracle() {
    let mut oracle = LeanOracle::start().unwrap();
    for fixture in CASES {
        let case = generated_case(fixture);
        let actual_edges = expected_contexts(&mut oracle, &case).unwrap();
        assert_eq!(
            actual_edges, fixture.expected_edges,
            "fixture {}",
            fixture.name
        );

        let actual_source = match oracle
            .observe(
                oracle_graph::traverse(oracle_csr(&case), case.frontier.clone())
                    .unwrap()
                    .map(oracle_graph::edge_id(), oracle_op::One)
                    .reduce_by_source(0, oracle_op::Add)
                    .unwrap(),
            )
            .unwrap()
        {
            Observation::SourceReduced(values) => values,
            observation => panic!("wrong source observation shape: {observation:?}"),
        };
        assert_eq!(
            actual_source, fixture.expected_source_counts,
            "fixture {}",
            fixture.name
        );

        let actual_destination = match oracle
            .observe(
                oracle_graph::traverse(oracle_csr(&case), case.frontier.clone())
                    .unwrap()
                    .map(oracle_graph::edge_id(), oracle_op::One)
                    .reduce_by_destination(0, oracle_op::Add)
                    .unwrap(),
            )
            .unwrap()
        {
            Observation::DestinationReduced(values) => values,
            observation => panic!("wrong destination observation shape: {observation:?}"),
        };
        assert_eq!(
            actual_destination, fixture.expected_destination_counts,
            "fixture {}",
            fixture.name
        );
    }
}

#[test]
fn proptest_massively_against_typed_lean_oracle() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let oracle = RefCell::new(LeanOracle::start().unwrap());
    let mut runner = TestRunner::new(Config {
        cases: property_cases(),
        ..Config::default()
    });
    runner
        .run(&graph_case(), |case| {
            compare_core(&mut oracle.borrow_mut(), &exec, &case)
        })
        .unwrap();
}

#[test]
fn proptest_edge_expression_and_terminal_semantics() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let oracle = RefCell::new(LeanOracle::start().unwrap());
    let mut runner = TestRunner::new(Config {
        cases: semantic_cases(),
        ..Config::default()
    });
    runner
        .run(&graph_case(), |case| {
            compare_fine_semantics(&mut oracle.borrow_mut(), &exec, &case)
        })
        .unwrap();
}

fn compare_cubecl_certificate(oracle: &mut LeanOracle, case: &GraphCase) -> TestCaseResult {
    let graph = oracle_csr(case);
    let machine = certificate_machine(case);
    let emit = oracle_graph::traverse(graph.clone(), case.frontier.clone())
        .unwrap()
        .map(oracle_graph::source_id(), oracle_op::Identity)
        .emit();
    let emit_cost = oracle
        .cubecl_certificate(&emit, machine, DestinationStrategy::SortReduce)
        .unwrap();
    let active_edges = active_edge_count(case);
    prop_assert_eq!(emit_cost.vertices, (case.offsets.len() - 1) as u64);
    prop_assert_eq!(emit_cost.topology_edges, case.destinations.len() as u64);
    prop_assert_eq!(emit_cost.frontier_occurrences, case.frontier.len() as u64);
    prop_assert_eq!(emit_cost.active_edges, active_edges);
    prop_assert_eq!(emit_cost.output_words, 1);
    prop_assert_eq!(
        emit_cost.fused.scalar_work,
        active_edges * (emit_cost.expression_work + emit_cost.output_words)
    );
    prop_assert_eq!(
        emit_cost.fused.global_loads,
        active_edges * emit_cost.global_load_words_per_edge
    );
    prop_assert_eq!(emit_cost.fused.global_stores, active_edges);
    prop_assert_eq!(emit_cost.fused.materializations, 0);
    let expected_control_work = emit_cost.topology_edges
        + emit_cost.vertices
        + 6 * emit_cost.frontier_occurrences
        + 11 * active_edges
        + 2
        + u64::from(!case.frontier.is_empty());
    prop_assert_eq!(
        emit_cost.materialized_csr_control.scalar_work,
        expected_control_work
    );
    prop_assert_eq!(
        emit_cost.materialized_csr_control.host_read_words,
        u64::from(!case.frontier.is_empty())
    );
    assert_sequential_cost(
        emit_cost.with_materialized_csr_control,
        emit_cost.materialized_csr_control,
        emit_cost.fused,
    )?;
    assert_machine_geometry(emit_cost.fused, machine)?;
    assert_machine_geometry(emit_cost.materialized_csr_control, machine)?;
    assert_machine_geometry(emit_cost.with_materialized_csr_control, machine)?;

    let source = oracle_graph::traverse(graph.clone(), case.frontier.clone())
        .unwrap()
        .map(oracle_graph::source_id(), oracle_op::Identity)
        .reduce_by_source(0, oracle_op::Add)
        .unwrap();
    let source_cost = oracle
        .cubecl_certificate(&source, machine, DestinationStrategy::SortReduce)
        .unwrap();
    prop_assert_eq!(
        source_cost.fused.scalar_work,
        source_cost.frontier_occurrences + active_edges * (source_cost.expression_work + 1)
    );
    assert_sequential_cost(
        source_cost.with_materialized_csr_control,
        source_cost.materialized_csr_control,
        source_cost.fused,
    )?;

    let destination = oracle_graph::traverse(graph, case.frontier.clone())
        .unwrap()
        .map(oracle_graph::source_id(), oracle_op::Identity)
        .reduce_by_destination(0, oracle_op::Add)
        .unwrap();
    let atomic = oracle
        .cubecl_certificate(&destination, machine, DestinationStrategy::Atomic)
        .unwrap();
    prop_assert_eq!(
        atomic.fused.scalar_work,
        atomic.vertices + active_edges * (atomic.expression_work + 1)
    );
    prop_assert_eq!(atomic.fused.atomic_operations, active_edges);
    prop_assert_eq!(atomic.fused.materializations, 0);

    let sort = oracle
        .cubecl_certificate(&destination, machine, DestinationStrategy::SortReduce)
        .unwrap();
    prop_assert_eq!(
        sort.fused.scalar_work,
        sort.vertices
            + active_edges * sort.expression_work
            + active_edges * reduction_depth(active_edges)
            + active_edges
    );
    assert_sequential_cost(
        sort.with_materialized_csr_control,
        sort.materialized_csr_control,
        sort.fused,
    )?;
    Ok(())
}

#[test]
fn proptest_cubecl_resource_certificates() {
    let oracle = RefCell::new(LeanOracle::start().unwrap());
    let mut runner = TestRunner::new(Config {
        cases: semantic_cases(),
        ..Config::default()
    });
    runner
        .run(&graph_case(), |case| {
            compare_cubecl_certificate(&mut oracle.borrow_mut(), &case)
        })
        .unwrap();
}

#[test]
fn scale_graph_crosses_previous_oracle_limits() {
    let vertices = std::env::var("TRAVERSAL_ALGEBRA_SCALE_VERTICES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_SCALE_VERTICES);
    assert!(
        vertices > 33,
        "scale graph must exceed the former 33-vertex cap"
    );
    let case = scale_case(vertices);
    assert!(case.destinations.len() > 8 * 33);

    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let mut oracle = LeanOracle::start().unwrap();
    compare_core(&mut oracle, &exec, &case).unwrap();
    compare_fine_semantics(&mut oracle, &exec, &case).unwrap();
}
