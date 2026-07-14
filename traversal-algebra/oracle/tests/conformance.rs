use std::{
    fmt::Write as _,
    path::PathBuf,
    process::{Command, Output},
};

use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{
    Executor,
    graph::{self, Csr},
    op::{Identity, ReductionOp, UnaryOp},
    unzip3, zip3,
};
use proptest::{
    prelude::*,
    test_runner::{Config, TestCaseError, TestCaseResult, TestRunner},
};
use traversal_algebra_oracle::{CASES, EdgeContext, OracleCase};

const DEFAULT_PROPERTY_CASES: u32 = 256;
const MAX_ROW_LENGTH: usize = 8;
const MAX_FRONTIER_LENGTH: usize = 40;

#[derive(Clone, Debug)]
struct GraphCase {
    offsets: Vec<u32>,
    destinations: Vec<u32>,
    frontier: Vec<u32>,
}

#[derive(Debug, PartialEq, Eq)]
struct Expected {
    edges: Vec<EdgeContext>,
    source_counts: Vec<u32>,
    destination_counts: Vec<u32>,
}

struct CountEdge;

#[cubecl::cube]
impl UnaryOp<u32> for CountEdge {
    type Output = u32;

    fn apply(_input: u32) -> u32 {
        1u32
    }
}

struct Add;

#[cubecl::cube]
impl ReductionOp<u32> for Add {
    fn apply(left: u32, right: u32) -> u32 {
        left + right
    }
}

fn vertex_count() -> impl Strategy<Value = usize> {
    prop_oneof![
        8 => 0usize..=8,
        2 => prop::sample::select(vec![15usize, 16, 17, 31, 32, 33]),
    ]
}

fn graph_case() -> impl Strategy<Value = GraphCase> {
    vertex_count().prop_flat_map(|vertices| {
        if vertices == 0 {
            return Just(GraphCase {
                offsets: vec![0],
                destinations: Vec::new(),
                frontier: Vec::new(),
            })
            .boxed();
        }

        (
            prop::collection::vec(
                prop::collection::vec(0u32..vertices as u32, 0..=MAX_ROW_LENGTH),
                vertices,
            ),
            prop::collection::vec(0u32..vertices as u32, 0..=MAX_FRONTIER_LENGTH),
        )
            .prop_map(|(rows, frontier)| {
                let mut offsets = Vec::with_capacity(rows.len() + 1);
                let mut destinations = Vec::new();
                offsets.push(0);
                for row in rows {
                    destinations.extend(row);
                    offsets.push(destinations.len() as u32);
                }
                GraphCase {
                    offsets,
                    destinations,
                    frontier,
                }
            })
            .boxed()
    })
}

fn property_cases() -> u32 {
    std::env::var("TRAVERSAL_ALGEBRA_PROPTEST_CASES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_PROPERTY_CASES)
}

fn lean_oracle_path() -> PathBuf {
    std::env::var_os("TRAVERSAL_ALGEBRA_LEAN_ORACLE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../proof/.lake/build/bin/oracle")
        })
}

fn csv(values: &[u32]) -> String {
    let mut output = String::new();
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            output.push(',');
        }
        write!(output, "{value}").unwrap();
    }
    output
}

fn run_lean(case: &GraphCase) -> Result<Expected, String> {
    let path = lean_oracle_path();
    let Output {
        status,
        stdout,
        stderr,
    } = Command::new(&path)
        .arg(csv(&case.offsets))
        .arg(csv(&case.destinations))
        .arg(csv(&case.frontier))
        .output()
        .map_err(|error| {
            format!(
                "failed to run Lean oracle at {}: {error}; run `just ta::proof` first",
                path.display()
            )
        })?;

    if !status.success() {
        return Err(format!(
            "Lean oracle rejected {case:?}: {}",
            String::from_utf8_lossy(&stderr).trim()
        ));
    }

    parse_expected(
        std::str::from_utf8(&stdout)
            .map_err(|error| format!("Lean oracle returned non-UTF-8 output: {error}"))?
            .trim(),
    )
}

fn parse_expected(output: &str) -> Result<Expected, String> {
    let fields = output.split('|').collect::<Vec<_>>();
    if fields.len() != 3 {
        return Err(format!("invalid Lean oracle output: {output:?}"));
    }
    Ok(Expected {
        edges: parse_edges(fields[0])?,
        source_counts: parse_u32s(fields[1])?,
        destination_counts: parse_u32s(fields[2])?,
    })
}

fn parse_u32s(input: &str) -> Result<Vec<u32>, String> {
    if input.is_empty() {
        return Ok(Vec::new());
    }
    input
        .split(',')
        .map(|value| {
            value
                .parse()
                .map_err(|error| format!("invalid Lean natural {value:?}: {error}"))
        })
        .collect()
}

fn parse_edges(input: &str) -> Result<Vec<EdgeContext>, String> {
    if input.is_empty() {
        return Ok(Vec::new());
    }
    input
        .split(';')
        .map(|edge| {
            let components = parse_u32s(edge)?;
            let [source, destination, edge] = components.as_slice() else {
                return Err(format!("invalid Lean edge context: {components:?}"));
            };
            Ok(EdgeContext {
                source: *source,
                destination: *destination,
                edge: *edge,
            })
        })
        .collect()
}

fn generated_case(case: &OracleCase) -> GraphCase {
    GraphCase {
        offsets: case.offsets.to_vec(),
        destinations: case.destinations.to_vec(),
        frontier: case.frontier.to_vec(),
    }
}

fn generated_expected(case: &OracleCase) -> Expected {
    Expected {
        edges: case.expected_edges.to_vec(),
        source_counts: case.expected_source_counts.to_vec(),
        destination_counts: case.expected_destination_counts.to_vec(),
    }
}

fn compare_with_massively(
    exec: &Executor<WgpuRuntime>,
    case: &GraphCase,
    expected: &Expected,
) -> TestCaseResult {
    let offsets = exec.to_device(&case.offsets);
    let destinations = exec.to_device(&case.destinations);
    let frontier = exec.to_device(&case.frontier);
    let csr = || Csr::new(destinations.slice(..), offsets.slice(..));

    let traversal = graph::traverse(exec, csr(), frontier.slice(..))
        .map_err(|error| TestCaseError::fail(format!("traverse failed: {error:?}")))?;
    prop_assert_eq!(traversal.edge_count() as usize, expected.edges.len());
    let emitted = traversal
        .map(
            zip3(
                graph::source_id(),
                graph::destination_id(),
                graph::edge_id(),
            ),
            Identity,
        )
        .emit(exec)
        .map_err(|error| TestCaseError::fail(format!("emit failed: {error:?}")))?;
    let (sources, emitted_destinations, edge_ids) = unzip3(emitted);
    let actual_edges = exec
        .to_host(&sources)
        .map_err(|error| TestCaseError::fail(format!("source readback failed: {error:?}")))?
        .into_iter()
        .zip(exec.to_host(&emitted_destinations).map_err(|error| {
            TestCaseError::fail(format!("destination readback failed: {error:?}"))
        })?)
        .zip(
            exec.to_host(&edge_ids)
                .map_err(|error| TestCaseError::fail(format!("edge readback failed: {error:?}")))?,
        )
        .map(|((source, destination), edge)| EdgeContext {
            source,
            destination,
            edge,
        })
        .collect::<Vec<_>>();
    prop_assert_eq!(&actual_edges, &expected.edges);

    let source_counts = graph::traverse(exec, csr(), frontier.slice(..))
        .map_err(|error| TestCaseError::fail(format!("source traverse failed: {error:?}")))?
        .map(graph::edge_id(), CountEdge)
        .reduce_by_source(exec, 0, Add)
        .map_err(|error| TestCaseError::fail(format!("source reduction failed: {error:?}")))?;
    let actual_source_counts = exec
        .to_host(&source_counts)
        .map_err(|error| TestCaseError::fail(format!("source readback failed: {error:?}")))?;
    prop_assert_eq!(&actual_source_counts, &expected.source_counts);

    let destination_counts = graph::traverse(exec, csr(), frontier.slice(..))
        .map_err(|error| TestCaseError::fail(format!("destination traverse failed: {error:?}")))?
        .map(graph::edge_id(), CountEdge)
        .reduce_by_destination(exec, 0, Add)
        .map_err(|error| TestCaseError::fail(format!("destination reduction failed: {error:?}")))?;
    let actual_destination_counts = exec
        .to_host(&destination_counts)
        .map_err(|error| TestCaseError::fail(format!("destination readback failed: {error:?}")))?;
    prop_assert_eq!(&actual_destination_counts, &expected.destination_counts);

    Ok(())
}

#[test]
fn generated_regressions_equal_the_compiled_lean_oracle() {
    for case in CASES {
        let actual = run_lean(&generated_case(case)).unwrap();
        assert_eq!(actual, generated_expected(case), "fixture {}", case.name);
    }
}

#[test]
fn proptest_massively_against_the_compiled_lean_oracle() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);

    for case in CASES {
        compare_with_massively(&exec, &generated_case(case), &generated_expected(case)).unwrap();
    }

    let mut runner = TestRunner::new(Config {
        cases: property_cases(),
        ..Config::default()
    });
    runner
        .run(&graph_case(), |case| {
            let expected = run_lean(&case).map_err(TestCaseError::fail)?;
            compare_with_massively(&exec, &case, &expected)
        })
        .unwrap();
}
