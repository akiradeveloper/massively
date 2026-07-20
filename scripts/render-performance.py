#!/usr/bin/env python3

import datetime
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def command_output(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def estimate_ns(path: Path) -> float:
    estimates = json.loads(path.read_text())
    estimate = estimates.get("slope") or estimates["mean"]
    return estimate["point_estimate"]


def benchmark_results():
    root = ROOT / "target" / "criterion" / "performance"
    paths = sorted(root.glob("*/new/estimates.json"))
    if not paths:
        raise SystemExit(f"no Criterion results found under {root.relative_to(ROOT)}")
    return [(path.parent.parent.name, estimate_ns(path)) for path in paths]


def decimal(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def duration(nanoseconds: float) -> str:
    if nanoseconds < 1_000:
        return f"{decimal(nanoseconds)} ns"
    if nanoseconds < 1_000_000:
        return f"{decimal(nanoseconds / 1_000)} us"
    if nanoseconds < 1_000_000_000:
        return f"{decimal(nanoseconds / 1_000_000)} ms"
    return f"{decimal(nanoseconds / 1_000_000_000)} s"


def main() -> None:
    adapter = os.environ.get("MASSIVELY_BENCH_DEVICE", "wgpu default adapter")
    revision = command_output("git", "rev-parse", "--short", "HEAD")
    rust = command_output("rustc", "--version")
    rows = [(api, duration(estimate)) for api, estimate in benchmark_results()]

    lines = [
        "# Performance",
        "",
        "Reference execution times for the vector API at N = 10,000,000.",
        "The table lists APIs covered by the dedicated performance benchmark.",
        "These are approximate, machine-specific values rather than performance guarantees.",
        "",
        "| Environment | Value |",
        "|---|---|",
        f"| GPU | {adapter} |",
        "| Runtime | CubeCL WGPU default device |",
        f"| Rust | `{rust}` |",
        f"| Revision | `{revision}` |",
        f"| Measured | {datetime.date.today().isoformat()} |",
        "",
        "| API | Time |",
        "|---|---:|",
    ]
    lines.extend(f"| `{api}` | {time} |" for api, time in rows)
    lines.extend(
        [
            "",
            "## Conditions",
            "",
            "Stored inputs are already device-resident, and input construction and host-to-device transfer are "
            "excluded. API-internal output allocation and synchronization required to observe completion are "
            "included; caller-provided output buffers are allocated before timing. Times are Criterion point "
            "estimates after warm-up (slope when available, otherwise mean).",
            "",
            "N is the length of each input range, so binary range algorithms process two inputs of N elements. The "
            "default element type is `f32` for value algorithms and `u32` for ordering, index, and key algorithms. "
            "Predicate and extremum queries use `lazy::counting<usize>`. Selection uses a 50% stencil, indexed "
            "operations use reverse indices, by-key algorithms use runs of eight equal keys, `scatter_reduce` maps "
            "four inputs to each output, and sorting uses deterministic shuffled keys.",
            "",
            "Regenerate this file with:",
            "",
            "```console",
            'MASSIVELY_BENCH_DEVICE="<GPU model>" just performance',
            "```",
            "",
        ]
    )
    (ROOT / "PERFORMANCE.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
