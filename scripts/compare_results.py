#!/usr/bin/env python3
"""compare_results.py — Compare Evo-Memory experiment results.

Usage:
    python scripts/compare_results.py results/comparison_*/

Takes one or more result directories, loads aggregated_results.json
from each, prints a comparison table, and generates:
  - A cumulative accuracy curve (text + optional matplotlib PNG)
  - A summary JSON with the comparison
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any


def find_result_dirs(base_path: str) -> List[Path]:
    """Find all directories containing aggregated_results.json."""
    base = Path(base_path)
    dirs = []

    if (base / "aggregated_results.json").exists():
        dirs.append(base)
        return dirs

    # Look one level down (e.g. comparison_dir/ganglion/, comparison_dir/exprag/)
    for child in sorted(base.iterdir()):
        if child.is_dir():
            # Check inside agent_name subdirectories
            for sub in sorted(child.iterdir()):
                if sub.is_dir() and (sub / "aggregated_results.json").exists():
                    dirs.append(sub)
            if (child / "aggregated_results.json").exists():
                dirs.append(child)

    return dirs


def load_aggregated(path: Path) -> Dict[str, Any]:
    """Load aggregated_results.json."""
    with open(path / "aggregated_results.json") as f:
        return json.load(f)


def load_stream_results(path: Path) -> List[Dict]:
    """Load per-stream result files for cumulative accuracy curves."""
    streams = []
    for f in sorted(path.glob("stream_*_results.json")):
        with open(f) as fh:
            streams.append(json.load(fh))
    return streams


def compute_cumulative_accuracy(stream_data: Dict) -> List[float]:
    """Compute cumulative accuracy at each task index."""
    task_results = stream_data.get("task_results", [])
    if not task_results:
        return []
    cumulative = []
    correct_so_far = 0
    for i, tr in enumerate(task_results):
        if tr.get("correct", False):
            correct_so_far += 1
        cumulative.append(correct_so_far / (i + 1))
    return cumulative


def print_comparison_table(results: Dict[str, Dict]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 72)
    print("COMPARISON TABLE")
    print("=" * 72)
    print(f"{'Agent':<20} {'Accuracy':>12} {'Learning Δ':>12} {'Memory':>10}")
    print("-" * 72)

    for name, data in results.items():
        acc_mean = data.get("accuracy_mean", 0)
        acc_std = data.get("accuracy_std", 0)
        improvement = data.get("learning_improvement_mean", 0)
        mem_size = data.get("final_memory_size_mean", 0)

        acc_str = f"{acc_mean:.4f} ± {acc_std:.4f}"
        imp_str = f"{improvement:+.4f}" if improvement else "N/A"
        mem_str = f"{mem_size:.0f}"

        print(f"{name:<20} {acc_str:>12} {imp_str:>12} {mem_str:>10}")

    print("=" * 72)


def print_cumulative_curve_ascii(all_curves: Dict[str, List[float]]):
    """Print a simple ASCII cumulative accuracy curve."""
    print("\nCUMULATIVE ACCURACY CURVES")
    print("-" * 72)

    if not all_curves:
        print("  No per-stream data available.")
        return

    max_len = max(len(c) for c in all_curves.values()) if all_curves else 0
    if max_len == 0:
        return

    # Sample at 10 evenly-spaced points
    n_points = min(10, max_len)
    indices = [int(i * (max_len - 1) / (n_points - 1)) for i in range(n_points)]

    header = f"{'Task →':<15}"
    for idx in indices:
        header += f"{idx + 1:>7}"
    print(header)
    print("-" * (15 + 7 * n_points))

    for name, curve in all_curves.items():
        row = f"{name:<15}"
        for idx in indices:
            if idx < len(curve):
                row += f"{curve[idx]:>7.3f}"
            else:
                row += f"{'—':>7}"
        print(row)

    print()


def try_plot_curves(all_curves: Dict[str, List[float]], output_path: Path):
    """Try to generate a matplotlib plot (non-fatal if unavailable)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        for name, curve in all_curves.items():
            ax.plot(range(1, len(curve) + 1), curve, label=name, linewidth=2)

        ax.set_xlabel("Task Number")
        ax.set_ylabel("Cumulative Accuracy")
        ax.set_title("Cumulative Accuracy Over Task Stream")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        plot_path = output_path / "cumulative_accuracy.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved to {plot_path}")
    except ImportError:
        print("(matplotlib not installed — skipping plot generation)")


def main():
    parser = argparse.ArgumentParser(
        description="Compare Evo-Memory experiment results"
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Result directories to compare",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for comparison summary (default: first path)",
    )
    args = parser.parse_args()

    # Collect all result directories
    all_dirs = []
    for p in args.paths:
        all_dirs.extend(find_result_dirs(p))

    if not all_dirs:
        print("ERROR: No result directories found with aggregated_results.json")
        sys.exit(1)

    print(f"Found {len(all_dirs)} result set(s):")
    for d in all_dirs:
        print(f"  - {d}")

    # Load results
    results = {}
    all_curves = {}

    for d in all_dirs:
        data = load_aggregated(d)
        config = data.get("config", {})
        agent_name = config.get("name", d.name)
        results[agent_name] = data

        # Load stream results for cumulative curves
        streams = load_stream_results(d)
        if streams:
            # Average cumulative accuracy across streams
            stream_curves = [compute_cumulative_accuracy(s) for s in streams]
            if stream_curves:
                max_len = max(len(c) for c in stream_curves)
                avg_curve = []
                for i in range(max_len):
                    vals = [c[i] for c in stream_curves if i < len(c)]
                    avg_curve.append(sum(vals) / len(vals))
                all_curves[agent_name] = avg_curve

    # Print comparison
    print_comparison_table(results)
    print_cumulative_curve_ascii(all_curves)

    # Determine output directory
    output_dir = Path(args.output) if args.output else Path(args.paths[0])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try to plot
    try_plot_curves(all_curves, output_dir)

    # Save summary JSON
    summary = {
        "agents": {},
        "cumulative_curves": {
            name: curve for name, curve in all_curves.items()
        },
    }

    for name, data in results.items():
        summary["agents"][name] = {
            "accuracy_mean": data.get("accuracy_mean", 0),
            "accuracy_std": data.get("accuracy_std", 0),
            "learning_improvement_mean": data.get("learning_improvement_mean", 0),
            "early_accuracy_mean": data.get("early_accuracy_mean", 0),
            "late_accuracy_mean": data.get("late_accuracy_mean", 0),
            "final_memory_size_mean": data.get("final_memory_size_mean", 0),
            "num_streams": data.get("num_streams", 0),
        }

    summary_path = output_dir / "comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
