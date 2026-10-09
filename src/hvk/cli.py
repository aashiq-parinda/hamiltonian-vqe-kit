"""Command-line interface for Hamiltonian VQE Kit (hvk).

Commands:
    hvk audit       - Hamiltonian locality & interaction graph audit
    hvk adapt       - Exact ADAPT-VQE execution with honest accounting
    hvk build       - Sparse ansatz synthesis from interaction graph
    hvk init        - Statevector initialization (random, HF, PA-HF, guarded ML)
    hvk diagnose    - Barren plateau and gradient variance diagnostics
    hvk benchmark   - Multi-molecule validation & noise benchmark
"""

import argparse
import sys

from hvk import __version__


def create_parser() -> argparse.ArgumentParser:
    """Create the top-level argument parser for the hvk CLI."""
    parser = argparse.ArgumentParser(
        prog="hvk",
        description="Hamiltonian VQE Kit: honest, validated quantum algorithm tools.",
    )
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # hvk audit
    audit_parser = subparsers.add_parser(
        "audit", help="Audit Hamiltonian locality & interaction graph"
    )
    audit_parser.add_argument("hamiltonian", help="Path to Hamiltonian file or molecule name")

    # hvk adapt
    adapt_parser = subparsers.add_parser("adapt", help="Run exact ADAPT-VQE with honest accounting")
    adapt_parser.add_argument("hamiltonian", help="Path to Hamiltonian file or molecule name")
    adapt_parser.add_argument(
        "--accuracy",
        type=float,
        default=0.0016,
        help="Target accuracy in Hartree (default: 0.0016 Ha = 1.6 mHa)",
    )

    # hvk build
    build_parser = subparsers.add_parser("build", help="Build sparse ansatz from interaction graph")
    build_parser.add_argument("hamiltonian", help="Path to Hamiltonian file or molecule name")

    # hvk init
    init_parser = subparsers.add_parser(
        "init", help="Generate parameter initialization (random, HF, PA-HF, guarded ML)"
    )
    init_parser.add_argument(
        "--method",
        choices=["random", "hf", "pa-hf", "ml"],
        default="hf",
        help="Initialization strategy (default: hf)",
    )

    # hvk diagnose
    diag_parser = subparsers.add_parser(
        "diagnose", help="Run gradient variance & barren plateau diagnostics"
    )
    diag_parser.add_argument("hamiltonian", help="Path to Hamiltonian file or molecule name")

    # hvk benchmark
    bench_parser = subparsers.add_parser(
        "benchmark", help="Run full multi-molecule validation and noise benchmarks"
    )
    bench_parser.add_argument(
        "--output-dir",
        default="benchmarks/results",
        help="Directory to save validated benchmark artifacts",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main CLI entrypoint."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help(sys.stderr)
        return 1

    # Stubs for Phase 0
    sys.stdout.write(f"hvk {args.command}: initialized (Phase 0 scaffold)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
