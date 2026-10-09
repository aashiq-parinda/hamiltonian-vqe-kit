"""Test basic package importing and CLI invocation."""

import pytest

import hvk
from hvk.cli import create_parser, main


def test_package_metadata() -> None:
    """Verify package version, author, and license metadata."""
    assert hasattr(hvk, "__version__")
    assert isinstance(hvk.__version__, str)
    assert hvk.__version__ == "0.1.0"
    assert hvk.__license__ == "Apache-2.0"
    assert "Ashraf Khan" in hvk.__author__


def test_cli_parser_creation() -> None:
    """Verify CLI parser has all required subcommands."""
    parser = create_parser()
    subparsers_actions = [action for action in parser._actions if action.dest == "command"]
    assert len(subparsers_actions) == 1
    choices = subparsers_actions[0].choices
    assert isinstance(choices, dict)
    expected_subcommands = {"audit", "adapt", "build", "init", "diagnose", "benchmark"}
    assert expected_subcommands.issubset(set(choices.keys()))


def test_cli_invocation_help() -> None:
    """Verify CLI returns exit code 1 when no subcommand is given."""
    exit_code = main([])
    assert exit_code == 1


def test_cli_subcommand_stubs(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify all subcommands execute without error in Phase 0."""
    subcommands = [
        ["audit", "H2"],
        ["adapt", "H2"],
        ["build", "H2"],
        ["init", "--method", "hf"],
        ["diagnose", "H2"],
        ["benchmark"],
    ]
    for cmd in subcommands:
        code = main(cmd)
        captured = capsys.readouterr()
        assert code == 0
        assert f"hvk {cmd[0]}: initialized" in captured.out
