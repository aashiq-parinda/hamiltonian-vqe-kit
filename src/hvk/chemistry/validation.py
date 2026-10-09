"""Rigorous chemical validation suite against PySCF reference energies.

Executes non-negotiable verification of:
  1. Hartree-Fock statevector expectation value <HF|H|HF> vs. PySCF RHF
  2. Qubit Hamiltonian exact diagonalization ground state E_exact vs. PySCF FCI/CASCI

Outputs verifiable benchmark tables with git commit metadata and strict tolerance checks.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np
import scipy

from hvk.chemistry.pyscf_loader import (
    MolecularSystem,
    load_beh2,
    load_h2,
    load_h4,
    load_lih,
)

DEFAULT_TOLERANCE_HA: float = 1e-6


def get_git_commit_hash() -> str:
    """Retrieve current git commit hash if in a git repository."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def validate_single_system(
    mol: MolecularSystem,
    tolerance: float = DEFAULT_TOLERANCE_HA,
) -> dict[str, Any]:
    """Validate a single molecular system against PySCF references.

    Parameters
    ----------
    mol : MolecularSystem
        System to validate.
    tolerance : float
        Maximum allowable energy discrepancy in Hartree (default: 1e-6 Ha).

    Returns
    -------
    dict
        Detailed comparison metrics and pass/fail boolean.
    """
    # 1. Evaluate Hartree-Fock expectation value on qubit Hamiltonian
    e_hf_qubit = mol.qubit_hamiltonian.expectation_value(mol.hf_statevector)
    hf_diff = abs(e_hf_qubit - mol.e_hf_pyscf)

    # 2. Evaluate exact ground state via matrix diagonalization
    e_exact_qubit, _ = mol.qubit_hamiltonian.exact_ground_state()
    fci_diff = abs(e_exact_qubit - mol.e_fci_pyscf)

    hf_pass = hf_diff <= tolerance
    fci_pass = fci_diff <= tolerance
    overall_pass = hf_pass and fci_pass

    return {
        "molecule": mol.name,
        "bond_distance_angstrom": mol.bond_distance,
        "n_qubits": mol.n_qubits,
        "n_electrons": mol.n_electrons,
        "n_pauli_terms": len(mol.qubit_hamiltonian),
        "e_hf_pyscf": mol.e_hf_pyscf,
        "e_hf_qubit": e_hf_qubit,
        "hf_diff_ha": hf_diff,
        "hf_diff_mha": hf_diff * 1000.0,
        "hf_passed": hf_pass,
        "e_fci_pyscf": mol.e_fci_pyscf,
        "e_exact_qubit": e_exact_qubit,
        "fci_diff_ha": fci_diff,
        "fci_diff_mha": fci_diff * 1000.0,
        "fci_passed": fci_pass,
        "overall_passed": overall_pass,
    }


def run_chemistry_validation_suite(
    output_dir: str = "benchmarks/results",
    tolerance: float = DEFAULT_TOLERANCE_HA,
    strict_fail: bool = True,
) -> dict[str, Any]:
    """Run full chemical validation across multiple molecular systems and bond lengths.

    Evaluates:
      - H2 at R = [0.5, 0.7414, 1.0, 1.5, 2.0, 2.5] Å
      - LiH at R = [1.0, 1.595, 2.0, 2.5] Å
      - BeH2 at R = [1.0, 1.326, 1.8, 2.4] Å
      - H4 chain at R = [0.8, 1.0, 1.5, 2.0] Å

    Writes reproducible artifacts to benchmarks/results/.
    """
    try:
        import pyscf

        pyscf_version = getattr(pyscf, "__version__", "unknown")
    except ImportError:
        pyscf_version = "not_installed"

    systems_to_test: list[MolecularSystem] = []

    # 1. H2 potential energy surface
    for r in [0.5, 0.7414, 1.0, 1.5, 2.0, 2.5]:
        systems_to_test.append(load_h2(bond_distance=r))

    # 2. LiH potential energy surface (CAS 2e, 3o -> 6 qubits)
    for r in [1.0, 1.595, 2.0, 2.5]:
        systems_to_test.append(load_lih(bond_distance=r))

    # 3. BeH2 potential energy surface (CAS 2e, 3o -> 6 qubits)
    for r in [1.0, 1.326, 1.8, 2.4]:
        systems_to_test.append(load_beh2(bond_distance=r))

    # 4. H4 chain (4e, 4o -> 8 qubits)
    for r in [0.8, 1.0, 1.5, 2.0]:
        systems_to_test.append(load_h4(bond_distance=r))

    results: list[dict[str, Any]] = []
    all_passed = True
    failures: list[dict[str, Any]] = []

    for mol in systems_to_test:
        rec = validate_single_system(mol, tolerance=tolerance)
        results.append(rec)
        if not rec["overall_passed"]:
            all_passed = False
            failures.append(rec)

    # Collect environment metadata
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit_hash(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "pyscf_version": pyscf_version,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "tolerance_ha": tolerance,
        "total_systems_tested": len(results),
        "total_passed": sum(1 for r in results if r["overall_passed"]),
        "all_passed": all_passed,
    }

    report_payload = {
        "metadata": metadata,
        "results": results,
    }

    # Write JSON artifact
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "chemistry_validation.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    # Write Markdown table artifact
    md_path = os.path.join(output_dir, "chemistry_validation.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Chemical Structure & Reference Parity Benchmark\n\n")
        f.write(f"**Date (UTC)**: {metadata['timestamp_utc']}  \n")
        f.write(f"**Git Commit**: `{metadata['git_commit']}`  \n")
        f.write(
            f"**PySCF Version**: `{metadata['pyscf_version']}` | **NumPy**: `{metadata['numpy_version']}` | **SciPy**: `{metadata['scipy_version']}`  \n"
        )
        f.write(f"**Tolerance**: `{tolerance:.1e} Ha`  \n\n")
        f.write("## Validation Results Table\n\n")
        f.write(
            "| Molecule | R (Å) | Qubits | Terms | PySCF HF (Ha) | Qubit HF (Ha) | |ΔHF| (mHa) | PySCF FCI (Ha) | Exact Qubit (Ha) | |ΔFCI| (mHa) | Status |\n"
        )
        f.write(
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
        )
        for res in results:
            status = "✅ PASS" if res["overall_passed"] else "❌ FAIL"
            f.write(
                f"| {res['molecule']} | {res['bond_distance_angstrom']:.4f} | {res['n_qubits']} | {res['n_pauli_terms']} | "
                f"{res['e_hf_pyscf']:.8f} | {res['e_hf_qubit']:.8f} | {res['hf_diff_mha']:.2e} | "
                f"{res['e_fci_pyscf']:.8f} | {res['e_exact_qubit']:.8f} | {res['fci_diff_mha']:.2e} | {status} |\n"
            )

    if strict_fail and not all_passed:
        fail_summary = "\n".join(
            f"  - {f['molecule']} (R={f['bond_distance_angstrom']}): HF diff={f['hf_diff_ha']:.2e}, FCI diff={f['fci_diff_ha']:.2e}"
            for f in failures
        )
        raise AssertionError(
            f"Chemistry validation failed for {len(failures)} system(s) beyond tolerance {tolerance:.1e} Ha:\n{fail_summary}"
        )

    return report_payload
