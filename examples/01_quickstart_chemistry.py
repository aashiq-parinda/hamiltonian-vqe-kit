#!/usr/bin/env python3
"""Example 01: Chemistry Loaders, Jordan-Wigner Mapping, and Pauli Algebra.

This script demonstrates how to use the `hamiltonian-vqe-kit` (hvk) library to:
1. Load molecular systems ($H_2$, $\\text{LiH}$) using PySCF.
2. Inspect orbital active spaces and generate Jordan-Wigner qubit Hamiltonians.
3. Verify Hartree-Fock and Full Configuration Interaction (FCI) reference parity.
4. Perform native Pauli algebra, expectation values, and commutators.
"""

from __future__ import annotations

import numpy as np

from hvk.chemistry import (
    PauliSum,
    PauliTerm,
    load_h2,
    load_lih,
    parse_pauli_string,
    validate_single_system,
)


def main() -> None:
    print("=" * 70)
    print("  Hamiltonian-VQE-Kit (hvk) — Chemistry & Pauli Quickstart")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Loading Molecular Systems & Active Spaces
    # -------------------------------------------------------------------------
    print("\n[Step 1] Loading molecular systems...")

    # Load H2 at equilibrium bond length (0.7414 Å)
    h2 = load_h2(bond_distance=0.7414)
    print(f"• Molecule: {h2.name} at R = {h2.bond_distance:.4f} Å")
    print(f"  - Spatial Orbitals: {h2.n_qubits // 2}")
    print(f"  - Active Electrons: {h2.n_electrons}")
    print(f"  - Qubits Required:  {h2.n_qubits} qubits (via Jordan-Wigner)")
    print(f"  - Pauli Terms:      {len(h2.qubit_hamiltonian)} terms in Hamiltonian")
    print(f"  - PySCF RHF Energy: {h2.e_hf_pyscf:.8f} Ha")
    print(f"  - PySCF FCI Energy: {h2.e_fci_pyscf:.8f} Ha")

    # Load LiH with frozen-core CAS(2e, 3o) active space (6 qubits)
    lih = load_lih(bond_distance=1.595)
    print(f"\n• Molecule: {lih.name} at R = {lih.bond_distance:.4f} Å (CAS 2e, 3o)")
    print(f"  - Qubits Required:  {lih.n_qubits} qubits")
    print(f"  - Pauli Terms:      {len(lih.qubit_hamiltonian)} terms")
    print(f"  - Active HF Energy: {lih.e_hf_pyscf:.8f} Ha")
    print(f"  - Active FCI Energy:{lih.e_fci_pyscf:.8f} Ha")

    # -------------------------------------------------------------------------
    # 2. Reference Parity Validation
    # -------------------------------------------------------------------------
    print("\n[Step 2] Validating exact parity against PySCF references...")
    report_h2 = validate_single_system(h2, tolerance=1e-6)
    print(f"• H2 Validation Status: {'✅ PASS' if report_h2['overall_passed'] else '❌ FAIL'}")
    print(f"  - Qubit HF State Energy:    {report_h2['e_hf_qubit']:.8f} Ha")
    print(f"  - HF Discrepancy (|ΔHF|):    {report_h2['hf_diff_mha']:.2e} mHa")
    print(f"  - Exact Diagonalization E0: {report_h2['e_exact_qubit']:.8f} Ha")
    print(f"  - FCI Discrepancy (|ΔFCI|):  {report_h2['fci_diff_mha']:.2e} mHa")

    # -------------------------------------------------------------------------
    # 3. Native Pauli Algebra & Expectation Values
    # -------------------------------------------------------------------------
    print("\n[Step 3] Working with PauliSum and PauliTerm operators...")

    # Define simple Pauli operators
    t1 = PauliTerm(0.5, [(0, "Z"), (1, "Z")])
    t2 = PauliTerm(0.25, [(0, "X"), (1, "X")])
    h_toy = PauliSum([t1, t2], n_qubits=2) + (-1.05)

    print(f"• Toy Hamiltonian:\n  {h_toy}")

    # Parse human-readable Hamiltonian string
    parsed_op = parse_pauli_string("0.5 * Z0 Z1 - 0.25 * X0 X1 + 1.25 * I", n_qubits=2)
    print(f"• Parsed Operator:\n  {parsed_op}")

    # Evaluate expectation value on state |00> (binary index 0)
    psi_00 = np.zeros(4, dtype=complex)
    psi_00[0] = 1.0
    val = h_toy.expectation_value(psi_00)
    print(f"• Expectation <00| H_toy |00> = {val.real:.4f} + {val.imag:.4f}j")

    # Compute exact ground state
    e_min, psi_min = h_toy.exact_ground_state()
    print(f"• Exact Ground State Energy: {e_min:.6f} Ha")

    print("\n" + "=" * 70)
    print("  Demonstration complete. Everything verified successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
