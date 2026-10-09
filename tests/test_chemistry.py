"""Comprehensive test suite for Phase 1 chemistry validation."""

import numpy as np
import pytest

from hvk.chemistry.jordan_wigner import (
    jordan_wigner_one_body,
    openfermion_to_pauli_sum,
)
from hvk.chemistry.pauli import (
    PauliSum,
    PauliTerm,
    parse_pauli_string,
)
from hvk.chemistry.pyscf_loader import (
    load_beh2,
    load_h2,
    load_h4,
    load_lih,
    load_molecule,
)
from hvk.chemistry.validation import validate_single_system

# =============================================================================
# 1. Pauli Algebra & Parsing Unit Tests
# =============================================================================


def test_pauli_term_creation_and_properties() -> None:
    """Test PauliTerm initialization, sorting, and edge cases."""
    # Identity term
    t_id = PauliTerm(1.5, [])
    assert t_id.is_identity
    assert t_id.max_qubit == -1
    assert "1.500000 * I" in str(t_id)

    # Multi-qubit term with sorting
    t_zx = PauliTerm(-0.25, [(2, "x"), (0, "Z")])
    assert not t_zx.is_identity
    assert t_zx.max_qubit == 2
    assert t_zx.ops == ((0, "Z"), (2, "X"))

    # Dense & Sparse Matrix representations
    mat_dense = t_zx.to_matrix(n_qubits=3)
    mat_sparse = t_zx.to_sparse_matrix(n_qubits=3).toarray()
    assert np.allclose(mat_dense, mat_sparse)
    assert mat_dense.shape == (8, 8)

    # Invalid Pauli label
    with pytest.raises(ValueError, match="Invalid Pauli label"):
        PauliTerm(1.0, [(0, "Q")])

    # n_qubits out of range
    with pytest.raises(ValueError, match="must be positive"):
        t_zx.to_matrix(0)
    with pytest.raises(ValueError, match="references qubit 2, but n_qubits is 2"):
        t_zx.to_matrix(2)


def test_pauli_sum_algebra_and_matrices() -> None:
    """Test PauliSum combination, scalar multiplication, and exact diagonalization."""
    t1 = PauliTerm(1.0, [(0, "Z")])
    t2 = PauliTerm(2.0, [(1, "Z")])
    t3 = PauliTerm(0.5, [(0, "X"), (1, "X")])

    h = PauliSum([t1, t2, t3], n_qubits=2)
    assert len(h) == 3
    assert h.n_qubits == 2

    # Linear combination
    h_double = 2.0 * h
    assert np.isclose(h_double.terms[0].coefficient, 2.0)

    h_sub = h - t1
    assert len(h_sub) == 2

    h_add_scalar = h + 1.5
    assert len(h_add_scalar) == 4

    # Dense & sparse equivalence
    m_dense = h.to_matrix()
    m_sparse = h.to_sparse_matrix().toarray()
    assert np.allclose(m_dense, m_sparse)

    # Expectation value on |00>
    state_00 = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    exp_00 = h.expectation_value(state_00)
    # <00| Z0 + 2 Z1 + 0.5 X0 X1 |00> = 1 + 2 + 0 = 3.0
    assert np.isclose(exp_00, 3.0)

    # Exact diagonalization
    e_min, psi_min = h.exact_ground_state()
    # Matrix eigenvalues analytically:
    # [[3, 0, 0, 0.5], [0, -1, 0.5, 0], [0, 0.5, 1, 0], [0.5, 0, 0, -3]]
    # Min eigenvalue is from the 2x2 block with -3 and 3, or -1 and 1
    eigvals = np.linalg.eigvalsh(m_dense)
    assert np.isclose(e_min, eigvals[0])
    assert psi_min.shape == (4,)


def test_parse_pauli_string() -> None:
    """Test string parser for human-readable Pauli Hamiltonian strings."""
    text = "0.5 * Z0 Z1 - 0.25 * X0 X1 + 1.25 * I"
    ps = parse_pauli_string(text, n_qubits=2)
    assert len(ps) == 3
    assert ps.n_qubits == 2

    # Empty string
    ps_empty = parse_pauli_string("")
    assert len(ps_empty) == 0

    # Invalid token format
    with pytest.raises(ValueError, match="Unrecognized Pauli factor"):
        parse_pauli_string("1.0 * FOO")


# =============================================================================
# 2. Jordan-Wigner Transformation Unit Tests
# =============================================================================


def test_jordan_wigner_one_body() -> None:
    """Test 1-body Jordan-Wigner operator mapping."""
    # Diagonal term a_0^dag a_0 = (I - Z_0) / 2
    terms_diag = jordan_wigner_one_body(0, 0, coeff=1.0)
    assert len(terms_diag) == 2
    sum_diag = PauliSum(terms_diag, n_qubits=1)
    # Expectation on |0> (unoccupied, bit 0) = 0
    # Expectation on |1> (occupied, bit 1) = 1
    assert np.isclose(sum_diag.expectation_value(np.array([1.0, 0.0])), 0.0)
    assert np.isclose(sum_diag.expectation_value(np.array([0.0, 1.0])), 1.0)

    # Off-diagonal hopping term a_0^dag a_1 + a_1^dag a_0
    terms_hop_01 = jordan_wigner_one_body(0, 1, coeff=1.0)
    terms_hop_10 = jordan_wigner_one_body(1, 0, coeff=1.0)
    hop_sum = PauliSum(terms_hop_01 + terms_hop_10, n_qubits=2)
    # Should be Hermitian
    mat = hop_sum.to_matrix()
    assert np.allclose(mat, mat.conj().T)


def test_openfermion_interoperability() -> None:
    """Test OpenFermion QubitOperator to native PauliSum conversion."""
    import openfermion as of

    of_op = (
        of.QubitOperator("X0 Y1", 0.75)
        + of.QubitOperator("Z0 Z1", -0.35)
        + of.QubitOperator("", 2.1)
    )
    ps = openfermion_to_pauli_sum(of_op, n_qubits=2)
    assert len(ps) == 3
    assert np.allclose(ps.to_matrix(), of.get_sparse_operator(of_op).toarray())


# =============================================================================
# 3. Chemical Validation Unit Tests (PySCF Reference Parity)
# =============================================================================


def test_h2_chemical_parity() -> None:
    """Verify H2 parity between PySCF (HF & FCI) and Jordan-Wigner qubit Hamiltonian."""
    mol = load_h2(bond_distance=0.7414)
    assert mol.n_qubits == 4
    assert mol.n_electrons == 2

    # Validate against tolerance
    rec = validate_single_system(mol, tolerance=1e-6)
    assert rec["hf_passed"], f"H2 HF mismatch: {rec['hf_diff_ha']} Ha"
    assert rec["fci_passed"], f"H2 FCI mismatch: {rec['fci_diff_ha']} Ha"
    assert rec["overall_passed"]


def test_lih_chemical_parity() -> None:
    """Verify LiH CAS(2e, 3o) parity between PySCF and Jordan-Wigner qubit Hamiltonian."""
    mol = load_lih(bond_distance=1.595)
    assert mol.n_qubits == 6
    assert mol.n_electrons == 2

    rec = validate_single_system(mol, tolerance=1e-6)
    assert rec["hf_passed"], f"LiH HF mismatch: {rec['hf_diff_ha']} Ha"
    assert rec["fci_passed"], f"LiH FCI mismatch: {rec['fci_diff_ha']} Ha"
    assert rec["overall_passed"]


def test_beh2_chemical_parity() -> None:
    """Verify BeH2 CAS(2e, 3o) parity between PySCF and Jordan-Wigner qubit Hamiltonian."""
    mol = load_beh2(bond_distance=1.326)
    assert mol.n_qubits == 6
    assert mol.n_electrons == 2

    rec = validate_single_system(mol, tolerance=1e-6)
    assert rec["hf_passed"], f"BeH2 HF mismatch: {rec['hf_diff_ha']} Ha"
    assert rec["fci_passed"], f"BeH2 FCI mismatch: {rec['fci_diff_ha']} Ha"
    assert rec["overall_passed"]


def test_h4_chain_chemical_parity() -> None:
    """Verify H4 chain (4e, 4o -> 8 qubits) parity between PySCF and Jordan-Wigner qubit Hamiltonian."""
    mol = load_h4(bond_distance=1.0)
    assert mol.n_qubits == 8
    assert mol.n_electrons == 4

    rec = validate_single_system(mol, tolerance=1e-6)
    assert rec["hf_passed"], f"H4 HF mismatch: {rec['hf_diff_ha']} Ha"
    assert rec["fci_passed"], f"H4 FCI mismatch: {rec['fci_diff_ha']} Ha"
    assert rec["overall_passed"]


def test_load_molecule_dispatch() -> None:
    """Test generic load_molecule helper and error on invalid molecule."""
    mol_h2 = load_molecule("H2", bond_distance=0.7414)
    assert mol_h2.name == "H2"

    mol_lih = load_molecule("LiH", bond_distance=1.595)
    assert mol_lih.name == "LiH"

    mol_beh2 = load_molecule("BeH2", bond_distance=1.326)
    assert mol_beh2.name == "BeH2"

    mol_h4 = load_molecule("H4", bond_distance=1.0)
    assert mol_h4.name in ("H4", "H4_chain")

    with pytest.raises(ValueError, match="Unknown molecule name"):
        load_molecule("NonExistentMolecule")


# =============================================================================
# 4. Validation Suite & Edge Case Unit Tests
# =============================================================================


def test_pauli_commutator_and_advanced_algebra() -> None:
    """Test commutator [A, B] = AB - BA and operator multiplication."""
    x0 = PauliSum([PauliTerm(1.0, [(0, "X")])], n_qubits=1)
    y0 = PauliSum([PauliTerm(1.0, [(0, "Y")])], n_qubits=1)
    z0 = PauliSum([PauliTerm(1.0, [(0, "Z")])], n_qubits=1)

    # [X, Y] = 2i Z
    # In matrix form: X Y - Y X = [[0, 1], [1, 0]] [[0, -1j], [1j, 0]] - ... = 2j * [[1, 0], [0, -1]]
    comm_mat = x0.to_matrix() @ y0.to_matrix() - y0.to_matrix() @ x0.to_matrix()
    expected = 2j * z0.to_matrix()
    assert np.allclose(comm_mat, expected)

    # Ground state calculation
    h = z0 + 0.5 * x0
    e_exact, psi_exact = h.exact_ground_state()
    # Analytical: eigenvalues of [[1, 0.5], [0.5, -1]] are +/- sqrt(1 + 0.25) = +/- sqrt(1.25) = -1.1180339887
    assert np.isclose(e_exact, -np.sqrt(1.25))
    assert psi_exact.shape == (2,)


def test_validation_suite_execution(tmp_path: pytest.TempPathFactory) -> None:
    """Test run_chemistry_validation_suite runs and writes artifact files."""
    import os
    from hvk.chemistry.validation import run_chemistry_validation_suite

    out_dir = str(tmp_path)
    report = run_chemistry_validation_suite(output_dir=out_dir, tolerance=1e-5)
    assert report["metadata"]["all_passed"]
    assert report["metadata"]["total_systems_tested"] > 0

    assert os.path.exists(os.path.join(out_dir, "chemistry_validation.json"))
    assert os.path.exists(os.path.join(out_dir, "chemistry_validation.md"))

