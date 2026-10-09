"""Native Pauli operator algebra, parsing, and exact diagonalization.

Provides clean representation of qubit operators, fast matrix synthesis,
expectation value evaluation, and exact matrix diagonalization.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

# Standard 2x2 Pauli matrices
I2 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
X2 = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
Y2 = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
Z2 = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

PAULI_MATRICES = {"I": I2, "X": X2, "Y": Y2, "Z": Z2}

# Sparse 2x2 Pauli matrices
sI2 = scipy.sparse.csr_matrix(I2)
sX2 = scipy.sparse.csr_matrix(X2)
sY2 = scipy.sparse.csr_matrix(Y2)
sZ2 = scipy.sparse.csr_matrix(Z2)

SPARSE_PAULI_MATRICES = {"I": sI2, "X": sX2, "Y": sY2, "Z": sZ2}


@dataclass(frozen=True)
class PauliTerm:
    """A single weighted Pauli product, e.g. -0.25 * X0 Z1.

    Represented internally as a mapping of qubit index -> 'X' | 'Y' | 'Z'.
    Identity factors on unlisted qubits are implicit.
    """

    coefficient: complex
    ops: tuple[tuple[int, str], ...]  # sorted by qubit index

    def __init__(self, coefficient: complex | float, ops: Sequence[tuple[int, str]] = ()):
        # Normalize and sort operations by qubit index
        ops_dict: dict[int, str] = {}
        for q, p in ops:
            p_upper = p.upper()
            if p_upper not in ("I", "X", "Y", "Z"):
                raise ValueError(f"Invalid Pauli label: {p}")
            if p_upper != "I":
                ops_dict[q] = p_upper

        sorted_ops = tuple(sorted(ops_dict.items(), key=lambda x: x[0]))
        object.__setattr__(self, "coefficient", complex(coefficient))
        object.__setattr__(self, "ops", sorted_ops)

    @property
    def is_identity(self) -> bool:
        """True if term has no non-identity Pauli factors."""
        return len(self.ops) == 0

    @property
    def max_qubit(self) -> int:
        """Maximum qubit index referenced, or -1 if pure identity."""
        if not self.ops:
            return -1
        return max(q for q, _ in self.ops)

    def to_matrix(self, n_qubits: int) -> np.ndarray:
        """Convert term to 2^N x 2^N dense matrix."""
        if n_qubits <= 0:
            raise ValueError(f"n_qubits must be positive, got {n_qubits}")
        if self.max_qubit >= n_qubits:
            raise ValueError(f"Term references qubit {self.max_qubit}, but n_qubits is {n_qubits}")

        mat = np.array([[1.0]], dtype=complex)
        ops_map = dict(self.ops)
        for q in range(n_qubits):
            p = ops_map.get(q, "I")
            mat = np.kron(mat, PAULI_MATRICES[p])
        return self.coefficient * mat

    def to_sparse_matrix(self, n_qubits: int) -> scipy.sparse.csr_matrix:
        """Convert term to 2^N x 2^N sparse CSR matrix."""
        if n_qubits <= 0:
            raise ValueError(f"n_qubits must be positive, got {n_qubits}")
        if self.max_qubit >= n_qubits:
            raise ValueError(f"Term references qubit {self.max_qubit}, but n_qubits is {n_qubits}")

        mat = scipy.sparse.csr_matrix([[1.0]], dtype=complex)
        ops_map = dict(self.ops)
        for q in range(n_qubits):
            p = ops_map.get(q, "I")
            mat = scipy.sparse.kron(mat, SPARSE_PAULI_MATRICES[p], format="csr")
        return self.coefficient * mat

    def __str__(self) -> str:
        coeff_str = (
            f"{self.coefficient.real:+.6f}"
            if abs(self.coefficient.imag) < 1e-12
            else f"({self.coefficient.real:+.6f}{self.coefficient.imag:+.6f}j)"
        )
        if not self.ops:
            return f"{coeff_str} * I"
        factors = " ".join(f"{p}{q}" for q, p in self.ops)
        return f"{coeff_str} * {factors}"


class PauliSum:
    """A linear combination of Pauli terms representing a Hamiltonian or observable."""

    def __init__(self, terms: Sequence[PauliTerm] = (), n_qubits: int | None = None):
        self._terms_map: dict[tuple[tuple[int, str], ...], complex] = {}
        max_q = -1
        for term in terms:
            if abs(term.coefficient) > 1e-14:
                self._terms_map[term.ops] = (
                    self._terms_map.get(term.ops, 0.0 + 0.0j) + term.coefficient
                )
                if term.max_qubit > max_q:
                    max_q = term.max_qubit

        # Filter near-zero coefficients after combination
        self._terms_map = {
            ops: coeff for ops, coeff in self._terms_map.items() if abs(coeff) > 1e-14
        }

        inferred_qubits = max_q + 1 if max_q >= 0 else 1
        if n_qubits is not None:
            if n_qubits < inferred_qubits:
                raise ValueError(
                    f"Specified n_qubits={n_qubits} is less than required {inferred_qubits}"
                )
            self.n_qubits = n_qubits
        else:
            self.n_qubits = inferred_qubits

    @property
    def terms(self) -> list[PauliTerm]:
        """Return list of canonical Pauli terms."""
        return [PauliTerm(coeff, ops) for ops, coeff in self._terms_map.items()]

    def __len__(self) -> int:
        return len(self._terms_map)

    def __iter__(self) -> Iterator[PauliTerm]:
        return iter(self.terms)

    def __add__(self, other: PauliSum | PauliTerm | float | complex) -> PauliSum:
        if isinstance(other, (int, float, complex)):
            other = PauliSum([PauliTerm(other, ())], n_qubits=self.n_qubits)
        elif isinstance(other, PauliTerm):
            other = PauliSum([other], n_qubits=max(self.n_qubits, other.max_qubit + 1))

        new_n_qubits = max(self.n_qubits, other.n_qubits)
        all_terms = list(self.terms) + list(other.terms)
        return PauliSum(all_terms, n_qubits=new_n_qubits)

    def __sub__(self, other: PauliSum | PauliTerm | float | complex) -> PauliSum:
        if isinstance(other, (int, float, complex)):
            other = PauliSum([PauliTerm(other, ())], n_qubits=self.n_qubits)
        elif isinstance(other, PauliTerm):
            other = PauliSum([other], n_qubits=max(self.n_qubits, other.max_qubit + 1))

        neg_other_terms = [PauliTerm(-t.coefficient, t.ops) for t in other.terms]
        new_n_qubits = max(self.n_qubits, other.n_qubits)
        all_terms = list(self.terms) + neg_other_terms
        return PauliSum(all_terms, n_qubits=new_n_qubits)

    def __mul__(self, scalar: int | float | complex) -> PauliSum:
        scale = complex(scalar)
        scaled_terms = [PauliTerm(t.coefficient * scale, t.ops) for t in self.terms]
        return PauliSum(scaled_terms, n_qubits=self.n_qubits)

    def __rmul__(self, scalar: int | float | complex) -> PauliSum:
        return self.__mul__(scalar)

    def to_matrix(self) -> np.ndarray:
        """Compute the full 2^N x 2^N dense matrix of the Pauli sum."""
        dim = 1 << self.n_qubits
        mat = np.zeros((dim, dim), dtype=complex)
        for term in self.terms:
            mat += term.to_matrix(self.n_qubits)
        return mat

    def to_sparse_matrix(self) -> scipy.sparse.csr_matrix:
        """Compute the 2^N x 2^N sparse CSR matrix representation."""
        dim = 1 << self.n_qubits
        mat = scipy.sparse.csr_matrix((dim, dim), dtype=complex)
        for term in self.terms:
            mat = mat + term.to_sparse_matrix(self.n_qubits)
        return mat

    def expectation_value(self, statevector: np.ndarray) -> float:
        """Calculate expectation value <psi | H | psi> for normalized statevector."""
        state = np.asarray(statevector, dtype=complex)
        expected_dim = 1 << self.n_qubits
        if state.shape != (expected_dim,):
            raise ValueError(
                f"Statevector shape {state.shape} does not match 2^{self.n_qubits}={expected_dim}"
            )

        norm = np.linalg.norm(state)
        if abs(norm - 1.0) > 1e-7:
            state = state / norm

        h_sparse = self.to_sparse_matrix()
        h_psi = h_sparse.dot(state)
        val = np.vdot(state, h_psi)
        return float(val.real)

    def exact_ground_state(self) -> tuple[float, np.ndarray]:
        """Compute the exact lowest eigenvalue and corresponding eigenvector via diagonalization.

        Returns:
            (ground_energy, ground_statevector)
        """
        dim = 1 << self.n_qubits
        if dim <= 256:
            # Dense Hermitian eigensolver
            mat = self.to_matrix()
            # Ensure numerical Hermiticity
            mat = (mat + mat.conj().T) / 2.0
            eigvals, eigvecs = scipy.linalg.eigh(mat)
            return float(eigvals[0]), eigvecs[:, 0]
        else:
            # Sparse eigensolver for larger Hilbert spaces
            mat_sparse = self.to_sparse_matrix()
            mat_sparse = (mat_sparse + mat_sparse.conj().T) / 2.0
            eigvals, eigvecs = scipy.sparse.linalg.eigsh(mat_sparse, k=1, which="SA")
            return float(eigvals[0]), eigvecs[:, 0]

    def __str__(self) -> str:
        if not self._terms_map:
            return "0.0 * I"
        return "\n".join(str(t) for t in self.terms)


def parse_pauli_string(text: str, n_qubits: int | None = None) -> PauliSum:
    """Parse a human-readable or serialized Pauli Hamiltonian string.

    Supported formats:
        - "0.5 * Z0 Z1 - 0.25 * X0 X1"
        - "-1.05 * I + 0.39 * Z0 + 0.39 * Z1 - 0.01 * Z0 Z1"
        - OpenFermion string formats: "[0.5 Z0 Z1] + [-0.25 X0 X1]"
    """
    cleaned = text.strip()
    if not cleaned:
        return PauliSum([], n_qubits=n_qubits)

    # Normalize whitespaces around signs: e.g. " - " or " - 0.25" -> " + -0.25"
    cleaned = re.sub(r"\s*-\s*", " + -", cleaned)
    tokens = [t.strip() for t in cleaned.split("+") if t.strip()]

    terms: list[PauliTerm] = []
    for token in tokens:
        token = token.strip("[] ")
        if not token:
            continue
        # Match coefficient and Pauli factors
        # Format: optional_coeff * P0 P1 ...
        if "*" in token:
            parts = token.split("*", 1)
            coeff_str = parts[0].strip().replace(" ", "")
            coeff = float(coeff_str)
            factor_str = parts[1].strip()
        else:
            # Check if it starts with a number (e.g. OpenFermion format "[0.5 Z0 Z1]" or "0.5 Z0 Z1")
            subwords = token.split()
            try:
                coeff = float(subwords[0].replace(" ", ""))
                factor_str = " ".join(subwords[1:])
            except (ValueError, IndexError):
                coeff = 1.0
                factor_str = token

        # Parse factors like Z0, X1, Y2, I
        ops: list[tuple[int, str]] = []
        for factor in factor_str.split():
            factor = factor.strip()
            if not factor or factor.upper() == "I":
                continue
            match = re.match(r"^([XYZxyz])(\d+)$", factor)
            if match:
                pauli = match.group(1).upper()
                q_idx = int(match.group(2))
                ops.append((q_idx, pauli))
            else:
                raise ValueError(f"Unrecognized Pauli factor: '{factor}' in token '{token}'")

        terms.append(PauliTerm(coeff, ops))

    return PauliSum(terms, n_qubits=n_qubits)
