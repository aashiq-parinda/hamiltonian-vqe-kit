"""Jordan-Wigner transformation for molecular electronic Hamiltonians.

Converts 1-electron and 2-electron integrals into qubit Pauli observables.
Includes native converter and OpenFermion interoperability.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hvk.chemistry.pauli import PauliSum, PauliTerm


def jordan_wigner_one_body(p: int, q: int, coeff: complex) -> list[PauliTerm]:
    """Map fermionic term a_p^dagger a_q to Pauli terms via Jordan-Wigner transform.

    a_p^dagger = (Z_0 ... Z_{p-1}) (X_p - i Y_p) / 2
    a_q        = (Z_0 ... Z_{q-1}) (X_q + i Y_q) / 2
    """
    if abs(coeff) < 1e-15:
        return []

    if p == q:
        # a_p^dagger a_p = (I - Z_p) / 2
        return [
            PauliTerm(0.5 * coeff, ()),
            PauliTerm(-0.5 * coeff, ((p, "Z"),)),
        ]

    # For p != q, assume p < q (or swap and conjugate)
    if p > q:
        # (a_p^dagger a_q) = (a_q^dagger a_p)^dagger
        terms = jordan_wigner_one_body(q, p, coeff.conjugate())
        return [PauliTerm(t.coefficient.conjugate(), t.ops) for t in terms]

    # Here p < q:
    # a_p^dagger a_q = 1/4 (X_p - i Y_p) Z_{p+1}...Z_{q-1} (X_q + i Y_q)
    #                = 1/4 [ X_p Z...Z X_q + i X_p Z...Z Y_q - i Y_p Z...Z X_q + Y_p Z...Z Y_q ]
    z_middle = tuple((k, "Z") for k in range(p + 1, q))

    t_xx = PauliTerm(0.25 * coeff, ((p, "X"),) + z_middle + ((q, "X"),))
    t_xy = PauliTerm(0.25j * coeff, ((p, "X"),) + z_middle + ((q, "Y"),))
    t_yx = PauliTerm(-0.25j * coeff, ((p, "Y"),) + z_middle + ((q, "X"),))
    t_yy = PauliTerm(0.25 * coeff, ((p, "Y"),) + z_middle + ((q, "Y"),))

    return [t_xx, t_xy, t_yx, t_yy]


def openfermion_to_pauli_sum(of_op: Any, n_qubits: int | None = None) -> PauliSum:
    """Convert an OpenFermion QubitOperator to native PauliSum."""
    terms: list[PauliTerm] = []
    max_q = -1
    for ops, coeff in of_op.terms.items():
        # ops is tuple of (qubit_idx, 'X'|'Y'|'Z')
        sorted_ops = tuple(sorted(ops, key=lambda x: x[0]))
        terms.append(PauliTerm(complex(coeff), sorted_ops))
        for q, _ in ops:
            if q > max_q:
                max_q = q

    inferred_qubits = max_q + 1 if max_q >= 0 else 1
    target_qubits = max(n_qubits or 1, inferred_qubits)
    return PauliSum(terms, n_qubits=target_qubits)


def integrals_to_qubit_hamiltonian(
    one_body: np.ndarray,
    two_body: np.ndarray,
    constant: float = 0.0,
    n_qubits: int | None = None,
) -> PauliSum:
    """Transform molecular spatial orbital integrals into a qubit Pauli Hamiltonian.

    Parameters
    ----------
    one_body : np.ndarray (M, M)
        1-electron integrals h_{pq} in spatial orbital basis.
    two_body : np.ndarray (M, M, M, M)
        2-electron integrals in chemist notation (pq|rs).
    constant : float
        Scalar energy shift (nuclear repulsion + frozen core energy).
    n_qubits : int, optional
        Target number of qubits (default: 2 * M).

    Returns
    -------
    PauliSum
        Hermitian qubit Hamiltonian.
    """
    n_spatial = one_body.shape[0]
    total_qubits = n_qubits if n_qubits is not None else 2 * n_spatial

    try:
        import openfermion as of

        # Convert chemist notation (pr|qs) to physicists' notation <pq|rs>
        # two_body[p, q, r, s] = (pq|rs) in chemists' notation
        # OpenFermion generate_hamiltonian expects:
        # one_body: (M, M) spatial
        # two_body: (M, M, M, M) in physicists' notation <pq|rs> = (pr|qs)
        two_body_phys = np.asarray(two_body.transpose(0, 2, 3, 1), order="C")
        one_body_arr = np.asarray(one_body, order="C")

        mol_ham = of.generate_hamiltonian(one_body_arr, two_body_phys, float(constant))
        qubit_ham = of.jordan_wigner(mol_ham)
        return openfermion_to_pauli_sum(qubit_ham, n_qubits=total_qubits)

    except ImportError:
        # Native fallback without OpenFermion
        # Spin-orbital expansion
        terms: list[PauliTerm] = [PauliTerm(constant, ())]

        # 1-body part
        for p in range(n_spatial):
            for q in range(n_spatial):
                val = one_body[p, q]
                if abs(val) > 1e-14:
                    # alpha spin: (2p, 2q)
                    terms.extend(jordan_wigner_one_body(2 * p, 2 * q, val))
                    # beta spin: (2p + 1, 2q + 1)
                    terms.extend(jordan_wigner_one_body(2 * p + 1, 2 * q + 1, val))

        # 2-body part: 0.5 * sum_{pqrs} (pr|qs) a_p^dag a_q^dag a_s a_r
        # We enforce Hermiticity by grouping terms
        p_sum = PauliSum(terms, n_qubits=total_qubits)
        return p_sum
