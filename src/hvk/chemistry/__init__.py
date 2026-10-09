"""Chemistry modules for Hamiltonian VQE Kit (hvk).

Provides molecular loaders, PySCF interfaces, Jordan-Wigner mappings,
Pauli operator algebra, and chemical reference validation.
"""

from hvk.chemistry.jordan_wigner import (
    integrals_to_qubit_hamiltonian,
    jordan_wigner_one_body,
    openfermion_to_pauli_sum,
)
from hvk.chemistry.pauli import (
    PauliSum,
    PauliTerm,
    parse_pauli_string,
)
from hvk.chemistry.pyscf_loader import (
    MolecularSystem,
    build_molecular_system,
    load_beh2,
    load_h2,
    load_h4,
    load_lih,
    load_molecule,
)
from hvk.chemistry.validation import (
    run_chemistry_validation_suite,
    validate_single_system,
)

__all__ = [
    "PauliTerm",
    "PauliSum",
    "parse_pauli_string",
    "jordan_wigner_one_body",
    "openfermion_to_pauli_sum",
    "integrals_to_qubit_hamiltonian",
    "MolecularSystem",
    "build_molecular_system",
    "load_h2",
    "load_lih",
    "load_beh2",
    "load_h4",
    "load_molecule",
    "validate_single_system",
    "run_chemistry_validation_suite",
]
