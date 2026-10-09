"""Molecular Hamiltonian loaders and PySCF electronic structure interfaces.

Constructs molecular systems, extracts 1-electron and 2-electron spatial
integrals, evaluates exact PySCF RHF and FCI/CASCI ground state energies,
and synthesizes qubit Hamiltonians with Hartree-Fock reference states.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hvk.chemistry.jordan_wigner import integrals_to_qubit_hamiltonian
from hvk.chemistry.pauli import PauliSum


@dataclass(frozen=True)
class MolecularSystem:
    """Complete specification of a molecular system and its qubit Hamiltonian.

    Attributes
    ----------
    name : str
        Human-readable molecule identifier (e.g., 'H2', 'LiH').
    geometry_str : str
        PySCF-compatible coordinate string.
    bond_distance : float
        Characteristic bond length in Angstroms.
    basis : str
        Atomic orbital basis set (e.g., 'sto-3g').
    n_qubits : int
        Number of qubits in Jordan-Wigner mapped space (2 * n_spatial_active).
    n_electrons : int
        Number of active electrons.
    e_hf_pyscf : float
        PySCF Hartree-Fock ground state energy in Hartree.
    e_fci_pyscf : float
        PySCF Full CI (or CASCI) ground state energy in Hartree.
    e_core : float
        Nuclear repulsion plus frozen-core scalar energy shift.
    h1e : np.ndarray
        Active 1-body integrals in spatial MO basis.
    h2e : np.ndarray
        Active 2-body integrals in chemists' notation (pq|rs).
    qubit_hamiltonian : PauliSum
        Jordan-Wigner mapped qubit Hamiltonian.
    hf_statevector : np.ndarray
        Normalized statevector representing the Hartree-Fock Slater determinant.
    hf_bitstring : str
        Binary occupation string for occupied spin-orbitals.
    """

    name: str
    geometry_str: str
    bond_distance: float
    basis: str
    n_qubits: int
    n_electrons: int
    e_hf_pyscf: float
    e_fci_pyscf: float
    e_core: float
    h1e: np.ndarray
    h2e: np.ndarray
    qubit_hamiltonian: PauliSum
    hf_statevector: np.ndarray
    hf_bitstring: str


def build_molecular_system(
    atom_spec: str,
    name: str,
    bond_distance: float,
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    active_space: tuple[int, int] | None = None,
) -> MolecularSystem:
    """Build a molecular system using PySCF and synthesize its qubit Hamiltonian.

    Parameters
    ----------
    atom_spec : str
        Atomic coordinates in PySCF format (e.g. 'H 0 0 0; H 0 0 0.7414').
    name : str
        Molecule label.
    bond_distance : float
        Bond distance in Angstroms.
    basis : str
        Basis set, default 'sto-3g'.
    charge : int
        Molecular charge, default 0.
    spin : int
        2S = N_alpha - N_beta, default 0 (singlet).
    active_space : tuple of (n_cas_electrons, n_cas_orbitals), optional
        Complete active space parameters. If None, uses all orbitals.

    Returns
    -------
    MolecularSystem
        Instantiated dataclass containing validated energies and qubit Hamiltonian.
    """
    try:
        from pyscf import ao2mo, fci, gto, mcscf, scf
    except ImportError as exc:
        raise ImportError(
            "PySCF is required for building molecular Hamiltonians. "
            "Install with `pip install hamiltonian-vqe-kit[chemistry]`"
        ) from exc

    mol = gto.M(
        atom=atom_spec,
        basis=basis,
        charge=charge,
        spin=spin,
        verbose=0,
    )

    mf = scf.RHF(mol).run()
    if not mf.converged:
        raise RuntimeError(f"PySCF RHF calculation failed to converge for {name}")

    e_hf = float(mf.e_tot)

    if active_space is None:
        # Full orbital space
        n_spatial = mol.nao
        n_elec = mol.nelectron
        cisolver = fci.FCI(mf)
        e_fci_res = cisolver.kernel()
        e_fci = float(e_fci_res[0] if isinstance(e_fci_res, tuple) else e_fci_res)

        h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
        h2e_raw = ao2mo.kernel(mol, mf.mo_coeff)
        h2e = ao2mo.restore(1, h2e_raw, n_spatial)
        e_core = float(mf.energy_nuc())
    else:
        # Complete Active Space (CASCI)
        n_cas_elec, n_cas_orb = active_space
        n_spatial = n_cas_orb
        n_elec = n_cas_elec

        cas = mcscf.CASCI(mf, ncas=n_cas_orb, nelecas=n_cas_elec)
        cas.fcisolver.spin = spin
        e_cas_res = cas.kernel()
        e_fci = float(e_cas_res[0] if isinstance(e_cas_res, tuple) else e_cas_res)

        h1eff, e_c = cas.get_h1eff()
        h2eff_raw = cas.get_h2eff()
        h1e = h1eff
        h2e = ao2mo.restore(1, h2eff_raw, n_cas_orb)
        e_core = float(e_c)

    n_qubits = 2 * n_spatial
    qubit_hamiltonian = integrals_to_qubit_hamiltonian(
        one_body=h1e,
        two_body=h2e,
        constant=e_core,
        n_qubits=n_qubits,
    )

    # Construct Hartree-Fock statevector:
    # First n_elec spin-orbitals are occupied.
    # In Jordan-Wigner with alternating spin (or spin-blocked in OpenFermion),
    # the lowest n_elec spin-orbitals are 0 .. n_elec - 1.
    # In big-endian computational basis, qubits 0 .. n_elec-1 have bit 1:
    hf_idx = 0
    bit_chars: list[str] = []
    for q in range(n_qubits):
        if q < n_elec:
            hf_idx |= 1 << (n_qubits - 1 - q)
            bit_chars.append("1")
        else:
            bit_chars.append("0")

    hf_bitstring = "".join(bit_chars)
    hf_statevector = np.zeros(1 << n_qubits, dtype=complex)
    hf_statevector[hf_idx] = 1.0

    return MolecularSystem(
        name=name,
        geometry_str=atom_spec,
        bond_distance=bond_distance,
        basis=basis,
        n_qubits=n_qubits,
        n_electrons=n_elec,
        e_hf_pyscf=e_hf,
        e_fci_pyscf=e_fci,
        e_core=e_core,
        h1e=h1e,
        h2e=h2e,
        qubit_hamiltonian=qubit_hamiltonian,
        hf_statevector=hf_statevector,
        hf_bitstring=hf_bitstring,
    )


def load_h2(bond_distance: float = 0.7414, basis: str = "sto-3g") -> MolecularSystem:
    """Load H2 molecule at specified bond distance in Angstroms."""
    atom_spec = f"H 0 0 0; H 0 0 {bond_distance}"
    return build_molecular_system(
        atom_spec=atom_spec,
        name="H2",
        bond_distance=bond_distance,
        basis=basis,
        active_space=None,  # 2 electrons, 2 spatial orbitals -> 4 qubits
    )


def load_lih(
    bond_distance: float = 1.595,
    basis: str = "sto-3g",
    active_space: tuple[int, int] = (2, 3),
) -> MolecularSystem:
    """Load LiH molecule with frozen-core CAS(2e, 3o) active space (6 qubits)."""
    atom_spec = f"Li 0 0 0; H 0 0 {bond_distance}"
    return build_molecular_system(
        atom_spec=atom_spec,
        name="LiH",
        bond_distance=bond_distance,
        basis=basis,
        active_space=active_space,  # freeze 1s on Li -> 6 qubits
    )


def load_beh2(
    bond_distance: float = 1.326,
    basis: str = "sto-3g",
    active_space: tuple[int, int] = (2, 3),
) -> MolecularSystem:
    """Load linear BeH2 molecule with frozen-core CAS(2e, 3o) active space (6 qubits)."""
    atom_spec = f"H 0 0 -{bond_distance}; Be 0 0 0; H 0 0 {bond_distance}"
    return build_molecular_system(
        atom_spec=atom_spec,
        name="BeH2",
        bond_distance=bond_distance,
        basis=basis,
        active_space=active_space,  # 6 qubits
    )


def load_h4(bond_distance: float = 1.0, basis: str = "sto-3g") -> MolecularSystem:
    """Load linear H4 chain at uniform interatomic distance in Angstroms (8 qubits)."""
    coords = [0.0, bond_distance, 2 * bond_distance, 3 * bond_distance]
    atom_spec = "; ".join(f"H 0 0 {z}" for z in coords)
    return build_molecular_system(
        atom_spec=atom_spec,
        name="H4_chain",
        bond_distance=bond_distance,
        basis=basis,
        active_space=None,  # 4 electrons, 4 spatial orbitals -> 8 qubits
    )


def load_molecule(
    molecule_name: str,
    bond_distance: float | None = None,
    basis: str = "sto-3g",
) -> MolecularSystem:
    """Generic loader for standard benchmark molecules by name.

    Supported names: 'H2', 'LiH', 'BeH2', 'H4' / 'H4_chain'.
    """
    mol_norm = molecule_name.upper().replace("-", "_").replace(" ", "_")
    if mol_norm == "H2":
        r = bond_distance if bond_distance is not None else 0.7414
        return load_h2(bond_distance=r, basis=basis)
    elif mol_norm == "LIH":
        r = bond_distance if bond_distance is not None else 1.595
        return load_lih(bond_distance=r, basis=basis)
    elif mol_norm in ("BEH2", "BE_H2"):
        r = bond_distance if bond_distance is not None else 1.326
        return load_beh2(bond_distance=r, basis=basis)
    elif mol_norm in ("H4", "H4_CHAIN"):
        r = bond_distance if bond_distance is not None else 1.0
        return load_h4(bond_distance=r, basis=basis)
    else:
        raise ValueError(f"Unknown molecule name '{molecule_name}'. Supported: H2, LiH, BeH2, H4.")
