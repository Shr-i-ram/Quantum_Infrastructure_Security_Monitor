"""Measurement observables and latent vector extraction for PI-TQSVDD."""

from __future__ import annotations

from typing import Sequence
import pennylane as qml


def get_pauli_z_measurements(wires: Sequence[int]):
    """Return Pauli-Z expectation values for all specified wires."""
    return [qml.expval(qml.PauliZ(w)) for w in wires]


def get_all_pauli_measurements(wires: Sequence[int]):
    """Return Pauli-X, Y, Z expectation values for enriched representation."""
    obs = []
    for w in wires:
        obs.append(qml.expval(qml.PauliX(w)))
        obs.append(qml.expval(qml.PauliY(w)))
        obs.append(qml.expval(qml.PauliZ(w)))
    return obs
