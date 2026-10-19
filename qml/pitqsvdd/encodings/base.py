"""Base interface for quantum data encodings in PI-TQSVDD."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence
import pennylane as qml
import torch


class BaseEncoding(ABC):
    """Abstract base class for quantum feature encodings."""

    def __init__(self, n_qubits: int, name: str = "base_encoding"):
        self.n_qubits = n_qubits
        self.name = name

    @abstractmethod
    def apply(self, inputs: torch.Tensor, wires: Sequence[int], **kwargs: Any) -> None:
        """Apply quantum encoding gates inside a PennyLane QNode.

        Args:
            inputs: 1D Tensor of input features for a single sample, or batch.
            wires: Sequence of qubit wire indices to apply the encoding on.
        """
        pass

    def get_info(self) -> dict[str, Any]:
        """Return encoding metadata and configuration."""
        return {
            "name": self.name,
            "n_qubits": self.n_qubits,
        }
