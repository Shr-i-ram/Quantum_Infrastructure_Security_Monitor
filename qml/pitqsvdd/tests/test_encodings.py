"""Tests for quantum data encodings across 2, 4, 6, and 8 qubits."""

import pytest
import torch

from qml.pitqsvdd.encodings import get_encoding


@pytest.mark.parametrize("n_q", [2, 4, 6, 8])
@pytest.mark.parametrize("enc_name", [
    "angle",
    "data_reuploading",
    "amplitude",
    "iqp_zz",
    "feature_group",
    "learned_projection",
])
def test_encoding_instantiation_and_metadata(n_q: int, enc_name: str):
    """Verify that every encoding initializes properly and returns valid metadata."""
    enc = get_encoding(enc_name, n_qubits=n_q)
    info = enc.get_info()
    assert info["name"] is not None
    assert info["n_qubits"] == n_q
