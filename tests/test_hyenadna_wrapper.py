"""Unit tests for the HyenaDNA wrapper.

These tests never download a model: they cover construction, device
fallback and the guard rails that run before load_model() is called.
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hyenadna_wrapper import HyenaDNAWrapper  # noqa: E402


@pytest.fixture()
def wrapper():
    return HyenaDNAWrapper(model_name="hyenadna-small-32k-seqlen", device="cpu")


def test_initialization_stores_model_name(wrapper):
    assert wrapper.model_name == "hyenadna-small-32k-seqlen"


def test_model_not_loaded_on_init(wrapper):
    assert wrapper.model is None
    assert wrapper.tokenizer is None


def test_cpu_device_requested(wrapper):
    assert wrapper.device == torch.device("cpu")


def test_cuda_request_falls_back_to_cpu_without_gpu():
    if torch.cuda.is_available():
        pytest.skip("GPU present")
    assert HyenaDNAWrapper(device="cuda").device == torch.device("cpu")


def test_model_info_reports_unloaded_state(wrapper):
    assert wrapper.get_model_info() == {"error": "Model not loaded"}


def test_tokenize_requires_loaded_model(wrapper):
    with pytest.raises(ValueError, match="Model not loaded"):
        wrapper.tokenize_sequence("ACGT")
