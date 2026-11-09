import importlib

import sparc.src.deepmd as deepmd_module


def test_get_backend_uses_tensorflow_when_pytorch_is_unavailable(monkeypatch):
    def import_backend(module_name):
        if module_name == "deepmd.pt":
            raise ModuleNotFoundError("No module named 'torch'")
        if module_name == "deepmd.tf":
            return object()
        raise AssertionError(f"Unexpected import: {module_name}")

    monkeypatch.setattr(importlib, "import_module", import_backend)

    assert deepmd_module.get_backend() == "tensorflow"
