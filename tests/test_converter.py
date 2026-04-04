"""Tests for converter service."""
import pytest


def test_converter_imports():
    """Test that converter module can be imported."""
    from services.converter import app
    assert app is not None


def test_converter_app_exists():
    """Test that FastAPI app is created."""
    from services.converter.app import app
    assert app is not None
    assert hasattr(app, "post")


def test_torch_builder_imports():
    """Test that torch_builder module can be imported."""
    from services.converter.torch_builder import build_torch_from_model_spec
    assert build_torch_from_model_spec is not None
