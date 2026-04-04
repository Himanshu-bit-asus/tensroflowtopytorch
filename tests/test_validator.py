"""Tests for validator service."""
import pytest


def test_validator_imports():
    """Test that validator module can be imported."""
    from services.validator import app
    assert app is not None


def test_validator_app_exists():
    """Test that FastAPI app is created."""
    from services.validator.app import app
    assert app is not None
    assert hasattr(app, "post")
