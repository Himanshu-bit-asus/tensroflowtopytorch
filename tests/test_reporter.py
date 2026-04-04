"""Tests for reporter service."""
import pytest


def test_reporter_imports():
    """Test that reporter module can be imported."""
    from services.reporter import app
    assert app is not None


def test_reporter_app_exists():
    """Test that FastAPI app is created."""
    from services.reporter.app import app
    assert app is not None
    assert hasattr(app, "post")
