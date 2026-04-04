"""Tests for analyzer service."""
import pytest


def test_analyzer_imports():
    """Test that analyzer module can be imported."""
    from services.analyzer import app
    assert app is not None


def test_analyzer_app_exists():
    """Test that FastAPI app is created."""
    from services.analyzer.app import app
    assert app is not None
    assert hasattr(app, "post")
