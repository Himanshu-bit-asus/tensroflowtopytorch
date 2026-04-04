"""Tests for common utilities."""
import pytest


def test_gcs_imports():
    """Test that gcs module can be imported."""
    from services.common.gcs import GcsPaths, _parse_gs_uri
    assert GcsPaths is not None
    assert _parse_gs_uri is not None


def test_tf_load_imports():
    """Test that tf_load module can be imported."""
    from services.common.tf_load import load_tf_model_from_gcs
    assert load_tf_model_from_gcs is not None


def test_parse_gs_uri():
    """Test GCS URI parsing."""
    from services.common.gcs import _parse_gs_uri
    
    bucket, blob = _parse_gs_uri("gs://my-bucket/path/to/file.json")
    assert bucket == "my-bucket"
    assert blob == "path/to/file.json"


def test_parse_gs_uri_invalid():
    """Test GCS URI parsing with invalid input."""
    from services.common.gcs import _parse_gs_uri
    
    with pytest.raises(ValueError):
        _parse_gs_uri("http://invalid-uri")
