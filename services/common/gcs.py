from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Tuple

from google.cloud import storage


def _parse_gs_uri(gs_uri: str) -> Tuple[str, str]:
    if not gs_uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {gs_uri}")
    _, _, rest = gs_uri.partition("gs://")
    bucket, _, blob = rest.partition("/")
    if not bucket or not blob:
        raise ValueError(f"Invalid gs:// URI: {gs_uri}")
    return bucket, blob


@dataclass(frozen=True)
class GcsPaths:
    job_id: str
    artifacts_prefix_uri: str

    def analysis_blueprint_uri(self) -> str:
        return f"{self.artifacts_prefix_uri.rstrip('/')}/{self.job_id}/analysis/model_blueprint.json"

    def convert_model_spec_uri(self) -> str:
        return f"{self.artifacts_prefix_uri.rstrip('/')}/{self.job_id}/convert/model_spec.json"

    def convert_state_dict_uri(self) -> str:
        return f"{self.artifacts_prefix_uri.rstrip('/')}/{self.job_id}/convert/state_dict.pt"

    def validate_report_uri(self) -> str:
        return f"{self.artifacts_prefix_uri.rstrip('/')}/{self.job_id}/validate/validation_report.json"

    def final_report_uri(self) -> str:
        return f"{self.artifacts_prefix_uri.rstrip('/')}/{self.job_id}/report/final_report.json"


def gcs_client() -> storage.Client:
    # Uses Application Default Credentials on Cloud Run / local gcloud auth.
    return storage.Client()


def download_to_temp(gs_uri: str, suffix: str = "") -> str:
    client = gcs_client()
    bucket_name, blob_name = _parse_gs_uri(gs_uri)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    blob.download_to_filename(path)
    return path


def download_bytes(gs_uri: str) -> bytes:
    client = gcs_client()
    bucket_name, blob_name = _parse_gs_uri(gs_uri)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.download_as_bytes()


def upload_file(local_path: str, gs_uri: str, content_type: str | None = None) -> str:
    client = gcs_client()
    bucket_name, blob_name = _parse_gs_uri(gs_uri)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path, content_type=content_type)
    return gs_uri


def upload_json(obj: Any, gs_uri: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    return upload_file(path, gs_uri, content_type="application/json")


def download_json(gs_uri: str) -> Any:
    path = download_to_temp(gs_uri, suffix=".json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

