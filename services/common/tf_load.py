from __future__ import annotations

import os
import tempfile
from typing import Tuple

import tensorflow as tf
from google.cloud import storage


def _parse_gs_uri(gs_uri: str) -> Tuple[str, str]:
    if not gs_uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {gs_uri}")
    _, _, rest = gs_uri.partition("gs://")
    bucket, _, blob = rest.partition("/")
    if not bucket or not blob:
        raise ValueError(f"Invalid gs:// URI: {gs_uri}")
    return bucket, blob


def _download_gcs_prefix(gs_uri_prefix: str, local_dir: str) -> str:
    client = storage.Client()
    bucket_name, prefix = _parse_gs_uri(gs_uri_prefix)
    bucket = client.bucket(bucket_name)

    # Ensure prefix ends with '/' to avoid partial matches.
    if not prefix.endswith("/"):
        prefix = prefix + "/"

    blobs = list(client.list_blobs(bucket, prefix=prefix))
    if not blobs:
        raise FileNotFoundError(f"No objects found under prefix: {gs_uri_prefix}")

    for blob in blobs:
        rel = blob.name[len(prefix) :]
        if not rel:
            continue
        dest = os.path.join(local_dir, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        blob.download_to_filename(dest)
    return local_dir


def load_tf_model_from_gcs(tf_model_uri: str) -> tf.types.experimental.GenericFunction | tf.keras.Model:
    """
    Supports:
      - Keras single-file formats: .h5, .keras
      - SavedModel directory in GCS (prefix)
    """
    lower = tf_model_uri.lower()
    if lower.endswith(".h5") or lower.endswith(".keras"):
        from .gcs import download_to_temp

        local_path = download_to_temp(tf_model_uri, suffix=os.path.splitext(lower)[1])
        return tf.keras.models.load_model(local_path, compile=False)

    tmpdir = tempfile.mkdtemp(prefix="tf_savedmodel_")
    local_dir = _download_gcs_prefix(tf_model_uri, tmpdir)
    return tf.saved_model.load(local_dir)

