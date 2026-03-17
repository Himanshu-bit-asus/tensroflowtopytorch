# Service contract (shared)

All agent services accept and return JSON. **All durable artifacts are written to GCS** and referenced by URI.

## Common fields

- `job_id` (string): stable identifier across the workflow. If omitted by caller, services may create one.
- `tf_model_uri` (string): `gs://...` pointing to a SavedModel directory or `.h5` file.
- `artifacts_prefix_uri` (string): `gs://.../<prefix>` where outputs are written.

## Analyzer (`POST /analyze`)

### Request

```json
{
  "job_id": "optional",
  "tf_model_uri": "gs://bucket/path/to/saved_model/",
  "artifacts_prefix_uri": "gs://bucket/tf2pt-artifacts",
  "extra": { "tags": ["optional"] }
}
```

### Response

```json
{
  "job_id": "abc123",
  "blueprint_uri": "gs://bucket/tf2pt-artifacts/abc123/analysis/model_blueprint.json",
  "warnings": ["..."]
}
```

## Converter (`POST /convert`)

### Request

```json
{
  "job_id": "abc123",
  "tf_model_uri": "gs://bucket/path/to/saved_model/",
  "blueprint_uri": "gs://bucket/tf2pt-artifacts/abc123/analysis/model_blueprint.json",
  "artifacts_prefix_uri": "gs://bucket/tf2pt-artifacts"
}
```

### Response

```json
{
  "job_id": "abc123",
  "model_spec_uri": "gs://bucket/tf2pt-artifacts/abc123/convert/model_spec.json",
  "state_dict_uri": "gs://bucket/tf2pt-artifacts/abc123/convert/state_dict.pt",
  "warnings": ["unsupported layer: ..."]
}
```

## Validator (`POST /validate`)

### Request

```json
{
  "job_id": "abc123",
  "tf_model_uri": "gs://bucket/path/to/saved_model/",
  "model_spec_uri": "gs://bucket/tf2pt-artifacts/abc123/convert/model_spec.json",
  "state_dict_uri": "gs://bucket/tf2pt-artifacts/abc123/convert/state_dict.pt",
  "artifacts_prefix_uri": "gs://bucket/tf2pt-artifacts",
  "validation": {
    "num_trials": 3,
    "atol": 1e-4,
    "rtol": 1e-3,
    "seed": 0
  }
}
```

### Response

```json
{
  "job_id": "abc123",
  "validation_report_uri": "gs://bucket/tf2pt-artifacts/abc123/validate/validation_report.json",
  "pass": true,
  "metrics": { "max_abs_diff": 0.00012, "cosine_similarity": 0.9999 }
}
```

## Reporter (`POST /report`)

### Request

```json
{
  "job_id": "abc123",
  "blueprint_uri": "gs://bucket/.../analysis/model_blueprint.json",
  "model_spec_uri": "gs://bucket/.../convert/model_spec.json",
  "validation_report_uri": "gs://bucket/.../validate/validation_report.json",
  "artifacts_prefix_uri": "gs://bucket/tf2pt-artifacts"
}
```

### Response

```json
{
  "job_id": "abc123",
  "final_report_uri": "gs://bucket/tf2pt-artifacts/abc123/report/final_report.json"
}
```

