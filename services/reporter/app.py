from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from services.common.gcs import GcsPaths, download_json, upload_json


class ReportRequest(BaseModel):
    job_id: Optional[str] = None
    blueprint_uri: str
    model_spec_uri: str
    validation_report_uri: str
    artifacts_prefix_uri: str


class ReportResponse(BaseModel):
    job_id: str
    final_report_uri: str


app = FastAPI(title="Reporting Agent", version="0.1")


@app.post("/report", response_model=ReportResponse)
def report(req: ReportRequest) -> ReportResponse:
    job_id = req.job_id or uuid.uuid4().hex
    paths = GcsPaths(job_id=job_id, artifacts_prefix_uri=req.artifacts_prefix_uri)

    blueprint = download_json(req.blueprint_uri)
    model_spec = download_json(req.model_spec_uri)
    validation = download_json(req.validation_report_uri)

    warnings: List[str] = []
    warnings.extend(blueprint.get("warnings", []) if isinstance(blueprint, dict) else [])
    warnings.extend(validation.get("warnings", []) if isinstance(validation, dict) else [])

    final: Dict[str, Any] = {
        "job_id": job_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "tf_model_uri": blueprint.get("tf_model_uri"),
            "blueprint_uri": req.blueprint_uri,
            "model_spec_uri": req.model_spec_uri,
            "validation_report_uri": req.validation_report_uri,
        },
        "summary": {
            "pass": validation.get("pass"),
            "aggregate_metrics": validation.get("aggregate", validation.get("metrics")),
            "warnings_count": len(warnings),
        },
        "analysis": blueprint,
        "conversion": model_spec,
        "validation": validation,
        "warnings": warnings,
        "env": {"service": "reporter", "region": os.environ.get("K_SERVICE", "")},
    }

    final_report_uri = upload_json(final, paths.final_report_uri())
    return ReportResponse(job_id=job_id, final_report_uri=final_report_uri)

