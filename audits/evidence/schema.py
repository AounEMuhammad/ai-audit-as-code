# audits/evidence/schema.py
from pydantic import BaseModel, Field, HttpUrl
from typing import Optional

class EvidenceItem(BaseModel):
    id: str
    type: str  # model_card, data_card, lineage, test_report, etc.
    uri: HttpUrl
    sha256: str
    created_at: str
    producer: str  # CI job or actor
    signed_attestation: Optional[str] = Field(default=None)
