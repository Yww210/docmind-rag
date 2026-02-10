from typing import List, Optional

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source: Optional[str] = "api"

class IngestResponse(BaseModel):
    chunks_indexed: int
    source: str

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    use_hyde: bool = False

class RetrievedChunk(BaseModel):
    source: str
    chunk_id: str
    score: float
    text: str

class QueryResponse(BaseModel):
    answer: str
    retrieved: List[RetrievedChunk]
