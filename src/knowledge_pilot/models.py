from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TextChunk:
    chunk_id: str
    document_id: str
    file_name: str
    page_number: int
    content: str
    content_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TextChunk":
        return cls(**data)


@dataclass
class RetrievalResult:
    chunk: TextChunk
    score: float

    def to_source(self) -> dict[str, Any]:
        return {
            "source_type": "knowledge_base",
            "title": self.chunk.file_name,
            "page": self.chunk.page_number,
            "content": self.chunk.content,
            "score": round(self.score, 4),
            "url": "",
        }


@dataclass
class WebResult:
    title: str
    url: str
    snippet: str
    score: float | None = None

    def to_source(self) -> dict[str, Any]:
        return {
            "source_type": "web",
            "title": self.title,
            "page": None,
            "content": self.snippet,
            "score": round(self.score, 4) if self.score is not None else None,
            "url": self.url,
        }


@dataclass
class WebSearchPlan:
    queries: list[str]
    search_depth: str = "advanced"
    country: str = ""
    include_domains: list[str] = field(default_factory=list)
    minimum_score: float = 0.35
    reason: str = ""


@dataclass
class RouteDecision:
    route: str
    reason: str
    confidence: float


@dataclass
class IntentDecision:
    intent: str
    rewritten_query: str
    reason: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class KnowledgeAssessment:
    sufficient: bool
    coverage: float
    knowledge_answer: str
    missing_points: list[str] = field(default_factory=list)
    reason: str = ""
    warning: str = ""


@dataclass
class AnswerSection:
    section_type: str
    title: str
    content: str
    sources: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentResult:
    answer: str
    route: str
    confidence: float
    sources: list[dict[str, Any]] = field(default_factory=list)
    sections: list[AnswerSection] = field(default_factory=list)
    intent: IntentDecision | None = None
    trace: list[str] = field(default_factory=list)
    warning: str = ""
