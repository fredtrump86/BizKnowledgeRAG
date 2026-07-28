from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    app_name: str
    llm_provider: str
    deepseek_api_key: str
    deepseek_base_url: str
    deepseek_model: str
    tavily_api_key: str
    tavily_base_url: str
    tavily_search_depth: str
    ollama_base_url: str
    ollama_model: str
    chunk_size: int
    chunk_overlap: int
    retrieval_top_k: int
    retrieval_score_threshold: float
    enable_web_search: bool
    request_timeout: int
    log_level: str

    @property
    def knowledge_base_dir(self) -> Path:
        return self.project_root / "data" / "knowledge_base"

    @property
    def index_dir(self) -> Path:
        return self.project_root / "data" / "index"

    @property
    def database_path(self) -> Path:
        return self.project_root / "data" / "app.db"

    @property
    def log_dir(self) -> Path:
        return self.project_root / "logs"

    @classmethod
    def load(cls, project_root: Path | None = None) -> "AppConfig":
        root = (project_root or PROJECT_ROOT).resolve()
        load_dotenv(root / ".env")
        chunk_size = int(os.getenv("CHUNK_SIZE", "700"))
        overlap = int(os.getenv("CHUNK_OVERLAP", "100"))
        if overlap >= chunk_size:
            raise ValueError("CHUNK_OVERLAP 必须小于 CHUNK_SIZE")
        tavily_search_depth = os.getenv(
            "TAVILY_SEARCH_DEPTH", "basic"
        ).strip().lower()
        if tavily_search_depth not in {
            "basic",
            "advanced",
            "fast",
            "ultra-fast",
        }:
            raise ValueError(
                "TAVILY_SEARCH_DEPTH 必须是 basic、advanced、"
                "fast 或 ultra-fast"
            )

        config = cls(
            project_root=root,
            app_name=os.getenv("APP_NAME", "KnowledgePilot"),
            llm_provider=os.getenv("LLM_PROVIDER", "deepseek").strip().lower(),
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", "").strip(),
            deepseek_base_url=os.getenv(
                "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
            ).rstrip("/"),
            deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            tavily_api_key=os.getenv("TAVILY_API_KEY", "").strip(),
            tavily_base_url=os.getenv(
                "TAVILY_BASE_URL", "https://api.tavily.com"
            ).rstrip("/"),
            tavily_search_depth=tavily_search_depth,
            ollama_base_url=os.getenv(
                "OLLAMA_BASE_URL", "http://localhost:11434"
            ).rstrip("/"),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "5")),
            retrieval_score_threshold=float(
                os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.08")
            ),
            enable_web_search=_as_bool(os.getenv("ENABLE_WEB_SEARCH"), True),
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", "30")),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )
        config.ensure_directories()
        return config

    def ensure_directories(self) -> None:
        for path in (
            self.knowledge_base_dir,
            self.knowledge_base_dir / "uploads",
            self.index_dir,
            self.log_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
