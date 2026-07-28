from __future__ import annotations

from pathlib import Path

import pytest

from knowledge_pilot.config import AppConfig


@pytest.fixture
def app_config(tmp_path: Path) -> AppConfig:
    config = AppConfig(
        project_root=tmp_path,
        app_name="TestPilot",
        llm_provider="extractive",
        deepseek_api_key="",
        deepseek_base_url="https://api.deepseek.com",
        deepseek_model="deepseek-chat",
        tavily_api_key="",
        tavily_base_url="https://api.tavily.com",
        tavily_search_depth="basic",
        ollama_base_url="http://localhost:11434",
        ollama_model="qwen2.5:7b",
        chunk_size=120,
        chunk_overlap=20,
        retrieval_top_k=3,
        retrieval_score_threshold=0.08,
        enable_web_search=True,
        request_timeout=2,
        log_level="INFO",
    )
    config.ensure_directories()
    return config
