from __future__ import annotations

import logging

import requests

from .models import WebResult


LOGGER = logging.getLogger(__name__)


class WebSearchService:
    def __init__(
        self,
        timeout: int = 30,
        tavily_api_key: str = "",
        tavily_base_url: str = "https://api.tavily.com",
        search_depth: str = "basic",
    ):
        self.timeout = timeout
        self.tavily_api_key = tavily_api_key.strip()
        self.tavily_base_url = tavily_base_url.rstrip("/")
        self.search_depth = search_depth
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "KnowledgePilot/1.0 (knowledge assistant)"}
        )

    def search(
        self,
        query: str,
        limit: int = 5,
        *,
        search_depth: str | None = None,
        country: str = "",
        include_domains: list[str] | None = None,
        minimum_score: float = 0.0,
    ) -> list[WebResult]:
        query = query.strip()
        if not query:
            return []
        if not self.tavily_api_key:
            LOGGER.warning(
                "未配置 TAVILY_API_KEY，跳过联网搜索，"
                "不会使用低质量搜索结果兜底。"
            )
            return []
        return self._tavily(
            query,
            limit,
            search_depth=search_depth,
            country=country,
            include_domains=include_domains,
            minimum_score=minimum_score,
        )

    def search_many(
        self,
        queries: list[str],
        per_query: int = 3,
        total_limit: int = 8,
        *,
        search_depth: str | None = None,
        country: str = "",
        include_domains: list[str] | None = None,
        minimum_score: float = 0.0,
    ) -> list[WebResult]:
        results_by_identity: dict[str, WebResult] = {}
        for query in queries:
            if not query.strip():
                continue
            for item in self.search(
                query,
                limit=per_query,
                search_depth=search_depth,
                country=country,
                include_domains=include_domains,
                minimum_score=minimum_score,
            ):
                identity = (item.url or item.title).strip().lower()
                if not identity or not item.snippet.strip():
                    continue
                existing = results_by_identity.get(identity)
                if existing is None or _score(item) > _score(existing):
                    results_by_identity[identity] = item
        return sorted(
            results_by_identity.values(),
            key=_score,
            reverse=True,
        )[:total_limit]

    def _tavily(
        self,
        query: str,
        limit: int,
        *,
        search_depth: str | None,
        country: str,
        include_domains: list[str] | None,
        minimum_score: float,
    ) -> list[WebResult]:
        try:
            effective_depth = search_depth or self.search_depth
            payload = {
                "query": query,
                "search_depth": effective_depth,
                "max_results": max(1, min(limit, 10)),
                "include_answer": False,
                "include_raw_content": False,
            }
            if effective_depth == "advanced":
                payload["chunks_per_source"] = 3
            if country:
                payload["country"] = country
            if include_domains:
                payload["include_domains"] = include_domains[:5]
            response = self.session.post(
                f"{self.tavily_base_url}/search",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.tavily_api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            results = []
            for item in data.get("results", []):
                title = str(item.get("title") or "").strip()
                url = str(item.get("url") or "").strip()
                snippet = str(item.get("content") or "").strip()
                if not title or not url or not snippet:
                    continue
                score_value = item.get("score")
                score = (
                    float(score_value)
                    if isinstance(score_value, (int, float))
                    else None
                )
                if minimum_score > 0 and (
                    score is None or score < minimum_score
                ):
                    continue
                results.append(
                    WebResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        score=score,
                    )
                )
            LOGGER.info(
                "Tavily 搜索完成：query=%r request_id=%s 返回=%d 合格=%d "
                "最低相关度=%.2f",
                query,
                data.get("request_id", ""),
                len(data.get("results", [])),
                len(results),
                minimum_score,
            )
            return results
        except (
            requests.RequestException,
            ValueError,
            TypeError,
            KeyError,
        ) as exc:
            LOGGER.warning("Tavily 搜索失败：%s", exc)
        return []


def _score(item: WebResult) -> float:
    return item.score if item.score is not None else -1.0
