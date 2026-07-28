import requests

from knowledge_pilot.models import WebResult
from knowledge_pilot.web_search import WebSearchService


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_search_uses_tavily_and_maps_ranked_results(monkeypatch):
    service = WebSearchService(
        timeout=7,
        tavily_api_key="tavily-test-key",
        tavily_base_url="https://api.tavily.test",
        search_depth="basic",
    )
    captured = {}

    def fake_post(url, json, headers, timeout):
        captured.update(
            url=url,
            json=json,
            headers=headers,
            timeout=timeout,
        )
        return FakeResponse(
            {
                "results": [
                    {
                        "title": "职工带薪年休假条例",
                        "url": "https://www.gov.cn/zwgk/2007-12/16/content_835527.htm",
                        "content": "职工累计工作已满1年不满10年的，年休假5天。",
                        "score": 0.97,
                    },
                    {
                        "title": "无摘要结果",
                        "url": "https://example.com/empty",
                        "content": "",
                        "score": 0.4,
                    },
                ]
            }
        )

    monkeypatch.setattr(service.session, "post", fake_post)
    results = service.search("国家年假标准", limit=3)

    assert len(results) == 1
    assert results[0].title == "职工带薪年休假条例"
    assert results[0].score == 0.97
    assert captured["url"] == "https://api.tavily.test/search"
    assert captured["json"]["query"] == "国家年假标准"
    assert captured["json"]["max_results"] == 3
    assert captured["json"]["search_depth"] == "basic"
    assert captured["headers"]["Authorization"] == "Bearer tavily-test-key"
    assert captured["timeout"] == 7


def test_search_without_tavily_key_returns_no_untrusted_fallback():
    service = WebSearchService(timeout=1, tavily_api_key="")
    assert service.search("国家年假标准") == []


def test_search_returns_empty_list_when_tavily_fails(monkeypatch):
    service = WebSearchService(timeout=1, tavily_api_key="tavily-test-key")

    def fake_post(*args, **kwargs):
        raise requests.RequestException("network error")

    monkeypatch.setattr(service.session, "post", fake_post)
    assert service.search("国家年假标准") == []


def test_search_applies_deepseek_plan_and_filters_low_scores(monkeypatch):
    service = WebSearchService(
        timeout=7,
        tavily_api_key="tavily-test-key",
        tavily_base_url="https://api.tavily.test",
        search_depth="basic",
    )
    captured = {}

    def fake_post(url, json, headers, timeout):
        captured.update(
            url=url,
            json=json,
            headers=headers,
            timeout=timeout,
        )
        return FakeResponse(
            {
                "request_id": "request-123",
                "results": [
                    {
                        "title": "低质量页面",
                        "url": "https://example.com/weak",
                        "content": "与问题关系很弱。",
                        "score": 0.12,
                    },
                    {
                        "title": "职工带薪年休假条例",
                        "url": "https://www.gov.cn/example",
                        "content": "累计工作满1年不满10年，年休假5天。",
                        "score": 0.93,
                    },
                ],
            }
        )

    monkeypatch.setattr(service.session, "post", fake_post)
    results = service.search(
        "中国职工带薪年休假条例",
        limit=5,
        search_depth="advanced",
        country="china",
        include_domains=["gov.cn", "mohrss.gov.cn"],
        minimum_score=0.45,
    )

    assert [item.title for item in results] == ["职工带薪年休假条例"]
    assert captured["json"]["search_depth"] == "advanced"
    assert captured["json"]["chunks_per_source"] == 3
    assert captured["json"]["country"] == "china"
    assert captured["json"]["include_domains"] == [
        "gov.cn",
        "mohrss.gov.cn",
    ]


def test_search_many_deduplicates_results(monkeypatch):
    service = WebSearchService(timeout=1, tavily_api_key="tavily-test-key")

    def fake_search(query, limit=5, **options):
        return [
            WebResult("相同结果", "https://example.com/a", f"{query} 摘要"),
            WebResult("空摘要", "https://example.com/empty", ""),
        ]

    monkeypatch.setattr(service, "search", fake_search)
    results = service.search_many(["问题一", "问题二"])
    assert len(results) == 1
    assert results[0].url == "https://example.com/a"
