import json
from dataclasses import replace

from knowledge_pilot.llm import AnswerGenerator
from knowledge_pilot.models import RetrievalResult, TextChunk


class FakeResponse:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": self.content}}]}


class FakeToolResponse:
    def __init__(self, arguments):
        self.arguments = arguments

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "plan_web_search",
                                    "arguments": json.dumps(
                                        self.arguments,
                                        ensure_ascii=False,
                                    ),
                                }
                            }
                        ],
                    }
                }
            ]
        }


def test_deepseek_uses_api_key_and_compatible_endpoint(
    monkeypatch, app_config
):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured.update(
            url=url,
            headers=headers,
            payload=json,
            timeout=timeout,
        )
        return FakeResponse("根据员工手册，正式员工有五天年假。")

    monkeypatch.setattr("knowledge_pilot.llm.requests.post", fake_post)
    config = replace(
        app_config,
        llm_provider="deepseek",
        deepseek_api_key="test-secret",
    )
    generation = AnswerGenerator(config).generate(
        "年假有几天？",
        ["正式员工每年享有五天带薪年假。"],
        "企业知识库",
    )

    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-secret"
    assert captured["payload"]["model"] == "deepseek-chat"
    assert "五天年假" in generation.text


def test_deepseek_classifies_intent_from_json(monkeypatch, app_config):
    payload = {
        "intent": "identity",
        "rewritten_query": "请介绍 KnowledgePilot",
        "reason": "用户询问项目身份",
        "confidence": 0.97,
    }
    monkeypatch.setattr(
        "knowledge_pilot.llm.requests.post",
        lambda *args, **kwargs: FakeResponse(json.dumps(payload)),
    )
    config = replace(
        app_config,
        llm_provider="deepseek",
        deepseek_api_key="test-secret",
    )
    decision = AnswerGenerator(config).classify_intent("你是谁？")
    assert decision.intent == "identity"
    assert decision.confidence == 0.97


def test_knowledge_assessment_returns_missing_points(
    monkeypatch, app_config
):
    payload = {
        "sufficient": False,
        "coverage": 0.45,
        "knowledge_answer": "知识库只说明公司年假为五天。",
        "missing_points": ["国家法定年假标准"],
        "reason": "只能回答公司内部规定",
    }
    monkeypatch.setattr(
        "knowledge_pilot.llm.requests.post",
        lambda *args, **kwargs: FakeResponse(json.dumps(payload)),
    )
    config = replace(
        app_config,
        llm_provider="deepseek",
        deepseek_api_key="test-secret",
    )
    chunk = TextChunk(
        "c1", "d1", "员工手册.md", 1, "公司年假为五天。", "hash"
    )
    assessment = AnswerGenerator(config).assess_knowledge(
        "公司年假和法定标准是什么？",
        [RetrievalResult(chunk, 0.5)],
    )
    assert assessment.sufficient is False
    assert assessment.missing_points == ["国家法定年假标准"]


def test_deepseek_plans_tavily_search_with_tool_call(
    monkeypatch, app_config
):
    captured = {}
    arguments = {
        "queries": [
            "中国《职工带薪年休假条例》第三条 5天 10天 15天"
        ],
        "search_depth": "advanced",
        "country": "china",
        "include_domains": ["gov.cn", "mohrss.gov.cn"],
        "minimum_score": 0.5,
        "reason": "法律问题优先检索中国政府网站",
    }

    def fake_post(url, headers, json, timeout):
        captured.update(url=url, headers=headers, payload=json, timeout=timeout)
        return FakeToolResponse(arguments)

    monkeypatch.setattr("knowledge_pilot.llm.requests.post", fake_post)
    config = replace(
        app_config,
        llm_provider="deepseek",
        deepseek_api_key="test-secret",
    )
    plan = AnswerGenerator(config).plan_web_search(
        "公司年假制度和国家基准年假标准分别是什么？",
        ["知识库未提供国家基准年假标准的具体内容"],
    )

    assert plan.queries == arguments["queries"]
    assert plan.search_depth == "advanced"
    assert plan.country == "china"
    assert plan.include_domains == ["gov.cn", "mohrss.gov.cn"]
    assert plan.minimum_score == 0.5
    assert captured["payload"]["tools"][0]["function"]["name"] == (
        "plan_web_search"
    )
    assert "tool_choice" not in captured["payload"]


def test_search_plan_fallback_keeps_original_question_context(app_config):
    plan = AnswerGenerator(app_config).plan_web_search(
        "公司年假制度和国家基准年假标准分别是什么？",
        ["知识库未提供国家基准年假标准的具体内容"],
    )
    assert plan.queries == [
        "公司年假制度和国家基准年假标准分别是什么？ "
        "知识库未提供国家基准年假标准的具体内容"
    ]


def test_invalid_router_json_falls_back_to_identity(monkeypatch, app_config):
    monkeypatch.setattr(
        "knowledge_pilot.llm.requests.post",
        lambda *args, **kwargs: FakeResponse("不是 JSON"),
    )
    config = replace(
        app_config,
        llm_provider="deepseek",
        deepseek_api_key="test-secret",
    )
    decision = AnswerGenerator(config).classify_intent("请问你是谁")
    assert decision.intent == "identity"
    assert "本地规则" in decision.reason


def test_deepseek_without_key_falls_back_to_extractive(app_config):
    config = replace(app_config, llm_provider="deepseek", deepseek_api_key="")
    generation = AnswerGenerator(config).generate(
        "年假有几天？",
        ["正式员工每年享有五天带薪年假。"],
        "企业知识库",
    )
    assert "五天带薪年假" in generation.text
    assert "本地抽取式回答" in generation.warning
