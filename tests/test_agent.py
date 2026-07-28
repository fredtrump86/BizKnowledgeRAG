from knowledge_pilot.agent import KnowledgeAgent
from knowledge_pilot.llm import Generation
from knowledge_pilot.models import (
    IntentDecision,
    KnowledgeAssessment,
    TextChunk,
    WebResult,
    WebSearchPlan,
)
from knowledge_pilot.retriever import LocalVectorRetriever


class StubWebSearch:
    def __init__(self):
        self.queries = []
        self.calls = []

    def search_many(
        self,
        queries,
        per_query=3,
        total_limit=8,
        **options,
    ):
        self.queries.extend(queries)
        self.calls.append(
            {
                "queries": queries,
                "per_query": per_query,
                "total_limit": total_limit,
                **options,
            }
        )
        return [
            WebResult(
                title="联网结果",
                url="https://example.com",
                snippet="公开资料中的补充信息",
            )
        ]


class StubGenerator:
    def __init__(self):
        self.plan_calls = []

    def classify_intent(self, question):
        if not question.strip():
            intent = "clarification"
        elif "谁" in question or "什么项目" in question:
            intent = "identity"
        elif "你好" in question:
            intent = "smalltalk"
        else:
            intent = "knowledge_query"
        return IntentDecision(intent, question, "测试路由", 0.95)

    def assess_knowledge(self, question, retrieved):
        if "法定标准" in question or "天气" in question:
            return KnowledgeAssessment(
                sufficient=False,
                coverage=0.4,
                knowledge_answer="知识库只能确认企业内部规定。",
                missing_points=["需要补充外部公开信息"],
                reason="知识库仅覆盖部分问题",
            )
        return KnowledgeAssessment(
            sufficient=True,
            coverage=0.9,
            knowledge_answer="正式员工每年享有五天带薪年假。",
            missing_points=[],
            reason="知识库完整覆盖",
        )

    def generate_smalltalk(self, question):
        return Generation("你好！我是 KnowledgePilot。")

    def generate_identity(self):
        return Generation("我是 KnowledgePilot 企业知识库 Agent。")

    def plan_web_search(
        self,
        question,
        missing_points,
        previous_queries=None,
    ):
        self.plan_calls.append(
            {
                "question": question,
                "missing_points": missing_points,
                "previous_queries": previous_queries or [],
            }
        )
        return WebSearchPlan(
            queries=["中国国家法定年假标准 职工带薪年休假条例"],
            search_depth="advanced",
            country="china",
            include_domains=["gov.cn", "mohrss.gov.cn"],
            minimum_score=0.45,
            reason="优先检索中国政府权威来源",
        )

    def generate_web_answer(self, question, missing_points, web_results):
        return Generation("根据互联网公开资料，可以补充外部规则。")


def make_retriever(tmp_path):
    retriever = LocalVectorRetriever(tmp_path / "index.joblib")
    retriever.build(
        [
            TextChunk(
                chunk_id="leave",
                document_id="doc",
                file_name="员工手册.md",
                page_number=1,
                content="正式员工每个自然年度享有五天带薪年假。",
                content_hash="hash",
            )
        ]
    )
    return retriever


def make_agent(tmp_path, app_config):
    web = StubWebSearch()
    generator = StubGenerator()
    agent = KnowledgeAgent(
        app_config,
        make_retriever(tmp_path),
        web,
        generator,
    )
    return agent, web


def test_agent_routes_identity_without_retrieval_or_web(tmp_path, app_config):
    agent, web = make_agent(tmp_path, app_config)
    result = agent.answer("你好，请问你是谁？")
    assert result.route == "identity"
    assert "KnowledgePilot" in result.answer
    assert web.queries == []


def test_agent_routes_greeting_to_smalltalk(tmp_path, app_config):
    agent, web = make_agent(tmp_path, app_config)
    result = agent.answer("你好")
    assert result.route == "smalltalk"
    assert web.queries == []


def test_agent_routes_relevant_question_to_knowledge_base(
    tmp_path, app_config
):
    agent, web = make_agent(tmp_path, app_config)
    result = agent.answer("正式员工有多少天年假？")
    assert result.route == "knowledge_base"
    assert result.sections[0].section_type == "knowledge_base"
    assert result.sections[0].sources[0]["source_type"] == "knowledge_base"
    assert "五天" in result.answer
    assert web.queries == []
    assert agent.generator.plan_calls == []


def test_agent_keeps_knowledge_and_web_sections_separate(
    tmp_path, app_config
):
    agent, web = make_agent(tmp_path, app_config)
    result = agent.answer("公司年假和国家法定标准分别是什么？")
    assert result.route == "knowledge_plus_web"
    assert [section.section_type for section in result.sections] == [
        "knowledge_base",
        "web",
    ]
    assert all(
        source["source_type"] == "knowledge_base"
        for source in result.sections[0].sources
    )
    assert all(
        source["source_type"] == "web"
        for source in result.sections[1].sources
    )
    assert web.queries == ["中国国家法定年假标准 职工带薪年休假条例"]
    assert web.calls[0]["search_depth"] == "advanced"
    assert web.calls[0]["country"] == "china"
    assert web.calls[0]["include_domains"] == ["gov.cn", "mohrss.gov.cn"]
    assert web.calls[0]["minimum_score"] == 0.45
    assert "实际搜索词：中国国家法定年假标准" in "\n".join(result.trace)


def test_agent_can_disable_web_supplement(tmp_path, app_config):
    agent, web = make_agent(tmp_path, app_config)
    result = agent.answer("今天北京天气如何？", allow_web=False)
    assert result.route == "knowledge_base_limited"
    assert len(result.sections) == 1
    assert web.queries == []


def test_agent_requests_clarification_for_empty_question(
    tmp_path, app_config
):
    agent, _ = make_agent(tmp_path, app_config)
    result = agent.answer("  ")
    assert result.route == "clarification"
