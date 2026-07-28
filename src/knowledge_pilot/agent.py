from __future__ import annotations

from .config import AppConfig
from .llm import AnswerGenerator
from .models import AgentResult, AnswerSection, WebSearchPlan
from .retriever import LocalVectorRetriever
from .web_search import WebSearchService


WEB_RESULTS_PER_QUERY = 5
WEB_RESULTS_TOTAL_LIMIT = 8
WEB_SEARCH_MAX_ROUNDS = 2


class KnowledgeAgent:
    def __init__(
        self,
        config: AppConfig,
        retriever: LocalVectorRetriever,
        web_search: WebSearchService,
        generator: AnswerGenerator,
    ):
        self.config = config
        self.retriever = retriever
        self.web_search = web_search
        self.generator = generator

    def answer(self, question: str, allow_web: bool = True) -> AgentResult:
        question = question.strip()
        trace = ["接收到用户问题", "调用模型进行意图识别"]
        intent = self.generator.classify_intent(question)
        trace.append(
            f"意图：{intent.intent}；置信度：{intent.confidence:.2f}；"
            f"原因：{intent.reason}"
        )

        if intent.intent == "clarification":
            return AgentResult(
                answer="请补充一个更完整、明确的问题，我才能继续检索和回答。",
                route="clarification",
                confidence=intent.confidence,
                intent=intent,
                trace=trace + ["请求用户补充问题"],
            )

        if intent.intent == "identity":
            generation = self.generator.generate_identity()
            return AgentResult(
                answer=generation.text,
                route="identity",
                confidence=intent.confidence,
                intent=intent,
                trace=trace + ["读取 KnowledgePilot 项目身份资料并回答"],
                warning=generation.warning,
            )

        if intent.intent == "smalltalk":
            generation = self.generator.generate_smalltalk(question)
            answer = generation.text or "你好！我是 KnowledgePilot，很高兴为你服务。"
            return AgentResult(
                answer=answer,
                route="smalltalk",
                confidence=intent.confidence,
                intent=intent,
                trace=trace + ["使用对话提示词生成闲聊回答"],
                warning=generation.warning,
            )

        query = intent.rewritten_query or question
        trace.append(f"使用规范化问题检索知识库：{query}")
        retrieved = self.retriever.search(
            query, top_k=self.config.retrieval_top_k
        )
        top_score = retrieved[0].score if retrieved else 0.0
        trace.append(
            f"知识库返回 {len(retrieved)} 个片段；最高相关度：{top_score:.4f}"
        )
        trace.append("调用模型判断知识库覆盖度并生成知识库部分")
        assessment = self.generator.assess_knowledge(question, retrieved)
        trace.append(
            f"知识库覆盖率：{assessment.coverage:.2f}；"
            f"是否足够：{assessment.sufficient}；原因：{assessment.reason}"
        )

        knowledge_sources = [item.to_source() for item in retrieved]
        knowledge_section = AnswerSection(
            section_type="knowledge_base",
            title="企业知识库内容",
            content=assessment.knowledge_answer,
            sources=knowledge_sources,
        )

        if assessment.sufficient:
            trace.append("知识库足以回答，不调用联网搜索")
            return AgentResult(
                answer=assessment.knowledge_answer,
                route="knowledge_base",
                confidence=max(intent.confidence, assessment.coverage),
                sources=knowledge_sources,
                sections=[knowledge_section],
                intent=intent,
                trace=trace,
                warning=assessment.warning,
            )

        if not allow_web or not self.config.enable_web_search:
            warning = _join_warnings(
                assessment.warning,
                "知识库信息不足，且联网补充已关闭。",
            )
            trace.append("知识库信息不足，但联网补充已关闭")
            return AgentResult(
                answer=assessment.knowledge_answer,
                route="knowledge_base_limited",
                confidence=assessment.coverage,
                sources=knowledge_sources,
                sections=[knowledge_section],
                intent=intent,
                trace=trace,
                warning=warning,
            )

        search_plan = self.generator.plan_web_search(
            question,
            assessment.missing_points,
        )
        trace.append(
            f"知识库不足，DeepSeek 生成 {len(search_plan.queries)} 个搜索词；"
            f"策略：{search_plan.reason}"
        )
        trace.append(f"实际搜索词：{' | '.join(search_plan.queries)}")
        web_results = self._execute_web_search(search_plan)
        trace.append(f"联网搜索返回 {len(web_results)} 个去重结果")

        if not web_results and WEB_SEARCH_MAX_ROUNDS > 1:
            retry_plan = self.generator.plan_web_search(
                question,
                assessment.missing_points,
                previous_queries=search_plan.queries,
            )
            if retry_plan.queries != search_plan.queries:
                trace.append(
                    "首轮没有达到相关度阈值，DeepSeek 已改写搜索词重试"
                )
                trace.append(
                    f"重试搜索词：{' | '.join(retry_plan.queries)}"
                )
                web_results = self._execute_web_search(retry_plan)
                trace.append(
                    f"重试联网搜索返回 {len(web_results)} 个去重结果"
                )

        if not web_results:
            web_section = AnswerSection(
                section_type="web",
                title="互联网补充内容",
                content="联网搜索没有返回可用于补充回答的有效内容。",
                sources=[],
            )
            warning = _join_warnings(
                assessment.warning,
                "知识库信息不足，联网搜索也未返回有效结果。",
            )
            return AgentResult(
                answer=_compose_sections([knowledge_section, web_section]),
                route="knowledge_base_limited",
                confidence=assessment.coverage,
                sources=knowledge_sources,
                sections=[knowledge_section, web_section],
                intent=intent,
                trace=trace + ["保留知识库部分并标记联网补充失败"],
                warning=warning,
            )

        trace.append("使用独立的联网提示词生成互联网补充部分")
        web_generation = self.generator.generate_web_answer(
            question,
            assessment.missing_points,
            web_results,
        )
        web_sources = [item.to_source() for item in web_results]
        web_section = AnswerSection(
            section_type="web",
            title="互联网补充内容",
            content=web_generation.text,
            sources=web_sources,
        )
        sections = [knowledge_section, web_section]
        trace.append("分别组装知识库部分和互联网部分，来源保持隔离")
        return AgentResult(
            answer=_compose_sections(sections),
            route="knowledge_plus_web",
            confidence=max(0.65, assessment.coverage),
            sources=knowledge_sources + web_sources,
            sections=sections,
            intent=intent,
            trace=trace,
            warning=_join_warnings(assessment.warning, web_generation.warning),
        )

    def _execute_web_search(
        self,
        plan: WebSearchPlan,
    ):
        return self.web_search.search_many(
            plan.queries,
            per_query=WEB_RESULTS_PER_QUERY,
            total_limit=WEB_RESULTS_TOTAL_LIMIT,
            search_depth=plan.search_depth,
            country=plan.country,
            include_domains=plan.include_domains,
            minimum_score=plan.minimum_score,
        )


def _compose_sections(sections: list[AnswerSection]) -> str:
    return "\n\n".join(
        f"### {section.title}\n\n{section.content}" for section in sections
    )


def _join_warnings(*warnings: str) -> str:
    return "；".join(item.strip("； ") for item in warnings if item.strip())
