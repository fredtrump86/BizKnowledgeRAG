from __future__ import annotations

import logging
import sys
from dataclasses import replace
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from knowledge_pilot.agent import KnowledgeAgent
from knowledge_pilot.config import AppConfig
from knowledge_pilot.documents import DocumentService
from knowledge_pilot.history import HistoryStore
from knowledge_pilot.llm import AnswerGenerator
from knowledge_pilot.retriever import LocalVectorRetriever
from knowledge_pilot.web_search import WebSearchService


st.set_page_config(
    page_title="KnowledgePilot",
    page_icon="🧭",
    layout="wide",
)


@st.cache_resource
def initialize_services():
    config = AppConfig.load(PROJECT_ROOT)
    logging.basicConfig(
        level=getattr(logging, config.log_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(
                config.log_dir / "app.log", encoding="utf-8"
            ),
            logging.StreamHandler(),
        ],
    )
    documents = DocumentService(
        config.knowledge_base_dir,
        config.chunk_size,
        config.chunk_overlap,
    )
    retriever = LocalVectorRetriever(config.index_dir / "tfidf.joblib")
    if not retriever.load():
        retriever.build(documents.build_chunks())
    history = HistoryStore(config.database_path)
    return config, documents, retriever, history


def rebuild_index(documents: DocumentService, retriever: LocalVectorRetriever):
    with st.spinner("正在解析文档并重建向量索引……"):
        retriever.build(documents.build_chunks())


def render_sources(
    sources: list[dict],
    label: str = "查看参考来源",
) -> None:
    if not sources:
        return
    with st.expander(f"{label}（{len(sources)} 条）"):
        for index, source in enumerate(sources, start=1):
            prefix = (
                "KB" if source.get("source_type") == "knowledge_base" else "WEB"
            )
            page = (
                f" · 第 {source['page']} 页"
                if source.get("page")
                else ""
            )
            score = (
                f" · 相关度 {source['score']:.3f}"
                if source.get("score") is not None
                else ""
            )
            st.markdown(
                f"**[{prefix}-{index}] {source['title']}**{page}{score}"
            )
            if source.get("url"):
                st.markdown(f"[打开网页]({source['url']})")
            st.caption(source.get("content", ""))
            if index < len(sources):
                st.divider()


def render_answer_sections(sections: list[dict]) -> None:
    for section in sections:
        section_type = section.get("section_type", "")
        if section_type == "knowledge_base":
            icon = "📚"
            source_label = "查看企业知识库来源"
            st.caption("以下内容仅根据本地企业知识库生成")
        else:
            icon = "🌐"
            source_label = "查看互联网来源"
            st.caption("以下内容来自外部公开网页，不代表企业内部规定")
        with st.container(border=True):
            st.markdown(f"#### {icon} {section.get('title', '回答内容')}")
            st.markdown(section.get("content", ""))
            render_sources(section.get("sources", []), source_label)


def render_trace(trace: list[str]) -> None:
    if trace:
        with st.expander("查看 Agent 执行轨迹"):
            for index, step in enumerate(trace, start=1):
                st.markdown(f"{index}. {step}")


config, documents, retriever, history = initialize_services()

st.title("🧭 KnowledgePilot")
st.caption("企业知识库问答 · 自主检索路由 · DeepSeek 生成 · 来源可追溯")

with st.sidebar:
    st.header("模型配置")
    use_deepseek = st.toggle(
        "使用 DeepSeek 生成回答",
        value=config.llm_provider == "deepseek",
    )
    allow_web = st.toggle(
        "知识库不足时启用联网补充",
        value=config.enable_web_search,
    )
    if use_deepseek and not config.deepseek_api_key:
        st.info("服务器未配置 DeepSeek 服务，将使用本地抽取式回答。")
    elif use_deepseek:
        st.success(f"已选择 {config.deepseek_model}")
    if allow_web and not config.tavily_api_key:
        st.info("服务器未配置联网搜索服务，互联网补充不会执行。")

    st.divider()
    st.header("知识库")
    paths = documents.list_documents()
    st.metric("文档数量", len(paths))
    st.metric("向量片段", len(retriever.chunks))

    uploaded_files = st.file_uploader(
        "上传 PDF、Markdown 或 TXT",
        type=["pdf", "md", "txt"],
        accept_multiple_files=True,
    )
    if st.button(
        "导入文件并重建索引",
        disabled=not uploaded_files,
        use_container_width=True,
    ):
        created = 0
        duplicates = 0
        for uploaded in uploaded_files:
            _, is_created = documents.save_upload(
                uploaded.name, uploaded.getvalue()
            )
            created += int(is_created)
            duplicates += int(not is_created)
        rebuild_index(documents, retriever)
        st.success(f"新增 {created} 个文件，跳过 {duplicates} 个重复文件。")
        st.rerun()

    if paths:
        selected_document = st.selectbox(
            "已入库文档",
            paths,
            format_func=lambda path: path.name,
        )
        col_rebuild, col_delete = st.columns(2)
        if col_rebuild.button("重建", use_container_width=True):
            rebuild_index(documents, retriever)
            st.success("索引重建完成。")
        if col_delete.button("删除", use_container_width=True):
            documents.delete_document(selected_document)
            rebuild_index(documents, retriever)
            st.success("文档已删除。")
            st.rerun()

    st.divider()
    st.header("会话")
    if st.button("＋ 新建会话", use_container_width=True):
        st.session_state.conversation_id = history.create_conversation()
        st.rerun()

    conversations = history.list_conversations()
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = (
            conversations[0]["id"]
            if conversations
            else history.create_conversation()
        )
        conversations = history.list_conversations()

    ids = [item["id"] for item in conversations]
    if st.session_state.conversation_id not in ids:
        st.session_state.conversation_id = (
            ids[0] if ids else history.create_conversation()
        )
        conversations = history.list_conversations()
        ids = [item["id"] for item in conversations]

    current_index = ids.index(st.session_state.conversation_id)
    selected_conversation = st.selectbox(
        "历史会话",
        ids,
        index=current_index,
        format_func=lambda item_id: next(
            item["title"] for item in conversations if item["id"] == item_id
        ),
    )
    if selected_conversation != st.session_state.conversation_id:
        st.session_state.conversation_id = selected_conversation
        st.rerun()

    if st.button("删除当前会话", use_container_width=True):
        history.delete_conversation(st.session_state.conversation_id)
        remaining = history.list_conversations()
        st.session_state.conversation_id = (
            remaining[0]["id"]
            if remaining
            else history.create_conversation()
        )
        st.rerun()


conversation_id = st.session_state.conversation_id
messages = history.load_messages(conversation_id)

if not messages:
    st.info(
        "知识库已内置员工手册、费用报销制度和 IT 安全规范。"
        "你可以先问：正式员工有多少天年假？"
    )

for message in messages:
    avatar = "👤" if message["role"] == "user" else "🧭"
    with st.chat_message(message["role"], avatar=avatar):
        if message["role"] == "assistant" and message.get("sections"):
            render_answer_sections(message["sections"])
        else:
            st.markdown(message["content"])
        if message.get("warning"):
            st.warning(message["warning"])
        if message["role"] == "assistant":
            route_labels = {
                "smalltalk": "自然对话",
                "identity": "项目介绍",
                "clarification": "问题澄清",
                "knowledge_base": "知识库回答",
                "knowledge_base_limited": "知识库信息有限",
                "knowledge_plus_web": "知识库 + 联网补充",
                "error_fallback": "降级回答",
            }
            st.caption(
                f"路由：{route_labels.get(message.get('route'), message.get('route'))}"
                f" · 置信度：{message.get('confidence', 0):.2f}"
            )
            if not message.get("sections"):
                render_sources(message.get("sources", []))
            render_trace(message.get("trace", []))

question = st.chat_input("请输入问题")
if question:
    history.save_message(conversation_id, "user", question)
    runtime_config = replace(
        config,
        llm_provider="deepseek" if use_deepseek else "extractive",
    )
    agent = KnowledgeAgent(
        runtime_config,
        retriever,
        WebSearchService(
            timeout=runtime_config.request_timeout,
            tavily_api_key=runtime_config.tavily_api_key,
            tavily_base_url=runtime_config.tavily_base_url,
            search_depth=runtime_config.tavily_search_depth,
        ),
        AnswerGenerator(runtime_config),
    )
    with st.spinner("Agent 正在检索并组织答案……"):
        result = agent.answer(question, allow_web=allow_web)
    history.save_message(
        conversation_id,
        "assistant",
        result.answer,
        route=result.route,
        confidence=result.confidence,
        sources=result.sources,
        sections=[section.to_dict() for section in result.sections],
        intent=result.intent.to_dict() if result.intent else {},
        trace=result.trace,
        warning=result.warning,
    )
    st.rerun()
