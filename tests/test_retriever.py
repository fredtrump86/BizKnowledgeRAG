from knowledge_pilot.models import TextChunk
from knowledge_pilot.retriever import LocalVectorRetriever


def make_chunk(chunk_id: str, content: str) -> TextChunk:
    return TextChunk(
        chunk_id=chunk_id,
        document_id="doc",
        file_name="制度.md",
        page_number=1,
        content=content,
        content_hash=chunk_id,
    )


def test_local_retriever_returns_relevant_chunk(tmp_path):
    retriever = LocalVectorRetriever(tmp_path / "index.joblib")
    retriever.build(
        [
            make_chunk("leave", "正式员工每年享有五天带薪年假。"),
            make_chunk("security", "员工账号必须启用多因素认证。"),
        ]
    )
    results = retriever.search("正式员工有多少天年假", top_k=2)
    assert results
    assert results[0].chunk.chunk_id == "leave"
    assert results[0].score > 0


def test_index_can_be_saved_and_loaded(tmp_path):
    index_path = tmp_path / "index.joblib"
    retriever = LocalVectorRetriever(index_path)
    retriever.build([make_chunk("one", "报销必须在十个工作日内提交。")])

    loaded = LocalVectorRetriever(index_path)
    assert loaded.load() is True
    assert loaded.search("报销期限")[0].chunk.chunk_id == "one"

