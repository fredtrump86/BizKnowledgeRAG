from knowledge_pilot.history import HistoryStore


def test_history_round_trip(tmp_path):
    history = HistoryStore(tmp_path / "history.db")
    conversation_id = history.create_conversation()
    history.save_message(conversation_id, "user", "年假有几天？")
    history.save_message(
        conversation_id,
        "assistant",
        "五天。",
        route="knowledge_base",
        confidence=0.9,
        sources=[{"title": "员工手册.md"}],
        sections=[
            {
                "section_type": "knowledge_base",
                "title": "企业知识库内容",
                "content": "五天。",
                "sources": [{"title": "员工手册.md"}],
            }
        ],
        intent={"intent": "knowledge_query"},
        trace=["执行检索"],
    )

    messages = history.load_messages(conversation_id)
    assert len(messages) == 2
    assert messages[1]["route"] == "knowledge_base"
    assert messages[1]["sources"][0]["title"] == "员工手册.md"
    assert messages[1]["sections"][0]["section_type"] == "knowledge_base"
    assert messages[1]["intent"]["intent"] == "knowledge_query"
    assert history.list_conversations()[0]["title"] == "年假有几天？"
