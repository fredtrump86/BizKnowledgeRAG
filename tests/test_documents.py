from pathlib import Path

from knowledge_pilot.documents import (
    DocumentService,
    safe_filename,
    split_text,
)


def test_split_text_preserves_content_with_overlap():
    text = "第一段内容。" * 80
    chunks = split_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(0 < len(chunk) <= 100 for chunk in chunks)


def test_save_upload_detects_duplicate(tmp_path: Path):
    service = DocumentService(tmp_path, chunk_size=100, overlap=20)
    first_path, first_created = service.save_upload(
        "制度.md", "# 年假\n正式员工年假为5天。".encode("utf-8")
    )
    second_path, second_created = service.save_upload(
        "另一个名字.md", "# 年假\n正式员工年假为5天。".encode("utf-8")
    )
    assert first_created is True
    assert second_created is False
    assert first_path == second_path


def test_safe_filename_removes_path_and_invalid_characters():
    assert safe_filename("../../危险:文件?.txt") == "危险_文件_.txt"

