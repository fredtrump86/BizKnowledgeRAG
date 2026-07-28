from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import BinaryIO, Iterable

from pypdf import PdfReader

from .models import TextChunk


ALLOWED_EXTENSIONS = {".pdf", ".md", ".txt"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_filename(name: str) -> str:
    cleaned = Path(name).name
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", cleaned)
    return cleaned.strip(" .") or "document.txt"


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            candidates = [
                text.rfind(mark, start + chunk_size // 2, end)
                for mark in ("\n", "。", "！", "？", ".", ";")
            ]
            boundary = max(candidates)
            if boundary > start:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


class DocumentService:
    def __init__(self, knowledge_base_dir: Path, chunk_size: int, overlap: int):
        self.knowledge_base_dir = knowledge_base_dir
        self.upload_dir = knowledge_base_dir / "uploads"
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def list_documents(self) -> list[Path]:
        return sorted(
            path
            for path in self.knowledge_base_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
        )

    def save_upload(self, file_name: str, file_data: bytes) -> tuple[Path, bool]:
        suffix = Path(file_name).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise ValueError("仅支持 PDF、Markdown 和 TXT 文件")
        if not file_data:
            raise ValueError("上传文件不能为空")
        digest = sha256_bytes(file_data)
        for existing in self.list_documents():
            if sha256_bytes(existing.read_bytes()) == digest:
                return existing, False
        destination = self.upload_dir / safe_filename(file_name)
        if destination.exists():
            destination = destination.with_stem(
                f"{destination.stem}_{digest[:8]}"
            )
        destination.write_bytes(file_data)
        return destination, True

    def delete_document(self, path: Path) -> None:
        resolved = path.resolve()
        root = self.knowledge_base_dir.resolve()
        if root not in resolved.parents:
            raise ValueError("只能删除知识库目录中的文件")
        if resolved.exists() and resolved.is_file():
            resolved.unlink()

    def build_chunks(self) -> list[TextChunk]:
        all_chunks: list[TextChunk] = []
        for path in self.list_documents():
            all_chunks.extend(self._chunks_from_file(path))
        return all_chunks

    def _chunks_from_file(self, path: Path) -> Iterable[TextChunk]:
        raw = path.read_bytes()
        document_id = sha256_bytes(raw)[:16]
        pages: list[tuple[int, str]]
        if path.suffix.lower() == ".pdf":
            reader = PdfReader(path)
            pages = [
                (number, page.extract_text() or "")
                for number, page in enumerate(reader.pages, start=1)
            ]
        else:
            text = raw.decode("utf-8-sig", errors="replace")
            pages = [(1, text)]

        chunks: list[TextChunk] = []
        sequence = 0
        for page_number, page_text in pages:
            for content in split_text(
                page_text, self.chunk_size, self.overlap
            ):
                sequence += 1
                content_hash = hashlib.sha256(
                    content.encode("utf-8")
                ).hexdigest()
                chunks.append(
                    TextChunk(
                        chunk_id=f"{document_id}_p{page_number}_c{sequence}",
                        document_id=document_id,
                        file_name=path.name,
                        page_number=page_number,
                        content=content,
                        content_hash=content_hash,
                    )
                )
        return chunks

