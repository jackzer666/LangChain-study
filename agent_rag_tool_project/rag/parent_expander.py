import json
import os
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document

from agent_rag_tool_project.rag.ranker import MultiQueryRankItem
from agent_rag_tool_project.utils.config_handler import chroma_conf, rag_conf
from agent_rag_tool_project.utils.path_tool import get_abs_path


@dataclass
class ParentRankItem:
    parent_key: tuple[Any, ...]
    source_file: str
    docs: list[Document] = field(default_factory=list)
    child_items: list[MultiQueryRankItem] = field(default_factory=list)
    best_child_item: MultiQueryRankItem | None = None
    best_child_score: float = 0.0
    final_score: float = 0.0
    matched_queries: set[str] = field(default_factory=set)
    child_chunk_ids: set[str] = field(default_factory=set)

    @property
    def child_hit_count(self) -> int:
        return len(self.child_chunk_ids)

    @property
    def query_hit_count(self) -> int:
        return len(self.matched_queries)


class HeadingParentExpander:
    """
    Markdown heading-aware parent retrieval.

    召回和排序仍然发生在child chunk级别；本类把child rank items按Markdown标题小节聚合为parent docs。
    """

    def __init__(self):
        self.chunk_file_path = get_abs_path(
            os.path.join(chroma_conf["persist_directory"], rag_conf["chunks_file"])
        )
        parent_conf = rag_conf.get("parent_retrieval", {})
        self.fallback_parent_chunk_count = parent_conf.get("fallback_parent_chunk_count", 3)
        self.chunk_by_key: dict[tuple[str, int], Document] = {}
        self.chunk_by_id: dict[str, Document] = {}
        self.docs_by_parent_key: dict[tuple[Any, ...], list[Document]] = {}
        self._load_chunks()

    def _load_chunks(self):
        if not os.path.exists(self.chunk_file_path):
            return

        with open(self.chunk_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)
                metadata = data.get("metadata", {})
                source_file = metadata.get("source_file")
                chunk_index = data.get("chunk_index")
                chunk_id = data.get("chunk_id")

                if source_file is None or chunk_index is None:
                    continue

                doc = Document(
                    page_content=data.get("content", ""),
                    metadata={
                        "h1": metadata.get("h1"),
                        "h2": metadata.get("h2"),
                        "h3": metadata.get("h3"),
                        "chunk_id": chunk_id,
                        "source_file": source_file,
                        "doc_id": data.get("doc_id"),
                        "chunk_index": int(chunk_index),
                        "parent_id": data.get("parent_id"),
                    },
                )

                self.chunk_by_key[(source_file, int(chunk_index))] = doc
                if chunk_id:
                    self.chunk_by_id[str(chunk_id)] = doc

                parent_key = self._parent_key(
                    doc,
                    fallback_parent_chunk_count=self.fallback_parent_chunk_count,
                )
                self.docs_by_parent_key.setdefault(parent_key, []).append(doc)

        for docs in self.docs_by_parent_key.values():
            docs.sort(key=lambda doc: int(doc.metadata.get("chunk_index") or 0))

    def expand_rank_items(
            self,
            rank_items: list[MultiQueryRankItem],
            final_k: int,
            max_parent_chars: int = 3000,
            fallback_parent_chunk_count: int = 3,
            query_hit_bonus: float = 0.15,
            child_hit_bonus: float = 0.1,
            max_bonus_hits: int = 3,
    ) -> list[Document]:
        parent_items: dict[tuple[Any, ...], ParentRankItem] = {}

        for item in rank_items:
            child_doc = self._resolve_child_doc(item.doc)
            if child_doc is None:
                continue

            parent_key = self._parent_key(
                child_doc,
                fallback_parent_chunk_count=fallback_parent_chunk_count,
            )
            source_file = child_doc.metadata.get("source_file")

            parent_item = parent_items.get(parent_key)
            if parent_item is None:
                parent_item = ParentRankItem(
                    parent_key=parent_key,
                    source_file=source_file,
                    docs=self.docs_by_parent_key.get(parent_key, [child_doc]),
                )
                parent_items[parent_key] = parent_item

            parent_item.child_items.append(item)

            if item.final_score > parent_item.best_child_score:
                parent_item.best_child_score = item.final_score
                parent_item.best_child_item = item

            child_chunk_id = child_doc.metadata.get("chunk_id")
            if child_chunk_id:
                parent_item.child_chunk_ids.add(str(child_chunk_id))

            for query in item.matched_queries or []:
                parent_item.matched_queries.add(query)

        for parent_item in parent_items.values():
            query_bonus = min(parent_item.query_hit_count, max_bonus_hits) * query_hit_bonus
            child_bonus = min(parent_item.child_hit_count, max_bonus_hits) * child_hit_bonus
            parent_item.final_score = parent_item.best_child_score + query_bonus + child_bonus

        sorted_parent_items = sorted(
            parent_items.values(),
            key=lambda item: item.final_score,
            reverse=True,
        )

        return [
            self._to_parent_document(parent_item, max_parent_chars)
            for parent_item in sorted_parent_items[:final_k]
        ]

    def _resolve_child_doc(self, doc: Document) -> Document | None:
        metadata = doc.metadata or {}

        source_file = metadata.get("source_file")
        chunk_index = metadata.get("chunk_index")
        if source_file is not None and chunk_index is not None:
            return doc

        chunk_id = metadata.get("chunk_id")
        if chunk_id:
            return self.chunk_by_id.get(str(chunk_id))

        return None

    def _parent_key(
            self,
            doc: Document,
            fallback_parent_chunk_count: int | None = None,
    ) -> tuple[Any, ...]:
        metadata = doc.metadata or {}
        source_file = metadata.get("source_file")
        h1 = metadata.get("h1")
        h2 = metadata.get("h2")
        h3 = metadata.get("h3")

        if h3:
            return source_file, "h3", h1, h2, h3
        if h2:
            return source_file, "h2", h1, h2
        if h1:
            return source_file, "h1", h1

        chunk_index = int(metadata.get("chunk_index") or 0)
        parent_chunk_count = fallback_parent_chunk_count or self.fallback_parent_chunk_count
        return source_file, "block", chunk_index // parent_chunk_count

    def _to_parent_document(self, parent_item: ParentRankItem, max_parent_chars: int) -> Document:
        docs = self._select_docs_for_parent(parent_item, max_parent_chars)
        content = "\n\n".join(doc.page_content for doc in docs)

        first_doc = docs[0] if docs else None
        metadata = dict(first_doc.metadata) if first_doc else {}

        best_child_doc = None
        if parent_item.best_child_item:
            best_child_doc = self._resolve_child_doc(parent_item.best_child_item.doc)
        best_child_metadata = best_child_doc.metadata if best_child_doc else {}

        metadata.update({
            "_parent_retrieval": True,
            "_parent_strategy": "heading",
            "_parent_key": " / ".join(str(part) for part in parent_item.parent_key if part),
            "_parent_score": parent_item.final_score,
            "_best_child_score": parent_item.best_child_score,
            "_best_child_chunk_id": best_child_metadata.get("chunk_id"),
            "_best_child_chunk_index": best_child_metadata.get("chunk_index"),
            "_child_hit_count": parent_item.child_hit_count,
            "_query_hit_count": parent_item.query_hit_count,
            "_matched_queries": sorted(parent_item.matched_queries),
            "_child_chunk_ids": sorted(parent_item.child_chunk_ids),
            "_parent_start_chunk_index": docs[0].metadata.get("chunk_index") if docs else None,
            "_parent_end_chunk_index": docs[-1].metadata.get("chunk_index") if docs else None,
            "_parent_trimmed": self._content_length(parent_item.docs) > max_parent_chars,
        })

        return Document(page_content=content, metadata=metadata)

    def _select_docs_for_parent(
            self,
            parent_item: ParentRankItem,
            max_parent_chars: int,
    ) -> list[Document]:
        docs = parent_item.docs
        if not docs:
            return []

        if self._content_length(docs) <= max_parent_chars:
            return docs

        best_doc = self._resolve_best_child_doc(parent_item)
        if best_doc is None:
            return self._take_until_limit(docs, max_parent_chars)

        best_index = int(best_doc.metadata.get("chunk_index") or 0)
        docs_by_index = {
            int(doc.metadata.get("chunk_index") or 0): doc
            for doc in docs
        }

        selected = []
        selected_indexes = set()
        total_chars = 0
        distance = 0

        while total_chars < max_parent_chars:
            candidates = [best_index] if distance == 0 else [best_index - distance, best_index + distance]
            added = False

            for index in candidates:
                doc = docs_by_index.get(index)
                if doc is None or index in selected_indexes:
                    continue

                doc_len = len(doc.page_content)
                if selected and total_chars + doc_len > max_parent_chars:
                    continue

                selected.append(doc)
                selected_indexes.add(index)
                total_chars += doc_len
                added = True

            if not added and distance > len(docs):
                break

            distance += 1

        selected.sort(key=lambda doc: int(doc.metadata.get("chunk_index") or 0))
        return selected or [best_doc]

    def _resolve_best_child_doc(self, parent_item: ParentRankItem) -> Document | None:
        if not parent_item.best_child_item:
            return None
        return self._resolve_child_doc(parent_item.best_child_item.doc)

    def _take_until_limit(self, docs: list[Document], max_parent_chars: int) -> list[Document]:
        selected = []
        total_chars = 0

        for doc in docs:
            doc_len = len(doc.page_content)
            if selected and total_chars + doc_len > max_parent_chars:
                break
            selected.append(doc)
            total_chars += doc_len

        return selected

    def _content_length(self, docs: list[Document]) -> int:
        return sum(len(doc.page_content) for doc in docs)
