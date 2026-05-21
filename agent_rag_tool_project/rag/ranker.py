"""
RAG检索排序工具。

该模块只负责分词、关键词打分、单query排序分和多query排序分的计算。
"""
from dataclasses import dataclass

import jieba
from langchain_core.documents import Document

from agent_rag_tool_project.utils.logger_handler import logger


BM25_WEIGHT = 0.7
VECTOR_WEIGHT = 0.3
HYBRID_RECALL_K = 15
HYBRID_RANK_BASE = 25
MULTI_QUERY_HIT_BONUS = 5


@dataclass
class HybridRankItem:
    """
    单个query混合召回阶段的排序单元。

    Attributes:
        doc: 被召回的文档片段。
        bm25_rank: BM25召回中的排名；未被BM25召回时为None。
        vector_rank: 向量召回中的排名；未被向量召回时为None。
        vector_distance: 向量召回返回的距离分数；距离越小通常越相似。
        score: 单query轻量rerank后的最终分数。
    """
    doc: Document
    bm25_rank: int | None = None
    vector_rank: int | None = None
    vector_distance: float | None = None
    score: float = 0.0


@dataclass
class MultiQueryRankItem:
    """
    多query合并阶段的排序单元。

    Attributes:
        doc: 被召回的文档片段。
        base_score: 多个query命中该文档时累计得到的基础分。
        final_score: 叠加命中次数奖励和原始query关键词分后的最终分。
        query_hit_count: 该文档被多少个query命中过。
        matched_queries: 命中过该文档的query列表，用于调试召回来源。
    """
    doc: Document
    base_score: float = 0.0
    final_score: float = 0.0
    query_hit_count: int = 0
    matched_queries: list[str] | None = None

    def add_match(self, query: str, score: float):
        """
        记录一次query命中，并把该query产生的分数累计到base_score。

        Args:
            query: 当前命中文档的检索query。
            score: 当前query对该文档贡献的基础分。
        """
        self.base_score += score
        self.query_hit_count += 1
        if self.matched_queries is None:
            self.matched_queries = []
        self.matched_queries.append(query)


class RagRanker:
    """
    RAG检索排序器。

    该类不访问向量库或模型，只做纯打分逻辑，方便单独测试和后续替换排序策略。
    """

    def tokens(self, text: str) -> set[str]:
        """
        使用jieba对文本做中文分词，并统一转为小写去重集合。

        Args:
            text: 待分词文本，通常是用户query。

        Returns:
            set[str]: 去空白、去重后的关键词集合。
        """
        tokens = {
            token.strip().lower()
            for token in jieba.cut(text, cut_all=False)
            if token.strip()
        }
        logger.info(f"[中文分词_tokens]:{text}\n{tokens}")
        return tokens

    def keyword_score(self, query: str, doc: Document) -> float:
        """
        将问题分词后，对文档片段标题和内容分别做匹配记分处理。

        Args:
            query: 原始问题或检索query。
            doc: 待评分的文档片段。

        Returns:
            float: 标题和正文关键词命中的加权分数。
        """
        return self.keyword_score_with_tokens(self.tokens(query), doc)

    def keyword_score_with_tokens(self, query_tokens: set[str], doc: Document) -> float:
        """
        使用已分好的query tokens计算文档关键词命中分。

        Args:
            query_tokens: 已分词、去重后的query关键词集合。
            doc: 待评分的文档片段。

        Returns:
            float: 关键词命中加权分数。
        """
        if not query_tokens:
            return 0.0

        title_text = " ".join(
            str(doc.metadata.get(key, "") or "").lower()
            for key in ["h1", "h2", "h3"]
        )
        content_text = doc.page_content.lower()

        title_hit = sum(1 for token in query_tokens if token in title_text)
        content_hit = sum(1 for token in query_tokens if token in content_text)
        return title_hit * 8 + content_hit * 3

    def rank_hybrid_items(
            self,
            merged: dict[str, HybridRankItem],
            query_tokens: set[str],
    ) -> list[HybridRankItem]:
        """
        对单query合并召回结果做轻量排序。

        Args:
            merged: BM25和向量召回合并后的结果。
            query_tokens: 当前query的分词结果。

        Returns:
            list[HybridRankItem]: 按score从高到低排序后的结果。
        """
        for item in merged.values():
            item.score = self.hybrid_score(item, query_tokens)
        return sorted(merged.values(), key=lambda item: item.score, reverse=True)

    def hybrid_score(self, item: HybridRankItem, query_tokens: set[str]) -> float:
        """
        计算单个文档片段在单query混合检索中的分数。

        Args:
            item: 单query排序单元。
            query_tokens: 当前query的分词结果。

        Returns:
            float: 单query混合排序分数。
        """
        score = 0.0
        if item.bm25_rank is not None:
            score += max(0, HYBRID_RANK_BASE - item.bm25_rank) * BM25_WEIGHT
        if item.vector_rank is not None:
            score += max(0, HYBRID_RANK_BASE - item.vector_rank) * VECTOR_WEIGHT
        score += self.keyword_score_with_tokens(query_tokens, item.doc)
        return score

    def rank_multi_query_items(
            self,
            merged: dict[str, MultiQueryRankItem],
            original_query_tokens: set[str],
    ) -> list[MultiQueryRankItem]:
        """
        对多query合并结果做最终排序。

        Args:
            merged: 多query召回结果容器。
            original_query_tokens: 用户原始问题的分词结果。

        Returns:
            list[MultiQueryRankItem]: 按final_score从高到低排序后的结果。
        """
        for item in merged.values():
            item.final_score = self.multi_query_score(item, original_query_tokens)
        return sorted(merged.values(), key=lambda item: item.final_score, reverse=True)

    def multi_query_score(
            self,
            item: MultiQueryRankItem,
            original_query_tokens: set[str],
    ) -> float:
        """
        计算多query合并阶段的最终分数。

        Args:
            item: 多query排序单元。
            original_query_tokens: 用户原始问题的分词结果。

        Returns:
            float: 多query最终排序分数。
        """
        return (
            item.base_score
            + item.query_hit_count * MULTI_QUERY_HIT_BONUS
            + self.keyword_score_with_tokens(original_query_tokens, item.doc)
        )

    def doc_key(self, doc: Document) -> str:
        """
        获取文档片段的去重key。

        Args:
            doc: 待生成key的文档片段。

        Returns:
            str: 优先使用chunk_id，没有chunk_id时用source_file和正文前100字符兜底。
        """
        chunk_id = doc.metadata.get("chunk_id")
        if chunk_id:
            return str(chunk_id)
        source_file = doc.metadata.get("source_file", "")
        return f"{source_file}:{doc.page_content[:100]}"
