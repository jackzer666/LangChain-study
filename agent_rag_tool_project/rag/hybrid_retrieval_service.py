"""
RAG混合检索服务。

负责BM25召回、向量召回、单query合并排序和多query合并排序。
"""
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

from agent_rag_tool_project.rag.bm25 import Bm25
from agent_rag_tool_project.rag.ranker import BM25_WEIGHT, HYBRID_RANK_BASE, HYBRID_RECALL_K, VECTOR_WEIGHT, \
    HybridRankItem, MultiQueryRankItem, RagRanker
from agent_rag_tool_project.rag.vector_store import VectorStoreService
from agent_rag_tool_project.utils.logger_handler import logger


class HybridRetrievalService:
    """
    混合检索服务。

    外部可调用hybrid_retriever()观察单query检索效果，或调用multi_query_hybrid_retriever()
    执行query rewrite后的多路召回合并。
    """

    def __init__(
            self,
            vector_store: VectorStoreService | None = None,
            bm25_retriever=None,
            ranker: RagRanker | None = None,
    ):
        """
        初始化混合检索依赖。

        Args:
            vector_store: 可注入的向量库服务；不传时自动创建。
            bm25_retriever: 可注入的BM25检索器；不传时从Bm25创建。
            ranker: 可注入的排序器；不传时自动创建。
        """
        self.vector_store = vector_store or VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.bm25_retriever = bm25_retriever or Bm25().get_bm25_retriever()
        self.ranker = ranker or RagRanker()
        self.ensemble_retriever = self.get_ensemble_retriever()

    def hybrid_retriever(self, query: str, final_k: int = 10) -> list[Document]:
        """
        自定义混合检索：BM25召回 + 向量召回 + 去重 + 轻量rerank。

        Args:
            query: 检索query。
            final_k: 最终返回的文档数量。

        Returns:
            list[Document]: 排序后的文档片段列表。
        """
        query_tokens = self.ranker.tokens(query)
        rank_items = self.ranker.rank_hybrid_items(
            self._merge_hybrid_results(query),
            query_tokens,
        )
        logger.info(f"[hybrid排序结果]:{rank_items}")
        return self._to_ranked_docs(rank_items[:final_k])

    def _merge_hybrid_results(self, query: str) -> dict[str, HybridRankItem]:
        """
        合并单个query下的BM25召回和向量召回结果。

        Args:
            query: 检索query。

        Returns:
            dict[str, HybridRankItem]: key为文档唯一标识，value为合并后的排序单元。
        """
        merged: dict[str, HybridRankItem] = {}
        self._merge_bm25_results(query, merged)
        self._merge_vector_results(query, merged)
        return merged

    def _merge_bm25_results(self, query: str, merged: dict[str, HybridRankItem]):
        """
        将BM25召回结果写入merged。

        Args:
            query: 检索query。
            merged: 单query召回结果容器，会被原地更新。
        """
        for rank, doc in enumerate(self.bm25_retriever.invoke(query), start=1):
            logger.info(f"[BM25 召回]{repr(doc)}")
            key = self.ranker.doc_key(doc)
            item = merged.setdefault(key, HybridRankItem(doc=doc))
            item.bm25_rank = rank

    def _merge_vector_results(self, query: str, merged: dict[str, HybridRankItem]):
        """
        将向量召回结果写入merged。

        Args:
            query: 检索query。
            merged: 单query召回结果容器，会被原地更新。
        """
        vector_results = self.vector_store.similarity_search_with_score(query, k=HYBRID_RECALL_K)
        for rank, (doc, distance) in enumerate(vector_results, start=1):
            logger.info(f"[向量召回内容]{repr(doc)}")
            logger.info(f"[向量召回distance]{distance}")
            key = self.ranker.doc_key(doc)
            item = merged.setdefault(key, HybridRankItem(doc=doc))
            item.vector_rank = rank
            item.vector_distance = distance

    def _to_ranked_docs(self, rank_items: list[HybridRankItem]) -> list[Document]:
        """
        将HybridRankItem列表转换回Document列表，并写入排序调试元数据。

        Args:
            rank_items: 已排序的单query排序单元列表。

        Returns:
            list[Document]: 带_rerank_score等元数据的Document列表。
        """
        docs = []
        for item in rank_items:
            doc = item.doc
            doc.metadata["_rerank_score"] = item.score
            doc.metadata["_bm25_rank"] = item.bm25_rank
            doc.metadata["_vector_rank"] = item.vector_rank
            doc.metadata["_vector_distance"] = item.vector_distance
            docs.append(doc)
        return docs

    def get_ensemble_retriever(self):
        """
        创建LangChain EnsembleRetriever。

        当前主链路使用自定义hybrid_retriever()，该方法保留用于兼容旧代码和临时调试。

        Returns:
            EnsembleRetriever: BM25和向量检索器按固定权重组合后的LangChain检索器。
        """
        return EnsembleRetriever(
            retrievers=[
                self.bm25_retriever,
                self.retriever,
            ],
            weights=[BM25_WEIGHT, VECTOR_WEIGHT],
        )

    def multi_query_hybrid_retriever(
            self,
            original_query: str,
            queries: list[str],
            final_k: int = 10,
    ) -> list[Document]:
        """
        多query混合检索。每个query分别走BM25+向量召回，再按chunk_id去重合并。

        Args:
            original_query: 用户原始问题，用于最终关键词相关性评分。
            queries: 实际用于召回的一组query。
            final_k: 最终返回的文档数量。

        Returns:
            list[Document]: 多query去重、合并、排序后的文档片段列表。
        """
        original_query_tokens = self.ranker.tokens(original_query)
        merged: dict[str, MultiQueryRankItem] = {}

        for query in queries:
            self._merge_multi_query_results(query, merged)

        rank_items = self.ranker.rank_multi_query_items(merged, original_query_tokens)
        docs = self._to_multi_query_docs(rank_items[:final_k])
        logger.info(f"[multi_query_retriever] final_docs={docs}")
        return docs

    def _merge_multi_query_results(
            self,
            query: str,
            merged: dict[str, MultiQueryRankItem],
    ):
        """
        将某一个query的hybrid_retriever结果合并到多query结果容器。

        Args:
            query: 当前正在执行召回的query。
            merged: 多query召回结果容器，会被原地更新。
        """
        docs = self.hybrid_retriever(query, final_k=HYBRID_RECALL_K)
        logger.info(
            f"[multi_query_retriever] query={query} recalled_chunk_ids="
            f"{[doc.metadata.get('chunk_id') for doc in docs]}"
        )

        for rank, doc in enumerate(docs, start=1):
            key = self.ranker.doc_key(doc)
            rerank_score = float(doc.metadata.get("_rerank_score") or 0)
            query_score = rerank_score + max(0, HYBRID_RANK_BASE - rank)
            item = merged.setdefault(key, MultiQueryRankItem(doc=doc))
            item.add_match(query, query_score)

    def _to_multi_query_docs(self, rank_items: list[MultiQueryRankItem]) -> list[Document]:
        """
        将MultiQueryRankItem列表转换回Document列表，并写入多query调试元数据。

        Args:
            rank_items: 已排序的多query排序单元列表。

        Returns:
            list[Document]: 带_multi_query_score、_query_hit_count等元数据的Document列表。
        """
        docs = []
        for item in rank_items:
            doc = item.doc
            doc.metadata["_multi_query_score"] = item.final_score
            doc.metadata["_query_hit_count"] = item.query_hit_count
            doc.metadata["_matched_queries"] = item.matched_queries or []
            docs.append(doc)
        return docs
