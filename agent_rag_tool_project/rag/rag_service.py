"""
总结服务类：用户提问，搜索参考资料，将提问和参考资料提交给模型，让模型总结回复
"""
import jieba
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from agent_rag_tool_project.model.factory import chat_model
from agent_rag_tool_project.rag.bm25 import Bm25
from agent_rag_tool_project.rag.vector_store import VectorStoreService
from agent_rag_tool_project.utils.logger_handler import logger
from agent_rag_tool_project.utils.prompt_loader import load_rag_prompts


class RagSummarizeService(object):
    def __init__(self):
        # 向量存储
        self.vector_store = VectorStoreService()
        # 检索器
        self.retriever = self.vector_store.get_retriever()
        # 提示词文本
        self.prompt_text = load_rag_prompts()
        # 提示词模板
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        # 模型
        self.model = chat_model
        # rag执行链
        self.chain = self._init_chain()

        # BM25检索器（关键词优先）
        self.bm25_retriever = Bm25().get_bm25_retriever()
        #
        self.ensemble_retriever = self.get_ensemble_retriever()

    def _init_chain(self):
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain

    def _tokens(self, text: str) -> set[str]:
        """
        中文分词器
        :param text:
        :return:
        """
        seg_list = set(token.strip().lower() for token in jieba.cut(text, cut_all=False) if token.strip())
        logger.info(f"[中文分词_tokens]:{text}\n{seg_list}")
        return seg_list

    def _keyword_score(self, query: str, doc: Document) -> float:
        """
        将问题分词后，对文档片段的标题、内容分别做匹配记分处理
        :param query:
        :param doc:
        :return:
        """
        query_tokens = self._tokens(query)
        if not query_tokens:
            return 0.0

        title_text = " ".join(
            str(doc.metadata.get(k, "") or "").lower()
            for k in ["h1", "h2", "h3"]
        )
        content_text = doc.page_content.lower()

        title_hit = sum(1 for token in query_tokens if token in title_text)
        content_hit = sum(1 for token in query_tokens if token in content_text)

        return title_hit * 8 + content_hit * 3


    def hybrid_retriever(self, query: str, final_k: int = 10) -> list[Document]:
        """
        自定义混合检索，主要是增加了去重和rerank的步骤
        :param query:
        :param final_k:
        :return:
        """
        merged = {}
        # 1. BM25 召回
        bm25_docs = self.bm25_retriever.invoke(query)
        for rank, doc in enumerate(bm25_docs, start=1):
            logger.info(f"[BM25 召回]{repr(doc)}")
            key = doc.metadata.get("chunk_id", "")
            if key not in merged:
                merged[key] = {
                    "doc": doc,
                    "bm25_rank": rank,
                    "vector_rank": None,
                    "vector_distance": None,
                }
            else:
                merged[key]["bm25_rank"] = rank

        # 2. 向量召回
        vector_results = self.vector_store.similarity_search_with_score(query, k=20)
        for rank, (doc, distance) in enumerate(vector_results, start=1):
            logger.info(f"[向量召回内容]{repr(doc)}")
            logger.info(f"[向量召回distance]{distance}")
            key = doc.metadata.get("chunk_id", "")
            if key not in merged:
                merged[key] = {
                    "doc": doc,
                    "bm25_rank": None,
                    "vector_rank": rank,
                    "vector_distance": distance,
                }
            else:
                merged[key]["vector_rank"] = rank
                merged[key]["vector_distance"] = distance

        # 3. 轻量rerank
        def score_item(item):
            doc = item["doc"]
            score = 0.0

            # BM25越靠前越加分
            if item["bm25_rank"] is not None:
                score += max(0, 25 - item["bm25_rank"]) * 0.7

            # 向量越靠前越加分
            if item["vector_rank"] is not None:
                score += max(0, 25 - item["vector_rank"]) * 0.3

            # 标题与正文关键词命中加分
            score += self._keyword_score(query, doc)

            return score


        rank_items = sorted(merged.values(), key=score_item, reverse=True)
        logger.info(f"[hybrid排序结果]:{rank_items}")

        docs = []
        for item in rank_items[:final_k]:
            doc = item["doc"]
            doc.metadata["_rerank_score"] = score_item(item)
            doc.metadata["_bm25_rank"] = item["bm25_rank"]
            doc.metadata["_vector_rank"] = item["vector_rank"]
            doc.metadata["_vector_distance"] = item["vector_distance"]
            docs.append(doc)

        return docs


    def get_ensemble_retriever(self):
        ensemble_retriever = EnsembleRetriever(
            retrievers = [
                self.bm25_retriever,
                self.retriever,
            ],
            weights = [0.7, 0.3]
        )
        return ensemble_retriever

    def retriever_docs(self, query: str) -> list[Document]:
        """
        检索文档函数
        :param query:
        :return:
        """
        rag_docs = self.hybrid_retriever(query, final_k=10)
        logger.info(f"[retriever_docs]rag检索结果：{rag_docs}")
        return rag_docs

    def rag_summarize(self, query: str) -> str:
        """
        rag总结
        :param query:
        :return:
        """
        context_docs = self.retriever_docs(query)

        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            context += f"【参考资料{counter}】：参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"

        return self.chain.invoke(
            {
                "input": query,
                "context": context
            }
        )


if __name__ == "__main__":
    rs = RagSummarizeService()
    res = rs.hybrid_retriever("在播放器项目中，对于window做了什么处理")
