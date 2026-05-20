"""
RAG总结门面服务。

该模块保留RagSummarizeService作为对外入口，内部委托给query规划、混合检索和参考资料格式化服务。
"""
from langchain_core.documents import Document

from agent_rag_tool_project.rag.hybrid_retrieval_service import HybridRetrievalService
from agent_rag_tool_project.rag.query_planner import QueryPlanner
from agent_rag_tool_project.rag.reference_formatter import RagReferenceFormatter
from agent_rag_tool_project.utils.logger_handler import logger


class RagSummarizeService(object):
    """
    RAG主门面服务。

    外部主要调用retriever_docs()获取文档，或调用rag_summarize()获取可交给Agent总结的参考资料。
    """

    def __init__(self):
        """
        初始化RAG门面依赖。

        会创建QueryPlanner、HybridRetrievalService和RagReferenceFormatter。
        """
        self.query_planner = QueryPlanner()
        self.retrieval_service = HybridRetrievalService()
        self.reference_formatter = RagReferenceFormatter()

    def retriever_docs(self, query: str) -> list[Document]:
        """
        RAG文档检索主入口。

        Args:
            query: 用户原始问题。

        Returns:
            list[Document]: 最终排序后的参考文档片段。
        """
        queries = self.query_planner.build_search_queries(query)
        rag_docs = self.retrieval_service.multi_query_hybrid_retriever(query, queries, final_k=10)
        logger.info(f"[retriever_docs]rag检索结果：{rag_docs}")
        return rag_docs

    def rag_summarize(self, query: str) -> str:
        """
        RAG资料检索入口。

        Args:
            query: 用户原始问题。

        Returns:
            str: 格式化后的参考资料文本，由Agent基于这些资料生成最终答案。
        """
        context_docs = self.retriever_docs(query)
        return self.reference_formatter.format_references(context_docs)


if __name__ == "__main__":
    rs = RagSummarizeService()
    res = rs.retriever_docs("在播放器项目中，对于window做了什么处理")
