"""
RAG检索query规划服务。

负责把用户原始问题和query rewrite结果组合成实际用于检索的query列表。
"""
from agent_rag_tool_project.rag.query_rewrite import QueryRewriteService
from agent_rag_tool_project.utils.config_handler import rag_conf
from agent_rag_tool_project.utils.logger_handler import logger


class QueryPlanner:
    """
    检索query规划器。

    外部调用build_search_queries()即可得到最终用于多路召回的query列表。
    """

    def __init__(self, query_rewriter: QueryRewriteService | None = None):
        """
        初始化query规划器。

        Args:
            query_rewriter: 可注入的QueryRewriteService；不传时自动创建。
        """
        self.query_rewriter = query_rewriter or QueryRewriteService()

    def build_search_queries(self, query: str) -> list[str]:
        """
        基于用户原始问题构建实际用于召回的query列表。

        Args:
            query: 用户原始问题。

        Returns:
            list[str]: 实际用于多路召回的query列表。
        """
        rewrite_result = self.query_rewriter.rewrite(query)
        rewrite_conf = rag_conf.get("query_rewrite", {})
        use_original_query = bool(rewrite_conf.get("use_original_query", True))

        queries = []
        if use_original_query:
            queries.append(query)
        queries.extend(rewrite_result.queries)
        if rewrite_result.keywords:
            queries.append(" ".join(rewrite_result.keywords))
        final_queries = self.unique_texts(queries)
        logger.info(f"[build_search_queries]多路召回最终关键字：{final_queries}")
        return final_queries

    def unique_texts(self, texts: list[str]) -> list[str]:
        """
        对字符串列表去空、去重，并保持原顺序。

        Args:
            texts: 待处理的字符串列表。

        Returns:
            list[str]: 清理后的字符串列表。
        """
        seen = set()
        results = []
        for text in texts:
            text = (text or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            results.append(text)
        return results
