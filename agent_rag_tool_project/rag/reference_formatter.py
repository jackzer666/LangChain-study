"""
RAG参考资料格式化服务。

负责把检索到的Document格式化为可返回给Agent的参考资料文本，不调用模型生成答案。
"""
import re

from langchain_core.documents import Document


class RagReferenceFormatter:
    """
    RAG参考资料格式化器。

    只处理“Document列表 -> 精简参考资料字符串”的流程。Agent拿到该字符串后负责最终总结回答。
    """

    def __init__(self, max_content_chars: int = 260):
        """
        初始化参考资料格式化器。

        Args:
            max_content_chars: 每条参考资料最多保留的正文字符数。
        """
        self.max_content_chars = max_content_chars

    def _compact_content(self, content: str) -> str:
        """
        压缩单条资料正文，只保留可供Agent判断相关性的短摘录。
        """
        normalized_content = re.sub(r"\s+", " ", content or "").strip()
        if len(normalized_content) <= self.max_content_chars:
            return normalized_content

        excerpt = normalized_content[: self.max_content_chars].rstrip()
        sentence_end_index = max(
            excerpt.rfind("。"),
            excerpt.rfind("；"),
            excerpt.rfind(";"),
            excerpt.rfind("."),
        )
        if sentence_end_index >= int(self.max_content_chars * 0.6):
            excerpt = excerpt[: sentence_end_index + 1]
        return f"{excerpt}..."

    def format_references(self, docs: list[Document]) -> str:
        """
        将检索到的Document列表格式化为工具返回的精简参考资料文本。

        Args:
            docs: 检索得到的参考文档片段列表。

        Returns:
            str: 带编号、来源、标题、chunk_id和正文摘录的参考资料文本。
        """
        if not docs:
            return "未检索到相关参考资料。"

        context_parts = []
        for index, doc in enumerate(docs, start=1):
            metadata = doc.metadata or {}
            heading = " / ".join(
                str(metadata.get(key) or "")
                for key in ["h1", "h2", "h3"]
                if metadata.get(key)
            )
            context_parts.append(
                "\n".join(
                    [
                        f"【参考资料{index}】",
                        f"来源：{metadata.get('source_file', '')}",
                        f"标题：{heading}",
                        f"chunk_id：{metadata.get('chunk_id', '')}",
                        f"内容摘录：{self._compact_content(doc.page_content)}",
                    ]
                )
            )
        return "\n\n".join(context_parts)
