import json
import re
import time
from dataclasses import dataclass, field

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from agent_rag_tool_project.model.factory import rewrite_model
from agent_rag_tool_project.utils.config_handler import rag_conf
from agent_rag_tool_project.utils.logger_handler import logger
from agent_rag_tool_project.utils.prompt_loader import load_query_rewrite_prompts


@dataclass
class QueryRewriteResult:
    original_query: str
    rewritten_query: str
    keywords: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)


class QueryRewriteService:
    def __init__(self):
        rewrite_conf = rag_conf.get("query_rewrite", {})
        self.enabled = bool(rewrite_conf.get("enabled", True))
        self.max_queries = int(rewrite_conf.get("max_queries", 3))
        self.prompt_template = PromptTemplate.from_template(load_query_rewrite_prompts())
        self.chain = self.prompt_template | rewrite_model | StrOutputParser()

    def rewrite(self, query: str) -> QueryRewriteResult:
        if not self.enabled:
            logger.info("[query_rewrite] disabled, skip model call")
            return QueryRewriteResult(
                original_query=query,
                rewritten_query=query,
                keywords=[],
                queries=[query],
            )

        try:
            started_at = time.perf_counter()
            content = self.chain.invoke(
                {
                    "query": query,
                    "max_queries": self.max_queries,
                }
            )
            elapsed = time.perf_counter() - started_at
            logger.info(f"[query_rewrite]模型调用耗时：{elapsed:.2f}s")
            data = self._loads_json(content)
            rewritten_query = self._clean_text(data.get("rewritten_query")) or query
            keywords = self._clean_list(data.get("keywords"))
            queries = self._clean_list(data.get("queries"))

            if rewritten_query not in queries:
                queries.insert(0, rewritten_query)
            queries = self._unique_texts(queries)[:self.max_queries]

            result = QueryRewriteResult(
                original_query=query,
                rewritten_query=rewritten_query,
                keywords=keywords,
                queries=queries,
            )
            logger.info(f"[query_rewrite] original_query={query}")
            logger.info(f"[query_rewrite] rewritten_query={result.rewritten_query}")
            logger.info(f"[query_rewrite] keywords={result.keywords}")
            logger.info(f"[query_rewrite] queries={result.queries}")
            return result
        except Exception as e:
            logger.error(f"[query_rewrite] query改写失败，使用原始query兜底：{str(e)}", exc_info=True)
            return QueryRewriteResult(
                original_query=query,
                rewritten_query=query,
                keywords=[],
                queries=[query],
            )

    def _loads_json(self, content: str) -> dict:
        text = content.strip()
        fenced_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
        if fenced_match:
            text = fenced_match.group(1).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            object_match = re.search(r"\{.*\}", text, re.S)
            if not object_match:
                raise
            return json.loads(object_match.group(0))

    def _clean_text(self, value) -> str:
        if not isinstance(value, str):
            return ""
        return value.strip()

    def _clean_list(self, value) -> list[str]:
        if not isinstance(value, list):
            return []
        return self._unique_texts([self._clean_text(item) for item in value])

    def _unique_texts(self, texts: list[str]) -> list[str]:
        seen = set()
        results = []
        for text in texts:
            if not text or text in seen:
                continue
            seen.add(text)
            results.append(text)
        return results
