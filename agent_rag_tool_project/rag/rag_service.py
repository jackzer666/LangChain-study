"""
总结服务类：用户提问，搜索参考资料，将提问和参考资料提交给模型，让模型总结回复
"""
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from agent_rag_tool_project.model.factory import chat_model
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

    def _init_chain(self):
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain

    def retriever_docs(self, query: str) -> list[Document]:
        """
        检索文档函数
        :param query:
        :return:
        """
        rag_docs = self.retriever.invoke(query)
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
    res = rs.rag_summarize("今天天气怎么样？")
    print(res)