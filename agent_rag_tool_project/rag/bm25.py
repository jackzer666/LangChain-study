import json
import os

from langchain_core.documents import Document

from agent_rag_tool_project.utils.config_handler import chroma_conf, rag_conf
from agent_rag_tool_project.utils.path_tool import get_abs_path
from langchain_community.retrievers import BM25Retriever


class Bm25:
    def __init__(self):
        self.chunk_file_path = get_abs_path(os.path.join(chroma_conf["persist_directory"], rag_conf["chunks_file"]))

    def save_to_jsonl(self, document: list[Document]):
        # 确保目录存在，只创建一次
        store_dir = get_abs_path(chroma_conf["persist_directory"])
        os.makedirs(store_dir, exist_ok=True)

        for doc in document:
            current_file = doc.metadata.get("source_file")
            chunk_index = doc.metadata.get("chunk_index")

            chunk_data = {
                # chunk_id作为主键放在最外层，访问效率更高
                "chunk_id": doc.metadata.get("chunk_id"),
                # 命名成content，便于后续兼容其他检索方案
                "content": doc.page_content,
                "metadata": {
                    "h1": doc.metadata.get("h1", None),
                    "h2": doc.metadata.get("h2", None),
                    "h3": doc.metadata.get("h3", None),
                    "source_file": current_file,
                },
                # 当前片段属于哪个文档，便于parent retrieval
                "doc_id": doc.metadata.get("doc_id") or current_file,
                # 这是文档中的第几个chunk，便于后续扩展上下文
                "chunk_index": chunk_index,

                # TODO: 以下后续实现
                "parent_id": "",
                "keywords": [],
                "token_count": 0,
            }

            with open(self.chunk_file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(chunk_data, ensure_ascii=False) + "\n")


    def get_chunks(self) -> list[Document]:
        """
        从jsonl文件读取所有chunks，封装成list[Document]格式。

        Returns:
            list[Document]: Document列表，每个Document包含:
                - page_content: chunk内容
                - metadata: 包含h1、h2、h3、chunk_id、source_file
        """
        # 文件不存在则创建空文件
        if not os.path.exists(self.chunk_file_path):
            os.makedirs(os.path.dirname(self.chunk_file_path), exist_ok=True)
            open(self.chunk_file_path, "w", encoding="utf-8").close()
            return []

        # 读取所有chunks，使用迭代器避免一次性加载全部内容
        documents = []
        with open(self.chunk_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                doc = Document(
                    page_content=data.get("content", ""),
                    metadata={
                        "h1": data.get("metadata", {}).get("h1"),
                        "h2": data.get("metadata", {}).get("h2"),
                        "h3": data.get("metadata", {}).get("h3"),
                        "chunk_id": data.get("chunk_id"),
                        "source_file": data.get("metadata", {}).get("source_file"),
                        "doc_id": data.get("doc_id"),
                        "chunk_index": data.get("chunk_index"),
                        "parent_id": data.get("parent_id"),
                    },
                )
                documents.append(doc)

        return documents

    def get_bm25_retriever(self):
        """
        创建BM25检索器
        :return:
        """
        documents = self.get_chunks()
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 20
        return bm25_retriever
