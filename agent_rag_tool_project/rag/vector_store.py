import json
import os
import uuid

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agent_rag_tool_project.model.factory import embed_model
from agent_rag_tool_project.utils.config_handler import chroma_conf, rag_conf
from agent_rag_tool_project.utils.file_handler import txt_loader, pdf_loader, listdir_width_allowed_type, \
    get_file_md5_hex, md_loader, restore_code_blocks
from agent_rag_tool_project.utils.logger_handler import logger
from agent_rag_tool_project.utils.path_tool import get_abs_path


class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name = chroma_conf["collection_name"],
            embedding_function = embed_model,
            persist_directory = get_abs_path(chroma_conf["persist_directory"]),
        )
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len
        )
        self.md_spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["md_chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["md_separators"],
            length_function=len
        )

    def get_retriever(self):
        """
        获取检索器对象
        :return:
        """
        return self.vector_store.as_retriever(
            search_kwargs = {"k": chroma_conf["k"]},
        )

    def load_document(self):
        """
        从数据文件夹内读取数据文件，转为向量存入向量数据库
        要计算文件的MD5去重
        :return:
        """

        def check_md5_hex(md5_for_check: str):
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w", encoding="utf-8").close()
                return False # md5没处理过

            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True # md5处理过

                return False # md5没处理过

        def save_md5_hex(md5_for_check: str):
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        def get_file_documents(read_path: str):
            """
            把文件内容变成document对象
            :param read_path:
            :return:
            """
            if read_path.endswith(".txt"):
                return txt_loader(read_path)

            if read_path.endswith(".pdf"):
                return pdf_loader(read_path)

            if read_path.endswith(".md"):
                return md_loader(read_path)

            return []

        def file_splitter(documents: list[Document], path: str) -> list[Document]:
            """
            不同类型文件的处理方式不同
            :return:
            """
            if path.endswith(".md"):
                split_document: list[Document] = []
                for doc in documents:
                    split_docs = [doc]
                    # 还是有问题的，没法语义化处理，可能还不如不要split_documents
                    if len(doc.page_content) > chroma_conf["chunk_size"]:
                        # 使用md_spliter而不是spliter
                        split_docs = self.md_spliter.split_documents([doc])

                    for split_doc in split_docs:
                        code_map = split_doc.metadata.get("_code_map", {})
                        # 还原代码块
                        split_doc.page_content = restore_code_blocks(split_doc.page_content, code_map)
                        split_doc.metadata = {k: v for k, v in split_doc.metadata.items() if k != "_code_map"}
                        split_document.append(split_doc)
            else:
                split_document: list[Document] = self.spliter.split_documents(documents)

            return split_document

        allowed_files_path: list[str] = listdir_width_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        for path in allowed_files_path:
            chunk_mappings = []

            # 获取文件md5
            md5_hex = get_file_md5_hex(path)
            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在知识库中，跳过")
                continue

            try:
                documents: list[Document] = get_file_documents(path)
                if not documents :
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue

                # print(documents)
                split_document: list[Document] = file_splitter(
                    documents,
                    path
                )

                # 为每个chunk添加metadata，尤其是chunk_id，为后续分析召回作准备
                for doc in split_document:
                    chunk_id = str(uuid.uuid4())
                    doc.metadata["chunk_id"] = chunk_id
                    doc.metadata["source_file"] = path # path可能是完整路径，后续考虑是否修改
                    chunk_mappings.append({
                        "chunk_id": chunk_id,
                        "source_file": path,
                        "content": doc.page_content,
                    })

                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue

                # 将内容存入向量库中（智谱AI每次最多64条）
                batch_size = rag_conf["embedding_batch_max_size"]
                for i in range(0, len(split_document), batch_size):
                    batch = split_document[i:i + batch_size]
                    self.vector_store.add_documents(batch)

                # 记录处理好的md5值
                save_md5_hex(md5_hex)

                logger.info(f"[加载知识库]{path} 内容加载成功并已经向量化")

                # 暂存chunk_id与内容的对应关系
                mapping_path = get_abs_path("chroma_db/mapping2.json")
                if os.path.exists(mapping_path):
                    with open(mapping_path, "r", encoding="utf-8") as f:
                        old_data = json.load(f)
                else:
                    old_data = []
                old_data.extend(chunk_mappings)
                with open(mapping_path, "w", encoding="utf-8") as f:
                    json.dump(old_data, f, ensure_ascii=False, indent=2)

            except Exception as e:
                # exc_info为true会记录详细的报错堆栈，如果是false仅记录报错信息本身
                logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)


    def load_document_verify(self):
        """
        临时函数，用于验证md文件切割的问题
        :return:
        """
        file_path = get_abs_path("data/frontend/project-player.md")
        documents: list[Document] = md_loader(file_path)
        logger.info(f"[load_document_verify]初步切割\n{documents}")

        split_document: list[Document] = []
        for doc in documents:
            split_docs = [doc]
            if len(doc.page_content) > chroma_conf["chunk_size"]:
                split_docs = self.md_spliter.split_documents([doc])

            for split_doc in split_docs:
                code_map = split_doc.metadata.get("_code_map", {})
                split_doc.page_content = restore_code_blocks(split_doc.page_content, code_map)
                split_doc.metadata = {k: v for k, v in split_doc.metadata.items() if k != "_code_map"}
                split_document.append(split_doc)

        logger.info(f"[load_document_verify]md_spliter切割\n{split_document}")


    def retrieval_debug(self, query: str):
        """
        临时函数，用于检测recall召回率
        :param query:
        :return:
        """
        retriever = self.get_retriever()
        docs = retriever.retrieve(query)

        logger.info(f"[召回率测试][QUERY]: {query}")

        for i, doc in enumerate(docs):
            rank = i + 1
            chunk_id = doc.metadata.get("chunk_id")
            source_file = doc.metadata.get("source_file")
            content = doc.page_content[:500]

            logger.info(f"[召回率测试][Rank:]: {rank}")
            logger.info(f"[召回率测试][Chunk ID:]: {chunk_id}")
            logger.info(f"[召回率测试][Source:]: {source_file}")
            logger.info(f"[召回率测试][Content:]: {content}")

        return docs


if __name__ == "__main__":
    vs = VectorStoreService()
    # vs.load_document_verify()
    vs.load_document()
    #
    # vs.retrieval_debug(
    #     "我想知道智己车机adb连接步骤是什么"
    # )

    # retriever = vs.get_retriever()
    # res = retriever.invoke("我想知道智己车机adb连接步骤是什么")
    # for r in res:
    #     print(r.page_content)
    #     print("-" * 20)