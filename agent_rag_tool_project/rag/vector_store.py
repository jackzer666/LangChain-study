import os

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agent_rag_tool_project.model.factory import embed_model
from agent_rag_tool_project.utils.config_handler import chroma_conf, rag_conf
from agent_rag_tool_project.utils.file_handler import txt_loader, pdf_loader, listdir_width_allowed_type, \
    get_file_md5_hex, md_loader
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


        allowed_files_path: list[str] = listdir_width_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        for path in allowed_files_path:
            # 获取文件md5
            md5_hex = get_file_md5_hex(path)
            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在知识库中，跳过")
                continue

            try:
                documents: list[Document] = get_file_documents(path)
                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue

                # print(documents)
                split_document: list[Document] = self.spliter.split_documents(documents)

                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue

                # print(split_document)

                # 将内容存入向量库中（智谱AI每次最多64条）
                batch_size = rag_conf["embedding_batch_max_size"]
                for i in range(0, len(split_document), batch_size):
                    batch = split_document[i:i + batch_size]
                    self.vector_store.add_documents(batch)

                # 记录处理好的md5值
                save_md5_hex(md5_hex)

                logger.info(f"[加载知识库]{path} 内容加载成功并已经向量化")

            except Exception as e:
                # exc_info为true会记录详细的报错堆栈，如果是false仅记录报错信息本身
                logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)


if __name__ == "__main__":
    vs = VectorStoreService()
    vs.load_document()


    # retriever = vs.get_retriever()
    # res = retriever.invoke("我想知道智己车机adb连接步骤是什么")
    # for r in res:
    #     print(r.page_content)
    #     print("-" * 20)