from langchain_chroma import Chroma

from RAG_project.config_data import collection_name, persist_directory, similarity_threshold


class VectorStoreService(object):
    def __init__(self, embedding):
        """
        :param embedding: 嵌入模型的转入
        """
        self.embedding = embedding
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding,
            persist_directory=persist_directory,
        )

    def get_retriever(self):
        """返回向量检索器，方便加入chain
        :return:
        """
        return self.vector_store.as_retriever(seach_kwargs={"k": similarity_threshold})