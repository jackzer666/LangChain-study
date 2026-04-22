from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import ZhipuAIEmbeddings
from env_utils import ZHIPUAI_API_KEY

vector_store = Chroma(
    collection_name="test", # 类似表名
    embedding_function=ZhipuAIEmbeddings(
        model="embedding-3",  # 指定使用的模型，例如 "embedding-3"
        # dimensions=1024     # 可选参数：指定输出向量的维度
        api_key=ZHIPUAI_API_KEY
    ),
    persist_directory="./chroma_db" # 指定数据存放的文件夹
)

# loader = CSVLoader(
#     file_path="/Users/jackzer/Downloads/Takeout 2/已保存/日本.csv",
#     encoding="utf-8",
#     source_column="标题"
# )
#
# docs = loader.load()
#
#  # ===== 分批添加文档，每批最多64条 =====
# batch_size = 64
# for i in range(0, len(docs), batch_size):
#     batch_docs = docs[i:i + batch_size]
#     # 生成当前批次的 id
#     batch_ids = ["id" + str(j) for j in range(i + 1, i + len(batch_docs) + 1)]
#     vector_store.add_documents(documents=batch_docs, ids=batch_ids)
#
# # 删除 传入[id]
# vector_store.delete(["id10"])

# 检索
result = vector_store.similarity_search(
    "我住的酒店是哪一个",
    3,
    # filter={"source": "沙吉诺宇酒店"}
)

print(result)