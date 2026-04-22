from langchain_community.embeddings import ZhipuAIEmbeddings
from env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, ZHIPUAI_API_KEY, ZHIPUAI_BASE_URL


# 1. 初始化嵌入模型
embeddings = ZhipuAIEmbeddings(
    model="embedding-3",  # 指定使用的模型，例如 "embedding-3"
    # dimensions=1024     # 可选参数：指定输出向量的维度
    api_key=ZHIPUAI_API_KEY
)

# 2. 将文本转换为向量
text = "天晴"
# single_vector = embeddings.embed_query(text)
single_vector2 = embeddings.embed_documents([text, '天晴了'])

print(single_vector2)