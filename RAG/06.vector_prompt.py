from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import ZhipuAIEmbeddings
from env_utils import ZHIPUAI_API_KEY
from init_llm import zhipu_llm


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "以我提供的已知参考资料为主，简洁和专业的回答用户问题，参考资料：{context}。"),
        ("user", "用户提问：{input}")
    ]
)

vector_store = InMemoryVectorStore(
    embedding=ZhipuAIEmbeddings(
        model="embedding-3",
        api_key=ZHIPUAI_API_KEY
    ),
)

# 准备资料
vector_store.add_texts([
    "今天天气晴天",
    "晴天应该去公园玩",
    "晴天也需要带伞，因为可以遮阳"
])

input_text = "今天我可以做什么"

# 检索向量库
result = vector_store.similarity_search(input_text, 3)

reference_text = "["
for doc in result:
    reference_text += doc.page_content
reference_text += "]"


# LCEL写法，管道自动传递，上一个结果直接作为下一个入参
chain = prompt | zhipu_llm | StrOutputParser()
res = chain.invoke({"input": input_text, "context": reference_text})
print(res)