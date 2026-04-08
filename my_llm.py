"""
创建各类LLM模型
"""
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI

from env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, ZHIPUAI_API_KEY, ZHIPUAI_BASE_URL

# Model Class 方式初始化模型
deepseek_llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    api_base=DEEPSEEK_BASE_URL
)


from langchain_community.chat_models import ChatZhipuAI
from langchain.messages import AIMessage, HumanMessage, SystemMessage

zhipu_llm = ChatZhipuAI(
    api_key=ZHIPUAI_API_KEY,
    model="glm-4",
)
 
# 所有的llm都可以使用openai的class来创建
zhipu_llm2 = ChatOpenAI(
    api_key=ZHIPUAI_API_KEY,
    base_url=ZHIPUAI_BASE_URL,
    model="glm-4",
)