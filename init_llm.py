"""
init_chat_model 初始化聊天模型
"""
from langchain.chat_models import init_chat_model

from env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, ZHIPUAI_API_KEY, ZHIPUAI_BASE_URL

deepseek_llm = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek", # 可省略，或者换直接写openai
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)

zhipu_llm = init_chat_model(
    model="glm-4",
    model_provider="openai",
    api_key=ZHIPUAI_API_KEY,
    base_url=ZHIPUAI_BASE_URL,
)