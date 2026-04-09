"""
推理模型
"""
from langchain.chat_models import init_chat_model

from env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

deepseek_llm = init_chat_model(
    model="deepseek-reasoner",
    model_provider="deepseek",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)

res = deepseek_llm.invoke("小明有4个苹果，吃了一个，还有几个")
print(res)
print(res.content_blocks)

reasoning_steps = [b for b in res.content_blocks if b["type"] == "reasoning"]
print(reasoning_steps)