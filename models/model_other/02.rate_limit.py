"""
速率限制
"""
import time

from langchain.chat_models import init_chat_model
from langchain_core.rate_limiters import InMemoryRateLimiter

from env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

rate_limiter = InMemoryRateLimiter(
    requests_per_second = 0.1, # 每秒一个请求，如果是0.1表示10秒一个请求
    check_every_n_seconds = 0.1, # 每隔多久检查一次
    max_bucket_size = 1, # 流量高峰时允许短时间最大请求数
)

deepseek_llm = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    rate_limiter=rate_limiter # 模型速率限制
)

for i in range(3):
    res = deepseek_llm.invoke("小明有4个苹果，吃了一个，还有几个")
    print(res)
    print(time.time())

