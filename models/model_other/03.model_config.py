"""
速率限制
"""
import time
from typing import Any
from uuid import UUID

from langchain.chat_models import init_chat_model
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.rate_limiters import InMemoryRateLimiter

from env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

# 要继承BaseCallbackHandler
class MyCustomCallbackHandler(BaseCallbackHandler):
    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        print(serialized)

    def on_llm_end(self, response, **kwargs: Any) -> None:
        print(response)

custom_handler = MyCustomCallbackHandler()

deepseek_llm = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    configurable_fields=("model", "temperature"), # 指定可以被后续覆盖的字段，如果不配置，即使后续配置了也不允许覆盖
)

config = {
    "run_name": "test_generation",   # 在LangSmith中此次运行会显示为test_generation
    "tags": ["tag1", "tag2"],        # 打上标签便于分类查找
    "metadata": {"user_id": "124"},  # 记录用户ID
    "callbacks": [custom_handler],   # 启用自定义回调
    "configurable": {
        "model": "deepseek-reasoner",
        "temperature": 0.7
    }
}

res = deepseek_llm.invoke("小明有4个苹果，吃了一个，还有几个", config=config)
# print(res)

