from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelResponse, ModelRequest
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

from env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, ZHIPUAI_API_KEY, ZHIPUAI_BASE_URL

@tool
def get_weather(local: str) -> str:
    """获取某地的天气情况"""
    return f"{local}天气是下雪"

@tool
def get_location() -> str:
    """获取本地地理位置"""
    return "这里是杭州"


# 定义了中间件，在模型调用前执行
@wrap_model_call
def change_model(request: ModelRequest, handler) -> ModelResponse:
    print(request)
    message_count = len(request.state["messages"])
    if message_count < 3:
        model = base_model
    else:
        model = advanced_model

    # 重置模型类型
    return handler(request.override(model=model))


base_model = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)

advanced_model = init_chat_model(
    model="glm-4",
    model_provider="openai",
    api_key=ZHIPUAI_API_KEY,
    base_url=ZHIPUAI_BASE_URL,
    temperature=1
)

agent = create_agent(
    model=base_model,
    tools=[get_weather, get_location],
    middleware=[change_model],
    system_prompt=SystemMessage('你是一个数学老师，只能回答数学相关问题')
)

res = agent.invoke({"messages": [{"role": "user", "content": "我这里天气怎么样"}]})

for msg in res["messages"]:
    msg.pretty_print()