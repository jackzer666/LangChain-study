from langchain.agents import create_agent
from langchain_core.tools import tool

from init_llm import deepseek_llm


@tool
def get_weather(local: str) -> str:
    """获取天气信息"""
    return f"{local}的天气是雨天"

agent = create_agent(
    model=deepseek_llm,
    tools=[get_weather],
)

res = agent.invoke({"messages": [{"role": "user", "content": "杭州天气怎么样"}]})
print(res)
