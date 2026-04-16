from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import SummarizationMiddleware, before_model, after_model
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolRuntime
from langchain_core.tools import tool
from langgraph.types import Command

from init_llm import deepseek_llm

"""
自定义状态步骤
1、需要定义一个类，这个类是AgentState，定义要存储的状态字段
2、构建agent时指定state_schema=
3、可以在调用Agent时，传入之定义状态 / 在Agent运行中通过中间件（@before_model @after_model）、tool方式设置
"""

class CustomAgentState(AgentState):
    name: str
    hobby: list


@tool
def get_weather(city: str) -> str:
    """根据城市查询天气
    """
    return f"城市{city}的天气是晴天"

@before_model
def before_model(state, runtime: ToolRuntime):
    print("before_model_state", state)

    messages = state["messages"]

    return None


@after_model
def after_model(state, runtime: ToolRuntime):
    print("after_model_state", state)

    messages = state["messages"]
    print(messages)


agent = create_agent(
    model=deepseek_llm,
    tools=[get_weather],
    checkpointer=InMemorySaver(),
    state_schema=CustomAgentState,
    middleware=[
        before_model,
        after_model,
        SummarizationMiddleware(
            model=deepseek_llm,
            trigger=("messages", 5), # 消息达到5条时总结
            keep=("messages", 2), # 保留最后两条消息
            summary_prompt="请摘要以下内容：{messages}"
        )
    ]
)

# 固定写法
config = {"configurable": {"thread_id": "session01"}}




res1 = agent.invoke({
    "messages": [{"role": "user", "content": "我是aa，你是谁？"}],
    "name": "aa",
    "hobby": "吃饭"
}, config=config)
print(res1)

print("-" * 50)

res2 = agent.invoke({"messages": [{"role": "user", "content": "今天北京的天气如何"}]}, config=config)
print(res2)

print("-" * 50)

res3 = agent.invoke({"messages": [{"role": "user", "content": "我的名字叫什么"}]}, config=config)
print(res3)
