from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolRuntime
from langchain_core.tools import tool

from init_llm import deepseek_llm

"""
自定义状态步骤
1、需要定义一个类，这个类是AgentState，定义要存储的状态字段
2、构建agent时指定state_schema=
3、可以在调用Agent时，传入之定义状态 / 在Agent运行中通过中间件（@before_model @after_model）、tool方式设置
"""

class CustomAgentState(AgentState):
    user_id: str
    hobby: list


@tool
def get_info(runtime: ToolRuntime) -> str:
    """获取用户信息
    Args:
        name 用户姓名
    Returns:
        用户信息
    """
    user_id = runtime.state["user_id"]
    return f"用户id是{user_id}"

agent = create_agent(
    model=deepseek_llm,
    tools=[get_info],
    checkpointer=InMemorySaver(),
    state_schema=CustomAgentState
)

# 固定写法
config = {"configurable": {"thread_id": "session01"}}




res1 = agent.invoke({
    "messages": [{"role": "user", "content": "我是aa，你是谁？"}],
    "user_id": "1",
    "test": "2"
}, config=config)
print(res1)

res2 = agent.invoke({"messages": [{"role": "user", "content": "我是谁？"}]}, config=config)
state = agent.get_state(config=config)
print(res2)
