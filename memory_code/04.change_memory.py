from langchain.agents import create_agent, AgentState
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
def get_info(name: str, runtime: ToolRuntime) -> str:
    """获取用户信息
    Args:
        name 用户姓名
    Returns:
        用户信息
    """
    name = runtime.state["name"]
    hobby = runtime.state["hobby"]
    return f"用户{name}的爱好是{hobby}"

@tool
def update_info(name: str, hobby: str, runtime: ToolRuntime) -> Command:
    """修改用户信息
    Args:
        name: 用户名字
        hobby: 爱好信息
    Returns:
        Command: 变更用户信息
    """
    if not hobby:
        return Command(
            update={
                "messages": [
                    ToolMessage(content="缺少爱好", tool_call_id=runtime.tool_call_id)
                ]
            }
        )

    update = {
        "hobby": hobby,
        "messages": [
            ToolMessage(
                content=f"用户{name}的爱好更新为{hobby}",
                tool_call_id=runtime.tool_call_id
            )
        ]
    }

    return Command(update=update)

agent = create_agent(
    model=deepseek_llm,
    tools=[get_info, update_info],
    checkpointer=InMemorySaver(),
    state_schema=CustomAgentState
)

# 固定写法
config = {"configurable": {"thread_id": "session01"}}




res1 = agent.invoke({
    "messages": [{"role": "user", "content": "我是aa，你是谁？"}],
    "name": "aa",
    "hobby": "吃饭"
}, config=config)
print(res1)

res2 = agent.invoke({"messages": [{"role": "user", "content": "我是aa，我的爱好还有跑步"}]}, config=config)
print(res2)


res3 = agent.invoke({"messages": [{"role": "user", "content": "打印我的信息"}]}, config=config)
print(res3["messages"][-1].content)
