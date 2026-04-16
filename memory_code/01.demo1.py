from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from init_llm import deepseek_llm

agent = create_agent(
    model=deepseek_llm,
    checkpointer=InMemorySaver()
)

# 固定写法
config = {"configurable": {"thread_id": "session01"}}

res1 = agent.invoke({"messages": [{"role": "user", "content": "我是aa，你是谁？"}]}, config=config)
print(res1["messages"][-1].content)

res2 = agent.invoke({"messages": [{"role": "user", "content": "我是谁？"}]}, config=config)
print(res2["messages"][-1].content)
