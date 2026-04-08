from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from init_llm import deepseek_llm

# 1. 创建工具 通过@tool注解告诉大模型这是一个工具函数
@tool
def get_weather(local: str) -> str:
    """获取天气信息"""
    return f"{local}的天气是晴天"

# 准备message
messages = []
humanMessage = HumanMessage(content="北京天气怎么样")
messages.append(humanMessage)

# 2. 工具告诉模型，绑定工具
model_with_tools = deepseek_llm.bind_tools([get_weather])

# 3. 模型不会调用工具，只是知道要调用工具，需要手动调用
res = model_with_tools.invoke(messages)
messages.append(res)

if res.tool_calls:
    for tool_call in res.tool_calls:
        print(tool_call)
        if tool_call["name"] == "get_weather":
            tool_res = get_weather.invoke(tool_call)
            messages.append(tool_res)

final_res = model_with_tools.invoke(messages)

# 4. 模型返回结果
print(final_res)