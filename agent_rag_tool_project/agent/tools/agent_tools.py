import random

from langchain_core.tools import tool
from agent_rag_tool_project.rag.rag_service import RagSummarizeService

rag = RagSummarizeService()

@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    return rag.rag_summarize(query)

@tool(description="获取指定城市的天气，以消息字符串的形式返回")
def get_weather(city: str) -> str:
    return f"城市{city}天气是晴天，气温26摄氏度，最近6小时降雨概率极低"

@tool(description="获取用户所在的城市名称，以字符串形式返回")
def get_user_location() -> str:
    return random.choice(["北京", "上海", "深圳"])

@tool(description="获取用户的id，以字符串形式返回")
def get_user_id() -> str:
    return random.choice(["1001", "1002", "1003"])

@tool(description="获取当前月份，以纯字符串形式返回")
def get_current_month() -> str:
    return random.choice(["1月", "2月", "3月"])

@tool(description="从外部系统中获取用户的使用记录，以字符串形式返回，如果未检索到返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    return f"{user_id}用户在{month}的使用数据是：用了两次"


@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report():
    return "fill_context_for_report已调用"