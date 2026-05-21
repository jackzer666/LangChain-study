import time
from typing import Callable

from langchain.agents import AgentState
from langchain.agents.middleware import wrap_model_call, wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from agent_rag_tool_project.utils.logger_handler import logger
from agent_rag_tool_project.utils.prompt_loader import load_report_prompts, load_system_prompts


@wrap_model_call
def monitor_model(request: ModelRequest, handler: Callable):
    """
    记录每次Agent模型请求耗时。
    """
    message_count = len(request.state.get("messages", []))
    started_at = time.perf_counter()
    try:
        return handler(request)
    finally:
        elapsed = time.perf_counter() - started_at
        logger.info(f"[monitor_model]模型调用耗时：{elapsed:.2f}s，消息数：{message_count}")


@wrap_tool_call
def monitor_tool(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command]
) -> ToolMessage | Command:
    """
    工具执行的监控 理解为拦截器
    :return:
    """

    logger.info(f"[monitor_tool]执行工具：{request.tool_call['name']}")
    logger.info(f"[monitor_tool]传入参数：{request.tool_call['args']}")



    try:
         result = handler(request)
         logger.info(f"[monitor_tool]工具{request.tool_call['name']}调用成功")

         if request.tool_call['name'] == "fill_context_for_report":
            request.runtime.context["report"] = True

         return result
    except Exception as e:
        logger.error(f"[monitor_tool]工具{request.tool_call['name']}调用失败，原因：{str(e)}")
        raise e


@before_model
def log_before_model(
        # 整个agent中的状态记录
        state: AgentState,
        # 记录了整个执行过程中的上下文信息
        runtime: Runtime,
):
    """
    在模型执行前输出日志
    :return:
    """
    logger.info(f"[log_before_model]即将调用模型，带有{len(state['messages'])}条消息")
    logger.debug(f"[log_before_model]{type(state['messages'][-1]).__name__} | {state['messages'][-1].content.strip()}")
    return None


@dynamic_prompt    # 每次在生成提示词之前，调用此函数
def report_prompt_switch(request: ModelRequest):
    """
    动态切换提示词
    :return:
    """
    is_report = request.runtime.context.get("report", False) # 获取不到默认是false

    if is_report:  # 是报告生成场景，返回报告生成提示词内容
        return load_report_prompts()

    return load_system_prompts()
