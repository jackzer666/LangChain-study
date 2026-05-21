import re

from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from agent_rag_tool_project.agent.tools.agent_tools import rag_summarize, fill_context_for_report
from agent_rag_tool_project.agent.tools.middleware import monitor_model, monitor_tool, log_before_model, report_prompt_switch
from agent_rag_tool_project.model.factory import chat_model
from agent_rag_tool_project.utils.prompt_loader import load_system_prompts


class ReactAgent:
    def __init__(self):
        self.qa_agent = self._create_agent(tools=[rag_summarize])
        self.report_agent = self._create_agent(tools=[rag_summarize, fill_context_for_report])

    def _create_agent(self, tools: list):
        return create_agent(
            model = chat_model,
            system_prompt = load_system_prompts(),
            tools = tools,
            middleware = [
                monitor_model,
                monitor_tool,
                log_before_model,
                report_prompt_switch,
            ]
        )

    def _is_report_request(self, query: str) -> bool:
        """
        只有用户明确要求生成报告时，才启用报告工具链。
        """
        return bool(re.search(r"(生成|写|出|做|撰写|整理).{0,8}报告|报告.{0,4}(生成|撰写)", query))

    def execute_stream(self, query: str):
        input_dict = {
            "messages": [
                {"role": "user", "content": query},
            ]
        }
        is_report_request = self._is_report_request(query)
        agent = self.report_agent if is_report_request else self.qa_agent

        # 第三个参数context就是上下文runtime中的信息，就是我们做提示词切换的标记
        for chunk in agent.stream(
            input_dict,
            stream_mode="values",
            context={"report": False}
        ):
            lastest_message = chunk["messages"][-1]
            if (
                    isinstance(lastest_message, AIMessage)
                    and not lastest_message.tool_calls
                    and lastest_message.content
            ):
                yield lastest_message.content.strip() + "\n"


if __name__ == '__main__':
    ra = ReactAgent()
    for chunk in ra.execute_stream("你帮我生成报告"):
        print(chunk, end="", flush=True)
