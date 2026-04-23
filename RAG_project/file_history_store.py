import json
import os
from typing import Sequence
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, message_to_dict
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory

from init_llm import deepseek_llm

# 获取 RAG_project 目录下的 chat_history 目录
_HISTORY_DIR = os.path.join(os.path.dirname(__file__), 'chat_history')


def get_history(session_id):
    return FileChatMessageHistory(session_id, _HISTORY_DIR)

class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id, storage_path):
        # 会话id
        self.session_id = session_id
        # 不同会话id的存储文件，所在的文件夹路径
        self.storage_path = storage_path
        # 完整的文件路径
        self.file_path = os.path.join(self.storage_path, self.session_id)
        # 确保文件夹是存在的
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def add_message(self, messages: Sequence[BaseMessage]) -> None:
        try:
            print("我准备好了")
            # Sequence序列 类似list tuple
            # 已有的消息列表
            all_messages = list(self.messages)
            # 新的和已有的融合成一个list
            # 注意：add_messages 对每条消息单独调用 add_message，
            # 所以 messages 可能是单条 BaseMessage，需要先转为 list
            if isinstance(messages, BaseMessage):
                all_messages.append(messages)
            else:
                all_messages.extend(messages)

            new_messages = [message_to_dict(message) for message in all_messages]
            # 将数据同步写入文件
            with open(self.file_path, "w", encoding='utf-8') as f:
                json.dump(new_messages, f, ensure_ascii=False)
        except Exception as e:
            print("错误信息:", e)


    @property # @property装饰器将messages方法变成成员属性用
    def messages(self) -> list[BaseMessage]:
        try:
            with open(self.file_path, "r", encoding='utf-8') as f:
                messages_data = json.load(f) # 返回值是list[字典]
                return messages_from_dict(messages_data)
        except FileNotFoundError:
            return []

    def clear(self) -> None:
        with open(self.file_path, "w", encoding='utf-8') as f:
            json.dump([], f)


# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "你需要根据会话历史回应用户问题，对话历史："),
#         MessagesPlaceholder('chat_history'),
#         ("human", "请回答下面问题：{input}")
#     ]
# )
#
# base_chain = prompt | deepseek_llm | StrOutputParser()
#
# conversation_chain = RunnableWithMessageHistory(
#     base_chain,
#     get_history,
#     input_messages_key="input",
#     history_messages_key="chat_history"
# )
#
# if __name__ == "__main__":
#     session_config = {
#         "configurable": {
#             "session_id": "user_001"
#         }
#     }
#     res1 = conversation_chain.invoke({"input": "我是zt, 你是谁"}, session_config)
#     print(res1)
#
#     res = conversation_chain.invoke({"input": "我是谁"}, session_config)
#     print(res)